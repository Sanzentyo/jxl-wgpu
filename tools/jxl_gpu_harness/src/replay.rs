use std::sync::Arc;
use std::time::Instant;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use jxl_gpu_protocol::{
    Border2d, ChromaAxis, EpfParams, EpfPass, Extent2d, FrameSessionDesc, GaborishParams, GroupId,
    GroupPayload, HostPlane, MemoryMode, OutputDesc, OutputId, OutputLayout, PlaneData, PlaneDesc,
    PlaneId, PlaneRole, PrecisionContract, PrecisionPolicy, RenderIntent, RenderNode, RenderOp,
    RenderPlan, ResourceData, ResourceId, ResourceUpdate, SampleType, SaveParams, Scale2d,
    UpsampleParams,
};

use crate::capture::{
    CaptureFile, DataType, OperationKind, SectionKind, TensorShape, decode_f32, decode_i32,
    encode_f32,
};
use crate::compare::{AccuracyThreshold, compare_f32};
use crate::error::{Error, Result};
use crate::reference;
use crate::report::{CaseReport, CaseStatus, TimingStatistics};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum BackendKind {
    Reference,
    Wgpu,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Wgpu => "wgpu",
        }
    }
}

pub trait ReplayBackend {
    fn kind(&self) -> BackendKind;
    fn execute(&mut self, capture: &CaptureFile) -> Result<Vec<f32>>;
}

#[derive(Debug, Default)]
pub struct ReferenceBackend;

impl ReplayBackend for ReferenceBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Reference
    }

    fn execute(&mut self, capture: &CaptureFile) -> Result<Vec<f32>> {
        reference::execute_capture(capture)
    }
}

#[derive(Debug)]
pub struct WgpuReplayBackend {
    backend: jxl_wgpu::WgpuBackend,
}

impl WgpuReplayBackend {
    pub fn request_default() -> Result<Self> {
        let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(
            jxl_wgpu::WgpuBackendConfig {
                enable_timestamps: false,
                ..jxl_wgpu::WgpuBackendConfig::default()
            },
        ))
        .map_err(|error| match error {
            jxl_wgpu::Error::NoAdapter => Error::BackendUnavailable(error.to_string()),
            other => Error::Verification(format!("failed to initialize wgpu: {other}")),
        })?;
        Ok(Self { backend })
    }
}

impl ReplayBackend for WgpuReplayBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Wgpu
    }

    fn execute(&mut self, capture: &CaptureFile) -> Result<Vec<f32>> {
        let job = WgpuReplayJob::from_capture(capture)?;
        let mut session = self
            .backend
            .create_session(&job.frame, Arc::new(job.plan))
            .map_err(map_wgpu_error)?;
        for update in job.resources {
            session.update_resource(update).map_err(map_wgpu_error)?;
        }
        session.enqueue(job.payload).map_err(map_wgpu_error)?;
        let token = session
            .submit(RenderIntent::Final)
            .map_err(map_wgpu_error)?;
        let frame = session.wait(token).map_err(map_wgpu_error)?;
        let output = frame
            .outputs
            .into_iter()
            .find(|output| output.id == OutputId(0))
            .ok_or_else(|| Error::Verification("wgpu replay returned no output".into()))?;
        match output.data {
            PlaneData::F32(values) => Ok(values),
            data => Err(Error::Verification(format!(
                "wgpu replay returned {:?}, expected F32",
                data.sample_type()
            ))),
        }
    }
}

struct WgpuReplayJob {
    frame: FrameSessionDesc,
    plan: RenderPlan,
    resources: Vec<ResourceUpdate>,
    payload: GroupPayload,
}

struct EpfReplayResource {
    sigma_plane: Option<PlaneDesc>,
    update: ResourceUpdate,
}

enum ReplayInput {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl ReplayInput {
    const fn sample_type(&self) -> SampleType {
        match self {
            Self::F32(_) => SampleType::F32,
            Self::I32(_) => SampleType::I32,
        }
    }

    fn plane_data(&self, start: usize, length: usize) -> Result<PlaneData> {
        let end = start.checked_add(length).ok_or(Error::LengthOverflow)?;
        match self {
            Self::F32(values) => values
                .get(start..end)
                .map(|values| PlaneData::F32(values.to_vec())),
            Self::I32(values) => values
                .get(start..end)
                .map(|values| PlaneData::I32(values.to_vec())),
        }
        .ok_or_else(|| Error::InvalidTensor("source plane is truncated".into()))
    }
}

impl WgpuReplayJob {
    fn from_capture(capture: &CaptureFile) -> Result<Self> {
        let (input_descriptor, input_bytes) = capture.section_by_kind(SectionKind::Input)?;
        let input_shape = dense_planar_shape(input_descriptor.tensor.as_ref(), "input")?;
        let (output_descriptor, _) = capture.section_by_kind(SectionKind::Expected)?;
        let output_shape = dense_planar_shape(output_descriptor.tensor.as_ref(), "expected")?;
        if output_descriptor.data_type != DataType::F32 {
            return Err(Error::UnsupportedOperation {
                backend: "wgpu",
                operation: "non-F32 expected output".into(),
            });
        }
        if input_shape.channels > u16::from(u8::MAX) {
            return Err(Error::InvalidTensor(format!(
                "{} channels exceed the output API limit",
                input_shape.channels
            )));
        }
        let input = match (&capture.metadata.operation.kind, input_descriptor.data_type) {
            (OperationKind::Affine, DataType::I32) => ReplayInput::I32(decode_i32(input_bytes)?),
            (_, DataType::F32) => ReplayInput::F32(decode_f32(input_bytes)?),
            (operation, data_type) => {
                return Err(Error::UnsupportedOperation {
                    backend: "wgpu",
                    operation: format!("{} with {data_type:?} input", operation.as_str()),
                });
            }
        };
        let input_sample_type = input.sample_type();
        let input_extent = Extent2d::new(input_shape.width, input_shape.height);
        let output_extent = Extent2d::new(output_shape.width, output_shape.height);
        let channel_count = u32::from(input_shape.channels);
        let source_ids = (0..channel_count).map(PlaneId).collect::<Vec<_>>();
        let output_ids = (0..channel_count)
            .map(|index| PlaneId(channel_count + index))
            .collect::<Vec<_>>();
        let epf_resource = epf_replay_resource(
            capture,
            input_shape,
            PlaneId(channel_count.checked_mul(2).ok_or(Error::LengthOverflow)?),
        )?;
        let mut planes = source_ids
            .iter()
            .copied()
            .map(|id| PlaneDesc {
                id,
                extent: input_extent,
                stride: input_shape.width,
                sample_type: input_sample_type,
                role: PlaneRole::Source,
            })
            .collect::<Vec<_>>();
        planes.extend(output_ids.iter().copied().map(|id| PlaneDesc {
            id,
            extent: output_extent,
            stride: output_shape.width,
            sample_type: SampleType::F32,
            role: PlaneRole::Intermediate,
        }));
        if let Some(sigma) = epf_resource
            .as_ref()
            .and_then(|resource| resource.sigma_plane.as_ref())
        {
            planes.push(sigma.clone());
        }
        let nodes = operation_nodes(
            capture,
            &source_ids,
            &output_ids,
            input_shape,
            output_shape,
            epf_resource
                .as_ref()
                .and_then(|resource| resource.sigma_plane.as_ref().map(|plane| plane.id)),
        )?;
        let output_id = OutputId(0);
        let mut nodes = nodes;
        nodes.push(RenderNode {
            name: "capture-save".into(),
            op: RenderOp::Save(SaveParams {
                output: output_id,
                sample_type: SampleType::F32,
                channels: output_ids.clone(),
                layout: OutputLayout::Planar,
                orientation: jxl_gpu_protocol::OutputOrientation::Identity,
            }),
            inputs: output_ids,
            outputs: Vec::new(),
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::default(),
            precision: PrecisionContract::Exact,
        });
        let plane_len =
            usize::try_from(input_shape.channel_stride).map_err(|_| Error::LengthOverflow)?;
        let host_planes = source_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(channel, id)| {
                let start = channel
                    .checked_mul(plane_len)
                    .ok_or(Error::LengthOverflow)?;
                Ok(HostPlane {
                    id,
                    extent: input_extent,
                    stride: input_shape.width,
                    origin: (0, 0),
                    data: input.plane_data(start, plane_len)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            frame: FrameSessionDesc {
                frame_extent: output_extent,
                group_extent: output_extent,
                group_count: 1,
                precision: PrecisionPolicy::F32Only,
                memory_mode: MemoryMode::Resident,
                max_resident_bytes: 512 * 1024 * 1024,
                max_scratch_bytes: 256 * 1024 * 1024,
            },
            plan: RenderPlan {
                planes,
                nodes,
                outputs: vec![OutputDesc {
                    id: output_id,
                    extent: output_extent,
                    sample_type: SampleType::F32,
                    channels: u8::try_from(input_shape.channels)
                        .map_err(|_| Error::LengthOverflow)?,
                    layout: OutputLayout::Planar,
                    color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
                }],
            },
            resources: epf_resource
                .into_iter()
                .map(|resource| resource.update)
                .collect(),
            payload: GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: host_planes,
                vardct: None,
            },
        })
    }
}

fn epf_replay_resource(
    capture: &CaptureFile,
    input_shape: &TensorShape,
    sigma_id: PlaneId,
) -> Result<Option<EpfReplayResource>> {
    if capture.metadata.operation.kind != OperationKind::Epf {
        return Ok(None);
    }
    if input_shape.channels != 3 {
        return Err(Error::InvalidTensor(
            "EPF replay requires exactly three input channels".into(),
        ));
    }
    let variable = reference::epf_uses_variable_sigma(&capture.metadata.operation)?;
    let sigma = capture.section_by_name(SectionKind::Parameter, "sigma")?;
    let resource = ResourceId(0);
    if !variable {
        if sigma.is_some() {
            return Err(Error::InvalidMetadata(
                "constant-sigma EPF capture must not contain a sigma parameter plane".into(),
            ));
        }
        let value = capture
            .metadata
            .operation
            .parameters
            .get("sigma")
            .copied()
            .unwrap_or(-0.58) as f32;
        if !value.is_finite() {
            return Err(Error::InvalidMetadata(
                "EPF constant sigma must be a finite f32 value".into(),
            ));
        }
        return Ok(Some(EpfReplayResource {
            sigma_plane: None,
            update: ResourceUpdate {
                id: resource,
                revision: 0,
                data: ResourceData::F32(vec![value]),
            },
        }));
    }

    let (descriptor, bytes) = sigma.ok_or_else(|| {
        Error::InvalidMetadata(
            "variable-sigma EPF capture is missing the named sigma parameter plane".into(),
        )
    })?;
    if descriptor.data_type != DataType::F32 {
        return Err(Error::InvalidTensor(
            "EPF sigma parameter plane must contain F32 values".into(),
        ));
    }
    let shape = dense_planar_shape(descriptor.tensor.as_ref(), "EPF sigma parameter")?;
    if shape.channels != 1
        || shape.width < input_shape.width.div_ceil(8)
        || shape.height < input_shape.height.div_ceil(8)
    {
        return Err(Error::InvalidTensor(format!(
            "EPF sigma plane must cover at least {}x{} blocks",
            input_shape.width.div_ceil(8),
            input_shape.height.div_ceil(8)
        )));
    }
    let values = decode_f32(bytes)?;
    let expected_values =
        usize::try_from(shape.minimum_elements()?).map_err(|_| Error::LengthOverflow)?;
    if values.len() != expected_values {
        return Err(Error::InvalidTensor(format!(
            "EPF sigma plane contains {} values but its tensor requires {expected_values}",
            values.len()
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidTensor(
            "EPF sigma plane contains a non-finite value".into(),
        ));
    }
    let extent = Extent2d::new(shape.width, shape.height);
    let sigma_plane = PlaneDesc {
        id: sigma_id,
        extent,
        stride: shape.width,
        sample_type: SampleType::F32,
        role: PlaneRole::Parameter,
    };
    Ok(Some(EpfReplayResource {
        sigma_plane: Some(sigma_plane),
        update: ResourceUpdate {
            id: resource,
            revision: 0,
            data: ResourceData::Plane(HostPlane {
                id: sigma_id,
                extent,
                stride: shape.width,
                origin: (0, 0),
                data: PlaneData::F32(values),
            }),
        },
    }))
}

fn operation_nodes(
    capture: &CaptureFile,
    inputs: &[PlaneId],
    outputs: &[PlaneId],
    input_shape: &TensorShape,
    output_shape: &TensorShape,
    epf_sigma_plane: Option<PlaneId>,
) -> Result<Vec<RenderNode>> {
    let operation = &capture.metadata.operation;
    let float_contract = PrecisionContract::Float {
        absolute: 2.0e-5,
        relative: 2.0e-5,
        rmse: 2.0e-6,
    };
    match operation.kind {
        OperationKind::Copy => Ok(inputs
            .iter()
            .copied()
            .zip(outputs.iter().copied())
            .enumerate()
            .map(|(channel, (input, output))| RenderNode {
                name: format!("capture-copy-{channel}").into(),
                op: RenderOp::Copy,
                inputs: vec![input],
                outputs: vec![output],
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: PrecisionContract::Exact,
            })
            .collect()),
        OperationKind::Gaborish => {
            let weight1 = operation.parameter("weight1")? as f32;
            let weight2 = operation.parameter("weight2")? as f32;
            let total = 1.0 + 4.0 * weight1 + 4.0 * weight2;
            if !total.is_finite() || total == 0.0 {
                return Err(Error::InvalidMetadata(
                    "invalid Gaborish weight normalization".into(),
                ));
            }
            Ok(inputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
                .enumerate()
                .map(|(channel, (input, output))| RenderNode {
                    name: format!("capture-gaborish-{channel}").into(),
                    op: RenderOp::Gaborish(GaborishParams {
                        channel: u16::try_from(channel).unwrap_or(u16::MAX),
                        weight0: 1.0 / total,
                        weight1: weight1 / total,
                        weight2: weight2 / total,
                    }),
                    inputs: vec![input],
                    outputs: vec![output],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::symmetric(1, 1),
                    precision: float_contract,
                })
                .collect())
        }
        OperationKind::Epf => {
            if inputs.len() != 3
                || outputs.len() != 3
                || input_shape != output_shape
                || input_shape.channels != 3
            {
                return Err(Error::InvalidTensor(
                    "EPF capture requires equal-sized three-channel input and output".into(),
                ));
            }
            let pass = operation.parameters.get("pass").copied().unwrap_or(1.0);
            let (pass, default_sigma_scale, border) = match pass {
                0.0 => (EpfPass::Pass0, 0.9, 3),
                1.0 => (EpfPass::Pass1, 1.0, 2),
                2.0 => (EpfPass::Pass2, 6.5, 1),
                _ => {
                    return Err(Error::InvalidMetadata(format!(
                        "EPF pass must be 0, 1, or 2; found {pass}"
                    )));
                }
            };
            let sigma_scale = operation
                .parameters
                .get("sigma_scale")
                .copied()
                .unwrap_or(default_sigma_scale) as f32;
            let border_sad_mul = operation
                .parameters
                .get("border_sad_mul")
                .copied()
                .unwrap_or(2.3 / 3.0) as f32;
            let channel_scale = [
                operation
                    .parameters
                    .get("channel_scale_x")
                    .copied()
                    .unwrap_or(40.0) as f32,
                operation
                    .parameters
                    .get("channel_scale_y")
                    .copied()
                    .unwrap_or(5.0) as f32,
                operation
                    .parameters
                    .get("channel_scale_b")
                    .copied()
                    .unwrap_or(3.5) as f32,
            ];
            if !sigma_scale.is_finite()
                || !border_sad_mul.is_finite()
                || channel_scale.iter().any(|value| !value.is_finite())
            {
                return Err(Error::InvalidMetadata(
                    "EPF parameters must be finite f32 values".into(),
                ));
            }
            let variable = reference::epf_uses_variable_sigma(operation)?;
            if variable != epf_sigma_plane.is_some() {
                return Err(Error::InvalidMetadata(
                    "EPF sigma representation disagrees with variable_sigma".into(),
                ));
            }
            let resource = ResourceId(0);
            Ok(vec![RenderNode {
                name: format!("capture-epf-pass-{}", pass as u8).into(),
                op: RenderOp::Epf(EpfParams {
                    pass,
                    sigma_scale,
                    border_sad_mul,
                    channel_scale,
                    sigma_resource: Some(resource),
                    sigma_plane: epf_sigma_plane,
                }),
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                resources: vec![resource],
                scale: Scale2d::IDENTITY,
                border: Border2d::symmetric(border, border),
                precision: float_contract,
            }])
        }
        OperationKind::Upsample => {
            let factor = operation.parameter("factor")?;
            if !factor.is_finite() || factor.fract() != 0.0 || !matches!(factor as u8, 2 | 4 | 8) {
                return Err(Error::InvalidMetadata(format!(
                    "unsupported upsample factor {factor}"
                )));
            }
            let factor = factor as u8;
            let mut weights = vec![0.0_f32; usize::from(factor) * usize::from(factor) * 25];
            weights
                .chunks_exact_mut(25)
                .for_each(|phase| phase[12] = 1.0);
            let params = UpsampleParams {
                factor,
                weights: weights.into(),
            };
            Ok(inputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
                .enumerate()
                .map(|(channel, (input, output))| RenderNode {
                    name: format!("capture-upsample-{channel}").into(),
                    op: RenderOp::Upsample(params.clone()),
                    inputs: vec![input],
                    outputs: vec![output],
                    resources: Vec::new(),
                    scale: Scale2d::new(factor, factor),
                    border: Border2d::symmetric(2, 2),
                    precision: float_contract,
                })
                .collect())
        }
        OperationKind::ChromaUpsample => {
            let axis = operation.parameter("axis")?;
            let (axis, scale, border, valid_extent) = match axis {
                0.0 => (
                    ChromaAxis::Horizontal,
                    Scale2d::new(2, 1),
                    Border2d::symmetric(1, 0),
                    output_shape.width.div_ceil(2) == input_shape.width
                        && output_shape.height == input_shape.height,
                ),
                1.0 => (
                    ChromaAxis::Vertical,
                    Scale2d::new(1, 2),
                    Border2d::symmetric(0, 1),
                    output_shape.width == input_shape.width
                        && output_shape.height.div_ceil(2) == input_shape.height,
                ),
                _ => {
                    return Err(Error::InvalidMetadata(format!(
                        "chroma axis must be 0 (horizontal) or 1 (vertical); found {axis}"
                    )));
                }
            };
            if !valid_extent || output_shape.channels != input_shape.channels {
                return Err(Error::InvalidTensor(
                    "chroma upsample output shape does not match its axis".into(),
                ));
            }
            Ok(inputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
                .enumerate()
                .map(|(channel, (input, output))| RenderNode {
                    name: format!("capture-chroma-upsample-{channel}").into(),
                    op: RenderOp::ChromaUpsample { axis },
                    inputs: vec![input],
                    outputs: vec![output],
                    resources: Vec::new(),
                    scale,
                    border,
                    precision: float_contract,
                })
                .collect())
        }
        OperationKind::YcbcrToRgb => {
            if inputs.len() != 3 || outputs.len() != 3 {
                return Err(Error::InvalidTensor(
                    "YCbCr capture must contain exactly three channels".into(),
                ));
            }
            Ok(vec![RenderNode {
                name: "capture-ycbcr-to-rgb".into(),
                op: RenderOp::YcbcrToRgb,
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: float_contract,
            }])
        }
        OperationKind::PremultiplyAlpha => {
            let alpha = operation
                .parameters
                .get("alpha_channel")
                .copied()
                .unwrap_or(f64::from(input_shape.channels - 1));
            if !alpha.is_finite()
                || alpha.fract() != 0.0
                || alpha < 0.0
                || alpha >= inputs.len() as f64
            {
                return Err(Error::InvalidMetadata(format!(
                    "invalid alpha channel {alpha}"
                )));
            }
            Ok(vec![RenderNode {
                name: "capture-premultiply-alpha".into(),
                op: RenderOp::PremultiplyAlpha {
                    alpha_plane: inputs[alpha as usize],
                },
                inputs: inputs.to_vec(),
                outputs: outputs.to_vec(),
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: float_contract,
            }])
        }
        OperationKind::Affine => {
            let multiplier = operation.parameter("scale")? as f32;
            let bias = operation.parameter("bias")? as f32;
            if !multiplier.is_finite() || !bias.is_finite() {
                return Err(Error::InvalidMetadata(
                    "Affine scale and bias must be representable as finite f32 values".into(),
                ));
            }
            Ok(inputs
                .iter()
                .copied()
                .zip(outputs.iter().copied())
                .enumerate()
                .map(|(channel, (input, output))| RenderNode {
                    name: format!("capture-affine-{channel}").into(),
                    op: RenderOp::ModularToF32 { multiplier, bias },
                    inputs: vec![input],
                    outputs: vec![output],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: float_contract,
                })
                .collect())
        }
    }
    .and_then(|nodes| {
        let scale_matches = !matches!(
            operation.kind,
            OperationKind::Upsample | OperationKind::ChromaUpsample
        ) || (output_shape.width >= input_shape.width
            && output_shape.height >= input_shape.height);
        if scale_matches {
            Ok(nodes)
        } else {
            Err(Error::InvalidTensor(
                "operation output shape is smaller than its input".into(),
            ))
        }
    })
}

fn dense_planar_shape<'a>(shape: Option<&'a TensorShape>, name: &str) -> Result<&'a TensorShape> {
    let shape = shape.ok_or_else(|| Error::InvalidTensor(format!("{name} has no tensor shape")))?;
    if shape != &TensorShape::planar(shape.width, shape.height, shape.channels)? {
        return Err(Error::UnsupportedOperation {
            backend: "wgpu",
            operation: format!("non-dense {name} tensor"),
        });
    }
    Ok(shape)
}

fn map_wgpu_error(error: jxl_wgpu::Error) -> Error {
    match error {
        jxl_wgpu::Error::NoAdapter => Error::BackendUnavailable(error.to_string()),
        jxl_wgpu::Error::Unsupported(_) => Error::UnsupportedOperation {
            backend: "wgpu",
            operation: error.to_string(),
        },
        other => Error::Verification(other.to_string()),
    }
}

pub fn create_backend(kind: BackendKind) -> Result<Box<dyn ReplayBackend>> {
    match kind {
        BackendKind::Reference => Ok(Box::<ReferenceBackend>::default()),
        BackendKind::Wgpu => Ok(Box::new(WgpuReplayBackend::request_default()?)),
    }
}

pub fn verify_capture(
    capture: &CaptureFile,
    backend: &mut dyn ReplayBackend,
    threshold: &AccuracyThreshold,
) -> Result<CaseReport> {
    let (expected_descriptor, expected_bytes) = capture.section_by_kind(SectionKind::Expected)?;
    if expected_descriptor.data_type != DataType::F32 {
        return Err(Error::UnsupportedOperation {
            backend: backend.kind().as_str(),
            operation: format!("{:?} comparison", expected_descriptor.data_type),
        });
    }
    let expected = decode_f32(expected_bytes)?;
    let start = Instant::now();
    let actual = backend.execute(capture)?;
    let elapsed = elapsed_ns(start);
    let peak = expected
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| f64::from(value.abs()))
        .fold(1.0_f64, f64::max);
    let metrics = compare_f32(&expected, &actual, peak)?;
    let evaluation = threshold.evaluate_f32(&expected, &actual, &metrics)?;
    let status = if evaluation.passed {
        CaseStatus::Passed
    } else {
        CaseStatus::Failed
    };
    let output_hash = blake3::hash(&encode_f32(&actual)).to_hex().to_string();
    Ok(CaseReport {
        case_id: capture.metadata.case_id.clone(),
        operation: capture.metadata.operation.kind.as_str().into(),
        backend: backend.kind().as_str().into(),
        status,
        output_hash: Some(output_hash),
        metrics: Some(metrics),
        threshold: Some(evaluation),
        timing: Some(TimingStatistics {
            samples: 1,
            minimum_ns: elapsed,
            median_ns: elapsed,
            p95_ns: elapsed,
            mean_ns: elapsed as f64,
            standard_deviation_ns: 0.0,
        }),
        message: None,
    })
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::capture::{OperationKind, PrecisionMode};
    use crate::config::SyntheticCaseConfig;
    use crate::synthetic::generate_case;

    use super::*;

    #[test]
    fn reference_replay_passes_exact_copy() {
        let capture = generate_case(&SyntheticCaseConfig {
            name: "copy".into(),
            operation: OperationKind::Copy,
            width: 17,
            height: 9,
            channels: 3,
            seed: 1,
            precision: PrecisionMode::Exact,
            parameters: BTreeMap::new(),
        })
        .unwrap();
        let mut backend = ReferenceBackend;
        let report = verify_capture(
            &capture,
            &mut backend,
            &AccuracyThreshold {
                require_exact: true,
                ..AccuracyThreshold::default()
            },
        )
        .unwrap();
        assert_eq!(report.status, CaseStatus::Passed);
    }

    #[test]
    fn epf_variable_sigma_resource_is_named_and_bounds_checked_before_gpu_use() {
        let capture = generate_case(&SyntheticCaseConfig {
            name: "epf-resource".into(),
            operation: OperationKind::Epf,
            width: 19,
            height: 11,
            channels: 3,
            seed: 77,
            precision: PrecisionMode::F32,
            parameters: BTreeMap::from([("pass".into(), 0.0), ("variable_sigma".into(), 1.0)]),
        })
        .unwrap();
        let job = WgpuReplayJob::from_capture(&capture).unwrap();
        assert_eq!(job.resources.len(), 1);
        let ResourceData::Plane(sigma) = &job.resources[0].data else {
            panic!("variable EPF sigma must lower to a resource plane");
        };
        assert_eq!(sigma.extent, Extent2d::new(3, 2));

        let mut missing = capture.clone();
        let sigma_id = missing
            .metadata
            .sections
            .iter()
            .find(|section| section.kind == SectionKind::Parameter && section.name == "sigma")
            .unwrap()
            .id;
        missing
            .metadata
            .sections
            .retain(|section| section.id != sigma_id);
        missing.sections.retain(|section| section.id != sigma_id);
        assert!(matches!(
            WgpuReplayJob::from_capture(&missing),
            Err(Error::InvalidMetadata(message)) if message.contains("missing")
        ));

        let mut undersized = capture;
        let descriptor = undersized
            .metadata
            .sections
            .iter_mut()
            .find(|section| section.kind == SectionKind::Parameter && section.name == "sigma")
            .unwrap();
        descriptor.tensor = Some(TensorShape::planar(2, 2, 1).unwrap());
        assert!(matches!(
            WgpuReplayJob::from_capture(&undersized),
            Err(Error::InvalidTensor(_))
        ));
    }

    #[test]
    fn wgpu_replay_executes_every_supported_capture_operation() {
        let mut backend = match WgpuReplayBackend::request_default() {
            Ok(backend) => backend,
            Err(Error::BackendUnavailable(message)) => {
                eprintln!("skipping wgpu integration test: {message}");
                return;
            }
            Err(error) => panic!("wgpu initialization failed: {error}"),
        };
        let mut upsample_parameters = BTreeMap::new();
        upsample_parameters.insert("factor".into(), 2.0);
        let mut chroma_horizontal_parameters = BTreeMap::new();
        chroma_horizontal_parameters.insert("axis".into(), 0.0);
        chroma_horizontal_parameters.insert("output_width".into(), 5.0);
        let mut chroma_vertical_parameters = BTreeMap::new();
        chroma_vertical_parameters.insert("axis".into(), 1.0);
        chroma_vertical_parameters.insert("output_height".into(), 5.0);
        let mut cases = vec![
            (
                OperationKind::Copy,
                7,
                5,
                2,
                PrecisionMode::Exact,
                BTreeMap::new(),
            ),
            (
                OperationKind::Affine,
                9,
                3,
                2,
                PrecisionMode::F32,
                BTreeMap::new(),
            ),
            (
                OperationKind::Gaborish,
                7,
                5,
                2,
                PrecisionMode::F32,
                BTreeMap::new(),
            ),
            (
                OperationKind::Upsample,
                5,
                3,
                1,
                PrecisionMode::F32,
                upsample_parameters,
            ),
            (
                OperationKind::ChromaUpsample,
                3,
                3,
                1,
                PrecisionMode::F32,
                chroma_horizontal_parameters,
            ),
            (
                OperationKind::ChromaUpsample,
                3,
                3,
                1,
                PrecisionMode::F32,
                chroma_vertical_parameters,
            ),
            (
                OperationKind::YcbcrToRgb,
                7,
                5,
                3,
                PrecisionMode::F32,
                BTreeMap::new(),
            ),
            (
                OperationKind::PremultiplyAlpha,
                7,
                5,
                4,
                PrecisionMode::F32,
                BTreeMap::new(),
            ),
        ];
        for pass in 0..=2 {
            for variable_sigma in [0.0, 1.0] {
                let mut parameters = BTreeMap::new();
                parameters.insert("pass".into(), f64::from(pass));
                parameters.insert("variable_sigma".into(), variable_sigma);
                cases.push((
                    OperationKind::Epf,
                    19,
                    11,
                    3,
                    PrecisionMode::F32,
                    parameters,
                ));
            }
        }
        for (index, (operation, width, height, channels, precision, parameters)) in
            cases.into_iter().enumerate()
        {
            let capture = generate_case(&SyntheticCaseConfig {
                name: format!("wgpu-{}", operation.as_str()),
                operation,
                width,
                height,
                channels,
                seed: u64::try_from(index).unwrap_or(u64::MAX),
                precision,
                parameters,
            })
            .unwrap();
            let threshold = if capture.metadata.operation.kind == OperationKind::Copy {
                AccuracyThreshold {
                    require_exact: true,
                    ..AccuracyThreshold::default()
                }
            } else {
                AccuracyThreshold::default()
            };
            let report =
                verify_capture(&capture, &mut backend, &threshold).unwrap_or_else(|error| {
                    panic!(
                        "{} must be implemented by wgpu replay: {error}",
                        capture.metadata.operation.kind.as_str()
                    )
                });
            assert_eq!(
                report.status,
                CaseStatus::Passed,
                "{} failed accuracy thresholds: {:?}",
                capture.metadata.operation.kind.as_str(),
                report.threshold
            );
        }
    }
}
