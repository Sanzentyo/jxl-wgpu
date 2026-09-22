//! GPU dispatch, artifact validation, and VarDCT encoder handles.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_wgpu::{KernelVariant, MemoryPermit};
use wgpu::util::DeviceExt;

use super::ac::{AcFragments, validate_blocks, validate_transform_fragments};
use super::bitstream::{build_frame_packet, image_header, pack_signed_control};
use super::entropy::{
    HfEntropyPlan, fixed_prefix_code, prefix_entries, read_fragment_slice,
    validate_fragment_padding,
};
use super::entropy::{UINT_SYMBOLS, VarDctPrefixCode};
use super::strategy_map::{TransformPlan, VarDctStrategyMap, VarDctTransform};
use super::types::{
    ARTIFACT_READY, ArtifactLayout, DcFragmentDescriptor, HEADER_WORDS, HF_QUANTIZATION,
    TiledVarDctGrid, VarDctArtifactData, VarDctArtifactHeader, VarDctColorEncoding,
    VarDctFrameLayout, VarDctKernelParams, VarDctLfMetadata, VarDctMemoryPlan, VarDctStrategy,
    VarDctTopology,
};
use super::{raw_matrices, saliency, transforms};
use crate::{
    AnimationHeader, BackendError, BitFragment, BufferImageSource, Determinism, EncodeError,
    EncodeProfile, EncoderCapabilities, FrameEncodeRequest, FrameIndex, FrameOptions,
    FrameSubmission, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameArtifacts, GpuFrameSource,
    KernelStage, ProfileCapability, ProgressivePlan, UnsupportedFeature, VarDctConfig,
    VarDctQuantization, WgpuContext, assemble_frame,
};

pub(super) const TILED_SHADER: &str = include_str!("tiled.wgsl");

pub(super) fn shader_source(entry_points: &str) -> String {
    format!(
        "{}\n{}\n{entry_points}",
        include_str!("common.wgsl"),
        include_str!("control.wgsl")
    )
}
pub(super) const FORWARD_KERNEL_KEY: &str = "vardct_encode_forward";
pub(super) const TILED_KERNEL_KEY: &str = "vardct_encode_quantize";
pub(super) const TILED_WORKGROUP_STORAGE_BYTES: u32 = 2 * 64 * 16 + 4;

#[derive(Clone, Copy, Debug)]
pub(super) struct VarDctDispatchPlan {
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    kernel: VarDctKernelPlan,
    memory: VarDctMemoryPlan,
    frame: VarDctFrameLayout,
}

#[derive(Clone, Copy, Debug)]
struct VarDctKernelPlan {
    params: VarDctKernelParams,
    layout: ArtifactLayout,
}

enum VarDctPipelines {
    Transforms(transforms::Pipeline),
    Tiled {
        quantize: Arc<wgpu::ComputePipeline>,
        serialize: Arc<wgpu::ComputePipeline>,
    },
}

/// GPU backend for standard VarDCT stills with fixed or mapped transform placement.
///
/// Sources match the selected transform or map extent; the optimized tiled-DCT8
/// constructor selects geometry at submission time. Pixels and coefficients remain
/// on the GPU until it has packed their entropy fragments.
pub struct VarDctBackend {
    pipelines: VarDctPipelines,
    workgroup_variant: KernelVariant,
    code: VarDctPrefixCode,
    hf_entropy: HfEntropyPlan,
    topology: VarDctTopology,
    transform_plan: Option<Arc<TransformPlan>>,
    tiled_metadata: Option<Vec<[u32; 6]>>,
    raw_matrix_plan: Option<Arc<raw_matrices::Plan>>,
    raw_matrix_pipeline: Option<raw_matrices::Pipeline>,
    saliency_pipeline: Option<saliency::Pipeline>,
    config: VarDctConfig,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
    storage_offset_alignment: u64,
}

impl VarDctBackend {
    /// Creates a standard VarDCT strategy backend and its compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// represented by the JPEG XL prefix-code writer.
    pub fn new(context: &WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        Self::new_with_config(context, strategy, VarDctConfig::default())
    }

    /// Creates a standard VarDCT strategy backend with explicit quantization and LF metadata.
    pub fn new_with_config(
        context: &WgpuContext,
        strategy: VarDctStrategy,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        Self::new_with_topology(context, VarDctTopology::SingleTransform(strategy), config)
    }

    /// Creates the bounded tiled-DCT8 profile used by [`TiledVarDctEncoder`].
    /// Every padded 8x8 block is an independent regular transform. The source
    /// extent selects the checked block, LF-group, and AC-group grids at
    /// submission time.
    pub fn new_tiled_dct8(context: &WgpuContext) -> Result<Self, EncodeError> {
        Self::new_tiled_dct8_with_config(context, VarDctConfig::default())
    }

    /// Creates the tiled DCT8 backend with explicit quantization and LF metadata.
    pub fn new_tiled_dct8_with_config(
        context: &WgpuContext,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        Self::new_with_topology(context, VarDctTopology::TiledDct8, config)
    }

    /// Creates an encoder for a validated image-wide transform map.
    pub fn new_with_strategy_map(
        context: &WgpuContext,
        map: VarDctStrategyMap,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        let plan = TransformPlan::new(map, &config)?;
        let mut backend = Self::new_with_topology(context, VarDctTopology::StrategyMap, config)?;
        backend.transform_plan = Some(Arc::new(plan));
        Ok(backend)
    }

    fn new_with_topology(
        context: &WgpuContext,
        topology: VarDctTopology,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        let code = fixed_prefix_code()?;
        let hf_entropy = HfEntropyPlan::single_cluster_prefix()?;
        let limits = context.device().limits();
        let raw_matrix_plan =
            raw_matrices::Plan::new(&config.dequant_matrices, &code)?.map(Arc::new);
        if let Some(plan) = &raw_matrix_plan {
            plan.validate_limits(&limits)?;
        }
        for (name, available) in [
            (
                "max_storage_buffer_binding_size",
                limits.max_storage_buffer_binding_size,
            ),
            ("max_buffer_size", limits.max_buffer_size),
        ] {
            let required = if topology == VarDctTopology::TiledDct8 {
                super::types::TILED_QUANTIZATION_BYTES
            } else {
                std::mem::size_of::<VarDctKernelParams>() as u64
            };
            if required > available {
                return Err(UnsupportedFeature::DeviceLimit {
                    name,
                    required,
                    available,
                }
                .into());
            }
        }
        let tiled_metadata = (topology == VarDctTopology::TiledDct8)
            .then(|| {
                config
                    .dequant_matrices
                    .metadata(VarDctStrategy::Dct8, &config.coefficient_orders)
            })
            .transpose()?;
        let (kernel_key, default_variant, workgroup_storage_bytes) =
            if topology == VarDctTopology::TiledDct8 {
                (
                    TILED_KERNEL_KEY,
                    KernelVariant::Lanes64,
                    TILED_WORKGROUP_STORAGE_BYTES,
                )
            } else {
                (FORWARD_KERNEL_KEY, KernelVariant::Lanes64, 4)
            };
        let workgroup_variant = context
            .kernel_policy()
            .variant_for(kernel_key, default_variant)?;
        workgroup_variant.validate_for(kernel_key, &limits, workgroup_storage_bytes)?;
        let (workgroup_x, _) = workgroup_variant.workgroup_size();
        let workgroup_constants = [("wg_x", f64::from(workgroup_x))];
        let pipelines = if !matches!(topology, VarDctTopology::TiledDct8) {
            VarDctPipelines::Transforms(transforms::Pipeline::new(
                context.device(),
                workgroup_variant,
            )?)
        } else {
            validate_tiled_device_limits(&limits)?;
            let module = context
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("jxl-wgpu tiled VarDCT kernel"),
                    source: wgpu::ShaderSource::Wgsl(shader_source(TILED_SHADER).into()),
                });
            VarDctPipelines::Tiled {
                quantize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu tiled VarDCT block quantization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("quantize_blocks"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &workgroup_constants,
                            ..Default::default()
                        },
                        cache: None,
                    },
                )),
                serialize: Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu tiled VarDCT control serialization"),
                        layout: None,
                        module: &module,
                        entry_point: Some("serialize_control"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    },
                )),
            }
        };
        let transform_plan = if let VarDctTopology::SingleTransform(strategy) = topology {
            let extent = strategy.pixel_extent();
            Some(Arc::new(TransformPlan::new(
                VarDctStrategyMap::new(
                    extent.width,
                    extent.height,
                    vec![VarDctTransform {
                        block_x: 0,
                        block_y: 0,
                        strategy,
                        hf_multiplier: None,
                    }],
                )?,
                &config,
            )?))
        } else {
            None
        };
        let saliency_pipeline = config
            .group_order
            .requires_saliency()
            .then(|| saliency::Pipeline::new(context.device(), workgroup_variant))
            .transpose()?;
        let mut implemented_stages = vec![
            KernelStage::InputNormalization,
            KernelStage::ColorTransform,
            KernelStage::ForwardTransform,
            KernelStage::Quantization,
            KernelStage::CoefficientTokenization,
            KernelStage::HistogramReduction,
        ];
        if saliency_pipeline.is_some() {
            implemented_stages.push(KernelStage::GroupOrderSelection);
        }
        Ok(Self {
            saliency_pipeline,
            raw_matrix_pipeline: raw_matrix_plan
                .as_ref()
                .map(|_| raw_matrices::Pipeline::new(context.device())),
            raw_matrix_plan,
            pipelines,
            workgroup_variant,
            code,
            hf_entropy,
            topology,
            transform_plan,
            tiled_metadata,
            config: config.clone(),
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::VarDct {
                    quantization: config.quantization,
                }],
                max_progressive_passes: ProgressivePlan::MAX_PASSES as u8,
                animation: false,
                determinism: Determinism::SameDevice,
                implemented_stages,
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        })
    }

    /// Selected linear workgroup for the parallel forward/quantization pass.
    ///
    /// The control serializer remains a separate fixed scalar pass because its DC
    /// prediction and bit-offset state are sequential.
    #[must_use]
    pub const fn workgroup_variant(&self) -> KernelVariant {
        self.workgroup_variant
    }

    #[must_use]
    pub const fn lf_metadata(&self) -> VarDctLfMetadata {
        self.config.lf_metadata
    }

    /// Computes the exact memory admission and source binding before a job is
    /// submitted.
    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    fn dispatch_plan(&self, source: &BufferImageSource) -> Result<VarDctDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let frame = match self.topology {
            VarDctTopology::SingleTransform(strategy) => {
                let frame = VarDctFrameLayout::single(strategy);
                if extent.width != frame.width || extent.height != frame.height {
                    return Err(EncodeError::InvalidSource(
                        "the VarDCT source extent must equal the selected transform extent",
                    ));
                }
                frame
            }
            VarDctTopology::StrategyMap => {
                let plan = self
                    .transform_plan
                    .as_ref()
                    .expect("strategy map backend plan");
                if extent != plan.map.extent() {
                    return Err(EncodeError::InvalidSource(
                        "VarDCT source extent differs from strategy map",
                    ));
                }
                plan.map.frame()?
            }
            VarDctTopology::TiledDct8 => {
                VarDctFrameLayout::tiled_dct8(extent.width, extent.height)?
            }
        };
        self.config.group_order.validate(frame)?;
        if source.layout.format != VarDctColorEncoding::SrgbD65.pixel_format()
            || source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing VarDCT RGB plane"))?;
        let row_bytes = u64::from(extent.width) * 3;
        if plane.row_bytes != row_bytes || plane.row_stride < row_bytes {
            return Err(EncodeError::InvalidSource(
                "the VarDCT RGB plane has an invalid row layout",
            ));
        }
        let row_stride = u32::try_from(plane.row_stride)
            .map_err(|_| EncodeError::InvalidSource("VarDCT row stride exceeds WGSL u32"))?;
        let sample_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .and_then(|rows| plane.offset.checked_add(rows))
            .and_then(|offset| offset.checked_add(row_bytes))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4).ok_or(EncodeError::InvalidSource(
            "VarDCT source binding size overflow",
        ))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "VarDCT source binding does not contain the final sample word",
            ));
        }
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        let source_binding_bytes =
            binding_end
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source binding range underflow",
                ))?;
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes).ok_or(
            EncodeError::InvalidSource("VarDCT source binding must not be empty"),
        )?;
        let relative_offset =
            plane
                .offset
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource(
                    "VarDCT source address arithmetic underflow",
                ))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|end| end.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "VarDCT source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source address exceeds the WGSL u32 space")
        })?;
        let byte_offset = u32::try_from(relative_offset).map_err(|_| {
            EncodeError::InvalidSource("VarDCT source offset exceeds the WGSL u32 space")
        })?;
        let blocks_x = frame.blocks_x;
        let blocks_y = frame.blocks_y;
        let (lf_quantization, lf_correlation) = self.config.lf_metadata.forward_quantization();
        let hf_correlation = self.config.lf_metadata.hf_correlation();
        let common_strategy = frame.topology.strategy_id();
        let (kernel, mut memory) = {
            let mut layout = match frame.topology {
                VarDctTopology::StrategyMap => self
                    .transform_plan
                    .as_ref()
                    .expect("strategy map backend plan")
                    .artifact_layout(frame, &self.code)?,
                VarDctTopology::SingleTransform(strategy) => {
                    ArtifactLayout::new(strategy, &self.code)?
                }
                VarDctTopology::TiledDct8 => {
                    ArtifactLayout::for_tiled_grid(frame, &self.code, &self.hf_entropy)?
                }
            }
            .with_passes(self.config.progressive.passes().len())?;
            if self.config.group_order.requires_saliency() {
                layout = layout.with_saliency(frame.ac_group_count()?)?;
            }
            let mut required_workgroup_axis = match frame.topology {
                VarDctTopology::TiledDct8 => blocks_x.max(blocks_y),
                VarDctTopology::SingleTransform(_) | VarDctTopology::StrategyMap => {
                    let groups = (frame.blocks_x * frame.blocks_y * 64)
                        .div_ceil(self.workgroup_variant.workgroup_size().0);
                    let columns = groups.min(self.max_compute_workgroups_per_dimension);
                    columns.max(groups.div_ceil(columns.max(1)))
                }
            };
            if self.config.group_order.requires_saliency() {
                required_workgroup_axis = required_workgroup_axis
                    .max(frame.ac_groups_x)
                    .max(frame.ac_groups_y);
            }
            if required_workgroup_axis > self.max_compute_workgroups_per_dimension {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_compute_workgroups_per_dimension",
                    required: u64::from(required_workgroup_axis),
                    available: u64::from(self.max_compute_workgroups_per_dimension),
                }
                .into());
            }
            let artifact_bytes = layout.artifact_bytes();
            if artifact_bytes > self.max_storage_binding_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_storage_buffer_binding_size",
                    required: artifact_bytes,
                    available: self.max_storage_binding_size,
                }
                .into());
            }
            if artifact_bytes > self.max_buffer_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_buffer_size",
                    required: artifact_bytes,
                    available: self.max_buffer_size,
                }
                .into());
            }
            (
                VarDctKernelPlan {
                    params: VarDctKernelParams {
                        row_stride,
                        byte_offset,
                        width: extent.width,
                        height: extent.height,
                        blocks_x,
                        blocks_y,
                        strategy: common_strategy,
                        global_scale: self.config.quantization.global_scale(),
                        quant_lf: self.config.quantization.quant_lf(),
                        hf_multiplier: self.config.quantization.hf_multiplier().get(),
                        raw_prefix: prefix_entries(&self.code),
                        strategy_offset: layout.strategy_offset,
                        dc_offset: layout.dc_offset,
                        token_offset: layout.token_offset,
                        extra_offset: layout.extra_offset,
                        fragment_offset: layout.fragment_offset,
                        fragment_word_capacity: layout.fragment_word_capacity,
                        artifact_words: layout.artifact_words,
                        topology: frame.topology.artifact_id(),
                        fragment_descriptor_offset: layout.fragment_descriptor_offset,
                        fragment_descriptor_len: layout.fragment_descriptor_len,
                        lf_groups_x: frame.lf_groups_x,
                        lf_groups_y: frame.lf_groups_y,
                        lf_quantization,
                        lf_correlation,
                        hf_prefix: self.hf_entropy.gpu_entries(),
                        hf_correlation,
                        hf_quantization: HF_QUANTIZATION,
                        ac_descriptor_offset: layout.ac_descriptor_offset,
                        ac_descriptor_len: layout.ac_descriptor_len,
                        ac_fragment_offset: layout.ac_fragment_offset,
                        ac_words_per_block: layout.ac_words_per_block,
                        ac_fragment_words: layout.ac_fragment_words,
                        workgroups_x: (frame.blocks_x * frame.blocks_y * 64)
                            .div_ceil(self.workgroup_variant.workgroup_size().0)
                            .min(self.max_compute_workgroups_per_dimension),
                        ac_pass_count: layout.ac_pass_count,
                        ac_pass_words: layout.ac_fragment_words / layout.ac_pass_count,
                        progressive: std::array::from_fn(|index| {
                            self.config
                                .progressive
                                .passes()
                                .get(index)
                                .map_or(0, |pass| {
                                    u32::from(pass.coefficient_square.get())
                                        | u32::from(pass.shift) << 8
                                })
                        }),
                        saliency_offset: layout.saliency_offset,
                        saliency_groups: layout.saliency_groups,
                        padding: [0; 7],
                    },
                    layout,
                },
                VarDctMemoryPlan::new(
                    source_binding_bytes,
                    artifact_bytes,
                    frame.topology.kernel_layout(),
                ),
            )
        };
        if kernel.layout.saliency_groups != 0 {
            memory.saliency_metadata_bytes =
                u64::from(kernel.layout.artifact_words - kernel.layout.saliency_offset) * 4;
        }
        if let Some(raw) = &self.raw_matrix_plan {
            memory.raw_matrix_input_bytes = raw.input_bytes();
            memory.raw_matrix_artifact_bytes = raw.artifact_bytes();
            memory.readback_bytes = memory
                .readback_bytes
                .checked_add(raw.artifact_bytes())
                .ok_or(EncodeError::InvalidConfiguration(
                    "raw matrix readback size overflow",
                ))?;
            if memory.readback_bytes > self.max_buffer_size {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_buffer_size",
                    required: memory.readback_bytes,
                    available: self.max_buffer_size,
                }
                .into());
            }
            let extra = raw.input_bytes() + 2 * raw.artifact_bytes();
            memory.owned_bytes_per_job = memory.owned_bytes_per_job.checked_add(extra).ok_or(
                EncodeError::InvalidConfiguration("raw matrix ownership size overflow"),
            )?;
            memory.addressed_bytes_per_job =
                memory.addressed_bytes_per_job.checked_add(extra).ok_or(
                    EncodeError::InvalidConfiguration("raw matrix addressed size overflow"),
                )?;
        }
        if let Some(plan) = &self.transform_plan {
            let transform = plan.memory;
            memory.owned_bytes_per_job += transform.total_bytes;
            memory.addressed_bytes_per_job += transform.total_bytes;
            memory.transform = Some(transform);
            let sizes = [
                transform.xyb_bytes,
                transform.coefficient_bytes,
                transform.lf_bytes,
                transform.quantized_bytes,
                transform.quantization_metadata_bytes,
                transform.task_metadata_bytes,
            ];
            let batch_sizes = plan.batches.iter().flat_map(|batch| {
                [
                    batch.memory.basis_bytes,
                    batch.memory.horizontal_bytes,
                    batch.memory.task_bytes,
                ]
            });
            for required in sizes.into_iter().chain(batch_sizes) {
                for (name, available) in [
                    ("max_buffer_size", self.max_buffer_size),
                    (
                        "max_storage_buffer_binding_size",
                        self.max_storage_binding_size,
                    ),
                ] {
                    if required > available {
                        return Err(UnsupportedFeature::DeviceLimit {
                            name,
                            required,
                            available,
                        }
                        .into());
                    }
                }
            }
        }
        Ok(VarDctDispatchPlan {
            source_binding_offset,
            source_binding_size,
            kernel,
            memory,
            frame,
        })
    }
}

pub(super) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn validate_tiled_device_limits(limits: &wgpu::Limits) -> Result<(), EncodeError> {
    let checks = [(
        "max_storage_buffers_per_shader_stage",
        4,
        u64::from(limits.max_storage_buffers_per_shader_stage),
    )];
    if let Some((name, required, available)) = checks
        .into_iter()
        .find(|(_, required, available)| required > available)
    {
        return Err(UnsupportedFeature::DeviceLimit {
            name,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

fn validate_vardct_request(
    request: &FrameEncodeRequest,
    frame: VarDctFrameLayout,
    config: &VarDctConfig,
) -> Result<(), EncodeError> {
    if request.frame_index != FrameIndex::new(0)
        || !request.is_last
        || request.animation != AnimationHeader::Still
        || request.canvas_width != frame.width
        || request.canvas_height != frame.height
        || request.options != FrameOptions::default()
    {
        return Err(EncodeError::InvalidConfiguration(
            "the VarDCT profile requires one full-canvas final transform-sized still frame",
        ));
    }
    if request.progressive != config.progressive {
        return Err(EncodeError::InvalidConfiguration(
            "the requested VarDCT passes do not match the backend configuration",
        ));
    }
    if request.profile
        != (EncodeProfile::VarDct {
            quantization: config.quantization,
        })
    {
        return Err(EncodeError::InvalidConfiguration(
            "the requested VarDCT quantization does not match the backend configuration",
        ));
    }
    Ok(())
}

impl GpuEncodeBackend for VarDctBackend {
    type Job = VarDctJob;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(source) = source else {
            return false;
        };
        self.dispatch_plan(source).is_ok()
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        validate_vardct_request(request, plan.frame, &self.config)?;
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let parameters = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT parameters"),
            size: plan.memory.parameter_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let artifact = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT artifact"),
            size: plan.memory.artifact_storage_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let readback = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu VarDCT readback"),
            size: plan.memory.readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        context
            .queue()
            .write_buffer(&parameters, 0, bytemuck::bytes_of(&plan.kernel.params));

        let tiled_quantization = self.tiled_metadata.as_ref().map(|metadata| {
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tiled VarDCT matrices and coefficient orders"),
                    contents: bytemuck::cast_slice(metadata),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        });

        let source_binding = wgpu::BufferBinding {
            buffer: &source.buffer,
            offset: plan.source_binding_offset,
            size: Some(plan.source_binding_size),
        };
        let params_binding_size = NonZeroU64::new(plan.memory.parameter_storage_bytes)
            .expect("the VarDCT parameter ABI is non-empty");
        let artifact_binding_size = NonZeroU64::new(plan.memory.artifact_storage_bytes)
            .expect("the VarDCT artifact ABI is non-empty");
        let create_bind_group = |pipeline: &wgpu::ComputePipeline, label| {
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(source_binding.clone()),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &parameters,
                                offset: 0,
                                size: Some(params_binding_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &artifact,
                                offset: 0,
                                size: Some(artifact_binding_size),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: tiled_quantization
                                .as_ref()
                                .expect("tiled quantization metadata")
                                .as_entire_binding(),
                        },
                    ],
                })
        };
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu VarDCT encode"),
                });
        commands.clear_buffer(&artifact, 0, None);
        let mut transform_scratch = None;
        let job_layout = match (&self.pipelines, plan.kernel) {
            (VarDctPipelines::Transforms(pipeline), VarDctKernelPlan { layout, .. }) => {
                transform_scratch = Some(
                    pipeline.encode(
                        context.device(),
                        &mut commands,
                        transforms::Inputs {
                            plan: self
                                .transform_plan
                                .as_ref()
                                .expect("general transform plan"),
                            source: source_binding.clone(),
                            parameters: &parameters,
                            artifact: &artifact,
                        },
                    )?,
                );
                layout
            }
            (
                VarDctPipelines::Tiled {
                    quantize,
                    serialize,
                },
                VarDctKernelPlan { params, layout },
            ) => {
                let quantize_bind_group =
                    create_bind_group(quantize, "jxl-wgpu tiled VarDCT quantization bindings");
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu tiled VarDCT block transform and entropy"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(quantize);
                    pass.set_bind_group(0, &quantize_bind_group, &[]);
                    pass.dispatch_workgroups(params.blocks_x, params.blocks_y, 1);
                }
                // A separate WebGPU pass is the explicit global storage
                // visibility boundary for all block workgroups before the
                // single deterministic prediction/serialization invocation.
                // The control entry point intentionally has no source binding;
                // automatic pipeline layouts therefore retain only bindings
                // 1 and 2 for this pass.
                let serialize_bind_group =
                    context
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("jxl-wgpu tiled VarDCT serialization bindings"),
                            layout: &serialize.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &parameters,
                                        offset: 0,
                                        size: Some(params_binding_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &artifact,
                                        offset: 0,
                                        size: Some(artifact_binding_size),
                                    }),
                                },
                            ],
                        });
                {
                    let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("jxl-wgpu tiled VarDCT control and entropy serialization"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(serialize);
                    pass.set_bind_group(0, &serialize_bind_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                layout
            }
        };
        if let Some(pipeline) = &self.saliency_pipeline {
            pipeline.encode(
                context.device(),
                &mut commands,
                source_binding,
                &parameters,
                &artifact,
                plan.frame,
            );
        }
        commands.copy_buffer_to_buffer(
            &artifact,
            0,
            &readback,
            0,
            plan.memory.artifact_storage_bytes,
        );

        let raw_scratch = self
            .raw_matrix_plan
            .as_ref()
            .zip(self.raw_matrix_pipeline.as_ref())
            .map(|(raw, pipeline)| {
                pipeline.encode(
                    context.device(),
                    &mut commands,
                    raw,
                    &readback,
                    plan.memory.artifact_storage_bytes,
                )
            });
        let completion = Arc::new(VarDctMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let readback_for_map = Arc::clone(&readback);
        let lifetime = Arc::new(VarDctJobLifetime {
            _parameters: parameters,
            _artifact: artifact,
            _transform: transform_scratch,
            _tiled_quantization: tiled_quantization,
            _raw_matrices: raw_scratch,
            readback,
            _memory_permit: memory_permit,
            mapped: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(
            &readback_for_map,
            wgpu::MapMode::Read,
            0..plan.memory.readback_bytes,
            move |result| {
                if result.is_ok() {
                    callback_lifetime.mapped.store(true, Ordering::Release);
                }
                callback_completion.complete(result.map_err(BackendError::ArtifactMapping));
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(BackendError::PollWorker(error)));
        }) {
            completion.complete(Err(BackendError::PollRegistration(error)));
        }

        Ok(VarDctJob {
            lifetime: Some(lifetime),
            completion,
            code: self.code.clone(),
            hf_entropy: self.hf_entropy.clone(),
            config: self.config.clone(),
            frame_layout: plan.frame,
            transform_plan: self.transform_plan.clone(),
            raw_matrix_plan: self.raw_matrix_plan.clone(),
            artifact_layout: job_layout,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
}

#[derive(Default)]
struct VarDctMapCompletion {
    state: Mutex<VarDctMapState>,
    condition: Condvar,
}

#[derive(Default)]
struct VarDctMapState {
    result: Option<Result<(), BackendError>>,
    waker: Option<Waker>,
}

impl VarDctMapCompletion {
    fn complete(&self, result: Result<(), BackendError>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), BackendError>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), BackendError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("VarDCT map completion was checked as present")
    }
}

struct VarDctJobLifetime {
    _raw_matrices: Option<raw_matrices::Scratch>,
    _transform: Option<transforms::Scratch>,
    _tiled_quantization: Option<wgpu::Buffer>,
    _parameters: Arc<wgpu::Buffer>,
    _artifact: Arc<wgpu::Buffer>,
    readback: Arc<wgpu::Buffer>,
    _memory_permit: MemoryPermit,
    mapped: AtomicBool,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.readback.unmap();
        }
    }
}

pub struct VarDctJob {
    raw_matrix_plan: Option<Arc<raw_matrices::Plan>>,
    lifetime: Option<Arc<VarDctJobLifetime>>,
    completion: Arc<VarDctMapCompletion>,
    code: VarDctPrefixCode,
    hf_entropy: HfEntropyPlan,
    config: VarDctConfig,
    frame_layout: VarDctFrameLayout,
    transform_plan: Option<Arc<TransformPlan>>,
    artifact_layout: ArtifactLayout,
    frame_index: FrameIndex,
    is_last: bool,
}

impl VarDctJob {
    #[cfg(test)]
    pub(super) fn wait_with_saliency_for_test(
        mut self,
    ) -> Result<(Vec<saliency::Record>, GpuFrameArtifacts), EncodeError> {
        self.completion.wait()?;
        let lifetime = self.lifetime.as_ref().expect("unconsumed test job");
        let mapped = lifetime
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(BackendError::ArtifactRange)?;
        let artifact = validate_artifact(
            &mapped[..self.artifact_layout.artifact_bytes() as usize],
            self.artifact_layout,
            &self.code,
            &self.hf_entropy,
            self.frame_layout,
            self.transform_plan.as_deref(),
        )?;
        let records = artifact
            .saliency
            .ok_or(BackendError::Invariant("test requires saliency"))?
            .to_vec();
        drop(mapped);
        let artifacts = self.finish(Ok(()))?;
        Ok((records, artifacts))
    }

    /// Completes once, retaining all GPU-compressed fragments for independent test oracles.
    #[cfg(test)]
    pub(super) fn wait_with_ac_fragments_for_test(
        mut self,
    ) -> Result<(Vec<u32>, Vec<u32>, GpuFrameArtifacts), EncodeError> {
        self.completion.wait()?;
        let lifetime = self.lifetime.as_ref().expect("unconsumed test job");
        let mapped = lifetime
            .readback
            .slice(..)
            .get_mapped_range()
            .map_err(BackendError::ArtifactRange)?;
        validate_artifact(
            &mapped[..self.artifact_layout.artifact_bytes() as usize],
            self.artifact_layout,
            &self.code,
            &self.hf_entropy,
            self.frame_layout,
            self.transform_plan.as_deref(),
        )?;
        let all_words = bytemuck::cast_slice::<u8, u32>(&mapped);
        let layout = self.artifact_layout;
        let words = artifact_words(
            all_words,
            layout.ac_fragment_offset,
            layout.ac_fragment_words,
        )?
        .to_vec();
        let lengths = artifact_words(
            all_words,
            layout.ac_descriptor_offset,
            layout.ac_descriptor_len * layout.ac_pass_count,
        )?
        .to_vec();
        drop(mapped);
        let artifacts = self.finish(Ok(()))?;
        Ok((words, lengths, artifacts))
    }

    #[cfg(test)]
    pub(super) fn wait_with_ac_for_test(
        self,
    ) -> Result<(Vec<u32>, u32, GpuFrameArtifacts), EncodeError> {
        let (words, lengths, artifacts) = self.wait_with_ac_fragments_for_test()?;
        let [length] = lengths.as_slice() else {
            return Err(BackendError::Invariant("test requires one transform").into());
        };
        Ok((words, *length, artifacts))
    }

    fn finish(
        &mut self,
        mapping: Result<(), BackendError>,
    ) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self.lifetime.take().ok_or(BackendError::Invariant(
            "VarDCT GPU job was already consumed",
        ))?;
        mapping?;
        let mapped = match lifetime.readback.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                lifetime.readback.unmap();
                lifetime.mapped.store(false, Ordering::Release);
                return Err(BackendError::ArtifactRange(error).into());
            }
        };
        let result = (|| {
            let boundary = self.artifact_layout.artifact_bytes() as usize;
            let mut artifact = validate_artifact(
                &mapped[..boundary],
                self.artifact_layout,
                &self.code,
                &self.hf_entropy,
                self.frame_layout,
                self.transform_plan.as_deref(),
            )?;
            if let Some(raw) = &self.raw_matrix_plan {
                artifact.raw_matrices = raw.validate(&mapped[boundary..])?;
            } else if mapped.len() != boundary {
                return Err(BackendError::InvalidArtifact("unexpected raw matrix artifact").into());
            }
            Ok(GpuFrameArtifacts {
                frame_index: self.frame_index,
                is_last: self.is_last,
                packets: build_frame_packet(
                    artifact,
                    &self.code,
                    &self.hf_entropy,
                    self.frame_layout,
                    &self.config,
                )?,
                acceleration: None,
            })
        })();
        drop(mapped);
        lifetime.readback.unmap();
        lifetime.mapped.store(false, Ordering::Release);
        drop(lifetime);
        result
    }
}

impl GpuEncodeJob for VarDctJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(BackendError::Invariant(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission",
            )
            .into())
        }
    }
}

pub(super) fn validate_artifact<'a>(
    mapped: &'a [u8],
    layout: ArtifactLayout,
    code: &VarDctPrefixCode,
    hf_entropy: &HfEntropyPlan,
    frame: VarDctFrameLayout,
    transform_plan: Option<&'a TransformPlan>,
) -> Result<VarDctArtifactData<'a>, BackendError> {
    let expected_bytes = usize::try_from(layout.artifact_bytes())
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact size does not fit usize"))?;
    if mapped.len() != expected_bytes {
        return Err(BackendError::InvalidArtifact(
            "VarDCT mapped artifact has the wrong byte length",
        ));
    }
    let words = bytemuck::try_cast_slice::<u8, u32>(mapped)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact word ABI alignment"))?;
    let header_bytes = mapped
        .get(..std::mem::size_of::<VarDctArtifactHeader>())
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT artifact header is truncated",
        ))?;
    let header = bytemuck::try_from_bytes::<VarDctArtifactHeader>(header_bytes)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT header ABI alignment"))?;
    let blocks_x = frame.blocks_x;
    let blocks_y = frame.blocks_y;
    let block_count = blocks_x
        .checked_mul(blocks_y)
        .ok_or(BackendError::InvalidArtifact("VarDCT block count overflow"))?;
    let dc_sample_count = block_count
        .checked_mul(3)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT sample count overflow",
        ))?;
    let strategy_id = frame.topology.strategy_id();
    let lf_group_count = frame
        .lf_group_count()
        .map_err(|_| BackendError::InvalidArtifact("VarDCT LF group count overflow"))?;
    if matches!(header.status, 0x4000_0000 | 0x8000_0000 | 0xc000_0000) {
        return Err(BackendError::VarDctQuantizationOverflow {
            low_frequency: header.status & 0x4000_0000 != 0,
            high_frequency: header.status & 0x8000_0000 != 0,
        });
    }
    if header.status != ARTIFACT_READY
        || header.block_count != block_count
        || header.dc_sample_count != dc_sample_count
        || header.strategy != strategy_id
        || header.ac_payload != u32::from(layout.ac_descriptor_len != 0)
        || header.strategy_offset != layout.strategy_offset
        || header.strategy_len != layout.strategy_len
        || header.dc_offset != layout.dc_offset
        || header.dc_len != layout.dc_len
        || header.token_offset != layout.token_offset
        || header.token_len != layout.token_len
        || header.extra_offset != layout.extra_offset
        || header.extra_len != layout.extra_len
        || header.fragment_offset != layout.fragment_offset
        || header.fragment_word_capacity != layout.fragment_word_capacity
        || header.artifact_words != layout.artifact_words
        || header.width != frame.width
        || header.height != frame.height
        || header.blocks_x != blocks_x
        || header.blocks_y != blocks_y
        || header.topology != frame.topology.artifact_id()
        || header.fragment_descriptor_offset != layout.fragment_descriptor_offset
        || header.fragment_descriptor_len != layout.fragment_descriptor_len
        || header.lf_groups_x != frame.lf_groups_x
        || header.lf_groups_y != frame.lf_groups_y
        || header.lf_group_count != lf_group_count
        || header.ac_descriptor_offset != layout.ac_descriptor_offset
        || header.ac_descriptor_len != layout.ac_descriptor_len
        || header.ac_fragment_offset != layout.ac_fragment_offset
        || header.ac_words_per_block != layout.ac_words_per_block
        || header.ac_fragment_words != layout.ac_fragment_words
        || header.ac_pass_count != layout.ac_pass_count
        || header.saliency_offset != layout.saliency_offset
        || header.saliency_groups != layout.saliency_groups
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT status, live counts, orientation, or layout metadata mismatch",
        ));
    }
    if header.dc_fragment_bit_len > layout.fragment_max_bits
        || header.dc_fragment_bit_len
            > layout
                .fragment_word_capacity
                .checked_mul(32)
                .ok_or(BackendError::InvalidArtifact(
                    "VarDCT fragment capacity overflow",
                ))?
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT entropy fragment exceeds its checked capacity",
        ));
    }

    let descriptor_words = artifact_words(
        words,
        layout.fragment_descriptor_offset,
        layout.fragment_descriptor_len,
    )?;
    let fragment_descriptors =
        bytemuck::try_cast_slice::<u32, DcFragmentDescriptor>(descriptor_words).map_err(|_| {
            BackendError::InvalidArtifact("VarDCT fragment descriptor ABI alignment")
        })?;
    if fragment_descriptors.len()
        != usize::try_from(lf_group_count).map_err(|_| {
            BackendError::InvalidArtifact("VarDCT LF group count does not fit usize")
        })?
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT fragment descriptor count mismatch",
        ));
    }
    let strategy_map = artifact_words(words, layout.strategy_offset, layout.strategy_len)?;
    let quantized_dc = artifact_words(words, layout.dc_offset, layout.dc_len)?;
    let raw_tokens = artifact_words(words, layout.token_offset, layout.token_len)?;
    let extra_bits = artifact_words(words, layout.extra_offset, layout.extra_len)?;
    let fragment_words =
        artifact_words(words, layout.fragment_offset, layout.fragment_word_capacity)?;
    validate_zero_gap(words, HEADER_WORDS, layout.fragment_descriptor_offset)?;
    validate_zero_gap(
        words,
        layout.fragment_descriptor_offset + layout.fragment_descriptor_len,
        layout.strategy_offset,
    )?;
    validate_zero_gap(
        words,
        layout.strategy_offset + layout.strategy_len,
        layout.dc_offset,
    )?;
    validate_zero_gap(words, layout.dc_offset + layout.dc_len, layout.token_offset)?;
    validate_zero_gap(
        words,
        layout.token_offset + layout.token_len,
        layout.extra_offset,
    )?;
    validate_zero_gap(
        words,
        layout.extra_offset + layout.extra_len,
        layout.fragment_offset,
    )?;
    let dc_end = layout.fragment_offset + layout.fragment_word_capacity;
    let entropy_end = if layout.saliency_groups == 0 {
        layout.artifact_words
    } else {
        layout.saliency_offset
    };
    let ac = if layout.ac_descriptor_len == 0 {
        validate_zero_gap(words, dc_end, entropy_end)?;
        AcFragments::Empty
    } else {
        validate_zero_gap(words, dc_end, layout.ac_descriptor_offset)?;
        validate_zero_gap(
            words,
            layout.ac_descriptor_offset + layout.ac_descriptor_len * layout.ac_pass_count,
            layout.ac_fragment_offset,
        )?;
        validate_zero_gap(
            words,
            layout.ac_fragment_offset + layout.ac_fragment_words,
            entropy_end,
        )?;
        let bit_lengths = artifact_words(
            words,
            layout.ac_descriptor_offset,
            layout.ac_descriptor_len * layout.ac_pass_count,
        )?;
        let ac_words = artifact_words(words, layout.ac_fragment_offset, layout.ac_fragment_words)?;
        match frame.topology {
            VarDctTopology::StrategyMap => {
                let plan =
                    transform_plan.ok_or(BackendError::Invariant("missing transform map"))?;
                for (words, lengths) in ac_words
                    .chunks_exact((layout.ac_fragment_words / layout.ac_pass_count) as usize)
                    .zip(bit_lengths.chunks_exact(layout.ac_descriptor_len as usize))
                {
                    plan.validate_ac(words, lengths, hf_entropy)?;
                }
                AcFragments::StrategyMap {
                    words: ac_words,
                    bit_lengths,
                    plan,
                }
            }
            VarDctTopology::TiledDct8 => {
                validate_blocks(ac_words, bit_lengths, layout.ac_words_per_block, hf_entropy)?;
                AcFragments::Dct8Blocks {
                    words: ac_words,
                    bit_lengths,
                    words_per_block: layout.ac_words_per_block,
                }
            }
            VarDctTopology::SingleTransform(_) => {
                validate_transform_fragments(
                    ac_words,
                    bit_lengths,
                    layout.ac_words_per_block,
                    block_count * 63,
                    hf_entropy,
                )?;
                AcFragments::Single {
                    words: ac_words,
                    bit_lengths,
                    words_per_pass: layout.ac_words_per_block,
                }
            }
        }
    };

    let expected_strategy = strategy_id;
    for (block, &value) in strategy_map.iter().enumerate() {
        if frame.topology == VarDctTopology::StrategyMap {
            let plan = transform_plan.ok_or(BackendError::Invariant("missing transform map"))?;
            if plan.map.block_map.get(block) != Some(&value) {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT GPU strategy map differs from plan",
                ));
            }
            continue;
        }
        let is_first = match frame.topology {
            VarDctTopology::StrategyMap => unreachable!("validated above"),
            VarDctTopology::SingleTransform(_) => block == 0,
            VarDctTopology::TiledDct8 => true,
        };
        let expected = expected_strategy | u32::from(is_first) << 8;
        if value != expected {
            return Err(BackendError::InvalidArtifact(
                "VarDCT GPU strategy map is malformed",
            ));
        }
    }

    let block_count_usize = usize::try_from(block_count)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT block count does not fit usize"))?;
    let entries = code.raw_entries();
    let mut expected_histogram = [0u32; UINT_SYMBOLS];
    let mut bit_offset = 0u32;
    for (group_index, descriptor) in fragment_descriptors.iter().enumerate() {
        let group_index = u32::try_from(group_index)
            .map_err(|_| BackendError::InvalidArtifact("VarDCT LF group index exceeds u32"))?;
        let group = frame
            .lf_group_blocks(group_index)
            .map_err(|_| BackendError::InvalidArtifact("VarDCT LF group geometry mismatch"))?;
        if descriptor.bit_offset != bit_offset {
            return Err(BackendError::InvalidArtifact(
                "VarDCT fragment descriptors are not contiguous",
            ));
        }
        for channel in 0..3usize {
            let base = channel * block_count_usize;
            for local_y in 0..group.height as usize {
                for local_x in 0..group.width as usize {
                    let block_x = group.origin_x as usize + local_x;
                    let block_y = group.origin_y as usize + local_y;
                    let block = block_y * blocks_x as usize + block_x;
                    let left = if local_x > 0 {
                        quantized_dc[base + block - 1] as i32
                    } else if local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize] as i32
                    } else {
                        0
                    };
                    let top = if local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize] as i32
                    } else {
                        left
                    };
                    let top_left = if local_x > 0 && local_y > 0 {
                        quantized_dc[base + block - blocks_x as usize - 1] as i32
                    } else {
                        left
                    };
                    let actual = quantized_dc[base + block] as i32;
                    let residual = gradient_residual_i32(actual, top, left, top_left);
                    let (token, extra_bit_count, extra) = signed_token(residual);
                    let slot = base + block;
                    if raw_tokens[slot] != token || extra_bits[slot] != extra {
                        return Err(BackendError::InvalidArtifact(
                            "VarDCT DC token does not match its predicted residual",
                        ));
                    }
                    let token_index = usize::try_from(token).map_err(|_| {
                        BackendError::InvalidArtifact("VarDCT DC token index does not fit usize")
                    })?;
                    let entry = entries
                        .get(token_index)
                        .ok_or(BackendError::InvalidArtifact(
                            "VarDCT DC token exceeds the fixed entropy alphabet",
                        ))?;
                    if read_fragment_slice(
                        fragment_words,
                        header.dc_fragment_bit_len,
                        bit_offset,
                        u32::from(entry.bit_len),
                    )? != u32::from(entry.bits)
                    {
                        return Err(BackendError::InvalidArtifact(
                            "VarDCT GPU prefix fragment does not match its token",
                        ));
                    }
                    bit_offset += u32::from(entry.bit_len);
                    if read_fragment_slice(
                        fragment_words,
                        header.dc_fragment_bit_len,
                        bit_offset,
                        extra_bit_count,
                    )? != extra
                    {
                        return Err(BackendError::InvalidArtifact(
                            "VarDCT GPU extra-bit fragment does not match its token",
                        ));
                    }
                    bit_offset += extra_bit_count;
                    expected_histogram[token_index] += 1;
                }
            }
        }
        if descriptor.bit_len != bit_offset - descriptor.bit_offset {
            return Err(BackendError::InvalidArtifact(
                "VarDCT fragment descriptor length mismatch",
            ));
        }
    }
    if bit_offset != header.dc_fragment_bit_len || header.raw_histogram != expected_histogram {
        return Err(BackendError::InvalidArtifact(
            "VarDCT entropy fragment length or histogram mismatch",
        ));
    }
    validate_fragment_padding(fragment_words, header.dc_fragment_bit_len)?;
    let saliency = if layout.saliency_groups == 0 {
        None
    } else {
        let records = artifact_words(words, layout.saliency_offset, layout.saliency_groups * 4)?;
        let records = bytemuck::try_cast_slice::<u32, saliency::Record>(records)
            .map_err(|_| BackendError::InvalidArtifact("saliency record ABI alignment"))?;
        saliency::validate(records, frame)?;
        validate_zero_gap(
            words,
            layout.saliency_offset + layout.saliency_groups * 4,
            layout.artifact_words,
        )?;
        Some(records)
    };
    Ok(VarDctArtifactData {
        saliency,
        raw_matrices: Default::default(),
        transform_plan,
        strategy: expected_strategy,
        dc_fragment_words: fragment_words,
        dc_fragment_bit_len: header.dc_fragment_bit_len,
        dc_fragment_descriptors: fragment_descriptors,
        ac,
    })
}

fn artifact_words(words: &[u32], offset: u32, len: u32) -> Result<&[u32], BackendError> {
    let start = usize::try_from(offset)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact offset does not fit usize"))?;
    let len = usize::try_from(len)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT artifact length does not fit usize"))?;
    let end = start.checked_add(len).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range overflow",
    ))?;
    words.get(start..end).ok_or(BackendError::InvalidArtifact(
        "VarDCT artifact range is out of bounds",
    ))
}

fn validate_zero_gap(words: &[u32], start: u32, end: u32) -> Result<(), BackendError> {
    if artifact_words(
        words,
        start,
        end.checked_sub(start).ok_or(BackendError::InvalidArtifact(
            "VarDCT artifact section order is invalid",
        ))?,
    )?
    .iter()
    .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "VarDCT artifact alignment padding is nonzero",
        ));
    }
    Ok(())
}

pub(super) fn clamped_gradient_i32(top: i32, left: i32, top_left: i32) -> i32 {
    let lower = top.min(left);
    let upper = top.max(left);
    if top_left >= upper {
        lower
    } else if top_left <= lower {
        upper
    } else {
        top.wrapping_add(left.wrapping_sub(top_left))
    }
}

pub(super) fn gradient_residual_i32(actual: i32, top: i32, left: i32, top_left: i32) -> i32 {
    actual.wrapping_sub(clamped_gradient_i32(top, left, top_left))
}

pub(super) fn signed_token(value: i32) -> (u32, u32, u32) {
    let value = pack_signed_control(value);
    if value == 0 {
        return (0, 0, 0);
    }
    let extra_bit_count = 31 - value.leading_zeros();
    let token = extra_bit_count + 1;
    (token, extra_bit_count, value - (1 << extra_bit_count))
}

/// GPU convenience encoder for one standard transform or an image-wide strategy map.
pub struct VarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
    frame: VarDctFrameLayout,
}

impl VarDctEncoder {
    /// Creates the profile backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed standard entropy tree cannot be
    /// constructed or the selected device cannot execute the strategy's
    /// checked storage/workgroup/dispatch requirements.
    pub fn new(context: WgpuContext, strategy: VarDctStrategy) -> Result<Self, EncodeError> {
        Self::new_with_config(context, strategy, VarDctConfig::default())
    }

    /// Creates the profile backend with explicit quantization and LF metadata.
    pub fn new_with_config(
        context: WgpuContext,
        strategy: VarDctStrategy,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new_with_config(&context, strategy, config)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
            frame: VarDctFrameLayout::single(strategy),
        })
    }

    /// Uses a caller-supplied strategy map across the whole image.
    pub fn new_with_strategy_map(
        context: WgpuContext,
        map: VarDctStrategyMap,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        let frame = map.frame()?;
        let backend = VarDctBackend::new_with_strategy_map(&context, map, config)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
            frame,
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    #[must_use]
    pub fn strategy_map(&self) -> &VarDctStrategyMap {
        &self
            .encoder
            .backend()
            .transform_plan
            .as_ref()
            .expect("VarDCT transform plan")
            .map
    }

    /// Workgroup selected for this encoder's parallel VarDCT pass.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub fn lf_metadata(&self) -> VarDctLfMetadata {
        self.encoder.backend().lf_metadata()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn quantization(&self) -> VarDctQuantization {
        self.encoder.backend().config.quantization
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.memory_plan(&source)?;
        let (width, height) = (self.frame.width, self.frame.height);
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                quantization: self.encoder.backend().config.quantization,
            },
            progressive: self.encoder.backend().config.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame),
            codestream_header: image_header(width, height)?,
            container,
        })
    }
}

/// GPU-only JPEG XL VarDCT encoder for a rectangular grid of independent
/// regular DCT8 transforms.
///
/// Accepts nonzero RGB8 dimensions through 16,384 pixels on each axis, with
/// partial edge blocks replicated on the GPU. Every block carries quantized
/// DC and AC, using default matrices, configurable coefficient orders and one prefix distribution.
/// The frame has every 2,048-pixel LF group and 256-pixel AC group; a single
/// AC group uses the standard fused packet. Exact quantizer settings are configurable;
/// perceptual distance and rate-control guarantees remain unimplemented.
pub struct TiledVarDctEncoder {
    encoder: GpuEncoder<VarDctBackend>,
}

impl TiledVarDctEncoder {
    /// Creates the tiled DCT8 backend.
    ///
    /// # Errors
    ///
    /// Returns an encoder error if the fixed entropy tree cannot be built or
    /// the device cannot execute the checked tiled kernel ABI.
    pub fn new(context: WgpuContext) -> Result<Self, EncodeError> {
        Self::new_with_config(context, VarDctConfig::default())
    }

    /// Creates the tiled DCT8 backend with explicit quantization and LF metadata.
    pub fn new_with_config(
        context: WgpuContext,
        config: VarDctConfig,
    ) -> Result<Self, EncodeError> {
        let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config)?;
        Ok(Self {
            encoder: GpuEncoder::new(context, backend),
        })
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Workgroup selected for block quantization. Control serialization remains scalar.
    #[must_use]
    pub fn workgroup_variant(&self) -> KernelVariant {
        self.encoder.backend().workgroup_variant()
    }

    #[must_use]
    pub fn lf_metadata(&self) -> VarDctLfMetadata {
        self.encoder.backend().lf_metadata()
    }

    #[must_use]
    pub const fn color_encoding(&self) -> VarDctColorEncoding {
        VarDctColorEncoding::SrgbD65
    }

    #[must_use]
    pub fn quantization(&self) -> VarDctQuantization {
        self.encoder.backend().config.quantization
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    pub fn memory_plan(&self, source: &BufferImageSource) -> Result<VarDctMemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    pub fn grid(&self, source: &BufferImageSource) -> Result<TiledVarDctGrid, EncodeError> {
        self.memory_plan(source)?;
        Ok(TiledVarDctGrid {
            passes: self.encoder.backend().config.progressive.passes().len() as u8,
            ..TiledVarDctGrid::new(source.layout.extent.width, source.layout.extent.height)?
        })
    }

    pub fn submit(&self, source: BufferImageSource) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: BufferImageSource,
    ) -> Result<VarDctSubmission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(&self, source: BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: BufferImageSource,
        container: bool,
    ) -> Result<VarDctSubmission, EncodeError> {
        let frame =
            VarDctFrameLayout::tiled_dct8(source.layout.extent.width, source.layout.extent.height)?;
        self.memory_plan(&source)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                quantization: self.encoder.backend().config.quantization,
            },
            progressive: self.encoder.backend().config.progressive.clone(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: frame.width,
            canvas_height: frame.height,
            options: FrameOptions::default(),
        };
        let frame_submission = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(VarDctSubmission {
            frame: Some(frame_submission),
            codestream_header: image_header(frame.width, frame.height)?,
            container,
        })
    }
}

/// Executor-independent future for a complete standard VarDCT codestream.
pub struct VarDctSubmission {
    frame: Option<FrameSubmission<VarDctJob>>,
    codestream_header: BitFragment,
    container: bool,
}

impl VarDctSubmission {
    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a VarDCT submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let encoded_frame = assemble_frame(frame.packets)?;
        let mut codestream = self.codestream_header.bytes().to_vec();
        codestream.extend_from_slice(encoded_frame.bytes());
        if self.container {
            Ok(jxl_gpu_bitstream::write_container(&codestream)?)
        } else {
            Ok(codestream)
        }
    }
}

impl Future for VarDctSubmission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a VarDCT submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
            }
        }
    }
}
