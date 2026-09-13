//! Image-owned ICC selection and completion-owned GPU presentation resources.

use std::num::NonZeroU64;

use jxl_gpu_formats::{ColorSpecification, ImageLayout};
use jxl_gpu_protocol::icc::IccTransform;
use jxl_gpu_protocol::{OutputOrientation, RgbColorEncoding, RgbColorSpace, WhitePointAdaptation};
use jxl_wgpu::{
    AlphaConversion, GpuBufferLease, ImageOutputGeometry, ImageOutputParams, ImageOutputSource,
    ResidentStorageBinding, WgpuBackend,
};
use wgpu::util::DeviceExt;

use super::super::icc_transform::{ColorBinding, Transform};
use super::super::submission::{GpuWork, completion_fence_bytes, submit_recorded, validate_size};
use super::{Surface, aligned, dispatch, pipeline};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::{Error, GpuOutputRequest, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

#[derive(Debug)]
pub(super) struct Presentation {
    source_encoding: FrameSurfaceEncoding,
    transform: Option<Transform>,
    working: FrameSurfaceLayout,
    output: ImageLayout,
    pipeline: wgpu::ComputePipeline,
    params: ImageOutputParams,
    dispatch: [u32; 2],
}

impl Presentation {
    pub(super) fn new(
        backend: &WgpuBackend,
        source: &FrameSurfaceLayout,
        source_encoding: &FrameSurfaceEncoding,
        request: &GpuOutputRequest,
        orientation: OutputOrientation,
        alpha: AlphaConversion,
        alpha_extra: Option<usize>,
    ) -> Result<Self> {
        let device = backend.device();
        if FrameSurfaceEncoding::from_format(&source.color.format).as_ref() != Some(source_encoding)
        {
            return Err(Error::EngineContract(
                "color presentation source layout mismatch",
            ));
        }
        if alpha_extra.is_some_and(|index| index >= source.extras.len()) {
            return Err(Error::EngineContract("color presentation alpha index"));
        }
        let output = ImageLayout::packed(
            orientation.map_extent(source.color.extent),
            request.format().clone(),
        )?;
        if matches!(output.format.color_spec, ColorSpecification::Defined(color)
            if matches!(color.transfer, jxl_gpu_formats::TransferFunction::Pq | jxl_gpu_formats::TransferFunction::Hlg))
        {
            return Err(crate::color_output::ColorOutputError::HdrLuminanceMappingRequired.into());
        }
        let (encoding, selected) =
            match (source_encoding, &output.format.color_spec) {
                (FrameSurfaceEncoding::Icc(profile), ColorSpecification::Icc(target)) => (
                    FrameSurfaceEncoding::Icc(target.clone()),
                    if target == profile {
                        None
                    } else {
                        Some(IccTransform::new(
                            profile,
                            target,
                            request.icc_rendering_intent(),
                        )?)
                    },
                ),
                (FrameSurfaceEncoding::Icc(profile), ColorSpecification::Defined(_)) => (
                    FrameSurfaceEncoding::Rgb(RgbColorEncoding::LINEAR_BT709),
                    Some(IccTransform::to_linear_rgb(
                        profile,
                        RgbColorSpace::Bt709,
                        request.icc_rendering_intent(),
                    )?),
                ),
                (FrameSurfaceEncoding::Rgb(encoding), ColorSpecification::Icc(target)) => {
                    if encoding.transfer != jxl_gpu_protocol::TransferFunction::Linear {
                        return Err(Error::EngineContract(
                            "ICC connection requires linear RGB input",
                        ));
                    }
                    (
                        FrameSurfaceEncoding::Icc(target.clone()),
                        Some(IccTransform::from_linear_rgb(
                            encoding.space,
                            target,
                            request.icc_rendering_intent(),
                        )?),
                    )
                }
                (FrameSurfaceEncoding::Rgb(_), ColorSpecification::Defined(_)) => {
                    (source_encoding.clone(), None)
                }
                _ => return Err(Error::UnsupportedOutputFormat(
                    "ICC color output requires an explicit target profile or enumerated encoding"
                        .into(),
                )),
            };
        if selected.is_some() && request.white_point_adaptation() != WhitePointAdaptation::Bradford
        {
            return Err(Error::UnsupportedOutputFormat(
                "ICC color conversion currently requires Bradford adaptation".into(),
            ));
        }
        let working = FrameSurfaceLayout::with_encoding(
            source.color.extent,
            source.extras.len(),
            encoding.clone(),
            &device.limits(),
        )?;
        let size = aligned(output.logical_size)?;
        validate_size(device, size)?;
        let dispatch = dispatch(device, size / 4)?;
        let geometry = ImageOutputGeometry {
            extent: source.color.extent,
            orientation,
            strides: [source.color.extent.width; 3],
        };
        let params = match &encoding {
            FrameSurfaceEncoding::Icc(target) => {
                ImageOutputParams::for_icc_device(&output, geometry, target, dispatch[0] * 64)?
            }
            FrameSurfaceEncoding::Rgb(encoding) => ImageOutputParams::new(
                &output,
                ImageOutputSource {
                    extent: geometry.extent,
                    orientation,
                    strides: geometry.strides,
                    encoding: *encoding,
                },
                dispatch[0] * 64,
                request.white_point_adaptation(),
            )?,
            FrameSurfaceEncoding::Encoded => unreachable!("resolved presentation color"),
        }
        .with_alpha_conversion(alpha);
        let color_count = working.color.planes.len() as u32;
        let alpha_channel = alpha_extra.map_or(u32::MAX, |index| color_count + index as u32);
        let shader = format!(
            "{}\n{}\n{}",
            jxl_wgpu::IMAGE_OUTPUT_SHADER,
            super::super::spot::shader(false),
            include_str!("../output.wgsl")
        );
        let pipeline = pipeline(
            device,
            "JPEG XL ICC presentation packing",
            &shader,
            &[
                ("wg_x", 64.0),
                ("wg_y", 1.0),
                (
                    "surface_plane_words",
                    (working.color_plane_bytes / 4) as f64,
                ),
                ("surface_alpha_channel", f64::from(alpha_channel)),
                ("surface_color_channels", f64::from(color_count)),
            ],
        );
        let transform = selected
            .map(|selected| Transform::new(backend, selected))
            .transpose()?;
        Ok(Self {
            source_encoding: source_encoding.clone(),
            transform,
            working,
            output,
            pipeline,
            params,
            dispatch,
        })
    }

    pub(super) fn pack(&self, backend: &WgpuBackend, source: &Surface) -> Result<GpuWork> {
        if source.encoding != self.source_encoding {
            return Err(Error::EngineContract(
                "color presentation received an unplanned source encoding",
            ));
        }
        let device = backend.device();
        let poll = backend.submission_poller().try_reserve()?;
        let output_size = aligned(self.output.logical_size)?;
        let output_permit = backend.transient_memory_budget().try_reserve(output_size)?;
        let transient_bytes = std::mem::size_of::<ImageOutputParams>() as u64
            + completion_fence_bytes()
            + self.transform.as_ref().map_or(0, |transform| {
                self.working.storage_bytes + transform.memory.dispatch_uniform_bytes
            });
        let permit = backend
            .transient_memory_budget()
            .try_reserve(transient_bytes)?;
        let program = self
            .transform
            .as_ref()
            .map(|transform| transform.resident(backend))
            .transpose()?;
        let output = GpuBufferLease::from_tracked(
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL ICC presentation"),
                size: output_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            output_permit,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut converted = None;
        let mut icc_uniform = None;
        if let (Some(transform), Some(program)) = (&self.transform, &program) {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("JPEG XL ICC converted device values"),
                size: self.working.storage_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            fn binding(buffer: &wgpu::Buffer, size: u64) -> ResidentStorageBinding<'_> {
                ResidentStorageBinding {
                    buffer,
                    offset: 0,
                    size: NonZeroU64::new(size).expect("nonempty ICC storage"),
                }
            }
            icc_uniform = Some(transform.encode(
                backend,
                &mut encoder,
                program,
                ColorBinding {
                    storage: binding(source.buffer.as_wgpu_buffer(), source.layout.storage_bytes),
                    layout: &source.layout.color,
                },
                ColorBinding {
                    storage: binding(&buffer, self.working.storage_bytes),
                    layout: &self.working.color,
                },
            )?);
            source.copy_extras(
                &mut encoder,
                binding(&buffer, self.working.storage_bytes),
                &self.working,
            )?;
            converted = Some(buffer);
        }
        let input = converted.as_ref().unwrap_or(source.buffer.as_wgpu_buffer());
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL ICC packing parameters"),
            contents: bytemuck::bytes_of(&self.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JPEG XL ICC packing bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_wgpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], 1);
        }
        submit_recorded(
            backend,
            encoder,
            output,
            vec![source.buffer.clone()],
            (uniform, converted, icc_uniform, program),
            permit,
            poll,
        )
    }
}
