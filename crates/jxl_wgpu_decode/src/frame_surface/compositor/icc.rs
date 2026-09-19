//! Image-owned ICC selection and completion-owned GPU presentation resources.

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::{ExtraChannelInventory, ExtraChannelTypeInventory};
use jxl_gpu_formats::{ColorSpecification, ImageLayout};
use jxl_gpu_protocol::icc::{IccProfile, IccSignature, IccTransform};
use jxl_gpu_protocol::{OutputOrientation, RgbColorEncoding, RgbColorSpace, WhitePointAdaptation};
use jxl_wgpu::{
    GpuBufferLease, ImageOutputGeometry, ImageOutputParams, ImageOutputSource,
    ResidentStorageBinding, WgpuBackend,
};
use wgpu::util::DeviceExt;

use super::super::icc_transform::{ColorBinding, Transform, Transforms};
use super::super::spot::render::Rendering;
use super::{NativeParams, Surface, aligned, dispatch, pipeline};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::gpu_submission::{
    GpuWork, IccWork, completion_fence_bytes, submit_icc_recorded, validate_size,
};
use crate::{Error, GpuOutputRequest, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(crate) enum Output<'a> {
    Color(&'a GpuOutputRequest),
    Numeric {
        request: &'a GpuOutputRequest,
        params: NativeParams,
        profile: &'a IccProfile,
    },
}

pub(crate) struct ImageMetadata<'a> {
    pub(crate) orientation: OutputOrientation,
    pub(crate) intensity: jxl_gpu_protocol::DisplayIntensity,
    pub(crate) tone_mapping: Option<jxl_gpu_protocol::ToneMapping>,
    pub(crate) extras: &'a [ExtraChannelInventory],
}

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) source_encoding: FrameSurfaceEncoding,
    transform: Option<Arc<Transform>>,
    spots: Option<Rendering>,
    working: FrameSurfaceLayout,
    working_encoding: FrameSurfaceEncoding,
    output: ImageLayout,
    pipeline: wgpu::ComputePipeline,
    params: Vec<u8>,
    bindings: [u32; 2],
    dispatch: [u32; 2],
}

impl Presentation {
    pub(crate) fn layout(&self) -> &ImageLayout {
        &self.output
    }

    pub(crate) fn new(
        backend: &WgpuBackend,
        source: &FrameSurfaceLayout,
        source_encoding: FrameSurfaceEncoding,
        output: Output<'_>,
        image: ImageMetadata<'_>,
        transforms: &mut Transforms,
    ) -> Result<Self> {
        let ImageMetadata {
            orientation,
            intensity,
            tone_mapping,
            extras,
        } = image;
        let (request, numeric) = match output {
            Output::Color(request) => (request, None),
            Output::Numeric {
                request,
                params,
                profile,
            } => (request, Some((params, profile))),
        };
        let device = backend.device();
        if source.color.format != source_encoding.format() {
            return Err(Error::EngineContract("color presentation source layout"));
        }
        if extras.len() != source.extras.len() {
            return Err(Error::EngineContract(
                "color presentation extra-channel count",
            ));
        }
        let alpha_extra = extras.iter().position(|extra| {
            matches!(extra.channel_type, ExtraChannelTypeInventory::Alpha { .. })
        });
        let spots = request
            .renders_spot_colors(extras)
            .then(|| Rendering::new(backend, source, extras))
            .transpose()?;
        let output = ImageLayout::packed(
            orientation.map_extent(source.color.extent),
            request.format().clone(),
        )?;
        let device_output = output.format.model == jxl_gpu_formats::ColorModel::IccDevice;
        let target_encoding = |profile: &jxl_gpu_protocol::icc::IccProfile| {
            if device_output || numeric.is_some() {
                FrameSurfaceEncoding::Device(profile.clone())
            } else {
                FrameSurfaceEncoding::Icc(profile.clone())
            }
        };
        let target = numeric.map_or_else(
            || output.format.color_spec.clone(),
            |(_, profile)| ColorSpecification::Icc(profile.clone()),
        );
        // Numeric reconstruction follows the suggested original profile's intent, independent
        // of presentation options. Complete device output keeps generated K separate from Black.
        let intent = numeric.map_or(request.icc_rendering_intent(), |(_, profile)| {
            profile.header().rendering_intent
        });
        let (encoding, selected) = match (&source_encoding, &target) {
            (
                FrameSurfaceEncoding::Icc(profile)
                | FrameSurfaceEncoding::Device(profile)
                | FrameSurfaceEncoding::Cmyk { profile, .. },
                ColorSpecification::Icc(target),
            ) => (
                if target == profile && tone_mapping.is_none() {
                    source_encoding.clone()
                } else {
                    target_encoding(target)
                },
                if target == profile && tone_mapping.is_none() {
                    None
                } else {
                    Some(IccTransform::new(profile, target, intent)?)
                },
            ),
            (
                FrameSurfaceEncoding::Icc(profile)
                | FrameSurfaceEncoding::Device(profile)
                | FrameSurfaceEncoding::Cmyk { profile, .. },
                ColorSpecification::Defined(_),
            ) => (
                FrameSurfaceEncoding::Rgb(RgbColorEncoding::LINEAR_BT709),
                Some(IccTransform::to_linear_rgb(
                    profile,
                    RgbColorSpace::Bt709,
                    intent,
                )?),
            ),
            (FrameSurfaceEncoding::Rgb(encoding), ColorSpecification::Icc(target)) => (
                target_encoding(target),
                Some(IccTransform::from_rgb_with_intensity(
                    *encoding, intensity, target, intent,
                )?),
            ),
            (FrameSurfaceEncoding::Rgb(_), ColorSpecification::Defined(_)) => {
                (source_encoding.clone(), None)
            }
            _ => {
                return Err(Error::UnsupportedOutputFormat(
                    "ICC color output requires an explicit target profile or enumerated encoding"
                        .into(),
                ));
            }
        };
        // Profile intent connection precedes luminance mapping in D50 PCS. Enumerated output
        // instead maps in the final RGB packer, after its selected profile conversion.
        let selected = if matches!(target, ColorSpecification::Icc(_)) {
            selected
                .map(|selected| match tone_mapping {
                    Some(mapping) => selected.with_tone_mapping(mapping),
                    None => Ok(selected),
                })
                .transpose()?
        } else {
            selected
        };
        if numeric.is_none()
            && selected.is_some()
            && request.white_point_adaptation() != WhitePointAdaptation::Bradford
        {
            return Err(Error::UnsupportedOutputFormat(
                "ICC color conversion currently requires Bradford adaptation".into(),
            ));
        }
        let working = if selected.is_none() {
            source.clone()
        } else {
            FrameSurfaceLayout::with_encoding(
                source.color.extent,
                source.extras.len(),
                encoding.clone(),
                &device.limits(),
            )?
        };
        let size = aligned(output.logical_size)?;
        validate_size(device, size)?;
        let dispatch = dispatch(device, size / 4)?;
        let geometry = ImageOutputGeometry {
            extent: source.color.extent,
            orientation,
            strides: [source.color.extent.width; 3],
        };
        let params = if let Some((mut params, profile)) = numeric {
            params.color[2] = working.color.planes.len() as u32;
            params.source[0] = (working.color_plane_bytes / 4) as u32;
            params.source[1] = alpha_extra.map_or(u32::MAX, |index| params.color[2] + index as u32);
            if matches!(encoding, FrameSurfaceEncoding::Device(_))
                && profile.header().device_space == IccSignature(*b"CMYK")
            {
                params.source[3] |= 4;
            }
            bytemuck::bytes_of(&params).to_vec()
        } else if device_output {
            let planes = working.icc_planes(&encoding)?;
            let alpha = alpha_extra.map(|index| {
                let plane = &working.extras[index].planes[0];
                jxl_wgpu::ResidentIccPlane {
                    offset: (plane.offset / 4) as u32,
                    stride: (plane.row_stride / 4) as u32,
                }
            });
            bytemuck::bytes_of(&jxl_wgpu::DeviceOutputParams::new(
                &output,
                jxl_wgpu::DeviceOutputSource {
                    profile: encoding
                        .icc_profile()
                        .ok_or(Error::EngineContract("ICC output has no device profile"))?,
                    extent: source.color.extent,
                    orientation,
                    planes: &planes,
                    alpha,
                    input_words: working.storage_bytes / 4,
                    sample_encoding: encoding.icc_sample_encoding(),
                    alpha_conversion: request.alpha_conversion(extras),
                },
                dispatch[0] * 64,
            )?)
            .to_vec()
        } else {
            bytemuck::bytes_of(
                &match &encoding {
                    FrameSurfaceEncoding::Icc(target) => ImageOutputParams::for_icc_device(
                        &output,
                        geometry,
                        target,
                        dispatch[0] * 64,
                    )?,
                    FrameSurfaceEncoding::Rgb(encoding) => {
                        let source = ImageOutputSource {
                            extent: geometry.extent,
                            orientation,
                            strides: geometry.strides,
                            encoding: *encoding,
                        };
                        match tone_mapping {
                            Some(mapping) => ImageOutputParams::new_with_tone_mapping(
                                &output,
                                source,
                                dispatch[0] * 64,
                                request.white_point_adaptation(),
                                mapping,
                            ),
                            None => ImageOutputParams::new_with_intensity_target(
                                &output,
                                source,
                                dispatch[0] * 64,
                                request.white_point_adaptation(),
                                intensity.nits(),
                            ),
                        }?
                        .with_gamut_mapping(request.gamut_mapping())?
                    }
                    FrameSurfaceEncoding::Encoded
                    | FrameSurfaceEncoding::Device(_)
                    | FrameSurfaceEncoding::Cmyk { .. } => {
                        unreachable!("resolved presentation color")
                    }
                }
                .with_alpha_conversion(request.alpha_conversion(extras)),
            )
            .to_vec()
        };
        if params.len() as u64 > device.limits().max_uniform_buffer_binding_size {
            return Err(Error::CompositionResourceLimit {
                resource: "ICC output uniform bytes",
                requested: params.len() as u64,
                limit: device.limits().max_uniform_buffer_binding_size,
            });
        }
        let color_count = working.color.planes.len() as u32;
        let alpha_channel = alpha_extra.map_or(u32::MAX, |index| color_count + index as u32);
        let shader = if numeric.is_some() {
            super::native_shader(super::super::spot::shader(false))
        } else if device_output {
            jxl_wgpu::DEVICE_OUTPUT_SHADER.to_owned()
        } else {
            format!(
                "{}\n{}\n{}",
                jxl_wgpu::IMAGE_OUTPUT_SHADER,
                super::super::spot::shader(false),
                include_str!("../output.wgsl")
            )
        };
        let mut constants = if numeric.is_some() {
            Vec::new()
        } else {
            vec![("wg_x", 64.0), ("wg_y", 1.0)]
        };
        if !device_output && numeric.is_none() {
            constants.extend([
                (
                    "surface_plane_words",
                    (working.color_plane_bytes / 4) as f64,
                ),
                ("surface_alpha_channel", f64::from(alpha_channel)),
                ("surface_color_channels", f64::from(color_count)),
            ]);
        }
        let pipeline = pipeline(
            device,
            "JPEG XL ICC presentation packing",
            &shader,
            &constants,
        );
        let transform = selected
            .map(|selected| transforms.select(backend, selected))
            .transpose()?;
        Ok(Self {
            source_encoding,
            transform,
            spots,
            working,
            working_encoding: encoding,
            output,
            pipeline,
            params,
            bindings: if numeric.is_some() { [1, 2] } else { [3, 4] },
            dispatch,
        })
    }

    pub(crate) fn pack(&self, backend: &WgpuBackend, source: &Surface) -> Result<GpuWork> {
        if source.encoding != self.source_encoding {
            return Err(Error::EngineContract(
                "color presentation received an unplanned source encoding",
            ));
        }
        let device = backend.device();
        let poll = backend.submission_poller().try_reserve()?;
        let output_size = aligned(self.output.logical_size)?;
        let output_permit = backend.transient_memory_budget().try_reserve(output_size)?;
        let transient_bytes = self.params.len() as u64
            + completion_fence_bytes()
            + self.spots.as_ref().map_or(0, Rendering::memory_bytes)
            + self.transform.as_ref().map_or(0, |transform| {
                self.working.storage_bytes + transform.memory.transient_bytes()
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
        let rendered = self
            .spots
            .as_ref()
            .map(|spots| spots.encode(device, &mut encoder, source))
            .transpose()?;
        let source_buffer = rendered
            .as_ref()
            .map_or(source.buffer.as_wgpu_buffer(), |rendered| &rendered.buffer);
        let mut converted = None;
        let mut icc_dispatch = None;
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
            icc_dispatch = Some(transform.encode(
                backend,
                &mut encoder,
                program,
                ColorBinding {
                    storage: binding(source_buffer, source.layout.storage_bytes),
                    layout: &source.layout,
                    encoding: &self.source_encoding,
                },
                ColorBinding {
                    storage: binding(&buffer, self.working.storage_bytes),
                    layout: &self.working,
                    encoding: &self.working_encoding,
                },
            )?);
            source.copy_extras(
                &mut encoder,
                binding(&buffer, self.working.storage_bytes),
                &self.working,
            )?;
            converted = Some(buffer);
        }
        let input = converted.as_ref().unwrap_or(source_buffer);
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JPEG XL ICC packing parameters"),
            contents: &self.params,
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
                    binding: self.bindings[0],
                    resource: output.as_wgpu_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: self.bindings[1],
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
        submit_icc_recorded(
            backend,
            encoder,
            output,
            vec![source.buffer.clone()],
            IccWork {
                resources: (uniform, converted, rendered, program),
                dispatch: icc_dispatch,
            },
            permit,
            poll,
        )
    }
}
