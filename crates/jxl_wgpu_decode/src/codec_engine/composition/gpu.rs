use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{
    ExtraChannelInventory, ExtraChannelTypeInventory, FrameInventory, SampleBitDepth,
};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_wgpu::{
    GpuBufferLease, GpuImageOutput, IMAGE_ORIENTATION_SHADER, IMAGE_OUTPUT_SHADER,
    ImageOutputParams, ImageOutputSource, WgpuBackend,
};

use super::blend::{BlendParams, blend_channels, intersection};
use super::spot::{SpotColor, shader as spot_shader, spot_colors};
use super::submission::{GpuWork, Submission, submit, validate_size};
use crate::frame_surface::{FrameSurfaceEncoding, FrameSurfaceLayout};
use crate::{Error, GpuOutputRequest, Result};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Unrounded RGB in an explicit domain, followed by independently normalized extra planes.
#[derive(Clone, Debug)]
pub(super) struct Surface {
    pub(super) buffer: GpuBufferLease,
    pub(super) extent: Extent2d,
    pub(super) plane_words: u32,
    pub(super) encoding: FrameSurfaceEncoding,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NativeParams {
    extent: [u32; 4],
    format: [u32; 4],
    output: [u32; 4],
    source: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<NativeParams>() == 64);
const _: () = assert!(std::mem::align_of::<NativeParams>() == 16);
const _: () = assert!(std::mem::offset_of!(NativeParams, source) == 48);

#[derive(Debug)]
enum Packing {
    Color(Box<[ImageOutputParams; 2]>),
    Native(NativeParams),
}

#[derive(Debug)]
pub(super) struct Compositor {
    backend: WgpuBackend,
    canvas: Extent2d,
    extras: Vec<ExtraChannelInventory>,
    surface: FrameSurfaceLayout,
    blend: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    packing: Packing,
    spots: Vec<SpotColor>,
    pub(super) layout: ImageLayout,
    blend_dispatch: [u32; 2],
    output_dispatch: [u32; 2],
}

impl Compositor {
    pub(super) fn new(
        backend: WgpuBackend,
        canvas: Extent2d,
        extras: &[ExtraChannelInventory],
        grayscale: bool,
        sample_bit_depth: SampleBitDepth,
        orientation: OutputOrientation,
        request: &GpuOutputRequest,
    ) -> Result<Self> {
        let orientation = request.orientation_policy().resolve(orientation);
        let layout = ImageLayout::packed(orientation.map_extent(canvas), request.format().clone())?;
        let device = backend.device();
        let surface = FrameSurfaceLayout::new(canvas, extras.len(), &device.limits())?;
        let output_size = aligned(layout.logical_size)?;
        validate_size(device, output_size)?;
        let bindings = device.limits().max_storage_buffers_per_shader_stage;
        if bindings < 7 {
            return Err(Error::CompositionResourceLimit {
                resource: "storage bindings",
                requested: 7,
                limit: u64::from(bindings),
            });
        }
        for (resource, required, available) in [
            (
                "workgroup invocations",
                64,
                device.limits().max_compute_invocations_per_workgroup,
            ),
            (
                "workgroup X size",
                64,
                device.limits().max_compute_workgroup_size_x,
            ),
            (
                "bind group entries",
                8,
                device.limits().max_bindings_per_bind_group,
            ),
            (
                "uniform bindings",
                1,
                device.limits().max_uniform_buffers_per_shader_stage,
            ),
        ] {
            if available < required {
                return Err(Error::CompositionResourceLimit {
                    resource,
                    requested: u64::from(required),
                    limit: u64::from(available),
                });
            }
        }
        let blend_dispatch = dispatch(
            device,
            u64::from(canvas.width) * u64::from(canvas.height) * (3 + extras.len() as u64),
        )?;
        let output_dispatch = dispatch(device, output_size / 4)?;
        let first_alpha = extras.iter().enumerate().find_map(|(index, extra)| {
            if let ExtraChannelTypeInventory::Alpha { associated } = extra.channel_type {
                Some((index, associated))
            } else {
                None
            }
        });
        let alpha_channel = first_alpha.map_or(u32::MAX, |(index, _)| 3 + index as u32);
        let alpha_conversion = request.alpha_conversion(extras);
        let color_channel = request.numeric_color_channel(grayscale)?;
        let selected = request
            .extra_channel()
            .map(|index| {
                extras
                    .get(index as usize)
                    .ok_or(Error::ExtraChannelIndex {
                        index,
                        count: extras.len() as u32,
                    })
                    .map(|extra| (3 + index, extra))
            })
            .transpose()?;
        if request.mapping() == crate::GpuOutputMapping::Color
            && extras
                .iter()
                .any(|extra| extra.channel_type == ExtraChannelTypeInventory::NonOptional)
        {
            return Err(crate::UnsupportedProfile::new(
                crate::UnsupportedCodestreamFeature::ExtraChannels,
                "non-optional extra-channel interpretation is not yet connected",
            )
            .into());
        }
        let spots = if request.renders_spot_colors(extras) {
            spot_colors(extras, (surface.plane_bytes / 4) as u32)
        } else {
            Vec::new()
        };
        if !spots.is_empty() {
            validate_size(device, std::mem::size_of_val(spots.as_slice()) as u64)?;
        }
        let spot_source = spot_shader(!spots.is_empty());
        let native = crate::model::native_modular_format(request.format());
        let source_depth = selected.map_or(sample_bit_depth, |(_, extra)| extra.bit_depth);
        let source_float = matches!(source_depth, SampleBitDepth::Float { .. });
        let wrong_numeric_type = match request.mapping() {
            crate::GpuOutputMapping::Numeric(crate::NumericSampleMapping::NativeFloat) => {
                !source_float
            }
            crate::GpuOutputMapping::Numeric(
                crate::NumericSampleMapping::NormalizedUnsigned
                | crate::NumericSampleMapping::NativeUnsigned,
            ) => source_float,
            _ => false,
        };
        if wrong_numeric_type {
            return Err(Error::UnsupportedOutputFormat(
                "numeric mapping does not match the declared source sample type".into(),
            ));
        }
        if request.mapping()
            == crate::GpuOutputMapping::Numeric(crate::NumericSampleMapping::NativeUnsigned)
            && native.is_none_or(|format| {
                source_depth
                    != (SampleBitDepth::Integer {
                        bits_per_sample: u32::from(format.bits_per_sample),
                    })
            })
        {
            return Err(Error::UnsupportedOutputFormat(
                "native composed scalar output must match its declared unsigned depth".into(),
            ));
        }
        let scalar_float = matches!(
            request.mapping(),
            crate::GpuOutputMapping::Numeric(
                crate::NumericSampleMapping::NormalizedUnsigned
                    | crate::NumericSampleMapping::NativeFloat
            )
        ) && matches!(jxl_gpu_formats::classify_pixel_format(request.format()), Ok(jxl_gpu_formats::PixelFormatClass::Numeric(n)) if n.components == 1 && n.sample_kind == jxl_gpu_formats::SampleKind::Float && n.bits_per_component == 32);
        let (packing, source) = if native.is_some() || scalar_float {
            if let (Some(native), Some((_, extra))) = (native, selected)
                && (native.channels != crate::ModularChannels::Gray
                    || extra.bit_depth
                        != (SampleBitDepth::Integer {
                            bits_per_sample: u32::from(native.bits_per_sample),
                        }))
            {
                return Err(Error::UnsupportedOutputFormat(
                    "native composed extra output must match its declared unsigned depth".into(),
                ));
            }
            let (channels, bits, sample_bytes) = native.map_or((1, 32, 4), |native| {
                (
                    native.channels.count(),
                    u32::from(native.bits_per_sample),
                    u32::from(native.storage_bits) / 8,
                )
            });
            (
                Packing::Native(NativeParams {
                    extent: [
                        layout.extent.width,
                        layout.extent.height,
                        canvas.width,
                        canvas.height,
                    ],
                    format: [
                        channels,
                        bits,
                        sample_bytes,
                        u32::try_from(layout.planes[0].row_stride).map_err(|_| address_error())?,
                    ],
                    output: [
                        u32::try_from(layout.logical_size).map_err(|_| address_error())?,
                        output_dispatch[0] * 64,
                        orientation.to_exif_value() - 1,
                        alpha_conversion as u32,
                    ],
                    source: [
                        (surface.plane_bytes / 4) as u32,
                        alpha_channel,
                        selected.map_or(color_channel.unwrap_or(0), |(index, _)| index),
                        u32::from(scalar_float),
                    ],
                }),
                format!(
                    "{IMAGE_ORIENTATION_SHADER}\n{}\n{spot_source}\n{}",
                    jxl_wgpu::ALPHA_OUTPUT_SHADER,
                    crate::modular_sample::shader(include_str!("native.wgsl"))
                ),
            )
        } else {
            if request.mapping() != crate::GpuOutputMapping::Color {
                return Err(Error::UnsupportedOutputFormat("composed numeric samples require matching native unsigned or scalar normalized F32 output".into()));
            }
            let params = |encoding: FrameSurfaceEncoding| -> Result<ImageOutputParams> {
                Ok(ImageOutputParams::new(
                    &layout,
                    ImageOutputSource {
                        extent: canvas,
                        orientation,
                        strides: [canvas.width; 3],
                        encoding: encoding.rgb_encoding(),
                    },
                    output_dispatch[0] * 64,
                )?
                .with_alpha_conversion(alpha_conversion))
            };
            (
                Packing::Color(Box::new([
                    params(FrameSurfaceEncoding::Srgb)?,
                    params(FrameSurfaceEncoding::Linear)?,
                ])),
                format!(
                    "{IMAGE_OUTPUT_SHADER}\n{spot_source}\n{}",
                    include_str!("output.wgsl")
                ),
            )
        };
        let blend = pipeline(
            device,
            "JPEG XL frame composition",
            include_str!("blend.wgsl"),
            &[],
        );
        let constants = if matches!(packing, Packing::Color(_)) {
            vec![
                ("wg_x", 64.0),
                ("wg_y", 1.0),
                ("surface_plane_words", (surface.plane_bytes / 4) as f64),
                ("surface_alpha_channel", f64::from(alpha_channel)),
            ]
        } else {
            Vec::new()
        };
        let pack = pipeline(device, "JPEG XL composed frame output", &source, &constants);
        Ok(Self {
            backend,
            canvas,
            extras: extras.to_vec(),
            surface,
            blend,
            pack,
            packing,
            spots,
            layout,
            blend_dispatch,
            output_dispatch,
        })
    }

    #[cfg(test)]
    pub(super) fn import(&self, outputs: Vec<GpuImageOutput>) -> Result<Surface> {
        self.import_with_encoding(outputs, None)
    }

    pub(super) fn import_with_encoding(
        &self,
        mut outputs: Vec<GpuImageOutput>,
        domain: Option<FrameSurfaceEncoding>,
    ) -> Result<Surface> {
        let first = outputs.first().ok_or(Error::EngineContract(
            "physical producer returned no surface",
        ))?;
        let extent = first.layout.extent;
        let encoding = domain
            .or_else(|| FrameSurfaceEncoding::from_format(&first.layout.format))
            .ok_or(Error::EngineContract(
                "physical producer returned an unknown RGB surface encoding",
            ))?;
        let expected = FrameSurfaceLayout::with_encoding(
            extent,
            self.extras.len(),
            encoding,
            &self.backend.device().limits(),
        )?;
        if outputs.len() != 1 + self.extras.len()
            || outputs.iter().zip(expected.layouts()).enumerate().any(
                |(index, (output, layout))| {
                    output.id != jxl_gpu_protocol::OutputId(index as u32)
                        || &output.layout != layout
                        || output.buffer.size() < expected.storage_bytes
                        || output.buffer.as_wgpu_buffer() != first.buffer.as_wgpu_buffer()
                },
            )
        {
            return Err(Error::EngineContract(
                "physical producer returned an invalid all-channel frame surface",
            ));
        }
        Ok(Surface {
            buffer: outputs.remove(0).buffer,
            extent,
            plane_words: (expected.plane_bytes / 4) as u32,
            encoding,
        })
    }

    pub(super) fn completed_surface(&self, buffer: GpuBufferLease) -> Surface {
        Surface {
            buffer,
            extent: self.canvas,
            plane_words: (self.surface.plane_bytes / 4) as u32,
            encoding: FrameSurfaceEncoding::Srgb,
        }
    }

    pub(super) fn blend(
        &self,
        foreground: &Surface,
        references: &[Option<Surface>; 4],
        frame: &FrameInventory,
    ) -> Result<GpuWork> {
        let references: &[Option<Surface>; 4] = &std::array::from_fn(|index| {
            std::iter::once(&frame.color_blend)
                .chain(&frame.extra_channel_blends)
                .any(|blend| blend.source as usize == index)
                .then(|| references[index].clone())
                .flatten()
        });
        if foreground.encoding != FrameSurfaceEncoding::Srgb
            || references
                .iter()
                .flatten()
                .any(|base| base.encoding != FrameSurfaceEncoding::Srgb)
        {
            return Err(Error::EngineContract(
                "blending requires original-encoding RGB surfaces",
            ));
        }
        if foreground.extent != Extent2d::new(frame.width, frame.height)
            || references.iter().flatten().any(|base| {
                base.extent.width < self.canvas.width || base.extent.height < self.canvas.height
            })
        {
            return Err(Error::EngineContract(
                "composition surface geometry disagrees with the frame plan",
            ));
        }
        let channels = blend_channels(frame, &self.extras)?;
        let (intersection, origin) = intersection(self.canvas, frame);
        let params = BlendParams {
            canvas: [
                self.canvas.width,
                self.canvas.height,
                (self.surface.plane_bytes / 4) as u32,
                3 + self.extras.len() as u32,
            ],
            intersection,
            source: [
                origin[0],
                origin[1],
                foreground.extent.width,
                foreground.plane_words,
            ],
            dispatch: [self.blend_dispatch[0] * 64, 0, 0, 0],
            references: std::array::from_fn(|index| {
                references[index].as_ref().map_or([0; 4], |surface| {
                    [
                        surface.extent.width,
                        surface.extent.height,
                        surface.plane_words,
                        1,
                    ]
                })
            }),
        };
        let inputs: Vec<_> = std::iter::once((0, &foreground.buffer))
            .chain(references.iter().enumerate().map(|(index, reference)| {
                (
                    (index + 1) as u32,
                    &reference.as_ref().unwrap_or(foreground).buffer,
                )
            }))
            .collect();
        submit(
            &self.backend,
            Submission {
                pipeline: &self.blend,
                params: bytemuck::bytes_of(&params),
                inputs: &inputs,
                metadata: Some((6, bytemuck::cast_slice(&channels))),
                output_binding: 5,
                uniform_binding: 7,
                size: self.surface.storage_bytes,
                dispatch: self.blend_dispatch,
            },
        )
    }

    pub(super) fn pack(&self, source: &Surface) -> Result<GpuWork> {
        if source.encoding == FrameSurfaceEncoding::Encoded {
            return Err(Error::EngineContract(
                "codec components cannot be presented as RGB",
            ));
        }
        if source.extent != self.canvas
            || source.plane_words != (self.surface.plane_bytes / 4) as u32
        {
            return Err(Error::EngineContract(
                "presentation canvas has the wrong extent or plane stride",
            ));
        }
        let mut native;
        let (params, output_binding, uniform_binding): (&[u8], _, _) = match &self.packing {
            Packing::Color(params) => (
                bytemuck::bytes_of(
                    &params[usize::from(source.encoding == FrameSurfaceEncoding::Linear)],
                ),
                3,
                4,
            ),
            Packing::Native(params) => {
                native = *params;
                native.source[3] |= u32::from(source.encoding == FrameSurfaceEncoding::Linear) << 1;
                (bytemuck::bytes_of(&native), 1, 2)
            }
        };
        submit(
            &self.backend,
            Submission {
                pipeline: &self.pack,
                params,
                inputs: &[(0, &source.buffer)],
                metadata: (!self.spots.is_empty()).then(|| (7, bytemuck::cast_slice(&self.spots))),
                output_binding,
                uniform_binding,
                size: aligned(self.layout.logical_size)?,
                dispatch: self.output_dispatch,
            },
        )
    }
}

pub(super) fn pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

fn address_error() -> Error {
    Error::CompositionResourceLimit {
        resource: "byte addressing",
        requested: u64::MAX,
        limit: u64::from(u32::MAX - 3),
    }
}

fn aligned(size: u64) -> Result<u64> {
    size.checked_add(3)
        .map(|n| n & !3)
        .ok_or_else(address_error)
}

pub(super) fn dispatch(device: &wgpu::Device, elements: u64) -> Result<[u32; 2]> {
    let limit = u64::from(device.limits().max_compute_workgroups_per_dimension);
    if elements == 0 || limit == 0 {
        return Err(Error::EngineContract(
            "composition requires a nonempty dispatch",
        ));
    }
    let x = elements.div_ceil(64).min(limit);
    let y = elements.div_ceil(x * 64);
    if y > limit {
        return Err(Error::CompositionResourceLimit {
            resource: "dispatch rows",
            requested: y,
            limit,
        });
    }
    Ok([x as u32, y as u32])
}
