use std::borrow::Cow;
use std::sync::Arc;

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

mod icc;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Unrounded color in an explicit domain, followed by independently normalized extra planes.
#[derive(Clone, Debug)]
pub(super) struct Surface {
    pub(super) buffer: GpuBufferLease,
    pub(super) layout: Arc<FrameSurfaceLayout>,
    pub(super) encoding: FrameSurfaceEncoding,
}

impl Surface {
    pub(super) fn extent(&self) -> Extent2d {
        self.layout.color.extent
    }

    /// Uniform strides are only valid after channel-specific resampling has finished.
    pub(super) fn uniform_plane_words(&self) -> Result<u32> {
        if !self.layout.has_uniform_extent() {
            return Err(Error::EngineContract(
                "frame operation requires equal channel extents",
            ));
        }
        Ok((self.layout.color_plane_bytes / 4) as u32)
    }

    pub(super) fn copy_extras(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: jxl_wgpu::ResidentStorageBinding<'_>,
        layout: &FrameSurfaceLayout,
    ) -> Result<()> {
        if self.layout.extras.len() != layout.extras.len() {
            return Err(Error::EngineContract(
                "color conversion must preserve all extra channels",
            ));
        }
        for (input, destination) in self.layout.extras.iter().zip(&layout.extras) {
            let plane = &input.planes[0];
            crate::frame_surface::copy::planes(
                encoder,
                &[jxl_wgpu::ResidentF32Plane {
                    storage: jxl_wgpu::ResidentStorageBinding {
                        buffer: self.buffer.as_wgpu_buffer(),
                        offset: plane.offset,
                        size: std::num::NonZeroU64::new(plane.end_offset()? - plane.offset)
                            .expect("nonempty extra"),
                    },
                    width: input.extent.width,
                    height: input.extent.height,
                    stride: (plane.row_stride / 4) as u32,
                }],
                output,
                destination,
            )?;
        }
        Ok(())
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NativeParams {
    extent: [u32; 4],
    format: [u32; 4],
    output: [u32; 4],
    color: [u32; 4],
    source: [u32; 4],
    luminance: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<NativeParams>() == 96);
const _: () = assert!(std::mem::align_of::<NativeParams>() == 16);
const _: () = assert!(std::mem::offset_of!(NativeParams, color) == 48);
const _: () = assert!(std::mem::offset_of!(NativeParams, source) == 64);
const _: () = assert!(std::mem::offset_of!(NativeParams, luminance) == 80);

#[derive(Debug)]
enum RasterPacking {
    Color {
        original: Box<ImageOutputParams>,
        linear: Box<ImageOutputParams>,
    },
    Native(NativeParams),
}

#[derive(Debug)]
enum Packing {
    Raster {
        pipeline: wgpu::ComputePipeline,
        params: RasterPacking,
        spots: Vec<SpotColor>,
    },
    Icc(Vec<icc::Presentation>),
}

/// Color domains actually consumed by this image's presentation and reference plan.
#[derive(Clone, Copy)]
pub(super) struct ColorUsage {
    pub(super) original: bool,
    pub(super) linear: bool,
    pub(super) reconstruct_original: bool,
}

impl ColorUsage {
    pub(super) const ORIGINAL: Self = Self {
        original: true,
        linear: false,
        reconstruct_original: true,
    };
    pub(super) const LINEAR: Self = Self {
        original: false,
        linear: true,
        reconstruct_original: false,
    };
}

#[derive(Debug)]
pub(super) struct Compositor {
    backend: WgpuBackend,
    canvas: Extent2d,
    pub(super) original: FrameSurfaceEncoding,
    pub(super) reconstruction: Option<Arc<super::icc_transform::Transform>>,
    extras: Vec<ExtraChannelInventory>,
    surface: FrameSurfaceLayout,
    blend: wgpu::ComputePipeline,
    packing: Packing,
    pub(super) layout: ImageLayout,
    blend_dispatch: [u32; 2],
    output_dispatch: [u32; 2],
}

impl Compositor {
    pub(super) fn new(
        backend: WgpuBackend,
        canvas: Extent2d,
        image: &jxl_gpu_bitstream::ImageHeaderInventory,
        request: &GpuOutputRequest,
        usage: ColorUsage,
    ) -> Result<Self> {
        let extras = &image.extra_channels;
        let grayscale = image.grayscale;
        let sample_bit_depth = image.bit_depth;
        let original = crate::image_color::original_domain(image)?;
        let intensity_target = image.tone_mapping.intensity_target.to_f32();
        let intensity = jxl_gpu_protocol::DisplayIntensity::new(intensity_target)
            .ok_or(crate::color_output::ColorOutputError::InvalidIntensityTarget)?;
        let tone_mapping = crate::tone_mapping::for_image(image, request)?;
        let orientation = OutputOrientation::from_exif_value(image.orientation).ok_or(
            Error::InvalidImageOrientation {
                value: image.orientation,
            },
        )?;
        let orientation = request.orientation_policy().resolve(orientation);

        let layout = ImageLayout::packed(orientation.map_extent(canvas), request.format().clone())?;
        let device = backend.device();
        let surface = FrameSurfaceLayout::with_encoding(
            canvas,
            extras.len(),
            original.clone(),
            &device.limits(),
        )?;
        let color_count = surface.color.planes.len() as u32;
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
            u64::from(canvas.width)
                * u64::from(canvas.height)
                * (u64::from(color_count) + extras.len() as u64),
        )?;
        let output_dispatch = dispatch(device, output_size / 4)?;
        let first_alpha = extras.iter().enumerate().find_map(|(index, extra)| {
            if let ExtraChannelTypeInventory::Alpha { associated } = extra.channel_type {
                Some((index, associated))
            } else {
                None
            }
        });
        let alpha_channel = first_alpha.map_or(u32::MAX, |(index, _)| color_count + index as u32);
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
                    .map(|extra| (color_count + index, extra))
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
        let native = crate::model::native_modular_format(request.format()).filter(|_| {
            request.uses_original_sample_domain()
                || (tone_mapping.is_none()
                    && request.gamut_mapping().is_none()
                    && original.rgb_encoding()
                        == Some(jxl_gpu_protocol::RgbColorEncoding::SRGB_BT709))
        });
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
        let mut transforms = super::icc_transform::Transforms::default();
        let reconstruction = if image.xyb_encoded
            && usage.reconstruct_original
            && let FrameSurfaceEncoding::Icc(profile) = &original
        {
            Some(transforms.select(
                &backend,
                jxl_gpu_protocol::icc::IccTransform::from_rgb_with_intensity(
                    jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
                    intensity,
                    profile,
                    profile.header().rendering_intent,
                )?,
            )?)
        } else {
            None
        };
        let packing = if (original.icc_profile().is_some()
            || matches!(
                request.format().color_spec,
                jxl_gpu_formats::ColorSpecification::Icc(_)
            ))
            && request.mapping() == crate::GpuOutputMapping::Color
        {
            let mut presentation = |encoding: FrameSurfaceEncoding| -> Result<_> {
                let source = FrameSurfaceLayout::with_encoding(
                    canvas,
                    extras.len(),
                    encoding.clone(),
                    &device.limits(),
                )?;
                icc::Presentation::new(
                    &backend,
                    &source,
                    encoding,
                    icc::Output::Color(request),
                    icc::ImageMetadata {
                        orientation,
                        intensity,
                        tone_mapping,
                        extras,
                    },
                    &mut transforms,
                )
            };
            let mut encodings = Vec::new();
            if usage.original {
                encodings.push(original.clone());
            }
            if usage.linear {
                let linear = FrameSurfaceEncoding::Rgb(original.rgb_encoding().map_or(
                    jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
                    crate::image_color::linear_encoding,
                ));
                // An originally linear image has one color encoding regardless of how a
                // physical frame reached it. Select by that encoding, not its history.
                if !encodings.contains(&linear) {
                    encodings.push(linear);
                }
            }
            Packing::Icc(
                encodings
                    .into_iter()
                    .map(&mut presentation)
                    .collect::<Result<_>>()?,
            )
        } else {
            let spots = if request.renders_spot_colors(extras) {
                spot_colors(extras, &surface.extras)
            } else {
                Vec::new()
            };
            if !spots.is_empty() {
                validate_size(device, std::mem::size_of_val(spots.as_slice()) as u64)?;
            }
            let spot_source = spot_shader(!spots.is_empty());
            let (packing, source) = if native.is_some() || scalar_float {
                if let (Some(native), Some((_, extra))) = (native, selected)
                    && (native.channels != crate::ModularChannels::Gray
                        || extra.bit_depth
                            != (SampleBitDepth::Integer {
                                bits_per_sample: u32::from(native.bits_per_sample),
                            }))
                {
                    return Err(Error::UnsupportedOutputFormat(
                        "native composed extra output must match its declared unsigned depth"
                            .into(),
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
                    RasterPacking::Native(NativeParams {
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
                            u32::try_from(layout.planes[0].row_stride)
                                .map_err(|_| address_error())?,
                        ],
                        output: [
                            u32::try_from(layout.logical_size).map_err(|_| address_error())?,
                            output_dispatch[0] * 64,
                            orientation.to_exif_value() - 1,
                            alpha_conversion as u32,
                        ],
                        color: [
                            match original.rgb_encoding().map(|encoding| encoding.transfer) {
                                None | Some(jxl_gpu_protocol::TransferFunction::Linear) => 0,
                                Some(jxl_gpu_protocol::TransferFunction::Srgb) => 1,
                                Some(jxl_gpu_protocol::TransferFunction::Bt709) => 2,
                                Some(jxl_gpu_protocol::TransferFunction::Pq) => 3,
                                Some(jxl_gpu_protocol::TransferFunction::Hlg) => 4,
                                Some(jxl_gpu_protocol::TransferFunction::Bt2020) => 5,
                                Some(jxl_gpu_protocol::TransferFunction::Gamma(_)) => 6,
                                Some(jxl_gpu_protocol::TransferFunction::Dci) => 7,
                            },
                            match original.rgb_encoding().map(|encoding| encoding.transfer) {
                                Some(jxl_gpu_protocol::TransferFunction::Gamma(exponent)) => {
                                    exponent.value().to_bits()
                                }
                                _ => 1.0f32.to_bits(),
                            },
                            color_count,
                            intensity_target.to_bits(),
                        ],
                        source: [
                            (surface.color_plane_bytes / 4) as u32,
                            alpha_channel,
                            selected.map_or(color_channel.unwrap_or(0), |(index, _)| index),
                            u32::from(scalar_float),
                        ],
                        luminance: original.rgb_encoding().map_or(Ok([0.0; 4]), |encoding| {
                            jxl_wgpu::display_luminance(encoding.space, intensity_target, true)
                        })?,
                    }),
                    native_shader(spot_source),
                )
            } else {
                if request.mapping() != crate::GpuOutputMapping::Color {
                    return Err(Error::UnsupportedOutputFormat("composed numeric samples require matching native unsigned or scalar normalized F32 output".into()));
                }
                let original = original
                    .rgb_encoding()
                    .ok_or(Error::EngineContract("enumerated packing requires RGB"))?;
                let params = |encoding: FrameSurfaceEncoding| -> Result<ImageOutputParams> {
                    let source = ImageOutputSource {
                        extent: canvas,
                        orientation,
                        strides: [canvas.width; 3],
                        encoding: encoding
                            .rgb_encoding()
                            .ok_or(Error::EngineContract("packing requires RGB"))?,
                    };
                    let params = match tone_mapping {
                        Some(mapping) => ImageOutputParams::new_with_tone_mapping(
                            &layout,
                            source,
                            output_dispatch[0] * 64,
                            request.white_point_adaptation(),
                            mapping,
                        ),
                        None => ImageOutputParams::new_with_intensity_target(
                            &layout,
                            source,
                            output_dispatch[0] * 64,
                            request.white_point_adaptation(),
                            intensity_target,
                        ),
                    }?
                    .with_gamut_mapping(request.gamut_mapping())?
                    .with_alpha_conversion(alpha_conversion);
                    Ok(
                        if encoding
                            == FrameSurfaceEncoding::Rgb(crate::image_color::linear_encoding(
                                original,
                            ))
                        {
                            if let Some(threshold) =
                                crate::image_color::reconstruction_black_threshold(
                                    original,
                                    &layout.format.color_spec,
                                )
                            {
                                params.with_linear_black_threshold(threshold)?
                            } else {
                                params
                            }
                        } else {
                            params
                        },
                    )
                };
                (
                    RasterPacking::Color {
                        original: Box::new(params(FrameSurfaceEncoding::Rgb(original))?),
                        linear: Box::new(params(FrameSurfaceEncoding::Rgb(
                            crate::image_color::linear_encoding(original),
                        ))?),
                    },
                    format!(
                        "{IMAGE_OUTPUT_SHADER}\n{spot_source}\n{}",
                        include_str!("output.wgsl")
                    ),
                )
            };
            let constants = if matches!(packing, RasterPacking::Color { .. }) {
                vec![
                    ("wg_x", 64.0),
                    ("wg_y", 1.0),
                    (
                        "surface_plane_words",
                        (surface.color_plane_bytes / 4) as f64,
                    ),
                    ("surface_alpha_channel", f64::from(alpha_channel)),
                ]
            } else {
                Vec::new()
            };
            if let RasterPacking::Native(params) = packing
                && color_channel.is_some()
                && let Some(profile) = original.icc_profile()
                && usage.linear
            {
                let linear =
                    FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709);
                let encodings = if usage.original {
                    vec![original.clone(), linear]
                } else {
                    vec![linear]
                };
                Packing::Icc(
                    encodings
                        .into_iter()
                        .map(|encoding| {
                            let source = FrameSurfaceLayout::with_encoding(
                                canvas,
                                extras.len(),
                                encoding.clone(),
                                &device.limits(),
                            )?;
                            icc::Presentation::new(
                                &backend,
                                &source,
                                encoding,
                                icc::Output::Numeric {
                                    request,
                                    params,
                                    profile,
                                },
                                icc::ImageMetadata {
                                    orientation,
                                    intensity,
                                    tone_mapping,
                                    extras,
                                },
                                &mut transforms,
                            )
                        })
                        .collect::<Result<_>>()?,
                )
            } else {
                let pipeline =
                    pipeline(device, "JPEG XL composed frame output", &source, &constants);
                Packing::Raster {
                    pipeline,
                    params: packing,
                    spots,
                }
            }
        };
        let blend = pipeline(
            device,
            "JPEG XL frame composition",
            include_str!("blend.wgsl"),
            &[],
        );
        Ok(Self {
            backend,
            canvas,
            original,
            reconstruction,
            extras: extras.to_vec(),
            surface,
            blend,
            packing,
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
                "physical producer returned an unknown surface color encoding",
            ))?;
        let expected = FrameSurfaceLayout::with_extra_extents(
            extent,
            outputs.iter().skip(1).map(|output| output.layout.extent),
            encoding.clone(),
            &self.backend.device().limits(),
        )?;
        if outputs.len() != 1 + self.extras.len()
            || (encoding != FrameSurfaceEncoding::Encoded && !expected.has_uniform_extent())
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
            layout: Arc::new(expected),
            encoding,
        })
    }

    pub(super) fn linear_encoding(&self) -> FrameSurfaceEncoding {
        match &self.original {
            FrameSurfaceEncoding::Rgb(original) => {
                FrameSurfaceEncoding::Rgb(crate::image_color::linear_encoding(*original))
            }
            FrameSurfaceEncoding::Icc(_)
            | FrameSurfaceEncoding::Device(_)
            | FrameSurfaceEncoding::Cmyk { .. } => {
                FrameSurfaceEncoding::Rgb(jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709)
            }
            FrameSurfaceEncoding::Encoded => unreachable!("original image color domain"),
        }
    }

    pub(super) fn completed_surface(&self, buffer: GpuBufferLease) -> Surface {
        Surface {
            buffer,
            layout: Arc::new(self.surface.clone()),
            encoding: self.original.clone(),
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
        if foreground.encoding != self.original
            || references
                .iter()
                .flatten()
                .any(|base| base.encoding != self.original)
        {
            return Err(Error::EngineContract(
                "blending requires original-color surfaces",
            ));
        }
        if foreground.extent() != Extent2d::new(frame.width, frame.height)
            || references.iter().flatten().any(|base| {
                base.extent().width < self.canvas.width || base.extent().height < self.canvas.height
            })
        {
            return Err(Error::EngineContract(
                "composition surface geometry disagrees with the frame plan",
            ));
        }
        let foreground_words = foreground.uniform_plane_words()?;
        let mut reference_geometry = [[0; 4]; 4];
        for (geometry, reference) in reference_geometry.iter_mut().zip(references) {
            if let Some(surface) = reference {
                *geometry = [
                    surface.extent().width,
                    surface.extent().height,
                    surface.uniform_plane_words()?,
                    1,
                ];
            }
        }
        let channels = blend_channels(frame, self.surface.color.planes.len(), &self.extras)?;
        let (intersection, origin) = intersection(self.canvas, frame);
        let params = BlendParams {
            canvas: [
                self.canvas.width,
                self.canvas.height,
                (self.surface.color_plane_bytes / 4) as u32,
                self.surface.color.planes.len() as u32 + self.extras.len() as u32,
            ],
            intersection,
            source: [
                origin[0],
                origin[1],
                foreground.extent().width,
                foreground_words,
            ],
            dispatch: [self.blend_dispatch[0] * 64, 0, 0, 0],
            references: reference_geometry,
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
        if source.encoding != self.original && source.encoding != self.linear_encoding() {
            return Err(Error::EngineContract(
                "presentation source is outside its original color domain",
            ));
        }
        if source.extent() != self.canvas
            || source.uniform_plane_words()? != (self.surface.color_plane_bytes / 4) as u32
        {
            return Err(Error::EngineContract(
                "presentation canvas has the wrong extent or plane stride",
            ));
        }
        let Packing::Raster {
            pipeline,
            params: packing,
            spots,
        } = &self.packing
        else {
            let Packing::Icc(presentations) = &self.packing else {
                unreachable!()
            };
            let icc = presentations
                .iter()
                .find(|presentation| presentation.source_encoding == source.encoding)
                .ok_or(Error::EngineContract("unplanned presentation color domain"))?;
            return icc.pack(&self.backend, source);
        };
        let mut native;
        let (params, output_binding, uniform_binding): (&[u8], _, _) = match packing {
            RasterPacking::Color { original, linear } => (
                bytemuck::bytes_of(if source.encoding == self.original {
                    original.as_ref()
                } else {
                    linear.as_ref()
                }),
                3,
                4,
            ),
            RasterPacking::Native(params) => {
                native = *params;
                let original_colors = native.color[2];
                let source_colors = source.layout.color.planes.len() as u32;
                if source.encoding != self.original
                    && self.original.icc_profile().is_some()
                    && native.source[2] < original_colors
                {
                    return Err(Error::EngineContract(
                        "original ICC color samples require device reconstruction",
                    ));
                }
                // LF previews can carry linear RGB even when the original ICC is Gray.
                // Extra selections and the first alpha follow the actual color planes.
                if native.source[1] != u32::MAX {
                    native.source[1] = native.source[1] - original_colors + source_colors;
                }
                if native.source[2] >= original_colors {
                    native.source[2] = native.source[2] - original_colors + source_colors;
                }
                native.color[2] = source_colors;
                native.source[3] |= u32::from(source.encoding != self.original) << 1;
                (bytemuck::bytes_of(&native), 1, 2)
            }
        };
        submit(
            &self.backend,
            Submission {
                pipeline,
                params,
                inputs: &[(0, &source.buffer)],
                metadata: (!spots.is_empty()).then(|| (7, bytemuck::cast_slice(spots))),
                output_binding,
                uniform_binding,
                size: aligned(self.layout.logical_size)?,
                dispatch: self.output_dispatch,
            },
        )
    }
}

fn native_shader(spot_source: &str) -> String {
    format!(
        "{IMAGE_ORIENTATION_SHADER}\n{}\n{}\n{spot_source}\n{}",
        jxl_wgpu::ALPHA_OUTPUT_SHADER,
        jxl_wgpu::IMAGE_TRANSFER_SHADER,
        crate::modular_sample::shader(include_str!("native.wgsl"))
    )
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
