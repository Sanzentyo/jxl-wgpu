//! Color presentation and independently selected scalar output share frame validation.
use super::types::VarDctDecodeError;
use crate::color_output::{ColorOutputConfig, ColorOutputPlan, ColorOutputTransform, InverseOpsin};
use crate::modular_scalar_output::{ModularScalarOutputConfig, ModularScalarOutputPlan};
use crate::vardct_frontend::VarDctColorTransform;
use crate::{GpuOutputMapping, GpuOutputRequest};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_wgpu::{KernelVariant, WgpuBackend};
use std::sync::Arc;

#[derive(Clone)]
pub(super) enum VarDctFrameOutput {
    Image(VarDctImageOutput),
    Extra {
        index: u32,
        plan: ModularScalarOutputPlan,
    },
}

#[derive(Clone)]
pub(super) enum VarDctImageOutput {
    Color {
        config: ColorOutputConfig,
        plan: ColorOutputPlan,
    },
    Components {
        extent: Extent2d,
        storage_bytes: u64,
    },
}

impl VarDctImageOutput {
    pub(super) fn extent(&self) -> Extent2d {
        match self {
            Self::Color { config, .. } => config.extent,
            Self::Components { extent, .. } => *extent,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct FrameOutputMemory {
    pub storage_bytes: u64,
    pub uniform_bytes: u64,
    pub status_bytes: u64,
}

impl VarDctFrameOutput {
    pub(super) fn requires_reconstruction(&self) -> bool {
        matches!(self, Self::Image(_))
    }
    pub(super) fn retains_components(&self) -> bool {
        matches!(self, Self::Image(VarDctImageOutput::Components { .. }))
    }
    pub(super) fn memory(&self) -> FrameOutputMemory {
        match self {
            Self::Image(VarDctImageOutput::Color { plan, .. }) => FrameOutputMemory {
                storage_bytes: plan.memory.output_storage_bytes,
                uniform_bytes: plan.memory.uniform_bytes,
                status_bytes: 0,
            },
            Self::Image(VarDctImageOutput::Components { storage_bytes, .. }) => FrameOutputMemory {
                storage_bytes: *storage_bytes,
                uniform_bytes: 0,
                status_bytes: 0,
            },
            Self::Extra { plan, .. } => FrameOutputMemory {
                storage_bytes: plan.storage_bytes,
                uniform_bytes: ModularScalarOutputPlan::UNIFORM_BYTES,
                status_bytes: ModularScalarOutputPlan::STATUS_BYTES,
            },
        }
    }
}

pub(super) struct VarDctPresentation {
    pub(super) output: VarDctFrameOutput,
    pub(super) layout: ImageLayout,
    pub(super) quant_biases: [f32; 4],
    pub(super) surface: Option<Arc<crate::frame_surface::FrameSurfaceLayout>>,
}

pub(super) fn prepare_presentation(
    backend: &WgpuBackend,
    inventory: &jxl_gpu_bitstream::CodestreamInventory,
    request: &GpuOutputRequest,
    profile: &crate::vardct_frontend::StandardVarDctProfile,
    variant: KernelVariant,
) -> Result<VarDctPresentation, VarDctDecodeError> {
    let orientation = OutputOrientation::from_exif_value(inventory.image_header.orientation)
        .ok_or(VarDctDecodeError::InvalidOrientation {
            orientation: inventory.image_header.orientation,
        })?;
    let orientation = request.orientation_policy().resolve(orientation);
    if let Some(index) = request.extra_channel() {
        let extra = inventory
            .image_header
            .extra_channels
            .get(index as usize)
            .ok_or(VarDctDecodeError::ExtraChannelIndex {
                index,
                count: inventory.image_header.extra_channels.len(),
            })?;
        let encoding = crate::modular_sample::ModularSampleEncoding::new(extra.bit_depth)
            .ok_or(VarDctDecodeError::UnsupportedOutput)?;
        let GpuOutputMapping::Numeric(mapping) = request.mapping() else {
            return Err(VarDctDecodeError::UnsupportedOutput);
        };
        let config = ModularScalarOutputConfig {
            extent: Extent2d::new(profile.output_width, profile.output_height),
            orientation,
            encoding,
            mapping,
        };
        let layout = ImageLayout::packed(
            orientation.map_extent(config.extent),
            request.format().clone(),
        )?;
        let plan =
            ModularScalarOutputPlan::new(config, &layout, &backend.device().limits(), variant)?;
        return Ok(VarDctPresentation {
            output: VarDctFrameOutput::Extra { index, plan },
            layout,
            quant_biases: [0.0; 4],
            surface: None,
        });
    }
    if request.mapping() != GpuOutputMapping::Color {
        return Err(VarDctDecodeError::UnsupportedOutput);
    }

    let quant_biases = match profile.color_transform {
        VarDctColorTransform::Xyb => {
            let opsin = inventory
                .image_header
                .opsin_inverse_matrix
                .ok_or(VarDctDecodeError::MissingInverseOpsin)?;
            [
                opsin.quant_bias[0].to_f32(),
                opsin.quant_bias[1].to_f32(),
                opsin.quant_bias[2].to_f32(),
                opsin.quant_bias_numerator.to_f32(),
            ]
        }
        VarDctColorTransform::Ycbcr | VarDctColorTransform::Rgb => {
            // Non-XYB image metadata omits the optional opsin object that otherwise carries these
            // TransformData defaults, but VarDCT coefficient biasing still uses their exact F32
            // roundings.
            [
                1.0 - 0.054_650_072,
                1.0 - 0.070_054_5,
                1.0 - 0.049_935_102,
                0.145,
            ]
        }
    };
    let extent = if request.defers_frame_features() {
        Extent2d::new(profile.width, profile.height)
    } else {
        Extent2d::new(profile.output_width, profile.output_height)
    };
    let surface = request
        .retains_frame_surface()
        .then(|| {
            let frame = &inventory.frames[0];
            let resampling = crate::frame_resampling::FrameResampling::new(
                Extent2d::new(profile.output_width, profile.output_height),
                profile.upsampling,
                &frame.extra_channel_upsampling,
            );
            // LF prediction can retain only XYB. Match the normalized-extra selection;
            // optional previews and producer features request their extra planes explicitly.
            let extra_factors: &[u32] = if profile.lf_level == 0 || request.retains_lf_extras() {
                &frame.extra_channel_upsampling
            } else {
                &[]
            };
            crate::frame_surface::FrameSurfaceLayout::with_extra_extents(
                orientation.map_extent(extent),
                extra_factors.iter().map(|&factor| {
                    resampling
                        .extra(factor, request.frame_render_stage())
                        .extent
                }),
                request.frame_surface_encoding(),
                &backend.device().limits(),
            )
            .map(Arc::new)
        })
        .transpose()?;
    let layout = match &surface {
        Some(surface) => surface.color.clone(),
        None => ImageLayout::packed(orientation.map_extent(extent), request.format().clone())?,
    };
    let output = if request.retains_frame_surface()
        && request.frame_surface_encoding() == crate::frame_surface::FrameSurfaceEncoding::Encoded
    {
        if orientation != OutputOrientation::Identity {
            return Err(VarDctDecodeError::UnsupportedOutput);
        }
        VarDctImageOutput::Components {
            extent,
            storage_bytes: layout.logical_size,
        }
    } else {
        let original = crate::image_color::original_encoding(&inventory.image_header)
            .ok_or(VarDctDecodeError::UnsupportedColorEncoding)?;
        let transform = match profile.color_transform {
            VarDctColorTransform::Xyb => ColorOutputTransform::Xyb(
                InverseOpsin::from_image(&inventory.image_header)
                    .ok_or(VarDctDecodeError::MissingInverseOpsin)?,
            ),
            VarDctColorTransform::Rgb => ColorOutputTransform::Rgb(original),
            VarDctColorTransform::Ycbcr => ColorOutputTransform::Ycbcr {
                channel_shifts: profile.channel_shifts,
                encoding: original.into(),
            },
        };
        let config = ColorOutputConfig {
            intensity_target: inventory
                .image_header
                .tone_mapping
                .intensity_target
                .to_f32(),
            linear_black_threshold: if inventory.image_header.xyb_encoded {
                crate::image_color::reconstruction_black_threshold(
                    original,
                    &request.format().color_spec,
                )
            } else {
                None
            },
            white_point_adaptation: request.white_point_adaptation(),
            extent,
            orientation,
            transform,
            alpha_conversion: if profile.lf_level != 0 {
                jxl_wgpu::AlphaConversion::Preserve
            } else {
                request.alpha_conversion(&inventory.image_header.extra_channels)
            },
        };
        config.validate_layout(&layout)?;
        let plan =
            ColorOutputPlan::for_limits_with_variant(&layout, &backend.device().limits(), variant)?;
        VarDctImageOutput::Color { config, plan }
    };
    Ok(VarDctPresentation {
        output: VarDctFrameOutput::Image(output),
        layout,
        quant_biases,
        surface,
    })
}

pub(super) fn selected_extra_indices(
    request: &GpuOutputRequest,
    extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
    profile: &crate::vardct_frontend::StandardVarDctProfile,
) -> Vec<usize> {
    // All LF extras validate. Patches and intermediate presentation need normalized planes too.
    if profile.lf_level != 0 && !request.retains_lf_extras() {
        return Vec::new();
    }
    if request.retains_frame_surface() || request.retains_lf_extras() {
        return (0..extras.len()).collect();
    }
    request
        .extra_channel()
        .map(|index| index as usize)
        .or_else(|| {
            extras.iter().position(|extra| {
                matches!(
                    extra.channel_type,
                    jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { .. }
                )
            })
        })
        .into_iter()
        .collect()
}
