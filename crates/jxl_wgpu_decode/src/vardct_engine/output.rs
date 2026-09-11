//! Color presentation and independently selected scalar output share frame validation.
use super::types::VarDctDecodeError;
use crate::color_output::{ColorOutputConfig, ColorOutputPlan, ColorOutputTransform, InverseOpsin};
use crate::modular_scalar_output::{ModularScalarOutputConfig, ModularScalarOutputPlan};
use crate::vardct_frontend::VarDctColorTransform;
use crate::{GpuOutputMapping, GpuOutputRequest};
use jxl_gpu_bitstream::{
    ColourEncodingInventory, ColourSpaceInventory, PrimariesInventory, TransferFunctionInventory,
    WhitePointInventory,
};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{Extent2d, OutputOrientation, RgbColorEncoding};
use jxl_wgpu::{KernelVariant, WgpuBackend};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum VarDctFrameOutput {
    Color {
        config: ColorOutputConfig,
        plan: ColorOutputPlan,
    },
    Extra {
        index: u32,
        plan: ModularScalarOutputPlan,
    },
}

#[derive(Clone, Copy)]
pub(super) struct FrameOutputMemory {
    pub storage_bytes: u64,
    pub uniform_bytes: u64,
    pub status_bytes: u64,
}

impl VarDctFrameOutput {
    pub(super) fn is_color(self) -> bool {
        matches!(self, Self::Color { .. })
    }
    pub(super) fn memory(self) -> FrameOutputMemory {
        match self {
            Self::Color { plan, .. } => FrameOutputMemory {
                storage_bytes: plan.memory.output_storage_bytes,
                uniform_bytes: plan.memory.uniform_bytes,
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

    if !matches!(
        inventory.image_header.colour_encoding,
        ColourEncodingInventory::Enumerated {
            colour_space: ColourSpaceInventory::Rgb | ColourSpaceInventory::Grey,
            white_point: WhitePointInventory::D65,
            primaries: PrimariesInventory::Srgb,
            transfer_function: TransferFunctionInventory::Srgb,
            rendering_intent: _,
        }
    ) {
        return Err(VarDctDecodeError::UnsupportedColorEncoding);
    }
    let (output_transform, quant_biases) = match profile.color_transform {
        VarDctColorTransform::Xyb => {
            let opsin = inventory
                .image_header
                .opsin_inverse_matrix
                .ok_or(VarDctDecodeError::MissingInverseOpsin)?;
            // Gray projection and stream-selected inverse opsin are shared with Modular.
            (
                ColorOutputTransform::Xyb(
                    InverseOpsin::from_image(&inventory.image_header)
                        .ok_or(VarDctDecodeError::MissingInverseOpsin)?,
                ),
                [
                    opsin.quant_bias[0].to_f32(),
                    opsin.quant_bias[1].to_f32(),
                    opsin.quant_bias[2].to_f32(),
                    opsin.quant_bias_numerator.to_f32(),
                ],
            )
        }
        VarDctColorTransform::Ycbcr | VarDctColorTransform::Rgb => (
            if profile.color_transform == VarDctColorTransform::Rgb {
                ColorOutputTransform::Rgb(RgbColorEncoding::SRGB_BT709)
            } else {
                ColorOutputTransform::Ycbcr {
                    // Execution resolves these shifts from the planes that reach presentation.
                    channel_shifts: profile.channel_shifts,
                }
            },
            // Non-XYB image metadata omits the optional opsin object that otherwise carries these
            // TransformData defaults, but VarDCT coefficient biasing still uses their exact F32
            // roundings.
            [
                1.0 - 0.054_650_072,
                1.0 - 0.070_054_5,
                1.0 - 0.049_935_102,
                0.145,
            ],
        ),
    };
    let output_config = ColorOutputConfig {
        extent: Extent2d::new(profile.output_width, profile.output_height),
        orientation,
        transform: output_transform,
        alpha_conversion: if profile.lf_level != 0 {
            jxl_wgpu::AlphaConversion::Preserve
        } else {
            request.alpha_conversion(&inventory.image_header.extra_channels)
        },
    };
    let surface = request
        .retains_frame_surface()
        .then(|| {
            crate::frame_surface::FrameSurfaceLayout::with_encoding(
                output_config.output_extent(),
                inventory.image_header.extra_channels.len(),
                request.frame_surface_encoding(),
                &backend.device().limits(),
            )
            .map(Arc::new)
        })
        .transpose()?;
    let layout = match &surface {
        Some(surface) => surface.color.clone(),
        None => ImageLayout::packed(output_config.output_extent(), request.format().clone())?,
    };
    output_config.validate_layout(&layout)?;
    let output_plan =
        ColorOutputPlan::for_limits_with_variant(&layout, &backend.device().limits(), variant)?;
    Ok(VarDctPresentation {
        output: VarDctFrameOutput::Color {
            config: output_config,
            plan: output_plan,
        },
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
    // All LF extras validate. Only an intermediate presentation retains their normalized planes.
    if profile.lf_level != 0 && !request.retains_lf_presentation() {
        return Vec::new();
    }
    if request.retains_frame_surface() || request.retains_lf_presentation() {
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
