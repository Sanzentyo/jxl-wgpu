//! Source-channel grids shared by the Modular frame and VarDCT extra-channel streams.

use jxl_gpu_bitstream::{FrameInventory, ImageHeaderInventory, SampleBitDepth};

use crate::modular_transform::{
    ModularChannelGeometry, ModularChannelTopology, ModularTransformLimits,
};
use crate::{Error, Result};

pub(crate) fn source_topology(
    image: &ImageHeaderInventory,
    frame: &FrameInventory,
    color_channels: u32,
) -> Result<ModularChannelTopology> {
    frame.validate_jpeg_sampling()?;
    if !matches!(color_channels, 0 | 1 | 3) {
        return Err(Error::EngineContract(
            "invalid Modular color component count",
        ));
    }
    let (width, height) = frame
        .color_sample_extent()
        .ok_or(Error::EngineContract("invalid Modular color grid"))?;
    let bits_per_sample = match image.bit_depth {
        SampleBitDepth::Integer { bits_per_sample }
        | SampleBitDepth::Float {
            bits_per_sample, ..
        } => bits_per_sample,
    };
    if !matches!(frame.upsampling, 1 | 2 | 4 | 8)
        || frame.extra_channel_upsampling.len() != image.extra_channels.len()
    {
        return Err(Error::EngineContract("invalid Modular upsampling factors"));
    }
    let lf_factor = frame
        .lf_level
        .checked_mul(3)
        .and_then(|shift| 1u32.checked_shl(shift))
        .ok_or(Error::EngineContract("invalid Modular LF grid"))?;
    let output_width = frame.width.div_ceil(lf_factor);
    let output_height = frame.height.div_ceil(lf_factor);
    let shifts = crate::jpeg_sampling::component_shifts(frame.jpeg_upsampling);
    let mut channels: Vec<_> = shifts[..color_channels as usize]
        .iter()
        .map(|shift| {
            let [width, height] = shift
                .shifted_extent(width, height)
                .expect("validated JPEG component sampling");
            ModularChannelGeometry::new(
                width,
                height,
                shift.horizontal as i32,
                shift.vertical as i32,
                bits_per_sample,
            )
        })
        .collect();
    for &factor in &frame.extra_channel_upsampling {
        if !factor.is_power_of_two() || factor > 64 || factor < frame.upsampling {
            return Err(Error::EngineContract("invalid Modular extra-channel grid"));
        }
        let shift = (factor.ilog2() - frame.upsampling.ilog2()) as i32;
        channels.push(ModularChannelGeometry::new(
            output_width.div_ceil(factor),
            output_height.div_ceil(factor),
            shift,
            shift,
            bits_per_sample,
        ));
    }
    ModularChannelTopology::new(channels, 0, ModularTransformLimits::default())
}
