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
    let (width, height) = frame
        .color_sample_extent()
        .ok_or(Error::EngineContract("invalid Modular color grid"))?;
    let SampleBitDepth::Integer { bits_per_sample } = image.bit_depth else {
        return Err(Error::EngineContract(
            "Modular integer topology requires integer precision",
        ));
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
    let mut channels = vec![
        ModularChannelGeometry::new(width, height, 0, 0, bits_per_sample);
        color_channels as usize
    ];
    for &factor in &frame.extra_channel_upsampling {
        if !matches!(factor, 1 | 2 | 4 | 8) || factor < frame.upsampling {
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
