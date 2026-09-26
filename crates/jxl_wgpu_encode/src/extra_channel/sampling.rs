//! The frame header and scalar dispatch share one resolved sampling contract.
use std::sync::Arc;

use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;

use super::MAX_EXTRA_CHANNELS;
use crate::{EncodeError, UpsamplingFactor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SampledExtraChannel {
    /// None names the alpha in the primary color source, at the color sample extent.
    pub(crate) source: Option<usize>,
    pub(crate) extent: Extent2d,
    /// Relative to the coded color grid, for LF/pass routing and group geometry.
    pub(crate) shift: u8,
    upsampling: UpsamplingFactor,
}

/// Caller policy is resolved before source binding, GPU allocation or wire emission.
/// All entries are bounded by the image's 256-channel profile limit, independent of pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtraChannelSamplingPlan {
    pub(crate) channels: Arc<[SampledExtraChannel]>,
}

impl ExtraChannelSamplingPlan {
    pub(crate) fn new(
        extent: Extent2d,
        samples: &crate::sample_format::ImageSamplePlan,
        color: UpsamplingFactor,
        requested: &[UpsamplingFactor],
    ) -> Result<Self, EncodeError> {
        let count = samples.extra_channels.len();
        if count > MAX_EXTRA_CHANNELS || (!requested.is_empty() && requested.len() != count) {
            return Err(EncodeError::InvalidConfiguration(
                "extra-channel upsampling count differs from the image declaration",
            ));
        }
        let primary_count = usize::from(samples.alpha.is_some()) + usize::from(samples.cmyk);
        if requested
            .iter()
            .take(primary_count)
            .any(|&value| value != color)
        {
            return Err(EncodeError::InvalidConfiguration(
                "primary alpha and Black must use the color upsampling factor",
            ));
        }
        let channels = samples
            .extra_channels
            .iter()
            .map(|channel| channel.dimension_shift())
            .enumerate()
            .map(|(index, intrinsic)| {
                let upsampling = requested.get(index).copied().unwrap_or(color);
                let effective_shift = intrinsic + upsampling.shift();
                let shift = effective_shift.checked_sub(color.shift()).ok_or(
                    EncodeError::InvalidConfiguration(
                        "effective extra-channel upsampling must be at least the color factor",
                    ),
                )?;
                // ExtraChannel checks intrinsic shifts <= 3; the wire factor contributes <= 3.
                let factor = 1u32 << effective_shift;
                Ok(SampledExtraChannel {
                    source: index.checked_sub(usize::from(samples.alpha.is_some())),
                    extent: Extent2d::new(
                        extent.width.div_ceil(factor),
                        extent.height.div_ceil(factor),
                    ),
                    shift,
                    upsampling,
                })
            })
            .collect::<Result<_, EncodeError>>()?;
        Ok(Self { channels })
    }

    pub(crate) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        for channel in &*self.channels {
            output.write_bits(u64::from(channel.upsampling.shift()), 2)?;
        }
        Ok(())
    }
}
