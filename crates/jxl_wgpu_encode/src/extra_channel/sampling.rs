//! The frame header and scalar dispatch share one resolved sampling contract.
use std::sync::Arc;

use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;

use super::{ExtraChannelUpsampling, MAX_EXTRA_CHANNELS};
use crate::{EncodeError, sample_format::ImageSamplePlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SampledExtraChannel {
    /// None names the full-resolution alpha in the primary color source.
    pub(crate) source: Option<usize>,
    pub(crate) extent: Extent2d,
    pub(crate) shift: u8,
    upsampling: ExtraChannelUpsampling,
}

/// Caller policy is resolved before source binding, GPU allocation or wire emission.
/// All entries are bounded by the image's 256-channel profile limit, independent of pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtraChannelSamplingPlan {
    pub(crate) channels: Arc<[SampledExtraChannel]>,
}

impl ExtraChannelSamplingPlan {
    pub(crate) fn for_image(
        image: &ImageSamplePlan,
        extent: Extent2d,
        requested: &[ExtraChannelUpsampling],
    ) -> Result<Self, EncodeError> {
        Self::new(
            extent,
            image
                .extra_channels
                .iter()
                .map(|channel| channel.dimension_shift()),
            image.alpha.is_some(),
            requested,
        )
    }

    pub(crate) fn for_packed_alpha(
        extent: Extent2d,
        has_alpha: bool,
        requested: &[ExtraChannelUpsampling],
    ) -> Result<Self, EncodeError> {
        Self::new(
            extent,
            (0..usize::from(has_alpha)).map(|_| 0),
            has_alpha,
            requested,
        )
    }

    fn new(
        extent: Extent2d,
        shifts: impl ExactSizeIterator<Item = u8>,
        packed_alpha: bool,
        requested: &[ExtraChannelUpsampling],
    ) -> Result<Self, EncodeError> {
        let count = shifts.len();
        if count > MAX_EXTRA_CHANNELS || (!requested.is_empty() && requested.len() != count) {
            return Err(EncodeError::InvalidConfiguration(
                "extra-channel upsampling count differs from the image declaration",
            ));
        }
        if packed_alpha
            && requested
                .first()
                .is_some_and(|&value| value != ExtraChannelUpsampling::One)
        {
            return Err(EncodeError::InvalidConfiguration(
                "packed alpha requires a per-frame upsampling factor of one",
            ));
        }
        let channels = shifts
            .enumerate()
            .map(|(index, intrinsic)| {
                let upsampling = requested.get(index).copied().unwrap_or_default();
                let shift = intrinsic + upsampling.shift();
                // ExtraChannel checks intrinsic shifts <= 3; the wire factor contributes <= 3.
                let factor = 1u32 << shift;
                SampledExtraChannel {
                    source: index.checked_sub(usize::from(packed_alpha)),
                    extent: Extent2d::new(
                        extent.width.div_ceil(factor),
                        extent.height.div_ceil(factor),
                    ),
                    shift,
                    upsampling,
                }
            })
            .collect();
        Ok(Self { channels })
    }

    pub(crate) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        for channel in &*self.channels {
            output.write_bits(u64::from(channel.upsampling.shift()), 2)?;
        }
        Ok(())
    }
}
