//! One checked geometry contract for the frame header and both codec backends.
use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;

use crate::extra_channel::sampling::ExtraChannelSamplingPlan;
use crate::{EncodeError, FrameEncodeRequest, sample_format::ImageSamplePlan};

/// JPEG XL's per-frame reconstruction factor. The encoder accepts already reduced
/// samples and signals reconstruction by the decoder; it does not resize source pixels.
/// Extra channels additionally apply their image-wide intrinsic dimension shift.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UpsamplingFactor {
    #[default]
    One,
    Two,
    Four,
    Eight,
}

impl UpsamplingFactor {
    #[must_use]
    pub const fn factor(self) -> u32 {
        1 << self.shift()
    }

    /// Required supplied color extent for a frame's displayed rectangle.
    #[must_use]
    pub const fn source_extent(self, frame: Extent2d) -> Extent2d {
        Extent2d::new(
            frame.width.div_ceil(self.factor()),
            frame.height.div_ceil(self.factor()),
        )
    }

    pub(crate) const fn shift(self) -> u8 {
        match self {
            Self::One => 0,
            Self::Two => 1,
            Self::Four => 2,
            Self::Eight => 3,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FrameSamplingPlan {
    pub(crate) color_extent: Extent2d,
    pub(crate) extras: ExtraChannelSamplingPlan,
    factor: UpsamplingFactor,
}

impl FrameSamplingPlan {
    pub(crate) fn for_request(
        request: &FrameEncodeRequest,
        source_extent: (u32, u32),
        intrinsic_shifts: impl ExactSizeIterator<Item = u8>,
        packed_alpha: bool,
    ) -> Result<Self, EncodeError> {
        let frame = request.options.crop.map_or(
            Extent2d::new(request.canvas_width, request.canvas_height),
            |crop| Extent2d::new(crop.width(), crop.height()),
        );
        let factor = request.options.upsampling;
        let color_extent = factor.source_extent(frame);
        if color_extent != Extent2d::new(source_extent.0, source_extent.1) {
            return Err(EncodeError::InvalidConfiguration(
                "GPU source extent must equal the frame rectangle divided by its upsampling factor (rounded up)",
            ));
        }
        Ok(Self {
            color_extent,
            factor,
            extras: ExtraChannelSamplingPlan::new(
                frame,
                intrinsic_shifts,
                packed_alpha,
                factor,
                &request.options.extra_channel_upsampling,
            )?,
        })
    }

    /// The source-only memory query has no presented rectangle or requested sampling.
    pub(crate) fn unscaled(
        extent: Extent2d,
        samples: &ImageSamplePlan,
    ) -> Result<Self, EncodeError> {
        Ok(Self {
            color_extent: extent,
            factor: UpsamplingFactor::One,
            extras: ExtraChannelSamplingPlan::new(
                extent,
                samples
                    .extra_channels
                    .iter()
                    .map(|channel| channel.dimension_shift()),
                samples.alpha.is_some(),
                UpsamplingFactor::One,
                &[],
            )?,
        })
    }

    pub(crate) const fn is_unscaled(&self) -> bool {
        matches!(self.factor, UpsamplingFactor::One)
    }

    pub(crate) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        output.write_bits(u64::from(self.factor.shift()), 2)?;
        self.extras.write(output)
    }
}
