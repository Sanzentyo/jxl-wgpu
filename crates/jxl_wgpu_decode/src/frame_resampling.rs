//! JPEG XL's single-filter resampling schedule around frame features.

use jxl_gpu_protocol::Extent2d;

use crate::frame_surface::FrameRenderStage;

/// One channel's reconstruction target and the filter applied to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChannelResampling {
    pub extent: Extent2d,
    pub factor: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameResampling {
    output_extent: Extent2d,
    color_factor: u32,
    late_extra_factor: u32,
}

impl FrameResampling {
    pub(crate) fn new(output_extent: Extent2d, color_factor: u32, extra_factors: &[u32]) -> Self {
        // All extras move together across the patch/spline boundary. Splitting an extra's
        // filter into an early quotient and a late color factor changes its samples.
        let late_extra_factor = if extra_factors.iter().all(|&factor| factor == color_factor) {
            color_factor
        } else {
            1
        };
        Self {
            output_extent,
            color_factor,
            late_extra_factor,
        }
    }

    pub(crate) fn color(self, stage: FrameRenderStage) -> ChannelResampling {
        self.channel(self.color_factor, self.color_factor, stage)
    }

    pub(crate) fn extra(self, factor: u32, stage: FrameRenderStage) -> ChannelResampling {
        self.channel(factor, self.late_extra_factor, stage)
    }

    pub(crate) fn late_extra_factor(self) -> u32 {
        self.late_extra_factor
    }

    pub(crate) fn extra_extent(self, stage: FrameRenderStage) -> Extent2d {
        self.extra(self.late_extra_factor, stage).extent
    }

    fn channel(self, factor: u32, late_factor: u32, stage: FrameRenderStage) -> ChannelResampling {
        let deferred = match stage {
            FrameRenderStage::Complete => 1,
            FrameRenderStage::BeforeFeatures => late_factor,
        };
        ChannelResampling {
            extent: Extent2d::new(
                self.output_extent.width.div_ceil(deferred),
                self.output_extent.height.div_ceil(deferred),
            ),
            factor: factor / deferred,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_channel_uses_one_full_filter_on_either_side_of_features() {
        let output = Extent2d::new(53, 37);
        for color_factor in [1, 2, 4, 8] {
            for extras in [vec![], vec![color_factor], vec![color_factor, 8]] {
                let plan = FrameResampling::new(output, color_factor, &extras);
                let color = plan.color(FrameRenderStage::BeforeFeatures);
                assert_eq!(color.factor, 1);
                assert_eq!(color.extent.width, output.width.div_ceil(color_factor));
                assert_eq!(plan.color(FrameRenderStage::Complete).extent, output);
                for factor in extras {
                    let early = plan.extra(factor, FrameRenderStage::BeforeFeatures);
                    assert!(early.factor == 1 || plan.late_extra_factor() == 1);
                    assert_eq!(early.factor * plan.late_extra_factor(), factor);
                    assert_eq!(
                        early.extent.width.div_ceil(early.factor),
                        output.width.div_ceil(factor)
                    );
                    assert_eq!(
                        early.extent.height.div_ceil(early.factor),
                        output.height.div_ceil(factor)
                    );
                    assert_eq!(
                        plan.extra(factor, FrameRenderStage::Complete),
                        ChannelResampling {
                            extent: output,
                            factor
                        }
                    );
                }
            }
        }
    }
}
