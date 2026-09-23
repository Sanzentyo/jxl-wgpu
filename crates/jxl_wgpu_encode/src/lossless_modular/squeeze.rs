use super::LosslessModularFormat;

/// Explicit separable Squeeze of every image channel within each Modular pass group, after
/// RCT and optional Palette. Palette's meta channel is excluded.
/// A fused single-group frame declares the same operation in DC-global. Residual channels are
/// appended in source-channel order. Axes of length one are skipped per group. No cross-group
/// filtering or image analysis is performed. The default preserves the existing unsqueezed bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum LosslessModularSqueeze {
    #[default]
    None = 0,
    Horizontal = 1,
    Vertical = 2,
    HorizontalThenVertical = 3,
    VerticalThenHorizontal = 4,
}

impl LosslessModularSqueeze {
    pub(super) const fn for_extent(self, width: u32, height: u32) -> Self {
        match self {
            Self::Horizontal if width == 1 => Self::None,
            Self::Vertical if height == 1 => Self::None,
            Self::HorizontalThenVertical | Self::VerticalThenHorizontal => {
                match (width > 1, height > 1) {
                    (true, true) => self,
                    (true, false) => Self::Horizontal,
                    (false, true) => Self::Vertical,
                    (false, false) => Self::None,
                }
            }
            _ => self,
        }
    }

    pub(super) const fn stages(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Horizontal | Self::Vertical => 1,
            Self::HorizontalThenVertical | Self::VerticalThenHorizontal => 2,
        }
    }

    pub(super) const fn first_horizontal(self) -> bool {
        matches!(self, Self::Horizontal | Self::HorizontalThenVertical)
    }

    pub(super) const fn channels(self, format: LosslessModularFormat) -> u32 {
        format.channel_count() << self.stages()
    }

    pub(super) fn extent(self, source: [u32; 2], channel: u32, components: u32) -> [u32; 2] {
        let mut extent = source;
        for stage in 0..self.stages() {
            let horizontal = self.first_horizontal() ^ (stage != 0);
            let axis = usize::from(!horizontal);
            let residual = (channel / components) & (1 << stage) != 0;
            extent[axis] = if residual {
                extent[axis] / 2
            } else {
                extent[axis].div_ceil(2)
            };
        }
        extent
    }
}
