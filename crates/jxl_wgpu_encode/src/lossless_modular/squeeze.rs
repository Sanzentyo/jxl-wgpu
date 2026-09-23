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
}
