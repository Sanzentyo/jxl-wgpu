use super::{LosslessModularFormat, LosslessModularSqueeze};
use crate::EncodeError;

/// Builds an exact first-occurrence palette of all components in each pass group on the GPU.
/// RCT precedes this transform; optional Squeeze operates on its index channel. A fused
/// single-group frame declares the palette in DC-global. Exceeding the caller's color limit
/// fails completion without returning a codestream. Delta and implicit palette entries are
/// not selected by this policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularPalette {
    max_colors: u32,
}

impl LosslessModularPalette {
    /// Largest explicit color count in the standard Palette header.
    pub const MAX_COLORS: u32 = 5376 + 65535;

    pub fn new(max_colors: u32) -> Result<Self, EncodeError> {
        if !(1..=Self::MAX_COLORS).contains(&max_colors) {
            return Err(EncodeError::InvalidModularPaletteLimit { max_colors });
        }
        Ok(Self { max_colors })
    }

    #[must_use]
    pub const fn max_colors(self) -> u32 {
        self.max_colors
    }

    pub(super) fn capacity(self, width: u32, height: u32) -> u32 {
        self.max_colors.min(width * height)
    }

    pub(super) fn scratch_words(capacity: u32, components: u32) -> u64 {
        // One count, a channel-major word table, and a <= 50%-full open-addressed hash table.
        1 + u64::from(capacity) * u64::from(components)
            + u64::from((capacity * 2).next_power_of_two())
    }
}

pub(super) const fn encoded_channels(
    format: LosslessModularFormat,
    squeeze: LosslessModularSqueeze,
    palette: Option<LosslessModularPalette>,
) -> u32 {
    if palette.is_some() {
        1 + (1 << squeeze.stages())
    } else {
        squeeze.channels(format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_limits_cover_the_complete_wire_domain() {
        for max_colors in [1, 255, 256, 1279, 1280, 5375, 5376, 70911] {
            assert_eq!(
                LosslessModularPalette::new(max_colors)
                    .unwrap()
                    .max_colors(),
                max_colors
            );
        }
        for max_colors in [0, 70912, u32::MAX] {
            assert!(
                matches!(LosslessModularPalette::new(max_colors), Err(EncodeError::InvalidModularPaletteLimit { max_colors: actual }) if actual == max_colors)
            );
        }
    }
}
