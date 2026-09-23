use super::{LosslessModularFormat, LosslessModularPredictor, LosslessModularSqueeze};
use crate::EncodeError;

/// Builds an exact first-occurrence palette of all components in each pass group on the GPU.
/// RCT precedes this transform; optional Squeeze operates on its index channel. A fused
/// single-group frame declares the palette in DC-global. Exceeding the caller's color limit
/// fails completion without returning a codestream. [`Self::deltas`] dictionaries contain
/// predictor residuals instead of colors. Implicit palette entries are not selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularPalette {
    max_colors: u32,
    delta_predictor: Option<LosslessModularPredictor>,
}

impl LosslessModularPalette {
    /// Largest explicit color count in the standard Palette header.
    pub const MAX_COLORS: u32 = 5376 + 65535;
    /// Largest explicit delta count in the standard Palette header.
    pub const MAX_DELTAS: u32 = 1281 + 65535;

    pub fn new(max_colors: u32) -> Result<Self, EncodeError> {
        if !(1..=Self::MAX_COLORS).contains(&max_colors) {
            return Err(EncodeError::InvalidModularPaletteLimit { max_colors });
        }
        Ok(Self {
            max_colors,
            delta_predictor: None,
        })
    }

    /// Builds a dictionary of exact wrapping residual tuples using any Modular predictor.
    /// The predictor operates independently on each post-RCT component and resets per group.
    /// Weighted uses the configuration's Weighted coefficients with its own row state.
    /// This predictor is independent of the predictor used to entropy-code table/index channels.
    pub fn deltas(
        max_deltas: u32,
        predictor: LosslessModularPredictor,
    ) -> Result<Self, EncodeError> {
        if !(1..=Self::MAX_DELTAS).contains(&max_deltas) {
            return Err(EncodeError::InvalidModularPaletteDeltaLimit { max_deltas });
        }
        Ok(Self {
            max_colors: max_deltas,
            delta_predictor: Some(predictor),
        })
    }

    /// Maximum dictionary entries (residual tuples for a delta palette).
    #[must_use]
    pub const fn max_colors(self) -> u32 {
        self.max_colors
    }

    #[must_use]
    pub const fn delta_predictor(self) -> Option<LosslessModularPredictor> {
        self.delta_predictor
    }

    pub(super) fn capacity(self, width: u32, height: u32) -> u32 {
        self.max_colors.min(width * height)
    }

    pub(super) fn scratch_words(
        self,
        capacity: u32,
        components: u32,
        width: u32,
        height: u32,
    ) -> u64 {
        // One count, a channel-major word table, and a <= 50%-full open-addressed hash table.
        1 + u64::from(capacity) * u64::from(components)
            + u64::from((capacity * 2).next_power_of_two())
            + if self.delta_predictor.is_some() {
                u64::from(width) * u64::from(height) * u64::from(components)
            } else {
                0
            }
            + if self.delta_predictor == Some(LosslessModularPredictor::Weighted) {
                5 * u64::from(width)
            } else {
                0
            }
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

    #[test]
    fn delta_palette_limits_cover_every_predictor_and_wire_bucket() {
        for predictor in LosslessModularPredictor::ALL {
            for max_deltas in [1, 256, 257, 1280, 1281, 66816] {
                let palette = LosslessModularPalette::deltas(max_deltas, predictor).unwrap();
                assert_eq!(palette.max_colors(), max_deltas);
                assert_eq!(palette.delta_predictor(), Some(predictor));
            }
        }
        for max_deltas in [0, 66817, 70911, u32::MAX] {
            assert!(
                matches!(LosslessModularPalette::deltas(max_deltas, LosslessModularPredictor::Zero),
                Err(EncodeError::InvalidModularPaletteDeltaLimit { max_deltas: actual }) if actual == max_deltas)
            );
        }
        assert_eq!(
            LosslessModularPalette::new(70911)
                .unwrap()
                .delta_predictor(),
            None
        );
    }
}
