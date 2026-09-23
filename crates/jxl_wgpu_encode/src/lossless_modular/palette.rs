use super::{LosslessModularFormat, LosslessModularPredictor, LosslessModularSqueeze};
use crate::EncodeError;

/// Builds an exact first-occurrence palette of all components in each pass group on the GPU.
/// RCT precedes this transform; optional Squeeze operates on its index channel. A fused
/// single-group frame declares the palette in DC-global. Exceeding the caller's color limit
/// fails completion without returning a codestream. [`Self::deltas`] dictionaries contain
/// predictor residuals instead of colors; [`Self::mixed`] retains both. [`Self::implicit`]
/// also selects exact implicit entries while keeping an explicit residual dictionary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularPalette {
    max_colors: u32,
    max_deltas: u32,
    delta_predictor: Option<LosslessModularPredictor>,
    implicit: bool,
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
            max_deltas: 0,
            delta_predictor: None,
            implicit: false,
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
            max_deltas,
            delta_predictor: Some(predictor),
            implicit: false,
        })
    }

    /// Keeps the first `max_colors` distinct post-RCT tuples as absolute entries, then uses
    /// exact predictor residuals for every other tuple. Absolute matches always take priority.
    /// Both dictionaries reset per group and have independent nonzero limits. Weighted state
    /// observes every original sample, including samples represented by absolute entries.
    pub fn mixed(
        max_colors: u32,
        max_deltas: u32,
        predictor: LosslessModularPredictor,
    ) -> Result<Self, EncodeError> {
        Self::new(max_colors)?;
        Self::deltas(max_deltas, predictor)?;
        Ok(Self {
            max_colors: max_colors + max_deltas,
            max_deltas,
            delta_predictor: Some(predictor),
            implicit: false,
        })
    }

    /// Uses exact implicit cube colors or signed delta entries before storing explicit residuals.
    /// The nonzero delta limit includes a zero entry which keeps single-channel streams on the
    /// interoperable delta inverse path. Cube selection uses working depths up to 24 bits;
    /// wider words still use implicit signed deltas or exact explicit residuals. No sample is
    /// quantized. All components, including alpha, must match the implicit tuple exactly.
    pub fn implicit(
        max_deltas: u32,
        predictor: LosslessModularPredictor,
    ) -> Result<Self, EncodeError> {
        Ok(Self {
            implicit: true,
            ..Self::deltas(max_deltas, predictor)?
        })
    }

    #[must_use]
    pub const fn uses_implicit_entries(self) -> bool {
        self.implicit
    }

    /// Maximum total dictionary entries (colors plus residual tuples for a mixed palette).
    #[must_use]
    pub const fn max_colors(self) -> u32 {
        self.max_colors
    }

    /// Maximum residual entries; zero for a color-only palette.
    #[must_use]
    pub const fn max_deltas(self) -> u32 {
        self.max_deltas
    }

    #[must_use]
    pub const fn delta_predictor(self) -> Option<LosslessModularPredictor> {
        self.delta_predictor
    }

    pub(super) fn capacity(self, width: u32, height: u32) -> u32 {
        (self.max_colors - self.max_deltas).min(width * height) + self.delta_capacity(width, height)
    }

    pub(super) fn delta_capacity(self, width: u32, height: u32) -> u32 {
        self.max_deltas
            .min(width * height + u32::from(self.implicit))
    }

    pub(super) fn hash_entries(self, capacity: u32) -> u32 {
        (2 * (capacity + if self.implicit { 143 } else { 0 })).next_power_of_two()
    }

    pub(super) fn scratch_words(
        self,
        capacity: u32,
        components: u32,
        width: u32,
        height: u32,
    ) -> u64 {
        // Total/delta counts, a channel-major table with delta/color partitions, and a
        // <= 50%-full open-addressed hash table. Equal words in different partitions differ.
        2 + u64::from(capacity) * u64::from(components)
            + u64::from(self.hash_entries(capacity))
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PaletteCounts {
    pub(super) entries: u32,
    pub(super) deltas: u32,
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

    #[test]
    fn mixed_palette_limits_are_independent_and_preserve_the_combined_capacity() {
        for predictor in LosslessModularPredictor::ALL {
            for colors in [
                1,
                255,
                256,
                1279,
                1280,
                5375,
                5376,
                LosslessModularPalette::MAX_COLORS,
            ] {
                for deltas in [1, 256, 257, 1280, 1281, LosslessModularPalette::MAX_DELTAS] {
                    let palette = LosslessModularPalette::mixed(colors, deltas, predictor).unwrap();
                    assert_eq!(palette.max_colors(), colors + deltas);
                    assert_eq!(palette.max_deltas(), deltas);
                    assert_eq!(palette.delta_predictor(), Some(predictor));
                    assert_eq!(palette.capacity(1024, 1024), colors + deltas);
                    assert_eq!(palette.capacity(1, 1), 2);
                    assert_eq!(palette.delta_capacity(1, 1), 1);
                }
            }
        }
        for colors in [0, LosslessModularPalette::MAX_COLORS + 1, u32::MAX] {
            assert!(
                matches!(LosslessModularPalette::mixed(colors, 1, LosslessModularPredictor::Zero),
                Err(EncodeError::InvalidModularPaletteLimit { max_colors }) if max_colors == colors)
            );
        }
        for deltas in [0, LosslessModularPalette::MAX_DELTAS + 1, u32::MAX] {
            assert!(
                matches!(LosslessModularPalette::mixed(1, deltas, LosslessModularPredictor::Zero),
                Err(EncodeError::InvalidModularPaletteDeltaLimit { max_deltas }) if max_deltas == deltas)
            );
        }
    }

    #[test]
    fn implicit_entries_reserve_the_zero_anchor_and_a_bounded_shared_hash() {
        for predictor in LosslessModularPredictor::ALL {
            for limit in [
                1,
                2,
                256,
                257,
                1280,
                1281,
                LosslessModularPalette::MAX_DELTAS,
            ] {
                let palette = LosslessModularPalette::implicit(limit, predictor).unwrap();
                assert!(palette.uses_implicit_entries());
                assert_eq!(palette.max_deltas(), limit);
                assert_eq!(palette.delta_predictor(), Some(predictor));
                assert_eq!(palette.capacity(1, 1), limit.min(2));
                assert_eq!(palette.capacity(1024, 1024), limit);
                assert!(palette.hash_entries(limit) >= 2 * (limit + 143));
                assert!(palette.hash_entries(limit).is_power_of_two());
            }
        }
        for limit in [0, LosslessModularPalette::MAX_DELTAS + 1, u32::MAX] {
            assert!(
                matches!(LosslessModularPalette::implicit(limit, LosslessModularPredictor::Zero),
                Err(EncodeError::InvalidModularPaletteDeltaLimit { max_deltas }) if max_deltas == limit)
            );
        }
        assert!(
            !LosslessModularPalette::new(1)
                .unwrap()
                .uses_implicit_entries()
        );
        assert!(
            !LosslessModularPalette::deltas(1, LosslessModularPredictor::Zero)
                .unwrap()
                .uses_implicit_entries()
        );
        assert!(
            !LosslessModularPalette::mixed(1, 1, LosslessModularPredictor::Zero)
                .unwrap()
                .uses_implicit_entries()
        );
    }
}
