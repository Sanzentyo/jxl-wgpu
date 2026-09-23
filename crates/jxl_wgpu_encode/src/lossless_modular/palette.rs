use super::{LosslessModularFormat, LosslessModularPredictor};
use crate::EncodeError;

/// Builds an exact first-occurrence palette of selected components in each pass group on the GPU.
/// RCT precedes this transform; optional Squeeze operates on the index and unselected channels. A fused
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
    component_range: Option<(u32, u32)>,
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
            component_range: None,
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
            component_range: None,
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
            component_range: None,
        })
    }

    /// Uses exact implicit cube colors or signed delta entries before storing explicit residuals.
    /// The nonzero delta limit includes a zero entry which keeps single-channel streams on the
    /// interoperable delta inverse path. Cube selection uses working depths up to 24 bits;
    /// wider words still use implicit signed deltas or exact explicit residuals. No sample is
    /// quantized. All selected components must match the implicit tuple exactly.
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

    /// Selects a nonempty contiguous range of post-RCT components. By default all source
    /// components participate. Unselected components remain independent image channels;
    /// optional Squeeze selects from them and the index channel, skipping the table.
    /// The range must fit the four-component input domain and each submitted source format.
    /// Implicit entry components are relative to this range, including when selecting alpha.
    pub fn with_components(mut self, begin: u32, count: u32) -> Result<Self, EncodeError> {
        validate_components(begin, count, 4)?;
        self.component_range = Some((begin, count));
        Ok(self)
    }

    /// Explicit post-RCT component range, or `None` when all source components are selected.
    #[must_use]
    pub fn component_range(self) -> Option<std::ops::Range<u32>> {
        self.component_range
            .map(|(begin, count)| begin..begin + count)
    }

    pub(super) const fn begin(self) -> u32 {
        match self.component_range {
            Some((begin, _)) => begin,
            None => 0,
        }
    }

    pub(super) const fn components(self, format: LosslessModularFormat) -> u32 {
        match self.component_range {
            Some((_, count)) => count,
            None => format.channel_count(),
        }
    }

    pub(super) fn validate(self, format: LosslessModularFormat) -> Result<(), EncodeError> {
        validate_components(
            self.begin(),
            self.components(format),
            format.channel_count(),
        )
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
}

fn validate_components(begin: u32, count: u32, channels: u32) -> Result<(), EncodeError> {
    if count != 0 && begin.checked_add(count).is_some_and(|end| end <= channels) {
        Ok(())
    } else {
        Err(EncodeError::InvalidModularPaletteComponents {
            begin,
            count,
            channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossless_modular::transform::{ModularTransformPlan, PlannedPalette};
    use crate::lossless_modular::{
        LosslessModularConfig, LosslessModularGroupGrid, LosslessModularGroupSize,
    };

    fn planned(policy: LosslessModularPalette, width: u32, height: u32) -> PlannedPalette {
        let grid = LosslessModularGroupGrid::for_extent(
            width,
            height,
            LosslessModularGroupSize::Pixels1024,
        )
        .unwrap();
        let plan = ModularTransformPlan::new(
            grid,
            LosslessModularFormat::Rgba,
            31,
            0,
            LosslessModularConfig {
                palette: Some(policy),
                ..Default::default()
            },
        )
        .unwrap();
        plan.group(grid.group(0).unwrap()).unwrap().palette.unwrap()
    }

    #[test]
    fn component_ranges_reject_empty_overflow_and_missing_source_components() {
        for palette in [
            LosslessModularPalette::new(4).unwrap(),
            LosslessModularPalette::deltas(4, LosslessModularPredictor::Weighted).unwrap(),
            LosslessModularPalette::mixed(4, 4, LosslessModularPredictor::West).unwrap(),
            LosslessModularPalette::implicit(4, LosslessModularPredictor::Zero).unwrap(),
        ] {
            assert_eq!(palette.component_range(), None);
            for (begin, count) in [(0, 0), (0, 5), (1, 4), (4, 1), (u32::MAX, 1), (1, u32::MAX)] {
                assert!(matches!(
                    palette.with_components(begin, count),
                    Err(EncodeError::InvalidModularPaletteComponents { .. })
                ));
            }
            for begin in 0..4 {
                for count in 1..=4 - begin {
                    let selected = palette.with_components(begin, count).unwrap();
                    assert_eq!(selected.component_range(), Some(begin..begin + count));
                    for format in [
                        LosslessModularFormat::Gray,
                        LosslessModularFormat::GrayAlpha,
                        LosslessModularFormat::Rgb,
                        LosslessModularFormat::Rgba,
                    ] {
                        assert_eq!(
                            selected.validate(format).is_ok(),
                            begin + count <= format.channel_count()
                        );
                    }
                }
            }
        }
    }

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
                    assert_eq!(
                        planned(palette, 1024, 1024).capacity.entries(),
                        colors + deltas
                    );
                    assert_eq!(planned(palette, 1, 1).capacity.entries(), 2);
                    assert_eq!(planned(palette, 1, 1).capacity.deltas, 1);
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
                assert_eq!(planned(palette, 1, 1).capacity.entries(), limit.min(2));
                assert_eq!(planned(palette, 1024, 1024).capacity.entries(), limit);
                assert!(planned(palette, 1024, 1024).hash_entries >= 2 * (limit + 143));
                assert!(planned(palette, 1024, 1024).hash_entries.is_power_of_two());
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
