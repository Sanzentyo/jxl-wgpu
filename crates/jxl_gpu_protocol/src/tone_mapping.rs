//! Explicit image and display luminance; these types never evaluate image samples.
use crate::DisplayIntensity;

#[cfg(test)]
mod tests;

/// Finite nonnegative black and positive unit-white luminance, in cd/m².
/// Equal endpoints represent a degenerate range and are retained without division.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LuminanceRange {
    black: u32,
    white: DisplayIntensity,
}

impl LuminanceRange {
    #[must_use]
    pub const fn new(black_nits: f32, white_nits: f32) -> Option<Self> {
        let Some(white) = DisplayIntensity::new(white_nits) else {
            return None;
        };
        if black_nits.is_finite() && black_nits >= 0.0 && black_nits <= white_nits {
            Some(Self {
                black: if black_nits == 0.0 {
                    0
                } else {
                    black_nits.to_bits()
                },
                white,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn black_nits(self) -> f32 {
        f32::from_bits(self.black)
    }

    #[must_use]
    pub const fn white(self) -> DisplayIntensity {
        self.white
    }
}

/// BT.2408 luminance mapping with the JPEG XL protected linear-light region.
/// Input/output linear unit whites are the source/target range maxima. The protected
/// threshold is absolute nits; a frontend resolves a relative threshold against target white.
/// Its F64 metadata preserves the exact product of a binary16 relative threshold and F32 display
/// white, before the backend lowers the final comparison boundary to GPU precision.
/// Gamut conversion and output quantization remain separate operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ToneMapping {
    source: LuminanceRange,
    target: LuminanceRange,
    linear_below: u64,
}

impl ToneMapping {
    #[must_use]
    pub const fn new(
        source: LuminanceRange,
        target: LuminanceRange,
        linear_below_nits: f64,
    ) -> Option<Self> {
        if linear_below_nits.is_finite() && linear_below_nits >= 0.0 {
            Some(Self {
                source,
                target,
                linear_below: if linear_below_nits == 0.0 {
                    0
                } else {
                    linear_below_nits.to_bits()
                },
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn source(self) -> LuminanceRange {
        self.source
    }

    #[must_use]
    pub const fn target(self) -> LuminanceRange {
        self.target
    }

    #[must_use]
    pub const fn linear_below_nits(self) -> f64 {
        f64::from_bits(self.linear_below)
    }
}
