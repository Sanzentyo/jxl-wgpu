//! Color parameters shared by render plans, pixel layouts and codec frontends.

pub(crate) mod matrix;
pub use matrix::{ColorMatrix, ColorMatrixError};

/// Finite CIE xy coordinates. Equality and hashing preserve the exact declared values.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Chromaticity {
    x: u64,
    y: u64,
}

impl Chromaticity {
    pub const D65: Self = Self::constant(0.3127, 0.3290);
    /// D50 from ICC's exact s15Fixed16 PCS illuminant (0xf6d6, 0x10000, 0xd32d).
    pub const ICC_D50: Self = Self::constant(
        0xf6d6 as f64 / (0xf6d6 + 0x10000 + 0xd32d) as f64,
        0x10000 as f64 / (0xf6d6 + 0x10000 + 0xd32d) as f64,
    );
    pub const E: Self = Self::constant(1.0 / 3.0, 1.0 / 3.0);
    pub const DCI: Self = Self::constant(0.314, 0.351);

    /// Creates a finite coordinate pair. Matrix construction separately validates the geometry.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Option<Self> {
        if x.is_finite() && y.is_finite() {
            Some(Self::constant(x, y))
        } else {
            None
        }
    }

    const fn constant(x: f64, y: f64) -> Self {
        Self {
            x: if x == 0.0 { 0 } else { x.to_bits() },
            y: if y == 0.0 { 0 } else { y.to_bits() },
        }
    }

    #[must_use]
    pub const fn x(self) -> f64 {
        f64::from_bits(self.x)
    }

    #[must_use]
    pub const fn y(self) -> f64 {
        f64::from_bits(self.y)
    }
}

/// RGB primaries together with their reference white point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbChromaticities {
    pub red: Chromaticity,
    pub green: Chromaticity,
    pub blue: Chromaticity,
    pub white: Chromaticity,
}

impl RgbChromaticities {
    pub const BT709: Self = Self {
        red: Chromaticity::constant(0.64, 0.33),
        green: Chromaticity::constant(0.30, 0.60),
        blue: Chromaticity::constant(0.15, 0.06),
        white: Chromaticity::D65,
    };
    pub const BT2020: Self = Self {
        red: Chromaticity::constant(0.708, 0.292),
        green: Chromaticity::constant(0.170, 0.797),
        blue: Chromaticity::constant(0.131, 0.046),
        white: Chromaticity::D65,
    };
    pub const DISPLAY_P3: Self = Self {
        red: Chromaticity::constant(0.680, 0.320),
        green: Chromaticity::constant(0.265, 0.690),
        blue: Chromaticity::constant(0.150, 0.060),
        white: Chromaticity::D65,
    };
}

/// A positive, finite exponent that maps linear-light values to gamma-encoded samples.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GammaExponent(u32);

impl GammaExponent {
    #[must_use]
    pub const fn new(value: f32) -> Option<Self> {
        if value.is_finite() && value > 0.0 && (1.0 / value).is_finite() {
            Some(Self(value.to_bits()))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn value(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// Positive finite luminance, in cd/m², represented by unit display-linear RGB.
/// This declares the image white; it does not select tone mapping or a display peak.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DisplayIntensity(u32);

impl DisplayIntensity {
    #[must_use]
    pub const fn new(nits: f32) -> Option<Self> {
        if nits.is_finite() && nits > 0.0 {
            Some(Self(nits.to_bits()))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn nits(self) -> f32 {
        f32::from_bits(self.0)
    }
}

impl std::fmt::Debug for DisplayIntensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DisplayIntensity")
            .field(&self.nits())
            .finish()
    }
}

/// Treatment of the reference whites during a colorimetric RGB conversion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WhitePointAdaptation {
    /// Adapt source white to target white using the Bradford cone-response matrix.
    #[default]
    Bradford,
    /// Preserve absolute XYZ values, including the difference between reference whites.
    None,
}

impl std::fmt::Debug for Chromaticity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chromaticity")
            .field("x", &self.x())
            .field("y", &self.y())
            .finish()
    }
}
impl std::fmt::Debug for GammaExponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GammaExponent").field(&self.value()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_parameters_reject_nonfinite_values_and_canonicalize_signed_zero() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Chromaticity::new(value, 0.3).is_none());
            assert!(Chromaticity::new(0.3, value).is_none());
        }
        assert_eq!(Chromaticity::new(-0.0, 0.0), Chromaticity::new(0.0, -0.0));
        for value in [0.0, -0.0, -1.0, f32::from_bits(1), f32::NAN, f32::INFINITY] {
            assert!(GammaExponent::new(value).is_none());
        }
        for value in [0.5, 1.0, 2.2] {
            assert_eq!(GammaExponent::new(value).unwrap().value(), value);
        }
    }
}
