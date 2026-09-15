//! Metadata-only selection of the rendition and its linear-light units.

use jxl_gpu_bitstream::gain_map::GainMapMetadata;
use jxl_gpu_protocol::DisplayIntensity;

use super::GainMapDecodeError;

/// Selects how much of the gain map to apply. Headroom is log2(nominal peak / reference white).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum GainMapRendition {
    /// Fully apply the map toward the alternate image, in either headroom direction.
    #[default]
    Alternate,
    /// Finite, nonnegative display headroom, in stops. Endpoints clamp to the base/alternate.
    DisplayHeadroom(f64),
}

/// Gain-map rendering policy, independent of the requested output format and transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GainMapRendering {
    pub rendition: GainMapRendition,
    /// Luminance represented by one in the gain equation, including its offsets.
    /// Defaults to 203 cd/m². Output linear units still follow the baseline image header.
    pub reference_white: DisplayIntensity,
}

impl Default for GainMapRendering {
    fn default() -> Self {
        Self {
            rendition: GainMapRendition::Alternate,
            reference_white: DisplayIntensity::new(203.0).expect("positive reference white"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Weight {
    Baseline,
    Apply(f32),
}

impl GainMapRendering {
    pub(super) fn weight(self, metadata: &GainMapMetadata) -> Result<Weight, GainMapDecodeError> {
        if let GainMapRendition::DisplayHeadroom(headroom) = self.rendition
            && (!headroom.is_finite() || headroom < 0.0)
        {
            return Err(GainMapDecodeError::InvalidRequest(
                "display headroom must be finite and nonnegative",
            ));
        }
        let base = metadata.base_hdr_headroom;
        let alternate = metadata.alternate_hdr_headroom;
        // Exact direction and separation, even when division rounds both fractions to one F64.
        let delta = i128::from(alternate.numerator) * i128::from(base.denominator)
            - i128::from(base.numerator) * i128::from(alternate.denominator);
        if delta == 0 {
            // Match libavif's identity policy for degenerate interpolation.
            return Ok(Weight::Baseline);
        }
        let sign = if delta < 0 { -1.0 } else { 1.0 };
        let GainMapRendition::DisplayHeadroom(headroom) = self.rendition else {
            return Ok(Weight::Apply(sign));
        };
        // FMA retains the residual against the exact rational endpoint. Subtracting two rounded
        // headrooms would erase it. Preserve Apply even if a positive weight underflows to F32
        // zero: the unequal offsets still apply at arbitrarily small nonzero interpolation.
        let distance = headroom.mul_add(f64::from(base.denominator), -f64::from(base.numerator));
        if distance == 0.0 || (distance < 0.0) != (delta < 0) {
            return Ok(Weight::Baseline);
        }
        let fraction = (distance * f64::from(alternate.denominator) / delta as f64).min(1.0);
        Ok(Weight::Apply(sign * fraction as f32))
    }
}

#[cfg(test)]
mod tests;
