//! Image-wide presentation declarations shared by every encoder and header writer.

use crate::EncodeError;
use jxl_gpu_bitstream::{BitWriter, FiniteF16};
use jxl_gpu_protocol::OutputOrientation;
use jxl_gpu_protocol::icc::IccRenderingIntent;

/// Image-wide declarations for encoded stills and animations.
///
/// These declare source color and presentation metadata; the coding transform applies separately.
/// They do not request tone mapping.
/// Primaries, white and transfer come from the source [`jxl_gpu_formats::PixelFormat`]. The default image white
/// is JPEG XL's 255 cd/m². HDR callers can supply their known image white explicitly.
///
/// ```
/// use jxl_gpu_bitstream::FiniteF16;
/// use jxl_wgpu_encode::{ImageOptions, IntrinsicSize, ToneMappingThreshold, VarDctConfig};
///
/// let options = ImageOptions {
///     intrinsic_size: Some(IntrinsicSize::new(1920, 1080)?),
///     intensity_target: FiniteF16::from_bits(0x63d0).unwrap(), // 1000 cd/m²
///     min_nits: FiniteF16::from_bits(0x2c00).unwrap(), // 1/16 cd/m²
///     linear_below: ToneMappingThreshold::DisplayFraction(
///         FiniteF16::from_bits(0x3000).unwrap(), // 1/8 of the requested display peak
///     ),
///     ..Default::default()
/// };
/// let config = VarDctConfig { image_options: options, ..Default::default() };
/// // LosslessModularEncoder::with_image_options accepts the same declaration.
/// # Ok::<(), jxl_wgpu_encode::EncodeError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageOptions {
    /// Presentation orientation, applied by the decoder after reconstruction/composition.
    /// Source pixels, canvas/crops, transform maps and memory plans stay in the encoded grid.
    pub orientation: OutputOrientation,
    /// For ICC input, this must equal the intent in the profile header; its original bytes
    /// are embedded unchanged. A conflicting declaration is rejected before admission.
    pub rendering_intent: IccRenderingIntent,
    /// Positive, exact binary16 luminance, in cd/m²; no implicit rounding is performed.
    pub intensity_target: FiniteF16,
    /// Nonnegative exact binary16 lower luminance bound, no greater than `intensity_target`.
    pub min_nits: FiniteF16,
    /// Protected light in the JPEG XL E.3 tone-mapping declaration; encoding does not apply it.
    pub linear_below: ToneMappingThreshold,
    /// Intended main-image display dimensions. Does not change source/crop/preview geometry.
    pub intrinsic_size: Option<crate::IntrinsicSize>,
}

/// Units of the protected-light threshold in image tone-mapping metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToneMappingThreshold {
    /// Absolute luminance, in cd/m². Must be nonnegative.
    AbsoluteNits(FiniteF16),
    /// Fraction of the requested display peak. Must be in 0..=1.
    DisplayFraction(FiniteF16),
}

impl Default for ToneMappingThreshold {
    fn default() -> Self {
        Self::AbsoluteNits(FiniteF16::from_bits(0).expect("zero is finite binary16"))
    }
}

impl ToneMappingThreshold {
    fn fields(self) -> (bool, FiniteF16) {
        match self {
            Self::AbsoluteNits(value) => (false, value),
            Self::DisplayFraction(value) => (true, value),
        }
    }
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            orientation: OutputOrientation::Identity,
            rendering_intent: IccRenderingIntent::Relative,
            intensity_target: FiniteF16::from_bits(0x5bf8).expect("255 is finite binary16"),
            min_nits: FiniteF16::from_bits(0).expect("zero is finite binary16"),
            linear_below: ToneMappingThreshold::default(),
            intrinsic_size: None,
        }
    }
}

impl ImageOptions {
    pub(crate) fn validate(self) -> Result<(), EncodeError> {
        if self.intensity_target.to_f32() <= 0.0 {
            return Err(EncodeError::InvalidConfiguration(
                "image intensity must be positive",
            ));
        }
        if self.min_nits.to_f32() < 0.0 || self.min_nits.to_f32() > self.intensity_target.to_f32() {
            return Err(EncodeError::InvalidConfiguration(
                "minimum image light must be between zero and the intensity target",
            ));
        }
        let (relative, threshold) = self.linear_below.fields();
        if threshold.to_f32() < 0.0 || (relative && threshold.to_f32() > 1.0) {
            return Err(EncodeError::InvalidConfiguration(
                "invalid protected-light threshold",
            ));
        }
        Ok(())
    }

    pub(crate) fn extra_fields(self) -> bool {
        self.orientation != OutputOrientation::Identity
            || self.intrinsic_size.is_some()
            || self.has_tone_metadata()
    }

    fn has_tone_metadata(self) -> bool {
        let defaults = Self::default();
        self.intensity_target != defaults.intensity_target
            || self.min_nits != defaults.min_nits
            || self.linear_below != defaults.linear_below
    }

    pub(crate) fn write_tone(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        output.write_bits(u64::from(!self.has_tone_metadata()), 1)?;
        if self.has_tone_metadata() {
            output.write_bits(u64::from(self.intensity_target.to_bits()), 16)?;
            output.write_bits(u64::from(self.min_nits.to_bits()), 16)?;
            let (relative, threshold) = self.linear_below.fields();
            output.write_bits(u64::from(relative), 1)?;
            output.write_bits(u64::from(threshold.to_bits()), 16)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_exact_image_white_is_required() {
        for bits in 0..=u16::MAX {
            if let Some(value) = FiniteF16::from_bits(bits) {
                let options = ImageOptions {
                    intensity_target: value,
                    ..Default::default()
                };
                assert_eq!(options.validate().is_ok(), bits > 0 && bits < 0x7c00);
            }
        }
    }

    #[test]
    fn every_finite_binary16_obeys_image_luminance_and_threshold_bounds() {
        for bits in 0..=u16::MAX {
            let Some(value) = FiniteF16::from_bits(bits) else {
                continue;
            };
            let scalar = value.to_f32();
            let options = ImageOptions {
                min_nits: value,
                ..Default::default()
            };
            assert_eq!(options.validate().is_ok(), (0.0..=255.0).contains(&scalar));
            for threshold in [
                ToneMappingThreshold::AbsoluteNits(value),
                ToneMappingThreshold::DisplayFraction(value),
            ] {
                let options = ImageOptions {
                    linear_below: threshold,
                    ..Default::default()
                };
                let valid = scalar >= 0.0
                    && (matches!(threshold, ToneMappingThreshold::AbsoluteNits(_))
                        || scalar <= 1.0);
                assert_eq!(options.validate().is_ok(), valid);
            }
            if scalar > 0.0 {
                assert!(
                    ImageOptions {
                        intensity_target: value,
                        min_nits: value,
                        ..Default::default()
                    }
                    .validate()
                    .is_ok()
                );
            }
        }
    }
}
