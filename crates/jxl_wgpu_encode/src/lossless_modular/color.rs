//! Modular image options and ownership metadata; shared source color syntax is codec-independent.
use super::types::AlphaAssociation;
use crate::{EncodeError, source_color::SourceColorEncoding};
use jxl_gpu_bitstream::{BitWriter, FiniteF16};
use jxl_gpu_protocol::icc::IccRenderingIntent;

/// Image-wide declarations for lossless Modular stills and animations.
///
/// These describe the source samples; they perform no color conversion or tone mapping.
/// Primaries, white and transfer come from the source [`jxl_gpu_formats::PixelFormat`]. The default image white
/// is JPEG XL's 255 cd/m². HDR callers can supply their known image white explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularColorOptions {
    /// For ICC input, this must equal the intent in the profile header; its original bytes
    /// are embedded unchanged. A conflicting declaration is rejected before admission.
    pub rendering_intent: IccRenderingIntent,
    /// Positive, exact binary16 luminance, in cd/m²; no implicit rounding is performed.
    pub intensity_target: FiniteF16,
}

impl Default for LosslessModularColorOptions {
    fn default() -> Self {
        Self {
            rendering_intent: IccRenderingIntent::Relative,
            intensity_target: FiniteF16::from_bits(0x5bf8).expect("255 is finite binary16"),
        }
    }
}

impl LosslessModularColorOptions {
    pub(super) fn validate(self) -> Result<(), EncodeError> {
        if self.intensity_target.to_f32() <= 0.0 {
            return Err(EncodeError::InvalidConfiguration(
                "Modular image intensity must be positive",
            ));
        }
        Ok(())
    }

    pub(super) fn extra_fields(self) -> bool {
        self.intensity_target != Self::default().intensity_target
    }

    pub(super) fn write_tone(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        output.write_bits(u64::from(!self.extra_fields()), 1)?;
        if self.extra_fields() {
            output.write_bits(u64::from(self.intensity_target.to_bits()), 16)?;
            output.write_bits(0, 16)?; // minimum light
            output.write_bits(0, 1)?; // absolute linear-below threshold
            output.write_bits(0, 16)?; // no protected threshold
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct ModularImageMetadata {
    pub(super) encoding: SourceColorEncoding,
    pub(super) options: LosslessModularColorOptions,
    pub(super) alpha: AlphaAssociation,
    pub(super) max_icc_profile_bytes: u64,
}

impl Default for ModularImageMetadata {
    fn default() -> Self {
        Self {
            encoding: SourceColorEncoding::default(),
            options: LosslessModularColorOptions::default(),
            alpha: AlphaAssociation::default(),
            max_icc_profile_bytes: super::icc::DEFAULT_PROFILE_LIMIT,
        }
    }
}

impl ModularImageMetadata {
    pub(super) fn new(
        encoding: SourceColorEncoding,
        options: LosslessModularColorOptions,
        alpha: AlphaAssociation,
        max_icc_profile_bytes: u64,
    ) -> Self {
        Self {
            encoding,
            options,
            alpha,
            max_icc_profile_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_exact_image_white_is_required() {
        for bits in 0..=u16::MAX {
            if let Some(value) = FiniteF16::from_bits(bits) {
                let options = LosslessModularColorOptions {
                    intensity_target: value,
                    ..Default::default()
                };
                assert_eq!(options.validate().is_ok(), bits > 0 && bits < 0x7c00);
            }
        }
    }
}
