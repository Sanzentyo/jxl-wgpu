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
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            orientation: OutputOrientation::Identity,
            rendering_intent: IccRenderingIntent::Relative,
            intensity_target: FiniteF16::from_bits(0x5bf8).expect("255 is finite binary16"),
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
        Ok(())
    }

    pub(crate) fn extra_fields(self) -> bool {
        self.orientation != OutputOrientation::Identity || self.has_tone_metadata()
    }

    fn has_tone_metadata(self) -> bool {
        self.intensity_target != Self::default().intensity_target
    }

    pub(crate) fn write_tone(self, output: &mut BitWriter) -> Result<(), EncodeError> {
        output.write_bits(u64::from(!self.has_tone_metadata()), 1)?;
        if self.has_tone_metadata() {
            output.write_bits(u64::from(self.intensity_target.to_bits()), 16)?;
            output.write_bits(0, 16)?; // minimum light
            output.write_bits(0, 1)?; // absolute linear-below threshold
            output.write_bits(0, 16)?; // no protected threshold
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
}
