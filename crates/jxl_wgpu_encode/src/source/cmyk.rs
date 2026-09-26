//! CMYK input conventions, independent of physical packing and ICC evaluation.

/// Meaning of CMYK samples in a GPU input buffer. Alpha is never complemented.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CmykSampleEncoding {
    /// ICC device amounts: zero means no ink. Integer samples are complemented exactly on GPU.
    /// Requires unassociated alpha; premultiplied inputs must use `Complemented`.
    #[default]
    InkAmounts,
    /// JPEG XL component words: zero means full ink. No sample conversion is performed.
    /// This preserves every integer or floating word, including signed zero and NaN payloads.
    /// Associated alpha describes these coded CMY words, without an association conversion.
    Complemented,
}

pub(super) const COMPLEMENT_INTEGER: u32 = 1 << 5;

impl crate::BufferImageSource {
    pub(crate) fn validate_alpha_association(
        &self,
        association: crate::AlphaAssociation,
    ) -> Result<(), crate::EncodeError> {
        if association == crate::AlphaAssociation::Associated
            && self.cmyk_encoding() == CmykSampleEncoding::InkAmounts
            && matches!(&self.layout.format.color_spec, jxl_gpu_formats::ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
        {
            // Complementing ink amounts does not commute with premultiplication.
            // Association belongs to the coded CMY domain; callers supply those words explicitly.
            return Err(crate::EncodeError::InvalidSource(
                "associated CMYK input requires explicit complemented samples",
            ));
        }
        Ok(())
    }
}

impl super::SourceSpec {
    pub(super) fn with_cmyk_encoding(
        mut self,
        encoding: CmykSampleEncoding,
    ) -> Result<Self, crate::EncodeError> {
        let Some(black) = self.black.as_mut() else {
            if encoding != CmykSampleEncoding::InkAmounts {
                return Err(crate::EncodeError::InvalidSource(
                    "a CMYK sample convention requires a CMYK ICC input",
                ));
            }
            return Ok(self);
        };
        if encoding == CmykSampleEncoding::InkAmounts {
            // 1-x is not reversible for all finite floating words (or signed zero).
            // The explicit complemented convention preserves the supplied words instead.
            if self.exponent_bits_per_sample != 0 {
                return Err(crate::EncodeError::InvalidSource(
                    "floating CMYK input requires explicit complemented samples",
                ));
            }
            for component in &mut self.components[..3] {
                component.bit_shift |= COMPLEMENT_INTEGER;
            }
            black.bit_shift |= COMPLEMENT_INTEGER;
        }
        Ok(self)
    }
}
