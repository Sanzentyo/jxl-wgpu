//! Explicit presentation gamut policy; no image samples are evaluated on the host.

/// Mix out-of-gamut linear RGB towards an equal-luminance neutral, then normalize highlights.
/// Zero favors luminance; one favors saturation. In-gamut RGB remains unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GamutMapping {
    preserve_saturation: u32,
}

impl GamutMapping {
    /// The saturation preference must be finite and between zero and one inclusive.
    #[must_use]
    pub const fn new(preserve_saturation: f32) -> Option<Self> {
        if preserve_saturation.is_finite()
            && preserve_saturation >= 0.0
            && preserve_saturation <= 1.0
        {
            Some(Self {
                preserve_saturation: if preserve_saturation == 0.0 {
                    0
                } else {
                    preserve_saturation.to_bits()
                },
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn preserve_saturation(self) -> f32 {
        f32::from_bits(self.preserve_saturation)
    }
}

impl Default for GamutMapping {
    fn default() -> Self {
        Self::new(0.1).expect("default gamut preference is in range")
    }
}

#[cfg(test)]
mod tests;
