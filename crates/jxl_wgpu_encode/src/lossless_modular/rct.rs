use super::types::LosslessModularFormat;
use crate::EncodeError;

/// One of JPEG XL's 42 reversible color operations and channel permutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularRctType(u8);

impl LosslessModularRctType {
    pub const IDENTITY: Self = Self(0);
    pub const YCOCG: Self = Self(6);

    /// Accepts the normative wire type in `0..=41`.
    pub fn new(rct_type: u32) -> Result<Self, EncodeError> {
        if rct_type >= 42 {
            return Err(EncodeError::InvalidModularRctType { rct_type });
        }
        Ok(Self(rct_type as u8))
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.0 as u32
    }
}

/// Reversible coding of the first three source components. Alpha stays independent.
/// Explicit RCT operates on integer words, including the raw bits of IEEE input, without
/// floating-point arithmetic or a change to the image's declared color encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LosslessModularColorTransform {
    /// Global YCoCg for integer RGB/RGBA; no transform for Gray/GrayAlpha or IEEE input.
    #[default]
    Auto,
    None,
    GlobalRct(LosslessModularRctType),
    /// A transform in each pass group, independent of its MA-tree placement.
    /// A single-group frame uses its fused DC-global section instead.
    LocalRct(LosslessModularRctType),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResolvedRct {
    pub(super) rct_type: LosslessModularRctType,
    pub(super) local: bool,
}

impl LosslessModularColorTransform {
    pub(super) fn resolve(
        self,
        format: LosslessModularFormat,
        exponent_bits: u8,
    ) -> Result<Option<ResolvedRct>, EncodeError> {
        let resolved = match self {
            Self::Auto if format.color_channel_count() == 3 && exponent_bits == 0 => {
                Some(ResolvedRct {
                    rct_type: LosslessModularRctType::YCOCG,
                    local: false,
                })
            }
            Self::Auto | Self::None => None,
            Self::GlobalRct(rct_type) | Self::LocalRct(rct_type) => {
                if format.color_channel_count() != 3 {
                    return Err(EncodeError::ModularRctColorChannels {
                        color_channels: format.color_channel_count(),
                    });
                }
                Some(ResolvedRct {
                    rct_type,
                    local: matches!(self, Self::LocalRct(_)),
                })
            }
        };
        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_domain_and_source_requirements_are_typed() {
        for value in 0..42 {
            assert_eq!(LosslessModularRctType::new(value).unwrap().value(), value);
        }
        for rct_type in [42, 255, u32::MAX] {
            assert!(
                matches!(LosslessModularRctType::new(rct_type), Err(EncodeError::InvalidModularRctType {rct_type: value}) if value == rct_type)
            );
        }
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::GrayAlpha,
        ] {
            for transform in [
                LosslessModularColorTransform::GlobalRct(LosslessModularRctType::IDENTITY),
                LosslessModularColorTransform::LocalRct(LosslessModularRctType::YCOCG),
            ] {
                assert!(matches!(
                    transform.resolve(format, 0),
                    Err(EncodeError::ModularRctColorChannels { color_channels: 1 })
                ));
            }
        }
    }
}
