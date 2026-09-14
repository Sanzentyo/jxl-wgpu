//! Exact ISO 21496-1 version-zero fractions; no pixel processing.

use super::{GainMapError, Reader};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedFraction {
    pub numerator: i32,
    pub denominator: u32,
}

impl SignedFraction {
    #[must_use]
    pub fn value(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsignedFraction {
    pub numerator: u32,
    pub denominator: u32,
}

impl UnsignedFraction {
    #[must_use]
    pub fn value(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

/// Per-channel log2 gain range, encoding gamma, and linear-light offsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainMapChannel {
    pub min: SignedFraction,
    pub max: SignedFraction,
    pub gamma: UnsignedFraction,
    pub base_offset: SignedFraction,
    pub alternate_offset: SignedFraction,
}

impl Default for GainMapChannel {
    fn default() -> Self {
        Self {
            min: SignedFraction {
                numerator: 0,
                denominator: 1,
            },
            max: SignedFraction {
                numerator: 1,
                denominator: 1,
            },
            gamma: UnsignedFraction {
                numerator: 1,
                denominator: 1,
            },
            base_offset: SignedFraction {
                numerator: 0,
                denominator: 1,
            },
            alternate_offset: SignedFraction {
                numerator: 0,
                denominator: 1,
            },
        }
    }
}

/// Exact metadata for RGB gain application. A single-channel record expands to three identical
/// entries. Fractions are deliberately not reduced, so their original precision is retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainMapMetadata {
    pub channels: [GainMapChannel; 3],
    pub base_hdr_headroom: UnsignedFraction,
    pub alternate_hdr_headroom: UnsignedFraction,
    pub backward_direction: bool,
    pub use_base_color_space: bool,
}

impl Default for GainMapMetadata {
    fn default() -> Self {
        Self {
            channels: [GainMapChannel::default(); 3],
            base_hdr_headroom: UnsignedFraction {
                numerator: 0,
                denominator: 1,
            },
            alternate_hdr_headroom: UnsignedFraction {
                numerator: 1,
                denominator: 1,
            },
            backward_direction: false,
            use_base_color_space: true,
        }
    }
}

impl GainMapMetadata {
    pub fn validate(&self) -> Result<(), GainMapError> {
        if self.base_hdr_headroom.denominator == 0 || self.alternate_hdr_headroom.denominator == 0 {
            return Err(GainMapError::Invalid("zero headroom denominator"));
        }
        for c in self.channels {
            if [
                c.min.denominator,
                c.max.denominator,
                c.gamma.denominator,
                c.base_offset.denominator,
                c.alternate_offset.denominator,
            ]
            .contains(&0)
            {
                return Err(GainMapError::Invalid("zero channel denominator"));
            }
            if c.gamma.numerator == 0 {
                return Err(GainMapError::Invalid("zero gamma"));
            }
            if i64::from(c.max.numerator) * i64::from(c.min.denominator)
                < i64::from(c.min.numerator) * i64::from(c.max.denominator)
            {
                return Err(GainMapError::Invalid(
                    "maximum gain is less than minimum gain",
                ));
            }
        }
        Ok(())
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, GainMapError> {
        let mut r = Reader::new(bytes);
        for scope in ["ISO 21496-1 minimum", "ISO 21496-1 writer"] {
            let version = r.u16()?;
            if version != 0 {
                return Err(GainMapError::Version { scope, version });
            }
        }
        let flags = r.u8()?;
        if flags & !0xcc != 0 {
            return Err(GainMapError::Invalid("reserved ISO metadata flags"));
        }
        let count = if flags & 0x80 != 0 { 3 } else { 1 };
        let common = if flags & 8 != 0 { Some(r.u32()?) } else { None };
        let base_hdr_headroom = read_unsigned(&mut r, common)?;
        let alternate_hdr_headroom = read_unsigned(&mut r, common)?;
        let mut channels = [GainMapChannel::default(); 3];
        for channel in &mut channels[..count] {
            *channel = GainMapChannel {
                min: read_signed(&mut r, common)?,
                max: read_signed(&mut r, common)?,
                gamma: read_unsigned(&mut r, common)?,
                base_offset: read_signed(&mut r, common)?,
                alternate_offset: read_signed(&mut r, common)?,
            };
        }
        if count == 1 {
            channels = [channels[0]; 3];
        }
        if !r.remaining().is_empty() {
            return Err(GainMapError::Invalid("trailing ISO metadata bytes"));
        }
        let result = Self {
            channels,
            base_hdr_headroom,
            alternate_hdr_headroom,
            backward_direction: flags & 4 != 0,
            use_base_color_space: flags & 0x40 != 0,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn encode(&self) -> Result<Vec<u8>, GainMapError> {
        self.validate()?;
        let count = if self.channels.iter().all(|c| *c == self.channels[0]) {
            1
        } else {
            3
        };
        let denominator = self.base_hdr_headroom.denominator;
        let common = self.alternate_hdr_headroom.denominator == denominator
            && self.channels[..count].iter().all(|c| {
                [
                    c.min.denominator,
                    c.max.denominator,
                    c.gamma.denominator,
                    c.base_offset.denominator,
                    c.alternate_offset.denominator,
                ]
                .iter()
                .all(|d| *d == denominator)
            });
        let mut out = Vec::new();
        out.try_reserve_exact(141)
            .map_err(|_| GainMapError::Allocation)?;
        out.extend_from_slice(&[0; 4]);
        out.push(
            (u8::from(count == 3) << 7)
                | (u8::from(self.use_base_color_space) << 6)
                | (u8::from(common) << 3)
                | (u8::from(self.backward_direction) << 2),
        );
        if common {
            out.extend_from_slice(&denominator.to_be_bytes());
        }
        for value in [self.base_hdr_headroom, self.alternate_hdr_headroom] {
            out.extend_from_slice(&value.numerator.to_be_bytes());
            if !common {
                out.extend_from_slice(&value.denominator.to_be_bytes());
            }
        }
        for c in &self.channels[..count] {
            for (numerator, denominator) in [
                (c.min.numerator as u32, c.min.denominator),
                (c.max.numerator as u32, c.max.denominator),
                (c.gamma.numerator, c.gamma.denominator),
                (c.base_offset.numerator as u32, c.base_offset.denominator),
                (
                    c.alternate_offset.numerator as u32,
                    c.alternate_offset.denominator,
                ),
            ] {
                out.extend_from_slice(&numerator.to_be_bytes());
                if !common {
                    out.extend_from_slice(&denominator.to_be_bytes());
                }
            }
        }
        Ok(out)
    }
}

fn read_unsigned(
    reader: &mut Reader<'_>,
    common: Option<u32>,
) -> Result<UnsignedFraction, GainMapError> {
    let numerator = reader.u32()?;
    let denominator = if let Some(d) = common {
        d
    } else {
        reader.u32()?
    };
    Ok(UnsignedFraction {
        numerator,
        denominator,
    })
}
fn read_signed(
    reader: &mut Reader<'_>,
    common: Option<u32>,
) -> Result<SignedFraction, GainMapError> {
    let numerator = reader.i32()?;
    let denominator = if let Some(d) = common {
        d
    } else {
        reader.u32()?
    };
    Ok(SignedFraction {
        numerator,
        denominator,
    })
}
