//! Exact ISO 21496-1 version-zero fractions; no pixel processing.

use super::{GainMapError, Reader, check_limit};

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GainMapMetadata {
    pub channels: [GainMapChannel; 3],
    pub base_hdr_headroom: UnsignedFraction,
    pub alternate_hdr_headroom: UnsignedFraction,
    pub use_base_color_space: bool,
    /// Version of the writer. The minimum supported reader version is always zero.
    pub writer_version: u16,
    /// Opaque trailing metadata from a newer, version-zero-compatible writer.
    /// Must be empty when `writer_version` is zero. Retained when serializing again.
    pub extensions: Vec<u8>,
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
            use_base_color_space: true,
            writer_version: 0,
            extensions: Vec::new(),
        }
    }
}

impl GainMapMetadata {
    pub fn validate(&self) -> Result<(), GainMapError> {
        if self.writer_version == 0 && !self.extensions.is_empty() {
            return Err(GainMapError::Invalid("trailing version-zero ISO metadata"));
        }
        check_limit(
            "ISO metadata bytes",
            self.encoded_len() as u64,
            u64::from(u16::MAX),
        )?;
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
        check_limit(
            "ISO metadata bytes",
            bytes.len() as u64,
            u64::from(u16::MAX),
        )?;
        let mut r = Reader::new(bytes);
        let version = r.u16()?;
        if version != 0 {
            return Err(GainMapError::Version {
                scope: "ISO 21496-1 minimum",
                version,
            });
        }
        let writer_version = r.u16()?;
        let flags = r.u8()?;
        if flags & !0xc0 != 0 {
            return Err(GainMapError::Invalid("reserved ISO metadata flags"));
        }
        let count = if flags & 0x80 != 0 { 3 } else { 1 };
        let base_hdr_headroom = read_unsigned(&mut r)?;
        let alternate_hdr_headroom = read_unsigned(&mut r)?;
        let mut channels = [GainMapChannel::default(); 3];
        for channel in &mut channels[..count] {
            *channel = GainMapChannel {
                min: read_signed(&mut r)?,
                max: read_signed(&mut r)?,
                gamma: read_unsigned(&mut r)?,
                base_offset: read_signed(&mut r)?,
                alternate_offset: read_signed(&mut r)?,
            };
        }
        if count == 1 {
            channels = [channels[0]; 3];
        }
        if writer_version == 0 && !r.remaining().is_empty() {
            return Err(GainMapError::Invalid("trailing ISO metadata bytes"));
        }
        let mut extensions = Vec::new();
        extensions
            .try_reserve_exact(r.remaining().len())
            .map_err(|_| GainMapError::Allocation)?;
        extensions.extend_from_slice(r.remaining());
        let result = Self {
            channels,
            base_hdr_headroom,
            alternate_hdr_headroom,
            use_base_color_space: flags & 0x40 != 0,
            writer_version,
            extensions,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn encode(&self) -> Result<Vec<u8>, GainMapError> {
        self.validate()?;
        let count = self.channel_count();
        let mut out = Vec::new();
        out.try_reserve_exact(self.encoded_len())
            .map_err(|_| GainMapError::Allocation)?;
        out.extend_from_slice(&0_u16.to_be_bytes());
        out.extend_from_slice(&self.writer_version.to_be_bytes());
        out.push((u8::from(count == 3) << 7) | (u8::from(self.use_base_color_space) << 6));
        for value in [self.base_hdr_headroom, self.alternate_hdr_headroom] {
            out.extend_from_slice(&value.numerator.to_be_bytes());
            out.extend_from_slice(&value.denominator.to_be_bytes());
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
                out.extend_from_slice(&denominator.to_be_bytes());
            }
        }
        out.extend_from_slice(&self.extensions);
        Ok(out)
    }

    fn channel_count(&self) -> usize {
        if self.channels.iter().all(|c| *c == self.channels[0]) {
            1
        } else {
            3
        }
    }

    pub(super) fn encoded_len(&self) -> usize {
        (21 + self.channel_count() * 40_usize).saturating_add(self.extensions.len())
    }
}

fn read_unsigned(reader: &mut Reader<'_>) -> Result<UnsignedFraction, GainMapError> {
    let numerator = reader.u32()?;
    let denominator = reader.u32()?;
    Ok(UnsignedFraction {
        numerator,
        denominator,
    })
}
fn read_signed(reader: &mut Reader<'_>) -> Result<SignedFraction, GainMapError> {
    let numerator = reader.i32()?;
    let denominator = reader.u32()?;
    Ok(SignedFraction {
        numerator,
        denominator,
    })
}
