//! Bounded JPEG XL HDR gain-map (`jhgm`) bundles and ISO 21496-1 metadata.
//!
//! Parsing reconstructs only color metadata. The auxiliary codestream remains borrowed and
//! requires an image decoder. The writer preserves supplied color/ICC/codestream bytes and
//! canonically serializes the exact rational gain-map parameters.

use std::sync::Arc;

use jxl_bitstream::Bitstream;
use jxl_image::color::ColourEncoding;
use jxl_oxide_common::Bundle;

use crate::{ColourEncodingInventory, InventoryLimits};

mod parameters;
pub use parameters::{GainMapChannel, GainMapMetadata, SignedFraction, UnsignedFraction};

pub const JHGM: [u8; 4] = *b"jhgm";

/// Host payload/ICC limits, independent of decoder GPU and codestream-inventory limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainMapLimits {
    pub max_payload_bytes: u64,
    pub max_codestream_bytes: u64,
    pub max_transformed_icc_bytes: u64,
    pub max_decoded_icc_bytes: u64,
}

impl Default for GainMapLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: 64 << 20,
            max_codestream_bytes: 64 << 20,
            max_transformed_icc_bytes: 16 << 20,
            max_decoded_icc_bytes: 16 << 20,
        }
    }
}

/// Validated version-zero bundle. Metadata remains independent of image decode support.
#[derive(Clone, Debug)]
pub struct GainMapBundle<'a> {
    metadata: GainMapMetadata,
    color_bytes: &'a [u8],
    color: Option<ColourEncodingInventory>,
    compressed_icc: &'a [u8],
    icc: Option<Arc<[u8]>>,
    transformed_icc_bytes: u64,
    codestream: &'a [u8],
}

impl<'a> GainMapBundle<'a> {
    pub fn parse(payload: &'a [u8], limits: GainMapLimits) -> Result<Self, GainMapError> {
        check_limit(
            "bundle bytes",
            payload.len() as u64,
            limits.max_payload_bytes,
        )?;
        let mut reader = Reader::new(payload);
        let version = reader.u8()?;
        if version != 0 {
            return Err(GainMapError::Version {
                scope: "jhgm",
                version: u16::from(version),
            });
        }
        let length = usize::from(reader.u16()?);
        let metadata = GainMapMetadata::parse(reader.take(length)?)?;
        let length = usize::from(reader.u8()?);
        let color = reader.take(length)?;
        let length = usize::try_from(reader.u32()?).map_err(|_| GainMapError::SizeOverflow)?;
        let icc = reader.take(length)?;
        Self::new(metadata, color, icc, reader.remaining(), limits)
    }

    /// Constructs a bundle from an optional serialized JPEG XL ColorEncoding, optional
    /// JPEG XL-compressed ICC profile, and raw auxiliary codestream. Empty slices omit color/ICC.
    pub fn new(
        metadata: GainMapMetadata,
        color: &'a [u8],
        compressed_icc: &'a [u8],
        codestream: &'a [u8],
        limits: GainMapLimits,
    ) -> Result<Self, GainMapError> {
        metadata.validate()?;
        check_limit(
            "color encoding bytes",
            color.len() as u64,
            u64::from(u8::MAX),
        )?;
        check_limit(
            "compressed ICC bytes",
            compressed_icc.len() as u64,
            u64::from(u32::MAX),
        )?;
        check_limit(
            "auxiliary codestream bytes",
            codestream.len() as u64,
            limits.max_codestream_bytes.min(u64::from(u32::MAX)),
        )?;
        let size = 8_u64
            .checked_add(metadata.encode()?.len() as u64)
            .and_then(|n| n.checked_add(color.len() as u64))
            .and_then(|n| n.checked_add(compressed_icc.len() as u64))
            .and_then(|n| n.checked_add(codestream.len() as u64))
            .ok_or(GainMapError::SizeOverflow)?;
        check_limit("bundle bytes", size, limits.max_payload_bytes)?;
        if !codestream.starts_with(&[0xff, 0x0a]) {
            return Err(GainMapError::Invalid(
                "gain map must contain a raw JPEG XL codestream",
            ));
        }
        let encoding = if color.is_empty() {
            None
        } else {
            let mut reader = Bitstream::new(color);
            let encoding = ColourEncoding::parse(&mut reader, ())
                .map_err(|error| GainMapError::Color(error.to_string()))?;
            finish_bits(color, reader.num_read_bits() as u64)?;
            Some(crate::inventory::colour_encoding_inventory(&encoding))
        };
        let (icc, transformed_icc_bytes) = if compressed_icc.is_empty() {
            (None, 0)
        } else {
            let icc = crate::inventory::parse_embedded_icc(
                compressed_icc,
                0,
                InventoryLimits {
                    max_encoded_icc_bytes: limits.max_transformed_icc_bytes,
                    max_decoded_icc_bytes: limits.max_decoded_icc_bytes,
                    ..InventoryLimits::default()
                },
            )
            .map_err(|error| GainMapError::Color(error.to_string()))?;
            finish_bits(compressed_icc, icc.bit_range.length)?;
            (Some(icc.profile), icc.encoded_byte_count)
        };
        if matches!(encoding, Some(ColourEncodingInventory::IccProfile { .. })) && icc.is_none() {
            return Err(GainMapError::Invalid(
                "alternate color encoding requires an ICC profile",
            ));
        }
        Ok(Self {
            metadata,
            color_bytes: color,
            color: encoding,
            compressed_icc,
            icc,
            transformed_icc_bytes,
            codestream,
        })
    }

    #[must_use]
    pub const fn metadata(&self) -> &GainMapMetadata {
        &self.metadata
    }

    #[must_use]
    pub const fn alternate_color_encoding(&self) -> Option<ColourEncodingInventory> {
        self.color
    }

    /// Exact reconstructed profile bytes; interpretation belongs to the color-management layer.
    #[must_use]
    pub fn alternate_icc(&self) -> Option<&Arc<[u8]>> {
        self.icc.as_ref()
    }

    #[must_use]
    pub const fn codestream(&self) -> &'a [u8] {
        self.codestream
    }

    pub fn encode(&self, limits: GainMapLimits) -> Result<Vec<u8>, GainMapError> {
        let metadata = self.metadata.encode()?;
        let size = 8_usize
            .checked_add(metadata.len())
            .and_then(|n| n.checked_add(self.color_bytes.len()))
            .and_then(|n| n.checked_add(self.compressed_icc.len()))
            .and_then(|n| n.checked_add(self.codestream.len()))
            .ok_or(GainMapError::SizeOverflow)?;
        check_limit("bundle bytes", size as u64, limits.max_payload_bytes)?;
        check_limit(
            "auxiliary codestream bytes",
            self.codestream.len() as u64,
            limits.max_codestream_bytes,
        )?;
        check_limit(
            "transformed ICC bytes",
            self.transformed_icc_bytes,
            limits.max_transformed_icc_bytes,
        )?;
        check_limit(
            "decoded ICC bytes",
            self.icc.as_ref().map_or(0, |profile| profile.len() as u64),
            limits.max_decoded_icc_bytes,
        )?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|_| GainMapError::Allocation)?;
        output.push(0);
        output.extend_from_slice(&(metadata.len() as u16).to_be_bytes());
        output.extend_from_slice(&metadata);
        output.push(self.color_bytes.len() as u8);
        output.extend_from_slice(self.color_bytes);
        output.extend_from_slice(&(self.compressed_icc.len() as u32).to_be_bytes());
        output.extend_from_slice(self.compressed_icc);
        output.extend_from_slice(self.codestream);
        Ok(output)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GainMapError {
    #[error("truncated gain-map metadata")]
    Truncated,
    #[error("unsupported {scope} version {version}")]
    Version { scope: &'static str, version: u16 },
    #[error("invalid gain-map metadata: {0}")]
    Invalid(&'static str),
    #[error("gain-map color metadata: {0}")]
    Color(String),
    #[error("gain-map {resource} requires {required}, limit is {limit}")]
    Limit {
        resource: &'static str,
        required: u64,
        limit: u64,
    },
    #[error("gain-map size overflow")]
    SizeOverflow,
    #[error("gain-map allocation failed")]
    Allocation,
}

fn check_limit(resource: &'static str, required: u64, limit: u64) -> Result<(), GainMapError> {
    if required > limit {
        Err(GainMapError::Limit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn finish_bits(bytes: &[u8], bits: u64) -> Result<(), GainMapError> {
    if bits.div_ceil(8) != bytes.len() as u64 {
        return Err(GainMapError::Invalid("trailing color metadata bytes"));
    }
    let used = bits % 8;
    if used != 0 && bytes.last().is_some_and(|value| value >> used != 0) {
        return Err(GainMapError::Invalid("nonzero color metadata padding"));
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
}
impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8], GainMapError> {
        let (head, tail) = self
            .bytes
            .split_at_checked(length)
            .ok_or(GainMapError::Truncated)?;
        self.bytes = tail;
        Ok(head)
    }
    const fn remaining(&self) -> &'a [u8] {
        self.bytes
    }
    fn u8(&mut self) -> Result<u8, GainMapError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, GainMapError> {
        Ok(u16::from_be_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, GainMapError> {
        Ok(u32::from_be_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn i32(&mut self) -> Result<i32, GainMapError> {
        Ok(i32::from_be_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
}

#[cfg(test)]
mod tests;
