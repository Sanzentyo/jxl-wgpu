//! Bounded export of original color metadata, independent of pixel decoding or CMS admission.
//!
//! Enumerated profiles use the libjxl 0.12.0 serialization contract. Embedded profiles retain
//! their original bytes, including profiles that a color-transform implementation may reject.
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Adapted metadata serialization: libjxl 0.12.0, BSD-3-Clause; see THIRD_PARTY.md.

use std::borrow::Cow;

use crate::{ColourEncodingInventory, ImageHeaderInventory};

mod encoding;
mod math;
mod tables;
mod writer;

use encoding::Encoding;
use writer::{Kind, Plan, Writer};

/// Output bytes are admitted before allocation. Generation also uses bounded metadata scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IccProfileLimits {
    pub max_profile_bytes: u64,
}

impl Default for IccProfileLimits {
    fn default() -> Self {
        Self {
            max_profile_bytes: 16 << 20,
        }
    }
}

/// Color declarations, resource admission and allocation have distinct failure categories.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IccProfileError {
    #[error("invalid ICC export metadata: {0}")]
    Invalid(&'static str),
    #[error("ICC export has no profile for {0}")]
    Unsupported(&'static str),
    #[error("ICC export needs {required} bytes, limit {limit}")]
    Limit { required: u64, limit: u64 },
    #[error("could not allocate {bytes} ICC profile bytes")]
    Allocation { bytes: u64 },
}

type Result<T> = std::result::Result<T, IccProfileError>;

fn admit(bytes: u64, limits: IccProfileLimits) -> Result<()> {
    let limit = limits.max_profile_bytes.min(u64::from(u32::MAX));
    if bytes > limit {
        return Err(IccProfileError::Limit {
            required: bytes,
            limit,
        });
    }
    Ok(())
}

impl ImageHeaderInventory {
    /// Returns the original profile declared by this image, without choosing a pixel output space.
    ///
    /// Embedded bytes are borrowed unchanged. Enumerated RGB, Gray and XYB declarations generate
    /// an owned ICCv4 profile. The result does not authorize a color transform or decoded pixels.
    pub fn original_icc_profile(&self, limits: IccProfileLimits) -> Result<Cow<'_, [u8]>> {
        match (self.colour_encoding, &self.embedded_icc) {
            (ColourEncodingInventory::IccProfile { .. }, Some(embedded)) => {
                admit(embedded.profile.len() as u64, limits)?;
                Ok(Cow::Borrowed(&embedded.profile))
            }
            (ColourEncodingInventory::Enumerated { .. }, None) => self
                .colour_encoding
                .generate_icc_profile(limits)
                .map(Cow::Owned),
            _ => Err(IccProfileError::Invalid("embedded ICC binding")),
        }
    }
}

impl ColourEncodingInventory {
    /// Serializes enumerated color metadata to ICCv4, under an explicit output-byte limit.
    ///
    /// An embedded-profile declaration alone has no profile bytes. Use the image header's
    /// original_icc_profile method to export a reconstructed embedded profile.
    pub fn generate_icc_profile(self, limits: IccProfileLimits) -> Result<Vec<u8>> {
        let encoding = Encoding::new(self)?;
        let plan = Plan::new(&encoding)?;
        admit(plan.bytes as u64, limits)?;
        let mut writer = Writer::new(plan.bytes)?;
        writer.header(&encoding);
        writer.u32_at(128, plan.tag_count as u32);
        let mut entry = 132;
        let mut offset = 132 + 12 * plan.tag_count;
        for tag in plan.tags() {
            for signature in tag.signatures() {
                writer.bytes_at(entry, signature);
                writer.u32_at(entry + 4, offset as u32);
                writer.u32_at(entry + 8, tag.size as u32);
                entry += 12;
            }
            writer.cursor = offset;
            match tag.kind {
                Kind::Description => writer.mluc(encoding.description.as_bytes()),
                Kind::Copyright => writer.mluc(b"CC0"),
                Kind::White => writer.xyz(encoding.white)?,
                Kind::Adaptation => writer.matrix(encoding.adaptation)?,
                Kind::Cicp => writer.cicp(encoding.cicp.expect("planned CICP")),
                Kind::Primary(c) => writer.xyz(encoding.primaries.map(|row| row[c]))?,
                Kind::Curve => tables::curve(&mut writer, &encoding)?,
                Kind::Xyb => tables::xyb(&mut writer)?,
                Kind::Hdr => tables::hdr(&mut writer, &encoding)?,
                Kind::Reverse => tables::reverse(&mut writer)?,
            }
            offset += tag.size;
            debug_assert_eq!(writer.cursor.next_multiple_of(4), offset);
        }
        debug_assert_eq!(offset, writer.bytes.len());
        Ok(writer.finish())
    }
}

#[cfg(test)]
mod tests;
