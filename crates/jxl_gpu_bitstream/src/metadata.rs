//! Opaque container metadata, independent of codestream rendering information.
//!
//! Retention preserves payload bytes, including their original Brotli representation. Decoding
//! is explicit and bounded separately. Exif orientation, XMP color information and JUMBF content
//! never override image headers. Rewriting normalizes box sizes and places metadata before `jxlc`.
//!
//! ```
//! use jxl_gpu_bitstream::{parse, ParseLimits};
//! use jxl_gpu_bitstream::metadata::{
//!     BrotliOptions, MetadataBox, MetadataCompression, MetadataLimits, MetadataSelection, XMP,
//! };
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let input = [0xff, 0x0a]; // Supply the complete codestream produced by the GPU encoder.
//! let parsed = parse(&input, ParseLimits::default())?;
//! let limits = MetadataLimits::default();
//! let mut metadata = parsed.metadata(&MetadataSelection::All, limits)?;
//! let xmp = MetadataBox::new(XMP, b"<xmp/>",
//!     MetadataCompression::Brotli(BrotliOptions::default()), limits)?;
//! metadata.replace(XMP, Some(xmp), limits)?;
//! let container = metadata.write_container(parsed.codestream())?;
//! let parsed = parse(&container, ParseLimits::default())?;
//! let retained = parsed.metadata(&MetadataSelection::Types(vec![XMP]), limits)?;
//! assert_eq!(retained.decode_all(limits)?[0].payload.as_ref(), b"<xmp/>");
//! # Ok(()) }
//! # example().unwrap();
//! ```

use std::borrow::Cow;

use crate::{ContainerBox, ContainerBoxRef, ParsedJxl};

mod codec;
mod collector;

pub(crate) use codec::{compress_with_prefix, decompress as decompress_body};
pub use collector::MetadataCollector;

pub const EXIF: [u8; 4] = *b"Exif";
pub const XMP: [u8; 4] = *b"xml ";
pub const JUMBF: [u8; 4] = *b"jumb";
const BROB: [u8; 4] = *b"brob";

/// Independent host limits; these do not consume the GPU image budget.
///
/// Byte limits count logical payloads, excluding box headers and allocator overhead. The
/// compressed-box payload includes its four-byte original type. Expansion ratio uses only the
/// Brotli stream length. Brotli scratch is separately bounded by its RFC 7932 window and grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetadataLimits {
    pub max_boxes: usize,
    pub max_encoded_box_bytes: u64,
    pub max_retained_bytes: u64,
    pub max_decoded_box_bytes: u64,
    pub max_total_decoded_bytes: u64,
    pub max_expansion_ratio: u32,
    pub max_brotli_window_bits: u8,
}

impl Default for MetadataLimits {
    fn default() -> Self {
        Self {
            max_boxes: 256,
            max_encoded_box_bytes: 64 << 20,
            max_retained_bytes: 128 << 20,
            max_decoded_box_bytes: 64 << 20,
            max_total_decoded_bytes: 128 << 20,
            max_expansion_ratio: 1024,
            max_brotli_window_bits: 24,
        }
    }
}

/// Selects by underlying box type, whether stored plainly or inside `brob`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetadataSelection {
    None,
    All,
    Types(Vec<[u8; 4]>),
}

impl MetadataSelection {
    fn includes(&self, box_type: [u8; 4]) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Types(types) => types.contains(&box_type),
        }
    }
}

/// Standard Brotli encoding parameters; no large-window or shared-dictionary extension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrotliOptions {
    quality: u8,
    window_bits: u8,
}

impl BrotliOptions {
    /// Quality is 0–11 and window size is 10–24 bits.
    #[must_use]
    pub const fn new(quality: u8, window_bits: u8) -> Option<Self> {
        if quality <= 11 && window_bits >= 10 && window_bits <= 24 {
            Some(Self {
                quality,
                window_bits,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn quality(self) -> u8 {
        self.quality
    }

    #[must_use]
    pub const fn window_bits(self) -> u8 {
        self.window_bits
    }
}

impl Default for BrotliOptions {
    fn default() -> Self {
        Self {
            quality: 6,
            window_bits: 22,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataCompression {
    #[default]
    None,
    Brotli(BrotliOptions),
}

/// One owned metadata payload in its original wire representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataBox {
    wire_type: [u8; 4],
    logical_type: [u8; 4],
    payload: Vec<u8>,
}

impl MetadataBox {
    /// Copies a wire payload after checking its type and encoded-size limit.
    ///
    /// Brotli syntax is checked only by [`Self::decode`], allowing opaque preservation without
    /// decompression. The four-byte `brob` type prefix is always validated.
    pub fn from_encoded(
        item: ContainerBoxRef<'_>,
        limits: MetadataLimits,
    ) -> Result<Self, MetadataError> {
        let logical_type = logical_type(item.box_type, item.payload)?;
        let mut payload = Vec::new();
        append(
            &mut payload,
            item.payload,
            limits.max_encoded_box_bytes,
            MetadataResource::EncodedBoxBytes,
        )?;
        Ok(Self {
            wire_type: item.box_type,
            logical_type,
            payload,
        })
    }

    /// Encodes an opaque payload. Exif includes its original four-byte TIFF-offset prefix.
    pub fn new(
        box_type: [u8; 4],
        payload: &[u8],
        compression: MetadataCompression,
        limits: MetadataLimits,
    ) -> Result<Self, MetadataError> {
        validate_type(box_type, false)?;
        if box_type == BROB {
            return Err(MetadataError::ForbiddenType(box_type));
        }
        check(
            MetadataResource::DecodedBoxBytes,
            payload.len() as u64,
            limits.max_decoded_box_bytes,
        )?;
        match compression {
            MetadataCompression::None => {
                Self::from_encoded(ContainerBoxRef { box_type, payload }, limits)
            }
            MetadataCompression::Brotli(options) => {
                validate_type(box_type, true)?;
                let encoded = codec::compress(box_type, payload, options, limits)?;
                Ok(Self {
                    wire_type: BROB,
                    logical_type: box_type,
                    payload: encoded,
                })
            }
        }
    }

    #[must_use]
    pub const fn box_type(&self) -> [u8; 4] {
        self.logical_type
    }

    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.wire_type == BROB
    }

    /// Borrowable directly by both complete and fragmented container writers.
    #[must_use]
    pub fn as_container_box(&self) -> ContainerBox<'_> {
        ContainerBox {
            box_type: self.wire_type,
            payload: &self.payload,
        }
    }

    /// Returns opaque decoded bytes; uncompressed payloads remain borrowed.
    pub fn decode(&self, limits: MetadataLimits) -> Result<Cow<'_, [u8]>, MetadataError> {
        check(
            MetadataResource::EncodedBoxBytes,
            self.payload.len() as u64,
            limits.max_encoded_box_bytes,
        )?;
        if self.is_compressed() {
            codec::decompress(&self.payload[4..], limits).map(Cow::Owned)
        } else {
            check(
                MetadataResource::DecodedBoxBytes,
                self.payload.len() as u64,
                limits.max_decoded_box_bytes,
            )?;
            Ok(Cow::Borrowed(&self.payload))
        }
    }
}

/// Ordered selected boxes. Duplicate types are retained until explicitly replaced or removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Metadata {
    boxes: Vec<MetadataBox>,
    retained_bytes: u64,
}

impl Metadata {
    #[must_use]
    pub fn boxes(&self) -> &[MetadataBox] {
        &self.boxes
    }

    #[must_use]
    pub const fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    pub fn boxes_of_type(&self, box_type: [u8; 4]) -> impl Iterator<Item = &MetadataBox> {
        self.boxes
            .iter()
            .filter(move |item| item.box_type() == box_type)
    }

    /// Adds a box atomically under per-box, retained-byte and count limits.
    pub fn push(&mut self, item: MetadataBox, limits: MetadataLimits) -> Result<(), MetadataError> {
        let bytes = self
            .retained_bytes
            .checked_add(item.payload.len() as u64)
            .ok_or(MetadataError::SizeOverflow)?;
        self.check_new_size(self.boxes.len() + 1, bytes, &item, limits)?;
        self.boxes
            .try_reserve(1)
            .map_err(|_| MetadataError::AllocationFailed)?;
        self.boxes.push(item);
        self.retained_bytes = bytes;
        Ok(())
    }

    /// Replaces all matching boxes at their first position, or appends when absent.
    /// `None` removes every matching box. Failure leaves the collection unchanged.
    pub fn replace(
        &mut self,
        box_type: [u8; 4],
        replacement: Option<MetadataBox>,
        limits: MetadataLimits,
    ) -> Result<(), MetadataError> {
        if replacement
            .as_ref()
            .is_some_and(|item| item.box_type() != box_type)
        {
            return Err(MetadataError::ReplacementType);
        }
        let position = self
            .boxes
            .iter()
            .position(|item| item.box_type() == box_type)
            .unwrap_or(self.boxes.len());
        let removed = self
            .boxes_of_type(box_type)
            .map(|item| item.payload.len() as u64)
            .sum::<u64>();
        let count = self.boxes.len() - self.boxes_of_type(box_type).count();
        let mut bytes = self.retained_bytes - removed;
        if let Some(item) = &replacement {
            bytes = bytes
                .checked_add(item.payload.len() as u64)
                .ok_or(MetadataError::SizeOverflow)?;
            self.check_new_size(count + 1, bytes, item, limits)?;
            self.boxes
                .try_reserve(1)
                .map_err(|_| MetadataError::AllocationFailed)?;
        }
        self.boxes.retain(|item| item.box_type() != box_type);
        if let Some(item) = replacement {
            self.boxes.insert(position, item);
        }
        self.retained_bytes = bytes;
        Ok(())
    }

    fn check_new_size(
        &self,
        count: usize,
        bytes: u64,
        item: &MetadataBox,
        limits: MetadataLimits,
    ) -> Result<(), MetadataError> {
        check(
            MetadataResource::BoxCount,
            count as u64,
            limits.max_boxes as u64,
        )?;
        check(
            MetadataResource::EncodedBoxBytes,
            item.payload.len() as u64,
            limits.max_encoded_box_bytes,
        )?;
        check(
            MetadataResource::RetainedBytes,
            bytes,
            limits.max_retained_bytes,
        )
    }

    /// Decodes the collection under both individual and aggregate output limits.
    /// No partially decoded collection is returned after an error.
    pub fn decode_all(
        &self,
        limits: MetadataLimits,
    ) -> Result<Vec<DecodedMetadataBox<'_>>, MetadataError> {
        let mut output = Vec::new();
        output
            .try_reserve(self.boxes.len())
            .map_err(|_| MetadataError::AllocationFailed)?;
        let mut total = 0_u64;
        for item in &self.boxes {
            let remaining = limits.max_total_decoded_bytes - total;
            let bounded = MetadataLimits {
                max_decoded_box_bytes: limits.max_decoded_box_bytes.min(remaining),
                ..limits
            };
            let payload = item.decode(bounded).map_err(|error| match error {
                MetadataError::Limit {
                    resource: MetadataResource::DecodedBoxBytes,
                    bytes,
                    ..
                } if remaining < limits.max_decoded_box_bytes => MetadataError::Limit {
                    resource: MetadataResource::TotalDecodedBytes,
                    bytes: total.saturating_add(bytes),
                    limit: limits.max_total_decoded_bytes,
                },
                other => other,
            })?;
            total += payload.len() as u64;
            output.push(DecodedMetadataBox {
                box_type: item.logical_type,
                payload,
            });
        }
        Ok(output)
    }

    /// Writes a canonical single-`jxlc` container with the selected original payloads.
    /// This preserves metadata order and payloads, not original box positions or size encodings.
    pub fn write_container(&self, codestream: &[u8]) -> Result<Vec<u8>, crate::Error> {
        let boxes = self
            .boxes
            .iter()
            .map(MetadataBox::as_container_box)
            .collect::<Vec<_>>();
        crate::write_container_with_boxes(codestream, &boxes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedMetadataBox<'a> {
    pub box_type: [u8; 4],
    pub payload: Cow<'a, [u8]>,
}

impl ParsedJxl<'_> {
    /// Copies only explicitly selected metadata; codestream storage is untouched.
    pub fn metadata(
        &self,
        selection: &MetadataSelection,
        limits: MetadataLimits,
    ) -> Result<Metadata, MetadataError> {
        let mut result = Metadata::default();
        for item in self.auxiliary_boxes() {
            let box_type = logical_type(item.box_type, item.payload)?;
            if selection.includes(box_type) {
                // Check the total before allocating a copy of the next encoded payload.
                check(
                    MetadataResource::BoxCount,
                    result.boxes.len() as u64 + 1,
                    limits.max_boxes as u64,
                )?;
                let total = result
                    .retained_bytes
                    .checked_add(item.payload.len() as u64)
                    .ok_or(MetadataError::SizeOverflow)?;
                check(
                    MetadataResource::RetainedBytes,
                    total,
                    limits.max_retained_bytes,
                )?;
                result.push(MetadataBox::from_encoded(*item, limits)?, limits)?;
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataResource {
    BoxCount,
    EncodedBoxBytes,
    RetainedBytes,
    DecodedBoxBytes,
    TotalDecodedBytes,
    BrotliWindowBits,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum MetadataError {
    #[error("metadata {resource:?} requires {bytes}, limit is {limit}")]
    Limit {
        resource: MetadataResource,
        bytes: u64,
        limit: u64,
    },
    #[error(
        "Brotli metadata expands {compressed} bytes to at least {decoded}, ratio limit is {max_ratio}"
    )]
    ExpansionLimit {
        compressed: u64,
        decoded: u64,
        max_ratio: u32,
    },
    #[error("box type {0:?} is not valid in this metadata representation")]
    ForbiddenType([u8; 4]),
    #[error("brob metadata is missing its original four-byte box type")]
    TruncatedBrotliType,
    #[error("Brotli metadata is truncated")]
    TruncatedBrotli,
    #[error("Brotli metadata is malformed or uses a nonstandard extension")]
    InvalidBrotli,
    #[error("Brotli metadata has trailing bytes after the compressed stream")]
    TrailingBrotliData,
    #[error("Brotli metadata compression failed")]
    CompressionFailed,
    #[error("replacement metadata has a different box type")]
    ReplacementType,
    #[error("metadata size arithmetic overflow")]
    SizeOverflow,
    #[error("metadata allocation failed")]
    AllocationFailed,
    #[error("metadata event order, type, offset or declared length is inconsistent")]
    EventContract,
    #[error("metadata collection requires the authoritative transport End event")]
    IncompleteTransport,
    #[error("metadata collector is poisoned by an earlier error")]
    CollectorFailed,
    #[error("metadata transport has already finished")]
    CollectorFinished,
}

fn validate_type(box_type: [u8; 4], compressed: bool) -> Result<(), MetadataError> {
    if matches!(&box_type, b"JXL " | b"ftyp" | b"jxlc" | b"jxlp")
        || (compressed && (box_type.starts_with(b"jxl") || matches!(&box_type, b"jbrd" | b"brob")))
    {
        return Err(MetadataError::ForbiddenType(box_type));
    }
    Ok(())
}

fn logical_type(wire_type: [u8; 4], payload: &[u8]) -> Result<[u8; 4], MetadataError> {
    validate_type(wire_type, false)?;
    if wire_type == BROB {
        let inner = payload
            .get(..4)
            .ok_or(MetadataError::TruncatedBrotliType)?
            .try_into()
            .expect("checked four-byte prefix");
        validate_type(inner, true)?;
        Ok(inner)
    } else {
        Ok(wire_type)
    }
}

fn check(resource: MetadataResource, bytes: u64, limit: u64) -> Result<(), MetadataError> {
    if bytes > limit {
        Err(MetadataError::Limit {
            resource,
            bytes,
            limit,
        })
    } else {
        Ok(())
    }
}

fn append(
    output: &mut Vec<u8>,
    bytes: &[u8],
    limit: u64,
    resource: MetadataResource,
) -> Result<(), MetadataError> {
    let length = output
        .len()
        .checked_add(bytes.len())
        .ok_or(MetadataError::SizeOverflow)?;
    check(resource, length as u64, limit)?;
    output
        .try_reserve(bytes.len())
        .map_err(|_| MetadataError::AllocationFailed)?;
    output.extend_from_slice(bytes);
    Ok(())
}

#[cfg(test)]
mod tests;
