//! Bounded `jbrd` metadata for byte-identical JPEG reconstruction.
//!
//! This module parses marker, table, scan and preservation metadata. It never decodes image
//! entropy or coefficients. A parsed record does not establish a reconstructible JPEG: the GPU
//! codec must still bind and validate the actual frame geometry, quantization and coefficients,
//! supply referenced ICC/Exif/XMP contents, and validate entropy output before publishing bytes.

use std::ops::Range;

use crate::metadata::{BrotliOptions, MetadataError};

mod decode;
mod encode;
mod validation;

#[cfg(test)]
mod tests;

pub const JBRD: [u8; 4] = *b"jbrd";

/// Host metadata limits, independent of GPU image admission.
///
/// Owned bytes count logical vector elements and the decoded opaque body. Borrowed encoded
/// input, inline records and allocator overhead are excluded. The standard Brotli window and
/// grammar scratch are bounded separately; the decoder uses a 4096-byte output scratch buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegReconstructionLimits {
    pub max_encoded_bytes: u64,
    pub max_owned_bytes: u64,
    pub max_markers: usize,
    pub max_entries: usize,
    pub max_decoded_body_bytes: u64,
    pub max_brotli_window_bits: u8,
    pub max_expansion_ratio: u32,
}

impl Default for JpegReconstructionLimits {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 64 << 20,
            max_owned_bytes: 64 << 20,
            max_markers: 16_384,
            max_entries: 1 << 20,
            max_decoded_body_bytes: 64 << 20,
            max_brotli_window_bits: 24,
            max_expansion_ratio: 1024,
        }
    }
}

/// Canonical metadata emission limits; image coefficients remain outside this operation.
///
/// The owned-byte limit includes the existing parsed metadata, compressed temporary body and
/// final encoded payload simultaneously. Allocator overhead and Brotli scratch are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegReconstructionEncodeOptions {
    pub max_encoded_bytes: u64,
    pub max_owned_bytes: u64,
    pub max_brotli_window_bits: u8,
    pub max_expansion_ratio: u32,
    pub brotli: BrotliOptions,
}

impl Default for JpegReconstructionEncodeOptions {
    fn default() -> Self {
        Self {
            max_encoded_bytes: 64 << 20,
            max_owned_bytes: 128 << 20,
            max_brotli_window_bits: 24,
            max_expansion_ratio: 1024,
            brotli: BrotliOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppMarkerKind {
    Unknown,
    Icc,
    Exif,
    Xmp,
}

#[derive(Debug, PartialEq, Eq)]
pub struct AppMarker {
    pub kind: AppMarkerKind,
    /// Record size including marker byte and two-byte length, but excluding the leading 0xff.
    pub size: u32,
    /// Unknown records refer to the opaque body; known records need separately supplied metadata.
    pub body: Option<Range<usize>>,
}

/// Quantization values themselves come from the GPU-decoded JPEG XL frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantizationTable {
    /// JPEG precision selector: zero for eight-bit entries, one for sixteen-bit entries.
    pub precision: u8,
    pub index: u8,
    pub last: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegComponent {
    pub id: u8,
    /// Index into the reconstruction record's quantization-table list.
    pub quant: u8,
}

#[derive(Debug, PartialEq, Eq)]
pub struct HuffmanTable {
    pub ac: bool,
    pub index: u8,
    pub last: bool,
    /// Counts include the synthetic terminal symbol; zero-length codes are rejected.
    pub counts: [u16; 17],
    /// Nonempty tables end in synthetic symbol 256. An empty record denotes an empty DHT marker.
    pub values: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanComponent {
    pub component: u8,
    pub ac: u8,
    pub dc: u8,
}

/// Scan declarations are not proof of valid progression or compatibility with an image grid.
#[derive(Debug, PartialEq, Eq)]
pub struct JpegScan {
    pub start: u8,
    pub end: u8,
    pub low: u8,
    pub high: u8,
    pub components: Vec<ScanComponent>,
    pub last_pass: u8,
    pub resets: Vec<u32>,
    pub extra_zeros: Vec<(u32, u8)>,
}

/// Immutable parsed metadata, without JPEG image or coefficient authority.
///
/// Only parsing constructs this type. Borrowed records cannot mutate its contents; emission
/// therefore operates on the same checked grammar and owned body. Byte-preservation records can
/// describe noncanonical JPEG inputs, so parsing also makes no JPEG syntax-conformance claim.
///
/// ```compile_fail
/// use jxl_gpu_bitstream::jpeg_reconstruction::JpegReconstructionMetadata;
/// fn alter_validated_table(metadata: &mut JpegReconstructionMetadata) {
///     metadata.huffman_tables()[0].values.clear();
/// }
/// ```
#[derive(Debug)]
pub struct JpegReconstructionMetadata {
    gray_hint: bool,
    markers: Vec<u8>,
    apps: Vec<AppMarker>,
    comments: Vec<Range<usize>>,
    quant: Vec<QuantizationTable>,
    components: Vec<JpegComponent>,
    huffman: Vec<HuffmanTable>,
    scans: Vec<JpegScan>,
    restart_interval: u16,
    intermarker: Vec<Range<usize>>,
    tail: Range<usize>,
    padding: Vec<u8>,
    padding_bits: u32,
    has_zero_padding: bool,
    body: Vec<u8>,
    header_bytes: usize,
    logical_owned_bytes: u64,
}

impl JpegReconstructionMetadata {
    pub fn parse(
        payload: &[u8],
        limits: JpegReconstructionLimits,
    ) -> Result<Self, JpegReconstructionError> {
        decode::parse(payload, limits)
    }

    /// Re-encodes metadata canonically; it does not reproduce the original compressed `jbrd`
    /// representation or serialize JPEG image entropy. The opaque decoded body stays exact.
    pub fn encode(
        &self,
        options: JpegReconstructionEncodeOptions,
    ) -> Result<EncodedJpegReconstruction, JpegReconstructionError> {
        encode::encode(self, options)
    }

    /// Initial grammar hint, superseded by the explicit component list when binding a frame.
    #[must_use]
    pub const fn grayscale_hint(&self) -> bool {
        self.gray_hint
    }

    #[must_use]
    pub fn markers(&self) -> &[u8] {
        &self.markers
    }

    #[must_use]
    pub fn app_markers(&self) -> &[AppMarker] {
        &self.apps
    }

    pub fn comments(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.comments.iter().map(|range| &self.body[range.clone()])
    }

    #[must_use]
    pub fn quantization_tables(&self) -> &[QuantizationTable] {
        &self.quant
    }

    #[must_use]
    pub fn components(&self) -> &[JpegComponent] {
        &self.components
    }

    #[must_use]
    pub fn huffman_tables(&self) -> &[HuffmanTable] {
        &self.huffman
    }

    #[must_use]
    pub fn scans(&self) -> &[JpegScan] {
        &self.scans
    }

    #[must_use]
    pub const fn restart_interval(&self) -> u16 {
        self.restart_interval
    }

    pub fn intermarker_data(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.intermarker
            .iter()
            .map(|range| &self.body[range.clone()])
    }

    #[must_use]
    pub fn tail(&self) -> &[u8] {
        &self.body[self.tail.clone()]
    }

    /// Whether explicit entropy-padding bits were stored, including the zero-count case.
    #[must_use]
    pub const fn has_preserved_padding(&self) -> bool {
        self.has_zero_padding
    }

    #[must_use]
    pub const fn padding_bit_count(&self) -> u32 {
        self.padding_bits
    }

    /// Least-significant-bit-first preserved padding. Unused bits in the last byte are zero.
    #[must_use]
    pub fn padding_bytes(&self) -> &[u8] {
        &self.padding
    }

    #[must_use]
    pub fn opaque_body(&self) -> &[u8] {
        &self.body
    }

    /// Byte offset of the Brotli body in the payload that was parsed.
    #[must_use]
    pub const fn source_header_bytes(&self) -> usize {
        self.header_bytes
    }

    #[must_use]
    pub const fn logical_owned_bytes(&self) -> u64 {
        self.logical_owned_bytes
    }
}

#[derive(Debug)]
pub struct EncodedJpegReconstruction {
    bytes: Vec<u8>,
    header_bytes: usize,
    compressed_body_bytes: usize,
    logical_peak_owned_bytes: u64,
}

impl EncodedJpegReconstruction {
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    #[must_use]
    pub const fn header_bytes(&self) -> usize {
        self.header_bytes
    }

    #[must_use]
    pub const fn compressed_body_bytes(&self) -> usize {
        self.compressed_body_bytes
    }

    #[must_use]
    pub const fn logical_peak_owned_bytes(&self) -> u64 {
        self.logical_peak_owned_bytes
    }
}

impl crate::ParsedJxl<'_> {
    /// Parses the unique `jbrd` in an already transport-validated input. Missing metadata returns
    /// `None`; duplicates and a forbidden `brob`-wrapped `jbrd` are rejected before body parsing.
    pub fn jpeg_reconstruction(
        &self,
        limits: JpegReconstructionLimits,
    ) -> Result<Option<JpegReconstructionMetadata>, JpegReconstructionError> {
        let mut payload = None;
        for item in self.auxiliary_boxes() {
            if item.box_type == JBRD && payload.replace(item.payload).is_some() {
                return Err(JpegReconstructionError::DuplicateBox);
            }
            if item.box_type == *b"brob" && item.payload.starts_with(&JBRD) {
                return Err(JpegReconstructionError::WrappedBox);
            }
        }
        payload
            .map(|bytes| JpegReconstructionMetadata::parse(bytes, limits))
            .transpose()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegReconstructionResource {
    EncodedBytes,
    OwnedBytes,
    Markers,
    Entries,
    DecodedBodyBytes,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum JpegReconstructionError {
    #[error(transparent)]
    Bits(#[from] crate::Error),
    #[error(transparent)]
    Metadata(#[from] MetadataError),
    #[error("invalid JPEG reconstruction metadata: {0}")]
    Invalid(&'static str),
    #[error("JPEG reconstruction {resource:?} requires {required}, limit is {limit}")]
    Limit {
        resource: JpegReconstructionResource,
        required: u64,
        limit: u64,
    },
    #[error("JPEG reconstruction metadata allocation failed")]
    Allocation,
    #[error("container contains more than one jbrd box")]
    DuplicateBox,
    #[error("jbrd must not be wrapped in brob")]
    WrappedBox,
}

fn check(
    resource: JpegReconstructionResource,
    required: u64,
    limit: u64,
) -> Result<(), JpegReconstructionError> {
    if required > limit {
        Err(JpegReconstructionError::Limit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
