//! Bounded JPEG XL container and bitstream orchestration.
//!
//! This crate deliberately does not decode or encode image samples. It validates the transport
//! container, exposes the contiguous JPEG XL codestream to a GPU codec, inventories standard image
//! and frame headers plus physical TOC section ranges, and provides deterministic bit/box writers
//! for final packet assembly.

#![deny(unsafe_code)]

use std::borrow::Cow;
use std::collections::BTreeMap;

use thiserror::Error;

mod acceleration;
mod inventory;

#[cfg(test)]
mod test_fixtures {
    fn hex_nibble(byte: u8) -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => panic!("invalid checked-in fixture hex digit"),
        }
    }

    fn decode_hex(input: &str) -> Vec<u8> {
        let digits = input
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]))
            .collect()
    }

    pub(crate) fn fragmented_animation() -> Vec<u8> {
        decode_hex(include_str!("../test-data/fragmented_animation.jxl.hex"))
    }

    pub(crate) fn basic() -> Vec<u8> {
        decode_hex(include_str!("../test-data/basic.jxl.hex"))
    }

    pub(crate) fn oddsize_ups() -> Vec<u8> {
        decode_hex(include_str!("../test-data/oddsize_ups.jxl.hex"))
    }

    pub(crate) fn green_queen_vardct() -> Vec<u8> {
        decode_hex(include_str!("../test-data/green_queen_vardct_e3.jxl.hex"))
    }

    pub(crate) fn animation_spline() -> Vec<u8> {
        decode_hex(include_str!("../test-data/animation_spline.jxl.hex"))
    }

    pub(crate) fn gpu_gray8_lossless() -> Vec<u8> {
        decode_hex(include_str!("../test-data/gpu_gray8_lossless.jxl.hex"))
    }

    pub(crate) fn has_permutation() -> Vec<u8> {
        decode_hex(include_str!("../test-data/has_permutation.jxl.hex"))
    }

    pub(crate) fn with_icc() -> Vec<u8> {
        decode_hex(include_str!("../test-data/with_icc.jxl.hex"))
    }
}

pub use acceleration::{
    ACCELERATION_INDEX_BOX_TYPE, AccelerationIndexError, Gray8AccelerationIndex, PrefixCodeEntry,
};
pub use inventory::{
    AnimationInventory, BitRange, ByteRange, CodestreamInventory, EmbeddedIccInventory,
    FrameEncoding, FrameInventory, FrameSection, FrameSectionKind, FrameType, ImageHeaderInventory,
    InventoryError, InventoryLimits, SampleBitDepth,
};

/// Raw JPEG XL codestream signature (`0xff 0x0a`).
pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xff, 0x0a];

/// Complete ISO BMFF signature box required at the beginning of a JPEG XL container.
pub const CONTAINER_SIGNATURE_BOX: [u8; 12] =
    [0, 0, 0, 12, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a];

/// Mandatory JPEG XL file-type box for delivery-order containers.
pub const CONTAINER_FILE_TYPE_BOX_V0: [u8; 20] = [
    0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x', b'l', b' ',
];

const JXLC: [u8; 4] = *b"jxlc";
const JXLP: [u8; 4] = *b"jxlp";
const FTYP: [u8; 4] = *b"ftyp";

/// Resource limits applied before allocating or concatenating container payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseLimits {
    /// Maximum total input bytes inspected by one parse operation.
    pub max_input_bytes: u64,
    pub max_boxes: usize,
    pub max_box_bytes: u64,
    pub max_codestream_bytes: u64,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1 << 30,
            max_boxes: 16_384,
            max_box_bytes: 1 << 30,
            max_codestream_bytes: 1 << 30,
        }
    }
}

/// Validated contiguous codestream. Raw inputs remain borrowed; fragmented containers are joined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedJxl<'input> {
    codestream: Cow<'input, [u8]>,
    container: bool,
    auxiliary_boxes: Vec<ContainerBoxRef<'input>>,
}

impl<'input> ParsedJxl<'input> {
    #[must_use]
    pub fn codestream(&self) -> &[u8] {
        &self.codestream
    }

    #[must_use]
    pub const fn is_container(&self) -> bool {
        self.container
    }

    #[must_use]
    pub fn into_codestream(self) -> Cow<'input, [u8]> {
        self.codestream
    }

    /// Non-codestream boxes after the mandatory signature and file-type boxes.
    ///
    /// Payloads remain borrowed from the original container. `jxlc`, `jxlp`, `JXL `, and `ftyp`
    /// transport boxes are never returned here.
    #[must_use]
    pub fn auxiliary_boxes(&self) -> &[ContainerBoxRef<'input>] {
        &self.auxiliary_boxes
    }

    /// Returns every auxiliary box with the requested ISO BMFF four-character type.
    pub fn boxes_of_type(
        &self,
        box_type: [u8; 4],
    ) -> impl Iterator<Item = ContainerBoxRef<'input>> + '_ {
        self.auxiliary_boxes
            .iter()
            .copied()
            .filter(move |item| item.box_type == box_type)
    }

    /// Inventories standard image/frame headers and physical TOC section ranges.
    ///
    /// The returned offsets are relative to [`Self::codestream`], regardless of whether the input
    /// was raw, a single `jxlc` box, or a reconstructed `jxlp` sequence. This operation parses only
    /// bounded grammar; it does not decode pixels or frame entropy.
    pub fn codestream_inventory(
        &self,
        limits: InventoryLimits,
    ) -> Result<CodestreamInventory, InventoryError> {
        inventory::parse_codestream_inventory(self.codestream(), limits)
    }
}

/// Borrowed auxiliary ISO BMFF box from a validated JPEG XL container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerBoxRef<'input> {
    pub box_type: [u8; 4],
    pub payload: &'input [u8],
}

/// Auxiliary ISO BMFF box to place before the codestream box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerBox<'payload> {
    pub box_type: [u8; 4],
    pub payload: &'payload [u8],
}

/// Validates a raw codestream or JPEG XL container and returns contiguous codestream bytes.
pub fn parse(input: &[u8], limits: ParseLimits) -> Result<ParsedJxl<'_>, Error> {
    if u64::try_from(input.len()).map_err(|_| Error::SizeOverflow)? > limits.max_input_bytes {
        return Err(Error::ResourceLimit("input size"));
    }
    if input.starts_with(&CODESTREAM_SIGNATURE) {
        validate_codestream_len(input.len(), limits)?;
        return Ok(ParsedJxl {
            codestream: Cow::Borrowed(input),
            container: false,
            auxiliary_boxes: Vec::new(),
        });
    }
    if !input.starts_with(&CONTAINER_SIGNATURE_BOX) {
        return Err(Error::InvalidSignature);
    }
    parse_container(input, limits)
}

fn parse_container(input: &[u8], limits: ParseLimits) -> Result<ParsedJxl<'_>, Error> {
    if limits.max_boxes < 2 {
        return Err(Error::ResourceLimit("container box count"));
    }
    let (mut cursor, file_type_version) = parse_file_type_box(input, limits)?;
    let mut box_count = 2usize;
    let mut codestream_box: Option<&[u8]> = None;
    let mut fragments = BTreeMap::<u32, &[u8]>::new();
    let mut last_fragment = None;
    let mut next_fragment_index = 0u32;
    let mut auxiliary_boxes = Vec::new();

    while cursor < input.len() {
        box_count = box_count.checked_add(1).ok_or(Error::SizeOverflow)?;
        if box_count > limits.max_boxes {
            return Err(Error::ResourceLimit("container box count"));
        }
        let header = parse_box_header(input, cursor)?;
        if header.size > limits.max_box_bytes {
            return Err(Error::ResourceLimit("container box size"));
        }
        let end = if header.extends_to_end {
            input.len()
        } else {
            cursor
                .checked_add(usize::try_from(header.size).map_err(|_| Error::SizeOverflow)?)
                .ok_or(Error::SizeOverflow)?
        };
        if end > input.len() {
            return Err(Error::TruncatedBox {
                box_type: header.box_type,
            });
        }
        let payload_start = cursor
            .checked_add(header.header_size)
            .ok_or(Error::SizeOverflow)?;
        let payload = input.get(payload_start..end).ok_or(Error::TruncatedBox {
            box_type: header.box_type,
        })?;
        match header.box_type {
            FTYP => return Err(Error::MisplacedFileTypeBox),
            JXLC => {
                if codestream_box.is_some() || !fragments.is_empty() {
                    return Err(Error::ConflictingCodestreamBoxes);
                }
                codestream_box = Some(payload);
            }
            JXLP => {
                if codestream_box.is_some() {
                    return Err(Error::ConflictingCodestreamBoxes);
                }
                let index_bytes: [u8; 4] = payload
                    .get(..4)
                    .ok_or(Error::TruncatedFragmentIndex)?
                    .try_into()
                    .expect("slice length was checked");
                let raw_index = u32::from_be_bytes(index_bytes);
                let is_last = raw_index & (1 << 31) != 0;
                let index = raw_index & !(1 << 31);
                if file_type_version == 0 && index != next_fragment_index {
                    return Err(Error::OutOfOrderFragment {
                        expected: next_fragment_index,
                        actual: index,
                    });
                }
                if fragments.insert(index, &payload[4..]).is_some() {
                    return Err(Error::DuplicateFragment(index));
                }
                if is_last && last_fragment.replace(index).is_some() {
                    return Err(Error::MultipleLastFragments);
                }
                next_fragment_index = next_fragment_index
                    .checked_add(1)
                    .ok_or(Error::SizeOverflow)?;
            }
            _ => auxiliary_boxes.push(ContainerBoxRef {
                box_type: header.box_type,
                payload,
            }),
        }
        cursor = end;
        if header.extends_to_end {
            break;
        }
    }

    let codestream = if let Some(codestream) = codestream_box {
        validate_codestream(codestream, limits)?;
        Cow::Borrowed(codestream)
    } else {
        Cow::Owned(join_fragments(fragments, last_fragment, limits)?)
    };
    Ok(ParsedJxl {
        codestream,
        container: true,
        auxiliary_boxes,
    })
}

fn parse_file_type_box(input: &[u8], limits: ParseLimits) -> Result<(usize, u32), Error> {
    let cursor = CONTAINER_SIGNATURE_BOX.len();
    let header = parse_box_header(input, cursor).map_err(|error| match error {
        Error::TruncatedBoxHeader => Error::MissingFileTypeBox,
        other => other,
    })?;
    if header.box_type != FTYP {
        return Err(Error::MissingFileTypeBox);
    }
    if header.extends_to_end || header.header_size != 8 || header.size != 20 {
        return Err(Error::InvalidFileTypeBox);
    }
    if header.size > limits.max_box_bytes {
        return Err(Error::ResourceLimit("container box size"));
    }
    let end = cursor.checked_add(20).ok_or(Error::SizeOverflow)?;
    let payload = input
        .get(cursor + header.header_size..end)
        .ok_or(Error::InvalidFileTypeBox)?;
    if payload.get(..4) != Some(b"jxl ") || payload.get(8..12) != Some(b"jxl ") {
        return Err(Error::InvalidFileTypeBox);
    }
    let version = u32::from_be_bytes(
        payload[4..8]
            .try_into()
            .expect("the ftyp payload length was checked"),
    );
    if version > 1 {
        return Err(Error::UnsupportedFileTypeVersion(version));
    }
    Ok((end, version))
}

fn validate_codestream(codestream: &[u8], limits: ParseLimits) -> Result<(), Error> {
    validate_codestream_len(codestream.len(), limits)?;
    if !codestream.starts_with(&CODESTREAM_SIGNATURE) {
        return Err(Error::InvalidCodestreamSignature);
    }
    Ok(())
}

fn validate_codestream_len(length: usize, limits: ParseLimits) -> Result<(), Error> {
    let length = u64::try_from(length).map_err(|_| Error::SizeOverflow)?;
    if length > limits.max_codestream_bytes {
        return Err(Error::ResourceLimit("codestream size"));
    }
    Ok(())
}

fn join_fragments(
    fragments: BTreeMap<u32, &[u8]>,
    last_fragment: Option<u32>,
    limits: ParseLimits,
) -> Result<Vec<u8>, Error> {
    let last = last_fragment.ok_or(Error::MissingCodestream)?;
    let expected_count = last.checked_add(1).ok_or(Error::SizeOverflow)?;
    if usize::try_from(expected_count).map_err(|_| Error::SizeOverflow)? != fragments.len() {
        return Err(Error::MissingFragment);
    }
    let mut total = 0u64;
    for (&index, payload) in &fragments {
        if index >= expected_count {
            return Err(Error::FragmentAfterLast(index));
        }
        total = total
            .checked_add(u64::try_from(payload.len()).map_err(|_| Error::SizeOverflow)?)
            .ok_or(Error::SizeOverflow)?;
    }
    if total > limits.max_codestream_bytes {
        return Err(Error::ResourceLimit("codestream size"));
    }
    let total = usize::try_from(total).map_err(|_| Error::SizeOverflow)?;
    let mut joined = Vec::new();
    joined
        .try_reserve_exact(total)
        .map_err(|_| Error::AllocationFailed("fragmented codestream"))?;
    for index in 0..expected_count {
        joined.extend_from_slice(fragments.get(&index).ok_or(Error::MissingFragment)?);
    }
    validate_codestream(&joined, limits)?;
    Ok(joined)
}

#[derive(Clone, Copy, Debug)]
struct BoxHeader {
    box_type: [u8; 4],
    size: u64,
    header_size: usize,
    extends_to_end: bool,
}

fn parse_box_header(input: &[u8], cursor: usize) -> Result<BoxHeader, Error> {
    let base = input
        .get(cursor..cursor.checked_add(8).ok_or(Error::SizeOverflow)?)
        .ok_or(Error::TruncatedBoxHeader)?;
    let size32 = u32::from_be_bytes(base[..4].try_into().expect("four-byte slice"));
    let box_type = base[4..8].try_into().expect("four-byte slice");
    match size32 {
        0 => Ok(BoxHeader {
            box_type,
            size: u64::try_from(input.len().saturating_sub(cursor))
                .map_err(|_| Error::SizeOverflow)?,
            header_size: 8,
            extends_to_end: true,
        }),
        1 => {
            let extended = input
                .get(
                    cursor.checked_add(8).ok_or(Error::SizeOverflow)?
                        ..cursor.checked_add(16).ok_or(Error::SizeOverflow)?,
                )
                .ok_or(Error::TruncatedBoxHeader)?;
            let size = u64::from_be_bytes(extended.try_into().expect("eight-byte slice"));
            if size < 16 {
                return Err(Error::InvalidBoxSize { box_type, size });
            }
            Ok(BoxHeader {
                box_type,
                size,
                header_size: 16,
                extends_to_end: false,
            })
        }
        size => {
            let size = u64::from(size);
            if size < 8 {
                return Err(Error::InvalidBoxSize { box_type, size });
            }
            Ok(BoxHeader {
                box_type,
                size,
                header_size: 8,
                extends_to_end: false,
            })
        }
    }
}

/// Checked little-endian bit reader used by JPEG XL metadata parsing.
#[derive(Clone, Debug)]
pub struct BitReader<'input> {
    bytes: &'input [u8],
    bit_offset: u64,
}

impl<'input> BitReader<'input> {
    #[must_use]
    pub const fn new(bytes: &'input [u8]) -> Self {
        Self {
            bytes,
            bit_offset: 0,
        }
    }

    #[must_use]
    pub const fn bit_offset(&self) -> u64 {
        self.bit_offset
    }

    #[must_use]
    pub fn remaining_bits(&self) -> u64 {
        u64::try_from(self.bytes.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(8)
            .saturating_sub(self.bit_offset)
    }

    pub fn read_bits(&mut self, count: u8) -> Result<u64, Error> {
        if count > 56 {
            return Err(Error::InvalidBitCount(count));
        }
        if self.remaining_bits() < u64::from(count) {
            return Err(Error::UnexpectedEndOfBits);
        }
        let mut value = 0u64;
        for shift in 0..count {
            let byte_index =
                usize::try_from(self.bit_offset / 8).map_err(|_| Error::SizeOverflow)?;
            let byte = self.bytes[byte_index];
            let bit = (byte >> (self.bit_offset % 8)) & 1;
            value |= u64::from(bit) << shift;
            self.bit_offset += 1;
        }
        Ok(value)
    }

    pub fn align_to_byte(&mut self) -> Result<(), Error> {
        let aligned = self.bit_offset.checked_add(7).ok_or(Error::SizeOverflow)? & !7;
        let available = u64::try_from(self.bytes.len())
            .map_err(|_| Error::SizeOverflow)?
            .checked_mul(8)
            .ok_or(Error::SizeOverflow)?;
        if aligned > available {
            return Err(Error::UnexpectedEndOfBits);
        }
        self.bit_offset = aligned;
        Ok(())
    }

    /// Advances by `count` bits without inspecting their values.
    pub fn skip_bits(&mut self, count: u64) -> Result<(), Error> {
        if self.remaining_bits() < count {
            return Err(Error::UnexpectedEndOfBits);
        }
        self.bit_offset = self
            .bit_offset
            .checked_add(count)
            .ok_or(Error::SizeOverflow)?;
        Ok(())
    }

    /// Consumes zero padding through the next byte boundary.
    pub fn zero_pad_to_byte(&mut self) -> Result<(), Error> {
        let count = (8 - (self.bit_offset & 7)) & 7;
        if self.read_bits(count as u8)? != 0 {
            return Err(Error::NonZeroPadding);
        }
        Ok(())
    }
}

/// Deterministic little-endian bit writer for header and packet assembly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitWriter {
    bytes: Vec<u8>,
    bit_offset: usize,
}

impl BitWriter {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_offset: 0,
        }
    }

    pub fn write_bits(&mut self, value: u64, count: u8) -> Result<(), Error> {
        if count > 56 {
            return Err(Error::InvalidBitCount(count));
        }
        if count < 64 && value >= (1u64 << count) && count != 0 {
            return Err(Error::BitValueOverflow { value, count });
        }
        for shift in 0..count {
            if self.bit_offset / 8 == self.bytes.len() {
                self.bytes.push(0);
            }
            let bit = ((value >> shift) & 1) as u8;
            self.bytes[self.bit_offset / 8] |= bit << (self.bit_offset % 8);
            self.bit_offset += 1;
        }
        Ok(())
    }

    pub fn align_to_byte(&mut self) -> Result<(), Error> {
        let aligned = self.bit_offset.checked_add(7).ok_or(Error::SizeOverflow)? & !7;
        if aligned / 8 > self.bytes.len() {
            self.bytes.resize(aligned / 8, 0);
        }
        self.bit_offset = aligned;
        Ok(())
    }

    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.bit_offset
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Wraps a complete raw codestream in the deterministic single-`jxlc` container form.
pub fn write_container(codestream: &[u8]) -> Result<Vec<u8>, Error> {
    write_container_with_boxes(codestream, &[])
}

/// Wraps a raw codestream and ordered auxiliary boxes in a deterministic JPEG XL container.
///
/// Auxiliary boxes are emitted after `ftyp` and before `jxlc`. Reserved transport types are
/// rejected so callers cannot create an ambiguous or structurally invalid container.
pub fn write_container_with_boxes(
    codestream: &[u8],
    boxes: &[ContainerBox<'_>],
) -> Result<Vec<u8>, Error> {
    if !codestream.starts_with(&CODESTREAM_SIGNATURE) {
        return Err(Error::InvalidCodestreamSignature);
    }
    let mut output = Vec::new();
    output.extend_from_slice(&CONTAINER_SIGNATURE_BOX);
    output.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
    for item in boxes {
        if matches!(item.box_type, JXLC | JXLP | FTYP) || item.box_type == *b"JXL " {
            return Err(Error::ReservedBoxType(item.box_type));
        }
        append_box(&mut output, item.box_type, item.payload)?;
    }
    append_box(&mut output, JXLC, codestream)?;
    Ok(output)
}

fn append_box(output: &mut Vec<u8>, box_type: [u8; 4], payload: &[u8]) -> Result<(), Error> {
    let compact_size = 8u64
        .checked_add(u64::try_from(payload.len()).map_err(|_| Error::SizeOverflow)?)
        .ok_or(Error::SizeOverflow)?;
    if let Ok(size32) = u32::try_from(compact_size) {
        output.extend_from_slice(&size32.to_be_bytes());
        output.extend_from_slice(&box_type);
    } else {
        let extended_size = compact_size.checked_add(8).ok_or(Error::SizeOverflow)?;
        output.extend_from_slice(&1u32.to_be_bytes());
        output.extend_from_slice(&box_type);
        output.extend_from_slice(&extended_size.to_be_bytes());
    }
    output.extend_from_slice(payload);
    Ok(())
}

/// Incremental deterministic `jxlp` container writer.
///
/// Encoders may append independently completed GPU packet batches without buffering the complete
/// codestream. The caller still owns packet ordering; this type assigns consecutive fragment
/// indices and marks exactly one final fragment.
#[derive(Clone, Debug)]
pub struct FragmentedContainerWriter {
    bytes: Vec<u8>,
    next_index: u32,
    finished: bool,
    codestream_signature: [u8; 2],
    codestream_signature_len: u8,
}

impl FragmentedContainerWriter {
    #[must_use]
    pub fn new() -> Self {
        let mut bytes = CONTAINER_SIGNATURE_BOX.to_vec();
        bytes.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        Self {
            bytes,
            next_index: 0,
            finished: false,
            codestream_signature: [0; 2],
            codestream_signature_len: 0,
        }
    }

    /// Appends an auxiliary box before the first codestream fragment.
    pub fn push_box(&mut self, item: ContainerBox<'_>) -> Result<(), Error> {
        if self.finished {
            return Err(Error::ContainerAlreadyFinished);
        }
        if self.next_index != 0 {
            return Err(Error::AuxiliaryBoxAfterCodestream);
        }
        if matches!(item.box_type, JXLC | JXLP | FTYP) || item.box_type == *b"JXL " {
            return Err(Error::ReservedBoxType(item.box_type));
        }
        append_box(&mut self.bytes, item.box_type, item.payload)
    }

    /// Appends one codestream fragment. `is_last` permanently closes this writer.
    pub fn push_fragment(&mut self, payload: &[u8], is_last: bool) -> Result<(), Error> {
        if self.finished {
            return Err(Error::ContainerAlreadyFinished);
        }
        if self.next_index >= 1 << 31 {
            return Err(Error::TooManyFragments);
        }
        let mut signature = self.codestream_signature;
        let mut signature_len = usize::from(self.codestream_signature_len);
        let needed = signature.len().saturating_sub(signature_len);
        let copied = needed.min(payload.len());
        signature[signature_len..signature_len + copied].copy_from_slice(&payload[..copied]);
        signature_len += copied;
        if (signature_len == signature.len() && signature != CODESTREAM_SIGNATURE)
            || (is_last && signature_len != signature.len())
        {
            return Err(Error::InvalidCodestreamSignature);
        }
        let index = self.next_index | if is_last { 1 << 31 } else { 0 };
        let payload_size = u64::try_from(payload.len()).map_err(|_| Error::SizeOverflow)?;
        let compact_size = 12u64.checked_add(payload_size).ok_or(Error::SizeOverflow)?;
        if let Ok(size32) = u32::try_from(compact_size) {
            self.bytes.extend_from_slice(&size32.to_be_bytes());
            self.bytes.extend_from_slice(&JXLP);
        } else {
            let extended_size = compact_size.checked_add(8).ok_or(Error::SizeOverflow)?;
            self.bytes.extend_from_slice(&1u32.to_be_bytes());
            self.bytes.extend_from_slice(&JXLP);
            self.bytes.extend_from_slice(&extended_size.to_be_bytes());
        }
        self.bytes.extend_from_slice(&index.to_be_bytes());
        self.bytes.extend_from_slice(payload);
        self.next_index = self.next_index.checked_add(1).ok_or(Error::SizeOverflow)?;
        self.finished = is_last;
        self.codestream_signature = signature;
        self.codestream_signature_len = signature_len as u8;
        Ok(())
    }

    /// Returns the complete container after a final fragment has been written.
    pub fn finish(self) -> Result<Vec<u8>, Error> {
        if !self.finished {
            return Err(Error::MissingFinalFragment);
        }
        Ok(self.bytes)
    }
}

impl Default for FragmentedContainerWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("input is neither a raw JPEG XL codestream nor a JPEG XL container")]
    InvalidSignature,
    #[error("JPEG XL container is missing its mandatory second ftyp box")]
    MissingFileTypeBox,
    #[error("JPEG XL ftyp box must be the canonical 20-byte form")]
    InvalidFileTypeBox,
    #[error("JPEG XL ftyp box may only occur as the second box")]
    MisplacedFileTypeBox,
    #[error("JPEG XL container file-type version {0} is not supported")]
    UnsupportedFileTypeVersion(u32),
    #[error("box type {0:?} is reserved for JPEG XL container transport")]
    ReservedBoxType([u8; 4]),
    #[error("embedded codestream has an invalid JPEG XL signature")]
    InvalidCodestreamSignature,
    #[error("container box header is truncated")]
    TruncatedBoxHeader,
    #[error("container box {box_type:?} is truncated")]
    TruncatedBox { box_type: [u8; 4] },
    #[error("container box {box_type:?} has invalid size {size}")]
    InvalidBoxSize { box_type: [u8; 4], size: u64 },
    #[error("container mixes or duplicates jxlc/jxlp codestream boxes")]
    ConflictingCodestreamBoxes,
    #[error("jxlp fragment is missing its four-byte index")]
    TruncatedFragmentIndex,
    #[error("duplicate jxlp fragment {0}")]
    DuplicateFragment(u32),
    #[error("container contains more than one final jxlp fragment")]
    MultipleLastFragments,
    #[error("ftyp v0 requires ordered jxlp boxes: expected {expected}, got {actual}")]
    OutOfOrderFragment { expected: u32, actual: u32 },
    #[error("container has no complete codestream box sequence")]
    MissingCodestream,
    #[error("container has a gap in its jxlp fragment sequence")]
    MissingFragment,
    #[error("jxlp fragment {0} occurs after the declared final fragment")]
    FragmentAfterLast(u32),
    #[error("resource limit exceeded for {0}")]
    ResourceLimit(&'static str),
    #[error("allocation failed while building {0}")]
    AllocationFailed(&'static str),
    #[error("size arithmetic overflow")]
    SizeOverflow,
    #[error("bit count {0} exceeds the supported 56-bit operation")]
    InvalidBitCount(u8),
    #[error("bit value {value} does not fit in {count} bits")]
    BitValueOverflow { value: u64, count: u8 },
    #[error("unexpected end of bitstream")]
    UnexpectedEndOfBits,
    #[error("byte-alignment padding contains non-zero bits")]
    NonZeroPadding,
    #[error("fragmented container writer is already finished")]
    ContainerAlreadyFinished,
    #[error("auxiliary boxes must be written before the first codestream fragment")]
    AuxiliaryBoxAfterCodestream,
    #[error("fragmented container has too many jxlp fragments")]
    TooManyFragments,
    #[error("fragmented container is missing its final fragment")]
    MissingFinalFragment,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_codestream_stays_borrowed() {
        let input = [0xff, 0x0a, 1, 2, 3];
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        assert!(!parsed.is_container());
        assert!(matches!(parsed.codestream, Cow::Borrowed(_)));
    }

    #[test]
    fn deterministic_container_roundtrip() {
        let codestream = [0xff, 0x0a, 4, 5, 6];
        let container = write_container(&codestream).unwrap();
        let parsed = parse(&container, ParseLimits::default()).unwrap();
        assert!(parsed.is_container());
        assert_eq!(parsed.codestream(), codestream);
    }

    #[test]
    fn auxiliary_boxes_are_ordered_borrowed_and_reserved_types_rejected() {
        let codestream = [0xff, 0x0a, 4, 5, 6];
        let container = write_container_with_boxes(
            &codestream,
            &[
                ContainerBox {
                    box_type: *b"Exif",
                    payload: &[0, 0, 0, 0, 1],
                },
                ContainerBox {
                    box_type: *b"jwgp",
                    payload: b"index-v1",
                },
            ],
        )
        .unwrap();
        let parsed = parse(&container, ParseLimits::default()).unwrap();
        assert_eq!(parsed.auxiliary_boxes().len(), 2);
        assert_eq!(
            parsed.boxes_of_type(*b"jwgp").collect::<Vec<_>>(),
            vec![ContainerBoxRef {
                box_type: *b"jwgp",
                payload: b"index-v1"
            }]
        );
        assert_eq!(
            write_container_with_boxes(
                &codestream,
                &[ContainerBox {
                    box_type: JXLC,
                    payload: b"ambiguous"
                }]
            )
            .unwrap_err(),
            Error::ReservedBoxType(JXLC)
        );
    }

    #[test]
    fn fragmented_container_is_ordered_by_index() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 1;
        input.extend_from_slice(&file_type);
        push_jxlp(&mut input, 1 | (1 << 31), &[2, 3]);
        push_jxlp(&mut input, 0, &[0xff, 0x0a, 1]);
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        assert_eq!(parsed.codestream(), [0xff, 0x0a, 1, 2, 3]);
    }

    #[test]
    fn fragmented_container_rejects_gaps() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 1;
        input.extend_from_slice(&file_type);
        push_jxlp(&mut input, 1 | (1 << 31), &[0xff, 0x0a]);
        assert_eq!(
            parse(&input, ParseLimits::default()).unwrap_err(),
            Error::MissingFragment
        );
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3).unwrap();
        writer.write_bits(0x1ff, 9).unwrap();
        writer.align_to_byte().unwrap();
        let mut reader = BitReader::new(writer.as_bytes());
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(9).unwrap(), 0x1ff);
        reader.align_to_byte().unwrap();
        assert_eq!(reader.bit_offset(), 16);
    }

    #[test]
    fn real_fragmented_animation_container_is_reassembled() {
        let input = crate::test_fixtures::fragmented_animation();
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        assert!(parsed.is_container());
        assert!(parsed.codestream().starts_with(&CODESTREAM_SIGNATURE));
        assert!(parsed.codestream().len() < input.len());
    }

    #[test]
    fn real_gpu_index_is_bound_to_its_standard_codestream() {
        let input = crate::test_fixtures::gpu_gray8_lossless();
        let parsed = parse(&input, ParseLimits::default()).unwrap();
        let boxes = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .collect::<Vec<_>>();
        assert_eq!(boxes.len(), 1);
        let index = Gray8AccelerationIndex::parse_bound(boxes[0].payload, parsed.codestream())
            .expect("checked-in acceleration index is valid");
        assert_eq!((index.width(), index.height()), (17, 13));
        assert_eq!(index.sample_count(), 221);
    }

    #[test]
    fn fragmented_writer_roundtrips_and_closes() {
        let mut writer = FragmentedContainerWriter::new();
        writer
            .push_box(ContainerBox {
                box_type: *b"jwgp",
                payload: b"index-v1",
            })
            .unwrap();
        writer.push_fragment(&[0xff, 0x0a, 1], false).unwrap();
        assert_eq!(
            writer
                .push_box(ContainerBox {
                    box_type: *b"Exif",
                    payload: &[],
                })
                .unwrap_err(),
            Error::AuxiliaryBoxAfterCodestream
        );
        writer.push_fragment(&[2, 3], true).unwrap();
        assert_eq!(
            writer.push_fragment(&[4], true).unwrap_err(),
            Error::ContainerAlreadyFinished
        );
        let bytes = writer.finish().unwrap();
        let parsed = parse(&bytes, ParseLimits::default()).unwrap();
        assert_eq!(parsed.codestream(), [0xff, 0x0a, 1, 2, 3]);
        assert_eq!(
            parsed.boxes_of_type(*b"jwgp").collect::<Vec<_>>(),
            vec![ContainerBoxRef {
                box_type: *b"jwgp",
                payload: b"index-v1"
            }]
        );
    }

    #[test]
    fn fragmented_writer_requires_a_final_fragment() {
        let mut writer = FragmentedContainerWriter::new();
        writer.push_fragment(&[0xff, 0x0a], false).unwrap();
        assert_eq!(writer.finish().unwrap_err(), Error::MissingFinalFragment);
    }

    #[test]
    fn fragmented_writer_validates_a_signature_split_across_fragments() {
        let mut writer = FragmentedContainerWriter::new();
        writer.push_fragment(&[0xff], false).unwrap();
        writer.push_fragment(&[0x0a, 7], true).unwrap();
        let bytes = writer.finish().unwrap();
        assert_eq!(
            parse(&bytes, ParseLimits::default()).unwrap().codestream(),
            [0xff, 0x0a, 7]
        );
    }

    #[test]
    fn fragmented_writer_rejects_an_invalid_or_truncated_signature() {
        let mut invalid = FragmentedContainerWriter::new();
        invalid.push_fragment(&[0xff], false).unwrap();
        assert_eq!(
            invalid.push_fragment(&[0x09], true).unwrap_err(),
            Error::InvalidCodestreamSignature
        );

        let mut truncated = FragmentedContainerWriter::new();
        assert_eq!(
            truncated.push_fragment(&[0xff], true).unwrap_err(),
            Error::InvalidCodestreamSignature
        );
    }

    #[test]
    fn truncated_real_container_prefixes_never_panic() {
        let input = crate::test_fixtures::fragmented_animation();
        for length in (0..input.len()).step_by(97) {
            let _ = parse(&input[..length], ParseLimits::default());
        }
        let _ = parse(&input, ParseLimits::default()).unwrap();
    }

    #[test]
    fn container_requires_canonical_file_type_box() {
        let mut missing = CONTAINER_SIGNATURE_BOX.to_vec();
        push_jxlp(&mut missing, 1 << 31, &[0xff, 0x0a]);
        assert_eq!(
            parse(&missing, ParseLimits::default()).unwrap_err(),
            Error::MissingFileTypeBox
        );

        let mut bad_version = CONTAINER_SIGNATURE_BOX.to_vec();
        let mut file_type = CONTAINER_FILE_TYPE_BOX_V0;
        file_type[15] = 2;
        bad_version.extend_from_slice(&file_type);
        push_jxlp(&mut bad_version, 1 << 31, &[0xff, 0x0a]);
        assert_eq!(
            parse(&bad_version, ParseLimits::default()).unwrap_err(),
            Error::UnsupportedFileTypeVersion(2)
        );
    }

    #[test]
    fn file_type_v0_rejects_out_of_order_fragments() {
        let mut input = CONTAINER_SIGNATURE_BOX.to_vec();
        input.extend_from_slice(&CONTAINER_FILE_TYPE_BOX_V0);
        push_jxlp(&mut input, 1 | (1 << 31), &[2, 3]);
        push_jxlp(&mut input, 0, &[0xff, 0x0a, 1]);
        assert_eq!(
            parse(&input, ParseLimits::default()).unwrap_err(),
            Error::OutOfOrderFragment {
                expected: 0,
                actual: 1
            }
        );
    }

    fn push_jxlp(output: &mut Vec<u8>, index: u32, payload: &[u8]) {
        let size = 12u32 + u32::try_from(payload.len()).unwrap();
        output.extend_from_slice(&size.to_be_bytes());
        output.extend_from_slice(&JXLP);
        output.extend_from_slice(&index.to_be_bytes());
        output.extend_from_slice(payload);
    }
}
