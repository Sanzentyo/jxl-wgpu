//! Bounded JPEG XL `jxli` wire metadata. Binding entries to real keyframes is a separate check.

use std::num::NonZeroU32;
use std::sync::Arc;

use crate::ParsedJxl;

pub const FRAME_INDEX_BOX_TYPE: [u8; 4] = *b"jxli";
const MAX_VARINT: u64 = i64::MAX as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameIndexLimits {
    pub max_payload_bytes: u64,
    pub max_entries: usize,
    pub max_frames: u64,
}

impl Default for FrameIndexLimits {
    fn default() -> Self {
        Self {
            max_payload_bytes: 1 << 20,
            max_entries: 16_384,
            max_frames: 16_384,
        }
    }
}

/// Absolute logical-codestream offset and the interval ending at the next entry or stream end.
/// `frames` counts displayed presentations, including the presentation at this entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameIndexEntry {
    pub codestream_offset: u64,
    pub duration_ticks: u64,
    pub frames: u64,
}

/// Structurally validated index. This alone does not authorize a decoder restart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameIndex {
    tick_numerator: u32,
    tick_denominator: NonZeroU32,
    entries: Arc<[FrameIndexEntry]>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FrameIndexError {
    #[error("jxli payload exceeds its byte limit")]
    PayloadLimit,
    #[error("jxli entry count exceeds its limit")]
    EntryLimit,
    #[error("jxli displayed-frame count exceeds its limit")]
    FrameLimit,
    #[error("jxli must contain its first frame")]
    Empty,
    #[error("jxli is truncated")]
    Truncated,
    #[error("jxli has trailing bytes")]
    TrailingBytes,
    #[error("jxli tick denominator is zero")]
    ZeroDenominator,
    #[error("jxli variable integer exceeds 63 bits or its bounded encoding")]
    InvalidVarint,
    #[error("jxli offsets are not strictly increasing")]
    OffsetOrder,
    #[error("jxli frame intervals must be nonempty")]
    EmptyFrameInterval,
    #[error("jxli cumulative offset, time, count or size overflows")]
    Overflow,
    #[error("a JPEG XL container may contain only one jxli box")]
    DuplicateBox,
    #[error("compressed jxli boxes are not supported")]
    CompressedBox,
    #[error("allocation failed for bounded jxli entries")]
    Allocation,
}

impl FrameIndex {
    pub fn new(
        tick_numerator: u32,
        tick_denominator: NonZeroU32,
        entries: Vec<FrameIndexEntry>,
        limits: FrameIndexLimits,
    ) -> Result<Self, FrameIndexError> {
        validate(&entries, limits)?;
        Ok(Self {
            tick_numerator,
            tick_denominator,
            entries: entries.into(),
        })
    }

    pub fn parse(payload: &[u8], limits: FrameIndexLimits) -> Result<Self, FrameIndexError> {
        if payload.len() as u64 > limits.max_payload_bytes {
            return Err(FrameIndexError::PayloadLimit);
        }
        let mut reader = Reader(payload);
        let count = usize::try_from(reader.varint()?).map_err(|_| FrameIndexError::EntryLimit)?;
        if count > limits.max_entries {
            return Err(FrameIndexError::EntryLimit);
        }
        if count as u64 > limits.max_frames {
            return Err(FrameIndexError::FrameLimit);
        }
        if count == 0 {
            return Err(FrameIndexError::Empty);
        }
        let numerator = reader.u32()?;
        let denominator = NonZeroU32::new(reader.u32()?).ok_or(FrameIndexError::ZeroDenominator)?;
        if count > reader.0.len() / 3 {
            return Err(FrameIndexError::Truncated);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(count)
            .map_err(|_| FrameIndexError::Allocation)?;
        let mut offset = 0_u64;
        for _ in 0..count {
            offset = offset
                .checked_add(reader.varint()?)
                .ok_or(FrameIndexError::Overflow)?;
            entries.push(FrameIndexEntry {
                codestream_offset: offset,
                duration_ticks: reader.varint()?,
                frames: reader.varint()?,
            });
        }
        if !reader.0.is_empty() {
            return Err(FrameIndexError::TrailingBytes);
        }
        Self::new(numerator, denominator, entries, limits)
    }

    /// Reads at most one plain `jxli`; ordinary image decoding need not request this metadata.
    pub fn from_container(
        parsed: &ParsedJxl<'_>,
        limits: FrameIndexLimits,
    ) -> Result<Option<Self>, FrameIndexError> {
        let mut index = None;
        for item in parsed.auxiliary_boxes() {
            if item.box_type == *b"brob" && item.payload.starts_with(&FRAME_INDEX_BOX_TYPE) {
                return Err(FrameIndexError::CompressedBox);
            }
            if item.box_type == FRAME_INDEX_BOX_TYPE {
                if index.is_some() {
                    return Err(FrameIndexError::DuplicateBox);
                }
                index = Some(Self::parse(item.payload, limits)?);
            }
        }
        Ok(index)
    }

    /// Emits canonical little-endian base-128 integers and big-endian tick-unit fields.
    pub fn encode(&self, limits: FrameIndexLimits) -> Result<Vec<u8>, FrameIndexError> {
        let size = validate(&self.entries, limits)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(size)
            .map_err(|_| FrameIndexError::Allocation)?;
        varint(&mut output, self.entries.len() as u64);
        output.extend_from_slice(&self.tick_numerator.to_be_bytes());
        output.extend_from_slice(&self.tick_denominator.get().to_be_bytes());
        let mut previous = 0;
        for entry in self.entries.iter() {
            varint(&mut output, entry.codestream_offset - previous);
            varint(&mut output, entry.duration_ticks);
            varint(&mut output, entry.frames);
            previous = entry.codestream_offset;
        }
        debug_assert_eq!(output.len(), size);
        Ok(output)
    }

    #[must_use]
    pub fn entries(&self) -> &[FrameIndexEntry] {
        &self.entries
    }

    #[must_use]
    pub const fn tick_numerator(&self) -> u32 {
        self.tick_numerator
    }

    #[must_use]
    pub const fn tick_denominator(&self) -> NonZeroU32 {
        self.tick_denominator
    }
}

fn validate(
    entries: &[FrameIndexEntry],
    limits: FrameIndexLimits,
) -> Result<usize, FrameIndexError> {
    if entries.is_empty() {
        return Err(FrameIndexError::Empty);
    }
    if entries.len() > limits.max_entries {
        return Err(FrameIndexError::EntryLimit);
    }
    let mut size = 8 + varint_size(entries.len() as u64);
    let (mut previous, mut ticks, mut frames) = (0_u64, 0_u64, 0_u64);
    for (index, entry) in entries.iter().enumerate() {
        if entry.codestream_offset > MAX_VARINT
            || entry.duration_ticks > MAX_VARINT
            || entry.frames > MAX_VARINT
        {
            return Err(FrameIndexError::InvalidVarint);
        }
        if index != 0 && entry.codestream_offset <= previous {
            return Err(FrameIndexError::OffsetOrder);
        }
        if entry.frames == 0 {
            return Err(FrameIndexError::EmptyFrameInterval);
        }
        ticks = ticks
            .checked_add(entry.duration_ticks)
            .ok_or(FrameIndexError::Overflow)?;
        frames = frames
            .checked_add(entry.frames)
            .ok_or(FrameIndexError::Overflow)?;
        if frames > limits.max_frames {
            return Err(FrameIndexError::FrameLimit);
        }
        size = size
            .checked_add(
                varint_size(entry.codestream_offset - previous)
                    + varint_size(entry.duration_ticks)
                    + varint_size(entry.frames),
            )
            .ok_or(FrameIndexError::Overflow)?;
        previous = entry.codestream_offset;
    }
    if size as u64 > limits.max_payload_bytes {
        return Err(FrameIndexError::PayloadLimit);
    }
    Ok(size)
}

fn varint_size(value: u64) -> usize {
    (64 - value.leading_zeros()).max(1).div_ceil(7) as usize
}

fn varint(output: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        output.push(value as u8 | 128);
        value >>= 7;
    }
    output.push(value as u8);
}

struct Reader<'a>(&'a [u8]);
impl Reader<'_> {
    fn u32(&mut self) -> Result<u32, FrameIndexError> {
        let bytes = self.0.get(..4).ok_or(FrameIndexError::Truncated)?;
        let value = u32::from_be_bytes(bytes.try_into().expect("four-byte range"));
        self.0 = &self.0[4..];
        Ok(value)
    }
    fn varint(&mut self) -> Result<u64, FrameIndexError> {
        let mut value = 0;
        for index in 0..10 {
            let (&byte, rest) = self.0.split_first().ok_or(FrameIndexError::Truncated)?;
            self.0 = rest;
            if index == 9 && byte != 0 {
                return Err(FrameIndexError::InvalidVarint);
            }
            value |= u64::from(byte & 127) << (7 * index);
            if byte < 128 {
                return Ok(value);
            }
        }
        Err(FrameIndexError::InvalidVarint)
    }
}

#[cfg(test)]
mod tests;
