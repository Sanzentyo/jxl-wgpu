//! Checked logical codestream spans used by metadata readers and bounded GPU uploads.

use std::ops::Range;
use std::sync::Arc;

use jxl_gpu_bitstream::StreamSlice;

use crate::{Error, Result, input_budget::IncrementalInputPermit, modular_tree::BitInput};

#[derive(Clone, Debug)]
struct CodestreamSpan {
    logical_offset: u64,
    bytes: StreamSlice,
}

/// A validated, logically contiguous codestream whose physical bytes may live in shared chunks.
///
/// The span table owns only shared ranges. It never assembles the complete codestream into a new
/// allocation; engines copy just the bounded metadata or GPU-upload range they are consuming.
#[derive(Clone, Debug)]
pub struct GpuCodestream {
    spans: Arc<[CodestreamSpan]>,
    logical_bytes: u64,
    container: bool,
    retained_input: Option<IncrementalInputPermit>,
}

impl GpuCodestream {
    pub(crate) fn from_shared(
        storage: Arc<[u8]>,
        byte_range: Range<usize>,
        container: bool,
    ) -> Result<Self> {
        let bytes = StreamSlice::from_shared_range(storage, byte_range).ok_or(
            Error::EngineContract("GPU codestream range is outside its shared storage"),
        )?;
        Self::from_spans_inner([(0, bytes)], container, None)
    }

    #[cfg(test)]
    pub(crate) fn from_spans(spans: impl IntoIterator<Item = (u64, StreamSlice)>) -> Result<Self> {
        Self::from_spans_inner(spans, false, None)
    }

    pub(crate) fn from_stream_spans(
        spans: impl IntoIterator<Item = (u64, StreamSlice)>,
        container: bool,
        retained_input: IncrementalInputPermit,
    ) -> Result<Self> {
        Self::from_spans_inner(spans, container, Some(retained_input))
    }

    fn from_spans_inner(
        spans: impl IntoIterator<Item = (u64, StreamSlice)>,
        container: bool,
        retained_input: Option<IncrementalInputPermit>,
    ) -> Result<Self> {
        let mut expected_offset = 0u64;
        let mut collected = Vec::new();
        for (logical_offset, bytes) in spans {
            if logical_offset != expected_offset {
                return Err(Error::EngineContract(
                    "codestream spans are not logically contiguous",
                ));
            }
            if bytes.is_empty() {
                continue;
            }
            expected_offset = expected_offset
                .checked_add(
                    u64::try_from(bytes.len())
                        .map_err(|_| Error::backend("codestream span length exceeds u64"))?,
                )
                .ok_or_else(|| Error::backend("codestream span offset overflow"))?;
            collected
                .try_reserve(1)
                .map_err(|_| Error::backend("codestream span table allocation failed"))?;
            collected.push(CodestreamSpan {
                logical_offset,
                bytes,
            });
        }
        if collected.is_empty() {
            return Err(Error::EngineContract("codestream span source is empty"));
        }
        Ok(Self {
            spans: collected.into(),
            logical_bytes: expected_offset,
            container,
            retained_input,
        })
    }

    #[must_use]
    pub const fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    #[must_use]
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    #[must_use]
    pub const fn is_container(&self) -> bool {
        self.container
    }

    /// Returns the complete logical codestream only when one physical span covers it.
    #[must_use]
    pub fn contiguous_bytes(&self) -> Option<&[u8]> {
        let span = self.spans.first()?;
        (self.spans.len() == 1 && span.logical_offset == 0).then(|| span.bytes.bytes())
    }

    /// Bytes admitted by the incremental-input budget and retained by this source.
    #[must_use]
    pub fn retained_input_bytes(&self) -> u64 {
        self.retained_input
            .as_ref()
            .map_or(0, IncrementalInputPermit::bytes)
    }

    pub(crate) fn logical_bits(&self) -> Result<u64> {
        self.logical_bytes
            .checked_mul(8)
            .ok_or_else(|| Error::backend("codestream bit length overflow"))
    }

    pub(crate) fn reader(&self) -> CodestreamBitReader<'_> {
        CodestreamBitReader::new(self)
    }

    pub fn for_each_range_chunk(
        &self,
        range: Range<u64>,
        mut visit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        if range.start > range.end || range.end > self.logical_bytes {
            return Err(Error::EngineContract(
                "codestream chunk range is outside the logical source",
            ));
        }
        let mut logical_cursor = range.start;
        let first_span = self
            .spans
            .partition_point(|span| span.logical_offset <= range.start)
            .saturating_sub(1);
        for span in &self.spans[first_span..] {
            if logical_cursor == range.end {
                break;
            }
            let span_end = span
                .logical_offset
                .checked_add(
                    u64::try_from(span.bytes.len())
                        .map_err(|_| Error::backend("codestream span length exceeds u64"))?,
                )
                .ok_or_else(|| Error::backend("codestream span end overflow"))?;
            if span_end <= logical_cursor || span.logical_offset >= range.end {
                continue;
            }
            let copy_start = logical_cursor.max(span.logical_offset);
            let copy_end = range.end.min(span_end);
            let source_start = usize::try_from(copy_start - span.logical_offset)
                .map_err(|_| Error::backend("codestream source offset exceeds host space"))?;
            let source_end = usize::try_from(copy_end - span.logical_offset)
                .map_err(|_| Error::backend("codestream source end exceeds host space"))?;
            visit(span.bytes.bytes().get(source_start..source_end).ok_or(
                Error::EngineContract("codestream span storage is truncated"),
            )?)?;
            logical_cursor = copy_end;
        }
        if logical_cursor != range.end {
            return Err(Error::EngineContract(
                "codestream span table does not cover the requested range",
            ));
        }
        Ok(())
    }

    pub fn copy_range(&self, range: Range<u64>, destination: &mut [u8]) -> Result<()> {
        let range_bytes = range
            .end
            .checked_sub(range.start)
            .ok_or_else(|| Error::backend("codestream copy range underflow"))?;
        if u64::try_from(destination.len())
            .map_err(|_| Error::backend("codestream copy destination exceeds u64"))?
            != range_bytes
        {
            return Err(Error::EngineContract(
                "codestream copy destination has the wrong length",
            ));
        }

        let mut destination_cursor = 0usize;
        self.for_each_range_chunk(range, |chunk| {
            let destination_end = destination_cursor
                .checked_add(chunk.len())
                .ok_or_else(|| Error::backend("codestream destination range overflow"))?;
            destination
                .get_mut(destination_cursor..destination_end)
                .ok_or(Error::EngineContract(
                    "codestream copy destination is truncated",
                ))?
                .copy_from_slice(chunk);
            destination_cursor = destination_end;
            Ok(())
        })?;
        if destination_cursor != destination.len() {
            return Err(Error::EngineContract(
                "codestream copy destination is truncated",
            ));
        }
        Ok(())
    }

    pub(crate) fn bits_are_zero(&self, start: u64, end: u64) -> Result<bool> {
        if start > end || end > self.logical_bits()? {
            return Ok(false);
        }
        let mut reader = self.reader();
        reader.skip_bits(start)?;
        for _ in start..end {
            if reader.read_bits(1)? != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn byte(&self, logical_offset: u64, hint: &mut usize) -> Result<u8> {
        while let Some(span) = self.spans.get(*hint) {
            let length = u64::try_from(span.bytes.len())
                .map_err(|_| Error::backend("codestream span length exceeds u64"))?;
            let end = span
                .logical_offset
                .checked_add(length)
                .ok_or_else(|| Error::backend("codestream span end overflow"))?;
            if logical_offset < span.logical_offset {
                break;
            }
            if logical_offset < end {
                let relative = usize::try_from(logical_offset - span.logical_offset)
                    .map_err(|_| Error::backend("codestream byte offset exceeds host space"))?;
                return span
                    .bytes
                    .bytes()
                    .get(relative)
                    .copied()
                    .ok_or(Error::EngineContract(
                        "codestream span storage is truncated",
                    ));
            }
            *hint = hint
                .checked_add(1)
                .ok_or_else(|| Error::backend("codestream span index overflow"))?;
        }
        Err(jxl_gpu_bitstream::Error::UnexpectedEndOfBits.into())
    }
}

/// Little-endian bit reader over a [`GpuCodestream`] span table.
#[derive(Clone, Debug)]
pub(crate) struct CodestreamBitReader<'source> {
    source: &'source GpuCodestream,
    bit_offset: u64,
    span_hint: usize,
}

impl<'source> CodestreamBitReader<'source> {
    const fn new(source: &'source GpuCodestream) -> Self {
        Self {
            source,
            bit_offset: 0,
            span_hint: 0,
        }
    }

    pub(crate) const fn bit_offset(&self) -> u64 {
        self.bit_offset
    }

    pub(crate) fn remaining_bits(&self) -> u64 {
        self.source
            .logical_bytes
            .saturating_mul(8)
            .saturating_sub(self.bit_offset)
    }

    pub(crate) fn read_bits(&mut self, count: u8) -> Result<u64> {
        if count > 56 {
            return Err(jxl_gpu_bitstream::Error::InvalidBitCount(count).into());
        }
        if self.remaining_bits() < u64::from(count) {
            return Err(jxl_gpu_bitstream::Error::UnexpectedEndOfBits.into());
        }
        let mut value = 0u64;
        for shift in 0..count {
            let byte = self.source.byte(self.bit_offset / 8, &mut self.span_hint)?;
            let bit = (byte >> (self.bit_offset & 7)) & 1;
            value |= u64::from(bit) << shift;
            self.bit_offset = self
                .bit_offset
                .checked_add(1)
                .ok_or_else(|| Error::backend("codestream bit offset overflow"))?;
        }
        Ok(value)
    }

    pub(crate) fn skip_bits(&mut self, count: u64) -> Result<()> {
        if self.remaining_bits() < count {
            return Err(jxl_gpu_bitstream::Error::UnexpectedEndOfBits.into());
        }
        self.bit_offset = self
            .bit_offset
            .checked_add(count)
            .ok_or_else(|| Error::backend("codestream bit offset overflow"))?;
        let byte_offset = self.bit_offset / 8;
        while let Some(span) = self.source.spans.get(self.span_hint) {
            let span_end = span
                .logical_offset
                .checked_add(
                    u64::try_from(span.bytes.len())
                        .map_err(|_| Error::backend("codestream span length exceeds u64"))?,
                )
                .ok_or_else(|| Error::backend("codestream span end overflow"))?;
            if byte_offset < span_end {
                break;
            }
            self.span_hint = self
                .span_hint
                .checked_add(1)
                .ok_or_else(|| Error::backend("codestream span index overflow"))?;
        }
        Ok(())
    }

    pub(crate) fn align_to_byte(&mut self) -> Result<()> {
        let aligned = self
            .bit_offset
            .checked_add(7)
            .ok_or_else(|| Error::backend("codestream byte alignment overflow"))?
            & !7;
        self.skip_bits(aligned - self.bit_offset)
    }
}

impl BitInput for CodestreamBitReader<'_> {
    fn bit_offset(&self) -> u64 {
        self.bit_offset()
    }

    fn read_bits(&mut self, count: u8) -> Result<u64> {
        self.read_bits(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_source(bytes: &[u8], split: usize) -> GpuCodestream {
        let storage: Arc<[u8]> = Arc::from(bytes);
        GpuCodestream::from_spans([
            (
                0,
                StreamSlice::from_shared_range(Arc::clone(&storage), 0..split).unwrap(),
            ),
            (
                split as u64,
                StreamSlice::from_shared_range(storage, split..bytes.len()).unwrap(),
            ),
        ])
        .unwrap()
    }

    #[test]
    fn bit_reader_and_range_copy_cross_every_chunk_split() {
        let bytes = [0x5a, 0xc3, 0x17, 0xe8, 0x4d];
        for split in 0..=bytes.len() {
            let source = split_source(&bytes, split);
            assert_eq!(source.logical_bits().unwrap(), 40);
            let mut reader = source.reader();
            assert_eq!(reader.read_bits(3).unwrap(), 2);
            assert_eq!(reader.read_bits(17).unwrap(), 0xf86b);
            reader.skip_bits(5).unwrap();
            assert_eq!(reader.bit_offset(), 25);
            assert_eq!(reader.read_bits(15).unwrap(), 0x26f4);
            assert_eq!(reader.remaining_bits(), 0);

            let mut copied = [0; 3];
            source.copy_range(1..4, &mut copied).unwrap();
            assert_eq!(copied, bytes[1..4]);

            let mut visited = Vec::new();
            source
                .for_each_range_chunk(1..4, |chunk| {
                    visited.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(visited, bytes[1..4]);
        }
        let storage: Arc<[u8]> = Arc::from(bytes);
        let byte_spans = (0..storage.len()).map(|offset| {
            (
                offset as u64,
                StreamSlice::from_shared_range(Arc::clone(&storage), offset..offset + 1).unwrap(),
            )
        });
        let source = GpuCodestream::from_spans(byte_spans).unwrap();
        let mut copied = [0; 3];
        source.copy_range(2..5, &mut copied).unwrap();
        assert_eq!(copied, bytes[2..5]);
    }

    #[test]
    fn rejects_gaps_overlaps_and_copy_contract_mismatches() {
        let bytes: Arc<[u8]> = Arc::from([1, 2, 3, 4]);
        let first = StreamSlice::from_shared_range(Arc::clone(&bytes), 0..2).unwrap();
        let second = StreamSlice::from_shared_range(bytes, 2..4).unwrap();
        assert!(GpuCodestream::from_spans([(0, first.clone()), (3, second.clone())]).is_err());
        assert!(GpuCodestream::from_spans([(0, first), (1, second)]).is_err());
        assert!(
            GpuCodestream::from_spans([(1, StreamSlice::from_shared(Arc::from(Vec::<u8>::new())))])
                .is_err()
        );

        let source = split_source(&[1, 2, 3, 4], 2);
        assert!(source.copy_range(1..3, &mut [0; 1]).is_err());
        assert!(source.copy_range(3..5, &mut [0; 2]).is_err());
    }

    #[test]
    fn checks_unaligned_zero_bits_across_span_boundaries() {
        for split in 0..=4 {
            let source = split_source(&[0xff, 0, 0, 0xff], split);
            assert!(source.bits_are_zero(8, 24).unwrap());
            assert!(!source.bits_are_zero(0, 9).unwrap());
            assert!(!source.bits_are_zero(7, 6).unwrap());
        }
        assert!(split_source(&[0b0000_0101], 0).bits_are_zero(3, 8).unwrap());
        assert!(!split_source(&[0b0001_0101], 1).bits_are_zero(3, 8).unwrap());
        assert!(
            split_source(&[0b0000_0111, 0, 0b1110_0000], 2)
                .bits_are_zero(3, 21)
                .unwrap()
        );
    }
}
