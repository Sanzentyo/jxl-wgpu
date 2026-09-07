//! Consumer-neutral bounded uploads for independent entropy-coded group streams.

use std::ops::Range;

use crate::{Error, Result};

pub(crate) const STREAM_SENTINEL_BYTES: u64 = 4;
// One complete output token consumes at most 94 bits: two 16-bit ANS refills and two 31-bit
// hybrid payloads in the LZ length/distance path. Sixteen bytes cover that token after the
// maximum seven-bit stream-start skew and give the following segment the same overshoot.
pub(crate) const STREAM_OVERLAP_BYTES: u64 = 16;
// Sentinel + two overlaps + one aligned four-byte core.
pub(crate) const MIN_STREAM_WINDOW_BYTES: u64 = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GroupEntropyRange {
    pub token_bit_offset: u64,
    pub token_bit_end: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GroupStreamSegment {
    pub group_index: usize,
    pub input_start: usize,
    pub input_end: usize,
    pub upload_offset: usize,
    pub window_logical_start: u32,
    pub window_upload_start: u32,
    pub available_token_end: u32,
    pub stream_token_end: u32,
    pub window_yield_end: u32,
    pub flags: u32,
}

impl GroupStreamSegment {
    pub const FIRST: u32 = 1 << 0;
    pub const FINAL: u32 = 1 << 1;
}

/// Constant-size geometry for a cursor-delimited stream. Windows are generated only as consumed,
/// so a short prefix inside a large packet does not allocate a table for its unconsumed suffix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EntropyStreamWindows {
    input_start: usize,
    packet_bytes: u64,
    token_length: u32,
    leading_bits: u32,
    core_bytes: u64,
    count: usize,
}

impl EntropyStreamWindows {
    pub(crate) fn new(codestream_bytes: u64, range: GroupEntropyRange, limit: u64) -> Result<Self> {
        if limit < MIN_STREAM_WINDOW_BYTES {
            return Err(Error::StreamWindowTooSmall {
                limit_bytes: limit,
                minimum_bytes: MIN_STREAM_WINDOW_BYTES,
            });
        }
        let token_length = range
            .token_bit_end
            .checked_sub(range.token_bit_offset)
            .and_then(|length| u32::try_from(length).ok())
            .ok_or_else(|| Error::backend("entropy stream length exceeds WGSL u32"))?;
        let input_start = usize::try_from(range.token_bit_offset / 8)
            .map_err(|_| Error::backend("entropy stream start exceeds host address space"))?;
        let end = range
            .token_bit_end
            .checked_add(7)
            .ok_or_else(|| Error::backend("entropy stream end overflow"))?
            / 8;
        if end > codestream_bytes || usize::try_from(end).is_err() {
            return Err(Error::backend(
                "entropy stream window exceeds the codestream",
            ));
        }
        let packet_bytes = end - input_start as u64;
        let whole_bytes = align4(packet_bytes)? + STREAM_SENTINEL_BYTES;
        let core_bytes = if whole_bytes <= limit {
            packet_bytes.max(1)
        } else {
            ((limit - STREAM_SENTINEL_BYTES) / 4) * 4 - 2 * STREAM_OVERLAP_BYTES
        };
        let count = usize::try_from(packet_bytes.div_ceil(core_bytes).max(1)).map_err(|_| {
            Error::backend("entropy stream window count exceeds host address space")
        })?;
        Ok(Self {
            input_start,
            packet_bytes,
            token_length,
            leading_bits: (range.token_bit_offset & 7) as u32,
            core_bytes,
            count,
        })
    }

    pub(crate) const fn len(self) -> usize {
        self.count
    }

    pub(crate) fn get(self, index: usize) -> Option<GroupStreamSegment> {
        if index >= self.count {
            return None;
        }
        let core_start = index as u64 * self.core_bytes;
        let core_end = (core_start + self.core_bytes).min(self.packet_bytes);
        let start = core_start.saturating_sub(STREAM_OVERLAP_BYTES);
        let end = (core_end + STREAM_OVERLAP_BYTES).min(self.packet_bytes);
        Some(GroupStreamSegment {
            group_index: 0,
            input_start: self.input_start + start as usize,
            input_end: self.input_start + end as usize,
            upload_offset: 0,
            window_logical_start: (start * 8) as u32,
            window_upload_start: self.leading_bits,
            available_token_end: (end * 8)
                .saturating_sub(u64::from(self.leading_bits))
                .min(u64::from(self.token_length)) as u32,
            stream_token_end: self.token_length,
            window_yield_end: (core_end * 8).min(u64::from(self.token_length)) as u32,
            flags: (u32::from(index == 0) * GroupStreamSegment::FIRST)
                | (u32::from(index + 1 == self.count) * GroupStreamSegment::FINAL),
        })
    }

    pub(crate) fn stream_bytes(self) -> u64 {
        // Both central windows cover the plateau even when the final core is partial.
        [0, (self.count - 1) / 2, self.count / 2, self.count - 1]
            .into_iter()
            .filter_map(|index| self.get(index))
            .map(|segment| (segment.input_end - segment.input_start) as u64)
            .map(|bytes| bytes.div_ceil(4) * 4 + STREAM_SENTINEL_BYTES)
            .max()
            .expect("an entropy stream always has a window")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StreamBatch {
    pub segments: Range<usize>,
    pub first_group: usize,
    pub group_count: usize,
}

pub(crate) fn build_stream_batches_for_len(
    codestream_bytes: u64,
    groups: &[GroupEntropyRange],
    stream_limit: u64,
    max_groups_per_batch: usize,
) -> Result<(Vec<GroupStreamSegment>, Vec<StreamBatch>, u64)> {
    if max_groups_per_batch == 0 {
        return Err(Error::backend(
            "bounded entropy stream batch has zero group lanes",
        ));
    }
    if stream_limit < MIN_STREAM_WINDOW_BYTES {
        return Err(Error::StreamWindowTooSmall {
            limit_bytes: stream_limit,
            minimum_bytes: MIN_STREAM_WINDOW_BYTES,
        });
    }
    let mut segments = Vec::with_capacity(groups.len());
    let mut batches = Vec::new();
    let mut batch_start_segment = 0usize;
    let mut batch_first_group = 0usize;
    let mut batch_group_count = 0usize;
    let mut upload_cursor = 0u64;
    let mut maximum_batch_bytes = 0u64;

    let flush_batch = |segments: &[GroupStreamSegment],
                       batches: &mut Vec<StreamBatch>,
                       batch_start_segment: usize,
                       batch_first_group: usize,
                       batch_group_count: usize,
                       upload_cursor: u64,
                       maximum_batch_bytes: &mut u64|
     -> Result<()> {
        if batch_group_count == 0 {
            return Ok(());
        }
        let bytes = align4(upload_cursor)?
            .checked_add(STREAM_SENTINEL_BYTES)
            .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
        *maximum_batch_bytes = (*maximum_batch_bytes).max(bytes);
        batches.push(StreamBatch {
            segments: batch_start_segment..segments.len(),
            first_group: batch_first_group,
            group_count: batch_group_count,
        });
        Ok(())
    };

    for (group_index, group) in groups.iter().copied().enumerate() {
        let input_start = usize::try_from(group.token_bit_offset / 8)
            .map_err(|_| Error::backend("group stream start exceeds host address space"))?;
        let input_end = usize::try_from(
            group
                .token_bit_end
                .checked_add(7)
                .ok_or_else(|| Error::backend("group stream end overflow"))?
                / 8,
        )
        .map_err(|_| Error::backend("group stream end exceeds host address space"))?;
        let input_end_u64 =
            u64::try_from(input_end).map_err(|_| Error::backend("group stream end exceeds u64"))?;
        if input_end_u64 > codestream_bytes {
            return Err(Error::backend("group stream window exceeds the codestream"));
        }
        let packet_bytes = u64::try_from(
            input_end
                .checked_sub(input_start)
                .ok_or_else(|| Error::backend("group stream byte range underflow"))?,
        )
        .map_err(|_| Error::backend("group stream size exceeds u64"))?;
        let group_packet_bytes = align4(packet_bytes)?
            .checked_add(STREAM_SENTINEL_BYTES)
            .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
        let token_length = group
            .token_bit_end
            .checked_sub(group.token_bit_offset)
            .and_then(|bits| u32::try_from(bits).ok())
            .ok_or_else(|| Error::backend("group stream length exceeds WGSL u32"))?;
        let leading_bits = u32::try_from(group.token_bit_offset & 7)
            .map_err(|_| Error::backend("group leading-bit count exceeds WGSL u32"))?;

        if group_packet_bytes > stream_limit {
            flush_batch(
                &segments,
                &mut batches,
                batch_start_segment,
                batch_first_group,
                batch_group_count,
                upload_cursor,
                &mut maximum_batch_bytes,
            )?;
            batch_group_count = 0;
            upload_cursor = 0;
            let windows = EntropyStreamWindows::new(codestream_bytes, group, stream_limit)?;
            maximum_batch_bytes = maximum_batch_bytes.max(windows.stream_bytes());
            for window_index in 0..windows.len() {
                let segment_index = segments.len();
                let mut segment = windows.get(window_index).expect("planned entropy window");
                segment.group_index = group_index;
                segments.push(segment);
                batches.push(StreamBatch {
                    segments: segment_index..segment_index + 1,
                    first_group: group_index,
                    group_count: 1,
                });
            }
            batch_start_segment = segments.len();
            continue;
        }

        let mut segment_start = align4(upload_cursor)?;
        let batch_bytes = segment_start
            .checked_add(packet_bytes)
            .and_then(|bytes| align4(bytes).ok())
            .and_then(|bytes| bytes.checked_add(STREAM_SENTINEL_BYTES))
            .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
        if batch_group_count != 0
            && (batch_bytes > stream_limit || batch_group_count >= max_groups_per_batch)
        {
            flush_batch(
                &segments,
                &mut batches,
                batch_start_segment,
                batch_first_group,
                batch_group_count,
                upload_cursor,
                &mut maximum_batch_bytes,
            )?;
            batch_start_segment = segments.len();
            batch_first_group = group_index;
            batch_group_count = 0;
            segment_start = 0;
        }
        if batch_group_count == 0 {
            batch_start_segment = segments.len();
            batch_first_group = group_index;
        }
        let segment_start_bits = segment_start
            .checked_mul(8)
            .ok_or_else(|| Error::backend("group stream bit offset overflow"))?;
        let window_upload_start = segment_start_bits
            .checked_add(u64::from(leading_bits))
            .and_then(|bits| u32::try_from(bits).ok())
            .ok_or_else(|| Error::backend("group stream start exceeds WGSL u32"))?;
        segments.push(GroupStreamSegment {
            group_index,
            input_start,
            input_end,
            upload_offset: usize::try_from(segment_start)
                .map_err(|_| Error::backend("group upload offset exceeds host address space"))?,
            window_logical_start: 0,
            window_upload_start,
            available_token_end: token_length,
            stream_token_end: token_length,
            window_yield_end: token_length,
            flags: GroupStreamSegment::FIRST | GroupStreamSegment::FINAL,
        });
        batch_group_count += 1;
        upload_cursor = segment_start
            .checked_add(packet_bytes)
            .ok_or_else(|| Error::backend("group stream batch cursor overflow"))?;
    }
    flush_batch(
        &segments,
        &mut batches,
        batch_start_segment,
        batch_first_group,
        batch_group_count,
        upload_cursor,
        &mut maximum_batch_bytes,
    )?;
    if segments.len() < groups.len() || batches.is_empty() || maximum_batch_bytes == 0 {
        return Err(Error::backend("entropy stream batch layout is empty"));
    }
    Ok((segments, batches, maximum_batch_bytes))
}

fn align4(value: u64) -> Result<u64> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| Error::backend("entropy stream alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_windows_cover_unaligned_empty_and_u32_boundary_streams() {
        for start in 0..8 {
            for bits in (0..=1024).chain([4096]) {
                let range = GroupEntropyRange {
                    token_bit_offset: start,
                    token_bit_end: start + bits,
                };
                for limit in [40, 41, 44, 45, 48, 52, 56, 60, 64, 68, 128, 512, u64::MAX] {
                    let plan = EntropyStreamWindows::new((start + bits).div_ceil(8), range, limit)
                        .unwrap();
                    let mut peak = 0;
                    let mut covered_until = 0;
                    for index in 0..plan.len() {
                        let segment = plan.get(index).unwrap();
                        assert_eq!(segment.window_upload_start, start as u32);
                        assert_eq!(segment.stream_token_end, bits as u32);
                        assert!(u64::from(segment.window_logical_start) <= covered_until);
                        assert!(segment.available_token_end >= segment.window_yield_end);
                        covered_until = u64::from(segment.available_token_end);
                        let size =
                            ((segment.input_end - segment.input_start) as u64).div_ceil(4) * 4 + 4;
                        peak = peak.max(size);
                    }
                    assert_eq!(covered_until, bits);
                    assert_eq!(peak, plan.stream_bytes());
                    assert!(peak <= limit);
                    assert!(plan.get(plan.len()).is_none());
                }
            }
        }
        // This describes over 134 million tiny windows without allocating a segment table.
        let range = GroupEntropyRange {
            token_bit_offset: 7,
            token_bit_end: 7 + u64::from(u32::MAX),
        };
        let plan = EntropyStreamWindows::new(range.token_bit_end.div_ceil(8), range, 40).unwrap();
        let last = plan.get(plan.len() - 1).unwrap();
        assert_eq!(last.available_token_end, u32::MAX);
        assert_eq!(last.window_yield_end, u32::MAX);
        assert_eq!(last.flags, GroupStreamSegment::FINAL);
        assert_eq!(plan.stream_bytes(), 40);
    }
}
