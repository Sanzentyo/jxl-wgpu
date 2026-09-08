//! Consumer-neutral bounded uploads for independent entropy-coded group streams.

mod plan;
pub(crate) use plan::{EntropyStreamPlan, StreamBatch};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
