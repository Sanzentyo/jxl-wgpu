use std::sync::Arc;

use super::{
    EntropyStreamWindows, GroupEntropyRange, GroupStreamSegment, STREAM_SENTINEL_BYTES, align4,
};
use crate::{Error, Result};

/// O(groups) storage, including when a group describes millions of overlapping windows.
#[derive(Clone, Debug, Default)]
pub(crate) struct EntropyStreamPlan {
    runs: Arc<[BatchRun]>,
    batch_count: usize,
    stream_bytes: u64,
    max_group_count: usize,
    uses_windows: bool,
}

#[derive(Debug)]
struct BatchRun {
    first_batch: usize,
    kind: BatchRunKind,
}

#[derive(Debug)]
enum BatchRunKind {
    Packed(Box<[GroupStreamSegment]>),
    Windowed {
        group_index: usize,
        windows: EntropyStreamWindows,
    },
}

/// A borrowed packed batch or one derived window. Neither form allocates on iteration.
#[derive(Clone, Copy, Debug)]
pub(crate) enum StreamBatch<'a> {
    Packed(&'a [GroupStreamSegment]),
    Window(GroupStreamSegment),
}

impl StreamBatch<'_> {
    pub(crate) fn segments(&self) -> &[GroupStreamSegment] {
        match self {
            Self::Packed(segments) => segments,
            Self::Window(segment) => std::slice::from_ref(segment),
        }
    }

    pub(crate) fn first_group(&self) -> usize {
        self.segments()[0].group_index
    }

    pub(crate) fn group_count(&self) -> usize {
        self.segments().len()
    }
}

impl BatchRun {
    fn len(&self) -> usize {
        match &self.kind {
            BatchRunKind::Packed(_) => 1,
            BatchRunKind::Windowed { windows, .. } => windows.len(),
        }
    }

    fn batch(&self, index: usize) -> Option<StreamBatch<'_>> {
        match &self.kind {
            BatchRunKind::Packed(segments) => (index == 0).then_some(StreamBatch::Packed(segments)),
            BatchRunKind::Windowed {
                group_index,
                windows,
            } => windows.get(index).map(|mut segment| {
                segment.group_index = *group_index;
                StreamBatch::Window(segment)
            }),
        }
    }
}

impl EntropyStreamPlan {
    pub(crate) fn new(
        codestream_bytes: u64,
        groups: &[GroupEntropyRange],
        stream_limit: u64,
        max_groups_per_batch: usize,
    ) -> Result<Self> {
        if max_groups_per_batch == 0 {
            return Err(Error::backend(
                "bounded entropy stream batch has zero group lanes",
            ));
        }
        if groups.is_empty() {
            return Err(Error::backend("entropy stream batch layout is empty"));
        }
        let mut runs = Vec::with_capacity(groups.len());
        let mut batch_count = 0usize;
        let mut stream_bytes = 0u64;
        let mut max_group_count = 0usize;
        let mut uses_windows = false;
        let mut push = |kind| -> Result<()> {
            let (count, lanes, bytes) = match &kind {
                BatchRunKind::Packed(segments) => {
                    let last = segments.last().ok_or(Error::EngineContract(
                        "packed entropy stream batch is empty",
                    ))?;
                    let end = (last.upload_offset as u64)
                        .checked_add((last.input_end - last.input_start) as u64)
                        .ok_or_else(|| Error::backend("group stream batch cursor overflow"))?;
                    (1, segments.len(), align4(end)? + STREAM_SENTINEL_BYTES)
                }
                BatchRunKind::Windowed { windows, .. } => {
                    uses_windows = true;
                    (windows.len(), 1, windows.stream_bytes())
                }
            };
            let first_batch = batch_count;
            batch_count = batch_count.checked_add(count).ok_or_else(|| {
                Error::backend("entropy stream batch count exceeds host address space")
            })?;
            stream_bytes = stream_bytes.max(bytes);
            max_group_count = max_group_count.max(lanes);
            runs.push(BatchRun { first_batch, kind });
            Ok(())
        };
        let mut packed = Vec::new();
        let mut upload_cursor = 0u64;
        for (group_index, range) in groups.iter().copied().enumerate() {
            let windows = EntropyStreamWindows::new(codestream_bytes, range, stream_limit)?;
            if windows.len() > 1 {
                if !packed.is_empty() {
                    push(BatchRunKind::Packed(
                        std::mem::take(&mut packed).into_boxed_slice(),
                    ))?;
                }
                upload_cursor = 0;
                push(BatchRunKind::Windowed {
                    group_index,
                    windows,
                })?;
                continue;
            }
            let mut segment = windows
                .get(0)
                .expect("an entropy stream has one or more windows");
            let packet_bytes = (segment.input_end - segment.input_start) as u64;
            let mut segment_start = align4(upload_cursor)?;
            let batch_bytes = segment_start
                .checked_add(packet_bytes)
                .and_then(|bytes| align4(bytes).ok())
                .and_then(|bytes| bytes.checked_add(STREAM_SENTINEL_BYTES))
                .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
            if !packed.is_empty()
                && (batch_bytes > stream_limit || packed.len() >= max_groups_per_batch)
            {
                push(BatchRunKind::Packed(
                    std::mem::take(&mut packed).into_boxed_slice(),
                ))?;
                segment_start = 0;
            }
            segment.group_index = group_index;
            segment.upload_offset = usize::try_from(segment_start)
                .map_err(|_| Error::backend("group upload offset exceeds host address space"))?;
            segment.window_upload_start = segment_start
                .checked_mul(8)
                .and_then(|bits| bits.checked_add(u64::from(segment.window_upload_start)))
                .and_then(|bits| u32::try_from(bits).ok())
                .ok_or_else(|| Error::backend("group stream start exceeds WGSL u32"))?;
            packed.push(segment);
            upload_cursor = segment_start
                .checked_add(packet_bytes)
                .ok_or_else(|| Error::backend("group stream batch cursor overflow"))?;
        }
        if !packed.is_empty() {
            push(BatchRunKind::Packed(packed.into_boxed_slice()))?;
        }
        Ok(Self {
            runs: runs.into(),
            batch_count,
            stream_bytes,
            max_group_count,
            uses_windows,
        })
    }

    pub(crate) fn batches(&self) -> impl Iterator<Item = StreamBatch<'_>> {
        self.runs.iter().flat_map(|run| {
            (0..run.len()).map(move |index| run.batch(index).expect("planned entropy batch"))
        })
    }

    pub(crate) fn batch(&self, index: usize) -> Option<StreamBatch<'_>> {
        if index >= self.batch_count {
            return None;
        }
        let run = &self.runs[self.runs.partition_point(|run| run.first_batch <= index) - 1];
        run.batch(index - run.first_batch)
    }

    pub(crate) const fn batch_count(&self) -> usize {
        self.batch_count
    }

    pub(crate) const fn stream_bytes(&self) -> u64 {
        self.stream_bytes
    }

    pub(crate) const fn max_group_count(&self) -> usize {
        self.max_group_count
    }

    pub(crate) const fn uses_windows(&self) -> bool {
        self.uses_windows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_batches_preserve_ranges_overlap_lanes_and_exact_peaks() {
        for skew in 0..8 {
            let mut offset = skew;
            let groups = [0, 1, 31, 32, 59, 2049, 0, 85, 8195, 7, 99].map(|length| {
                let range = GroupEntropyRange {
                    token_bit_offset: offset,
                    token_bit_end: offset + length,
                };
                offset += length + 3;
                range
            });
            for cap in [40, 41, 44, 64, 256, 1024, 4096] {
                for lanes in [1, 2, 4, 8] {
                    let plan =
                        EntropyStreamPlan::new(offset.div_ceil(8), &groups, cap, lanes).unwrap();
                    let mut covered = [0; 11];
                    let mut seen = [0; 11];
                    let mut final_counts = [0; 11];
                    let mut peak = 0;
                    let mut max_lanes = 0;
                    let mut batch_count = 0;
                    let mut uses_windows = false;
                    for (index, batch) in plan.batches().enumerate() {
                        assert_eq!(plan.batch(index).unwrap().segments(), batch.segments());
                        assert!(batch.group_count() <= lanes);
                        max_lanes = max_lanes.max(batch.group_count());
                        batch_count += 1;
                        let mut end = 0;
                        for (lane, segment) in batch.segments().iter().enumerate() {
                            let group = segment.group_index;
                            let range = groups[group];
                            assert_eq!(group, batch.first_group() + lane);
                            assert_eq!(segment.upload_offset % 4, 0);
                            assert!(segment.upload_offset >= end);
                            end = segment.upload_offset + segment.input_end - segment.input_start;
                            assert_eq!(
                                u64::from(segment.window_upload_start),
                                segment.upload_offset as u64 * 8 + (range.token_bit_offset & 7)
                            );
                            assert_eq!(
                                segment.input_start as u64,
                                range.token_bit_offset / 8
                                    + u64::from(segment.window_logical_start) / 8
                            );
                            assert_eq!(
                                u64::from(segment.stream_token_end),
                                range.token_bit_end - range.token_bit_offset
                            );
                            assert_eq!(
                                segment.flags & GroupStreamSegment::FIRST != 0,
                                seen[group] == 0
                            );
                            assert!(segment.window_logical_start <= covered[group]);
                            assert!(segment.available_token_end >= segment.window_yield_end);
                            covered[group] = segment.available_token_end;
                            seen[group] += 1;
                            final_counts[group] +=
                                usize::from(segment.flags & GroupStreamSegment::FINAL != 0);
                            uses_windows |= segment.flags
                                != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL);
                        }
                        let bytes = (end as u64).div_ceil(4) * 4 + STREAM_SENTINEL_BYTES;
                        assert!(bytes <= cap);
                        peak = peak.max(bytes);
                    }
                    for (index, range) in groups.iter().enumerate() {
                        assert!(seen[index] != 0);
                        assert_eq!(final_counts[index], 1);
                        assert_eq!(
                            u64::from(covered[index]),
                            range.token_bit_end - range.token_bit_offset
                        );
                    }
                    assert_eq!(plan.batch_count(), batch_count);
                    assert_eq!(plan.stream_bytes(), peak);
                    assert_eq!(plan.max_group_count(), max_lanes);
                    assert_eq!(plan.uses_windows(), uses_windows);
                    assert!(plan.batch(batch_count).is_none());
                    assert!(plan.runs.len() <= groups.len());
                }
            }
        }
    }

    #[test]
    fn huge_window_counts_retain_only_group_geometry_and_support_direct_lookup() {
        let end = 7 + u64::from(u32::MAX);
        let groups = [
            GroupEntropyRange {
                token_bit_offset: 0,
                token_bit_end: 3,
            },
            GroupEntropyRange {
                token_bit_offset: 7,
                token_bit_end: end,
            },
            GroupEntropyRange {
                token_bit_offset: end + 3,
                token_bit_end: end + 131,
            },
        ];
        let plan = EntropyStreamPlan::new((end + 131).div_ceil(8), &groups, 40, 8).unwrap();
        assert!(plan.batch_count() > 134_000_000);
        assert_eq!(plan.runs.len(), 3);
        assert_eq!(plan.stream_bytes(), 40);
        assert_eq!(plan.max_group_count(), 1);
        assert_eq!(plan.batch(0).unwrap().first_group(), 0);
        let first = plan.batch(1).unwrap();
        assert_eq!(first.first_group(), 1);
        assert_eq!(first.segments()[0].flags, GroupStreamSegment::FIRST);
        let last = plan.batch(plan.batch_count() - 2).unwrap();
        assert_eq!(last.first_group(), 1);
        assert_eq!(last.segments()[0].flags, GroupStreamSegment::FINAL);
        assert_eq!(last.segments()[0].available_token_end, u32::MAX);
        assert_eq!(plan.batch(plan.batch_count() - 1).unwrap().first_group(), 2);
        assert!(plan.batch(plan.batch_count()).is_none());
        let cloned = plan.clone();
        assert!(Arc::ptr_eq(&plan.runs, &cloned.runs));
    }
}
