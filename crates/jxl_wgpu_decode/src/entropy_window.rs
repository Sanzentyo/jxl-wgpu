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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StreamBatch {
    pub segments: Range<usize>,
    pub first_group: usize,
    pub group_count: usize,
}

pub(crate) fn build_stream_batches(
    codestream: &[u8],
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
        let input = codestream
            .get(input_start..input_end)
            .ok_or_else(|| Error::backend("group stream window exceeds the codestream"))?;
        let packet_bytes = u64::try_from(input.len())
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
            let maximum_input_bytes = ((stream_limit - STREAM_SENTINEL_BYTES) / 4) * 4;
            let core_bytes = maximum_input_bytes
                .checked_sub(2 * STREAM_OVERLAP_BYTES)
                .ok_or(Error::StreamWindowTooSmall {
                    limit_bytes: stream_limit,
                    minimum_bytes: MIN_STREAM_WINDOW_BYTES,
                })?;
            let mut core_start = 0u64;
            while core_start < packet_bytes {
                let core_end = core_start.saturating_add(core_bytes).min(packet_bytes);
                let relative_input_start = core_start.saturating_sub(STREAM_OVERLAP_BYTES);
                let relative_input_end = core_end
                    .saturating_add(STREAM_OVERLAP_BYTES)
                    .min(packet_bytes);
                let input_len = relative_input_end - relative_input_start;
                let uploaded_bytes = align4(input_len)?
                    .checked_add(STREAM_SENTINEL_BYTES)
                    .ok_or_else(|| Error::backend("group stream segment size overflow"))?;
                if uploaded_bytes > stream_limit {
                    return Err(Error::backend(
                        "bounded entropy overlap exceeded the resolved stream window",
                    ));
                }
                let logical_start = relative_input_start
                    .checked_mul(8)
                    .and_then(|bits| u32::try_from(bits).ok())
                    .ok_or_else(|| Error::backend("stream segment start exceeds WGSL u32"))?;
                let available_bits = input_len
                    .checked_mul(8)
                    .and_then(|bits| bits.checked_sub(u64::from(leading_bits)))
                    .and_then(|bits| bits.checked_add(u64::from(logical_start)))
                    .and_then(|bits| u32::try_from(bits).ok())
                    .ok_or_else(|| Error::backend("stream segment end exceeds WGSL u32"))?;
                let yield_end = core_end
                    .checked_mul(8)
                    .and_then(|bits| u32::try_from(bits).ok())
                    .ok_or_else(|| Error::backend("stream yield boundary exceeds WGSL u32"))?
                    .min(token_length);
                let is_first = core_start == 0;
                let is_final = core_end == packet_bytes;
                let segment_index = segments.len();
                segments.push(GroupStreamSegment {
                    group_index,
                    input_start: input_start
                        .checked_add(usize::try_from(relative_input_start).map_err(|_| {
                            Error::backend("stream segment start exceeds host address space")
                        })?)
                        .ok_or_else(|| Error::backend("stream segment start overflow"))?,
                    input_end: input_start
                        .checked_add(usize::try_from(relative_input_end).map_err(|_| {
                            Error::backend("stream segment end exceeds host address space")
                        })?)
                        .ok_or_else(|| Error::backend("stream segment end overflow"))?,
                    upload_offset: 0,
                    window_logical_start: logical_start,
                    window_upload_start: leading_bits,
                    available_token_end: available_bits.min(token_length),
                    stream_token_end: token_length,
                    window_yield_end: yield_end,
                    flags: (u32::from(is_first) * GroupStreamSegment::FIRST)
                        | (u32::from(is_final) * GroupStreamSegment::FINAL),
                });
                batches.push(StreamBatch {
                    segments: segment_index..segment_index + 1,
                    first_group: group_index,
                    group_count: 1,
                });
                maximum_batch_bytes = maximum_batch_bytes.max(uploaded_bytes);
                core_start = core_end;
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
