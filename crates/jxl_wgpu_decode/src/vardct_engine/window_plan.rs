use std::sync::Arc;

use crate::entropy_window::{
    GroupEntropyRange, GroupStreamSegment, MIN_STREAM_WINDOW_BYTES, StreamBatch,
    build_stream_batches_for_len,
};
use crate::vardct_packet::{
    BoundedHfMetadataContinuation, BoundedVarDctPacketPlan, VarDctModularParams,
};
use crate::vardct_pass_group::HfCoefficientExecutionPlan;
use crate::{Error as DecodeError, GpuCodestream};

use super::types::{VarDctDecodeError, VarDctDecodeMemoryStats};

pub(super) struct VarDctEntropyPlanSelection {
    pub(super) stream_limit: u64,
    pub(super) lf_packet_windows: Option<LfPacketWindowExecutionPlan>,
    pub(super) combined_packet_windows: Option<CombinedPacketWindowExecutionPlan>,
    pub(super) hf_coefficients: Option<HfCoefficientExecutionPlan>,
    pub(super) memory: VarDctDecodeMemoryStats,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AdaptiveStreamMemory {
    pub(super) total_frame_bytes: u64,
    pub(super) packet_stream_window_bytes: u64,
    pub(super) hf_stream_window_bytes: u64,
}

impl From<VarDctDecodeMemoryStats> for AdaptiveStreamMemory {
    fn from(memory: VarDctDecodeMemoryStats) -> Self {
        Self {
            total_frame_bytes: memory.total_frame_bytes,
            packet_stream_window_bytes: memory.packet_stream_window_bytes,
            hf_stream_window_bytes: memory.hf_stream_window_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdaptiveStreamLimitDecision {
    Selected(u64),
    BudgetTooSmall { required_bytes: u64 },
}

pub(super) fn select_budget_adaptive_stream_limit(
    configured_limit: u64,
    memory_limit_bytes: u64,
    mut memory_at_limit: impl FnMut(u64) -> Result<AdaptiveStreamMemory, VarDctDecodeError>,
) -> Result<AdaptiveStreamLimitDecision, VarDctDecodeError> {
    let configured_limit = configured_limit & !3;
    let configured = memory_at_limit(configured_limit)?;
    if configured.total_frame_bytes <= memory_limit_bytes {
        return Ok(AdaptiveStreamLimitDecision::Selected(configured_limit));
    }

    let minimum = memory_at_limit(MIN_STREAM_WINDOW_BYTES)?;
    if minimum.total_frame_bytes > memory_limit_bytes {
        return Ok(AdaptiveStreamLimitDecision::BudgetTooSmall {
            required_bytes: minimum.total_frame_bytes,
        });
    }

    let active_stream_windows = u64::from(minimum.packet_stream_window_bytes != 0)
        + u64::from(minimum.hf_stream_window_bytes != 0);
    debug_assert!(active_stream_windows != 0);
    let non_stream_bytes = minimum
        .total_frame_bytes
        .checked_sub(minimum.packet_stream_window_bytes)
        .and_then(|bytes| bytes.checked_sub(minimum.hf_stream_window_bytes))
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "minimum-window VarDCT non-stream bytes",
        })?;
    let available_stream_bytes = memory_limit_bytes.saturating_sub(non_stream_bytes);
    let suggested_limit = available_stream_bytes
        .checked_div(active_stream_windows)
        .unwrap_or(MIN_STREAM_WINDOW_BYTES)
        .min(configured_limit)
        .max(MIN_STREAM_WINDOW_BYTES)
        & !3;

    let mut best_limit = MIN_STREAM_WINDOW_BYTES;
    let mut failing_limit = configured_limit;
    if suggested_limit > best_limit && suggested_limit < failing_limit {
        let suggested = memory_at_limit(suggested_limit)?;
        if suggested.total_frame_bytes <= memory_limit_bytes {
            best_limit = suggested_limit;
        } else {
            failing_limit = suggested_limit;
        }
    }
    for _ in 0..32 {
        let remaining_steps = failing_limit.saturating_sub(best_limit) / 4;
        if remaining_steps <= 1 {
            break;
        }
        let midpoint = best_limit.checked_add((remaining_steps / 2) * 4).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "adaptive VarDCT stream-window midpoint",
            },
        )?;
        let candidate = memory_at_limit(midpoint)?;
        if candidate.total_frame_bytes <= memory_limit_bytes {
            best_limit = midpoint;
        } else {
            failing_limit = midpoint;
        }
    }
    Ok(AdaptiveStreamLimitDecision::Selected(best_limit))
}

#[derive(Clone, Debug)]
pub(super) struct LfPacketWindowExecutionPlan {
    pub(super) stream_segments: Arc<[GroupStreamSegment]>,
    pub(super) stream_batches: Arc<[StreamBatch]>,
    pub(super) segment_params: Arc<[VarDctModularParams]>,
    pub(super) stream_bytes: u64,
}

#[derive(Clone, Debug)]
pub(super) struct CombinedPacketWindowExecutionPlan {
    pub(super) stream_segments: Arc<[GroupStreamSegment]>,
    pub(super) stream_batches: Arc<[StreamBatch]>,
    pub(super) segment_params: Arc<[VarDctModularParams]>,
    pub(super) stream_bytes: u64,
}

#[derive(Clone, Debug)]
pub(super) struct HfPacketWindowExecutionPlan {
    pub(super) stream_segments: Arc<[GroupStreamSegment]>,
    pub(super) stream_batches: Arc<[StreamBatch]>,
    pub(super) segment_params: Arc<[VarDctModularParams]>,
    pub(super) stream_bytes: u64,
}

fn map_packet_window_plan_error(error: DecodeError) -> VarDctDecodeError {
    match error {
        DecodeError::StreamWindowTooSmall {
            limit_bytes,
            minimum_bytes,
        } => VarDctDecodeError::EntropyStreamWindowTooSmall {
            limit_bytes,
            minimum_bytes,
        },
        source => VarDctDecodeError::EntropyWindowPlan {
            source: Box::new(source),
        },
    }
}

pub(super) fn map_codestream_source_error(source: DecodeError) -> VarDctDecodeError {
    VarDctDecodeError::CodestreamSource {
        source: Box::new(source),
    }
}

pub(super) fn copy_stream_segment(
    codestream: &GpuCodestream,
    segment: GroupStreamSegment,
    upload: &mut [u8],
    detail: &'static str,
) -> Result<(), VarDctDecodeError> {
    let input_len = segment
        .input_end
        .checked_sub(segment.input_start)
        .ok_or(VarDctDecodeError::EntropyWindowContract { detail })?;
    let output_end = segment.upload_offset.checked_add(input_len).ok_or(
        VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream upload end",
        },
    )?;
    let output = upload
        .get_mut(segment.upload_offset..output_end)
        .ok_or(VarDctDecodeError::EntropyWindowContract { detail })?;
    let input_start =
        u64::try_from(segment.input_start).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream input start",
        })?;
    let input_end =
        u64::try_from(segment.input_end).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "bounded stream input end",
        })?;
    if input_end > codestream.logical_bytes() {
        return Err(VarDctDecodeError::EntropyWindowContract { detail });
    }
    codestream
        .copy_range(input_start..input_end, output)
        .map_err(map_codestream_source_error)
}

impl LfPacketWindowExecutionPlan {
    pub(super) fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let ranges = packet
            .groups
            .iter()
            .map(|group| {
                let token_bit_end =
                    group
                        .lf_group
                        .end()
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "LF packet stream end",
                        })?;
                Ok(GroupEntropyRange {
                    token_bit_offset: u64::from(group.lf_entropy_bit_offset),
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, _) =
            build_stream_batches_for_len(codestream_bytes, &ranges, stream_limit, 1)
                .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment references an absent group",
                },
            )?;
            // Local-tree staging reserves the conservative predictor/LZ layout because the HF
            // descriptor is discovered only after this stage. The LF state itself records only
            // the predictor fields selected by its parsed tree.
            let state_offset = group.packet_execution_state_offset_words(true)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(group.lf_modular.lz77_window_words)
                    .with_self_correcting(group.lf_modular.needs_self_correcting)
                    .with_stream_segment(segment, group.lf_entropy_bit_offset, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes: stream_limit & !3,
        }))
    }

    pub(super) fn batch_count(&self) -> usize {
        self.stream_batches.len()
    }
}

impl CombinedPacketWindowExecutionPlan {
    pub(super) fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let mut stream_bases = Vec::with_capacity(packet.groups.len());
        let ranges = packet
            .groups
            .iter()
            .map(|group| {
                let control = group.packet_control(packet)?;
                let token_bit_offset = u64::from(control.section_bits[0]);
                let token_bit_end = u64::from(control.section_bits[1]);
                stream_bases.push(control.section_bits[0]);
                Ok(GroupEntropyRange {
                    token_bit_offset,
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, stream_bytes) = build_stream_batches_for_len(
            codestream_bytes,
            &ranges,
            stream_limit,
            packet.groups.len().max(1),
        )
        .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet segment references an absent group",
                },
            )?;
            let stream_base_bit = *stream_bases.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet segment has no stream base",
                },
            )?;
            let state_offset =
                group.packet_execution_state_offset_words(packet.needs_self_correcting)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(group.lz77_window_words)
                    .with_self_correcting(packet.needs_self_correcting)
                    .with_stream_segment(segment, stream_base_bit, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes,
        }))
    }

    pub(super) fn batch_count(&self) -> usize {
        self.stream_batches.len()
    }
}

impl HfPacketWindowExecutionPlan {
    pub(super) fn new(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        continuations: &[BoundedHfMetadataContinuation],
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        if packet.groups.len() != continuations.len() {
            return Err(VarDctDecodeError::GroupPlanCount {
                component: "HF packet continuation",
                expected: packet.groups.len(),
                actual: continuations.len(),
            });
        }
        let ranges = packet
            .groups
            .iter()
            .zip(continuations)
            .map(|(group, continuation)| {
                let token_bit_end =
                    group
                        .lf_group
                        .end()
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "HF packet stream end",
                        })?;
                Ok(GroupEntropyRange {
                    token_bit_offset: u64::from(continuation.token_bit_offset),
                    token_bit_end,
                })
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;
        let (segments, batches, stream_bytes) = build_stream_batches_for_len(
            codestream_bytes,
            &ranges,
            stream_limit,
            packet.groups.len().max(1),
        )
        .map_err(map_packet_window_plan_error)?;
        let uses_windows = segments.iter().any(|segment| {
            segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
        });
        if !uses_windows {
            return Ok(None);
        }
        let mut segment_params = Vec::with_capacity(segments.len());
        for &segment in &segments {
            let group = packet.groups.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "HF packet segment references an absent group",
                },
            )?;
            let continuation = continuations.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "HF packet segment references an absent continuation",
                },
            )?;
            let state_offset = group.packet_execution_state_offset_words(true)?;
            segment_params.push(
                VarDctModularParams::default()
                    .with_lz77_window(continuation.modular.lz77_window_words)
                    .with_self_correcting(continuation.modular.needs_self_correcting)
                    .with_stream_segment(segment, continuation.token_bit_offset, state_offset),
            );
        }
        Ok(Some(Self {
            stream_segments: segments.into(),
            stream_batches: batches.into(),
            segment_params: segment_params.into(),
            stream_bytes,
        }))
    }
}
