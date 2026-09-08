use std::sync::Arc;

use crate::entropy_window::{
    EntropyStreamPlan, EntropyStreamWindows, GroupEntropyRange, GroupStreamSegment,
    MIN_STREAM_WINDOW_BYTES,
};
use crate::vardct_packet::{
    BoundedHfMetadataContinuation, BoundedVarDctGroupEntry, BoundedVarDctGroupPlan,
    BoundedVarDctPacketPlan, VarDctModularParams,
};
use crate::vardct_pass_group::HfCoefficientExecutionPlan;
use crate::{Error as DecodeError, GpuCodestream};

use super::types::{VarDctDecodeError, VarDctDecodeMemoryStats};

pub(super) struct VarDctEntropyPlanSelection {
    pub(super) stream_limit: u64,
    pub(super) packet_windows: Option<PacketWindowExecutionPlan>,
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

/// The entropy entry point is independent of whether a reusable upload is needed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PacketStage {
    Lf,
    Hf,
    Combined,
}

#[derive(Clone, Debug)]
pub(super) struct PacketWindowExecutionPlan {
    pub(super) stage: PacketStage,
    pub(super) streams: EntropyStreamPlan,
    groups: Arc<[PacketWindowGroup]>,
    pub(super) stream_bytes: u64,
}

#[derive(Clone, Debug)]
struct PacketWindowGroup {
    params: VarDctModularParams,
    stream_base_bit: u32,
    state_offset: u32,
}

struct PacketStreamDescriptor {
    range: GroupEntropyRange,
    params: VarDctModularParams,
}

impl PacketStreamDescriptor {
    fn group(
        group: &BoundedVarDctGroupPlan,
        token_bit_offset: u32,
        params: VarDctModularParams,
    ) -> Result<Self, VarDctDecodeError> {
        Ok(Self {
            range: GroupEntropyRange {
                token_bit_offset: u64::from(token_bit_offset),
                token_bit_end: group.lf_group.end().ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "packet stream end",
                    },
                )?,
            },
            params,
        })
    }
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

/// Reserve a bounded upload for an HF descriptor hidden behind extra-channel entropy. The
/// enclosing packet is a conservative range: the eventual HF stream is always a suffix of it.
pub(super) fn deferred_hf_stream_window_bytes(
    codestream_bytes: u64,
    packet: &BoundedVarDctPacketPlan,
    stream_limit: u64,
) -> Result<u64, VarDctDecodeError> {
    if !packet.requires_lf_extra_staging() {
        return Ok(0);
    }
    let mut bytes = 0;
    for group in &packet.groups {
        let range = GroupEntropyRange {
            token_bit_offset: u64::from(group.entry.bit_offset()),
            token_bit_end: group
                .lf_group
                .end()
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF packet stream end",
                })?,
        };
        // Capacity planning needs only the range geometry, never a table of its future windows.
        let windows = EntropyStreamWindows::new(codestream_bytes, range, stream_limit)
            .map_err(map_packet_window_plan_error)?;
        if windows.len() > 1 {
            bytes = stream_limit & !3;
        }
    }
    Ok(bytes)
}

impl PacketWindowExecutionPlan {
    pub(super) fn initial(
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let stage = if packet.requires_lf_extra_staging() {
            return Ok(None);
        } else if packet.requires_lf_staging() {
            PacketStage::Lf
        } else if packet.profile.uses_lf_frame {
            PacketStage::Hf
        } else if packet.pending_raw_hf_dequant_side_image().is_some() {
            return Ok(None);
        } else {
            PacketStage::Combined
        };
        let descriptors = packet
            .groups
            .iter()
            .map(|group| match stage {
                PacketStage::Lf | PacketStage::Hf => {
                    let modular = match (&group.entry, stage) {
                        (
                            BoundedVarDctGroupEntry::LfCoefficients { modular, .. },
                            PacketStage::Lf,
                        ) => modular,
                        (BoundedVarDctGroupEntry::HfMetadata(continuation), PacketStage::Hf) => {
                            &continuation.modular
                        }
                        _ => {
                            return Err(VarDctDecodeError::EntropyWindowContract {
                                detail: "initial packet stage disagrees with its entropy entry",
                            });
                        }
                    };
                    PacketStreamDescriptor::group(
                        group,
                        group.entry.bit_offset(),
                        VarDctModularParams::default()
                            .with_lz77_window(modular.lz77_window_words)
                            .with_self_correcting(modular.needs_self_correcting),
                    )
                }
                PacketStage::Combined => {
                    let control = group.packet_control(packet)?;
                    Ok(PacketStreamDescriptor {
                        range: GroupEntropyRange {
                            token_bit_offset: u64::from(control.section_bits[0]),
                            token_bit_end: u64::from(control.section_bits[1]),
                        },
                        params: VarDctModularParams::default()
                            .with_lz77_window(group.lz77_window_words)
                            .with_self_correcting(packet.needs_self_correcting),
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(stage, codestream_bytes, packet, descriptors, stream_limit)
    }

    pub(super) fn hf(
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
        let descriptors = packet
            .groups
            .iter()
            .zip(continuations)
            .map(|(group, continuation)| {
                PacketStreamDescriptor::group(
                    group,
                    continuation.token_bit_offset,
                    VarDctModularParams::default()
                        .with_lz77_window(continuation.modular.lz77_window_words)
                        .with_self_correcting(continuation.modular.needs_self_correcting),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            PacketStage::Hf,
            codestream_bytes,
            packet,
            descriptors,
            stream_limit,
        )
    }

    fn new(
        stage: PacketStage,
        codestream_bytes: u64,
        packet: &BoundedVarDctPacketPlan,
        descriptors: Vec<PacketStreamDescriptor>,
        stream_limit: u64,
    ) -> Result<Option<Self>, VarDctDecodeError> {
        let ranges = descriptors
            .iter()
            .map(|descriptor| descriptor.range)
            .collect::<Vec<_>>();
        let streams = EntropyStreamPlan::new(
            codestream_bytes,
            &ranges,
            stream_limit,
            if stage == PacketStage::Lf {
                1
            } else {
                packet.groups.len().max(1)
            },
        )
        .map_err(map_packet_window_plan_error)?;
        if !streams.uses_windows() {
            return Ok(None);
        }
        if descriptors.len() != packet.groups.len() {
            return Err(VarDctDecodeError::GroupPlanCount {
                component: "packet stream descriptor",
                expected: packet.groups.len(),
                actual: descriptors.len(),
            });
        }
        let mut groups = Vec::with_capacity(descriptors.len());
        for (group, descriptor) in packet.groups.iter().zip(descriptors) {
            // Undiscovered HF descriptors need conservative predictor capacity. Eager HF-only
            // entries use the exact same compact layout as their admitted GPU allocation.
            let state_offset = group.packet_execution_state_offset_words(
                packet.needs_self_correcting || packet.requires_hf_metadata_staging(),
            )?;
            let stream_base_bit =
                u32::try_from(descriptor.range.token_bit_offset).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "packet stream base bit",
                    }
                })?;
            groups.push(PacketWindowGroup {
                params: descriptor.params,
                stream_base_bit,
                state_offset,
            });
        }
        let stream_bytes = streams.stream_bytes();
        Ok(Some(Self {
            stage,
            streams,
            groups: groups.into(),
            // A clipped LF window can be smaller than a later HF suffix or packed HF batch.
            // Those descriptors are still unknown, so reserve the selected shared capacity.
            stream_bytes: if stage == PacketStage::Lf {
                stream_limit & !3
            } else {
                stream_bytes
            },
        }))
    }

    pub(super) fn batch_count(&self) -> usize {
        self.streams.batch_count()
    }

    pub(super) fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub(super) fn params_for_segment(
        &self,
        segment: GroupStreamSegment,
    ) -> Result<VarDctModularParams, VarDctDecodeError> {
        let group = self.groups.get(segment.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "packet segment has no parameter record",
            },
        )?;
        Ok(group
            .params
            .with_stream_segment(segment, group.stream_base_bit, group.state_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vardct_frontend::VarDctFrameRole;
    use jxl_gpu_bitstream::{CodestreamInventory, FrameType, parse};

    fn decode_hex(hex: &str) -> Vec<u8> {
        let hex = hex.split_whitespace().collect::<String>();
        hex.as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn packet_planning_keeps_one_parameter_record_for_a_huge_stream() {
        let bytes = decode_hex(include_str!(
            "../../test-data/jpeg_transcode_raw_matrix_local_packets.jxl.hex"
        ));
        let parsed = parse(&bytes, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let mut packet = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
        packet.groups.truncate(1);
        let range = GroupEntropyRange {
            token_bit_offset: 7,
            token_bit_end: 7 + u64::from(u32::MAX),
        };
        let plan = PacketWindowExecutionPlan::new(
            PacketStage::Lf,
            range.token_bit_end.div_ceil(8),
            &packet,
            vec![PacketStreamDescriptor {
                range,
                params: VarDctModularParams::default(),
            }],
            40,
        )
        .unwrap()
        .unwrap();
        assert!(plan.batch_count() > 134_000_000);
        assert_eq!(plan.group_count(), 1);
        assert_eq!(plan.stream_bytes, 40);
        for index in [0, plan.batch_count() / 2, plan.batch_count() - 1] {
            let batch = plan.streams.batch(index).unwrap();
            let segment = batch.segments()[0];
            let params = plan.params_for_segment(segment).unwrap();
            let phase_flags = GroupStreamSegment::FIRST | GroupStreamSegment::FINAL;
            assert_eq!(params.window_contract()[4] & phase_flags, segment.flags);
            assert_ne!(params.window_contract()[4] & !phase_flags, 0);
            assert_eq!(
                params.window_contract()[5],
                packet.groups[0]
                    .packet_execution_state_offset_words(true)
                    .unwrap()
            );
        }
    }

    #[test]
    fn staged_lf_reserves_capacity_for_larger_hf_suffix_windows() {
        let bytes = decode_hex(include_str!(
            "../../test-data/jpeg_transcode_raw_matrix_local_packets.jxl.hex"
        ));
        let parsed = parse(&bytes, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let mut packet = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
        assert!(packet.requires_lf_staging());
        packet.groups = vec![packet.groups[0].clone(); 2];

        // Geometry-only counterexample using two copies of a real predictor layout. Each LF
        // range clips to two windows with a 240-byte peak. One later HF suffix fits whole but
        // needs 256 bytes; the other still needs splitting, so both use the shared HF upload.
        let descriptors = |ranges: [(u64, u64); 2]| {
            ranges
                .into_iter()
                .map(|(start, end)| PacketStreamDescriptor {
                    range: GroupEntropyRange {
                        token_bit_offset: start * 8,
                        token_bit_end: end * 8,
                    },
                    params: VarDctModularParams::default(),
                })
                .collect()
        };
        let lf = PacketWindowExecutionPlan::new(
            PacketStage::Lf,
            520,
            &packet,
            descriptors([(0, 260), (260, 520)]),
            256,
        )
        .unwrap()
        .unwrap();
        let lf_peak = lf
            .streams
            .batches()
            .flat_map(|batch| batch.segments().to_vec())
            .map(|segment| (segment.input_end - segment.input_start).div_ceil(4) * 4 + 4)
            .max()
            .unwrap();
        assert_eq!(lf_peak, 240);
        let hf = PacketWindowExecutionPlan::new(
            PacketStage::Hf,
            520,
            &packet,
            descriptors([(10, 260), (261, 520)]),
            256,
        )
        .unwrap()
        .unwrap();
        assert_eq!(hf.stream_bytes, 256);
        assert!(hf.stream_bytes > lf_peak as u64);
        assert!(hf.stream_bytes <= lf.stream_bytes);
    }

    #[test]
    fn lf_consumers_bound_initial_hf_packets_with_their_admitted_predictor_layout() {
        let bytes = decode_hex(include_str!(
            "../../test-data/testsrc_vardct_progressive_dc_ac.jxl.hex"
        ));
        let parsed = parse(&bytes, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let source = GpuCodestream::from_shared(
            parsed.codestream().into(),
            0..parsed.codestream().len(),
            false,
        )
        .unwrap();
        let mut bounded = 0;
        let mut generic_bounded = 0;
        for frame in inventory
            .frames
            .iter()
            .filter(|frame| frame.uses_lf_frame())
        {
            let projected = CodestreamInventory {
                frames: vec![frame.clone()],
                ..inventory.clone()
            };
            let packet = BoundedVarDctPacketPlan::parse_frame_source(
                &source,
                &projected,
                if frame.frame_type == FrameType::LowFrequency {
                    VarDctFrameRole::LowFrequency
                } else {
                    VarDctFrameRole::Frame
                },
            )
            .unwrap();
            assert!(
                packet
                    .groups
                    .iter()
                    .all(|group| matches!(group.entry, BoundedVarDctGroupEntry::HfMetadata(_)))
            );
            assert!(!packet.requires_hf_metadata_staging());
            assert!(
                PacketWindowExecutionPlan::initial(source.logical_bytes(), &packet, u64::MAX)
                    .unwrap()
                    .is_none()
            );
            for limit in [40, 64, 128] {
                let Some(plan) =
                    PacketWindowExecutionPlan::initial(source.logical_bytes(), &packet, limit)
                        .unwrap()
                else {
                    continue;
                };
                assert_eq!(plan.stage, PacketStage::Hf);
                assert!(plan.stream_bytes <= limit);
                assert!(plan.batch_count() > 1);
                for segment in plan
                    .streams
                    .batches()
                    .flat_map(|batch| batch.segments().to_vec())
                {
                    let params = plan.params_for_segment(segment).unwrap();
                    let group = &packet.groups[segment.group_index];
                    assert!(
                        segment.upload_offset + segment.input_end - segment.input_start + 4
                            <= plan.stream_bytes as usize
                    );
                    // The generic predictor allocation is intentionally smaller than the weighted
                    // one. A hardcoded conservative resume offset would write outside this buffer.
                    let words = group
                        .reconstructed_words(packet.needs_self_correcting)
                        .unwrap();
                    let state_words = crate::vardct_packet::packet_execution_state_bytes(
                        packet.needs_self_correcting,
                    ) / 4;
                    assert_eq!(
                        u64::from(params.window_contract()[5]) + state_words,
                        u64::from(words)
                    );
                }
                bounded += 1;
                generic_bounded += usize::from(!packet.needs_self_correcting);
            }
        }
        assert!(
            bounded >= 1,
            "the checked LF consumers must actually require HF windows"
        );
        assert!(
            generic_bounded > 0,
            "a compact generic predictor allocation is required"
        );
    }
}
