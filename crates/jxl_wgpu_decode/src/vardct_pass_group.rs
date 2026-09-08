//! Bounded GPU entropy decode and accumulation for VarDCT coefficient passes.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{EntropyStreamPlan, GroupEntropyRange, GroupStreamSegment};
use crate::vardct_artifact::{
    HF_ORDER_CHANNELS, HF_ORDER_COUNT, HfCoefficientSinkParams, VarDctArtifactLayout,
};
use crate::vardct_packet::{BoundedVarDctPacketPlan, HfCoefficientEntropyPlan};

const SHADER_TEMPLATE: &str = include_str!("vardct_pass_group.wgsl");
const ENTROPY_ABI: &str = include_str!("modular_entropy_abi.wgsl");
const ENTROPY: &str = include_str!("modular_entropy.wgsl");
const BLOCK_CONTEXT: &str = include_str!("vardct_block_context.wgsl");
const COEFFICIENT_SINK: &str = include_str!("vardct_hf_coefficient_sink.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const BLOCK_CONTEXT_MARKER: &str = "/*__JXL_VARDCT_BLOCK_CONTEXT__*/";
const COEFFICIENT_SINK_MARKER: &str = "/*__JXL_HF_COEFFICIENT_SINK__*/";

pub const HF_COEFFICIENT_STATUS_BYTES: u64 = 32;
pub const HF_COEFFICIENT_EXECUTION_STATE_WORDS: u32 = 116;
pub const HF_COEFFICIENT_EXECUTION_STATE_BYTES: u64 =
    HF_COEFFICIENT_EXECUTION_STATE_WORDS as u64 * 4;

/// Whether the coefficient stream ends its packet or precedes another entropy consumer.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HfCoefficientStreamEnd {
    /// Validate at most seven zero padding bits and consume the complete packet.
    #[default]
    Packet = 0,
    /// Validate the entropy coder's terminal state and return the unaligned next bit cursor.
    Continuation = 1,
}

/// Exact 48-byte storage ABI locating the variable-length HF block-context tables.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HfBlockContextTables {
    pub block_context_map_offset_words: u32,
    pub qf_threshold_offset_words: u32,
    pub qf_threshold_count: u32,
    pub lf0_threshold_offset_words: u32,
    pub lf0_threshold_count: u32,
    pub lf1_threshold_offset_words: u32,
    pub lf1_threshold_count: u32,
    pub lf2_threshold_offset_words: u32,
    pub lf2_threshold_count: u32,
    pub _reserved: [u32; 3],
}

/// One 160-byte pass-group entropy invocation. A 92-byte stream/geometry prefix is followed by
/// 48-byte block-context locations, component shifts, per-pass table bases, and a spatial group ID.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HfCoefficientPassParams {
    entropy: EntropyStreamParams,
    window_logical_start: u32,
    window_upload_start: u32,
    stream_token_end: u32,
    window_yield_end: u32,
    window_flags: u32,
    execution_state_base_words: u32,
    status_index: u32,
    block_origin_x: u32,
    block_origin_y: u32,
    block_width: u32,
    block_height: u32,
    blocks_per_row: u32,
    block_task_map_offset_words: u32,
    num_hf_presets: u32,
    num_block_clusters: u32,
    context_map_offset_words: u32,
    lf_plane_stride_words: u32,
    lz77_window_base_words: u32,
    coeff_shift: u32,
    global_group_index: u32,
    block_context: HfBlockContextTables,
    channel_shifts: u32,
    metadata_base_words: u32,
    order_base_words: u32,
    spatial_group_index: u32,
    stream_end: u32,
}

/// Exact 464-byte resume record for one serial HF coefficient consumer.
///
/// The common prefix preserves bit/ANS/LZ state. Consumer words retain the nested block/channel/
/// coefficient loop and the coefficient sink error. The 96-word tail is the three-channel
/// nonzero-neighbour grid required by the JPEG XL coefficient contexts.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HfCoefficientExecutionState {
    common: [u32; 8],
    consumer: [u32; 10],
    nonzero_grid: [u32; 96],
    _reserved: [u32; 2],
}

/// One LF group's pass-group parameters and coefficient sink.
#[derive(Clone, Debug)]
pub struct HfCoefficientGroupExecutionPlan {
    pub lf_group_index: u32,
    pub params: Vec<HfCoefficientPassParams>,
    pub sink_params: HfCoefficientSinkParams,
    pub lz77_scratch_words: u32,
    pub(crate) streams: EntropyStreamPlan,
}

/// Shared immutable entropy/order tables plus independently bounded LF-group jobs.
#[derive(Clone, Debug)]
pub struct HfCoefficientExecutionPlan {
    pub entropy_words: Vec<u32>,
    pub order_words: Vec<u32>,
    pub groups: Vec<HfCoefficientGroupExecutionPlan>,
}

fn append_block_context_tables(
    words: &mut Vec<u32>,
    block_context_map: &[u32],
    qf_thresholds: &[u32],
    lf_thresholds: &[Vec<i32>; 3],
) -> Result<HfBlockContextTables, HfCoefficientPlanError> {
    let offset = |words: &[u32], field: &'static str| {
        u32::try_from(words.len()).map_err(|_| HfCoefficientPlanError::ArithmeticOverflow { field })
    };
    let count = |values: usize, field: &'static str| {
        u32::try_from(values).map_err(|_| HfCoefficientPlanError::ArithmeticOverflow { field })
    };

    let block_context_map_offset_words = offset(words, "block-context map offset")?;
    words.extend_from_slice(block_context_map);
    let qf_threshold_offset_words = offset(words, "QF threshold offset")?;
    words.extend_from_slice(qf_thresholds);
    let lf0_threshold_offset_words = offset(words, "LF0 threshold offset")?;
    words.extend(lf_thresholds[0].iter().map(|&threshold| threshold as u32));
    let lf1_threshold_offset_words = offset(words, "LF1 threshold offset")?;
    words.extend(lf_thresholds[1].iter().map(|&threshold| threshold as u32));
    let lf2_threshold_offset_words = offset(words, "LF2 threshold offset")?;
    words.extend(lf_thresholds[2].iter().map(|&threshold| threshold as u32));

    Ok(HfBlockContextTables {
        block_context_map_offset_words,
        qf_threshold_offset_words,
        qf_threshold_count: count(qf_thresholds.len(), "QF threshold count")?,
        lf0_threshold_offset_words,
        lf0_threshold_count: count(lf_thresholds[0].len(), "LF0 threshold count")?,
        lf1_threshold_offset_words,
        lf1_threshold_count: count(lf_thresholds[1].len(), "LF1 threshold count")?,
        lf2_threshold_offset_words,
        lf2_threshold_count: count(lf_thresholds[2].len(), "LF2 threshold count")?,
        _reserved: [0; 3],
    })
}

impl HfCoefficientExecutionPlan {
    /// Changes one logical pass group's termination rule, including every bounded-window resume.
    /// The caller must consume and validate the suffix before publishing a decoded frame.
    pub fn set_stream_end(
        &mut self,
        pass_group: u32,
        end: HfCoefficientStreamEnd,
    ) -> Result<(), HfCoefficientPlanError> {
        let mut found = false;
        for group in &mut self.groups {
            for params in &mut group.params {
                if params.global_group_index == pass_group {
                    params.stream_end = end as u32;
                    found = true;
                }
            }
        }
        if !found {
            return Err(HfCoefficientPlanError::MissingPassGroup { group: pass_group });
        }
        Ok(())
    }

    pub fn new(
        packet: &BoundedVarDctPacketPlan,
        entropy: &HfCoefficientEntropyPlan,
        artifacts: &[VarDctArtifactLayout],
        codestream_bytes: u64,
        stream_limit: u64,
    ) -> Result<Self, HfCoefficientPlanError> {
        let offset = |words: &[u32], field| {
            u32::try_from(words.len())
                .map_err(|_| HfCoefficientPlanError::ArithmeticOverflow { field })
        };
        let mut entropy_words = Vec::new();
        let mut order_words = Vec::new();
        let block_context = append_block_context_tables(
            &mut entropy_words,
            &entropy.block_context_map,
            &entropy.qf_thresholds,
            &entropy.lf_thresholds,
        )?;

        if entropy.passes.len() != packet.profile.coefficient_shifts.len() {
            return Err(HfCoefficientPlanError::PassCount {
                expected: packet.profile.coefficient_shifts.len(),
                actual: entropy.passes.len(),
            });
        }
        let mut pass_tables = Vec::with_capacity(entropy.passes.len());
        for (pass, &shift) in entropy
            .passes
            .iter()
            .zip(&packet.profile.coefficient_shifts)
        {
            if pass.coefficient_shift != shift {
                return Err(
                    crate::vardct_frontend::VarDctFrontendError::InvalidPassSchedule.into(),
                );
            }
            if pass.pass_groups.len() as u64 != packet.profile.group_count {
                return Err(HfCoefficientPlanError::PassGroupCount {
                    expected: packet.profile.group_count,
                    actual: pass.pass_groups.len() as u64,
                });
            }
            let metadata_base = crate::modular_tree::PackedModularMetadata {
                words: pass.metadata.clone(),
            }
            .append_to(&mut entropy_words)
            .map_err(|error| {
                crate::vardct_packet::BoundedVarDctPacketError::ModularTree(error.to_string())
            })?;
            let context_map_offset = offset(&entropy_words, "pass context-map offset")?;
            entropy_words.extend_from_slice(&pass.context_map);
            let order_base = offset(&order_words, "pass order base")?;
            order_words.extend_from_slice(&pass.order_words);
            pass_tables.push((metadata_base, context_map_offset, order_base));
        }
        offset(&entropy_words, "entropy bundle words")?;
        offset(&order_words, "order bundle words")?;
        let num_block_clusters = entropy.num_block_clusters;
        if artifacts.len() != packet.groups.len() {
            return Err(HfCoefficientPlanError::LfGroupCount {
                expected: packet.groups.len(),
                actual: artifacts.len(),
            });
        }
        let mut groups = Vec::with_capacity(packet.groups.len());
        for (lf_group, &artifact) in packet.groups.iter().zip(artifacts) {
            let [blocks_per_row, block_rows] = lf_group.block_extent();
            let lf_plane_stride_words = blocks_per_row.checked_mul(block_rows).ok_or(
                HfCoefficientPlanError::ArithmeticOverflow {
                    field: "quantized LF plane stride",
                },
            )?;
            let lz77_scratch_base_words =
                lf_group.reconstructed_words(packet.needs_self_correcting)?;
            let mut params = Vec::new();
            let mut stream_ranges = Vec::new();
            let mut lz77_scratch_words = 0u32;
            for (
                pass_index,
                (pass, &(metadata_base_words, context_map_offset_words, order_base_words)),
            ) in entropy.passes.iter().zip(&pass_tables).enumerate()
            {
                for (global_group_index, range) in pass.pass_groups.iter().copied().enumerate() {
                    let global_group_index = u32::try_from(global_group_index).map_err(|_| {
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group index",
                        }
                    })?;
                    if packet
                        .profile
                        .low_frequency_group_index_for_pass_group(u64::from(global_group_index))?
                        != lf_group.index
                    {
                        continue;
                    }
                    let rect = packet
                        .profile
                        .pass_group_rect(u64::from(global_group_index))?;
                    let [block_width, block_height] =
                        packet.profile.padded_group_block_extent(rect)?;
                    let local_x = rect.x.checked_sub(lf_group.rect.x).ok_or(
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "local pass-group x origin",
                        },
                    )?;
                    let local_y = rect.y.checked_sub(lf_group.rect.y).ok_or(
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "local pass-group y origin",
                        },
                    )?;
                    let token_start = u32::try_from(range.offset).map_err(|_| {
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group bit start",
                        }
                    })?;
                    let token_end = range.end().and_then(|end| u32::try_from(end).ok()).ok_or(
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group bit end",
                        },
                    )?;
                    let local_group_index = u32::try_from(params.len()).map_err(|_| {
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "local pass-group index",
                        }
                    })?;
                    params.push(HfCoefficientPassParams {
                        entropy: EntropyStreamParams {
                            token_start,
                            token_end,
                            lz77_window_mask: pass.lz77_window_words.saturating_sub(1),
                        },
                        window_logical_start: 0,
                        window_upload_start: 0,
                        stream_token_end: token_end,
                        window_yield_end: token_end,
                        window_flags: 3,
                        execution_state_base_words: 0,
                        status_index: local_group_index,
                        block_origin_x: local_x / 8,
                        block_origin_y: local_y / 8,
                        block_width,
                        block_height,
                        blocks_per_row,
                        block_task_map_offset_words: artifact.block_task_map_offset_words,
                        num_hf_presets: entropy.num_hf_presets,
                        num_block_clusters,
                        context_map_offset_words,
                        lf_plane_stride_words,
                        lz77_window_base_words: lz77_scratch_base_words
                            .checked_add(lz77_scratch_words)
                            .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                                field: "pass-group LZ77 scratch offset",
                            })?,
                        coeff_shift: pass.coefficient_shift,
                        global_group_index: u32::try_from(
                            pass_index as u64 * packet.profile.group_count
                                + u64::from(global_group_index),
                        )
                        .map_err(|_| {
                            HfCoefficientPlanError::ArithmeticOverflow {
                                field: "logical pass-group index",
                            }
                        })?,
                        block_context,
                        metadata_base_words,
                        order_base_words,
                        spatial_group_index: (local_y / packet.profile.group_dimension)
                            * blocks_per_row.div_ceil(packet.profile.group_dimension / 8)
                            + local_x / packet.profile.group_dimension,
                        stream_end: packet
                            .extra_channels
                            .as_ref()
                            .map(|extras| {
                                extras.ac_stream_end(
                                    &packet.profile,
                                    pass_index,
                                    global_group_index,
                                )
                            })
                            .transpose()?
                            .unwrap_or_default() as u32,
                        channel_shifts: packet.profile.channel_shifts.into_iter().enumerate().fold(
                            0u32,
                            |packed, (channel, shift)| {
                                packed
                                    | shift.horizontal << (channel as u32 * 2)
                                    | shift.vertical << (channel as u32 * 2 + 1)
                            },
                        ),
                    });
                    lz77_scratch_words = lz77_scratch_words
                        .checked_add(pass.lz77_window_words)
                        .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group LZ77 scratch words",
                        })?;
                    stream_ranges.push(GroupEntropyRange {
                        token_bit_offset: range.offset,
                        token_bit_end: range.end().ok_or(
                            HfCoefficientPlanError::ArithmeticOverflow {
                                field: "pass-group stream end",
                            },
                        )?,
                    });
                }
            }
            let execution_state_base_words = lz77_scratch_base_words
                .checked_add(lz77_scratch_words)
                .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                    field: "HF execution-state base",
                })?;
            for (index, params) in params.iter_mut().enumerate() {
                params.execution_state_base_words = u32::try_from(index)
                    .ok()
                    .and_then(|index| index.checked_mul(HF_COEFFICIENT_EXECUTION_STATE_WORDS))
                    .and_then(|offset| execution_state_base_words.checked_add(offset))
                    .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                        field: "HF execution-state offset",
                    })?;
            }
            let streams = EntropyStreamPlan::new(
                codestream_bytes,
                &stream_ranges,
                stream_limit,
                params.len(),
            )
            .map_err(|error| HfCoefficientPlanError::EntropyWindow {
                message: error.to_string(),
            })?;
            groups.push(HfCoefficientGroupExecutionPlan {
                lf_group_index: lf_group.index,
                params,
                sink_params: HfCoefficientSinkParams {
                    task_metadata_offset_words: artifact.task_metadata_offset_words,
                    task_count: lf_group.task_capacity,
                    coefficient_words: lf_group.coefficient_words(),
                    order_descriptor_count: (HF_ORDER_COUNT * HF_ORDER_CHANNELS) as u32,
                    order_coordinate_offset_words: (HF_ORDER_COUNT * HF_ORDER_CHANNELS * 4) as u32,
                    _reserved: [0; 3],
                },
                lz77_scratch_words,
                streams,
            });
        }

        Ok(Self {
            entropy_words,
            order_words,
            groups,
        })
    }

    #[must_use]
    pub fn status_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(HfCoefficientGroupExecutionPlan::status_bytes)
            .sum()
    }

    #[must_use]
    pub fn lz77_scratch_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(HfCoefficientGroupExecutionPlan::lz77_scratch_bytes)
            .sum()
    }

    #[must_use]
    pub fn execution_state_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(HfCoefficientGroupExecutionPlan::execution_state_bytes)
            .sum()
    }

    #[must_use]
    pub fn uses_bounded_stream_windows(&self) -> bool {
        self.groups.iter().any(|group| group.streams.uses_windows())
    }

    #[must_use]
    pub fn stream_window_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(|group| group.streams.stream_bytes())
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn stream_batch_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.streams.batch_count())
            .sum()
    }

    #[must_use]
    pub fn reusable_params_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(|group| {
                group.streams.max_group_count() as u64
                    * std::mem::size_of::<HfCoefficientPassParams>() as u64
            })
            .max()
            .unwrap_or(0)
    }
}

impl HfCoefficientGroupExecutionPlan {
    pub(crate) fn params_for_segment(
        &self,
        segment: GroupStreamSegment,
    ) -> Option<HfCoefficientPassParams> {
        let mut params = *self.params.get(segment.group_index)?;
        params.entropy.token_start = 0;
        params.entropy.token_end = segment.available_token_end;
        params.window_logical_start = segment.window_logical_start;
        params.window_upload_start = segment.window_upload_start;
        params.stream_token_end = segment.stream_token_end;
        params.window_yield_end = segment.window_yield_end;
        params.window_flags = segment.flags;
        Some(params)
    }

    #[must_use]
    pub fn status_bytes(&self) -> u64 {
        self.params.len() as u64 * HF_COEFFICIENT_STATUS_BYTES
    }

    #[must_use]
    pub fn lz77_scratch_bytes(&self) -> u64 {
        u64::from(self.lz77_scratch_words) * 4
    }

    #[must_use]
    pub fn execution_state_bytes(&self) -> u64 {
        self.params.len() as u64 * HF_COEFFICIENT_EXECUTION_STATE_BYTES
    }

    pub(crate) fn global_group_indices(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.params.iter().map(|params| params.global_group_index)
    }
}

#[derive(Debug, Error)]
pub enum HfCoefficientPlanError {
    #[error("HF coefficient plan has no logical pass group {group}")]
    MissingPassGroup { group: u32 },
    #[error(transparent)]
    Frontend(#[from] crate::vardct_frontend::VarDctFrontendError),
    #[error(transparent)]
    Packet(#[from] crate::vardct_packet::BoundedVarDctPacketError),
    #[error("HF coefficient plan has {actual} pass groups; expected {expected}")]
    PassGroupCount { expected: u64, actual: u64 },
    #[error("HF coefficient plan has {actual} passes; expected {expected}")]
    PassCount { expected: usize, actual: usize },
    #[error("HF coefficient plan has {actual} LF groups; expected {expected}")]
    LfGroupCount { expected: usize, actual: usize },
    #[error("HF coefficient plan arithmetic overflowed while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("HF coefficient entropy window planning failed: {message}")]
    EntropyWindow { message: String },
}

/// One 32-byte status record written by a pass-group invocation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuHfCoefficientStatus {
    pub error_code: u32,
    pub bit_cursor: u32,
    pub token_end: u32,
    pub decoded_symbols: u32,
    pub selected_preset: u32,
    /// Logical pass-group index: `pass_index * spatial_group_count + spatial_group_index`.
    pub group_index: u32,
    pub nonzero_coefficients: u32,
    pub sink_error: u32,
}

impl GpuHfCoefficientStatus {
    pub fn validate(self, expected_group: u32) -> Result<(), GpuHfCoefficientError> {
        self.validate_end(expected_group, self.bit_cursor == self.token_end)
    }

    /// Validates a continuation against host-owned packet bounds and returns its next bit.
    pub fn validate_cursor(
        self,
        expected_group: u32,
        token_start: u32,
        token_end: u32,
    ) -> Result<u32, GpuHfCoefficientError> {
        self.validate_end(
            expected_group,
            token_start <= self.bit_cursor
                && self.bit_cursor <= token_end
                && self.token_end == token_end,
        )?;
        Ok(self.bit_cursor)
    }

    fn validate_end(
        self,
        expected_group: u32,
        cursor_valid: bool,
    ) -> Result<(), GpuHfCoefficientError> {
        let error = match self.error_code {
            1 if self.group_index == expected_group && cursor_valid => {
                return Ok(());
            }
            1 => GpuHfCoefficientError::StatusContract {
                expected_group,
                actual_group: self.group_index,
                bit_cursor: self.bit_cursor,
                token_end: self.token_end,
            },
            2 => GpuHfCoefficientError::TruncatedBits {
                group: expected_group,
            },
            3 => GpuHfCoefficientError::PrefixCode {
                group: expected_group,
            },
            5 => GpuHfCoefficientError::Lz77State {
                group: expected_group,
            },
            7 => GpuHfCoefficientError::TrailingBits {
                group: expected_group,
            },
            10 => GpuHfCoefficientError::AnsState {
                group: expected_group,
            },
            11 => GpuHfCoefficientError::EntropyCluster {
                group: expected_group,
            },
            20 => GpuHfCoefficientError::Preset {
                group: expected_group,
            },
            21 => GpuHfCoefficientError::GroupGeometry {
                group: expected_group,
            },
            22 => GpuHfCoefficientError::MissingTask {
                group: expected_group,
            },
            23 => GpuHfCoefficientError::TaskShape {
                group: expected_group,
            },
            24 => GpuHfCoefficientError::NonzeroCount {
                group: expected_group,
            },
            25 => GpuHfCoefficientError::ContextMap {
                group: expected_group,
            },
            code if code >= 32 => GpuHfCoefficientError::CoefficientSink {
                group: expected_group,
                sink_code: self.sink_error,
            },
            code => GpuHfCoefficientError::Unknown {
                group: expected_group,
                code,
            },
        };
        Err(error)
    }
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GpuHfCoefficientError {
    #[error("HF pass group {group} ran out of coefficient bits")]
    TruncatedBits { group: u32 },
    #[error("HF pass group {group} contains an invalid prefix code")]
    PrefixCode { group: u32 },
    #[error("HF pass group {group} entered an invalid LZ77 state")]
    Lz77State { group: u32 },
    #[error("HF pass group {group} contains non-padding trailing bits")]
    TrailingBits { group: u32 },
    #[error("HF pass group {group} ended with an invalid ANS state")]
    AnsState { group: u32 },
    #[error("HF pass group {group} selected an invalid entropy cluster")]
    EntropyCluster { group: u32 },
    #[error("HF pass group {group} selected an invalid HF preset")]
    Preset { group: u32 },
    #[error("HF pass group {group} has unsupported GPU geometry")]
    GroupGeometry { group: u32 },
    #[error("HF pass group {group} references a missing VarDCT task")]
    MissingTask { group: u32 },
    #[error("HF pass group {group} references a non-DCT8 task")]
    TaskShape { group: u32 },
    #[error("HF pass group {group} decoded an impossible nonzero count")]
    NonzeroCount { group: u32 },
    #[error("HF pass group {group} indexed outside its coefficient-context map")]
    ContextMap { group: u32 },
    #[error("HF pass group {group} coefficient sink failed with code {sink_code}")]
    CoefficientSink { group: u32, sink_code: u32 },
    #[error(
        "HF pass-group status contract failed: group {actual_group}/{expected_group}, bits {bit_cursor}/{token_end}"
    )]
    StatusContract {
        expected_group: u32,
        actual_group: u32,
        bit_cursor: u32,
        token_end: u32,
    },
    #[error("HF pass group {group} returned unknown status {code}")]
    Unknown { group: u32, code: u32 },
}

pub struct HfCoefficientPipeline {
    pipeline: wgpu::ComputePipeline,
}

pub struct HfCoefficientBuffers<'a> {
    pub codestream: &'a wgpu::Buffer,
    pub entropy_bundle: &'a wgpu::Buffer,
    /// Reconstruction workspace followed by disjoint per-group HF LZ77 scratch slices.
    pub reconstruction: &'a wgpu::Buffer,
    pub params: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub artifact: &'a wgpu::Buffer,
    pub order_table: &'a wgpu::Buffer,
    pub coefficients: &'a wgpu::Buffer,
    pub sink_params: &'a wgpu::Buffer,
}

impl HfCoefficientPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu HF coefficient entropy"),
            source: wgpu::ShaderSource::Wgsl(shader_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu HF coefficient entropy"),
            layout: None,
            module: &module,
            entry_point: Some("decode_hf_coefficients"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: HfCoefficientBuffers<'_>,
        group_count: u32,
    ) {
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu HF coefficient entropy inputs"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, buffers.codestream),
                binding(1, buffers.entropy_bundle),
                binding(2, buffers.reconstruction),
                binding(3, buffers.params),
                binding(4, buffers.status),
            ],
        });
        let group1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu HF coefficient sink"),
            layout: &self.pipeline.get_bind_group_layout(1),
            entries: &[
                binding(0, buffers.artifact),
                binding(1, buffers.order_table),
                binding(2, buffers.coefficients),
                binding(3, buffers.sink_params),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu HF coefficient entropy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group0, &[]);
        pass.set_bind_group(1, &group1, &[]);
        pass.dispatch_workgroups(group_count, 1, 1);
    }
}

fn shader_source() -> String {
    SHADER_TEMPLATE
        .replace(ENTROPY_ABI_MARKER, ENTROPY_ABI)
        .replace(BLOCK_CONTEXT_MARKER, BLOCK_CONTEXT)
        .replace(ENTROPY_MARKER, ENTROPY)
        .replace(COEFFICIENT_SINK_MARKER, COEFFICIENT_SINK)
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

const _: () = {
    assert!(std::mem::size_of::<HfCoefficientPassParams>() == 160);
    assert!(std::mem::align_of::<HfCoefficientPassParams>() == 16);
    assert!(std::mem::size_of::<HfBlockContextTables>() == 48);
    assert!(std::mem::align_of::<HfBlockContextTables>() == 4);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, block_context) == 92);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, metadata_base_words) == 144);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, order_base_words) == 148);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, spatial_group_index) == 152);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, stream_end) == 156);
    assert!(std::mem::size_of::<HfCoefficientExecutionState>() == 464);
    assert!(std::mem::align_of::<HfCoefficientExecutionState>() == 16);
    assert!(std::mem::offset_of!(HfCoefficientExecutionState, nonzero_grid) == 72);
    assert!(std::mem::size_of::<GpuHfCoefficientStatus>() == HF_COEFFICIENT_STATUS_BYTES as usize);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_group_shader_is_portable_wgsl() {
        let module = naga::front::wgsl::parse_str(&shader_source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn block_context_tables_are_word_exact_and_ordered() {
        let mut words = vec![99, 98];
        let tables = append_block_context_tables(
            &mut words,
            &[7, 8, 9],
            &[4, 10],
            &[vec![-2, 5], vec![0], vec![-10, 3]],
        )
        .unwrap();
        assert_eq!(tables.block_context_map_offset_words, 2);
        assert_eq!(tables.qf_threshold_offset_words, 5);
        assert_eq!(tables.lf0_threshold_offset_words, 7);
        assert_eq!(tables.lf1_threshold_offset_words, 9);
        assert_eq!(tables.lf2_threshold_offset_words, 10);
        assert_eq!(
            &words,
            &[
                99,
                98,
                7,
                8,
                9,
                4,
                10,
                (-2i32) as u32,
                5,
                0,
                (-10i32) as u32,
                3
            ]
        );
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "vardct_pass_group/tests.rs"]
mod continuation_tests;
