//! GPU entropy decode for single-pass regular DCT8 pass groups.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::entropy::EntropyStreamParams;
use crate::entropy_window::{
    GroupEntropyRange, GroupStreamSegment, StreamBatch, build_stream_batches_for_len,
};
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

/// One pass-group entropy invocation. The first 96 bytes carry stream-window, progress-storage,
/// geometry, and entropy-table bounds; the trailing 48 bytes carry the block-context table ABI.
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
    _reserved: u32,
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
    pub(crate) stream_segments: Vec<GroupStreamSegment>,
    pub(crate) stream_batches: Vec<StreamBatch>,
    pub(crate) segment_params: Vec<HfCoefficientPassParams>,
    pub(crate) stream_bytes: u64,
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
    pub fn new(
        packet: &BoundedVarDctPacketPlan,
        entropy: &HfCoefficientEntropyPlan,
        artifacts: &[VarDctArtifactLayout],
        codestream_bytes: u64,
        stream_limit: u64,
    ) -> Result<Self, HfCoefficientPlanError> {
        let metadata_words = u32::try_from(entropy.metadata.len()).map_err(|_| {
            HfCoefficientPlanError::ArithmeticOverflow {
                field: "metadata words",
            }
        })?;
        let context_map_offset_words = metadata_words;
        let mut entropy_words = Vec::with_capacity(
            entropy
                .metadata
                .len()
                .checked_add(entropy.context_map.len())
                .and_then(|words| words.checked_add(entropy.block_context_map.len()))
                .and_then(|words| words.checked_add(entropy.qf_thresholds.len()))
                .and_then(|words| words.checked_add(entropy.lf_thresholds[0].len()))
                .and_then(|words| words.checked_add(entropy.lf_thresholds[1].len()))
                .and_then(|words| words.checked_add(entropy.lf_thresholds[2].len()))
                .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                    field: "entropy bundle words",
                })?,
        );
        entropy_words.extend_from_slice(&entropy.metadata);
        entropy_words.extend_from_slice(&entropy.context_map);
        let block_context = append_block_context_tables(
            &mut entropy_words,
            &entropy.block_context_map,
            &entropy.qf_thresholds,
            &entropy.lf_thresholds,
        )?;

        let group_count = u32::try_from(entropy.pass_groups.len()).map_err(|_| {
            HfCoefficientPlanError::ArithmeticOverflow {
                field: "pass-group count",
            }
        })?;
        if u64::from(group_count) != packet.profile.group_count {
            return Err(HfCoefficientPlanError::PassGroupCount {
                expected: packet.profile.group_count,
                actual: u64::from(group_count),
            });
        }
        let lz77_window_mask = entropy.lz77_window_words.saturating_sub(1);
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
            for (global_group_index, range) in entropy.pass_groups.iter().copied().enumerate() {
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
                        lz77_window_mask,
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
                    block_width: rect.width.div_ceil(8),
                    block_height: rect.height.div_ceil(8),
                    blocks_per_row,
                    block_task_map_offset_words: artifact.block_task_map_offset_words,
                    num_hf_presets: entropy.num_hf_presets,
                    num_block_clusters,
                    context_map_offset_words,
                    lf_plane_stride_words,
                    lz77_window_base_words: local_group_index
                        .checked_mul(entropy.lz77_window_words)
                        .and_then(|offset| lz77_scratch_base_words.checked_add(offset))
                        .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group LZ77 scratch offset",
                        })?,
                    coeff_shift: 0,
                    global_group_index,
                    block_context,
                    _reserved: 0,
                });
                stream_ranges.push(GroupEntropyRange {
                    token_bit_offset: range.offset,
                    token_bit_end: range.end().ok_or(
                        HfCoefficientPlanError::ArithmeticOverflow {
                            field: "pass-group stream end",
                        },
                    )?,
                });
            }
            let local_group_count = u32::try_from(params.len()).map_err(|_| {
                HfCoefficientPlanError::ArithmeticOverflow {
                    field: "LF-group pass-group count",
                }
            })?;
            let lz77_scratch_words = entropy
                .lz77_window_words
                .checked_mul(local_group_count)
                .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                    field: "LF-group pass-group LZ77 scratch words",
                })?;
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
            let (stream_segments, stream_batches, stream_bytes) = build_stream_batches_for_len(
                codestream_bytes,
                &stream_ranges,
                stream_limit,
                params.len(),
            )
            .map_err(|error| HfCoefficientPlanError::EntropyWindow {
                message: error.to_string(),
            })?;
            let segment_params = stream_segments
                .iter()
                .map(|segment| {
                    let mut params = params[segment.group_index];
                    params.entropy.token_start = 0;
                    params.entropy.token_end = segment.available_token_end;
                    params.window_logical_start = segment.window_logical_start;
                    params.window_upload_start = segment.window_upload_start;
                    params.stream_token_end = segment.stream_token_end;
                    params.window_yield_end = segment.window_yield_end;
                    params.window_flags = segment.flags;
                    params
                })
                .collect();
            groups.push(HfCoefficientGroupExecutionPlan {
                lf_group_index: lf_group.index,
                params,
                sink_params: HfCoefficientSinkParams {
                    task_metadata_offset_words: artifact.task_metadata_offset_words,
                    task_count: lf_group.task_capacity,
                    coefficient_words: lf_group.coefficient_words(),
                    order_descriptor_count: (HF_ORDER_COUNT * HF_ORDER_CHANNELS) as u32,
                    order_coordinate_offset_words: entropy.order_coordinate_offset_words,
                    _reserved: [0; 3],
                },
                lz77_scratch_words,
                stream_segments,
                stream_batches,
                segment_params,
                stream_bytes,
            });
        }

        let order_words = entropy.order_words.clone();
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
        self.groups.iter().any(|group| {
            group.stream_segments.iter().any(|segment| {
                segment.flags != (GroupStreamSegment::FIRST | GroupStreamSegment::FINAL)
            })
        })
    }

    #[must_use]
    pub fn stream_window_bytes(&self) -> u64 {
        self.groups
            .iter()
            .map(|group| group.stream_bytes)
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn stream_batch_count(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.stream_batches.len())
            .sum()
    }

    #[must_use]
    pub fn reusable_params_bytes(&self) -> u64 {
        self.groups
            .iter()
            .flat_map(|group| &group.stream_batches)
            .map(|batch| {
                batch.group_count as u64 * std::mem::size_of::<HfCoefficientPassParams>() as u64
            })
            .max()
            .unwrap_or(0)
    }
}

impl HfCoefficientGroupExecutionPlan {
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
    #[error(transparent)]
    Frontend(#[from] crate::vardct_frontend::VarDctFrontendError),
    #[error(transparent)]
    Packet(#[from] crate::vardct_packet::BoundedVarDctPacketError),
    #[error("HF coefficient plan has {actual} pass groups; expected {expected}")]
    PassGroupCount { expected: u64, actual: u64 },
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
    pub group_index: u32,
    pub nonzero_coefficients: u32,
    pub sink_error: u32,
}

impl GpuHfCoefficientStatus {
    pub fn validate(self, expected_group: u32) -> Result<(), GpuHfCoefficientError> {
        let error = match self.error_code {
            1 if self.group_index == expected_group && self.bit_cursor == self.token_end => {
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
    assert!(std::mem::size_of::<HfCoefficientPassParams>() == 144);
    assert!(std::mem::align_of::<HfCoefficientPassParams>() == 16);
    assert!(std::mem::size_of::<HfBlockContextTables>() == 48);
    assert!(std::mem::align_of::<HfBlockContextTables>() == 4);
    assert!(std::mem::offset_of!(HfCoefficientPassParams, block_context) == 92);
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
