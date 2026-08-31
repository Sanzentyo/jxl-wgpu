//! GPU entropy decode for single-pass regular DCT8 pass groups.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::entropy::EntropyStreamParams;
use crate::vardct_artifact::{HfCoefficientSinkParams, VarDctArtifactLayout};
use crate::vardct_packet::{BoundedVarDctPacketPlan, HfCoefficientEntropyPlan};

const SHADER_TEMPLATE: &str = include_str!("vardct_pass_group.wgsl");
const ENTROPY_ABI: &str = include_str!("modular_entropy_abi.wgsl");
const ENTROPY: &str = include_str!("modular_entropy.wgsl");
const COEFFICIENT_SINK: &str = include_str!("vardct_hf_coefficient_sink.wgsl");
const ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
const ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const COEFFICIENT_SINK_MARKER: &str = "/*__JXL_HF_COEFFICIENT_SINK__*/";

pub const HF_COEFFICIENT_STATUS_BYTES: u64 = 32;

/// One pass-group entropy invocation. The 64-byte array stride is valid for storage buffers and
/// keeps each lane's bounds and scratch base in one naturally aligned cache line.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HfCoefficientPassParams {
    entropy: EntropyStreamParams,
    block_origin_x: u32,
    block_origin_y: u32,
    block_width: u32,
    block_height: u32,
    blocks_per_row: u32,
    block_task_map_offset_words: u32,
    num_hf_presets: u32,
    num_block_clusters: u32,
    context_map_offset_words: u32,
    block_context_map_offset_words: u32,
    lz77_window_base_words: u32,
    coeff_shift: u32,
    _reserved: u32,
}

/// Packed immutable input for all pass groups. Offsets are word-relative to `entropy_words`.
#[derive(Clone, Debug)]
pub struct HfCoefficientExecutionPlan {
    pub entropy_words: Vec<u32>,
    pub order_words: Vec<u32>,
    pub params: Vec<HfCoefficientPassParams>,
    pub sink_params: HfCoefficientSinkParams,
    pub lz77_scratch_words: u32,
}

impl HfCoefficientExecutionPlan {
    pub fn new(
        packet: &BoundedVarDctPacketPlan,
        entropy: &HfCoefficientEntropyPlan,
        artifact: VarDctArtifactLayout,
    ) -> Result<Self, HfCoefficientPlanError> {
        let metadata_words = u32::try_from(entropy.metadata.len()).map_err(|_| {
            HfCoefficientPlanError::ArithmeticOverflow {
                field: "metadata words",
            }
        })?;
        let context_map_offset_words = metadata_words;
        let context_words = u32::try_from(entropy.context_map.len()).map_err(|_| {
            HfCoefficientPlanError::ArithmeticOverflow {
                field: "context-map words",
            }
        })?;
        let block_context_map_offset_words = context_map_offset_words
            .checked_add(context_words)
            .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                field: "block-context map offset",
            })?;
        let mut entropy_words = Vec::with_capacity(
            entropy
                .metadata
                .len()
                .checked_add(entropy.context_map.len())
                .and_then(|words| words.checked_add(entropy.block_context_map.len()))
                .ok_or(HfCoefficientPlanError::ArithmeticOverflow {
                    field: "entropy bundle words",
                })?,
        );
        entropy_words.extend_from_slice(&entropy.metadata);
        entropy_words.extend_from_slice(&entropy.context_map);
        entropy_words.extend_from_slice(&entropy.block_context_map);

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
        let lz77_scratch_words = entropy.lz77_window_words.checked_mul(group_count).ok_or(
            HfCoefficientPlanError::ArithmeticOverflow {
                field: "pass-group LZ77 scratch words",
            },
        )?;
        let [blocks_per_row, _] = packet.block_extent();
        let lz77_window_mask = entropy.lz77_window_words.saturating_sub(1);
        let num_block_clusters = entropy.num_block_clusters;
        let mut params = Vec::with_capacity(entropy.pass_groups.len());
        for (group_index, range) in entropy.pass_groups.iter().copied().enumerate() {
            let group_index = u32::try_from(group_index).map_err(|_| {
                HfCoefficientPlanError::ArithmeticOverflow {
                    field: "pass-group index",
                }
            })?;
            let rect = packet.profile.pass_group_rect(u64::from(group_index))?;
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
            params.push(HfCoefficientPassParams {
                entropy: EntropyStreamParams {
                    token_start,
                    token_end,
                    lz77_window_mask,
                },
                block_origin_x: rect.x / 8,
                block_origin_y: rect.y / 8,
                block_width: rect.width.div_ceil(8),
                block_height: rect.height.div_ceil(8),
                blocks_per_row,
                block_task_map_offset_words: artifact.block_task_map_offset_words,
                num_hf_presets: entropy.num_hf_presets,
                num_block_clusters,
                context_map_offset_words,
                block_context_map_offset_words,
                lz77_window_base_words: group_index.checked_mul(entropy.lz77_window_words).ok_or(
                    HfCoefficientPlanError::ArithmeticOverflow {
                        field: "pass-group LZ77 scratch offset",
                    },
                )?,
                coeff_shift: 0,
                _reserved: 0,
            });
        }

        let order_words = entropy.order_words.clone();
        Ok(Self {
            entropy_words,
            order_words,
            params,
            sink_params: HfCoefficientSinkParams {
                task_metadata_offset_words: artifact.task_metadata_offset_words,
                task_count: packet.task_count,
                coefficient_words: packet.coefficient_words(),
                order_descriptor_count: 3,
                order_coordinate_offset_words: entropy.order_coordinate_offset_words,
                _reserved: [0; 3],
            },
            lz77_scratch_words,
        })
    }

    #[must_use]
    pub fn status_bytes(&self) -> u64 {
        self.params.len() as u64 * HF_COEFFICIENT_STATUS_BYTES
    }

    #[must_use]
    pub fn lz77_scratch_bytes(&self) -> u64 {
        u64::from(self.lz77_scratch_words.max(1)) * 4
    }
}

#[derive(Debug, Error)]
pub enum HfCoefficientPlanError {
    #[error(transparent)]
    Frontend(#[from] crate::vardct_frontend::VarDctFrontendError),
    #[error("HF coefficient plan has {actual} pass groups; expected {expected}")]
    PassGroupCount { expected: u64, actual: u64 },
    #[error("HF coefficient plan arithmetic overflowed while computing {field}")]
    ArithmeticOverflow { field: &'static str },
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
    pub lz77_scratch: &'a wgpu::Buffer,
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
                binding(2, buffers.lz77_scratch),
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
    assert!(std::mem::size_of::<HfCoefficientPassParams>() == 64);
    assert!(std::mem::align_of::<HfCoefficientPassParams>() == 16);
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
}
