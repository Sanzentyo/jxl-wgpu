//! GPU-resident artifacts between JPEG XL entropy reconstruction and inverse VarDCT.
//!
//! The generic Modular decoder reconstructs the four HF-metadata channels into one `i32`
//! storage buffer.  This module lowers the variable-length `block_info` channel into the exact
//! 64-byte task ABI consumed by `jxl_wgpu`'s general VarDCT shaders.  Coefficients, tasks, and
//! indirect dispatch arguments remain GPU resident; callers supply all buffers so the decode
//! session can reserve them against its shared transient budget before encoding commands.
//!
//! The 64-byte bridge task still contains the historical single `correlation_index`.  The status
//! requirement bit identifies any varblock that spans more than one 64x64 HF correlation cell;
//! such an artifact must not be submitted to a backend that lacks per-frequency CfL grid lookup.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use thiserror::Error;

pub const VAR_DCT_STRATEGY_COUNT: usize = 27;
pub const HF_ORDER_COUNT: usize = 13;
pub const HF_ORDER_CHANNELS: usize = 3;
pub const GENERAL_TASK_BYTES: u64 = 64;
pub const INDIRECT_STAGE_COUNT: usize = 3;
pub const ARTIFACT_STATUS_WORDS: u32 = 16;
pub const BACKEND_REQUIREMENT_FREQUENCY_CFL_GRID: u32 = 1;

const WORD_BYTES: u64 = 4;
const BUCKET_WORDS: u64 = 4;
const TASK_WORDS: u64 = GENERAL_TASK_BYTES / WORD_BYTES;
const TASK_METADATA_WORDS: u64 = 12;
const INDIRECT_WORDS: u64 = 3;
const PORTABLE_BINDING_ALIGNMENT: u64 = 256;
const COEFFICIENTS_PER_BLOCK: u64 = 8 * 8 * 3;

/// Failure while planning or encoding a GPU-resident VarDCT artifact.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VarDctArtifactError {
    #[error("VarDCT {axis} block dimension must be non-zero")]
    ZeroDimension { axis: &'static str },
    #[error("HF metadata declares zero varblock entries")]
    ZeroBlockInfoEntries,
    #[error("pass-group block dimension must be non-zero")]
    ZeroPassGroupDimension,
    #[error("invalid VarDCT artifact geometry: {field}")]
    InvalidGeometry { field: &'static str },
    #[error("VarDCT artifact arithmetic overflow while computing {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error("{resource} requires {required} bytes, exceeding the device buffer limit {maximum}")]
    BufferLimit {
        resource: &'static str,
        required: u64,
        maximum: u64,
    },
    #[error(
        "{resource} requires a {required}-byte storage binding, exceeding the device limit {maximum}"
    )]
    StorageBindingLimit {
        resource: &'static str,
        required: u64,
        maximum: u64,
    },
    #[error("raw HF metadata range {end} exceeds its {available}-word GPU buffer")]
    RawMetadataRange { end: u64, available: u64 },
    #[error("storage binding alignment {alignment} is not a non-zero power of two")]
    InvalidBindingAlignment { alignment: u64 },
    #[error("GPU VarDCT bucket contains invalid {field}")]
    InvalidBucket { field: &'static str },
}

/// Device limits relevant to the buffers in [`VarDctArtifactLayout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctArtifactDeviceLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub storage_binding_alignment: u64,
}

impl VarDctArtifactDeviceLimits {
    #[must_use]
    pub fn from_wgpu(limits: &wgpu::Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            storage_binding_alignment: u64::from(limits.min_storage_buffer_offset_alignment)
                .max(PORTABLE_BINDING_ALIGNMENT),
        }
    }
}

/// Geometry and source-channel offsets for one decoded LF group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HfMetadataArtifactConfig {
    pub blocks_width: u32,
    pub blocks_height: u32,
    /// Number of entries in each row of the raw `block_info` channel.
    pub block_info_entries: u32,
    /// Word offset of the strategy row in the generic Modular reconstruction buffer.
    pub strategy_offset_words: u32,
    /// Word offset of the `hf_mul - 1` row in the reconstruction buffer.
    pub hf_mul_offset_words: u32,
    /// Number of words in the reconstruction buffer bound at group 0, binding 0.
    pub raw_metadata_words: u64,
    /// JPEG XL pass groups are 32 by 32 8x8 blocks.  This remains explicit for bounded tests and
    /// for possible future codestream levels with a different group dimension.
    pub pass_group_dim_blocks: u32,
    pub lf_stride: u32,
    pub correlation_width: u32,
    pub correlation_height: u32,
    pub destination_origin: [u32; 2],
    /// Offset of the immutable AFV basis in the general VarDCT resource-vector buffer.
    pub afv_basis_offset: u32,
    /// Resource-vector offsets for the dequant matrix selected by each wire strategy.
    pub matrix_offsets: [u32; VAR_DCT_STRATEGY_COUNT],
}

/// Byte/word offsets for the single GPU artifact buffer and its coefficient/workspace buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctArtifactLayout {
    pub status_offset_words: u32,
    pub buckets_offset_words: u32,
    pub tasks_offset_words: u32,
    pub indirect_offset_words: u32,
    pub task_metadata_offset_words: u32,
    pub block_task_map_offset_words: u32,
    pub artifact_bytes: u64,
    pub occupancy_bytes: u64,
    pub coefficient_bytes: u64,
    pub task_capacity: u32,
    pub block_count: u32,
    pub coefficient_capacity_words: u32,
    binding_alignment: u64,
}

impl VarDctArtifactLayout {
    pub fn plan(
        config: &HfMetadataArtifactConfig,
        limits: VarDctArtifactDeviceLimits,
    ) -> Result<Self, VarDctArtifactError> {
        if config.blocks_width == 0 {
            return Err(VarDctArtifactError::ZeroDimension { axis: "horizontal" });
        }
        if config.blocks_height == 0 {
            return Err(VarDctArtifactError::ZeroDimension { axis: "vertical" });
        }
        if config.block_info_entries == 0 {
            return Err(VarDctArtifactError::ZeroBlockInfoEntries);
        }
        if config.pass_group_dim_blocks == 0 {
            return Err(VarDctArtifactError::ZeroPassGroupDimension);
        }
        if config.lf_stride < config.blocks_width {
            return Err(VarDctArtifactError::InvalidGeometry {
                field: "LF stride is smaller than the block width",
            });
        }
        if config.correlation_width < config.blocks_width.div_ceil(8)
            || config.correlation_height < config.blocks_height.div_ceil(8)
        {
            return Err(VarDctArtifactError::InvalidGeometry {
                field: "correlation grid does not cover the LF group",
            });
        }
        for (origin, blocks, axis) in [
            (
                config.destination_origin[0],
                config.blocks_width,
                "horizontal destination extent",
            ),
            (
                config.destination_origin[1],
                config.blocks_height,
                "vertical destination extent",
            ),
        ] {
            origin
                .checked_add(
                    blocks
                        .checked_mul(8)
                        .ok_or(VarDctArtifactError::ArithmeticOverflow { field: axis })?,
                )
                .ok_or(VarDctArtifactError::ArithmeticOverflow { field: axis })?;
        }
        let alignment = limits
            .storage_binding_alignment
            .max(PORTABLE_BINDING_ALIGNMENT);
        if !alignment.is_power_of_two() {
            return Err(VarDctArtifactError::InvalidBindingAlignment { alignment });
        }

        let block_count_u64 = u64::from(config.blocks_width)
            .checked_mul(u64::from(config.blocks_height))
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "block count",
            })?;
        let block_count = u32::try_from(block_count_u64).map_err(|_| {
            VarDctArtifactError::ArithmeticOverflow {
                field: "GPU block count",
            }
        })?;
        let task_capacity = config.block_info_entries.min(block_count);
        let coefficient_capacity_words_u64 = block_count_u64
            .checked_mul(COEFFICIENTS_PER_BLOCK)
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "coefficient word count",
            })?;
        let coefficient_capacity_words =
            u32::try_from(coefficient_capacity_words_u64).map_err(|_| {
                VarDctArtifactError::ArithmeticOverflow {
                    field: "GPU coefficient word count",
                }
            })?;
        let coefficient_bytes = coefficient_capacity_words_u64
            .checked_mul(WORD_BYTES)
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "coefficient bytes",
            })?;

        let status_offset_bytes = 0;
        let buckets_offset_bytes = align_up(
            u64::from(ARTIFACT_STATUS_WORDS) * WORD_BYTES,
            alignment,
            "bucket offset",
        )?;
        let bucket_bytes = (VAR_DCT_STRATEGY_COUNT as u64)
            .checked_mul(BUCKET_WORDS * WORD_BYTES)
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "bucket bytes",
            })?;
        let tasks_offset_bytes = align_up(
            buckets_offset_bytes.checked_add(bucket_bytes).ok_or(
                VarDctArtifactError::ArithmeticOverflow {
                    field: "task offset",
                },
            )?,
            alignment,
            "task offset",
        )?;
        let task_bytes = u64::from(task_capacity)
            .checked_mul(TASK_WORDS * WORD_BYTES)
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "task bytes",
            })?;
        let indirect_offset_bytes = align_up(
            tasks_offset_bytes.checked_add(task_bytes).ok_or(
                VarDctArtifactError::ArithmeticOverflow {
                    field: "indirect offset",
                },
            )?,
            alignment,
            "indirect offset",
        )?;
        let indirect_bytes = (VAR_DCT_STRATEGY_COUNT as u64)
            .checked_mul(INDIRECT_STAGE_COUNT as u64)
            .and_then(|count| count.checked_mul(INDIRECT_WORDS * WORD_BYTES))
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "indirect bytes",
            })?;
        let task_metadata_offset_bytes = align_up(
            indirect_offset_bytes.checked_add(indirect_bytes).ok_or(
                VarDctArtifactError::ArithmeticOverflow {
                    field: "task metadata offset",
                },
            )?,
            alignment,
            "task metadata offset",
        )?;
        let task_metadata_bytes = u64::from(task_capacity)
            .checked_mul(TASK_METADATA_WORDS * WORD_BYTES)
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "task metadata bytes",
            })?;
        let block_task_map_offset_bytes = align_up(
            task_metadata_offset_bytes
                .checked_add(task_metadata_bytes)
                .ok_or(VarDctArtifactError::ArithmeticOverflow {
                    field: "block-task map offset",
                })?,
            alignment,
            "block-task map offset",
        )?;
        let block_task_map_bytes = block_count_u64.checked_mul(WORD_BYTES).ok_or(
            VarDctArtifactError::ArithmeticOverflow {
                field: "block-task map bytes",
            },
        )?;
        let artifact_bytes = align_up(
            block_task_map_offset_bytes
                .checked_add(block_task_map_bytes)
                .ok_or(VarDctArtifactError::ArithmeticOverflow {
                    field: "artifact bytes",
                })?,
            alignment,
            "artifact bytes",
        )?;
        let occupancy_words = block_count_u64.div_ceil(32);
        let occupancy_bytes = occupancy_words.checked_mul(WORD_BYTES).ok_or(
            VarDctArtifactError::ArithmeticOverflow {
                field: "occupancy bytes",
            },
        )?;

        check_buffer_limit("VarDCT artifact", artifact_bytes, limits.max_buffer_size)?;
        check_storage_limit(
            "VarDCT artifact",
            artifact_bytes,
            limits.max_storage_buffer_binding_size,
        )?;
        check_buffer_limit(
            "VarDCT occupancy workspace",
            occupancy_bytes,
            limits.max_buffer_size,
        )?;
        check_storage_limit(
            "VarDCT occupancy workspace",
            occupancy_bytes,
            limits.max_storage_buffer_binding_size,
        )?;
        check_buffer_limit(
            "VarDCT coefficients",
            coefficient_bytes,
            limits.max_buffer_size,
        )?;
        check_storage_limit(
            "VarDCT coefficients",
            coefficient_bytes,
            limits.max_storage_buffer_binding_size,
        )?;

        let raw_end = u64::from(config.strategy_offset_words)
            .checked_add(u64::from(config.block_info_entries))
            .and_then(|strategy_end| {
                u64::from(config.hf_mul_offset_words)
                    .checked_add(u64::from(config.block_info_entries))
                    .map(|hf_end| strategy_end.max(hf_end))
            })
            .ok_or(VarDctArtifactError::ArithmeticOverflow {
                field: "raw HF metadata range",
            })?;
        if raw_end > config.raw_metadata_words {
            return Err(VarDctArtifactError::RawMetadataRange {
                end: raw_end,
                available: config.raw_metadata_words,
            });
        }

        Ok(Self {
            status_offset_words: word_offset(status_offset_bytes, "status offset")?,
            buckets_offset_words: word_offset(buckets_offset_bytes, "bucket offset")?,
            tasks_offset_words: word_offset(tasks_offset_bytes, "task offset")?,
            indirect_offset_words: word_offset(indirect_offset_bytes, "indirect offset")?,
            task_metadata_offset_words: word_offset(
                task_metadata_offset_bytes,
                "task metadata offset",
            )?,
            block_task_map_offset_words: word_offset(
                block_task_map_offset_bytes,
                "block-task map offset",
            )?,
            artifact_bytes,
            occupancy_bytes,
            coefficient_bytes,
            task_capacity,
            block_count,
            coefficient_capacity_words,
            binding_alignment: alignment,
        })
    }

    #[must_use]
    pub fn persistent_bytes(self) -> u64 {
        self.artifact_bytes.saturating_add(self.coefficient_bytes)
    }

    #[must_use]
    pub fn transient_bytes(self) -> u64 {
        self.occupancy_bytes
            .saturating_add(std::mem::size_of::<HfMetadataLoweringParams>() as u64)
    }

    #[must_use]
    pub fn task_binding(self) -> (u64, u64) {
        (
            u64::from(self.tasks_offset_words) * WORD_BYTES,
            u64::from(self.task_capacity) * GENERAL_TASK_BYTES,
        )
    }

    #[must_use]
    pub fn indirect_offset(self, strategy: usize, stage: HfDispatchStage) -> Option<u64> {
        if strategy >= VAR_DCT_STRATEGY_COUNT {
            return None;
        }
        let index = strategy
            .checked_mul(INDIRECT_STAGE_COUNT)?
            .checked_add(stage as usize)?;
        Some(
            u64::from(self.indirect_offset_words) * WORD_BYTES
                + index as u64 * INDIRECT_WORDS * WORD_BYTES,
        )
    }

    #[must_use]
    pub fn binding_alignment(self) -> u64 {
        self.binding_alignment
    }

    /// Validate one GPU-produced bucket and resolve its typed inverse-transform dispatches.
    pub fn bucket_dispatch(
        self,
        bucket: GpuVarDctBucket,
    ) -> Result<VarDctBucketDispatch, VarDctArtifactError> {
        let strategy =
            usize::try_from(bucket.strategy).map_err(|_| VarDctArtifactError::InvalidBucket {
                field: "wire strategy",
            })?;
        let Some(&transform) = TransformKind::ALL.get(strategy) else {
            return Err(VarDctArtifactError::InvalidBucket {
                field: "wire strategy",
            });
        };
        let task_end = bucket.task_offset.checked_add(bucket.task_count).ok_or(
            VarDctArtifactError::InvalidBucket {
                field: "task range",
            },
        )?;
        if task_end > self.task_capacity {
            return Err(VarDctArtifactError::InvalidBucket {
                field: "task range",
            });
        }
        let dequantize_indirect_offset = self
            .indirect_offset(strategy, HfDispatchStage::Dequantize)
            .ok_or(VarDctArtifactError::InvalidBucket {
                field: "indirect offset",
            })?;
        let horizontal_indirect_offset = self
            .indirect_offset(strategy, HfDispatchStage::Horizontal)
            .ok_or(VarDctArtifactError::InvalidBucket {
                field: "indirect offset",
            })?;
        let vertical_indirect_offset = self
            .indirect_offset(strategy, HfDispatchStage::Vertical)
            .ok_or(VarDctArtifactError::InvalidBucket {
                field: "indirect offset",
            })?;
        if dequantize_indirect_offset / WORD_BYTES != u64::from(bucket.indirect_word_offset) {
            return Err(VarDctArtifactError::InvalidBucket {
                field: "indirect offset",
            });
        }
        Ok(VarDctBucketDispatch {
            transform,
            task_offset: bucket.task_offset,
            task_count: bucket.task_count,
            dequantize_indirect_offset,
            horizontal_indirect_offset,
            vertical_indirect_offset,
        })
    }
}

/// Typed, buffer-relative dispatch view for one transform bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctBucketDispatch {
    pub transform: TransformKind,
    pub task_offset: u32,
    pub task_count: u32,
    pub dequantize_indirect_offset: u64,
    pub horizontal_indirect_offset: u64,
    pub vertical_indirect_offset: u64,
}

/// General-VarDCT dispatch stage represented by one standard 12-byte indirect tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HfDispatchStage {
    Dequantize = 0,
    Horizontal = 1,
    Vertical = 2,
}

/// Status written by the metadata-lowering shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuVarDctArtifactStatus {
    pub error_code: u32,
    pub error_x: u32,
    pub error_y: u32,
    pub error_value: u32,
    pub task_count: u32,
    pub coefficient_words: u32,
    pub covered_blocks: u32,
    pub consumed_block_info_entries: u32,
    pub backend_requirements: u32,
    pub _reserved: [u32; 7],
}

impl GpuVarDctArtifactStatus {
    pub fn validate(self) -> Result<(), GpuVarDctLoweringError> {
        match self.error_code {
            0 => Ok(()),
            1 => Err(GpuVarDctLoweringError::InvalidStrategy {
                x: self.error_x,
                y: self.error_y,
                value: self.error_value,
            }),
            2 => Err(GpuVarDctLoweringError::NonPositiveHfMultiplier {
                x: self.error_x,
                y: self.error_y,
                value: self.error_value as i32,
            }),
            3 => Err(GpuVarDctLoweringError::BlockInfoExhausted {
                x: self.error_x,
                y: self.error_y,
                consumed: self.error_value,
            }),
            4 => Err(GpuVarDctLoweringError::TransformOutsideLfGroup {
                x: self.error_x,
                y: self.error_y,
                strategy: self.error_value,
            }),
            5 => Err(GpuVarDctLoweringError::PassGroupCrossing {
                x: self.error_x,
                y: self.error_y,
                strategy: self.error_value,
            }),
            6 => Err(GpuVarDctLoweringError::TransformOverlap {
                x: self.error_x,
                y: self.error_y,
                strategy: self.error_value,
            }),
            7 => Err(GpuVarDctLoweringError::TaskCapacity {
                required: self.error_value,
            }),
            8 => Err(GpuVarDctLoweringError::CoefficientCapacity {
                required_words: self.error_value,
            }),
            code => Err(GpuVarDctLoweringError::UnknownStatus { code }),
        }
    }
}

/// Typed validation failure reported by the metadata-lowering compute pass.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum GpuVarDctLoweringError {
    #[error("HF varblock at ({x}, {y}) selects invalid strategy {value}")]
    InvalidStrategy { x: u32, y: u32, value: u32 },
    #[error("HF varblock at ({x}, {y}) has non-positive multiplier source {value}")]
    NonPositiveHfMultiplier { x: u32, y: u32, value: i32 },
    #[error("HF block-info list is exhausted at ({x}, {y}) after {consumed} entries")]
    BlockInfoExhausted { x: u32, y: u32, consumed: u32 },
    #[error("strategy {strategy} at ({x}, {y}) does not fit in its LF group")]
    TransformOutsideLfGroup { x: u32, y: u32, strategy: u32 },
    #[error("strategy {strategy} at ({x}, {y}) crosses a pass-group border")]
    PassGroupCrossing { x: u32, y: u32, strategy: u32 },
    #[error("strategy {strategy} overlaps an occupied block at ({x}, {y})")]
    TransformOverlap { x: u32, y: u32, strategy: u32 },
    #[error("GPU task artifact needs {required} task records")]
    TaskCapacity { required: u32 },
    #[error("GPU coefficient artifact needs {required_words} words")]
    CoefficientCapacity { required_words: u32 },
    #[error("GPU VarDCT lowering returned unknown status {code}")]
    UnknownStatus { code: u32 },
}

/// One transform bucket in wire-strategy order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuVarDctBucket {
    pub strategy: u32,
    pub task_offset: u32,
    pub task_count: u32,
    pub indirect_word_offset: u32,
}

/// Exact 64-byte storage ABI consumed by `vardct_general.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuGeneralVarDctTask {
    pub coefficient_offset: u32,
    pub scratch_or_basis_offset: u32,
    pub matrix_offset: u32,
    pub quant_index: u32,
    pub correlation_index: u32,
    pub lf_offset: u32,
    pub channel_mask: u32,
    pub _pad0: u32,
    pub destination_x_x: u32,
    pub destination_y_x: u32,
    pub destination_x_y: u32,
    pub destination_y_y: u32,
    pub destination_x_b: u32,
    pub destination_y_b: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Entropy-side metadata parallel to the compact general-transform task array.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuHfTaskMetadata {
    pub strategy: u32,
    pub raster_index: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_width: u32,
    pub block_height: u32,
    pub hf_mul: u32,
    pub pass_group: u32,
    pub coefficient_offset: u32,
    pub coefficient_words: u32,
    pub order_id: u32,
    /// Bit 0 is coefficient transposition; bit 1 selects a special 8x8 inverse.
    pub flags: u32,
}

/// Fixed-size indirect dispatch tuple accepted by WebGPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuDispatchIndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Uniform consumed by `vardct_artifact.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct HfMetadataLoweringParams {
    pub dimensions: [u32; 4],
    pub capacities: [u32; 4],
    pub image: [u32; 4],
    pub artifact_offsets: [u32; 4],
    pub metadata_offsets: [u32; 4],
    pub source_offsets: [u32; 4],
    pub matrix_offsets: [[u32; 4]; 7],
}

impl HfMetadataLoweringParams {
    pub fn new(
        config: &HfMetadataArtifactConfig,
        layout: VarDctArtifactLayout,
    ) -> Result<Self, VarDctArtifactError> {
        let groups_per_row = config.blocks_width.div_ceil(config.pass_group_dim_blocks);
        let mut matrix_offsets = [[0u32; 4]; 7];
        for (strategy, &offset) in config.matrix_offsets.iter().enumerate() {
            matrix_offsets[strategy / 4][strategy % 4] = offset;
        }
        Ok(Self {
            dimensions: [
                config.blocks_width,
                config.blocks_height,
                config.block_info_entries,
                config.pass_group_dim_blocks,
            ],
            capacities: [
                layout.task_capacity,
                layout.coefficient_capacity_words,
                groups_per_row,
                config.correlation_width,
            ],
            image: [
                config.lf_stride,
                config.destination_origin[0],
                config.destination_origin[1],
                config.afv_basis_offset,
            ],
            artifact_offsets: [
                layout.status_offset_words,
                layout.buckets_offset_words,
                layout.tasks_offset_words,
                layout.indirect_offset_words,
            ],
            metadata_offsets: [
                layout.task_metadata_offset_words,
                layout.block_task_map_offset_words,
                0,
                0,
            ],
            source_offsets: [
                config.strategy_offset_words,
                config.hf_mul_offset_words,
                0,
                0,
            ],
            matrix_offsets,
        })
    }
}

/// Buffers supplied to [`HfMetadataLoweringPipeline::encode`].
pub struct HfMetadataLoweringBuffers<'a> {
    pub raw_metadata: &'a wgpu::Buffer,
    pub artifact: &'a wgpu::Buffer,
    pub occupancy: &'a wgpu::Buffer,
    pub params: &'a wgpu::Buffer,
}

/// Reusable metadata-to-task compute pipeline.
pub struct HfMetadataLoweringPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl HfMetadataLoweringPipeline {
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jxl-wgpu VarDCT artifact lowering"),
            source: wgpu::ShaderSource::Wgsl(VAR_DCT_ARTIFACT_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("jxl-wgpu VarDCT artifact lowering"),
            layout: None,
            module: &module,
            entry_point: Some("lower_hf_metadata"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline }
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: HfMetadataLoweringBuffers<'_>,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu VarDCT artifact lowering bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                binding(0, buffers.raw_metadata),
                binding(1, buffers.artifact),
                binding(2, buffers.occupancy),
                binding(3, buffers.params),
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu VarDCT artifact lowering"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// One channel/order range in a GPU-populated JPEG XL coefficient-order table.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuHfOrderDescriptor {
    pub offset: u32,
    pub len: u32,
    pub width: u32,
    pub height: u32,
}

/// Layout for entropy-decoded custom order permutations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HfOrderTableLayout {
    pub used_orders: u16,
    pub descriptors: [GpuHfOrderDescriptor; HF_ORDER_COUNT * HF_ORDER_CHANNELS],
    pub coordinate_words: u32,
}

impl HfOrderTableLayout {
    pub fn new(used_orders: u16) -> Result<Self, VarDctArtifactError> {
        let mut descriptors = [GpuHfOrderDescriptor::zeroed(); HF_ORDER_COUNT * HF_ORDER_CHANNELS];
        let mut cursor = 0u32;
        for order_id in 0..HF_ORDER_COUNT {
            let [width, height] = HF_ORDER_EXTENTS[order_id];
            let len = width
                .checked_mul(height)
                .ok_or(VarDctArtifactError::ArithmeticOverflow {
                    field: "HF order area",
                })?;
            for channel in 0..HF_ORDER_CHANNELS {
                let descriptor = &mut descriptors[order_id * HF_ORDER_CHANNELS + channel];
                *descriptor = GpuHfOrderDescriptor {
                    offset: cursor,
                    len,
                    width,
                    height,
                };
                cursor =
                    cursor
                        .checked_add(len)
                        .ok_or(VarDctArtifactError::ArithmeticOverflow {
                            field: "HF order coordinate words",
                        })?;
            }
        }
        Ok(Self {
            used_orders: used_orders & ((1 << HF_ORDER_COUNT) - 1),
            descriptors,
            coordinate_words: cursor,
        })
    }

    #[must_use]
    pub fn descriptor(&self, order_id: usize, channel: usize) -> Option<GpuHfOrderDescriptor> {
        if order_id >= HF_ORDER_COUNT || channel >= HF_ORDER_CHANNELS {
            return None;
        }
        Some(self.descriptors[order_id * HF_ORDER_CHANNELS + channel])
    }

    #[must_use]
    pub fn custom(&self, order_id: usize) -> bool {
        order_id < HF_ORDER_COUNT && (self.used_orders & (1 << order_id)) != 0
    }
}

/// Uniform used by the composable HF coefficient sink fragment.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct HfCoefficientSinkParams {
    pub task_metadata_offset_words: u32,
    pub task_count: u32,
    pub coefficient_words: u32,
    pub order_descriptor_count: u32,
}

pub const VAR_DCT_ARTIFACT_SHADER: &str = include_str!("vardct_artifact.wgsl");
pub const HF_COEFFICIENT_SINK_SHADER: &str = include_str!("vardct_hf_coefficient_sink.wgsl");

pub const HF_ORDER_EXTENTS: [[u32; 2]; HF_ORDER_COUNT] = [
    [8, 8],
    [8, 8],
    [16, 16],
    [32, 32],
    [16, 8],
    [32, 8],
    [32, 16],
    [64, 64],
    [64, 32],
    [128, 128],
    [128, 64],
    [256, 256],
    [256, 128],
];

const _: () = {
    assert!(std::mem::size_of::<GpuVarDctArtifactStatus>() == 64);
    assert!(std::mem::size_of::<GpuVarDctBucket>() == 16);
    assert!(std::mem::size_of::<GpuGeneralVarDctTask>() == 64);
    assert!(std::mem::size_of::<GpuHfTaskMetadata>() == 48);
    assert!(std::mem::size_of::<GpuDispatchIndirectArgs>() == 12);
    assert!(std::mem::size_of::<GpuHfOrderDescriptor>() == 16);
    assert!(std::mem::size_of::<HfCoefficientSinkParams>() == 16);
    assert!(std::mem::size_of::<HfMetadataLoweringParams>() == 208);
    assert!(std::mem::align_of::<HfMetadataLoweringParams>() == 16);
};

fn align_up(value: u64, alignment: u64, field: &'static str) -> Result<u64, VarDctArtifactError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(VarDctArtifactError::ArithmeticOverflow { field })
}

fn word_offset(value: u64, field: &'static str) -> Result<u32, VarDctArtifactError> {
    debug_assert_eq!(value % WORD_BYTES, 0);
    u32::try_from(value / WORD_BYTES).map_err(|_| VarDctArtifactError::ArithmeticOverflow { field })
}

fn check_buffer_limit(
    resource: &'static str,
    required: u64,
    maximum: u64,
) -> Result<(), VarDctArtifactError> {
    if required > maximum {
        return Err(VarDctArtifactError::BufferLimit {
            resource,
            required,
            maximum,
        });
    }
    Ok(())
}

fn check_storage_limit(
    resource: &'static str,
    required: u64,
    maximum: u64,
) -> Result<(), VarDctArtifactError> {
    if required > maximum {
        return Err(VarDctArtifactError::StorageBindingLimit {
            resource,
            required,
            maximum,
        });
    }
    Ok(())
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
