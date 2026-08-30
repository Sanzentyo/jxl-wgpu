// The JPEG XL header and fast-lossless control-plane construction in this module is derived
// from the permissively licensed zune-jpegxl 0.5.2 encoder. See `THIRD_PARTY.md` and
// `LICENSES/zune-jpegxl-MIT.txt` in this crate.

use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, BitWriter, ContainerBox, Gray8AccelerationIndex,
    write_container_with_boxes,
};
use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use jxl_wgpu::MemoryPermit;
#[cfg(test)]
use wgpu::util::DeviceExt;

use crate::buffer_pool::EncoderBufferPool;
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BitFragment, DEFAULT_ENCODER_BUFFER_POOL_BYTES, Determinism, EncodeError,
    EncodeProfile, EncoderBufferPoolStats, EncoderCapabilities, FrameEncodeRequest,
    FrameGroupLayout, FrameIndex, FrameOptions, FramePacketSet, FrameSubmission,
    GpuAccelerationArtifact, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameArtifacts,
    GpuFrameSource, GroupPacket, GroupPacketKind, KernelStage, ProfileCapability, ProgressivePlan,
    UnsupportedFeature, WgpuContext, assemble_frame,
};

/// JPEG XL's default Modular pass-group edge length.
pub const LOSSLESS_GRAY8_GROUP_DIMENSION: u32 = 256;
const LOSSLESS_GRAY8_LF_GROUP_DIMENSION: u32 = LOSSLESS_GRAY8_GROUP_DIMENSION * 8;
const SHADER: &str = include_str!("lossless_gray8.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8Params {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
    output_word_offset: u32,
}

/// Fixed storage-buffer header written by `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8ArtifactHeader {
    event_count: u32,
    raw_counts: [u32; RAW_SYMBOLS],
    lz77_counts: [u32; LZ77_SYMBOLS],
}

/// Fixed storage-buffer event written after [`Gray8ArtifactHeader`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8Event {
    kind: u32,
    token: u32,
    extra_bit_count: u32,
    extra_bits: u32,
}

const OUTPUT_HEADER_WORDS: usize = std::mem::size_of::<Gray8ArtifactHeader>() / 4;
const EVENT_WORDS: usize = std::mem::size_of::<Gray8Event>() / 4;

const _: () = {
    assert!(std::mem::size_of::<Gray8Params>() == 20);
    assert!(std::mem::align_of::<Gray8Params>() == 4);
    assert!(std::mem::size_of::<Gray8ArtifactHeader>() == 53 * 4);
    assert!(std::mem::align_of::<Gray8ArtifactHeader>() == 4);
    assert!(std::mem::size_of::<Gray8Event>() == 16);
    assert!(std::mem::align_of::<Gray8Event>() == 4);
};

/// Row-major JPEG XL pass-group grid used by one Gray8 frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8GroupGrid {
    pub width: u32,
    pub height: u32,
    pub columns: u32,
    pub rows: u32,
    pub groups: u32,
    pub lf_columns: u32,
    pub lf_rows: u32,
    pub lf_groups: u32,
}

impl LosslessGray8GroupGrid {
    fn for_extent(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width >= (1 << 30) || height >= (1 << 30) {
            return Err(EncodeError::InvalidConfiguration(
                "gray8 dimensions must be in 1..2^30",
            ));
        }
        let columns = width.div_ceil(LOSSLESS_GRAY8_GROUP_DIMENSION);
        let rows = height.div_ceil(LOSSLESS_GRAY8_GROUP_DIMENSION);
        let groups = columns
            .checked_mul(rows)
            .ok_or(EncodeError::InvalidSource("Gray8 group count overflow"))?;
        let lf_columns = width.div_ceil(LOSSLESS_GRAY8_LF_GROUP_DIMENSION);
        let lf_rows = height.div_ceil(LOSSLESS_GRAY8_LF_GROUP_DIMENSION);
        let lf_groups = lf_columns
            .checked_mul(lf_rows)
            .ok_or(EncodeError::InvalidSource("Gray8 LF group count overflow"))?;
        // FrameGroupLayout performs the normative TOC-entry bound as well. Do it here so an
        // impossible grid is rejected before any driver allocation or queue interaction.
        FrameGroupLayout::new(lf_groups, groups, 1)?;
        Ok(Self {
            width,
            height,
            columns,
            rows,
            groups,
            lf_columns,
            lf_rows,
            lf_groups,
        })
    }

    /// Resolves a canonical row-major pass-group index to its exact pixel rectangle.
    #[must_use]
    pub fn group(self, index: u32) -> Option<LosslessGray8Group> {
        if index >= self.groups {
            return None;
        }
        let column = index % self.columns;
        let row = index / self.columns;
        let x = column.checked_mul(LOSSLESS_GRAY8_GROUP_DIMENSION)?;
        let y = row.checked_mul(LOSSLESS_GRAY8_GROUP_DIMENSION)?;
        Some(LosslessGray8Group {
            index,
            column,
            row,
            x,
            y,
            width: (self.width - x).min(LOSSLESS_GRAY8_GROUP_DIMENSION),
            height: (self.height - y).min(LOSSLESS_GRAY8_GROUP_DIMENSION),
        })
    }

    /// Iterates the standard JPEG XL TOC PassGroup order.
    pub fn ordered_groups(self) -> impl ExactSizeIterator<Item = LosslessGray8Group> {
        (0..self.groups).map(move |index| {
            self.group(index)
                .expect("an index from the checked group range is valid")
        })
    }
}

/// One GPU workgroup and its standard row-major JPEG XL PassGroup destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8Group {
    pub index: u32,
    pub column: u32,
    pub row: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Checked memory accounting for one concrete Gray8 submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8MemoryPlan {
    pub group_grid: LosslessGray8GroupGrid,
    pub source_binding_bytes: u64,
    pub parameter_storage_bytes: u64,
    pub artifact_storage_bytes: u64,
    pub readback_bytes: u64,
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

/// Total memory exposure for a caller-selected maximum number of in-flight jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8InFlightMemory {
    pub max_in_flight_jobs: u32,
    pub total_owned_bytes: u64,
    pub total_addressed_bytes: u64,
}

/// Device limits that bound concrete Gray8 source and artifact bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8MemoryLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_compute_workgroups_per_dimension: u32,
}

impl LosslessGray8MemoryPlan {
    pub fn for_in_flight(
        self,
        max_in_flight_jobs: u32,
    ) -> Result<LosslessGray8InFlightMemory, EncodeError> {
        if max_in_flight_jobs == 0 {
            return Err(EncodeError::InvalidConfiguration(
                "max in-flight job count must be non-zero",
            ));
        }
        let jobs = u64::from(max_in_flight_jobs);
        let total_owned_bytes =
            self.owned_bytes_per_job
                .checked_mul(jobs)
                .ok_or(EncodeError::InvalidConfiguration(
                    "in-flight encoder memory size overflow",
                ))?;
        let total_addressed_bytes = self.addressed_bytes_per_job.checked_mul(jobs).ok_or(
            EncodeError::InvalidConfiguration("in-flight encoder memory size overflow"),
        )?;
        Ok(LosslessGray8InFlightMemory {
            max_in_flight_jobs,
            total_owned_bytes,
            total_addressed_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Gray8GroupPlan {
    width: u32,
    height: u32,
    artifact_byte_offset: u64,
    output_size: u64,
    max_events: usize,
}

#[derive(Clone, Debug)]
struct Gray8DispatchPlan {
    width: u32,
    height: u32,
    group_grid: LosslessGray8GroupGrid,
    parameters: Vec<Gray8Params>,
    groups: Vec<Gray8GroupPlan>,
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    output_size: u64,
    memory: LosslessGray8MemoryPlan,
}

/// GPU lossless 8-bit grayscale Modular encoding with row-major 256x256 pass groups.
///
/// It never reads source pixels on the CPU. The source buffer must contain one
/// `PixelFormat::non_color(Unsigned, 8, &[X])` plane. The GPU emits predictor
/// residual tokens and histograms; the host only serializes those artifacts.
pub struct LosslessGray8Backend {
    pipeline: Arc<wgpu::ComputePipeline>,
    buffer_pool: Arc<EncoderBufferPool>,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    storage_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u32,
}

impl LosslessGray8Backend {
    #[must_use]
    pub fn new(context: &WgpuContext) -> Self {
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu lossless gray8 token kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu lossless gray8 token pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        let limits = context.device().limits();
        Self {
            pipeline,
            buffer_pool: EncoderBufferPool::new(DEFAULT_ENCODER_BUFFER_POOL_BYTES),
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::ModularLossless {
                    min_bits_per_sample: 8,
                    max_bits_per_sample: 8,
                }],
                max_progressive_passes: 1,
                animation: false,
                determinism: Determinism::CrossDevice,
                implemented_stages: vec![
                    KernelStage::ModularPrediction,
                    KernelStage::ModularResidualTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }

    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessGray8MemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessGray8MemoryLimits {
        LosslessGray8MemoryLimits {
            max_storage_buffer_binding_size: self.max_storage_binding_size,
            max_buffer_size: self.max_buffer_size,
            min_storage_buffer_offset_alignment: self.storage_offset_alignment,
            max_compute_workgroups_per_dimension: self.max_compute_workgroups_per_dimension,
        }
    }

    /// Reports encoder-owned parameter, artifact, and readback buffers retained for reuse.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> EncoderBufferPoolStats {
        self.buffer_pool.stats()
    }

    /// Changes the maximum idle allocation bytes retained by this backend.
    ///
    /// A value of zero disables retention. Reducing the limit immediately evicts oldest idle
    /// sets; resources already leased to GPU jobs follow the new limit when they complete.
    pub fn set_buffer_pool_limit(&self, limit_bytes: u64) {
        self.buffer_pool.set_limit(limit_bytes);
    }

    /// Drops all idle buffers and prevents currently leased sets from re-entering the pool.
    ///
    /// In-flight buffers remain exclusively owned by their submissions until their GPU mapping
    /// callback finishes, then are discarded instead of being retained.
    pub fn clear_buffer_pool(&self) {
        self.buffer_pool.clear();
    }

    fn dispatch_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<Gray8DispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let group_grid = LosslessGray8GroupGrid::for_extent(extent.width, extent.height)?;
        if group_grid.groups > self.max_compute_workgroups_per_dimension {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(group_grid.groups),
                available: u64::from(self.max_compute_workgroups_per_dimension),
            }
            .into());
        }
        if source.layout.format != PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X])
            || source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !source.buffer.size().is_multiple_of(4)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing grayscale plane"))?;
        let row_stride = u32::try_from(plane.row_stride).map_err(|_| {
            EncodeError::InvalidSource("row stride exceeds the Gray8 profile limit")
        })?;
        if plane.row_stride < u64::from(extent.width) {
            return Err(EncodeError::InvalidSource(
                "row stride is smaller than the grayscale row width",
            ));
        }
        let preceding_rows = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let sample_end = plane
            .offset
            .checked_add(preceding_rows)
            .and_then(|value| value.checked_add(u64::from(extent.width)))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let _full_stride_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height))
            .and_then(|value| plane.offset.checked_add(value))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4)
            .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "source binding does not contain the final addressable sample word",
            ));
        }
        // A storage array of u32 also needs a word-aligned base even on a
        // hypothetical device reporting a smaller dynamic-offset alignment.
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        if !source_binding_offset.is_multiple_of(alignment) {
            return Err(EncodeError::InvalidSource(
                "source storage binding offset is not device-aligned",
            ));
        }
        let source_binding_bytes = binding_end
            .checked_sub(source_binding_offset)
            .ok_or(EncodeError::InvalidSource("source binding range underflow"))?;
        if !source_binding_bytes.is_multiple_of(4) {
            return Err(EncodeError::InvalidSource(
                "source storage binding size is not word-aligned",
            ));
        }
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("source binding is empty"))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|value| value.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("source address exceeds the WGSL u32 address space")
        })?;

        let group_count = usize::try_from(group_grid.groups)
            .map_err(|_| EncodeError::InvalidSource("Gray8 group count overflow"))?;
        let mut parameters = Vec::with_capacity(group_count);
        let mut groups = Vec::with_capacity(group_count);
        let mut output_size = 0u64;
        for group in group_grid.ordered_groups() {
            let x = group.x;
            let y = group.y;
            let width = group.width;
            let height = group.height;
            let pixel_count = usize::try_from(u64::from(width) * u64::from(height))
                .map_err(|_| EncodeError::InvalidSource("group dimensions overflow"))?;
            let max_events = event_capacity(pixel_count)?;
            let output_words = OUTPUT_HEADER_WORDS
                .checked_add(
                    max_events
                        .checked_mul(EVENT_WORDS)
                        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?,
                )
                .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            let group_output_size = u64::try_from(output_words)
                .ok()
                .and_then(|words| words.checked_mul(4))
                .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            let output_word_offset = u32::try_from(output_size / 4).map_err(|_| {
                EncodeError::InvalidSource("artifact buffer exceeds WGSL u32 indexing")
            })?;
            let tile_byte_offset = plane
                .offset
                .checked_add(plane.row_stride.checked_mul(u64::from(y)).ok_or(
                    EncodeError::InvalidSource("source address arithmetic overflow"),
                )?)
                .and_then(|value| value.checked_add(u64::from(x)))
                .and_then(|value| value.checked_sub(source_binding_offset))
                .ok_or(EncodeError::InvalidSource(
                    "source address arithmetic overflow",
                ))?;
            let byte_offset = u32::try_from(tile_byte_offset).map_err(|_| {
                EncodeError::InvalidSource("source address exceeds the WGSL u32 address space")
            })?;
            parameters.push(Gray8Params {
                width,
                height,
                row_stride,
                byte_offset,
                output_word_offset,
            });
            groups.push(Gray8GroupPlan {
                width,
                height,
                artifact_byte_offset: output_size,
                output_size: group_output_size,
                max_events,
            });
            output_size = output_size
                .checked_add(group_output_size)
                .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
        }
        let output_words = output_size / 4;
        if output_words > u64::from(u32::MAX) + 1 {
            return Err(EncodeError::InvalidSource(
                "artifact buffer exceeds WGSL u32 indexing",
            ));
        }
        if output_size > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: output_size,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        if output_size > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: output_size,
                available: self.max_buffer_size,
            }
            .into());
        }
        let parameter_storage_bytes = u64::try_from(parameters.len())
            .ok()
            .and_then(|count| {
                count.checked_mul(u64::try_from(std::mem::size_of::<Gray8Params>()).ok()?)
            })
            .ok_or(EncodeError::InvalidSource(
                "group parameter storage size overflow",
            ))?;
        if parameter_storage_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: parameter_storage_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        if parameter_storage_bytes > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: parameter_storage_bytes,
                available: self.max_buffer_size,
            }
            .into());
        }
        let owned_bytes_per_job = output_size
            .checked_mul(2)
            .and_then(|value| value.checked_add(parameter_storage_bytes))
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let addressed_bytes_per_job = owned_bytes_per_job
            .checked_add(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let memory = LosslessGray8MemoryPlan {
            group_grid,
            source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes: output_size,
            readback_bytes: output_size,
            owned_bytes_per_job,
            addressed_bytes_per_job,
        };
        Ok(Gray8DispatchPlan {
            width: extent.width,
            height: extent.height,
            group_grid,
            parameters,
            groups,
            source_binding_offset,
            source_binding_size,
            output_size,
            memory,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn event_capacity(pixel_count: usize) -> Result<usize, EncodeError> {
    pixel_count
        .checked_add(pixel_count.div_ceil(8))
        .and_then(|value| value.checked_add(1))
        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))
}

impl GpuEncodeBackend for LosslessGray8Backend {
    type Job = LosslessGray8Job;

    fn capabilities(&self) -> &EncoderCapabilities {
        &self.capabilities
    }

    fn supports_input(&self, source: &GpuFrameSource) -> bool {
        let GpuFrameSource::Buffer(source) = source else {
            return false;
        };
        self.dispatch_plan(source).is_ok()
    }

    fn submit(
        &self,
        context: &WgpuContext,
        source: GpuFrameSource,
        request: &FrameEncodeRequest,
    ) -> Result<Self::Job, EncodeError> {
        if request.animation != AnimationHeader::Still
            || request.frame_index != FrameIndex::new(0)
            || !request.is_last
        {
            return Err(UnsupportedFeature::Animation.into());
        }
        if request.options != FrameOptions::default() {
            return Err(EncodeError::InvalidConfiguration(
                "the Gray8 profile only supports default still-frame options",
            ));
        }
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let buffer_lease = self.buffer_pool.checkout(
            context.device(),
            plan.memory.parameter_storage_bytes,
            plan.output_size,
        );
        let buffers = buffer_lease.buffers();
        context.queue().write_buffer(
            &buffers.parameters,
            0,
            bytemuck::cast_slice(&plan.parameters),
        );
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu lossless gray8 bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &source.buffer,
                            offset: plan.source_binding_offset,
                            size: Some(plan.source_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.artifact.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.parameters.as_entire_binding(),
                    },
                ],
            });
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu lossless gray8 encode"),
                });
        commands.clear_buffer(&buffers.artifact, 0, None);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu lossless gray8 tokenization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(plan.group_grid.groups, 1, 1);
        }
        commands.copy_buffer_to_buffer(
            &buffers.artifact,
            0,
            &buffers.readback,
            0,
            plan.output_size,
        );

        let completion = Arc::new(MapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let readback_for_map = Arc::clone(&buffers.readback);
        let lifetime = Arc::new(EncodeJobLifetime {
            buffer_lease,
            _memory_permit: memory_permit,
            mapped: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(
            &readback_for_map,
            wgpu::MapMode::Read,
            0..plan.output_size,
            move |result| {
                if result.is_ok() {
                    callback_lifetime.mapped.store(true, Ordering::Release);
                }
                callback_completion.complete(
                    result.map_err(|error| format!("GPU artifact mapping failed: {error}")),
                );
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!("GPU poll registration failed: {error}")));
        }

        Ok(LosslessGray8Job {
            lifetime: Some(lifetime),
            completion,
            output_size: plan.output_size,
            group_grid: plan.group_grid,
            groups: plan.groups,
            width: plan.width,
            height: plan.height,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
}

#[derive(Default)]
struct MapCompletion {
    state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<Waker>,
}

impl MapCompletion {
    fn complete(&self, result: Result<(), String>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.condition.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("map completion was checked as present")
    }
}

/// Runtime-neutral completion for the concrete GPU lossless profile.
pub struct LosslessGray8Job {
    lifetime: Option<Arc<EncodeJobLifetime>>,
    completion: Arc<MapCompletion>,
    output_size: u64,
    group_grid: LosslessGray8GroupGrid,
    groups: Vec<Gray8GroupPlan>,
    width: u32,
    height: u32,
    frame_index: FrameIndex,
    is_last: bool,
}

struct EncodeJobLifetime {
    buffer_lease: crate::buffer_pool::EncoderBufferLease,
    _memory_permit: MemoryPermit,
    mapped: AtomicBool,
}

impl Drop for EncodeJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.buffer_lease.buffers().readback.unmap();
        }
    }
}

impl LosslessGray8Job {
    fn finish(&mut self, mapping: Result<(), String>) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self
            .lifetime
            .take()
            .ok_or_else(|| EncodeError::Backend("GPU job was already consumed".into()))?;
        mapping.map_err(EncodeError::Backend)?;
        let readback = &lifetime.buffer_lease.buffers().readback;
        let mapped = readback
            .slice(0..self.output_size)
            .get_mapped_range()
            .map_err(|error| {
                EncodeError::Backend(format!("invalid mapped artifact range: {error}"))
            })?;
        let expected = usize::try_from(self.output_size)
            .map_err(|_| EncodeError::Backend("mapped artifact size overflow".into()))?;
        let bytes = mapped
            .get(..expected)
            .ok_or_else(|| EncodeError::Backend("mapped artifact buffer was truncated".into()))?;
        let result = build_packets(
            self.width,
            self.height,
            self.group_grid,
            &self.groups,
            bytes,
        );
        drop(mapped);
        readback.unmap();
        lifetime.mapped.store(false, Ordering::Release);
        drop(lifetime);
        let (packets, acceleration) = result?;
        Ok(GpuFrameArtifacts {
            frame_index: self.frame_index,
            is_last: self.is_last,
            packets,
            acceleration,
        })
    }
}

impl GpuEncodeJob for LosslessGray8Job {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(EncodeError::Backend(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission".into(),
            ))
        }
    }
}

/// Convenience API that produces a complete raw codestream or deterministic
/// `jxlc` container from a GPU-resident grayscale buffer.
pub struct LosslessGray8Encoder {
    encoder: GpuEncoder<LosslessGray8Backend>,
}

impl LosslessGray8Encoder {
    #[must_use]
    pub fn new(context: WgpuContext) -> Self {
        let backend = LosslessGray8Backend::new(&context);
        Self {
            encoder: GpuEncoder::new(context, backend),
        }
    }

    /// Creates an encoder with an application-selected idle buffer retention limit.
    ///
    /// The limit is independent of the context's live-job [`jxl_wgpu::MemoryBudget`]. A value of
    /// zero creates buffers on demand and drops them immediately after each mapping callback.
    #[must_use]
    pub fn with_buffer_pool_limit(context: WgpuContext, limit_bytes: u64) -> Self {
        let backend = LosslessGray8Backend::new(&context);
        backend.set_buffer_pool_limit(limit_bytes);
        Self {
            encoder: GpuEncoder::new(context, backend),
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Reports aggregate owned bytes retained by currently live encode jobs.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> jxl_wgpu::MemoryBudgetSnapshot {
        self.encoder.memory_stats()
    }

    /// Computes all source, artifact, and readback bytes before submission.
    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessGray8MemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessGray8MemoryLimits {
        self.encoder.backend().memory_limits()
    }

    /// Reports reusable encoder-owned GPU buffers and cumulative reuse counters.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> EncoderBufferPoolStats {
        self.encoder.backend().buffer_pool_stats()
    }

    /// Changes the maximum idle allocation bytes retained for later submissions.
    pub fn set_buffer_pool_limit(&self, limit_bytes: u64) {
        self.encoder.backend().set_buffer_pool_limit(limit_bytes);
    }

    /// Clears idle buffers; in-flight sets from before the clear are discarded on completion.
    pub fn clear_buffer_pool(&self) {
        self.encoder.backend().clear_buffer_pool();
    }

    pub fn submit(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: crate::BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: crate::BufferImageSource,
        container: bool,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        // Preserve typed address/device-limit failures before the generic
        // backend admission predicate maps unsupported inputs to InputFormat.
        self.memory_plan(&source)?;
        let width = source.layout.extent.width;
        let height = source.layout.extent.height;
        let group_grid = LosslessGray8GroupGrid::for_extent(width, height)?;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless { bits_per_sample: 8 },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::CrossDevice,
            animation: AnimationHeader::Still,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(LosslessGray8Submission {
            frame: Some(frame),
            codestream_header: image_header(width, height)?,
            container,
            group_grid,
        })
    }
}

/// A `Future` with an executor-independent blocking counterpart.
pub struct LosslessGray8Submission {
    frame: Option<FrameSubmission<LosslessGray8Job>>,
    codestream_header: BitFragment,
    container: bool,
    group_grid: LosslessGray8GroupGrid,
}

impl LosslessGray8Submission {
    /// Exact row-major group grid dispatched by this submission.
    #[must_use]
    pub const fn group_grid(&self) -> LosslessGray8GroupGrid {
        self.group_grid
    }

    /// Canonical descriptors for the independently executed GPU workgroups.
    pub fn ordered_groups(&self) -> impl ExactSizeIterator<Item = LosslessGray8Group> {
        self.group_grid.ordered_groups()
    }

    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a lossless submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let acceleration = frame.acceleration;
        let fused_group_size = acceleration
            .as_ref()
            .map(|_| {
                frame
                    .packets
                    .packets()
                    .first()
                    .ok_or_else(|| EncodeError::Backend("gray8 frame has no group packet".into()))
                    .map(|packet| packet.payload.len())
            })
            .transpose()?;
        let encoded_frame = assemble_frame(frame.packets)?;
        let mut codestream = self.codestream_header.bytes().to_vec();
        codestream.extend_from_slice(encoded_frame.bytes());
        if !self.container {
            return Ok(codestream);
        }

        let Some(acceleration) = acceleration else {
            // The current private acceleration-index schema describes one contiguous token span.
            // Multi-group output remains a fully standard deterministic `jxlc` container, without
            // inventing an incompatible extension record.
            return Ok(write_container_with_boxes(&codestream, &[])?);
        };
        let group_size = fused_group_size.ok_or_else(|| {
            EncodeError::Backend("gray8 acceleration metadata requires a fused group".into())
        })?;
        let bytes_before_group = encoded_frame
            .bytes()
            .len()
            .checked_sub(group_size)
            .ok_or_else(|| EncodeError::Backend("gray8 group size exceeds frame size".into()))?;
        let group_start = self
            .codestream_header
            .bytes()
            .len()
            .checked_add(bytes_before_group)
            .ok_or_else(|| EncodeError::Backend("gray8 codestream size overflow".into()))?;

        let GpuAccelerationArtifact::Gray8Prefix {
            width,
            height,
            token_bit_offset_in_group,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        } = acceleration;
        let group_start_bits = u64::try_from(group_start)
            .ok()
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let token_bit_offset = group_start_bits
            .checked_add(token_bit_offset_in_group)
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let index = Gray8AccelerationIndex::new(
            &codestream,
            width,
            height,
            token_bit_offset,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        )?;
        let payload = index.serialize();
        Ok(write_container_with_boxes(
            &codestream,
            &[ContainerBox {
                box_type: ACCELERATION_INDEX_BOX_TYPE,
                payload: &payload,
            }],
        )?)
    }
}

impl Future for LosslessGray8Submission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a lossless submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
            }
        }
    }
}

fn build_packets(
    width: u32,
    height: u32,
    group_grid: LosslessGray8GroupGrid,
    group_plans: &[Gray8GroupPlan],
    bytes: &[u8],
) -> Result<(FramePacketSet, Option<GpuAccelerationArtifact>), EncodeError> {
    if group_plans.len() != group_grid.groups as usize {
        return Err(EncodeError::Backend(
            "GPU group plan does not match the frame grid".into(),
        ));
    }
    let mut artifacts = Vec::with_capacity(group_plans.len());
    let mut aggregate_raw = [0u64; RAW_SYMBOLS];
    let mut aggregate_lz77 = [0u64; LZ77_SYMBOLS];
    for plan in group_plans {
        let start = usize::try_from(plan.artifact_byte_offset)
            .map_err(|_| EncodeError::Backend("GPU artifact offset overflow".into()))?;
        let end = plan
            .artifact_byte_offset
            .checked_add(plan.output_size)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| EncodeError::Backend("GPU artifact range overflow".into()))?;
        let artifact_bytes = bytes
            .get(start..end)
            .ok_or_else(|| EncodeError::Backend("GPU group artifact is truncated".into()))?;
        let artifact =
            parse_group_artifact(plan.width, plan.height, plan.max_events, artifact_bytes)?;
        for (total, count) in aggregate_raw.iter_mut().zip(artifact.header.raw_counts) {
            *total = total
                .checked_add(u64::from(count))
                .ok_or_else(|| invalid_gpu_artifact("aggregate raw histogram overflow"))?;
        }
        for (total, count) in aggregate_lz77.iter_mut().zip(artifact.header.lz77_counts) {
            *total = total
                .checked_add(u64::from(count))
                .ok_or_else(|| invalid_gpu_artifact("aggregate LZ77 histogram overflow"))?;
        }
        artifacts.push(artifact);
    }

    let primary = PrefixCode::from_aggregated_counts(&aggregate_raw, &aggregate_lz77)?;
    let unused = PrefixCode::fixed_unused_channel();
    let codes = [primary.clone(), unused.clone(), unused.clone(), unused];
    if group_grid.groups == 1 {
        let mut group = BitWriter::new();
        write_dc_global(&mut group, &codes)?;
        let token_bit_offset_in_group = u64::try_from(group.bit_len())
            .map_err(|_| EncodeError::Backend("gray8 token offset overflow".into()))?;
        write_events(&mut group, &codes[0], artifacts[0].events)?;
        let token_bit_end = u64::try_from(group.bit_len())
            .map_err(|_| EncodeError::Backend("gray8 token length overflow".into()))?;
        let token_bit_len = token_bit_end
            .checked_sub(token_bit_offset_in_group)
            .ok_or_else(|| EncodeError::Backend("gray8 token length underflow".into()))?;
        group.align_to_byte()?;
        let packets = FramePacketSet::new(
            frame_header()?,
            FrameGroupLayout::new(1, 1, 1)?,
            [GroupPacket::new(
                GroupPacketKind::Single,
                group.into_bytes(),
            )],
        )?;
        let acceleration = GpuAccelerationArtifact::Gray8Prefix {
            width,
            height,
            token_bit_offset_in_group,
            token_bit_len,
            raw_prefix: codes[0].raw_entries(),
            lz77_prefix: codes[0].lz77_entries(),
        };
        return Ok((packets, Some(acceleration)));
    }

    let layout = FrameGroupLayout::new(group_grid.lf_groups, group_grid.groups, 1)?;
    let mut packets = Vec::with_capacity(layout.toc_entries());
    let mut dc_global = BitWriter::new();
    write_dc_global(&mut dc_global, &codes)?;
    dc_global.align_to_byte()?;
    packets.push(GroupPacket::new(
        GroupPacketKind::DcGlobal,
        dc_global.into_bytes(),
    ));
    for group in 0..group_grid.lf_groups {
        packets.push(GroupPacket::new(
            GroupPacketKind::DcGroup(group),
            Vec::new(),
        ));
    }
    // Lossless Modular has no VarDCT HF-global payload.
    packets.push(GroupPacket::new(GroupPacketKind::AcGlobal, Vec::new()));
    for (group, artifact) in artifacts.iter().enumerate() {
        let mut pass_group = BitWriter::new();
        // GroupHeader: use the LF-global tree, default weighted predictor, no transforms.
        pass_group.write_bits(1, 1)?;
        pass_group.write_bits(1, 1)?;
        pass_group.write_bits(0, 2)?;
        write_events(&mut pass_group, &codes[0], artifact.events)?;
        pass_group.align_to_byte()?;
        packets.push(GroupPacket::new(
            GroupPacketKind::AcGroup {
                pass: 0,
                group: u32::try_from(group)
                    .map_err(|_| EncodeError::Backend("Gray8 group index overflow".into()))?,
            },
            pass_group.into_bytes(),
        ));
    }
    Ok((FramePacketSet::new(frame_header()?, layout, packets)?, None))
}

#[derive(Clone, Copy)]
struct ValidatedGray8Artifact<'a> {
    header: Gray8ArtifactHeader,
    events: &'a [Gray8Event],
}

fn parse_group_artifact<'a>(
    width: u32,
    height: u32,
    max_events: usize,
    bytes: &'a [u8],
) -> Result<ValidatedGray8Artifact<'a>, EncodeError> {
    let header_bytes = bytes
        .get(..std::mem::size_of::<Gray8ArtifactHeader>())
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let header = bytemuck::try_cast_slice::<u8, Gray8ArtifactHeader>(header_bytes)
        .map_err(|_| EncodeError::Backend("GPU artifact header has an invalid ABI layout".into()))?
        .first()
        .copied()
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let event_count = usize::try_from(header.event_count)
        .map_err(|_| EncodeError::Backend("GPU event count overflow".into()))?;
    if event_count > max_events {
        return Err(EncodeError::Backend(
            "GPU emitted more token events than the output allocation".into(),
        ));
    }
    let event_bytes = event_count
        .checked_mul(std::mem::size_of::<Gray8Event>())
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let required_bytes = std::mem::size_of::<Gray8ArtifactHeader>()
        .checked_add(event_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let events = bytes
        .get(std::mem::size_of::<Gray8ArtifactHeader>()..required_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event stream is truncated".into()))?;
    let events = bytemuck::try_cast_slice::<u8, Gray8Event>(events)
        .map_err(|_| EncodeError::Backend("GPU event stream has an invalid ABI layout".into()))?;

    validate_gpu_artifacts(width, height, &header, events)?;
    Ok(ValidatedGray8Artifact { header, events })
}

fn write_events(
    output: &mut BitWriter,
    code: &PrefixCode,
    events: &[Gray8Event],
) -> Result<(), EncodeError> {
    for event in events {
        match event.kind {
            0 => code.write_raw(output, event.token, event.extra_bit_count, event.extra_bits)?,
            1 => code.write_run(output, event.token, event.extra_bit_count, event.extra_bits)?,
            _ => {
                return Err(EncodeError::Backend(
                    "GPU emitted an unknown token kind".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_gpu_artifacts(
    width: u32,
    height: u32,
    header: &Gray8ArtifactHeader,
    events: &[Gray8Event],
) -> Result<(), EncodeError> {
    let mut raw_counts = [0u32; RAW_SYMBOLS];
    let mut lz77_counts = [0u32; LZ77_SYMBOLS];
    let mut sample_count = 0u64;

    for event in events {
        match event.kind {
            0 => {
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("raw token overflow"))?;
                if token > 9 {
                    return Err(invalid_gpu_artifact("impossible raw token"));
                }
                let expected_nbits = event.token.saturating_sub(1);
                if event.extra_bit_count != expected_nbits
                    || !canonical_extra_bits(event.extra_bit_count, event.extra_bits)
                {
                    return Err(invalid_gpu_artifact("non-canonical raw token"));
                }
                raw_counts[token] = raw_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("raw histogram overflow"))?;
                sample_count = sample_count
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("sample count overflow"))?;
            }
            1 => {
                let token = usize::try_from(event.token)
                    .map_err(|_| invalid_gpu_artifact("LZ77 token overflow"))?;
                if token > 27 {
                    return Err(invalid_gpu_artifact("impossible LZ77 token"));
                }
                let expected_nbits = if event.token < 16 {
                    0
                } else {
                    event.token - 12
                };
                if event.extra_bit_count != expected_nbits
                    || !canonical_extra_bits(event.extra_bit_count, event.extra_bits)
                {
                    return Err(invalid_gpu_artifact("non-canonical LZ77 token"));
                }
                raw_counts[0] = raw_counts[0]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("raw histogram overflow"))?;
                lz77_counts[token] = lz77_counts[token]
                    .checked_add(1)
                    .ok_or_else(|| invalid_gpu_artifact("LZ77 histogram overflow"))?;
                let encoded_value = if event.token < 16 {
                    u64::from(event.token)
                } else {
                    (1u64 << event.extra_bit_count) + u64::from(event.extra_bits)
                };
                sample_count = sample_count
                    .checked_add(encoded_value + 8)
                    .ok_or_else(|| invalid_gpu_artifact("sample count overflow"))?;
            }
            _ => return Err(invalid_gpu_artifact("unknown token kind")),
        }
    }

    if raw_counts != header.raw_counts || lz77_counts != header.lz77_counts {
        return Err(invalid_gpu_artifact(
            "token histograms do not match the event stream",
        ));
    }
    let expected_samples = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| invalid_gpu_artifact("image sample count overflow"))?;
    if sample_count != expected_samples {
        return Err(invalid_gpu_artifact(
            "event stream does not cover the image exactly",
        ));
    }
    Ok(())
}

fn canonical_extra_bits(nbits: u32, bits: u32) -> bool {
    match nbits {
        0 => bits == 0,
        1..=31 => bits < (1u32 << nbits),
        _ => false,
    }
}

fn invalid_gpu_artifact(reason: &'static str) -> EncodeError {
    EncodeError::Backend(format!("invalid GPU artifact: {reason}"))
}

fn write_dc_global(output: &mut BitWriter, codes: &[PrefixCode; 4]) -> Result<(), EncodeError> {
    // Handcrafted Modular metadata adapted from zune-jpegxl 0.5.2. See this crate's
    // `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt`.
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    output.write_bits(0b100011, 6)?;
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0b1010, 4)?;
    output.write_bits(4, 4)?;
    output.write_bits(0, 3)?;
    output.write_bits(0, 3)?;
    output.write_bits(1, 1)?;
    output.write_bits(3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        output.write_bits(context, 3)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    output.write_bits(1, 5)?;
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        // libjxl's U32 selector stores the low eight bits of 256 here.
        output.write_bits(0, 8)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    for code in codes {
        code.write_tree(output)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    Ok(())
}

fn image_header(width: u32, height: u32) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?;
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(1, 2)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0b10, 2)?;
    output.write_bits(11, 4)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes())?)
}

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(1..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "gray8 dimensions must be in 1..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

fn frame_header() -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    let bit_len = output.bit_len();
    BitFragment::new(output.into_bytes(), bit_len).map_err(Into::into)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
    use jxl_gpu_protocol::Extent2d;
    use std::process::Command;

    fn checked_in_gpu_gray8_lossless() -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = include_str!("../test-data/gpu_gray8_lossless.jxl.hex")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    struct ReentrantWake {
        completion: Arc<MapCompletion>,
        observed_unlocked: Arc<AtomicBool>,
    }

    impl std::task::Wake for ReentrantWake {
        fn wake(self: Arc<Self>) {
            let _guard = self
                .completion
                .state
                .try_lock()
                .expect("completion mutex must be unlocked before invoking a waker");
            self.observed_unlocked.store(true, Ordering::Release);
        }
    }

    #[test]
    fn completion_wakes_after_releasing_its_mutex() {
        let completion = Arc::new(MapCompletion::default());
        let observed_unlocked = Arc::new(AtomicBool::new(false));
        let waker = std::task::Waker::from(Arc::new(ReentrantWake {
            completion: Arc::clone(&completion),
            observed_unlocked: Arc::clone(&observed_unlocked),
        }));
        let context = Context::from_waker(&waker);
        assert!(completion.poll(&context).is_none());
        completion.complete(Ok(()));
        assert!(observed_unlocked.load(Ordering::Acquire));
    }

    #[test]
    fn gray8_params_abi_matches_wgsl_storage_array() {
        assert_eq!(std::mem::size_of::<Gray8Params>(), 20);
        assert_eq!(std::mem::align_of::<Gray8Params>(), 4);
        let params = Gray8Params {
            width: 1,
            height: 2,
            row_stride: 3,
            byte_offset: 4,
            output_word_offset: 5,
        };
        assert_eq!(
            bytemuck::cast::<Gray8Params, [u32; 5]>(params),
            [1, 2, 3, 4, 5]
        );
        assert!(SHADER.contains(
            "struct Params {\n    width: u32,\n    height: u32,\n    row_stride: u32,\n    byte_offset: u32,\n    output_word_offset: u32,\n}"
        ));
        assert!(SHADER.contains("var<storage, read> group_params: array<Params>;"));
    }

    #[test]
    fn gray8_artifact_abi_matches_wgsl_word_schema() {
        assert_eq!(std::mem::size_of::<Gray8ArtifactHeader>(), 53 * 4);
        assert_eq!(std::mem::align_of::<Gray8ArtifactHeader>(), 4);
        assert_eq!(std::mem::size_of::<Gray8Event>(), 4 * 4);
        assert_eq!(std::mem::align_of::<Gray8Event>(), 4);

        let header = Gray8ArtifactHeader {
            event_count: 7,
            raw_counts: std::array::from_fn(|index| 100 + index as u32),
            lz77_counts: std::array::from_fn(|index| 200 + index as u32),
        };
        let words = bytemuck::cast::<Gray8ArtifactHeader, [u32; 53]>(header);
        assert_eq!(words[0], 7);
        assert_eq!(words[1..20], header.raw_counts);
        assert_eq!(words[20..53], header.lz77_counts);

        let event = Gray8Event {
            kind: 1,
            token: 2,
            extra_bit_count: 3,
            extra_bits: 4,
        };
        assert_eq!(bytemuck::cast::<Gray8Event, [u32; 4]>(event), [1, 2, 3, 4]);
        assert!(SHADER.contains("Word 0 is the event count, words 1..20 are raw-token counts"));
        assert!(SHADER.contains("// (kind, token, extra-bit count, extra bits)."));
        assert!(SHADER.contains("const OUTPUT_HEADER_WORDS: u32 = 53u;"));
        assert!(SHADER.contains("const EVENT_WORDS: u32 = 4u;"));
    }

    #[test]
    fn group_grid_is_row_major_and_covers_edge_tiles_exactly() {
        let grid = LosslessGray8GroupGrid::for_extent(513, 257).unwrap();
        assert_eq!(
            grid,
            LosslessGray8GroupGrid {
                width: 513,
                height: 257,
                columns: 3,
                rows: 2,
                groups: 6,
                lf_columns: 1,
                lf_rows: 1,
                lf_groups: 1,
            }
        );
        let groups = grid.ordered_groups().collect::<Vec<_>>();
        assert_eq!(groups.len(), 6);
        assert_eq!(
            groups[0],
            LosslessGray8Group {
                index: 0,
                column: 0,
                row: 0,
                x: 0,
                y: 0,
                width: 256,
                height: 256,
            }
        );
        assert_eq!(groups[2].x, 512);
        assert_eq!(groups[2].width, 1);
        assert_eq!(groups[3].y, 256);
        assert_eq!(groups[3].height, 1);
        assert_eq!((groups[5].x, groups[5].y), (512, 256));
        assert!(grid.group(6).is_none());

        assert_eq!(LosslessGray8GroupGrid::for_extent(1, 1).unwrap().groups, 1);
        assert!(LosslessGray8GroupGrid::for_extent(0, 1).is_err());
        assert!(LosslessGray8GroupGrid::for_extent(1, 0).is_err());
    }

    fn artifact_bytes(header: Gray8ArtifactHeader, events: &[Gray8Event]) -> Vec<u8> {
        let mut bytes = bytemuck::bytes_of(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(events));
        bytes
    }

    #[test]
    fn packet_builder_rejects_impossible_histogram_bins() {
        let mut header = Gray8ArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        header.raw_counts[12] = 1;
        let bytes = artifact_bytes(
            header,
            &[Gray8Event {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &bytes).is_err());
    }

    #[test]
    fn packet_builder_rejects_noncanonical_events_and_histogram_mismatches() {
        let mut header = Gray8ArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[2] = 1;
        let malformed = artifact_bytes(
            header,
            &[Gray8Event {
                kind: 0,
                token: 2,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &malformed).is_err());

        header.raw_counts = [0; RAW_SYMBOLS];
        header.raw_counts[1] = 1;
        let mismatched = artifact_bytes(
            header,
            &[Gray8Event {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(1, 1, 1, &mismatched).is_err());
    }

    #[test]
    fn packet_builder_rejects_event_streams_with_the_wrong_sample_count() {
        let mut header = Gray8ArtifactHeader {
            event_count: 1,
            raw_counts: [0; RAW_SYMBOLS],
            lz77_counts: [0; LZ77_SYMBOLS],
        };
        header.raw_counts[0] = 1;
        let bytes = artifact_bytes(
            header,
            &[Gray8Event {
                kind: 0,
                token: 0,
                extra_bit_count: 0,
                extra_bits: 0,
            }],
        );
        assert!(parse_group_artifact(2, 1, 1, &bytes).is_err());
    }

    /// Mirrors only the event-admission control flow in `encode` WGSL. A
    /// `true` sample is a zero packed residual; the actual token value is
    /// irrelevant to the number of four-word event records.
    fn simulated_shader_event_count(
        width: usize,
        height: usize,
        is_zero: impl Fn(usize) -> bool,
    ) -> usize {
        let mut run = 0usize;
        let mut events = 0usize;
        for y in 0..height {
            for chunk_x in (0..width).step_by(8) {
                let count = 8.min(width - chunk_x);
                let mut prefix = 0usize;
                while prefix < count && is_zero(y * width + chunk_x + prefix) {
                    prefix += 1;
                }
                if prefix == count && (run > 0 || prefix > 7) {
                    run += prefix;
                } else if prefix + run > 7 {
                    events += usize::from(run + prefix > 0);
                    events += count - prefix;
                    run = 0;
                } else {
                    events += count;
                }
            }
        }
        events + usize::from(run > 0)
    }

    #[test]
    fn event_allocation_bounds_every_shader_write() {
        // Exhaust every zero/non-zero residual stream up to 16 samples and
        // vary row boundaries because a run is intentionally frame-global.
        for width in 1usize..=16 {
            for height in 1usize..=(16 / width) {
                let pixels = width * height;
                let capacity = event_capacity(pixels).expect("small capacity is representable");
                for mask in 0u32..(1u32 << pixels) {
                    let events = simulated_shader_event_count(width, height, |index| {
                        mask & (1u32 << index) != 0
                    });
                    assert!(events <= capacity, "{width}x{height}, mask={mask:#x}");
                }
            }
        }

        let pixels = usize::try_from(
            u64::from(LOSSLESS_GRAY8_GROUP_DIMENSION) * u64::from(LOSSLESS_GRAY8_GROUP_DIMENSION),
        )
        .expect("maximum Gray8 profile dimensions fit usize");
        let capacity = event_capacity(pixels).expect("maximum event capacity fits usize");
        for events in [
            simulated_shader_event_count(pixels, 1, |_| false),
            simulated_shader_event_count(pixels, 1, |_| true),
            simulated_shader_event_count(pixels, 1, |index| index % 2 == 0),
            simulated_shader_event_count(pixels, 1, |index| index % 17 < 8),
        ] {
            assert!(events <= capacity);
        }

        let words = OUTPUT_HEADER_WORDS + capacity * EVENT_WORDS;
        let last_event_word = OUTPUT_HEADER_WORDS + (capacity - 1) * EVENT_WORDS + 3;
        assert!(last_event_word < words);
        assert_eq!(words * 4 % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
    }

    fn test_context() -> Option<WgpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu lossless encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        WgpuContext::new(Arc::new(device), Arc::new(queue)).ok()
    }

    fn packed_test_source(
        context: &WgpuContext,
        width: u32,
        height: u32,
    ) -> crate::BufferImageSource {
        let extent = Extent2d::new(width, height);
        let layout = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        )
        .unwrap();
        let pixels = packed_test_pixels(width, height);
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu encoder pool test source"),
                contents: &pixels,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        crate::BufferImageSource::new(buffer, layout).unwrap()
    }

    fn packed_test_pixels(width: u32, height: u32) -> Vec<u8> {
        let byte_count = usize::try_from(u64::from(width) * u64::from(height))
            .expect("test source size fits usize");
        (0..byte_count)
            .map(|index| ((index * 29 + index / 7) & 255) as u8)
            .collect()
    }

    fn expected_artifact_storage_bytes(width: u32, height: u32) -> u64 {
        LosslessGray8GroupGrid::for_extent(width, height)
            .unwrap()
            .ordered_groups()
            .map(|group| {
                let pixels =
                    usize::try_from(u64::from(group.width) * u64::from(group.height)).unwrap();
                let words = OUTPUT_HEADER_WORDS + event_capacity(pixels).unwrap() * EVENT_WORDS;
                u64::try_from(words).unwrap() * 4
            })
            .sum()
    }

    #[test]
    fn pool_exclusively_leases_real_gpu_buffer_sets_and_clear_invalidates_live_returns() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder pool lease test: no wgpu adapter");
            return;
        };
        let pool = EncoderBufferPool::new(64 * 1024);
        let first = pool.checkout(context.device(), 16, 1024);
        let first_artifact = Arc::clone(&first.buffers().artifact);
        let second = pool.checkout(context.device(), 16, 1024);
        assert!(!Arc::ptr_eq(&first_artifact, &second.buffers().artifact));
        assert_eq!(pool.stats().leased_buffer_sets, 2);
        assert_eq!(pool.stats().allocation_misses, 2);

        drop(first);
        let third = pool.checkout(context.device(), 16, 1024);
        assert!(Arc::ptr_eq(&first_artifact, &third.buffers().artifact));
        assert_eq!(pool.stats().reuse_hits, 1);
        assert_eq!(pool.stats().leased_buffer_sets, 2);

        pool.clear();
        drop(second);
        drop(third);
        let stats = pool.stats();
        assert_eq!(stats.leased_buffer_sets, 0);
        assert_eq!(stats.idle_buffer_sets, 0);
        assert_eq!(stats.idle_buffers, 0);
        assert_eq!(stats.evicted_buffer_sets, 2);
        assert_eq!(stats.evicted_buffers, 6);
    }

    #[test]
    fn sequential_gpu_jobs_reuse_exact_buffer_sets_and_enforce_the_idle_limit() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessGray8Encoder::with_buffer_pool_limit(context, 8 * 1024 * 1024);
        let allocation_bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;

        encoder.submit(source.clone()).unwrap().wait().unwrap();
        let first = encoder.buffer_pool_stats();
        assert_eq!(first.allocation_misses, 1);
        assert_eq!(first.reuse_hits, 0);
        assert_eq!(first.idle_buffer_sets, 1);
        assert_eq!(first.idle_buffers, 3);
        assert_eq!(first.idle_bytes, allocation_bytes);

        encoder.submit(source).unwrap().wait().unwrap();
        let reused = encoder.buffer_pool_stats();
        assert_eq!(reused.allocation_misses, 1);
        assert_eq!(reused.reuse_hits, 1);
        assert_eq!(reused.idle_buffer_sets, 1);

        encoder.set_buffer_pool_limit(allocation_bytes - 1);
        let evicted = encoder.buffer_pool_stats();
        assert_eq!(evicted.limit_bytes, allocation_bytes - 1);
        assert_eq!(evicted.idle_bytes, 0);
        assert_eq!(evicted.idle_buffer_sets, 0);
        assert_eq!(evicted.evicted_buffer_sets, 1);
        assert_eq!(evicted.evicted_buffers, 3);
        assert_eq!(evicted.evicted_bytes, allocation_bytes);
    }

    #[test]
    fn abandoned_gpu_future_returns_buffers_and_live_memory_after_mapping() {
        let Some(context) = test_context() else {
            eprintln!("skipping abandoned GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessGray8Encoder::with_buffer_pool_limit(
            context.clone(),
            DEFAULT_ENCODER_BUFFER_POOL_BYTES,
        );
        let dropped = encoder.submit(source.clone()).unwrap();
        assert_eq!(encoder.buffer_pool_stats().leased_buffer_sets, 1);
        assert_eq!(encoder.buffer_pool_stats().idle_buffer_sets, 0);
        assert!(encoder.in_flight_memory_stats().reserved_bytes > 0);
        drop(dropped);

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let pool = encoder.buffer_pool_stats();
            let memory = encoder.in_flight_memory_stats();
            if pool.leased_buffer_sets == 0
                && pool.idle_buffer_sets == 1
                && memory.reserved_bytes == 0
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "abandoned submission did not release resources: pool={pool:?}, memory={memory:?}"
            );
            let _ = context.device().poll(wgpu::PollType::Poll);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        encoder.submit(source).unwrap().wait().unwrap();
        let stats = encoder.buffer_pool_stats();
        assert_eq!(stats.allocation_misses, 1);
        assert_eq!(stats.reuse_hits, 1);
        assert_eq!(stats.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_real_gpu_submissions_reuse_only_completed_buffer_sets() {
        let Some(context) = test_context() else {
            eprintln!("skipping concurrent GPU encoder reuse test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 71, 121);
        let encoder = LosslessGray8Encoder::with_buffer_pool_limit(context, 32 * 1024 * 1024);
        let per_job = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, per_job * 8);
        let first_outputs = jobs
            .into_iter()
            .map(LosslessGray8Submission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(first_outputs.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

        let first_stats = encoder.buffer_pool_stats();
        assert_eq!(first_stats.reuse_hits + first_stats.allocation_misses, 8);
        assert_eq!(first_stats.leased_buffer_sets, 0);
        assert!(first_stats.idle_buffer_sets >= 1);
        let guaranteed_hits = first_stats.idle_buffer_sets;

        let jobs = (0..8)
            .map(|_| encoder.submit(source.clone()))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let second_outputs = jobs
            .into_iter()
            .map(LosslessGray8Submission::wait)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            second_outputs
                .iter()
                .all(|encoded| encoded == &first_outputs[0])
        );
        let second_stats = encoder.buffer_pool_stats();
        assert!(second_stats.reuse_hits >= first_stats.reuse_hits + guaranteed_hits);
        assert_eq!(second_stats.reuse_hits + second_stats.allocation_misses, 16);
        assert_eq!(second_stats.leased_buffer_sets, 0);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn poll_admission_failure_happens_before_submit_and_returns_the_pool_lease() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder poll admission test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 17, 13);
        let encoder = LosslessGray8Encoder::new(context.clone());
        let permits = (0..jxl_wgpu::SUBMISSION_POLLER_CAPACITY)
            .map(|_| context.submission_poller().try_reserve().unwrap())
            .collect::<Vec<_>>();

        let error = match encoder.submit(source.clone()) {
            Ok(_) => panic!("saturated poll admission must reject before queue submission"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            EncodeError::PollBackpressure(jxl_wgpu::SubmissionPollerError::Full { .. })
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let rejected = encoder.buffer_pool_stats();
        assert_eq!(rejected.leased_buffer_sets, 0);
        assert_eq!(rejected.idle_buffer_sets, 1);
        assert_eq!(rejected.allocation_misses, 1);

        drop(permits);
        encoder.submit(source).unwrap().wait().unwrap();
        let recovered = encoder.buffer_pool_stats();
        assert_eq!(recovered.reuse_hits, 1);
        assert_eq!(recovered.leased_buffer_sets, 0);
    }

    #[test]
    fn concurrent_encoder_jobs_use_owned_byte_backpressure() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU encoder backpressure test: no wgpu adapter");
            return;
        };
        let source = packed_test_source(&context, 2, 2);
        let plan = LosslessGray8Backend::new(&context)
            .memory_plan(&source)
            .unwrap();
        let limited = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = LosslessGray8Encoder::new(limited);

        let first = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            plan.owned_bytes_per_job
        );
        assert!(matches!(
            encoder.submit(source.clone()),
            Err(EncodeError::MemoryBackpressure(
                jxl_wgpu::MemoryBudgetError::Exhausted { .. }
            ))
        ));
        first.wait().unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        encoder.submit(source).unwrap().wait().unwrap();
    }

    fn decode_gray8(encoded: &[u8]) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let mut pixels = vec![0u8; size.0 * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    fn decode_with_djxl_if_available(encoded: &[u8]) -> Option<Result<Vec<u8>, String>> {
        static DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let id = DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("jxl-wgpu-gray8-{}-{id}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pgm");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pgm = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PGM: {error}"))?;
            parse_pgm(&pgm)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_pgm(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut cursor = 0usize;
        let mut token = || -> Result<&[u8], String> {
            loop {
                while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                    cursor += 1;
                }
                if bytes.get(cursor) == Some(&b'#') {
                    while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                        cursor += 1;
                    }
                    continue;
                }
                break;
            }
            let start = cursor;
            while bytes
                .get(cursor)
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            bytes
                .get(start..cursor)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "truncated PGM header".into())
        };
        if token()? != b"P5" {
            return Err("djxl did not emit a binary grayscale PGM".into());
        }
        let width = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let height = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if token()? != b"255" {
            return Err("djxl PGM did not contain 8-bit samples".into());
        }
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let pixels = bytes
            .get(cursor..)
            .ok_or_else(|| "truncated PGM pixels".to_string())?;
        if pixels.len() != width * height {
            return Err(format!(
                "djxl PGM has {} samples, expected {}",
                pixels.len(),
                width * height
            ));
        }
        Ok(pixels.to_vec())
    }

    #[test]
    fn gpu_groups_cover_safe_boundary_extents_and_decode_exactly() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group lossless encode test: no wgpu adapter");
            return;
        };
        let encoder = LosslessGray8Encoder::new(context.clone());
        let extents = [
            (1, 1),
            (17, 13),
            (257, 1),
            (1, 257),
            (257, 257),
            (513, 3),
            (3, 513),
            (513, 513),
            (4_097, 1),
            (1, 4_097),
        ];
        for (width, height) in extents {
            let expected = packed_test_pixels(width, height);
            let source = packed_test_source(&context, width, height);
            let memory = encoder.memory_plan(&source).unwrap();
            assert_eq!(
                memory.parameter_storage_bytes,
                u64::from(memory.group_grid.groups) * 20
            );
            assert_eq!(
                memory.artifact_storage_bytes,
                expected_artifact_storage_bytes(width, height)
            );
            assert_eq!(memory.readback_bytes, memory.artifact_storage_bytes);
            assert_eq!(
                memory.owned_bytes_per_job,
                memory.parameter_storage_bytes + memory.artifact_storage_bytes * 2
            );
            assert_eq!(
                memory.addressed_bytes_per_job,
                memory.owned_bytes_per_job + memory.source_binding_bytes
            );

            let submission = encoder.submit(source.clone()).unwrap();
            assert_eq!(
                submission.ordered_groups().collect::<Vec<_>>(),
                memory.group_grid.ordered_groups().collect::<Vec<_>>()
            );
            let encoded = submission.wait().unwrap();
            let (size, decoded) = decode_gray8(&encoded)
                .unwrap_or_else(|error| panic!("Rust jxl rejected {width}x{height}: {error}"));
            assert_eq!(size, (width as usize, height as usize));
            assert_eq!(decoded, expected, "Rust jxl mismatch for {width}x{height}");
            let container = encoder.encode_container(source).unwrap();
            let parsed =
                jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                    .unwrap();
            assert_eq!(parsed.codestream(), encoded);
            let (_, container_decoded) = decode_gray8(&container).unwrap_or_else(|error| {
                panic!("Rust jxl rejected {width}x{height} container: {error}")
            });
            assert_eq!(container_decoded, expected);
            if let Some(decoded) = decode_with_djxl_if_available(&container) {
                assert_eq!(
                    decoded.unwrap_or_else(|error| {
                        panic!("djxl rejected {width}x{height} container: {error}")
                    }),
                    expected,
                    "djxl mismatch for {width}x{height}"
                );
            }
        }
    }

    #[test]
    fn multi_group_container_and_runtime_neutral_future_are_deterministic() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU multi-group container test: no wgpu adapter");
            return;
        };
        let width = 257;
        let height = 257;
        let expected = packed_test_pixels(width, height);
        let source = packed_test_source(&context, width, height);
        let encoder = LosslessGray8Encoder::new(context);
        let raw = encoder.encode(source.clone()).unwrap();
        let async_raw = pollster::block_on(encoder.submit(source.clone()).unwrap()).unwrap();
        assert_eq!(async_raw, raw);

        let container = encoder.encode_container(source.clone()).unwrap();
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap();
        assert_eq!(parsed.codestream(), raw);
        assert_eq!(
            parsed.boxes_of_type(ACCELERATION_INDEX_BOX_TYPE).count(),
            0,
            "the current private acceleration index is intentionally single-group"
        );
        let (size, decoded) = decode_gray8(&container).unwrap();
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(decoded.unwrap(), expected);
        }
        assert_eq!(encoder.encode_container(source).unwrap(), container);
    }

    #[test]
    fn gpu_tokens_form_a_reference_decodable_lossless_codestream() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU lossless encode test: no wgpu adapter");
            return;
        };
        let width = 17u32;
        let height = 13u32;
        let row_stride = 20u64;
        let binding_alignment = u64::from(
            context
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        )
        .max(4);
        let offset = binding_alignment + 4;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4)
            .expect("test allocation size is representable");
        let mut allocation = vec![0u8; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                };
                allocation[(offset + u64::from(y) * row_stride + u64::from(x)) as usize] = value;
                expected.push(value);
            }
        }
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu gray8 test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        let extent = Extent2d::new(width, height);
        let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
        let layout = ImageLayout::from_planes(
            extent,
            format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes: u64::from(width),
            }],
        )
        .expect("test image layout is valid");
        let source = crate::BufferImageSource::new(buffer, layout).expect("test source is valid");
        let encoder = LosslessGray8Encoder::new(context);
        let memory = encoder
            .memory_plan(&source)
            .expect("test source has a checked memory plan");
        let pixel_count = usize::try_from(width * height).expect("test dimensions fit usize");
        let expected_output_words = OUTPUT_HEADER_WORDS
            + event_capacity(pixel_count).expect("test event capacity") * EVENT_WORDS;
        let expected_output_bytes =
            u64::try_from(expected_output_words * 4).expect("test artifact size fits u64");
        assert_eq!(memory.group_grid.groups, 1);
        assert_eq!(memory.parameter_storage_bytes, 20);
        assert_eq!(memory.artifact_storage_bytes, expected_output_bytes);
        assert_eq!(memory.readback_bytes, expected_output_bytes);
        assert_eq!(memory.owned_bytes_per_job, 20 + expected_output_bytes * 2);
        assert_eq!(
            memory.addressed_bytes_per_job,
            memory.owned_bytes_per_job + memory.source_binding_bytes
        );
        let in_flight = memory
            .for_in_flight(4)
            .expect("four-job memory total is representable");
        assert_eq!(in_flight.max_in_flight_jobs, 4);
        assert_eq!(in_flight.total_owned_bytes, memory.owned_bytes_per_job * 4);
        assert_eq!(
            in_flight.total_addressed_bytes,
            memory.addressed_bytes_per_job * 4
        );
        let limits = encoder.memory_limits();
        assert_eq!(
            limits.min_storage_buffer_offset_alignment.max(4),
            binding_alignment
        );
        let encoded = encoder
            .encode(source.clone())
            .expect("GPU lossless encode succeeds");
        let (size, decoded) = decode_gray8(&encoded).expect("jxl reference decoder accepts output");
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&encoded) {
            assert_eq!(decoded.expect("djxl accepts GPU codestream"), expected);
        }
        let submission = encoder
            .submit(source.clone())
            .expect("runtime-neutral Future submission succeeds");
        let async_encoded =
            pollster::block_on(submission).expect("runtime-neutral Future encode succeeds");
        assert_eq!(async_encoded, encoded);

        let container = encoder
            .encode_container(source.clone())
            .expect("GPU lossless container encode succeeds");
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .expect("container is structurally valid");
        assert_eq!(parsed.codestream(), encoded);
        let boxes = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .collect::<Vec<_>>();
        assert_eq!(boxes.len(), 1);
        let index = Gray8AccelerationIndex::parse_bound(boxes[0].payload, parsed.codestream())
            .expect("jwgp index is bound to the exact codestream");
        assert_eq!(index.width(), width);
        assert_eq!(index.height(), height);
        assert_eq!(index.sample_count(), width * height);
        let (_, decoded) =
            decode_gray8(&container).expect("jxl reference decoder ignores the private box");
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(
                decoded.expect("djxl ignores jwgp and decodes jxlc"),
                expected
            );
        }
        let second = encoder
            .encode_container(source)
            .expect("second deterministic container encode succeeds");
        assert_eq!(container, second);
        assert_eq!(container, checked_in_gpu_gray8_lossless());
        if let Some(path) = std::env::var_os("JXL_WGPU_WRITE_FIXTURE") {
            std::fs::write(path, &container).expect("requested fixture path is writable");
        }
    }
}
