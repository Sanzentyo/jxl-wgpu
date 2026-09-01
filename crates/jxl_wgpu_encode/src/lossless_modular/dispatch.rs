use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use jxl_gpu_bitstream::BitWriter;

use super::grid::LosslessModularGroupGrid;
use super::memory::{
    LosslessModularMemoryLimits, LosslessModularMemoryPlan, align_up, event_capacity,
};
use super::serializer::{ModularFrameHeader, pack_signed, write_animation_header};
use super::streaming::{
    EncodeJobLifetime, LosslessModularJob, LosslessModularJobState, MapCompletion,
    ResidentLosslessModularJob,
};
use super::types::{
    EVENT_WORDS, LosslessModularFormat, LosslessModularTreeMode,
    MAX_DISPATCHES_PER_ARTIFACT_BINDING, ModularParams, OUTPUT_HEADER_WORDS, SHADER,
    lossless_modular_source_spec,
};
use crate::buffer_pool::EncoderBufferPool;
use crate::{
    AnimationHeader, BackendError, DEFAULT_ENCODER_BUFFER_POOL_BYTES, Determinism, EncodeError,
    EncodeProfile, EncoderBufferPoolStats, EncoderCapabilities, FrameEncodeRequest, FrameIndex,
    FrameOptions, GpuEncodeBackend, GpuFrameSource, KernelStage, ProfileCapability,
    UnsupportedFeature, WgpuContext,
};

#[derive(Clone, Copy, Debug)]
pub(super) struct ModularGroupPlan {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channel: u32,
    pub(super) artifact_byte_offset: u64,
    pub(super) output_size: u64,
    pub(super) max_events: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModularDispatchBatch {
    pub(super) first_dispatch: usize,
    pub(super) dispatch_count: usize,
    pub(super) artifact_byte_offset: u64,
    pub(super) artifact_binding_size: NonZeroU64,
    pub(super) source_binding_offset: u64,
    pub(super) source_binding_size: NonZeroU64,
}

#[derive(Clone, Debug)]
pub(super) struct ModularDispatchPlan {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) group_grid: LosslessModularGroupGrid,
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) parameters: Vec<ModularParams>,
    pub(super) groups: Vec<ModularGroupPlan>,
    pub(super) batches: Vec<ModularDispatchBatch>,
    pub(super) output_size: u64,
    pub(super) memory: LosslessModularMemoryPlan,
}

/// GPU lossless 1-16-bit integer Modular encoding with row-major 256x256 pass groups.
///
/// It never reads source pixels on the CPU. The source buffer may contain packed Gray, RGB, or
/// RGBA unsigned samples in canonical native `u8`/`u16` storage. RGB samples use the normative
/// reversible YCoCg transform in WGSL before prediction. The GPU emits predictor
/// residual tokens and histograms; the host only serializes those artifacts.
pub struct LosslessModularBackend {
    pub(super) pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) buffer_pool: Arc<EncoderBufferPool>,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    storage_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u32,
    pub(super) direct_mapping: bool,
    tree_mode: LosslessModularTreeMode,
}

impl LosslessModularBackend {
    #[must_use]
    pub fn new(context: &WgpuContext) -> Self {
        Self::with_tree_mode(context, LosslessModularTreeMode::SharedGlobal)
    }

    /// Creates a backend with an explicit multi-group MA-tree placement policy.
    #[must_use]
    pub fn with_tree_mode(context: &WgpuContext, tree_mode: LosslessModularTreeMode) -> Self {
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu lossless modular token kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu lossless modular token pipeline"),
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
                    min_bits_per_sample: 1,
                    max_bits_per_sample: 16,
                }],
                max_progressive_passes: 1,
                animation: true,
                determinism: Determinism::CrossDevice,
                implemented_stages: vec![
                    KernelStage::ColorTransform,
                    KernelStage::ModularTransform,
                    KernelStage::ModularPrediction,
                    KernelStage::ModularResidualTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            direct_mapping: context.direct_mapping_enabled(),
            tree_mode,
        }
    }

    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessModularMemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessModularMemoryLimits {
        LosslessModularMemoryLimits {
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

    pub(super) fn dispatch_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<ModularDispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        let group_grid = LosslessModularGroupGrid::for_extent(extent.width, extent.height)?;
        let source_spec = lossless_modular_source_spec(&source.layout.format)?;
        let format = source_spec.format;
        let channels = format.channel_count();
        let dispatches =
            group_grid
                .groups
                .checked_mul(channels)
                .ok_or(EncodeError::InvalidSource(
                    "Modular dispatch count overflow",
                ))?;
        if dispatches > self.max_compute_workgroups_per_dimension {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(dispatches),
                available: u64::from(self.max_compute_workgroups_per_dimension),
            }
            .into());
        }
        if source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !source.buffer.size().is_multiple_of(4)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing Modular plane"))?;
        let row_stride = u32::try_from(plane.row_stride).map_err(|_| {
            EncodeError::InvalidSource("row stride exceeds the Modular profile limit")
        })?;
        let row_bytes = u64::from(extent.width)
            .checked_mul(u64::from(channels))
            .and_then(|value| value.checked_mul(u64::from(source_spec.bytes_per_sample)))
            .ok_or(EncodeError::InvalidSource("source row size overflow"))?;
        if plane.row_stride < row_bytes {
            return Err(EncodeError::InvalidSource(
                "row stride is smaller than the packed Modular row width",
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
            .and_then(|value| value.checked_add(row_bytes))
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
        let dispatch_count = usize::try_from(dispatches)
            .map_err(|_| EncodeError::InvalidSource("Modular dispatch count overflow"))?;
        if !256_u64.is_multiple_of(self.storage_offset_alignment.max(1)) {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "min_storage_buffer_offset_alignment",
                required: self.storage_offset_alignment,
                available: 256,
            }
            .into());
        }
        let artifact_alignment = self.storage_offset_alignment.max(4);
        let mut parameters = Vec::with_capacity(dispatch_count);
        let mut groups = Vec::with_capacity(dispatch_count);
        let mut absolute_source_offsets = Vec::with_capacity(dispatch_count);
        let mut batches =
            Vec::with_capacity(dispatch_count.div_ceil(MAX_DISPATCHES_PER_ARTIFACT_BINDING));
        let mut output_size = 0u64;
        let mut batch_first_dispatch = 0usize;
        let mut batch_artifact_offset = 0u64;
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
            let mut proposed_output_size = output_size;
            for _ in 0..channels {
                proposed_output_size = align_up(proposed_output_size, artifact_alignment)
                    .and_then(|value| value.checked_add(group_output_size))
                    .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            }
            let batch_dispatches = parameters.len() - batch_first_dispatch;
            let proposed_batch_bytes = proposed_output_size
                .checked_sub(batch_artifact_offset)
                .ok_or(EncodeError::InvalidSource(
                    "artifact batch offset underflow",
                ))?;
            if batch_dispatches != 0
                && (batch_dispatches
                    .checked_add(usize::try_from(channels).map_err(|_| {
                        EncodeError::InvalidSource("Modular channel count overflow")
                    })?)
                    .is_none_or(|count| count > MAX_DISPATCHES_PER_ARTIFACT_BINDING)
                    || proposed_batch_bytes > self.max_storage_binding_size)
            {
                let batch_end_dispatch = parameters.len();
                batches.push(modular_dispatch_batch(
                    batch_first_dispatch..batch_end_dispatch,
                    batch_artifact_offset..output_size,
                    &mut ModularBatchFinalizeContext {
                        minimum_source_binding_offset: source_binding_offset,
                        absolute_source_offsets: &absolute_source_offsets,
                        parameters: &mut parameters,
                        groups: &groups,
                        source_alignment: artifact_alignment,
                        max_storage_binding_size: self.max_storage_binding_size,
                    },
                )?);
                batch_first_dispatch = parameters.len();
                output_size = align_up(output_size, artifact_alignment).ok_or(
                    EncodeError::InvalidSource("artifact batch alignment overflow"),
                )?;
                batch_artifact_offset = output_size;
                proposed_output_size = output_size;
                for _ in 0..channels {
                    proposed_output_size = align_up(proposed_output_size, artifact_alignment)
                        .and_then(|value| value.checked_add(group_output_size))
                        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
                }
            }
            if proposed_output_size
                .checked_sub(batch_artifact_offset)
                .is_none_or(|bytes| bytes > self.max_storage_binding_size)
            {
                return Err(UnsupportedFeature::DeviceLimit {
                    name: "max_storage_buffer_binding_size",
                    required: proposed_output_size.saturating_sub(batch_artifact_offset),
                    available: self.max_storage_binding_size,
                }
                .into());
            }
            let tile_byte_offset = plane
                .offset
                .checked_add(plane.row_stride.checked_mul(u64::from(y)).ok_or(
                    EncodeError::InvalidSource("source address arithmetic overflow"),
                )?)
                .and_then(|value| {
                    u64::from(x)
                        .checked_mul(u64::from(channels))
                        .and_then(|value| {
                            value.checked_mul(u64::from(source_spec.bytes_per_sample))
                        })
                        .and_then(|x_offset| value.checked_add(x_offset))
                })
                .ok_or(EncodeError::InvalidSource(
                    "source address arithmetic overflow",
                ))?;
            for channel in 0..channels {
                output_size = align_up(output_size, artifact_alignment).ok_or(
                    EncodeError::InvalidSource("artifact group alignment overflow"),
                )?;
                let batch_word_offset = output_size.checked_sub(batch_artifact_offset).ok_or(
                    EncodeError::InvalidSource("artifact batch offset underflow"),
                )? / 4;
                let output_word_offset = u32::try_from(batch_word_offset).map_err(|_| {
                    EncodeError::InvalidSource("artifact binding exceeds WGSL u32 indexing")
                })?;
                parameters.push(ModularParams {
                    width,
                    height,
                    row_stride,
                    byte_offset: 0,
                    output_word_offset,
                    channel,
                    channels,
                    bytes_per_sample: u32::from(source_spec.bytes_per_sample),
                    sample_mask: (1u32 << source_spec.bits_per_sample) - 1,
                    _padding: [0; 55],
                });
                groups.push(ModularGroupPlan {
                    width,
                    height,
                    channel,
                    artifact_byte_offset: output_size,
                    output_size: group_output_size,
                    max_events,
                });
                absolute_source_offsets.push(tile_byte_offset);
                output_size = output_size
                    .checked_add(group_output_size)
                    .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
            }
        }
        if parameters.len() > batch_first_dispatch {
            let batch_end_dispatch = parameters.len();
            batches.push(modular_dispatch_batch(
                batch_first_dispatch..batch_end_dispatch,
                batch_artifact_offset..output_size,
                &mut ModularBatchFinalizeContext {
                    minimum_source_binding_offset: source_binding_offset,
                    absolute_source_offsets: &absolute_source_offsets,
                    parameters: &mut parameters,
                    groups: &groups,
                    source_alignment: artifact_alignment,
                    max_storage_binding_size: self.max_storage_binding_size,
                },
            )?);
        }
        let parameter_storage_bytes = batches
            .iter()
            .map(|batch| batch.dispatch_count)
            .max()
            .and_then(|count| u64::try_from(count).ok())
            .and_then(|count| count.checked_mul(std::mem::size_of::<ModularParams>() as u64))
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
        let artifact_storage_bytes = batches
            .iter()
            .map(|batch| batch.artifact_binding_size.get())
            .max()
            .ok_or(EncodeError::InvalidSource("artifact batch plan is empty"))?;
        if artifact_storage_bytes > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: artifact_storage_bytes,
                available: self.max_buffer_size,
            }
            .into());
        }
        let total_artifact_bytes = batches
            .iter()
            .try_fold(0u64, |total, batch| {
                total.checked_add(batch.artifact_binding_size.get())
            })
            .ok_or(EncodeError::InvalidSource("total artifact size overflow"))?;
        let peak_source_binding_bytes = batches
            .iter()
            .map(|batch| batch.source_binding_size.get())
            .max()
            .ok_or(EncodeError::InvalidSource("source batch plan is empty"))?;
        let readback_bytes = if self.direct_mapping {
            0
        } else {
            artifact_storage_bytes
        };
        let owned_bytes_per_job = artifact_storage_bytes
            .checked_add(readback_bytes)
            .and_then(|value| value.checked_add(parameter_storage_bytes))
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let addressed_bytes_per_job = owned_bytes_per_job
            .checked_add(peak_source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let batch_count = u32::try_from(batches.len())
            .map_err(|_| EncodeError::InvalidSource("artifact batch count overflow"))?;
        let streaming = batch_count > 1;
        let gpu_submission_count = if streaming {
            batch_count
                .checked_mul(2)
                .ok_or(EncodeError::InvalidSource("GPU submission count overflow"))?
        } else {
            1
        };
        let memory = LosslessModularMemoryPlan {
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
            bytes_per_sample: source_spec.bytes_per_sample,
            channel_count: channels,
            source_binding_bytes,
            peak_source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes,
            total_artifact_bytes,
            readback_bytes,
            direct_readback: self.direct_mapping,
            batch_count,
            gpu_submission_count,
            streaming,
            owned_bytes_per_job,
            addressed_bytes_per_job,
        };
        Ok(ModularDispatchPlan {
            width: extent.width,
            height: extent.height,
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
            tree_mode: self.tree_mode,
            parameters,
            groups,
            batches,
            output_size,
            memory,
        })
    }
}
struct ModularBatchFinalizeContext<'a> {
    minimum_source_binding_offset: u64,
    absolute_source_offsets: &'a [u64],
    parameters: &'a mut [ModularParams],
    groups: &'a [ModularGroupPlan],
    source_alignment: u64,
    max_storage_binding_size: u64,
}

fn modular_dispatch_batch(
    dispatches: std::ops::Range<usize>,
    artifact_range: std::ops::Range<u64>,
    context: &mut ModularBatchFinalizeContext<'_>,
) -> Result<ModularDispatchBatch, EncodeError> {
    let first_dispatch = dispatches.start;
    let end_dispatch = dispatches.end;
    let artifact_byte_offset = artifact_range.start;
    let artifact_end = artifact_range.end;
    let dispatch_count =
        end_dispatch
            .checked_sub(first_dispatch)
            .ok_or(EncodeError::InvalidSource(
                "artifact batch dispatch range underflow",
            ))?;
    if dispatch_count == 0 {
        return Err(EncodeError::InvalidSource(
            "artifact batch must contain at least one dispatch",
        ));
    }
    let artifact_binding_bytes =
        artifact_end
            .checked_sub(artifact_byte_offset)
            .ok_or(EncodeError::InvalidSource(
                "artifact batch byte range underflow",
            ))?;
    if artifact_binding_bytes > context.max_storage_binding_size {
        return Err(UnsupportedFeature::DeviceLimit {
            name: "max_storage_buffer_binding_size",
            required: artifact_binding_bytes,
            available: context.max_storage_binding_size,
        }
        .into());
    }
    let artifact_binding_size = NonZeroU64::new(artifact_binding_bytes).ok_or(
        EncodeError::InvalidSource("artifact batch binding must not be empty"),
    )?;
    let batch_offsets = context
        .absolute_source_offsets
        .get(first_dispatch..end_dispatch)
        .ok_or(EncodeError::InvalidSource(
            "source batch dispatch range is invalid",
        ))?;
    let mut source_binding_offset = *batch_offsets
        .iter()
        .min()
        .ok_or(EncodeError::InvalidSource("source batch is empty"))?;
    source_binding_offset -= source_binding_offset % context.source_alignment;
    if source_binding_offset < context.minimum_source_binding_offset {
        return Err(EncodeError::InvalidSource(
            "source batch begins before the declared image plane",
        ));
    }
    let mut source_binding_end = source_binding_offset;
    for (index, &absolute_source_offset) in context
        .absolute_source_offsets
        .iter()
        .enumerate()
        .take(end_dispatch)
        .skip(first_dispatch)
    {
        let params = context
            .parameters
            .get(index)
            .ok_or(EncodeError::InvalidSource(
                "parameter batch range is invalid",
            ))?;
        let group = context.groups.get(index).ok_or(EncodeError::InvalidSource(
            "artifact batch range is invalid",
        ))?;
        let row_bytes = u64::from(group.width)
            .checked_mul(u64::from(params.channels))
            .and_then(|value| value.checked_mul(u64::from(params.bytes_per_sample)))
            .ok_or(EncodeError::InvalidSource("source batch row size overflow"))?;
        let source_end = absolute_source_offset
            .checked_add(
                u64::from(params.row_stride)
                    .checked_mul(u64::from(group.height.saturating_sub(1)))
                    .ok_or(EncodeError::InvalidSource("source batch address overflow"))?,
            )
            .and_then(|value| value.checked_add(row_bytes))
            .ok_or(EncodeError::InvalidSource("source batch address overflow"))?;
        source_binding_end = source_binding_end.max(source_end);
    }
    source_binding_end = align_up(source_binding_end, 4)
        .ok_or(EncodeError::InvalidSource("source batch size overflow"))?;
    let source_binding_bytes = source_binding_end
        .checked_sub(source_binding_offset)
        .ok_or(EncodeError::InvalidSource("source batch range underflow"))?;
    if source_binding_bytes > context.max_storage_binding_size {
        return Err(UnsupportedFeature::DeviceLimit {
            name: "max_storage_buffer_binding_size",
            required: source_binding_bytes,
            available: context.max_storage_binding_size,
        }
        .into());
    }
    let source_binding_size = NonZeroU64::new(source_binding_bytes)
        .ok_or(EncodeError::InvalidSource("source batch binding is empty"))?;
    let shader_last_byte = source_binding_bytes
        .checked_sub(1)
        .ok_or(EncodeError::InvalidSource("source batch binding is empty"))?;
    u32::try_from(shader_last_byte).map_err(|_| {
        EncodeError::InvalidSource("source batch exceeds the WGSL u32 address space")
    })?;
    for (index, &absolute_source_offset) in context
        .absolute_source_offsets
        .iter()
        .enumerate()
        .take(end_dispatch)
        .skip(first_dispatch)
    {
        context.parameters[index].byte_offset = u32::try_from(
            absolute_source_offset
                .checked_sub(source_binding_offset)
                .ok_or(EncodeError::InvalidSource("source batch address underflow"))?,
        )
        .map_err(|_| EncodeError::InvalidSource("source batch address exceeds WGSL u32"))?;
    }
    Ok(ModularDispatchBatch {
        first_dispatch,
        dispatch_count,
        artifact_byte_offset,
        artifact_binding_size,
        source_binding_offset,
        source_binding_size,
    })
}
pub(super) fn validate_modular_frame_request(
    request: &FrameEncodeRequest,
    plan: &ModularDispatchPlan,
) -> Result<(), EncodeError> {
    if request.canvas_width == 0 || request.canvas_height == 0 {
        return Err(EncodeError::InvalidConfiguration(
            "the JPEG XL animation canvas must be non-empty",
        ));
    }
    match request.animation {
        AnimationHeader::Still => {
            if request.frame_index != FrameIndex::new(0)
                || !request.is_last
                || request.options != FrameOptions::default()
                || request.canvas_width != plan.width
                || request.canvas_height != plan.height
            {
                return Err(EncodeError::InvalidConfiguration(
                    "a still lossless Modular request must be one full-canvas final frame",
                ));
            }
        }
        AnimationHeader::Animation { have_timecodes, .. } => {
            write_animation_header(&mut BitWriter::new(), request.animation)?;
            if request.options.timing.timecode.is_some() != have_timecodes {
                return Err(EncodeError::InvalidConfiguration(
                    "frame timecode presence must match the animation header",
                ));
            }
            let (frame_width, frame_height) = request
                .options
                .crop
                .map_or((request.canvas_width, request.canvas_height), |crop| {
                    (crop.width(), crop.height())
                });
            if let Some(crop) = request.options.crop {
                for value in [
                    pack_signed(crop.x()),
                    pack_signed(crop.y()),
                    crop.width(),
                    crop.height(),
                ] {
                    if value >= 18_688 + (1 << 30) {
                        return Err(EncodeError::InvalidConfiguration(
                            "animation frame crop coordinate exceeds the JPEG XL limit",
                        ));
                    }
                }
            }
            if frame_width != plan.width || frame_height != plan.height {
                return Err(EncodeError::InvalidConfiguration(
                    "the GPU source extent must match the animation frame crop",
                ));
            }
            let extra_channels = usize::from(plan.format.has_alpha());
            if !request.options.extra_channel_blends.is_empty()
                && request.options.extra_channel_blends.len() != extra_channels
            {
                return Err(EncodeError::InvalidConfiguration(
                    "animation extra-channel blend count does not match the source format",
                ));
            }
            if !plan.format.has_alpha()
                && matches!(
                    request.options.color_blend.mode,
                    crate::BlendMode::Blend | crate::BlendMode::MultiplyAdd
                )
            {
                return Err(EncodeError::InvalidConfiguration(
                    "alpha-weighted animation blending requires an RGBA source",
                ));
            }
            let color_uses_clamp = request.options.color_blend.mode == crate::BlendMode::Multiply
                || (plan.format.has_alpha()
                    && matches!(
                        request.options.color_blend.mode,
                        crate::BlendMode::Blend | crate::BlendMode::MultiplyAdd
                    ));
            if request.options.color_blend.clamp && !color_uses_clamp {
                return Err(EncodeError::InvalidConfiguration(
                    "the selected JPEG XL color blend mode has no clamp field",
                ));
            }
            if request.options.extra_channel_blends.iter().any(|blend| {
                blend.clamp
                    && !matches!(
                        blend.mode,
                        crate::BlendMode::Blend
                            | crate::BlendMode::MultiplyAdd
                            | crate::BlendMode::Multiply
                    )
            }) {
                return Err(EncodeError::InvalidConfiguration(
                    "the selected JPEG XL extra-channel blend mode has no clamp field",
                ));
            }
            if request.is_last && request.options.save_as_reference != Default::default() {
                return Err(EncodeError::InvalidConfiguration(
                    "the final JPEG XL frame cannot be saved as a reference",
                ));
            }
            let full_frame = frame_covers_canvas(
                request.options.crop,
                request.canvas_width,
                request.canvas_height,
            );
            let resets_canvas =
                request.options.color_blend.mode == crate::BlendMode::Replace && full_frame;
            let can_be_referenced = !request.is_last
                && (request.options.timing.duration_ticks == 0
                    || request.options.save_as_reference.get() != 0);
            let writes_save_before = resets_canvas && can_be_referenced;
            if request.options.save_before_color_transform && !writes_save_before {
                return Err(EncodeError::InvalidConfiguration(
                    "save-before-color-transform is not present for this frame contract",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn frame_covers_canvas(
    crop: Option<crate::FrameCrop>,
    canvas_width: u32,
    canvas_height: u32,
) -> bool {
    let Some(crop) = crop else {
        return true;
    };
    i64::from(crop.x()) <= 0
        && i64::from(crop.y()) <= 0
        && i64::from(crop.x()) + i64::from(crop.width()) >= i64::from(canvas_width)
        && i64::from(crop.y()) + i64::from(crop.height()) >= i64::from(canvas_height)
}

impl GpuEncodeBackend for LosslessModularBackend {
    type Job = LosslessModularJob;

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
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;
        validate_modular_frame_request(request, &plan)?;
        if request.profile
            != (EncodeProfile::ModularLossless {
                bits_per_sample: plan.bits_per_sample,
            })
        {
            return Err(EncodeError::InvalidConfiguration(
                "requested Modular depth does not match the source valid bits",
            ));
        }
        if plan.memory.streaming {
            return self.submit_streaming(context, source, plan, request.clone());
        }
        let memory_permit = context
            .memory_budget()
            .try_reserve(plan.memory.owned_bytes_per_job)?;

        let buffer_lease = self.buffer_pool.checkout(
            context.device(),
            plan.memory.parameter_storage_bytes,
            plan.output_size,
            self.direct_mapping,
        );
        let buffers = buffer_lease.buffers();
        context.queue().write_buffer(
            &buffers.parameters,
            0,
            bytemuck::cast_slice(&plan.parameters),
        );
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_groups = plan
            .batches
            .iter()
            .map(|batch| {
                let parameter_offset = u64::try_from(batch.first_dispatch)
                    .ok()
                    .and_then(|index| {
                        index.checked_mul(std::mem::size_of::<ModularParams>() as u64)
                    })
                    .ok_or(EncodeError::InvalidSource(
                        "artifact batch parameter offset overflow",
                    ))?;
                let parameter_size = u64::try_from(batch.dispatch_count)
                    .ok()
                    .and_then(|count| {
                        count.checked_mul(std::mem::size_of::<ModularParams>() as u64)
                    })
                    .and_then(NonZeroU64::new)
                    .ok_or(EncodeError::InvalidSource(
                        "artifact batch parameter size overflow",
                    ))?;
                Ok((
                    context
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("jxl-wgpu lossless modular batch bindings"),
                            layout: &bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &source.buffer,
                                        offset: batch.source_binding_offset,
                                        size: Some(batch.source_binding_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &buffers.artifact,
                                        offset: batch.artifact_byte_offset,
                                        size: Some(batch.artifact_binding_size),
                                    }),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                        buffer: &buffers.parameters,
                                        offset: parameter_offset,
                                        size: Some(parameter_size),
                                    }),
                                },
                            ],
                        }),
                    u32::try_from(batch.dispatch_count).map_err(|_| {
                        EncodeError::InvalidSource("artifact batch dispatch count overflow")
                    })?,
                ))
            })
            .collect::<Result<Vec<_>, EncodeError>>()?;
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu lossless modular encode"),
                });
        commands.clear_buffer(&buffers.artifact, 0, None);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu lossless modular tokenization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (bind_group, dispatch_count) in &bind_groups {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(*dispatch_count, 1, 1);
            }
        }
        if !self.direct_mapping {
            commands.copy_buffer_to_buffer(
                &buffers.artifact,
                0,
                &buffers.readback,
                0,
                plan.output_size,
            );
        }

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
                callback_completion.complete(result.map_err(BackendError::ArtifactMapping));
                drop(callback_lifetime);
            },
        );
        let poll_permit = context.submission_poller().try_reserve()?;
        let submission_index = context.queue().submit([commands.finish()]);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission_index, move |error| {
            poll_completion.complete(Err(BackendError::PollWorker(error)));
        }) {
            completion.complete(Err(BackendError::PollRegistration(error)));
        }

        Ok(LosslessModularJob {
            state: LosslessModularJobState::Resident(ResidentLosslessModularJob {
                lifetime: Some(lifetime),
                completion,
                output_size: plan.output_size,
                group_grid: plan.group_grid,
                groups: plan.groups,
                format: plan.format,
                bits_per_sample: plan.bits_per_sample,
                tree_mode: plan.tree_mode,
                width: plan.width,
                height: plan.height,
                frame_index: request.frame_index,
                is_last: request.is_last,
                header: ModularFrameHeader {
                    animation: request.animation,
                    canvas_width: request.canvas_width,
                    canvas_height: request.canvas_height,
                    options: request.options.clone(),
                    is_last: request.is_last,
                },
            }),
        })
    }
}
