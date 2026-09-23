use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use jxl_gpu_bitstream::BitWriter;

use super::grid::LosslessModularGroupGrid;
use super::lz77::LosslessModularLz77;
use super::memory::{
    LosslessModularMemoryLimits, LosslessModularMemoryPlan, align_up, event_capacity,
};
use super::palette::{LosslessModularPalette, encoded_channels};
use super::predictor::{LosslessModularPredictor, LosslessModularWeightedPredictor};
use super::rct::ResolvedRct;
use super::serializer::{ModularFrameHeader, pack_signed, write_animation_header};
use super::source::{ModularSourceLayout, ModularSourceWindows};
use super::squeeze::LosslessModularSqueeze;
use super::streaming::{
    EncodeJobLifetime, LosslessModularJob, LosslessModularJobState, MapCompletion,
    ResidentLosslessModularJob,
};
use super::types::{
    EVENT_WORDS, LosslessModularConfig, LosslessModularFormat, LosslessModularTreeMode,
    MAX_DISPATCHES_PER_ARTIFACT_BINDING, ModularParams, OUTPUT_HEADER_WORDS, SHADER,
    modular_sample_depth,
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
    pub(super) group_index: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channel: u32,
    pub(super) artifact_byte_offset: u64,
    pub(super) output_size: u64,
    pub(super) max_events: usize,
    pub(super) palette_counts_byte_offset: Option<u64>,
    pub(super) palette_delta_capacity: u32,
}

struct ModularChannelLayout {
    width: u32,
    height: u32,
    max_events: usize,
    output_words: usize,
    weighted_words: u64,
    output_size: u64,
}

impl ModularChannelLayout {
    fn new(width: u32, height: u32, config: LosslessModularConfig) -> Result<Self, EncodeError> {
        let pixels = usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| EncodeError::InvalidSource("group dimensions overflow"))?;
        let max_events = event_capacity(pixels)?;
        let output_words = max_events
            .checked_mul(EVENT_WORDS)
            .and_then(|words| words.checked_add(OUTPUT_HEADER_WORDS))
            .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
        let weighted_words = if config.predictor == LosslessModularPredictor::Weighted {
            5 * u64::from(width)
        } else {
            0
        };
        let output_size = u64::try_from(output_words)
            .ok()
            .and_then(|words| words.checked_add(weighted_words))
            .and_then(|words| words.checked_add(config.lz77.scratch_words(width * height)))
            .and_then(|words| words.checked_mul(4))
            .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
        Ok(Self {
            width,
            height,
            max_events,
            output_words,
            weighted_words,
            output_size,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ModularDispatchBatch {
    pub(super) first_dispatch: usize,
    pub(super) dispatch_count: usize,
    pub(super) artifact_byte_offset: u64,
    pub(super) artifact_binding_size: NonZeroU64,
    pub(super) source_windows: ModularSourceWindows,
}

#[derive(Clone, Debug)]
pub(super) struct ModularDispatchPlan {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) group_grid: LosslessModularGroupGrid,
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) exponent_bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) rct: Option<ResolvedRct>,
    pub(super) squeeze: LosslessModularSqueeze,
    pub(super) palette: Option<LosslessModularPalette>,
    pub(super) predictor: LosslessModularPredictor,
    pub(super) weighted_predictor: LosslessModularWeightedPredictor,
    pub(super) lz77: LosslessModularLz77,
    pub(super) parameters: Vec<ModularParams>,
    pub(super) groups: Vec<ModularGroupPlan>,
    pub(super) batches: Vec<ModularDispatchBatch>,
    pub(super) output_size: u64,
    pub(super) memory: LosslessModularMemoryPlan,
}

/// GPU lossless integer/IEEE floating Modular encoding with configurable row-major pass groups.
///
/// It never reads source pixels on the CPU. Gray, GrayAlpha, RGB and RGBA components may occupy packed,
/// planar or split storage with explicit swizzles, bit positions and word byte order. Samples
/// have one common 1-31-bit integer or binary16/binary32 precision.
/// The selected reversible color transform operates on source words. The GPU
/// emits predictor residual tokens and histograms; the host only serializes those artifacts.
pub struct LosslessModularBackend {
    pipeline: Option<Arc<wgpu::ComputePipeline>>,
    pub(super) buffer_pool: Arc<EncoderBufferPool>,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_storage_buffers_per_shader_stage: u32,
    max_buffer_size: u64,
    storage_offset_alignment: u64,
    max_compute_workgroups_per_dimension: u32,
    pub(super) direct_mapping: bool,
    config: LosslessModularConfig,
}

impl LosslessModularBackend {
    #[must_use]
    pub fn new(context: &WgpuContext) -> Self {
        Self::with_config(context, Default::default())
    }

    /// Creates a backend with an explicit multi-group MA-tree placement policy.
    #[must_use]
    pub fn with_tree_mode(context: &WgpuContext, tree_mode: LosslessModularTreeMode) -> Self {
        Self::with_config(
            context,
            LosslessModularConfig {
                tree_mode,
                ..Default::default()
            },
        )
    }

    /// Selects group geometry, MA-tree placement and RCT policy before any source is submitted.
    #[must_use]
    pub fn with_config(context: &WgpuContext, config: LosslessModularConfig) -> Self {
        let limits = context.device().limits();
        let pipeline =
            (limits.max_storage_buffers_per_shader_stage >= 6).then(|| {
                let module = context
                    .device()
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("jxl-wgpu lossless modular token kernel"),
                        source: wgpu::ShaderSource::Wgsl(
                            jxl_wgpu::modular_prediction_shader(SHADER).into(),
                        ),
                    });
                Arc::new(context.device().create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("jxl-wgpu lossless modular token pipeline"),
                        layout: None,
                        module: &module,
                        entry_point: Some("encode"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[
                                (
                                    "squeeze_enabled",
                                    f64::from(u32::from(
                                        config.squeeze != LosslessModularSqueeze::None,
                                    )),
                                ),
                                (
                                    "palette_enabled",
                                    f64::from(u32::from(config.palette.is_some())),
                                ),
                            ],
                            ..Default::default()
                        },
                        cache: None,
                    },
                ))
            });
        Self {
            pipeline,
            buffer_pool: EncoderBufferPool::new(DEFAULT_ENCODER_BUFFER_POOL_BYTES),
            capabilities: EncoderCapabilities {
                profiles: [(1, 31, 0), (16, 16, 5), (32, 32, 8)]
                    .map(
                        |(min_bits_per_sample, max_bits_per_sample, exponent_bits_per_sample)| {
                            ProfileCapability::ModularLossless {
                                min_bits_per_sample,
                                max_bits_per_sample,
                                exponent_bits_per_sample,
                            }
                        },
                    )
                    .to_vec(),
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
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_buffer_size: limits.max_buffer_size,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            direct_mapping: context.direct_mapping_enabled(),
            config,
        }
    }

    #[must_use]
    pub const fn config(&self) -> LosslessModularConfig {
        self.config
    }

    pub(super) fn pipeline(&self) -> Result<&Arc<wgpu::ComputePipeline>, EncodeError> {
        self.pipeline.as_ref().ok_or_else(|| {
            UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffers_per_shader_stage",
                required: 6,
                available: u64::from(self.max_storage_buffers_per_shader_stage),
            }
            .into()
        })
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
        self.pipeline()?;
        let extent = source.layout.extent;
        let group_grid = LosslessModularGroupGrid::for_extent(
            extent.width,
            extent.height,
            self.config.group_size,
        )?;
        let source_layout = ModularSourceLayout::new(
            &source.layout,
            source.buffer.size(),
            self.storage_offset_alignment,
        )?;
        let source_spec = &source_layout.spec;
        let format = source_spec.format;
        let rct = self
            .config
            .color_transform
            .resolve(format, source_spec.exponent_bits_per_sample)?;
        let max_squeeze = self.config.squeeze.for_extent(
            extent.width.min(self.config.group_size.dimension()),
            extent.height.min(self.config.group_size.dimension()),
        );
        let channels = encoded_channels(format, max_squeeze, self.config.palette);
        let dispatches = group_grid.ordered_groups().try_fold(0u32, |count, group| {
            count
                .checked_add(encoded_channels(
                    format,
                    self.config.squeeze.for_extent(group.width, group.height),
                    self.config.palette,
                ))
                .ok_or(EncodeError::InvalidSource(
                    "Modular dispatch count overflow",
                ))
        })?;
        if dispatches > self.max_compute_workgroups_per_dimension {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_compute_workgroups_per_dimension",
                required: u64::from(dispatches),
                available: u64::from(self.max_compute_workgroups_per_dimension),
            }
            .into());
        }
        if !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !source.buffer.size().is_multiple_of(4)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let source_binding_bytes = source_layout.full_windows.addressed_bytes()?;
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
        let mut batch_source_windows = ModularSourceWindows::default();
        for group in group_grid.ordered_groups() {
            let squeeze = self.config.squeeze.for_extent(group.width, group.height);
            let channels = encoded_channels(format, squeeze, self.config.palette);
            let palette_capacity = self
                .config
                .palette
                .map_or(0, |palette| palette.capacity(group.width, group.height));
            let palette_delta_capacity = self.config.palette.map_or(0, |palette| {
                palette.delta_capacity(group.width, group.height)
            });
            let palette_scratch_bytes = self.config.palette.map_or(0, |palette| {
                4 * palette.scratch_words(
                    palette_capacity,
                    format.channel_count(),
                    group.width,
                    group.height,
                )
            });
            let group_source = source_layout.group(group)?;
            group_source
                .windows
                .validate(self.max_storage_binding_size)?;
            let proposed_source_windows = batch_source_windows.merge(group_source.windows);
            let mut palette_counts_byte_offset = None;
            let layouts = (0..channels)
                .map(|channel| {
                    let [width, height] = if palette_capacity != 0 && channel == 0 {
                        [palette_capacity, format.channel_count()]
                    } else {
                        squeeze.extent(
                            [group.width, group.height],
                            channel - u32::from(palette_capacity != 0),
                            if palette_capacity == 0 {
                                format.channel_count()
                            } else {
                                1
                            },
                        )
                    };
                    let mut layout = ModularChannelLayout::new(width, height, self.config)?;
                    if palette_capacity != 0 && channel == 0 {
                        palette_counts_byte_offset = Some(layout.output_size);
                        layout.output_size = layout
                            .output_size
                            .checked_add(palette_scratch_bytes)
                            .ok_or(EncodeError::InvalidSource("palette scratch size overflow"))?;
                    }
                    Ok(layout)
                })
                .collect::<Result<Vec<_>, EncodeError>>()?;
            let mut proposed_output_size = output_size;
            for layout in &layouts {
                proposed_output_size = align_up(proposed_output_size, artifact_alignment)
                    .and_then(|value| value.checked_add(layout.output_size))
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
                    || proposed_batch_bytes > self.max_storage_binding_size
                    || proposed_source_windows.maximum_bytes() > self.max_storage_binding_size)
            {
                let batch_end_dispatch = parameters.len();
                batches.push(modular_dispatch_batch(
                    batch_first_dispatch..batch_end_dispatch,
                    batch_artifact_offset..output_size,
                    &mut ModularBatchFinalizeContext {
                        source_windows: batch_source_windows,
                        absolute_source_offsets: &absolute_source_offsets,
                        parameters: &mut parameters,
                        max_storage_binding_size: self.max_storage_binding_size,
                    },
                )?);
                batch_first_dispatch = parameters.len();
                batch_source_windows = ModularSourceWindows::default();
                output_size = align_up(output_size, artifact_alignment).ok_or(
                    EncodeError::InvalidSource("artifact batch alignment overflow"),
                )?;
                batch_artifact_offset = output_size;
                proposed_output_size = output_size;
                for layout in &layouts {
                    proposed_output_size = align_up(proposed_output_size, artifact_alignment)
                        .and_then(|value| value.checked_add(layout.output_size))
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
            batch_source_windows = batch_source_windows.merge(group_source.windows);
            let palette_scratch_word_offset = if let Some(offset) = palette_counts_byte_offset {
                align_up(output_size, artifact_alignment)
                    .and_then(|size| size.checked_sub(batch_artifact_offset))
                    .and_then(|size| size.checked_add(offset))
                    .and_then(|size| u32::try_from(size / 4).ok())
                    .ok_or(EncodeError::InvalidSource(
                        "palette scratch offset exceeds WGSL u32 indexing",
                    ))?
            } else {
                0
            };
            for (channel, layout) in layouts.into_iter().enumerate() {
                let ModularChannelLayout {
                    width,
                    height,
                    max_events,
                    output_words,
                    weighted_words,
                    output_size: group_output_size,
                } = layout;
                let channel = channel as u32;
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
                    output_word_offset,
                    channel,
                    channels: format.channel_count(),
                    sample_mask: u32::MAX >> (32 - source_spec.bits_per_sample),
                    rct_type: rct.map_or(42, |rct| rct.rct_type.value()),
                    big_endian: u32::from(source_spec.big_endian),
                    sources: group_source.components,
                    predictor: self.config.predictor.value(),
                    wp_scratch_word_offset: u32::try_from(
                        u64::from(output_word_offset) + output_words as u64,
                    )
                    .map_err(|_| {
                        EncodeError::InvalidSource(
                            "weighted scratch offset exceeds WGSL u32 indexing",
                        )
                    })?,
                    wp_coefficients: self.config.weighted_predictor.coefficients().map(u32::from),
                    wp_max_weights: self.config.weighted_predictor.max_weights().map(u32::from),
                    lz77_mode: self.config.lz77 as u32,
                    lz77_scratch_word_offset: u32::try_from(
                        u64::from(output_word_offset) + output_words as u64 + weighted_words,
                    )
                    .map_err(|_| {
                        EncodeError::InvalidSource("LZ77 scratch offset exceeds WGSL u32 indexing")
                    })?,
                    lz77_hash_mask: self
                        .config
                        .lz77
                        .hash_entries(width * height)
                        .saturating_sub(1),
                    squeeze: squeeze as u32,
                    source_width: group.width,
                    source_height: group.height,
                    palette_capacity,
                    palette_scratch_word_offset,
                    palette_hash_mask: if palette_capacity == 0 {
                        0
                    } else {
                        (palette_capacity * 2).next_power_of_two() - 1
                    },
                    palette_channels: if palette_capacity == 0 { 0 } else { channels },
                    palette_delta_predictor: self
                        .config
                        .palette
                        .and_then(LosslessModularPalette::delta_predictor)
                        .map_or(14, LosslessModularPredictor::value),
                    palette_delta_capacity,
                    _padding: [0; 7],
                });
                groups.push(ModularGroupPlan {
                    group_index: group.index,
                    width,
                    height,
                    channel,
                    artifact_byte_offset: output_size,
                    output_size: group_output_size,
                    max_events,
                    palette_counts_byte_offset: if channel == 0 {
                        palette_counts_byte_offset
                    } else {
                        None
                    },
                    palette_delta_capacity,
                });
                absolute_source_offsets.push(group_source.offsets);
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
                    source_windows: batch_source_windows,
                    absolute_source_offsets: &absolute_source_offsets,
                    parameters: &mut parameters,
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
        let weighted_predictor_scratch_bytes = if self.config.predictor
            == LosslessModularPredictor::Weighted
        {
            batches
                .iter()
                .map(|batch| {
                    parameters[batch.first_dispatch..batch.first_dispatch + batch.dispatch_count]
                        .iter()
                        .map(|params| 20 * u64::from(params.width))
                        .sum::<u64>()
                })
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let lz77_scratch_bytes = batches
            .iter()
            .map(|batch| {
                parameters[batch.first_dispatch..batch.first_dispatch + batch.dispatch_count]
                    .iter()
                    .map(|params| 4 * self.config.lz77.scratch_words(params.width * params.height))
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
        let palette_scratch_bytes = batches
            .iter()
            .map(|batch| {
                parameters[batch.first_dispatch..batch.first_dispatch + batch.dispatch_count]
                    .iter()
                    .filter(|params| params.palette_capacity != 0 && params.channel == 0)
                    .map(|params| {
                        self.config.palette.map_or(0, |palette| {
                            4 * palette.scratch_words(
                                params.palette_capacity,
                                params.channels,
                                params.source_width,
                                params.source_height,
                            )
                        })
                    })
                    .sum::<u64>()
            })
            .max()
            .unwrap_or(0);
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
        let peak_source_binding_bytes = batches.iter().try_fold(0u64, |peak, batch| {
            batch
                .source_windows
                .addressed_bytes()
                .map(|bytes| peak.max(bytes))
        })?;
        let readback_bytes = if self.direct_mapping {
            0
        } else {
            artifact_storage_bytes
        };
        let owned_bytes_per_job = artifact_storage_bytes
            .checked_add(readback_bytes)
            .and_then(|value| value.checked_add(parameter_storage_bytes))
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let icc_profile_bytes = source_spec
            .color
            .icc_profile()
            .map_or(0, |profile| profile.bytes().len() as u64);
        let addressed_bytes_per_job = owned_bytes_per_job
            .checked_add(peak_source_binding_bytes)
            .and_then(|bytes| bytes.checked_add(icc_profile_bytes))
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
            exponent_bits_per_sample: source_spec.exponent_bits_per_sample,
            bytes_per_sample: source_spec.bytes_per_sample,
            channel_count: channels,
            source_binding_bytes,
            peak_source_binding_bytes,
            parameter_storage_bytes,
            artifact_storage_bytes,
            weighted_predictor_scratch_bytes,
            lz77_scratch_bytes,
            palette_scratch_bytes,
            total_artifact_bytes,
            readback_bytes,
            direct_readback: self.direct_mapping,
            batch_count,
            gpu_submission_count,
            streaming,
            icc_profile_bytes,
            icc_storage_bytes: 0,
            owned_bytes_per_job,
            addressed_bytes_per_job,
        };
        Ok(ModularDispatchPlan {
            width: extent.width,
            height: extent.height,
            group_grid,
            format,
            bits_per_sample: source_spec.bits_per_sample,
            exponent_bits_per_sample: source_spec.exponent_bits_per_sample,
            tree_mode: self.config.tree_mode,
            rct,
            squeeze: self.config.squeeze,
            palette: self.config.palette,
            predictor: self.config.predictor,
            weighted_predictor: self.config.weighted_predictor,
            lz77: self.config.lz77,
            parameters,
            groups,
            batches,
            output_size,
            memory,
        })
    }
}
struct ModularBatchFinalizeContext<'a> {
    source_windows: ModularSourceWindows,
    absolute_source_offsets: &'a [[u64; 4]],
    parameters: &'a mut [ModularParams],
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
    context
        .source_windows
        .validate(context.max_storage_binding_size)?;
    for index in dispatches {
        context.source_windows.rebase(
            &mut context.parameters[index],
            context.absolute_source_offsets[index],
        )?;
    }
    Ok(ModularDispatchBatch {
        first_dispatch,
        dispatch_count,
        artifact_byte_offset,
        artifact_binding_size,
        source_windows: context.source_windows,
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
                    "alpha-weighted animation blending requires a GrayAlpha or RGBA source",
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
                sample_bit_depth: modular_sample_depth(
                    plan.bits_per_sample,
                    plan.exponent_bits_per_sample,
                ),
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
        let bind_group_layout = self.pipeline()?.get_bind_group_layout(0);
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
                let [source0, source1, source2, source3] =
                    batch.source_windows.entries(&source.buffer);
                Ok((
                    context
                        .device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("jxl-wgpu lossless modular batch bindings"),
                            layout: &bind_group_layout,
                            entries: &[
                                source0,
                                source1,
                                source2,
                                source3,
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
            pass.set_pipeline(self.pipeline()?);
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
            state: LosslessModularJobState::Resident(Box::new(ResidentLosslessModularJob {
                lifetime: Some(lifetime),
                completion,
                output_size: plan.output_size,
                group_grid: plan.group_grid,
                groups: plan.groups,
                format: plan.format,
                bits_per_sample: plan.bits_per_sample,
                exponent_bits_per_sample: plan.exponent_bits_per_sample,
                tree_mode: plan.tree_mode,
                rct: plan.rct,
                squeeze: plan.squeeze,
                palette: plan.palette,
                predictor: plan.predictor,
                weighted_predictor: plan.weighted_predictor,
                lz77: plan.lz77,
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
            })),
        })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod source_window_tests {
    use super::*;
    use crate::{
        BufferImageSource, CodestreamAssembler, GpuEncodeJob, LosslessModularEncoder,
        ProgressivePlan,
    };
    use jxl_gpu_formats::{ImageLayout, PackingField, PackingWord};
    use jxl_gpu_protocol::Extent2d;
    use wgpu::util::DeviceExt;

    #[test]
    fn every_group_size_checks_artifact_capacity_before_admission() {
        let render = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
            .expect("required GPU adapter");
        let context = WgpuContext::from_backend(&render);
        for group_size in super::super::types::LosslessModularGroupSize::ALL {
            let edge = group_size.dimension();
            let layout = ImageLayout::packed(
                Extent2d::new(edge, edge),
                LosslessModularFormat::Gray.pixel_format(8).unwrap(),
            )
            .unwrap();
            let buffer = context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("full Modular group source"),
                size: layout.logical_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let input = BufferImageSource::new(Arc::new(buffer), layout).unwrap();
            for (lz77, predictor) in [
                (
                    LosslessModularLz77::ZeroRuns,
                    LosslessModularPredictor::Gradient,
                ),
                (
                    LosslessModularLz77::Greedy,
                    LosslessModularPredictor::Gradient,
                ),
                (
                    LosslessModularLz77::Greedy,
                    LosslessModularPredictor::Weighted,
                ),
            ] {
                let mut backend = LosslessModularBackend::with_config(
                    &context,
                    LosslessModularConfig {
                        group_size,
                        lz77,
                        predictor,
                        ..Default::default()
                    },
                );
                let plan = backend.memory_plan(&input).unwrap();
                let expected = 400
                    + 16 * (u64::from(edge) * u64::from(edge)
                        + (u64::from(edge) * u64::from(edge)).div_ceil(8)
                        + 1)
                    + if lz77 == LosslessModularLz77::Greedy {
                        8 * u64::from(edge * edge) + 4 * u64::from((edge * edge).min(65536))
                    } else {
                        0
                    }
                    + if predictor == LosslessModularPredictor::Weighted {
                        20 * u64::from(edge)
                    } else {
                        0
                    };
                assert_eq!(plan.artifact_storage_bytes, expected);
                backend.max_storage_binding_size = expected;
                assert!(backend.memory_plan(&input).is_ok());
                backend.max_storage_binding_size = expected - 1;
                assert!(
                    matches!(backend.memory_plan(&input),Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {name:"max_storage_buffer_binding_size",required,available})) if required == expected && available == expected-1)
                );
                assert_eq!(backend.buffer_pool_stats().allocation_misses, 0);
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }

    #[test]
    fn source_span_alone_splits_gpu_batches_and_rejects_an_oversized_group() {
        let render = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
            .expect("required GPU adapter");
        let context = WgpuContext::from_backend(&render);
        let extent = Extent2d::new(513, 1);
        let mut format = LosslessModularFormat::Gray.pixel_format(8).unwrap();
        format.planes[0].words.extend((0..127).map(|_| PackingWord {
            fields: vec![PackingField::padding(8)],
        }));
        let layout = ImageLayout::packed(extent, format).unwrap();
        let mut bytes = vec![0xa5; layout.logical_size as usize];
        let expected: Vec<u8> = (0..513).map(|x| (x * 71) as u8).collect();
        for (x, &value) in expected.iter().enumerate() {
            bytes[x * 128] = value;
        }
        let buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("source-driven batch splitting"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let input = BufferImageSource::new(Arc::new(buffer), layout).unwrap();
        let mut backend = LosslessModularBackend::new(&context);
        backend.max_storage_binding_size = 40 * 1024;
        let plan = backend.dispatch_plan(&input).unwrap();
        assert_eq!(plan.batches.len(), 2);
        assert!(plan.memory.source_binding_bytes > backend.max_storage_binding_size);
        assert!(
            plan.batches
                .iter()
                .all(|batch| batch.source_windows.maximum_bytes() <= 40 * 1024)
        );
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless {
                sample_bit_depth: plan.memory.sample_bit_depth(),
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::CrossDevice,
            animation: AnimationHeader::Still,
            canvas_width: extent.width,
            canvas_height: extent.height,
            options: FrameOptions::default(),
        };
        let job = backend
            .submit(&context, GpuFrameSource::Buffer(input.clone()), &request)
            .unwrap();
        let mut assembly = CodestreamAssembler::new(
            super::super::serializer::image_header(
                extent.width,
                extent.height,
                LosslessModularFormat::Gray,
                8,
                0,
                AnimationHeader::Still,
                Default::default(),
            )
            .unwrap()
            .finish(context.memory_budget())
            .unwrap()
            .0,
        )
        .unwrap();
        assembly.insert(job.wait().unwrap()).unwrap();
        let encoded = assembly.finish_raw().unwrap();
        let canonical = LosslessModularEncoder::new(context.clone())
            .encode(input.clone())
            .unwrap();
        assert_eq!(encoded, canonical);
        let decoded = jxl_test_support::oracles::modular_integer::original_planes(&encoded, 0);
        assert_eq!(
            decoded,
            vec![expected.into_iter().map(i32::from).collect::<Vec<_>>()]
        );
        assert_eq!(context.memory_stats().reserved_bytes, 0);
        backend.max_storage_binding_size = 4096;
        assert!(matches!(
            backend.memory_plan(&input),
            Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                ..
            }))
        ));
    }

    #[test]
    fn insufficient_storage_bindings_are_typed_before_pipeline_creation() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("required GPU adapter");
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 5,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .unwrap();
        let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
        let backend = LosslessModularBackend::new(&context);
        assert!(matches!(
            backend.pipeline(),
            Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffers_per_shader_stage",
                required: 6,
                available: 5
            }))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, 0);
        assert_eq!(backend.buffer_pool_stats().allocation_misses, 0);
    }
}
