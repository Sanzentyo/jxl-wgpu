use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, OnceLock};

use jxl_gpu_bitstream::{CodestreamInventory, InventoryLimits, ParseLimits};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{KernelVariant, MemoryBudget, MemoryBudgetSnapshot, WgpuBackend};

use crate::buffer_pool::{DecodeBufferPool, WgpuDecodeBufferPoolLimits, WgpuDecodeBufferPoolStats};
use crate::model::native_modular_pixel_format;
use crate::modular_finalize::{
    DEFAULT_MODULAR_FINALIZE_VARIANT, MODULAR_FINALIZE_KERNEL_KEY, ModularFinalizeF64Path,
    ModularFinalizeParams, ModularFinalizePipeline,
};
use crate::modular_inverse::ModularInverseJob;
use crate::modular_palette::{
    DEFAULT_MODULAR_PALETTE_VARIANT, MODULAR_PALETTE_KERNEL_KEY, ModularPalettePipeline,
};
use crate::modular_rct::{DEFAULT_MODULAR_RCT_VARIANT, MODULAR_RCT_KERNEL_KEY, ModularRctPipeline};
use crate::modular_squeeze::ModularSqueezePipeline;
use crate::modular_tree::MaTreeNodeIr;
use crate::profile::{
    StandardModularProfile, parse_progressive_dc_modular_profile, parse_standard_modular_profile,
};
use crate::progressive_dc::ProgressiveDcPipeline;
use crate::{
    AnimationMetadata, DecodeProfile, Error, GpuCodestream, GpuOutputRequest, GpuSubmissionEngine,
    ModularPredictionProfile, PreparedGpuSession, Result,
};

use super::execution::{
    DeviceAdmissionOptions, GroupDispatchLayout, GroupDispatchOptions, ModularMetadataInventory,
    OutputPlan, modular_finalize_params, modular_frame_finalize_params, validate_device_limits,
};
use super::lifetime::DecodeSource;
use super::session::WgpuDecodeSession;
use super::types::{
    ATOMIC_OUTPUT_WORDS_TYPE, ATOMIC_WRITE_BYTE_WORD, ATOMIC_WRITE_FULL_WORD,
    DEFAULT_MODULAR_GROUP_VARIANT, DecodePipelineCache, F64_BINDING_MARKER, F64_EXACT_F32_WIDENING,
    F64_NATIVE_ARITHMETIC, F64_NATIVE_BINDING, F64_OUTPUT_MARKER, F64OutputPath,
    MODULAR_ENTROPY_ABI_MARKER, MODULAR_ENTROPY_ABI_SHADER, MODULAR_ENTROPY_MARKER,
    MODULAR_ENTROPY_SHADER, MODULAR_FIXED_GRADIENT_SHADER, MODULAR_RECONSTRUCT_MARKER,
    MODULAR_RECONSTRUCT_SHADER, MODULAR_RESUME_MARKER, MODULAR_RESUME_SHADER,
    ModularInversePipelineCache, ModularInversePipelines, ModularReconstructionSpecialization,
    OUTPUT_WORDS_TYPE_MARKER, OutputWritePath, SHADER_TEMPLATE, WORD_ALIGNED_OUTPUT_WORDS_TYPE,
    WORD_ALIGNED_WRITE_BYTE_WORD, WORD_ALIGNED_WRITE_FULL_WORD, WRITE_BYTE_WORD_MARKER,
    WRITE_FULL_WORD_MARKER, WgpuDecodeCapabilities, WgpuSubmissionEngine,
};
use crate::ModularPredictor;
pub(super) fn reconstruction_specialization(
    profile: &StandardModularProfile,
) -> ModularReconstructionSpecialization {
    if uses_generalized_channel_layout(profile) {
        return ModularReconstructionSpecialization::DescriptorMetaAdaptive;
    }
    let mut candidates = profile.resident_entropy_plans.iter().map(|plan| {
        let ma_config = plan.ma_config.resolve(&profile.ma_config);
        channel_fixed_gradient_specialization(
            &ma_config.nodes,
            profile.channels.count(),
            ma_config.needs_self_correcting(),
        )
    });
    let Some(first) = candidates.next() else {
        return ModularReconstructionSpecialization::GenericMetaAdaptive;
    };
    if candidates.all(|candidate| candidate == first) {
        first
    } else {
        ModularReconstructionSpecialization::GenericMetaAdaptive
    }
}

pub(super) fn uses_generalized_channel_layout(profile: &StandardModularProfile) -> bool {
    profile.generalized_channels
}

pub(super) fn channel_fixed_gradient_specialization(
    nodes: &[MaTreeNodeIr],
    source_channels: u32,
    needs_self_correcting: bool,
) -> ModularReconstructionSpecialization {
    let generic = ModularReconstructionSpecialization::GenericMetaAdaptive;
    let Ok(channel_count) = u8::try_from(source_channels) else {
        return generic;
    };
    if !(1..=4).contains(&channel_count)
        || needs_self_correcting
        || nodes.is_empty()
        || nodes.iter().any(|node| {
            matches!(
                node,
                MaTreeNodeIr::Decision {
                    property,
                    ..
                } if *property != 0
            )
        })
    {
        return generic;
    }
    let mut clusters = [0u8; 4];
    for (channel, cluster) in [0i32, 1, 2, 3].into_iter().zip(clusters.iter_mut()) {
        let mut node_index = 0u32;
        let mut leaf = None;
        for _ in 0..nodes.len() {
            let Some(node) = usize::try_from(node_index)
                .ok()
                .and_then(|index| nodes.get(index))
            else {
                break;
            };
            match *node {
                MaTreeNodeIr::Decision {
                    property: 0,
                    threshold,
                    left,
                    right,
                } => {
                    node_index = if channel > threshold { left } else { right };
                }
                MaTreeNodeIr::Leaf {
                    cluster,
                    predictor: 5,
                    offset: 0,
                    multiplier: 1,
                } => {
                    leaf = Some(cluster);
                    break;
                }
                _ => break,
            }
        }
        let Some(resolved) = leaf else {
            return generic;
        };
        *cluster = resolved;
    }
    ModularReconstructionSpecialization::ChannelFixed {
        predictor: ModularPredictor::Gradient,
        offset: 0,
        multiplier: 1,
        channel_count,
        clusters,
    }
}

impl std::fmt::Debug for WgpuSubmissionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuSubmissionEngine")
            .field("backend", &self.backend)
            .field("memory", &self.memory.snapshot())
            .field("buffer_pool", &self.buffers.stats())
            .field("stream_window_limit", &self.stream_window_limit)
            .finish_non_exhaustive()
    }
}

impl WgpuSubmissionEngine {
    #[must_use]
    pub fn new(backend: WgpuBackend) -> Self {
        let memory_budget = backend.transient_memory_budget().clone();
        Self::with_memory_budget(backend, memory_budget)
    }

    /// Constructs an engine with an explicitly supplied aggregate reservation budget.
    ///
    /// Passing a clone of another component's [`MemoryBudget`] makes decode jobs participate in
    /// that component's admission bound. [`Self::new`] uses the backend-wide transient budget and
    /// is the normal constructor.
    #[must_use]
    pub fn with_memory_budget(backend: WgpuBackend, memory_budget: MemoryBudget) -> Self {
        // Pipelines are selected after the output layout is validated. Lazy caches avoid
        // compiling the atomic fallback or native-f64 variants when a workload never needs them.
        let pipelines = Arc::new(DecodePipelineCache::default());
        let native_f64_pipelines = backend
            .native_f64_enabled()
            .then(|| Arc::new(DecodePipelineCache::default()));
        let buffers = DecodeBufferPool::new(
            backend.device().clone(),
            WgpuDecodeBufferPoolLimits::default(),
        );
        Self {
            backend,
            pipelines,
            native_f64_pipelines,
            inverse_pipelines: Arc::new(ModularInversePipelineCache::default()),
            progressive_dc_pipeline: Arc::new(OnceLock::new()),
            memory: memory_budget,
            buffers,
            stream_window_limit: None,
        }
    }

    /// Caps the reusable GPU upload used for Modular entropy streams.
    ///
    /// A legal channel-fixed Gradient group larger than this bound is split into ordered,
    /// overlapping windows while its entropy and LZ77 state remains GPU-resident. Device limits
    /// and the shared byte budget may resolve a smaller effective bound. An undersized value is
    /// rejected with [`Error::StreamWindowTooSmall`] when a session is opened.
    #[must_use]
    pub fn with_stream_window_limit(mut self, limit: NonZeroU64) -> Self {
        self.stream_window_limit = Some(limit);
        self
    }

    /// Returns the caller-supplied Modular stream-window cap, if one was configured.
    #[must_use]
    pub const fn stream_window_limit(&self) -> Option<NonZeroU64> {
        self.stream_window_limit
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        &self.backend
    }

    #[must_use]
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory.snapshot().limit_bytes
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    #[must_use]
    pub fn capabilities(&self) -> WgpuDecodeCapabilities {
        WgpuDecodeCapabilities {
            native_f64_arithmetic: self.native_f64_pipelines.is_some(),
        }
    }

    /// Current idle-cache limits. Active allocations remain governed by `MemoryBudget`.
    #[must_use]
    pub fn buffer_pool_limits(&self) -> WgpuDecodeBufferPoolLimits {
        self.buffers.limits()
    }

    /// Changes idle-cache limits and immediately evicts allocations outside the new bounds.
    pub fn set_buffer_pool_limits(&self, limits: WgpuDecodeBufferPoolLimits) {
        self.buffers.set_limits(limits);
    }

    /// Drops every idle allocation and invalidates all currently leased pool generations.
    ///
    /// In-flight jobs remain valid. Their transient buffers are discarded, rather than cached,
    /// after GPU completion. The returned generation is also published in
    /// [`Self::buffer_pool_stats`].
    pub fn clear_buffer_pool(&self) -> u64 {
        self.buffers.clear()
    }

    /// Reports physical idle/leased buffer reuse independently from logical byte admission.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> WgpuDecodeBufferPoolStats {
        self.buffers.stats()
    }

    pub(crate) fn open_with_inventory(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> Result<PreparedGpuSession<WgpuDecodeSession>> {
        let codestream = Arc::new(codestream);
        self.open_with_inventory_data(codestream, request, inventory)
    }

    pub(crate) fn open_with_inventory_data(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> Result<PreparedGpuSession<WgpuDecodeSession>> {
        let profile = parse_standard_modular_profile(&codestream, inventory)?;
        self.open_profile(codestream, request, profile)
    }

    pub(crate) fn open_progressive_dc_with_inventory_data(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> Result<PreparedGpuSession<WgpuDecodeSession>> {
        let profile = parse_progressive_dc_modular_profile(&codestream, inventory)?;
        let internal_request = GpuOutputRequest::color(native_modular_pixel_format(
            profile.channels,
            profile.bits_per_sample,
        )?)?
        .with_max_frame_slots(request.max_frame_slots());
        self.open_profile(codestream, &internal_request, profile)
    }

    fn open_profile(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        profile: StandardModularProfile,
    ) -> Result<PreparedGpuSession<WgpuDecodeSession>> {
        if profile.resident_entropy_plans.len() != profile.entropy_groups.len() {
            return Err(Error::EngineContract(
                "Modular entropy plans do not match the LF/pass stream inventory",
            ));
        }
        let mut modular_metadata = Vec::new();
        let global_metadata_offset = profile
            .ma_config
            .pack_gpu_metadata()?
            .append_to(&mut modular_metadata)?;
        if global_metadata_offset != 0 {
            return Err(Error::EngineContract(
                "global Modular metadata did not start at word zero",
            ));
        }
        let mut unique_local_metadata = Vec::<(crate::modular_tree::MaConfigIr, u32)>::new();
        let ma_metadata_offsets: Arc<[u32]> = profile
            .resident_entropy_plans
            .iter()
            .map(|plan| match &plan.ma_config {
                crate::profile::ModularMaConfig::Global => Ok::<u32, Error>(0),
                crate::profile::ModularMaConfig::Local(local) => {
                    if let Some((_, offset)) = unique_local_metadata
                        .iter()
                        .find(|(candidate, _)| candidate == local)
                    {
                        return Ok(*offset);
                    }
                    let offset = local
                        .pack_gpu_metadata()?
                        .append_to(&mut modular_metadata)?;
                    unique_local_metadata.push((local.clone(), offset));
                    Ok(offset)
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into();
        let global_ma_metadata_offset = profile
            .resident_frame_plan
            .as_ref()
            .map(|plan| match &plan.ma_config {
                crate::profile::ModularMaConfig::Global => Ok::<u32, Error>(0),
                crate::profile::ModularMaConfig::Local(local) => {
                    if let Some((_, offset)) = unique_local_metadata
                        .iter()
                        .find(|(candidate, _)| candidate == local)
                    {
                        return Ok(*offset);
                    }
                    let offset = local
                        .pack_gpu_metadata()?
                        .append_to(&mut modular_metadata)?;
                    unique_local_metadata.push((local.clone(), offset));
                    Ok(offset)
                }
            })
            .transpose()?;
        let local_ma_stream_count = ma_metadata_offsets
            .iter()
            .filter(|&&offset| offset != global_metadata_offset)
            .count();
        let unique_ma_config_count = unique_local_metadata
            .len()
            .checked_add(1)
            .ok_or_else(|| Error::backend("Modular MA configuration count overflow"))?;
        let metadata_inventory = ModularMetadataInventory {
            local_ma_stream_count,
            unique_ma_config_count,
        };
        let generalized_channels = uses_generalized_channel_layout(&profile);
        let channel_layout_offsets: Arc<[u32]> = if generalized_channels {
            let mut unique = Vec::<(usize, u32)>::new();
            let mut offsets = Vec::with_capacity(profile.resident_entropy_plans.len());
            for (group_index, plan) in profile.resident_entropy_plans.iter().enumerate() {
                if let Some((_, offset)) = unique.iter().find(|(candidate_index, _)| {
                    let candidate = &profile.resident_entropy_plans[*candidate_index];
                    candidate.channel_metadata == plan.channel_metadata
                        && candidate.inverse_plan == plan.inverse_plan
                }) {
                    offsets.push(*offset);
                    continue;
                }
                let offset = plan.channel_metadata.append_to(
                    &mut modular_metadata,
                    plan.inverse_plan.arena_words(),
                    &plan.inverse_plan.final_gpu_layouts(),
                )?;
                unique.push((group_index, offset));
                offsets.push(offset);
            }
            offsets.into()
        } else {
            Arc::from([])
        };
        let global_channel_layout_offset = profile
            .resident_frame_plan
            .as_ref()
            .map(|plan| {
                plan.channel_metadata.append_to(
                    &mut modular_metadata,
                    plan.inverse_plan.arena_words(),
                    &[],
                )
            })
            .transpose()?;
        let modular_metadata: Arc<[u32]> = modular_metadata.into();
        let extent = Extent2d::new(profile.width, profile.height);
        let output = OutputPlan::new(
            extent,
            request,
            profile.channels,
            profile.bits_per_sample,
            self.capabilities(),
        )?;
        let output_write_path = if generalized_channels {
            OutputWritePath::AtomicBytes
        } else {
            output.write_path_for_groups(&profile.groups)?
        };
        let reconstruction_specialization = reconstruction_specialization(&profile);
        let kernel_variant = self
            .backend
            .kernel_policy()
            .variant_for("lossless_gray8", DEFAULT_MODULAR_GROUP_VARIANT)?;
        kernel_variant.validate_for("lossless_gray8", &self.backend.device().limits(), 0)?;
        let (pipelines, pipeline_f64_path) = match output.f64_output_path {
            Some(F64OutputPath::NativeArithmetic) => (
                self.native_f64_pipelines
                    .as_ref()
                    .ok_or(Error::NativeF64Unavailable)?,
                F64OutputPath::NativeArithmetic,
            ),
            Some(F64OutputPath::ExactF32Widening) | None => {
                (&self.pipelines, F64OutputPath::ExactF32Widening)
            }
        };
        let inverse_pipelines = generalized_channels
            .then(|| {
                let needs_squeeze = profile
                    .resident_entropy_plans
                    .iter()
                    .flat_map(|plan| plan.inverse_plan.jobs())
                    .chain(
                        profile
                            .resident_frame_plan
                            .iter()
                            .flat_map(|plan| plan.inverse_plan.jobs()),
                    )
                    .any(|job| matches!(job, ModularInverseJob::Squeeze { .. }));
                let needs_rct = profile
                    .resident_entropy_plans
                    .iter()
                    .flat_map(|plan| plan.inverse_plan.jobs())
                    .chain(
                        profile
                            .resident_frame_plan
                            .iter()
                            .flat_map(|plan| plan.inverse_plan.jobs()),
                    )
                    .any(|job| matches!(job, ModularInverseJob::Rct { .. }));
                let needs_palette = profile
                    .resident_entropy_plans
                    .iter()
                    .flat_map(|plan| plan.inverse_plan.jobs())
                    .chain(
                        profile
                            .resident_frame_plan
                            .iter()
                            .flat_map(|plan| plan.inverse_plan.jobs()),
                    )
                    .any(|job| matches!(job, ModularInverseJob::Palette { .. }));
                self.inverse_pipelines.get(
                    &self.backend,
                    pipeline_f64_path,
                    needs_palette,
                    needs_squeeze,
                    needs_rct,
                )
            })
            .transpose()?;
        let finalize_params: Arc<[ModularFinalizeParams]> = if profile.progressive_dc.is_some() {
            Arc::from([])
        } else if let Some(frame_plan) = &profile.resident_frame_plan {
            vec![modular_frame_finalize_params(
                &profile, &output, frame_plan,
            )?]
            .into()
        } else if generalized_channels {
            profile
                .groups
                .iter()
                .enumerate()
                .map(|(group_index, group)| {
                    modular_finalize_params(&profile, &output, group_index, *group)
                })
                .collect::<Result<Vec<_>>>()?
                .into()
        } else {
            Arc::from([])
        };
        let pipeline = pipelines.get_or_init(
            &self.backend,
            pipeline_f64_path,
            output_write_path,
            reconstruction_specialization,
            kernel_variant,
        );
        let f64_output_path = output.f64_output_path;
        let memory_limit_bytes = self.memory.snapshot().limit_bytes;
        let dispatch_layout = GroupDispatchLayout::new(
            self.backend.device(),
            codestream.logical_bytes(),
            &profile,
            &modular_metadata,
            &output,
            GroupDispatchOptions {
                requested_frame_slots: request.max_frame_slots().get(),
                memory_limit_bytes,
                kernel_variant,
                stream_window_limit: self.stream_window_limit,
            },
        )?;
        let memory_stats = validate_device_limits(
            self.backend.device(),
            &modular_metadata,
            metadata_inventory,
            &dispatch_layout,
            &output,
            DeviceAdmissionOptions {
                requested_frame_slots: request.max_frame_slots().get(),
                memory_limit_bytes,
                progressive_dc: profile.progressive_dc.is_some(),
            },
        )?;
        let progressive_dc_pipeline = profile
            .progressive_dc
            .map(|_| {
                match self.progressive_dc_pipeline.get_or_init(|| {
                    ProgressiveDcPipeline::with_policy(
                        self.backend.device(),
                        self.backend.kernel_policy(),
                    )
                    .map(Arc::new)
                }) {
                    Ok(pipeline) => Ok(Arc::clone(pipeline)),
                    Err(error) => Err(error.clone()),
                }
            })
            .transpose()?;
        let resolved_frame_slots = NonZeroUsize::new(memory_stats.max_frame_slots)
            .expect("device admission always resolves at least one frame slot");
        let reported_ma_config = profile
            .resident_entropy_plans
            .iter()
            .map(|plan| plan.ma_config.resolve(&profile.ma_config))
            .chain(
                profile
                    .resident_frame_plan
                    .iter()
                    .map(|plan| plan.ma_config.resolve(&profile.ma_config)),
            )
            .max_by_key(|config| {
                (
                    config.nodes.len(),
                    config.max_depth,
                    config.needs_self_correcting(),
                )
            })
            .ok_or(Error::EngineContract(
                "Modular frame has no group MA configuration",
            ))?;
        let node_count = u32::try_from(reported_ma_config.nodes.len())
            .map_err(|_| Error::backend("MA tree node count exceeds public profile bounds"))?;
        let decision_node_count = u32::try_from(
            reported_ma_config
                .nodes
                .iter()
                .filter(|node| matches!(node, MaTreeNodeIr::Decision { .. }))
                .count(),
        )
        .map_err(|_| Error::backend("MA decision node count exceeds public profile bounds"))?;
        let leaf_context_count = node_count
            .checked_sub(decision_node_count)
            .ok_or_else(|| Error::backend("MA leaf context count underflow"))?;
        let max_depth = u32::try_from(reported_ma_config.max_depth)
            .map_err(|_| Error::backend("MA tree depth exceeds public profile bounds"))?;
        let prediction = ModularPredictionProfile::MetaAdaptive {
            node_count,
            decision_node_count,
            leaf_context_count,
            max_depth,
            uses_self_correcting: reported_ma_config.needs_self_correcting(),
        };
        Ok(PreparedGpuSession::new(
            DecodeProfile::ModularLossless {
                bits_per_sample: profile.bits_per_sample,
                channels: profile.channels,
                prediction,
                grouping: if profile.groups.len() == 1 {
                    crate::ModularGrouping::SingleGroup
                } else {
                    crate::ModularGrouping::MultipleGroups {
                        columns: profile.group_columns,
                        rows: profile.group_rows,
                    }
                },
                passes: profile.pass_count,
            },
            AnimationMetadata::still(extent),
            WgpuDecodeSession {
                backend: self.backend.clone(),
                pipeline,
                source: Some(DecodeSource {
                    codestream,
                    profile,
                    dispatch_layout,
                    modular_metadata,
                    ma_metadata_offsets,
                    global_ma_metadata_offset,
                    channel_layout_offsets,
                    global_channel_layout_offset,
                    finalize_params,
                    output,
                }),
                memory_stats,
                memory_budget: self.memory.clone(),
                buffers: Arc::clone(&self.buffers),
                f64_output_path,
                inverse_pipelines,
                progressive_dc_pipeline,
            },
        )
        .with_resolved_frame_slots(resolved_frame_slots))
    }
}

impl ModularInversePipelineCache {
    pub(super) fn get(
        &self,
        backend: &WgpuBackend,
        f64_path: F64OutputPath,
        needs_palette: bool,
        needs_squeeze: bool,
        needs_rct: bool,
    ) -> Result<Arc<ModularInversePipelines>> {
        let palette = if needs_palette {
            let variant = backend
                .kernel_policy()
                .variant_for(MODULAR_PALETTE_KERNEL_KEY, DEFAULT_MODULAR_PALETTE_VARIANT)?;
            Some(
                match self.palette.get_or_init(|| {
                    ModularPalettePipeline::with_variant(backend.device(), variant).map(Arc::new)
                }) {
                    Ok(pipeline) => Arc::clone(pipeline),
                    Err(error) => return Err(error.clone().into()),
                },
            )
        } else {
            None
        };
        let squeeze = if needs_squeeze {
            let variant = backend
                .kernel_policy()
                .variant_for("modular_squeeze", KernelVariant::Lanes64)?;
            Some(
                match self.squeeze.get_or_init(|| {
                    ModularSqueezePipeline::with_variant(backend.device(), variant).map(Arc::new)
                }) {
                    Ok(pipeline) => Arc::clone(pipeline),
                    Err(error) => {
                        return Err(crate::ModularInversePlanError::Squeeze(error.clone()).into());
                    }
                },
            )
        } else {
            None
        };
        let rct = if needs_rct {
            let variant = backend
                .kernel_policy()
                .variant_for(MODULAR_RCT_KERNEL_KEY, DEFAULT_MODULAR_RCT_VARIANT)?;
            Some(
                match self.rct.get_or_init(|| {
                    ModularRctPipeline::with_variant(backend.device(), variant).map(Arc::new)
                }) {
                    Ok(pipeline) => Arc::clone(pipeline),
                    Err(error) => {
                        return Err(crate::ModularInversePlanError::Rct(error.clone()).into());
                    }
                },
            )
        } else {
            None
        };
        let finalize_variant = backend.kernel_policy().variant_for(
            MODULAR_FINALIZE_KERNEL_KEY,
            DEFAULT_MODULAR_FINALIZE_VARIANT,
        )?;
        let (finalize_cache, finalize_path) = match f64_path {
            F64OutputPath::NativeArithmetic => (
                &self.finalize_native,
                ModularFinalizeF64Path::NativeArithmetic,
            ),
            F64OutputPath::ExactF32Widening => (
                &self.finalize_exact,
                ModularFinalizeF64Path::ExactF32Widening,
            ),
        };
        let finalize = match finalize_cache.get_or_init(|| {
            ModularFinalizePipeline::with_variant(backend.device(), finalize_variant, finalize_path)
                .map(Arc::new)
        }) {
            Ok(pipeline) => Arc::clone(pipeline),
            Err(error) => return Err(error.clone().into()),
        };
        Ok(Arc::new(ModularInversePipelines {
            palette,
            squeeze,
            rct,
            finalize,
        }))
    }
}

impl DecodePipelineCache {
    fn get_or_init(
        &self,
        backend: &WgpuBackend,
        f64_path: F64OutputPath,
        output_write_path: OutputWritePath,
        reconstruction: ModularReconstructionSpecialization,
        variant: KernelVariant,
    ) -> Arc<wgpu::ComputePipeline> {
        let fixed_gradient = matches!(
            reconstruction,
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                ..
            }
        );
        let descriptor = matches!(
            reconstruction,
            ModularReconstructionSpecialization::DescriptorMetaAdaptive
        );
        let pipeline = match (fixed_gradient, descriptor, output_write_path) {
            (false, false, OutputWritePath::AtomicBytes) => &self.generic_atomic,
            (false, false, OutputWritePath::WordAligned) => &self.generic_word_aligned,
            (false, true, OutputWritePath::AtomicBytes) => &self.descriptor_atomic,
            (false, true, OutputWritePath::WordAligned) => &self.descriptor_word_aligned,
            (true, false, OutputWritePath::AtomicBytes) => &self.fixed_gradient_atomic,
            (true, false, OutputWritePath::WordAligned) => &self.fixed_gradient_word_aligned,
            (true, true, _) => unreachable!("descriptor reconstruction is never channel-fixed"),
        };
        Arc::clone(pipeline.get_or_init(|| {
            let label = match (f64_path, output_write_path, fixed_gradient, descriptor) {
                (F64OutputPath::ExactF32Widening, OutputWritePath::AtomicBytes, false, false) => {
                    "jxl-wgpu decode generic Modular atomic output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::WordAligned, false, false) => {
                    "jxl-wgpu decode generic Modular word-aligned output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::AtomicBytes, false, false) => {
                    "jxl-wgpu decode generic Modular native-f64 atomic output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::WordAligned, false, false) => {
                    "jxl-wgpu decode generic Modular native-f64 word-aligned output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::AtomicBytes, false, true) => {
                    "jxl-wgpu decode descriptor Modular atomic arena"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::WordAligned, false, true) => {
                    "jxl-wgpu decode descriptor Modular word-aligned arena"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::AtomicBytes, false, true) => {
                    "jxl-wgpu decode descriptor Modular native-f64 atomic arena"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::WordAligned, false, true) => {
                    "jxl-wgpu decode descriptor Modular native-f64 word-aligned arena"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::AtomicBytes, true, false) => {
                    "jxl-wgpu decode fixed-Gradient Modular atomic output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::WordAligned, true, false) => {
                    "jxl-wgpu decode fixed-Gradient Modular word-aligned output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::AtomicBytes, true, false) => {
                    "jxl-wgpu decode fixed-Gradient Modular native-f64 atomic output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::WordAligned, true, false) => {
                    "jxl-wgpu decode fixed-Gradient Modular native-f64 word-aligned output"
                }
                (_, _, true, true) => {
                    unreachable!("descriptor reconstruction is never channel-fixed")
                }
            };
            Arc::new(create_decode_pipeline(
                backend,
                label,
                &shader_source(f64_path, output_write_path, reconstruction),
                variant,
                descriptor,
            ))
        }))
    }
}

pub(super) fn shader_source(
    path: F64OutputPath,
    output_write_path: OutputWritePath,
    reconstruction: ModularReconstructionSpecialization,
) -> String {
    let (implementation, binding) = match path {
        F64OutputPath::NativeArithmetic => (F64_NATIVE_ARITHMETIC, F64_NATIVE_BINDING),
        F64OutputPath::ExactF32Widening => (F64_EXACT_F32_WIDENING, ""),
    };
    let (output_words_type, write_byte_word, write_full_word) = match output_write_path {
        OutputWritePath::AtomicBytes => (
            ATOMIC_OUTPUT_WORDS_TYPE,
            ATOMIC_WRITE_BYTE_WORD,
            ATOMIC_WRITE_FULL_WORD,
        ),
        OutputWritePath::WordAligned => (
            WORD_ALIGNED_OUTPUT_WORDS_TYPE,
            WORD_ALIGNED_WRITE_BYTE_WORD,
            WORD_ALIGNED_WRITE_FULL_WORD,
        ),
    };
    let reconstruction_shader = match reconstruction {
        ModularReconstructionSpecialization::GenericMetaAdaptive
        | ModularReconstructionSpecialization::DescriptorMetaAdaptive => MODULAR_RECONSTRUCT_SHADER,
        ModularReconstructionSpecialization::ChannelFixed {
            predictor: ModularPredictor::Gradient,
            ..
        } => MODULAR_FIXED_GRADIENT_SHADER,
        ModularReconstructionSpecialization::ChannelFixed { .. } => MODULAR_RECONSTRUCT_SHADER,
    };
    let resume_shader = match reconstruction {
        ModularReconstructionSpecialization::GenericMetaAdaptive
        | ModularReconstructionSpecialization::DescriptorMetaAdaptive => MODULAR_RESUME_SHADER,
        ModularReconstructionSpecialization::ChannelFixed { .. } => "",
    };
    SHADER_TEMPLATE
        .replace(MODULAR_ENTROPY_ABI_MARKER, MODULAR_ENTROPY_ABI_SHADER)
        .replace(MODULAR_ENTROPY_MARKER, MODULAR_ENTROPY_SHADER)
        .replace(MODULAR_RESUME_MARKER, resume_shader)
        .replace(MODULAR_RECONSTRUCT_MARKER, reconstruction_shader)
        .replace(F64_OUTPUT_MARKER, implementation)
        .replace(F64_BINDING_MARKER, binding)
        .replace(OUTPUT_WORDS_TYPE_MARKER, output_words_type)
        .replace(WRITE_BYTE_WORD_MARKER, write_byte_word)
        .replace(WRITE_FULL_WORD_MARKER, write_full_word)
}

pub(super) fn create_decode_pipeline(
    backend: &WgpuBackend,
    label: &str,
    shader: &str,
    variant: KernelVariant,
    descriptor_channels: bool,
) -> wgpu::ComputePipeline {
    let module = backend
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
    let (workgroup_x, workgroup_y) = variant.workgroup_size();
    let constants = [
        ("wg_x", f64::from(workgroup_x)),
        ("wg_y", f64::from(workgroup_y)),
        ("descriptor_channels", f64::from(descriptor_channels)),
    ];
    backend
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("decode"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            },
            cache: None,
        })
}

impl GpuSubmissionEngine for WgpuSubmissionEngine {
    type Session = WgpuDecodeSession;

    fn parse_limits(&self) -> ParseLimits {
        let policy_ceiling = ParseLimits::default().max_codestream_bytes;
        let budget_scaled = self.memory.snapshot().limit_bytes.saturating_mul(4);
        let device_scaled = self
            .backend
            .device()
            .limits()
            .max_buffer_size
            .saturating_mul(4);
        let host_codestream_limit = policy_ceiling.min(budget_scaled.max(device_scaled));
        ParseLimits {
            max_input_bytes: host_codestream_limit,
            max_boxes: 32,
            max_box_bytes: host_codestream_limit,
            max_codestream_bytes: host_codestream_limit,
        }
    }

    fn inventory_limits(&self) -> InventoryLimits {
        InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: self.parse_limits().max_codestream_bytes,
            ..InventoryLimits::default()
        }
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        self.open_with_inventory(codestream, request, &inventory)
    }
}
