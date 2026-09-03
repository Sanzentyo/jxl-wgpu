use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::{Arc, atomic::AtomicUsize};

use jxl_gpu_bitstream::{CodestreamInventory, InventoryLimits, ParseLimits};
use jxl_wgpu::{
    KernelVariant, MemoryBudget, MemoryBudgetSnapshot, ResidentChromaUpsamplePipeline,
    ResidentEpfPipeline, ResidentGaborishPipeline, ResidentImageUpsamplePipeline,
    ResidentVarDctRenderer, WgpuBackend,
};

use crate::progressive_dc::ProgressiveDcPipeline;
use crate::vardct_artifact::HfMetadataLoweringPipeline;
use crate::vardct_epf::EpfSigmaPipeline;
use crate::vardct_lf::AdaptiveLfPipeline;
use crate::vardct_output::VarDctOutputPacker;
use crate::vardct_packet::VarDctPacketPipeline;
use crate::vardct_pass_group::HfCoefficientPipeline;
use crate::vardct_resource::VarDctResourcePipeline;
use crate::wgpu_engine::RawHfDequantSideImagePipeline;
use crate::{
    AnimationMetadata, DecodeProfile, GpuCodestream, GpuOutputRequest, GpuSubmissionEngine,
    PreparedGpuSession, Result as DecodeResult,
};

use super::execution::{VarDctDecodeSession, VarDctRuntimeStats};
use super::source::{VarDctPrepareOptions, VarDctSource, prepare_source};
use super::types::{SubsampledAdaptiveLfPolicy, VAR_DCT_PARSE_LIMIT_BYTES, VarDctDecodeError};

pub(super) struct VarDctPipelines {
    pub(super) packet: VarDctPacketPipeline,
    pub(super) resource: VarDctResourcePipeline,
    pub(super) adaptive_lf: AdaptiveLfPipeline,
    pub(super) artifact: HfMetadataLoweringPipeline,
    pub(super) hf_coefficients: HfCoefficientPipeline,
    pub(super) renderer: ResidentVarDctRenderer,
    pub(super) chroma_upsample: ResidentChromaUpsamplePipeline,
    pub(super) image_upsample: ResidentImageUpsamplePipeline,
    pub(super) gaborish: ResidentGaborishPipeline,
    pub(super) epf_sigma: EpfSigmaPipeline,
    pub(super) epf: ResidentEpfPipeline,
    pub(super) output: VarDctOutputPacker,
    pub(super) progressive_dc: ProgressiveDcPipeline,
    pub(super) raw_hf_dequant: RawHfDequantSideImagePipeline,
    pub(super) output_variant: KernelVariant,
}

impl VarDctPipelines {
    pub(super) fn new(backend: &WgpuBackend) -> Result<Self, VarDctDecodeError> {
        let resource_variant =
            resolve_kernel_variant(backend, "vardct_resource", KernelVariant::Lanes64)?;
        let output_variant =
            resolve_kernel_variant(backend, "vardct_output", KernelVariant::Lanes256)?;
        let gaborish_variant =
            resolve_kernel_variant(backend, "vardct_gaborish", KernelVariant::Tile16x16)?;
        let chroma_upsample_variant =
            resolve_kernel_variant(backend, "vardct_chroma_upsample", KernelVariant::Tile16x16)?;
        let image_upsample_variant =
            resolve_kernel_variant(backend, "vardct_image_upsample", KernelVariant::Tile16x16)?;
        let epf_sigma_variant =
            resolve_kernel_variant(backend, "vardct_epf_sigma", KernelVariant::Lanes64)?;
        let epf_variant = resolve_kernel_variant(backend, "vardct_epf", KernelVariant::Tile16x16)?;
        let raw_hf_dequant_variant =
            resolve_kernel_variant(backend, "vardct_raw_matrix", KernelVariant::Lanes64)?;
        if raw_hf_dequant_variant.workgroup_size().1 != 1 {
            return Err(VarDctDecodeError::KernelPolicy {
                kernel: "vardct_raw_matrix",
                message: "raw matrix decode requires a linear workgroup".to_owned(),
            });
        }
        let device = backend.device();
        Ok(Self {
            packet: VarDctPacketPipeline::new(device),
            resource: VarDctResourcePipeline::with_variant(device, resource_variant)?,
            adaptive_lf: AdaptiveLfPipeline::new(device),
            artifact: HfMetadataLoweringPipeline::new(device),
            hf_coefficients: HfCoefficientPipeline::new(device),
            renderer: ResidentVarDctRenderer::new(device),
            chroma_upsample: ResidentChromaUpsamplePipeline::with_variant(
                device,
                chroma_upsample_variant,
            )?,
            image_upsample: ResidentImageUpsamplePipeline::with_variant(
                device,
                image_upsample_variant,
            )?,
            gaborish: ResidentGaborishPipeline::with_variant(device, gaborish_variant)?,
            epf_sigma: EpfSigmaPipeline::with_variant(device, epf_sigma_variant)?,
            epf: ResidentEpfPipeline::with_variant(device, epf_variant)?,
            output: VarDctOutputPacker::with_variant(device, output_variant)?,
            progressive_dc: ProgressiveDcPipeline::with_policy(device, backend.kernel_policy())?,
            raw_hf_dequant: RawHfDequantSideImagePipeline::new(backend, raw_hf_dequant_variant),
            output_variant,
        })
    }
}

fn resolve_kernel_variant(
    backend: &WgpuBackend,
    kernel: &'static str,
    default: KernelVariant,
) -> Result<KernelVariant, VarDctDecodeError> {
    let variant = backend
        .kernel_policy()
        .variant_for(kernel, default)
        .map_err(|error| VarDctDecodeError::KernelPolicy {
            kernel,
            message: error.to_string(),
        })?;
    variant
        .validate_for(kernel, &backend.device().limits(), 0)
        .map_err(|error| VarDctDecodeError::KernelPolicy {
            kernel,
            message: error.to_string(),
        })?;
    Ok(variant)
}

/// GPU-only submission engine for the bounded standard regular-VarDCT profile.
#[derive(Clone)]
pub struct VarDctSubmissionEngine {
    backend: WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory: MemoryBudget,
    stream_window_limit: Option<NonZeroU64>,
    subsampled_adaptive_lf_policy: SubsampledAdaptiveLfPolicy,
}

impl VarDctSubmissionEngine {
    pub fn new(backend: WgpuBackend) -> Result<Self, VarDctDecodeError> {
        let memory = backend.transient_memory_budget().clone();
        Self::with_memory_budget(backend, memory)
    }

    /// Uses an explicitly shared byte budget for output, entropy, render, and validation buffers.
    pub fn with_memory_budget(
        backend: WgpuBackend,
        memory: MemoryBudget,
    ) -> Result<Self, VarDctDecodeError> {
        let pipelines = Arc::new(VarDctPipelines::new(&backend)?);
        Ok(Self {
            backend,
            pipelines,
            memory,
            stream_window_limit: None,
            subsampled_adaptive_lf_policy: SubsampledAdaptiveLfPolicy::default(),
        })
    }

    #[must_use]
    pub fn with_subsampled_adaptive_lf_policy(
        mut self,
        policy: SubsampledAdaptiveLfPolicy,
    ) -> Self {
        self.subsampled_adaptive_lf_policy = policy;
        self
    }

    #[must_use]
    pub const fn subsampled_adaptive_lf_policy(&self) -> SubsampledAdaptiveLfPolicy {
        self.subsampled_adaptive_lf_policy
    }

    /// Caps reusable VarDCT entropy uploads.
    ///
    /// Combined/global-tree packets, staged local-tree LF/HF packets, and AC pass groups enforce
    /// this caller upper bound. Device limits and the shared per-frame byte budget may resolve a
    /// smaller four-byte-aligned cap. Recursive entropy streams will adopt the same policy with
    /// their resume state.
    #[must_use]
    pub fn with_stream_window_limit(mut self, limit: NonZeroU64) -> Self {
        self.stream_window_limit = Some(limit);
        self
    }

    /// Returns the caller-supplied upper bound, not a session's budget-resolved cap. The latter is
    /// reported by [`VarDctDecodeMemoryStats::resolved_stream_window_limit_bytes`].
    #[must_use]
    pub const fn stream_window_limit(&self) -> Option<NonZeroU64> {
        self.stream_window_limit
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        &self.backend
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    pub(crate) fn open_with_inventory(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        self.open_with_inventory_data(codestream, request, inventory)
    }

    pub(crate) fn open_with_inventory_data(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        let source = prepare_source(
            &self.backend,
            codestream,
            request,
            inventory,
            VarDctPrepareOptions {
                output_variant: self.pipelines.output_variant,
                stream_window_limit: self.stream_window_limit,
                memory_limit_bytes: self.memory.snapshot().limit_bytes,
                progressive_dc_final: None,
                subsampled_adaptive_lf_policy: self.subsampled_adaptive_lf_policy,
            },
        )?;
        self.open_source(source)
    }

    pub(crate) fn open_progressive_dc_with_inventory_data(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
        is_final: bool,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        let source = prepare_source(
            &self.backend,
            codestream,
            request,
            inventory,
            VarDctPrepareOptions {
                output_variant: self.pipelines.output_variant,
                stream_window_limit: self.stream_window_limit,
                memory_limit_bytes: self.memory.snapshot().limit_bytes,
                progressive_dc_final: Some(is_final),
                subsampled_adaptive_lf_policy: self.subsampled_adaptive_lf_policy,
            },
        )?;
        self.open_source(source)
    }

    fn open_source(
        &self,
        source: VarDctSource,
    ) -> DecodeResult<PreparedGpuSession<VarDctDecodeSession>> {
        let extent = source.layout.extent;
        let profile = DecodeProfile::VarDct { bits_per_sample: 8 };
        let submissions_per_frame = source.submissions_per_frame();
        let runtime_stats = Arc::new(VarDctRuntimeStats {
            submissions_per_frame: Arc::new(AtomicUsize::new(submissions_per_frame)),
            hf_packet_stream_batch_count: AtomicUsize::new(0),
        });
        Ok(PreparedGpuSession::new(
            profile,
            AnimationMetadata::still(extent),
            VarDctDecodeSession {
                backend: self.backend.clone(),
                pipelines: Arc::clone(&self.pipelines),
                memory_stats: source.memory,
                runtime_stats,
                adaptive_lf: source.adaptive_lf,
                source: Some(source),
                memory: self.memory.clone(),
            },
        )
        .with_resolved_frame_slots(NonZeroUsize::new(1).expect("one is nonzero")))
    }
}

impl std::fmt::Debug for VarDctSubmissionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctSubmissionEngine")
            .field("backend", &self.backend)
            .field("memory", &self.memory.snapshot())
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionEngine for VarDctSubmissionEngine {
    type Session = VarDctDecodeSession;

    fn parse_limits(&self) -> ParseLimits {
        ParseLimits {
            max_input_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
            max_boxes: 32,
            max_box_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
            max_codestream_bytes: VAR_DCT_PARSE_LIMIT_BYTES,
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
    ) -> DecodeResult<PreparedGpuSession<Self::Session>> {
        self.open_with_inventory(codestream, request, &inventory)
    }
}
