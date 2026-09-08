use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::SubmissionToken;
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, MemoryBudget, MemoryPermit, ResidentChromaShift,
    ResidentChromaUpsampleInputs, ResidentEpfInputs, ResidentF32Plane, ResidentGaborishInputs,
    ResidentStorageBinding, ResidentUpsampleInputs, ResidentUpsampleWeights, ResidentVarDctInputs,
    ResidentVarDctRenderConfig, ResidentVarDctScratch, SubmissionPollPermit,
    UnvalidatedGpuImageFrame, WgpuBackend,
};
use wgpu::util::DeviceExt;

use crate::color_output::{ColorOutputInputs, ColorOutputPlane, ColorOutputScratch};
use crate::progressive_dc::{
    ProgressiveDcGpuError, ProgressiveDcPackInputs, ProgressiveDcXybPlanes,
};
use crate::vardct_artifact::{GpuVarDctArtifactStatus, HfMetadataLoweringBuffers};
use crate::vardct_lf::{AdaptiveLfBuffers, AdaptiveLfParams};
use crate::vardct_packet::{
    GpuVarDctPacketStatus, VarDctModularParams, VarDctPacketBuffers, VarDctPacketValidation,
};
use crate::vardct_pass_group::{
    GpuHfCoefficientStatus, HfCoefficientBuffers, HfCoefficientExecutionPlan,
    HfCoefficientGroupExecutionPlan,
};
use crate::vardct_resource::VarDctResourceBuffers;
use crate::{
    Error as DecodeError, FrameDuration, FrameMetadata, GpuCodestream, GpuPendingFrame,
    GpuSubmissionSession, Result as DecodeResult, SubmittedGpuFrame,
};

use super::output::VarDctFrameOutput;
use super::pipeline::VarDctPipelines;
use super::restoration::RestorationCursor;
use super::source::{VarDctSource, check_limit};
use super::types::{
    ARTIFACT_STATUS_BYTES, PACKET_STATUS_BYTES, VarDctDecodeError, VarDctDecodeMemoryStats,
};
use super::window_plan::{PacketStage, PacketWindowExecutionPlan, map_codestream_source_error};

mod coefficients;
use coefficients::{HfCoefficientBatchSubmission, prepare_hf_batches};
mod extra;
mod packet;

use packet::{
    PacketBatchSubmission, prepare_packet_windows, submit_packet_batches, submit_packet_commands,
};
mod raw_matrix;
use extra::{ExtraLifetime, ExtraWork, HfValidation, map_extra_error};
use raw_matrix::{RawMatrixLifetime, RawMatrixWork};

/// One-frame submission state for [`crate::VarDctSubmissionEngine`].
pub struct FrameDecodeSession {
    pub(super) backend: WgpuBackend,
    pub(super) pipelines: Arc<VarDctPipelines>,
    pub(super) memory_stats: VarDctDecodeMemoryStats,
    pub(super) runtime_stats: Arc<VarDctRuntimeStats>,
    pub(super) source: Option<VarDctSource>,
    pub(super) memory: MemoryBudget,
}

#[derive(Debug)]
pub(super) struct VarDctRuntimeStats {
    pub(super) submissions_per_frame: Arc<AtomicUsize>,
    pub(super) hf_packet_stream_batch_count: AtomicUsize,
}

impl FrameDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> VarDctDecodeMemoryStats {
        self.memory_stats
    }

    pub(crate) fn set_progressive_dc_source(
        &mut self,
        planes: ProgressiveDcXybPlanes,
    ) -> Result<(), VarDctDecodeError> {
        let source = self
            .source
            .as_mut()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        if !source.packet.profile.uses_lf_frame {
            return Err(VarDctDecodeError::UnexpectedProgressiveDcSource);
        }
        planes.validate_extent(source.packet.block_extent())?;
        source.external_lf = Some(planes);
        Ok(())
    }
}

impl std::fmt::Debug for FrameDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FrameDecodeSession")
            .field("submitted", &self.source.is_none())
            .field("memory_stats", &self.memory_stats())
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for FrameDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = FramePendingFrame;

    fn submit_next(&mut self) -> DecodeResult<Option<Self::Pending>> {
        let Some(source) = self.source.as_ref() else {
            return Ok(None);
        };
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let output_permit = self.memory.try_reserve(source.memory.output_lease_bytes)?;
        let extra_permit = (source.memory.extra_arena_bytes != 0)
            .then(|| self.memory.try_reserve(source.memory.extra_arena_bytes))
            .transpose()?;
        let transient_permit = self
            .memory
            .try_reserve(source.memory.transient_bytes - source.memory.extra_arena_bytes)?;
        let source = self
            .source
            .take()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let pending = submit_vardct(
            &self.backend,
            Arc::clone(&self.pipelines),
            self.memory.clone(),
            Arc::clone(&self.runtime_stats),
            source,
            VarDctMemoryPermits {
                output: output_permit,
                transient: transient_permit,
                extra: extra_permit,
            },
            poll_permit,
        )?;
        Ok(Some(pending))
    }
}

struct VarDctMemoryPermits {
    output: MemoryPermit,
    transient: MemoryPermit,
    extra: Option<MemoryPermit>,
}

struct HfCoefficientJobBuffers {
    entropy_bundle: wgpu::Buffer,
    order_table: wgpu::Buffer,
    stream_window: Option<wgpu::Buffer>,
    params_window: Option<wgpu::Buffer>,
    groups: Vec<HfCoefficientGroupJobBuffers>,
}

struct HfCoefficientGroupJobBuffers {
    params: Option<wgpu::Buffer>,
    status: wgpu::Buffer,
    sink_params: wgpu::Buffer,
}

fn create_hf_coefficient_job_buffers(
    device: &wgpu::Device,
    plan: &HfCoefficientExecutionPlan,
) -> HfCoefficientJobBuffers {
    let windowed = plan.uses_bounded_stream_windows();
    let storage = |label, size, extra| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    HfCoefficientJobBuffers {
        entropy_bundle: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu HF entropy bundle"),
            contents: bytemuck::cast_slice(&plan.entropy_words),
            usage: wgpu::BufferUsages::STORAGE,
        }),
        order_table: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu HF natural-order table"),
            contents: bytemuck::cast_slice(&plan.order_words),
            usage: wgpu::BufferUsages::STORAGE,
        }),
        stream_window: windowed.then(|| {
            storage(
                "jxl-wgpu reusable HF coefficient stream window",
                plan.stream_window_bytes(),
                wgpu::BufferUsages::COPY_DST,
            )
        }),
        params_window: windowed.then(|| {
            storage(
                "jxl-wgpu reusable HF coefficient parameter window",
                plan.reusable_params_bytes(),
                wgpu::BufferUsages::COPY_DST,
            )
        }),
        groups: plan
            .groups
            .iter()
            .map(|group| HfCoefficientGroupJobBuffers {
                params: (!windowed).then(|| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jxl-wgpu LF-group HF pass-group params"),
                        contents: bytemuck::cast_slice(&group.params),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
                }),
                status: storage(
                    "jxl-wgpu LF-group HF pass-group status",
                    group.status_bytes(),
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                ),
                sink_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu LF-group HF coefficient sink params"),
                    contents: bytemuck::bytes_of(&group.sink_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
            })
            .collect(),
    }
}

enum PacketCommands {
    Whole(wgpu::CommandBuffer),
    Windowed(Vec<PacketBatchSubmission>),
}

enum VarDctDownstreamCommands {
    Whole(wgpu::CommandBuffer),
    Windowed {
        before_coefficients: wgpu::CommandBuffer,
        coefficient_batches: Vec<HfCoefficientBatchSubmission>,
        device: wgpu::Device,
        pipelines: Arc<VarDctPipelines>,
        after_coefficients: wgpu::CommandBuffer,
    },
}

struct DeferredHfGlobalCommands {
    before_coefficients: Option<wgpu::CommandBuffer>,
    after_coefficients: wgpu::CommandBuffer,
}

enum PostLfCommands {
    Direct(VarDctDownstreamCommands),
    DeferredHfGlobal(DeferredHfGlobalCommands),
}

enum VarDctPendingContinuation {
    LfExtras {
        source: Box<VarDctSource>,
        commands: PostLfCommands,
    },
    AcExtra {
        source: Box<VarDctSource>,
    },
    ModularExtra {
        work: Box<ExtraWork>,
        lifetime: Arc<ExtraLifetime>,
    },
    LocalLf {
        source: Box<VarDctSource>,
        commands: PostLfCommands,
    },
    HfGlobal {
        source: Box<VarDctSource>,
        commands: DeferredHfGlobalCommands,
    },
    RawHfDequant {
        work: Box<RawMatrixWork>,
        lifetime: Arc<RawMatrixLifetime>,
    },
}

fn submit_vardct_downstream(
    queue: &wgpu::Queue,
    mut prefix: Vec<wgpu::CommandBuffer>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match downstream {
        VarDctDownstreamCommands::Whole(commands) => {
            prefix.push(commands);
            Ok(queue.submit(prefix))
        }
        VarDctDownstreamCommands::Windowed {
            before_coefficients,
            coefficient_batches,
            device,
            pipelines,
            after_coefficients,
        } => {
            prefix.push(before_coefficients);
            queue.submit(prefix);
            let retained = lock_unpoisoned(&lifetime._hf_coefficients);
            let buffers = retained
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed AC commands have no retained coefficient buffers",
                })?;
            let stream =
                buffers
                    .stream_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC commands have no stream upload",
                    })?;
            let params =
                buffers
                    .params_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC commands have no parameter upload",
                    })?;
            for batch in coefficient_batches {
                queue.write_buffer(stream, 0, &batch.stream_upload);
                queue.write_buffer(params, 0, &batch.params_upload);
                queue.submit([batch.record(&device, &pipelines, buffers, &lifetime._groups)?]);
            }
            Ok(queue.submit([after_coefficients]))
        }
    }
}

struct VarDctGroupJobBuffers {
    reconstructed: wgpu::Buffer,
    raw_metadata: wgpu::Buffer,
    coefficients: wgpu::Buffer,
    packet_status: wgpu::Buffer,
    packet_control: wgpu::Buffer,
    modular_params: wgpu::Buffer,
    artifact: wgpu::Buffer,
    occupancy: wgpu::Buffer,
    artifact_uniform: wgpu::Buffer,
}

#[derive(Default)]
struct PostTransformJobBuffers {
    _restoration_planes: Option<[wgpu::Buffer; 3]>,
    _pre_restoration_planes: Option<[wgpu::Buffer; 3]>,
    _pre_restoration_uniforms: Vec<wgpu::Buffer>,
    _gaborish_uniform: Option<wgpu::Buffer>,
    _epf_sigma: Option<wgpu::Buffer>,
    _epf_sigma_uniforms: Vec<wgpu::Buffer>,
    _epf_uniforms: Vec<wgpu::Buffer>,
    _frame_upsample_planes: Option<[wgpu::Buffer; 3]>,
    _frame_upsample_weights: Option<ResidentUpsampleWeights>,
    _frame_upsample_uniforms: Vec<wgpu::Buffer>,
}

enum FrameOutputScratch {
    Color {
        _scratch: ColorOutputScratch,
    },
    Extra {
        scratch: crate::modular_scalar_output::ModularScalarOutputScratch,
    },
}

impl FrameOutputScratch {
    fn status_bytes(&self) -> u64 {
        match self {
            Self::Color { .. } => 0,
            Self::Extra { .. } => {
                crate::modular_scalar_output::ModularScalarOutputPlan::STATUS_BYTES
            }
        }
    }

    fn copy_status(&self, encoder: &mut wgpu::CommandEncoder, staging: &wgpu::Buffer) {
        if let Self::Extra { scratch } = self {
            encoder.copy_buffer_to_buffer(
                &scratch.status,
                0,
                staging,
                staging.size() - self.status_bytes(),
                self.status_bytes(),
            );
        }
    }
}

struct VarDctJobLifetime {
    output: GpuBufferLease,
    status_staging: wgpu::Buffer,
    status_mapped: AtomicBool,
    _transient_permits: Mutex<Vec<MemoryPermit>>,
    _codestream: wgpu::Buffer,
    _packet_stream_window: Option<wgpu::Buffer>,
    _modular_metadata: Mutex<Vec<wgpu::Buffer>>,
    _groups: Vec<VarDctGroupJobBuffers>,
    _lf_temporary: Option<wgpu::Buffer>,
    _resources: wgpu::Buffer,
    _resource_uniforms: Vec<wgpu::Buffer>,
    _adaptive_lf_uniform: Option<wgpu::Buffer>,
    _progressive_dc_uniform: Option<wgpu::Buffer>,
    _external_lf: Option<ProgressiveDcXybPlanes>,
    _extra_planes: Vec<super::staging::ResidentModularPlane>,
    extra_frame: Option<GpuBufferLease>,
    _extra_prefix: Option<GpuBufferLease>,
    _extra_uniforms: Vec<wgpu::Buffer>,
    _rendered_extra: Option<crate::modular_render::ModularRenderBuffers>,
    _hf_coefficients: Mutex<Option<HfCoefficientJobBuffers>>,
    _resident_planes: Option<[wgpu::Buffer; 3]>,
    lf_planes: Option<ProgressiveDcXybPlanes>,
    _post_transform: PostTransformJobBuffers,
    _resident_scratch: Vec<ResidentVarDctScratch>,
    _output_scratch: FrameOutputScratch,
}

impl Drop for VarDctJobLifetime {
    fn drop(&mut self) {
        if self.status_mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.unmap();
        }
    }
}

#[derive(Clone, Debug)]
struct VarDctGroupValidation {
    expected_lf_samples: u32,
    expected_coefficients: u32,
    expected_blocks: u32,
    correlation_samples: u32,
    task_capacity: u32,
    expected_global_scale: u32,
    expected_quant_lf: u32,
    expected_extra_precision: u8,
}

/// Submitted VarDCT frame awaiting one aggregate map of every LF/pass-group status record.
pub struct FramePendingFrame {
    pub(super) backend: WgpuBackend,
    pub(super) pipelines: Arc<VarDctPipelines>,
    pub(super) memory: MemoryBudget,
    pub(super) runtime_stats: Arc<VarDctRuntimeStats>,
    lifetime: Option<Arc<VarDctJobLifetime>>,
    stage: VarDctPendingStage,
    token: SubmissionToken,
    layout: ImageLayout,
    surface: Option<Arc<crate::frame_surface::FrameSurfaceLayout>>,
    frame_name: String,
    expected_groups: Vec<VarDctGroupValidation>,
    expected_hf: Vec<HfValidation>,
    extra_output_commands: Option<wgpu::CommandBuffer>,
    hf_metadata_stop: bool,
}

enum VarDctPendingStage {
    LfExtras {
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
        commands: Option<PostLfCommands>,
    },
    LocalLf {
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
        commands: Option<PostLfCommands>,
    },
    HfGlobal {
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
        commands: Option<DeferredHfGlobalCommands>,
    },
    RawHfDequant {
        completion: Arc<MapCompletion>,
        work: Box<RawMatrixWork>,
        lifetime: Arc<RawMatrixLifetime>,
    },
    AcExtra {
        completion: Arc<MapCompletion>,
        source: Box<VarDctSource>,
    },
    ModularExtra {
        completion: Arc<MapCompletion>,
        work: Box<ExtraWork>,
        lifetime: Arc<ExtraLifetime>,
    },
    Final {
        completion: Arc<MapCompletion>,
    },
}

impl std::fmt::Debug for FramePendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FramePendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("lf_group_count", &self.expected_groups.len())
            .field(
                "stage",
                &match &self.stage {
                    VarDctPendingStage::LfExtras { .. } => "lf-extras",
                    VarDctPendingStage::LocalLf { .. } => "local-lf",
                    VarDctPendingStage::HfGlobal { .. } => "hf-global",
                    VarDctPendingStage::RawHfDequant { .. } => "raw-hf-dequant",
                    VarDctPendingStage::AcExtra { .. } => "ac-extra",
                    VarDctPendingStage::ModularExtra { .. } => "modular-extra",
                    VarDctPendingStage::Final { .. } => "final",
                },
            )
            .finish_non_exhaustive()
    }
}

impl FramePendingFrame {
    #[must_use]
    pub(crate) fn dependency_submission_ready(&self) -> bool {
        matches!(self.stage, VarDctPendingStage::Final { .. })
    }

    pub(crate) fn progressive_dc_planes(
        &self,
    ) -> Result<ProgressiveDcXybPlanes, VarDctDecodeError> {
        if !self.dependency_submission_ready() {
            return Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted);
        }
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        lifetime
            .lf_planes
            .clone()
            .ok_or(VarDctDecodeError::UnvalidatedOutputNotSubmitted)
    }

    /// Same-queue, budget-tracked access before packet/artifact status becomes authoritative.
    pub fn unvalidated_gpu_frame(&self) -> DecodeResult<UnvalidatedGpuImageFrame> {
        if !self.dependency_submission_ready() {
            return Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted.into());
        }
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: crate::frame_surface::unvalidated_outputs(
                &self.layout,
                self.surface.as_deref(),
                &lifetime.output,
            ),
        })
    }

    fn stage_completion(&self) -> Arc<MapCompletion> {
        match &self.stage {
            VarDctPendingStage::LfExtras { completion, .. }
            | VarDctPendingStage::LocalLf { completion, .. }
            | VarDctPendingStage::HfGlobal { completion, .. }
            | VarDctPendingStage::RawHfDequant { completion, .. }
            | VarDctPendingStage::AcExtra { completion, .. }
            | VarDctPendingStage::ModularExtra { completion, .. }
            | VarDctPendingStage::Final { completion } => Arc::clone(completion),
        }
    }

    fn take_staged_packet(&mut self) -> Option<VarDctPendingContinuation> {
        let placeholder = VarDctPendingStage::Final {
            completion: Arc::new(MapCompletion::default()),
        };
        let stage = std::mem::replace(&mut self.stage, placeholder);
        match stage {
            VarDctPendingStage::LfExtras {
                source,
                mut commands,
                ..
            } => commands
                .take()
                .map(|commands| VarDctPendingContinuation::LfExtras { source, commands }),
            VarDctPendingStage::LocalLf {
                source,
                mut commands,
                ..
            } => commands
                .take()
                .map(|commands| VarDctPendingContinuation::LocalLf { source, commands }),
            VarDctPendingStage::HfGlobal {
                source,
                mut commands,
                ..
            } => commands
                .take()
                .map(|commands| VarDctPendingContinuation::HfGlobal { source, commands }),
            VarDctPendingStage::RawHfDequant { work, lifetime, .. } => {
                Some(VarDctPendingContinuation::RawHfDequant { work, lifetime })
            }
            VarDctPendingStage::AcExtra { source, .. } => {
                Some(VarDctPendingContinuation::AcExtra { source })
            }
            VarDctPendingStage::ModularExtra { work, lifetime, .. } => {
                Some(VarDctPendingContinuation::ModularExtra { work, lifetime })
            }
            final_stage @ VarDctPendingStage::Final { .. } => {
                self.stage = final_stage;
                None
            }
        }
    }

    fn advance_staged_packet(&mut self, mapping: Result<(), String>) -> DecodeResult<bool> {
        match self.take_staged_packet() {
            Some(VarDctPendingContinuation::LfExtras { source, commands }) => {
                mapping.map_err(DecodeError::backend)?;
                let lifetime = self
                    .lifetime
                    .as_ref()
                    .ok_or(VarDctDecodeError::CompletionConsumed)?;
                // This mapping fences initial arena copies, not a nonexistent LF decode. The
                // section starts are known without consulting a GPU coefficient status record.
                lifetime.status_staging.unmap();
                lifetime.status_mapped.store(false, Ordering::Release);
                let cursors = source
                    .packet
                    .groups
                    .iter()
                    .map(|group| group.entry.bit_offset())
                    .collect();
                self.start_lf_extras(source, commands, cursors)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::LocalLf { source, commands }) => {
                self.submit_hf_stage(mapping, source, commands)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::HfGlobal { source, commands }) => {
                self.submit_hf_global_stage(mapping, source, commands)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::RawHfDequant { work, lifetime }) => {
                self.finish_raw_hf_dequant_stage(mapping, work, lifetime)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::AcExtra { source }) => {
                self.finish_ac_extra(mapping, source)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::ModularExtra { work, lifetime }) => {
                self.finish_extra_subimage(mapping, work, lifetime)?;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wait_until_dependency_submitted(&mut self) -> DecodeResult<()> {
        while !matches!(self.stage, VarDctPendingStage::Final { .. }) {
            let mapping = self.stage_completion().wait();
            if !self.advance_staged_packet(mapping)? {
                return Err(VarDctDecodeError::EngineContract {
                    detail: "VarDCT dependency stage made no progress",
                }
                .into());
            }
        }
        Ok(())
    }

    pub(crate) fn poll_until_dependency_submitted(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<()>> {
        loop {
            if matches!(self.stage, VarDctPendingStage::Final { .. }) {
                return Poll::Ready(Ok(()));
            }
            if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(DecodeError::backend(error)));
            }
            let Some(mapping) = self.stage_completion().poll(context) else {
                return Poll::Pending;
            };
            if let Err(error) = self.advance_staged_packet(mapping) {
                return Poll::Ready(Err(error));
            }
        }
    }

    fn submit_hf_stage(
        &mut self,
        mapping: Result<(), String>,
        source: Box<VarDctSource>,
        post_lf: PostLfCommands,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = lifetime
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let mut cursors = Vec::with_capacity(self.expected_groups.len());
        for (index, expected) in self.expected_groups.iter().enumerate() {
            let offset = index.checked_mul(PACKET_STATUS_BYTES as usize).ok_or(
                VarDctDecodeError::StatusAbi {
                    status: "LF cursor offset",
                },
            )?;
            let status: GpuVarDctPacketStatus = mapped
                .get(offset..offset + PACKET_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "LF cursor",
                })?;
            cursors.push(
                status
                    .validate_lf_stage(
                        expected.expected_lf_samples,
                        expected.expected_global_scale,
                        expected.expected_quant_lf,
                        expected.expected_extra_precision,
                    )
                    .map_err(VarDctDecodeError::from)?,
            );
        }
        drop(mapped);
        lifetime.status_staging.unmap();
        lifetime.status_mapped.store(false, Ordering::Release);

        if source
            .packet
            .extra_channels
            .as_ref()
            .is_some_and(|plan| plan.has_lf())
        {
            self.start_lf_extras(source, post_lf, cursors)
        } else {
            self.submit_hf_from_cursors(source, post_lf, cursors)
        }
    }

    fn submit_hf_from_cursors(
        &mut self,
        source: Box<VarDctSource>,
        post_lf: PostLfCommands,
        cursors: Vec<u32>,
    ) -> DecodeResult<()> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let continuations = source
            .packet
            .groups
            .iter()
            .zip(cursors)
            .map(|(group, cursor)| {
                source
                    .packet
                    .parse_hf_continuation_source(&source.codestream, group, cursor)
                    .map_err(VarDctDecodeError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hf_packet_windows = PacketWindowExecutionPlan::hf(
            source.codestream.logical_bytes(),
            &source.packet,
            &continuations,
            source.stream_limit,
        )?;
        if let Some(plan) = &hf_packet_windows {
            let available = lifetime
                ._packet_stream_window
                .as_ref()
                .map_or(0, wgpu::Buffer::size);
            if plan.stream_bytes > available {
                return Err(VarDctDecodeError::DeviceLimit {
                    resource: "shared local-tree packet stream window",
                    required: plan.stream_bytes,
                    available,
                }
                .into());
            }
        }
        let hf_metadata_bytes = continuations
            .iter()
            .try_fold(0_u64, |total, continuation| {
                let words = u64::try_from(continuation.modular.metadata.len()).map_err(|_| {
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "HF-local Modular metadata length",
                    }
                })?;
                total
                    .checked_add(words.checked_mul(4).ok_or(
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "HF-local Modular metadata bytes",
                        },
                    )?)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF-local Modular metadata total",
                    })
            })?;
        let additional_permit = hf_metadata_bytes
            .checked_sub(source.memory.modular_metadata_bytes)
            .filter(|&bytes| bytes != 0)
            .map(|bytes| self.memory.try_reserve(bytes))
            .transpose()?;
        let limits = self.backend.device().limits();
        for continuation in &continuations {
            let bytes = u64::try_from(continuation.modular.metadata.len())
                .ok()
                .and_then(|words| words.checked_mul(4))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "HF-local Modular metadata binding",
                })?;
            check_limit("HF-local Modular metadata", bytes, limits.max_buffer_size)?;
            check_limit(
                "HF-local Modular metadata",
                bytes,
                limits.max_storage_buffer_binding_size,
            )?;
        }
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let device = self.backend.device();
        lock_unpoisoned(&lifetime._modular_metadata).clear();
        let mut metadata_buffers = Vec::with_capacity(continuations.len());
        for continuation in &continuations {
            metadata_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu VarDCT HF-local Modular metadata"),
                    contents: bytemuck::cast_slice(&continuation.modular.metadata),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );
        }
        {
            let mut retained = lock_unpoisoned(&lifetime._modular_metadata);
            retained.extend(metadata_buffers.iter().cloned());
        }
        if let Some(permit) = additional_permit {
            lock_unpoisoned(&lifetime._transient_permits).push(permit);
        }
        let controls = source
            .packet
            .groups
            .iter()
            .zip(&source.groups)
            .zip(&continuations)
            .map(|((packet_group, group), continuation)| {
                let control = packet_group
                    .hf_stage_control(&source.packet, continuation)
                    .map_err(VarDctDecodeError::from)?;
                debug_assert_eq!(group.control.geometry, control.geometry);
                Ok(control)
            })
            .collect::<Result<Vec<_>, VarDctDecodeError>>()?;

        let deferred_hf_global = matches!(&post_lf, PostLfCommands::DeferredHfGlobal(_));
        // Local packet trees can discover a raw matrix after LF preparation. Final validation
        // must match the entry point actually submitted, including its HF-metadata stop status.
        self.hf_metadata_stop = deferred_hf_global;

        let windowed_batches = hf_packet_windows
            .as_ref()
            .map(|plan| {
                prepare_packet_windows(
                    plan,
                    &source.codestream,
                    &controls,
                    deferred_hf_global,
                    None,
                )
            })
            .transpose()?;
        let completion = Arc::new(MapCompletion::default());
        let (submission, deferred_commands) = if let Some(batches) = windowed_batches {
            let batch_count = batches.len();
            let additional_submissions =
                batch_count
                    .checked_sub(1)
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed HF packet execution has no batches",
                    })?;
            let current_submissions = self
                .runtime_stats
                .submissions_per_frame
                .load(Ordering::Acquire);
            let total_submissions = current_submissions
                .checked_add(additional_submissions)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "VarDCT dynamic submission count",
                })?;
            self.runtime_stats
                .hf_packet_stream_batch_count
                .store(batch_count, Ordering::Release);
            self.runtime_stats
                .submissions_per_frame
                .store(total_submissions, Ordering::Release);
            match post_lf {
                PostLfCommands::Direct(downstream) => (
                    submit_packet_batches(
                        &self.backend,
                        &self.pipelines,
                        batches,
                        Some(downstream),
                        lifetime,
                    )?,
                    None,
                ),
                PostLfCommands::DeferredHfGlobal(deferred) => (
                    submit_packet_batches(&self.backend, &self.pipelines, batches, None, lifetime)?,
                    Some(deferred),
                ),
            }
        } else {
            let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT HF-local stage"),
            });
            for (((buffers, continuation), metadata), control) in lifetime
                ._groups
                .iter()
                .zip(&continuations)
                .zip(&metadata_buffers)
                .zip(&controls)
            {
                let params = VarDctModularParams::default()
                    .with_lz77_window(continuation.modular.lz77_window_words)
                    .with_self_correcting(continuation.modular.needs_self_correcting);
                self.backend.queue().write_buffer(
                    &buffers.packet_control,
                    0,
                    bytemuck::bytes_of(control),
                );
                self.backend.queue().write_buffer(
                    &buffers.modular_params,
                    0,
                    bytemuck::bytes_of(&params),
                );
                let packet_buffers = VarDctPacketBuffers {
                    codestream: &lifetime._codestream,
                    modular_metadata: metadata,
                    reconstructed_lf: &buffers.reconstructed,
                    raw_hf_metadata: &buffers.raw_metadata,
                    coefficients: &buffers.coefficients,
                    status: &buffers.packet_status,
                    control: &buffers.packet_control,
                    modular_params: &buffers.modular_params,
                };
                if deferred_hf_global {
                    self.pipelines
                        .packet
                        .encode_hf_metadata(device, &mut commands, packet_buffers);
                } else {
                    self.pipelines
                        .packet
                        .encode_hf(device, &mut commands, packet_buffers);
                }
            }
            if deferred_hf_global {
                for (index, buffers) in lifetime._groups.iter().enumerate() {
                    commands.copy_buffer_to_buffer(
                        &buffers.packet_status,
                        0,
                        &lifetime.status_staging,
                        u64::try_from(index).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "deferred HF metadata status index",
                            }
                        })? * PACKET_STATUS_BYTES,
                        PACKET_STATUS_BYTES,
                    );
                }
            }
            match post_lf {
                PostLfCommands::Direct(downstream) => (
                    submit_vardct_downstream(
                        self.backend.queue(),
                        vec![commands.finish()],
                        downstream,
                        lifetime,
                    )?,
                    None,
                ),
                PostLfCommands::DeferredHfGlobal(deferred) => (
                    self.backend.queue().submit([commands.finish()]),
                    Some(deferred),
                ),
            }
        };
        arm_status_map(
            lifetime,
            &completion,
            if deferred_commands.is_some() {
                "VarDCT HF-global cursor mapping"
            } else {
                "VarDCT final validation mapping"
            },
        );
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!("VarDCT GPU poll registration failed: {error}")));
        }
        self.stage = if let Some(commands) = deferred_commands {
            VarDctPendingStage::HfGlobal {
                completion,
                source,
                commands: Some(commands),
            }
        } else {
            self.after_coefficients_stage(completion, source)
        };
        Ok(())
    }

    fn submit_hf_global_stage(
        &mut self,
        mapping: Result<(), String>,
        mut source: Box<VarDctSource>,
        commands: DeferredHfGlobalCommands,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = lifetime
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let mut cursors = Vec::with_capacity(self.expected_groups.len());
        for (index, expected) in self.expected_groups.iter().enumerate() {
            let offset = index.checked_mul(PACKET_STATUS_BYTES as usize).ok_or(
                VarDctDecodeError::StatusAbi {
                    status: "HF-metadata cursor offset",
                },
            )?;
            let status: GpuVarDctPacketStatus = mapped
                .get(offset..offset + PACKET_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "HF-metadata cursor",
                })?;
            cursors.push(
                status
                    .validate_hf_metadata_stage(VarDctPacketValidation {
                        expected_strategy: None,
                        expected_lf_samples: expected.expected_lf_samples,
                        block_count: expected.expected_blocks,
                        correlation_samples: expected.correlation_samples,
                        task_capacity: expected.task_capacity,
                        expected_global_scale: expected.expected_global_scale,
                        expected_quant_lf: expected.expected_quant_lf,
                        expected_extra_precision: expected.expected_extra_precision,
                    })
                    .map_err(VarDctDecodeError::from)?,
            );
        }
        drop(mapped);
        lifetime.status_staging.unmap();
        lifetime.status_mapped.store(false, Ordering::Release);

        if source.packet.pending_raw_hf_dequant_side_image().is_some() {
            self.start_raw_hf_dequant_stage(source, commands, None)?;
            return Ok(());
        }
        let [cursor] = cursors.as_slice() else {
            return Err(VarDctDecodeError::GroupPlanCount {
                component: "single-entry progressive-DC packet",
                expected: 1,
                actual: cursors.len(),
            }
            .into());
        };
        source
            .packet
            .parse_single_hf_global_continuation_source(&source.codestream, *cursor)
            .map_err(VarDctDecodeError::from)?;
        if source.packet.pending_raw_hf_dequant_side_image().is_some() {
            self.start_raw_hf_dequant_stage(source, commands, None)?;
            return Ok(());
        }
        self.submit_deferred_hf_coefficients(source, commands)
    }

    fn submit_deferred_hf_coefficients(
        &mut self,
        source: Box<VarDctSource>,
        mut commands: DeferredHfGlobalCommands,
    ) -> DecodeResult<()> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let entropy =
            source
                .packet
                .hf_coefficients
                .as_ref()
                .ok_or(VarDctDecodeError::EngineContract {
                    detail: "deferred HF-global parse did not produce coefficient metadata",
                })?;
        let artifacts = source
            .groups
            .iter()
            .map(|group| group.artifact_layout)
            .collect::<Vec<_>>();
        let plan = HfCoefficientExecutionPlan::new(
            &source.packet,
            entropy,
            &artifacts,
            source.codestream.logical_bytes(),
            source.stream_limit,
        )
        .map_err(VarDctDecodeError::from)?;
        if let Some(words) = &entropy.dequant_matrix_words {
            source
                .resource_layout
                .validate_dequant_matrix_words(words)
                .map_err(VarDctDecodeError::from)?;
            let raw_matrices = entropy
                .raw_dequant_matrices
                .iter()
                .map(|matrix| matrix.matrix_index)
                .collect::<Vec<_>>();
            let layout = source.resource_layout;
            for range in layout.host_dequant_matrix_ranges(&raw_matrices) {
                let start = (range.start - layout.matrix_offsets[0]) as usize;
                let end = (range.end - layout.matrix_offsets[0]) as usize;
                self.backend.queue().write_buffer(
                    &lifetime._resources,
                    u64::from(range.start) * 16,
                    bytemuck::cast_slice(&words[start..end]),
                );
            }
        }
        let deferred = source
            .deferred_hf
            .as_ref()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "deferred HF-global stage has no admitted scratch layout",
            })?;
        if plan.groups.len() != deferred.groups.len() {
            return Err(VarDctDecodeError::GroupPlanCount {
                component: "deferred HF coefficient",
                expected: deferred.groups.len(),
                actual: plan.groups.len(),
            }
            .into());
        }
        for (actual, admitted) in plan.groups.iter().zip(&deferred.groups) {
            if actual.lz77_scratch_bytes() > admitted.lz77_scratch_bytes
                || actual.execution_state_bytes() > admitted.execution_state_bytes
            {
                return Err(VarDctDecodeError::EngineContract {
                    detail: "deferred HF coefficient scratch exceeded its conservative admission",
                }
                .into());
            }
        }
        if plan.status_bytes() > deferred.status_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "deferred HF coefficient status exceeded its conservative admission",
            }
            .into());
        }
        let checked_words = |words: usize, field: &'static str| {
            u64::try_from(words)
                .ok()
                .and_then(|words| words.checked_mul(4))
                .ok_or(VarDctDecodeError::ArithmeticOverflow { field })
        };
        let entropy_bytes = checked_words(plan.entropy_words.len(), "deferred HF entropy bytes")?;
        let order_bytes = checked_words(plan.order_words.len(), "deferred HF order bytes")?;
        let stream_bytes = if plan.uses_bounded_stream_windows() {
            plan.stream_window_bytes()
        } else {
            0
        };
        let params_bytes = if plan.uses_bounded_stream_windows() {
            plan.reusable_params_bytes()
        } else {
            plan.groups.iter().try_fold(0_u64, |total, group| {
                u64::try_from(group.params.len())
                    .ok()
                    .and_then(|count| {
                        count.checked_mul(std::mem::size_of::<
                            crate::vardct_pass_group::HfCoefficientPassParams,
                        >() as u64)
                    })
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "deferred HF parameter bytes",
                    })
            })?
        };
        if params_bytes > deferred.params_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "deferred HF parameters exceeded their conservative admission",
            }
            .into());
        }
        let sink_bytes = u64::try_from(plan.groups.len())
            .ok()
            .and_then(|groups| {
                groups.checked_mul(std::mem::size_of::<
                    crate::vardct_artifact::HfCoefficientSinkParams,
                >() as u64)
            })
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF sink bytes",
            })?;
        if sink_bytes > deferred.sink_uniform_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "deferred HF sink uniforms exceeded their conservative admission",
            }
            .into());
        }
        let dynamic_bytes = entropy_bytes
            .checked_add(order_bytes)
            .and_then(|bytes| bytes.checked_add(stream_bytes))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF dynamic bytes",
            })?;
        let dynamic_permit = self.memory.try_reserve(dynamic_bytes)?;
        let limits = self.backend.device().limits();
        for (resource, required, storage) in [
            ("deferred HF entropy bundle", entropy_bytes, true),
            ("deferred HF order table", order_bytes, true),
            ("deferred HF stream window", stream_bytes, true),
            ("deferred HF parameters", params_bytes, true),
            ("deferred HF status", plan.status_bytes(), true),
            ("deferred HF sink uniform", sink_bytes, false),
        ] {
            check_limit(resource, required, limits.max_buffer_size)?;
            if storage {
                check_limit(resource, required, limits.max_storage_buffer_binding_size)?;
            } else {
                check_limit(resource, required, limits.max_uniform_buffer_binding_size)?;
            }
        }
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(DecodeError::PollBackpressure)?;
        let device = self.backend.device();
        let buffers = create_hf_coefficient_job_buffers(device, &plan);
        let mut coefficient_batches = Vec::new();
        let mut whole_coefficients = None;
        if plan.uses_bounded_stream_windows() {
            coefficient_batches = prepare_hf_batches(&source.codestream, &plan)?;
        } else {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu deferred whole-range HF coefficients"),
            });
            for ((group_plan, hf_buffers), group_buffers) in plan
                .groups
                .iter()
                .zip(&buffers.groups)
                .zip(&lifetime._groups)
            {
                let params =
                    hf_buffers
                        .params
                        .as_ref()
                        .ok_or(VarDctDecodeError::EntropyWindowContract {
                            detail: "deferred whole-range HF plan has no parameters",
                        })?;
                self.pipelines.hf_coefficients.encode(
                    device,
                    &mut encoder,
                    HfCoefficientBuffers {
                        codestream: &lifetime._codestream,
                        entropy_bundle: &buffers.entropy_bundle,
                        reconstruction: &group_buffers.reconstructed,
                        params,
                        status: &hf_buffers.status,
                        artifact: &group_buffers.artifact,
                        order_table: &buffers.order_table,
                        coefficients: &group_buffers.coefficients,
                        sink_params: &hf_buffers.sink_params,
                    },
                    u32::try_from(group_plan.params.len()).map_err(|_| {
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "deferred HF dispatch count",
                        }
                    })?,
                );
            }
            whole_coefficients = Some(encoder.finish());
        }
        let mut status_commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu deferred HF status aggregation"),
        });
        let group_count = u64::try_from(lifetime._groups.len()).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF status group count",
            }
        })?;
        let mut status_offset = group_count
            .checked_mul(PACKET_STATUS_BYTES + ARTIFACT_STATUS_BYTES)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "deferred HF status offset",
            })?;
        for group in &buffers.groups {
            let bytes = group.status.size();
            status_commands.copy_buffer_to_buffer(
                &group.status,
                0,
                &lifetime.status_staging,
                status_offset,
                bytes,
            );
            status_offset =
                status_offset
                    .checked_add(bytes)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "deferred HF status end",
                    })?;
        }
        if status_offset + lifetime._output_scratch.status_bytes()
            != source.memory.validation_staging_bytes
        {
            return Err(VarDctDecodeError::EngineContract {
                detail: "deferred HF status aggregation disagrees with admitted staging",
            }
            .into());
        }
        self.expected_hf = HfValidation::plan(&source, &plan).map_err(map_extra_error)?;
        {
            let mut retained = lock_unpoisoned(&lifetime._hf_coefficients);
            if retained.is_some() {
                return Err(VarDctDecodeError::EngineContract {
                    detail: "deferred HF coefficient buffers were already installed",
                }
                .into());
            }
            *retained = Some(buffers);
        }
        lock_unpoisoned(&lifetime._transient_permits).push(dynamic_permit);

        let submission =
            if let Some(whole_coefficients) = whole_coefficients {
                let mut submissions = Vec::with_capacity(4);
                if let Some(before_coefficients) = commands.before_coefficients.take() {
                    submissions.push(before_coefficients);
                }
                submissions.extend([
                    whole_coefficients,
                    commands.after_coefficients,
                    status_commands.finish(),
                ]);
                self.backend.queue().submit(submissions)
            } else {
                let retained = lock_unpoisoned(&lifetime._hf_coefficients);
                let buffers = retained.as_ref().ok_or(VarDctDecodeError::EngineContract {
                    detail: "deferred HF coefficient buffers disappeared before submission",
                })?;
                let stream = buffers.stream_window.as_ref().ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "deferred HF coefficient buffers have no stream window",
                    },
                )?;
                let params = buffers.params_window.as_ref().ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "deferred HF coefficient buffers have no parameter window",
                    },
                )?;
                let mut batches = coefficient_batches.into_iter();
                let first = batches
                    .next()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "deferred windowed HF plan has no batches",
                    })?;
                self.backend
                    .queue()
                    .write_buffer(stream, 0, &first.stream_upload);
                self.backend
                    .queue()
                    .write_buffer(params, 0, &first.params_upload);
                let first_commands =
                    first.record(device, &self.pipelines, buffers, &lifetime._groups)?;
                if let Some(before_coefficients) = commands.before_coefficients.take() {
                    self.backend
                        .queue()
                        .submit([before_coefficients, first_commands]);
                } else {
                    self.backend.queue().submit([first_commands]);
                }
                let mut batch_count = 1_usize;
                for batch in batches {
                    self.backend
                        .queue()
                        .write_buffer(stream, 0, &batch.stream_upload);
                    self.backend
                        .queue()
                        .write_buffer(params, 0, &batch.params_upload);
                    self.backend.queue().submit([batch.record(
                        device,
                        &self.pipelines,
                        buffers,
                        &lifetime._groups,
                    )?]);
                    batch_count = batch_count.checked_add(1).ok_or(
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "deferred HF batch count",
                        },
                    )?;
                }
                // The known count already includes LF/HF packets and any preceding Modular
                // stages. Each newly discovered coefficient batch adds one queue submission.
                let total_submissions = self
                    .runtime_stats
                    .submissions_per_frame
                    .load(Ordering::Acquire)
                    .checked_add(batch_count)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "deferred HF submission count",
                    })?;
                self.runtime_stats
                    .submissions_per_frame
                    .store(total_submissions, Ordering::Release);
                drop(retained);
                self.backend
                    .queue()
                    .submit([commands.after_coefficients, status_commands.finish()])
            };
        let completion = Arc::new(MapCompletion::default());
        arm_status_map(
            lifetime,
            &completion,
            "VarDCT deferred HF validation mapping",
        );
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!(
                "VarDCT deferred HF GPU poll registration failed: {error}"
            )));
        }
        self.stage = self.after_coefficients_stage(completion, source);
        Ok(())
    }

    fn finish(
        &mut self,
        mapping: Result<(), String>,
    ) -> DecodeResult<SubmittedGpuFrame<GpuImageFrame>> {
        mapping.map_err(DecodeError::backend)?;
        let lifetime = self
            .lifetime
            .take()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let mapped = lifetime
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(DecodeError::backend)?;
        let group_count = self.expected_groups.len();
        let packet_bytes = group_count
            .checked_mul(PACKET_STATUS_BYTES as usize)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "packet count",
            })?;
        let artifact_bytes = group_count
            .checked_mul(ARTIFACT_STATUS_BYTES as usize)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "artifact count",
            })?;
        let hf_offset =
            packet_bytes
                .checked_add(artifact_bytes)
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "aggregate offset",
                })?;
        for (index, expected) in self.expected_groups.iter().enumerate() {
            let packet_offset = index * PACKET_STATUS_BYTES as usize;
            let packet_status: GpuVarDctPacketStatus = mapped
                .get(packet_offset..packet_offset + PACKET_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi { status: "packet" })?;
            let artifact_offset = packet_bytes + index * ARTIFACT_STATUS_BYTES as usize;
            let artifact: GpuVarDctArtifactStatus = mapped
                .get(artifact_offset..artifact_offset + ARTIFACT_STATUS_BYTES as usize)
                .and_then(|bytes| bytemuck::try_pod_read_unaligned(bytes).ok())
                .ok_or(VarDctDecodeError::StatusAbi { status: "artifact" })?;
            let validation = VarDctPacketValidation {
                expected_strategy: None,
                expected_lf_samples: expected.expected_lf_samples,
                block_count: expected.expected_blocks,
                correlation_samples: expected.correlation_samples,
                task_capacity: expected.task_capacity,
                expected_global_scale: expected.expected_global_scale,
                expected_quant_lf: expected.expected_quant_lf,
                expected_extra_precision: expected.expected_extra_precision,
            };
            let first_blocks = if self.hf_metadata_stop {
                packet_status
                    .validate_hf_metadata_stage(validation)
                    .map_err(VarDctDecodeError::from)?;
                packet_status.first_blocks
            } else {
                packet_status
                    .validate(validation)
                    .map_err(VarDctDecodeError::from)?
                    .first_blocks
            };
            if packet_status.coefficient_words != expected.expected_coefficients {
                return Err(VarDctDecodeError::ArtifactStatus {
                    field: "packet coefficient_words",
                    expected: expected.expected_coefficients,
                    actual: packet_status.coefficient_words,
                }
                .into());
            }
            artifact.validate().map_err(VarDctDecodeError::from)?;
            for (field, expected, actual) in [
                ("task_count", first_blocks, artifact.task_count),
                (
                    "coefficient_words",
                    expected.expected_coefficients,
                    artifact.coefficient_words,
                ),
                (
                    "covered_blocks",
                    expected.expected_blocks,
                    artifact.covered_blocks,
                ),
                (
                    "consumed_block_info_entries",
                    first_blocks,
                    artifact.consumed_block_info_entries,
                ),
                ("backend_requirements", 0, artifact.backend_requirements),
            ] {
                if actual != expected {
                    return Err(VarDctDecodeError::ArtifactStatus {
                        field,
                        expected,
                        actual,
                    }
                    .into());
                }
            }
        }
        let hf_end = mapped.len() - lifetime._output_scratch.status_bytes() as usize;
        let hf_status_bytes =
            mapped
                .get(hf_offset..hf_end)
                .ok_or(VarDctDecodeError::StatusAbi {
                    status: "HF coefficient",
                })?;
        let hf_statuses = bytemuck::try_cast_slice::<u8, GpuHfCoefficientStatus>(hf_status_bytes)
            .map_err(|_| VarDctDecodeError::StatusAbi {
            status: "HF coefficient",
        })?;
        if hf_statuses.len() != self.expected_hf.len() {
            return Err(VarDctDecodeError::StatusAbi {
                status: "HF coefficient count",
            }
            .into());
        }
        for (expected, status) in self.expected_hf.iter().zip(hf_statuses.iter().copied()) {
            expected.validate(status)?;
        }
        if matches!(lifetime._output_scratch, FrameOutputScratch::Extra { .. }) {
            crate::modular_scalar_output::ModularScalarOutputScratch::validate_status(
                &mapped[hf_end..],
            )
            .map_err(VarDctDecodeError::from)?;
        }
        drop(mapped);
        Ok(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                presentation_ticks: 0,
                timecode: None,
                is_last: true,
                is_keyframe: true,
                name: std::mem::take(&mut self.frame_name),
            },
            GpuImageFrame {
                token: self.token,
                outputs: crate::frame_surface::outputs(
                    &self.layout,
                    self.surface.as_deref(),
                    &lifetime.output,
                ),
                changed: crate::frame_surface::changed_regions(
                    &self.layout,
                    self.surface.as_deref(),
                ),
            },
        ))
    }
}

impl GpuPendingFrame for FramePendingFrame {
    type Frame = GpuImageFrame;

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> DecodeResult<SubmittedGpuFrame<Self::Frame>> {
        loop {
            let mapping = self.stage_completion().wait();
            if self.dependency_submission_ready() {
                return self.finish(mapping);
            }
            self.advance_staged_packet(mapping)?;
        }
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(DecodeError::backend(error)));
        }
        let Some(mapping) = self.stage_completion().poll(context) else {
            return Poll::Pending;
        };
        if self.dependency_submission_ready() {
            return Poll::Ready(self.finish(mapping));
        }
        if let Err(error) = self.advance_staged_packet(mapping) {
            return Poll::Ready(Err(error));
        }
        // Yield at every descriptor boundary so cancellation stays responsive in browsers.
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

fn resident_binding(
    buffer: &wgpu::Buffer,
) -> Result<ResidentStorageBinding<'_>, VarDctDecodeError> {
    Ok(ResidentStorageBinding::entire(buffer)?)
}

fn resident_image_planes<'a>(
    buffers: &'a [wgpu::Buffer; 3],
    width: u32,
    height: u32,
    stride: u32,
) -> Result<[ResidentF32Plane<'a>; 3], VarDctDecodeError> {
    Ok([
        ResidentF32Plane {
            storage: resident_binding(&buffers[0])?,
            width,
            height,
            stride,
        },
        ResidentF32Plane {
            storage: resident_binding(&buffers[1])?,
            width,
            height,
            stride,
        },
        ResidentF32Plane {
            storage: resident_binding(&buffers[2])?,
            width,
            height,
            stride,
        },
    ])
}

fn resident_shifted_image_planes<'a>(
    buffers: &'a [wgpu::Buffer; 3],
    width: u32,
    height: u32,
    shifts: [crate::vardct_frontend::VarDctChannelShift; 3],
) -> Result<[ResidentF32Plane<'a>; 3], VarDctDecodeError> {
    let plane = |channel: usize| -> Result<ResidentF32Plane<'a>, VarDctDecodeError> {
        let shift = shifts[channel];
        let [plane_width, plane_height] =
            shift
                .shifted_extent(width, height)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "resident shifted channel extent",
                })?;
        Ok(ResidentF32Plane {
            storage: resident_binding(&buffers[channel])?,
            width: plane_width,
            height: plane_height,
            stride: plane_width,
        })
    };
    Ok([plane(0)?, plane(1)?, plane(2)?])
}

fn upload_codestream(
    codestream: &GpuCodestream,
    buffer: &wgpu::Buffer,
    padded_bytes: u64,
) -> Result<(), VarDctDecodeError> {
    if padded_bytes < codestream.logical_bytes() || !padded_bytes.is_multiple_of(4) {
        return Err(VarDctDecodeError::EntropyWindowContract {
            detail: "GPU codestream buffer does not cover an aligned logical source",
        });
    }
    let logical_size = usize::try_from(codestream.logical_bytes()).map_err(|_| {
        VarDctDecodeError::ArithmeticOverflow {
            field: "codestream upload length",
        }
    })?;
    let mut mapped = buffer
        .get_mapped_range_mut(..)
        .map_err(|source| VarDctDecodeError::CodestreamMap { source })?;
    let upload_result = (|| {
        let mut mapped_cursor = 0usize;
        codestream
            .for_each_range_chunk(0..codestream.logical_bytes(), |chunk| -> DecodeResult<()> {
                let mapped_end = mapped_cursor
                    .checked_add(chunk.len())
                    .ok_or_else(|| DecodeError::backend("codestream mapped offset overflow"))?;
                if mapped_end > mapped.len() {
                    return Err(DecodeError::EngineContract(
                        "codestream span exceeds the mapped GPU buffer",
                    ));
                }
                mapped
                    .slice(mapped_cursor..mapped_end)
                    .copy_from_slice(chunk);
                mapped_cursor = mapped_end;
                Ok(())
            })
            .map_err(map_codestream_source_error)?;
        if mapped_cursor != logical_size {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "codestream spans did not fill the logical mapped range",
            });
        }
        if logical_size < mapped.len() {
            mapped.slice(logical_size..).fill(0);
        }
        Ok(())
    })();
    drop(mapped);
    buffer.unmap();
    upload_result
}

fn submit_vardct(
    backend: &WgpuBackend,
    pipelines: Arc<VarDctPipelines>,
    memory: MemoryBudget,
    runtime_stats: Arc<VarDctRuntimeStats>,
    mut source: VarDctSource,
    mut permits: VarDctMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<FramePendingFrame, VarDctDecodeError> {
    let device = backend.device();
    let codestream_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT codestream"),
        size: source.memory.codestream_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    upload_codestream(
        &source.codestream,
        &codestream_buffer,
        source.memory.codestream_bytes,
    )?;
    let staged_lf = source.packet.requires_lf_staging();
    let staged_lf_extras = source.packet.requires_lf_extra_staging();
    // Eager HF-only windows must finish before a sectioned raw matrix begins. Their validated
    // metadata stop uses the same continuation as a fused HF-global cursor.
    let staged_hf_global = (source.packet.requires_hf_global_staging()
        && !source.fuses_packet_stages())
        || (source.packet.pending_raw_hf_dequant_side_image().is_some()
            && source.packet_window_batches(PacketStage::Hf) != 0);
    let group_specific_metadata = staged_lf || source.packet.profile.uses_lf_frame;
    let modular_metadata = if group_specific_metadata {
        source
            .packet
            .groups
            .iter()
            .filter_map(|group| group.entry.modular())
            .map(|modular| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu VarDCT LF-local Modular metadata"),
                    contents: bytemuck::cast_slice(&modular.metadata),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            })
            .collect::<Vec<_>>()
    } else {
        vec![
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu VarDCT global Modular metadata"),
                contents: bytemuck::cast_slice(&source.packet.modular_metadata),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        ]
    };
    let storage = |label: &'static str, size: u64, extra: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    };
    let extra_frame =
        extra::prepare_frame_arena(backend, &mut source, permits.extra).map_err(map_extra_error)?;
    let packet_stream_window =
        NonZeroU64::new(source.memory.packet_stream_window_bytes).map(|bytes| {
            storage(
                "jxl-wgpu reusable packet entropy stream window",
                bytes.get(),
                wgpu::BufferUsages::COPY_DST,
            )
        });
    if source.groups.len() != source.packet.groups.len() {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "packet source",
            expected: source.packet.groups.len(),
            actual: source.groups.len(),
        });
    }
    if let Some(plan) = &source.hf_coefficients
        && plan.groups.len() != source.packet.groups.len()
    {
        return Err(VarDctDecodeError::GroupPlanCount {
            component: "HF coefficient",
            expected: source.packet.groups.len(),
            actual: plan.groups.len(),
        });
    }
    let mut group_buffers = Vec::with_capacity(source.groups.len());
    for (index, (packet_group, group)) in
        source.packet.groups.iter().zip(&source.groups).enumerate()
    {
        let predictor_capacity =
            source.packet.needs_self_correcting || source.packet.requires_hf_metadata_staging();
        let reconstructed_bytes = u64::from(packet_group.reconstructed_words(predictor_capacity)?)
            .checked_mul(4)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group reconstruction bytes",
            })?;
        let hf_lz77_bytes = source
            .hf_coefficients
            .as_ref()
            .and_then(|plan| plan.groups.get(index))
            .map(HfCoefficientGroupExecutionPlan::lz77_scratch_bytes)
            .or_else(|| {
                source
                    .deferred_hf
                    .as_ref()
                    .and_then(|plan| plan.groups.get(index))
                    .map(|group| group.lz77_scratch_bytes)
            })
            .unwrap_or(0);
        let hf_execution_state_bytes = source
            .hf_coefficients
            .as_ref()
            .and_then(|plan| plan.groups.get(index))
            .map(HfCoefficientGroupExecutionPlan::execution_state_bytes)
            .or_else(|| {
                source
                    .deferred_hf
                    .as_ref()
                    .and_then(|plan| plan.groups.get(index))
                    .map(|group| group.execution_state_bytes)
            })
            .unwrap_or(0);
        let reconstructed = storage(
            "jxl-wgpu VarDCT LF-group reconstruction",
            reconstructed_bytes
                .checked_add(hf_lz77_bytes)
                .and_then(|bytes| bytes.checked_add(hf_execution_state_bytes))
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group reconstruction, HF LZ77, and execution-state bytes",
                })?,
            wgpu::BufferUsages::COPY_DST,
        );
        let raw_metadata = storage(
            "jxl-wgpu VarDCT LF-group raw HF metadata",
            u64::from(group.control.capacities[1]) * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let coefficients = storage(
            "jxl-wgpu VarDCT LF-group coefficients",
            u64::from(packet_group.coefficient_words()) * 4,
            wgpu::BufferUsages::COPY_DST,
        );
        let packet_status = storage(
            "jxl-wgpu VarDCT LF-group packet status",
            PACKET_STATUS_BYTES,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let packet_control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group packet control"),
            contents: bytemuck::bytes_of(&group.control),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let modular = packet_group.entry.modular();
        let params = VarDctModularParams::default()
            .with_lz77_window(if group_specific_metadata {
                modular.map_or(0, |plan| plan.lz77_window_words)
            } else {
                packet_group.lz77_window_words
            })
            .with_self_correcting(if group_specific_metadata {
                modular.is_some_and(|plan| plan.needs_self_correcting)
            } else {
                source.packet.needs_self_correcting
            });
        let modular_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group Modular params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let artifact = storage(
            "jxl-wgpu VarDCT LF-group resident artifact",
            group.artifact_layout.artifact_bytes,
            wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let occupancy = storage(
            "jxl-wgpu VarDCT LF-group artifact occupancy",
            group.artifact_layout.occupancy_bytes,
            wgpu::BufferUsages::COPY_DST,
        );
        let artifact_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu VarDCT LF-group artifact params"),
            contents: bytemuck::bytes_of(&group.artifact_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        group_buffers.push(VarDctGroupJobBuffers {
            reconstructed,
            raw_metadata,
            coefficients,
            packet_status,
            packet_control,
            modular_params,
            artifact,
            occupancy,
            artifact_uniform,
        });
    }
    let lf_temporary = (source.memory.lf_temporary_bytes != 0).then(|| {
        storage(
            "jxl-wgpu VarDCT dequantized LF temporary",
            source.memory.lf_temporary_bytes,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        )
    });
    let mut resource_values = source.resource_layout.initial_values()?;
    if let Some(words) = source
        .packet
        .hf_coefficients
        .as_ref()
        .and_then(|entropy| entropy.dequant_matrix_words.as_deref())
    {
        source
            .resource_layout
            .install_dequant_matrix_words(&mut resource_values, words)?;
    }
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT resource vectors"),
        contents: bytemuck::cast_slice(&resource_values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let hf_coefficient_buffers = source
        .hf_coefficients
        .as_ref()
        .map(|plan| create_hf_coefficient_job_buffers(device, plan));
    let image_labels = [
        "jxl-wgpu VarDCT X plane",
        "jxl-wgpu VarDCT Y plane",
        "jxl-wgpu VarDCT B plane",
    ];
    let resident_planes = source.output.is_color().then(|| {
        std::array::from_fn(|channel| {
            storage(
                image_labels[channel],
                source.memory.resident_plane_bytes[channel],
                wgpu::BufferUsages::COPY_DST,
            )
        })
    });
    let pre_restoration_planes = (source.memory.pre_restoration_upsample_bytes != 0).then(|| {
        let labels = [
            "jxl-wgpu VarDCT pre-restoration X plane",
            "jxl-wgpu VarDCT pre-restoration Y plane",
            "jxl-wgpu VarDCT pre-restoration B plane",
        ];
        let shifted_channels = source
            .packet
            .profile
            .channel_shifts
            .into_iter()
            .filter(|shift| shift.is_subsampled())
            .count() as u64;
        let full_plane_bytes = source.memory.pre_restoration_upsample_bytes / shifted_channels;
        std::array::from_fn(|channel| {
            if source.packet.profile.channel_shifts[channel].is_subsampled() {
                storage(
                    labels[channel],
                    full_plane_bytes,
                    wgpu::BufferUsages::empty(),
                )
            } else {
                resident_planes.as_ref().expect("color planes")[channel].clone()
            }
        })
    });
    let restoration_planes = (source.gaborish.is_some() || source.epf.is_some()).then(|| {
        let labels = [
            "jxl-wgpu VarDCT restoration scratch X plane",
            "jxl-wgpu VarDCT restoration scratch Y plane",
            "jxl-wgpu VarDCT restoration scratch B plane",
        ];
        std::array::from_fn(|channel| {
            storage(
                labels[channel],
                source.memory.restoration_scratch_bytes / 3,
                wgpu::BufferUsages::empty(),
            )
        })
    });
    let frame_upsample_planes = source.frame_upsample.as_ref().map(|_| {
        std::array::from_fn(|_| {
            storage(
                "jxl-wgpu VarDCT upsampled frame plane",
                source.memory.frame_upsample_bytes / 3,
                wgpu::BufferUsages::empty(),
            )
        })
    });
    let frame_upsample_weights = source
        .frame_upsample
        .as_ref()
        .map(|kernel| kernel.upload(device))
        .transpose()?;
    let epf_sigma = source.epf.as_ref().map(|_| {
        storage(
            "jxl-wgpu VarDCT EPF inverse-sigma plane",
            source.memory.epf_sigma_bytes,
            wgpu::BufferUsages::COPY_DST,
        )
    });
    let mut output_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    if backend.direct_readback_enabled() {
        output_usage |= wgpu::BufferUsages::MAP_READ;
    }
    let output = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT packed output"),
        size: source.memory.output_lease_bytes,
        usage: output_usage,
        mapped_at_creation: false,
    }));
    let status_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu VarDCT aggregate validation staging"),
        size: source.memory.validation_staging_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut packet_commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jxl-wgpu bounded VarDCT packet stage"),
    });
    if let (Some(plan), Some(prefix), Some(arena)) = (
        source.packet.extra_channels.as_ref(),
        source.global_extra_prefix.as_ref(),
        extra_frame.as_ref(),
    ) {
        crate::modular_assembly::encode_plane_copies(
            &mut packet_commands,
            prefix.as_wgpu_buffer(),
            arena.as_wgpu_buffer(),
            0,
            plan.global_targets()
                .iter()
                .copied()
                .map(|layout| (layout, layout)),
        )
        .map_err(map_extra_error)?;
    }
    if let Some(lf_temporary) = &lf_temporary {
        packet_commands.clear_buffer(lf_temporary, 0, None);
    }
    if let Some(planes) = &resident_planes {
        for buffer in planes {
            packet_commands.clear_buffer(buffer, 0, None);
        }
    }
    packet_commands.clear_buffer(output.as_ref(), 0, None);
    for group in &group_buffers {
        for buffer in [
            &group.reconstructed,
            &group.raw_metadata,
            &group.coefficients,
            &group.packet_status,
            &group.artifact,
            &group.occupancy,
        ] {
            packet_commands.clear_buffer(buffer, 0, None);
        }
    }
    if let Some(buffers) = &hf_coefficient_buffers {
        for group in &buffers.groups {
            packet_commands.clear_buffer(&group.status, 0, None);
        }
    }
    if let Some(sigma) = &epf_sigma {
        packet_commands.clear_buffer(sigma, 0, None);
    }
    let (packet_stage_commands, packet_batches, mut commands) = if staged_lf_extras {
        packet_commands.clear_buffer(&status_staging, 0, None);
        (
            Some(PacketCommands::Whole(packet_commands.finish())),
            None,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu VarDCT downstream after LF extras"),
            }),
        )
    } else if let Some(plan) = &source.packet_windows {
        let controls = source
            .groups
            .iter()
            .map(|group| group.control)
            .collect::<Vec<_>>();
        let staged = staged_lf || staged_hf_global;
        let submissions = prepare_packet_windows(
            plan,
            &source.codestream,
            &controls,
            staged,
            Some(packet_commands),
        )?;
        let commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu VarDCT downstream after packet windows"),
        });
        if staged {
            (Some(PacketCommands::Windowed(submissions)), None, commands)
        } else {
            (None, Some(submissions), commands)
        }
    } else {
        for (index, buffers) in group_buffers.iter().enumerate() {
            let metadata = if group_specific_metadata {
                modular_metadata
                    .get(index)
                    .ok_or(VarDctDecodeError::GroupPlanCount {
                        component: "LF-local Modular metadata",
                        expected: group_buffers.len(),
                        actual: modular_metadata.len(),
                    })?
            } else {
                modular_metadata
                    .first()
                    .ok_or(VarDctDecodeError::GroupPlanCount {
                        component: "global Modular metadata",
                        expected: 1,
                        actual: 0,
                    })?
            };
            let buffers = VarDctPacketBuffers {
                codestream: &codestream_buffer,
                modular_metadata: metadata,
                reconstructed_lf: &buffers.reconstructed,
                raw_hf_metadata: &buffers.raw_metadata,
                coefficients: &buffers.coefficients,
                status: &buffers.packet_status,
                control: &buffers.packet_control,
                modular_params: &buffers.modular_params,
            };
            if source.packet.profile.uses_lf_frame {
                if staged_hf_global {
                    pipelines
                        .packet
                        .encode_hf_metadata(device, &mut packet_commands, buffers);
                } else {
                    pipelines
                        .packet
                        .encode_hf(device, &mut packet_commands, buffers);
                }
            } else if staged_lf || staged_hf_global {
                pipelines
                    .packet
                    .encode_lf(device, &mut packet_commands, buffers);
            } else {
                pipelines
                    .packet
                    .encode(device, &mut packet_commands, buffers);
            }
        }
        if staged_lf || staged_hf_global {
            for (index, buffers) in group_buffers.iter().enumerate() {
                packet_commands.copy_buffer_to_buffer(
                    &buffers.packet_status,
                    0,
                    &status_staging,
                    u64::try_from(index).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                        field: "staged packet status index",
                    })? * PACKET_STATUS_BYTES,
                    PACKET_STATUS_BYTES,
                );
            }
        }
        if staged_lf || staged_hf_global {
            (
                Some(PacketCommands::Whole(packet_commands.finish())),
                None,
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded VarDCT downstream stage"),
                }),
            )
        } else {
            (None, None, packet_commands)
        }
    };
    let [blocks_x, blocks_y] = source.packet.block_extent();
    let external_lf = source.external_lf.clone();
    let (resource_uniforms, adaptive_lf_uniform, progressive_dc_uniform) =
        if source.packet.profile.uses_lf_frame {
            let planes = external_lf
                .as_ref()
                .ok_or(VarDctDecodeError::MissingProgressiveDcSource)?;
            let uniform = pipelines.progressive_dc.encode_pack(
                device,
                &mut commands,
                ProgressiveDcPackInputs {
                    planes,
                    resources: resident_binding(&resources)?,
                    lf_offset: source.resource_layout.lf_offsets[0],
                    lf_stride: blocks_x,
                },
            )?;
            (Vec::new(), None, Some(uniform))
        } else {
            let lf_destination = if source.packet.profile.adaptive_lf_smoothing {
                lf_temporary
                    .as_ref()
                    .ok_or(VarDctDecodeError::EngineContract {
                        detail: "adaptive LF smoothing has no temporary buffer",
                    })?
            } else {
                &resources
            };
            let mut resource_uniforms = Vec::with_capacity(source.groups.len());
            for (group, buffers) in source.groups.iter().zip(&group_buffers) {
                resource_uniforms.push(pipelines.resource.encode(
                    device,
                    &mut commands,
                    VarDctResourceBuffers {
                        quantized_lf: &buffers.reconstructed,
                        dequantized_lf: lf_destination,
                    },
                    group.resource_params,
                ));
            }
            let smoothing_thresholds = source
                .groups
                .first()
                .ok_or(VarDctDecodeError::GroupPlanCount {
                    component: "packet source",
                    expected: 1,
                    actual: 0,
                })?
                .resource_params
                .smoothing_thresholds();
            let adaptive_lf_uniform = if source.packet.profile.adaptive_lf_smoothing {
                Some(pipelines.adaptive_lf.encode(
                    device,
                    &mut commands,
                    AdaptiveLfBuffers {
                        input: lf_destination,
                        output: &resources,
                    },
                    AdaptiveLfParams::new(
                        blocks_x,
                        blocks_y,
                        0,
                        source.resource_layout.lf_offsets[0],
                        smoothing_thresholds,
                    ),
                ))
            } else {
                None
            };
            (resource_uniforms, adaptive_lf_uniform, None)
        };
    for buffers in &group_buffers {
        pipelines.artifact.encode(
            device,
            &mut commands,
            HfMetadataLoweringBuffers {
                raw_metadata: &buffers.raw_metadata,
                artifact: &buffers.artifact,
                occupancy: &buffers.occupancy,
                resources: &resources,
                params: &buffers.artifact_uniform,
            },
        );
    }
    let mut epf_sigma_uniforms = Vec::new();
    match (source.epf.as_ref(), epf_sigma.as_ref()) {
        (Some(plan), Some(sigma)) => {
            if plan.sigma_groups.len() != group_buffers.len() {
                return Err(VarDctDecodeError::GroupPlanCount {
                    component: "EPF sigma",
                    expected: group_buffers.len(),
                    actual: plan.sigma_groups.len(),
                });
            }
            epf_sigma_uniforms.reserve(plan.sigma_groups.len());
            for (&config, buffers) in plan.sigma_groups.iter().zip(&group_buffers) {
                epf_sigma_uniforms.push(pipelines.epf_sigma.encode(
                    device,
                    &mut commands,
                    &buffers.raw_metadata,
                    &buffers.artifact,
                    sigma,
                    config,
                )?);
            }
        }
        (None, None) => {}
        _ => unreachable!("EPF plan and sigma buffer are constructed together"),
    }
    let deferred_before_coefficients = if source.deferred_hf.is_some() {
        let before = commands.finish();
        commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu deferred HF-global post-coefficient stage"),
        });
        Some(before)
    } else {
        None
    };
    let mut windowed_before_coefficients = None;
    let mut windowed_coefficient_batches = Vec::new();
    if let (Some(plan), Some(buffers)) = (
        source.hf_coefficients.as_ref(),
        hf_coefficient_buffers.as_ref(),
    ) {
        if plan.uses_bounded_stream_windows() {
            windowed_before_coefficients = Some(commands.finish());
            commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT post-coefficient stage"),
            });
            windowed_coefficient_batches = prepare_hf_batches(&source.codestream, plan)?;
        } else {
            for ((group_plan, hf_buffers), group_buffers) in
                plan.groups.iter().zip(&buffers.groups).zip(&group_buffers)
            {
                let params =
                    hf_buffers
                        .params
                        .as_ref()
                        .ok_or(VarDctDecodeError::EntropyWindowContract {
                            detail: "whole-range AC plan has no parameter buffer",
                        })?;
                pipelines.hf_coefficients.encode(
                    device,
                    &mut commands,
                    HfCoefficientBuffers {
                        codestream: &codestream_buffer,
                        entropy_bundle: &buffers.entropy_bundle,
                        reconstruction: &group_buffers.reconstructed,
                        params,
                        status: &hf_buffers.status,
                        artifact: &group_buffers.artifact,
                        order_table: &buffers.order_table,
                        coefficients: &group_buffers.coefficients,
                        sink_params: &hf_buffers.sink_params,
                    },
                    u32::try_from(group_plan.params.len()).map_err(|_| {
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "LF-group HF pass-group dispatch count",
                        }
                    })?,
                );
            }
        }
    }
    let mut extra_coefficient_commands = extra_frame.as_ref().map(|_| {
        std::mem::replace(
            &mut commands,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu assembled extra channels and frame output"),
            }),
        )
    });
    let mut extra_uniforms = if let (Some(plan), Some(arena)) =
        (source.packet.extra_channels.as_ref(), extra_frame.as_ref())
    {
        pipelines
            .raw_hf_dequant
            .modular()
            .encode_inverse(
                backend,
                &mut commands,
                arena.as_wgpu_buffer(),
                &plan.inverse,
                plan.wp_header,
            )
            .map_err(map_extra_error)?
    } else {
        Vec::new()
    };
    let padded_width = blocks_x
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output width",
        })?;
    let mut resident_scratch = Vec::with_capacity(source.groups.len());
    let rendered_extra =
        source
            .extra_render
            .as_ref()
            .map(|plan| {
                let extra = source.extra_planes.first().ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "resampled output lacks its extra plane",
                    },
                )?;
                let buffers = plan.allocate(device)?;
                let planes = source
                    .extra_planes
                    .iter()
                    .map(|extra| {
                        crate::modular_sample::ModularOutputPlane::new(extra.plane, extra.encoding)
                    })
                    .collect::<Vec<_>>();
                extra_uniforms.extend(pipelines.modular_render.encode(
                    device,
                    &mut commands,
                    plan,
                    &buffers,
                    resident_binding(extra.arena.as_wgpu_buffer())?,
                    &planes,
                )?);
                Ok::<_, VarDctDecodeError>(buffers)
            })
            .transpose()?;
    let (output_scratch, post_transform_buffers, lf_planes) = match source.output {
        VarDctFrameOutput::Color { config, plan } => {
            let resident_planes =
                resident_planes
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "color output lacks its resident planes",
                    })?;
            let padded_height =
                blocks_y
                    .checked_mul(8)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "padded output height",
                    })?;
            let correlation_width = source.packet.profile.width.div_ceil(64);
            let correlation_height = source.packet.profile.height.div_ceil(64);

            for ((packet_group, group), buffers) in source
                .packet
                .groups
                .iter()
                .zip(&source.groups)
                .zip(&group_buffers)
            {
                resident_scratch.push(pipelines.renderer.encode(
                    device,
                    &mut commands,
                    ResidentVarDctInputs {
                        coefficients: resident_binding(&buffers.coefficients)?,
                        artifact: resident_binding(&buffers.artifact)?,
                        resources: resident_binding(&resources)?,
                        outputs: resident_shifted_image_planes(
                            resident_planes,
                            padded_width,
                            padded_height,
                            source.packet.profile.channel_shifts,
                        )?,
                        indirect: &buffers.artifact,
                        indirect_base_offset: u64::from(
                            group.artifact_layout.indirect_offset_words,
                        ) * 4,
                        config: ResidentVarDctRenderConfig {
                            task_capacity: packet_group.task_capacity,
                            scratch_scalars: packet_group.coefficient_words(),
                            task_word_offset: group.artifact_layout.tasks_offset_words,
                            bucket_word_offset: group.artifact_layout.buckets_offset_words,
                            quant_offset: group.quant_offset,
                            correlation_offset: source.resource_layout.correlation_offset,
                            lf_offsets: source.resource_layout.lf_offsets,
                            lf_strides: source.resource_layout.lf_strides,
                            correlation_width,
                            correlation_height,
                            quant_biases: source.quant_biases,
                        },
                    },
                )?);
            }
            let image_width = source.packet.profile.width;
            let image_height = source.packet.profile.height;
            let mut pre_restoration_uniforms = Vec::new();
            if let Some(upsampled) = &pre_restoration_planes {
                for channel in 0..3 {
                    let shift = source.packet.profile.channel_shifts[channel];
                    if !shift.is_subsampled() {
                        continue;
                    }
                    let [input_width, input_height] = shift
                        .shifted_extent(image_width, image_height)
                        .ok_or(VarDctDecodeError::ArithmeticOverflow {
                            field: "pre-restoration input extent",
                        })?;
                    pre_restoration_uniforms.push(pipelines.chroma_upsample.encode(
                        device,
                        &mut commands,
                        ResidentChromaUpsampleInputs {
                            input: ResidentF32Plane {
                                storage: resident_binding(&resident_planes[channel])?,
                                width: input_width,
                                height: input_height,
                                stride: padded_width >> shift.horizontal,
                            },
                            output: ResidentF32Plane {
                                storage: resident_binding(&upsampled[channel])?,
                                width: image_width,
                                height: image_height,
                                stride: padded_width,
                            },
                            shift: ResidentChromaShift {
                                horizontal: shift.horizontal != 0,
                                vertical: shift.vertical != 0,
                            },
                        },
                    )?);
                }
            }
            let restoration_source = pre_restoration_planes.as_ref().unwrap_or(resident_planes);
            let mut restoration = restoration_planes
                .as_ref()
                .map(|scratch| RestorationCursor::new(restoration_source, scratch));
            let gaborish_uniform = match (source.gaborish, restoration.as_mut()) {
                (Some(weights), Some(restoration)) => {
                    let (input_buffers, output_buffers) = restoration.advance();
                    let uniform = pipelines.gaborish.encode(
                        device,
                        &mut commands,
                        ResidentGaborishInputs {
                            inputs: resident_image_planes(
                                input_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            outputs: resident_image_planes(
                                output_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            weights,
                        },
                    )?;
                    Some(uniform)
                }
                (None, _) => None,
                (Some(_), None) => unreachable!("Gaborish requires restoration scratch planes"),
            };
            let mut epf_uniforms =
                Vec::with_capacity(source.epf.as_ref().map_or(0, |plan| plan.passes.len()));
            if let Some(epf) = &source.epf {
                let restoration = restoration
                    .as_mut()
                    .unwrap_or_else(|| unreachable!("EPF requires restoration scratch planes"));
                let sigma_buffer = epf_sigma
                    .as_ref()
                    .unwrap_or_else(|| unreachable!("EPF requires a sigma plane"));
                let sigma = ResidentF32Plane {
                    storage: resident_binding(sigma_buffer)?,
                    width: blocks_x,
                    height: blocks_y,
                    stride: blocks_x,
                };
                for &parameters in &epf.passes {
                    let (input_buffers, output_buffers) = restoration.advance();
                    epf_uniforms.push(pipelines.epf.encode(
                        device,
                        &mut commands,
                        ResidentEpfInputs {
                            inputs: resident_image_planes(
                                input_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            outputs: resident_image_planes(
                                output_buffers,
                                image_width,
                                image_height,
                                padded_width,
                            )?,
                            sigma: jxl_wgpu::ResidentEpfSigma::Plane(sigma),
                            parameters,
                        },
                    )?);
                }
            }
            let presentation_planes = restoration
                .as_ref()
                .map_or(restoration_source, RestorationCursor::current);
            let mut frame_upsample_uniforms = Vec::new();
            if let (Some(upsampled), Some(weights)) =
                (&frame_upsample_planes, &frame_upsample_weights)
            {
                for channel in 0..3 {
                    frame_upsample_uniforms.push(pipelines.frame_upsample.encode(
                        device,
                        &mut commands,
                        ResidentUpsampleInputs {
                            input: ResidentF32Plane {
                                storage: resident_binding(&presentation_planes[channel])?,
                                width: image_width,
                                height: image_height,
                                stride: padded_width,
                            },
                            output: ResidentF32Plane {
                                storage: resident_binding(&upsampled[channel])?,
                                width: source.packet.profile.output_width,
                                height: source.packet.profile.output_height,
                                stride: source.packet.profile.output_width,
                            },
                            weights,
                        },
                    )?);
                }
            }
            let presentation_planes = frame_upsample_planes
                .as_ref()
                .unwrap_or(presentation_planes);
            let presentation_shifts = if restoration.is_some() || frame_upsample_planes.is_some() {
                [crate::vardct_frontend::VarDctChannelShift::default(); 3]
            } else {
                source.packet.profile.channel_shifts
            };
            let output_width = source.packet.profile.output_width;
            let output_height = source.packet.profile.output_height;
            let presentation_stride = if frame_upsample_planes.is_some() {
                output_width
            } else {
                padded_width
            };
            let presentation_geometry = presentation_shifts.map(|shift| {
                shift.shifted_extent(output_width, output_height).ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "presentation channel extent",
                    },
                )
            });
            let [geometry_x, geometry_y, geometry_b] = presentation_geometry;
            let presentation_geometry = [geometry_x?, geometry_y?, geometry_b?];
            let presentation_strides = presentation_shifts.map(|shift| {
                presentation_stride.checked_shr(shift.horizontal).ok_or(
                    VarDctDecodeError::ArithmeticOverflow {
                        field: "presentation channel stride",
                    },
                )
            });
            let [stride_x, stride_y, stride_b] = presentation_strides;
            let presentation_strides = [stride_x?, stride_y?, stride_b?];
            let output_scratch = pipelines.output.encode(
                device,
                &mut commands,
                ColorOutputInputs {
                    alpha: if source.surface.is_some() {
                        None
                    } else if let (Some(plan), Some(buffers)) =
                        (&source.extra_render, &rendered_extra)
                    {
                        let plane = plan.planes()[0];
                        Some(crate::color_output::ColorOutputAlpha {
                            domain: crate::ModularSampleDomain::DecodedF32,
                            storage: resident_binding(&buffers.output)?,
                            width: plane.layout.width,
                            height: plane.layout.height,
                            stride: plane.layout.row_stride_words,
                            word_offset: plane.layout.word_offset,
                            sample_bit_depth: plane.encoding.depth(),
                        })
                    } else {
                        source
                            .extra_planes
                            .first()
                            .map(super::staging::ResidentModularPlane::alpha_binding)
                            .transpose()?
                    },
                    planes: [
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[0])?,
                            width: presentation_geometry[0][0],
                            height: presentation_geometry[0][1],
                            stride: presentation_strides[0],
                        },
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[1])?,
                            width: presentation_geometry[1][0],
                            height: presentation_geometry[1][1],
                            stride: presentation_strides[1],
                        },
                        ColorOutputPlane {
                            storage: resident_binding(&presentation_planes[2])?,
                            width: presentation_geometry[2][0],
                            height: presentation_geometry[2][1],
                            stride: presentation_strides[2],
                        },
                    ],
                    output: resident_binding(&output)?,
                    layout: &source.layout,
                    config,
                },
            )?;
            debug_assert_eq!(output_scratch.plan, plan);
            // An LF slot contains the complete pre-color-transform image, including restoration
            // and frame upsampling. Retain only these final allocations after validation.
            let lf_planes = if source.packet.profile.lf_level != 0 {
                let mut tracked = |buffer: &wgpu::Buffer| {
                    let permit = permits
                        .transient
                        .split_off(buffer.size())
                        .map_err(ProgressiveDcGpuError::from)?;
                    Ok::<_, VarDctDecodeError>(GpuBufferLease::from_tracked(buffer.clone(), permit))
                };
                Some(ProgressiveDcXybPlanes::from_leases(
                    [
                        tracked(&presentation_planes[0])?,
                        tracked(&presentation_planes[1])?,
                        tracked(&presentation_planes[2])?,
                    ],
                    output_width,
                    output_height,
                    presentation_stride,
                )?)
            } else {
                None
            };
            let post_transform_buffers = PostTransformJobBuffers {
                _restoration_planes: restoration_planes,
                _pre_restoration_planes: pre_restoration_planes,
                _pre_restoration_uniforms: pre_restoration_uniforms,
                _gaborish_uniform: gaborish_uniform,
                _epf_sigma: epf_sigma,
                _epf_sigma_uniforms: epf_sigma_uniforms,
                _epf_uniforms: epf_uniforms,
                _frame_upsample_planes: frame_upsample_planes,
                _frame_upsample_weights: frame_upsample_weights,
                _frame_upsample_uniforms: frame_upsample_uniforms,
            };
            (
                FrameOutputScratch::Color {
                    _scratch: output_scratch,
                },
                post_transform_buffers,
                lf_planes,
            )
        }
        VarDctFrameOutput::Extra { index, plan } => {
            let extra =
                source
                    .extra_planes
                    .first()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "scalar output lacks its selected resident extra plane",
                    })?;
            if plan.config.encoding != extra.encoding || index != extra.index {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "scalar output precision differs from the selected extra plane",
                });
            }
            let (plane, domain, arena) =
                if let (Some(plan), Some(buffers)) = (&source.extra_render, &rendered_extra) {
                    (
                        plan.planes()[0].layout,
                        crate::ModularSampleDomain::DecodedF32,
                        &buffers.output,
                    )
                } else {
                    (
                        extra.plane,
                        crate::ModularSampleDomain::Encoded,
                        extra.arena.as_wgpu_buffer(),
                    )
                };
            let scratch = pipelines.scalar_output.encode(
                device,
                &mut commands,
                plan,
                crate::modular_scalar_output::ModularScalarOutputInputs {
                    plane,
                    domain,
                    arena: resident_binding(arena)?,
                    output: resident_binding(&output)?,
                },
            )?;
            (
                FrameOutputScratch::Extra { scratch },
                PostTransformJobBuffers::default(),
                None,
            )
        }
    };
    if let Some(surface) = &source.surface
        && !surface.extras.is_empty()
    {
        let plan =
            source
                .extra_render
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "frame surface lacks normalized extra planes",
                })?;
        let buffers = rendered_extra
            .as_ref()
            .ok_or(VarDctDecodeError::EntropyWindowContract {
                detail: "frame surface lacks normalized extra storage",
            })?;
        if plan.planes().len() != surface.extras.len() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "frame surface extra plane count changed",
            });
        }
        for (plane, layout) in plan.planes().iter().zip(&surface.extras) {
            let plane = plane.layout;
            commands.copy_buffer_to_buffer(
                &buffers.output,
                u64::from(plane.word_offset) * 4,
                &output,
                layout.planes[0].offset,
                u64::from(plane.width) * u64::from(plane.height) * 4,
            );
        }
    }
    let packet_status_end = source.memory.packet_status_bytes;
    let artifact_status_end = packet_status_end
        .checked_add(
            u64::try_from(group_buffers.len())
                .map_err(|_| VarDctDecodeError::ArithmeticOverflow {
                    field: "LF-group status count",
                })?
                .checked_mul(ARTIFACT_STATUS_BYTES)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "artifact status staging bytes",
                })?,
        )
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "artifact status staging end",
        })?;
    for (index, (group, buffers)) in source.groups.iter().zip(&group_buffers).enumerate() {
        let index = u64::try_from(index).map_err(|_| VarDctDecodeError::ArithmeticOverflow {
            field: "LF-group status index",
        })?;
        commands.copy_buffer_to_buffer(
            &buffers.packet_status,
            0,
            &status_staging,
            index * PACKET_STATUS_BYTES,
            PACKET_STATUS_BYTES,
        );
        commands.copy_buffer_to_buffer(
            &buffers.artifact,
            u64::from(group.artifact_layout.status_offset_words) * 4,
            &status_staging,
            packet_status_end + index * ARTIFACT_STATUS_BYTES,
            ARTIFACT_STATUS_BYTES,
        );
    }
    if let Some(buffers) = &hf_coefficient_buffers {
        let mut offset = artifact_status_end;
        for group in &buffers.groups {
            let status_bytes = group.status.size();
            commands.copy_buffer_to_buffer(&group.status, 0, &status_staging, offset, status_bytes);
            if let Some(ac) = &mut extra_coefficient_commands {
                ac.copy_buffer_to_buffer(&group.status, 0, &status_staging, offset, status_bytes);
            }
            offset =
                offset
                    .checked_add(status_bytes)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF status staging offset",
                    })?;
        }
        debug_assert_eq!(
            offset + output_scratch.status_bytes(),
            source.memory.validation_staging_bytes
        );
    }
    output_scratch.copy_status(&mut commands, &status_staging);

    let mut after_coefficients = commands.finish();
    let extra_output_commands = extra_coefficient_commands
        .map(|ac| std::mem::replace(&mut after_coefficients, ac.finish()));
    let (downstream_commands, mut deferred_commands) =
        if let Some(before_coefficients) = deferred_before_coefficients {
            debug_assert!(windowed_before_coefficients.is_none());
            debug_assert!(windowed_coefficient_batches.is_empty());
            (
                None,
                Some(DeferredHfGlobalCommands {
                    before_coefficients: Some(before_coefficients),
                    after_coefficients,
                }),
            )
        } else {
            let downstream = if let Some(before_coefficients) = windowed_before_coefficients {
                VarDctDownstreamCommands::Windowed {
                    before_coefficients,
                    coefficient_batches: windowed_coefficient_batches,
                    device: device.clone(),
                    pipelines: Arc::clone(&pipelines),
                    after_coefficients,
                }
            } else {
                VarDctDownstreamCommands::Whole(after_coefficients)
            };
            (Some(downstream), None)
        };
    let lifetime = Arc::new(VarDctJobLifetime {
        output: GpuBufferLease::from_tracked(output.as_ref().clone(), permits.output),
        status_staging,
        status_mapped: AtomicBool::new(false),
        _transient_permits: Mutex::new(vec![permits.transient]),
        _codestream: codestream_buffer,
        _packet_stream_window: packet_stream_window,
        _modular_metadata: Mutex::new(modular_metadata),
        _groups: group_buffers,
        _lf_temporary: lf_temporary,
        _resources: resources,
        _resource_uniforms: resource_uniforms,
        _adaptive_lf_uniform: adaptive_lf_uniform,
        _progressive_dc_uniform: progressive_dc_uniform,
        _external_lf: external_lf,
        _extra_planes: std::mem::take(&mut source.extra_planes),
        extra_frame,
        _extra_prefix: source.global_extra_prefix.take(),
        _extra_uniforms: extra_uniforms,
        _rendered_extra: rendered_extra,
        _hf_coefficients: Mutex::new(hf_coefficient_buffers),
        _resident_planes: resident_planes,
        lf_planes,
        _post_transform: post_transform_buffers,
        _resident_scratch: resident_scratch,
        _output_scratch: output_scratch,
    });
    let mut expected_groups = Vec::with_capacity(source.packet.groups.len());
    for (group, group_source) in source.packet.groups.iter().zip(&source.groups) {
        let [group_blocks_x, group_blocks_y] = group.block_extent();
        let expected_blocks = group_blocks_x.checked_mul(group_blocks_y).ok_or(
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group validation block count",
            },
        )?;
        let correlation_samples = group
            .rect
            .width
            .div_ceil(64)
            .checked_mul(group.rect.height.div_ceil(64))
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "LF-group validation correlation samples",
            })?;
        expected_groups.push(VarDctGroupValidation {
            expected_lf_samples: if source.packet.profile.uses_lf_frame {
                0
            } else {
                group_source
                    .resource_params
                    .source_geometry
                    .into_iter()
                    .try_fold(0u32, |total, [width, height, _, _]| {
                        width
                            .checked_mul(height)
                            .and_then(|samples| total.checked_add(samples))
                    })
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "LF-group validation sample count",
                    })?
            },
            expected_coefficients: group.coefficient_words(),
            expected_blocks,
            correlation_samples,
            task_capacity: group.task_capacity,
            expected_global_scale: source.packet.global_scale,
            expected_quant_lf: source.packet.quant_lf,
            expected_extra_precision: group.extra_precision(),
        });
    }
    let expected_hf = source
        .hf_coefficients
        .as_ref()
        .map(|plan| HfValidation::plan(&source, plan))
        .transpose()
        .map_err(map_extra_error)?
        .unwrap_or_default();
    let layout = source.layout.clone();
    let frame_name = source.frame_name.clone();
    let mut pending = FramePendingFrame {
        backend: backend.clone(),
        pipelines,
        memory,
        runtime_stats,
        lifetime: Some(Arc::clone(&lifetime)),
        stage: VarDctPendingStage::Final {
            completion: Arc::new(MapCompletion::default()),
        },
        token: SubmissionToken(1),
        layout,
        surface: source.surface.clone(),
        frame_name,
        expected_groups,
        expected_hf,
        extra_output_commands,
        hf_metadata_stop: staged_hf_global,
    };
    if source.packet.pending_raw_hf_dequant_side_image().is_some()
        && packet_stage_commands.is_none()
    {
        if packet_batches.is_some() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "raw HF dequant side images require packet completion before their first window",
            });
        }
        if downstream_commands.is_some() {
            return Err(VarDctDecodeError::EngineContract {
                detail: "raw HF dequant side image unexpectedly has eager downstream commands",
            });
        }
        let commands = deferred_commands
            .take()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "raw HF dequant side image has no deferred coefficient commands",
            })?;
        pending.start_raw_hf_dequant_stage(Box::new(source), commands, Some(poll_permit))?;
        return Ok(pending);
    }
    let completion = Arc::new(MapCompletion::default());
    let (submission, local_commands, deferred_commands) =
        if let Some(packet_stage_commands) = packet_stage_commands {
            let submission = submit_packet_commands(
                backend,
                &pending.pipelines,
                packet_stage_commands,
                &lifetime,
            )?;
            if staged_hf_global || source.packet.pending_raw_hf_dequant_side_image().is_some() {
                if source.packet.profile.uses_lf_frame && !staged_lf_extras {
                    (submission, None, deferred_commands)
                } else {
                    let deferred = deferred_commands.ok_or(VarDctDecodeError::EngineContract {
                        detail: "regular fused packet has no deferred HF-global commands",
                    })?;
                    (
                        submission,
                        Some(PostLfCommands::DeferredHfGlobal(deferred)),
                        None,
                    )
                }
            } else {
                (
                    submission,
                    downstream_commands.map(PostLfCommands::Direct),
                    None,
                )
            }
        } else if let Some(batches) = packet_batches {
            let downstream = downstream_commands.ok_or(VarDctDecodeError::EngineContract {
                detail: "windowed packet execution is missing downstream commands",
            })?;
            (
                submit_packet_batches(
                    backend,
                    &pending.pipelines,
                    batches,
                    Some(downstream),
                    &lifetime,
                )?,
                None,
                None,
            )
        } else {
            let downstream = downstream_commands.ok_or(VarDctDecodeError::EngineContract {
                detail: "VarDCT execution is missing downstream commands",
            })?;
            (
                submit_vardct_downstream(backend.queue(), Vec::new(), downstream, &lifetime)?,
                None,
                None,
            )
        };
    arm_status_map(
        &lifetime,
        &completion,
        if staged_lf_extras {
            "VarDCT LF-extra arena mapping"
        } else if staged_lf {
            "VarDCT LF cursor mapping"
        } else if staged_hf_global {
            "VarDCT HF-global cursor mapping"
        } else {
            "VarDCT validation mapping"
        },
    );
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll_permit.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(format!("VarDCT GPU poll registration failed: {error}")));
    }
    pending.stage = if let Some(commands) = local_commands {
        if staged_lf_extras {
            VarDctPendingStage::LfExtras {
                completion,
                source: Box::new(source),
                commands: Some(commands),
            }
        } else {
            VarDctPendingStage::LocalLf {
                completion,
                source: Box::new(source),
                commands: Some(commands),
            }
        }
    } else if let Some(commands) = deferred_commands {
        VarDctPendingStage::HfGlobal {
            completion,
            source: Box::new(source),
            commands: Some(commands),
        }
    } else {
        pending.after_coefficients_stage(completion, Box::new(source))
    };
    Ok(pending)
}

#[derive(Default)]
pub(super) struct MapCompletion {
    state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
struct MapState {
    result: Option<Result<(), String>>,
    waker: Option<Waker>,
}

impl MapCompletion {
    pub(super) fn complete(&self, result: Result<(), String>) {
        let waker = {
            let mut state = lock_unpoisoned(&self.state);
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

    pub(super) fn poll(&self, context: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.waker = Some(context.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(&self) -> Result<(), String> {
        let mut state = lock_unpoisoned(&self.state);
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state
            .result
            .take()
            .expect("mapping result was checked as present")
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn arm_status_map(
    lifetime: &Arc<VarDctJobLifetime>,
    completion: &Arc<MapCompletion>,
    stage: &'static str,
) {
    let callback_lifetime = Arc::clone(lifetime);
    let callback_completion = Arc::clone(completion);
    lifetime
        .status_staging
        .map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() {
                callback_lifetime
                    .status_mapped
                    .store(true, Ordering::Release);
            }
            drop(callback_lifetime);
            callback_completion
                .complete(result.map_err(|error| format!("{stage} failed: {error}")));
        });
}
