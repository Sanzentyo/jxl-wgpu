use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{
    ChangedRegions, Extent2d, OutputId, Region, SubmissionToken, TransformKind,
};
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, GpuImageOutput, MemoryBudget, MemoryBudgetSnapshot,
    MemoryPermit, ResidentChromaShift, ResidentChromaUpsampleInputs, ResidentEpfInputs,
    ResidentF32Plane, ResidentGaborishInputs, ResidentImageUpsampleInputs,
    ResidentImageUpsampleResources, ResidentStorageBinding, ResidentVarDctInputs,
    ResidentVarDctRenderConfig, ResidentVarDctScratch, SubmissionPollPermit,
    UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput, WgpuBackend,
};
use wgpu::util::DeviceExt;

use crate::progressive_dc::{
    ProgressiveDcGpuError, ProgressiveDcPackInputs, ProgressiveDcXybPlanes,
};
use crate::vardct_artifact::{GpuVarDctArtifactStatus, HfMetadataLoweringBuffers};
use crate::vardct_lf::{AdaptiveLfBuffers, AdaptiveLfParams};
use crate::vardct_output::{
    VarDctOutputConfig, VarDctOutputInputs, VarDctOutputPlane, VarDctOutputScratch,
    VarDctOutputTransform,
};
use crate::vardct_packet::{
    GpuVarDctPacketStatus, VarDctModularParams, VarDctPacketBuffers, VarDctPacketControl,
    VarDctPacketValidation,
};
use crate::vardct_pass_group::{
    GpuHfCoefficientStatus, HfCoefficientBuffers, HfCoefficientExecutionPlan,
    HfCoefficientGroupExecutionPlan,
};
use crate::vardct_resource::VarDctResourceBuffers;
use crate::wgpu_engine::{
    RawHfDequantSideImageJob, RawHfDequantSideImageStatus, raw_matrix_status_ok,
    raw_matrix_value_error,
};
use crate::{
    Error as DecodeError, FrameDuration, FrameMetadata, GpuCodestream, GpuPendingFrame,
    GpuSubmissionSession, Result as DecodeResult, SubmittedGpuFrame,
};

use super::pipeline::VarDctPipelines;
use super::restoration::RestorationCursor;
use super::source::{VarDctSource, check_limit};
use super::types::{
    ARTIFACT_STATUS_BYTES, AdaptiveLfDisposition, PACKET_STATUS_BYTES, VarDctDecodeError,
    VarDctDecodeMemoryStats,
};
use super::window_plan::{
    HfPacketWindowExecutionPlan, copy_stream_segment, map_codestream_source_error,
};

/// One-frame submission state for [`VarDctSubmissionEngine`].
pub struct VarDctDecodeSession {
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

impl VarDctDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> VarDctDecodeMemoryStats {
        self.memory_stats
    }

    #[must_use]
    pub fn adaptive_lf_signaled(&self) -> bool {
        self.source.as_ref().map_or(
            self.memory_stats.adaptive_lf_signaled,
            VarDctSource::adaptive_lf_signaled,
        )
    }

    #[must_use]
    pub fn adaptive_lf_disposition(&self) -> AdaptiveLfDisposition {
        self.source.as_ref().map_or(
            self.memory_stats.adaptive_lf_disposition,
            VarDctSource::adaptive_lf_disposition,
        )
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    #[must_use]
    pub fn submissions_per_frame(&self) -> usize {
        self.runtime_stats
            .submissions_per_frame
            .load(Ordering::Acquire)
    }

    /// Exact staged HF packet batch count once the LF cursor map has completed. It is zero before
    /// that dynamic plan exists and when every HF packet binds the retained whole codestream.
    #[must_use]
    pub fn hf_packet_stream_batch_count(&self) -> usize {
        self.runtime_stats
            .hf_packet_stream_batch_count
            .load(Ordering::Acquire)
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
        let [expected_width, expected_height] = source.packet.block_extent();
        for (plane, actual) in planes.planes.iter().enumerate() {
            if actual.width != expected_width || actual.height != expected_height {
                return Err(ProgressiveDcGpuError::PlaneExtent {
                    plane,
                    actual_width: actual.width,
                    actual_height: actual.height,
                    expected_width,
                    expected_height,
                }
                .into());
            }
        }
        source.external_lf = Some(planes);
        Ok(())
    }
}

impl std::fmt::Debug for VarDctDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctDecodeSession")
            .field("submitted", &self.source.is_none())
            .field("memory_stats", &self.memory_stats())
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for VarDctDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = VarDctPendingFrame;

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
        let transient_permit = self.memory.try_reserve(source.memory.transient_bytes)?;
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
            },
            poll_permit,
        )?;
        Ok(Some(pending))
    }
}

struct VarDctMemoryPermits {
    output: MemoryPermit,
    transient: MemoryPermit,
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

struct HfCoefficientBatchSubmission {
    stream_upload: Box<[u8]>,
    params_upload: Box<[u8]>,
    commands: wgpu::CommandBuffer,
}

struct LfPacketBatchSubmission {
    group_index: usize,
    stream_upload: Box<[u8]>,
    params: VarDctModularParams,
    commands: wgpu::CommandBuffer,
}

struct CombinedPacketGroupUpload {
    group_index: usize,
    params: VarDctModularParams,
}

struct CombinedPacketBatchSubmission {
    stream_upload: Box<[u8]>,
    groups: Box<[CombinedPacketGroupUpload]>,
    commands: wgpu::CommandBuffer,
}

struct HfPacketGroupUpload {
    group_index: usize,
    control: VarDctPacketControl,
    params: VarDctModularParams,
}

struct HfPacketBatchSubmission {
    stream_upload: Box<[u8]>,
    groups: Box<[HfPacketGroupUpload]>,
    commands: wgpu::CommandBuffer,
}

enum LfPacketCommands {
    Whole(wgpu::CommandBuffer),
    Windowed(Vec<LfPacketBatchSubmission>),
}

enum VarDctDownstreamCommands {
    Whole(wgpu::CommandBuffer),
    Windowed {
        before_coefficients: wgpu::CommandBuffer,
        coefficient_batches: Vec<HfCoefficientBatchSubmission>,
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
    LocalLf {
        source: Box<VarDctSource>,
        commands: PostLfCommands,
    },
    HfGlobal {
        source: Box<VarDctSource>,
        commands: DeferredHfGlobalCommands,
    },
    RawHfDequant {
        source: Box<VarDctSource>,
        commands: DeferredHfGlobalCommands,
        job: Box<RawHfDequantSideImageJob>,
        permit: MemoryPermit,
    },
}

fn submit_lf_packet_commands(
    queue: &wgpu::Queue,
    commands: LfPacketCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    match commands {
        LfPacketCommands::Whole(commands) => Ok(queue.submit([commands])),
        LfPacketCommands::Windowed(batches) => {
            let stream = lifetime._packet_stream_window.as_ref().ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed LF packet commands have no retained stream upload",
                },
            )?;
            let mut last_submission = None;
            for batch in batches {
                let group = lifetime._groups.get(batch.group_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed LF packet batch references an absent group",
                    },
                )?;
                queue.write_buffer(stream, 0, &batch.stream_upload);
                queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&batch.params));
                last_submission = Some(queue.submit([batch.commands]));
            }
            last_submission.ok_or(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed LF packet execution has no batches",
            })
        }
    }
}

fn write_combined_packet_batch(
    queue: &wgpu::Queue,
    stream: &wgpu::Buffer,
    batch: &CombinedPacketBatchSubmission,
    lifetime: &VarDctJobLifetime,
) -> Result<(), VarDctDecodeError> {
    queue.write_buffer(stream, 0, &batch.stream_upload);
    for upload in &batch.groups {
        let group = lifetime._groups.get(upload.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "windowed combined packet batch references an absent group",
            },
        )?;
        queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&upload.params));
    }
    Ok(())
}

fn submit_combined_packet_commands(
    queue: &wgpu::Queue,
    mut batches: Vec<CombinedPacketBatchSubmission>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed combined packet commands have no retained stream upload",
        },
    )?;
    let final_batch = batches
        .pop()
        .ok_or(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed combined packet execution has no batches",
        })?;
    for batch in batches {
        write_combined_packet_batch(queue, stream, &batch, lifetime)?;
        queue.submit([batch.commands]);
    }
    write_combined_packet_batch(queue, stream, &final_batch, lifetime)?;
    submit_vardct_downstream(queue, vec![final_batch.commands], downstream, lifetime)
}

fn write_hf_packet_batch(
    queue: &wgpu::Queue,
    stream: &wgpu::Buffer,
    batch: &HfPacketBatchSubmission,
    lifetime: &VarDctJobLifetime,
) -> Result<(), VarDctDecodeError> {
    queue.write_buffer(stream, 0, &batch.stream_upload);
    for upload in &batch.groups {
        let group = lifetime._groups.get(upload.group_index).ok_or(
            VarDctDecodeError::EntropyWindowContract {
                detail: "windowed HF packet batch references an absent group",
            },
        )?;
        queue.write_buffer(
            &group.packet_control,
            0,
            bytemuck::bytes_of(&upload.control),
        );
        queue.write_buffer(&group.modular_params, 0, bytemuck::bytes_of(&upload.params));
    }
    Ok(())
}

fn submit_hf_packet_commands(
    queue: &wgpu::Queue,
    mut batches: Vec<HfPacketBatchSubmission>,
    downstream: VarDctDownstreamCommands,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF packet commands have no retained stream upload",
        },
    )?;
    let final_batch = batches
        .pop()
        .ok_or(VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF packet execution has no batches",
        })?;
    for batch in batches {
        write_hf_packet_batch(queue, stream, &batch, lifetime)?;
        queue.submit([batch.commands]);
    }
    write_hf_packet_batch(queue, stream, &final_batch, lifetime)?;
    submit_vardct_downstream(queue, vec![final_batch.commands], downstream, lifetime)
}

fn submit_hf_metadata_packet_commands(
    queue: &wgpu::Queue,
    batches: Vec<HfPacketBatchSubmission>,
    lifetime: &VarDctJobLifetime,
) -> Result<wgpu::SubmissionIndex, VarDctDecodeError> {
    let stream = lifetime._packet_stream_window.as_ref().ok_or(
        VarDctDecodeError::EntropyWindowContract {
            detail: "windowed HF-metadata commands have no retained stream upload",
        },
    )?;
    let mut last_submission = None;
    for batch in batches {
        write_hf_packet_batch(queue, stream, &batch, lifetime)?;
        last_submission = Some(queue.submit([batch.commands]));
    }
    last_submission.ok_or(VarDctDecodeError::EntropyWindowContract {
        detail: "windowed HF-metadata execution has no batches",
    })
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
                queue.submit([batch.commands]);
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

struct RestorationJobBuffers {
    _planes: [wgpu::Buffer; 3],
    _gaborish_uniform: Option<wgpu::Buffer>,
    _epf_sigma: Option<wgpu::Buffer>,
    _epf_sigma_uniforms: Vec<wgpu::Buffer>,
    _epf_uniforms: Vec<wgpu::Buffer>,
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
    _hf_coefficients: Mutex<Option<HfCoefficientJobBuffers>>,
    _resident_planes: [wgpu::Buffer; 3],
    _component_upsample_planes: Option<[wgpu::Buffer; 3]>,
    _component_upsample_uniforms: Vec<wgpu::Buffer>,
    _restoration: Option<RestorationJobBuffers>,
    _frame_upsample_planes: Option<[wgpu::Buffer; 3]>,
    _frame_upsample_resources: Option<ResidentImageUpsampleResources>,
    _resident_scratch: Vec<ResidentVarDctScratch>,
    _output_scratch: VarDctOutputScratch,
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
    uniform_transform: Option<TransformKind>,
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
pub struct VarDctPendingFrame {
    pub(super) backend: WgpuBackend,
    pub(super) pipelines: Arc<VarDctPipelines>,
    pub(super) memory: MemoryBudget,
    pub(super) runtime_stats: Arc<VarDctRuntimeStats>,
    lifetime: Option<Arc<VarDctJobLifetime>>,
    stage: VarDctPendingStage,
    token: SubmissionToken,
    layout: ImageLayout,
    frame_name: String,
    expected_groups: Vec<VarDctGroupValidation>,
    expected_hf_group_indices: Vec<u32>,
    deferred_hf_global: bool,
    progressive_dc_extent: Extent2d,
    progressive_dc_stride: u32,
}

enum VarDctPendingStage {
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
        source: Box<VarDctSource>,
        commands: Option<DeferredHfGlobalCommands>,
        job: Option<Box<RawHfDequantSideImageJob>>,
        permit: Option<MemoryPermit>,
    },
    Final {
        completion: Arc<MapCompletion>,
    },
}

impl std::fmt::Debug for VarDctPendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VarDctPendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("lf_group_count", &self.expected_groups.len())
            .field(
                "stage",
                &match &self.stage {
                    VarDctPendingStage::LocalLf { .. } => "local-lf",
                    VarDctPendingStage::HfGlobal { .. } => "hf-global",
                    VarDctPendingStage::RawHfDequant { .. } => "raw-hf-dequant",
                    VarDctPendingStage::Final { .. } => "final",
                },
            )
            .finish_non_exhaustive()
    }
}

impl VarDctPendingFrame {
    pub(crate) fn submissions_per_frame_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.runtime_stats.submissions_per_frame)
    }

    #[must_use]
    pub(crate) fn dependency_submission_ready(&self) -> bool {
        matches!(self.stage, VarDctPendingStage::Final { .. })
    }

    pub(crate) fn progressive_dc_planes(
        &self,
    ) -> Result<ProgressiveDcXybPlanes, VarDctDecodeError> {
        if matches!(
            self.stage,
            VarDctPendingStage::LocalLf { .. }
                | VarDctPendingStage::HfGlobal { .. }
                | VarDctPendingStage::RawHfDequant { .. }
        ) {
            return Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted);
        }
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        ProgressiveDcXybPlanes::from_buffers(
            lifetime._resident_planes.clone(),
            self.progressive_dc_extent.width,
            self.progressive_dc_extent.height,
            self.progressive_dc_stride,
        )
        .map_err(Into::into)
    }

    /// Same-queue, budget-tracked access before packet/artifact status becomes authoritative.
    pub fn unvalidated_gpu_frame(&self) -> DecodeResult<UnvalidatedGpuImageFrame> {
        if matches!(
            self.stage,
            VarDctPendingStage::LocalLf { .. }
                | VarDctPendingStage::HfGlobal { .. }
                | VarDctPendingStage::RawHfDequant { .. }
        ) {
            return Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted.into());
        }
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: vec![UnvalidatedGpuImageOutput {
                id: OutputId(0),
                layout: self.layout.clone(),
                buffer: lifetime.output.clone(),
            }],
        })
    }

    fn stage_completion(&self) -> Arc<MapCompletion> {
        match &self.stage {
            VarDctPendingStage::LocalLf { completion, .. }
            | VarDctPendingStage::HfGlobal { completion, .. }
            | VarDctPendingStage::RawHfDequant { completion, .. }
            | VarDctPendingStage::Final { completion } => Arc::clone(completion),
        }
    }

    fn take_staged_packet(&mut self) -> Option<VarDctPendingContinuation> {
        let placeholder = VarDctPendingStage::Final {
            completion: Arc::new(MapCompletion::default()),
        };
        let stage = std::mem::replace(&mut self.stage, placeholder);
        match stage {
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
            VarDctPendingStage::RawHfDequant {
                source,
                mut commands,
                mut job,
                mut permit,
                ..
            } => commands.take().zip(job.take()).zip(permit.take()).map(
                |((commands, job), permit)| VarDctPendingContinuation::RawHfDequant {
                    source,
                    commands,
                    job,
                    permit,
                },
            ),
            final_stage @ VarDctPendingStage::Final { .. } => {
                self.stage = final_stage;
                None
            }
        }
    }

    fn advance_staged_packet(&mut self, mapping: Result<(), String>) -> DecodeResult<bool> {
        match self.take_staged_packet() {
            Some(VarDctPendingContinuation::LocalLf { source, commands }) => {
                self.submit_hf_stage(mapping, source, commands)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::HfGlobal { source, commands }) => {
                self.submit_hf_global_stage(mapping, source, commands)?;
                Ok(true)
            }
            Some(VarDctPendingContinuation::RawHfDequant {
                source,
                commands,
                job,
                permit,
            }) => {
                self.finish_raw_hf_dequant_stage(mapping, source, commands, job, permit)?;
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
        let hf_packet_windows = HfPacketWindowExecutionPlan::new(
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

        let windowed_batches = if let Some(plan) = &hf_packet_windows {
            let stream = lifetime._packet_stream_window.as_ref().ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed HF packet plan has no shared stream buffer",
                },
            )?;
            let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF packet stream window host length",
                }
            })?;
            let mut submissions = Vec::with_capacity(plan.stream_batches.len());
            for batch in plan.stream_batches.iter() {
                if batch.group_count == 0 || batch.segments.is_empty() {
                    return Err(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF packet batch contains no segment",
                    }
                    .into());
                }
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded HF packet stream batch"),
                });
                let mut stream_upload = vec![0_u8; upload_len];
                let mut group_uploads = Vec::with_capacity(batch.group_count);
                for segment_index in batch.segments.clone() {
                    let segment = *plan.stream_segments.get(segment_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet batch references an absent segment",
                        },
                    )?;
                    let buffers = lifetime._groups.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references an absent GPU group",
                        },
                    )?;
                    let metadata = metadata_buffers.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references absent Modular metadata",
                        },
                    )?;
                    let control = *controls.get(segment.group_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment references an absent control record",
                        },
                    )?;
                    let params = *plan.segment_params.get(segment_index).ok_or(
                        VarDctDecodeError::EntropyWindowContract {
                            detail: "HF packet segment has no parameter record",
                        },
                    )?;
                    copy_stream_segment(
                        &source.codestream,
                        segment,
                        &mut stream_upload,
                        "HF packet segment exceeds the source or reusable upload",
                    )?;
                    let packet_buffers = VarDctPacketBuffers {
                        codestream: stream,
                        modular_metadata: metadata,
                        reconstructed_lf: &buffers.reconstructed,
                        raw_hf_metadata: &buffers.raw_metadata,
                        coefficients: &buffers.coefficients,
                        status: &buffers.packet_status,
                        control: &buffers.packet_control,
                        modular_params: &buffers.modular_params,
                    };
                    if deferred_hf_global {
                        self.pipelines.packet.encode_hf_metadata(
                            device,
                            &mut encoder,
                            packet_buffers,
                        );
                        encoder.copy_buffer_to_buffer(
                            &buffers.packet_status,
                            0,
                            &lifetime.status_staging,
                            u64::try_from(segment.group_index).map_err(|_| {
                                VarDctDecodeError::ArithmeticOverflow {
                                    field: "windowed HF-metadata status index",
                                }
                            })? * PACKET_STATUS_BYTES,
                            PACKET_STATUS_BYTES,
                        );
                    } else {
                        self.pipelines
                            .packet
                            .encode_hf(device, &mut encoder, packet_buffers);
                    }
                    group_uploads.push(HfPacketGroupUpload {
                        group_index: segment.group_index,
                        control,
                        params,
                    });
                }
                if group_uploads.len() != batch.group_count {
                    return Err(VarDctDecodeError::EntropyWindowContract {
                        detail: "HF packet batch group count disagrees with its segments",
                    }
                    .into());
                }
                submissions.push(HfPacketBatchSubmission {
                    stream_upload: stream_upload.into_boxed_slice(),
                    groups: group_uploads.into_boxed_slice(),
                    commands: encoder.finish(),
                });
            }
            Some(submissions)
        } else {
            None
        };
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
                    submit_hf_packet_commands(self.backend.queue(), batches, downstream, lifetime)?,
                    None,
                ),
                PostLfCommands::DeferredHfGlobal(deferred) => (
                    submit_hf_metadata_packet_commands(self.backend.queue(), batches, lifetime)?,
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
            VarDctPendingStage::Final { completion }
        };
        Ok(())
    }

    fn start_raw_hf_dequant_stage(
        &mut self,
        source: Box<VarDctSource>,
        mut commands: DeferredHfGlobalCommands,
        poll_permit: Option<SubmissionPollPermit>,
    ) -> Result<(), VarDctDecodeError> {
        let lifetime = self
            .lifetime
            .as_ref()
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        let plan = source
            .packet
            .pending_raw_hf_dequant_side_image()
            .cloned()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "raw HF dequant stage has no pending side image",
            })?;
        let packet_end = source.packet.pending_raw_hf_dequant_packet_end().ok_or(
            VarDctDecodeError::EngineContract {
                detail: "raw HF dequant stage has no bounded packet end",
            },
        )?;
        let memory_bytes = self
            .pipelines
            .raw_hf_dequant
            .memory_bytes(&plan, packet_end)
            .map_err(|source| VarDctDecodeError::RawHfDequantGpu {
                matrix: plan.matrix_index,
                source: Box::new(source),
            })?;
        let permit = self.memory.try_reserve(memory_bytes)?;
        let mut job = self
            .pipelines
            .raw_hf_dequant
            .prepare(
                &self.backend,
                &lifetime._codestream,
                &lifetime._resources,
                source.resource_layout,
                &plan,
                packet_end,
            )
            .map_err(|source| VarDctDecodeError::RawHfDequantGpu {
                matrix: plan.matrix_index,
                source: Box::new(source),
            })?;
        if job.memory_bytes() != memory_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "raw HF dequant allocation disagrees with its byte admission",
            });
        }
        let poll_permit = match poll_permit {
            Some(permit) => permit,
            None => self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(VarDctDecodeError::PollBackpressure)?,
        };
        let mut submission_commands = Vec::with_capacity(2);
        if let Some(before_coefficients) = commands.before_coefficients.take() {
            submission_commands.push(before_coefficients);
        }
        submission_commands.push(job.take_commands().map_err(|source| {
            VarDctDecodeError::RawHfDequantGpu {
                matrix: plan.matrix_index,
                source: Box::new(source),
            }
        })?);
        let submission = self.backend.queue().submit(submission_commands);
        let completion = Arc::new(MapCompletion::default());
        arm_raw_hf_dequant_status_map(&job, &completion);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission, move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!(
                "raw HF dequant GPU poll registration failed: {error}"
            )));
        }
        let submissions = self
            .runtime_stats
            .submissions_per_frame
            .load(Ordering::Acquire)
            .checked_add(1)
            .ok_or(VarDctDecodeError::ArithmeticOverflow {
                field: "raw HF dequant submission count",
            })?;
        self.runtime_stats
            .submissions_per_frame
            .store(submissions, Ordering::Release);
        self.stage = VarDctPendingStage::RawHfDequant {
            completion,
            source,
            commands: Some(commands),
            job: Some(Box::new(job)),
            permit: Some(permit),
        };
        Ok(())
    }

    fn finish_raw_hf_dequant_stage(
        &mut self,
        mapping: Result<(), String>,
        mut source: Box<VarDctSource>,
        commands: DeferredHfGlobalCommands,
        job: Box<RawHfDequantSideImageJob>,
        permit: MemoryPermit,
    ) -> DecodeResult<()> {
        mapping.map_err(DecodeError::backend)?;
        let plan = source
            .packet
            .pending_raw_hf_dequant_side_image()
            .cloned()
            .ok_or(VarDctDecodeError::EngineContract {
                detail: "completed raw HF dequant stage has no parser continuation",
            })?;
        let packet_end = source.packet.pending_raw_hf_dequant_packet_end().ok_or(
            VarDctDecodeError::EngineContract {
                detail: "completed raw HF dequant stage has no bounded packet end",
            },
        )?;
        let status = job
            .finish_status()
            .map_err(|source| VarDctDecodeError::RawHfDequantGpu {
                matrix: plan.matrix_index,
                source: Box::new(source),
            })?;
        validate_raw_hf_dequant_status(&plan, packet_end, status)?;
        drop(job);
        drop(permit);
        source
            .packet
            .resume_hf_global_after_raw_side_image_source(&source.codestream, status.cursor)
            .map_err(VarDctDecodeError::from)?;
        if source.packet.pending_raw_hf_dequant_side_image().is_some() {
            self.start_raw_hf_dequant_stage(source, commands, None)?;
            Ok(())
        } else {
            self.submit_deferred_hf_coefficients(source, commands)
        }
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
                        expected_strategy: expected.uniform_transform,
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
            self.backend.queue().write_buffer(
                &lifetime._resources,
                source.resource_layout.dequant_matrix_byte_offset(),
                bytemuck::cast_slice(words),
            );
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
            let upload_len = usize::try_from(plan.stream_window_bytes()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "deferred HF stream window host length",
                }
            })?;
            for ((group_plan, hf_buffers), group_buffers) in plan
                .groups
                .iter()
                .zip(&buffers.groups)
                .zip(&lifetime._groups)
            {
                for batch in &group_plan.stream_batches {
                    let mut stream_upload = vec![0_u8; upload_len];
                    for segment in &group_plan.stream_segments[batch.segments.clone()] {
                        copy_stream_segment(
                            &source.codestream,
                            *segment,
                            &mut stream_upload,
                            "deferred HF stream segment exceeds the source or reusable upload",
                        )?;
                    }
                    let params_upload =
                        bytemuck::cast_slice(&group_plan.segment_params[batch.segments.clone()])
                            .to_vec()
                            .into_boxed_slice();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jxl-wgpu deferred HF coefficient stream batch"),
                        });
                    self.pipelines.hf_coefficients.encode(
                        device,
                        &mut encoder,
                        HfCoefficientBuffers {
                            codestream: buffers.stream_window.as_ref().ok_or(
                                VarDctDecodeError::EntropyWindowContract {
                                    detail: "deferred HF plan has no stream window",
                                },
                            )?,
                            entropy_bundle: &buffers.entropy_bundle,
                            reconstruction: &group_buffers.reconstructed,
                            params: buffers.params_window.as_ref().ok_or(
                                VarDctDecodeError::EntropyWindowContract {
                                    detail: "deferred HF plan has no parameter window",
                                },
                            )?,
                            status: &hf_buffers.status,
                            artifact: &group_buffers.artifact,
                            order_table: &buffers.order_table,
                            coefficients: &group_buffers.coefficients,
                            sink_params: &hf_buffers.sink_params,
                        },
                        u32::try_from(batch.group_count).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "deferred HF batch dispatch count",
                            }
                        })?,
                    );
                    coefficient_batches.push(HfCoefficientBatchSubmission {
                        stream_upload: stream_upload.into_boxed_slice(),
                        params_upload,
                        commands: encoder.finish(),
                    });
                }
            }
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
        if status_offset != source.memory.validation_staging_bytes {
            return Err(VarDctDecodeError::EngineContract {
                detail: "deferred HF status aggregation disagrees with admitted staging",
            }
            .into());
        }
        self.expected_hf_group_indices = plan
            .groups
            .iter()
            .flat_map(HfCoefficientGroupExecutionPlan::global_group_indices)
            .collect();
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
                if let Some(before_coefficients) = commands.before_coefficients.take() {
                    self.backend
                        .queue()
                        .submit([before_coefficients, first.commands]);
                } else {
                    self.backend.queue().submit([first.commands]);
                }
                let mut batch_count = 1_usize;
                for batch in batches {
                    self.backend
                        .queue()
                        .write_buffer(stream, 0, &batch.stream_upload);
                    self.backend
                        .queue()
                        .write_buffer(params, 0, &batch.params_upload);
                    self.backend.queue().submit([batch.commands]);
                    batch_count = batch_count.checked_add(1).ok_or(
                        VarDctDecodeError::ArithmeticOverflow {
                            field: "deferred HF batch count",
                        },
                    )?;
                }
                let total_submissions = source
                    .staged_lf_submission_count()
                    .checked_add(batch_count)
                    .and_then(|count| count.checked_add(2))
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
        self.stage = VarDctPendingStage::Final { completion };
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
                expected_strategy: expected.uniform_transform,
                expected_lf_samples: expected.expected_lf_samples,
                block_count: expected.expected_blocks,
                correlation_samples: expected.correlation_samples,
                task_capacity: expected.task_capacity,
                expected_global_scale: expected.expected_global_scale,
                expected_quant_lf: expected.expected_quant_lf,
                expected_extra_precision: expected.expected_extra_precision,
            };
            let first_blocks = if self.deferred_hf_global {
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
        let hf_status_bytes = mapped
            .get(hf_offset..)
            .ok_or(VarDctDecodeError::StatusAbi {
                status: "HF coefficient",
            })?;
        let hf_statuses = bytemuck::try_cast_slice::<u8, GpuHfCoefficientStatus>(hf_status_bytes)
            .map_err(|_| VarDctDecodeError::StatusAbi {
            status: "HF coefficient",
        })?;
        if hf_statuses.len() != self.expected_hf_group_indices.len() {
            return Err(VarDctDecodeError::StatusAbi {
                status: "HF coefficient count",
            }
            .into());
        }
        for (&group, status) in self
            .expected_hf_group_indices
            .iter()
            .zip(hf_statuses.iter().copied())
        {
            status.validate(group).map_err(VarDctDecodeError::from)?;
        }
        drop(mapped);
        let output_id = OutputId(0);
        let mut regions = BTreeMap::new();
        regions.insert(
            output_id,
            vec![Region::new(
                0,
                0,
                self.layout.extent.width,
                self.layout.extent.height,
            )],
        );
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
                outputs: vec![GpuImageOutput {
                    id: output_id,
                    layout: self.layout.clone(),
                    buffer: lifetime.output.clone(),
                }],
                changed: ChangedRegions { outputs: regions },
            },
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for VarDctPendingFrame {
    type Frame = GpuImageFrame;

    fn wait(mut self) -> DecodeResult<SubmittedGpuFrame<Self::Frame>> {
        loop {
            let mapping = self.stage_completion().wait();
            match self.take_staged_packet() {
                Some(VarDctPendingContinuation::LocalLf { source, commands }) => {
                    self.submit_hf_stage(mapping, source, commands)?;
                }
                Some(VarDctPendingContinuation::HfGlobal { source, commands }) => {
                    self.submit_hf_global_stage(mapping, source, commands)?;
                }
                Some(VarDctPendingContinuation::RawHfDequant {
                    source,
                    commands,
                    job,
                    permit,
                }) => {
                    self.finish_raw_hf_dequant_stage(mapping, source, commands, job, permit)?;
                }
                None => return self.finish(mapping),
            }
        }
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        loop {
            if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(DecodeError::backend(error)));
            }
            let Some(mapping) = self.stage_completion().poll(context) else {
                return Poll::Pending;
            };
            match self.take_staged_packet() {
                Some(VarDctPendingContinuation::LocalLf { source, commands }) => {
                    if let Err(error) = self.submit_hf_stage(mapping, source, commands) {
                        return Poll::Ready(Err(error));
                    }
                }
                Some(VarDctPendingContinuation::HfGlobal { source, commands }) => {
                    if let Err(error) = self.submit_hf_global_stage(mapping, source, commands) {
                        return Poll::Ready(Err(error));
                    }
                }
                Some(VarDctPendingContinuation::RawHfDequant {
                    source,
                    commands,
                    job,
                    permit,
                }) => {
                    if let Err(error) =
                        self.finish_raw_hf_dequant_stage(mapping, source, commands, job, permit)
                    {
                        return Poll::Ready(Err(error));
                    }
                }
                None => return Poll::Ready(self.finish(mapping)),
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for VarDctPendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<DecodeResult<SubmittedGpuFrame<Self::Frame>>> {
        loop {
            if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
                return Poll::Ready(Err(DecodeError::backend(error)));
            }
            let Some(mapping) = self.stage_completion().poll(context) else {
                return Poll::Pending;
            };
            match self.take_staged_packet() {
                Some(VarDctPendingContinuation::LocalLf { source, commands }) => {
                    if let Err(error) = self.submit_hf_stage(mapping, source, commands) {
                        return Poll::Ready(Err(error));
                    }
                }
                Some(VarDctPendingContinuation::HfGlobal { source, commands }) => {
                    if let Err(error) = self.submit_hf_global_stage(mapping, source, commands) {
                        return Poll::Ready(Err(error));
                    }
                }
                Some(VarDctPendingContinuation::RawHfDequant {
                    source,
                    commands,
                    job,
                    permit,
                }) => {
                    if let Err(error) =
                        self.finish_raw_hf_dequant_stage(mapping, source, commands, job, permit)
                    {
                        return Poll::Ready(Err(error));
                    }
                }
                None => return Poll::Ready(self.finish(mapping)),
            }
        }
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
    source: VarDctSource,
    permits: VarDctMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<VarDctPendingFrame, VarDctDecodeError> {
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
    let staged_local_trees = source.packet.requires_local_tree_staging();
    let staged_hf_global =
        source.packet.requires_hf_global_staging() && source.combined_packet_windows.is_none();
    let group_specific_metadata = staged_local_trees || source.packet.profile.uses_lf_frame;
    let modular_metadata = if group_specific_metadata {
        source
            .packet
            .groups
            .iter()
            .map(|group| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu VarDCT LF-local Modular metadata"),
                    contents: bytemuck::cast_slice(&group.lf_modular.metadata),
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
    let packet_stream_window_bytes = source
        .lf_packet_windows
        .as_ref()
        .map(|plan| plan.stream_bytes)
        .or_else(|| {
            source
                .combined_packet_windows
                .as_ref()
                .map(|plan| plan.stream_bytes)
        });
    let packet_stream_window = packet_stream_window_bytes.map(|bytes| {
        storage(
            "jxl-wgpu reusable packet entropy stream window",
            bytes,
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
        let predictor_capacity = source.packet.needs_self_correcting || group_specific_metadata;
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
        let modular = &packet_group.lf_modular;
        let params = VarDctModularParams::default()
            .with_lz77_window(if group_specific_metadata {
                modular.lz77_window_words
            } else {
                packet_group.lz77_window_words
            })
            .with_self_correcting(if group_specific_metadata {
                modular.needs_self_correcting
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
    let resident_planes = std::array::from_fn(|channel| {
        storage(
            image_labels[channel],
            source.memory.resident_plane_bytes[channel],
            wgpu::BufferUsages::COPY_DST,
        )
    });
    let component_upsample_planes = (source.memory.component_upsample_bytes != 0).then(|| {
        let labels = [
            "jxl-wgpu VarDCT component upsample X plane",
            "jxl-wgpu VarDCT component upsample Y plane",
            "jxl-wgpu VarDCT component upsample B plane",
        ];
        let shifted = source
            .packet
            .profile
            .channel_shifts
            .into_iter()
            .filter(|shift| shift.is_subsampled())
            .count() as u64;
        let full_plane_bytes = source
            .memory
            .component_upsample_bytes
            .checked_div(shifted)
            .unwrap_or(0);
        std::array::from_fn(|channel| {
            if source.packet.profile.channel_shifts[channel].is_subsampled() {
                storage(
                    labels[channel],
                    full_plane_bytes,
                    wgpu::BufferUsages::empty(),
                )
            } else {
                resident_planes[channel].clone()
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
    let frame_upsample_planes = (source.memory.frame_upsample_image_bytes != 0).then(|| {
        let labels = [
            "jxl-wgpu VarDCT frame upsample X plane",
            "jxl-wgpu VarDCT frame upsample Y plane",
            "jxl-wgpu VarDCT frame upsample B plane",
        ];
        std::array::from_fn(|channel| {
            storage(
                labels[channel],
                source.memory.frame_upsample_image_bytes / 3,
                wgpu::BufferUsages::empty(),
            )
        })
    });
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
        label: Some("jxl-wgpu VarDCT packed RGB8 output"),
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
    if let Some(lf_temporary) = &lf_temporary {
        packet_commands.clear_buffer(lf_temporary, 0, None);
    }
    for buffer in [
        &resident_planes[0],
        &resident_planes[1],
        &resident_planes[2],
        output.as_ref(),
    ] {
        packet_commands.clear_buffer(buffer, 0, None);
    }
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
    let (packet_stage_commands, combined_packet_batches, mut commands) = if let Some(plan) =
        &source.lf_packet_windows
    {
        if !staged_local_trees {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "LF packet windows require staged local trees",
            });
        }
        let stream =
            packet_stream_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed LF packet plan has no stream buffer",
                })?;
        let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "LF packet stream window host length",
            }
        })?;
        let mut first_commands = Some(packet_commands);
        let mut submissions = Vec::with_capacity(plan.stream_batches.len());
        for (batch_index, batch) in plan.stream_batches.iter().enumerate() {
            if batch.group_count != 1 || batch.segments.end != batch.segments.start + 1 {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "serial LF packet batch does not contain exactly one segment",
                });
            }
            let segment_index = batch.segments.start;
            let segment = *plan.stream_segments.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet batch references an absent segment",
                },
            )?;
            if segment.group_index != batch.first_group {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment and batch group indices disagree",
                });
            }
            let buffers = group_buffers.get(segment.group_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment references an absent GPU group",
                },
            )?;
            let metadata = modular_metadata.get(segment.group_index).ok_or(
                VarDctDecodeError::GroupPlanCount {
                    component: "LF-local Modular metadata",
                    expected: group_buffers.len(),
                    actual: modular_metadata.len(),
                },
            )?;
            let mut encoder = first_commands.take().unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded LF packet stream batch"),
                })
            });
            pipelines.packet.encode_lf(
                device,
                &mut encoder,
                VarDctPacketBuffers {
                    codestream: stream,
                    modular_metadata: metadata,
                    reconstructed_lf: &buffers.reconstructed,
                    raw_hf_metadata: &buffers.raw_metadata,
                    coefficients: &buffers.coefficients,
                    status: &buffers.packet_status,
                    control: &buffers.packet_control,
                    modular_params: &buffers.modular_params,
                },
            );
            if batch_index + 1 == plan.stream_batches.len() {
                for (index, buffers) in group_buffers.iter().enumerate() {
                    encoder.copy_buffer_to_buffer(
                        &buffers.packet_status,
                        0,
                        &status_staging,
                        u64::try_from(index).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "LF staging status index",
                            }
                        })? * PACKET_STATUS_BYTES,
                        PACKET_STATUS_BYTES,
                    );
                }
            }
            let mut stream_upload = vec![0_u8; upload_len];
            copy_stream_segment(
                &source.codestream,
                segment,
                &mut stream_upload,
                "LF packet segment exceeds the source or reusable upload",
            )?;
            let params = *plan.segment_params.get(segment_index).ok_or(
                VarDctDecodeError::EntropyWindowContract {
                    detail: "LF packet segment has no parameter record",
                },
            )?;
            submissions.push(LfPacketBatchSubmission {
                group_index: segment.group_index,
                stream_upload: stream_upload.into_boxed_slice(),
                params,
                commands: encoder.finish(),
            });
        }
        if first_commands.is_some() || submissions.is_empty() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed LF packet execution has no dispatch",
            });
        }
        (
            Some(LfPacketCommands::Windowed(submissions)),
            None,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT downstream stage"),
            }),
        )
    } else if let Some(plan) = &source.combined_packet_windows {
        if staged_local_trees {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "combined packet windows cannot stage local trees",
            });
        }
        let stream =
            packet_stream_window
                .as_ref()
                .ok_or(VarDctDecodeError::EntropyWindowContract {
                    detail: "windowed combined packet plan has no stream buffer",
                })?;
        let metadata = modular_metadata
            .first()
            .ok_or(VarDctDecodeError::GroupPlanCount {
                component: "global Modular metadata",
                expected: 1,
                actual: 0,
            })?;
        let upload_len = usize::try_from(plan.stream_bytes).map_err(|_| {
            VarDctDecodeError::ArithmeticOverflow {
                field: "combined packet stream window host length",
            }
        })?;
        let mut first_commands = Some(packet_commands);
        let mut submissions = Vec::with_capacity(plan.stream_batches.len());
        for batch in plan.stream_batches.iter() {
            if batch.group_count == 0 || batch.segments.is_empty() {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet batch contains no segment",
                });
            }
            let mut encoder = first_commands.take().unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu bounded combined packet stream batch"),
                })
            });
            let mut stream_upload = vec![0_u8; upload_len];
            let mut group_uploads = Vec::with_capacity(batch.group_count);
            for segment_index in batch.segments.clone() {
                let segment = *plan.stream_segments.get(segment_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet batch references an absent segment",
                    },
                )?;
                let buffers = group_buffers.get(segment.group_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet segment references an absent GPU group",
                    },
                )?;
                let params = *plan.segment_params.get(segment_index).ok_or(
                    VarDctDecodeError::EntropyWindowContract {
                        detail: "combined packet segment has no parameter record",
                    },
                )?;
                copy_stream_segment(
                    &source.codestream,
                    segment,
                    &mut stream_upload,
                    "combined packet segment exceeds the source or reusable upload",
                )?;
                pipelines.packet.encode(
                    device,
                    &mut encoder,
                    VarDctPacketBuffers {
                        codestream: stream,
                        modular_metadata: metadata,
                        reconstructed_lf: &buffers.reconstructed,
                        raw_hf_metadata: &buffers.raw_metadata,
                        coefficients: &buffers.coefficients,
                        status: &buffers.packet_status,
                        control: &buffers.packet_control,
                        modular_params: &buffers.modular_params,
                    },
                );
                group_uploads.push(CombinedPacketGroupUpload {
                    group_index: segment.group_index,
                    params,
                });
            }
            if group_uploads.len() != batch.group_count {
                return Err(VarDctDecodeError::EntropyWindowContract {
                    detail: "combined packet batch group count disagrees with its segments",
                });
            }
            submissions.push(CombinedPacketBatchSubmission {
                stream_upload: stream_upload.into_boxed_slice(),
                groups: group_uploads.into_boxed_slice(),
                commands: encoder.finish(),
            });
        }
        if first_commands.is_some() || submissions.is_empty() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "windowed combined packet execution has no dispatch",
            });
        }
        (
            None,
            Some(submissions),
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu bounded VarDCT downstream stage"),
            }),
        )
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
            } else if staged_local_trees || staged_hf_global {
                pipelines
                    .packet
                    .encode_lf(device, &mut packet_commands, buffers);
            } else {
                pipelines
                    .packet
                    .encode(device, &mut packet_commands, buffers);
            }
        }
        if staged_local_trees || staged_hf_global {
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
        if staged_local_trees || staged_hf_global {
            (
                Some(LfPacketCommands::Whole(packet_commands.finish())),
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
            let lf_destination = if source.adaptive_lf_smoothing {
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
            let adaptive_lf_uniform = if source.adaptive_lf_smoothing {
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
            let stream_window =
                buffers
                    .stream_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC plan has no stream buffer",
                    })?;
            let params_window =
                buffers
                    .params_window
                    .as_ref()
                    .ok_or(VarDctDecodeError::EntropyWindowContract {
                        detail: "windowed AC plan has no parameter buffer",
                    })?;
            let upload_len = usize::try_from(plan.stream_window_bytes()).map_err(|_| {
                VarDctDecodeError::ArithmeticOverflow {
                    field: "HF stream window host length",
                }
            })?;
            for ((group_plan, hf_buffers), group_buffers) in
                plan.groups.iter().zip(&buffers.groups).zip(&group_buffers)
            {
                for batch in &group_plan.stream_batches {
                    let mut stream_upload = vec![0_u8; upload_len];
                    for segment in &group_plan.stream_segments[batch.segments.clone()] {
                        copy_stream_segment(
                            &source.codestream,
                            *segment,
                            &mut stream_upload,
                            "HF stream segment exceeds the source or reusable upload",
                        )?;
                    }
                    let params_upload =
                        bytemuck::cast_slice(&group_plan.segment_params[batch.segments.clone()])
                            .to_vec()
                            .into_boxed_slice();
                    let mut batch_commands =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("jxl-wgpu bounded HF coefficient stream batch"),
                        });
                    pipelines.hf_coefficients.encode(
                        device,
                        &mut batch_commands,
                        HfCoefficientBuffers {
                            codestream: stream_window,
                            entropy_bundle: &buffers.entropy_bundle,
                            reconstruction: &group_buffers.reconstructed,
                            params: params_window,
                            status: &hf_buffers.status,
                            artifact: &group_buffers.artifact,
                            order_table: &buffers.order_table,
                            coefficients: &group_buffers.coefficients,
                            sink_params: &hf_buffers.sink_params,
                        },
                        u32::try_from(batch.group_count).map_err(|_| {
                            VarDctDecodeError::ArithmeticOverflow {
                                field: "HF stream batch dispatch count",
                            }
                        })?,
                    );
                    windowed_coefficient_batches.push(HfCoefficientBatchSubmission {
                        stream_upload: stream_upload.into_boxed_slice(),
                        params_upload,
                        commands: batch_commands.finish(),
                    });
                }
            }
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
    let padded_width = blocks_x
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output width",
        })?;
    let padded_height = blocks_y
        .checked_mul(8)
        .ok_or(VarDctDecodeError::ArithmeticOverflow {
            field: "padded output height",
        })?;
    let correlation_width = source.packet.profile.width.div_ceil(64);
    let correlation_height = source.packet.profile.height.div_ceil(64);
    let mut resident_scratch = Vec::with_capacity(source.groups.len());
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
                    &resident_planes,
                    padded_width,
                    padded_height,
                    source.packet.profile.channel_shifts,
                )?,
                indirect: &buffers.artifact,
                indirect_base_offset: u64::from(group.artifact_layout.indirect_offset_words) * 4,
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
    let mut component_upsample_uniforms = Vec::new();
    if let Some(upsampled) = &component_upsample_planes {
        for channel in 0..3 {
            let shift = source.packet.profile.channel_shifts[channel];
            if !shift.is_subsampled() {
                continue;
            }
            let [input_width, input_height] = shift
                .shifted_extent(image_width, image_height)
                .ok_or(VarDctDecodeError::ArithmeticOverflow {
                    field: "component upsample input extent",
                })?;
            component_upsample_uniforms.push(pipelines.chroma_upsample.encode(
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
    let restoration_source = component_upsample_planes
        .as_ref()
        .unwrap_or(&resident_planes);
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
                    sigma,
                    parameters,
                },
            )?);
        }
    }
    let restored_planes = restoration
        .as_ref()
        .map_or(restoration_source, RestorationCursor::current);
    let (presentation_planes, frame_upsample_resources) =
        if let (Some(weights), Some(output_buffers)) = (
            source.frame_upsampling.as_ref(),
            frame_upsample_planes.as_ref(),
        ) {
            let presentation_width = source.packet.profile.presentation_width;
            let presentation_height = source.packet.profile.presentation_height;
            let inputs =
                resident_image_planes(restored_planes, image_width, image_height, padded_width)?;
            let outputs = [
                ResidentF32Plane {
                    storage: resident_binding(&output_buffers[0])?,
                    width: presentation_width,
                    height: presentation_height,
                    stride: presentation_width,
                },
                ResidentF32Plane {
                    storage: resident_binding(&output_buffers[1])?,
                    width: presentation_width,
                    height: presentation_height,
                    stride: presentation_width,
                },
                ResidentF32Plane {
                    storage: resident_binding(&output_buffers[2])?,
                    width: presentation_width,
                    height: presentation_height,
                    stride: presentation_width,
                },
            ];
            let resources = pipelines.image_upsample.encode(
                device,
                &mut commands,
                ResidentImageUpsampleInputs {
                    inputs,
                    outputs,
                    weights,
                },
            )?;
            (output_buffers, Some(resources))
        } else {
            (restored_planes, None)
        };
    let components_are_full_resolution = source.frame_upsampling.is_some()
        || restoration.is_some()
        || component_upsample_planes.is_some();
    let presentation_shifts = if components_are_full_resolution {
        [crate::vardct_frontend::VarDctChannelShift::default(); 3]
    } else {
        source.packet.profile.channel_shifts
    };
    let (output_geometry, output_strides) = if source.frame_upsampling.is_some() {
        (
            [[
                source.packet.profile.presentation_width,
                source.packet.profile.presentation_height,
            ]; 3],
            [source.packet.profile.presentation_width; 3],
        )
    } else {
        let presentation_geometry = presentation_shifts.map(|shift| {
            shift.shifted_extent(image_width, image_height).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "presentation channel extent",
                },
            )
        });
        let [geometry_x, geometry_y, geometry_b] = presentation_geometry;
        let presentation_geometry = [geometry_x?, geometry_y?, geometry_b?];
        let presentation_strides = presentation_shifts.map(|shift| {
            padded_width.checked_shr(shift.horizontal).ok_or(
                VarDctDecodeError::ArithmeticOverflow {
                    field: "presentation channel stride",
                },
            )
        });
        let [stride_x, stride_y, stride_b] = presentation_strides;
        let presentation_strides = [stride_x?, stride_y?, stride_b?];
        (presentation_geometry, presentation_strides)
    };
    let output_transform = match source.output_transform {
        VarDctOutputTransform::Ycbcr { .. } => VarDctOutputTransform::Ycbcr {
            channel_shifts: presentation_shifts,
        },
        transform => transform,
    };
    let output_scratch = pipelines.output.encode(
        device,
        &mut commands,
        VarDctOutputInputs {
            planes: [
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[0])?,
                    width: output_geometry[0][0],
                    height: output_geometry[0][1],
                    stride: output_strides[0],
                },
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[1])?,
                    width: output_geometry[1][0],
                    height: output_geometry[1][1],
                    stride: output_strides[1],
                },
                VarDctOutputPlane {
                    storage: resident_binding(&presentation_planes[2])?,
                    width: output_geometry[2][0],
                    height: output_geometry[2][1],
                    stride: output_strides[2],
                },
            ],
            output: resident_binding(&output)?,
            config: VarDctOutputConfig {
                width: source.packet.profile.presentation_width,
                height: source.packet.profile.presentation_height,
                format: source.output_format,
                transform: output_transform,
            },
        },
    )?;
    debug_assert_eq!(output_scratch.plan, source.output_plan);
    let restoration_buffers = restoration_planes.map(|planes| RestorationJobBuffers {
        _planes: planes,
        _gaborish_uniform: gaborish_uniform,
        _epf_sigma: epf_sigma,
        _epf_sigma_uniforms: epf_sigma_uniforms,
        _epf_uniforms: epf_uniforms,
    });
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
            offset =
                offset
                    .checked_add(status_bytes)
                    .ok_or(VarDctDecodeError::ArithmeticOverflow {
                        field: "HF status staging offset",
                    })?;
        }
        debug_assert_eq!(offset, source.memory.validation_staging_bytes);
    }

    let after_coefficients = commands.finish();
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
        _hf_coefficients: Mutex::new(hf_coefficient_buffers),
        _resident_planes: resident_planes,
        _component_upsample_planes: component_upsample_planes,
        _component_upsample_uniforms: component_upsample_uniforms,
        _restoration: restoration_buffers,
        _frame_upsample_planes: frame_upsample_planes,
        _frame_upsample_resources: frame_upsample_resources,
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
            uniform_transform: source.packet.uniform_transform,
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
            expected_extra_precision: group.extra_precision,
        });
    }
    let expected_hf_group_indices = source
        .hf_coefficients
        .iter()
        .flat_map(|plan| &plan.groups)
        .flat_map(HfCoefficientGroupExecutionPlan::global_group_indices)
        .collect();
    let layout = source.layout.clone();
    let frame_name = source.frame_name.clone();
    let progressive_dc_extent = Extent2d {
        width: source.packet.profile.width,
        height: source.packet.profile.height,
    };
    let mut pending = VarDctPendingFrame {
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
        frame_name,
        expected_groups,
        expected_hf_group_indices,
        deferred_hf_global: staged_hf_global,
        progressive_dc_extent,
        progressive_dc_stride: padded_width,
    };
    if source.packet.pending_raw_hf_dequant_side_image().is_some()
        && packet_stage_commands.is_none()
    {
        if combined_packet_batches.is_some() {
            return Err(VarDctDecodeError::EntropyWindowContract {
                detail: "raw HF dequant side images cannot use combined packet stream windows",
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
            let submission =
                submit_lf_packet_commands(backend.queue(), packet_stage_commands, &lifetime)?;
            if staged_hf_global || source.packet.pending_raw_hf_dequant_side_image().is_some() {
                if source.packet.profile.uses_lf_frame {
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
        } else if let Some(batches) = combined_packet_batches {
            let downstream = downstream_commands.ok_or(VarDctDecodeError::EngineContract {
                detail: "combined packet execution is missing downstream commands",
            })?;
            (
                submit_combined_packet_commands(backend.queue(), batches, downstream, &lifetime)?,
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
        if staged_local_trees {
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
        VarDctPendingStage::LocalLf {
            completion,
            source: Box::new(source),
            commands: Some(commands),
        }
    } else if let Some(commands) = deferred_commands {
        VarDctPendingStage::HfGlobal {
            completion,
            source: Box::new(source),
            commands: Some(commands),
        }
    } else {
        VarDctPendingStage::Final { completion }
    };
    Ok(pending)
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

    fn poll(&self, context: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.waker = Some(context.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), String> {
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

fn validate_raw_hf_dequant_status(
    plan: &crate::vardct_side_image::RawHfDequantSideImagePlan,
    packet_end: u32,
    status: RawHfDequantSideImageStatus,
) -> Result<(), VarDctDecodeError> {
    if raw_matrix_value_error(status.code) {
        return Err(VarDctDecodeError::RawHfDequantValue {
            matrix: plan.matrix_index,
        });
    }
    if !raw_matrix_status_ok(status.code)
        || status.decoded_samples != plan.decoded_words
        || status.cursor < plan.token_bit_offset
        || status.cursor > packet_end
        || status.expected_cursor != packet_end
    {
        return Err(VarDctDecodeError::RawHfDequantStatus {
            matrix: plan.matrix_index,
            code: status.code,
            decoded_samples: status.decoded_samples,
            expected_samples: plan.decoded_words,
            cursor: status.cursor,
            expected_cursor: status.expected_cursor,
        });
    }
    Ok(())
}

fn arm_raw_hf_dequant_status_map(job: &RawHfDequantSideImageJob, completion: &Arc<MapCompletion>) {
    job.mark_status_mapped();
    let callback_completion = Arc::clone(completion);
    job.status_staging()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            callback_completion.complete(result.map_err(|error| error.to_string()));
        });
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
