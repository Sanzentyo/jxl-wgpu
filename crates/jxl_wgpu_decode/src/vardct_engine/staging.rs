//! A frame is prepared in the order in which its GPU-owned entropy cursors become known.

use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{CodestreamInventory, SampleBitDepth};
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, MemoryBudget, MemoryBudgetSnapshot, MemoryPermit,
    ResidentStorageBinding, SubmissionPollPermit, UnvalidatedGpuImageFrame, WgpuBackend,
};

use crate::modular_transform::GpuModularChannelLayout;
use crate::progressive_dc::ProgressiveDcXybPlanes;
use crate::vardct_output::VarDctOutputAlpha;
use crate::vardct_packet::PendingGlobalModular;
use crate::wgpu_engine::{
    ModularSideImageJob, ModularSideImagePipeline, ModularSideImageStreamPlan,
};
use crate::{
    AnimationMetadata, DecodeProfile, Error, GpuCodestream, GpuOutputRequest, GpuPendingFrame,
    GpuSubmissionSession, PreparedGpuSession, Result, SubmittedGpuFrame,
};

use super::execution::{FrameDecodeSession, FramePendingFrame, MapCompletion, VarDctRuntimeStats};
use super::pipeline::VarDctPipelines;
use super::source::{VarDctPrepareOptions, prepare_packet_source};
use super::types::{VarDctDecodeError, VarDctDecodeMemoryStats};

/// Exact buffer requirements for the initial global Modular stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarDctGlobalModularMemoryStats {
    /// Reused input binding, including overlap and the four-byte read sentinel.
    pub stream_bytes: u64,
    /// Arena retained through color output when an opacity plane is present.
    pub arena_bytes: u64,
    /// All buffers allocated by the initial stage, including its arena and 16-byte status map.
    pub total_bytes: u64,
}

pub(super) struct ResidentModularPlane {
    pub(super) index: u32,
    pub(super) arena: GpuBufferLease,
    pub(super) plane: GpuModularChannelLayout,
    pub(super) bits: u32,
}

impl ResidentModularPlane {
    pub(super) fn alpha_binding(
        &self,
    ) -> std::result::Result<VarDctOutputAlpha<'_>, VarDctDecodeError> {
        Ok(VarDctOutputAlpha {
            domain: crate::ModularSampleDomain::SignedInteger,
            storage: ResidentStorageBinding::entire(self.arena.as_wgpu_buffer())?,
            width: self.plane.width,
            height: self.plane.height,
            stride: self.plane.row_stride_words,
            word_offset: self.plane.word_offset,
            bits_per_sample: self.bits,
        })
    }
}

struct GlobalSource {
    codestream: GpuCodestream,
    inventory: CodestreamInventory,
    request: GpuOutputRequest,
    packet: PendingGlobalModular,
    options: VarDctPrepareOptions,
    pipelines: Arc<VarDctPipelines>,
    memory: VarDctGlobalModularMemoryStats,
    stream: ModularSideImageStreamPlan,
    next_window: usize,
}

enum PreparedStage {
    Frame(Box<FrameDecodeSession>),
    Global(Box<GlobalSource>),
}

/// One frame whose preparation may require a global Modular GPU cursor before its remaining frame plan.
pub struct VarDctDecodeSession {
    backend: WgpuBackend,
    memory: MemoryBudget,
    prepared: Option<PreparedStage>,
    runtime: Arc<VarDctRuntimeStats>,
    frame_memory: Arc<Mutex<Option<VarDctDecodeMemoryStats>>>,
    global_memory: Option<VarDctGlobalModularMemoryStats>,
}

impl VarDctDecodeSession {
    pub(super) fn ready(session: FrameDecodeSession) -> Self {
        Self {
            backend: session.backend.clone(),
            memory: session.memory.clone(),
            runtime: Arc::clone(&session.runtime_stats),
            frame_memory: Arc::new(Mutex::new(Some(session.memory_stats()))),
            global_memory: None,
            prepared: Some(PreparedStage::Frame(Box::new(session))),
        }
    }

    pub(super) fn global(
        engine: &super::pipeline::VarDctSubmissionEngine,
        codestream: GpuCodestream,
        inventory: &CodestreamInventory,
        request: &GpuOutputRequest,
        packet: PendingGlobalModular,
        options: VarDctPrepareOptions,
    ) -> Result<PreparedGpuSession<Self>> {
        let backend = engine.backend.clone();
        let memory = engine.memory.clone();
        let pipelines = Arc::clone(&engine.pipelines);
        let profile = packet.profile();
        let presentation = super::output::prepare_presentation(
            &backend,
            inventory,
            request,
            profile,
            options.output_variant,
        )?;
        let extent = presentation.layout.extent;
        let decode_profile = DecodeProfile::VarDct {
            bits_per_sample: profile.bits_per_sample as u8,
        };
        let mut metadata = AnimationMetadata::still(extent);
        metadata.extra_channels = inventory.image_header.extra_channels.clone();
        let end = packet.packet_end().map_err(VarDctDecodeError::from)?;
        let limits = backend.device().limits();
        let stream_limit = options
            .stream_window_limit
            .map_or(u64::MAX, |value| value.get())
            .min(limits.max_buffer_size)
            .min(limits.max_storage_buffer_binding_size);
        let mut stream = pipelines.raw_hf_dequant.modular().plan_source(
            &codestream,
            &packet.image,
            end,
            stream_limit,
        )?;
        if stream.memory_bytes > options.memory_limit_bytes {
            let fixed_bytes = stream.memory_bytes - stream.stream_bytes;
            let minimum_stream = stream
                .stream_bytes
                .min(crate::entropy_window::MIN_STREAM_WINDOW_BYTES);
            let available = options.memory_limit_bytes.saturating_sub(fixed_bytes) & !3;
            if available < minimum_stream {
                return Err(VarDctDecodeError::MemoryBudgetTooSmall {
                    required_bytes: fixed_bytes + minimum_stream,
                    limit_bytes: options.memory_limit_bytes,
                }
                .into());
            }
            stream = pipelines.raw_hf_dequant.modular().plan_source(
                &codestream,
                &packet.image,
                end,
                stream_limit.min(available),
            )?;
        }
        let global_memory = VarDctGlobalModularMemoryStats {
            stream_bytes: stream.stream_bytes,
            arena_bytes: ModularSideImagePipeline::arena_bytes(&packet.image)?,
            total_bytes: stream.memory_bytes,
        };
        if global_memory.total_bytes > options.memory_limit_bytes {
            return Err(VarDctDecodeError::MemoryBudgetTooSmall {
                required_bytes: global_memory.total_bytes,
                limit_bytes: options.memory_limit_bytes,
            }
            .into());
        }
        let session = Self {
            backend,
            memory,
            runtime: Arc::new(VarDctRuntimeStats {
                submissions_per_frame: Arc::new(AtomicUsize::new(1)),
                hf_packet_stream_batch_count: AtomicUsize::new(0),
            }),
            frame_memory: Arc::new(Mutex::new(None)),
            global_memory: Some(global_memory),
            prepared: Some(PreparedStage::Global(Box::new(GlobalSource {
                codestream,
                inventory: inventory.clone(),
                request: request.clone(),
                packet,
                options,
                pipelines,
                memory: global_memory,
                stream,
                next_window: 1,
            }))),
        };
        Ok(PreparedGpuSession::new(decode_profile, metadata, session)
            .with_resolved_frame_slots(NonZeroUsize::new(1).expect("one is nonzero")))
    }

    /// Frame-stage allocation plan. It becomes available after global Modular cursor validation.
    /// This excludes the independently tracked global arena; use the budget snapshot for live bytes.
    #[must_use]
    pub fn memory_stats(&self) -> Option<VarDctDecodeMemoryStats> {
        *self
            .frame_memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    #[must_use]
    pub const fn global_modular_memory_stats(&self) -> Option<VarDctGlobalModularMemoryStats> {
        self.global_memory
    }
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }
    /// Submissions known so far, including the global Modular stage when present.
    #[must_use]
    pub fn submissions_per_frame(&self) -> usize {
        self.runtime.submissions_per_frame.load(Ordering::Acquire)
    }
    #[must_use]
    pub fn hf_packet_stream_batch_count(&self) -> usize {
        self.runtime
            .hf_packet_stream_batch_count
            .load(Ordering::Acquire)
    }
    pub(crate) fn set_progressive_dc_source(
        &mut self,
        planes: ProgressiveDcXybPlanes,
    ) -> std::result::Result<(), VarDctDecodeError> {
        match self.prepared.as_mut() {
            Some(PreparedStage::Frame(session)) => session.set_progressive_dc_source(planes),
            _ => Err(VarDctDecodeError::UnexpectedProgressiveDcSource),
        }
    }
}

impl std::fmt::Debug for VarDctDecodeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarDctDecodeSession")
            .field("submitted", &self.prepared.is_none())
            .field("frame_memory", &self.memory_stats())
            .field("global_memory", &self.global_memory)
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for VarDctDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = VarDctPendingFrame;
    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        let Some(prepared) = self.prepared.as_mut() else {
            return Ok(None);
        };
        let state = match prepared {
            PreparedStage::Frame(session) => {
                let Some(pending) = session.submit_next()? else {
                    return Ok(None);
                };
                PendingStage::Frame(Box::new(pending))
            }
            PreparedStage::Global(source) => {
                let poll = self
                    .backend
                    .submission_poller()
                    .try_reserve()
                    .map_err(Error::PollBackpressure)?;
                let arena_permit = self.memory.try_reserve(source.memory.arena_bytes)?;
                let transient = self
                    .memory
                    .try_reserve(source.memory.total_bytes - source.memory.arena_bytes)?;
                let mut job = source
                    .pipelines
                    .raw_hf_dequant
                    .modular()
                    .record_source(
                        &self.backend,
                        &source.codestream,
                        &source.packet.image,
                        &source.stream,
                    )?
                    .finish();
                debug_assert_eq!(job.memory_bytes(), source.memory.total_bytes);
                let commands = job.take_commands()?;
                let lifetime = Arc::new(GlobalLifetime {
                    arena: GpuBufferLease::from_tracked(job.arena().clone(), arena_permit),
                    job,
                    _transient: transient,
                });
                let completion = submit_global(&self.backend, &lifetime, poll, commands);
                let Some(PreparedStage::Global(source)) = self.prepared.take() else {
                    unreachable!("global source is retained through submission");
                };
                PendingStage::Global {
                    source,
                    lifetime,
                    completion,
                }
            }
        };
        self.prepared = None;
        Ok(Some(VarDctPendingFrame {
            backend: self.backend.clone(),
            memory: self.memory.clone(),
            runtime: Arc::clone(&self.runtime),
            frame_memory: Arc::clone(&self.frame_memory),
            state,
        }))
    }
}

struct GlobalLifetime {
    job: ModularSideImageJob,
    arena: GpuBufferLease,
    _transient: MemoryPermit,
}

fn submit_global(
    backend: &WgpuBackend,
    lifetime: &Arc<GlobalLifetime>,
    poll: SubmissionPollPermit,
    commands: wgpu::CommandBuffer,
) -> Arc<MapCompletion> {
    let submission = backend.queue().submit([commands]);
    let completion = Arc::new(MapCompletion::default());
    lifetime.job.mark_status_mapped();
    let callback_lifetime = Arc::clone(lifetime);
    let callback_completion = Arc::clone(&completion);
    lifetime
        .job
        .status_staging()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // Completion publication also hands exclusive ownership back to the continuation.
            drop(callback_lifetime);
            callback_completion.complete(result.map_err(|error| error.to_string()));
        });
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(error.to_string()));
    }
    completion
}

enum PendingStage {
    Global {
        source: Box<GlobalSource>,
        lifetime: Arc<GlobalLifetime>,
        completion: Arc<MapCompletion>,
    },
    Frame(Box<FramePendingFrame>),
    Consumed,
}

/// A submitted VarDCT frame, including any preceding Modular cursor continuation.
pub struct VarDctPendingFrame {
    backend: WgpuBackend,
    memory: MemoryBudget,
    runtime: Arc<VarDctRuntimeStats>,
    frame_memory: Arc<Mutex<Option<VarDctDecodeMemoryStats>>>,
    state: PendingStage,
}

impl std::fmt::Debug for VarDctPendingFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarDctPendingFrame")
            .field(
                "stage",
                &match self.state {
                    PendingStage::Global { .. } => "global-modular",
                    PendingStage::Frame(_) => "frame",
                    PendingStage::Consumed => "consumed",
                },
            )
            .finish_non_exhaustive()
    }
}

impl VarDctPendingFrame {
    fn resume_global(&mut self, mapping: std::result::Result<(), String>) -> Result<()> {
        let state = std::mem::replace(&mut self.state, PendingStage::Consumed);
        let PendingStage::Global {
            mut source,
            lifetime,
            ..
        } = state
        else {
            return Err(VarDctDecodeError::CompletionConsumed.into());
        };
        mapping.map_err(Error::backend)?;
        let status = lifetime.job.finish_status()?;
        let end = source
            .packet
            .packet_end()
            .map_err(VarDctDecodeError::from)?;
        let plan = &source.packet.image;
        let segment =
            source
                .stream
                .segments
                .get(source.next_window - 1)
                .ok_or(Error::EngineContract(
                    "Modular side-image has no submitted window",
                ))?;
        if !(status.is_ok() || status.is_in_progress())
            || status.decoded_samples > plan.decoded_words
            || (status.is_ok() && status.decoded_samples != plan.decoded_words)
            || status.cursor < plan.token_bit_offset
            || status.cursor > plan.token_bit_offset + segment.available_token_end
            || status.expected_cursor != end
        {
            return Err(VarDctDecodeError::GlobalModularStatus {
                code: status.code,
                decoded_samples: status.decoded_samples,
                expected_samples: plan.decoded_words,
                cursor: status.cursor,
                packet_end: end,
            }
            .into());
        }
        let mut lifetime = Arc::try_unwrap(lifetime).map_err(|_| {
            Error::EngineContract("Modular side-image callback retained resources after completion")
        })?;
        let commands = if status.is_in_progress() {
            let segment =
                source
                    .stream
                    .segments
                    .get(source.next_window)
                    .ok_or(Error::EngineContract(
                        "Modular side-image yielded after its final window",
                    ))?;
            // Reserve the polling slot before recording uploads or consuming the next window.
            let poll = self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(Error::PollBackpressure)?;
            let commands =
                lifetime
                    .job
                    .record_next_window(&self.backend, &source.codestream, segment)?;
            source.next_window += 1;
            Some((commands, poll))
        } else if lifetime.job.has_inverse_commands() {
            let poll = self
                .backend
                .submission_poller()
                .try_reserve()
                .map_err(Error::PollBackpressure)?;
            lifetime
                .job
                .take_inverse_commands()
                .map(|commands| (commands, poll))
        } else {
            None
        };
        if let Some((commands, poll)) = commands {
            let lifetime = Arc::new(lifetime);
            let completion = submit_global(&self.backend, &lifetime, poll, commands);
            self.runtime
                .submissions_per_frame
                .fetch_add(1, Ordering::AcqRel);
            self.state = PendingStage::Global {
                source,
                lifetime,
                completion,
            };
            return Ok(());
        }
        let distributed = source.packet.has_distributed_extras();
        let global_extra_prefix = distributed.then(|| lifetime.arena.clone());
        let extra_planes = super::output::selected_extra_indices(
            &source.request,
            &source.inventory.image_header.extra_channels,
        )
        .into_iter()
        .filter(|_| !distributed)
        .map(|index| {
            let SampleBitDepth::Integer { bits_per_sample } =
                source.inventory.image_header.extra_channels[index].bit_depth
            else {
                unreachable!("integer extra-channel profile")
            };
            ResidentModularPlane {
                index: index as u32,
                arena: lifetime.arena.clone(),
                plane: plan.final_planes[index],
                bits: bits_per_sample,
            }
        })
        .collect::<Vec<_>>();
        drop(lifetime);
        let mut options = source.options;
        options.memory_limit_bytes = options.memory_limit_bytes.saturating_sub(
            global_extra_prefix
                .as_ref()
                .map_or(0, GpuBufferLease::reserved_bytes)
                + extra_planes
                    .first()
                    .map_or(0, |plane| plane.arena.reserved_bytes()),
        );
        let packet = source
            .packet
            .resume(&source.codestream, status.cursor)
            .map_err(VarDctDecodeError::from)?;
        let mut frame = prepare_packet_source(
            &self.backend,
            source.codestream,
            &source.request,
            &source.inventory,
            options,
            packet,
        )?;
        frame.extra_planes = extra_planes;
        frame.global_extra_prefix = global_extra_prefix;
        *self
            .frame_memory
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(frame.memory);
        self.runtime
            .submissions_per_frame
            .fetch_add(frame.submissions_per_frame(), Ordering::AcqRel);
        let mut session = FrameDecodeSession {
            backend: self.backend.clone(),
            pipelines: source.pipelines,
            memory_stats: frame.memory,
            runtime_stats: Arc::clone(&self.runtime),
            source: Some(frame),
            memory: self.memory.clone(),
        };
        let pending = session
            .submit_next()?
            .ok_or(VarDctDecodeError::CompletionConsumed)?;
        self.state = PendingStage::Frame(Box::new(pending));
        Ok(())
    }
    pub(crate) fn submissions_per_frame_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.runtime.submissions_per_frame)
    }
    pub(crate) fn dependency_submission_ready(&self) -> bool {
        matches!(&self.state, PendingStage::Frame(pending) if pending.dependency_submission_ready())
    }
    pub(crate) fn progressive_dc_planes(
        &self,
    ) -> std::result::Result<ProgressiveDcXybPlanes, VarDctDecodeError> {
        match &self.state {
            PendingStage::Frame(pending) => pending.progressive_dc_planes(),
            _ => Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted),
        }
    }
    pub fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match &self.state {
            PendingStage::Frame(pending) => pending.unvalidated_gpu_frame(),
            _ => Err(VarDctDecodeError::UnvalidatedOutputNotSubmitted.into()),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn wait_until_dependency_submitted(&mut self) -> Result<()> {
        while let PendingStage::Global { completion, .. } = &self.state {
            let mapping = completion.wait();
            self.resume_global(mapping)?;
        }
        match &mut self.state {
            PendingStage::Frame(pending) => pending.wait_until_dependency_submitted(),
            _ => Err(VarDctDecodeError::CompletionConsumed.into()),
        }
    }
    pub(crate) fn poll_until_dependency_submitted(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<()>> {
        if let PendingStage::Global { completion, .. } = &self.state {
            self.backend
                .device()
                .poll(wgpu::PollType::Poll)
                .map_err(Error::backend)?;
            let Some(mapping) = completion.poll(context) else {
                return Poll::Pending;
            };
            self.resume_global(mapping)?;
            if matches!(self.state, PendingStage::Global { .. }) {
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
        }
        match &mut self.state {
            PendingStage::Frame(pending) => pending.poll_until_dependency_submitted(context),
            _ => Poll::Ready(Err(VarDctDecodeError::CompletionConsumed.into())),
        }
    }
}

impl GpuPendingFrame for VarDctPendingFrame {
    type Frame = GpuImageFrame;
    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        while let PendingStage::Global { completion, .. } = &self.state {
            let mapping = completion.wait();
            self.resume_global(mapping)?;
        }
        match self.state {
            PendingStage::Frame(pending) => (*pending).wait(),
            _ => Err(VarDctDecodeError::CompletionConsumed.into()),
        }
    }
    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if let PendingStage::Global { completion, .. } = &self.state {
            self.backend
                .device()
                .poll(wgpu::PollType::Poll)
                .map_err(Error::backend)?;
            let Some(mapping) = completion.poll(context) else {
                return Poll::Pending;
            };
            self.resume_global(mapping)?;
            // Yield between windows even if a fast adapter already completed the next map.
            // This keeps cancellation responsive and avoids monopolizing a browser executor.
            if matches!(self.state, PendingStage::Global { .. }) {
                context.waker().wake_by_ref();
                return Poll::Pending;
            }
        }
        match &mut self.state {
            PendingStage::Frame(pending) => Pin::new(pending.as_mut()).poll_complete(context),
            _ => Poll::Ready(Err(VarDctDecodeError::CompletionConsumed.into())),
        }
    }
}
