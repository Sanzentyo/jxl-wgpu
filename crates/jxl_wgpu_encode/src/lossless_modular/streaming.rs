use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_wgpu::MemoryPermit;

use super::dispatch::{
    LosslessModularBackend, ModularDispatchBatch, ModularDispatchPlan, ModularGroupPlan,
};
use super::grid::LosslessModularGroupGrid;
use super::serializer::{
    ModularFrameHeader, ModularPacketAssembler, ModularPacketConfig, PacketBuildInput,
    ValidatedModularArtifact, accumulate_artifact_histograms, build_packets, build_prefix_codes,
    parse_group_artifact, parse_group_artifact_header,
};
use super::types::{LosslessModularFormat, LosslessModularTreeMode, ModularParams};
use crate::buffer_pool::EncoderBufferPool;
use crate::prefix::{LZ77_SYMBOLS, RAW_SYMBOLS};
use crate::{
    BackendError, EncodeError, FrameEncodeRequest, FrameIndex, GpuEncodeJob, GpuFrameArtifacts,
    WgpuContext,
};

#[cfg(not(target_arch = "wasm32"))]
impl LosslessModularBackend {
    pub(super) fn submit_streaming(
        &self,
        context: &WgpuContext,
        source: crate::BufferImageSource,
        plan: ModularDispatchPlan,
        request: FrameEncodeRequest,
    ) -> Result<LosslessModularJob, EncodeError> {
        let completion = Arc::new(StreamingCompletion::default());
        let worker_completion = Arc::clone(&completion);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker = StreamingModularWorker {
            context: context.clone(),
            pipeline: Arc::clone(&self.pipeline),
            buffer_pool: Arc::clone(&self.buffer_pool),
            direct_mapping: self.direct_mapping,
            source,
            plan,
            request,
            cancelled: Arc::clone(&cancelled),
        };
        std::thread::Builder::new()
            .name("jxl-wgpu-modular-stream".into())
            .spawn(move || {
                worker_completion.complete(worker.run());
            })
            .map_err(BackendError::StreamingWorkerStart)?;
        Ok(LosslessModularJob {
            state: LosslessModularJobState::Streaming(StreamingLosslessModularJob {
                completion,
                cancelled,
            }),
        })
    }
}

#[cfg(target_arch = "wasm32")]
impl LosslessModularBackend {
    pub(super) fn submit_streaming(
        &self,
        context: &WgpuContext,
        source: crate::BufferImageSource,
        plan: ModularDispatchPlan,
        request: FrameEncodeRequest,
    ) -> Result<LosslessModularJob, EncodeError> {
        Ok(LosslessModularJob {
            state: LosslessModularJobState::Streaming(Box::new(
                BrowserStreamingLosslessModularJob::new(
                    context.clone(),
                    Arc::clone(&self.pipeline),
                    Arc::clone(&self.buffer_pool),
                    self.direct_mapping,
                    source,
                    plan,
                    request,
                )?,
            )),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct StreamingModularWorker {
    context: WgpuContext,
    pipeline: Arc<wgpu::ComputePipeline>,
    buffer_pool: Arc<EncoderBufferPool>,
    direct_mapping: bool,
    source: crate::BufferImageSource,
    plan: ModularDispatchPlan,
    request: FrameEncodeRequest,
    cancelled: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
impl StreamingModularWorker {
    fn run(&self) -> Result<GpuFrameArtifacts, EncodeError> {
        let mut aggregate_raw = [[0u64; RAW_SYMBOLS]; 4];
        let mut aggregate_lz77 = [[0u64; LZ77_SYMBOLS]; 4];
        for batch in &self.plan.batches {
            ensure_streaming_job_active(&self.cancelled)?;
            self.with_batch(batch, |bytes| {
                accumulate_streaming_batch_histograms(
                    &self.plan,
                    batch,
                    bytes,
                    &mut aggregate_raw,
                    &mut aggregate_lz77,
                )
            })?;
        }

        let codes = build_prefix_codes(
            self.plan.format,
            self.plan.bits_per_sample,
            &aggregate_raw,
            &aggregate_lz77,
        )?;
        let frame = ModularFrameHeader {
            animation: self.request.animation,
            canvas_width: self.request.canvas_width,
            canvas_height: self.request.canvas_height,
            options: self.request.options.clone(),
            is_last: self.request.is_last,
        };
        let mut assembler = ModularPacketAssembler::new(
            ModularPacketConfig {
                width: self.plan.width,
                height: self.plan.height,
                group_grid: self.plan.group_grid,
                format: self.plan.format,
                bits_per_sample: self.plan.bits_per_sample,
                tree_mode: self.plan.tree_mode,
                frame,
            },
            codes,
        )?;
        for batch in &self.plan.batches {
            ensure_streaming_job_active(&self.cancelled)?;
            self.with_batch(batch, |bytes| {
                serialize_streaming_batch(&self.plan, batch, bytes, &mut assembler)
            })?;
        }
        let (packets, acceleration) = assembler.finish()?;
        Ok(GpuFrameArtifacts {
            frame_index: self.request.frame_index,
            is_last: self.request.is_last,
            packets,
            acceleration,
        })
    }

    fn with_batch<T>(
        &self,
        batch: &ModularDispatchBatch,
        inspect: impl FnOnce(&[u8]) -> Result<T, EncodeError>,
    ) -> Result<T, EncodeError> {
        let pending = submit_streaming_batch(StreamingBatchContext {
            context: &self.context,
            pipeline: &self.pipeline,
            buffer_pool: &self.buffer_pool,
            direct_mapping: self.direct_mapping,
            source: &self.source,
            plan: &self.plan,
            batch,
        })?;
        let mapping = pending.completion.wait();
        pending.finish(mapping, inspect)
    }
}

struct StreamingBatchContext<'a> {
    context: &'a WgpuContext,
    pipeline: &'a wgpu::ComputePipeline,
    buffer_pool: &'a Arc<EncoderBufferPool>,
    direct_mapping: bool,
    source: &'a crate::BufferImageSource,
    plan: &'a ModularDispatchPlan,
    batch: &'a ModularDispatchBatch,
}

struct PendingStreamingBatch {
    completion: Arc<MapCompletion>,
    lifetime: Arc<EncodeJobLifetime>,
    artifact_bytes: u64,
}

impl PendingStreamingBatch {
    fn finish<T>(
        self,
        mapping: Result<(), BackendError>,
        inspect: impl FnOnce(&[u8]) -> Result<T, EncodeError>,
    ) -> Result<T, EncodeError> {
        mapping?;
        let readback = &self.lifetime.buffer_lease.buffers().readback;
        let mapped = readback
            .slice(0..self.artifact_bytes)
            .get_mapped_range()
            .map_err(BackendError::ArtifactRange)?;
        let expected = usize::try_from(self.artifact_bytes)
            .map_err(|_| EncodeError::Backend("mapped artifact size overflow".into()))?;
        let bytes = mapped
            .get(..expected)
            .ok_or_else(|| EncodeError::Backend("mapped artifact buffer was truncated".into()))?;
        let result = inspect(bytes);
        drop(mapped);
        readback.unmap();
        self.lifetime.mapped.store(false, Ordering::Release);
        drop(self.lifetime);
        result
    }
}

fn submit_streaming_batch(
    submission: StreamingBatchContext<'_>,
) -> Result<PendingStreamingBatch, EncodeError> {
    let StreamingBatchContext {
        context,
        pipeline,
        buffer_pool,
        direct_mapping,
        source,
        plan,
        batch,
    } = submission;
    let parameter_bytes = u64::try_from(batch.dispatch_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<ModularParams>() as u64))
        .ok_or(EncodeError::InvalidSource(
            "streaming parameter buffer size overflow",
        ))?;
    let artifact_bytes = batch.artifact_binding_size.get();
    let owned_bytes = artifact_bytes
        .checked_add(if direct_mapping { 0 } else { artifact_bytes })
        .and_then(|value| value.checked_add(parameter_bytes))
        .ok_or(EncodeError::InvalidSource(
            "streaming batch memory size overflow",
        ))?;
    let memory_permit = context.memory_budget().try_reserve(owned_bytes)?;
    let buffer_lease = buffer_pool.checkout(
        context.device(),
        parameter_bytes,
        artifact_bytes,
        direct_mapping,
    );
    let buffers = buffer_lease.buffers();
    let end_dispatch = batch
        .first_dispatch
        .checked_add(batch.dispatch_count)
        .ok_or(EncodeError::InvalidSource(
            "streaming parameter range overflow",
        ))?;
    let parameters = plan
        .parameters
        .get(batch.first_dispatch..end_dispatch)
        .ok_or(EncodeError::InvalidSource(
            "streaming parameter range is invalid",
        ))?;
    context
        .queue()
        .write_buffer(&buffers.parameters, 0, bytemuck::cast_slice(parameters));
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu streamed lossless modular bindings"),
            layout: &pipeline.get_bind_group_layout(0),
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
                    resource: buffers.artifact.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.parameters.as_entire_binding(),
                },
            ],
        });
    let mut commands = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu streamed lossless modular encode"),
        });
    commands.clear_buffer(&buffers.artifact, 0, None);
    {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu streamed lossless modular tokenization"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            u32::try_from(batch.dispatch_count)
                .map_err(|_| EncodeError::InvalidSource("streaming dispatch count overflow"))?,
            1,
            1,
        );
    }
    if !direct_mapping {
        commands.copy_buffer_to_buffer(&buffers.artifact, 0, &buffers.readback, 0, artifact_bytes);
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
        0..artifact_bytes,
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
    Ok(PendingStreamingBatch {
        completion,
        lifetime,
        artifact_bytes,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_streaming_job_active(cancelled: &AtomicBool) -> Result<(), EncodeError> {
    if cancelled.load(Ordering::Acquire) {
        return Err(BackendError::Invariant("streamed Modular encode was cancelled").into());
    }
    Ok(())
}

fn streaming_artifact_bytes<'a>(
    group: &ModularGroupPlan,
    batch: &ModularDispatchBatch,
    bytes: &'a [u8],
) -> Result<&'a [u8], EncodeError> {
    let start = group
        .artifact_byte_offset
        .checked_sub(batch.artifact_byte_offset)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| EncodeError::Backend("streaming artifact offset overflow".into()))?;
    let end = start
        .checked_add(
            usize::try_from(group.output_size)
                .map_err(|_| EncodeError::Backend("streaming artifact size overflow".into()))?,
        )
        .ok_or_else(|| EncodeError::Backend("streaming artifact range overflow".into()))?;
    bytes
        .get(start..end)
        .ok_or_else(|| EncodeError::Backend("streaming GPU artifact is truncated".into()))
}

fn accumulate_streaming_batch_histograms(
    plan: &ModularDispatchPlan,
    batch: &ModularDispatchBatch,
    bytes: &[u8],
    aggregate_raw: &mut [[u64; RAW_SYMBOLS]; 4],
    aggregate_lz77: &mut [[u64; LZ77_SYMBOLS]; 4],
) -> Result<(), EncodeError> {
    let channels = usize::try_from(plan.format.channel_count())
        .map_err(|_| EncodeError::Backend("Modular channel count overflow".into()))?;
    let end_dispatch = batch
        .first_dispatch
        .checked_add(batch.dispatch_count)
        .ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range overflow",
        ))?;
    for dispatch in batch.first_dispatch..end_dispatch {
        let group_plan = plan.groups.get(dispatch).ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range is invalid",
        ))?;
        let artifact_bytes = streaming_artifact_bytes(group_plan, batch, bytes)?;
        let header = parse_group_artifact_header(group_plan.max_events, artifact_bytes)?;
        accumulate_artifact_histograms(
            dispatch % channels,
            &ValidatedModularArtifact {
                header,
                events: &[],
            },
            aggregate_raw,
            aggregate_lz77,
        )?;
    }
    Ok(())
}

fn serialize_streaming_batch(
    plan: &ModularDispatchPlan,
    batch: &ModularDispatchBatch,
    bytes: &[u8],
    assembler: &mut ModularPacketAssembler,
) -> Result<(), EncodeError> {
    let channels = usize::try_from(plan.format.channel_count())
        .map_err(|_| EncodeError::Backend("Modular channel count overflow".into()))?;
    if !batch.first_dispatch.is_multiple_of(channels)
        || !batch.dispatch_count.is_multiple_of(channels)
    {
        return Err(EncodeError::Backend(
            "streaming batch splits a Modular channel group".into(),
        ));
    }
    let end_dispatch = batch
        .first_dispatch
        .checked_add(batch.dispatch_count)
        .ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range overflow",
        ))?;
    for first_channel in (batch.first_dispatch..end_dispatch).step_by(channels) {
        let mut artifacts = Vec::with_capacity(channels);
        for dispatch in first_channel..first_channel + channels {
            let group_plan = plan.groups.get(dispatch).ok_or(EncodeError::InvalidSource(
                "artifact batch dispatch range is invalid",
            ))?;
            artifacts.push(parse_group_artifact(
                group_plan.width,
                group_plan.height,
                group_plan.max_events,
                streaming_artifact_bytes(group_plan, batch, bytes)?,
            )?);
        }
        let group_index = u32::try_from(first_channel / channels)
            .map_err(|_| EncodeError::Backend("Modular group index overflow".into()))?;
        assembler.push_group(group_index, &artifacts)?;
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct MapCompletion {
    pub(super) state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
pub(super) struct MapState {
    result: Option<Result<(), BackendError>>,
    waker: Option<Waker>,
}

impl MapCompletion {
    pub(super) fn complete(&self, result: Result<(), BackendError>) {
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

    pub(super) fn poll(&self, cx: &Context<'_>) -> Option<Result<(), BackendError>> {
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
    fn wait(&self) -> Result<(), BackendError> {
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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub(super) struct StreamingCompletion {
    state: Mutex<StreamingCompletionState>,
    condition: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct StreamingCompletionState {
    result: Option<Result<GpuFrameArtifacts, EncodeError>>,
    waker: Option<Waker>,
}

#[cfg(not(target_arch = "wasm32"))]
impl StreamingCompletion {
    pub(super) fn complete(&self, result: Result<GpuFrameArtifacts, EncodeError>) {
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

    fn poll(&self, cx: &Context<'_>) -> Option<Result<GpuFrameArtifacts, EncodeError>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    fn wait(&self) -> Result<GpuFrameArtifacts, EncodeError> {
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
            .expect("streaming completion was checked as present")
    }
}

#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StreamingPass {
    Histogram,
    Serialize,
}

#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StreamingCursor {
    pub(super) pass: StreamingPass,
    pub(super) batch_index: usize,
    pub(super) batch_count: usize,
}

#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StreamingAdvance {
    SubmitNext,
    BeginSerialization,
    Complete,
}

#[cfg(any(test, target_arch = "wasm32"))]
impl StreamingCursor {
    pub(super) fn new(batch_count: usize) -> Result<Self, EncodeError> {
        if batch_count == 0 {
            return Err(BackendError::Invariant("streaming dispatch plan has no batches").into());
        }
        Ok(Self {
            pass: StreamingPass::Histogram,
            batch_index: 0,
            batch_count,
        })
    }

    pub(super) fn advance(&mut self) -> StreamingAdvance {
        if self.batch_index + 1 < self.batch_count {
            self.batch_index += 1;
            return StreamingAdvance::SubmitNext;
        }
        match self.pass {
            StreamingPass::Histogram => {
                self.pass = StreamingPass::Serialize;
                self.batch_index = 0;
                StreamingAdvance::BeginSerialization
            }
            StreamingPass::Serialize => StreamingAdvance::Complete,
        }
    }
}

/// Runtime-neutral completion for the concrete GPU lossless profile.
pub struct LosslessModularJob {
    pub(super) state: LosslessModularJobState,
}

pub(super) enum LosslessModularJobState {
    Resident(ResidentLosslessModularJob),
    #[cfg(not(target_arch = "wasm32"))]
    Streaming(StreamingLosslessModularJob),
    #[cfg(target_arch = "wasm32")]
    Streaming(Box<BrowserStreamingLosslessModularJob>),
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct StreamingLosslessModularJob {
    pub(super) completion: Arc<StreamingCompletion>,
    pub(super) cancelled: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for StreamingLosslessModularJob {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

pub(super) struct ResidentLosslessModularJob {
    pub(super) lifetime: Option<Arc<EncodeJobLifetime>>,
    pub(super) completion: Arc<MapCompletion>,
    pub(super) output_size: u64,
    pub(super) group_grid: LosslessModularGroupGrid,
    pub(super) groups: Vec<ModularGroupPlan>,
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frame_index: FrameIndex,
    pub(super) is_last: bool,
    pub(super) header: ModularFrameHeader,
}

pub(super) struct EncodeJobLifetime {
    pub(super) buffer_lease: crate::buffer_pool::EncoderBufferLease,
    pub(super) _memory_permit: MemoryPermit,
    pub(super) mapped: AtomicBool,
}

impl Drop for EncodeJobLifetime {
    fn drop(&mut self) {
        if self.mapped.swap(false, Ordering::AcqRel) {
            self.buffer_lease.buffers().readback.unmap();
        }
    }
}

/// Browser WebGPU streams one mapped batch at a time from its event loop. The future itself is
/// the scheduler: map callbacks only publish completion and wake the caller's executor, while the
/// next queue submission is recorded by the following poll. No Web Worker or async runtime is
/// required, and abandoning the future leaves the active map callback owning its budgeted lease.
#[cfg(target_arch = "wasm32")]
pub(super) struct BrowserStreamingLosslessModularJob {
    context: WgpuContext,
    pipeline: Arc<wgpu::ComputePipeline>,
    buffer_pool: Arc<EncoderBufferPool>,
    direct_mapping: bool,
    source: crate::BufferImageSource,
    plan: ModularDispatchPlan,
    request: FrameEncodeRequest,
    cursor: StreamingCursor,
    pending: Option<PendingStreamingBatch>,
    aggregate_raw: [[u64; RAW_SYMBOLS]; 4],
    aggregate_lz77: [[u64; LZ77_SYMBOLS]; 4],
    assembler: Option<ModularPacketAssembler>,
}

#[cfg(target_arch = "wasm32")]
impl BrowserStreamingLosslessModularJob {
    pub(super) fn new(
        context: WgpuContext,
        pipeline: Arc<wgpu::ComputePipeline>,
        buffer_pool: Arc<EncoderBufferPool>,
        direct_mapping: bool,
        source: crate::BufferImageSource,
        plan: ModularDispatchPlan,
        request: FrameEncodeRequest,
    ) -> Result<Self, EncodeError> {
        let cursor = StreamingCursor::new(plan.batches.len())?;
        let mut job = Self {
            context,
            pipeline,
            buffer_pool,
            direct_mapping,
            source,
            plan,
            request,
            cursor,
            pending: None,
            aggregate_raw: [[0; RAW_SYMBOLS]; 4],
            aggregate_lz77: [[0; LZ77_SYMBOLS]; 4],
            assembler: None,
        };
        job.submit_current_batch()?;
        Ok(job)
    }

    fn submit_current_batch(&mut self) -> Result<(), EncodeError> {
        if self.pending.is_some() {
            return Err(BackendError::Invariant(
                "browser Modular scheduler already has an active batch",
            )
            .into());
        }
        let batch =
            self.plan
                .batches
                .get(self.cursor.batch_index)
                .ok_or(BackendError::Invariant(
                    "browser Modular scheduler batch index is out of range",
                ))?;
        self.pending = Some(submit_streaming_batch(StreamingBatchContext {
            context: &self.context,
            pipeline: &self.pipeline,
            buffer_pool: &self.buffer_pool,
            direct_mapping: self.direct_mapping,
            source: &self.source,
            plan: &self.plan,
            batch,
        })?);
        Ok(())
    }

    fn begin_serialization(&mut self) -> Result<(), EncodeError> {
        let codes = build_prefix_codes(
            self.plan.format,
            self.plan.bits_per_sample,
            &self.aggregate_raw,
            &self.aggregate_lz77,
        )?;
        self.assembler = Some(ModularPacketAssembler::new(
            ModularPacketConfig {
                width: self.plan.width,
                height: self.plan.height,
                group_grid: self.plan.group_grid,
                format: self.plan.format,
                bits_per_sample: self.plan.bits_per_sample,
                tree_mode: self.plan.tree_mode,
                frame: ModularFrameHeader {
                    animation: self.request.animation,
                    canvas_width: self.request.canvas_width,
                    canvas_height: self.request.canvas_height,
                    options: self.request.options.clone(),
                    is_last: self.request.is_last,
                },
            },
            codes,
        )?);
        Ok(())
    }

    fn finish(&mut self) -> Result<GpuFrameArtifacts, EncodeError> {
        let assembler = self.assembler.take().ok_or(BackendError::Invariant(
            "browser Modular serialization finished without an assembler",
        ))?;
        let (packets, acceleration) = assembler.finish()?;
        Ok(GpuFrameArtifacts {
            frame_index: self.request.frame_index,
            is_last: self.request.is_last,
            packets,
            acceleration,
        })
    }

    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        loop {
            let mapping = match self.pending.as_ref() {
                Some(pending) => match pending.completion.poll(cx) {
                    Some(mapping) => mapping,
                    None => return Poll::Pending,
                },
                None => {
                    return Poll::Ready(Err(BackendError::Invariant(
                        "browser Modular scheduler has no active batch",
                    )
                    .into()));
                }
            };
            let pending = self
                .pending
                .take()
                .expect("the browser Modular batch was checked as present");
            let batch = match self.plan.batches.get(self.cursor.batch_index) {
                Some(batch) => *batch,
                None => {
                    return Poll::Ready(Err(BackendError::Invariant(
                        "browser Modular scheduler batch index is out of range",
                    )
                    .into()));
                }
            };
            let inspected = match self.cursor.pass {
                StreamingPass::Histogram => pending.finish(mapping, |bytes| {
                    accumulate_streaming_batch_histograms(
                        &self.plan,
                        &batch,
                        bytes,
                        &mut self.aggregate_raw,
                        &mut self.aggregate_lz77,
                    )
                }),
                StreamingPass::Serialize => {
                    let Some(assembler) = self.assembler.as_mut() else {
                        return Poll::Ready(Err(BackendError::Invariant(
                            "browser Modular serialization has no assembler",
                        )
                        .into()));
                    };
                    pending.finish(mapping, |bytes| {
                        serialize_streaming_batch(&self.plan, &batch, bytes, assembler)
                    })
                }
            };
            if let Err(error) = inspected {
                return Poll::Ready(Err(error));
            }
            match self.cursor.advance() {
                StreamingAdvance::SubmitNext => {}
                StreamingAdvance::BeginSerialization => {
                    if let Err(error) = self.begin_serialization() {
                        return Poll::Ready(Err(error));
                    }
                }
                StreamingAdvance::Complete => {
                    return Poll::Ready(self.finish());
                }
            }
            if let Err(error) = self.submit_current_batch() {
                return Poll::Ready(Err(error));
            }
            // Register the current executor waker with the newly submitted map before returning.
            // If WebGPU completed it synchronously, consume it in this same poll instead.
        }
    }
}

impl ResidentLosslessModularJob {
    fn finish(
        &mut self,
        mapping: Result<(), BackendError>,
    ) -> Result<GpuFrameArtifacts, EncodeError> {
        let lifetime = self
            .lifetime
            .take()
            .ok_or_else(|| EncodeError::Backend("GPU job was already consumed".into()))?;
        mapping?;
        let readback = &lifetime.buffer_lease.buffers().readback;
        let mapped = readback
            .slice(0..self.output_size)
            .get_mapped_range()
            .map_err(BackendError::ArtifactRange)?;
        let expected = usize::try_from(self.output_size)
            .map_err(|_| EncodeError::Backend("mapped artifact size overflow".into()))?;
        let bytes = mapped
            .get(..expected)
            .ok_or_else(|| EncodeError::Backend("mapped artifact buffer was truncated".into()))?;
        let result = build_packets(PacketBuildInput {
            width: self.width,
            height: self.height,
            group_grid: self.group_grid,
            format: self.format,
            bits_per_sample: self.bits_per_sample,
            tree_mode: self.tree_mode,
            frame: &self.header,
            group_plans: &self.groups,
            bytes,
        });
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

impl GpuEncodeJob for LosslessModularJob {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match &mut self.state {
            LosslessModularJobState::Resident(job) => match job.completion.poll(cx) {
                Some(result) => Poll::Ready(job.finish(result)),
                None => Poll::Pending,
            },
            #[cfg(not(target_arch = "wasm32"))]
            LosslessModularJobState::Streaming(job) => match job.completion.poll(cx) {
                Some(result) => Poll::Ready(result),
                None => Poll::Pending,
            },
            #[cfg(target_arch = "wasm32")]
            LosslessModularJobState::Streaming(job) => job.poll_complete(cx),
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            match self.state {
                LosslessModularJobState::Resident(mut job) => {
                    let result = job.completion.wait();
                    job.finish(result)
                }
                LosslessModularJobState::Streaming(job) => job.completion.wait(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(EncodeError::Backend(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission".into(),
            ))
        }
    }
}
