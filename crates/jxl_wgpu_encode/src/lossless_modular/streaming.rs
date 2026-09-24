use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_wgpu::MemoryPermit;

use super::dispatch::{
    LosslessModularBackend, ModularDispatchBatch, ModularDispatchPlan, ModularGroupPlan,
};
use super::entropy::{
    AnsCodebook, AnsPipelines, EntropyCode, FrameHistograms, validate_encoded_group,
};
use super::grid::LosslessModularGroupGrid;
use super::lz77::LosslessModularLz77;
use super::predictor::{LosslessModularPredictor, LosslessModularWeightedPredictor};
use super::serializer::{
    ModularPacketAssembler, ModularPacketConfig, PacketBuildInput, ValidatedModularArtifact,
    accumulate_artifact_histograms, build_packets, parse_group_artifact_header,
    parse_planned_artifact,
};
use super::transform::ModularTransformPlan;
use super::types::{LosslessModularFormat, LosslessModularTreeMode, ModularParams};
use crate::buffer_pool::EncoderBufferPool;
use crate::frame_header::FrameHeaderPlan;
use crate::{BackendError, EncodeError, GpuEncodeJob, GpuFrameArtifacts, WgpuContext};

#[cfg(not(target_arch = "wasm32"))]
impl LosslessModularBackend {
    pub(super) fn submit_streaming(
        &self,
        context: &WgpuContext,
        source: crate::BufferImageSource,
        plan: ModularDispatchPlan,
        header: FrameHeaderPlan,
    ) -> Result<LosslessModularJob, EncodeError> {
        let completion = Arc::new(StreamingCompletion::default());
        let worker_completion = Arc::clone(&completion);
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker = StreamingModularWorker {
            context: context.clone(),
            pipeline: Arc::clone(self.pipeline()?),
            ans_pipeline: self.ans_pipeline.clone(),
            buffer_pool: Arc::clone(&self.buffer_pool),
            direct_mapping: self.direct_mapping,
            source,
            plan,
            header,
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
        header: FrameHeaderPlan,
    ) -> Result<LosslessModularJob, EncodeError> {
        Ok(LosslessModularJob {
            state: LosslessModularJobState::Streaming(Box::new(
                BrowserStreamingLosslessModularJob::new(
                    context.clone(),
                    self,
                    source,
                    plan,
                    header,
                )?,
            )),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct StreamingModularWorker {
    context: WgpuContext,
    pipeline: Arc<wgpu::ComputePipeline>,
    ans_pipeline: Option<Arc<AnsPipelines>>,
    buffer_pool: Arc<EncoderBufferPool>,
    direct_mapping: bool,
    source: crate::BufferImageSource,
    plan: ModularDispatchPlan,
    header: FrameHeaderPlan,
    cancelled: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
impl StreamingModularWorker {
    // Consume the worker so its source and plan are released before completion is published.
    fn run(self) -> Result<GpuFrameArtifacts, EncodeError> {
        let mut histograms = FrameHistograms::default();
        for batch in &self.plan.batches {
            ensure_streaming_job_active(&self.cancelled)?;
            self.with_batch(batch, None, |bytes| {
                accumulate_streaming_batch_histograms(&self.plan, batch, bytes, &mut histograms)
            })?;
        }

        let entropy = Arc::new(EntropyCode::from_histograms(&self.plan, &histograms)?);
        let frame = self.header.clone();
        let mut assembler = ModularPacketAssembler::new(
            ModularPacketConfig {
                width: self.plan.width,
                height: self.plan.height,
                group_grid: self.plan.group_grid,
                format: self.plan.format,
                bits_per_sample: self.plan.bits_per_sample,
                exponent_bits_per_sample: self.plan.exponent_bits_per_sample,
                tree_mode: self.plan.tree_mode,
                transforms: Arc::clone(&self.plan.transforms),
                predictor: self.plan.predictor,
                weighted_predictor: self.plan.weighted_predictor,
                lz77: self.plan.lz77,
                frame,
            },
            Arc::clone(&entropy),
        )?;
        for batch in &self.plan.batches {
            ensure_streaming_job_active(&self.cancelled)?;
            self.with_batch(batch, entropy.ans(), |bytes| {
                serialize_streaming_batch(&self.plan, batch, bytes, &mut assembler)
            })?;
        }
        let (packets, acceleration) = assembler.finish()?;
        Ok(GpuFrameArtifacts {
            frame_index: self.header.frame_index(),
            is_last: self.header.is_last(),
            packets,
            acceleration,
        })
    }

    fn with_batch<T>(
        &self,
        batch: &ModularDispatchBatch,
        codebook: Option<&AnsCodebook>,
        inspect: impl FnOnce(&[u8]) -> Result<T, EncodeError>,
    ) -> Result<T, EncodeError> {
        let pending = submit_streaming_batch(StreamingBatchContext {
            context: &self.context,
            pipeline: &self.pipeline,
            entropy: entropy_stage(self.ans_pipeline.as_deref(), codebook)?,
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

fn entropy_stage<'a>(
    pipeline: Option<&'a AnsPipelines>,
    codebook: Option<&'a AnsCodebook>,
) -> Result<Option<(&'a wgpu::ComputePipeline, Option<&'a AnsCodebook>)>, EncodeError> {
    match (pipeline, codebook) {
        (Some(pipelines), codebook) => Ok(Some((
            if codebook.is_some() {
                &pipelines.encode
            } else {
                &pipelines.profile
            },
            codebook,
        ))),
        (None, None) => Ok(None),
        (None, Some(_)) => {
            Err(BackendError::Invariant("ANS serialization pipeline is absent").into())
        }
    }
}

struct StreamingBatchContext<'a> {
    context: &'a WgpuContext,
    pipeline: &'a wgpu::ComputePipeline,
    entropy: Option<(&'a wgpu::ComputePipeline, Option<&'a AnsCodebook>)>,
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
        entropy,
        buffer_pool,
        direct_mapping,
        source,
        plan,
        batch,
    } = submission;
    let parameter_bytes = batch.parameter_bytes;
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
    let uploads = plan.upload_parameters(context.queue(), &buffers.parameters, *batch)?;
    let [source0, source1, source2, source3] = batch.source_windows.entries(&source.buffer);
    let bind_group = context
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu streamed lossless modular bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                source0,
                source1,
                source2,
                source3,
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.artifact.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers.parameters,
                        offset: 0,
                        size: std::num::NonZeroU64::new(
                            batch.dispatch_count as u64
                                * std::mem::size_of::<ModularParams>() as u64,
                        ),
                    }),
                },
            ],
        });
    let mut commands = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu streamed lossless modular encode"),
        });
    commands.clear_buffer(&buffers.artifact, 0, None);
    for upload in uploads {
        upload.record(&mut commands, &buffers.parameters, &buffers.artifact);
    }
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
    if let Some((pipeline, codebook)) = entropy {
        super::entropy::record(
            super::entropy::AnsSubmission {
                plan,
                batch,
                codebook,
                context,
                pipeline,
                parameters: &buffers.parameters,
                artifact: &buffers.artifact,
            },
            &mut commands,
        )?;
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
            callback_completion.complete_mapping(
                callback_lifetime,
                result.map_err(BackendError::ArtifactMapping),
            );
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
    aggregate: &mut FrameHistograms,
) -> Result<(), EncodeError> {
    let end_dispatch = batch
        .first_dispatch
        .checked_add(batch.dispatch_count)
        .ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range overflow",
        ))?;
    let mut histograms = FrameHistograms::default();
    for dispatch in batch.first_dispatch..end_dispatch {
        let group = plan.groups.get(dispatch).ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range is invalid",
        ))?;
        let data = streaming_artifact_bytes(group, batch, bytes)?;
        let artifact = if batch.entropy.is_some() {
            parse_planned_artifact(group, data)?
        } else {
            ValidatedModularArtifact {
                header: parse_group_artifact_header(group.max_events, data)?,
                events: &[],
                palette_counts: None,
            }
        };
        accumulate_artifact_histograms(
            group.channel as usize,
            &artifact,
            &mut histograms.raw,
            &mut histograms.lz77,
            &mut histograms.distance,
        )?;
    }
    if let Some(entropy) = batch.entropy {
        histograms.read_profiles(
            plan.lz77,
            entropy.profile_byte_offset,
            batch.dispatch_count,
            bytes,
        )?;
    }
    aggregate.accumulate(histograms)
}

fn serialize_streaming_batch(
    plan: &ModularDispatchPlan,
    batch: &ModularDispatchBatch,
    bytes: &[u8],
    assembler: &mut ModularPacketAssembler,
) -> Result<(), EncodeError> {
    let end_dispatch = batch
        .first_dispatch
        .checked_add(batch.dispatch_count)
        .ok_or(EncodeError::InvalidSource(
            "artifact batch dispatch range overflow",
        ))?;
    let group_plans =
        plan.groups
            .get(batch.first_dispatch..end_dispatch)
            .ok_or(BackendError::Invariant(
                "artifact batch dispatch range is invalid",
            ))?;
    for channels in group_plans.chunk_by(|a, b| a.group_index == b.group_index) {
        let mut artifacts = Vec::with_capacity(channels.len());
        for (channel, group_plan) in channels.iter().enumerate() {
            if group_plan.channel != channel as u32 {
                return Err(BackendError::Invariant(
                    "streaming batch splits a Modular channel group",
                )
                .into());
            }
            artifacts.push(parse_planned_artifact(
                group_plan,
                streaming_artifact_bytes(group_plan, batch, bytes)?,
            )?);
        }
        let encoded = channels[0]
            .entropy
            .map(|entropy| {
                validate_encoded_group(
                    entropy,
                    streaming_artifact_bytes(&channels[0], batch, bytes)?,
                    &artifacts,
                )
            })
            .transpose()?;
        assembler.push_group(channels[0].group_index, &artifacts, encoded)?;
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
    pub(super) fn complete_mapping(
        &self,
        lifetime: Arc<EncodeJobLifetime>,
        result: Result<(), BackendError>,
    ) {
        if result.is_ok() {
            lifetime.mapped.store(true, Ordering::Release);
        }
        // A waiter can run synchronously from wake(), or on another thread immediately.
        // It must own the only remaining job reference before it can finish or reuse the budget.
        drop(lifetime);
        self.complete(result);
    }

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
    Resident(Box<ResidentLosslessModularJob>),
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
    pub(super) exponent_bits_per_sample: u8,
    pub(super) tree_mode: LosslessModularTreeMode,
    pub(super) transforms: Arc<ModularTransformPlan>,
    pub(super) predictor: LosslessModularPredictor,
    pub(super) weighted_predictor: LosslessModularWeightedPredictor,
    pub(super) lz77: LosslessModularLz77,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) header: FrameHeaderPlan,
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
    ans_pipeline: Option<Arc<AnsPipelines>>,
    buffer_pool: Arc<EncoderBufferPool>,
    direct_mapping: bool,
    source: crate::BufferImageSource,
    plan: ModularDispatchPlan,
    header: FrameHeaderPlan,
    cursor: StreamingCursor,
    pending: Option<PendingStreamingBatch>,
    histograms: FrameHistograms,
    assembler: Option<ModularPacketAssembler>,
}

#[cfg(target_arch = "wasm32")]
impl BrowserStreamingLosslessModularJob {
    pub(super) fn new(
        context: WgpuContext,
        backend: &LosslessModularBackend,
        source: crate::BufferImageSource,
        plan: ModularDispatchPlan,
        header: FrameHeaderPlan,
    ) -> Result<Self, EncodeError> {
        let cursor = StreamingCursor::new(plan.batches.len())?;
        let mut job = Self {
            context,
            pipeline: Arc::clone(backend.pipeline()?),
            ans_pipeline: backend.ans_pipeline.clone(),
            buffer_pool: Arc::clone(&backend.buffer_pool),
            direct_mapping: backend.direct_mapping,
            source,
            plan,
            header,
            cursor,
            pending: None,
            histograms: FrameHistograms::default(),
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
            entropy: entropy_stage(
                self.ans_pipeline.as_deref(),
                self.assembler
                    .as_ref()
                    .and_then(|assembler| assembler.entropy().ans()),
            )?,
            buffer_pool: &self.buffer_pool,
            direct_mapping: self.direct_mapping,
            source: &self.source,
            plan: &self.plan,
            batch,
        })?);
        Ok(())
    }

    fn begin_serialization(&mut self) -> Result<(), EncodeError> {
        let entropy = Arc::new(EntropyCode::from_histograms(&self.plan, &self.histograms)?);
        self.assembler = Some(ModularPacketAssembler::new(
            ModularPacketConfig {
                width: self.plan.width,
                height: self.plan.height,
                group_grid: self.plan.group_grid,
                format: self.plan.format,
                bits_per_sample: self.plan.bits_per_sample,
                exponent_bits_per_sample: self.plan.exponent_bits_per_sample,
                tree_mode: self.plan.tree_mode,
                transforms: Arc::clone(&self.plan.transforms),
                predictor: self.plan.predictor,
                weighted_predictor: self.plan.weighted_predictor,
                lz77: self.plan.lz77,
                frame: self.header.clone(),
            },
            entropy,
        )?);
        Ok(())
    }

    fn finish(&mut self) -> Result<GpuFrameArtifacts, EncodeError> {
        let assembler = self.assembler.take().ok_or(BackendError::Invariant(
            "browser Modular serialization finished without an assembler",
        ))?;
        let (packets, acceleration) = assembler.finish()?;
        Ok(GpuFrameArtifacts {
            frame_index: self.header.frame_index(),
            is_last: self.header.is_last(),
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
                        &mut self.histograms,
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
            exponent_bits_per_sample: self.exponent_bits_per_sample,
            tree_mode: self.tree_mode,
            transforms: Arc::clone(&self.transforms),
            predictor: self.predictor,
            weighted_predictor: self.weighted_predictor,
            lz77: self.lz77,
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
            frame_index: self.header.frame_index(),
            is_last: self.header.is_last(),
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lz77_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod completion_tests;
