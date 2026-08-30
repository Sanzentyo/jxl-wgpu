// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::future::Future;
use std::num::NonZeroU64;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

#[cfg(not(target_arch = "wasm32"))]
use jxl_gpu_protocol::PlaneData;
use jxl_gpu_protocol::{Extent2d, OutputDesc, OutputId, RenderedOutput, SampleType};

use crate::upload::aligned_buffer_size;
use crate::{
    CpuImageFrame, CpuImageOutput, Error, GpuBufferLease, GpuImageFrame, ImageLayout, MemoryBudget,
    MemoryBudgetSnapshot, MemoryPermit, Result, UnvalidatedGpuImageFrame,
};

#[derive(Debug)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct ReadbackRequest {
    pub id: OutputId,
    pub extent: Extent2d,
    pub sample_type: SampleType,
    pub channels: u8,
    pub logical_size: u64,
    pub buffer: Arc<wgpu::Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    mapping: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

pub(crate) fn stage_output(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    output: &OutputDesc,
    source: &Arc<wgpu::Buffer>,
    logical_size: u64,
    direct: bool,
) -> Result<ReadbackRequest> {
    let buffer = if direct {
        Arc::clone(source)
    } else {
        let padded_size = aligned_buffer_size(logical_size)?;
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu output readback"),
            size: padded_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, padded_size);
        buffer
    };
    #[cfg(not(target_arch = "wasm32"))]
    let mapping = {
        let (sender, receiver) = mpsc::sync_channel(1);
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        receiver
    };
    Ok(ReadbackRequest {
        id: output.id,
        extent: output.extent,
        sample_type: output.sample_type,
        channels: output.channels,
        logical_size,
        buffer,
        #[cfg(not(target_arch = "wasm32"))]
        mapping,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_outputs(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    requests: Vec<ReadbackRequest>,
) -> Result<Vec<RenderedOutput>> {
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;

    requests
        .into_iter()
        .map(|request| {
            request.mapping.recv().map_err(|_| {
                Error::Execution(format!(
                    "mapping callback was dropped for output {:?}",
                    request.id
                ))
            })??;
            decode_output(request)
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_outputs(
    _device: &wgpu::Device,
    _submission: wgpu::SubmissionIndex,
    _requests: Vec<ReadbackRequest>,
) -> Result<Vec<RenderedOutput>> {
    Err(Error::Unsupported(
        "the synchronous FrameSession::wait API cannot block browser WebGPU".into(),
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_output(request: ReadbackRequest) -> Result<RenderedOutput> {
    let byte_len = usize::try_from(request.logical_size).map_err(|_| Error::BufferSizeOverflow)?;
    let slice = request.buffer.slice(..);
    let mapped = slice
        .get_mapped_range()
        .map_err(|error| Error::Execution(format!("mapped output range is invalid: {error}")))?;
    let bytes = mapped
        .get(..byte_len)
        .ok_or_else(|| Error::Execution("mapped output was shorter than requested".into()))?;
    let expected_samples = request
        .extent
        .area()
        .and_then(|area| area.checked_mul(usize::from(request.channels)))
        .ok_or(Error::BufferSizeOverflow)?;
    let data = decode_plane_data(request.sample_type, bytes, expected_samples)?;
    drop(mapped);
    request.buffer.unmap();
    Ok(RenderedOutput {
        id: request.id,
        extent: request.extent,
        data,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_plane_data(
    sample_type: SampleType,
    bytes: &[u8],
    expected_samples: usize,
) -> Result<PlaneData> {
    let invalid_length = || {
        Error::Execution(format!(
            "readback byte count {} does not encode {expected_samples} {sample_type:?} samples",
            bytes.len()
        ))
    };
    match sample_type {
        SampleType::I32 => bytemuck::try_cast_slice::<u8, i32>(bytes)
            .map(|samples| PlaneData::I32(samples.to_vec()))
            .map_err(|_| invalid_length())
            .and_then(|data| ensure_sample_count(data, expected_samples)),
        SampleType::F32 => bytemuck::try_cast_slice::<u8, f32>(bytes)
            .map(|samples| PlaneData::F32(samples.to_vec()))
            .map_err(|_| invalid_length())
            .and_then(|data| ensure_sample_count(data, expected_samples)),
        SampleType::F16 => bytemuck::try_cast_slice::<u8, u16>(bytes)
            .map(|samples| PlaneData::F16(samples.to_vec()))
            .map_err(|_| invalid_length())
            .and_then(|data| ensure_sample_count(data, expected_samples)),
        SampleType::U16 => bytemuck::try_cast_slice::<u8, u16>(bytes)
            .map(|samples| PlaneData::U16(samples.to_vec()))
            .map_err(|_| invalid_length())
            .and_then(|data| ensure_sample_count(data, expected_samples)),
        SampleType::U8 => ensure_sample_count(PlaneData::U8(bytes.to_vec()), expected_samples),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_sample_count(data: PlaneData, expected: usize) -> Result<PlaneData> {
    if data.len() == expected {
        Ok(data)
    } else {
        Err(Error::Execution(format!(
            "readback produced {} samples, expected {expected}",
            data.len()
        )))
    }
}

/// Memory limit applied to one aggregate generic-image readback submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageReadbackLimits {
    /// Maximum staging bytes in one readback submission.
    pub max_transient_bytes: u64,
    /// Maximum staging bytes held by all concurrent submissions sharing this pipeline budget.
    pub max_in_flight_bytes: u64,
}

impl Default for ImageReadbackLimits {
    fn default() -> Self {
        Self {
            max_transient_bytes: 256 * 1024 * 1024,
            max_in_flight_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Byte accounting known before a generic-image readback is submitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageReadbackStats {
    /// Number of source frames represented by this submission.
    pub frame_count: usize,
    pub output_count: usize,
    /// Sum of the addressable bytes declared by all image layouts.
    pub logical_bytes: u64,
    /// Size of the single aggregate staging buffer, including four-byte copy padding, or zero for
    /// an in-place direct map.
    pub staging_bytes: u64,
    /// Bytes present only to satisfy four-byte buffer-copy alignment.
    pub padding_bytes: u64,
    /// `true` when a single caller-visible output was mapped in place on a supported native
    /// unified-memory backend instead of allocating and copying to staging memory.
    pub direct_mapped: bool,
}

/// Completed explicit CPU readback of one decoder-owned GPU image frame.
#[derive(Clone, Debug)]
pub struct ImageReadbackResult {
    pub frame: CpuImageFrame,
    pub stats: ImageReadbackStats,
}

/// Completed explicit CPU readback of multiple decoder-owned GPU image frames.
#[derive(Clone, Debug)]
pub struct ImageReadbackBatchResult {
    pub frames: Vec<CpuImageFrame>,
    pub stats: ImageReadbackStats,
}

/// Completed explicit readback whose producing codec frame has not been validated.
///
/// No changed regions or authoritative frame metadata are exposed. The bytes must be discarded if
/// the producer later reports validation failure.
#[derive(Clone, Debug)]
pub struct UnvalidatedImageReadbackResult {
    pub token: jxl_gpu_protocol::SubmissionToken,
    pub outputs: Vec<CpuImageOutput>,
    pub stats: ImageReadbackStats,
}

struct ImageReadbackPipelineInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    poller: crate::SubmissionPoller,
    limits: ImageReadbackLimits,
    memory_budget: MemoryBudget,
}

/// Reusable aggregate readback state for generic pitch-linear decoder output.
///
/// [`Self::submit_frames`] copies every output across every supplied [`GpuImageFrame`] into one
/// aggregate `MAP_READ` buffer and records all copies in one command buffer. [`Self::submit`]
/// instead maps a sole output in place when its producer supplied `MAP_READ` usage. Source and destination
/// copy ranges are padded independently to four bytes, while returned byte vectors contain only
/// each layout's `logical_size` bytes. This is an explicit host transfer of GPU decode results; it
/// is not a CPU codec fallback or a claim that the codec submissions themselves were batched.
#[derive(Clone)]
pub struct ImageReadbackPipeline {
    inner: Arc<ImageReadbackPipelineInner>,
}

impl std::fmt::Debug for ImageReadbackPipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageReadbackPipeline")
            .field("limits", &self.inner.limits)
            .finish_non_exhaustive()
    }
}

impl ImageReadbackPipeline {
    /// Uses the backend's transient-memory policy and exact device/queue pair.
    #[must_use]
    pub fn new(backend: &crate::WgpuBackend) -> Self {
        Self {
            inner: Arc::new(ImageReadbackPipelineInner {
                device: backend.device().clone(),
                queue: backend.queue().clone(),
                poller: backend.submission_poller().clone(),
                limits: ImageReadbackLimits {
                    max_transient_bytes: backend.config.memory.max_transient_bytes,
                    max_in_flight_bytes: backend.config.memory.max_in_flight_transient_bytes,
                },
                memory_budget: backend.transient_memory_budget().clone(),
            }),
        }
    }

    /// Creates an application-owned readback pipeline for outputs from the same device and queue.
    ///
    /// # Errors
    ///
    /// Returns an execution error if the native bounded poll worker cannot be created.
    pub fn from_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: ImageReadbackLimits,
    ) -> Result<Self> {
        let memory_limit = NonZeroU64::new(limits.max_in_flight_bytes).ok_or_else(|| {
            Error::ResourceLimit("readback max_in_flight_bytes must be greater than zero".into())
        })?;
        let poller = crate::SubmissionPoller::new(device.clone()).map_err(|error| {
            Error::Execution(format!(
                "could not start the bounded GPU poll worker: {error}"
            ))
        })?;
        Ok(Self {
            inner: Arc::new(ImageReadbackPipelineInner {
                device,
                queue,
                poller,
                limits,
                memory_budget: MemoryBudget::new(memory_limit),
            }),
        })
    }

    #[must_use]
    pub fn limits(&self) -> ImageReadbackLimits {
        self.inner.limits
    }

    /// Returns aggregate byte-weighted admission state shared by all pipeline clones.
    #[must_use]
    pub fn memory_stats(&self) -> MemoryBudgetSnapshot {
        self.inner.memory_budget.snapshot()
    }

    /// Records one frame's outputs as one aggregate copy and returns without a host wait.
    ///
    /// Use [`Self::submit_frames`] when multiple already-produced GPU frames should share one
    /// staging allocation, queue submission, mapping callback, and completion future.
    pub fn submit(&self, frame: &GpuImageFrame) -> Result<ImageReadbackSubmission> {
        if frame.outputs.len() == 1
            && frame.outputs[0]
                .buffer
                .usage()
                .contains(wgpu::BufferUsages::MAP_READ)
        {
            return self
                .submit_direct(frame)
                .map(ImageReadbackSubmission::from_batch);
        }
        self.submit_frames(std::slice::from_ref(frame))
            .map(ImageReadbackSubmission::from_batch)
    }

    fn submit_direct(&self, frame: &GpuImageFrame) -> Result<ImageReadbackBatchSubmission> {
        let mut plan = ReadbackPlan::new(
            &self.inner.device,
            ImageReadbackLimits {
                // The caller-visible allocation already passed the producer's device and memory
                // admission checks. No readback staging allocation is created on this path.
                max_transient_bytes: u64::MAX,
                max_in_flight_bytes: self.inner.limits.max_in_flight_bytes,
            },
            std::slice::from_ref(frame),
            false,
        )?;
        debug_assert_eq!(plan.entries.len(), 1);
        plan.stats.staging_bytes = 0;
        plan.stats.padding_bytes = 0;
        plan.stats.direct_mapped = true;

        let source_lease = frame.outputs[0].buffer.clone();
        let mapped = source_lease.as_wgpu_buffer().clone();
        let commands = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("jxl-wgpu direct image readback"),
            });
        let completion = Arc::new(BatchMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let lifetime = Arc::new(ImageReadbackLifetime {
            mapped,
            source_leases: vec![source_lease],
            memory_permit: None,
            mapping_submitted: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(&lifetime.mapped, wgpu::MapMode::Read, .., move |result| {
            callback_completion
                .complete(result.map_err(|error| format!("direct image mapping failed: {error}")));
            drop(callback_lifetime);
        });
        let poll_permit = self.inner.poller.try_reserve()?;
        let submission = self.inner.queue.submit([commands.finish()]);
        lifetime.mapping_submitted.store(true, Ordering::Release);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission.clone(), move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!(
                "GPU direct-readback poll registration failed: {error}"
            )));
        }

        Ok(ImageReadbackBatchSubmission {
            device: self.inner.device.clone(),
            submission,
            lifetime: Some(lifetime),
            completion,
            entries: plan.entries,
            frames: plan.frames,
            stats: plan.stats,
        })
    }

    /// Records one explicitly unvalidated frame for readback without waiting for codec validation.
    ///
    /// The pipeline must use the producer's device and queue. The copy is then ordered after the
    /// producer submission. Completing this readback does not validate the codec frame; if later
    /// validation fails, the returned bytes are non-authoritative and must be discarded.
    pub fn submit_unvalidated(
        &self,
        frame: &UnvalidatedGpuImageFrame,
    ) -> Result<UnvalidatedImageReadbackSubmission> {
        let transport_frame = GpuImageFrame {
            token: frame.token,
            outputs: frame
                .outputs
                .iter()
                .map(|output| crate::GpuImageOutput {
                    id: output.id,
                    layout: output.layout.clone(),
                    buffer: output.buffer.clone(),
                })
                .collect(),
            // This value is private transport metadata only. The public unvalidated result does
            // not expose it as an authoritative changed-region statement.
            changed: jxl_gpu_protocol::ChangedRegions::default(),
        };
        self.submit(&transport_frame)
            .map(UnvalidatedImageReadbackSubmission::new)
    }

    /// Records all outputs from multiple GPU frames into one aggregate readback submission.
    ///
    /// Frame order, output order, submission tokens, changed regions, layouts, and logical byte
    /// boundaries are preserved in [`ImageReadbackBatchResult`]. This method only coalesces the
    /// explicit buffer-to-host transport; it does not coalesce the codec submissions that produced
    /// the frames.
    pub fn submit_frames(&self, frames: &[GpuImageFrame]) -> Result<ImageReadbackBatchSubmission> {
        let plan = ReadbackPlan::new(&self.inner.device, self.inner.limits, frames, true)?;
        let memory_permit = self
            .inner
            .memory_budget
            .try_reserve(plan.stats.staging_bytes)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu aggregate image readback"),
            size: plan.stats.staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut commands =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu aggregate image readback"),
                });
        for (output, entry) in frames
            .iter()
            .flat_map(|frame| &frame.outputs)
            .zip(&plan.entries)
        {
            commands.copy_buffer_to_buffer(
                output.buffer.as_wgpu_buffer(),
                0,
                &staging,
                entry.staging_offset,
                entry.copy_size,
            );
        }

        let completion = Arc::new(BatchMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        let lifetime = Arc::new(ImageReadbackLifetime {
            mapped: staging,
            source_leases: frames
                .iter()
                .flat_map(|frame| &frame.outputs)
                .map(|output| output.buffer.clone())
                .collect(),
            memory_permit: Some(memory_permit),
            mapping_submitted: AtomicBool::new(false),
        });
        let callback_lifetime = Arc::clone(&lifetime);
        commands.map_buffer_on_submit(&lifetime.mapped, wgpu::MapMode::Read, .., move |result| {
            callback_completion.complete(
                result.map_err(|error| format!("aggregate image mapping failed: {error}")),
            );
            drop(callback_lifetime);
        });
        let poll_permit = self.inner.poller.try_reserve()?;
        let submission = self.inner.queue.submit([commands.finish()]);
        lifetime.mapping_submitted.store(true, Ordering::Release);
        let poll_completion = Arc::clone(&completion);
        if let Err(error) = poll_permit.register(submission.clone(), move |error| {
            poll_completion.complete(Err(error));
        }) {
            completion.complete(Err(format!(
                "GPU readback poll registration failed: {error}"
            )));
        }

        Ok(ImageReadbackBatchSubmission {
            device: self.inner.device.clone(),
            submission,
            lifetime: Some(lifetime),
            completion,
            entries: plan.entries,
            frames: plan.frames,
            stats: plan.stats,
        })
    }
}

#[derive(Clone, Debug)]
struct ReadbackEntry {
    id: OutputId,
    layout: ImageLayout,
    staging_offset: u64,
    copy_size: u64,
}

#[derive(Clone, Debug)]
struct ReadbackFrame {
    token: jxl_gpu_protocol::SubmissionToken,
    changed: jxl_gpu_protocol::ChangedRegions,
    entries: Range<usize>,
}

struct ReadbackPlan {
    entries: Vec<ReadbackEntry>,
    frames: Vec<ReadbackFrame>,
    stats: ImageReadbackStats,
}

impl ReadbackPlan {
    fn new(
        device: &wgpu::Device,
        limits: ImageReadbackLimits,
        source_frames: &[GpuImageFrame],
        require_copy_source: bool,
    ) -> Result<Self> {
        if source_frames.is_empty() {
            return Err(Error::ImageReadbackNoFrames);
        }

        let mut metadata = Vec::new();
        let mut logical_sizes = Vec::with_capacity(source_frames.len());
        for (frame_index, frame) in source_frames.iter().enumerate() {
            if frame.outputs.is_empty() {
                return Err(Error::ImageReadbackFrameEmpty { frame: frame_index });
            }
            let mut frame_sizes = Vec::with_capacity(frame.outputs.len());
            for (output_index, output) in frame.outputs.iter().enumerate() {
                let layout = ImageLayout::from_planes(
                    output.layout.extent,
                    output.layout.format.clone(),
                    output.layout.planes.clone(),
                )?;
                if require_copy_source
                    && !output.buffer.usage().contains(wgpu::BufferUsages::COPY_SRC)
                {
                    return Err(Error::ImageReadbackSourceUsage {
                        frame: frame_index,
                        output: output_index,
                    });
                }
                let copy_size = aligned_buffer_size(layout.logical_size)?;
                if output.buffer.size() < copy_size {
                    return Err(Error::ImageReadbackSourceSize {
                        frame: frame_index,
                        output: output_index,
                        required: copy_size,
                        actual: output.buffer.size(),
                    });
                }
                frame_sizes.push(layout.logical_size);
                metadata.push((output.id, layout));
            }
            logical_sizes.push(frame_sizes);
        }

        let packing =
            ReadbackPacking::new(&logical_sizes, limits, device.limits().max_buffer_size)?;
        let entries = metadata
            .into_iter()
            .zip(&packing.entries)
            .map(|((id, layout), packed)| ReadbackEntry {
                id,
                layout,
                staging_offset: packed.staging_offset,
                copy_size: packed.copy_size,
            })
            .collect();
        let frames = source_frames
            .iter()
            .zip(packing.frame_entries)
            .map(|(frame, entries)| ReadbackFrame {
                token: frame.token,
                changed: frame.changed.clone(),
                entries,
            })
            .collect();
        Ok(Self {
            entries,
            frames,
            stats: packing.stats,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedReadbackEntry {
    staging_offset: u64,
    copy_size: u64,
}

#[derive(Debug)]
struct ReadbackPacking {
    entries: Vec<PackedReadbackEntry>,
    frame_entries: Vec<Range<usize>>,
    stats: ImageReadbackStats,
}

impl ReadbackPacking {
    /// Pure checked packing logic, deliberately independent of a GPU device for deterministic
    /// limit and offset tests.
    fn new(
        logical_sizes: &[Vec<u64>],
        limits: ImageReadbackLimits,
        device_limit: u64,
    ) -> Result<Self> {
        if logical_sizes.is_empty() {
            return Err(Error::ImageReadbackNoFrames);
        }

        let mut entries = Vec::new();
        let mut frame_entries = Vec::with_capacity(logical_sizes.len());
        let mut logical_bytes = 0_u64;
        let mut staging_bytes = 0_u64;
        for (frame_index, frame_sizes) in logical_sizes.iter().enumerate() {
            if frame_sizes.is_empty() {
                return Err(Error::ImageReadbackFrameEmpty { frame: frame_index });
            }
            let first_entry = entries.len();
            for &logical_size in frame_sizes {
                let copy_size = aligned_buffer_size(logical_size)?;
                entries.push(PackedReadbackEntry {
                    staging_offset: staging_bytes,
                    copy_size,
                });
                staging_bytes = staging_bytes
                    .checked_add(copy_size)
                    .ok_or(Error::BufferSizeOverflow)?;
                logical_bytes = logical_bytes
                    .checked_add(logical_size)
                    .ok_or(Error::BufferSizeOverflow)?;
            }
            frame_entries.push(first_entry..entries.len());
        }
        if staging_bytes > limits.max_transient_bytes {
            return Err(Error::ImageReadbackTransientLimit {
                required: staging_bytes,
                limit: limits.max_transient_bytes,
            });
        }
        if staging_bytes > device_limit {
            return Err(Error::ImageReadbackDeviceLimit {
                required: staging_bytes,
                limit: device_limit,
            });
        }
        Ok(Self {
            entries,
            frame_entries,
            stats: ImageReadbackStats {
                frame_count: logical_sizes.len(),
                output_count: logical_sizes.iter().map(Vec::len).sum(),
                logical_bytes,
                staging_bytes,
                padding_bytes: staging_bytes
                    .checked_sub(logical_bytes)
                    .ok_or(Error::BufferSizeOverflow)?,
                direct_mapped: false,
            },
        })
    }
}

#[derive(Default)]
struct BatchMapCompletion {
    state: Mutex<BatchMapState>,
    condition: Condvar,
}

#[derive(Default)]
struct BatchMapState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

impl BatchMapCompletion {
    fn complete(&self, result: std::result::Result<(), String>) {
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

    fn poll(&self, context: &Context<'_>) -> Option<std::result::Result<(), String>> {
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.waker = Some(context.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> std::result::Result<(), String> {
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

/// In-flight aggregate readback for multiple GPU frames.
///
/// Await it or call [`Self::wait`] exactly once. Dropping the future before completion is safe:
/// the mapping callback retains the staging buffer, source leases, and memory permit until the GPU
/// has finished the copy.
#[must_use = "readback submissions do nothing useful unless awaited or waited"]
pub struct ImageReadbackBatchSubmission {
    device: wgpu::Device,
    submission: wgpu::SubmissionIndex,
    lifetime: Option<Arc<ImageReadbackLifetime>>,
    completion: Arc<BatchMapCompletion>,
    entries: Vec<ReadbackEntry>,
    frames: Vec<ReadbackFrame>,
    stats: ImageReadbackStats,
}

/// In-flight aggregate readback for one GPU frame.
///
/// This is the single-frame convenience returned by [`ImageReadbackPipeline::submit`]. Use
/// [`ImageReadbackPipeline::submit_frames`] to coalesce transport for multiple frames.
#[must_use = "readback submissions do nothing useful unless awaited or waited"]
pub struct ImageReadbackSubmission {
    batch: ImageReadbackBatchSubmission,
}

/// In-flight explicit readback of one unvalidated GPU frame.
///
/// Dropping the future is safe. The callback retains its staging buffer, source buffer leases, and
/// byte-budget permit until GPU completion, just like [`ImageReadbackSubmission`].
#[must_use = "unvalidated readback submissions do nothing useful unless awaited or waited"]
pub struct UnvalidatedImageReadbackSubmission {
    submission: ImageReadbackSubmission,
}

struct ImageReadbackLifetime {
    mapped: wgpu::Buffer,
    source_leases: Vec<GpuBufferLease>,
    memory_permit: Option<MemoryPermit>,
    mapping_submitted: AtomicBool,
}

impl Drop for ImageReadbackLifetime {
    fn drop(&mut self) {
        // This also makes an abandoned direct-map future safe: after the callback releases its
        // final lifetime owner, the caller-visible buffer is returned to the unmapped state.
        if self.mapping_submitted.load(Ordering::Acquire) {
            self.mapped.unmap();
        }
    }
}

impl std::fmt::Debug for ImageReadbackLifetime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageReadbackLifetime")
            .field("mapped_bytes", &self.mapped.size())
            .field("source_count", &self.source_leases.len())
            .field(
                "reserved_bytes",
                &self.memory_permit.as_ref().map_or(0, MemoryPermit::bytes),
            )
            .finish()
    }
}

impl std::fmt::Debug for ImageReadbackBatchSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageReadbackBatchSubmission")
            .field("submission", &self.submission)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ImageReadbackSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageReadbackSubmission")
            .field("submission", &self.batch.submission)
            .field("stats", &self.batch.stats)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for UnvalidatedImageReadbackSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("UnvalidatedImageReadbackSubmission")
            .field("submission", self.submission.submission())
            .field("stats", &self.submission.stats())
            .finish_non_exhaustive()
    }
}

impl ImageReadbackBatchSubmission {
    #[must_use]
    pub const fn stats(&self) -> ImageReadbackStats {
        self.stats
    }

    #[must_use]
    pub const fn submission(&self) -> &wgpu::SubmissionIndex {
        &self.submission
    }

    pub fn wait(self) -> Result<ImageReadbackBatchResult> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut submission = self;
            let mapping = submission.completion.wait();
            submission.finish(mapping)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(Error::Unsupported(
                "blocking generic image readback is unavailable on browser WebGPU; await the submission"
                    .into(),
            ))
        }
    }

    fn finish(
        &mut self,
        mapping: std::result::Result<(), String>,
    ) -> Result<ImageReadbackBatchResult> {
        let lifetime = self
            .lifetime
            .take()
            .ok_or_else(|| Error::Execution("batch image readback was already consumed".into()))?;
        mapping.map_err(Error::Execution)?;
        let mapped = lifetime
            .mapped
            .slice(..)
            .get_mapped_range()
            .map_err(|error| Error::Execution(format!("mapped image range is invalid: {error}")))?;
        let frames = self
            .frames
            .iter()
            .map(|frame| {
                let entries = self.entries.get(frame.entries.clone()).ok_or_else(|| {
                    Error::Execution("batch readback frame range escaped its output plan".into())
                })?;
                let outputs = entries
                    .iter()
                    .map(|entry| {
                        let start = usize::try_from(entry.staging_offset)
                            .map_err(|_| Error::BufferSizeOverflow)?;
                        let logical_size = usize::try_from(entry.layout.logical_size)
                            .map_err(|_| Error::BufferSizeOverflow)?;
                        let end = start
                            .checked_add(logical_size)
                            .ok_or(Error::BufferSizeOverflow)?;
                        let bytes = mapped
                            .get(start..end)
                            .ok_or_else(|| {
                                Error::Execution(
                                    "aggregate mapped output was shorter than its readback plan"
                                        .into(),
                                )
                            })?
                            .to_vec();
                        Ok(CpuImageOutput {
                            id: entry.id,
                            layout: entry.layout.clone(),
                            bytes,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(CpuImageFrame {
                    token: frame.token,
                    outputs,
                    changed: frame.changed.clone(),
                })
            })
            .collect::<Result<Vec<_>>>();
        drop(mapped);
        drop(lifetime);
        Ok(ImageReadbackBatchResult {
            frames: frames?,
            stats: self.stats,
        })
    }
}

impl Future for ImageReadbackBatchSubmission {
    type Output = Result<ImageReadbackBatchResult>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        if let Err(error) = submission.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(error.into()));
        }
        match submission.completion.poll(context) {
            Some(mapping) => Poll::Ready(submission.finish(mapping)),
            None => Poll::Pending,
        }
    }
}

impl ImageReadbackSubmission {
    fn from_batch(batch: ImageReadbackBatchSubmission) -> Self {
        debug_assert_eq!(batch.stats.frame_count, 1);
        Self { batch }
    }

    #[must_use]
    pub const fn stats(&self) -> ImageReadbackStats {
        self.batch.stats()
    }

    #[must_use]
    pub const fn submission(&self) -> &wgpu::SubmissionIndex {
        self.batch.submission()
    }

    pub fn wait(self) -> Result<ImageReadbackResult> {
        single_readback_result(self.batch.wait()?)
    }
}

impl Future for ImageReadbackSubmission {
    type Output = Result<ImageReadbackResult>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        match Pin::new(&mut submission.batch).poll(context) {
            Poll::Ready(result) => Poll::Ready(result.and_then(single_readback_result)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl UnvalidatedImageReadbackSubmission {
    fn new(submission: ImageReadbackSubmission) -> Self {
        Self { submission }
    }

    #[must_use]
    pub const fn stats(&self) -> ImageReadbackStats {
        self.submission.stats()
    }

    #[must_use]
    pub const fn submission(&self) -> &wgpu::SubmissionIndex {
        self.submission.submission()
    }

    pub fn wait(self) -> Result<UnvalidatedImageReadbackResult> {
        Ok(unvalidated_readback_result(self.submission.wait()?))
    }
}

impl Future for UnvalidatedImageReadbackSubmission {
    type Output = Result<UnvalidatedImageReadbackResult>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        match Pin::new(&mut submission.submission).poll(context) {
            Poll::Ready(result) => Poll::Ready(result.map(unvalidated_readback_result)),
            Poll::Pending => Poll::Pending,
        }
    }
}

fn unvalidated_readback_result(result: ImageReadbackResult) -> UnvalidatedImageReadbackResult {
    UnvalidatedImageReadbackResult {
        token: result.frame.token,
        outputs: result.frame.outputs,
        stats: result.stats,
    }
}

fn single_readback_result(mut batch: ImageReadbackBatchResult) -> Result<ImageReadbackResult> {
    if batch.frames.len() != 1 {
        return Err(Error::Execution(format!(
            "single-frame readback completed with {} frames",
            batch.frames.len()
        )));
    }
    Ok(ImageReadbackResult {
        frame: batch.frames.remove(0),
        stats: batch.stats,
    })
}

#[cfg(test)]
mod batch_tests {
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicBool, Ordering};

    use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
    use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, Region, SubmissionToken};
    use wgpu::util::DeviceExt;

    use super::*;
    use crate::{GpuImageOutput, WgpuBackendConfig};

    struct ReentrantWake {
        completion: Arc<BatchMapCompletion>,
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

    fn backend() -> Option<crate::WgpuBackend> {
        match pollster::block_on(crate::WgpuBackend::request_default(WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        })) {
            Ok(backend) => Some(backend),
            Err(Error::NoAdapter) => None,
            Err(error) => panic!("failed to create test adapter: {error}"),
        }
    }

    fn output(
        backend: &crate::WgpuBackend,
        id: u32,
        extent: Extent2d,
        logical_bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> GpuImageOutput {
        let layout = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        )
        .unwrap();
        assert_eq!(layout.logical_size as usize, logical_bytes.len());
        let mut bytes = logical_bytes.to_vec();
        bytes.resize(bytes.len().div_ceil(4) * 4, 0);
        let buffer = Arc::new(backend.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu batch readback test source"),
                contents: &bytes,
                usage,
            },
        ));
        GpuImageOutput {
            id: OutputId(id),
            layout,
            buffer: crate::GpuBufferLease::new(buffer),
        }
    }

    fn frame_with_token(token: u64, outputs: Vec<GpuImageOutput>) -> GpuImageFrame {
        GpuImageFrame {
            token: SubmissionToken(token),
            outputs,
            changed: ChangedRegions::default(),
        }
    }

    fn frame(outputs: Vec<GpuImageOutput>) -> GpuImageFrame {
        frame_with_token(7, outputs)
    }

    #[test]
    fn pure_cross_frame_packing_reports_ranges_offsets_and_padding() {
        let packing = ReadbackPacking::new(
            &[vec![9, 4], vec![1, 6]],
            ImageReadbackLimits {
                max_transient_bytes: 28,
                max_in_flight_bytes: 28,
            },
            28,
        )
        .unwrap();
        assert_eq!(packing.frame_entries, [0..2, 2..4]);
        assert_eq!(
            packing.entries,
            [
                PackedReadbackEntry {
                    staging_offset: 0,
                    copy_size: 12,
                },
                PackedReadbackEntry {
                    staging_offset: 12,
                    copy_size: 4,
                },
                PackedReadbackEntry {
                    staging_offset: 16,
                    copy_size: 4,
                },
                PackedReadbackEntry {
                    staging_offset: 20,
                    copy_size: 8,
                },
            ]
        );
        assert_eq!(
            packing.stats,
            ImageReadbackStats {
                frame_count: 2,
                output_count: 4,
                logical_bytes: 20,
                staging_bytes: 28,
                padding_bytes: 8,
                direct_mapped: false,
            }
        );
    }

    #[test]
    fn single_map_read_output_uses_no_staging_allocation_or_copy_source_usage() {
        let Some(backend) = backend() else {
            return;
        };
        let logical = [1, 2, 3, 4, 5, 6];
        let source = frame(vec![output(
            &backend,
            90,
            Extent2d::new(3, 2),
            &logical,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )]);
        let pipeline = ImageReadbackPipeline::new(&backend);
        let pending = pipeline.submit(&source).unwrap();
        assert_eq!(
            pending.stats(),
            ImageReadbackStats {
                frame_count: 1,
                output_count: 1,
                logical_bytes: 6,
                staging_bytes: 0,
                padding_bytes: 0,
                direct_mapped: true,
            }
        );
        let result = pending.wait().unwrap();
        assert_eq!(result.frame.outputs[0].bytes, logical);
        assert_eq!(pipeline.memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn pure_cross_frame_packing_typed_rejects_shape_limits_and_overflow() {
        let limits = ImageReadbackLimits {
            max_transient_bytes: u64::MAX,
            max_in_flight_bytes: u64::MAX,
        };
        assert!(matches!(
            ReadbackPacking::new(&[], limits, u64::MAX),
            Err(Error::ImageReadbackNoFrames)
        ));
        assert!(matches!(
            ReadbackPacking::new(&[vec![4], vec![]], limits, u64::MAX),
            Err(Error::ImageReadbackFrameEmpty { frame: 1 })
        ));
        assert!(matches!(
            ReadbackPacking::new(
                &[vec![9], vec![4]],
                ImageReadbackLimits {
                    max_transient_bytes: 15,
                    max_in_flight_bytes: 100,
                },
                100,
            ),
            Err(Error::ImageReadbackTransientLimit {
                required: 16,
                limit: 15,
            })
        ));
        assert!(matches!(
            ReadbackPacking::new(&[vec![9], vec![4]], limits, 15),
            Err(Error::ImageReadbackDeviceLimit {
                required: 16,
                limit: 15,
            })
        ));
        assert!(matches!(
            ReadbackPacking::new(&[vec![u64::MAX]], limits, u64::MAX),
            Err(Error::BufferSizeOverflow)
        ));
    }

    #[test]
    fn aggregate_wait_preserves_output_order_and_padding_stats() {
        let Some(backend) = backend() else {
            return;
        };
        let first = (0..9).collect::<Vec<u8>>();
        let second = vec![90, 91, 92, 93];
        let frame = frame(vec![
            output(
                &backend,
                1,
                Extent2d::new(3, 3),
                &first,
                wgpu::BufferUsages::COPY_SRC,
            ),
            output(
                &backend,
                2,
                Extent2d::new(2, 2),
                &second,
                wgpu::BufferUsages::COPY_SRC,
            ),
        ]);
        let submission = ImageReadbackPipeline::new(&backend).submit(&frame).unwrap();
        assert_eq!(
            submission.stats(),
            ImageReadbackStats {
                frame_count: 1,
                output_count: 2,
                logical_bytes: 13,
                staging_bytes: 16,
                padding_bytes: 3,
                direct_mapped: false,
            }
        );
        let result = submission.wait().unwrap();
        assert_eq!(result.frame.token, frame.token);
        assert_eq!(result.frame.outputs[0].bytes, first);
        assert_eq!(result.frame.outputs[1].bytes, second);
        assert_eq!(result.stats.logical_bytes, 13);
    }

    #[test]
    fn multi_frame_future_uses_one_submission_and_preserves_every_boundary() {
        let Some(backend) = backend() else {
            return;
        };
        let first_a = (10..19).collect::<Vec<u8>>();
        let first_b = vec![21, 22, 23, 24];
        let second_a = vec![31];
        let second_b = vec![41, 42, 43, 44, 45, 46];
        let mut frames = vec![
            frame_with_token(
                100,
                vec![
                    output(
                        &backend,
                        10,
                        Extent2d::new(3, 3),
                        &first_a,
                        wgpu::BufferUsages::COPY_SRC,
                    ),
                    output(
                        &backend,
                        11,
                        Extent2d::new(2, 2),
                        &first_b,
                        wgpu::BufferUsages::COPY_SRC,
                    ),
                ],
            ),
            frame_with_token(
                200,
                vec![
                    output(
                        &backend,
                        20,
                        Extent2d::new(1, 1),
                        &second_a,
                        wgpu::BufferUsages::COPY_SRC,
                    ),
                    output(
                        &backend,
                        21,
                        Extent2d::new(3, 2),
                        &second_b,
                        wgpu::BufferUsages::COPY_SRC,
                    ),
                ],
            ),
        ];
        frames[0]
            .changed
            .outputs
            .insert(OutputId(10), vec![Region::new(1, 2, 3, 4)]);
        frames[1]
            .changed
            .outputs
            .insert(OutputId(21), vec![Region::new(-1, 0, 2, 2)]);
        let pipeline = ImageReadbackPipeline::new(&backend);
        let pending = pipeline.submit_frames(&frames).unwrap();
        assert_eq!(
            pending.stats(),
            ImageReadbackStats {
                frame_count: 2,
                output_count: 4,
                logical_bytes: 20,
                staging_bytes: 28,
                padding_bytes: 8,
                direct_mapped: false,
            }
        );
        let result = pollster::block_on(pending).unwrap();
        assert_eq!(result.frames.len(), 2);
        assert_eq!(result.frames[0].token, SubmissionToken(100));
        assert_eq!(result.frames[1].token, SubmissionToken(200));
        assert_eq!(result.frames[0].outputs[0].id, OutputId(10));
        assert_eq!(result.frames[0].outputs[0].bytes, first_a);
        assert_eq!(result.frames[0].outputs[1].bytes, first_b);
        assert_eq!(result.frames[1].outputs[0].id, OutputId(20));
        assert_eq!(result.frames[1].outputs[0].bytes, second_a);
        assert_eq!(result.frames[1].outputs[1].bytes, second_b);
        assert_eq!(
            result.frames[0].changed.outputs[&OutputId(10)],
            [Region::new(1, 2, 3, 4)]
        );
        assert_eq!(
            result.frames[1].changed.outputs[&OutputId(21)],
            [Region::new(-1, 0, 2, 2)]
        );
        assert_eq!(pipeline.memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn completion_wakes_after_releasing_its_mutex() {
        let completion = Arc::new(BatchMapCompletion::default());
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
    fn aggregate_submission_is_a_runtime_neutral_future() {
        let Some(backend) = backend() else {
            return;
        };
        let expected = vec![1, 3, 5, 7, 9, 11];
        let frame = frame(vec![output(
            &backend,
            3,
            Extent2d::new(3, 2),
            &expected,
            wgpu::BufferUsages::COPY_SRC,
        )]);
        let submission = ImageReadbackPipeline::new(&backend).submit(&frame).unwrap();
        let result = pollster::block_on(submission).unwrap();
        assert_eq!(result.frame.outputs[0].bytes, expected);
        assert_eq!(result.stats.staging_bytes, 8);
    }

    #[test]
    fn aggregate_plan_typed_rejects_limits_usage_and_empty_frames() {
        let Some(backend) = backend() else {
            return;
        };
        let valid = frame(vec![output(
            &backend,
            4,
            Extent2d::new(3, 3),
            &[0; 9],
            wgpu::BufferUsages::COPY_SRC,
        )]);
        let limited = ImageReadbackPipeline::from_device_queue(
            backend.device().clone(),
            backend.queue().clone(),
            ImageReadbackLimits {
                max_transient_bytes: 11,
                max_in_flight_bytes: 11,
            },
        )
        .unwrap();
        assert!(matches!(
            limited.submit(&valid),
            Err(Error::ImageReadbackTransientLimit {
                required: 12,
                limit: 11
            })
        ));

        let wrong_usage = frame(vec![output(
            &backend,
            5,
            Extent2d::new(2, 2),
            &[1; 4],
            wgpu::BufferUsages::STORAGE,
        )]);
        assert!(matches!(
            ImageReadbackPipeline::new(&backend).submit(&wrong_usage),
            Err(Error::ImageReadbackSourceUsage {
                frame: 0,
                output: 0,
            })
        ));
        assert!(matches!(
            ImageReadbackPipeline::new(&backend).submit(&frame(Vec::new())),
            Err(Error::ImageReadbackFrameEmpty { frame: 0 })
        ));
        assert!(matches!(
            ImageReadbackPipeline::new(&backend).submit_frames(&[]),
            Err(Error::ImageReadbackNoFrames)
        ));
    }

    #[test]
    fn abandoned_batch_future_releases_budget_after_gpu_completion() {
        let Some(backend) = backend() else {
            return;
        };
        let frame = frame(vec![output(
            &backend,
            30,
            Extent2d::new(3, 3),
            &[9; 9],
            wgpu::BufferUsages::COPY_SRC,
        )]);
        let pipeline = ImageReadbackPipeline::new(&backend);
        let pending = pipeline.submit_frames(&[frame]).unwrap();
        let submission = pending.submission().clone();
        assert_eq!(pipeline.memory_stats().reserved_bytes, 12);
        drop(pending);
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("wait for abandoned readback callback");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while pipeline.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            backend
                .device()
                .poll(wgpu::PollType::Poll)
                .expect("drive abandoned readback callback");
            std::thread::yield_now();
        }
        assert_eq!(pipeline.memory_stats().reserved_bytes, 0);
    }

    #[test]
    fn aggregate_readback_retains_only_explicit_source_buffer_leases() {
        let Some(backend) = backend() else {
            return;
        };
        let extent = Extent2d::new(3, 3);
        let layout = ImageLayout::packed(
            extent,
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        )
        .unwrap();
        let source = Arc::new(backend.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu tracked readback source"),
                contents: &[5; 12],
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        ));
        let source_budget = MemoryBudget::new(NonZeroU64::new(12).unwrap());
        let source_permit = source_budget.try_reserve(12).unwrap();
        let frame = frame(vec![GpuImageOutput {
            id: OutputId(32),
            layout,
            buffer: GpuBufferLease::with_memory_permit(source, source_permit),
        }]);

        let pipeline = ImageReadbackPipeline::new(&backend);
        let pending = pipeline
            .submit_frames(std::slice::from_ref(&frame))
            .unwrap();
        drop(frame);
        assert_eq!(source_budget.snapshot().reserved_bytes, 12);

        pending.wait().unwrap();
        assert_eq!(source_budget.snapshot().reserved_bytes, 0);
    }

    #[test]
    fn concurrent_submissions_use_byte_weighted_backpressure() {
        let Some(backend) = backend() else {
            return;
        };
        let frame = frame(vec![output(
            &backend,
            6,
            Extent2d::new(3, 3),
            &[7; 9],
            wgpu::BufferUsages::COPY_SRC,
        )]);
        let pipeline = ImageReadbackPipeline::from_device_queue(
            backend.device().clone(),
            backend.queue().clone(),
            ImageReadbackLimits {
                max_transient_bytes: 12,
                max_in_flight_bytes: 12,
            },
        )
        .unwrap();

        let first = pipeline.submit(&frame).unwrap();
        assert_eq!(pipeline.memory_stats().reserved_bytes, 12);
        assert!(matches!(
            pipeline.submit(&frame),
            Err(Error::MemoryBackpressure(
                crate::MemoryBudgetError::Exhausted {
                    requested_bytes: 12,
                    reserved_bytes: 12,
                    limit_bytes: 12,
                }
            ))
        ));
        first.wait().unwrap();
        assert_eq!(pipeline.memory_stats().reserved_bytes, 0);
        pipeline.submit(&frame).unwrap().wait().unwrap();
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn saturated_poll_admission_rejects_before_readback_submission() {
        let Some(backend) = backend() else {
            return;
        };
        let permits = (0..crate::SUBMISSION_POLLER_CAPACITY)
            .map(|_| backend.submission_poller().try_reserve().unwrap())
            .collect::<Vec<_>>();
        let frame = frame(vec![output(
            &backend,
            31,
            Extent2d::new(3, 3),
            &[4; 9],
            wgpu::BufferUsages::COPY_SRC,
        )]);
        let pipeline = ImageReadbackPipeline::new(&backend);

        assert!(matches!(
            pipeline.submit(&frame),
            Err(Error::PollAdmission(crate::SubmissionPollerError::Full {
                capacity: crate::SUBMISSION_POLLER_CAPACITY
            }))
        ));
        assert_eq!(pipeline.memory_stats().reserved_bytes, 0);

        drop(permits);
        assert_eq!(backend.submission_poller().in_flight(), 0);
        pipeline.submit(&frame).unwrap().wait().unwrap();
    }
}
