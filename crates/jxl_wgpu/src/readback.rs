// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::future::Future;
use std::pin::Pin;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

#[cfg(not(target_arch = "wasm32"))]
use jxl_gpu_protocol::PlaneData;
use jxl_gpu_protocol::{Extent2d, OutputDesc, OutputId, RenderedOutput, SampleType};

use crate::upload::aligned_buffer_size;
use crate::{CpuImageFrame, CpuImageOutput, Error, GpuImageFrame, ImageLayout, Result};

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
    pub max_transient_bytes: u64,
}

impl Default for ImageReadbackLimits {
    fn default() -> Self {
        Self {
            max_transient_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Byte accounting known before a generic-image readback is submitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageReadbackStats {
    pub output_count: usize,
    /// Sum of the addressable bytes declared by all image layouts.
    pub logical_bytes: u64,
    /// Size of the single aggregate staging buffer, including four-byte copy padding.
    pub staging_bytes: u64,
}

/// Completed explicit CPU readback of one decoder-owned GPU image frame.
#[derive(Clone, Debug)]
pub struct ImageReadbackResult {
    pub frame: CpuImageFrame,
    pub stats: ImageReadbackStats,
}

struct ImageReadbackPipelineInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: ImageReadbackLimits,
}

/// Reusable batch readback state for generic pitch-linear decoder output.
///
/// Every output in a [`GpuImageFrame`] is copied into one aggregate `MAP_READ` buffer. Source and
/// destination copy ranges are padded independently to four bytes, while returned byte vectors
/// contain only each layout's `logical_size` bytes. This is an explicit host transfer of GPU
/// decode results; it is not a CPU codec fallback.
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
        Self::from_device_queue(
            backend.device().clone(),
            backend.queue().clone(),
            ImageReadbackLimits {
                max_transient_bytes: backend.config.memory.max_transient_bytes,
            },
        )
    }

    /// Creates an application-owned readback pipeline for outputs from the same device and queue.
    #[must_use]
    pub fn from_device_queue(
        device: wgpu::Device,
        queue: wgpu::Queue,
        limits: ImageReadbackLimits,
    ) -> Self {
        Self {
            inner: Arc::new(ImageReadbackPipelineInner {
                device,
                queue,
                limits,
            }),
        }
    }

    #[must_use]
    pub fn limits(&self) -> ImageReadbackLimits {
        self.inner.limits
    }

    /// Records one aggregate copy and returns immediately without a host wait.
    pub fn submit(&self, frame: &GpuImageFrame) -> Result<ImageReadbackSubmission> {
        let plan = ReadbackPlan::new(&self.inner.device, self.inner.limits, frame)?;
        let staging = Arc::new(self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu aggregate image readback"),
            size: plan.stats.staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        let mut commands =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu aggregate image readback"),
                });
        for (output, entry) in frame.outputs.iter().zip(&plan.entries) {
            commands.copy_buffer_to_buffer(
                &output.buffer,
                0,
                &staging,
                entry.staging_offset,
                entry.copy_size,
            );
        }

        let completion = Arc::new(BatchMapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        commands.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            callback_completion.complete(
                result.map_err(|error| format!("aggregate image mapping failed: {error}")),
            );
        });
        let submission = self.inner.queue.submit([commands.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let poll_device = self.inner.device.clone();
            let poll_submission = submission.clone();
            let poll_completion = Arc::clone(&completion);
            std::thread::spawn(move || {
                if let Err(error) = poll_device.poll(wgpu::PollType::Wait {
                    submission_index: Some(poll_submission),
                    timeout: None,
                }) {
                    poll_completion.complete(Err(format!("GPU readback poll failed: {error}")));
                }
            });
        }

        Ok(ImageReadbackSubmission {
            device: self.inner.device.clone(),
            submission,
            staging: Some(staging),
            completion,
            entries: plan.entries,
            stats: plan.stats,
            token: frame.token,
            changed: frame.changed.clone(),
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

struct ReadbackPlan {
    entries: Vec<ReadbackEntry>,
    stats: ImageReadbackStats,
}

impl ReadbackPlan {
    fn new(
        device: &wgpu::Device,
        limits: ImageReadbackLimits,
        frame: &GpuImageFrame,
    ) -> Result<Self> {
        if frame.outputs.is_empty() {
            return Err(Error::ImageReadbackEmpty);
        }
        let mut logical_bytes = 0_u64;
        let mut staging_bytes = 0_u64;
        let mut entries = Vec::with_capacity(frame.outputs.len());
        for (output_index, output) in frame.outputs.iter().enumerate() {
            let layout = ImageLayout::from_planes(
                output.layout.extent,
                output.layout.format.clone(),
                output.layout.planes.clone(),
            )?;
            if !output.buffer.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                return Err(Error::ImageReadbackSourceUsage {
                    output: output_index,
                });
            }
            let copy_size = aligned_buffer_size(layout.logical_size)?;
            if output.buffer.size() < copy_size {
                return Err(Error::ImageReadbackSourceSize {
                    output: output_index,
                    required: copy_size,
                    actual: output.buffer.size(),
                });
            }
            let staging_offset = staging_bytes;
            staging_bytes = staging_bytes
                .checked_add(copy_size)
                .ok_or(Error::BufferSizeOverflow)?;
            logical_bytes = logical_bytes
                .checked_add(layout.logical_size)
                .ok_or(Error::BufferSizeOverflow)?;
            entries.push(ReadbackEntry {
                id: output.id,
                layout,
                staging_offset,
                copy_size,
            });
        }
        if staging_bytes > limits.max_transient_bytes {
            return Err(Error::ImageReadbackTransientLimit {
                required: staging_bytes,
                limit: limits.max_transient_bytes,
            });
        }
        let device_limit = device.limits().max_buffer_size;
        if staging_bytes > device_limit {
            return Err(Error::ImageReadbackDeviceLimit {
                required: staging_bytes,
                limit: device_limit,
            });
        }
        Ok(Self {
            entries,
            stats: ImageReadbackStats {
                output_count: frame.outputs.len(),
                logical_bytes,
                staging_bytes,
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
        let mut state = lock_unpoisoned(&self.state);
        if state.result.is_none() {
            state.result = Some(result);
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
            self.condition.notify_all();
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

/// In-flight aggregate readback. Await it or call [`Self::wait`] exactly once.
#[must_use = "readback submissions do nothing useful unless awaited or waited"]
pub struct ImageReadbackSubmission {
    device: wgpu::Device,
    submission: wgpu::SubmissionIndex,
    staging: Option<Arc<wgpu::Buffer>>,
    completion: Arc<BatchMapCompletion>,
    entries: Vec<ReadbackEntry>,
    stats: ImageReadbackStats,
    token: jxl_gpu_protocol::SubmissionToken,
    changed: jxl_gpu_protocol::ChangedRegions,
}

impl std::fmt::Debug for ImageReadbackSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ImageReadbackSubmission")
            .field("submission", &self.submission)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl ImageReadbackSubmission {
    #[must_use]
    pub const fn stats(&self) -> ImageReadbackStats {
        self.stats
    }

    #[must_use]
    pub const fn submission(&self) -> &wgpu::SubmissionIndex {
        &self.submission
    }

    pub fn wait(self) -> Result<ImageReadbackResult> {
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

    fn finish(&mut self, mapping: std::result::Result<(), String>) -> Result<ImageReadbackResult> {
        mapping.map_err(Error::Execution)?;
        let staging = self
            .staging
            .take()
            .ok_or_else(|| Error::Execution("image readback was already consumed".into()))?;
        let mapped = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|error| Error::Execution(format!("mapped image range is invalid: {error}")))?;
        let outputs = self
            .entries
            .iter()
            .map(|entry| {
                let start =
                    usize::try_from(entry.staging_offset).map_err(|_| Error::BufferSizeOverflow)?;
                let logical_size = usize::try_from(entry.layout.logical_size)
                    .map_err(|_| Error::BufferSizeOverflow)?;
                let end = start
                    .checked_add(logical_size)
                    .ok_or(Error::BufferSizeOverflow)?;
                let bytes = mapped
                    .get(start..end)
                    .ok_or_else(|| {
                        Error::Execution(
                            "aggregate mapped output was shorter than its readback plan".into(),
                        )
                    })?
                    .to_vec();
                Ok(CpuImageOutput {
                    id: entry.id,
                    layout: entry.layout.clone(),
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>();
        drop(mapped);
        staging.unmap();
        let outputs = outputs?;
        Ok(ImageReadbackResult {
            frame: CpuImageFrame {
                token: self.token,
                outputs,
                changed: self.changed.clone(),
            },
            stats: self.stats,
        })
    }
}

impl Future for ImageReadbackSubmission {
    type Output = Result<ImageReadbackResult>;

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

#[cfg(test)]
mod batch_tests {
    use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
    use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, SubmissionToken};
    use wgpu::util::DeviceExt;

    use super::*;
    use crate::{GpuImageOutput, WgpuBackendConfig};

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
            buffer,
        }
    }

    fn frame(outputs: Vec<GpuImageOutput>) -> GpuImageFrame {
        GpuImageFrame {
            token: SubmissionToken(7),
            outputs,
            changed: ChangedRegions::default(),
        }
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
                output_count: 2,
                logical_bytes: 13,
                staging_bytes: 16,
            }
        );
        let result = submission.wait().unwrap();
        assert_eq!(result.frame.token, frame.token);
        assert_eq!(result.frame.outputs[0].bytes, first);
        assert_eq!(result.frame.outputs[1].bytes, second);
        assert_eq!(result.stats.logical_bytes, 13);
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
            },
        );
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
            Err(Error::ImageReadbackSourceUsage { output: 0 })
        ));
        assert!(matches!(
            ImageReadbackPipeline::new(&backend).submit(&frame(Vec::new())),
            Err(Error::ImageReadbackEmpty)
        ));
    }
}
