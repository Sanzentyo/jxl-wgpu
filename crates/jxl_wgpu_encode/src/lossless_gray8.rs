use std::future::Future;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, BitWriter, ContainerBox, Gray8AccelerationIndex,
    write_container_with_boxes,
};
use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use wgpu::util::DeviceExt;

use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{
    AnimationHeader, BitFragment, Determinism, EncodeError, EncodeProfile, EncoderCapabilities,
    FrameEncodeRequest, FrameGroupLayout, FrameIndex, FrameOptions, FramePacketSet,
    FrameSubmission, GpuAccelerationArtifact, GpuEncodeBackend, GpuEncodeJob, GpuEncoder,
    GpuFrameArtifacts, GpuFrameSource, GroupPacket, GroupPacketKind, KernelStage,
    ProfileCapability, ProgressivePlan, UnsupportedFeature, WgpuContext, assemble_frame,
};

const MAX_DIMENSION: u32 = 256;
const SHADER: &str = include_str!("lossless_gray8.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8Params {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
}

/// Fixed storage-buffer header written by `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8ArtifactHeader {
    event_count: u32,
    raw_counts: [u32; RAW_SYMBOLS],
    lz77_counts: [u32; LZ77_SYMBOLS],
}

/// Fixed storage-buffer event written after [`Gray8ArtifactHeader`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Gray8Event {
    kind: u32,
    token: u32,
    extra_bit_count: u32,
    extra_bits: u32,
}

const OUTPUT_HEADER_WORDS: usize = std::mem::size_of::<Gray8ArtifactHeader>() / 4;
const EVENT_WORDS: usize = std::mem::size_of::<Gray8Event>() / 4;

const _: () = {
    assert!(std::mem::size_of::<Gray8Params>() == 16);
    assert!(std::mem::align_of::<Gray8Params>() == 4);
    assert!(std::mem::size_of::<Gray8ArtifactHeader>() == 53 * 4);
    assert!(std::mem::align_of::<Gray8ArtifactHeader>() == 4);
    assert!(std::mem::size_of::<Gray8Event>() == 16);
    assert!(std::mem::align_of::<Gray8Event>() == 4);
};

/// Checked memory accounting for one concrete Gray8 submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8MemoryPlan {
    pub source_binding_bytes: u64,
    pub uniform_bytes: u64,
    pub artifact_storage_bytes: u64,
    pub readback_bytes: u64,
    pub owned_bytes_per_job: u64,
    pub addressed_bytes_per_job: u64,
}

/// Total memory exposure for a caller-selected maximum number of in-flight jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8InFlightMemory {
    pub max_in_flight_jobs: u32,
    pub total_owned_bytes: u64,
    pub total_addressed_bytes: u64,
}

/// Device limits that bound concrete Gray8 source and artifact bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessGray8MemoryLimits {
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub min_storage_buffer_offset_alignment: u64,
}

impl LosslessGray8MemoryPlan {
    pub fn for_in_flight(
        self,
        max_in_flight_jobs: u32,
    ) -> Result<LosslessGray8InFlightMemory, EncodeError> {
        if max_in_flight_jobs == 0 {
            return Err(EncodeError::InvalidConfiguration(
                "max in-flight job count must be non-zero",
            ));
        }
        let jobs = u64::from(max_in_flight_jobs);
        let total_owned_bytes =
            self.owned_bytes_per_job
                .checked_mul(jobs)
                .ok_or(EncodeError::InvalidConfiguration(
                    "in-flight encoder memory size overflow",
                ))?;
        let total_addressed_bytes = self.addressed_bytes_per_job.checked_mul(jobs).ok_or(
            EncodeError::InvalidConfiguration("in-flight encoder memory size overflow"),
        )?;
        Ok(LosslessGray8InFlightMemory {
            max_in_flight_jobs,
            total_owned_bytes,
            total_addressed_bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Gray8DispatchPlan {
    width: u32,
    height: u32,
    row_stride: u32,
    shader_byte_offset: u32,
    source_binding_offset: u64,
    source_binding_size: NonZeroU64,
    output_size: u64,
    max_events: usize,
    memory: LosslessGray8MemoryPlan,
}

/// The first executable encoder profile: one 8-bit grayscale Modular group.
///
/// It never reads source pixels on the CPU. The source buffer must contain one
/// `PixelFormat::non_color(Unsigned, 8, &[X])` plane. The GPU emits predictor
/// residual tokens and histograms; the host only serializes those artifacts.
pub struct LosslessGray8Backend {
    pipeline: Arc<wgpu::ComputePipeline>,
    capabilities: EncoderCapabilities,
    max_storage_binding_size: u64,
    max_buffer_size: u64,
    storage_offset_alignment: u64,
}

impl LosslessGray8Backend {
    #[must_use]
    pub fn new(context: &WgpuContext) -> Self {
        let module = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu lossless gray8 token kernel"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(context.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu lossless gray8 token pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("encode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        let limits = context.device().limits();
        Self {
            pipeline,
            capabilities: EncoderCapabilities {
                profiles: vec![ProfileCapability::ModularLossless {
                    min_bits_per_sample: 8,
                    max_bits_per_sample: 8,
                }],
                max_progressive_passes: 1,
                animation: false,
                determinism: Determinism::CrossDevice,
                implemented_stages: vec![
                    KernelStage::ModularPrediction,
                    KernelStage::ModularResidualTokenization,
                    KernelStage::HistogramReduction,
                ],
            },
            max_storage_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            storage_offset_alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        }
    }

    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessGray8MemoryPlan, EncodeError> {
        Ok(self.dispatch_plan(source)?.memory)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessGray8MemoryLimits {
        LosslessGray8MemoryLimits {
            max_storage_buffer_binding_size: self.max_storage_binding_size,
            max_buffer_size: self.max_buffer_size,
            min_storage_buffer_offset_alignment: self.storage_offset_alignment,
        }
    }

    fn dispatch_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<Gray8DispatchPlan, EncodeError> {
        let extent = source.layout.extent;
        if extent.width < 2
            || extent.height < 2
            || extent.width > MAX_DIMENSION
            || extent.height > MAX_DIMENSION
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        if source.layout.format != PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X])
            || source.layout.planes.len() != 1
            || !source.buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            || !source.buffer.size().is_multiple_of(4)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        let plane = source
            .layout
            .plane(0)
            .ok_or(EncodeError::InvalidSource("missing grayscale plane"))?;
        let row_stride = u32::try_from(plane.row_stride)
            .map_err(|_| EncodeError::InvalidSource("row stride exceeds the prototype limit"))?;
        if plane.row_stride < u64::from(extent.width) {
            return Err(EncodeError::InvalidSource(
                "row stride is smaller than the grayscale row width",
            ));
        }
        let preceding_rows = plane
            .row_stride
            .checked_mul(u64::from(extent.height - 1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let sample_end = plane
            .offset
            .checked_add(preceding_rows)
            .and_then(|value| value.checked_add(u64::from(extent.width)))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let _full_stride_end = plane
            .row_stride
            .checked_mul(u64::from(extent.height))
            .and_then(|value| plane.offset.checked_add(value))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic overflow",
            ))?;
        let binding_end = align_up(sample_end, 4)
            .ok_or(EncodeError::InvalidSource("source binding size overflow"))?;
        if binding_end > source.buffer.size() {
            return Err(EncodeError::InvalidSource(
                "source binding does not contain the final addressable sample word",
            ));
        }
        // A storage array of u32 also needs a word-aligned base even on a
        // hypothetical device reporting a smaller dynamic-offset alignment.
        let alignment = self.storage_offset_alignment.max(4);
        let source_binding_offset = plane.offset - plane.offset % alignment;
        if !source_binding_offset.is_multiple_of(alignment) {
            return Err(EncodeError::InvalidSource(
                "source storage binding offset is not device-aligned",
            ));
        }
        let source_binding_bytes = binding_end
            .checked_sub(source_binding_offset)
            .ok_or(EncodeError::InvalidSource("source binding range underflow"))?;
        if !source_binding_bytes.is_multiple_of(4) {
            return Err(EncodeError::InvalidSource(
                "source storage binding size is not word-aligned",
            ));
        }
        if source_binding_bytes > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: source_binding_bytes,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        let source_binding_size = NonZeroU64::new(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("source binding is empty"))?;
        let shader_byte_offset = u32::try_from(plane.offset - source_binding_offset)
            .map_err(|_| EncodeError::InvalidSource("relative plane offset exceeds u32"))?;
        let shader_last_byte = sample_end
            .checked_sub(source_binding_offset)
            .and_then(|value| value.checked_sub(1))
            .ok_or(EncodeError::InvalidSource(
                "source address arithmetic underflow",
            ))?;
        u32::try_from(shader_last_byte).map_err(|_| {
            EncodeError::InvalidSource("source address exceeds the WGSL u32 address space")
        })?;

        let pixel_count = usize::try_from(u64::from(extent.width) * u64::from(extent.height))
            .map_err(|_| EncodeError::InvalidSource("image dimensions overflow"))?;
        let max_events = event_capacity(pixel_count)?;
        let output_words = OUTPUT_HEADER_WORDS
            .checked_add(
                max_events
                    .checked_mul(EVENT_WORDS)
                    .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?,
            )
            .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
        let output_size = u64::try_from(output_words)
            .ok()
            .and_then(|words| words.checked_mul(4))
            .ok_or(EncodeError::InvalidSource("event buffer size overflow"))?;
        if !output_size.is_multiple_of(4) {
            return Err(EncodeError::InvalidSource(
                "artifact buffer is not copy/map aligned",
            ));
        }
        if output_size > self.max_storage_binding_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_storage_buffer_binding_size",
                required: output_size,
                available: self.max_storage_binding_size,
            }
            .into());
        }
        if output_size > self.max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: output_size,
                available: self.max_buffer_size,
            }
            .into());
        }
        let uniform_bytes = u64::try_from(std::mem::size_of::<Gray8Params>())
            .map_err(|_| EncodeError::InvalidSource("uniform size overflow"))?;
        let owned_bytes_per_job = output_size
            .checked_mul(2)
            .and_then(|value| value.checked_add(uniform_bytes))
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let addressed_bytes_per_job = owned_bytes_per_job
            .checked_add(source_binding_bytes)
            .ok_or(EncodeError::InvalidSource("per-job memory size overflow"))?;
        let memory = LosslessGray8MemoryPlan {
            source_binding_bytes,
            uniform_bytes,
            artifact_storage_bytes: output_size,
            readback_bytes: output_size,
            owned_bytes_per_job,
            addressed_bytes_per_job,
        };
        Ok(Gray8DispatchPlan {
            width: extent.width,
            height: extent.height,
            row_stride,
            shader_byte_offset,
            source_binding_offset,
            source_binding_size,
            output_size,
            max_events,
            memory,
        })
    }
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    let adjustment = alignment.checked_sub(1)?;
    value
        .checked_add(adjustment)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

fn event_capacity(pixel_count: usize) -> Result<usize, EncodeError> {
    pixel_count
        .checked_add(pixel_count.div_ceil(8))
        .and_then(|value| value.checked_add(1))
        .ok_or(EncodeError::InvalidSource("event buffer size overflow"))
}

impl GpuEncodeBackend for LosslessGray8Backend {
    type Job = LosslessGray8Job;

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
        if request.animation != AnimationHeader::Still
            || request.frame_index != FrameIndex::new(0)
            || !request.is_last
        {
            return Err(UnsupportedFeature::Animation.into());
        }
        if request.options != FrameOptions::default() {
            return Err(EncodeError::InvalidConfiguration(
                "the gray8 prototype only supports default still-frame options",
            ));
        }
        let GpuFrameSource::Buffer(source) = source else {
            return Err(UnsupportedFeature::InputFormat.into());
        };
        let plan = self.dispatch_plan(&source)?;

        let params = Gray8Params {
            width: plan.width,
            height: plan.height,
            row_stride: plan.row_stride,
            byte_offset: plan.shader_byte_offset,
        };
        let params = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu lossless gray8 parameters"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let output = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu lossless gray8 GPU artifacts"),
            size: plan.output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu lossless gray8 artifact readback"),
            size: plan.output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        let bind_group = context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu lossless gray8 bindings"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &source.buffer,
                            offset: plan.source_binding_offset,
                            size: Some(plan.source_binding_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let mut commands =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("jxl-wgpu lossless gray8 encode"),
                });
        commands.clear_buffer(&output, 0, None);
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu lossless gray8 tokenization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        commands.copy_buffer_to_buffer(&output, 0, &staging, 0, plan.output_size);

        let completion = Arc::new(MapCompletion::default());
        let callback_completion = Arc::clone(&completion);
        commands.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            callback_completion
                .complete(result.map_err(|error| format!("GPU artifact mapping failed: {error}")));
        });
        let submission_index = context.queue().submit([commands.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let poll_context = context.clone();
            let poll_completion = Arc::clone(&completion);
            std::thread::spawn(move || {
                if let Err(error) = poll_context.device().poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index),
                    timeout: None,
                }) {
                    poll_completion.complete(Err(format!("GPU submission failed: {error}")));
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        let _ = submission_index;

        Ok(LosslessGray8Job {
            staging: Some(staging),
            completion,
            output_size: plan.output_size,
            max_events: plan.max_events,
            width: plan.width,
            height: plan.height,
            frame_index: request.frame_index,
            is_last: request.is_last,
        })
    }
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
        let mut state = self.state.lock().expect("map completion mutex poisoned");
        if state.result.is_none() {
            state.result = Some(result);
            if let Some(waker) = state.waker.take() {
                waker.wake();
            }
            self.condition.notify_all();
        }
    }

    fn poll(&self, cx: &Context<'_>) -> Option<Result<(), String>> {
        let mut state = self.state.lock().expect("map completion mutex poisoned");
        if state.result.is_none() {
            state.waker = Some(cx.waker().clone());
        }
        state.result.take()
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) -> Result<(), String> {
        let mut state = self.state.lock().expect("map completion mutex poisoned");
        while state.result.is_none() {
            state = self
                .condition
                .wait(state)
                .expect("map completion mutex poisoned while waiting");
        }
        state
            .result
            .take()
            .expect("map completion was checked as present")
    }
}

/// Runtime-neutral completion for the concrete GPU lossless profile.
pub struct LosslessGray8Job {
    staging: Option<Arc<wgpu::Buffer>>,
    completion: Arc<MapCompletion>,
    output_size: u64,
    max_events: usize,
    width: u32,
    height: u32,
    frame_index: FrameIndex,
    is_last: bool,
}

impl LosslessGray8Job {
    fn finish(&mut self, mapping: Result<(), String>) -> Result<GpuFrameArtifacts, EncodeError> {
        mapping.map_err(EncodeError::Backend)?;
        let staging = self
            .staging
            .take()
            .ok_or_else(|| EncodeError::Backend("GPU job was already consumed".into()))?;
        let mapped = staging.slice(..).get_mapped_range().map_err(|error| {
            EncodeError::Backend(format!("invalid mapped artifact range: {error}"))
        })?;
        let expected = usize::try_from(self.output_size)
            .map_err(|_| EncodeError::Backend("mapped artifact size overflow".into()))?;
        let bytes = mapped
            .get(..expected)
            .ok_or_else(|| EncodeError::Backend("mapped artifact buffer was truncated".into()))?;
        let result = build_packets(self.width, self.height, self.max_events, bytes);
        drop(mapped);
        staging.unmap();
        let (packets, acceleration) = result?;
        Ok(GpuFrameArtifacts {
            frame_index: self.frame_index,
            is_last: self.is_last,
            packets,
            acceleration: Some(acceleration),
        })
    }
}

impl GpuEncodeJob for LosslessGray8Job {
    fn poll_complete(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        match self.completion.poll(cx) {
            Some(result) => Poll::Ready(self.finish(result)),
            None => Poll::Pending,
        }
    }

    fn wait(self) -> Result<GpuFrameArtifacts, EncodeError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut job = self;
            let result = job.completion.wait();
            job.finish(result)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(EncodeError::Backend(
                "blocking GPU waits are unavailable on browser WebGPU; await the submission".into(),
            ))
        }
    }
}

/// Convenience API that produces a complete raw codestream or deterministic
/// `jxlc` container from a GPU-resident grayscale buffer.
pub struct LosslessGray8Encoder {
    encoder: GpuEncoder<LosslessGray8Backend>,
}

impl LosslessGray8Encoder {
    #[must_use]
    pub fn new(context: WgpuContext) -> Self {
        let backend = LosslessGray8Backend::new(&context);
        Self {
            encoder: GpuEncoder::new(context, backend),
        }
    }

    #[must_use]
    pub fn capabilities(&self) -> &EncoderCapabilities {
        self.encoder.capabilities()
    }

    /// Computes all source, artifact, and readback bytes before submission.
    pub fn memory_plan(
        &self,
        source: &crate::BufferImageSource,
    ) -> Result<LosslessGray8MemoryPlan, EncodeError> {
        self.encoder.backend().memory_plan(source)
    }

    #[must_use]
    pub fn memory_limits(&self) -> LosslessGray8MemoryLimits {
        self.encoder.backend().memory_limits()
    }

    pub fn submit(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        self.submit_inner(source, false)
    }

    pub fn submit_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        self.submit_inner(source, true)
    }

    pub fn encode(&self, source: crate::BufferImageSource) -> Result<Vec<u8>, EncodeError> {
        self.submit(source)?.wait()
    }

    pub fn encode_container(
        &self,
        source: crate::BufferImageSource,
    ) -> Result<Vec<u8>, EncodeError> {
        self.submit_container(source)?.wait()
    }

    fn submit_inner(
        &self,
        source: crate::BufferImageSource,
        container: bool,
    ) -> Result<LosslessGray8Submission, EncodeError> {
        // Preserve typed address/device-limit failures before the generic
        // backend admission predicate maps unsupported inputs to InputFormat.
        self.memory_plan(&source)?;
        let width = source.layout.extent.width;
        let height = source.layout.extent.height;
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless { bits_per_sample: 8 },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::CrossDevice,
            animation: AnimationHeader::Still,
            options: FrameOptions::default(),
        };
        let frame = self
            .encoder
            .submit_frame(GpuFrameSource::Buffer(source), request)?;
        Ok(LosslessGray8Submission {
            frame: Some(frame),
            codestream_header: image_header(width, height)?,
            container,
        })
    }
}

/// A `Future` with an executor-independent blocking counterpart.
pub struct LosslessGray8Submission {
    frame: Option<FrameSubmission<LosslessGray8Job>>,
    codestream_header: BitFragment,
    container: bool,
}

impl LosslessGray8Submission {
    pub fn wait(mut self) -> Result<Vec<u8>, EncodeError> {
        let frame = self
            .frame
            .take()
            .expect("a lossless submission can only complete once")
            .wait()?;
        self.assemble(frame)
    }

    fn assemble(&self, frame: GpuFrameArtifacts) -> Result<Vec<u8>, EncodeError> {
        let group_size = frame
            .packets
            .packets()
            .first()
            .ok_or_else(|| EncodeError::Backend("gray8 frame has no group packet".into()))?
            .payload
            .len();
        let acceleration = frame
            .acceleration
            .ok_or_else(|| EncodeError::Backend("gray8 acceleration artifact is missing".into()))?;
        let encoded_frame = assemble_frame(frame.packets)?;
        let bytes_before_group = encoded_frame
            .bytes()
            .len()
            .checked_sub(group_size)
            .ok_or_else(|| EncodeError::Backend("gray8 group size exceeds frame size".into()))?;
        let group_start = self
            .codestream_header
            .bytes()
            .len()
            .checked_add(bytes_before_group)
            .ok_or_else(|| EncodeError::Backend("gray8 codestream size overflow".into()))?;
        let mut codestream = self.codestream_header.bytes().to_vec();
        codestream.extend_from_slice(encoded_frame.bytes());
        if !self.container {
            return Ok(codestream);
        }

        let GpuAccelerationArtifact::Gray8Prefix {
            width,
            height,
            token_bit_offset_in_group,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        } = acceleration;
        let group_start_bits = u64::try_from(group_start)
            .ok()
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let token_bit_offset = group_start_bits
            .checked_add(token_bit_offset_in_group)
            .ok_or_else(|| EncodeError::Backend("gray8 token offset overflow".into()))?;
        let index = Gray8AccelerationIndex::new(
            &codestream,
            width,
            height,
            token_bit_offset,
            token_bit_len,
            raw_prefix,
            lz77_prefix,
        )?;
        let payload = index.serialize();
        Ok(write_container_with_boxes(
            &codestream,
            &[ContainerBox {
                box_type: ACCELERATION_INDEX_BOX_TYPE,
                payload: &payload,
            }],
        )?)
    }
}

impl Future for LosslessGray8Submission {
    type Output = Result<Vec<u8>, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let frame = submission
            .frame
            .as_mut()
            .expect("a lossless submission must not be polled after completion");
        match Pin::new(frame).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.frame.take();
                Poll::Ready(result.and_then(|frame| submission.assemble(frame)))
            }
        }
    }
}

fn build_packets(
    width: u32,
    height: u32,
    max_events: usize,
    bytes: &[u8],
) -> Result<(FramePacketSet, GpuAccelerationArtifact), EncodeError> {
    let header_bytes = bytes
        .get(..std::mem::size_of::<Gray8ArtifactHeader>())
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let header = bytemuck::try_cast_slice::<u8, Gray8ArtifactHeader>(header_bytes)
        .map_err(|_| EncodeError::Backend("GPU artifact header has an invalid ABI layout".into()))?
        .first()
        .copied()
        .ok_or_else(|| EncodeError::Backend("GPU artifact header is truncated".into()))?;
    let event_count = usize::try_from(header.event_count)
        .map_err(|_| EncodeError::Backend("GPU event count overflow".into()))?;
    if event_count > max_events {
        return Err(EncodeError::Backend(
            "GPU emitted more token events than the output allocation".into(),
        ));
    }
    let event_bytes = event_count
        .checked_mul(std::mem::size_of::<Gray8Event>())
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let required_bytes = std::mem::size_of::<Gray8ArtifactHeader>()
        .checked_add(event_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event count overflow".into()))?;
    let events = bytes
        .get(std::mem::size_of::<Gray8ArtifactHeader>()..required_bytes)
        .ok_or_else(|| EncodeError::Backend("GPU event stream is truncated".into()))?;
    let events = bytemuck::try_cast_slice::<u8, Gray8Event>(events)
        .map_err(|_| EncodeError::Backend("GPU event stream has an invalid ABI layout".into()))?;

    let primary = PrefixCode::from_gpu_counts(&header.raw_counts, &header.lz77_counts);
    let unused = PrefixCode::fixed_unused_channel();
    let codes = [primary.clone(), unused.clone(), unused.clone(), unused];
    let mut group = BitWriter::new();
    write_dc_global(&mut group, &codes)?;
    let token_bit_offset_in_group = u64::try_from(group.bit_len())
        .map_err(|_| EncodeError::Backend("gray8 token offset overflow".into()))?;
    for event in events {
        match event.kind {
            0 => codes[0].write_raw(
                &mut group,
                event.token,
                event.extra_bit_count,
                event.extra_bits,
            )?,
            1 => codes[0].write_run(
                &mut group,
                event.token,
                event.extra_bit_count,
                event.extra_bits,
            )?,
            _ => {
                return Err(EncodeError::Backend(
                    "GPU emitted an unknown token kind".into(),
                ));
            }
        }
    }
    let token_bit_end = u64::try_from(group.bit_len())
        .map_err(|_| EncodeError::Backend("gray8 token length overflow".into()))?;
    let token_bit_len = token_bit_end
        .checked_sub(token_bit_offset_in_group)
        .ok_or_else(|| EncodeError::Backend("gray8 token length underflow".into()))?;
    group.align_to_byte()?;

    let packets = FramePacketSet::new(
        frame_header()?,
        FrameGroupLayout::new(1, 1, 1)?,
        [GroupPacket::new(
            GroupPacketKind::Single,
            group.into_bytes(),
        )],
    )?;
    let acceleration = GpuAccelerationArtifact::Gray8Prefix {
        width,
        height,
        token_bit_offset_in_group,
        token_bit_len,
        raw_prefix: codes[0].raw_entries(),
        lz77_prefix: codes[0].lz77_entries(),
    };
    Ok((packets, acceleration))
}

fn write_dc_global(output: &mut BitWriter, codes: &[PrefixCode; 4]) -> Result<(), EncodeError> {
    // Handcrafted Modular metadata adapted from zune-jpegxl 0.5.2. See
    // `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt` at repository root.
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    output.write_bits(0b100011, 6)?;
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0b1010, 4)?;
    output.write_bits(4, 4)?;
    output.write_bits(0, 3)?;
    output.write_bits(0, 3)?;
    output.write_bits(1, 1)?;
    output.write_bits(3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        output.write_bits(context, 3)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    output.write_bits(1, 5)?;
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        // libjxl's U32 selector stores the low eight bits of 256 here.
        output.write_bits(0, 8)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    for code in codes {
        code.write_tree(output)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    Ok(())
}

fn image_header(width: u32, height: u32) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?;
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(1, 2)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0b10, 2)?;
    output.write_bits(11, 4)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes()))
}

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(2..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "gray8 dimensions must be in 2..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

fn frame_header() -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(0, 2)?;
    let bit_len = output.bit_len();
    BitFragment::new(output.into_bytes(), bit_len).map_err(Into::into)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use jxl::api::{
        JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
        JxlPixelFormat, ProcessingResult, states,
    };
    use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
    use jxl_gpu_protocol::Extent2d;
    use std::process::Command;

    #[test]
    fn gray8_params_abi_matches_wgsl_uniform() {
        assert_eq!(std::mem::size_of::<Gray8Params>(), 16);
        assert_eq!(std::mem::align_of::<Gray8Params>(), 4);
        let params = Gray8Params {
            width: 1,
            height: 2,
            row_stride: 3,
            byte_offset: 4,
        };
        assert_eq!(
            bytemuck::cast::<Gray8Params, [u32; 4]>(params),
            [1, 2, 3, 4]
        );
        assert!(SHADER.contains(
            "struct Params {\n    width: u32,\n    height: u32,\n    row_stride: u32,\n    byte_offset: u32,\n}"
        ));
    }

    #[test]
    fn gray8_artifact_abi_matches_wgsl_word_schema() {
        assert_eq!(std::mem::size_of::<Gray8ArtifactHeader>(), 53 * 4);
        assert_eq!(std::mem::align_of::<Gray8ArtifactHeader>(), 4);
        assert_eq!(std::mem::size_of::<Gray8Event>(), 4 * 4);
        assert_eq!(std::mem::align_of::<Gray8Event>(), 4);

        let header = Gray8ArtifactHeader {
            event_count: 7,
            raw_counts: std::array::from_fn(|index| 100 + index as u32),
            lz77_counts: std::array::from_fn(|index| 200 + index as u32),
        };
        let words = bytemuck::cast::<Gray8ArtifactHeader, [u32; 53]>(header);
        assert_eq!(words[0], 7);
        assert_eq!(words[1..20], header.raw_counts);
        assert_eq!(words[20..53], header.lz77_counts);

        let event = Gray8Event {
            kind: 1,
            token: 2,
            extra_bit_count: 3,
            extra_bits: 4,
        };
        assert_eq!(bytemuck::cast::<Gray8Event, [u32; 4]>(event), [1, 2, 3, 4]);
        assert!(SHADER.contains("Word 0 is the event count, words 1..20 are raw-token counts"));
        assert!(SHADER.contains("// (kind, token, extra-bit count, extra bits)."));
        assert!(SHADER.contains("const OUTPUT_HEADER_WORDS: u32 = 53u;"));
        assert!(SHADER.contains("const EVENT_WORDS: u32 = 4u;"));
    }

    /// Mirrors only the event-admission control flow in `encode` WGSL. A
    /// `true` sample is a zero packed residual; the actual token value is
    /// irrelevant to the number of four-word event records.
    fn simulated_shader_event_count(
        width: usize,
        height: usize,
        is_zero: impl Fn(usize) -> bool,
    ) -> usize {
        let mut run = 0usize;
        let mut events = 0usize;
        for y in 0..height {
            for chunk_x in (0..width).step_by(8) {
                let count = 8.min(width - chunk_x);
                let mut prefix = 0usize;
                while prefix < count && is_zero(y * width + chunk_x + prefix) {
                    prefix += 1;
                }
                if prefix == count && (run > 0 || prefix > 7) {
                    run += prefix;
                } else if prefix + run > 7 {
                    events += usize::from(run + prefix > 0);
                    events += count - prefix;
                    run = 0;
                } else {
                    events += count;
                }
            }
        }
        events + usize::from(run > 0)
    }

    #[test]
    fn event_allocation_bounds_every_shader_write() {
        // Exhaust every zero/non-zero residual stream up to 16 samples and
        // vary row boundaries because a run is intentionally frame-global.
        for width in 1usize..=16 {
            for height in 1usize..=(16 / width) {
                let pixels = width * height;
                let capacity = event_capacity(pixels).expect("small capacity is representable");
                for mask in 0u32..(1u32 << pixels) {
                    let events = simulated_shader_event_count(width, height, |index| {
                        mask & (1u32 << index) != 0
                    });
                    assert!(events <= capacity, "{width}x{height}, mask={mask:#x}");
                }
            }
        }

        let pixels = usize::try_from(u64::from(MAX_DIMENSION) * u64::from(MAX_DIMENSION))
            .expect("maximum prototype dimensions fit usize");
        let capacity = event_capacity(pixels).expect("maximum event capacity fits usize");
        for events in [
            simulated_shader_event_count(pixels, 1, |_| false),
            simulated_shader_event_count(pixels, 1, |_| true),
            simulated_shader_event_count(pixels, 1, |index| index % 2 == 0),
            simulated_shader_event_count(pixels, 1, |index| index % 17 < 8),
        ] {
            assert!(events <= capacity);
        }

        let words = OUTPUT_HEADER_WORDS + capacity * EVENT_WORDS;
        let last_event_word = OUTPUT_HEADER_WORDS + (capacity - 1) * EVENT_WORDS + 3;
        assert!(last_event_word < words);
        assert_eq!(words * 4 % wgpu::COPY_BUFFER_ALIGNMENT as usize, 0);
    }

    fn test_context() -> Option<WgpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("jxl-wgpu lossless encoder test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some(WgpuContext::new(Arc::new(device), Arc::new(queue)))
    }

    fn decode_gray8(encoded: &[u8]) -> Result<((usize, usize), Vec<u8>), String> {
        let mut input = encoded;
        let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before image info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let size = decoder.basic_info().size;
        decoder.set_pixel_format(JxlPixelFormat {
            color_type: JxlColorType::Grayscale,
            color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
            extra_channel_format: Vec::new(),
        });
        let mut frame = loop {
            match decoder
                .process(&mut input, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("decoder needed more input before frame info".into());
                    }
                    decoder = fallback;
                }
            }
        };
        let mut pixels = vec![0u8; size.0 * size.1];
        {
            let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0)];
            loop {
                match frame
                    .process(&mut input, &mut buffers, None)
                    .map_err(|error| error.to_string())?
                {
                    ProcessingResult::Complete { .. } => break,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            return Err("decoder needed more input while rendering".into());
                        }
                        frame = fallback;
                    }
                }
            }
        }
        Ok((size, pixels))
    }

    fn decode_with_djxl_if_available(encoded: &[u8]) -> Option<Result<Vec<u8>, String>> {
        if Command::new("djxl").arg("-V").output().is_err() {
            return None;
        }
        let directory = std::env::temp_dir().join(format!("jxl-wgpu-gray8-{}", std::process::id()));
        if let Err(error) = std::fs::create_dir(&directory) {
            return Some(Err(format!(
                "could not create djxl test directory: {error}"
            )));
        }
        let input = directory.join("gpu.jxl");
        let output = directory.join("gpu.pgm");
        let result = (|| {
            std::fs::write(&input, encoded)
                .map_err(|error| format!("could not write djxl input: {error}"))?;
            let command = Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .arg("--quiet")
                .output()
                .map_err(|error| format!("could not execute djxl: {error}"))?;
            if !command.status.success() {
                return Err(format!(
                    "djxl rejected GPU codestream: {}",
                    String::from_utf8_lossy(&command.stderr)
                ));
            }
            let pgm = std::fs::read(&output)
                .map_err(|error| format!("could not read djxl PGM: {error}"))?;
            parse_pgm(&pgm)
        })();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_dir(directory);
        Some(result)
    }

    fn parse_pgm(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let mut cursor = 0usize;
        let mut token = || -> Result<&[u8], String> {
            loop {
                while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
                    cursor += 1;
                }
                if bytes.get(cursor) == Some(&b'#') {
                    while bytes.get(cursor).is_some_and(|byte| *byte != b'\n') {
                        cursor += 1;
                    }
                    continue;
                }
                break;
            }
            let start = cursor;
            while bytes
                .get(cursor)
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                cursor += 1;
            }
            bytes
                .get(start..cursor)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "truncated PGM header".into())
        };
        if token()? != b"P5" {
            return Err("djxl did not emit a binary grayscale PGM".into());
        }
        let width = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        let height = std::str::from_utf8(token()?)
            .map_err(|error| error.to_string())?
            .parse::<usize>()
            .map_err(|error| error.to_string())?;
        if token()? != b"255" {
            return Err("djxl PGM did not contain 8-bit samples".into());
        }
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let pixels = bytes
            .get(cursor..)
            .ok_or_else(|| "truncated PGM pixels".to_string())?;
        if pixels.len() != width * height {
            return Err(format!(
                "djxl PGM has {} samples, expected {}",
                pixels.len(),
                width * height
            ));
        }
        Ok(pixels.to_vec())
    }

    #[test]
    fn gpu_tokens_form_a_reference_decodable_lossless_codestream() {
        let Some(context) = test_context() else {
            eprintln!("skipping GPU lossless encode test: no wgpu adapter");
            return;
        };
        let width = 17u32;
        let height = 13u32;
        let row_stride = 20u64;
        let binding_alignment = u64::from(
            context
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        )
        .max(4);
        let offset = binding_alignment + 4;
        let allocation_size = align_up(offset + row_stride * u64::from(height), 4)
            .expect("test allocation size is representable");
        let mut allocation = vec![0u8; allocation_size as usize];
        let mut expected = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                let value = if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                };
                allocation[(offset + u64::from(y) * row_stride + u64::from(x)) as usize] = value;
                expected.push(value);
            }
        }
        let buffer = Arc::new(context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu gray8 test source"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        let extent = Extent2d::new(width, height);
        let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
        let layout = ImageLayout::from_planes(
            extent,
            format,
            vec![PitchLinearPlaneLayout {
                plane_index: 0,
                offset,
                row_stride,
                sample_extent: extent,
                row_bytes: u64::from(width),
            }],
        )
        .expect("test image layout is valid");
        let source = crate::BufferImageSource::new(buffer, layout).expect("test source is valid");
        let encoder = LosslessGray8Encoder::new(context);
        let memory = encoder
            .memory_plan(&source)
            .expect("test source has a checked memory plan");
        let pixel_count = usize::try_from(width * height).expect("test dimensions fit usize");
        let expected_output_words = OUTPUT_HEADER_WORDS
            + event_capacity(pixel_count).expect("test event capacity") * EVENT_WORDS;
        let expected_output_bytes =
            u64::try_from(expected_output_words * 4).expect("test artifact size fits u64");
        assert_eq!(memory.uniform_bytes, 16);
        assert_eq!(memory.artifact_storage_bytes, expected_output_bytes);
        assert_eq!(memory.readback_bytes, expected_output_bytes);
        assert_eq!(memory.owned_bytes_per_job, 16 + expected_output_bytes * 2);
        assert_eq!(
            memory.addressed_bytes_per_job,
            memory.owned_bytes_per_job + memory.source_binding_bytes
        );
        let in_flight = memory
            .for_in_flight(4)
            .expect("four-job memory total is representable");
        assert_eq!(in_flight.max_in_flight_jobs, 4);
        assert_eq!(in_flight.total_owned_bytes, memory.owned_bytes_per_job * 4);
        assert_eq!(
            in_flight.total_addressed_bytes,
            memory.addressed_bytes_per_job * 4
        );
        let limits = encoder.memory_limits();
        assert_eq!(
            limits.min_storage_buffer_offset_alignment.max(4),
            binding_alignment
        );
        let encoded = encoder
            .encode(source.clone())
            .expect("GPU lossless encode succeeds");
        let (size, decoded) = decode_gray8(&encoded).expect("jxl reference decoder accepts output");
        assert_eq!(size, (width as usize, height as usize));
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&encoded) {
            assert_eq!(decoded.expect("djxl accepts GPU codestream"), expected);
        }
        let submission = encoder
            .submit(source.clone())
            .expect("runtime-neutral Future submission succeeds");
        let async_encoded =
            pollster::block_on(submission).expect("runtime-neutral Future encode succeeds");
        assert_eq!(async_encoded, encoded);

        let container = encoder
            .encode_container(source.clone())
            .expect("GPU lossless container encode succeeds");
        let parsed =
            jxl_gpu_bitstream::parse(&container, jxl_gpu_bitstream::ParseLimits::default())
                .expect("container is structurally valid");
        assert_eq!(parsed.codestream(), encoded);
        let boxes = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .collect::<Vec<_>>();
        assert_eq!(boxes.len(), 1);
        let index = Gray8AccelerationIndex::parse_bound(boxes[0].payload, parsed.codestream())
            .expect("jwgp index is bound to the exact codestream");
        assert_eq!(index.width(), width);
        assert_eq!(index.height(), height);
        assert_eq!(index.sample_count(), width * height);
        let (_, decoded) =
            decode_gray8(&container).expect("jxl reference decoder ignores the private box");
        assert_eq!(decoded, expected);
        if let Some(decoded) = decode_with_djxl_if_available(&container) {
            assert_eq!(
                decoded.expect("djxl ignores jwgp and decodes jxlc"),
                expected
            );
        }
        let second = encoder
            .encode_container(source)
            .expect("second deterministic container encode succeeds");
        assert_eq!(container, second);
        assert_eq!(
            container,
            include_bytes!("../../../fixtures/gpu_gray8_lossless.jxl")
        );
        if let Some(path) = std::env::var_os("JXL_WGPU_WRITE_FIXTURE") {
            std::fs::write(path, &container).expect("requested fixture path is writable");
        }
    }
}
