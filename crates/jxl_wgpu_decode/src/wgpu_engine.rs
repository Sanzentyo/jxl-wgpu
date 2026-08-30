use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{Gray8AccelerationIndex, PrefixCodeEntry};
use jxl_gpu_formats::{
    Channel, ChromaOrder, ColorRange, ColorSpecification, ImageLayout, PixelFormat, SampleKind,
    TransferFunction,
};
use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, Region, SubmissionToken};
use jxl_wgpu::{GpuImageFrame, GpuImageOutput, WgpuBackend};
use wgpu::util::DeviceExt;

use crate::profile::validate_gray8_envelope;
use crate::{
    AnimationMetadata, DecodeProfile, Error, FixedModularPredictor, FrameDuration, FrameMetadata,
    GpuCodestream, GpuDecoder, GpuOutputRequest, GpuSubmissionEngine, GpuSubmissionSession,
    PreparedGpuSession, Result, SubmittedGpuFrame, UnsupportedCodestreamFeature,
    UnsupportedProfile,
};

const SHADER: &str = include_str!("lossless_gray8.wgsl");
const LOOKUP_BITS: u8 = 15;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;
const STATUS_OK: u32 = 1;
const STREAM_SENTINEL_BYTES: u64 = 4;
const MAX_SESSION_RESERVATION_BYTES: u64 = 64 * 1024 * 1024;
const DEFAULT_CONCURRENT_MEMORY_BUDGET: u64 = 256 * 1024 * 1024;

/// Conservative GPU allocation accounting for one open stock decode session.
///
/// The reservation multiplies the complete per-frame allocation estimate by the requested
/// bounded in-flight count. It remains held until the session is dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecodeMemoryStats {
    pub per_frame_bytes: u64,
    pub max_in_flight: usize,
    pub reserved_bytes: u64,
}

/// CPU/WGSL ABI for `Params` in `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    token_start: u32,
    token_end: u32,
    width: u32,
    height: u32,
    sample_count: u32,
    output_mode: u32,
    transfer: u32,
    limited_range: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    chroma_width: u32,
    chroma_height: u32,
}

/// Fixed storage-buffer status written by `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DecodeStatus {
    code: u32,
    decoded_samples: u32,
    cursor: u32,
    expected_cursor: u32,
}

const STATUS_BYTES: u64 = std::mem::size_of::<DecodeStatus>() as u64;

const _: () = {
    assert!(std::mem::size_of::<ShaderParams>() == 64);
    assert!(std::mem::align_of::<ShaderParams>() == 4);
    assert!(std::mem::size_of::<DecodeStatus>() == 16);
    assert!(std::mem::align_of::<DecodeStatus>() == 4);
};

struct EngineMemoryBudget {
    limit: u64,
    reserved: Mutex<u64>,
}

impl EngineMemoryBudget {
    fn reserve(self: &Arc<Self>, bytes: u64) -> Result<EngineMemoryReservation> {
        let mut reserved = lock_unpoisoned(&self.reserved);
        let next = reserved
            .checked_add(bytes)
            .ok_or_else(|| Error::backend("concurrent GPU decode memory budget overflow"))?;
        if next > self.limit {
            return Err(Error::backend(format!(
                "concurrent GPU decode sessions would reserve {next} bytes, exceeding the {}-byte engine budget",
                self.limit
            )));
        }
        *reserved = next;
        Ok(EngineMemoryReservation {
            budget: Arc::clone(self),
            bytes,
        })
    }

    fn reserved(&self) -> u64 {
        *lock_unpoisoned(&self.reserved)
    }
}

struct EngineMemoryReservation {
    budget: Arc<EngineMemoryBudget>,
    bytes: u64,
}

impl Drop for EngineMemoryReservation {
    fn drop(&mut self) {
        let mut reserved = lock_unpoisoned(&self.budget.reserved);
        *reserved = reserved.saturating_sub(self.bytes);
    }
}

/// Stock GPU-only decoder for indexed single-group lossless Gray8 codestreams.
///
/// The `jwgp` box contains only SHA-bound bit offsets and canonical prefix tables. The shader
/// reads entropy tokens from the actual `jxlc` bytes, expands the profile's distance-one zero
/// runs, unpacks signed residuals, applies the Gradient predictor, and writes the requested GPU
/// image layout. No CPU pixel or entropy fallback is present.
#[derive(Clone)]
pub struct WgpuSubmissionEngine {
    backend: WgpuBackend,
    pipeline: Arc<wgpu::ComputePipeline>,
    memory: Arc<EngineMemoryBudget>,
}

impl std::fmt::Debug for WgpuSubmissionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuSubmissionEngine")
            .field("backend", &self.backend)
            .field("memory_budget_bytes", &self.memory.limit)
            .field("reserved_session_bytes", &self.memory.reserved())
            .finish_non_exhaustive()
    }
}

impl WgpuSubmissionEngine {
    #[must_use]
    pub fn new(backend: WgpuBackend) -> Self {
        Self::with_memory_budget(
            backend,
            NonZeroU64::new(DEFAULT_CONCURRENT_MEMORY_BUDGET)
                .expect("the default concurrent memory budget is non-zero"),
        )
    }

    /// Constructs an engine with an explicit aggregate reservation bound across cloned engines
    /// and concurrently open decode sessions.
    #[must_use]
    pub fn with_memory_budget(backend: WgpuBackend, memory_budget: NonZeroU64) -> Self {
        let module = backend
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("jxl-wgpu decode lossless gray8"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = Arc::new(backend.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("jxl-wgpu decode lossless gray8 pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("decode"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        Self {
            backend,
            pipeline,
            memory: Arc::new(EngineMemoryBudget {
                limit: memory_budget.get(),
                reserved: Mutex::new(0),
            }),
        }
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        &self.backend
    }

    #[must_use]
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory.limit
    }

    #[must_use]
    pub fn reserved_session_bytes(&self) -> u64 {
        self.memory.reserved()
    }
}

impl GpuSubmissionEngine for WgpuSubmissionEngine {
    type Session = WgpuDecodeSession;

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        let index = codestream.acceleration_index().cloned().ok_or_else(|| {
            UnsupportedProfile::new(
                UnsupportedCodestreamFeature::AccelerationIndex,
                "the stock GPU frontend currently requires a SHA-bound jwgp v1 index; generic JPEG XL parsing is not silently substituted",
            )
        })?;
        if !codestream.is_container() {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::AccelerationIndex,
                "jwgp acceleration metadata is carried by a JPEG XL container",
            )
            .into());
        }
        validate_gray8_envelope(codestream.bytes(), &index)?;
        let extent = Extent2d::new(index.width(), index.height());
        let output = OutputPlan::new(extent, request.format.clone())?;
        let memory_stats = validate_device_limits(
            self.backend.device(),
            codestream.bytes(),
            &index,
            &output,
            request.max_in_flight.get(),
        )?;
        let memory_reservation = self.memory.reserve(memory_stats.reserved_bytes)?;
        let predictor = FixedModularPredictor::new(Gray8AccelerationIndex::PREDICTOR)
            .expect("the shared Gray8 schema uses a valid JPEG XL predictor");
        Ok(PreparedGpuSession::new(
            DecodeProfile::prototype_8bit(predictor),
            AnimationMetadata::still(extent),
            WgpuDecodeSession {
                backend: self.backend.clone(),
                pipeline: Arc::clone(&self.pipeline),
                source: Some(DecodeSource {
                    codestream: codestream.shared_bytes(),
                    index,
                    output,
                }),
                pending: None,
                emitted: false,
                memory_stats,
                _memory_reservation: memory_reservation,
            },
        ))
    }
}

/// One-frame runtime-neutral GPU decode session for the indexed Gray8 profile.
pub struct WgpuDecodeSession {
    backend: WgpuBackend,
    pipeline: Arc<wgpu::ComputePipeline>,
    source: Option<DecodeSource>,
    pending: Option<PendingDecode>,
    emitted: bool,
    memory_stats: WgpuDecodeMemoryStats,
    _memory_reservation: EngineMemoryReservation,
}

impl std::fmt::Debug for WgpuDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuDecodeSession")
            .field("submitted", &self.pending.is_some())
            .field("emitted", &self.emitted)
            .field("memory_stats", &self.memory_stats)
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for WgpuDecodeSession {
    type Frame = GpuImageFrame;

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>> {
        if self.emitted {
            return Ok(None);
        }
        self.ensure_submitted()?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let completion = Arc::clone(&self.pending_ref()?.completion);
            let mapping = completion.wait();
            self.finish(mapping).map(Some)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(Error::backend(
                "blocking GPU decode waits are unavailable on browser WebGPU; poll the frame future",
            ))
        }
    }

    fn poll_next_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>> {
        if self.emitted {
            return Poll::Ready(Ok(None));
        }
        if let Err(error) = self.ensure_submitted() {
            return Poll::Ready(Err(error));
        }
        if let Err(error) = self.backend.device().poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::backend(error)));
        }
        let completion = match self.pending_ref() {
            Ok(pending) => Arc::clone(&pending.completion),
            Err(error) => return Poll::Ready(Err(error)),
        };
        match completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping).map(Some)),
            None => Poll::Pending,
        }
    }
}

impl WgpuDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> WgpuDecodeMemoryStats {
        self.memory_stats
    }

    /// Conservative allocation reservation held by this session, including its requested bound.
    #[must_use]
    pub const fn reserved_gpu_bytes(&self) -> u64 {
        self.memory_stats.reserved_bytes
    }

    fn ensure_submitted(&mut self) -> Result<()> {
        if self.pending.is_some() {
            return Ok(());
        }
        let source = self.source.take().ok_or(Error::EngineContract(
            "Gray8 decode source was consumed without a pending GPU job",
        ))?;
        self.pending = Some(submit_decode(&self.backend, &self.pipeline, source)?);
        Ok(())
    }

    fn pending_ref(&self) -> Result<&PendingDecode> {
        self.pending.as_ref().ok_or(Error::EngineContract(
            "Gray8 GPU completion was queried before submission",
        ))
    }

    fn finish(
        &mut self,
        mapping: std::result::Result<(), String>,
    ) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        mapping.map_err(Error::backend)?;
        let pending = self.pending.take().ok_or(Error::EngineContract(
            "Gray8 GPU completion was consumed more than once",
        ))?;
        let mapped = pending
            .status_staging
            .slice(..)
            .get_mapped_range()
            .map_err(Error::backend)?;
        let status_bytes = mapped
            .get(..STATUS_BYTES as usize)
            .ok_or_else(|| Error::backend("GPU status buffer was truncated"))?;
        let status = bytemuck::try_cast_slice::<u8, DecodeStatus>(status_bytes)
            .map_err(|_| Error::backend("GPU status buffer has an invalid ABI layout"))
            .and_then(|statuses| {
                statuses
                    .first()
                    .copied()
                    .ok_or_else(|| Error::backend("GPU status buffer was truncated"))
            });
        drop(mapped);
        pending.status_staging.unmap();
        let status = status?;
        if status.code != STATUS_OK
            || status.decoded_samples != pending.sample_count
            || status.cursor != status.expected_cursor
        {
            return Err(Error::backend(format!(
                "Gray8 GPU decode rejected entropy stream: status={}, decoded={}/{}, cursor={}/{}",
                status.code,
                status.decoded_samples,
                pending.sample_count,
                status.cursor,
                status.expected_cursor
            )));
        }

        let output_id = OutputId(0);
        let mut regions = BTreeMap::new();
        regions.insert(
            output_id,
            vec![Region::new(
                0,
                0,
                pending.layout.extent.width,
                pending.layout.extent.height,
            )],
        );
        self.emitted = true;
        Ok(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                is_last: true,
                is_keyframe: true,
                name: String::new(),
            },
            GpuImageFrame {
                token: SubmissionToken(1),
                outputs: vec![GpuImageOutput {
                    id: output_id,
                    layout: pending.layout,
                    buffer: pending.output,
                }],
                changed: ChangedRegions { outputs: regions },
            },
        ))
    }
}

struct DecodeSource {
    codestream: Arc<[u8]>,
    index: Gray8AccelerationIndex,
    output: OutputPlan,
}

struct PendingDecode {
    output: Arc<wgpu::Buffer>,
    layout: ImageLayout,
    status_staging: Arc<wgpu::Buffer>,
    completion: Arc<MapCompletion>,
    sample_count: u32,
}

#[derive(Clone, Copy)]
enum OutputMode {
    NonColor = 0,
    Luma = 1,
    Semiplanar = 2,
    Planar = 3,
}

struct OutputPlan {
    layout: ImageLayout,
    mode: OutputMode,
    transfer: u32,
    limited_range: bool,
}

impl OutputPlan {
    fn new(extent: Extent2d, format: PixelFormat) -> Result<Self> {
        let non_color = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
        if format == non_color {
            return Ok(Self {
                layout: ImageLayout::packed(extent, format)?,
                mode: OutputMode::NonColor,
                transfer: 0,
                limited_range: false,
            });
        }

        let mode = if format == PixelFormat::luma(8, format.color_spec) {
            OutputMode::Luma
        } else if [ChromaOrder::CbCr, ChromaOrder::CrCb]
            .into_iter()
            .filter_map(|order| {
                PixelFormat::yuv_semiplanar(
                    format.chroma_subsampling,
                    8,
                    8,
                    order,
                    format.color_spec,
                )
                .ok()
            })
            .any(|candidate| candidate == format)
        {
            OutputMode::Semiplanar
        } else if PixelFormat::yuv_planar(format.chroma_subsampling, 8, 8, format.color_spec)
            .is_ok_and(|candidate| candidate == format)
        {
            OutputMode::Planar
        } else {
            return Err(Error::UnsupportedOutputFormat(format!("{format:?}")));
        };
        let (transfer, limited_range) = color_conversion(&format)?;
        Ok(Self {
            layout: ImageLayout::packed(extent, format)?,
            mode,
            transfer,
            limited_range,
        })
    }
}

fn color_conversion(format: &PixelFormat) -> Result<(u32, bool)> {
    let ColorSpecification::Defined(spec) = format.color_spec else {
        return Err(Error::UnsupportedOutputFormat(
            "YCbCr output requires an explicit color specification".into(),
        ));
    };
    let transfer = match spec.transfer {
        TransferFunction::Srgb | TransferFunction::Sycc => 0,
        TransferFunction::Bt709 | TransferFunction::Bt2020 => 1,
        TransferFunction::Linear => 2,
        transfer => {
            return Err(Error::UnsupportedOutputFormat(format!(
                "the Gray8 GPU kernel does not implement {transfer:?} output transfer"
            )));
        }
    };
    Ok((transfer, spec.range == ColorRange::Limited))
}

fn validate_device_limits(
    device: &wgpu::Device,
    codestream: &[u8],
    index: &Gray8AccelerationIndex,
    output: &OutputPlan,
    max_in_flight: usize,
) -> Result<WgpuDecodeMemoryStats> {
    let token_end = index
        .token_bit_offset()
        .checked_add(index.token_bit_len())
        .ok_or_else(|| Error::backend("indexed token range overflow"))?;
    if token_end > u64::from(u32::MAX) {
        return Err(Error::backend(
            "indexed token offsets exceed the portable WGSL u32 address space",
        ));
    }
    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let codestream_bytes = stream_allocation_size(codestream.len())?;
    let lookup_bytes = u64::try_from(LOOKUP_SIZE * 4)
        .map_err(|_| Error::backend("prefix lookup size overflow"))?;
    let sample_bytes = u64::from(index.sample_count())
        .checked_mul(4)
        .ok_or_else(|| Error::backend("reconstruction buffer size overflow"))?;
    let output_bytes = align4(output.layout.logical_size)?;
    for (name, required) in [
        ("codestream", codestream_bytes),
        ("prefix lookup", lookup_bytes),
        ("reconstructed samples", sample_bytes),
        ("requested output", output_bytes),
    ] {
        if required > storage_limit || required > device.limits().max_buffer_size {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device limit"
            )));
        }
    }
    let per_frame = [
        codestream_bytes,
        lookup_bytes,
        sample_bytes,
        output_bytes,
        STATUS_BYTES,
        STATUS_BYTES,
        std::mem::size_of::<ShaderParams>() as u64,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| Error::backend("Gray8 GPU memory budget overflow"))?;
    let reserved = per_frame
        .checked_mul(
            u64::try_from(max_in_flight)
                .map_err(|_| Error::backend("max-in-flight count overflow"))?,
        )
        .ok_or_else(|| Error::backend("bounded in-flight GPU memory budget overflow"))?;
    if reserved > MAX_SESSION_RESERVATION_BYTES {
        return Err(Error::backend(format!(
            "bounded Gray8 session reserves {reserved} bytes ({per_frame} per frame), exceeding the {}-byte session limit",
            MAX_SESSION_RESERVATION_BYTES
        )));
    }
    Ok(WgpuDecodeMemoryStats {
        per_frame_bytes: per_frame,
        max_in_flight,
        reserved_bytes: reserved,
    })
}

fn submit_decode(
    backend: &WgpuBackend,
    pipeline: &wgpu::ComputePipeline,
    source: DecodeSource,
) -> Result<PendingDecode> {
    let device = backend.device();
    let mut codestream_bytes = source.codestream.to_vec();
    codestream_bytes.resize(
        usize::try_from(stream_allocation_size(codestream_bytes.len())?)
            .map_err(|_| Error::backend("codestream allocation size overflow"))?,
        0,
    );
    let stream = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu decode codestream"),
        contents: &codestream_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let lookup = build_prefix_lookup(&source.index)?;
    let lookup = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu decode prefix lookup"),
        contents: bytemuck::cast_slice(&lookup),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let sample_bytes = u64::from(source.index.sample_count())
        .checked_mul(4)
        .ok_or_else(|| Error::backend("reconstruction buffer size overflow"))?;
    let reconstructed = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decoded Gray8 samples"),
        size: sample_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_size = align4(source.output.layout.logical_size)?;
    let output = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decoded image output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decode status"),
        size: STATUS_BYTES,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let status_staging = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decode status readback"),
        size: STATUS_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    }));

    let params = build_params(&source.index, &source.output)?;
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu decode Gray8 parameters"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu decode Gray8 bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lookup.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: reconstructed.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: status.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jxl-wgpu decode Gray8 submission"),
    });
    commands.clear_buffer(&reconstructed, 0, None);
    commands.clear_buffer(&output, 0, None);
    commands.clear_buffer(&status, 0, None);
    {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu entropy and Gradient reconstruction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    commands.copy_buffer_to_buffer(&status, 0, &status_staging, 0, STATUS_BYTES);

    let completion = Arc::new(MapCompletion::default());
    let callback_completion = Arc::clone(&completion);
    commands.map_buffer_on_submit(&status_staging, wgpu::MapMode::Read, .., move |result| {
        callback_completion
            .complete(result.map_err(|error| format!("GPU status mapping failed: {error}")));
    });
    let submission = backend.queue().submit([commands.finish()]);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let poll_device = device.clone();
        let poll_completion = Arc::clone(&completion);
        std::thread::spawn(move || {
            if let Err(error) = poll_device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            }) {
                poll_completion.complete(Err(format!("GPU submission failed: {error}")));
            }
        });
    }
    #[cfg(target_arch = "wasm32")]
    let _ = submission;

    Ok(PendingDecode {
        output,
        layout: source.output.layout,
        status_staging,
        completion,
        sample_count: source.index.sample_count(),
    })
}

fn build_prefix_lookup(index: &Gray8AccelerationIndex) -> Result<Vec<u32>> {
    let mut lookup = vec![0u32; LOOKUP_SIZE];
    for (symbol, entry) in index
        .raw_prefix()
        .iter()
        .copied()
        .enumerate()
        .map(|(symbol, entry)| (symbol as u32, entry))
        .chain(
            index
                .lz77_prefix()
                .iter()
                .copied()
                .enumerate()
                .map(|(symbol, entry)| (224 + symbol as u32, entry)),
        )
    {
        insert_prefix(&mut lookup, symbol, entry)?;
    }
    Ok(lookup)
}

fn insert_prefix(lookup: &mut [u32], symbol: u32, entry: PrefixCodeEntry) -> Result<()> {
    if entry.bit_len == 0 {
        return Ok(());
    }
    if entry.bit_len > LOOKUP_BITS {
        return Err(Error::backend(format!(
            "validated jwgp prefix length {} exceeds the {LOOKUP_BITS}-bit GPU lookup",
            entry.bit_len
        )));
    }
    let suffix_bits = LOOKUP_BITS - entry.bit_len;
    for suffix in 0..1usize << suffix_bits {
        let index = usize::from(entry.bits) | (suffix << entry.bit_len);
        if lookup[index] != 0 {
            return Err(Error::backend(
                "validated jwgp prefix entries collide in the GPU lookup table",
            ));
        }
        lookup[index] = (symbol << 8) | u32::from(entry.bit_len);
    }
    Ok(())
}

fn build_params(index: &Gray8AccelerationIndex, output: &OutputPlan) -> Result<ShaderParams> {
    let plane0 = output
        .layout
        .plane(0)
        .ok_or_else(|| Error::backend("requested output has no first plane"))?;
    let plane1 = output.layout.plane(1);
    let plane2 = output.layout.plane(2);
    let token_end = index
        .token_bit_offset()
        .checked_add(index.token_bit_len())
        .ok_or_else(|| Error::backend("indexed token range overflow"))?;
    let to_u32 = |value: u64, name: &'static str| {
        u32::try_from(value).map_err(|_| Error::backend(format!("{name} exceeds WGSL u32")))
    };
    Ok(ShaderParams {
        token_start: to_u32(index.token_bit_offset(), "token start")?,
        token_end: to_u32(token_end, "token end")?,
        width: index.width(),
        height: index.height(),
        sample_count: index.sample_count(),
        output_mode: output.mode as u32,
        transfer: output.transfer,
        limited_range: u32::from(output.limited_range),
        plane0_offset: to_u32(plane0.offset, "plane 0 offset")?,
        plane0_stride: to_u32(plane0.row_stride, "plane 0 row stride")?,
        plane1_offset: to_u32(plane1.map_or(0, |plane| plane.offset), "plane 1 offset")?,
        plane1_stride: to_u32(
            plane1.map_or(0, |plane| plane.row_stride),
            "plane 1 row stride",
        )?,
        plane2_offset: to_u32(plane2.map_or(0, |plane| plane.offset), "plane 2 offset")?,
        plane2_stride: to_u32(
            plane2.map_or(0, |plane| plane.row_stride),
            "plane 2 row stride",
        )?,
        chroma_width: plane1.map_or(0, |plane| plane.sample_extent.width),
        chroma_height: plane1.map_or(0, |plane| plane.sample_extent.height),
    })
}

fn stream_allocation_size(byte_len: usize) -> Result<u64> {
    let byte_len = u64::try_from(byte_len)
        .map_err(|_| Error::backend("codestream allocation size overflow"))?;
    align4(byte_len)?
        .checked_add(STREAM_SENTINEL_BYTES)
        .ok_or_else(|| Error::backend("codestream sentinel allocation overflow"))
}

fn align4(value: u64) -> Result<u64> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| Error::backend("GPU buffer size overflow"))
}

#[derive(Default)]
struct MapCompletion {
    state: Mutex<MapState>,
    condition: Condvar,
}

#[derive(Default)]
struct MapState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

impl MapCompletion {
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

impl GpuDecoder<WgpuSubmissionEngine> {
    /// Constructs the GPU-required facade around an application's existing wgpu backend.
    #[must_use]
    pub fn wgpu(backend: WgpuBackend) -> Self {
        Self::new(WgpuSubmissionEngine::new(backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_negotiation_rejects_unimplemented_packed_rgb() {
        let format = PixelFormat::rgb8(
            jxl_gpu_formats::RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Undefined,
        );
        assert!(matches!(
            OutputPlan::new(Extent2d::new(2, 2), format),
            Err(Error::UnsupportedOutputFormat(_))
        ));
    }

    #[test]
    fn lookup_table_uses_lsb_first_prefixes() {
        let mut lookup = vec![0u32; LOOKUP_SIZE];
        insert_prefix(
            &mut lookup,
            7,
            PrefixCodeEntry {
                bit_len: 3,
                bits: 0b101,
            },
        )
        .unwrap();
        assert_eq!(lookup[0b101] >> 8, 7);
        assert_eq!(lookup[0b1_0101] >> 8, 7);
        assert_eq!(lookup[0b101] & 0xff, 3);
        assert!(matches!(
            insert_prefix(
                &mut lookup,
                8,
                PrefixCodeEntry {
                    bit_len: LOOKUP_BITS + 1,
                    bits: 0,
                },
            ),
            Err(Error::Backend(_))
        ));
    }

    #[test]
    fn shader_abi_and_stream_sentinel_are_explicit() {
        assert_eq!(std::mem::size_of::<ShaderParams>(), 64);
        assert_eq!(std::mem::align_of::<ShaderParams>(), 4);
        let params = ShaderParams {
            token_start: 1,
            token_end: 2,
            width: 3,
            height: 4,
            sample_count: 5,
            output_mode: 6,
            transfer: 7,
            limited_range: 8,
            plane0_offset: 9,
            plane0_stride: 10,
            plane1_offset: 11,
            plane1_stride: 12,
            plane2_offset: 13,
            plane2_stride: 14,
            chroma_width: 15,
            chroma_height: 16,
        };
        assert_eq!(
            bytemuck::cast::<ShaderParams, [u32; 16]>(params),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert!(SHADER.contains(
            "struct Params {\n    token_start: u32,\n    token_end: u32,\n    width: u32,\n    height: u32,"
        ));

        assert_eq!(std::mem::size_of::<DecodeStatus>(), 16);
        assert_eq!(std::mem::align_of::<DecodeStatus>(), 4);
        let status = DecodeStatus {
            code: 1,
            decoded_samples: 2,
            cursor: 3,
            expected_cursor: 4,
        };
        assert_eq!(
            bytemuck::cast::<DecodeStatus, [u32; 4]>(status),
            [1, 2, 3, 4]
        );
        assert!(SHADER.contains("status[0] = STATUS_OK;"));
        assert!(SHADER.contains("status[1] = decoded;"));
        assert!(SHADER.contains("status[2] = bit_cursor;"));
        assert!(SHADER.contains("status[3] = params.token_end;"));
        assert_eq!(stream_allocation_size(4).unwrap(), 8);
        assert_eq!(stream_allocation_size(5).unwrap(), 12);
    }

    #[test]
    fn aggregate_memory_reservations_are_bounded_and_released() {
        let budget = Arc::new(EngineMemoryBudget {
            limit: 10,
            reserved: Mutex::new(0),
        });
        let first = budget.reserve(6).unwrap();
        assert_eq!(budget.reserved(), 6);
        assert!(matches!(budget.reserve(5), Err(Error::Backend(_))));
        assert_eq!(budget.reserved(), 6);
        drop(first);
        assert_eq!(budget.reserved(), 0);
    }
}
