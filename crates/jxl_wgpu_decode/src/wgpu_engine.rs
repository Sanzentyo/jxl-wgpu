use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{InventoryLimits, ParseLimits, PrefixCodeEntry};
use jxl_gpu_formats::{
    ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification, ImageLayout, Packed422Order,
    PixelFormat, PixelFormatClass, RgbChannelOrder, RgbStorage, SampleKind, TransferFunction,
    classify_pixel_format,
};
use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, Region, SubmissionToken};
use jxl_wgpu::{
    GpuBufferLease, GpuImageFrame, GpuImageOutput, MemoryBudget, MemoryBudgetSnapshot,
    MemoryPermit, SubmissionPollPermit, UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput,
    WgpuBackend,
};

use crate::buffer_pool::{
    DecodeBufferLease, DecodeBufferPool, WgpuDecodeBufferPoolLimits, WgpuDecodeBufferPoolStats,
};
use crate::model::native_modular_format;
use crate::profile::{ModularGroup, StandardModularProfile, parse_standard_modular_profile};
use crate::{
    AnimationMetadata, DecodeProfile, Error, F64OutputPolicy, FixedModularPredictor, FrameDuration,
    FrameMetadata, GpuCodestream, GpuDecoder, GpuOutputMapping, GpuOutputRequest, GpuPendingFrame,
    GpuSubmissionEngine, GpuSubmissionSession, NumericSampleMapping, PreparedGpuSession, Result,
    SubmittedGpuFrame,
};

const SHADER_TEMPLATE: &str = include_str!("lossless_gray8.wgsl");
const F64_OUTPUT_MARKER: &str = "/*__JXL_F64_OUTPUT__*/";
const F64_BINDING_MARKER: &str = "/*__JXL_F64_BINDING__*/";
const F64_EXACT_F32_WIDENING: &str = r#"
                if params.numeric_mapping != 1u {
                    decode_error = ERROR_OUTPUT_MAPPING;
                } else {
                    let words = widen_normalized_f32_to_f64_words(normalized_bits);
                    write_word(offset, words.x);
                    write_word(offset + 4u, words.y);
                }
"#;
const F64_NATIVE_ARITHMETIC: &str = r#"
                if params.numeric_mapping != 2u {
                    decode_error = ERROR_OUTPUT_MAPPING;
                } else {
                    if (offset & 7u) != 0u || offset > params.logical_size
                        || params.logical_size - offset < 8u {
                        decode_error = ERROR_OUTPUT_BOUNDS;
                    } else {
                        output_f64[offset >> 3u] = f64(sample) / 255.0;
                    }
                }
"#;
const F64_NATIVE_BINDING: &str =
    "@group(0) @binding(6) var<storage, read_write> output_f64: array<f64>;";
const LOOKUP_BITS: u8 = 15;
const LOOKUP_SIZE: usize = 1 << LOOKUP_BITS;
const STATUS_OK: u32 = 1;
const STREAM_SENTINEL_BYTES: u64 = 4;
const NATIVE_F64_DUMMY_WORD_BYTES: u64 = 4;
const MAX_SESSION_IN_FLIGHT_BYTES: u64 = 64 * 1024 * 1024;

/// Conservative GPU allocation accounting for the stock decoder's bounded frame window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecodeMemoryStats {
    pub per_frame_bytes: u64,
    /// Bytes that remain reserved with the caller-owned output buffer.
    pub output_lease_bytes: u64,
    /// Per-frame bytes released when status readback completes.
    pub transient_bytes: u64,
    pub max_frame_slots: usize,
    /// Maximum exposure implied by `per_frame_bytes * max_frame_slots`.
    pub max_frame_window_bytes: u64,
}

/// F64 production path resolved for one output request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F64OutputPath {
    /// The shader evaluates the normalization with native f64 arithmetic.
    NativeArithmetic,
    /// The shader constructs the exact binary64 widening of a correctly-rounded f32 value.
    ExactF32Widening,
}

/// Capabilities of the stock wgpu decode engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecodeCapabilities {
    /// Whether `SHADER_F64` is enabled on the device and the native F64 pipeline is available.
    pub native_f64_arithmetic: bool,
}

/// CPU/WGSL ABI for `Params` in `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    token_start: u32,
    token_end: u32,
    width: u32,
    height: u32,
    origin_x: u32,
    origin_y: u32,
    sample_count: u32,
    initialize_chroma: u32,
    source_channels: u32,
    source_bits: u32,
    source_mask: u32,
    _source_padding: u32,
    output_kind: u32,
    transfer: u32,
    limited_range: u32,
    channels: u32,
    order: u32,
    bits: u32,
    storage_bits: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    plane3_offset: u32,
    plane3_stride: u32,
    chroma_width: u32,
    chroma_height: u32,
    logical_size: u32,
    numeric_mapping: u32,
    _padding: u32,
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
    assert!(std::mem::size_of::<ShaderParams>() == 128);
    assert!(std::mem::align_of::<ShaderParams>() == 4);
    assert!(std::mem::size_of::<DecodeStatus>() == 16);
    assert!(std::mem::align_of::<DecodeStatus>() == 4);
};

/// Stock GPU-only decoder for the standard lossless 1-16-bit Gray/RGB/RGBA Modular profile.
///
/// The frontend inventories standard frame sections and parses only bounded prefix metadata. The
/// shader reads entropy tokens from the actual `jxlc` bytes, expands distance-one zero runs,
/// unpacks signed residuals, applies the Gradient predictor, and writes the requested GPU image
/// layout. No private index, CPU pixel decoder, or CPU entropy fallback is required.
#[derive(Clone)]
pub struct WgpuSubmissionEngine {
    backend: WgpuBackend,
    pipeline: Arc<wgpu::ComputePipeline>,
    native_f64_pipeline: Option<Arc<OnceLock<Arc<wgpu::ComputePipeline>>>>,
    memory: MemoryBudget,
    buffers: Arc<DecodeBufferPool>,
}

impl std::fmt::Debug for WgpuSubmissionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuSubmissionEngine")
            .field("backend", &self.backend)
            .field("memory", &self.memory.snapshot())
            .field("buffer_pool", &self.buffers.stats())
            .finish_non_exhaustive()
    }
}

impl WgpuSubmissionEngine {
    #[must_use]
    pub fn new(backend: WgpuBackend) -> Self {
        let memory_budget = backend.transient_memory_budget().clone();
        Self::with_memory_budget(backend, memory_budget)
    }

    /// Constructs an engine with an explicitly supplied aggregate reservation budget.
    ///
    /// Passing a clone of another component's [`MemoryBudget`] makes decode jobs participate in
    /// that component's admission bound. [`Self::new`] uses the backend-wide transient budget and
    /// is the normal constructor.
    #[must_use]
    pub fn with_memory_budget(backend: WgpuBackend, memory_budget: MemoryBudget) -> Self {
        let pipeline = Arc::new(create_decode_pipeline(
            &backend,
            "jxl-wgpu decode lossless modular",
            &shader_source(F64OutputPath::ExactF32Widening),
        ));
        // Native f64 compilation is intentionally lazy: adapters that enable SHADER_F64 but
        // never request F64 output do not pay the additional pipeline initialization cost.
        let native_f64_pipeline = backend
            .native_f64_enabled()
            .then(|| Arc::new(OnceLock::new()));
        let buffers = DecodeBufferPool::new(
            backend.device().clone(),
            WgpuDecodeBufferPoolLimits::default(),
        );
        Self {
            backend,
            pipeline,
            native_f64_pipeline,
            memory: memory_budget,
            buffers,
        }
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        &self.backend
    }

    #[must_use]
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory.snapshot().limit_bytes
    }

    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    #[must_use]
    pub fn capabilities(&self) -> WgpuDecodeCapabilities {
        WgpuDecodeCapabilities {
            native_f64_arithmetic: self.native_f64_pipeline.is_some(),
        }
    }

    /// Current idle-cache limits. Active allocations remain governed by `MemoryBudget`.
    #[must_use]
    pub fn buffer_pool_limits(&self) -> WgpuDecodeBufferPoolLimits {
        self.buffers.limits()
    }

    /// Changes idle-cache limits and immediately evicts allocations outside the new bounds.
    pub fn set_buffer_pool_limits(&self, limits: WgpuDecodeBufferPoolLimits) {
        self.buffers.set_limits(limits);
    }

    /// Drops every idle allocation and invalidates all currently leased pool generations.
    ///
    /// In-flight jobs remain valid. Their transient buffers are discarded, rather than cached,
    /// after GPU completion. The returned generation is also published in
    /// [`Self::buffer_pool_stats`].
    pub fn clear_buffer_pool(&self) -> u64 {
        self.buffers.clear()
    }

    /// Reports physical idle/leased buffer reuse independently from logical byte admission.
    #[must_use]
    pub fn buffer_pool_stats(&self) -> WgpuDecodeBufferPoolStats {
        self.buffers.stats()
    }
}

fn shader_source(path: F64OutputPath) -> String {
    let (implementation, binding) = match path {
        F64OutputPath::NativeArithmetic => (F64_NATIVE_ARITHMETIC, F64_NATIVE_BINDING),
        F64OutputPath::ExactF32Widening => (F64_EXACT_F32_WIDENING, ""),
    };
    let source = SHADER_TEMPLATE
        .replace(F64_OUTPUT_MARKER, implementation)
        .replace(F64_BINDING_MARKER, binding);
    debug_assert!(!source.contains(F64_OUTPUT_MARKER));
    debug_assert!(!source.contains(F64_BINDING_MARKER));
    source
}

fn create_decode_pipeline(
    backend: &WgpuBackend,
    label: &str,
    shader: &str,
) -> wgpu::ComputePipeline {
    let module = backend
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
    backend
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("decode"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

impl GpuSubmissionEngine for WgpuSubmissionEngine {
    type Session = WgpuDecodeSession;

    fn parse_limits(&self) -> ParseLimits {
        ParseLimits {
            max_input_bytes: 16 * 1024 * 1024,
            max_boxes: 32,
            max_box_bytes: 16 * 1024 * 1024,
            max_codestream_bytes: 16 * 1024 * 1024,
        }
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        let parsed = jxl_gpu_bitstream::parse(codestream.bytes(), self.parse_limits())?;
        let inventory = parsed
            .codestream_inventory(InventoryLimits {
                max_frames: 1,
                max_total_section_bytes: u64::try_from(codestream.bytes().len())
                    .map_err(|_| Error::backend("codestream size exceeds u64"))?,
                ..InventoryLimits::default()
            })
            .map_err(Error::CodestreamInventory)?;
        let profile = parse_standard_modular_profile(codestream.bytes(), &inventory)?;
        let prefix_lookup: Arc<[u32]> = build_prefix_lookup(&profile)?.into();
        let extent = Extent2d::new(profile.width, profile.height);
        let output = OutputPlan::new(
            extent,
            request,
            profile.channels,
            profile.bits_per_sample,
            self.capabilities(),
        )?;
        let pipeline = match output.f64_output_path {
            Some(F64OutputPath::NativeArithmetic) => self
                .native_f64_pipeline
                .as_ref()
                .ok_or(Error::NativeF64Unavailable)?
                .get_or_init(|| {
                    Arc::new(create_decode_pipeline(
                        &self.backend,
                        "jxl-wgpu decode lossless modular native f64",
                        &shader_source(F64OutputPath::NativeArithmetic),
                    ))
                })
                .clone(),
            Some(F64OutputPath::ExactF32Widening) | None => Arc::clone(&self.pipeline),
        };
        let f64_output_path = output.f64_output_path;
        let dispatch_layout = GroupDispatchLayout::new(self.backend.device(), &profile)?;
        let memory_stats = validate_device_limits(
            self.backend.device(),
            codestream.bytes(),
            &profile,
            &dispatch_layout,
            &output,
            request.max_frame_slots().get(),
        )?;
        let predictor = FixedModularPredictor::new(5)
            .expect("the standard Modular profile uses the valid Gradient predictor index");
        Ok(PreparedGpuSession::new(
            DecodeProfile::ModularLossless {
                bits_per_sample: profile.bits_per_sample,
                channels: profile.channels,
                predictor,
                grouping: if profile.groups.len() == 1 {
                    crate::ModularGrouping::SingleGroup
                } else {
                    crate::ModularGrouping::MultipleGroups {
                        columns: profile.group_columns,
                        rows: profile.group_rows,
                    }
                },
            },
            AnimationMetadata::still(extent),
            WgpuDecodeSession {
                backend: self.backend.clone(),
                pipeline,
                source: Some(DecodeSource {
                    codestream_storage: codestream.shared_storage(),
                    codestream_range: codestream.storage_range(),
                    profile,
                    dispatch_layout,
                    prefix_lookup,
                    output,
                }),
                memory_stats,
                memory_budget: self.memory.clone(),
                buffers: Arc::clone(&self.buffers),
                f64_output_path,
            },
        ))
    }
}

/// One-frame runtime-neutral GPU decode session for the standard lossless Modular profile.
pub struct WgpuDecodeSession {
    backend: WgpuBackend,
    pipeline: Arc<wgpu::ComputePipeline>,
    source: Option<DecodeSource>,
    memory_stats: WgpuDecodeMemoryStats,
    memory_budget: MemoryBudget,
    buffers: Arc<DecodeBufferPool>,
    f64_output_path: Option<F64OutputPath>,
}

impl std::fmt::Debug for WgpuDecodeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuDecodeSession")
            .field("submitted", &self.source.is_none())
            .field("memory_stats", &self.memory_stats)
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionSession for WgpuDecodeSession {
    type Frame = GpuImageFrame;
    type Pending = WgpuPendingFrame;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        let Some(source) = self.source.as_ref() else {
            return Ok(None);
        };
        // Admission must precede Queue::submit and source consumption. Saturation leaves the
        // exact decode source available for a later prefetch attempt.
        let poll_permit = self
            .backend
            .submission_poller()
            .try_reserve()
            .map_err(Error::PollBackpressure)?;
        let output_permit = self
            .memory_budget
            .try_reserve(self.memory_stats.output_lease_bytes)?;
        let transient_permit = self
            .memory_budget
            .try_reserve(self.memory_stats.transient_bytes)?;
        let pending = submit_decode(
            &self.backend,
            &self.pipeline,
            source,
            &self.buffers,
            DecodeMemoryPermits {
                output: output_permit,
                transient: transient_permit,
            },
            poll_permit,
        )?;
        self.source = None;
        Ok(Some(pending))
    }
}

impl WgpuDecodeSession {
    #[must_use]
    pub const fn memory_stats(&self) -> WgpuDecodeMemoryStats {
        self.memory_stats
    }

    /// Maximum byte exposure allowed by this session's requested frame window.
    #[must_use]
    pub const fn max_frame_window_gpu_bytes(&self) -> u64 {
        self.memory_stats.max_frame_window_bytes
    }

    /// Reports allocations currently retained by jobs and output leases across engine clones.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory_budget.snapshot()
    }

    /// Resolved F64 path for this session, or `None` when the requested output is not F64.
    #[must_use]
    pub const fn f64_output_path(&self) -> Option<F64OutputPath> {
        self.f64_output_path
    }
}

/// One submitted stock Modular frame. Queue submission has completed, while mapped validation may
/// still be pending.
pub struct WgpuPendingFrame {
    device: wgpu::Device,
    lifetime: Option<Arc<DecodeJobLifetime>>,
    token: SubmissionToken,
    layout: ImageLayout,
    completion: Arc<MapCompletion>,
    group_sample_counts: Arc<[u32]>,
    status_stride: u64,
}

impl std::fmt::Debug for WgpuPendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuPendingFrame")
            .field("token", &self.token)
            .field("layout", &self.layout)
            .field("group_sample_counts", &self.group_sample_counts)
            .finish_non_exhaustive()
    }
}

impl WgpuPendingFrame {
    /// Clones a budget-tracked lease to the queue-submitted output before validation completes.
    ///
    /// Submit consumers only to the same [`WgpuBackend`] device and queue that created this decode
    /// session. Queue ordering then permits display, readback, or custom GPU work without a host
    /// wait. This value deliberately has no authoritative frame metadata or changed regions. If
    /// [`GpuDecodeSession::next_frame`](crate::GpuDecodeSession::next_frame) later returns an error,
    /// already-submitted consumer work cannot be rolled back and all derived data must be
    /// discarded.
    ///
    /// The returned [`GpuBufferLease`] clone retains the output allocation's shared byte-budget
    /// permit. Keep that lease alive instead of cloning its raw wgpu buffer handle.
    pub fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        let lifetime = self.lifetime.as_ref().ok_or(Error::EngineContract(
            "Modular GPU pending frame was already consumed",
        ))?;
        Ok(UnvalidatedGpuImageFrame {
            token: self.token,
            outputs: vec![UnvalidatedGpuImageOutput {
                id: OutputId(0),
                layout: self.layout.clone(),
                buffer: GpuBufferLease::with_memory_permit(
                    Arc::clone(&lifetime.output),
                    lifetime.output_permit.clone(),
                ),
            }],
        })
    }

    fn finish(
        &mut self,
        mapping: std::result::Result<(), String>,
    ) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        mapping.map_err(Error::backend)?;
        let lifetime = self.lifetime.take().ok_or(Error::EngineContract(
            "Modular GPU completion was consumed more than once",
        ))?;
        let mapped = lifetime
            .status_staging
            .buffer()
            .slice(..)
            .get_mapped_range()
            .map_err(Error::backend)?;
        let statuses = self
            .group_sample_counts
            .iter()
            .copied()
            .enumerate()
            .map(|(group_index, expected_samples)| {
                let start = u64::try_from(group_index)
                    .ok()
                    .and_then(|index| index.checked_mul(self.status_stride))
                    .and_then(|offset| usize::try_from(offset).ok())
                    .ok_or_else(|| Error::backend("GPU status offset overflow"))?;
                let end = start
                    .checked_add(STATUS_BYTES as usize)
                    .ok_or_else(|| Error::backend("GPU status range overflow"))?;
                let bytes = mapped
                    .get(start..end)
                    .ok_or_else(|| Error::backend("GPU status buffer was truncated"))?;
                let status = bytemuck::try_cast_slice::<u8, DecodeStatus>(bytes)
                    .map_err(|_| Error::backend("GPU status buffer has an invalid ABI layout"))?
                    .first()
                    .copied()
                    .ok_or_else(|| Error::backend("GPU status buffer was truncated"))?;
                Ok((group_index, expected_samples, status))
            })
            .collect::<Result<Vec<_>>>();
        drop(mapped);
        for (group_index, expected_samples, status) in statuses? {
            if status.code != STATUS_OK
                || status.decoded_samples != expected_samples
                || status.cursor != status.expected_cursor
            {
                return Err(Error::backend(format!(
                    "Modular GPU group {group_index} rejected entropy stream: status={}, decoded={}/{}, cursor={}/{}",
                    status.code,
                    status.decoded_samples,
                    expected_samples,
                    status.cursor,
                    status.expected_cursor
                )));
            }
        }

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
                name: String::new(),
            },
            GpuImageFrame {
                token: self.token,
                outputs: vec![GpuImageOutput {
                    id: output_id,
                    layout: self.layout.clone(),
                    buffer: GpuBufferLease::with_memory_permit(
                        Arc::clone(&lifetime.output),
                        lifetime.output_permit.clone(),
                    ),
                }],
                changed: ChangedRegions { outputs: regions },
            },
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for WgpuPendingFrame {
    type Frame = GpuImageFrame;

    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        let mapping = self.completion.wait();
        self.finish(mapping)
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for WgpuPendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if let Err(error) = self.device.poll(wgpu::PollType::Poll) {
            return Poll::Ready(Err(Error::backend(error)));
        }
        match self.completion.poll(context) {
            Some(mapping) => Poll::Ready(self.finish(mapping)),
            None => Poll::Pending,
        }
    }
}

struct DecodeSource {
    codestream_storage: Arc<[u8]>,
    codestream_range: std::ops::Range<usize>,
    profile: StandardModularProfile,
    dispatch_layout: GroupDispatchLayout,
    // Immutable within the session. All independently decoded groups share the standard
    // DC-global prefix set without sharing mutable GPU transient allocations.
    prefix_lookup: Arc<[u32]>,
    output: OutputPlan,
}

struct DecodeJobLifetime {
    output: Arc<wgpu::Buffer>,
    _lookup: DecodeBufferLease,
    _reconstructed: DecodeBufferLease,
    _native_f64_dummy_words: Option<DecodeBufferLease>,
    _status: DecodeBufferLease,
    status_staging: DecodeBufferLease,
    status_mapped: AtomicBool,
    _params: DecodeBufferLease,
    output_permit: MemoryPermit,
    _transient_permit: MemoryPermit,
}

impl Drop for DecodeJobLifetime {
    fn drop(&mut self) {
        // A successful map remains mapped until explicitly released. This also covers abandoned
        // sessions/Futures: the callback owns the final Arc until mapping has completed, then this
        // drop runs and unmaps before field destruction returns the staging lease to the pool.
        if self.status_mapped.swap(false, Ordering::AcqRel) {
            self.status_staging.buffer().unmap();
        }
    }
}

struct DecodeMemoryPermits {
    output: MemoryPermit,
    transient: MemoryPermit,
}

#[derive(Clone, Debug)]
struct GroupDispatchLayout {
    reconstructed_offsets: Arc<[u64]>,
    reconstructed_bytes: u64,
    status_stride: u64,
    status_bytes: u64,
    params_stride: u64,
    params_bytes: u64,
}

impl GroupDispatchLayout {
    fn new(device: &wgpu::Device, profile: &StandardModularProfile) -> Result<Self> {
        let limits = device.limits();
        let storage_alignment = u64::from(limits.min_storage_buffer_offset_alignment.max(4));
        let uniform_alignment = u64::from(limits.min_uniform_buffer_offset_alignment.max(16));
        let mut reconstructed_offsets = Vec::with_capacity(profile.groups.len());
        let mut reconstructed_bytes = 0u64;
        for group in &profile.groups {
            reconstructed_bytes = align_to(reconstructed_bytes, storage_alignment)?;
            reconstructed_offsets.push(reconstructed_bytes);
            reconstructed_bytes = reconstructed_bytes
                .checked_add(
                    u64::from(group.sample_count()?)
                        .checked_mul(u64::from(profile.channels.count()))
                        .and_then(|samples| samples.checked_mul(4))
                        .ok_or_else(|| {
                            Error::backend("group reconstruction buffer size overflow")
                        })?,
                )
                .ok_or_else(|| Error::backend("reconstruction buffer size overflow"))?;
        }
        reconstructed_bytes = align4(reconstructed_bytes)?;
        let group_count = u64::try_from(profile.groups.len())
            .map_err(|_| Error::backend("Modular group count exceeds u64"))?;
        let status_stride = align_to(STATUS_BYTES, storage_alignment)?;
        let status_bytes = status_stride
            .checked_mul(group_count.saturating_sub(1))
            .and_then(|bytes| bytes.checked_add(STATUS_BYTES))
            .and_then(|bytes| align4(bytes).ok())
            .ok_or_else(|| Error::backend("group status buffer size overflow"))?;
        let params_stride = align_to(
            std::mem::size_of::<ShaderParams>() as u64,
            uniform_alignment,
        )?;
        let params_bytes = params_stride
            .checked_mul(group_count.saturating_sub(1))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<ShaderParams>() as u64))
            .and_then(|bytes| align4(bytes).ok())
            .ok_or_else(|| Error::backend("group parameter buffer size overflow"))?;
        Ok(Self {
            reconstructed_offsets: reconstructed_offsets.into(),
            reconstructed_bytes,
            status_stride,
            status_bytes,
            params_stride,
            params_bytes,
        })
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputKind {
    NumericUnsigned = 0,
    Luma = 1,
    YuvSemiplanar = 2,
    YuvPlanar = 3,
    Yuv422Packed = 4,
    RgbInterleaved = 5,
    RgbPlanar = 6,
    NumericSigned = 7,
    NumericFloat = 8,
    NativeModular = 9,
}

struct OutputPlan {
    layout: ImageLayout,
    kind: OutputKind,
    transfer: u32,
    limited_range: bool,
    channels: u32,
    order: u32,
    bits: u32,
    storage_bits: u32,
    numeric_mapping: u32,
    f64_output_path: Option<F64OutputPath>,
}

impl OutputPlan {
    fn new(
        extent: Extent2d,
        request: &GpuOutputRequest,
        source_channels: crate::ModularChannels,
        source_bits: u8,
        capabilities: WgpuDecodeCapabilities,
    ) -> Result<Self> {
        let format = request.format().clone();
        if let Some(native) = native_modular_format(&format) {
            let native_mapping = matches!(
                (native.channels, request.mapping()),
                (
                    crate::ModularChannels::Gray,
                    GpuOutputMapping::Numeric(NumericSampleMapping::NativeUnsigned)
                ) | (
                    crate::ModularChannels::Rgb | crate::ModularChannels::Rgba,
                    GpuOutputMapping::Color
                )
            );
            if native_mapping {
                if native.channels != source_channels || native.bits_per_sample != source_bits {
                    return Err(Error::UnsupportedOutputFormat(format!(
                        "native Modular output {native:?} does not match {:?} {}-bit source",
                        source_channels, source_bits
                    )));
                }
                let output = Self {
                    layout: ImageLayout::packed(extent, format)?,
                    kind: OutputKind::NativeModular,
                    transfer: 0,
                    limited_range: false,
                    channels: native.channels.count(),
                    order: 0,
                    bits: u32::from(native.bits_per_sample),
                    storage_bits: u32::from(native.storage_bits),
                    numeric_mapping: 3,
                    f64_output_path: None,
                };
                output.validate_shader_layout()?;
                return Ok(output);
            }
        }
        if source_channels != crate::ModularChannels::Gray || source_bits != 8 {
            return Err(Error::UnsupportedOutputFormat(
                "RGB/RGBA and non-8-bit Modular sources currently require their exact canonical native output descriptor"
                    .into(),
            ));
        }
        let class = classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?;
        let (
            kind,
            transfer,
            limited_range,
            channels,
            order,
            bits,
            storage_bits,
            numeric_mapping,
            f64_output_path,
        ) = match (class, request.mapping()) {
            (
                PixelFormatClass::Numeric(numeric),
                GpuOutputMapping::Numeric(NumericSampleMapping::NormalizedGray8),
            ) => {
                if numeric.sample_kind == SampleKind::Float && numeric.bits_per_component == 64 {
                    return Err(Error::F64OutputPolicyRequired);
                }
                let kind = match numeric.sample_kind {
                    SampleKind::Unsigned => OutputKind::NumericUnsigned,
                    SampleKind::Signed => OutputKind::NumericSigned,
                    SampleKind::Float => OutputKind::NumericFloat,
                };
                (
                    kind,
                    0,
                    false,
                    u32::from(numeric.components),
                    0,
                    u32::from(numeric.bits_per_component),
                    u32::from(numeric.bits_per_component),
                    1,
                    None,
                )
            }
            (
                PixelFormatClass::Numeric(numeric),
                GpuOutputMapping::Numeric(NumericSampleMapping::NormalizedGray8F64(policy)),
            ) => {
                if numeric.sample_kind != SampleKind::Float
                    || numeric.bits_per_component != 64
                    || numeric.components != 1
                {
                    return Err(Error::F64OutputPolicyForNonF64);
                }
                let path = resolve_f64_output_path(policy, capabilities)?;
                (
                    OutputKind::NumericFloat,
                    0,
                    false,
                    1,
                    0,
                    64,
                    64,
                    match path {
                        F64OutputPath::ExactF32Widening => 1,
                        F64OutputPath::NativeArithmetic => 2,
                    },
                    Some(path),
                )
            }
            (PixelFormatClass::Numeric(_), GpuOutputMapping::Color) => {
                return Err(Error::NumericMappingRequired);
            }
            (
                PixelFormatClass::Numeric(_),
                GpuOutputMapping::Numeric(NumericSampleMapping::NativeUnsigned),
            ) => {
                return Err(Error::UnsupportedOutputFormat(
                    "native unsigned output descriptor does not match the Modular source".into(),
                ));
            }
            (PixelFormatClass::Color(_), GpuOutputMapping::Numeric(_)) => {
                return Err(Error::NumericMappingForColorOutput);
            }
            (PixelFormatClass::Color(color), GpuOutputMapping::Color) => {
                let (transfer, limited_range) = color_conversion(&format)?;
                let (kind, channels, order, bits, storage_bits) = match color {
                    ColorFormatClass::Rgb8 { storage, order } => {
                        if limited_range {
                            return Err(Error::UnsupportedOutputFormat(
                                "RGB output requires an explicit full-range color specification"
                                    .into(),
                            ));
                        }
                        let (channels, order) = rgb_output_shape(order);
                        let kind = match storage {
                            RgbStorage::Interleaved => OutputKind::RgbInterleaved,
                            RgbStorage::Planar => OutputKind::RgbPlanar,
                        };
                        (kind, channels, order, 8, 8)
                    }
                    ColorFormatClass::Luma { bits, storage_bits }
                        if matches!((bits, storage_bits), (8, 8) | (16, 16)) =>
                    {
                        (
                            OutputKind::Luma,
                            1,
                            0,
                            u32::from(bits),
                            u32::from(storage_bits),
                        )
                    }
                    ColorFormatClass::YuvSemiplanar {
                        bits: 8,
                        storage_bits: 8,
                        chroma_order,
                        ..
                    } => (
                        OutputKind::YuvSemiplanar,
                        3,
                        match chroma_order {
                            ChromaOrder::CbCr => 0,
                            ChromaOrder::CrCb => 1,
                        },
                        8,
                        8,
                    ),
                    ColorFormatClass::YuvPlanar {
                        bits: 8,
                        storage_bits: 8,
                        ..
                    } => (OutputKind::YuvPlanar, 3, 0, 8, 8),
                    ColorFormatClass::Yuv422Packed { order } => (
                        OutputKind::Yuv422Packed,
                        3,
                        match order {
                            Packed422Order::Yuyv => 0,
                            Packed422Order::Uyvy => 1,
                        },
                        8,
                        8,
                    ),
                    unsupported => {
                        return Err(Error::UnsupportedOutputFormat(format!(
                            "the 8-bit Gray GPU conversion path does not implement color storage {unsupported:?}"
                        )));
                    }
                };
                (
                    kind,
                    transfer,
                    limited_range,
                    channels,
                    order,
                    bits,
                    storage_bits,
                    0,
                    None,
                )
            }
        };
        let output = Self {
            layout: ImageLayout::packed(extent, format)?,
            kind,
            transfer,
            limited_range,
            channels,
            order,
            bits,
            storage_bits,
            numeric_mapping,
            f64_output_path,
        };
        output.validate_shader_layout()?;
        Ok(output)
    }

    fn validate_shader_layout(&self) -> Result<()> {
        let expected_planes = match self.kind {
            OutputKind::NumericUnsigned
            | OutputKind::NumericSigned
            | OutputKind::NumericFloat
            | OutputKind::Luma
            | OutputKind::Yuv422Packed
            | OutputKind::RgbInterleaved
            | OutputKind::NativeModular => 1,
            OutputKind::YuvSemiplanar => 2,
            OutputKind::YuvPlanar => 3,
            OutputKind::RgbPlanar => usize::try_from(self.channels)
                .map_err(|_| Error::backend("RGB plane count overflow"))?,
        };
        if self.layout.planes.len() != expected_planes || expected_planes > 4 {
            return Err(Error::backend(format!(
                "requested output has {} planes; {:?} requires {expected_planes}",
                self.layout.planes.len(),
                self.kind
            )));
        }
        u32::try_from(self.layout.logical_size)
            .map_err(|_| Error::backend("requested output exceeds the WGSL u32 address space"))?;
        for plane in &self.layout.planes {
            for (name, value) in [
                ("offset", plane.offset),
                ("row stride", plane.row_stride),
                ("row bytes", plane.row_bytes),
                ("end offset", plane.end_offset()?),
            ] {
                u32::try_from(value).map_err(|_| {
                    Error::backend(format!(
                        "output plane {} {name} exceeds the WGSL u32 address space",
                        plane.plane_index
                    ))
                })?;
            }
        }
        if matches!(self.kind, OutputKind::Yuv422Packed)
            || (self.kind == OutputKind::RgbInterleaved && self.channels == 4)
        {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4) {
                return Err(Error::backend(
                    "four-byte packed output requires four-byte-aligned rows",
                ));
            }
        }
        if matches!(
            self.kind,
            OutputKind::NumericUnsigned | OutputKind::NumericSigned | OutputKind::NumericFloat
        ) && self.bits >= 32
        {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4) {
                return Err(Error::backend(
                    "32/64-bit numeric output requires four-byte-aligned rows",
                ));
            }
        }
        if self.kind == OutputKind::NumericFloat && self.bits == 64 {
            let plane = &self.layout.planes[0];
            if !plane.offset.is_multiple_of(8) || !plane.row_stride.is_multiple_of(8) {
                return Err(Error::backend(
                    "F64 numeric output requires eight-byte-aligned rows",
                ));
            }
        }
        Ok(())
    }
}

fn resolve_f64_output_path(
    policy: F64OutputPolicy,
    capabilities: WgpuDecodeCapabilities,
) -> Result<F64OutputPath> {
    match policy {
        F64OutputPolicy::NativeRequired => capabilities
            .native_f64_arithmetic
            .then_some(F64OutputPath::NativeArithmetic)
            .ok_or(Error::NativeF64Unavailable),
        F64OutputPolicy::NativeOrExactF32Widening => Ok(if capabilities.native_f64_arithmetic {
            F64OutputPath::NativeArithmetic
        } else {
            F64OutputPath::ExactF32Widening
        }),
        F64OutputPolicy::ExactF32Widening => Ok(F64OutputPath::ExactF32Widening),
    }
}

fn rgb_output_shape(order: RgbChannelOrder) -> (u32, u32) {
    match order {
        RgbChannelOrder::Rgb => (3, 0),
        RgbChannelOrder::Bgr => (3, 1),
        RgbChannelOrder::Rgba => (4, 2),
        RgbChannelOrder::Bgra => (4, 3),
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
                "the 8-bit Gray GPU conversion path does not implement {transfer:?} output transfer"
            )));
        }
    };
    Ok((transfer, spec.range == ColorRange::Limited))
}

fn validate_device_limits(
    device: &wgpu::Device,
    codestream: &[u8],
    profile: &StandardModularProfile,
    dispatch: &GroupDispatchLayout,
    output: &OutputPlan,
    max_frame_slots: usize,
) -> Result<WgpuDecodeMemoryStats> {
    if profile
        .groups
        .iter()
        .any(|group| group.token_bit_end > u64::from(u32::MAX))
    {
        return Err(Error::backend(
            "standard group token offsets exceed the portable WGSL u32 address space",
        ));
    }
    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let buffer_limit = device.limits().max_buffer_size;
    let codestream_bytes = stream_allocation_size(codestream.len())?;
    let lookup_bytes = u64::try_from(LOOKUP_SIZE)
        .ok()
        .and_then(|entries| entries.checked_mul(u64::from(profile.channels.count())))
        .and_then(|entries| entries.checked_mul(4))
        .ok_or_else(|| Error::backend("prefix lookup size overflow"))?;
    let output_bytes = align4(output.layout.logical_size)?;
    let native_f64_dummy_bytes = if output.f64_output_path == Some(F64OutputPath::NativeArithmetic)
    {
        NATIVE_F64_DUMMY_WORD_BYTES
    } else {
        0
    };
    for (name, required) in [
        ("codestream", codestream_bytes),
        ("prefix lookup", lookup_bytes),
        ("requested output", output_bytes),
    ] {
        if required > storage_limit || required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device limit"
            )));
        }
    }
    if profile.groups.iter().any(|group| {
        u64::from(group.width)
            .checked_mul(u64::from(group.height))
            .and_then(|samples| samples.checked_mul(u64::from(profile.channels.count())))
            .and_then(|samples| samples.checked_mul(4))
            .is_none_or(|bytes| bytes > storage_limit)
    }) {
        return Err(Error::backend(
            "a Modular group reconstruction binding exceeds the device storage-binding limit",
        ));
    }
    for (name, required) in [
        ("reconstructed samples", dispatch.reconstructed_bytes),
        ("group statuses", dispatch.status_bytes),
        ("group status readback", dispatch.status_bytes),
        ("group parameters", dispatch.params_bytes),
    ] {
        if required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device buffer limit"
            )));
        }
    }
    let per_frame = [
        codestream_bytes,
        lookup_bytes,
        dispatch.reconstructed_bytes,
        output_bytes,
        native_f64_dummy_bytes,
        dispatch.status_bytes,
        dispatch.status_bytes,
        dispatch.params_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| Error::backend("Modular GPU memory budget overflow"))?;
    let max_frame_window_bytes = per_frame
        .checked_mul(
            u64::try_from(max_frame_slots)
                .map_err(|_| Error::backend("frame-slot count overflow"))?,
        )
        .ok_or_else(|| Error::backend("bounded in-flight GPU memory budget overflow"))?;
    if max_frame_window_bytes > MAX_SESSION_IN_FLIGHT_BYTES {
        return Err(Error::backend(format!(
            "bounded Modular session exposes {max_frame_window_bytes} bytes ({per_frame} per frame), exceeding the {}-byte session limit",
            MAX_SESSION_IN_FLIGHT_BYTES
        )));
    }
    let transient_bytes = per_frame
        .checked_sub(output_bytes)
        .ok_or_else(|| Error::backend("Modular transient memory accounting underflow"))?;
    Ok(WgpuDecodeMemoryStats {
        per_frame_bytes: per_frame,
        output_lease_bytes: output_bytes,
        transient_bytes,
        max_frame_slots,
        max_frame_window_bytes,
    })
}

fn submit_decode(
    backend: &WgpuBackend,
    pipeline: &wgpu::ComputePipeline,
    source: &DecodeSource,
    buffers: &Arc<DecodeBufferPool>,
    memory_permits: DecodeMemoryPermits,
    poll_permit: SubmissionPollPermit,
) -> Result<WgpuPendingFrame> {
    let device = backend.device();
    let codestream = source
        .codestream_storage
        .get(source.codestream_range.clone())
        .ok_or_else(|| Error::backend("codestream storage range is invalid"))?;
    // The raw source is intentionally not pooled: it is session data, not a reusable transient
    // shape. Upload aligned spans directly from the shared input Arc rather than allocating a
    // second full codestream Vec on the host.
    let stream = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decode codestream"),
        size: stream_allocation_size(codestream.len())?,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    upload_codestream(backend.queue(), &stream, codestream)?;

    let lookup_bytes = u64::try_from(source.prefix_lookup.len())
        .ok()
        .and_then(|entries| entries.checked_mul(u64::try_from(std::mem::size_of::<u32>()).ok()?))
        .ok_or_else(|| Error::backend("prefix lookup size overflow"))?;
    let lookup_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let lookup_buffer = buffers.checkout(
        "jxl-wgpu decode prefix lookup",
        lookup_bytes,
        lookup_usage,
        std::mem::align_of::<u32>() as u64,
    );
    backend.queue().write_buffer(
        lookup_buffer.buffer(),
        0,
        bytemuck::cast_slice(source.prefix_lookup.as_ref()),
    );

    let reconstructed_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let reconstructed = buffers.checkout(
        "jxl-wgpu decoded Modular samples",
        source.dispatch_layout.reconstructed_bytes,
        reconstructed_usage,
        std::mem::align_of::<u32>() as u64,
    );
    let output_size = align4(source.output.layout.logical_size)?;
    let output = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decoded image output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    // The native shader declares the caller-visible allocation exactly once as `array<f64>`.
    // Its otherwise-unused raw-word binding receives a distinct dummy allocation, avoiding two
    // writable storage aliases for the same buffer.
    let native_f64_dummy_words =
        (source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic)).then(|| {
            buffers.checkout(
                "jxl-wgpu native F64 dummy word output",
                NATIVE_F64_DUMMY_WORD_BYTES,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                std::mem::align_of::<u32>() as u64,
            )
        });
    let status_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let status = buffers.checkout(
        "jxl-wgpu decode status",
        source.dispatch_layout.status_bytes,
        status_usage,
        std::mem::align_of::<DecodeStatus>() as u64,
    );
    let status_staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
    let status_staging = buffers.checkout(
        "jxl-wgpu decode status readback",
        source.dispatch_layout.status_bytes,
        status_staging_usage,
        wgpu::COPY_BUFFER_ALIGNMENT,
    );

    let params_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let params_buffer = buffers.checkout(
        "jxl-wgpu decode Modular parameters",
        source.dispatch_layout.params_bytes,
        params_usage,
        16,
    );
    let mut params_upload = vec![
        0u8;
        usize::try_from(source.dispatch_layout.params_bytes).map_err(
            |_| Error::backend("group parameter upload exceeds host address space")
        )?
    ];
    for (index, &group) in source.profile.groups.iter().enumerate() {
        let params = build_params(
            group,
            source.profile.channels,
            source.profile.bits_per_sample,
            &source.output,
            index == 0,
        )?;
        let offset = u64::try_from(index)
            .ok()
            .and_then(|index| index.checked_mul(source.dispatch_layout.params_stride))
            .and_then(|offset| usize::try_from(offset).ok())
            .ok_or_else(|| Error::backend("group parameter offset overflow"))?;
        let end = offset
            .checked_add(std::mem::size_of::<ShaderParams>())
            .ok_or_else(|| Error::backend("group parameter range overflow"))?;
        params_upload
            .get_mut(offset..end)
            .ok_or_else(|| Error::backend("group parameter buffer is truncated"))?
            .copy_from_slice(bytemuck::bytes_of(&params));
    }
    backend
        .queue()
        .write_buffer(params_buffer.buffer(), 0, &params_upload);

    let word_output_binding = native_f64_dummy_words.as_ref().map_or_else(
        || output.as_entire_binding(),
        |buffer| buffer.buffer().as_entire_binding(),
    );
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut bindings = Vec::with_capacity(source.profile.groups.len());
    for (group_index, group) in source.profile.groups.iter().enumerate() {
        let index = u64::try_from(group_index)
            .map_err(|_| Error::backend("group binding index exceeds u64"))?;
        let reconstructed_offset = *source
            .dispatch_layout
            .reconstructed_offsets
            .get(group_index)
            .ok_or_else(|| Error::backend("missing group reconstruction offset"))?;
        let reconstructed_size = u64::from(group.sample_count()?)
            .checked_mul(u64::from(source.profile.channels.count()))
            .and_then(|samples| samples.checked_mul(4))
            .and_then(NonZeroU64::new)
            .ok_or_else(|| Error::backend("invalid group reconstruction binding size"))?;
        let status_offset = index
            .checked_mul(source.dispatch_layout.status_stride)
            .ok_or_else(|| Error::backend("group status binding offset overflow"))?;
        let params_offset = index
            .checked_mul(source.dispatch_layout.params_stride)
            .ok_or_else(|| Error::backend("group parameter binding offset overflow"))?;
        let reconstructed_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: reconstructed.buffer(),
            offset: reconstructed_offset,
            size: Some(reconstructed_size),
        });
        let status_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: status.buffer(),
            offset: status_offset,
            size: NonZeroU64::new(STATUS_BYTES),
        });
        let params_binding = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: params_buffer.buffer(),
            offset: params_offset,
            size: NonZeroU64::new(std::mem::size_of::<ShaderParams>() as u64),
        });
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: stream.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: lookup_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: reconstructed_binding,
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: word_output_binding.clone(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: status_binding,
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_binding,
            },
        ];
        if source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: output.as_entire_binding(),
            });
        }
        bindings.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("jxl-wgpu decode Modular group bindings"),
            layout: &bind_group_layout,
            entries: &entries,
        }));
    }
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("jxl-wgpu decode Modular submission"),
    });
    commands.clear_buffer(reconstructed.buffer(), 0, None);
    commands.clear_buffer(&output, 0, None);
    if let Some(dummy) = &native_f64_dummy_words {
        commands.clear_buffer(dummy.buffer(), 0, None);
    }
    commands.clear_buffer(status.buffer(), 0, None);
    {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("jxl-wgpu entropy and Gradient reconstruction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for binding in &bindings {
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
    commands.copy_buffer_to_buffer(
        status.buffer(),
        0,
        status_staging.buffer(),
        0,
        source.dispatch_layout.status_bytes,
    );

    let completion = Arc::new(MapCompletion::default());
    let callback_completion = Arc::clone(&completion);
    let lifetime = Arc::new(DecodeJobLifetime {
        output,
        _lookup: lookup_buffer,
        _reconstructed: reconstructed,
        _native_f64_dummy_words: native_f64_dummy_words,
        _status: status,
        status_staging,
        status_mapped: AtomicBool::new(false),
        _params: params_buffer,
        output_permit: memory_permits.output,
        _transient_permit: memory_permits.transient,
    });
    let callback_lifetime = Arc::clone(&lifetime);
    commands.map_buffer_on_submit(
        lifetime.status_staging.buffer(),
        wgpu::MapMode::Read,
        ..,
        move |result| {
            // Release the callback's ownership before waking a waiter. The pending frame keeps
            // the job alive through status validation; an abandoned pending frame instead makes
            // this the final Arc, so staging is unmapped and recycled at this proven boundary.
            if result.is_ok() {
                callback_lifetime
                    .status_mapped
                    .store(true, Ordering::Release);
            }
            drop(callback_lifetime);
            callback_completion
                .complete(result.map_err(|error| format!("GPU status mapping failed: {error}")));
        },
    );
    let submission = backend.queue().submit([commands.finish()]);
    let poll_completion = Arc::clone(&completion);
    if let Err(error) = poll_permit.register(submission, move |error| {
        poll_completion.complete(Err(error));
    }) {
        completion.complete(Err(format!("GPU poll registration failed: {error}")));
    }

    Ok(WgpuPendingFrame {
        device: backend.device().clone(),
        lifetime: Some(lifetime),
        token: SubmissionToken(1),
        layout: source.output.layout.clone(),
        completion,
        group_sample_counts: source
            .profile
            .groups
            .iter()
            .copied()
            .map(ModularGroup::sample_count)
            .map(|samples| {
                samples.and_then(|samples| {
                    samples
                        .checked_mul(source.profile.channels.count())
                        .ok_or_else(|| Error::backend("group decoded sample count overflow"))
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into(),
        status_stride: source.dispatch_layout.status_stride,
    })
}

fn build_prefix_lookup(profile: &StandardModularProfile) -> Result<Vec<u32>> {
    let channel_count = usize::try_from(profile.channels.count())
        .map_err(|_| Error::backend("Modular channel count exceeds usize"))?;
    let mut lookup = vec![0u32; LOOKUP_SIZE * channel_count];
    for channel in 0..channel_count {
        let table_start = channel
            .checked_mul(LOOKUP_SIZE)
            .ok_or_else(|| Error::backend("prefix lookup channel offset overflow"))?;
        let table_end = table_start
            .checked_add(LOOKUP_SIZE)
            .ok_or_else(|| Error::backend("prefix lookup channel range overflow"))?;
        let table = lookup
            .get_mut(table_start..table_end)
            .ok_or_else(|| Error::backend("prefix lookup channel range is truncated"))?;
        for (symbol, entry) in profile.raw_prefix[channel]
            .iter()
            .copied()
            .enumerate()
            .map(|(symbol, entry)| (symbol as u32, entry))
            .chain(
                profile.lz77_prefix[channel]
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(symbol, entry)| (224 + symbol as u32, entry)),
            )
        {
            insert_prefix(table, symbol, entry)?;
        }
    }
    Ok(lookup)
}

fn insert_prefix(lookup: &mut [u32], symbol: u32, entry: PrefixCodeEntry) -> Result<()> {
    if entry.bit_len == 0 {
        return Ok(());
    }
    if entry.bit_len > LOOKUP_BITS {
        return Err(Error::backend(format!(
            "validated standard prefix length {} exceeds the {LOOKUP_BITS}-bit GPU lookup",
            entry.bit_len
        )));
    }
    let suffix_bits = LOOKUP_BITS - entry.bit_len;
    for suffix in 0..1usize << suffix_bits {
        let index = usize::from(entry.bits) | (suffix << entry.bit_len);
        if lookup[index] != 0 {
            return Err(Error::backend(
                "validated standard prefix entries collide in the GPU lookup table",
            ));
        }
        lookup[index] = (symbol << 8) | u32::from(entry.bit_len);
    }
    Ok(())
}

fn build_params(
    group: ModularGroup,
    source_channels: crate::ModularChannels,
    source_bits: u8,
    output: &OutputPlan,
    initialize_chroma: bool,
) -> Result<ShaderParams> {
    let to_u32 = |value: u64, name: &'static str| {
        u32::try_from(value).map_err(|_| Error::backend(format!("{name} exceeds WGSL u32")))
    };
    let plane = |index: usize| -> Result<(u32, u32)> {
        output.layout.plane(index).map_or(Ok((0, 0)), |plane| {
            Ok((
                to_u32(plane.offset, "plane offset")?,
                to_u32(plane.row_stride, "plane row stride")?,
            ))
        })
    };
    let (plane0_offset, plane0_stride) = plane(0)?;
    let (plane1_offset, plane1_stride) = plane(1)?;
    let (plane2_offset, plane2_stride) = plane(2)?;
    let (plane3_offset, plane3_stride) = plane(3)?;
    let chroma = output.layout.plane(1);
    Ok(ShaderParams {
        token_start: to_u32(group.token_bit_offset, "token start")?,
        token_end: to_u32(group.token_bit_end, "token end")?,
        width: group.width,
        height: group.height,
        origin_x: group.x,
        origin_y: group.y,
        sample_count: group.sample_count()?,
        initialize_chroma: u32::from(initialize_chroma),
        source_channels: source_channels.count(),
        source_bits: u32::from(source_bits),
        source_mask: (1u32 << source_bits) - 1,
        _source_padding: 0,
        output_kind: output.kind as u32,
        transfer: output.transfer,
        limited_range: u32::from(output.limited_range),
        channels: output.channels,
        order: output.order,
        bits: output.bits,
        storage_bits: output.storage_bits,
        plane0_offset,
        plane0_stride,
        plane1_offset,
        plane1_stride,
        plane2_offset,
        plane2_stride,
        plane3_offset,
        plane3_stride,
        chroma_width: chroma.map_or(0, |plane| plane.sample_extent.width),
        chroma_height: chroma.map_or(0, |plane| plane.sample_extent.height),
        logical_size: to_u32(output.layout.logical_size, "output logical size")?,
        numeric_mapping: output.numeric_mapping,
        _padding: 0,
    })
}

fn stream_allocation_size(byte_len: usize) -> Result<u64> {
    let byte_len = u64::try_from(byte_len)
        .map_err(|_| Error::backend("codestream allocation size overflow"))?;
    align4(byte_len)?
        .checked_add(STREAM_SENTINEL_BYTES)
        .ok_or_else(|| Error::backend("codestream sentinel allocation overflow"))
}

fn upload_codestream(
    queue: &wgpu::Queue,
    destination: &wgpu::Buffer,
    codestream: &[u8],
) -> Result<()> {
    let aligned_prefix = codestream.len() & !(wgpu::COPY_BUFFER_ALIGNMENT as usize - 1);
    if aligned_prefix != 0 {
        queue.write_buffer(destination, 0, &codestream[..aligned_prefix]);
    }

    let remainder = &codestream[aligned_prefix..];
    if !remainder.is_empty() {
        let mut tail = [0u8; wgpu::COPY_BUFFER_ALIGNMENT as usize];
        tail[..remainder.len()].copy_from_slice(remainder);
        queue.write_buffer(
            destination,
            u64::try_from(aligned_prefix)
                .map_err(|_| Error::backend("codestream upload offset overflow"))?,
            &tail,
        );
    }

    let sentinel_offset = align4(
        u64::try_from(codestream.len())
            .map_err(|_| Error::backend("codestream sentinel offset overflow"))?,
    )?;
    queue.write_buffer(
        destination,
        sentinel_offset,
        &[0u8; wgpu::COPY_BUFFER_ALIGNMENT as usize],
    );
    Ok(())
}

fn align4(value: u64) -> Result<u64> {
    value
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or_else(|| Error::backend("GPU buffer size overflow"))
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::backend(
            "wgpu reported a non-power-of-two buffer offset alignment",
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| Error::backend("GPU buffer alignment overflow"))
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
    use jxl_gpu_formats::vpi::VpiPitchLinearFormat as Vpi;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::Wake;

    struct ReentrantCompletionWake {
        completion: Arc<MapCompletion>,
        entered: AtomicBool,
    }

    impl Wake for ReentrantCompletionWake {
        fn wake(self: Arc<Self>) {
            self.entered.store(true, Ordering::SeqCst);
            let waker = Waker::from(Arc::clone(&self));
            let context = Context::from_waker(&waker);
            assert_eq!(self.completion.poll(&context), Some(Ok(())));
        }
    }

    const PORTABLE_CAPABILITIES: WgpuDecodeCapabilities = WgpuDecodeCapabilities {
        native_f64_arithmetic: false,
    };

    #[test]
    fn output_negotiation_rejects_rgb_without_explicit_transfer_and_range() {
        let format = PixelFormat::rgb8(
            jxl_gpu_formats::RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Undefined,
        );
        let request = GpuOutputRequest::color(format).unwrap();
        assert!(matches!(
            OutputPlan::new(
                Extent2d::new(2, 2),
                &request,
                crate::ModularChannels::Gray,
                8,
                PORTABLE_CAPABILITIES,
            ),
            Err(Error::UnsupportedOutputFormat(_))
        ));
    }

    #[test]
    fn output_negotiation_rejects_shader_address_overflow() {
        let request = GpuOutputRequest::color(Vpi::Rgba8.pixel_format()).unwrap();
        assert!(matches!(
            OutputPlan::new(
                Extent2d::new(u32::MAX, 1),
                &request,
                crate::ModularChannels::Gray,
                8,
                PORTABLE_CAPABILITIES,
            ),
            Err(Error::Backend(_))
        ));
    }

    #[test]
    fn output_negotiation_covers_all_vpi_pitch_linear_formats() {
        let color_formats = [
            (Vpi::Y8, OutputKind::Luma, 1, 0, 8, 1, true, 1),
            (Vpi::Y8Er, OutputKind::Luma, 1, 0, 8, 1, false, 1),
            (Vpi::Y16, OutputKind::Luma, 1, 0, 16, 1, true, 1),
            (Vpi::Y16Er, OutputKind::Luma, 1, 0, 16, 1, false, 1),
            (Vpi::Nv12, OutputKind::YuvSemiplanar, 3, 0, 8, 2, true, 1),
            (Vpi::Nv12Er, OutputKind::YuvSemiplanar, 3, 0, 8, 2, false, 1),
            (Vpi::Nv24, OutputKind::YuvSemiplanar, 3, 0, 8, 2, true, 1),
            (Vpi::Nv24Er, OutputKind::YuvSemiplanar, 3, 0, 8, 2, false, 1),
            (Vpi::Uyvy, OutputKind::Yuv422Packed, 3, 1, 8, 1, true, 1),
            (Vpi::UyvyEr, OutputKind::Yuv422Packed, 3, 1, 8, 1, false, 1),
            (Vpi::Yuyv, OutputKind::Yuv422Packed, 3, 0, 8, 1, true, 1),
            (Vpi::YuyvEr, OutputKind::Yuv422Packed, 3, 0, 8, 1, false, 1),
            (Vpi::Rgb8, OutputKind::RgbInterleaved, 3, 0, 8, 1, false, 2),
            (Vpi::Bgr8, OutputKind::RgbInterleaved, 3, 1, 8, 1, false, 2),
            (Vpi::Rgba8, OutputKind::RgbInterleaved, 4, 2, 8, 1, false, 2),
            (Vpi::Bgra8, OutputKind::RgbInterleaved, 4, 3, 8, 1, false, 2),
            (Vpi::Rgb8Planar, OutputKind::RgbPlanar, 3, 0, 8, 3, false, 2),
            (Vpi::Bgr8Planar, OutputKind::RgbPlanar, 3, 1, 8, 3, false, 2),
            (
                Vpi::Rgba8Planar,
                OutputKind::RgbPlanar,
                4,
                2,
                8,
                4,
                false,
                2,
            ),
            (
                Vpi::Bgra8Planar,
                OutputKind::RgbPlanar,
                4,
                3,
                8,
                4,
                false,
                2,
            ),
        ];
        assert_eq!(color_formats.len(), 20);
        for (format, kind, channels, order, bits, planes, limited, transfer) in color_formats {
            let pixel_format = format.pixel_format();
            assert!(matches!(
                classify_pixel_format(&pixel_format),
                Ok(PixelFormatClass::Color(_))
            ));
            let request = GpuOutputRequest::color(pixel_format).unwrap();
            let output = OutputPlan::new(
                Extent2d::new(5, 3),
                &request,
                crate::ModularChannels::Gray,
                8,
                PORTABLE_CAPABILITIES,
            )
            .unwrap_or_else(|error| panic!("{} must be supported: {error}", format.name()));
            assert_eq!(output.kind, kind, "{} kind", format.name());
            assert_eq!(output.channels, channels, "{} channels", format.name());
            assert_eq!(output.order, order, "{} order", format.name());
            assert_eq!(output.bits, bits, "{} bits", format.name());
            assert_eq!(output.storage_bits, bits, "{} storage bits", format.name());
            assert_eq!(
                output.layout.planes.len(),
                planes,
                "{} planes",
                format.name()
            );
            assert_eq!(output.limited_range, limited, "{} range", format.name());
            assert_eq!(output.transfer, transfer, "{} transfer", format.name());
            assert!(output.layout.logical_size <= u64::from(u32::MAX));
        }

        let numeric_formats = [
            Vpi::U8,
            Vpi::S8,
            Vpi::U16,
            Vpi::U32,
            Vpi::S32,
            Vpi::S16,
            Vpi::TwoS16,
            Vpi::F32,
            Vpi::F64,
            Vpi::TwoF32,
        ];
        assert_eq!(numeric_formats.len(), 10);
        for format in numeric_formats {
            let mapping = if format == Vpi::F64 {
                NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening)
            } else {
                NumericSampleMapping::NormalizedGray8
            };
            let request = GpuOutputRequest::numeric(format.pixel_format(), mapping).unwrap();
            let output = OutputPlan::new(
                Extent2d::new(5, 3),
                &request,
                crate::ModularChannels::Gray,
                8,
                PORTABLE_CAPABILITIES,
            )
            .unwrap_or_else(|error| panic!("{} must be supported: {error}", format.name()));
            let numeric = classify_pixel_format(request.format())
                .unwrap()
                .numeric()
                .unwrap();
            assert_eq!(
                output.kind,
                match numeric.sample_kind {
                    SampleKind::Unsigned => OutputKind::NumericUnsigned,
                    SampleKind::Signed => OutputKind::NumericSigned,
                    SampleKind::Float => OutputKind::NumericFloat,
                },
                "{} kind",
                format.name()
            );
            assert_eq!(output.channels, u32::from(numeric.components));
            assert_eq!(output.bits, u32::from(numeric.bits_per_component));
            assert_eq!(output.numeric_mapping, 1);
            assert_eq!(output.layout.planes.len(), 1);
            if format == Vpi::F64 {
                let plane = &output.layout.planes[0];
                assert_eq!(output.layout.logical_size, 5 * 3 * 8);
                assert!(plane.offset.is_multiple_of(8));
                assert!(plane.row_stride.is_multiple_of(8));
                assert_eq!(
                    output.f64_output_path,
                    Some(F64OutputPath::ExactF32Widening)
                );
            }
        }
    }

    #[test]
    fn output_request_requires_mapping_to_match_the_format_class() {
        assert!(matches!(
            GpuOutputRequest::color(Vpi::U8.pixel_format()),
            Err(Error::NumericMappingRequired)
        ));
        assert!(matches!(
            GpuOutputRequest::numeric(
                Vpi::Rgba8.pixel_format(),
                NumericSampleMapping::NormalizedGray8,
            ),
            Err(Error::NumericMappingForColorOutput)
        ));
        assert!(matches!(
            GpuOutputRequest::numeric(
                Vpi::F64.pixel_format(),
                NumericSampleMapping::NormalizedGray8,
            ),
            Err(Error::F64OutputPolicyRequired)
        ));
        assert!(matches!(
            GpuOutputRequest::numeric(
                Vpi::U8.pixel_format(),
                NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::NativeRequired),
            ),
            Err(Error::F64OutputPolicyForNonF64)
        ));
    }

    #[test]
    fn f64_policy_resolution_never_silently_downgrades_native_required() {
        assert!(matches!(
            resolve_f64_output_path(F64OutputPolicy::NativeRequired, PORTABLE_CAPABILITIES),
            Err(Error::NativeF64Unavailable)
        ));
        assert_eq!(
            resolve_f64_output_path(
                F64OutputPolicy::NativeOrExactF32Widening,
                PORTABLE_CAPABILITIES,
            )
            .unwrap(),
            F64OutputPath::ExactF32Widening
        );
        let native = WgpuDecodeCapabilities {
            native_f64_arithmetic: true,
        };
        assert_eq!(
            resolve_f64_output_path(F64OutputPolicy::NativeRequired, native).unwrap(),
            F64OutputPath::NativeArithmetic
        );
        assert_eq!(
            resolve_f64_output_path(F64OutputPolicy::NativeOrExactF32Widening, native).unwrap(),
            F64OutputPath::NativeArithmetic
        );
        assert_eq!(
            resolve_f64_output_path(F64OutputPolicy::ExactF32Widening, native).unwrap(),
            F64OutputPath::ExactF32Widening
        );
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
    fn map_completion_wakes_after_releasing_state_lock() {
        let completion = Arc::new(MapCompletion::default());
        let wake = Arc::new(ReentrantCompletionWake {
            completion: Arc::clone(&completion),
            entered: AtomicBool::new(false),
        });
        let waker = Waker::from(Arc::clone(&wake));
        let context = Context::from_waker(&waker);
        assert_eq!(completion.poll(&context), None);

        completion.complete(Ok(()));
        assert!(wake.entered.load(Ordering::SeqCst));
    }

    #[test]
    fn shader_abi_and_stream_sentinel_are_explicit() {
        assert_eq!(std::mem::size_of::<ShaderParams>(), 128);
        assert_eq!(std::mem::align_of::<ShaderParams>(), 4);
        let params = ShaderParams {
            token_start: 1,
            token_end: 2,
            width: 3,
            height: 4,
            origin_x: 5,
            origin_y: 6,
            sample_count: 7,
            initialize_chroma: 8,
            source_channels: 9,
            source_bits: 10,
            source_mask: 11,
            _source_padding: 12,
            output_kind: 13,
            transfer: 14,
            limited_range: 15,
            channels: 16,
            order: 17,
            bits: 18,
            storage_bits: 19,
            plane0_offset: 20,
            plane0_stride: 21,
            plane1_offset: 22,
            plane1_stride: 23,
            plane2_offset: 24,
            plane2_stride: 25,
            plane3_offset: 26,
            plane3_stride: 27,
            chroma_width: 28,
            chroma_height: 29,
            logical_size: 30,
            numeric_mapping: 31,
            _padding: 32,
        };
        assert_eq!(
            bytemuck::cast::<ShaderParams, [u32; 32]>(params),
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ]
        );
        assert!(SHADER_TEMPLATE.contains(
            "struct Params {\n    token_start: u32,\n    token_end: u32,\n    width: u32,\n    height: u32,\n    origin_x: u32,\n    origin_y: u32,\n    sample_count: u32,\n    initialize_chroma: u32,\n    source_channels: u32,\n    source_bits: u32,\n    source_mask: u32,\n    _source_padding: u32,\n    output_kind: u32,\n    transfer: u32,\n    limited_range: u32,\n    channels: u32,\n    order: u32,\n    bits: u32,\n    storage_bits: u32,\n    plane0_offset: u32,\n    plane0_stride: u32,\n    plane1_offset: u32,\n    plane1_stride: u32,\n    plane2_offset: u32,\n    plane2_stride: u32,\n    plane3_offset: u32,\n    plane3_stride: u32,\n    chroma_width: u32,\n    chroma_height: u32,\n    logical_size: u32,\n    numeric_mapping: u32,\n    _padding: u32,\n};"
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
        assert!(SHADER_TEMPLATE.contains("status[0] = STATUS_OK;"));
        assert!(SHADER_TEMPLATE.contains("status[1] = decoded;"));
        assert!(SHADER_TEMPLATE.contains("status[2] = bit_cursor;"));
        assert!(SHADER_TEMPLATE.contains("status[3] = params.token_end;"));
        assert_eq!(stream_allocation_size(4).unwrap(), 8);
        assert_eq!(stream_allocation_size(5).unwrap(), 12);
    }

    #[test]
    fn portable_and_native_f64_shader_sources_validate_with_exact_capabilities() {
        let portable = shader_source(F64OutputPath::ExactF32Widening);
        let native = shader_source(F64OutputPath::NativeArithmetic);
        assert!(!portable.contains(F64_OUTPUT_MARKER));
        assert!(!native.contains(F64_OUTPUT_MARKER));
        assert!(!portable.contains(F64_BINDING_MARKER));
        assert!(!native.contains(F64_BINDING_MARKER));
        assert!(!portable.contains("f64(sample)"));
        assert!(native.contains("f64(sample) / 255.0"));
        assert!(native.contains("output_f64: array<f64>"));

        let native_without_capability = naga::front::wgsl::parse_str(&native)
            .expect("native F64 WGSL syntax must parse before capability validation");
        let error = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&native_without_capability)
        .expect_err("native F64 WGSL must be rejected without Naga FLOAT64 capability");
        assert!(format!("{error:?}").contains("FLOAT64"));

        for (name, source, capabilities) in [
            ("portable", portable, naga::valid::Capabilities::empty()),
            ("native-f64", native, naga::valid::Capabilities::FLOAT64),
        ] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name} WGSL did not parse: {error}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} WGSL did not validate: {error}"));
        }
    }

    #[test]
    fn aggregate_memory_reservations_are_bounded_and_released() {
        let budget = MemoryBudget::new(NonZeroU64::new(10).unwrap());
        let first = budget.try_reserve(6).unwrap();
        assert_eq!(budget.snapshot().reserved_bytes, 6);
        assert!(matches!(
            budget.try_reserve(5),
            Err(jxl_wgpu::MemoryBudgetError::Exhausted { .. })
        ));
        assert_eq!(budget.snapshot().reserved_bytes, 6);
        drop(first);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}
