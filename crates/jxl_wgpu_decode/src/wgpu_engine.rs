use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::task::{Context, Poll, Waker};

use jxl_gpu_bitstream::{InventoryLimits, ParseLimits};
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
use crate::modular_tree::MaTreeNodeIr;
use crate::profile::{ModularGroup, StandardModularProfile, parse_standard_modular_profile};
use crate::{
    AnimationMetadata, DecodeProfile, Error, F64OutputPolicy, FrameDuration, FrameMetadata,
    GpuCodestream, GpuDecoder, GpuOutputMapping, GpuOutputRequest, GpuPendingFrame,
    GpuSubmissionEngine, GpuSubmissionSession, ModularPredictionProfile, ModularPredictor,
    NumericSampleMapping, PreparedGpuSession, Result, SubmittedGpuFrame,
};

const SHADER_TEMPLATE: &str = include_str!("lossless_gray8.wgsl");
const MODULAR_ENTROPY_SHADER: &str = include_str!("modular_entropy.wgsl");
const MODULAR_RECONSTRUCT_SHADER: &str = include_str!("modular_reconstruct.wgsl");
const MODULAR_FIXED_GRADIENT_SHADER: &str = include_str!("modular_fixed_gradient.wgsl");
const MODULAR_ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
const MODULAR_RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";
const F64_OUTPUT_MARKER: &str = "/*__JXL_F64_OUTPUT__*/";
const F64_BINDING_MARKER: &str = "/*__JXL_F64_BINDING__*/";
const WORKGROUP_SIZE_MARKER: &str = "/*__JXL_WORKGROUP_SIZE__*/";
const OUTPUT_WORDS_TYPE_MARKER: &str = "/*__JXL_OUTPUT_WORDS_TYPE__*/";
const WRITE_BYTE_WORD_MARKER: &str = "/*__JXL_WRITE_BYTE_WORD__*/";
const WRITE_FULL_WORD_MARKER: &str = "/*__JXL_WRITE_FULL_WORD__*/";
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
const ATOMIC_OUTPUT_WORDS_TYPE: &str = "array<atomic<u32>>";
const WORD_ALIGNED_OUTPUT_WORDS_TYPE: &str = "array<u32>";
const ATOMIC_WRITE_BYTE_WORD: &str = r#"
    var previous = atomicLoad(&output_words[word_index]);
    loop {
        let updated = (previous & ~mask) | ((value & 0xffu) << shift);
        let exchange = atomicCompareExchangeWeak(&output_words[word_index], previous, updated);
        if exchange.exchanged {
            break;
        }
        previous = exchange.old_value;
    }
"#;
const WORD_ALIGNED_WRITE_BYTE_WORD: &str = r#"
    let previous = output_words[word_index];
    output_words[word_index] = (previous & ~mask) | ((value & 0xffu) << shift);
"#;
const ATOMIC_WRITE_FULL_WORD: &str = "atomicStore(&output_words[offset >> 2u], value);";
const WORD_ALIGNED_WRITE_FULL_WORD: &str = "output_words[offset >> 2u] = value;";
const STATUS_OK: u32 = 1;
const STREAM_SENTINEL_BYTES: u64 = 4;
const NATIVE_F64_DUMMY_WORD_BYTES: u64 = 4;
const MODULAR_GROUP_WORKGROUP_SIZE: u32 = 64;
// Each lane is one serial reconstruction invocation which may process a full 256x256 group. Keep
// a finite watchdog-oriented ceiling even when the adapter and shared byte budget allow more.
const WATCHDOG_PARALLEL_GROUP_LANE_CAP: usize = 512;

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
    /// Peak reusable codestream window selected from the byte budget and device binding limit.
    pub stream_window_bytes: u64,
    /// All parallel reconstruction lanes in the reusable scratch allocation.
    pub reconstruction_scratch_bytes: u64,
    /// Byte stride of one reconstruction lane.
    pub reconstruction_lane_stride_bytes: u64,
    /// Largest descriptor-derived LZ history ring used by one group lane.
    pub max_lz77_window_words: u32,
    /// Largest physical LZ history ring stored in one reconstruction lane.
    ///
    /// A logical one-word ring uses invocation-private state and therefore reports zero here.
    pub max_lz77_scratch_words: u32,
    /// Bounded stream uploads required for one frame. Each batch is one ordered queue submission.
    pub stream_batch_count: usize,
    /// Actual codec queue submissions per decoded frame.
    pub submissions_per_frame: usize,
    /// Scratch-isolated Modular groups decoded concurrently by one compute dispatch.
    pub parallel_group_lanes: usize,
    /// Logical group invocations packed into one portable compute workgroup.
    pub group_workgroup_size: u32,
    /// Largest compute-workgroup count submitted by one bounded stream batch.
    pub max_dispatch_workgroups: u32,
    /// Output write implementation selected after proving the complete group/plane layout.
    pub output_write_path: OutputWritePath,
    /// Reconstruction kernel selected from the fully validated MA-tree IR.
    pub reconstruction_specialization: ModularReconstructionSpecialization,
}

/// GPU output update strategy selected for a validated frame layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputWritePath {
    /// Byte updates use atomics because distinct group rows may share a storage word.
    AtomicBytes,
    /// Every plane row and internal group boundary is word-aligned, allowing ordinary RMW/store.
    WordAligned,
}

/// Typed Modular reconstruction specialization selected for one decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularReconstructionSpecialization {
    /// The complete MA tree and predictor family are evaluated per sample.
    GenericMetaAdaptive,
    /// Every channel resolves through channel-only decisions to one fixed leaf.
    ChannelFixed {
        predictor: ModularPredictor,
        offset: i32,
        multiplier: u32,
        channel_count: u8,
        clusters: [u8; 4],
    },
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
    needs_self_correcting: u32,
    lz77_window_mask: u32,
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
    status_index: u32,
    stream_index: u32,
    fixed_leaf_predictor: u32,
    fixed_leaf_offset: u32,
    fixed_leaf_multiplier: u32,
    fixed_leaf_cluster0: u32,
    fixed_leaf_cluster1: u32,
    fixed_leaf_cluster2: u32,
    fixed_leaf_cluster3: u32,
    wp_p1: u32,
    wp_p2: u32,
    wp_p3a: u32,
    wp_p3b: u32,
    wp_p3c: u32,
    wp_p3d: u32,
    wp_p3e: u32,
    wp_w0: u32,
    wp_w1: u32,
    wp_w2: u32,
    wp_w3: u32,
}

/// CPU/WGSL ABI selecting one bounded parallel group wave.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchControl {
    first_group: u32,
    group_count: u32,
    lane_stride_words: u32,
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
    assert!(std::mem::size_of::<ShaderParams>() == 208);
    assert!(std::mem::align_of::<ShaderParams>() == 4);
    assert!(std::mem::size_of::<DecodeStatus>() == 16);
    assert!(std::mem::align_of::<DecodeStatus>() == 4);
};

/// Stock GPU-only decoder for the standard lossless 1-16-bit Gray/RGB/RGBA Modular profile.
///
/// The frontend inventories standard frame sections and parses only bounded entropy and MA-tree
/// metadata. The shader reads Prefix or ANS tokens from the actual `jxlc` bytes, applies LZ77,
/// reconstructs the selected Modular predictors, and writes the requested GPU image layout. A
/// host-proven channel-fixed Gradient tree uses a specialized kernel; all other accepted trees use
/// the generic MA-tree kernel. No private index, CPU pixel decoder, or CPU entropy fallback is
/// required.
#[derive(Clone)]
pub struct WgpuSubmissionEngine {
    backend: WgpuBackend,
    pipelines: Arc<DecodePipelineCache>,
    native_f64_pipelines: Option<Arc<DecodePipelineCache>>,
    memory: MemoryBudget,
    buffers: Arc<DecodeBufferPool>,
}

#[derive(Default)]
struct DecodePipelineCache {
    generic_atomic: OnceLock<Arc<wgpu::ComputePipeline>>,
    generic_word_aligned: OnceLock<Arc<wgpu::ComputePipeline>>,
    fixed_gradient_atomic: OnceLock<Arc<wgpu::ComputePipeline>>,
    fixed_gradient_word_aligned: OnceLock<Arc<wgpu::ComputePipeline>>,
}

fn reconstruction_specialization(
    profile: &StandardModularProfile,
) -> ModularReconstructionSpecialization {
    channel_fixed_gradient_specialization(
        &profile.ma_config.nodes,
        profile.channels.count(),
        profile.ma_config.needs_self_correcting(),
    )
}

fn channel_fixed_gradient_specialization(
    nodes: &[MaTreeNodeIr],
    source_channels: u32,
    needs_self_correcting: bool,
) -> ModularReconstructionSpecialization {
    let generic = ModularReconstructionSpecialization::GenericMetaAdaptive;
    let Ok(channel_count) = u8::try_from(source_channels) else {
        return generic;
    };
    if !(1..=4).contains(&channel_count)
        || needs_self_correcting
        || nodes.is_empty()
        || nodes.iter().any(|node| {
            matches!(
                node,
                MaTreeNodeIr::Decision {
                    property,
                    ..
                } if *property != 0
            )
        })
    {
        return generic;
    }
    let mut clusters = [0u8; 4];
    for (channel, cluster) in [0i32, 1, 2, 3].into_iter().zip(clusters.iter_mut()) {
        let mut node_index = 0u32;
        let mut leaf = None;
        for _ in 0..nodes.len() {
            let Some(node) = usize::try_from(node_index)
                .ok()
                .and_then(|index| nodes.get(index))
            else {
                break;
            };
            match *node {
                MaTreeNodeIr::Decision {
                    property: 0,
                    threshold,
                    left,
                    right,
                } => {
                    node_index = if channel > threshold { left } else { right };
                }
                MaTreeNodeIr::Leaf {
                    cluster,
                    predictor: 5,
                    offset: 0,
                    multiplier: 1,
                } => {
                    leaf = Some(cluster);
                    break;
                }
                _ => break,
            }
        }
        let Some(resolved) = leaf else {
            return generic;
        };
        *cluster = resolved;
    }
    ModularReconstructionSpecialization::ChannelFixed {
        predictor: ModularPredictor::Gradient,
        offset: 0,
        multiplier: 1,
        channel_count,
        clusters,
    }
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
        // Pipelines are selected after the output layout is validated. Lazy caches avoid
        // compiling the atomic fallback or native-f64 variants when a workload never needs them.
        let pipelines = Arc::new(DecodePipelineCache::default());
        let native_f64_pipelines = backend
            .native_f64_enabled()
            .then(|| Arc::new(DecodePipelineCache::default()));
        let buffers = DecodeBufferPool::new(
            backend.device().clone(),
            WgpuDecodeBufferPoolLimits::default(),
        );
        Self {
            backend,
            pipelines,
            native_f64_pipelines,
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
            native_f64_arithmetic: self.native_f64_pipelines.is_some(),
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

impl DecodePipelineCache {
    fn get_or_init(
        &self,
        backend: &WgpuBackend,
        f64_path: F64OutputPath,
        output_write_path: OutputWritePath,
        reconstruction: ModularReconstructionSpecialization,
    ) -> Arc<wgpu::ComputePipeline> {
        let fixed_gradient = matches!(
            reconstruction,
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                ..
            }
        );
        let pipeline = match (fixed_gradient, output_write_path) {
            (false, OutputWritePath::AtomicBytes) => &self.generic_atomic,
            (false, OutputWritePath::WordAligned) => &self.generic_word_aligned,
            (true, OutputWritePath::AtomicBytes) => &self.fixed_gradient_atomic,
            (true, OutputWritePath::WordAligned) => &self.fixed_gradient_word_aligned,
        };
        Arc::clone(pipeline.get_or_init(|| {
            let label = match (f64_path, output_write_path, fixed_gradient) {
                (F64OutputPath::ExactF32Widening, OutputWritePath::AtomicBytes, false) => {
                    "jxl-wgpu decode generic Modular atomic output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::WordAligned, false) => {
                    "jxl-wgpu decode generic Modular word-aligned output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::AtomicBytes, false) => {
                    "jxl-wgpu decode generic Modular native-f64 atomic output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::WordAligned, false) => {
                    "jxl-wgpu decode generic Modular native-f64 word-aligned output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::AtomicBytes, true) => {
                    "jxl-wgpu decode fixed-Gradient Modular atomic output"
                }
                (F64OutputPath::ExactF32Widening, OutputWritePath::WordAligned, true) => {
                    "jxl-wgpu decode fixed-Gradient Modular word-aligned output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::AtomicBytes, true) => {
                    "jxl-wgpu decode fixed-Gradient Modular native-f64 atomic output"
                }
                (F64OutputPath::NativeArithmetic, OutputWritePath::WordAligned, true) => {
                    "jxl-wgpu decode fixed-Gradient Modular native-f64 word-aligned output"
                }
            };
            Arc::new(create_decode_pipeline(
                backend,
                label,
                &shader_source(f64_path, output_write_path, reconstruction),
            ))
        }))
    }
}

fn shader_source(
    path: F64OutputPath,
    output_write_path: OutputWritePath,
    reconstruction: ModularReconstructionSpecialization,
) -> String {
    let (implementation, binding) = match path {
        F64OutputPath::NativeArithmetic => (F64_NATIVE_ARITHMETIC, F64_NATIVE_BINDING),
        F64OutputPath::ExactF32Widening => (F64_EXACT_F32_WIDENING, ""),
    };
    let (output_words_type, write_byte_word, write_full_word) = match output_write_path {
        OutputWritePath::AtomicBytes => (
            ATOMIC_OUTPUT_WORDS_TYPE,
            ATOMIC_WRITE_BYTE_WORD,
            ATOMIC_WRITE_FULL_WORD,
        ),
        OutputWritePath::WordAligned => (
            WORD_ALIGNED_OUTPUT_WORDS_TYPE,
            WORD_ALIGNED_WRITE_BYTE_WORD,
            WORD_ALIGNED_WRITE_FULL_WORD,
        ),
    };
    let reconstruction_shader = match reconstruction {
        ModularReconstructionSpecialization::GenericMetaAdaptive => MODULAR_RECONSTRUCT_SHADER,
        ModularReconstructionSpecialization::ChannelFixed {
            predictor: ModularPredictor::Gradient,
            ..
        } => MODULAR_FIXED_GRADIENT_SHADER,
        ModularReconstructionSpecialization::ChannelFixed { .. } => MODULAR_RECONSTRUCT_SHADER,
    };
    let source = SHADER_TEMPLATE
        .replace(MODULAR_ENTROPY_MARKER, MODULAR_ENTROPY_SHADER)
        .replace(MODULAR_RECONSTRUCT_MARKER, reconstruction_shader)
        .replace(F64_OUTPUT_MARKER, implementation)
        .replace(F64_BINDING_MARKER, binding)
        .replace(OUTPUT_WORDS_TYPE_MARKER, output_words_type)
        .replace(WRITE_BYTE_WORD_MARKER, write_byte_word)
        .replace(WRITE_FULL_WORD_MARKER, write_full_word)
        .replace(
            WORKGROUP_SIZE_MARKER,
            &MODULAR_GROUP_WORKGROUP_SIZE.to_string(),
        );
    debug_assert!(!source.contains(MODULAR_ENTROPY_MARKER));
    debug_assert!(!source.contains(MODULAR_RECONSTRUCT_MARKER));
    debug_assert!(!source.contains(F64_OUTPUT_MARKER));
    debug_assert!(!source.contains(F64_BINDING_MARKER));
    debug_assert!(!source.contains(OUTPUT_WORDS_TYPE_MARKER));
    debug_assert!(!source.contains(WRITE_BYTE_WORD_MARKER));
    debug_assert!(!source.contains(WRITE_FULL_WORD_MARKER));
    debug_assert!(!source.contains(WORKGROUP_SIZE_MARKER));
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
        let policy_ceiling = ParseLimits::default().max_codestream_bytes;
        let budget_scaled = self.memory.snapshot().limit_bytes.saturating_mul(4);
        let device_scaled = self
            .backend
            .device()
            .limits()
            .max_buffer_size
            .saturating_mul(4);
        let host_codestream_limit = policy_ceiling.min(budget_scaled.max(device_scaled));
        ParseLimits {
            max_input_bytes: host_codestream_limit,
            max_boxes: 32,
            max_box_bytes: host_codestream_limit,
            max_codestream_bytes: host_codestream_limit,
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
        let modular_metadata: Arc<[u32]> = profile.ma_config.pack_gpu_metadata()?.words.into();
        let extent = Extent2d::new(profile.width, profile.height);
        let output = OutputPlan::new(
            extent,
            request,
            profile.channels,
            profile.bits_per_sample,
            self.capabilities(),
        )?;
        let output_write_path = output.write_path_for_groups(&profile.groups)?;
        let reconstruction_specialization = reconstruction_specialization(&profile);
        let (pipelines, pipeline_f64_path) = match output.f64_output_path {
            Some(F64OutputPath::NativeArithmetic) => (
                self.native_f64_pipelines
                    .as_ref()
                    .ok_or(Error::NativeF64Unavailable)?,
                F64OutputPath::NativeArithmetic,
            ),
            Some(F64OutputPath::ExactF32Widening) | None => {
                (&self.pipelines, F64OutputPath::ExactF32Widening)
            }
        };
        let pipeline = pipelines.get_or_init(
            &self.backend,
            pipeline_f64_path,
            output_write_path,
            reconstruction_specialization,
        );
        let f64_output_path = output.f64_output_path;
        let memory_limit_bytes = self.memory.snapshot().limit_bytes;
        let dispatch_layout = GroupDispatchLayout::new(
            self.backend.device(),
            codestream.bytes(),
            &profile,
            &modular_metadata,
            &output,
            request.max_frame_slots().get(),
            memory_limit_bytes,
        )?;
        let memory_stats = validate_device_limits(
            self.backend.device(),
            &modular_metadata,
            &dispatch_layout,
            &output,
            request.max_frame_slots().get(),
            memory_limit_bytes,
        )?;
        let resolved_frame_slots = NonZeroUsize::new(memory_stats.max_frame_slots)
            .expect("device admission always resolves at least one frame slot");
        let node_count = u32::try_from(profile.ma_config.nodes.len())
            .map_err(|_| Error::backend("MA tree node count exceeds public profile bounds"))?;
        let decision_node_count = u32::try_from(
            profile
                .ma_config
                .nodes
                .iter()
                .filter(|node| matches!(node, MaTreeNodeIr::Decision { .. }))
                .count(),
        )
        .map_err(|_| Error::backend("MA decision node count exceeds public profile bounds"))?;
        let leaf_context_count = node_count
            .checked_sub(decision_node_count)
            .ok_or_else(|| Error::backend("MA leaf context count underflow"))?;
        let max_depth = u32::try_from(profile.ma_config.max_depth)
            .map_err(|_| Error::backend("MA tree depth exceeds public profile bounds"))?;
        let prediction = ModularPredictionProfile::MetaAdaptive {
            node_count,
            decision_node_count,
            leaf_context_count,
            max_depth,
            uses_self_correcting: profile.ma_config.needs_self_correcting(),
        };
        Ok(PreparedGpuSession::new(
            DecodeProfile::ModularLossless {
                bits_per_sample: profile.bits_per_sample,
                channels: profile.channels,
                prediction,
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
                    modular_metadata,
                    output,
                }),
                memory_stats,
                memory_budget: self.memory.clone(),
                buffers: Arc::clone(&self.buffers),
                f64_output_path,
            },
        )
        .with_resolved_frame_slots(resolved_frame_slots))
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
    // DC-global MA/entropy descriptor without sharing mutable GPU transient allocations.
    modular_metadata: Arc<[u32]>,
    output: OutputPlan,
}

struct DecodeJobLifetime {
    output: Arc<wgpu::Buffer>,
    _modular_metadata: DecodeBufferLease,
    _reconstructed: DecodeBufferLease,
    _native_f64_dummy_words: Option<DecodeBufferLease>,
    _status: DecodeBufferLease,
    status_staging: DecodeBufferLease,
    status_mapped: AtomicBool,
    _params: DecodeBufferLease,
    _dispatch_control: DecodeBufferLease,
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
    reconstruction_lane_stride: u64,
    max_lz77_window_words: u32,
    max_lz77_scratch_words: u32,
    parallel_group_lanes: usize,
    reconstructed_bytes: u64,
    stream_windows: Arc<[GroupStreamWindow]>,
    stream_batches: Arc<[std::ops::Range<usize>]>,
    stream_bytes: u64,
    status_stride: u64,
    status_bytes: u64,
    params_stride: u64,
    params_bytes: u64,
    output_write_path: OutputWritePath,
    reconstruction_specialization: ModularReconstructionSpecialization,
}

#[derive(Clone, Copy, Debug)]
struct GroupStreamWindow {
    input_start: usize,
    input_end: usize,
    upload_offset: usize,
    token_start: u32,
    token_end: u32,
}

impl GroupDispatchLayout {
    fn new(
        device: &wgpu::Device,
        codestream: &[u8],
        profile: &StandardModularProfile,
        modular_metadata: &[u32],
        output: &OutputPlan,
        requested_frame_slots: usize,
        memory_limit_bytes: u64,
    ) -> Result<Self> {
        let output_write_path = output.write_path_for_groups(&profile.groups)?;
        let reconstruction_specialization = reconstruction_specialization(profile);
        let limits = device.limits();
        if limits.max_compute_invocations_per_workgroup < MODULAR_GROUP_WORKGROUP_SIZE
            || limits.max_compute_workgroup_size_x < MODULAR_GROUP_WORKGROUP_SIZE
        {
            return Err(Error::backend(format!(
                "device cannot run the portable {MODULAR_GROUP_WORKGROUP_SIZE}-invocation Modular workgroup"
            )));
        }
        let mut reconstruction_lane_stride = 0u64;
        let mut max_lz77_window_words = 0u32;
        let mut max_lz77_scratch_words = 0u32;
        for group in &profile.groups {
            let decoded_symbol_count = group_decoded_symbol_count(profile, *group)?;
            let lz77_window_words = group_lz77_window_words(profile, *group, decoded_symbol_count)?;
            max_lz77_window_words = max_lz77_window_words.max(lz77_window_words);
            max_lz77_scratch_words =
                max_lz77_scratch_words.max(lz77_scratch_words(lz77_window_words));
            reconstruction_lane_stride = reconstruction_lane_stride
                .max(align4(group_reconstructed_bytes(profile, *group)?)?);
        }
        if reconstruction_lane_stride == 0 {
            return Err(Error::backend("Modular reconstruction lane is empty"));
        }
        let stream_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let group_count = u64::try_from(profile.groups.len())
            .map_err(|_| Error::backend("Modular group count exceeds u64"))?;
        let status_stride = STATUS_BYTES;
        let status_bytes = status_stride
            .checked_mul(group_count)
            .ok_or_else(|| Error::backend("group status buffer size overflow"))?;
        let params_stride = std::mem::size_of::<ShaderParams>() as u64;
        let params_bytes = params_stride
            .checked_mul(group_count)
            .ok_or_else(|| Error::backend("group parameter buffer size overflow"))?;
        let fixed_bytes = [
            modular_metadata_bytes(modular_metadata)?,
            align4(output.layout.logical_size)?,
            if output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
                NATIVE_F64_DUMMY_WORD_BYTES
            } else {
                0
            },
            status_bytes,
            status_bytes,
            params_bytes,
            std::mem::size_of::<DispatchControl>() as u64,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| Error::backend("parallel Modular fixed memory size overflow"))?;
        let device_lane_cap = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / reconstruction_lane_stride;
        let device_lane_cap = usize::try_from(device_lane_cap).unwrap_or(usize::MAX);
        let workgroup_cap = u64::from(limits.max_compute_workgroups_per_dimension)
            .checked_mul(u64::from(MODULAR_GROUP_WORKGROUP_SIZE))
            .and_then(|lanes| usize::try_from(lanes).ok())
            .unwrap_or(usize::MAX);
        let lane_cap = WATCHDOG_PARALLEL_GROUP_LANE_CAP
            .min(profile.groups.len())
            .min(device_lane_cap)
            .min(workgroup_cap);
        if lane_cap == 0 {
            return Err(Error::backend(
                "device limits cannot bind one Modular reconstruction lane",
            ));
        }
        let requested_slots = u64::try_from(requested_frame_slots.max(1))
            .map_err(|_| Error::backend("requested frame-slot count exceeds u64"))?;
        let requested_target = memory_limit_bytes / requested_slots;
        let selected = match select_parallel_group_layout(
            codestream,
            &profile.groups,
            stream_limit,
            lane_cap,
            reconstruction_lane_stride,
            fixed_bytes,
            requested_target,
        )? {
            Some(selected) => Some(selected),
            None => {
            select_parallel_group_layout(
                codestream,
                &profile.groups,
                stream_limit,
                lane_cap,
                reconstruction_lane_stride,
                fixed_bytes,
                memory_limit_bytes,
            )
            ?
            }
        }
        .ok_or_else(|| {
            Error::backend(format!(
                "one bounded Modular lane plus fixed allocations exceeds the shared {memory_limit_bytes}-byte budget"
            ))
        })?;
        let (parallel_group_lanes, stream_windows, stream_batches, stream_bytes) = selected;
        let reconstructed_bytes = reconstruction_lane_stride
            .checked_mul(u64::try_from(parallel_group_lanes).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::backend("parallel Modular scratch size overflow"))?;
        Ok(Self {
            reconstruction_lane_stride,
            max_lz77_window_words,
            max_lz77_scratch_words,
            parallel_group_lanes,
            reconstructed_bytes,
            stream_windows: stream_windows.into(),
            stream_batches: stream_batches.into(),
            stream_bytes,
            status_stride,
            status_bytes,
            params_stride,
            params_bytes,
            output_write_path,
            reconstruction_specialization,
        })
    }
}

fn build_stream_batches(
    codestream: &[u8],
    groups: &[ModularGroup],
    stream_limit: u64,
    max_groups_per_batch: usize,
) -> Result<(Vec<GroupStreamWindow>, Vec<std::ops::Range<usize>>, u64)> {
    if max_groups_per_batch == 0 {
        return Err(Error::backend(
            "bounded Modular stream batch has zero group lanes",
        ));
    }
    if stream_limit < STREAM_SENTINEL_BYTES + 4 {
        return Err(Error::backend(
            "device storage limit is too small for a bounded Modular stream window",
        ));
    }
    let mut windows = Vec::with_capacity(groups.len());
    let mut batches = Vec::new();
    let mut batch_start = 0usize;
    let mut upload_cursor = 0u64;
    let mut maximum_batch_bytes = 0u64;
    for (index, group) in groups.iter().copied().enumerate() {
        let input_start = usize::try_from(group.token_bit_offset / 8)
            .map_err(|_| Error::backend("group stream start exceeds host address space"))?;
        let input_end = usize::try_from(
            group
                .token_bit_end
                .checked_add(7)
                .ok_or_else(|| Error::backend("group stream end overflow"))?
                / 8,
        )
        .map_err(|_| Error::backend("group stream end exceeds host address space"))?;
        let input = codestream
            .get(input_start..input_end)
            .ok_or_else(|| Error::backend("group stream window exceeds the codestream"))?;
        let packet_bytes = u64::try_from(input.len())
            .map_err(|_| Error::backend("group stream size exceeds u64"))?;
        let mut segment_start = align4(upload_cursor)?;
        let mut batch_bytes = segment_start
            .checked_add(packet_bytes)
            .and_then(|bytes| align4(bytes).ok())
            .and_then(|bytes| bytes.checked_add(STREAM_SENTINEL_BYTES))
            .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
        if index != batch_start
            && (batch_bytes > stream_limit || index - batch_start >= max_groups_per_batch)
        {
            batches.push(batch_start..index);
            maximum_batch_bytes = maximum_batch_bytes.max(
                align4(upload_cursor)?
                    .checked_add(STREAM_SENTINEL_BYTES)
                    .ok_or_else(|| Error::backend("group stream batch size overflow"))?,
            );
            batch_start = index;
            segment_start = 0;
            batch_bytes = align4(packet_bytes)?
                .checked_add(STREAM_SENTINEL_BYTES)
                .ok_or_else(|| Error::backend("group stream batch size overflow"))?;
        }
        if batch_bytes > stream_limit {
            return Err(Error::backend(format!(
                "one Modular group stream requires {batch_bytes} bytes, exceeding the bounded {stream_limit}-byte GPU window"
            )));
        }
        let segment_start_bits = segment_start
            .checked_mul(8)
            .ok_or_else(|| Error::backend("group stream bit offset overflow"))?;
        let leading_bits = group.token_bit_offset & 7;
        let token_start = segment_start_bits
            .checked_add(leading_bits)
            .and_then(|bits| u32::try_from(bits).ok())
            .ok_or_else(|| Error::backend("group stream start exceeds WGSL u32"))?;
        let token_end = u64::from(token_start)
            .checked_add(group.token_bit_end - group.token_bit_offset)
            .and_then(|bits| u32::try_from(bits).ok())
            .ok_or_else(|| Error::backend("group stream end exceeds WGSL u32"))?;
        windows.push(GroupStreamWindow {
            input_start,
            input_end,
            upload_offset: usize::try_from(segment_start)
                .map_err(|_| Error::backend("group upload offset exceeds host address space"))?,
            token_start,
            token_end,
        });
        upload_cursor = segment_start
            .checked_add(packet_bytes)
            .ok_or_else(|| Error::backend("group stream batch cursor overflow"))?;
    }
    if batch_start < groups.len() {
        batches.push(batch_start..groups.len());
        maximum_batch_bytes = maximum_batch_bytes.max(
            align4(upload_cursor)?
                .checked_add(STREAM_SENTINEL_BYTES)
                .ok_or_else(|| Error::backend("group stream batch size overflow"))?,
        );
    }
    if windows.len() != groups.len() || batches.is_empty() || maximum_batch_bytes == 0 {
        return Err(Error::backend("Modular stream batch layout is empty"));
    }
    Ok((windows, batches, maximum_batch_bytes))
}

type ParallelGroupLayout = (
    usize,
    Vec<GroupStreamWindow>,
    Vec<std::ops::Range<usize>>,
    u64,
);

fn select_parallel_group_layout(
    codestream: &[u8],
    groups: &[ModularGroup],
    stream_limit: u64,
    lane_cap: usize,
    lane_stride: u64,
    fixed_bytes: u64,
    per_frame_target: u64,
) -> Result<Option<ParallelGroupLayout>> {
    let available = match per_frame_target.checked_sub(fixed_bytes) {
        Some(available) => available,
        None => return Ok(None),
    };
    let budget_lane_cap = usize::try_from(available / lane_stride).unwrap_or(usize::MAX);
    let mut lanes = lane_cap.min(budget_lane_cap);
    while lanes != 0 {
        let (windows, batches, stream_bytes) =
            build_stream_batches(codestream, groups, stream_limit, lanes)?;
        let scratch_bytes = lane_stride
            .checked_mul(u64::try_from(lanes).unwrap_or(u64::MAX))
            .ok_or_else(|| Error::backend("parallel Modular scratch size overflow"))?;
        let required = fixed_bytes
            .checked_add(stream_bytes)
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .ok_or_else(|| Error::backend("parallel Modular memory target overflow"))?;
        if required <= per_frame_target {
            return Ok(Some((lanes, windows, batches, stream_bytes)));
        }
        lanes -= 1;
    }
    Ok(None)
}

fn group_reconstructed_bytes(profile: &StandardModularProfile, group: ModularGroup) -> Result<u64> {
    let sample_words = group_decoded_symbol_count(profile, group)?;
    let predictor_words = if profile.ma_config.needs_self_correcting() {
        u64::from(group.width)
            .checked_mul(5)
            .ok_or_else(|| Error::backend("weighted predictor workspace overflow"))?
    } else {
        0
    };
    let entropy_words = u64::from(lz77_scratch_words(group_lz77_window_words(
        profile,
        group,
        sample_words,
    )?));
    u64::from(sample_words)
        .checked_add(predictor_words)
        .and_then(|words| words.checked_add(entropy_words))
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| Error::backend("group reconstruction workspace size overflow"))
}

fn group_decoded_symbol_count(
    profile: &StandardModularProfile,
    group: ModularGroup,
) -> Result<u32> {
    group
        .sample_count()?
        .checked_mul(profile.channels.count())
        .ok_or_else(|| Error::backend("group reconstruction sample count overflow"))
}

fn group_lz77_window_words(
    profile: &StandardModularProfile,
    group: ModularGroup,
    decoded_symbol_count: u32,
) -> Result<u32> {
    profile
        .ma_config
        .entropy
        .lz77_window_words(group.width, decoded_symbol_count)
}

const fn lz77_scratch_words(window_words: u32) -> u32 {
    if window_words <= 1 { 0 } else { window_words }
}

fn modular_metadata_bytes(metadata: &[u32]) -> Result<u64> {
    u64::try_from(metadata.len())
        .ok()
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| Error::backend("Modular metadata size overflow"))
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

    fn write_path_for_groups(&self, groups: &[ModularGroup]) -> Result<OutputWritePath> {
        if groups.is_empty() {
            return Err(Error::backend("Modular output has no pass groups"));
        }
        if self
            .layout
            .planes
            .iter()
            .any(|plane| !plane.offset.is_multiple_of(4) || !plane.row_stride.is_multiple_of(4))
        {
            return Ok(OutputWritePath::AtomicBytes);
        }
        for &group in groups {
            if !self.group_row_span_is_word_isolated(group)? {
                return Ok(OutputWritePath::AtomicBytes);
            }
        }
        Ok(OutputWritePath::WordAligned)
    }

    /// Proves that one group's row writes cannot share a storage word with a horizontal neighbor.
    /// Plane offsets and strides are checked separately by [`Self::write_path_for_groups`].
    fn group_row_span_is_word_isolated(&self, group: ModularGroup) -> Result<bool> {
        let end_x = group
            .x
            .checked_add(group.width)
            .ok_or_else(|| Error::backend("Modular group horizontal extent overflow"))?;
        if end_x > self.layout.extent.width {
            return Err(Error::backend("Modular group exceeds the output width"));
        }
        let internal_right_boundary = end_x != self.layout.extent.width;
        if self.kind == OutputKind::Yuv422Packed {
            // Each output word owns a pair. An odd internal edge would make adjacent groups write
            // the same pair even though both plane rows themselves begin on word boundaries.
            return Ok(
                group.x.is_multiple_of(2) && (!internal_right_boundary || end_x.is_multiple_of(2))
            );
        }
        let bytes_per_pixel = match self.kind {
            OutputKind::NumericUnsigned | OutputKind::NumericSigned | OutputKind::NumericFloat => {
                u64::from(self.channels)
                    .checked_mul(u64::from(self.bits / 8))
                    .ok_or_else(|| Error::backend("numeric output pixel size overflow"))?
            }
            OutputKind::Luma | OutputKind::YuvSemiplanar | OutputKind::YuvPlanar => {
                u64::from(self.storage_bits / 8)
            }
            OutputKind::RgbInterleaved => u64::from(self.channels),
            OutputKind::RgbPlanar => 1,
            OutputKind::NativeModular => u64::from(self.channels)
                .checked_mul(u64::from(self.storage_bits / 8))
                .ok_or_else(|| Error::backend("native Modular output pixel size overflow"))?,
            OutputKind::Yuv422Packed => unreachable!("packed 4:2:2 was handled above"),
        };
        let start = u64::from(group.x)
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| Error::backend("Modular group output start overflow"))?;
        let end = u64::from(end_x)
            .checked_mul(bytes_per_pixel)
            .ok_or_else(|| Error::backend("Modular group output end overflow"))?;
        Ok(start.is_multiple_of(4) && (!internal_right_boundary || end.is_multiple_of(4)))
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
    modular_metadata: &[u32],
    dispatch: &GroupDispatchLayout,
    output: &OutputPlan,
    requested_frame_slots: usize,
    memory_limit_bytes: u64,
) -> Result<WgpuDecodeMemoryStats> {
    let storage_limit = device.limits().max_storage_buffer_binding_size;
    let buffer_limit = device.limits().max_buffer_size;
    let stream_bytes = dispatch.stream_bytes;
    let metadata_bytes = modular_metadata_bytes(modular_metadata)?;
    let output_bytes = align4(output.layout.logical_size)?;
    let dispatch_control_bytes = std::mem::size_of::<DispatchControl>() as u64;
    let native_f64_dummy_bytes = if output.f64_output_path == Some(F64OutputPath::NativeArithmetic)
    {
        NATIVE_F64_DUMMY_WORD_BYTES
    } else {
        0
    };
    for (name, required) in [
        ("bounded group stream window", stream_bytes),
        ("Modular metadata", metadata_bytes),
        (
            "parallel reconstructed samples",
            dispatch.reconstructed_bytes,
        ),
        ("requested output", output_bytes),
        ("group statuses", dispatch.status_bytes),
        ("group parameters", dispatch.params_bytes),
    ] {
        if required > storage_limit || required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device limit"
            )));
        }
    }
    for (name, required) in [
        ("group status readback", dispatch.status_bytes),
        ("parallel dispatch control", dispatch_control_bytes),
    ] {
        if required > buffer_limit {
            return Err(Error::backend(format!(
                "{name} buffer requires {required} bytes, exceeding the device buffer limit"
            )));
        }
    }
    if dispatch_control_bytes > device.limits().max_uniform_buffer_binding_size {
        return Err(Error::backend(
            "parallel dispatch control exceeds the device uniform-binding limit",
        ));
    }
    let per_frame = [
        stream_bytes,
        metadata_bytes,
        dispatch.reconstructed_bytes,
        output_bytes,
        native_f64_dummy_bytes,
        dispatch.status_bytes,
        dispatch.status_bytes,
        dispatch.params_bytes,
        dispatch_control_bytes,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| Error::backend("Modular GPU memory budget overflow"))?;
    let affordable_slots = memory_limit_bytes / per_frame;
    if affordable_slots == 0 {
        return Err(Error::backend(format!(
            "one Modular GPU frame requires {per_frame} bytes, exceeding the shared {memory_limit_bytes}-byte budget"
        )));
    }
    let max_frame_slots =
        requested_frame_slots.min(usize::try_from(affordable_slots).unwrap_or(usize::MAX));
    let max_frame_window_bytes = per_frame
        .checked_mul(
            u64::try_from(max_frame_slots)
                .map_err(|_| Error::backend("resolved frame-slot count exceeds u64"))?,
        )
        .ok_or_else(|| Error::backend("bounded in-flight GPU memory budget overflow"))?;
    let transient_bytes = per_frame
        .checked_sub(output_bytes)
        .ok_or_else(|| Error::backend("Modular transient memory accounting underflow"))?;
    let max_dispatch_workgroups =
        dispatch
            .stream_batches
            .iter()
            .try_fold(0u32, |maximum, batch| {
                u32::try_from(batch.len())
                    .map(|groups| maximum.max(groups.div_ceil(MODULAR_GROUP_WORKGROUP_SIZE)))
                    .map_err(|_| Error::backend("batch group count exceeds WGSL u32"))
            })?;
    if max_dispatch_workgroups == 0 {
        return Err(Error::backend("Modular stream batch layout is empty"));
    }
    Ok(WgpuDecodeMemoryStats {
        per_frame_bytes: per_frame,
        output_lease_bytes: output_bytes,
        transient_bytes,
        max_frame_slots,
        max_frame_window_bytes,
        stream_window_bytes: dispatch.stream_bytes,
        reconstruction_scratch_bytes: dispatch.reconstructed_bytes,
        reconstruction_lane_stride_bytes: dispatch.reconstruction_lane_stride,
        max_lz77_window_words: dispatch.max_lz77_window_words,
        max_lz77_scratch_words: dispatch.max_lz77_scratch_words,
        stream_batch_count: dispatch.stream_batches.len(),
        submissions_per_frame: dispatch.stream_batches.len(),
        parallel_group_lanes: dispatch.parallel_group_lanes,
        group_workgroup_size: MODULAR_GROUP_WORKGROUP_SIZE,
        max_dispatch_workgroups,
        output_write_path: dispatch.output_write_path,
        reconstruction_specialization: dispatch.reconstruction_specialization,
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
    // Only a bounded batch of pass-group packets is storage-bound at once. The host keeps the
    // validated codestream Arc, while queue ordering lets every batch reuse this one GPU window.
    let stream = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu decode bounded group stream window"),
        size: source.dispatch_layout.stream_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let metadata_bytes = u64::try_from(source.modular_metadata.len())
        .ok()
        .and_then(|entries| entries.checked_mul(u64::try_from(std::mem::size_of::<u32>()).ok()?))
        .ok_or_else(|| Error::backend("Modular metadata size overflow"))?;
    let metadata_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let metadata_buffer = buffers.checkout(
        "jxl-wgpu decode Modular metadata",
        metadata_bytes,
        metadata_usage,
        std::mem::align_of::<u32>() as u64,
    );
    backend.queue().write_buffer(
        metadata_buffer.buffer(),
        0,
        bytemuck::cast_slice(source.modular_metadata.as_ref()),
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

    let params_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let params_buffer = buffers.checkout(
        "jxl-wgpu decode Modular parameters",
        source.dispatch_layout.params_bytes,
        params_usage,
        std::mem::align_of::<u32>() as u64,
    );
    let mut params_upload = vec![
        0u8;
        usize::try_from(source.dispatch_layout.params_bytes).map_err(
            |_| Error::backend("group parameter upload exceeds host address space")
        )?
    ];
    for (index, (&group, window)) in source
        .profile
        .groups
        .iter()
        .zip(source.dispatch_layout.stream_windows.iter())
        .enumerate()
    {
        let status_index = u32::try_from(index)
            .map_err(|_| Error::backend("group status index exceeds WGSL u32"))?;
        let params = build_params(
            group,
            window.token_start,
            window.token_end,
            status_index,
            &source.profile,
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

    let dispatch_control = buffers.checkout(
        "jxl-wgpu decode Modular dispatch control",
        std::mem::size_of::<DispatchControl>() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        std::mem::align_of::<DispatchControl>() as u64,
    );

    let word_output_binding = native_f64_dummy_words.as_ref().map_or_else(
        || output.as_entire_binding(),
        |buffer| buffer.buffer().as_entire_binding(),
    );
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: stream.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: metadata_buffer.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: reconstructed.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 3,
            resource: word_output_binding,
        },
        wgpu::BindGroupEntry {
            binding: 4,
            resource: status.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 5,
            resource: params_buffer.buffer().as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 7,
            resource: dispatch_control.buffer().as_entire_binding(),
        },
    ];
    if source.output.f64_output_path == Some(F64OutputPath::NativeArithmetic) {
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: output.as_entire_binding(),
        });
    }
    let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu decode parallel Modular group bindings"),
        layout: &bind_group_layout,
        entries: &entries,
    });
    let completion = Arc::new(MapCompletion::default());
    let lifetime = Arc::new(DecodeJobLifetime {
        output: Arc::clone(&output),
        _modular_metadata: metadata_buffer,
        _reconstructed: reconstructed,
        _native_f64_dummy_words: native_f64_dummy_words,
        _status: status,
        status_staging,
        status_mapped: AtomicBool::new(false),
        _params: params_buffer,
        _dispatch_control: dispatch_control,
        output_permit: memory_permits.output,
        _transient_permit: memory_permits.transient,
    });
    let upload_len = usize::try_from(source.dispatch_layout.stream_bytes)
        .map_err(|_| Error::backend("bounded stream upload exceeds host address space"))?;
    let mut stream_upload = vec![0u8; upload_len];
    let mut final_submission = None;
    for (batch_index, batch) in source.dispatch_layout.stream_batches.iter().enumerate() {
        stream_upload.fill(0);
        for group_index in batch.clone() {
            let window = source
                .dispatch_layout
                .stream_windows
                .get(group_index)
                .ok_or_else(|| Error::backend("group stream window is missing"))?;
            let input = codestream
                .get(window.input_start..window.input_end)
                .ok_or_else(|| Error::backend("group stream input range is truncated"))?;
            let end = window
                .upload_offset
                .checked_add(input.len())
                .ok_or_else(|| Error::backend("group stream upload range overflow"))?;
            stream_upload
                .get_mut(window.upload_offset..end)
                .ok_or_else(|| Error::backend("group stream upload range is truncated"))?
                .copy_from_slice(input);
        }
        backend.queue().write_buffer(&stream, 0, &stream_upload);
        let control = DispatchControl {
            first_group: u32::try_from(batch.start)
                .map_err(|_| Error::backend("batch group index exceeds WGSL u32"))?,
            group_count: u32::try_from(batch.len())
                .map_err(|_| Error::backend("batch group count exceeds WGSL u32"))?,
            lane_stride_words: u32::try_from(source.dispatch_layout.reconstruction_lane_stride / 4)
                .map_err(|_| Error::backend("reconstruction lane stride exceeds WGSL u32"))?,
            _padding: 0,
        };
        backend.queue().write_buffer(
            lifetime._dispatch_control.buffer(),
            0,
            bytemuck::bytes_of(&control),
        );

        let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu decode bounded Modular batch"),
        });
        if batch_index == 0 {
            commands.clear_buffer(lifetime._reconstructed.buffer(), 0, None);
            commands.clear_buffer(&lifetime.output, 0, None);
            if let Some(dummy) = &lifetime._native_f64_dummy_words {
                commands.clear_buffer(dummy.buffer(), 0, None);
            }
            commands.clear_buffer(lifetime._status.buffer(), 0, None);
        }
        {
            let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("jxl-wgpu generic Modular entropy and MA reconstruction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &binding, &[]);
            pass.dispatch_workgroups(
                control.group_count.div_ceil(MODULAR_GROUP_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        let final_batch = batch_index + 1 == source.dispatch_layout.stream_batches.len();
        if final_batch {
            commands.copy_buffer_to_buffer(
                lifetime._status.buffer(),
                0,
                lifetime.status_staging.buffer(),
                0,
                source.dispatch_layout.status_bytes,
            );
            let callback_lifetime = Arc::clone(&lifetime);
            let callback_completion = Arc::clone(&completion);
            commands.map_buffer_on_submit(
                lifetime.status_staging.buffer(),
                wgpu::MapMode::Read,
                ..,
                move |result| {
                    // Release the callback's ownership before waking a waiter. The pending frame
                    // keeps the job alive through validation; an abandoned pending frame instead
                    // makes this the final Arc and safely unmaps/recycles staging.
                    if result.is_ok() {
                        callback_lifetime
                            .status_mapped
                            .store(true, Ordering::Release);
                    }
                    drop(callback_lifetime);
                    callback_completion.complete(
                        result.map_err(|error| format!("GPU status mapping failed: {error}")),
                    );
                },
            );
        }
        let submission = backend.queue().submit([commands.finish()]);
        if final_batch {
            final_submission = Some(submission);
        }
    }
    let submission = final_submission
        .ok_or_else(|| Error::backend("bounded Modular stream produced no GPU submission"))?;
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

fn build_params(
    group: ModularGroup,
    token_start: u32,
    token_end: u32,
    status_index: u32,
    profile: &StandardModularProfile,
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
    let (fixed_leaf_predictor, fixed_leaf_offset, fixed_leaf_multiplier, fixed_leaf_clusters) =
        match reconstruction_specialization(profile) {
            ModularReconstructionSpecialization::ChannelFixed {
                predictor,
                offset,
                multiplier,
                clusters,
                ..
            } => (
                u32::from(predictor.index()),
                u32::from_ne_bytes(offset.to_ne_bytes()),
                multiplier,
                clusters.map(u32::from),
            ),
            ModularReconstructionSpecialization::GenericMetaAdaptive => (0, 0, 0, [0; 4]),
        };
    Ok(ShaderParams {
        token_start,
        token_end,
        width: group.width,
        height: group.height,
        origin_x: group.x,
        origin_y: group.y,
        sample_count: group.sample_count()?,
        initialize_chroma: u32::from(initialize_chroma),
        source_channels: profile.channels.count(),
        source_bits: u32::from(profile.bits_per_sample),
        source_mask: (1u32 << profile.bits_per_sample) - 1,
        needs_self_correcting: u32::from(profile.ma_config.needs_self_correcting()),
        lz77_window_mask: group_lz77_window_words(
            profile,
            group,
            group_decoded_symbol_count(profile, group)?,
        )?
        .saturating_sub(1),
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
        status_index,
        stream_index: group.stream_index,
        fixed_leaf_predictor,
        fixed_leaf_offset,
        fixed_leaf_multiplier,
        fixed_leaf_cluster0: fixed_leaf_clusters[0],
        fixed_leaf_cluster1: fixed_leaf_clusters[1],
        fixed_leaf_cluster2: fixed_leaf_clusters[2],
        fixed_leaf_cluster3: fixed_leaf_clusters[3],
        wp_p1: profile.wp_header.p1,
        wp_p2: profile.wp_header.p2,
        wp_p3a: profile.wp_header.p3a,
        wp_p3b: profile.wp_header.p3b,
        wp_p3c: profile.wp_header.p3c,
        wp_p3d: profile.wp_header.p3d,
        wp_p3e: profile.wp_header.p3e,
        wp_w0: profile.wp_header.w0,
        wp_w1: profile.wp_header.w1,
        wp_w2: profile.wp_header.w2,
        wp_w3: profile.wp_header.w3,
    })
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
    use jxl_wgpu_encode::LosslessModularFormat;
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
    const GENERIC_RECONSTRUCTION: ModularReconstructionSpecialization =
        ModularReconstructionSpecialization::GenericMetaAdaptive;
    const FIXED_GRADIENT_RECONSTRUCTION: ModularReconstructionSpecialization =
        ModularReconstructionSpecialization::ChannelFixed {
            predictor: ModularPredictor::Gradient,
            offset: 0,
            multiplier: 1,
            channel_count: 4,
            clusters: [1, 2, 3, 4],
        };

    #[test]
    fn stream_batches_rebase_unaligned_group_bits_and_respect_peak_window() {
        let codestream = vec![0u8; 32];
        let group = |start, end| ModularGroup {
            token_bit_offset: start,
            token_bit_end: end,
            x: 0,
            y: 0,
            width: 1,
            height: 1,
            stream_index: 0,
        };
        let groups = [group(3, 67), group(75, 139), group(147, 211)];
        let (windows, batches, peak) =
            build_stream_batches(&codestream, &groups, 20, usize::MAX).unwrap();
        assert_eq!(batches, [0..1, 1..2, 2..3]);
        assert_eq!(peak, 16);
        for (window, original) in windows.iter().zip(groups) {
            assert_eq!(window.upload_offset, 0);
            assert_eq!(window.token_start, (original.token_bit_offset & 7) as u32);
            assert_eq!(
                window.token_end - window.token_start,
                u32::try_from(original.token_bit_end - original.token_bit_offset).unwrap()
            );
            assert_eq!(window.input_start, (original.token_bit_offset / 8) as usize);
            assert_eq!(
                window.input_end,
                original.token_bit_end.div_ceil(8) as usize
            );
        }
    }

    #[test]
    fn stream_batches_never_alias_more_groups_than_scratch_lanes() {
        let codestream = vec![0u8; 32];
        let groups = (0..5)
            .map(|index| ModularGroup {
                token_bit_offset: index * 16 + 3,
                token_bit_end: index * 16 + 11,
                x: index as u32,
                y: 0,
                width: 1,
                height: 1,
                stream_index: index as u32,
            })
            .collect::<Vec<_>>();
        let (windows, batches, _) = build_stream_batches(&codestream, &groups, 1024, 2).unwrap();
        assert_eq!(batches, [0..2, 2..4, 4..5]);
        assert_eq!(windows[0].upload_offset, 0);
        assert_eq!(windows[2].upload_offset, 0);
        assert_eq!(windows[4].upload_offset, 0);
    }

    #[test]
    fn adaptive_stream_layout_coalesces_or_trades_lanes_for_the_byte_budget() {
        let codestream = vec![0u8; 8 * 1024];
        let groups = (0..8)
            .map(|index| ModularGroup {
                token_bit_offset: index * 8 * 1024,
                token_bit_end: (index + 1) * 8 * 1024,
                x: index as u32,
                y: 0,
                width: 1,
                height: 1,
                stream_index: index as u32,
            })
            .collect::<Vec<_>>();
        let (lanes, _, batches, peak) =
            select_parallel_group_layout(&codestream, &groups, 64 * 1024, 8, 4096, 1024, 64 * 1024)
                .unwrap()
                .unwrap();
        assert_eq!(lanes, 8);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], 0..8);
        assert_eq!(peak, 8 * 1024 + STREAM_SENTINEL_BYTES);

        let (lanes, _, batches, peak) =
            select_parallel_group_layout(&codestream, &groups, 64 * 1024, 8, 4096, 1024, 20 * 1024)
                .unwrap()
                .unwrap();
        assert_eq!(lanes, 3);
        assert_eq!(batches, [0..3, 3..6, 6..8]);
        assert_eq!(peak, 3 * 1024 + STREAM_SENTINEL_BYTES);
    }

    #[test]
    fn aligned_output_requires_word_isolated_plane_rows_and_internal_group_edges() {
        let extent = Extent2d::new(516, 3);
        let groups = [
            ModularGroup {
                token_bit_offset: 0,
                token_bit_end: 1,
                x: 0,
                y: 0,
                width: 256,
                height: 3,
                stream_index: 0,
            },
            ModularGroup {
                token_bit_offset: 1,
                token_bit_end: 2,
                x: 256,
                y: 0,
                width: 256,
                height: 3,
                stream_index: 1,
            },
            ModularGroup {
                token_bit_offset: 2,
                token_bit_end: 3,
                x: 512,
                y: 0,
                width: 4,
                height: 3,
                stream_index: 2,
            },
        ];
        let mut cases = Vpi::ALL
            .iter()
            .filter_map(|&format| {
                let pixel_format = format.pixel_format();
                let request = match classify_pixel_format(&pixel_format).ok()? {
                    PixelFormatClass::Numeric(numeric) => {
                        let mapping = if numeric.sample_kind == SampleKind::Float
                            && numeric.bits_per_component == 64
                        {
                            NumericSampleMapping::NormalizedGray8F64(
                                F64OutputPolicy::ExactF32Widening,
                            )
                        } else {
                            NumericSampleMapping::NormalizedGray8
                        };
                        GpuOutputRequest::numeric(pixel_format, mapping).ok()?
                    }
                    PixelFormatClass::Color(_) => GpuOutputRequest::color(pixel_format).ok()?,
                };
                Some((format.name(), request, crate::ModularChannels::Gray))
            })
            .collect::<Vec<_>>();
        cases.extend([
            (
                "native-gray8",
                GpuOutputRequest::numeric(
                    LosslessModularFormat::Gray.pixel_format(8).unwrap(),
                    NumericSampleMapping::NativeUnsigned,
                )
                .unwrap(),
                crate::ModularChannels::Gray,
            ),
            (
                "native-rgb8",
                GpuOutputRequest::color(LosslessModularFormat::Rgb.pixel_format(8).unwrap())
                    .unwrap(),
                crate::ModularChannels::Rgb,
            ),
            (
                "native-rgba8",
                GpuOutputRequest::color(LosslessModularFormat::Rgba.pixel_format(8).unwrap())
                    .unwrap(),
                crate::ModularChannels::Rgba,
            ),
        ]);
        for (name, request, source_channels) in cases {
            let output =
                OutputPlan::new(extent, &request, source_channels, 8, PORTABLE_CAPABILITIES)
                    .unwrap_or_else(|error| panic!("{name} output plan failed: {error}"));
            assert_eq!(
                output.write_path_for_groups(&groups).unwrap(),
                OutputWritePath::WordAligned,
                "{name} standard 256-pixel group boundaries"
            );
        }

        let rgb_request = GpuOutputRequest::color(Vpi::Rgb8.pixel_format()).unwrap();
        let mut rgb = OutputPlan::new(
            extent,
            &rgb_request,
            crate::ModularChannels::Gray,
            8,
            PORTABLE_CAPABILITIES,
        )
        .unwrap();
        let nonisolated = [ModularGroup {
            token_bit_offset: 0,
            token_bit_end: 1,
            x: 1,
            y: 0,
            width: 255,
            height: 3,
            stream_index: 0,
        }];
        assert_eq!(
            rgb.write_path_for_groups(&nonisolated).unwrap(),
            OutputWritePath::AtomicBytes
        );
        rgb.layout.planes[0].row_stride += 1;
        assert_eq!(
            rgb.write_path_for_groups(&groups).unwrap(),
            OutputWritePath::AtomicBytes
        );
    }

    #[test]
    fn distance_one_lz_history_uses_no_storage_scratch() {
        assert_eq!(lz77_scratch_words(0), 0);
        assert_eq!(lz77_scratch_words(1), 0);
        assert_eq!(lz77_scratch_words(2), 2);
        assert_eq!(lz77_scratch_words(1 << 20), 1 << 20);
    }

    #[test]
    fn channel_fixed_gradient_proof_pins_channel_cluster_order_and_fallbacks() {
        let leaf = |cluster| MaTreeNodeIr::Leaf {
            cluster,
            predictor: ModularPredictor::Gradient.index(),
            offset: 0,
            multiplier: 1,
        };
        let nodes = [
            MaTreeNodeIr::Decision {
                property: 0,
                threshold: 1,
                left: 1,
                right: 4,
            },
            MaTreeNodeIr::Decision {
                property: 0,
                threshold: 2,
                left: 2,
                right: 3,
            },
            leaf(4),
            leaf(3),
            MaTreeNodeIr::Decision {
                property: 0,
                threshold: 0,
                left: 5,
                right: 6,
            },
            leaf(2),
            leaf(1),
        ];
        assert_eq!(
            channel_fixed_gradient_specialization(&nodes, 4, false),
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                offset: 0,
                multiplier: 1,
                channel_count: 4,
                clusters: [1, 2, 3, 4],
            }
        );

        let mut non_channel = nodes;
        non_channel[0] = MaTreeNodeIr::Decision {
            property: 3,
            threshold: 1,
            left: 1,
            right: 4,
        };
        assert_eq!(
            channel_fixed_gradient_specialization(&non_channel, 4, false),
            ModularReconstructionSpecialization::GenericMetaAdaptive
        );

        let mut bad_unused_channel = nodes;
        bad_unused_channel[2] = MaTreeNodeIr::Leaf {
            cluster: 4,
            predictor: ModularPredictor::West.index(),
            offset: 0,
            multiplier: 1,
        };
        assert_eq!(
            channel_fixed_gradient_specialization(&bad_unused_channel, 1, false),
            ModularReconstructionSpecialization::GenericMetaAdaptive
        );

        let cycle = [MaTreeNodeIr::Decision {
            property: 0,
            threshold: 0,
            left: 0,
            right: 0,
        }];
        assert_eq!(
            channel_fixed_gradient_specialization(&cycle, 1, false),
            ModularReconstructionSpecialization::GenericMetaAdaptive
        );
        assert_eq!(
            channel_fixed_gradient_specialization(&nodes, 4, true),
            ModularReconstructionSpecialization::GenericMetaAdaptive
        );
    }

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
        assert_eq!(std::mem::size_of::<ShaderParams>(), 208);
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
            needs_self_correcting: 12,
            lz77_window_mask: 13,
            output_kind: 14,
            transfer: 15,
            limited_range: 16,
            channels: 17,
            order: 18,
            bits: 19,
            storage_bits: 20,
            plane0_offset: 21,
            plane0_stride: 22,
            plane1_offset: 23,
            plane1_stride: 24,
            plane2_offset: 25,
            plane2_stride: 26,
            plane3_offset: 27,
            plane3_stride: 28,
            chroma_width: 29,
            chroma_height: 30,
            logical_size: 31,
            numeric_mapping: 32,
            status_index: 33,
            stream_index: 34,
            fixed_leaf_predictor: 35,
            fixed_leaf_offset: 36,
            fixed_leaf_multiplier: 37,
            fixed_leaf_cluster0: 38,
            fixed_leaf_cluster1: 39,
            fixed_leaf_cluster2: 40,
            fixed_leaf_cluster3: 41,
            wp_p1: 42,
            wp_p2: 43,
            wp_p3a: 44,
            wp_p3b: 45,
            wp_p3c: 46,
            wp_p3d: 47,
            wp_p3e: 48,
            wp_w0: 49,
            wp_w1: 50,
            wp_w2: 51,
            wp_w3: 52,
        };
        assert_eq!(
            bytemuck::cast::<ShaderParams, [u32; 52]>(params),
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                45, 46, 47, 48, 49, 50, 51, 52,
            ]
        );
        assert!(SHADER_TEMPLATE.contains("needs_self_correcting: u32,"));
        assert!(SHADER_TEMPLATE.contains("lz77_window_mask: u32,"));
        assert!(SHADER_TEMPLATE.contains("stream_index: u32,"));
        assert!(SHADER_TEMPLATE.contains("wp_w3: u32,"));
        assert!(SHADER_TEMPLATE.contains("params_table: array<Params>"));
        let portable = shader_source(
            F64OutputPath::ExactF32Widening,
            OutputWritePath::WordAligned,
            FIXED_GRADIENT_RECONSTRUCTION,
        );
        assert!(portable.contains("@compute @workgroup_size(64)"));
        assert!(portable.contains("@builtin(global_invocation_id)"));
        assert!(portable.contains("let lane_index = global_invocation_id.x;"));

        assert_eq!(std::mem::size_of::<DispatchControl>(), 16);
        assert_eq!(std::mem::align_of::<DispatchControl>(), 4);
        let control = DispatchControl {
            first_group: 1,
            group_count: 2,
            lane_stride_words: 3,
            _padding: 4,
        };
        assert_eq!(
            bytemuck::cast::<DispatchControl, [u32; 4]>(control),
            [1, 2, 3, 4]
        );

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
        assert!(SHADER_TEMPLATE.contains("let status_base = params.status_index * 4u;"));
        assert!(SHADER_TEMPLATE.contains("status[status_base] = STATUS_OK;"));
        assert!(SHADER_TEMPLATE.contains("status[status_base + 1u] = decoded;"));
        assert!(SHADER_TEMPLATE.contains("status[status_base + 2u] = bit_cursor;"));
        assert!(SHADER_TEMPLATE.contains("status[status_base + 3u] = params.token_end;"));
    }

    #[test]
    fn every_reconstruction_write_and_f64_shader_variant_validates() {
        let portable_atomic = shader_source(
            F64OutputPath::ExactF32Widening,
            OutputWritePath::AtomicBytes,
            GENERIC_RECONSTRUCTION,
        );
        let portable_aligned = shader_source(
            F64OutputPath::ExactF32Widening,
            OutputWritePath::WordAligned,
            GENERIC_RECONSTRUCTION,
        );
        let native_atomic = shader_source(
            F64OutputPath::NativeArithmetic,
            OutputWritePath::AtomicBytes,
            GENERIC_RECONSTRUCTION,
        );
        let native_aligned = shader_source(
            F64OutputPath::NativeArithmetic,
            OutputWritePath::WordAligned,
            GENERIC_RECONSTRUCTION,
        );
        let fixed_gradient_atomic = shader_source(
            F64OutputPath::ExactF32Widening,
            OutputWritePath::AtomicBytes,
            FIXED_GRADIENT_RECONSTRUCTION,
        );
        let fixed_gradient_aligned = shader_source(
            F64OutputPath::ExactF32Widening,
            OutputWritePath::WordAligned,
            FIXED_GRADIENT_RECONSTRUCTION,
        );
        let native_fixed_gradient_atomic = shader_source(
            F64OutputPath::NativeArithmetic,
            OutputWritePath::AtomicBytes,
            FIXED_GRADIENT_RECONSTRUCTION,
        );
        let native_fixed_gradient_aligned = shader_source(
            F64OutputPath::NativeArithmetic,
            OutputWritePath::WordAligned,
            FIXED_GRADIENT_RECONSTRUCTION,
        );
        assert!(!portable_aligned.contains(F64_OUTPUT_MARKER));
        assert!(!native_aligned.contains(F64_OUTPUT_MARKER));
        assert!(!portable_aligned.contains(F64_BINDING_MARKER));
        assert!(!native_aligned.contains(F64_BINDING_MARKER));
        assert!(!portable_aligned.contains("f64(sample)"));
        assert!(native_aligned.contains("f64(sample) / 255.0"));
        assert!(native_aligned.contains("output_f64: array<f64>"));
        assert!(portable_atomic.contains("array<atomic<u32>>"));
        assert!(portable_atomic.contains("atomicCompareExchangeWeak"));
        assert!(native_atomic.contains("atomicStore"));
        assert!(portable_aligned.contains("output_words: array<u32>"));
        assert!(!portable_aligned.contains("atomicCompareExchangeWeak"));
        assert!(!native_aligned.contains("atomicStore"));
        assert!(fixed_gradient_atomic.contains("atomicCompareExchangeWeak"));
        assert!(fixed_gradient_aligned.contains("fn fixed_leaf_cluster"));
        assert!(!fixed_gradient_aligned.contains("fn ma_leaf"));
        assert!(!MODULAR_FIXED_GRADIENT_SHADER.contains(" % "));
        assert!(!MODULAR_FIXED_GRADIENT_SHADER.contains(" / params.width"));

        let native_without_capability = naga::front::wgsl::parse_str(&native_aligned)
            .expect("native F64 WGSL syntax must parse before capability validation");
        let error = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&native_without_capability)
        .expect_err("native F64 WGSL must be rejected without Naga FLOAT64 capability");
        assert!(format!("{error:?}").contains("FLOAT64"));

        for (name, source, capabilities) in [
            (
                "portable-atomic",
                portable_atomic,
                naga::valid::Capabilities::empty(),
            ),
            (
                "portable-aligned",
                portable_aligned,
                naga::valid::Capabilities::empty(),
            ),
            (
                "native-f64-atomic",
                native_atomic,
                naga::valid::Capabilities::FLOAT64,
            ),
            (
                "native-f64-aligned",
                native_aligned,
                naga::valid::Capabilities::FLOAT64,
            ),
            (
                "portable-fixed-gradient-atomic",
                fixed_gradient_atomic,
                naga::valid::Capabilities::empty(),
            ),
            (
                "portable-fixed-gradient-aligned",
                fixed_gradient_aligned,
                naga::valid::Capabilities::empty(),
            ),
            (
                "native-f64-fixed-gradient-atomic",
                native_fixed_gradient_atomic,
                naga::valid::Capabilities::FLOAT64,
            ),
            (
                "native-f64-fixed-gradient-aligned",
                native_fixed_gradient_aligned,
                naga::valid::Capabilities::FLOAT64,
            ),
        ] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name} WGSL did not parse: {error}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap_or_else(|error| panic!("{name} WGSL did not validate: {error:?}"));
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
