use std::num::NonZeroU64;
use std::sync::{Arc, OnceLock};

use jxl_wgpu::{KernelVariant, MemoryBudget, WgpuBackend};

use crate::ModularPredictor;
use crate::buffer_pool::DecodeBufferPool;
use crate::entropy::EntropyStreamParams;
use crate::modular_finalize::ModularFinalizePipeline;
use crate::modular_palette::ModularPalettePipeline;
use crate::modular_rct::ModularRctPipeline;
use crate::modular_squeeze::ModularSqueezePipeline;
use crate::progressive_dc::{ProgressiveDcGpuError, ProgressiveDcPipeline};
pub(super) const SHADER_TEMPLATE: &str = include_str!("../lossless_gray8.wgsl");
pub(super) const MODULAR_ENTROPY_ABI_SHADER: &str = include_str!("../modular_entropy_abi.wgsl");
pub(super) const MODULAR_ENTROPY_SHADER: &str = include_str!("../modular_entropy.wgsl");
pub(super) const MODULAR_RECONSTRUCT_SHADER: &str = include_str!("../modular_reconstruct.wgsl");
pub(super) const MODULAR_RESUME_SHADER: &str = include_str!("../modular_resume.wgsl");
pub(super) const MODULAR_FIXED_GRADIENT_SHADER: &str =
    include_str!("../modular_fixed_gradient.wgsl");
pub(super) const MODULAR_ENTROPY_ABI_MARKER: &str = "/*__JXL_MODULAR_ENTROPY_ABI__*/";
pub(super) const MODULAR_ENTROPY_MARKER: &str = "/*__JXL_MODULAR_ENTROPY__*/";
pub(super) const MODULAR_RESUME_MARKER: &str = "/*__JXL_MODULAR_RESUME__*/";
pub(super) const MODULAR_RECONSTRUCT_MARKER: &str = "/*__JXL_MODULAR_RECONSTRUCT__*/";
pub(super) const F64_OUTPUT_MARKER: &str = "/*__JXL_F64_OUTPUT__*/";
pub(super) const F64_BINDING_MARKER: &str = "/*__JXL_F64_BINDING__*/";
pub(super) const OUTPUT_WORDS_TYPE_MARKER: &str = "/*__JXL_OUTPUT_WORDS_TYPE__*/";
pub(super) const WRITE_BYTE_WORD_MARKER: &str = "/*__JXL_WRITE_BYTE_WORD__*/";
pub(super) const WRITE_FULL_WORD_MARKER: &str = "/*__JXL_WRITE_FULL_WORD__*/";
pub(super) const F64_EXACT_F32_WIDENING: &str = r#"
                if params.numeric_mapping != 1u {
                    decode_error = ERROR_OUTPUT_MAPPING;
                } else {
                    let words = widen_normalized_f32_to_f64_words(normalized_bits);
                    write_word(offset, words.x);
                    write_word(offset + 4u, words.y);
                }
"#;
pub(super) const F64_NATIVE_ARITHMETIC: &str = r#"
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
pub(super) const F64_NATIVE_BINDING: &str =
    "@group(0) @binding(6) var<storage, read_write> output_f64: array<f64>;";
pub(super) const ATOMIC_OUTPUT_WORDS_TYPE: &str = "array<atomic<u32>>";
pub(super) const WORD_ALIGNED_OUTPUT_WORDS_TYPE: &str = "array<u32>";
pub(super) const ATOMIC_WRITE_BYTE_WORD: &str = r#"
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
pub(super) const WORD_ALIGNED_WRITE_BYTE_WORD: &str = r#"
    let previous = output_words[word_index];
    output_words[word_index] = (previous & ~mask) | ((value & 0xffu) << shift);
"#;
pub(super) const ATOMIC_WRITE_FULL_WORD: &str = "atomicStore(&output_words[offset >> 2u], value);";
pub(super) const WORD_ALIGNED_WRITE_FULL_WORD: &str = "output_words[offset >> 2u] = value;";
pub(super) const STATUS_OK: u32 = 1;
pub(super) const ENTROPY_EXECUTION_STATE_WORDS: u64 = 8;
pub(super) const ENTROPY_EXECUTION_STATE_BYTES: u64 = ENTROPY_EXECUTION_STATE_WORDS * 4;
// Generic MA reconstruction additionally persists Property-8 gradient history. The weighted
// predictor extends that tail with four true errors and twelve subprediction-error accumulators.
// Both layouts end on a 16-byte boundary so adjacent scratch lanes cannot alias.
pub(super) const GENERIC_PREDICTOR_EXECUTION_STATE_BYTES: u64 =
    std::mem::size_of::<GenericPredictorExecutionState>() as u64;
pub(super) const GENERIC_WEIGHTED_EXECUTION_STATE_BYTES: u64 =
    std::mem::size_of::<WeightedModularExecutionState>() as u64;
pub(super) const NATIVE_F64_DUMMY_WORD_BYTES: u64 = 4;
pub(super) const DEFAULT_MODULAR_GROUP_VARIANT: KernelVariant = KernelVariant::Lanes64;
// Each lane is one serial reconstruction invocation which may process a full 256x256 group. Keep
// a finite watchdog-oriented ceiling even when the adapter and shared byte budget allow more.
pub(super) const WATCHDOG_PARALLEL_GROUP_LANE_CAP: usize = 512;

/// Conservative GPU allocation accounting for the stock decoder's bounded frame window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuDecodeMemoryStats {
    pub per_frame_bytes: u64,
    /// Packed global and deduplicated local MA/entropy descriptors retained on the GPU.
    pub modular_metadata_bytes: u64,
    /// LF/pass subimage streams that select a local rather than the outer global MA configuration.
    pub local_ma_stream_count: usize,
    /// Distinct MA configurations resident in `modular_metadata_bytes`, including the global one.
    pub unique_ma_config_count: usize,
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
    /// Per-lane 16-byte-aligned entropy and reconstruction resume record included in the stride.
    pub execution_state_bytes_per_lane: u64,
    /// Entropy representation parsed from the frame's Modular MA configuration.
    pub entropy_coding: ModularEntropyCoding,
    /// Largest decoded sample count represented by one logical group stream.
    pub max_logical_reconstruction_sample_words: u32,
    /// Largest sample workspace physically retained by one reconstruction lane.
    ///
    /// Fixed-Gradient normalized Gray8 groups with invocation-private LZ history retain two rows;
    /// direct generic streams retain their complete logical samples, while generalized streams
    /// report the resident inverse arena high-water.
    pub max_physical_reconstruction_sample_words: u32,
    /// Resident transformed-sample arena contained in one generalized reconstruction lane.
    pub resident_modular_arena_bytes: u64,
    /// Frame-resident transformed arena and entropy state retained across all LF/pass batches.
    pub frame_modular_arena_bytes: u64,
    /// Samples reconstructed from the DC-global stream before LF/pass-subimage assembly.
    pub global_reconstruction_sample_words: u32,
    /// Nonempty LF-group Modular entropy streams scheduled before pass groups.
    pub low_frequency_group_stream_count: usize,
    /// Progressive Modular passes declared by the frame header, including empty passes.
    pub progressive_pass_count: u32,
    /// Number of ordered inverse RCT/Palette/Squeeze dispatches after entropy reconstruction.
    pub inverse_transform_count: usize,
    /// Palette dispatches included in `inverse_transform_count`, including bounded serial chunks.
    pub palette_dispatch_count: usize,
    /// Aggregate per-dispatch uniform allocation retained through inverse submission.
    pub inverse_transform_uniform_bytes: u64,
    /// Final source-plane packing uniform retained through the same submission.
    pub final_output_uniform_bytes: u64,
    /// Three planar F32 XYB dependency buffers retained by a progressive-DC producer.
    pub progressive_dc_plane_bytes: u64,
    /// Modular-to-XYB conversion uniform retained through the producer submission.
    pub progressive_dc_uniform_bytes: u64,
    /// Largest descriptor-derived LZ history ring used by one group lane.
    pub max_lz77_window_words: u32,
    /// Largest physical LZ history ring stored in one reconstruction lane.
    ///
    /// A logical one-word ring uses invocation-private state and therefore reports zero here.
    pub max_lz77_scratch_words: u32,
    /// Bounded stream uploads required for one frame, including a DC-global entropy job when
    /// present. Each batch is one ordered queue submission.
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
    /// Output traversal selected after proving the source, output, and MA-tree contracts.
    pub output_specialization: ModularOutputSpecialization,
    /// Reconstruction kernel selected from the fully validated MA-tree IR.
    pub reconstruction_specialization: ModularReconstructionSpecialization,
}

/// Entropy representation used by the decoded Modular image stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularEntropyCoding {
    /// Canonical prefix-code histograms.
    Prefix,
    /// JPEG XL's 12-bit asymmetric numeral system.
    Ans,
    /// Independently coded groups use both Prefix and ANS descriptors.
    Mixed,
}

/// GPU output update strategy selected for a validated frame layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputWritePath {
    /// Byte updates use atomics because distinct group rows may share a storage word.
    AtomicBytes,
    /// Every plane row and internal group boundary is word-aligned, allowing ordinary RMW/store.
    WordAligned,
}

/// Typed Modular output traversal selected for one decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularOutputSpecialization {
    /// Reconstruct the complete group before converting it into the requested output layout.
    FinalizePass,
    /// Emit proven normalized U8 Gray8 samples from the fixed-Gradient reconstruction loop.
    ///
    /// Groups with invocation-private LZ history additionally retain only the current and previous
    /// reconstruction rows; the logical and physical workspace fields in
    /// [`WgpuDecodeMemoryStats`] report whether that compaction was selected.
    DirectNormalizedGray8,
}

/// Typed Modular reconstruction specialization selected for one decoded frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularReconstructionSpecialization {
    /// The complete MA tree and predictor family are evaluated per sample.
    GenericMetaAdaptive,
    /// The complete MA tree is evaluated against descriptor-addressed transformed channels.
    DescriptorMetaAdaptive,
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
pub(super) struct ShaderParams {
    pub(super) entropy: EntropyStreamParams,
    pub(super) window_logical_start: u32,
    pub(super) window_upload_start: u32,
    pub(super) stream_token_end: u32,
    pub(super) window_yield_end: u32,
    pub(super) window_flags: u32,
    pub(super) entropy_state_offset: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) origin_x: u32,
    pub(super) origin_y: u32,
    pub(super) sample_count: u32,
    pub(super) initialize_chroma: u32,
    pub(super) source_channels: u32,
    pub(super) channel_layout_offset: u32,
    pub(super) metadata_base: u32,
    pub(super) source_bits: u32,
    pub(super) source_mask: u32,
    pub(super) needs_self_correcting: u32,
    pub(super) output_kind: u32,
    pub(super) transfer: u32,
    pub(super) limited_range: u32,
    pub(super) channels: u32,
    pub(super) order: u32,
    pub(super) bits: u32,
    pub(super) storage_bits: u32,
    pub(super) plane0_offset: u32,
    pub(super) plane0_stride: u32,
    pub(super) plane1_offset: u32,
    pub(super) plane1_stride: u32,
    pub(super) plane2_offset: u32,
    pub(super) plane2_stride: u32,
    pub(super) plane3_offset: u32,
    pub(super) plane3_stride: u32,
    pub(super) chroma_width: u32,
    pub(super) chroma_height: u32,
    pub(super) logical_size: u32,
    pub(super) numeric_mapping: u32,
    pub(super) status_index: u32,
    pub(super) stream_index: u32,
    pub(super) fixed_leaf_predictor: u32,
    pub(super) fixed_leaf_offset: u32,
    pub(super) fixed_leaf_multiplier: u32,
    pub(super) fixed_leaf_cluster0: u32,
    pub(super) fixed_leaf_cluster1: u32,
    pub(super) fixed_leaf_cluster2: u32,
    pub(super) fixed_leaf_cluster3: u32,
    pub(super) fixed_output_mode: u32,
    pub(super) wp_p1: u32,
    pub(super) wp_p2: u32,
    pub(super) wp_p3a: u32,
    pub(super) wp_p3b: u32,
    pub(super) wp_p3c: u32,
    pub(super) wp_p3d: u32,
    pub(super) wp_p3e: u32,
    pub(super) wp_w0: u32,
    pub(super) wp_w1: u32,
    pub(super) wp_w2: u32,
    pub(super) wp_w3: u32,
}

/// CPU/WGSL ABI selecting one bounded parallel group wave.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DispatchControl {
    pub(super) first_group: u32,
    pub(super) group_count: u32,
    pub(super) lane_stride_words: u32,
    pub(super) _padding: u32,
}

/// Persistent per-lane state used to resume one entropy consumer after a bounded upload window.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EntropyExecutionState {
    pub(super) bit_cursor: u32,
    pub(super) ans_state: u32,
    pub(super) copy_remaining: u32,
    pub(super) copy_position: u32,
    pub(super) entropy_decoded: u32,
    pub(super) last_value: u32,
    pub(super) consumer_decoded: u32,
    pub(super) error_code: u32,
}

/// Persistent state for a generic MA consumer without SelfCorrecting prediction.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GenericPredictorExecutionState {
    pub(super) entropy: EntropyExecutionState,
    pub(super) predictor_prev_grad: i32,
    pub(super) _padding: [u32; 3],
}

/// Complete persistent state for a generic MA consumer using SelfCorrecting prediction.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WeightedModularExecutionState {
    pub(super) entropy: EntropyExecutionState,
    pub(super) predictor_prev_grad: i32,
    pub(super) wp_true_err_w: i32,
    pub(super) wp_true_err_nw: i32,
    pub(super) wp_true_err_n: i32,
    pub(super) wp_true_err_ne: i32,
    pub(super) wp_subpred_nw_ww: [u32; 4],
    pub(super) wp_subpred_n_w: [u32; 4],
    pub(super) wp_subpred_ne: [u32; 4],
    pub(super) _padding: [u32; 3],
}

/// Fixed storage-buffer status written by `lossless_gray8.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct DecodeStatus {
    pub(super) code: u32,
    pub(super) decoded_samples: u32,
    pub(super) cursor: u32,
    pub(super) expected_cursor: u32,
}

pub(super) const STATUS_BYTES: u64 = std::mem::size_of::<DecodeStatus>() as u64;

const _: () = {
    assert!(std::mem::size_of::<ShaderParams>() == 244);
    assert!(std::mem::align_of::<ShaderParams>() == 4);
    assert!(std::mem::size_of::<EntropyExecutionState>() == 32);
    assert!(std::mem::align_of::<EntropyExecutionState>() == 16);
    assert!(std::mem::size_of::<GenericPredictorExecutionState>() == 48);
    assert!(std::mem::align_of::<GenericPredictorExecutionState>() == 16);
    assert!(std::mem::size_of::<WeightedModularExecutionState>() == 112);
    assert!(std::mem::align_of::<WeightedModularExecutionState>() == 16);
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
    pub(super) backend: WgpuBackend,
    pub(super) pipelines: Arc<DecodePipelineCache>,
    pub(super) native_f64_pipelines: Option<Arc<DecodePipelineCache>>,
    pub(super) inverse_pipelines: Arc<ModularInversePipelineCache>,
    pub(super) progressive_dc_pipeline:
        Arc<OnceLock<std::result::Result<Arc<ProgressiveDcPipeline>, ProgressiveDcGpuError>>>,
    pub(super) memory: MemoryBudget,
    pub(super) buffers: Arc<DecodeBufferPool>,
    pub(super) stream_window_limit: Option<NonZeroU64>,
}

#[derive(Default)]
pub(super) struct DecodePipelineCache {
    pub(super) generic_atomic: OnceLock<Arc<wgpu::ComputePipeline>>,
    pub(super) generic_word_aligned: OnceLock<Arc<wgpu::ComputePipeline>>,
    pub(super) descriptor_atomic: OnceLock<Arc<wgpu::ComputePipeline>>,
    pub(super) descriptor_word_aligned: OnceLock<Arc<wgpu::ComputePipeline>>,
    pub(super) fixed_gradient_atomic: OnceLock<Arc<wgpu::ComputePipeline>>,
    pub(super) fixed_gradient_word_aligned: OnceLock<Arc<wgpu::ComputePipeline>>,
}

#[derive(Default)]
pub(super) struct ModularInversePipelineCache {
    pub(super) palette: OnceLock<
        std::result::Result<
            Arc<ModularPalettePipeline>,
            crate::modular_palette::ModularPaletteError,
        >,
    >,
    pub(super) squeeze: OnceLock<
        std::result::Result<
            Arc<ModularSqueezePipeline>,
            crate::modular_squeeze::ModularSqueezeError,
        >,
    >,
    pub(super) rct:
        OnceLock<std::result::Result<Arc<ModularRctPipeline>, crate::modular_rct::ModularRctError>>,
    pub(super) finalize_exact:
        OnceLock<std::result::Result<Arc<ModularFinalizePipeline>, crate::ModularFinalizeError>>,
    pub(super) finalize_native:
        OnceLock<std::result::Result<Arc<ModularFinalizePipeline>, crate::ModularFinalizeError>>,
}

pub(super) struct ModularInversePipelines {
    pub(super) palette: Option<Arc<ModularPalettePipeline>>,
    pub(super) squeeze: Option<Arc<ModularSqueezePipeline>>,
    pub(super) rct: Option<Arc<ModularRctPipeline>>,
    pub(super) finalize: Arc<ModularFinalizePipeline>,
}
