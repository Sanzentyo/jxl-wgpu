//! GPU-required JPEG XL encoder orchestration.
//!
//! This crate has no CPU image encoder and no CPU fallback. A backend must
//! record its image, transform, quantization, tokenization, and histogram work
//! through `wgpu`. The CPU side is limited to job coordination and deterministic
//! JPEG XL bitstream/container assembly.
//!
//! [`LosslessModularEncoder`] implements standard multi-group lossless Modular Gray/GrayAlpha/RGB/RGBA
//! for every unsigned integer depth in `1..=31` and IEEE binary16/binary32. Packed, planar and
//! split pitch-linear buffers support component swizzles and explicit word bit/byte order.
//! [`LosslessModularConfig`] selects group geometry, all 42 RCT types and all 14 predictors,
//! including checked custom Weighted/SelfCorrecting parameters. [`LosslessModularLz77`] selects
//! zero-run coding or bounded GPU hash-chain search for arbitrary residual sequences.
//! Full-range enumerated source color is retained, with explicit intent and image-white options.
//! [`AlphaAssociation`] declares source association without changing samples or invisible color.
//! Samples remain GPU-resident through reversible
//! color transform, prediction, residual tokenization, and histogram collection. The generic
//! [`GpuEncoder`] advertises only profiles implemented by its backend.
//! [`LosslessModularAnimationSession`] adds standard timebases, exact frame durations and
//! timecodes, signed crop rectangles, all five blend modes, alpha extra-channel blending, and four
//! reference slots. Its independent frame submissions support both blocking waits and a
//! runtime-neutral [`Future`], so a caller can keep multiple frames in flight. Multi-batch browser
//! jobs advance one bounded map callback at a time from that same future and do not require a
//! specific executor or Web Worker.
//! [`VarDctEncoder`] executes all 27 standard strategies singly or in a validated
//! [`VarDctStrategyMap`]. Per-strategy GPU batches share resident image/coefficient/LF arenas,
//! quantize real AC, and pack one bounded entropy fragment per transform. Partial source edges
//! are replicated on GPU; mixed maps may span multiple LF and AC groups. Quantization uses
//! explicit global/LF controls and per-transform HF multipliers; content-adaptive strategy search,
//! distance control and progressive encoding
//! remain incomplete. [`TiledVarDctEncoder`] provides an optimized DCT8 workgroup path.
//! [`VarDctCoefficientOrders`] selects independent X/Y/B permutations for every standard size class;
//! GPU serializers consume them without exposing image coefficients to the host.
//!
//! Fixed CPU/WGSL ABI records use `#[repr(C)]` plus `bytemuck::Pod`. WGSL defines
//! host-shareable numeric values as little-endian, so this crate rejects big-endian targets at
//! compile time rather than silently reinterpreting native-endian `Pod` bytes.

#![deny(unsafe_code)]
// Public zero-copy sources intentionally keep Arc-based types on every target. Browser WebGPU
// handles are main-thread-local even though the same native handles are Send + Sync.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]

#[cfg(not(target_endian = "little"))]
compile_error!(
    "jxl_wgpu_encode requires a little-endian target because WGSL host-shareable buffer values are little-endian"
);

mod buffer_pool;
mod capability;
mod error;
mod gpu;
mod lossless_modular;
mod packet;
mod permutation;
mod prefix;
mod session;
mod vardct_encoder;

pub use buffer_pool::{
    DEFAULT_ENCODER_BUFFER_POOL_BYTES, EncoderBufferPoolStats, MAX_ENCODER_BUFFER_POOL_IDLE_SETS,
};
pub use capability::{
    Determinism, EncodeProfile, EncoderCapabilities, KernelStage, ProfileCapability,
    ProgressiveDownsampling, ProgressivePass, ProgressivePlan,
};
pub use error::{BackendError, EncodeError, PacketError, UnsupportedFeature};
pub use gpu::{
    BufferImageSource, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameSource,
    TextureImageSource, WgpuContext,
};
pub use jxl_gpu_bitstream::FiniteF16;
pub use lossless_modular::{
    AlphaAssociation, LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularAnimationDescriptor,
    LosslessModularAnimationSession, LosslessModularBackend, LosslessModularColorOptions,
    LosslessModularColorTransform, LosslessModularConfig, LosslessModularEncoder,
    LosslessModularFormat, LosslessModularGroup, LosslessModularGroupGrid,
    LosslessModularGroupSize, LosslessModularInFlightMemory, LosslessModularJob,
    LosslessModularLz77, LosslessModularMemoryLimits, LosslessModularMemoryPlan,
    LosslessModularPalette, LosslessModularPredictor, LosslessModularRctType,
    LosslessModularSqueeze, LosslessModularSubmission, LosslessModularTreeMode,
    LosslessModularWeightedPredictor,
};
pub use packet::{
    BitFragment, EncodedFrame, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind,
    assemble_frame,
};
pub use session::{
    AnimationHeader, BlendMode, CodestreamAssembler, EncodeSession, FrameBlend, FrameCrop,
    FrameEncodeRequest, FrameIndex, FrameOptions, FrameSubmission, FrameTiming,
    GpuAccelerationArtifact, GpuFrameArtifacts, ReferenceSlot, SessionDescriptor,
};
pub use vardct_encoder::{
    TiledVarDctEncoder, TiledVarDctGrid, VarDctBackend, VarDctCoefficientOrders,
    VarDctColorEncoding, VarDctConfig, VarDctDequantMatrices, VarDctEncoder, VarDctGroupOrder,
    VarDctHfMultiplier, VarDctJob, VarDctKernelLayout, VarDctLfMetadata, VarDctMatrixEncoding,
    VarDctMemoryPlan, VarDctQuantization, VarDctRawMatrix, VarDctStrategy, VarDctStrategyMap,
    VarDctSubmission, VarDctTransform, VarDctTransformMemoryPlan,
};
