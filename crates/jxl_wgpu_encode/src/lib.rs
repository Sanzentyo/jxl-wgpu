//! GPU-required JPEG XL encoder orchestration.
//!
//! This crate has no CPU image encoder and no CPU fallback. A backend must
//! record its image, transform, quantization, tokenization, and histogram work
//! through `wgpu`. The CPU side is limited to job coordination and deterministic
//! JPEG XL bitstream/container assembly.
//!
//! [`LosslessModularEncoder`] implements standard multi-group lossless Modular Gray, RGB, and RGBA
//! for every unsigned integer depth in `1..=16`. Samples remain GPU-resident through reversible
//! color transform, prediction, residual tokenization, and histogram collection. The generic
//! [`GpuEncoder`] advertises only profiles implemented by its backend.
//! [`LosslessModularAnimationSession`] adds standard timebases, exact frame durations and
//! timecodes, signed crop rectangles, all five blend modes, alpha extra-channel blending, and four
//! reference slots. Its independent frame submissions support both blocking waits and a
//! runtime-neutral [`Future`], so a caller can keep multiple frames in flight. Multi-batch browser
//! jobs advance one bounded map callback at a time from that same future and do not require a
//! specific executor or Web Worker.
//! [`VarDctEncoder`] executes all 27 standard strategy identifiers. Strategies through 32x32 use a
//! fixed diagnostic transform artifact; the 64x64 through 256x256 families use a scalable
//! per-8x8-block GPU DC reduction followed by a bounded GPU control/entropy pass. Its fixed
//! distance-25 profile retains DC and quantizes AC to zero.
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
mod prefix;
mod session;
mod vardct_encoder;

pub use buffer_pool::{
    DEFAULT_ENCODER_BUFFER_POOL_BYTES, EncoderBufferPoolStats, MAX_ENCODER_BUFFER_POOL_IDLE_SETS,
};
pub use capability::{
    Determinism, EncodeProfile, EncoderCapabilities, KernelStage, PerceptualDistance,
    ProfileCapability, ProgressivePass, ProgressivePlan,
};
pub use error::{BackendError, EncodeError, PacketError, UnsupportedFeature};
pub use gpu::{
    BufferImageSource, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameSource,
    TextureImageSource, WgpuContext,
};
pub use lossless_modular::{
    LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularAnimationDescriptor,
    LosslessModularAnimationSession, LosslessModularBackend, LosslessModularEncoder,
    LosslessModularFormat, LosslessModularGroup, LosslessModularGroupGrid,
    LosslessModularInFlightMemory, LosslessModularJob, LosslessModularMemoryLimits,
    LosslessModularMemoryPlan, LosslessModularSubmission,
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
    VarDctBackend, VarDctColorEncoding, VarDctEncoder, VarDctJob, VarDctKernelLayout,
    VarDctMemoryPlan, VarDctStrategy, VarDctSubmission,
};
