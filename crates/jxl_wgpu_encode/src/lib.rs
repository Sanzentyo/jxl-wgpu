//! GPU-required JPEG XL encoder orchestration.
//!
//! This crate has no CPU image encoder and no CPU fallback. A backend must
//! record its image, transform, quantization, tokenization, and histogram work
//! through `wgpu`. The CPU side is limited to job coordination and deterministic
//! JPEG XL bitstream/container assembly.
//!
//! [`LosslessGray8Encoder`] implements standard multi-group lossless Modular Gray8 while the
//! generic [`GpuEncoder`] advertises only profiles implemented by its selected backend.
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
mod lossless_gray8;
mod packet;
mod prefix;
mod session;

pub use buffer_pool::{
    DEFAULT_ENCODER_BUFFER_POOL_BYTES, EncoderBufferPoolStats, MAX_ENCODER_BUFFER_POOL_IDLE_SETS,
};
pub use capability::{
    Determinism, EncodeProfile, EncoderCapabilities, KernelStage, PerceptualDistance,
    ProfileCapability, ProgressivePass, ProgressivePlan,
};
pub use error::{EncodeError, PacketError, UnsupportedFeature};
pub use gpu::{
    BufferImageSource, GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameSource,
    TextureImageSource, WgpuContext,
};
pub use lossless_gray8::{
    LOSSLESS_GRAY8_GROUP_DIMENSION, LosslessGray8Backend, LosslessGray8Encoder, LosslessGray8Group,
    LosslessGray8GroupGrid, LosslessGray8InFlightMemory, LosslessGray8Job,
    LosslessGray8MemoryLimits, LosslessGray8MemoryPlan, LosslessGray8Submission,
};
pub use packet::{
    BitFragment, EncodedFrame, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind,
    assemble_frame,
};
pub use session::{
    AnimationHeader, BlendMode, CodestreamAssembler, EncodeSession, FrameEncodeRequest, FrameIndex,
    FrameOptions, FrameSubmission, FrameTiming, GpuAccelerationArtifact, GpuFrameArtifacts,
    ReferenceSlot, SessionDescriptor,
};
