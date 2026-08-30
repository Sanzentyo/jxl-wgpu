//! GPU-required JPEG XL encoder orchestration.
//!
//! This crate has no CPU image encoder and no CPU fallback. A backend must
//! record its image, transform, quantization, tokenization, and histogram work
//! through `wgpu`. The CPU side is limited to job coordination and deterministic
//! JPEG XL bitstream/container assembly.
//!
//! The first revision intentionally exposes the backend boundary and the exact
//! group-packet assembly primitive before claiming complete JPEG XL encoding.
//! [`GpuEncoder`] can only advertise profiles implemented by its backend.

#![deny(unsafe_code)]
// Public zero-copy sources intentionally keep Arc-based types on every target. Browser WebGPU
// handles are main-thread-local even though the same native handles are Send + Sync.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]

mod capability;
mod error;
mod gpu;
mod lossless_gray8;
mod packet;
mod prefix;
mod session;

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
    LosslessGray8Backend, LosslessGray8Encoder, LosslessGray8InFlightMemory, LosslessGray8Job,
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
