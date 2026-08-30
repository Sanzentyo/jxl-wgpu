//! GPU-required JPEG XL decode orchestration.
//!
//! This crate has no production CPU pixel/entropy decoder and no fallback policy. Container and
//! SHA-bound `jwgp` acceleration-index validation is performed by `jxl_gpu_bitstream`. The stock
//! [`WgpuSubmissionEngine`] decodes the indexed single-group lossless Gray8 profile from actual
//! `jxlc` token bits in WGSL and returns GPU-resident frames in the requested generic
//! [`PixelFormat`]. Unsupported profiles and incomplete generic frontend stages are typed errors.

#![deny(unsafe_code)]
// Keep one Arc-based GPU output API across native and browser WebGPU. WebGPU handles are
// main-thread-local on wasm32, so Clippy's cross-thread Arc heuristic does not apply there.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]

mod error;
mod inflight;
mod model;
mod profile;
mod session;
mod wgpu_engine;

pub use error::{
    Error, FrontendIncomplete, FrontendStage, Result, UnsupportedCodestreamFeature,
    UnsupportedProfile,
};
pub use inflight::{Acquire, InFlightLimiter, InFlightPermit};
pub use jxl_gpu_bitstream::ParseLimits;
pub use jxl_gpu_formats::{ImageLayout, PixelFormat};
pub use model::{
    AnimationMetadata, DecodeProfile, FixedModularPredictor, FrameDuration, FrameMetadata,
    FrameTimebase, GpuCodestream, GpuOutputRequest, ModularGrouping,
};
pub use session::{
    GpuDecodeSession, GpuDecoder, GpuFrameLease, GpuSubmissionEngine, GpuSubmissionSession,
    NextGpuFrame, PreparedGpuSession, SubmittedGpuFrame,
};
pub use wgpu_engine::{WgpuDecodeMemoryStats, WgpuDecodeSession, WgpuSubmissionEngine};
