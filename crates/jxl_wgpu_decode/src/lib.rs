//! GPU-required JPEG XL decode orchestration.
//!
//! This crate has no production CPU pixel/entropy decoder and no fallback policy. The stock
//! [`WgpuSubmissionEngine`] inventories standard JPEG XL frame sections, parses only bounded
//! Modular prefix metadata, and decodes single- or multi-group lossless 1-16-bit Gray/RGB/RGBA
//! token streams in WGSL. Single-root-group streams may use arbitrary resident
//! RCT/Palette/Squeeze stacks; no decoded sample crosses a CPU or mapped-buffer boundary.
//! It returns GPU-resident frames in the requested generic [`PixelFormat`]; no private sidecar box
//! is required. Unsupported profiles and incomplete generic frontend stages are typed errors.
//! Submission and completion are separate: sessions can prefetch an ordered bounded queue of
//! [`GpuPendingFrame`] values before synchronously waiting or runtime-neutrally polling its front.
//!
//! CPU/WGSL transport structs use fixed `repr(C)` + `bytemuck::Pod` layouts. WGSL host-shareable
//! scalar bytes are little-endian, so this crate deliberately supports little-endian hosts only;
//! that same byte order pins portable and native F64 output word order.

#![deny(unsafe_code)]
// Keep one Arc-based GPU output API across native and browser WebGPU. WebGPU handles are
// main-thread-local on wasm32, so Clippy's cross-thread Arc heuristic does not apply there.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]

#[cfg(not(target_endian = "little"))]
compile_error!(
    "jxl_wgpu_decode requires a little-endian host because WGSL host-shareable values are little-endian while bytemuck uses native-endian repr(C) values"
);

mod buffer_pool;
mod codec_engine;
mod codestream_data;
mod entropy;
mod entropy_window;
mod error;
mod inflight;
mod input_budget;
mod model;
mod modular_finalize;
mod modular_inverse;
mod modular_palette;
pub mod modular_rct;
pub mod modular_squeeze;
mod modular_transform;
mod modular_tree;
mod profile;
mod progressive_dc;
mod session;
mod vardct_artifact;
mod vardct_engine;
mod vardct_epf;
mod vardct_frontend;
mod vardct_lf;
mod vardct_output;
mod vardct_packet;
mod vardct_pass_group;
mod vardct_resource;
mod vardct_side_image;
mod wgpu_engine;

pub mod vardct;

pub use buffer_pool::{
    DEFAULT_DECODE_BUFFER_POOL_BUFFERS, DEFAULT_DECODE_BUFFER_POOL_BUFFERS_PER_KEY,
    DEFAULT_DECODE_BUFFER_POOL_BYTES, WgpuDecodeBufferPoolLimits, WgpuDecodeBufferPoolStats,
};
pub use codec_engine::{
    ProgressiveDcPlan, ProgressiveDcStage, WgpuDecodeEngine, WgpuDecodePendingFrame,
    WgpuDecodeSubmissionSession,
};
pub use codestream_data::GpuCodestream;
pub use error::{
    Error, FrontendIncomplete, FrontendStage, ModularInversePlanError, ModularTransformError,
    ModularTransformFeature, ModularTreeError, ProgressiveDcError, Result,
    UnsupportedCodestreamFeature, UnsupportedProfile,
};
pub use inflight::{Acquire, InFlightLimiter, InFlightPermit};
pub use input_budget::{
    IncrementalInputBudget, IncrementalInputBudgetError, IncrementalInputBudgetSnapshot,
};
pub use jxl_gpu_bitstream::ParseLimits;
pub use jxl_gpu_formats::{ImageLayout, PixelFormat};
pub use jxl_wgpu::{UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput};
pub use model::{
    AnimationMetadata, DecodeProfile, F64OutputPolicy, FrameDuration, FrameMetadata, FrameTimebase,
    GpuOutputMapping, GpuOutputRequest, ModularChannels, ModularGrouping, ModularPredictionProfile,
    ModularPredictor, NumericSampleMapping,
};
pub use modular_finalize::ModularFinalizeError;
pub use modular_palette::ModularPaletteError;
pub use progressive_dc::ProgressiveDcGpuError;
pub use session::{
    GpuDecodeSession, GpuDecodeStream, GpuDecodeStreamStats, GpuDecoder, GpuFrameLease,
    GpuPendingFrame, GpuSubmissionEngine, GpuSubmissionSession, NextGpuFrame, PrefetchBackpressure,
    PrefetchGpuFrames, PrefetchProgress, PreparedGpuSession, SubmittedGpuFrame,
};
pub use vardct_engine::{
    VarDctDecodeError, VarDctDecodeMemoryStats, VarDctDecodeSession, VarDctPendingFrame,
    VarDctSubmissionEngine, vardct_rgb8_format,
};
pub use wgpu_engine::{
    F64OutputPath, ModularEntropyCoding, ModularOutputSpecialization,
    ModularReconstructionSpecialization, OutputWritePath, WgpuDecodeCapabilities,
    WgpuDecodeMemoryStats, WgpuDecodeSession, WgpuPendingFrame, WgpuSubmissionEngine,
};
