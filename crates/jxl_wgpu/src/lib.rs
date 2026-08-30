// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]
// The public zero-copy contract intentionally uses Arc on every target so callers can retain the
// same output type. Browser WebGPU handles are single-threaded even though native handles are
// Send + Sync, making Clippy's generic Arc heuristic inapplicable on wasm32.
#![cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]

//! A portable `wgpu` implementation of [`jxl_gpu_protocol::RenderBackend`].
//!
//! Fixed CPU/WGSL ABI records use `#[repr(C)]` plus `bytemuck::Pod`. WGSL defines
//! host-shareable numeric values as little-endian, so this crate rejects big-endian targets at
//! compile time rather than silently reinterpreting native-endian `Pod` bytes.

#[cfg(not(target_endian = "little"))]
compile_error!(
    "jxl_wgpu requires a little-endian target because WGSL host-shareable buffer values are little-endian"
);

mod arena;
mod autotune;
mod buffer_pool;
#[cfg(not(target_arch = "wasm32"))]
mod capability;
mod context;
mod display;
mod error;
mod memory_budget;
mod metrics;
mod pipeline_cache;
mod planner;
mod poller;
mod readback;
mod scheduler;
mod session;
mod upload;
mod vardct;
mod vardct_general;
mod video;

pub use arena::{ArenaAllocation, ArenaPlan, ArenaPlanner};
pub use autotune::{AdapterFingerprint, AutotuneProfile, KernelVariant, TunedKernel};
pub use buffer_pool::WgpuBufferPoolStats;
pub use context::{
    DirectReadbackPolicy, ShaderF64Policy, WgpuBackend, WgpuBackendConfig, WgpuMemoryPolicy,
};
pub use display::{
    DisplayColorEncoding, DisplayPipeline, DisplayPipelineCacheStats, DisplaySubmission,
    DisplayTexture, DisplayTextureDescriptor, NumericDisplayChannels, NumericDisplayClamp,
    NumericDisplayContract, NumericDisplayError, NumericDisplayPrecision, NumericDisplaySource,
    NumericDisplayTransfer, NumericNonFinitePolicy,
};
pub use error::{Error, Result};
pub use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaLocation, ChromaLocation2d, ChromaOrder, ChromaSubsampling,
    ColorModel, ColorRange, ColorSpace, ColorSpec, ColorSpecification, ImageLayout, Packed422Order,
    PackingField, PackingFieldKind, PackingWord, PitchLinearPlaneLayout, PixelFormat, PlaneFormat,
    PlaneSampling, RgbChannelOrder, SampleKind, Swizzle, SwizzleComponent, TransferFunction,
    YcbcrEncoding,
};
pub use jxl_gpu_protocol::{OutputColorEncoding, RgbColorEncoding, RgbPrimaries};
pub use memory_budget::{MemoryBudget, MemoryBudgetError, MemoryBudgetSnapshot, MemoryPermit};
pub use metrics::{AccuracyMetrics, TimingBreakdown};
pub use planner::{ExecutionPlan, FusedKernel, PlannedDispatch, Planner};
pub use poller::{
    SUBMISSION_POLLER_CAPACITY, SubmissionPollPermit, SubmissionPoller, SubmissionPollerError,
};
pub use readback::{
    ImageReadbackBatchResult, ImageReadbackBatchSubmission, ImageReadbackLimits,
    ImageReadbackPipeline, ImageReadbackResult, ImageReadbackStats, ImageReadbackSubmission,
    UnvalidatedImageReadbackResult, UnvalidatedImageReadbackSubmission,
};
pub use session::{
    GpuFrame, GpuOutputBuffer, SubmissionMode, WgpuFrameSession, WgpuSubmissionStats,
};
pub use video::{
    CpuImageFrame, CpuImageOutput, GpuBufferLease, GpuImageFrame, GpuImageOutput,
    ImageOutputRequest, UnvalidatedGpuImageFrame, UnvalidatedGpuImageOutput,
};
