//! Coding-mode-neutral GPU decode selection.
//!
//! This layer owns the one public mode decision. Modular and VarDCT keep independent pipeline
//! caches and submission state because their storage bindings and render phases are intentionally
//! different. Both paths share the backend-wide byte budget and return the same GPU frame type.

use std::num::NonZeroU64;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{FrameEncoding, InventoryLimits, ParseLimits};
use jxl_wgpu::{
    GpuImageFrame, MemoryBudget, MemoryBudgetSnapshot, UnvalidatedGpuImageFrame, WgpuBackend,
};

use crate::{
    Error, GpuCodestream, GpuDecoder, GpuOutputRequest, GpuPendingFrame, GpuSubmissionEngine,
    GpuSubmissionSession, PreparedGpuSession, Result, SubmittedGpuFrame, VarDctDecodeSession,
    VarDctPendingFrame, VarDctSubmissionEngine, WgpuDecodeSession, WgpuPendingFrame,
    WgpuSubmissionEngine, codestream_data::CodestreamData,
};

/// Stock GPU decoder engine that selects Modular or VarDCT from the standard frame header.
#[derive(Clone)]
pub struct WgpuDecodeEngine {
    modular: WgpuSubmissionEngine,
    vardct: VarDctSubmissionEngine,
    memory: MemoryBudget,
}

impl WgpuDecodeEngine {
    /// Builds both coding-mode pipeline sets around one backend and one aggregate byte budget.
    pub fn new(backend: WgpuBackend) -> Result<Self> {
        let memory = backend.transient_memory_budget().clone();
        let modular = WgpuSubmissionEngine::with_memory_budget(backend.clone(), memory.clone());
        let vardct = VarDctSubmissionEngine::with_memory_budget(backend, memory.clone())?;
        Ok(Self {
            modular,
            vardct,
            memory,
        })
    }

    #[must_use]
    pub const fn backend(&self) -> &WgpuBackend {
        self.modular.backend()
    }

    /// Aggregate reservations retained by Modular, VarDCT, readback, and output leases.
    #[must_use]
    pub fn in_flight_memory_stats(&self) -> MemoryBudgetSnapshot {
        self.memory.snapshot()
    }

    #[must_use]
    pub fn memory_budget_bytes(&self) -> u64 {
        self.memory.snapshot().limit_bytes
    }

    /// Applies one entropy-window cap to both coding-mode engines.
    #[must_use]
    pub fn with_stream_window_limit(mut self, limit: NonZeroU64) -> Self {
        self.modular = self.modular.with_stream_window_limit(limit);
        self.vardct = self.vardct.with_stream_window_limit(limit);
        self
    }

    /// Explicit low-level Modular engine access for diagnostics and cache policy.
    #[must_use]
    pub const fn modular_engine(&self) -> &WgpuSubmissionEngine {
        &self.modular
    }

    /// Explicit low-level VarDCT engine access for diagnostics.
    #[must_use]
    pub const fn vardct_engine(&self) -> &VarDctSubmissionEngine {
        &self.vardct
    }

    pub(crate) fn open_with_inventory_data(
        &self,
        codestream: Arc<CodestreamData>,
        request: &GpuOutputRequest,
        inventory: &jxl_gpu_bitstream::CodestreamInventory,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        let encoding = inventory
            .frames
            .first()
            .ok_or(Error::MissingImageFrame)?
            .encoding;
        match encoding {
            FrameEncoding::Modular => {
                validate_codestream_limit(codestream.logical_bytes(), self.modular.parse_limits())?;
                map_modular(
                    self.modular
                        .open_with_inventory_data(codestream, request, inventory)?,
                )
            }
            FrameEncoding::VarDct => {
                validate_codestream_limit(codestream.logical_bytes(), self.vardct.parse_limits())?;
                map_vardct(self.vardct.open_with_inventory_data(
                    (*codestream).clone(),
                    request,
                    inventory,
                )?)
            }
        }
    }
}

impl std::fmt::Debug for WgpuDecodeEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WgpuDecodeEngine")
            .field("backend", self.backend())
            .field("memory", &self.memory.snapshot())
            .finish_non_exhaustive()
    }
}

impl GpuSubmissionEngine for WgpuDecodeEngine {
    type Session = WgpuDecodeSubmissionSession;

    fn parse_limits(&self) -> ParseLimits {
        maximum_limits(self.modular.parse_limits(), self.vardct.parse_limits())
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
        let codestream = Arc::new(CodestreamData::from_gpu_codestream(codestream)?);
        self.open_with_inventory_data(codestream, request, &inventory)
    }
}

fn maximum_limits(left: ParseLimits, right: ParseLimits) -> ParseLimits {
    ParseLimits {
        max_input_bytes: left.max_input_bytes.max(right.max_input_bytes),
        max_boxes: left.max_boxes.max(right.max_boxes),
        max_box_bytes: left.max_box_bytes.max(right.max_box_bytes),
        max_codestream_bytes: left.max_codestream_bytes.max(right.max_codestream_bytes),
    }
}

fn validate_codestream_limit(length: u64, limits: ParseLimits) -> Result<()> {
    if length > limits.max_input_bytes || length > limits.max_codestream_bytes {
        return Err(jxl_gpu_bitstream::Error::ResourceLimit("codestream size").into());
    }
    Ok(())
}

fn map_modular(
    prepared: PreparedGpuSession<WgpuDecodeSession>,
) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
    let resolved = prepared.resolved_frame_slots();
    let mut mapped = PreparedGpuSession::new(
        prepared.profile,
        prepared.metadata,
        WgpuDecodeSubmissionSession::Modular(Box::new(prepared.session)),
    );
    if let Some(slots) = resolved {
        mapped = mapped.with_resolved_frame_slots(slots);
    }
    Ok(mapped)
}

fn map_vardct(
    prepared: PreparedGpuSession<VarDctDecodeSession>,
) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
    let resolved = prepared.resolved_frame_slots();
    let mut mapped = PreparedGpuSession::new(
        prepared.profile,
        prepared.metadata,
        WgpuDecodeSubmissionSession::VarDct(Box::new(prepared.session)),
    );
    if let Some(slots) = resolved {
        mapped = mapped.with_resolved_frame_slots(slots);
    }
    Ok(mapped)
}

/// Per-codestream state selected once from the standard frame coding mode.
pub enum WgpuDecodeSubmissionSession {
    Modular(Box<WgpuDecodeSession>),
    VarDct(Box<VarDctDecodeSession>),
}

impl WgpuDecodeSubmissionSession {
    #[must_use]
    pub const fn modular(&self) -> Option<&WgpuDecodeSession> {
        match self {
            Self::Modular(session) => Some(session),
            Self::VarDct(_) => None,
        }
    }

    #[must_use]
    pub const fn vardct(&self) -> Option<&VarDctDecodeSession> {
        match self {
            Self::Modular(_) => None,
            Self::VarDct(session) => Some(session),
        }
    }

    /// Number of queue submissions issued for one visible frame in the selected mode. A staged
    /// VarDCT local-tree session updates this count after its LF cursor map determines the exact HF
    /// packet window plan.
    #[must_use]
    pub fn submissions_per_frame(&self) -> usize {
        match self {
            Self::Modular(session) => session.memory_stats().submissions_per_frame,
            Self::VarDct(session) => session.submissions_per_frame(),
        }
    }
}

impl std::fmt::Debug for WgpuDecodeSubmissionSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modular(session) => formatter.debug_tuple("Modular").field(session).finish(),
            Self::VarDct(session) => formatter.debug_tuple("VarDct").field(session).finish(),
        }
    }
}

impl GpuSubmissionSession for WgpuDecodeSubmissionSession {
    type Frame = GpuImageFrame;
    type Pending = WgpuDecodePendingFrame;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        match self {
            Self::Modular(session) => session
                .submit_next()
                .map(|pending| pending.map(WgpuDecodePendingFrame::Modular)),
            Self::VarDct(session) => session.submit_next().map(|pending| {
                pending.map(|pending| WgpuDecodePendingFrame::VarDct(Box::new(pending)))
            }),
        }
    }
}

/// One submitted frame from either stock GPU coding-mode pipeline.
pub enum WgpuDecodePendingFrame {
    Modular(WgpuPendingFrame),
    VarDct(Box<VarDctPendingFrame>),
}

impl WgpuDecodePendingFrame {
    /// Same-queue, budget-tracked output access before validation completes.
    pub fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match self {
            Self::Modular(pending) => pending.unvalidated_gpu_frame(),
            Self::VarDct(pending) => pending.unvalidated_gpu_frame(),
        }
    }
}

impl std::fmt::Debug for WgpuDecodePendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modular(pending) => formatter.debug_tuple("Modular").field(pending).finish(),
            Self::VarDct(pending) => formatter.debug_tuple("VarDct").field(pending).finish(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for WgpuDecodePendingFrame {
    type Frame = GpuImageFrame;

    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        match self {
            Self::Modular(pending) => pending.wait(),
            Self::VarDct(pending) => (*pending).wait(),
        }
    }

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        match self.get_mut() {
            Self::Modular(pending) => Pin::new(pending).poll_complete(context),
            Self::VarDct(pending) => Pin::new(pending.as_mut()).poll_complete(context),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for WgpuDecodePendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        match self.get_mut() {
            Self::Modular(pending) => Pin::new(pending).poll_complete(context),
            Self::VarDct(pending) => Pin::new(pending.as_mut()).poll_complete(context),
        }
    }
}

impl GpuDecoder<WgpuDecodeEngine> {
    /// Constructs the GPU-only decoder that selects Modular or VarDCT automatically.
    pub fn wgpu(backend: WgpuBackend) -> Result<Self> {
        Ok(Self::new(WgpuDecodeEngine::new(backend)?))
    }
}
