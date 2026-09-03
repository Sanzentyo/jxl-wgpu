//! Coding-mode-neutral GPU decode selection.
//!
//! This layer owns the one public mode decision. Modular and VarDCT keep independent pipeline
//! caches and submission state because their storage bindings and render phases are intentionally
//! different. Both paths share the backend-wide byte budget and return the same GPU frame type.

use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    CodestreamInventory, FrameEncoding, FrameInventory, FrameType, InventoryLimits, ParseLimits,
};
use jxl_wgpu::{
    GpuImageFrame, MemoryBudget, MemoryBudgetSnapshot, UnvalidatedGpuImageFrame, WgpuBackend,
};

use crate::{
    Error, GpuCodestream, GpuDecoder, GpuOutputRequest, GpuPendingFrame, GpuSubmissionEngine,
    GpuSubmissionSession, PreparedGpuSession, ProgressiveDcError, Result, SubmittedGpuFrame,
    SubsampledAdaptiveLfPolicy, VarDctDecodeSession, VarDctPendingFrame, VarDctSubmissionEngine,
    WgpuDecodeSession, WgpuPendingFrame, WgpuSubmissionEngine,
};

/// One physical frame in a progressive-DC dependency chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgressiveDcStage {
    pub frame_index: u32,
    pub source_frame: Option<u32>,
    pub encoding: FrameEncoding,
    pub lf_level: u32,
    pub width: u32,
    pub height: u32,
    pub is_final: bool,
}

/// Coarse-to-fine execution order for one final still image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgressiveDcPlan {
    pub stages: Vec<ProgressiveDcStage>,
}

impl ProgressiveDcPlan {
    /// Resolves a complete recursive LF chain. Ordinary one-frame stills return `Ok(None)`.
    pub fn negotiate(
        inventory: &CodestreamInventory,
    ) -> std::result::Result<Option<Self>, ProgressiveDcError> {
        negotiate_progressive_dc(&inventory.frames)
    }
}

fn negotiate_progressive_dc(
    frames: &[FrameInventory],
) -> std::result::Result<Option<ProgressiveDcPlan>, ProgressiveDcError> {
    if frames.len() == 1
        && frames[0].frame_type == FrameType::Regular
        && frames[0].lf_source_frame.is_none()
    {
        return Ok(None);
    }
    let final_indices = frames
        .iter()
        .enumerate()
        .filter_map(|(index, frame)| frame.is_last.then_some(index))
        .collect::<Vec<_>>();
    let [final_index] = final_indices.as_slice() else {
        return Err(ProgressiveDcError::MissingFinalFrame);
    };
    if *final_index + 1 != frames.len() {
        return Err(ProgressiveDcError::MissingFinalFrame);
    }
    let final_frame = &frames[*final_index];
    if final_frame.frame_type != FrameType::Regular
        || final_frame.encoding != FrameEncoding::VarDct
        || final_frame.lf_level != 0
        || final_frame.lf_source_frame.is_none()
    {
        return Err(ProgressiveDcError::UnsupportedFrame {
            frame_index: final_frame.frame_index,
        });
    }

    let mut visited = vec![false; frames.len()];
    let mut reverse = Vec::with_capacity(frames.len());
    let mut cursor = *final_index;
    loop {
        let frame = &frames[cursor];
        if visited[cursor] {
            return Err(ProgressiveDcError::DependencyCycle {
                frame_index: frame.frame_index,
            });
        }
        visited[cursor] = true;
        let (width, height) =
            frame
                .color_sample_extent()
                .ok_or(ProgressiveDcError::InvalidExtent {
                    frame_index: frame.frame_index,
                })?;
        reverse.push(ProgressiveDcStage {
            frame_index: frame.frame_index,
            source_frame: frame.lf_source_frame,
            encoding: frame.encoding,
            lf_level: frame.lf_level,
            width,
            height,
            is_final: cursor == *final_index,
        });

        let Some(source_index) = frame.lf_source_frame else {
            break;
        };
        let source_index_usize =
            usize::try_from(source_index).map_err(|_| ProgressiveDcError::InvalidSource {
                frame_index: frame.frame_index,
                source_frame: source_index,
            })?;
        if source_index_usize >= cursor
            || frames
                .get(source_index_usize)
                .is_none_or(|source| source.frame_index != source_index)
        {
            return Err(ProgressiveDcError::InvalidSource {
                frame_index: frame.frame_index,
                source_frame: source_index,
            });
        }
        let source = &frames[source_index_usize];
        if source.frame_type != FrameType::LowFrequency
            || source.is_last
            || frame
                .lf_level
                .checked_add(1)
                .is_none_or(|source_level| source.lf_level != source_level)
            || (frame.encoding != FrameEncoding::VarDct)
        {
            return Err(ProgressiveDcError::UnsupportedFrame {
                frame_index: frame.frame_index,
            });
        }
        let (source_width, source_height) =
            source
                .color_sample_extent()
                .ok_or(ProgressiveDcError::InvalidExtent {
                    frame_index: source.frame_index,
                })?;
        let required_width = width.div_ceil(8);
        let required_height = height.div_ceil(8);
        if (source_width, source_height) != (required_width, required_height) {
            return Err(ProgressiveDcError::ExtentMismatch {
                frame_index: frame.frame_index,
                source_frame: source.frame_index,
                source_width,
                source_height,
                required_width,
                required_height,
            });
        }
        cursor = source_index_usize;
    }

    if let Some((frame_index, _)) = visited.iter().enumerate().find(|(_, used)| !**used) {
        return Err(ProgressiveDcError::UnusedFrame {
            frame_index: frames[frame_index].frame_index,
        });
    }
    reverse.reverse();
    Ok(Some(ProgressiveDcPlan { stages: reverse }))
}

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

    /// Sets the policy for handling unstandardized subsampled Adaptive LF smoothing.
    #[must_use]
    pub fn with_subsampled_adaptive_lf_policy(
        mut self,
        policy: SubsampledAdaptiveLfPolicy,
    ) -> Self {
        self.vardct = self.vardct.with_subsampled_adaptive_lf_policy(policy);
        self
    }

    #[must_use]
    pub const fn subsampled_adaptive_lf_policy(&self) -> SubsampledAdaptiveLfPolicy {
        self.vardct.subsampled_adaptive_lf_policy()
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
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &jxl_gpu_bitstream::CodestreamInventory,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        if let Some(plan) = ProgressiveDcPlan::negotiate(inventory)? {
            return self.open_progressive_dc(codestream, request, inventory, plan);
        }
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

    fn open_progressive_dc(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
        plan: ProgressiveDcPlan,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        validate_codestream_limit(codestream.logical_bytes(), self.parse_limits())?;
        let mut stages = Vec::with_capacity(plan.stages.len());
        let mut submissions_per_frame = 0_usize;
        let mut final_profile = None;
        let mut final_metadata = None;
        for stage in plan.stages {
            let stage_inventory = project_frame_inventory(inventory, stage.frame_index)?;
            match stage.encoding {
                FrameEncoding::Modular => {
                    let prepared = self.modular.open_progressive_dc_with_inventory_data(
                        Arc::clone(&codestream),
                        request,
                        &stage_inventory,
                    )?;
                    submissions_per_frame = submissions_per_frame
                        .checked_add(prepared.session.memory_stats().submissions_per_frame)
                        .ok_or(Error::EngineContract(
                            "progressive-DC submission count overflowed",
                        ))?;
                    if stage.is_final {
                        final_profile = Some(prepared.profile);
                        final_metadata = Some(prepared.metadata.clone());
                    }
                    stages.push(ProgressiveDcStageSession::Modular(Box::new(
                        prepared.session,
                    )));
                }
                FrameEncoding::VarDct => {
                    let prepared = self.vardct.open_progressive_dc_with_inventory_data(
                        (*codestream).clone(),
                        request,
                        &stage_inventory,
                        stage.is_final,
                    )?;
                    submissions_per_frame = submissions_per_frame
                        .checked_add(prepared.session.submissions_per_frame())
                        .ok_or(Error::EngineContract(
                            "progressive-DC submission count overflowed",
                        ))?;
                    if stage.is_final {
                        final_profile = Some(prepared.profile);
                        final_metadata = Some(prepared.metadata.clone());
                    }
                    stages.push(ProgressiveDcStageSession::VarDct(Box::new(
                        prepared.session,
                    )));
                }
            }
        }
        let profile = final_profile.ok_or(Error::EngineContract(
            "progressive-DC plan has no final presentation stage",
        ))?;
        let metadata = final_metadata.ok_or(Error::EngineContract(
            "progressive-DC plan has no final presentation metadata",
        ))?;
        Ok(PreparedGpuSession::new(
            profile,
            metadata,
            WgpuDecodeSubmissionSession::ProgressiveDc(Box::new(ProgressiveDcSubmissionSession {
                stages: Some(stages),
                submissions_per_frame: Arc::new(AtomicUsize::new(submissions_per_frame)),
            })),
        )
        .with_resolved_frame_slots(NonZeroUsize::new(1).expect("one is nonzero")))
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

    fn inventory_limits(&self) -> InventoryLimits {
        InventoryLimits {
            // Four LF slots plus one final frame are representable by the JPEG XL frame header.
            max_frames: 5,
            max_total_section_bytes: self.parse_limits().max_codestream_bytes,
            ..InventoryLimits::default()
        }
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
        inventory: Arc<jxl_gpu_bitstream::CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        let codestream = Arc::new(codestream);
        self.open_with_inventory_data(codestream, request, &inventory)
    }
}

fn project_frame_inventory(
    inventory: &CodestreamInventory,
    frame_index: u32,
) -> Result<CodestreamInventory> {
    let frame = inventory
        .frames
        .iter()
        .find(|frame| frame.frame_index == frame_index)
        .cloned()
        .ok_or(Error::EngineContract(
            "progressive-DC plan references a missing physical frame",
        ))?;
    let mut projected = inventory.clone();
    projected.frames = vec![frame];
    Ok(projected)
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
    ProgressiveDc(Box<ProgressiveDcSubmissionSession>),
}

enum ProgressiveDcStageSession {
    Modular(Box<WgpuDecodeSession>),
    VarDct(Box<VarDctDecodeSession>),
}

pub struct ProgressiveDcSubmissionSession {
    stages: Option<Vec<ProgressiveDcStageSession>>,
    submissions_per_frame: Arc<AtomicUsize>,
}

impl WgpuDecodeSubmissionSession {
    #[must_use]
    pub const fn modular(&self) -> Option<&WgpuDecodeSession> {
        match self {
            Self::Modular(session) => Some(session),
            Self::VarDct(_) | Self::ProgressiveDc(_) => None,
        }
    }

    #[must_use]
    pub const fn vardct(&self) -> Option<&VarDctDecodeSession> {
        match self {
            Self::Modular(_) | Self::ProgressiveDc(_) => None,
            Self::VarDct(session) => Some(session),
        }
    }

    /// Number of queue submissions issued for one visible frame in the selected mode. Staged
    /// VarDCT and recursive progressive-DC sessions update this count after a cursor map determines
    /// the exact packet or AC window plan.
    #[must_use]
    pub fn submissions_per_frame(&self) -> usize {
        match self {
            Self::Modular(session) => session.memory_stats().submissions_per_frame,
            Self::VarDct(session) => session.submissions_per_frame(),
            Self::ProgressiveDc(session) => session.submissions_per_frame.load(Ordering::Acquire),
        }
    }
}

impl std::fmt::Debug for WgpuDecodeSubmissionSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modular(session) => formatter.debug_tuple("Modular").field(session).finish(),
            Self::VarDct(session) => formatter.debug_tuple("VarDct").field(session).finish(),
            Self::ProgressiveDc(session) => formatter
                .debug_struct("ProgressiveDc")
                .field("submitted", &session.stages.is_none())
                .field(
                    "submissions_per_frame",
                    &session.submissions_per_frame.load(Ordering::Acquire),
                )
                .finish(),
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
            Self::ProgressiveDc(session) => session.submit_next(),
        }
    }
}

impl ProgressiveDcSubmissionSession {
    fn submit_next(&mut self) -> Result<Option<WgpuDecodePendingFrame>> {
        let Some(stages) = self.stages.take() else {
            return Ok(None);
        };
        let stage_count = stages.len();
        let mut remaining = VecDeque::from(stages);
        let root = remaining
            .pop_front()
            .ok_or(Error::EngineContract("progressive-DC plan has no stages"))?;
        let ProgressiveDcStageSession::Modular(mut root) = root else {
            return Err(Error::EngineContract(
                "a progressive-DC plan must start with a Modular LF producer",
            ));
        };
        let root_pending = root.submit_next()?.ok_or(Error::EngineContract(
            "progressive-DC Modular stage produced no submission",
        ))?;
        let dependency = root_pending.progressive_dc_planes()?;
        let mut pending = ProgressiveDcPendingFrame {
            hidden: VecDeque::with_capacity(stage_count.saturating_sub(1)),
            active_dependency: None,
            remaining,
            final_pending: None,
            final_planned_submissions: None,
            final_submission_counter: None,
            submissions_per_frame: Arc::clone(&self.submissions_per_frame),
        };
        pending
            .hidden
            .push_back(ProgressiveDcPhysicalPending::Modular(root_pending));
        pending.submit_ready_stages(dependency)?;
        Ok(Some(WgpuDecodePendingFrame::ProgressiveDc(Box::new(
            pending,
        ))))
    }
}

/// One submitted frame from either stock GPU coding-mode pipeline.
pub enum WgpuDecodePendingFrame {
    Modular(WgpuPendingFrame),
    VarDct(Box<VarDctPendingFrame>),
    ProgressiveDc(Box<ProgressiveDcPendingFrame>),
}

enum ProgressiveDcPhysicalPending {
    Modular(WgpuPendingFrame),
    VarDct(Box<VarDctPendingFrame>),
}

pub struct ProgressiveDcPendingFrame {
    hidden: VecDeque<ProgressiveDcPhysicalPending>,
    active_dependency: Option<ProgressiveDcActiveDependency>,
    remaining: VecDeque<ProgressiveDcStageSession>,
    final_pending: Option<Box<VarDctPendingFrame>>,
    final_planned_submissions: Option<usize>,
    final_submission_counter: Option<Arc<AtomicUsize>>,
    submissions_per_frame: Arc<AtomicUsize>,
}

struct ProgressiveDcActiveDependency {
    pending: Box<VarDctPendingFrame>,
    planned_submissions: usize,
}

impl ProgressiveDcPendingFrame {
    fn submit_ready_stages(
        &mut self,
        mut dependency: crate::progressive_dc::ProgressiveDcXybPlanes,
    ) -> Result<()> {
        while let Some(stage) = self.remaining.pop_front() {
            let is_final = self.remaining.is_empty();
            let ProgressiveDcStageSession::VarDct(mut session) = stage else {
                return Err(Error::EngineContract(
                    "only the root progressive-DC stage may use Modular encoding",
                ));
            };
            let planned_submissions = session.submissions_per_frame();
            session.set_progressive_dc_source(dependency)?;
            let pending = session.submit_next()?.ok_or(Error::EngineContract(
                "progressive-DC VarDCT stage produced no submission",
            ))?;
            if is_final {
                self.final_submission_counter = Some(pending.submissions_per_frame_counter());
                self.final_planned_submissions = Some(planned_submissions);
                self.final_pending = Some(Box::new(pending));
                return Ok(());
            }
            if pending.dependency_submission_ready() {
                self.reconcile_submission_count(
                    planned_submissions,
                    pending
                        .submissions_per_frame_counter()
                        .load(Ordering::Acquire),
                )?;
                dependency = pending.progressive_dc_planes()?;
                self.hidden
                    .push_back(ProgressiveDcPhysicalPending::VarDct(Box::new(pending)));
            } else {
                self.active_dependency = Some(ProgressiveDcActiveDependency {
                    pending: Box::new(pending),
                    planned_submissions,
                });
                return Ok(());
            }
        }
        Err(Error::EngineContract(
            "progressive-DC chain did not submit a final presentation frame",
        ))
    }

    fn reconcile_submission_count(&self, planned: usize, actual: usize) -> Result<()> {
        self.submissions_per_frame
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_sub(planned)?.checked_add(actual)
            })
            .map(|_| ())
            .map_err(|_| Error::EngineContract("progressive-DC submission count update overflowed"))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn submit_deferred_dependencies_blocking(&mut self) -> Result<()> {
        while let Some(mut active) = self.active_dependency.take() {
            active.pending.wait_until_dependency_submitted()?;
            self.reconcile_submission_count(
                active.planned_submissions,
                active
                    .pending
                    .submissions_per_frame_counter()
                    .load(Ordering::Acquire),
            )?;
            let dependency = active.pending.progressive_dc_planes()?;
            self.hidden
                .push_back(ProgressiveDcPhysicalPending::VarDct(active.pending));
            self.submit_ready_stages(dependency)?;
        }
        Ok(())
    }

    fn poll_deferred_dependencies(&mut self, context: &mut Context<'_>) -> Poll<Result<()>> {
        while let Some(active) = self.active_dependency.as_mut() {
            match active.pending.poll_until_dependency_submitted(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(())) => {}
            }
            let Some(active) = self.active_dependency.take() else {
                return Poll::Ready(Err(Error::EngineContract(
                    "progressive-DC active dependency disappeared while being progressed",
                )));
            };
            if let Err(error) = self.reconcile_submission_count(
                active.planned_submissions,
                active
                    .pending
                    .submissions_per_frame_counter()
                    .load(Ordering::Acquire),
            ) {
                return Poll::Ready(Err(error));
            }
            let dependency = match active.pending.progressive_dc_planes() {
                Ok(dependency) => dependency,
                Err(error) => return Poll::Ready(Err(error.into())),
            };
            self.hidden
                .push_back(ProgressiveDcPhysicalPending::VarDct(active.pending));
            if let Err(error) = self.submit_ready_stages(dependency) {
                return Poll::Ready(Err(error));
            }
        }
        Poll::Ready(Ok(()))
    }
}

impl WgpuDecodePendingFrame {
    /// Same-queue, budget-tracked output access before validation completes.
    pub fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match self {
            Self::Modular(pending) => pending.unvalidated_gpu_frame(),
            Self::VarDct(pending) => pending.unvalidated_gpu_frame(),
            Self::ProgressiveDc(pending) => pending
                .final_pending
                .as_ref()
                .ok_or(crate::vardct_engine::VarDctDecodeError::UnvalidatedOutputNotSubmitted)?
                .unvalidated_gpu_frame(),
        }
    }
}

impl std::fmt::Debug for WgpuDecodePendingFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Modular(pending) => formatter.debug_tuple("Modular").field(pending).finish(),
            Self::VarDct(pending) => formatter.debug_tuple("VarDct").field(pending).finish(),
            Self::ProgressiveDc(pending) => formatter
                .debug_struct("ProgressiveDc")
                .field("hidden_pending", &pending.hidden.len())
                .field(
                    "has_active_dependency",
                    &pending.active_dependency.is_some(),
                )
                .field("remaining_stages", &pending.remaining.len())
                .field("has_final", &pending.final_pending.is_some())
                .finish(),
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
            Self::ProgressiveDc(pending) => pending.wait(),
        }
    }

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        match self.get_mut() {
            Self::Modular(pending) => Pin::new(pending).poll_complete(context),
            Self::VarDct(pending) => Pin::new(pending.as_mut()).poll_complete(context),
            Self::ProgressiveDc(pending) => Pin::new(pending.as_mut()).poll_complete(context),
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
            Self::ProgressiveDc(pending) => Pin::new(pending.as_mut()).poll_complete(context),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for ProgressiveDcPendingFrame {
    type Frame = GpuImageFrame;

    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        self.submit_deferred_dependencies_blocking()?;
        while let Some(pending) = self.hidden.pop_front() {
            match pending {
                ProgressiveDcPhysicalPending::Modular(pending) => {
                    drop(pending.wait()?);
                }
                ProgressiveDcPhysicalPending::VarDct(pending) => {
                    drop((*pending).wait()?);
                }
            }
        }
        let final_pending = self.final_pending.take().ok_or(Error::EngineContract(
            "progressive-DC final frame completion was already consumed",
        ))?;
        let result = final_pending.wait();
        let planned = self
            .final_planned_submissions
            .take()
            .ok_or(Error::EngineContract(
                "progressive-DC final frame has no planned submission count",
            ))?;
        let actual = self
            .final_submission_counter
            .take()
            .ok_or(Error::EngineContract(
                "progressive-DC final frame has no submission counter",
            ))?
            .load(Ordering::Acquire);
        self.reconcile_submission_count(planned, actual)?;
        result
    }

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        poll_progressive_dc(self.get_mut(), context)
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for ProgressiveDcPendingFrame {
    type Frame = GpuImageFrame;

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        poll_progressive_dc(self.get_mut(), context)
    }
}

fn poll_progressive_dc(
    pending: &mut ProgressiveDcPendingFrame,
    context: &mut Context<'_>,
) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
    match pending.poll_deferred_dependencies(context) {
        Poll::Pending => return Poll::Pending,
        Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
        Poll::Ready(Ok(())) => {}
    }
    while let Some(front) = pending.hidden.front_mut() {
        let result = match front {
            ProgressiveDcPhysicalPending::Modular(frame) => Pin::new(frame).poll_complete(context),
            ProgressiveDcPhysicalPending::VarDct(frame) => {
                Pin::new(frame.as_mut()).poll_complete(context)
            }
        };
        match result {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(frame)) => {
                drop(frame);
                pending.hidden.pop_front();
            }
            Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
        }
    }
    let Some(final_pending) = pending.final_pending.as_mut() else {
        return Poll::Ready(Err(Error::EngineContract(
            "progressive-DC final frame completion was already consumed",
        )));
    };
    let result = Pin::new(final_pending.as_mut()).poll_complete(context);
    if result.is_ready() {
        pending.final_pending = None;
        let Some(planned) = pending.final_planned_submissions.take() else {
            return Poll::Ready(Err(Error::EngineContract(
                "progressive-DC final frame has no planned submission count",
            )));
        };
        let Some(counter) = pending.final_submission_counter.take() else {
            return Poll::Ready(Err(Error::EngineContract(
                "progressive-DC final frame has no submission counter",
            )));
        };
        if let Err(error) =
            pending.reconcile_submission_count(planned, counter.load(Ordering::Acquire))
        {
            return Poll::Ready(Err(error));
        }
    }
    result
}

impl GpuDecoder<WgpuDecodeEngine> {
    /// Constructs the GPU-only decoder that selects Modular or VarDCT automatically.
    pub fn wgpu(backend: WgpuBackend) -> Result<Self> {
        Ok(Self::new(WgpuDecodeEngine::new(backend)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_frame() -> FrameInventory {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }
        let digits = include_str!("../test-data/basic.jxl.hex")
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        let encoded = digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect::<Vec<_>>();
        let parsed = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default()).unwrap();
        parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap()
            .frames
            .remove(0)
    }

    fn progressive_frames() -> Vec<FrameInventory> {
        let mut root = base_frame();
        root.frame_index = 0;
        root.frame_type = FrameType::LowFrequency;
        root.encoding = FrameEncoding::Modular;
        root.lf_level = 2;
        root.lf_source_frame = None;
        root.width = 1_024;
        root.height = 128;
        root.is_last = false;

        let mut refinement = root.clone();
        refinement.frame_index = 1;
        refinement.encoding = FrameEncoding::VarDct;
        refinement.lf_level = 1;
        refinement.lf_source_frame = Some(0);

        let mut final_frame = refinement.clone();
        final_frame.frame_index = 2;
        final_frame.frame_type = FrameType::Regular;
        final_frame.lf_level = 0;
        final_frame.lf_source_frame = Some(1);
        final_frame.is_last = true;
        vec![root, refinement, final_frame]
    }

    #[test]
    fn recursive_progressive_dc_plan_is_coarse_to_fine() {
        let plan = negotiate_progressive_dc(&progressive_frames())
            .unwrap()
            .unwrap();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (
                    stage.frame_index,
                    stage.source_frame,
                    stage.encoding,
                    stage.lf_level,
                    (stage.width, stage.height),
                    stage.is_final,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, None, FrameEncoding::Modular, 2, (16, 2), false),
                (1, Some(0), FrameEncoding::VarDct, 1, (128, 16), false),
                (2, Some(1), FrameEncoding::VarDct, 0, (1_024, 128), true),
            ]
        );
    }

    #[test]
    fn ordinary_single_frame_is_not_a_progressive_dc_plan() {
        assert_eq!(negotiate_progressive_dc(&[base_frame()]).unwrap(), None);
    }

    #[test]
    fn progressive_dc_plan_rejects_extent_and_ownership_mismatches() {
        let mut mismatched = progressive_frames();
        mismatched[0].width = 2_048;
        assert!(matches!(
            negotiate_progressive_dc(&mismatched),
            Err(ProgressiveDcError::ExtentMismatch {
                frame_index: 1,
                source_frame: 0,
                ..
            })
        ));

        let mut unused = progressive_frames();
        let mut detached = unused[0].clone();
        detached.frame_index = 2;
        unused.insert(2, detached);
        unused[3].frame_index = 3;
        assert!(matches!(
            negotiate_progressive_dc(&unused),
            Err(ProgressiveDcError::UnusedFrame { frame_index: 2 })
        ));

        let mut invalid_source = progressive_frames();
        invalid_source[2].lf_source_frame = Some(2);
        assert!(matches!(
            negotiate_progressive_dc(&invalid_source),
            Err(ProgressiveDcError::InvalidSource {
                frame_index: 2,
                source_frame: 2,
            })
        ));
    }
}
