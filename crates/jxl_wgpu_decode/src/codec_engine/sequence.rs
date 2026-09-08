use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding, FrameType};
use jxl_wgpu::{GpuImageFrame, UnvalidatedGpuImageFrame};

use crate::{
    DecodeProfile, Error, FrameExecutionPlan, GpuCodestream, GpuOutputRequest, GpuPendingFrame,
    PreparedGpuSession, Result, SubmittedGpuFrame,
};

use super::composition::{CompositionPending, CompositionSession};
use super::{
    ProgressiveDcPlan, WgpuDecodeEngine, WgpuDecodePendingFrame, WgpuDecodeSubmissionSession,
    map_modular, map_vardct, project_frame_inventory, validate_codestream_limit,
};
use crate::GpuSubmissionEngine;

mod independent;
use independent::{IndependentPending, IndependentSession};

impl WgpuDecodeEngine {
    pub(super) fn open_sequence(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
        plan: FrameExecutionPlan,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        validate_codestream_limit(codestream.logical_bytes(), self.parse_limits())?;
        let (execution, slots) = if super::composition::needs_surface(inventory, request, &plan) {
            (
                SequenceExecution::Composed(CompositionSession::new(
                    self.clone(),
                    codestream,
                    inventory,
                    request,
                    &plan,
                )?),
                request.max_frame_slots(),
            )
        } else {
            let source = Arc::new(SequenceSource {
                engine: self.clone(),
                codestream,
                inventory: inventory.clone(),
                request: request.clone(),
                surface_encodings: None,
            });
            let (session, slots) = IndependentSession::new(source, &plan)?;
            (SequenceExecution::Independent(session), slots)
        };
        Ok(PreparedGpuSession::new(
            DecodeProfile::FrameSequence {
                physical_frames: plan.nodes.len(),
                presentation_frames: plan.presentations.len(),
            },
            plan.metadata.clone(),
            WgpuDecodeSubmissionSession::Sequence(Box::new(FrameSequenceSession {
                execution,
                next_index: 0,
                plan,
            })),
        )
        .with_resolved_frame_slots(slots))
    }
}

#[derive(Debug)]
pub(super) struct SequenceSource {
    pub(super) engine: WgpuDecodeEngine,
    pub(super) codestream: Arc<GpuCodestream>,
    pub(super) inventory: CodestreamInventory,
    pub(super) request: GpuOutputRequest,
    pub(super) surface_encodings: Option<Arc<[crate::frame_surface::FrameSurfaceEncoding]>>,
}

impl SequenceSource {
    /// Color producers execute their recursive LF dependency closure. Both output paths must
    /// visit every color layer, even when a later full-canvas Replace overwrites its pixels.
    pub(super) fn next_producer(&self, start: usize, end: usize) -> Result<usize> {
        (start..end)
            .find(|&i| self.inventory.frames[i].frame_type != FrameType::LowFrequency)
            .ok_or(Error::EngineContract("presentation has no color producer"))
    }

    pub(super) fn prepare_physical(
        &self,
        index: usize,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        let frame_index = self.inventory.frames[index].frame_index;
        let request = self.surface_encodings.as_ref().map_or_else(
            || self.request.clone(),
            |encodings| self.request.clone().for_frame_surface(encodings[index]),
        );
        if let Some(dc) = ProgressiveDcPlan::for_frame(&self.inventory, frame_index)? {
            self.engine.open_progressive_dc(
                Arc::clone(&self.codestream),
                &request,
                &self.inventory,
                dc,
            )
        } else {
            let projected = project_frame_inventory(&self.inventory, frame_index)?;
            match projected.frames[0].encoding {
                FrameEncoding::Modular => {
                    map_modular(self.engine.modular.open_frame_with_inventory_data(
                        Arc::clone(&self.codestream),
                        &request,
                        &projected,
                    )?)
                }
                FrameEncoding::VarDct => {
                    map_vardct(self.engine.vardct.open_frame_with_inventory_data(
                        (*self.codestream).clone(),
                        &request,
                        &projected,
                    )?)
                }
            }
        }
    }
}

/// An ordered frame graph. Output buffers and source reservations use the underlying coding-mode
/// leases, so independently pending presentations remain accounted by the shared byte budget.
#[derive(Debug)]
pub struct FrameSequenceSession {
    execution: SequenceExecution,
    next_index: usize,
    plan: FrameExecutionPlan,
}

#[derive(Debug)]
enum SequenceExecution {
    Independent(IndependentSession),
    Composed(CompositionSession),
}

impl FrameSequenceSession {
    #[must_use]
    pub const fn execution_plan(&self) -> &FrameExecutionPlan {
        &self.plan
    }

    pub(super) fn submissions_per_frame(&self) -> usize {
        match &self.execution {
            SequenceExecution::Independent(session) => session.submissions(),
            SequenceExecution::Composed(session) => session.submissions(),
        }
    }

    pub(super) fn submit_next(&mut self) -> Result<Option<WgpuDecodePendingFrame>> {
        if self.next_index == self.plan.presentations.len() {
            return Ok(None);
        }
        let inner = match &mut self.execution {
            SequenceExecution::Independent(session) => {
                SequencePending::Independent(Box::new(session.submit(&self.plan, self.next_index)?))
            }
            SequenceExecution::Composed(session) => {
                SequencePending::Composed(Box::new(session.submit(&self.plan, self.next_index)?))
            }
        };
        self.next_index += 1;
        Ok(Some(WgpuDecodePendingFrame::Sequence(Box::new(
            FrameSequencePending { inner },
        ))))
    }
}

pub(super) fn submission_counter(
    pending: &WgpuDecodePendingFrame,
    planned: usize,
) -> Arc<AtomicUsize> {
    match pending {
        WgpuDecodePendingFrame::Modular(_) => Arc::new(AtomicUsize::new(planned)),
        WgpuDecodePendingFrame::VarDct(frame) => frame.submissions_per_frame_counter(),
        WgpuDecodePendingFrame::ProgressiveDc(frame) => Arc::clone(&frame.submissions_per_frame),
        WgpuDecodePendingFrame::Sequence(_) => Arc::new(AtomicUsize::new(planned)),
    }
}

/// Pending pixels plus their exact coalesced presentation contract. The same completion function
/// is used by blocking and runtime-neutral polling paths.
#[derive(Debug)]
pub struct FrameSequencePending {
    inner: SequencePending,
}

#[derive(Debug)]
enum SequencePending {
    Independent(Box<IndependentPending>),
    Composed(Box<CompositionPending>),
}

impl FrameSequencePending {
    pub(super) fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match &self.inner {
            SequencePending::Independent(pending) => pending.unvalidated(),
            SequencePending::Composed(pending) => pending.unvalidated(),
        }
    }

    fn poll_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        match &mut self.inner {
            SequencePending::Composed(pending) => pending.poll(context),
            SequencePending::Independent(pending) => pending.poll(context),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for FrameSequencePending {
    type Frame = GpuImageFrame;

    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        match self.inner {
            SequencePending::Composed(pending) => pending.wait(),
            SequencePending::Independent(pending) => pending.wait(),
        }
    }

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        self.get_mut().poll_frame(context)
    }
}

#[cfg(target_arch = "wasm32")]
impl GpuPendingFrame for FrameSequencePending {
    type Frame = GpuImageFrame;

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        self.get_mut().poll_frame(context)
    }
}
