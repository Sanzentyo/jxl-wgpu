use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding, FrameType};
use jxl_wgpu::{GpuImageFrame, UnvalidatedGpuImageFrame};

use crate::{
    DecodeProfile, FrameExecutionPlan, GpuCodestream, GpuOutputRequest, GpuPendingFrame,
    PreparedGpuSession, Result, SubmittedGpuFrame,
};

use super::composition::{DependentPending, DependentSession};
use super::{
    WgpuDecodeEngine, WgpuDecodePendingFrame, WgpuDecodeSubmissionSession, map_modular, map_vardct,
    project_frame_inventory, validate_codestream_limit,
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
        let (execution, slots) = if super::composition::needs_surface(inventory, request, &plan)
            || inventory
                .frames
                .iter()
                .any(|frame| frame.frame_type == FrameType::LowFrequency)
        {
            (
                SequenceExecution::Dependent(DependentSession::new(
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
    pub(super) fn prepare_physical(
        &self,
        index: usize,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        let frame_index = self.inventory.frames[index].frame_index;
        let request = self.surface_encodings.as_ref().map_or_else(
            || self.request.clone(),
            |encodings| self.request.clone().for_frame_surface(encodings[index]),
        );
        let projected = project_frame_inventory(&self.inventory, frame_index)?;
        match projected.frames[0].encoding {
            FrameEncoding::Modular => {
                let prepared = if projected.frames[0].frame_type == FrameType::LowFrequency {
                    self.engine
                        .modular
                        .open_progressive_dc_with_inventory_data(
                            Arc::clone(&self.codestream),
                            &request,
                            &projected,
                        )?
                } else {
                    self.engine.modular.open_frame_with_inventory_data(
                        Arc::clone(&self.codestream),
                        &request,
                        &projected,
                    )?
                };
                map_modular(prepared)
            }
            FrameEncoding::VarDct => {
                let request = if projected.frames[0].frame_type == FrameType::LowFrequency {
                    GpuOutputRequest::color(crate::vardct_rgb8_format())?
                        .with_max_frame_slots(request.max_frame_slots())
                } else {
                    request
                };
                map_vardct(self.engine.vardct.open_frame_with_inventory_data(
                    (*self.codestream).clone(),
                    &request,
                    &projected,
                )?)
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
    Dependent(DependentSession),
}

impl FrameSequenceSession {
    #[must_use]
    pub const fn execution_plan(&self) -> &FrameExecutionPlan {
        &self.plan
    }

    pub(super) fn submissions_per_frame(&self) -> usize {
        match &self.execution {
            SequenceExecution::Independent(session) => session.submissions(),
            SequenceExecution::Dependent(session) => session.submissions(),
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
            SequenceExecution::Dependent(session) => {
                SequencePending::Dependent(Box::new(session.submit(&self.plan, self.next_index)?))
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
    Dependent(Box<DependentPending>),
}

impl FrameSequencePending {
    pub(super) fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match &self.inner {
            SequencePending::Independent(pending) => pending.unvalidated(),
            SequencePending::Dependent(pending) => pending.unvalidated(),
        }
    }

    fn poll_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        match &mut self.inner {
            SequencePending::Dependent(pending) => pending.poll(context),
            SequencePending::Independent(pending) => pending.poll(context),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for FrameSequencePending {
    type Frame = GpuImageFrame;

    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        match self.inner {
            SequencePending::Dependent(pending) => pending.wait(),
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
