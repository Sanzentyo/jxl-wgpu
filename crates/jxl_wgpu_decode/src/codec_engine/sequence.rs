use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding};
use jxl_wgpu::{GpuImageFrame, UnvalidatedGpuImageFrame};

use crate::{
    DecodeProfile, Error, FrameExecutionPlan, FrameMetadata, GpuCodestream, GpuOutputRequest,
    GpuPendingFrame, GpuSubmissionSession, PreparedGpuSession, Result, SubmittedGpuFrame,
};

use super::{
    ProgressiveDcPlan, WgpuDecodeEngine, WgpuDecodePendingFrame, WgpuDecodeSubmissionSession,
    map_modular, map_vardct, project_frame_inventory, validate_codestream_limit,
};
use crate::GpuSubmissionEngine;

impl WgpuDecodeEngine {
    pub(super) fn open_sequence(
        &self,
        codestream: Arc<GpuCodestream>,
        request: &GpuOutputRequest,
        inventory: &CodestreamInventory,
        plan: FrameExecutionPlan,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        validate_codestream_limit(codestream.logical_bytes(), self.parse_limits())?;
        plan.validate_independent_frames(inventory)?;
        let source = SequenceSource {
            engine: self.clone(),
            codestream,
            inventory: inventory.clone(),
            request: request.clone(),
        };
        let prepared = source.prepare(&plan, 0)?;
        let slots = prepared
            .resolved_frame_slots()
            .unwrap_or(request.max_frame_slots());
        Ok(PreparedGpuSession::new(
            DecodeProfile::FrameSequence {
                physical_frames: plan.nodes.len(),
                presentation_frames: plan.presentations.len(),
            },
            plan.metadata.clone(),
            WgpuDecodeSubmissionSession::Sequence(Box::new(FrameSequenceSession {
                source: Some(source),
                current: Some(prepared.session),
                next_index: 0,
                plan,
                last_submissions: Arc::new(AtomicUsize::new(0)),
            })),
        )
        .with_resolved_frame_slots(slots))
    }
}

#[derive(Debug)]
struct SequenceSource {
    engine: WgpuDecodeEngine,
    codestream: Arc<GpuCodestream>,
    inventory: CodestreamInventory,
    request: GpuOutputRequest,
}

impl SequenceSource {
    fn prepare(
        &self,
        plan: &FrameExecutionPlan,
        index: usize,
    ) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
        // Prepare only one upcoming presentation. Entropy descriptors and scratch plans must not
        // grow with the animation length. Full-canvas Replace removes overwritten zero-duration
        // layers; only the visible producer and its LF dependency closure need image decoding.
        let frame_index =
            self.inventory.frames[plan.presentations[index].physical_frames.end - 1].frame_index;
        let prepared = if let Some(dc) = ProgressiveDcPlan::for_frame(&self.inventory, frame_index)?
        {
            self.engine.open_progressive_dc(
                Arc::clone(&self.codestream),
                &self.request,
                &self.inventory,
                dc,
            )?
        } else {
            let projected = project_frame_inventory(&self.inventory, frame_index)?;
            match projected.frames[0].encoding {
                FrameEncoding::Modular => {
                    map_modular(self.engine.modular.open_frame_with_inventory_data(
                        Arc::clone(&self.codestream),
                        &self.request,
                        &projected,
                    )?)?
                }
                FrameEncoding::VarDct => {
                    map_vardct(self.engine.vardct.open_frame_with_inventory_data(
                        (*self.codestream).clone(),
                        &self.request,
                        &projected,
                    )?)?
                }
            }
        };
        if prepared.metadata.extent != plan.metadata.extent {
            return Err(Error::EngineContract(
                "frame producer disagrees with the presentation extent",
            ));
        }
        Ok(prepared)
    }
}

/// An ordered frame graph. Output buffers and source reservations use the underlying coding-mode
/// leases, so independently pending presentations remain accounted by the shared byte budget.
#[derive(Debug)]
pub struct FrameSequenceSession {
    source: Option<SequenceSource>,
    current: Option<WgpuDecodeSubmissionSession>,
    next_index: usize,
    plan: FrameExecutionPlan,
    last_submissions: Arc<AtomicUsize>,
}

impl FrameSequenceSession {
    #[must_use]
    pub const fn execution_plan(&self) -> &FrameExecutionPlan {
        &self.plan
    }

    pub(super) fn submissions_per_frame(&self) -> usize {
        self.current.as_ref().map_or_else(
            || self.last_submissions.load(Ordering::Acquire),
            |frame| frame.submissions_per_frame(),
        )
    }

    pub(super) fn submit_next(&mut self) -> Result<Option<WgpuDecodePendingFrame>> {
        if self.next_index == self.plan.presentations.len() {
            return Ok(None);
        }
        if self.current.is_none() {
            self.current = Some(
                self.source
                    .as_ref()
                    .ok_or(Error::EngineContract("frame sequence lost its source"))?
                    .prepare(&self.plan, self.next_index)?
                    .session,
            );
        }
        let frame = self
            .current
            .as_mut()
            .expect("the next producer was prepared");
        // Keep the producer and metadata at the queue front until admission succeeds. Retrying
        // frame-slot or shared-byte pressure must not consume or reorder a presentation.
        let pending = frame.submit_next()?.ok_or(Error::EngineContract(
            "frame graph producer returned no frame",
        ))?;
        self.last_submissions = submission_counter(&pending, frame.submissions_per_frame());
        let metadata = self.plan.presentations[self.next_index].metadata.clone();
        self.next_index += 1;
        self.current = None;
        if self.next_index == self.plan.presentations.len() {
            self.source = None;
        }
        Ok(Some(WgpuDecodePendingFrame::Sequence(Box::new(
            FrameSequencePending {
                pending: Box::new(pending),
                metadata,
            },
        ))))
    }
}

fn submission_counter(pending: &WgpuDecodePendingFrame, planned: usize) -> Arc<AtomicUsize> {
    match pending {
        WgpuDecodePendingFrame::Modular(_) => Arc::new(AtomicUsize::new(planned)),
        WgpuDecodePendingFrame::VarDct(frame) => frame.submissions_per_frame_counter(),
        WgpuDecodePendingFrame::ProgressiveDc(frame) => Arc::clone(&frame.submissions_per_frame),
        WgpuDecodePendingFrame::Sequence(frame) => submission_counter(&frame.pending, planned),
    }
}

/// Pending pixels plus their exact coalesced presentation contract. The same completion function
/// is used by blocking and runtime-neutral polling paths.
#[derive(Debug)]
pub struct FrameSequencePending {
    pending: Box<WgpuDecodePendingFrame>,
    metadata: FrameMetadata,
}

impl FrameSequencePending {
    pub(super) fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        self.pending.unvalidated_gpu_frame()
    }

    fn poll_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        Pin::new(self.pending.as_mut())
            .poll_complete(context)
            .map(|result| {
                result.map(|frame| SubmittedGpuFrame::new(self.metadata.clone(), frame.output))
            })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for FrameSequencePending {
    type Frame = GpuImageFrame;

    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        self.pending
            .wait()
            .map(|frame| SubmittedGpuFrame::new(self.metadata, frame.output))
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
