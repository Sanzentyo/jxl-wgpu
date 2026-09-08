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

use super::composition::{CompositionPending, CompositionSession};
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
        if request.renders_spot_colors(&inventory.image_header.extra_channels)
            || plan.nodes.iter().any(|node| node.needs_composition)
            || inventory
                .frames
                .iter()
                .any(|frame| frame.frame_type == jxl_gpu_bitstream::FrameType::ReferenceOnly)
        {
            let composition =
                CompositionSession::new(self.clone(), codestream, inventory, request, &plan)?;
            return Ok(PreparedGpuSession::new(
                DecodeProfile::FrameSequence {
                    physical_frames: plan.nodes.len(),
                    presentation_frames: plan.presentations.len(),
                },
                plan.metadata.clone(),
                WgpuDecodeSubmissionSession::Sequence(Box::new(FrameSequenceSession {
                    source: None,
                    current: None,
                    next_index: 0,
                    plan,
                    last_submissions: Arc::new(AtomicUsize::new(0)),
                    composition: Some(composition),
                })),
            )
            .with_resolved_frame_slots(request.max_frame_slots()));
        }
        let source = SequenceSource {
            engine: self.clone(),
            codestream,
            inventory: inventory.clone(),
            request: request.clone(),
            surface_encodings: None,
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
                composition: None,
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
        let prepared = self.prepare_physical(frame_index as usize)?;
        if prepared.metadata.extent != plan.metadata.extent {
            return Err(Error::EngineContract(
                "frame producer disagrees with the presentation extent",
            ));
        }
        Ok(prepared)
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
    source: Option<SequenceSource>,
    current: Option<WgpuDecodeSubmissionSession>,
    next_index: usize,
    plan: FrameExecutionPlan,
    last_submissions: Arc<AtomicUsize>,
    composition: Option<CompositionSession>,
}

impl FrameSequenceSession {
    #[must_use]
    pub const fn execution_plan(&self) -> &FrameExecutionPlan {
        &self.plan
    }

    pub(super) fn submissions_per_frame(&self) -> usize {
        if let Some(composition) = &self.composition {
            return composition.submissions();
        }
        self.current.as_ref().map_or_else(
            || self.last_submissions.load(Ordering::Acquire),
            |frame| frame.submissions_per_frame(),
        )
    }

    pub(super) fn submit_next(&mut self) -> Result<Option<WgpuDecodePendingFrame>> {
        if self.next_index == self.plan.presentations.len() {
            return Ok(None);
        }
        if let Some(composition) = &mut self.composition {
            let pending = composition.submit(&self.plan, self.next_index)?;
            self.next_index += 1;
            return Ok(Some(WgpuDecodePendingFrame::Sequence(Box::new(
                FrameSequencePending {
                    inner: SequencePending::Composed(Box::new(pending)),
                },
            ))));
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
                inner: SequencePending::Independent {
                    pending: Box::new(pending),
                    metadata,
                },
            },
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
    Independent {
        pending: Box<WgpuDecodePendingFrame>,
        metadata: FrameMetadata,
    },
    Composed(Box<CompositionPending>),
}

impl FrameSequencePending {
    pub(super) fn unvalidated_gpu_frame(&self) -> Result<UnvalidatedGpuImageFrame> {
        match &self.inner {
            SequencePending::Independent { pending, .. } => pending.unvalidated_gpu_frame(),
            SequencePending::Composed(pending) => pending.unvalidated(),
        }
    }

    fn poll_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        match &mut self.inner {
            SequencePending::Composed(pending) => pending.poll(context),
            SequencePending::Independent { pending, metadata } => Pin::new(pending.as_mut())
                .poll_complete(context)
                .map(|result| {
                    result.map(|frame| SubmittedGpuFrame::new(metadata.clone(), frame.output))
                }),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuPendingFrame for FrameSequencePending {
    type Frame = GpuImageFrame;

    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        match self.inner {
            SequencePending::Composed(pending) => pending.wait(),
            SequencePending::Independent { pending, metadata } => pending
                .wait()
                .map(|frame| SubmittedGpuFrame::new(metadata, frame.output)),
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
