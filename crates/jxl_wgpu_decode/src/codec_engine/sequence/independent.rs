//! Full-canvas Replace presentations retain native producer output. Overwritten layers still
//! execute and validate, one at a time, without a floating-point composition surface.

use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{GpuImageFrame, UnvalidatedGpuImageFrame};

use super::{SequenceSource, submission_counter};
use crate::{
    Error, FrameExecutionPlan, FrameMetadata, GpuPendingFrame, GpuSubmissionSession,
    PreparedGpuSession, Result, SubmittedGpuFrame, SubmittedGpuUpdate, WgpuDecodePendingFrame,
    WgpuDecodeSubmissionSession,
};

fn prepare(
    source: &SequenceSource,
    index: usize,
    extent: Extent2d,
    progressive: bool,
) -> Result<PreparedGpuSession<WgpuDecodeSubmissionSession>> {
    let prepared = source.prepare_physical(index, progressive)?;
    if prepared.metadata.extent != extent {
        return Err(Error::EngineContract(
            "frame producer disagrees with the presentation extent",
        ));
    }
    Ok(prepared)
}

#[derive(Debug)]
pub(super) struct IndependentSession {
    source: Option<Arc<SequenceSource>>,
    prepared: Option<WgpuDecodeSubmissionSession>,
    submissions: Arc<AtomicUsize>,
}

impl IndependentSession {
    pub(super) fn new(
        source: Arc<SequenceSource>,
        plan: &FrameExecutionPlan,
    ) -> Result<(Self, NonZeroUsize)> {
        let range = &plan.presentations[0].physical_frames;
        let first = range.start;
        let prepared = prepare(&source, first, plan.metadata.extent, first + 1 == range.end)?;
        let slots = prepared
            .resolved_frame_slots()
            .unwrap_or(source.request.max_frame_slots());
        Ok((
            Self {
                source: Some(source),
                submissions: Arc::new(AtomicUsize::new(prepared.session.submissions_per_frame())),
                prepared: Some(prepared.session),
            },
            slots,
        ))
    }

    pub(super) fn submissions(&self) -> usize {
        self.prepared.as_ref().map_or_else(
            || self.submissions.load(Ordering::Acquire),
            |producer| producer.submissions_per_frame(),
        )
    }

    pub(super) fn submit(
        &mut self,
        plan: &FrameExecutionPlan,
        index: usize,
    ) -> Result<IndependentPending> {
        let source = self
            .source
            .as_ref()
            .ok_or(Error::EngineContract("frame sequence lost its source"))?;
        let presentation = &plan.presentations[index];
        let range = &presentation.physical_frames;
        let physical = range.start;
        if self.prepared.is_none() {
            self.prepared = Some(
                prepare(
                    source,
                    physical,
                    plan.metadata.extent,
                    physical + 1 == range.end,
                )?
                .session,
            );
        }
        let producer = self.prepared.as_mut().expect("physical producer prepared");
        // Admission failure leaves this producer and its presentation available for retry.
        let pending = producer
            .submit_next()?
            .ok_or(Error::EngineContract("physical producer returned no frame"))?;
        let count = submission_counter(&pending, producer.submissions_per_frame());
        self.submissions = Arc::new(AtomicUsize::new(count.load(Ordering::Acquire)));
        self.prepared = None;
        let source = Arc::clone(source);
        if presentation.metadata.is_last {
            self.source = None;
        }
        Ok(IndependentPending {
            source: Some(source),
            extent: plan.metadata.extent,
            physical,
            end: range.end,
            pending: Some(Box::new(pending)),
            count,
            completed_submissions: 0,
            submissions: Arc::clone(&self.submissions),
            metadata: presentation.metadata.clone(),
        })
    }
}

#[derive(Debug)]
pub(super) struct IndependentPending {
    source: Option<Arc<SequenceSource>>,
    extent: Extent2d,
    physical: usize,
    end: usize,
    pending: Option<Box<WgpuDecodePendingFrame>>,
    count: Arc<AtomicUsize>,
    completed_submissions: usize,
    submissions: Arc<AtomicUsize>,
    metadata: FrameMetadata,
}

impl IndependentPending {
    pub(super) fn unvalidated(&self) -> Result<UnvalidatedGpuImageFrame> {
        if self.physical + 1 != self.end {
            return Err(Error::UnvalidatedOutputNotSubmitted);
        }
        self.pending
            .as_ref()
            .ok_or(Error::EngineContract(
                "independent presentation was consumed",
            ))?
            .unvalidated_gpu_frame()
    }

    fn update_count(&self) -> Result<usize> {
        let count = self
            .completed_submissions
            .checked_add(self.count.load(Ordering::Acquire))
            .ok_or(Error::EngineContract(
                "sequence submission count overflowed",
            ))?;
        self.submissions.store(count, Ordering::Release);
        Ok(count)
    }

    /// Both completion APIs discard an overwritten allocation before admitting the next layer.
    fn decoded(
        &mut self,
        frame: SubmittedGpuFrame<GpuImageFrame>,
    ) -> Result<Option<SubmittedGpuFrame<GpuImageFrame>>> {
        self.completed_submissions = self.update_count()?;
        if self.physical + 1 == self.end {
            self.source = None;
            return Ok(Some(SubmittedGpuFrame::new(
                self.metadata.clone(),
                frame.output,
            )));
        }
        drop(frame);
        let source = self.source.as_ref().ok_or(Error::EngineContract(
            "independent presentation lost its source",
        ))?;
        self.physical += 1;
        let mut producer = prepare(
            source,
            self.physical,
            self.extent,
            self.physical + 1 == self.end,
        )?
        .session;
        let pending = producer
            .submit_next()?
            .ok_or(Error::EngineContract("physical producer returned no frame"))?;
        self.count = submission_counter(&pending, producer.submissions_per_frame());
        self.pending = Some(Box::new(pending));
        self.update_count()?;
        Ok(None)
    }

    pub(super) fn poll(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<GpuImageFrame>>> {
        self.poll_update(context, false)
            .map(|result| match result? {
                SubmittedGpuUpdate::Complete(frame) => Ok(frame),
                SubmittedGpuUpdate::Intermediate { .. } => Err(Error::EngineContract(
                    "final-only completion returned an intermediate",
                )),
            })
    }

    pub(super) fn poll_update(
        &mut self,
        context: &mut Context<'_>,
        emit_intermediates: bool,
    ) -> Poll<Result<SubmittedGpuUpdate<GpuImageFrame>>> {
        loop {
            let pending = self.pending.as_mut().ok_or(Error::EngineContract(
                "independent presentation was consumed",
            ))?;
            let result = if emit_intermediates && self.physical + 1 == self.end {
                Pin::new(pending.as_mut()).poll_next_update(context)
            } else {
                Pin::new(pending.as_mut())
                    .poll_complete(context)
                    .map(|result| result.map(SubmittedGpuUpdate::Complete))
            };
            self.update_count()?;
            let update = match result {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result?,
            };
            let frame = match update {
                SubmittedGpuUpdate::Complete(frame) => frame,
                SubmittedGpuUpdate::Intermediate {
                    mut frame,
                    progression,
                } => {
                    frame.metadata = self.metadata.clone();
                    return Poll::Ready(Ok(SubmittedGpuUpdate::Intermediate {
                        frame,
                        progression,
                    }));
                }
            };
            self.pending = None;
            if let Some(frame) = self.decoded(frame)? {
                return Poll::Ready(Ok(SubmittedGpuUpdate::Complete(frame)));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn wait(mut self) -> Result<SubmittedGpuFrame<GpuImageFrame>> {
        loop {
            let frame = self
                .pending
                .take()
                .ok_or(Error::EngineContract(
                    "independent presentation was consumed",
                ))?
                .wait()?;
            if let Some(frame) = self.decoded(frame)? {
                return Ok(frame);
            }
        }
    }
}
