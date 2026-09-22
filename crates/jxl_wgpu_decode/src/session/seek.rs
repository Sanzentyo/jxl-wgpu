//! One-target seeking through the ordinary validated GPU session and its ownership contracts.
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::FrameIndexLimits;

mod stream;
pub use stream::GpuDecodeSeekStream;

use super::{
    GpuDecodeSession, GpuDecoder, GpuFrameLease, GpuSubmissionEngine, GpuSubmissionSession,
};
use crate::{
    BoundFrameIndex, FrameSeekError, FrameSeekLimits, FrameSeekPlan, GpuOutputRequest,
    ImageSelection, Result,
};

impl<E: GpuSubmissionEngine> GpuDecoder<E> {
    /// Opens one main-image presentation from the latest dependency-complete indexed anchor.
    /// The entire input container/header inventory is checked first. This is bounded GPU restart,
    /// not a byte-range input API. Missing indexes are generated from the same header graph;
    /// malformed or inconsistent indexes are errors, never silently ignored.
    ///
    /// ```no_run
    /// # use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine, vardct_rgb8_format};
    /// # fn example(decoder: &GpuDecoder<WgpuDecodeEngine>, bytes: &[u8]) -> jxl_wgpu_decode::Result<()> {
    /// let request = GpuOutputRequest::color(vardct_rgb8_format())?;
    /// let mut seek = decoder.open_seek(bytes, request, 2, Default::default(), Default::default())?;
    /// let frame = seek.next_frame()?.expect("requested presentation");
    /// assert_eq!(frame.metadata.index, 2);
    /// assert!(seek.next_frame()?.is_none()); // Ends this seek, regardless of frame.metadata.is_last.
    /// # Ok(()) }
    /// ```
    pub fn open_seek(
        &self,
        encoded: &[u8],
        request: GpuOutputRequest,
        target: usize,
        index_limits: FrameIndexLimits,
        seek_limits: FrameSeekLimits,
    ) -> Result<GpuSeekSession<E::Session>> {
        super::validate_total_input_size(encoded.len(), self.parse_limits)?;
        self.open_seek_shared(
            Arc::from(encoded),
            request,
            target,
            index_limits,
            seek_limits,
        )
    }

    pub fn open_seek_shared(
        &self,
        encoded: Arc<[u8]>,
        request: GpuOutputRequest,
        target: usize,
        index_limits: FrameIndexLimits,
        seek_limits: FrameSeekLimits,
    ) -> Result<GpuSeekSession<E::Session>> {
        request.format().validate()?;
        if request.image_selection() != ImageSelection::Main {
            return Err(FrameSeekError::PreviewSelection.into());
        }
        let (codestream, inventory, index) = super::parse_shared_indexed(
            encoded,
            self.parse_limits,
            self.engine.inventory_limits(),
            Some(index_limits),
        )?;
        let bound = BoundFrameIndex::new(inventory, index, index_limits)?;
        let plan = bound.seek(target, seek_limits)?;
        open_seek_plan(self.engine.as_ref(), codestream, request, plan)
    }
}

fn open_seek_plan<E: GpuSubmissionEngine>(
    engine: &E,
    codestream: crate::GpuCodestream,
    request: GpuOutputRequest,
    plan: FrameSeekPlan,
) -> Result<GpuSeekSession<E::Session>> {
    let prepared = engine.open(codestream, &request, plan.selection.clone())?;
    Ok(GpuSeekSession::new(
        GpuDecodeSession::new(prepared, request)?,
        plan,
    ))
}

/// Decodes and drops the bounded preroll, then exposes only the requested presentation.
/// Returned metadata keeps the original index, clock, timecode and stream finality. `None`
/// after that presentation ends this seek operation; it does not mark a nonfinal frame as final.
/// Dropping this session cancels through the existing GPU completion and byte reservations.
pub struct GpuSeekSession<S: GpuSubmissionSession> {
    inner: Option<GpuDecodeSession<S>>,
    plan: FrameSeekPlan,
    skipped: usize,
    submitted: usize,
}

impl<S: GpuSubmissionSession> GpuSeekSession<S> {
    fn new(inner: GpuDecodeSession<S>, plan: FrameSeekPlan) -> Self {
        Self {
            inner: Some(inner),
            plan,
            skipped: 0,
            submitted: 0,
        }
    }

    #[must_use]
    pub const fn plan(&self) -> &FrameSeekPlan {
        &self.plan
    }

    #[must_use]
    pub fn frames_submitted(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(self.submitted, GpuDecodeSession::frames_submitted)
    }

    /// Existing backend diagnostics while this seek still owns an active decode session.
    #[must_use]
    pub fn submission_session(&self) -> Option<&S> {
        self.inner
            .as_ref()
            .map(GpuDecodeSession::submission_session)
    }

    pub fn next_frame(&mut self) -> Result<Option<GpuFrameLease<S::Frame>>> {
        self.next(false)
    }

    pub fn next_update(&mut self) -> Result<Option<GpuFrameLease<S::Frame>>> {
        self.next(true)
    }

    fn next(&mut self, updates: bool) -> Result<Option<GpuFrameLease<S::Frame>>> {
        let Some(inner) = &mut self.inner else {
            return Ok(None);
        };
        while self.skipped < self.plan.preroll_presentations() {
            let frame = inner.next_frame()?.ok_or(crate::Error::MissingFinalFrame)?;
            drop(frame);
            self.skipped += 1;
        }
        let frame = if updates {
            inner.next_update()?
        } else {
            inner.next_frame()?
        };
        self.complete(frame)
    }

    pub fn poll_next_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<GpuFrameLease<S::Frame>>>> {
        self.poll_next(context, false)
    }

    pub fn poll_next_update(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<GpuFrameLease<S::Frame>>>> {
        self.poll_next(context, true)
    }

    fn poll_next(
        &mut self,
        context: &mut Context<'_>,
        updates: bool,
    ) -> Poll<Result<Option<GpuFrameLease<S::Frame>>>> {
        let Some(inner) = &mut self.inner else {
            return Poll::Ready(Ok(None));
        };
        while self.skipped < self.plan.preroll_presentations() {
            let frame = std::task::ready!(inner.poll_next_frame(context))?
                .ok_or(crate::Error::MissingFinalFrame)?;
            drop(frame);
            self.skipped += 1;
        }
        let frame = std::task::ready!(if updates {
            inner.poll_next_update(context)
        } else {
            inner.poll_next_frame(context)
        })?;
        Poll::Ready(self.complete(frame))
    }

    fn complete(
        &mut self,
        frame: Option<GpuFrameLease<S::Frame>>,
    ) -> Result<Option<GpuFrameLease<S::Frame>>> {
        let mut frame = frame.ok_or(crate::Error::MissingFinalFrame)?;
        frame.metadata = self.plan.target().clone();
        if frame.is_complete() {
            self.submitted = self.inner.as_ref().expect("active seek").frames_submitted();
            // References and temporary input are unnecessary once the final target validates.
            self.inner = None;
        }
        Ok(Some(frame))
    }

    pub fn next_frame_async(&mut self) -> NextSeekFrame<'_, S> {
        NextSeekFrame {
            session: self,
            updates: false,
        }
    }
    pub fn next_update_async(&mut self) -> NextSeekFrame<'_, S> {
        NextSeekFrame {
            session: self,
            updates: true,
        }
    }
}

pub struct NextSeekFrame<'a, S: GpuSubmissionSession> {
    session: &'a mut GpuSeekSession<S>,
    updates: bool,
}

impl<S: GpuSubmissionSession> Future for NextSeekFrame<'_, S> {
    type Output = Result<Option<GpuFrameLease<S::Frame>>>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        this.session.poll_next(context, this.updates)
    }
}
