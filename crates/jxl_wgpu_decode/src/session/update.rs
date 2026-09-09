//! Refinements share presentation identity and frame-slot ownership without advancing animation.

use super::*;

/// The completely decoded coefficient boundary represented by an intermediate image.
/// Pixel storage keeps the requested full canvas extent; downsampling describes intended detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameProgression {
    /// Original physical frame index, including hidden and LF frames.
    pub physical_frame_index: u32,
    /// Zero denotes DC only. Otherwise all logical groups in these AC passes are complete.
    pub completed_passes: u8,
    pub total_passes: u8,
    /// A nonzero power of two. This does not change the output buffer's dimensions.
    pub intended_downsampling: u32,
}

/// One update from a pending engine frame. Only `Complete` advances presentation state.
#[derive(Debug)]
pub enum SubmittedGpuUpdate<F> {
    Intermediate {
        frame: SubmittedGpuFrame<F>,
        progression: FrameProgression,
    },
    Complete(SubmittedGpuFrame<F>),
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn wait_next_update<P: GpuPendingFrame + ?Sized>(
    pending: &mut P,
) -> Result<SubmittedGpuUpdate<P::Frame>> {
    struct ThreadWake(std::thread::Thread);
    impl std::task::Wake for ThreadWake {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match Pin::new(&mut *pending).poll_next_update(&mut context) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

impl<S: GpuSubmissionSession> GpuDecodeSession<S> {
    /// Returns the next validated intermediate image or final frame. All updates of one
    /// presentation share its metadata and frame slot. Caller-held images remain immutable and
    /// retain their own GPU byte reservations while later updates are decoded.
    ///
    /// Engines may provide only final frames. Use [`Self::next_frame`] to skip refinements.
    pub fn next_update(&mut self) -> Result<Option<GpuFrameLease<S::Frame>>> {
        if self.failed {
            return Err(Error::SessionPoisoned);
        }
        if self.finished {
            return Ok(None);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.ensure_front_sync()?;
            let result = self
                .pending
                .front_mut()
                .ok_or(Error::EngineContract("update request has no pending frame"))?
                .1
                .wait_next_update();
            self.finish_update(result).map(Some)
        }
        #[cfg(target_arch = "wasm32")]
        Err(Error::BlockingWaitUnavailable)
    }

    /// Polls one update without consuming its pending frame until the final image is validated.
    pub fn poll_next_update(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<GpuFrameLease<S::Frame>>>> {
        if self.failed {
            return Poll::Ready(Err(Error::SessionPoisoned));
        }
        if self.finished {
            return Poll::Ready(Ok(None));
        }
        if self.pending.is_empty() {
            let target = NonZeroUsize::new(1).expect("one is nonzero");
            match self.poll_prefetch(target, context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(progress)) => {
                    if let Some(pressure) = progress.backpressure {
                        return Poll::Ready(Err(prefetch_backpressure_error(pressure)));
                    }
                }
            }
        }
        let Some((_, pending)) = self.pending.front_mut() else {
            self.failed = true;
            return Poll::Ready(Err(Error::MissingFinalFrame));
        };
        match Pin::new(pending).poll_next_update(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => Poll::Ready(self.finish_update(result).map(Some)),
        }
    }

    #[must_use = "futures do nothing unless polled"]
    pub const fn next_update_async(&mut self) -> NextGpuUpdate<'_, S> {
        NextGpuUpdate { session: self }
    }

    fn finish_update(
        &mut self,
        update: Result<SubmittedGpuUpdate<S::Frame>>,
    ) -> Result<GpuFrameLease<S::Frame>> {
        let result = match update {
            Ok(SubmittedGpuUpdate::Complete(frame)) => {
                let (permit, _) = self.pending.pop_front().ok_or(Error::EngineContract(
                    "completed update has no pending frame",
                ))?;
                self.finish_frame(permit, frame)
            }
            Ok(SubmittedGpuUpdate::Intermediate { frame, progression }) => {
                self.finish_intermediate(frame, progression)
            }
            Err(error) => Err(error),
        };
        if result.is_err() {
            self.failed = true;
            self.pending.clear();
            self.last_progression = None;
        }
        result
    }

    fn finish_intermediate(
        &mut self,
        frame: SubmittedGpuFrame<S::Frame>,
        progression: FrameProgression,
    ) -> Result<GpuFrameLease<S::Frame>> {
        self.validate_frame_metadata(&frame.metadata)?;
        if progression.total_passes == 0
            || progression.completed_passes >= progression.total_passes
            || !progression.intended_downsampling.is_power_of_two()
        {
            return Err(Error::EngineContract(
                "invalid intermediate coefficient boundary",
            ));
        }
        if let Some((_, previous)) = &self.last_progression {
            let same_source = progression.physical_frame_index == previous.physical_frame_index;
            if progression.physical_frame_index < previous.physical_frame_index
                || progression.intended_downsampling > previous.intended_downsampling
                || (same_source
                    && (progression.completed_passes <= previous.completed_passes
                        || progression.total_passes != previous.total_passes))
            {
                return Err(Error::EngineContract(
                    "progressive image updates are out of order",
                ));
            }
        }
        let permit = self
            .pending
            .front()
            .ok_or(Error::EngineContract(
                "intermediate update has no pending frame",
            ))?
            .0
            .clone();
        self.last_progression = Some((frame.metadata.clone(), progression));
        Ok(GpuFrameLease {
            metadata: frame.metadata,
            output: frame.output,
            progression: Some(progression),
            _permit: permit,
        })
    }
}

#[must_use = "futures do nothing unless polled"]
pub struct NextGpuUpdate<'session, S: GpuSubmissionSession> {
    session: &'session mut GpuDecodeSession<S>,
}

impl<S: GpuSubmissionSession> Future for NextGpuUpdate<'_, S> {
    type Output = Result<Option<GpuFrameLease<S::Frame>>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().session.poll_next_update(context)
    }
}
