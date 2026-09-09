//! Refinements share presentation identity and frame-slot ownership without advancing animation.

use super::*;

/// The validated physical boundary represented by an intermediate image. Completing an LF
/// dependency does not complete its presentation. Every image keeps the requested canvas extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameProgression {
    /// Zero completed passes denotes DC; otherwise every logical group in these passes is valid.
    Coefficients {
        physical_frame_index: u32,
        completed_passes: u8,
        total_passes: u8,
        /// A nonzero power of two describing detail, not the output buffer's dimensions.
        intended_downsampling: u32,
    },
    /// A completely decoded LF dependency, including its restoration and frame resampling.
    LowFrequency {
        physical_frame_index: u32,
        /// JPEG XL LF level, from 1 through 4; intended downsampling is 8 to this power.
        level: u8,
    },
}

impl FrameProgression {
    pub const fn physical_frame_index(self) -> u32 {
        match self {
            Self::Coefficients {
                physical_frame_index,
                ..
            }
            | Self::LowFrequency {
                physical_frame_index,
                ..
            } => physical_frame_index,
        }
    }

    /// Returns the intended detail divisor, or zero for an invalid LF level.
    pub const fn intended_downsampling(self) -> u32 {
        match self {
            Self::Coefficients {
                intended_downsampling,
                ..
            } => intended_downsampling,
            Self::LowFrequency {
                level: level @ 1..=4,
                ..
            } => 1 << (3 * level),
            Self::LowFrequency { .. } => 0,
        }
    }

    /// Completed coefficient passes; complete LF frames have no partial coefficient boundary.
    pub const fn completed_passes(self) -> Option<u8> {
        match self {
            Self::Coefficients {
                completed_passes, ..
            } => Some(completed_passes),
            Self::LowFrequency { .. } => None,
        }
    }

    pub const fn total_passes(self) -> Option<u8> {
        match self {
            Self::Coefficients { total_passes, .. } => Some(total_passes),
            Self::LowFrequency { .. } => None,
        }
    }

    fn valid(self) -> bool {
        match self {
            Self::Coefficients {
                completed_passes,
                total_passes,
                intended_downsampling,
                ..
            } => {
                total_passes != 0
                    && completed_passes < total_passes
                    && intended_downsampling.is_power_of_two()
            }
            Self::LowFrequency { level, .. } => (1..=4).contains(&level),
        }
    }

    fn follows(self, previous: Self) -> bool {
        if self.physical_frame_index() < previous.physical_frame_index()
            || self.intended_downsampling() > previous.intended_downsampling()
        {
            return false;
        }
        if self.physical_frame_index() != previous.physical_frame_index() {
            return true;
        }
        matches!((previous, self),
            (Self::Coefficients { completed_passes: before, total_passes: before_total, .. },
             Self::Coefficients { completed_passes: after, total_passes: after_total, .. })
                if after > before && after_total == before_total)
    }
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
        if !progression.valid() {
            return Err(Error::EngineContract(
                "invalid intermediate physical boundary",
            ));
        }
        if let Some((_, previous)) = &self.last_progression
            && !progression.follows(*previous)
        {
            return Err(Error::EngineContract(
                "progressive image updates are out of order",
            ));
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

#[cfg(test)]
mod tests {
    use super::FrameProgression;

    #[test]
    fn complete_lf_boundaries_have_real_levels_and_precede_coefficient_refinements() {
        let lf = |physical_frame_index, level| FrameProgression::LowFrequency {
            physical_frame_index,
            level,
        };
        for level in 0..=u8::MAX {
            let boundary = lf(5, level);
            assert_eq!(boundary.valid(), (1..=4).contains(&level));
            assert_eq!(boundary.completed_passes(), None);
            assert_eq!(boundary.total_passes(), None);
            assert_eq!(
                boundary.intended_downsampling(),
                if (1..=4).contains(&level) {
                    8_u32.pow(u32::from(level))
                } else {
                    0
                }
            );
        }
        let levels = [
            lf(5, 4),
            lf(6, 3),
            lf(7, 2),
            lf(8, 1),
            FrameProgression::Coefficients {
                physical_frame_index: 9,
                completed_passes: 0,
                total_passes: 3,
                intended_downsampling: 8,
            },
            FrameProgression::Coefficients {
                physical_frame_index: 9,
                completed_passes: 1,
                total_passes: 3,
                intended_downsampling: 2,
            },
        ];
        for (index, current) in levels.iter().enumerate() {
            assert!(current.valid());
            assert!(!current.follows(*current), "duplicate boundary");
            for prior in &levels[..index] {
                assert!(current.follows(*prior));
                assert!(!prior.follows(*current));
            }
        }
        assert!(
            !lf(8, 1).follows(lf(8, 2)),
            "one physical LF cannot have two levels"
        );
        assert!(
            !lf(9, 2).follows(lf(8, 1)),
            "later LF may not reduce detail"
        );
    }
}
