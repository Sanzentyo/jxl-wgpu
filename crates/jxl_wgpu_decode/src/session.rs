use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{CODESTREAM_SIGNATURE, ParseLimits};

use crate::{
    AnimationMetadata, DecodeProfile, Error, FrameMetadata, GpuCodestream, GpuOutputRequest,
    InFlightLimiter, InFlightPermit, Result,
};

/// GPU-resident frame returned by an engine before bounded lease wrapping.
#[derive(Debug)]
pub struct SubmittedGpuFrame<F> {
    pub metadata: FrameMetadata,
    pub output: F,
}

impl<F> SubmittedGpuFrame<F> {
    #[must_use]
    pub const fn new(metadata: FrameMetadata, output: F) -> Self {
        Self { metadata, output }
    }
}

/// GPU session plus metadata produced after header/profile negotiation.
#[derive(Debug)]
pub struct PreparedGpuSession<S> {
    pub profile: DecodeProfile,
    pub metadata: AnimationMetadata,
    pub session: S,
    resolved_frame_slots: Option<NonZeroUsize>,
}

impl<S> PreparedGpuSession<S> {
    #[must_use]
    pub const fn new(profile: DecodeProfile, metadata: AnimationMetadata, session: S) -> Self {
        Self {
            profile,
            metadata,
            session,
            resolved_frame_slots: None,
        }
    }

    /// Narrows the caller's requested frame window to the number of slots this prepared backend
    /// session can actually keep resident within its device and shared-memory budgets.
    #[must_use]
    pub const fn with_resolved_frame_slots(mut self, slots: NonZeroUsize) -> Self {
        self.resolved_frame_slots = Some(slots);
        self
    }

    /// Backend-resolved frame-slot bound, or `None` when the request remains unchanged.
    #[must_use]
    pub const fn resolved_frame_slots(&self) -> Option<NonZeroUsize> {
        self.resolved_frame_slots
    }
}

/// One already-submitted GPU frame whose host-visible validation has not necessarily completed.
///
/// Submission and completion are deliberately separate: codec sessions can enqueue several
/// animation frames before waiting for the oldest one. Completion must not reorder the public
/// frame stream; [`GpuDecodeSession`] always consumes pending values from its queue front.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuPendingFrame: Send + Unpin + 'static {
    type Frame: Send + 'static;

    /// Blocks until this exact submission completes and returns its validated GPU frame.
    fn wait(self) -> Result<SubmittedGpuFrame<Self::Frame>>;

    /// Polls this exact submission without blocking an executor thread.
    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>>;
}

/// Browser WebGPU pending handles are main-thread-local and complete through the event loop.
#[cfg(target_arch = "wasm32")]
pub trait GpuPendingFrame: Unpin + 'static {
    type Frame: 'static;

    fn poll_complete(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>>;
}

/// Per-codestream GPU submission state. `submit_next` records queue work only and never waits for
/// host-visible completion.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuSubmissionSession: Send + 'static {
    type Frame: Send + 'static;
    type Pending: GpuPendingFrame<Frame = Self::Frame>;

    /// Enqueues the next visible frame. `None` means the submission stream reached its end.
    fn submit_next(&mut self) -> Result<Option<Self::Pending>>;
}

/// Browser WebGPU session handles are main-thread-local. The async contract is
/// otherwise identical to the native session contract above.
#[cfg(target_arch = "wasm32")]
pub trait GpuSubmissionSession: 'static {
    type Frame: 'static;
    type Pending: GpuPendingFrame<Frame = Self::Frame>;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>>;
}

/// Exact reason a prefetch attempt stopped below its requested queue depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchBackpressure {
    /// Pending frames plus caller-owned frame leases exhausted the configured frame slots.
    FrameSlots { limit: usize },
    /// The shared byte-weighted GPU memory budget rejected the next submission.
    Memory(jxl_wgpu::MemoryBudgetError),
    /// The backend's bounded native submission-poll worker rejected the next submission.
    SubmissionPoller(jxl_wgpu::SubmissionPollerError),
}

/// Observable result of filling the ordered pending-frame queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefetchProgress {
    /// Total frames successfully submitted by this decode session since it opened.
    pub submitted: usize,
    /// Frames currently queued in submission order and not yet returned to the caller.
    pub queued: usize,
    /// Whether the submission engine has explicitly returned end-of-stream.
    pub end_reached: bool,
    /// Why the requested depth could not be reached. `None` means target depth or stream end.
    pub backpressure: Option<PrefetchBackpressure>,
}

/// Opens validated codestreams as GPU-only frame submission sessions.
///
/// Implementations own header/entropy/group parsing and GPU packet submission. Returning CPU pixel
/// data through `Frame` violates this crate's contract; unsupported input must instead return
/// [`Error::UnsupportedProfile`] or [`Error::FrontendIncomplete`].
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuSubmissionEngine: Send + Sync + 'static {
    type Session: GpuSubmissionSession;

    /// Hard parser limits for this engine's implemented GPU profiles.
    fn parse_limits(&self) -> ParseLimits {
        ParseLimits::default()
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
    ) -> Result<PreparedGpuSession<Self::Session>>;
}

/// Browser WebGPU engines stay on the JavaScript main thread and therefore do
/// not require `Send + Sync`.
#[cfg(target_arch = "wasm32")]
pub trait GpuSubmissionEngine: 'static {
    type Session: GpuSubmissionSession;

    /// Hard parser limits for this engine's implemented GPU profiles.
    fn parse_limits(&self) -> ParseLimits {
        ParseLimits::default()
    }

    fn open(
        &self,
        codestream: GpuCodestream,
        request: &GpuOutputRequest,
    ) -> Result<PreparedGpuSession<Self::Session>>;
}

/// GPU-required JPEG XL decoder. There is no CPU constructor or fallback policy.
pub struct GpuDecoder<E: GpuSubmissionEngine> {
    engine: Arc<E>,
    parse_limits: ParseLimits,
}

impl<E: GpuSubmissionEngine> GpuDecoder<E> {
    #[must_use]
    pub fn new(engine: E) -> Self {
        let parse_limits = engine.parse_limits();
        Self {
            engine: Arc::new(engine),
            parse_limits,
        }
    }

    #[must_use]
    pub fn from_shared(engine: Arc<E>) -> Self {
        let parse_limits = engine.parse_limits();
        Self {
            engine,
            parse_limits,
        }
    }

    #[must_use]
    pub fn with_parse_limits(mut self, parse_limits: ParseLimits) -> Self {
        let engine_limits = self.engine.parse_limits();
        self.parse_limits = intersect_parse_limits(parse_limits, engine_limits);
        self
    }

    #[must_use]
    pub const fn parse_limits(&self) -> ParseLimits {
        self.parse_limits
    }

    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Validates/canonicalizes the container and opens a GPU-only decode session.
    pub fn open(
        &self,
        encoded: &[u8],
        request: GpuOutputRequest,
    ) -> Result<GpuDecodeSession<E::Session>> {
        validate_total_input_size(encoded.len(), self.parse_limits)?;
        self.open_shared(Arc::from(encoded), request)
    }

    /// Shared-input form that avoids copying a raw codestream before engine handoff.
    pub fn open_shared(
        &self,
        encoded: Arc<[u8]>,
        request: GpuOutputRequest,
    ) -> Result<GpuDecodeSession<E::Session>> {
        request.format().validate()?;
        let codestream = parse_shared(encoded, self.parse_limits)?;
        let prepared = self.engine.open(codestream, &request)?;
        GpuDecodeSession::new(prepared, request)
    }
}

const fn intersect_parse_limits(requested: ParseLimits, engine: ParseLimits) -> ParseLimits {
    ParseLimits {
        max_input_bytes: if requested.max_input_bytes < engine.max_input_bytes {
            requested.max_input_bytes
        } else {
            engine.max_input_bytes
        },
        max_boxes: if requested.max_boxes < engine.max_boxes {
            requested.max_boxes
        } else {
            engine.max_boxes
        },
        max_box_bytes: if requested.max_box_bytes < engine.max_box_bytes {
            requested.max_box_bytes
        } else {
            engine.max_box_bytes
        },
        max_codestream_bytes: if requested.max_codestream_bytes < engine.max_codestream_bytes {
            requested.max_codestream_bytes
        } else {
            engine.max_codestream_bytes
        },
    }
}

fn validate_total_input_size(length: usize, limits: ParseLimits) -> Result<()> {
    if u64::try_from(length).map_err(|_| jxl_gpu_bitstream::Error::SizeOverflow)?
        > limits.max_input_bytes
    {
        return Err(jxl_gpu_bitstream::Error::ResourceLimit("input size").into());
    }
    Ok(())
}

impl<E: GpuSubmissionEngine> Clone for GpuDecoder<E> {
    fn clone(&self) -> Self {
        Self {
            engine: Arc::clone(&self.engine),
            parse_limits: self.parse_limits,
        }
    }
}

impl<E: GpuSubmissionEngine + fmt::Debug> fmt::Debug for GpuDecoder<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuDecoder")
            .field("engine", &self.engine)
            .field("parse_limits", &self.parse_limits)
            .finish()
    }
}

fn parse_shared(encoded: Arc<[u8]>, limits: ParseLimits) -> Result<GpuCodestream> {
    if encoded.starts_with(&CODESTREAM_SIGNATURE) {
        jxl_gpu_bitstream::parse(&encoded, limits)?;
        let length = encoded.len();
        return Ok(GpuCodestream::new(encoded, 0..length, false));
    }
    let parsed = jxl_gpu_bitstream::parse(&encoded, limits)?;
    let is_container = parsed.is_container();
    let (storage, byte_range) = match parsed.into_codestream() {
        Cow::Borrowed(codestream) => {
            let storage_start = encoded.as_ptr() as usize;
            let codestream_start = codestream.as_ptr() as usize;
            let start =
                codestream_start
                    .checked_sub(storage_start)
                    .ok_or(Error::EngineContract(
                        "borrowed codestream is outside its input storage",
                    ))?;
            let end = start
                .checked_add(codestream.len())
                .filter(|&end| end <= encoded.len())
                .ok_or(Error::EngineContract(
                    "borrowed codestream range exceeds its input storage",
                ))?;
            (Arc::clone(&encoded), start..end)
        }
        Cow::Owned(codestream) => {
            let length = codestream.len();
            (Arc::from(codestream), 0..length)
        }
    };
    Ok(GpuCodestream::new(storage, byte_range, is_container))
}

/// Bounded ownership wrapper for one GPU-resident output.
///
/// This wrapper alone owns the session's frame-slot permit. Borrowing `output` and cloning a nested
/// GPU handle does not retain that count slot. The stock wgpu frame/output containers are
/// intentionally non-cloneable; custom submission engines must apply the same distinction to
/// their own output types.
pub struct GpuFrameLease<F> {
    pub metadata: FrameMetadata,
    output: F,
    _permit: InFlightPermit,
}

impl<F> GpuFrameLease<F> {
    #[must_use]
    pub fn output(&self) -> &F {
        &self.output
    }
}

impl<F: fmt::Debug> fmt::Debug for GpuFrameLease<F> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuFrameLease")
            .field("metadata", &self.metadata)
            .field("output", &self.output)
            .finish()
    }
}

/// Ordered GPU-only decode session with bounded pending and returned frame ownership.
pub struct GpuDecodeSession<S: GpuSubmissionSession> {
    engine: S,
    request: GpuOutputRequest,
    profile: DecodeProfile,
    metadata: AnimationMetadata,
    limiter: InFlightLimiter,
    pending: VecDeque<(InFlightPermit, S::Pending)>,
    submitted_count: usize,
    next_index: usize,
    next_presentation_ticks: u64,
    engine_end_reached: bool,
    finished: bool,
    failed: bool,
}

impl<S: GpuSubmissionSession> GpuDecodeSession<S> {
    fn new(prepared: PreparedGpuSession<S>, request: GpuOutputRequest) -> Result<Self> {
        validate_stream_metadata(&prepared.metadata)?;
        validate_profile(prepared.profile)?;
        let resolved_frame_slots = prepared
            .resolved_frame_slots
            .unwrap_or_else(|| request.max_frame_slots());
        if resolved_frame_slots > request.max_frame_slots() {
            return Err(Error::EngineContract(
                "resolved frame-slot limit exceeds the caller's request",
            ));
        }
        Ok(Self {
            engine: prepared.session,
            profile: prepared.profile,
            metadata: prepared.metadata,
            limiter: InFlightLimiter::new(resolved_frame_slots),
            request,
            pending: VecDeque::new(),
            submitted_count: 0,
            next_index: 0,
            next_presentation_ticks: 0,
            engine_end_reached: false,
            finished: false,
            failed: false,
        })
    }

    #[must_use]
    pub const fn profile(&self) -> DecodeProfile {
        self.profile
    }

    #[must_use]
    pub const fn metadata(&self) -> &AnimationMetadata {
        &self.metadata
    }

    #[must_use]
    pub const fn request(&self) -> &GpuOutputRequest {
        &self.request
    }

    /// Effective frame-slot limit after backend memory/device admission narrows the request.
    #[must_use]
    pub fn resolved_frame_slots(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.limiter.limit()).expect("the frame-slot limiter is nonzero")
    }

    /// Concrete GPU submission state, for backend-specific diagnostics such as memory accounting.
    #[must_use]
    pub const fn submission_session(&self) -> &S {
        &self.engine
    }

    #[must_use]
    pub const fn frames_submitted(&self) -> usize {
        self.submitted_count
    }

    #[must_use]
    pub fn queued_frames(&self) -> usize {
        self.pending.len()
    }

    /// Borrows the submitted pending frames in public presentation order.
    ///
    /// This read-only view does not complete, remove, or reorder submissions and does not expose
    /// the session's frame-slot permits. Backend-specific pending types may use the borrow for
    /// explicitly unvalidated same-queue handoff. Authoritative frame metadata remains available
    /// only from [`Self::next_frame`] or [`Self::next_frame_async`].
    pub fn pending_frames(
        &self,
    ) -> impl DoubleEndedIterator<Item = &S::Pending> + ExactSizeIterator {
        self.pending.iter().map(|(_, pending)| pending)
    }

    /// Borrows the oldest submitted pending frame without completing it.
    #[must_use]
    pub fn front_pending_frame(&self) -> Option<&S::Pending> {
        self.pending.front().map(|(_, pending)| pending)
    }

    /// Number of slots currently occupied by queued submissions and caller-held frame leases.
    #[must_use]
    pub fn active_frame_slots(&self) -> usize {
        self.limiter.active_slots()
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Submits frames without waiting until the ordered pending queue reaches `target_depth`, the
    /// engine reports end-of-stream, or a typed admission boundary is encountered.
    ///
    /// The target is a queue depth, not an additional count, and cannot exceed the request's
    /// backend-resolved frame-slot bound. Already returned frame leases retain their slots and can
    /// therefore make a synchronous attempt report [`PrefetchBackpressure::FrameSlots`].
    pub fn prefetch(&mut self, target_depth: NonZeroUsize) -> Result<PrefetchProgress> {
        self.validate_prefetch(target_depth)?;
        if self.failed {
            return Err(Error::SessionPoisoned);
        }
        if self.finished {
            self.engine_end_reached = true;
            return Ok(self.prefetch_progress(None));
        }
        while self.pending.len() < target_depth.get() && !self.engine_end_reached {
            let Some(permit) = self.limiter.try_acquire() else {
                return Ok(
                    self.prefetch_progress(Some(PrefetchBackpressure::FrameSlots {
                        limit: self.limiter.limit(),
                    })),
                );
            };
            if let Some(progress) = self.submit_one(permit)? {
                return Ok(progress);
            }
        }
        Ok(self.prefetch_progress(None))
    }

    /// Nonblocking prefetch counterpart. `Pending` is returned only while waiting for a retained
    /// frame lease to release a slot; memory and poll-worker admission failures are returned as a
    /// ready [`PrefetchProgress`] because those external budgets do not own task wakers.
    pub fn poll_prefetch(
        &mut self,
        target_depth: NonZeroUsize,
        context: &mut Context<'_>,
    ) -> Poll<Result<PrefetchProgress>> {
        if let Err(error) = self.validate_prefetch(target_depth) {
            return Poll::Ready(Err(error));
        }
        if self.failed {
            return Poll::Ready(Err(Error::SessionPoisoned));
        }
        if self.finished {
            self.engine_end_reached = true;
            return Poll::Ready(Ok(self.prefetch_progress(None)));
        }
        while self.pending.len() < target_depth.get() && !self.engine_end_reached {
            let permit = match self.limiter.poll_acquire(context) {
                Poll::Ready(permit) => permit,
                Poll::Pending => return Poll::Pending,
            };
            match self.submit_one(permit) {
                Ok(Some(progress)) => return Poll::Ready(Ok(progress)),
                Ok(None) => {}
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
        Poll::Ready(Ok(self.prefetch_progress(None)))
    }

    #[must_use = "futures do nothing unless polled"]
    pub const fn prefetch_async(&mut self, target_depth: NonZeroUsize) -> PrefetchGpuFrames<'_, S> {
        PrefetchGpuFrames {
            session: self,
            target_depth,
        }
    }

    /// Waits for and returns the oldest submitted GPU frame. If the queue is empty, exactly one
    /// frame is submitted first.
    pub fn next_frame(&mut self) -> Result<Option<GpuFrameLease<S::Frame>>> {
        if self.failed {
            return Err(Error::SessionPoisoned);
        }
        if self.finished {
            return Ok(None);
        }
        self.ensure_front_sync()?;
        let (permit, pending) = self.pending.pop_front().ok_or(Error::EngineContract(
            "prefetch reported a frame without queueing a pending submission",
        ))?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            match pending.wait() {
                Ok(submitted) => self.finish_frame(permit, submitted).map(Some),
                Err(error) => {
                    self.failed = true;
                    Err(error)
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            drop((permit, pending));
            Err(Error::BlockingWaitUnavailable)
        }
    }

    /// Polls GPU submission/completion without blocking an executor thread.
    pub fn poll_next_frame(
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
                    if let Some(backpressure) = progress.backpressure {
                        return Poll::Ready(Err(prefetch_backpressure_error(backpressure)));
                    }
                }
            }
        }
        if self.pending.is_empty() {
            self.failed = true;
            return Poll::Ready(Err(Error::MissingFinalFrame));
        }
        let completion = {
            let (_, pending) = self
                .pending
                .front_mut()
                .expect("the pending queue was checked as non-empty");
            Pin::new(pending).poll_complete(context)
        };
        match completion {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                let (permit, _) = self
                    .pending
                    .pop_front()
                    .expect("the completed pending frame remains at the queue front");
                match result {
                    Ok(submitted) => Poll::Ready(self.finish_frame(permit, submitted).map(Some)),
                    Err(error) => {
                        self.failed = true;
                        Poll::Ready(Err(error))
                    }
                }
            }
        }
    }

    pub const fn next_frame_async(&mut self) -> NextGpuFrame<'_, S> {
        NextGpuFrame { session: self }
    }

    fn validate_prefetch(&self, target_depth: NonZeroUsize) -> Result<()> {
        if target_depth.get() > self.limiter.limit() {
            return Err(Error::PrefetchDepthExceedsLimit {
                requested: target_depth.get(),
                limit: self.limiter.limit(),
            });
        }
        Ok(())
    }

    fn submit_one(&mut self, permit: InFlightPermit) -> Result<Option<PrefetchProgress>> {
        let next_submitted_count = self.submitted_count.checked_add(1).ok_or_else(|| {
            self.failed = true;
            Error::EngineContract("submitted frame count overflow")
        })?;
        match self.engine.submit_next() {
            Ok(Some(pending)) => {
                self.submitted_count = next_submitted_count;
                self.pending.push_back((permit, pending));
                Ok(None)
            }
            Ok(None) => {
                self.engine_end_reached = true;
                drop(permit);
                Ok(Some(self.prefetch_progress(None)))
            }
            Err(Error::MemoryBackpressure(error)) => {
                drop(permit);
                Ok(Some(self.prefetch_progress(Some(
                    PrefetchBackpressure::Memory(error),
                ))))
            }
            Err(Error::PollBackpressure(error @ jxl_wgpu::SubmissionPollerError::Full { .. })) => {
                drop(permit);
                Ok(Some(self.prefetch_progress(Some(
                    PrefetchBackpressure::SubmissionPoller(error),
                ))))
            }
            Err(error) => {
                drop(permit);
                self.failed = true;
                Err(error)
            }
        }
    }

    fn prefetch_progress(&self, backpressure: Option<PrefetchBackpressure>) -> PrefetchProgress {
        PrefetchProgress {
            submitted: self.submitted_count,
            queued: self.pending.len(),
            end_reached: self.engine_end_reached || self.finished,
            backpressure,
        }
    }

    fn ensure_front_sync(&mut self) -> Result<()> {
        if !self.pending.is_empty() {
            return Ok(());
        }
        let progress = self.prefetch(NonZeroUsize::new(1).expect("one is nonzero"))?;
        if let Some(backpressure) = progress.backpressure {
            return Err(prefetch_backpressure_error(backpressure));
        }
        if self.pending.is_empty() {
            self.failed = true;
            return Err(Error::MissingFinalFrame);
        }
        Ok(())
    }

    fn finish_frame(
        &mut self,
        permit: InFlightPermit,
        submitted: SubmittedGpuFrame<S::Frame>,
    ) -> Result<GpuFrameLease<S::Frame>> {
        if submitted.metadata.index != self.next_index {
            self.failed = true;
            return Err(Error::UnexpectedFrameIndex {
                expected: self.next_index,
                actual: submitted.metadata.index,
            });
        }
        if submitted.metadata.duration.timebase != self.metadata.timebase {
            self.failed = true;
            return Err(Error::FrameTimebaseMismatch {
                index: submitted.metadata.index,
            });
        }
        if self.metadata.timebase.is_none() && submitted.metadata.duration.ticks != 0 {
            self.failed = true;
            return Err(Error::FrameTimebaseMismatch {
                index: submitted.metadata.index,
            });
        }
        if submitted.metadata.presentation_ticks != self.next_presentation_ticks {
            self.failed = true;
            return Err(Error::FramePresentationTicksMismatch {
                index: submitted.metadata.index,
                expected: self.next_presentation_ticks,
                actual: submitted.metadata.presentation_ticks,
            });
        }
        let stream_has_timecodes = self.metadata.has_timecodes.unwrap_or(false);
        let frame_has_timecode = submitted.metadata.timecode.is_some();
        if frame_has_timecode != stream_has_timecodes {
            self.failed = true;
            return Err(Error::FrameTimecodePresenceMismatch {
                index: submitted.metadata.index,
                stream_has_timecodes,
                frame_has_timecode,
            });
        }
        if self.metadata.timebase.is_none()
            && (submitted.metadata.index != 0 || !submitted.metadata.is_last)
        {
            self.failed = true;
            return Err(Error::EngineContract(
                "still-image metadata requires exactly one final visible frame",
            ));
        }
        let Some(next_index) = self.next_index.checked_add(1) else {
            self.failed = true;
            return Err(Error::EngineContract("visible frame index overflow"));
        };
        let next_presentation_ticks = self
            .next_presentation_ticks
            .checked_add(u64::from(submitted.metadata.duration.ticks))
            .ok_or_else(|| {
                self.failed = true;
                Error::FramePresentationTicksOverflow {
                    index: submitted.metadata.index,
                }
            })?;
        if let Some(hint) = self.metadata.frame_count_hint
            && (next_index > hint || (submitted.metadata.is_last && next_index != hint))
        {
            self.failed = true;
            return Err(Error::FrameCountMismatch {
                hint,
                actual: next_index,
            });
        }
        if submitted.metadata.is_last && !self.pending.is_empty() {
            self.failed = true;
            return Err(Error::EngineContract(
                "submission engine queued visible frames after the final frame",
            ));
        }
        self.next_index = next_index;
        self.next_presentation_ticks = next_presentation_ticks;
        self.finished = submitted.metadata.is_last;
        if self.finished {
            self.engine_end_reached = true;
        }
        Ok(GpuFrameLease {
            metadata: submitted.metadata,
            output: submitted.output,
            _permit: permit,
        })
    }
}

fn prefetch_backpressure_error(backpressure: PrefetchBackpressure) -> Error {
    match backpressure {
        PrefetchBackpressure::FrameSlots { limit } => Error::Backpressure { limit },
        PrefetchBackpressure::Memory(error) => Error::MemoryBackpressure(error),
        PrefetchBackpressure::SubmissionPoller(error) => Error::PollBackpressure(error),
    }
}

#[must_use = "futures do nothing unless polled"]
pub struct PrefetchGpuFrames<'session, S: GpuSubmissionSession> {
    session: &'session mut GpuDecodeSession<S>,
    target_depth: NonZeroUsize,
}

impl<S: GpuSubmissionSession> Future for PrefetchGpuFrames<'_, S> {
    type Output = Result<PrefetchProgress>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let future = self.get_mut();
        future.session.poll_prefetch(future.target_depth, context)
    }
}

#[must_use = "futures do nothing unless polled"]
pub struct NextGpuFrame<'session, S: GpuSubmissionSession> {
    session: &'session mut GpuDecodeSession<S>,
}

impl<S: GpuSubmissionSession> Future for NextGpuFrame<'_, S> {
    type Output = Result<Option<GpuFrameLease<S::Frame>>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().session.poll_next_frame(context)
    }
}

fn validate_stream_metadata(metadata: &AnimationMetadata) -> Result<()> {
    if metadata.extent.is_empty() {
        return Err(Error::EngineContract("stream extent must be non-empty"));
    }
    let timing_is_still = metadata.timebase.is_none()
        && metadata.loop_count.is_none()
        && metadata.has_timecodes.is_none();
    let timing_is_animation = metadata.timebase.is_some()
        && metadata.loop_count.is_some()
        && metadata.has_timecodes.is_some();
    if !timing_is_still && !timing_is_animation {
        return Err(Error::EngineContract(
            "animation timebase, loop count, and timecode flag must be present together",
        ));
    }
    if metadata.frame_count_hint == Some(0) {
        return Err(Error::EngineContract(
            "frame count hint cannot declare zero visible frames",
        ));
    }
    Ok(())
}

fn validate_profile(profile: DecodeProfile) -> Result<()> {
    match profile {
        DecodeProfile::ModularLossless {
            bits_per_sample: 1..=16,
            prediction,
            ..
        } => match prediction {
            crate::ModularPredictionProfile::Fixed { .. } => Ok(()),
            crate::ModularPredictionProfile::MetaAdaptive {
                node_count,
                decision_node_count,
                leaf_context_count,
                max_depth,
                ..
            } if node_count != 0
                && leaf_context_count != 0
                && max_depth != 0
                && decision_node_count.checked_add(leaf_context_count) == Some(node_count) =>
            {
                Ok(())
            }
            crate::ModularPredictionProfile::MetaAdaptive { .. } => Err(Error::EngineContract(
                "MA prediction profile has inconsistent node/context/depth metadata",
            )),
        },
        DecodeProfile::ModularLossless { .. } => Err(Error::EngineContract(
            "lossless Modular profile must use 1 through 16 bits per sample",
        )),
        DecodeProfile::VarDctRegular {
            bits_per_sample: 8,
            transform,
        } if !transform.is_special() => Ok(()),
        DecodeProfile::VarDctRegular { .. } => Err(Error::EngineContract(
            "the bounded VarDCT profile requires 8-bit samples and one regular transform",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(input: &str) -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = input
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }

    #[test]
    fn raw_and_single_codestream_container_share_the_caller_allocation() {
        for encoded in [
            fixture(include_str!("../test-data/basic.jxl.hex")),
            fixture(include_str!("../test-data/gpu_gray8_lossless.jxl.hex")),
        ] {
            let storage: Arc<[u8]> = Arc::from(encoded);
            let parsed = parse_shared(Arc::clone(&storage), ParseLimits::default()).unwrap();
            assert!(Arc::ptr_eq(&storage, &parsed.shared_storage()));
            assert!(parsed.bytes().starts_with(&CODESTREAM_SIGNATURE));
            assert_eq!(
                &parsed.shared_storage()[parsed.storage_range()],
                parsed.bytes()
            );
        }
    }

    #[test]
    fn requested_parser_limits_can_only_tighten_engine_limits() {
        let engine = ParseLimits {
            max_input_bytes: 16,
            max_boxes: 8,
            max_box_bytes: 12,
            max_codestream_bytes: 14,
        };
        let requested = ParseLimits {
            max_input_bytes: 32,
            max_boxes: 4,
            max_box_bytes: 24,
            max_codestream_bytes: 7,
        };
        assert_eq!(
            intersect_parse_limits(requested, engine),
            ParseLimits {
                max_input_bytes: 16,
                max_boxes: 4,
                max_box_bytes: 12,
                max_codestream_bytes: 7,
            }
        );
        assert!(validate_total_input_size(17, engine).is_err());
        assert!(validate_total_input_size(16, engine).is_ok());
    }
}
