use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::{
    ACCELERATION_INDEX_BOX_TYPE, CODESTREAM_SIGNATURE, Gray8AccelerationIndex, ParseLimits,
};

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
}

impl<S> PreparedGpuSession<S> {
    #[must_use]
    pub const fn new(profile: DecodeProfile, metadata: AnimationMetadata, session: S) -> Self {
        Self {
            profile,
            metadata,
            session,
        }
    }
}

/// Per-codestream GPU submission state.
///
/// `next_frame` may synchronously wait for GPU completion. `poll_next_frame` must not block: it
/// registers `context.waker()` before returning `Pending` and wakes it after GPU callback or input
/// progress makes another poll useful.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuSubmissionSession: Send + 'static {
    type Frame: Send + 'static;

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>>;

    fn poll_next_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>>;
}

/// Browser WebGPU session handles are main-thread-local. The async contract is
/// otherwise identical to the native session contract above.
#[cfg(target_arch = "wasm32")]
pub trait GpuSubmissionSession: 'static {
    type Frame: 'static;

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>>;

    fn poll_next_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>>;
}

/// Opens validated codestreams as GPU-only frame submission sessions.
///
/// Implementations own header/entropy/group parsing and GPU packet submission. Returning CPU pixel
/// data through `Frame` violates this crate's contract; unsupported input must instead return
/// [`Error::UnsupportedProfile`] or [`Error::FrontendIncomplete`].
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuSubmissionEngine: Send + Sync + 'static {
    type Session: GpuSubmissionSession;

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
        Self {
            engine: Arc::new(engine),
            parse_limits: ParseLimits::default(),
        }
    }

    #[must_use]
    pub fn from_shared(engine: Arc<E>) -> Self {
        Self {
            engine,
            parse_limits: ParseLimits::default(),
        }
    }

    #[must_use]
    pub const fn with_parse_limits(mut self, parse_limits: ParseLimits) -> Self {
        self.parse_limits = parse_limits;
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
        self.open_shared(Arc::from(encoded), request)
    }

    /// Shared-input form that avoids copying a raw codestream before engine handoff.
    pub fn open_shared(
        &self,
        encoded: Arc<[u8]>,
        request: GpuOutputRequest,
    ) -> Result<GpuDecodeSession<E::Session>> {
        request.format.validate()?;
        let codestream = parse_shared(encoded, self.parse_limits)?;
        let prepared = self.engine.open(codestream, &request)?;
        GpuDecodeSession::new(prepared, request)
    }
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
        return Ok(GpuCodestream::new(encoded, false, None));
    }
    let parsed = jxl_gpu_bitstream::parse(&encoded, limits)?;
    let is_container = parsed.is_container();
    let mut index_boxes = parsed.boxes_of_type(ACCELERATION_INDEX_BOX_TYPE);
    let acceleration_index = index_boxes
        .next()
        .map(|item| Gray8AccelerationIndex::parse_bound(item.payload, parsed.codestream()))
        .transpose()?;
    if index_boxes.next().is_some() {
        return Err(Error::DuplicateAccelerationIndex);
    }
    let codestream = Arc::from(parsed.codestream());
    Ok(GpuCodestream::new(
        codestream,
        is_container,
        acceleration_index,
    ))
}

/// Bounded ownership wrapper for one GPU-resident output.
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

/// Sequential GPU-only decode session with bounded returned frame ownership.
pub struct GpuDecodeSession<S: GpuSubmissionSession> {
    engine: S,
    request: GpuOutputRequest,
    profile: DecodeProfile,
    metadata: AnimationMetadata,
    limiter: InFlightLimiter,
    pending_permit: Option<InFlightPermit>,
    next_index: usize,
    finished: bool,
    failed: bool,
}

impl<S: GpuSubmissionSession> GpuDecodeSession<S> {
    fn new(prepared: PreparedGpuSession<S>, request: GpuOutputRequest) -> Result<Self> {
        validate_stream_metadata(&prepared.metadata)?;
        validate_profile(prepared.profile)?;
        Ok(Self {
            engine: prepared.session,
            profile: prepared.profile,
            metadata: prepared.metadata,
            limiter: InFlightLimiter::new(request.max_in_flight),
            request,
            pending_permit: None,
            next_index: 0,
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

    /// Concrete GPU submission state, for backend-specific diagnostics such as memory accounting.
    #[must_use]
    pub const fn submission_session(&self) -> &S {
        &self.engine
    }

    #[must_use]
    pub const fn frames_submitted(&self) -> usize {
        self.next_index
    }

    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.limiter.in_flight()
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    /// Synchronously submits/waits for the next GPU frame according to the engine contract.
    pub fn next_frame(&mut self) -> Result<Option<GpuFrameLease<S::Frame>>> {
        if self.failed {
            return Err(Error::SessionPoisoned);
        }
        if self.finished {
            return Ok(None);
        }
        if self.pending_permit.is_some() {
            return Err(Error::OperationInProgress);
        }
        let permit = self.limiter.try_acquire().ok_or(Error::Backpressure {
            limit: self.limiter.limit(),
        })?;
        let submitted = match self.engine.next_frame() {
            Ok(submitted) => submitted,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        self.finish_frame(permit, submitted)
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
        let permit = match self.pending_permit.take() {
            Some(permit) => permit,
            None => match self.limiter.poll_acquire(context) {
                Poll::Ready(permit) => permit,
                Poll::Pending => return Poll::Pending,
            },
        };
        match self.engine.poll_next_frame(context) {
            Poll::Pending => {
                self.pending_permit = Some(permit);
                Poll::Pending
            }
            Poll::Ready(Ok(submitted)) => Poll::Ready(self.finish_frame(permit, submitted)),
            Poll::Ready(Err(error)) => {
                self.failed = true;
                Poll::Ready(Err(error))
            }
        }
    }

    pub const fn next_frame_async(&mut self) -> NextGpuFrame<'_, S> {
        NextGpuFrame { session: self }
    }

    fn finish_frame(
        &mut self,
        permit: InFlightPermit,
        submitted: Option<SubmittedGpuFrame<S::Frame>>,
    ) -> Result<Option<GpuFrameLease<S::Frame>>> {
        let Some(submitted) = submitted else {
            self.failed = true;
            return Err(Error::MissingFinalFrame);
        };
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
        if let Some(hint) = self.metadata.frame_count_hint
            && (next_index > hint || (submitted.metadata.is_last && next_index != hint))
        {
            self.failed = true;
            return Err(Error::FrameCountMismatch {
                hint,
                actual: next_index,
            });
        }
        self.next_index = next_index;
        self.finished = submitted.metadata.is_last;
        Ok(Some(GpuFrameLease {
            metadata: submitted.metadata,
            output: submitted.output,
            _permit: permit,
        }))
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
            bits_per_sample: 8 | 16,
            ..
        } => Ok(()),
        DecodeProfile::ModularLossless { .. } => Err(Error::EngineContract(
            "prototype Modular profile must use 8 or 16 bits per sample",
        )),
    }
}
