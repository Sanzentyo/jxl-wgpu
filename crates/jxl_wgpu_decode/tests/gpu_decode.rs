use std::collections::VecDeque;
use std::future::Future;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use jxl_gpu_bitstream::{
    CodestreamInventory, ContainerStreamEvent, ContainerStreamLimits, ContainerStreamScanner,
    InventoryLimits, ParseLimits,
};
use jxl_gpu_formats::{ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, PixelFormat};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu_decode::{
    AnimationMetadata, DecodeProfile, Error, FrameDuration, FrameMetadata, FrameTimebase,
    FrontendIncomplete, FrontendStage, GpuCodestream, GpuDecoder, GpuOutputRequest,
    GpuPendingFrame, GpuSubmissionEngine, GpuSubmissionSession, IncrementalInputBudget,
    ModularChannels, ModularGrouping, ModularPredictionProfile, ModularPredictor,
    PrefetchBackpressure, PreparedGpuSession, Result, SubmittedGpuFrame,
    UnsupportedCodestreamFeature, UnsupportedProfile,
};

mod common;

use common::{basic as raw_still, fragmented_animation};

#[derive(Clone, Debug, PartialEq, Eq)]
struct MockGpuFrame {
    resource_id: u64,
}

fn output_request(limit: usize) -> GpuOutputRequest {
    let color = ColorSpecification::Defined(ColorSpec::bt709(
        ColorRange::Limited,
        ChromaLocation2d::CENTER,
    ));
    GpuOutputRequest::color(PixelFormat::nv12(color))
        .unwrap()
        .with_max_frame_slots(NonZeroUsize::new(limit).unwrap())
}

fn assert_jxl_signature(codestream: &GpuCodestream) {
    let mut signature = [0; 2];
    codestream.copy_range(0..2, &mut signature).unwrap();
    assert_eq!(signature, [0xff, 0x0a]);
}

fn fixed_profile(bits_per_sample: u8, predictor: ModularPredictor) -> DecodeProfile {
    DecodeProfile::ModularLossless {
        bits_per_sample,
        channels: ModularChannels::Gray,
        prediction: ModularPredictionProfile::Fixed { predictor },
        grouping: ModularGrouping::SingleGroup,
    }
}

fn timebase() -> FrameTimebase {
    FrameTimebase {
        ticks_per_second_numerator: NonZeroU32::new(1_000).unwrap(),
        ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
    }
}

fn frame(index: usize, is_last: bool) -> SubmittedGpuFrame<MockGpuFrame> {
    SubmittedGpuFrame::new(
        FrameMetadata {
            index,
            duration: FrameDuration::animation(20, timebase()),
            presentation_ticks: u64::try_from(index).unwrap() * 20,
            timecode: None,
            is_last,
            is_keyframe: index == 0,
            name: format!("frame-{index}"),
        },
        MockGpuFrame {
            resource_id: index as u64,
        },
    )
}

#[derive(Debug)]
struct ReadyPending(Option<SubmittedGpuFrame<MockGpuFrame>>);

impl ReadyPending {
    fn take(&mut self) -> Result<SubmittedGpuFrame<MockGpuFrame>> {
        self.0
            .take()
            .ok_or(Error::EngineContract("mock pending frame completed twice"))
    }
}

impl GpuPendingFrame for ReadyPending {
    type Frame = MockGpuFrame;

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        self.take()
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        _context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        Poll::Ready(self.take())
    }
}

#[derive(Debug)]
struct ReadyEngine;

#[derive(Debug)]
struct ReadySession {
    frames: VecDeque<SubmittedGpuFrame<MockGpuFrame>>,
}

impl GpuSubmissionEngine for ReadyEngine {
    type Session = ReadySession;

    fn open(
        &self,
        codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        assert_jxl_signature(&codestream);
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            AnimationMetadata::animation(Extent2d::new(8, 6), timebase(), 0, false, Some(2)),
            ReadySession {
                frames: VecDeque::from([frame(0, false), frame(1, true)]),
            },
        ))
    }
}

impl GpuSubmissionSession for ReadySession {
    type Frame = MockGpuFrame;
    type Pending = ReadyPending;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        Ok(self
            .frames
            .pop_front()
            .map(|frame| ReadyPending(Some(frame))))
    }
}

#[test]
fn sync_gpu_frames_are_bounded_and_keep_exact_timing() {
    let decoder = GpuDecoder::new(ReadyEngine);
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    assert_eq!(session.profile(), fixed_profile(8, ModularPredictor::Zero));
    assert_eq!(session.metadata().extent, Extent2d::new(8, 6));

    let first = session.next_frame().unwrap().unwrap();
    assert_eq!(first.metadata.index, 0);
    assert_eq!(first.metadata.duration.ticks, 20);
    assert_eq!(first.metadata.duration.timebase, Some(timebase()));
    assert_eq!(first.metadata.presentation_ticks, 0);
    assert_eq!(first.metadata.timecode, None);
    assert_eq!(first.output().resource_id, 0);
    assert!(matches!(
        session.next_frame(),
        Err(Error::Backpressure { limit: 1 })
    ));
    drop(first);

    let last = session.next_frame().unwrap().unwrap();
    assert!(last.metadata.is_last);
    assert_eq!(last.metadata.presentation_ticks, 20);
    assert_eq!(last.output().resource_id, 1);
    // EOF does not require another output slot, even while the final lease remains live.
    assert!(session.next_frame().unwrap().is_none());
    assert_eq!(session.frames_submitted(), 2);
}

#[derive(Debug)]
struct TimecodeEngine {
    frame_timecode: Option<u32>,
    presentation_ticks: u64,
}

#[derive(Debug)]
struct TimecodeSession {
    frame_timecode: Option<u32>,
    presentation_ticks: u64,
    emitted: bool,
}

impl GpuSubmissionEngine for TimecodeEngine {
    type Session = TimecodeSession;

    fn open(
        &self,
        _codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            AnimationMetadata::animation(Extent2d::new(4, 3), timebase(), 1, true, Some(1)),
            TimecodeSession {
                frame_timecode: self.frame_timecode,
                presentation_ticks: self.presentation_ticks,
                emitted: false,
            },
        ))
    }
}

impl GpuSubmissionSession for TimecodeSession {
    type Frame = MockGpuFrame;
    type Pending = ReadyPending;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(Some(ReadyPending(Some(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::animation(25, timebase()),
                presentation_ticks: self.presentation_ticks,
                timecode: self.frame_timecode,
                is_last: true,
                is_keyframe: true,
                name: "timecoded".into(),
            },
            MockGpuFrame { resource_id: 99 },
        )))))
    }
}

#[test]
fn animation_frame_preserves_exact_bitstream_timecode() {
    let decoder = GpuDecoder::new(TimecodeEngine {
        frame_timecode: Some(0x1020_3040),
        presentation_ticks: 0,
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    assert_eq!(frame.metadata.timecode, Some(0x1020_3040));
}

#[test]
fn frame_timecode_presence_must_match_animation_header() {
    let decoder = GpuDecoder::new(TimecodeEngine {
        frame_timecode: None,
        presentation_ticks: 0,
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    assert!(matches!(
        session.next_frame(),
        Err(Error::FrameTimecodePresenceMismatch {
            index: 0,
            stream_has_timecodes: true,
            frame_has_timecode: false,
        })
    ));
}

#[test]
fn frame_presentation_ticks_must_equal_accumulated_durations() {
    let decoder = GpuDecoder::new(TimecodeEngine {
        frame_timecode: Some(7),
        presentation_ticks: 1,
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    assert!(matches!(
        session.next_frame(),
        Err(Error::FramePresentationTicksMismatch {
            index: 0,
            expected: 0,
            actual: 1,
        })
    ));
}

#[derive(Debug, Default)]
struct PendingControl {
    ready: AtomicBool,
    waker: Mutex<Option<Waker>>,
    #[cfg(not(target_arch = "wasm32"))]
    wait_lock: Mutex<()>,
    condition: Condvar,
}

impl PendingControl {
    fn complete(&self) {
        self.ready.store(true, Ordering::Release);
        self.condition.notify_all();
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(&self) {
        let mut guard = self.wait_lock.lock().unwrap();
        while !self.ready.load(Ordering::Acquire) {
            guard = self.condition.wait(guard).unwrap();
        }
    }
}

#[derive(Debug)]
struct ControlledPending {
    control: Arc<PendingControl>,
    frame: Option<SubmittedGpuFrame<MockGpuFrame>>,
}

impl ControlledPending {
    fn take(&mut self) -> Result<SubmittedGpuFrame<MockGpuFrame>> {
        self.frame
            .take()
            .ok_or(Error::EngineContract("mock pending frame completed twice"))
    }
}

impl GpuPendingFrame for ControlledPending {
    type Frame = MockGpuFrame;

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        self.control.wait();
        self.take()
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        if !self.control.ready.load(Ordering::Acquire) {
            *self.control.waker.lock().unwrap() = Some(context.waker().clone());
            return Poll::Pending;
        }
        Poll::Ready(self.take())
    }
}

#[derive(Clone, Debug)]
struct PendingEngine {
    control: Arc<PendingControl>,
}

struct PendingSession {
    control: Arc<PendingControl>,
    emitted: bool,
}

impl GpuSubmissionEngine for PendingEngine {
    type Session = PendingSession;

    fn open(
        &self,
        _codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        Ok(PreparedGpuSession::new(
            fixed_profile(16, ModularPredictor::West),
            AnimationMetadata::still(Extent2d::new(2, 2)),
            PendingSession {
                control: Arc::clone(&self.control),
                emitted: false,
            },
        ))
    }
}

impl GpuSubmissionSession for PendingSession {
    type Frame = MockGpuFrame;
    type Pending = ControlledPending;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(Some(ControlledPending {
            control: Arc::clone(&self.control),
            frame: Some(SubmittedGpuFrame::new(
                FrameMetadata {
                    index: 0,
                    duration: FrameDuration::still(),
                    presentation_ticks: 0,
                    timecode: None,
                    is_last: true,
                    is_keyframe: true,
                    name: String::new(),
                },
                MockGpuFrame { resource_id: 7 },
            )),
        }))
    }
}

#[derive(Default)]
struct WakeCounter(AtomicUsize);

impl Wake for WakeCounter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn future_is_runtime_neutral_and_woken_by_gpu_completion() {
    let control = Arc::new(PendingControl::default());
    let decoder = GpuDecoder::new(PendingEngine {
        control: Arc::clone(&control),
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(session.next_frame_async());

    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    control.complete();
    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    let frame = match future.as_mut().poll(&mut context) {
        Poll::Ready(Ok(Some(frame))) => frame,
        other => panic!("expected completed GPU frame, got {other:?}"),
    };
    assert_eq!(frame.output().resource_id, 7);
}

#[test]
fn cancelled_async_wait_can_be_resumed_synchronously() {
    let control = Arc::new(PendingControl::default());
    let decoder = GpuDecoder::new(PendingEngine {
        control: Arc::clone(&control),
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    let waker = Waker::from(Arc::new(WakeCounter::default()));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(session.next_frame_async());
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);

    control.complete();
    let frame = session
        .next_frame()
        .expect("sync wait resumes the submitted async operation")
        .expect("the resumed operation returns its frame");
    assert_eq!(frame.output().resource_id, 7);
}

#[derive(Clone, Debug)]
struct PrefetchAnimationEngine {
    controls: Arc<Vec<Arc<PendingControl>>>,
    submitted: Arc<AtomicUsize>,
}

#[derive(Debug)]
struct PrefetchAnimationSession {
    controls: VecDeque<Arc<PendingControl>>,
    submitted: Arc<AtomicUsize>,
    next_index: usize,
    frame_count: usize,
}

impl PrefetchAnimationEngine {
    fn new(frame_count: usize) -> Self {
        Self {
            controls: Arc::new(
                (0..frame_count)
                    .map(|_| Arc::new(PendingControl::default()))
                    .collect(),
            ),
            submitted: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl GpuSubmissionEngine for PrefetchAnimationEngine {
    type Session = PrefetchAnimationSession;

    fn open(
        &self,
        _codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            AnimationMetadata::animation(
                Extent2d::new(8, 6),
                timebase(),
                2,
                true,
                Some(self.controls.len()),
            ),
            PrefetchAnimationSession {
                controls: self.controls.iter().cloned().collect(),
                submitted: Arc::clone(&self.submitted),
                next_index: 0,
                frame_count: self.controls.len(),
            },
        ))
    }
}

impl GpuSubmissionSession for PrefetchAnimationSession {
    type Frame = MockGpuFrame;
    type Pending = ControlledPending;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        let Some(control) = self.controls.pop_front() else {
            return Ok(None);
        };
        let index = self.next_index;
        self.next_index += 1;
        self.submitted.fetch_add(1, Ordering::SeqCst);
        Ok(Some(ControlledPending {
            control,
            frame: Some(SubmittedGpuFrame::new(
                FrameMetadata {
                    index,
                    duration: FrameDuration::animation(10, timebase()),
                    presentation_ticks: u64::try_from(index).unwrap() * 10,
                    timecode: Some(0x1000 + u32::try_from(index).unwrap()),
                    is_last: index + 1 == self.frame_count,
                    is_keyframe: index == 0,
                    name: format!("prefetched-{index}"),
                },
                MockGpuFrame {
                    resource_id: u64::try_from(index).unwrap(),
                },
            )),
        }))
    }
}

#[test]
fn animation_prefetch_submits_before_completion_and_preserves_order_metadata_and_permits() {
    let engine = PrefetchAnimationEngine::new(4);
    let controls = Arc::clone(&engine.controls);
    let submitted = Arc::clone(&engine.submitted);
    let decoder = GpuDecoder::new(engine);
    let mut session = decoder.open(raw_still(), output_request(4)).unwrap();

    let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
    assert_eq!(progress.submitted, 3);
    assert_eq!(progress.queued, 3);
    assert!(!progress.end_reached);
    assert_eq!(progress.backpressure, None);
    assert_eq!(submitted.load(Ordering::SeqCst), 3);
    assert_eq!(session.active_frame_slots(), 3);

    let counter = Arc::new(WakeCounter::default());
    let waker = Waker::from(Arc::clone(&counter));
    let mut context = Context::from_waker(&waker);
    let mut first_future = Box::pin(session.next_frame_async());
    assert!(matches!(
        first_future.as_mut().poll(&mut context),
        Poll::Pending
    ));
    controls[2].complete();
    assert_eq!(counter.0.load(Ordering::SeqCst), 0);
    assert!(matches!(
        first_future.as_mut().poll(&mut context),
        Poll::Pending
    ));
    controls[0].complete();
    let first = match first_future.as_mut().poll(&mut context) {
        Poll::Ready(Ok(Some(frame))) => frame,
        other => panic!("expected ordered first frame, got {other:?}"),
    };
    drop(first_future);
    assert_eq!(first.metadata.index, 0);
    assert_eq!(first.metadata.presentation_ticks, 0);
    assert_eq!(first.metadata.timecode, Some(0x1000));
    assert_eq!(session.queued_frames(), 2);
    assert_eq!(
        session.active_frame_slots(),
        3,
        "two pending plus one returned lease"
    );

    let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
    assert_eq!(progress.submitted, 4);
    assert_eq!(progress.queued, 3);
    assert_eq!(session.active_frame_slots(), 4);
    let blocked = session.prefetch(NonZeroUsize::new(4).unwrap()).unwrap();
    assert_eq!(
        blocked.backpressure,
        Some(PrefetchBackpressure::FrameSlots { limit: 4 })
    );
    assert_eq!(blocked.queued, 3);

    drop(first);
    let end = session.prefetch(NonZeroUsize::new(4).unwrap()).unwrap();
    assert!(end.end_reached);
    assert_eq!(end.queued, 3);
    assert_eq!(end.backpressure, None);

    controls[1].complete();
    controls[3].complete();
    for expected in 1..4 {
        let frame = session.next_frame().unwrap().unwrap();
        assert_eq!(frame.metadata.index, expected);
        assert_eq!(
            frame.metadata.presentation_ticks,
            u64::try_from(expected).unwrap() * 10
        );
        assert_eq!(
            frame.metadata.timecode,
            Some(0x1000 + u32::try_from(expected).unwrap())
        );
        drop(frame);
    }
    assert!(session.next_frame().unwrap().is_none());
    assert_eq!(session.queued_frames(), 0);
    assert_eq!(session.active_frame_slots(), 0);
}

#[test]
fn prefetch_future_is_runtime_neutral_and_does_not_wait_for_frame_completion() {
    let engine = PrefetchAnimationEngine::new(3);
    let submitted = Arc::clone(&engine.submitted);
    let decoder = GpuDecoder::new(engine);
    let mut session = decoder.open(raw_still(), output_request(3)).unwrap();
    let waker = Waker::from(Arc::new(WakeCounter::default()));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(session.prefetch_async(NonZeroUsize::new(3).unwrap()));
    let progress = match future.as_mut().poll(&mut context) {
        Poll::Ready(Ok(progress)) => progress,
        other => panic!("prefetch should only submit queue work, got {other:?}"),
    };
    assert_eq!(progress.submitted, 3);
    assert_eq!(progress.queued, 3);
    assert_eq!(submitted.load(Ordering::SeqCst), 3);
}

#[test]
fn abandoned_prefetch_future_keeps_partial_queue_and_sync_prefetch_resumes_it() {
    let engine = PrefetchAnimationEngine::new(3);
    let controls = Arc::clone(&engine.controls);
    let decoder = GpuDecoder::new(engine);
    let mut session = decoder.open(raw_still(), output_request(2)).unwrap();

    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    controls[0].complete();
    let first = session.next_frame().unwrap().unwrap();
    let waker = Waker::from(Arc::new(WakeCounter::default()));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(session.prefetch_async(NonZeroUsize::new(2).unwrap()));
    assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
    drop(future);
    assert_eq!(session.frames_submitted(), 2);
    assert_eq!(session.queued_frames(), 1);

    drop(first);
    let resumed = session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
    assert_eq!(resumed.submitted, 3);
    assert_eq!(resumed.queued, 2);
    assert_eq!(resumed.backpressure, None);
    controls[1].complete();
    controls[2].complete();
    assert_eq!(session.next_frame().unwrap().unwrap().metadata.index, 1);
    assert_eq!(session.next_frame().unwrap().unwrap().metadata.index, 2);
}

#[test]
fn prefetch_depth_cannot_exceed_frame_slot_limit() {
    let decoder = GpuDecoder::new(ReadyEngine);
    let mut session = decoder.open(raw_still(), output_request(2)).unwrap();
    assert!(matches!(
        session.prefetch(NonZeroUsize::new(3).unwrap()),
        Err(Error::PrefetchDepthExceedsLimit {
            requested: 3,
            limit: 2,
        })
    ));
}

#[derive(Debug)]
struct ResolvedSlotEngine;

impl GpuSubmissionEngine for ResolvedSlotEngine {
    type Session = ReadySession;

    fn open(
        &self,
        _codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            AnimationMetadata::animation(Extent2d::new(8, 6), timebase(), 0, false, Some(2)),
            ReadySession {
                frames: VecDeque::from([frame(0, false), frame(1, true)]),
            },
        )
        .with_resolved_frame_slots(NonZeroUsize::new(1).unwrap()))
    }
}

#[test]
fn backend_resolved_frame_slots_drive_limiter_and_prefetch() {
    let decoder = GpuDecoder::new(ResolvedSlotEngine);
    let mut session = decoder.open(raw_still(), output_request(2)).unwrap();

    assert_eq!(session.request().max_frame_slots().get(), 2);
    assert_eq!(session.resolved_frame_slots().get(), 1);
    assert!(matches!(
        session.prefetch(NonZeroUsize::new(2).unwrap()),
        Err(Error::PrefetchDepthExceedsLimit {
            requested: 2,
            limit: 1,
        })
    ));

    let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(progress.queued, 1);
    assert_eq!(session.active_frame_slots(), 1);
    let first = session.next_frame().unwrap().unwrap();
    let blocked = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(
        blocked.backpressure,
        Some(PrefetchBackpressure::FrameSlots { limit: 1 })
    );
    drop(first);
    assert_eq!(session.active_frame_slots(), 0);
}

#[derive(Debug)]
struct TypedRejectEngine {
    expected_container: bool,
    unsupported: bool,
}

#[derive(Debug)]
struct RejectSession;

impl GpuSubmissionEngine for TypedRejectEngine {
    type Session = RejectSession;

    fn open(
        &self,
        codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        _inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        assert_eq!(codestream.is_container(), self.expected_container);
        assert_jxl_signature(&codestream);
        if self.unsupported {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::VarDct,
                "stock backend accepts only single-group fixed-predictor lossless Modular",
            )
            .into());
        }
        Err(FrontendIncomplete::new(
            FrontendStage::EntropyGroups,
            "GPU entropy packetization is not implemented for this fixture",
        )
        .into())
    }
}

impl GpuSubmissionSession for RejectSession {
    type Frame = MockGpuFrame;
    type Pending = ReadyPending;

    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        unreachable!()
    }
}

#[test]
fn real_raw_fixture_reaches_typed_unsupported_profile_without_fallback() {
    let decoder = GpuDecoder::new(TypedRejectEngine {
        expected_container: false,
        unsupported: true,
    });
    let result = decoder.open(raw_still(), output_request(1));
    assert!(matches!(result, Err(Error::UnsupportedProfile(_))));
}

#[test]
fn real_fragmented_container_is_joined_before_typed_frontend_reject() {
    let decoder = GpuDecoder::new(TypedRejectEngine {
        expected_container: true,
        unsupported: false,
    });
    let result = decoder.open(fragmented_animation(), output_request(1));
    assert!(matches!(
        result,
        Err(Error::FrontendIncomplete(FrontendIncomplete {
            stage: FrontendStage::EntropyGroups,
            ..
        }))
    ));
}

#[derive(Clone, Debug)]
struct CapturingEngine {
    opened: Arc<Mutex<Option<OpenedSource>>>,
}

#[derive(Clone, Debug)]
struct OpenedSource {
    logical_bytes: u64,
    spans: usize,
    retained_input_bytes: u64,
    is_container: bool,
    is_contiguous: bool,
    inventory: CodestreamInventory,
}

impl GpuSubmissionEngine for CapturingEngine {
    type Session = ReadySession;

    fn open(
        &self,
        codestream: GpuCodestream,
        _request: &GpuOutputRequest,
        inventory: Arc<CodestreamInventory>,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        assert_jxl_signature(&codestream);
        *self.opened.lock().unwrap() = Some(OpenedSource {
            logical_bytes: codestream.logical_bytes(),
            spans: codestream.span_count(),
            retained_input_bytes: codestream.retained_input_bytes(),
            is_container: codestream.is_container(),
            is_contiguous: codestream.contiguous_bytes().is_some(),
            inventory: (*inventory).clone(),
        });
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            AnimationMetadata::animation(Extent2d::new(8, 6), timebase(), 0, false, Some(2)),
            ReadySession {
                frames: VecDeque::from([frame(0, false), frame(1, true)]),
            },
        ))
    }
}

fn push_transport_ranges(
    stream: &mut jxl_wgpu_decode::GpuDecodeStream<CapturingEngine>,
    limits: ContainerStreamLimits,
    input: &[u8],
    ranges: impl IntoIterator<Item = std::ops::Range<usize>>,
) {
    let mut transport = ContainerStreamScanner::new(limits);
    for range in ranges {
        for event in transport.push_chunk(Arc::from(&input[range])).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
}

#[test]
fn incremental_frontend_matches_contiguous_inventory_at_every_split() {
    let input = raw_still();
    let parsed = jxl_gpu_bitstream::parse(input, ParseLimits::default()).unwrap();
    let expected = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let codestream_bytes = expected.codestream_bytes;

    for split in 0..=input.len() {
        let opened = Arc::new(Mutex::new(None));
        let decoder = GpuDecoder::new(CapturingEngine {
            opened: Arc::clone(&opened),
        });
        let mut stream = decoder.stream(output_request(2)).unwrap();
        push_transport_ranges(
            &mut stream,
            decoder.container_stream_limits(),
            input,
            [0..split, split..input.len()],
        );
        let stats = stream.stats();
        assert!(stream.is_ready());
        assert_eq!(stats.retained_codestream_bytes, codestream_bytes);
        assert_eq!(stats.input_budget.reserved_bytes, codestream_bytes);

        let _session = stream.finish().unwrap();
        let actual = opened.lock().unwrap().take().unwrap();
        assert_eq!(actual.logical_bytes, codestream_bytes);
        assert_eq!(actual.retained_input_bytes, codestream_bytes);
        assert!(actual.spans >= 2);
        assert!(!actual.is_container);
        assert!(!actual.is_contiguous);
        assert_eq!(actual.inventory, expected);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn byte_drip_fragmented_container_reaches_engine_as_shared_spans() {
    let input = fragmented_animation();
    let parsed = jxl_gpu_bitstream::parse(input, ParseLimits::default()).unwrap();
    let expected = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let opened = Arc::new(Mutex::new(None));
    let decoder = GpuDecoder::new(CapturingEngine {
        opened: Arc::clone(&opened),
    });
    let mut stream = decoder.stream(output_request(2)).unwrap();
    push_transport_ranges(
        &mut stream,
        decoder.container_stream_limits(),
        input,
        (0..input.len()).map(|offset| offset..offset + 1),
    );
    let span_count = stream.stats().retained_spans;
    assert!(span_count > 2);
    let _session = stream.finish().unwrap();

    let actual = opened.lock().unwrap().take().unwrap();
    assert_eq!(actual.logical_bytes, expected.codestream_bytes);
    assert_eq!(actual.retained_input_bytes, expected.codestream_bytes);
    assert_eq!(actual.spans, span_count);
    assert!(actual.is_container);
    assert!(!actual.is_contiguous);
    assert_eq!(actual.inventory, expected);
}

#[test]
fn incremental_input_backpressure_is_retryable_and_cancel_releases_budget() {
    let input = raw_still();
    let codestream_bytes = jxl_gpu_bitstream::parse(input, ParseLimits::default())
        .unwrap()
        .codestream()
        .len() as u64;
    let budget = IncrementalInputBudget::new(NonZeroU64::new(codestream_bytes).unwrap());
    let decoder = GpuDecoder::new(CapturingEngine {
        opened: Arc::new(Mutex::new(None)),
    })
    .with_incremental_input_budget(budget.clone());
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    let events = transport.push_chunk(Arc::from(input)).unwrap();
    let mut first = decoder.stream(output_request(2)).unwrap();
    for event in &events {
        first.push_transport_event(event).unwrap();
    }
    assert_eq!(budget.snapshot().reserved_bytes, codestream_bytes);

    let first_chunk = events
        .iter()
        .find(|event| matches!(event, ContainerStreamEvent::CodestreamChunk { .. }))
        .unwrap();
    let mut second = decoder.stream(output_request(2)).unwrap();
    assert!(matches!(
        second.push_transport_event(first_chunk),
        Err(Error::IncrementalInputBudget(_))
    ));
    assert_eq!(second.stats().codestream.codestream_bytes_received, 0);

    drop(first);
    assert_eq!(budget.snapshot().reserved_bytes, 0);
    second.push_transport_event(first_chunk).unwrap();
    assert!(matches!(
        second.finish(),
        Err(Error::IncrementalInputIncomplete)
    ));
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}

#[test]
fn malformed_input_fails_before_the_gpu_engine() {
    let decoder = GpuDecoder::new(ReadyEngine);
    let result = decoder.open(b"not-jxl", output_request(1));
    assert!(matches!(result, Err(Error::Bitstream(_))));
}
