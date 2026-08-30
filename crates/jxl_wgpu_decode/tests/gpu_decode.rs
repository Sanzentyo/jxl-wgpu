use std::collections::VecDeque;
use std::future::Future;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use jxl_gpu_formats::{ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, PixelFormat};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu_decode::{
    AnimationMetadata, DecodeProfile, Error, FixedModularPredictor, FrameDuration, FrameMetadata,
    FrameTimebase, FrontendIncomplete, FrontendStage, GpuCodestream, GpuDecoder, GpuOutputRequest,
    GpuSubmissionEngine, GpuSubmissionSession, PreparedGpuSession, Result, SubmittedGpuFrame,
    UnsupportedCodestreamFeature, UnsupportedProfile,
};

const RAW_STILL: &[u8] = include_bytes!("../../../fixtures/basic.jxl");
const FRAGMENTED_ANIMATION: &[u8] = include_bytes!("../../../fixtures/fragmented_animation.jxl");

#[derive(Clone, Debug, PartialEq, Eq)]
struct MockGpuFrame {
    resource_id: u64,
}

fn output_request(limit: usize) -> GpuOutputRequest {
    let color = ColorSpecification::Defined(ColorSpec::bt709(
        ColorRange::Limited,
        ChromaLocation2d::CENTER,
    ));
    GpuOutputRequest::new(PixelFormat::nv12(color))
        .with_max_in_flight(NonZeroUsize::new(limit).unwrap())
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
    ) -> Result<PreparedGpuSession<Self::Session>> {
        assert!(codestream.bytes().starts_with(&[0xff, 0x0a]));
        Ok(PreparedGpuSession::new(
            DecodeProfile::prototype_8bit(FixedModularPredictor::new(0).unwrap()),
            AnimationMetadata::animation(Extent2d::new(8, 6), timebase(), 0, false, Some(2)),
            ReadySession {
                frames: VecDeque::from([frame(0, false), frame(1, true)]),
            },
        ))
    }
}

impl GpuSubmissionSession for ReadySession {
    type Frame = MockGpuFrame;

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>> {
        Ok(self.frames.pop_front())
    }

    fn poll_next_frame(
        &mut self,
        _context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>> {
        Poll::Ready(self.next_frame())
    }
}

#[test]
fn sync_gpu_frames_are_bounded_and_keep_exact_timing() {
    let decoder = GpuDecoder::new(ReadyEngine);
    let mut session = decoder.open(RAW_STILL, output_request(1)).unwrap();
    assert_eq!(
        session.profile(),
        DecodeProfile::prototype_8bit(FixedModularPredictor::new(0).unwrap())
    );
    assert_eq!(session.metadata().extent, Extent2d::new(8, 6));

    let first = session.next_frame().unwrap().unwrap();
    assert_eq!(first.metadata.index, 0);
    assert_eq!(first.metadata.duration.ticks, 20);
    assert_eq!(first.metadata.duration.timebase, Some(timebase()));
    assert_eq!(first.output().resource_id, 0);
    assert!(matches!(
        session.next_frame(),
        Err(Error::Backpressure { limit: 1 })
    ));
    drop(first);

    let last = session.next_frame().unwrap().unwrap();
    assert!(last.metadata.is_last);
    assert_eq!(last.output().resource_id, 1);
    // EOF does not require another output slot, even while the final lease remains live.
    assert!(session.next_frame().unwrap().is_none());
    assert_eq!(session.frames_submitted(), 2);
}

#[derive(Debug, Default)]
struct PendingControl {
    ready: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl PendingControl {
    fn complete(&self) {
        self.ready.store(true, Ordering::Release);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
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
    ) -> Result<PreparedGpuSession<Self::Session>> {
        Ok(PreparedGpuSession::new(
            DecodeProfile::prototype_16bit(FixedModularPredictor::new(1).unwrap()),
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

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>> {
        Err(Error::backend("mock GPU callback is not ready"))
    }

    fn poll_next_frame(
        &mut self,
        context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>> {
        if !self.control.ready.load(Ordering::Acquire) {
            *self.control.waker.lock().unwrap() = Some(context.waker().clone());
            return Poll::Pending;
        }
        if self.emitted {
            return Poll::Ready(Ok(None));
        }
        self.emitted = true;
        Poll::Ready(Ok(Some(SubmittedGpuFrame::new(
            FrameMetadata {
                index: 0,
                duration: FrameDuration::still(),
                is_last: true,
                is_keyframe: true,
                name: String::new(),
            },
            MockGpuFrame { resource_id: 7 },
        ))))
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
    let mut session = decoder.open(RAW_STILL, output_request(1)).unwrap();
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
    ) -> Result<PreparedGpuSession<Self::Session>> {
        assert_eq!(codestream.is_container(), self.expected_container);
        assert!(codestream.bytes().starts_with(&[0xff, 0x0a]));
        if self.unsupported {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::VarDct,
                "prototype accepts only single-group fixed-predictor lossless Modular",
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

    fn next_frame(&mut self) -> Result<Option<SubmittedGpuFrame<Self::Frame>>> {
        unreachable!()
    }

    fn poll_next_frame(
        &mut self,
        _context: &mut Context<'_>,
    ) -> Poll<Result<Option<SubmittedGpuFrame<Self::Frame>>>> {
        unreachable!()
    }
}

#[test]
fn real_raw_fixture_reaches_typed_unsupported_profile_without_fallback() {
    let decoder = GpuDecoder::new(TypedRejectEngine {
        expected_container: false,
        unsupported: true,
    });
    let result = decoder.open(RAW_STILL, output_request(1));
    assert!(matches!(result, Err(Error::UnsupportedProfile(_))));
}

#[test]
fn real_fragmented_container_is_joined_before_typed_frontend_reject() {
    let decoder = GpuDecoder::new(TypedRejectEngine {
        expected_container: true,
        unsupported: false,
    });
    let result = decoder.open(FRAGMENTED_ANIMATION, output_request(1));
    assert!(matches!(
        result,
        Err(Error::FrontendIncomplete(FrontendIncomplete {
            stage: FrontendStage::EntropyGroups,
            ..
        }))
    ));
}

#[test]
fn malformed_input_fails_before_the_gpu_engine() {
    let decoder = GpuDecoder::new(ReadyEngine);
    let result = decoder.open(b"not-jxl", output_request(1));
    assert!(matches!(result, Err(Error::Bitstream(_))));
}
