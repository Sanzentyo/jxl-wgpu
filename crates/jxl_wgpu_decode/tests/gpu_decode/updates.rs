use super::*;
use jxl_wgpu_decode::{FrameProgression, SubmittedGpuUpdate};

struct UpdatePending {
    updates: VecDeque<Result<SubmittedGpuUpdate<MockGpuFrame>>>,
    yield_once: bool,
}

impl GpuPendingFrame for UpdatePending {
    type Frame = MockGpuFrame;

    #[cfg(not(target_arch = "wasm32"))]
    fn wait(mut self) -> Result<SubmittedGpuFrame<Self::Frame>> {
        loop {
            if let SubmittedGpuUpdate::Complete(frame) = self.wait_next_update()? {
                return Ok(frame);
            }
        }
    }

    fn poll_complete(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuFrame<Self::Frame>>> {
        loop {
            match self.as_mut().poll_next_update(context) {
                Poll::Ready(Ok(SubmittedGpuUpdate::Complete(frame))) => {
                    return Poll::Ready(Ok(frame));
                }
                Poll::Ready(Ok(SubmittedGpuUpdate::Intermediate { .. })) => {}
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Pending => return Poll::Pending,
            }
        }
    }

    fn poll_next_update(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Result<SubmittedGpuUpdate<Self::Frame>>> {
        if self.yield_once {
            self.yield_once = false;
            context.waker().wake_by_ref();
            return Poll::Pending;
        }
        self.yield_once = true;
        Poll::Ready(
            self.updates
                .pop_front()
                .expect("completion must consume the pending frame"),
        )
    }
}

struct UpdateSession(VecDeque<UpdatePending>);

impl GpuSubmissionSession for UpdateSession {
    type Frame = MockGpuFrame;
    type Pending = UpdatePending;
    fn submit_next(&mut self) -> Result<Option<Self::Pending>> {
        Ok(self.0.pop_front())
    }
}

struct UpdateEngine {
    still: bool,
    fault: Option<u8>,
}

impl GpuSubmissionEngine for UpdateEngine {
    type Session = UpdateSession;
    fn open(
        &self,
        _: GpuCodestream,
        _: &GpuOutputRequest,
        _: jxl_wgpu_decode::SelectedImageInventory,
    ) -> Result<PreparedGpuSession<Self::Session>> {
        let count = if self.still { 1 } else { 2 };
        let frames = (0..count)
            .map(|index| {
                let make_frame = |resource_id| {
                    let mut value = frame(index, index + 1 == count);
                    if self.still {
                        value.metadata.duration = FrameDuration::still();
                    }
                    value.output.resource_id = resource_id;
                    value
                };
                let mut second = make_frame(index as u64 * 10 + 1);
                let mut progression = FrameProgression {
                    physical_frame_index: index as u32 + 3,
                    completed_passes: 1,
                    total_passes: 3,
                    intended_downsampling: 4,
                };
                let mut last = make_frame(index as u64 * 10 + 2);
                match self.fault {
                    Some(0) => progression.completed_passes = 0,
                    Some(1) => progression.total_passes = 0,
                    Some(2) => progression.completed_passes = 3,
                    Some(3) => progression.intended_downsampling = 0,
                    Some(4) => progression.intended_downsampling = 3,
                    Some(5) => progression.intended_downsampling = 16,
                    Some(6) => progression.physical_frame_index = 0,
                    Some(7) => second.metadata.presentation_ticks += 1,
                    Some(8) => last.metadata.name.push('!'),
                    Some(9) => progression.total_passes = 2,
                    _ => {}
                }
                let second = if self.fault == Some(10) {
                    Err(Error::EngineContract("late entropy corruption"))
                } else {
                    Ok(SubmittedGpuUpdate::Intermediate {
                        frame: second,
                        progression,
                    })
                };
                UpdatePending {
                    updates: VecDeque::from([
                        Ok(SubmittedGpuUpdate::Intermediate {
                            frame: make_frame(index as u64 * 10),
                            progression: FrameProgression {
                                physical_frame_index: index as u32 + 3,
                                completed_passes: 0,
                                total_passes: 3,
                                intended_downsampling: 8,
                            },
                        }),
                        second,
                        Ok(SubmittedGpuUpdate::Complete(last)),
                    ]),
                    yield_once: true,
                }
            })
            .collect();
        Ok(PreparedGpuSession::new(
            fixed_profile(8, ModularPredictor::Zero),
            if self.still {
                AnimationMetadata::still(Extent2d::new(8, 6))
            } else {
                AnimationMetadata::animation(Extent2d::new(8, 6), timebase(), 0, false, Some(count))
            },
            UpdateSession(frames),
        ))
    }
}

#[test]
fn intermediate_leases_share_one_slot_and_do_not_advance_the_animation_clock() {
    let decoder = GpuDecoder::new(UpdateEngine {
        still: false,
        fault: None,
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    let dc = session.next_update().unwrap().unwrap();
    assert!(!dc.is_complete());
    assert_eq!(dc.progression().unwrap().completed_passes, 0);
    assert_eq!(session.frames_submitted(), 1);
    assert_eq!(session.queued_frames(), 1);
    assert_eq!(session.active_frame_slots(), 1);
    let ac = session.next_update().unwrap().unwrap();
    let final_frame = session.next_update().unwrap().unwrap();
    assert!(final_frame.is_complete());
    assert_eq!(dc.metadata, ac.metadata);
    assert_eq!(dc.metadata, final_frame.metadata);
    assert_eq!(dc.metadata.presentation_ticks, 0);
    assert_eq!(dc.output().resource_id, 0);
    assert_eq!(ac.output().resource_id, 1);
    assert_eq!(final_frame.output().resource_id, 2);
    assert_eq!(session.queued_frames(), 0);
    assert_eq!(session.active_frame_slots(), 1);
    drop(final_frame);
    drop(ac);
    assert!(matches!(
        session.next_update(),
        Err(Error::Backpressure { limit: 1 })
    ));
    drop(dc);
    let next_dc = session.next_update().unwrap().unwrap();
    assert_eq!(next_dc.metadata.index, 1);
    assert_eq!(next_dc.metadata.presentation_ticks, 20);
    let last = session.next_frame().unwrap().unwrap();
    assert!(last.is_complete());
    assert_eq!(last.output().resource_id, 12);
    assert!(session.is_finished());
    assert!(session.next_update().unwrap().is_none());
    assert_eq!(session.frames_submitted(), 2);
}

#[test]
fn async_updates_survive_future_cancellation_and_keep_prefetched_presentations_ordered() {
    let decoder = GpuDecoder::new(UpdateEngine {
        still: false,
        fault: None,
    });
    let mut session = decoder.open(raw_still(), output_request(2)).unwrap();
    session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
    let mut cancelled = session.next_update_async();
    assert!(
        Pin::new(&mut cancelled)
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(cancelled);
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
        held.push(update);
    }
    assert_eq!(held.len(), 6);
    for (index, updates) in held.chunks_exact(3).enumerate() {
        assert!(updates.iter().all(|u| u.metadata == updates[0].metadata));
        assert_eq!(updates[0].metadata.index, index);
        assert_eq!(updates[0].metadata.presentation_ticks, index as u64 * 20);
        assert!(!updates[0].is_complete());
        assert!(!updates[1].is_complete());
        assert!(updates[2].is_complete());
    }
    assert_eq!(session.active_frame_slots(), 2);
    drop(held);
    assert_eq!(session.active_frame_slots(), 0);
}

#[test]
fn still_refinements_do_not_mark_the_session_finished_and_final_only_engines_still_work() {
    let decoder = GpuDecoder::new(UpdateEngine {
        still: true,
        fault: None,
    });
    let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
    let dc = session.next_update().unwrap().unwrap();
    assert!(dc.metadata.is_last);
    assert!(!dc.is_complete());
    assert!(!session.is_finished());
    let last = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    assert!(last.is_complete());
    assert_eq!(last.metadata, dc.metadata);
    assert!(session.is_finished());
    let mut final_only = GpuDecoder::new(ReadyEngine)
        .open(raw_still(), output_request(2))
        .unwrap();
    let first = final_only.next_update().unwrap().unwrap();
    let last = pollster::block_on(final_only.next_update_async())
        .unwrap()
        .unwrap();
    assert!(first.is_complete() && last.is_complete());
    assert_eq!(last.metadata.presentation_ticks, 20);
    assert!(final_only.next_update().unwrap().is_none());
}

#[test]
fn malformed_or_corrupt_refinements_poison_the_session_without_invalidating_prior_leases() {
    for fault in 0..=10 {
        let decoder = GpuDecoder::new(UpdateEngine {
            still: true,
            fault: Some(fault),
        });
        let mut session = decoder.open(raw_still(), output_request(1)).unwrap();
        let prior = session.next_update().unwrap().unwrap();
        if fault == 8 {
            drop(session.next_update().unwrap().unwrap());
        }
        assert!(session.next_update().is_err(), "fault {fault}");
        assert!(matches!(session.next_update(), Err(Error::SessionPoisoned)));
        assert!(matches!(session.next_frame(), Err(Error::SessionPoisoned)));
        assert_eq!(prior.output().resource_id, 0);
        drop(session);
        assert_eq!(prior.output().resource_id, 0);
    }
}
