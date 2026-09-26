use super::*;
use crate::{BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind};
use std::num::NonZeroU64;

struct ReadyJob(Option<GpuFrameArtifacts>, Arc<()>);
impl GpuEncodeJob for ReadyJob {
    fn poll_complete(
        &mut self,
        _: &mut Context<'_>,
    ) -> Poll<Result<GpuFrameArtifacts, EncodeError>> {
        Poll::Ready(Ok(self.0.take().unwrap()))
    }
    fn wait(mut self) -> Result<GpuFrameArtifacts, EncodeError> {
        // Keep the lifetime probe until the job is consumed.
        assert_eq!(Arc::strong_count(&self.1), 1);
        Ok(self.0.take().unwrap())
    }
}

fn artifacts() -> GpuFrameArtifacts {
    let mut payload = Vec::with_capacity(512);
    payload.extend_from_slice(&[17; 31]);
    GpuFrameArtifacts {
        frame_index: FrameIndex::new(0),
        is_last: true,
        acceleration: None,
        packets: FramePacketSet::new(
            BitFragment::new(vec![0], 1).unwrap(),
            FrameGroupLayout::new(1, 1, 1).unwrap(),
            [GroupPacket::new(GroupPacketKind::Single, payload)],
        )
        .unwrap(),
    }
}

fn submission(
    artifacts: GpuFrameArtifacts,
    budget: &MemoryBudget,
) -> (PreviewSubmission<ReadyJob>, std::sync::Weak<()>) {
    let owner = Arc::new(());
    let weak = Arc::downgrade(&owner);
    let frame = FrameSubmission::new(ReadyJob(Some(artifacts), owner), FrameIndex::new(0), true);
    let mut state = PreviewState::new(PreviewSize::new(1, 1).unwrap());
    (state.submitted(frame, budget), weak)
}

#[test]
fn corrupt_completion_metadata_cannot_publish_or_keep_the_job_alive() {
    let budget = MemoryBudget::new(NonZeroU64::new(4096).unwrap());
    for case in 0..3 {
        let mut artifacts = artifacts();
        match case {
            0 => artifacts.frame_index = FrameIndex::new(1),
            1 => artifacts.is_last = false,
            _ => artifacts.packets.layout = FrameGroupLayout::new(1, 2, 1).unwrap(),
        }
        let (mut future, owner) = submission(artifacts, &budget);
        let result = pollster::block_on(&mut future);
        assert!(matches!(
            (case, result),
            (0, Err(EncodeError::Packet(PacketError::FrameIndexMismatch)))
                | (1, Err(EncodeError::Packet(PacketError::FinalFlagMismatch)))
                | (2, Err(EncodeError::Packet(_)))
        ));
        assert!(owner.upgrade().is_none());
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn completion_storage_admission_is_exact_and_retains_only_live_capacity() {
    let budget = MemoryBudget::new(NonZeroU64::new(4096).unwrap());
    let plan = crate::packet::PreparedFrame::new(artifacts().packets).unwrap();
    let peak = plan.peak_bytes().unwrap();
    drop(plan);
    let blocker = budget.try_reserve(4096 - peak + 1).unwrap();
    let (mut rejected, owner) = submission(artifacts(), &budget);
    assert!(matches!(
        pollster::block_on(&mut rejected),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    assert!(owner.upgrade().is_none());
    assert_eq!(budget.snapshot().reserved_bytes, blocker.bytes());
    drop(blocker);
    let blocker = budget.try_reserve(4096 - peak).unwrap();
    for blocking in [true, false] {
        let (mut accepted, owner) = submission(artifacts(), &budget);
        let preview = if blocking {
            accepted.wait().unwrap()
        } else {
            pollster::block_on(&mut accepted).unwrap()
        };
        assert!(owner.upgrade().is_none());
        assert_eq!(
            preview.reserved_bytes(),
            preview.frame.storage_bytes() as u64
        );
        assert!(preview.reserved_bytes() < peak);
        assert_eq!(
            budget.snapshot().reserved_bytes,
            blocker.bytes() + preview.reserved_bytes()
        );
        drop(preview);
        assert_eq!(budget.snapshot().reserved_bytes, blocker.bytes());
    }
    drop(blocker);
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}
