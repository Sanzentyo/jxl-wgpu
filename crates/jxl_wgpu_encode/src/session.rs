use std::collections::BTreeMap;
use std::future::Future;
use std::num::NonZeroU32;
use std::pin::Pin;
use std::task::{Context, Poll};

use jxl_gpu_bitstream::PrefixCodeEntry;

use crate::{
    BitFragment, Determinism, EncodeError, EncodeProfile, EncodedFrame, FramePacketSet,
    GpuEncodeBackend, GpuEncodeJob, GpuEncoder, GpuFrameSource, PacketError, ProgressivePlan,
    assemble_frame,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameIndex(u32);

impl FrameIndex {
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimationHeader {
    Still,
    Animation {
        ticks_per_second_numerator: NonZeroU32,
        ticks_per_second_denominator: NonZeroU32,
        num_loops: u32,
        have_timecodes: bool,
    },
}

impl AnimationHeader {
    #[must_use]
    pub const fn is_animation(self) -> bool {
        matches!(self, Self::Animation { .. })
    }

    #[must_use]
    pub const fn has_timecodes(self) -> bool {
        matches!(
            self,
            Self::Animation {
                have_timecodes: true,
                ..
            }
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTiming {
    pub duration_ticks: u32,
    pub timecode: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    #[default]
    Replace,
    Add,
    Blend,
    MultiplyAdd,
    Multiply,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReferenceSlot(u8);

impl ReferenceSlot {
    pub const MAX: u8 = 3;

    pub fn new(slot: u8) -> Result<Self, EncodeError> {
        if slot > Self::MAX {
            return Err(EncodeError::InvalidConfiguration(
                "JPEG XL reference slot must be in 0..=3",
            ));
        }
        Ok(Self(slot))
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameOptions {
    pub timing: FrameTiming,
    pub blend_mode: BlendMode,
    pub source_reference: Option<ReferenceSlot>,
    pub save_as_reference: Option<ReferenceSlot>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionDescriptor {
    pub profile: EncodeProfile,
    pub progressive: ProgressivePlan,
    pub minimum_determinism: Determinism,
    pub animation: AnimationHeader,
}

impl SessionDescriptor {
    #[must_use]
    pub fn still(profile: EncodeProfile) -> Self {
        Self {
            profile,
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::Assembly,
            animation: AnimationHeader::Still,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameEncodeRequest {
    pub frame_index: FrameIndex,
    pub is_last: bool,
    pub profile: EncodeProfile,
    pub progressive: ProgressivePlan,
    pub minimum_determinism: Determinism,
    pub animation: AnimationHeader,
    pub options: FrameOptions,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuAccelerationArtifact {
    Gray8Prefix {
        width: u32,
        height: u32,
        token_bit_offset_in_group: u64,
        token_bit_len: u64,
        raw_prefix: [PrefixCodeEntry; 19],
        lz77_prefix: [PrefixCodeEntry; 33],
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuFrameArtifacts {
    pub frame_index: FrameIndex,
    pub is_last: bool,
    pub packets: FramePacketSet,
    pub acceleration: Option<GpuAccelerationArtifact>,
}

pub struct FrameSubmission<J> {
    job: Option<J>,
    expected_index: FrameIndex,
    expected_last: bool,
}

impl<J: GpuEncodeJob> FrameSubmission<J> {
    pub(crate) fn new(job: J, expected_index: FrameIndex, expected_last: bool) -> Self {
        Self {
            job: Some(job),
            expected_index,
            expected_last,
        }
    }

    pub fn wait(mut self) -> Result<GpuFrameArtifacts, EncodeError> {
        let job = self
            .job
            .take()
            .expect("a frame submission can only complete once");
        validate_artifacts(job.wait()?, self.expected_index, self.expected_last)
    }
}

impl<J: GpuEncodeJob> Future for FrameSubmission<J> {
    type Output = Result<GpuFrameArtifacts, EncodeError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let submission = self.get_mut();
        let job = submission
            .job
            .as_mut()
            .expect("a frame submission must not be polled after completion");
        match job.poll_complete(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                submission.job.take();
                Poll::Ready(result.and_then(|artifacts| {
                    validate_artifacts(
                        artifacts,
                        submission.expected_index,
                        submission.expected_last,
                    )
                }))
            }
        }
    }
}

fn validate_artifacts(
    artifacts: GpuFrameArtifacts,
    expected_index: FrameIndex,
    expected_last: bool,
) -> Result<GpuFrameArtifacts, EncodeError> {
    if artifacts.frame_index != expected_index {
        return Err(PacketError::FrameIndexMismatch.into());
    }
    if artifacts.is_last != expected_last {
        return Err(PacketError::FinalFlagMismatch.into());
    }
    Ok(artifacts)
}

/// Reusable session that can keep multiple GPU frame jobs in flight. Returned
/// submissions are independent futures, so callers may await them concurrently
/// on any executor or block with [`FrameSubmission::wait`].
pub struct EncodeSession<B> {
    encoder: GpuEncoder<B>,
    descriptor: SessionDescriptor,
    next_frame: u32,
    closed: bool,
}

impl<B: GpuEncodeBackend> EncodeSession<B> {
    pub(crate) fn new(encoder: GpuEncoder<B>, descriptor: SessionDescriptor) -> Self {
        Self {
            encoder,
            descriptor,
            next_frame: 0,
            closed: false,
        }
    }

    #[must_use]
    pub const fn next_frame_index(&self) -> FrameIndex {
        FrameIndex(self.next_frame)
    }

    pub fn submit_frame(
        &mut self,
        source: GpuFrameSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<B::Job>, EncodeError> {
        if !self.descriptor.animation.is_animation() {
            return Err(EncodeError::InvalidConfiguration(
                "a still-image session must use submit_last_frame",
            ));
        }
        self.submit(source, options, false)
    }

    pub fn submit_last_frame(
        &mut self,
        source: GpuFrameSource,
        options: FrameOptions,
    ) -> Result<FrameSubmission<B::Job>, EncodeError> {
        let submission = self.submit(source, options, true)?;
        self.closed = true;
        Ok(submission)
    }

    pub fn ensure_closed(&self) -> Result<(), EncodeError> {
        if self.closed {
            Ok(())
        } else {
            Err(EncodeError::MissingFinalFrame)
        }
    }

    fn submit(
        &mut self,
        source: GpuFrameSource,
        options: FrameOptions,
        is_last: bool,
    ) -> Result<FrameSubmission<B::Job>, EncodeError> {
        if self.closed {
            return Err(EncodeError::SessionClosed);
        }
        validate_frame_options(self.descriptor.animation, options)?;
        let frame_index = FrameIndex(self.next_frame);
        let request = FrameEncodeRequest {
            frame_index,
            is_last,
            profile: self.descriptor.profile,
            progressive: self.descriptor.progressive.clone(),
            minimum_determinism: self.descriptor.minimum_determinism,
            animation: self.descriptor.animation,
            options,
        };
        let submission = self.encoder.submit_frame(source, request)?;
        self.next_frame =
            self.next_frame
                .checked_add(1)
                .ok_or(EncodeError::InvalidConfiguration(
                    "too many animation frames",
                ))?;
        Ok(submission)
    }
}

fn validate_frame_options(
    animation: AnimationHeader,
    options: FrameOptions,
) -> Result<(), EncodeError> {
    if !animation.is_animation()
        && (options.timing.duration_ticks != 0 || options.timing.timecode.is_some())
    {
        return Err(EncodeError::InvalidConfiguration(
            "still images cannot have animation timing",
        ));
    }
    if options.timing.timecode.is_some() != animation.has_timecodes() {
        return Err(EncodeError::InvalidConfiguration(
            "frame timecode presence must match the animation header",
        ));
    }
    if options.blend_mode != BlendMode::Replace && options.source_reference.is_none() {
        return Err(EncodeError::InvalidConfiguration(
            "non-replace blend modes require a source reference slot",
        ));
    }
    Ok(())
}

/// Orders independently completed frames and creates a raw codestream or
/// deterministic `jxlc` container. The header is produced by the CPU metadata
/// serializer and must already include the `0xff 0x0a` codestream signature.
pub struct CodestreamAssembler {
    codestream_header: BitFragment,
    frames: BTreeMap<FrameIndex, (bool, EncodedFrame)>,
}

impl CodestreamAssembler {
    pub fn new(codestream_header: BitFragment) -> Result<Self, PacketError> {
        if !codestream_header.is_byte_aligned()
            || !codestream_header
                .bytes()
                .starts_with(&jxl_gpu_bitstream::CODESTREAM_SIGNATURE)
        {
            return Err(PacketError::InvalidCodestreamHeader);
        }
        Ok(Self {
            codestream_header,
            frames: BTreeMap::new(),
        })
    }

    pub fn insert(&mut self, artifacts: GpuFrameArtifacts) -> Result<(), PacketError> {
        let frame = assemble_frame(artifacts.packets)?;
        if self
            .frames
            .insert(artifacts.frame_index, (artifacts.is_last, frame))
            .is_some()
        {
            return Err(PacketError::DuplicateFrame(artifacts.frame_index.get()));
        }
        Ok(())
    }

    pub fn finish_raw(self) -> Result<Vec<u8>, PacketError> {
        let mut output = self.codestream_header.bytes().to_vec();
        let mut saw_last = false;
        let frame_count = self.frames.len();
        for expected in 0..frame_count {
            let index = FrameIndex(u32::try_from(expected).map_err(|_| PacketError::SizeOverflow)?);
            let (is_last, frame) = self
                .frames
                .get(&index)
                .ok_or(PacketError::MissingFrame(index.get()))?;
            if *is_last != (expected + 1 == frame_count) || (saw_last && *is_last) {
                return Err(PacketError::InvalidFinalFrame);
            }
            saw_last |= *is_last;
            output
                .try_reserve(frame.bytes().len())
                .map_err(|_| PacketError::SizeOverflow)?;
            output.extend_from_slice(frame.bytes());
        }
        if frame_count == 0 || !saw_last {
            return Err(PacketError::InvalidFinalFrame);
        }
        Ok(output)
    }

    pub fn finish_container(self) -> Result<Vec<u8>, EncodeError> {
        let codestream = self.finish_raw()?;
        Ok(jxl_gpu_bitstream::write_container(&codestream)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::{FrameGroupLayout, GroupPacket, GroupPacketKind};

    use super::*;

    fn artifacts(index: u32, last: bool, byte: u8) -> GpuFrameArtifacts {
        let layout = FrameGroupLayout::new(1, 1, 1).unwrap();
        GpuFrameArtifacts {
            frame_index: FrameIndex(index),
            is_last: last,
            packets: FramePacketSet::new(
                BitFragment::new(Vec::new(), 0).unwrap(),
                layout,
                [GroupPacket::new(GroupPacketKind::Single, vec![byte])],
            )
            .unwrap(),
            acceleration: None,
        }
    }

    #[test]
    fn assembler_orders_concurrently_completed_frames() {
        let mut assembler =
            CodestreamAssembler::new(BitFragment::byte_aligned(vec![0xff, 0x0a])).unwrap();
        assembler.insert(artifacts(1, true, 2)).unwrap();
        assembler.insert(artifacts(0, false, 1)).unwrap();
        let raw = assembler.finish_raw().unwrap();
        assert!(raw.starts_with(&[0xff, 0x0a]));
        assert_eq!(*raw.last().unwrap(), 2);
    }

    #[test]
    fn assembler_rejects_early_final_frame() {
        let mut assembler =
            CodestreamAssembler::new(BitFragment::byte_aligned(vec![0xff, 0x0a])).unwrap();
        assembler.insert(artifacts(0, true, 1)).unwrap();
        assembler.insert(artifacts(1, false, 2)).unwrap();
        assert_eq!(
            assembler.finish_raw().unwrap_err(),
            PacketError::InvalidFinalFrame
        );
    }
}
