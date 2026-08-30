use thiserror::Error;

use crate::{EncodeProfile, KernelStage};

#[derive(Clone, Debug, Error, PartialEq)]
pub enum UnsupportedFeature {
    #[error("the backend does not implement profile {0:?}")]
    Profile(EncodeProfile),
    #[error("the backend does not accept this pitch-linear input format")]
    InputFormat,
    #[error("the backend supports at most {supported} progressive passes, requested {requested}")]
    ProgressivePasses { supported: u8, requested: u8 },
    #[error("the backend does not implement animation encoding")]
    Animation,
    #[error("the backend does not provide deterministic assembly artifacts")]
    DeterministicAssembly,
    #[error("the backend is missing the required GPU kernel stage {0:?}")]
    Kernel(KernelStage),
    #[error("the device limit {name} is {available}, but at least {required} is required")]
    DeviceLimit {
        name: &'static str,
        required: u64,
        available: u64,
    },
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PacketError {
    #[error("bit length {bit_len} requires {expected} bytes, received {actual}")]
    BitLength {
        bit_len: usize,
        expected: usize,
        actual: usize,
    },
    #[error("unused high bits in the final byte must be zero")]
    NonZeroPadding,
    #[error("group layout must contain at least one DC group, AC group, and pass")]
    EmptyLayout,
    #[error("group layout exceeds the JPEG XL TOC entry limit")]
    TooManyGroups,
    #[error("packet kind {kind:?} is not valid for layout {layout:?}")]
    InvalidKind {
        kind: crate::GroupPacketKind,
        layout: crate::FrameGroupLayout,
    },
    #[error("packet {0:?} occurs more than once")]
    Duplicate(crate::GroupPacketKind),
    #[error("packet {0:?} is missing")]
    Missing(crate::GroupPacketKind),
    #[error("group packet is larger than the JPEG XL TOC representation")]
    PacketTooLarge,
    #[error("bitstream size arithmetic overflow")]
    SizeOverflow,
    #[error("the frame index embedded in GPU artifacts does not match the submitted frame")]
    FrameIndexMismatch,
    #[error("the final-frame flag embedded in GPU artifacts does not match the submission")]
    FinalFlagMismatch,
    #[error("frame {0} was already inserted")]
    DuplicateFrame(u32),
    #[error("frame sequence is missing index {0}")]
    MissingFrame(u32),
    #[error("exactly one final frame is required")]
    InvalidFinalFrame,
    #[error("a raw JPEG XL codestream must begin with 0xff 0x0a")]
    InvalidCodestreamHeader,
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error(transparent)]
    Unsupported(#[from] UnsupportedFeature),
    #[error(transparent)]
    Packet(#[from] PacketError),
    #[error(transparent)]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error(transparent)]
    AccelerationIndex(#[from] jxl_gpu_bitstream::AccelerationIndexError),
    #[error("invalid encoder configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid GPU frame source: {0}")]
    InvalidSource(&'static str),
    #[error("GPU encoder job failed: {0}")]
    Backend(String),
    #[error("the encode session is already closed")]
    SessionClosed,
    #[error("the encode session must be closed with a final frame")]
    MissingFinalFrame,
}
