use thiserror::Error;

use crate::{EncodeProfile, KernelStage};

/// Failures reported by the concrete GPU encoder implementation.
///
/// Typed errors from `wgpu`, the operating system, and the shared submission
/// poller are retained as sources so callers can inspect the complete error
/// chain. [`Self::PollWorker`] carries a message because the shared poller
/// deliberately erases backend-specific poll errors at its callback boundary.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("GPU encoder backend invariant failed: {0}")]
    Invariant(&'static str),
    #[error("invalid GPU artifact: {0}")]
    InvalidArtifact(&'static str),
    #[error("GPU artifact mapping failed")]
    ArtifactMapping(#[source] wgpu::BufferAsyncError),
    #[error("the mapped GPU artifact range is invalid")]
    ArtifactRange(#[source] wgpu::MapRangeError),
    #[error("GPU submission polling failed: {0}")]
    PollWorker(String),
    #[error("GPU submission poll registration failed")]
    PollRegistration(#[source] jxl_wgpu::SubmissionPollerError),
    #[error("could not start the bounded GPU poll worker")]
    PollWorkerStart(#[source] std::io::Error),
    #[error("could not start a streamed Modular encode worker")]
    StreamingWorkerStart(#[source] std::io::Error),
}

impl From<&'static str> for BackendError {
    fn from(message: &'static str) -> Self {
        Self::Invariant(message)
    }
}

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
    #[error(
        "tiled VarDCT dimensions {width}x{height} exceed the checked {max_dimension}px-per-axis profile"
    )]
    TiledVarDctDimensions {
        width: u32,
        height: u32,
        max_dimension: u32,
    },
    #[error(
        "tiled VarDCT requires at least two AC groups so its section topology is unambiguous; {width}x{height} fits one {group_dimension}px group"
    )]
    TiledVarDctSingleAcGroup {
        width: u32,
        height: u32,
        group_dimension: u32,
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
    #[error("GPU encoder kernel policy failed: {0}")]
    KernelPolicy(#[from] jxl_wgpu::Error),
    #[error("invalid encoder configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid GPU frame source: {0}")]
    InvalidSource(&'static str),
    #[error("VarDCT LF dequantization multiplier for {channel} is too small: {value}")]
    VarDctLfDequantization { channel: &'static str, value: f32 },
    #[error("VarDCT channel correlation colour factor {value} is outside 2..=65793")]
    VarDctColourFactor { value: u32 },
    #[error("VarDCT base channel correlation for {channel} is outside [-4, 4]: {value}")]
    VarDctBaseCorrelation { channel: &'static str, value: f32 },
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("GPU encoder memory backpressure: {0}")]
    MemoryBackpressure(#[from] jxl_wgpu::MemoryBudgetError),
    #[error("GPU encoder submission-poll backpressure: {0}")]
    PollBackpressure(#[from] jxl_wgpu::SubmissionPollerError),
    #[error("the encode session is already closed")]
    SessionClosed,
    #[error("the encode session must be closed with a final frame")]
    MissingFinalFrame,
}
