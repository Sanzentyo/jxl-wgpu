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
    #[error("Modular Squeeze residual exceeds the lossless signed 32-bit representation")]
    ModularSqueezeOverflow,
    #[error("Modular palette exceeds the configured distinct-color capacity")]
    ModularPaletteOverflow,
    #[error(
        "VarDCT quantization exceeds signed 32-bit coefficients (LF: {low_frequency}, HF: {high_frequency})"
    )]
    VarDctQuantizationOverflow {
        low_frequency: bool,
        high_frequency: bool,
    },
    #[error("VarDCT source contains a non-finite floating sample")]
    VarDctNonFiniteSource,
    #[error("VarDCT ICC source conversion produced a non-finite working component")]
    VarDctColorConversionNonFinite,
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
    #[error("packet order requires {expected} entries, received {actual}")]
    OrderLength { expected: usize, actual: usize },
    #[error("packet {0:?} occurs more than once in the requested order")]
    DuplicateOrder(crate::GroupPacketKind),
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
    #[error("the declared preview has no validated completed output")]
    MissingPreview,
    #[error("preview output belongs to another sequence or was already inserted")]
    UnexpectedPreview,
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("invalid Modular transform count {count} (program: 1..=273; complete header: 0..=273)")]
    InvalidModularTransformCount { count: usize },
    #[error("Modular RCT begin {begin} must fit 0..=9287")]
    InvalidModularRctBegin { begin: u32 },
    #[error(
        "RCT operation {operation} range beginning at {begin} needs three of {channels} image channels"
    )]
    InvalidModularRctChannels {
        operation: u32,
        begin: u32,
        channels: u32,
    },
    #[error(
        "RCT operation {operation} beginning at {begin} requires equal image-channel extents and shifts"
    )]
    UnequalModularRctChannels { operation: u32, begin: u32 },
    #[error("explicit Squeeze sequence requires 1..=296 steps, got {count}")]
    InvalidModularSqueezeStepCount { count: usize },
    #[error("explicit Squeeze begin {begin} must fit 0..=9287 and count {count} must fit 1..=19")]
    InvalidModularSqueezeStep { begin: u32, count: u32 },
    #[error("Squeeze step {step} targets empty image channel {channel}")]
    EmptyModularSqueezeChannel { step: u32, channel: u32 },
    #[error(
        "Squeeze step {step} targets image channel {channel} with shifts {horizontal}/{vertical} exceeding 30"
    )]
    ModularSqueezeShiftLimit {
        step: u32,
        channel: u32,
        horizontal: u8,
        vertical: u8,
    },
    #[error(
        "Modular Squeeze range beginning at {begin} with {count} channels must be nonempty and fit {channels} post-Palette image channels"
    )]
    InvalidModularSqueezeChannels {
        begin: u32,
        count: u32,
        channels: u32,
    },

    #[error("Modular palette color limit must be in 1..=70911, got {max_colors}")]
    InvalidModularPaletteLimit { max_colors: u32 },

    #[error("Modular delta palette limit {max_deltas} is outside 1..=66816")]
    InvalidModularPaletteDeltaLimit { max_deltas: u32 },

    #[error(
        "Modular palette range beginning at {begin} with {count} components must be nonempty and fit {channels} source components"
    )]
    InvalidModularPaletteComponents {
        begin: u32,
        count: u32,
        channels: u32,
    },
    #[error(transparent)]
    VarDctMatrix(#[from] jxl_gpu_protocol::VarDctMatrixError),
    #[error("invalid VarDCT coefficient order for family {family}, channel {channel}: {reason}")]
    VarDctCoefficientOrder {
        family: u8,
        channel: u8,
        reason: &'static str,
    },
    #[error(transparent)]
    Unsupported(#[from] UnsupportedFeature),
    #[error(transparent)]
    Packet(#[from] PacketError),
    #[error(transparent)]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error(transparent)]
    Inventory(#[from] jxl_gpu_bitstream::InventoryError),
    #[error(transparent)]
    FramePlan(#[from] jxl_gpu_bitstream::FramePlanError),
    #[error(transparent)]
    FrameIndex(#[from] jxl_gpu_bitstream::FrameIndexError),
    #[error(transparent)]
    AccelerationIndex(#[from] jxl_gpu_bitstream::AccelerationIndexError),
    #[error("GPU encoder kernel policy failed: {0}")]
    KernelPolicy(#[from] jxl_wgpu::Error),
    #[error(transparent)]
    ForwardVarDct(#[from] jxl_wgpu::ForwardVarDctError),
    #[error("invalid encoder configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("weighted predictor {name}[{index}] is {value}, maximum is {maximum}")]
    WeightedPredictorParameter {
        name: &'static str,
        index: usize,
        value: u8,
        maximum: u8,
    },
    #[error("Modular RCT type {rct_type} is invalid; valid types are 0 through 41")]
    InvalidModularRctType { rct_type: u32 },
    #[error("Modular RCT requires three color channels, received {color_channels}")]
    ModularRctColorChannels { color_channels: u32 },
    #[error(transparent)]
    Icc(#[from] jxl_gpu_protocol::icc::IccError),
    #[error(transparent)]
    ResidentIcc(#[from] jxl_wgpu::ResidentIccError),
    #[error("ICC {resource} requires {required} bytes, limit {limit}")]
    IccLimit {
        resource: &'static str,
        required: u64,
        limit: u64,
    },
    #[error("invalid GPU frame source: {0}")]
    InvalidSource(&'static str),
    #[error("invalid encoder source layout: {0}")]
    SourceLayout(#[from] jxl_gpu_formats::LayoutError),
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
    #[error(transparent)]
    ImageSelection(#[from] jxl_gpu_bitstream::ImageSelectionError),
}
