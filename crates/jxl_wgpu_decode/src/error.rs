use thiserror::Error;

/// Codestream feature recognized by a frontend but outside its GPU profile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnsupportedCodestreamFeature {
    VarDct,
    ModularBitDepth(u8),
    AdaptiveModularPredictor,
    MultiplePasses,
    ExtraChannels,
    Patches,
    Splines,
    Noise,
    ModularTransform(ModularTransformFeature),
    AnimationReferences,
    Other(String),
}

/// Standard Modular transform that has not yet been lowered to the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularTransformFeature {
    ReversibleColor { begin_channel: u32, rct_type: u32 },
    Palette,
    Squeeze,
    Invalid,
}

/// Invalid or resource-exhausting standard Modular metadata.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularTreeError {
    #[error("Modular metadata exceeded the bounded {resource} limit of {limit}")]
    LimitExceeded {
        resource: &'static str,
        limit: usize,
    },
    #[error("invalid Modular entropy descriptor: {reason}")]
    InvalidEntropy { reason: &'static str },
    #[error(
        "invalid Modular hybrid integer config at bit {bit_offset} for log alphabet {log_alphabet_size}: split exponent {split_exponent}, MSB {msb_in_token}, LSB {lsb_in_token}"
    )]
    InvalidHybridConfig {
        bit_offset: u64,
        log_alphabet_size: u32,
        split_exponent: u32,
        msb_in_token: u32,
        lsb_in_token: u32,
    },
    #[error("invalid Modular MA tree: {reason}")]
    InvalidTree { reason: &'static str },
    #[error("Modular MA tree predictor index {predictor} is outside 0 through 13")]
    InvalidPredictor { predictor: u32 },
    #[error("Modular MA tree depth {depth} exceeds the bounded limit {limit}")]
    TreeDepthExceeded { depth: usize, limit: usize },
}

/// Invalid or resource-exhausting JPEG XL Modular transform metadata.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ModularTransformError {
    #[error("Modular transform count {actual} exceeds the bounded limit {limit}")]
    TransformLimitExceeded { actual: usize, limit: usize },
    #[error("Modular transformed channel count {actual} exceeds the bounded limit {limit}")]
    ChannelLimitExceeded { actual: usize, limit: usize },
    #[error("Modular squeeze parameter count {actual} exceeds the bounded limit {limit}")]
    SqueezeLimitExceeded { actual: usize, limit: usize },
    #[error("Modular transform topology work {actual} exceeds the bounded limit {limit}")]
    TopologyWorkLimitExceeded { actual: usize, limit: usize },
    #[error("invalid Modular transform id {id}")]
    InvalidTransformId { id: u32 },
    #[error("invalid Modular RCT type {rct_type}; valid types are 0 through 41")]
    InvalidRctType { rct_type: u32 },
    #[error("invalid Modular palette predictor {predictor}; valid predictors are 0 through 13")]
    InvalidPalettePredictor { predictor: u32 },
    #[error(
        "{transform} channel range {begin}..{end} exceeds the {available} transformed channels"
    )]
    ChannelRange {
        transform: &'static str,
        begin: u32,
        end: u32,
        available: usize,
    },
    #[error("{transform} cannot mix meta and non-meta channels")]
    MixedMetaChannels { transform: &'static str },
    #[error("{transform} requires channels with identical geometry and bit depth")]
    UnequalChannels { transform: &'static str },
    #[error("a Modular squeeze of meta channels must place residuals in-place")]
    MetaSqueezeRequiresInPlace,
    #[error("default Modular squeeze requires at least one non-meta channel")]
    MissingDataChannel,
    #[error("Modular channel {channel} has zero extent before a squeeze operation")]
    ZeroSizedSqueezeChannel { channel: usize },
    #[error("Modular channel {channel} has exceeded the maximum squeeze shift of 30")]
    TooManySqueezes { channel: usize },
    #[error("Modular palette dimensions overflow the GPU address space")]
    PaletteDimensionOverflow,
    #[error("Modular transformed sample storage exceeds the portable WGSL u32 address space")]
    GpuAddressSpaceOverflow,
    #[error("cannot reverse {transform} topology: {reason}")]
    InvalidInverseTopology {
        transform: &'static str,
        reason: &'static str,
    },
    #[error("reversing the Modular transform stack did not recover its source topology")]
    InverseTopologyMismatch,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("unsupported GPU decode profile feature {feature:?}: {detail}")]
pub struct UnsupportedProfile {
    pub feature: UnsupportedCodestreamFeature,
    pub detail: String,
}

impl UnsupportedProfile {
    #[must_use]
    pub fn new(feature: UnsupportedCodestreamFeature, detail: impl Into<String>) -> Self {
        Self {
            feature,
            detail: detail.into(),
        }
    }
}

/// GPU frontend stage that is not implemented yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrontendStage {
    CodestreamHeader,
    EntropyGroups,
    ModularResiduals,
    RenderPlan,
    GpuSubmission,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("GPU decode frontend is incomplete at {stage:?}: {detail}")]
pub struct FrontendIncomplete {
    pub stage: FrontendStage,
    pub detail: String,
}

impl FrontendIncomplete {
    #[must_use]
    pub fn new(stage: FrontendStage, detail: impl Into<String>) -> Self {
        Self {
            stage,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("JPEG XL container/codestream validation failed: {0}")]
    Bitstream(#[from] jxl_gpu_bitstream::Error),
    #[error("JPEG XL standard codestream inventory failed: {0}")]
    CodestreamInventory(#[from] jxl_gpu_bitstream::InventoryError),
    #[error("incremental JPEG XL codestream inventory failed: {0}")]
    CodestreamStream(#[from] jxl_gpu_bitstream::CodestreamStreamError),
    #[error(transparent)]
    IncrementalInputBudget(#[from] crate::IncrementalInputBudgetError),
    #[error("invalid requested pixel format: {0}")]
    PixelFormat(#[from] jxl_gpu_formats::PixelFormatError),
    #[error("invalid requested image layout: {0}")]
    ImageLayout(#[from] jxl_gpu_formats::LayoutError),
    #[error("the stock GPU decoder cannot produce the requested pixel format: {0}")]
    UnsupportedOutputFormat(String),
    #[error("a non-color numeric output requires an explicit NumericSampleMapping")]
    NumericMappingRequired,
    #[error("a numeric sample mapping cannot be attached to a color output format")]
    NumericMappingForColorOutput,
    #[error("F64 output requires an explicit F64OutputPolicy")]
    F64OutputPolicyRequired,
    #[error("an F64 output policy cannot be attached to a non-F64 numeric format")]
    F64OutputPolicyForNonF64,
    #[error("native F64 output was required but the wgpu device lacks enabled SHADER_F64")]
    NativeF64Unavailable,
    #[error("GPU kernel policy failed: {0}")]
    KernelPolicy(#[from] jxl_wgpu::Error),
    #[error(transparent)]
    UnsupportedProfile(#[from] UnsupportedProfile),
    #[error(transparent)]
    FrontendIncomplete(#[from] FrontendIncomplete),
    #[error(transparent)]
    ModularTree(#[from] ModularTreeError),
    #[error(transparent)]
    ModularTransform(#[from] ModularTransformError),
    #[error(transparent)]
    VarDct(#[from] crate::vardct_engine::VarDctDecodeError),
    #[error("GPU decode backend failed: {0}")]
    Backend(String),
    #[error(
        "the bounded entropy stream window is {limit_bytes} bytes, but at least {minimum_bytes} bytes are required"
    )]
    StreamWindowTooSmall {
        limit_bytes: u64,
        minimum_bytes: u64,
    },
    #[error(
        "Modular GPU group {group_index} rejected entropy stream: status={status}, decoded={decoded_samples}/{expected_samples}, cursor={cursor}/{expected_cursor}"
    )]
    ModularEntropyRejected {
        group_index: usize,
        status: u32,
        decoded_samples: u32,
        expected_samples: u32,
        cursor: u32,
        expected_cursor: u32,
    },
    #[error("GPU decode memory backpressure: {0}")]
    MemoryBackpressure(#[from] jxl_wgpu::MemoryBudgetError),
    #[error("GPU decode submission-poll backpressure: {0}")]
    PollBackpressure(#[from] jxl_wgpu::SubmissionPollerError),
    #[error("GPU decode engine violated its public contract: {0}")]
    EngineContract(&'static str),
    #[error("JPEG XL codestream inventory contains no image frame")]
    MissingImageFrame,
    #[error("all {limit} bounded GPU frame slots are in flight")]
    Backpressure { limit: usize },
    #[error("prefetch depth {requested} exceeds the configured frame-slot limit {limit}")]
    PrefetchDepthExceedsLimit { requested: usize, limit: usize },
    #[error("blocking GPU completion waits are unavailable on browser WebGPU")]
    BlockingWaitUnavailable,
    #[error("expected visible frame index {expected}, engine returned {actual}")]
    UnexpectedFrameIndex { expected: usize, actual: usize },
    #[error("frame {index} timing does not use the stream timebase")]
    FrameTimebaseMismatch { index: usize },
    #[error("frame {index} presentation tick is {actual}, expected {expected}")]
    FramePresentationTicksMismatch {
        index: usize,
        expected: u64,
        actual: u64,
    },
    #[error("presentation tick accumulation overflowed after frame {index}")]
    FramePresentationTicksOverflow { index: usize },
    #[error(
        "frame {index} timecode presence ({frame_has_timecode}) does not match the stream declaration ({stream_has_timecodes})"
    )]
    FrameTimecodePresenceMismatch {
        index: usize,
        stream_has_timecodes: bool,
        frame_has_timecode: bool,
    },
    #[error("stream advertised {hint} visible frames but produced {actual}")]
    FrameCountMismatch { hint: usize, actual: usize },
    #[error("GPU frontend ended without returning a final visible frame")]
    MissingFinalFrame,
    #[error("GPU decode session cannot continue after a previous engine failure")]
    SessionPoisoned,
    #[error("incremental GPU decode input ended before an authoritative transport End event")]
    IncrementalInputIncomplete,
    #[error("incremental GPU decode input cannot continue after a previous failure")]
    IncrementalInputPoisoned,
}

impl Error {
    #[must_use]
    pub fn backend(error: impl std::fmt::Display) -> Self {
        Self::Backend(error.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
