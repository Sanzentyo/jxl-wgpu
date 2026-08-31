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
    VarDct(#[from] crate::vardct_engine::VarDctDecodeError),
    #[error("GPU decode backend failed: {0}")]
    Backend(String),
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
}

impl Error {
    #[must_use]
    pub fn backend(error: impl std::fmt::Display) -> Self {
        Self::Backend(error.to_string())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
