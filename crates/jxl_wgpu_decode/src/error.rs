use thiserror::Error;

/// Codestream feature recognized by a frontend but outside its GPU profile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnsupportedCodestreamFeature {
    AccelerationIndex,
    VarDct,
    ModularBitDepth(u8),
    AdaptiveModularPredictor,
    MultipleGroups,
    MultiplePasses,
    ExtraChannels,
    Patches,
    Splines,
    Noise,
    AnimationReferences,
    Other(String),
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
    #[error("invalid jwgp GPU acceleration index: {0}")]
    AccelerationIndex(#[from] jxl_gpu_bitstream::AccelerationIndexError),
    #[error("invalid requested pixel format: {0}")]
    PixelFormat(#[from] jxl_gpu_formats::PixelFormatError),
    #[error("invalid requested image layout: {0}")]
    ImageLayout(#[from] jxl_gpu_formats::LayoutError),
    #[error("the input contains more than one jwgp GPU acceleration index")]
    DuplicateAccelerationIndex,
    #[error("the stock GPU decoder cannot produce the requested pixel format: {0}")]
    UnsupportedOutputFormat(String),
    #[error(transparent)]
    UnsupportedProfile(#[from] UnsupportedProfile),
    #[error(transparent)]
    FrontendIncomplete(#[from] FrontendIncomplete),
    #[error("GPU decode backend failed: {0}")]
    Backend(String),
    #[error("GPU decode engine violated its public contract: {0}")]
    EngineContract(&'static str),
    #[error("all {limit} bounded GPU frame slots are in flight")]
    Backpressure { limit: usize },
    #[error("an asynchronously-polled GPU frame is already pending")]
    OperationInProgress,
    #[error("expected visible frame index {expected}, engine returned {actual}")]
    UnexpectedFrameIndex { expected: usize, actual: usize },
    #[error("frame {index} timing does not use the stream timebase")]
    FrameTimebaseMismatch { index: usize },
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
