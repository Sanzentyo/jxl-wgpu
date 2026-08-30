use std::num::NonZeroU8;

use crate::{EncodeError, FrameEncodeRequest, UnsupportedFeature};

/// How widely a backend guarantees repeatable output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Determinism {
    /// Group ordering, TOC, and container bytes are deterministic for identical
    /// GPU artifacts. This is the minimum accepted by this crate.
    Assembly,
    /// Complete bytes are stable on the same adapter/driver pair.
    SameDevice,
    /// Complete bytes are stable across conforming adapters.
    CrossDevice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelStage {
    InputNormalization,
    ColorTransform,
    ModularTransform,
    ModularPrediction,
    AdaptiveQuantization,
    AcStrategy,
    ForwardTransform,
    Quantization,
    ProgressiveSplit,
    ModularResidualTokenization,
    CoefficientTokenization,
    HistogramReduction,
}

/// Validated JPEG XL perceptual distance.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PerceptualDistance(f32);

impl PerceptualDistance {
    pub fn new(value: f32) -> Result<Self, EncodeError> {
        if !value.is_finite() || value <= 0.0 || value > 25.0 {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT distance must be finite and in (0, 25]",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EncodeProfile {
    ModularLossless { bits_per_sample: u8 },
    VarDct { distance: PerceptualDistance },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileCapability {
    ModularLossless {
        min_bits_per_sample: u8,
        max_bits_per_sample: u8,
    },
    VarDct {
        min_distance: PerceptualDistance,
        max_distance: PerceptualDistance,
    },
}

impl ProfileCapability {
    #[must_use]
    pub fn supports(self, profile: EncodeProfile) -> bool {
        match (self, profile) {
            (
                Self::ModularLossless {
                    min_bits_per_sample,
                    max_bits_per_sample,
                },
                EncodeProfile::ModularLossless { bits_per_sample },
            ) => (min_bits_per_sample..=max_bits_per_sample).contains(&bits_per_sample),
            (
                Self::VarDct {
                    min_distance,
                    max_distance,
                },
                EncodeProfile::VarDct { distance },
            ) => (min_distance.get()..=max_distance.get()).contains(&distance.get()),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgressivePass {
    pub coefficient_square: NonZeroU8,
    pub shift: u8,
}

/// Spectral/quantized AC passes. JPEG XL permits at most 11 passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgressivePlan {
    passes: Vec<ProgressivePass>,
}

impl ProgressivePlan {
    pub const MAX_PASSES: usize = 11;

    #[must_use]
    pub fn single() -> Self {
        Self {
            passes: vec![ProgressivePass {
                coefficient_square: NonZeroU8::new(8).expect("eight is non-zero"),
                shift: 0,
            }],
        }
    }

    pub fn new(passes: Vec<ProgressivePass>) -> Result<Self, EncodeError> {
        if passes.is_empty() || passes.len() > Self::MAX_PASSES {
            return Err(EncodeError::InvalidConfiguration(
                "progressive plan must contain 1..=11 passes",
            ));
        }
        let mut previous = (1u8, u8::MAX);
        for pass in &passes {
            let coefficients = pass.coefficient_square.get();
            if coefficients > 8
                || coefficients < previous.0
                || (coefficients == previous.0 && pass.shift >= previous.1)
            {
                return Err(EncodeError::InvalidConfiguration(
                    "progressive passes must add coefficients or reduce shift",
                ));
            }
            previous = (coefficients, pass.shift);
        }
        let last = passes.last().expect("non-empty plan was checked");
        if last.coefficient_square.get() != 8 || last.shift != 0 {
            return Err(EncodeError::InvalidConfiguration(
                "final progressive pass must contain the full unshifted 8x8 spectrum",
            ));
        }
        Ok(Self { passes })
    }

    #[must_use]
    pub fn passes(&self) -> &[ProgressivePass] {
        &self.passes
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EncoderCapabilities {
    pub profiles: Vec<ProfileCapability>,
    pub max_progressive_passes: u8,
    pub animation: bool,
    pub determinism: Determinism,
    pub implemented_stages: Vec<KernelStage>,
}

impl EncoderCapabilities {
    #[must_use]
    pub fn prototype() -> Self {
        Self {
            profiles: Vec::new(),
            max_progressive_passes: 0,
            animation: false,
            determinism: Determinism::Assembly,
            implemented_stages: Vec::new(),
        }
    }

    pub fn negotiate(&self, request: &FrameEncodeRequest) -> Result<(), UnsupportedFeature> {
        if !self
            .profiles
            .iter()
            .copied()
            .any(|capability| capability.supports(request.profile))
        {
            return Err(UnsupportedFeature::Profile(request.profile));
        }
        let requested_passes = u8::try_from(request.progressive.passes().len())
            .expect("progressive plans are bounded to 11 entries");
        if requested_passes > self.max_progressive_passes {
            return Err(UnsupportedFeature::ProgressivePasses {
                supported: self.max_progressive_passes,
                requested: requested_passes,
            });
        }
        if request.frame_index.get() != 0 && !self.animation {
            return Err(UnsupportedFeature::Animation);
        }
        if self.determinism < request.minimum_determinism {
            return Err(UnsupportedFeature::DeterministicAssembly);
        }
        Ok(())
    }

    #[must_use]
    pub fn has_stage(&self, stage: KernelStage) -> bool {
        self.implemented_stages.contains(&stage)
    }
}
