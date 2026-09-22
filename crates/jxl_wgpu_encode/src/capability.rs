use std::num::NonZeroU8;

use crate::{EncodeError, FrameEncodeRequest, UnsupportedFeature, VarDctQuantization};

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
    GroupOrderSelection,
    ModularResidualTokenization,
    CoefficientTokenization,
    HistogramReduction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EncodeProfile {
    ModularLossless {
        sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth,
    },
    VarDct {
        quantization: VarDctQuantization,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileCapability {
    ModularLossless {
        min_bits_per_sample: u8,
        max_bits_per_sample: u8,
        /// Zero for integer samples, otherwise the declared floating exponent width.
        exponent_bits_per_sample: u8,
    },
    VarDct {
        quantization: VarDctQuantization,
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
                    exponent_bits_per_sample,
                },
                EncodeProfile::ModularLossless { sample_bit_depth },
            ) => {
                let (bits, exponent) = match (sample_bit_depth, exponent_bits_per_sample) {
                    (jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample }, 0) => {
                        (bits_per_sample, 0)
                    }
                    (
                        jxl_gpu_bitstream::SampleBitDepth::Float {
                            bits_per_sample,
                            exponent_bits_per_sample,
                        },
                        exponent,
                    ) if exponent != 0 => (bits_per_sample, exponent_bits_per_sample),
                    _ => return false,
                };
                (u32::from(min_bits_per_sample)..=u32::from(max_bits_per_sample)).contains(&bits)
                    && exponent == u32::from(exponent_bits_per_sample)
            }
            (
                Self::VarDct { quantization },
                EncodeProfile::VarDct {
                    quantization: requested,
                },
            ) => quantization == requested,
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgressivePass {
    /// Spectral rectangle size in eighths of each canonical transform axis (1..=8).
    /// The longer axis is horizontal, independently of the serialized coefficient order.
    pub coefficient_square: NonZeroU8,
    /// Divide the remaining signed coefficients by `2^shift`, rounding toward zero (0..=3).
    pub shift: u8,
}

/// Spectral/quantized AC passes. JPEG XL permits at most 11 passes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProgressivePlan {
    passes: Vec<ProgressivePass>,
    downsampling: Vec<ProgressiveDownsampling>,
}

/// A completed AC pass after which a decoder may stop at the intended resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProgressiveDownsampling {
    /// Intended downsampling factor: 1, 2, 4 or 8.
    pub factor: u8,
    /// Zero-based last required pass, limited by the wire syntax to 0..=7.
    pub last_pass: u8,
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
            downsampling: Vec::new(),
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
                || pass.shift > 3
                || coefficients < previous.0
                || (coefficients == previous.0 && pass.shift >= previous.1)
            {
                return Err(EncodeError::InvalidConfiguration(
                    "progressive passes require shifts in 0..=3 and must add coefficients or reduce shift",
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
        Ok(Self {
            passes,
            downsampling: Vec::new(),
        })
    }

    /// Replaces the optional resolution stopping points without changing coefficient passes.
    /// Factors must decrease and their last-pass indices must increase. The final full-resolution
    /// endpoint and initial DC downsampling of eight remain implicit when not specified.
    pub fn with_downsampling(
        mut self,
        downsampling: Vec<ProgressiveDownsampling>,
    ) -> Result<Self, EncodeError> {
        if downsampling.len() > 4
            || downsampling.len() > self.passes.len()
            || (self.passes.len() == 1 && !downsampling.is_empty())
        {
            return Err(EncodeError::InvalidConfiguration(
                "progressive downsampling requires a multi-pass plan and at most four endpoints",
            ));
        }
        let mut previous = None::<ProgressiveDownsampling>;
        for &point in &downsampling {
            if !matches!(point.factor, 1 | 2 | 4 | 8)
                || point.last_pass > 7
                || usize::from(point.last_pass) >= self.passes.len()
                || previous.is_some_and(|old| {
                    point.factor >= old.factor || point.last_pass <= old.last_pass
                })
            {
                return Err(EncodeError::InvalidConfiguration(
                    "progressive factors must decrease through 8/4/2/1 at increasing encoded pass indices",
                ));
            }
            previous = Some(point);
        }
        self.downsampling = downsampling;
        Ok(self)
    }

    #[must_use]
    pub fn downsampling(&self) -> &[ProgressiveDownsampling] {
        &self.downsampling
    }

    #[must_use]
    pub fn passes(&self) -> &[ProgressivePass] {
        &self.passes
    }
}

impl Default for ProgressivePlan {
    fn default() -> Self {
        Self::single()
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
    pub fn empty() -> Self {
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
        if (request.animation.is_animation() || request.frame_index.get() != 0) && !self.animation {
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::{AnimationHeader, FrameIndex, FrameOptions};

    use super::*;

    #[test]
    fn modular_profiles_distinguish_numeric_type_and_exponent_width() {
        use jxl_gpu_bitstream::SampleBitDepth::{Float, Integer};

        let profiles = [(1, 31, 0), (16, 16, 5), (32, 32, 8)].map(|(min, max, exponent)| {
            ProfileCapability::ModularLossless {
                min_bits_per_sample: min,
                max_bits_per_sample: max,
                exponent_bits_per_sample: exponent,
            }
        });
        for (sample_bit_depth, expected) in [
            (
                Integer {
                    bits_per_sample: 16,
                },
                [true, false, false],
            ),
            (
                Integer {
                    bits_per_sample: 32,
                },
                [false, false, false],
            ),
            (
                Float {
                    bits_per_sample: 16,
                    exponent_bits_per_sample: 5,
                },
                [false, true, false],
            ),
            (
                Float {
                    bits_per_sample: 32,
                    exponent_bits_per_sample: 8,
                },
                [false, false, true],
            ),
            (
                Float {
                    bits_per_sample: 16,
                    exponent_bits_per_sample: 0,
                },
                [false, false, false],
            ),
            (
                Float {
                    bits_per_sample: 16,
                    exponent_bits_per_sample: 8,
                },
                [false, false, false],
            ),
            (
                Float {
                    bits_per_sample: 24,
                    exponent_bits_per_sample: 8,
                },
                [false, false, false],
            ),
        ] {
            let request = EncodeProfile::ModularLossless { sample_bit_depth };
            assert_eq!(profiles.map(|profile| profile.supports(request)), expected);
        }
    }

    #[test]
    fn animation_header_is_negotiated_even_for_frame_zero() {
        let capabilities = EncoderCapabilities {
            profiles: vec![ProfileCapability::ModularLossless {
                min_bits_per_sample: 8,
                max_bits_per_sample: 8,
                exponent_bits_per_sample: 0,
            }],
            max_progressive_passes: 1,
            animation: false,
            determinism: Determinism::Assembly,
            implemented_stages: Vec::new(),
        };
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::ModularLossless {
                sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample: 8 },
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::Assembly,
            animation: AnimationHeader::Animation {
                ticks_per_second_numerator: NonZeroU32::new(24).unwrap(),
                ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                num_loops: 0,
                have_timecodes: false,
            },
            canvas_width: 1,
            canvas_height: 1,
            options: FrameOptions::default(),
        };
        assert_eq!(
            capabilities.negotiate(&request),
            Err(UnsupportedFeature::Animation)
        );
    }
}
