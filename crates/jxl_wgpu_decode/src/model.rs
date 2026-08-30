use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use jxl_gpu_bitstream::Gray8AccelerationIndex;
use jxl_gpu_formats::{PixelFormat, PixelFormatClass, classify_pixel_format};
use jxl_gpu_protocol::Extent2d;

use crate::{Error, Result};

/// A GPU decode profile negotiated before any frame is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeProfile {
    /// One-group, lossless Modular data reconstructed by a fixed predictor GPU kernel.
    ModularLossless {
        bits_per_sample: u8,
        predictor: FixedModularPredictor,
        grouping: ModularGrouping,
    },
}

impl DecodeProfile {
    #[must_use]
    pub const fn modular_lossless_8bit(predictor: FixedModularPredictor) -> Self {
        Self::ModularLossless {
            bits_per_sample: 8,
            predictor,
            grouping: ModularGrouping::SingleGroup,
        }
    }

    #[must_use]
    pub const fn modular_lossless_16bit(predictor: FixedModularPredictor) -> Self {
        Self::ModularLossless {
            bits_per_sample: 16,
            predictor,
            grouping: ModularGrouping::SingleGroup,
        }
    }
}

/// Fixed JPEG XL Modular predictor selected for every sample in the negotiated profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FixedModularPredictor(u8);

impl FixedModularPredictor {
    /// Creates a fixed predictor from the JPEG XL predictor index (0..=13).
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if index <= 13 { Some(Self(index)) } else { None }
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }
}

/// Grouping supported by the fixed-predictor GPU entropy/group frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularGrouping {
    SingleGroup,
}

/// Validated contiguous codestream handed to a GPU-only frontend.
#[derive(Clone, Debug)]
pub struct GpuCodestream {
    storage: Arc<[u8]>,
    byte_range: Range<usize>,
    container: bool,
    acceleration_index: Option<Gray8AccelerationIndex>,
}

impl GpuCodestream {
    pub(crate) fn new(
        storage: Arc<[u8]>,
        byte_range: Range<usize>,
        container: bool,
        acceleration_index: Option<Gray8AccelerationIndex>,
    ) -> Self {
        Self {
            storage,
            byte_range,
            container,
            acceleration_index,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.storage[self.byte_range.clone()]
    }

    #[must_use]
    pub fn shared_storage(&self) -> Arc<[u8]> {
        Arc::clone(&self.storage)
    }

    #[must_use]
    pub fn storage_range(&self) -> Range<usize> {
        self.byte_range.clone()
    }

    #[must_use]
    pub const fn is_container(&self) -> bool {
        self.container
    }

    /// Validated private acceleration metadata bound to these exact codestream bytes.
    #[must_use]
    pub const fn acceleration_index(&self) -> Option<&Gray8AccelerationIndex> {
        self.acceleration_index.as_ref()
    }
}

/// Exact JPEG XL animation clock information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameTimebase {
    pub ticks_per_second_numerator: NonZeroU32,
    pub ticks_per_second_denominator: NonZeroU32,
}

impl FrameTimebase {
    #[must_use]
    pub fn seconds_for_ticks(self, ticks: u32) -> f64 {
        f64::from(ticks) * f64::from(self.ticks_per_second_denominator.get())
            / f64::from(self.ticks_per_second_numerator.get())
    }
}

/// Exact frame duration. Still frames have zero ticks and no timebase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameDuration {
    pub ticks: u32,
    pub timebase: Option<FrameTimebase>,
}

impl FrameDuration {
    #[must_use]
    pub const fn still() -> Self {
        Self {
            ticks: 0,
            timebase: None,
        }
    }

    #[must_use]
    pub const fn animation(ticks: u32, timebase: FrameTimebase) -> Self {
        Self {
            ticks,
            timebase: Some(timebase),
        }
    }

    #[must_use]
    pub fn as_seconds(self) -> f64 {
        self.timebase
            .map_or(0.0, |timebase| timebase.seconds_for_ticks(self.ticks))
    }

    #[must_use]
    pub fn as_std(self) -> Option<Duration> {
        Duration::try_from_secs_f64(self.as_seconds()).ok()
    }
}

/// Metadata attached to one GPU-resident presentation frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameMetadata {
    pub index: usize,
    pub duration: FrameDuration,
    /// Presentation start time in stream timebase ticks, accumulated from preceding durations.
    /// Still images use zero.
    pub presentation_ticks: u64,
    /// Exact JPEG XL frame timecode when the animation header enables timecodes.
    ///
    /// This is the bitstream value, not a timestamp derived from preceding frame durations.
    pub timecode: Option<u32>,
    pub is_last: bool,
    pub is_keyframe: bool,
    pub name: String,
}

/// Stream-wide presentation and animation metadata parsed by the GPU frontend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnimationMetadata {
    pub extent: Extent2d,
    pub timebase: Option<FrameTimebase>,
    /// JPEG XL loop count. `None` denotes a still image; zero denotes infinite animation looping.
    pub loop_count: Option<u32>,
    pub has_timecodes: Option<bool>,
    pub frame_count_hint: Option<usize>,
}

impl AnimationMetadata {
    #[must_use]
    pub const fn still(extent: Extent2d) -> Self {
        Self {
            extent,
            timebase: None,
            loop_count: None,
            has_timecodes: None,
            frame_count_hint: Some(1),
        }
    }

    #[must_use]
    pub const fn animation(
        extent: Extent2d,
        timebase: FrameTimebase,
        loop_count: u32,
        has_timecodes: bool,
        frame_count_hint: Option<usize>,
    ) -> Self {
        Self {
            extent,
            timebase: Some(timebase),
            loop_count: Some(loop_count),
            has_timecodes: Some(has_timecodes),
            frame_count_hint,
        }
    }

    #[must_use]
    pub const fn is_animation(&self) -> bool {
        self.timebase.is_some()
    }
}

/// Numeric interpretation applied while converting the decoded Gray8 code into a non-color
/// output sample.
///
/// This mapping is explicit because a [`PixelFormat`] with `ColorModel::NonColor` carries storage
/// shape, not normalization semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericSampleMapping {
    /// Maps the decoded integer code `gray` in `[0, 255]` across the destination's nonnegative
    /// range. Unsigned integers use `[0, MAX]`; signed integers use `[0, MAX]` (never negative);
    /// floating-point values use the normalized `f32` value `gray / 255`. Two-component formats
    /// receive the same value in both components.
    ///
    /// F64 uses a separate policy-bearing variant so precision cannot silently depend on the
    /// selected adapter.
    NormalizedGray8,
    /// The same normalized mapping for an F64 destination, with an explicit precision policy.
    NormalizedGray8F64(F64OutputPolicy),
}

/// Precision policy for normalized F64 output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum F64OutputPolicy {
    /// Require device-enabled `wgpu::Features::SHADER_F64` and evaluate `f64(gray) / 255.0` in the
    /// shader. The request is rejected when native arithmetic is unavailable.
    NativeRequired,
    /// Use native shader f64 when enabled; otherwise use the explicitly permitted compatibility
    /// path described by [`F64OutputPolicy::ExactF32Widening`].
    NativeOrExactF32Widening,
    /// Produce the exact IEEE-754 binary64 widening of the correctly-rounded f32 value
    /// `gray / 255`. This is deterministic binary64 storage, but it is not native f64 arithmetic
    /// and does not preserve the additional precision of evaluating the division in f64.
    ExactF32Widening,
}

/// Semantic side of a [`GpuOutputRequest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuOutputMapping {
    /// The format's explicit color specification determines conversion and packing.
    Color,
    /// A non-color numeric image uses the supplied, explicit sample mapping.
    Numeric(NumericSampleMapping),
}

/// Generic GPU output request. No CPU-readable fallback representation exists.
///
/// Construction is deliberately split between [`GpuOutputRequest::color`] and
/// [`GpuOutputRequest::numeric`]. There is no implicit numeric interpretation and no compatibility
/// constructor which guesses one from the pixel format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOutputRequest {
    format: PixelFormat,
    mapping: GpuOutputMapping,
    max_frame_slots: NonZeroUsize,
}

impl GpuOutputRequest {
    /// Creates a color-bearing output request after semantic classification.
    pub fn color(format: PixelFormat) -> Result<Self> {
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Color(_) => Ok(Self::from_parts(format, GpuOutputMapping::Color)),
            PixelFormatClass::Numeric(_) => Err(Error::NumericMappingRequired),
        }
    }

    /// Creates a non-color output request with an explicit sample mapping.
    pub fn numeric(format: PixelFormat, mapping: NumericSampleMapping) -> Result<Self> {
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Numeric(numeric) => {
                let is_f64 = numeric.sample_kind == jxl_gpu_formats::SampleKind::Float
                    && numeric.bits_per_component == 64;
                match (is_f64, mapping) {
                    (true, NumericSampleMapping::NormalizedGray8) => {
                        return Err(Error::F64OutputPolicyRequired);
                    }
                    (false, NumericSampleMapping::NormalizedGray8F64(_)) => {
                        return Err(Error::F64OutputPolicyForNonF64);
                    }
                    _ => {}
                }
                Ok(Self::from_parts(format, GpuOutputMapping::Numeric(mapping)))
            }
            PixelFormatClass::Color(_) => Err(Error::NumericMappingForColorOutput),
        }
    }

    fn from_parts(format: PixelFormat, mapping: GpuOutputMapping) -> Self {
        Self {
            format,
            mapping,
            max_frame_slots: NonZeroUsize::new(2).expect("two is nonzero"),
        }
    }

    #[must_use]
    pub const fn format(&self) -> &PixelFormat {
        &self.format
    }

    #[must_use]
    pub const fn mapping(&self) -> GpuOutputMapping {
        self.mapping
    }

    /// Maximum number of slots jointly occupied by queued submissions and caller-held frame
    /// leases. This count is independent from the byte-weighted GPU memory budget.
    #[must_use]
    pub const fn max_frame_slots(&self) -> NonZeroUsize {
        self.max_frame_slots
    }

    /// Sets the maximum number of queued-or-caller-held frame slots.
    #[must_use]
    pub const fn with_max_frame_slots(mut self, max_frame_slots: NonZeroUsize) -> Self {
        self.max_frame_slots = max_frame_slots;
        self
    }
}
