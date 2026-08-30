use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::time::Duration;

use jxl_gpu_bitstream::Gray8AccelerationIndex;
use jxl_gpu_formats::PixelFormat;
use jxl_gpu_protocol::Extent2d;

/// The first end-to-end GPU decode target.
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
    pub const fn prototype_8bit(predictor: FixedModularPredictor) -> Self {
        Self::ModularLossless {
            bits_per_sample: 8,
            predictor,
            grouping: ModularGrouping::SingleGroup,
        }
    }

    #[must_use]
    pub const fn prototype_16bit(predictor: FixedModularPredictor) -> Self {
        Self::ModularLossless {
            bits_per_sample: 16,
            predictor,
            grouping: ModularGrouping::SingleGroup,
        }
    }
}

/// Fixed JPEG XL Modular predictor selected for every sample in the prototype profile.
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

/// Grouping supported by the prototype GPU entropy/group frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularGrouping {
    SingleGroup,
}

/// Validated contiguous codestream handed to a GPU-only frontend.
#[derive(Clone, Debug)]
pub struct GpuCodestream {
    bytes: Arc<[u8]>,
    container: bool,
    acceleration_index: Option<Gray8AccelerationIndex>,
}

impl GpuCodestream {
    pub(crate) const fn new(
        bytes: Arc<[u8]>,
        container: bool,
        acceleration_index: Option<Gray8AccelerationIndex>,
    ) -> Self {
        Self {
            bytes,
            container,
            acceleration_index,
        }
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn shared_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
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

/// Generic GPU output request. No CPU-readable fallback representation exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOutputRequest {
    pub format: PixelFormat,
    pub max_in_flight: NonZeroUsize,
}

impl GpuOutputRequest {
    #[must_use]
    pub fn new(format: PixelFormat) -> Self {
        Self {
            format,
            max_in_flight: NonZeroUsize::new(2).expect("two is nonzero"),
        }
    }

    #[must_use]
    pub const fn with_max_in_flight(mut self, max_in_flight: NonZeroUsize) -> Self {
        self.max_in_flight = max_in_flight;
        self
    }
}
