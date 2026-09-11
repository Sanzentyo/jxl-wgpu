use std::num::{NonZeroU32, NonZeroUsize};
use std::time::Duration;

use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingFieldKind, PackingWord, PixelFormat, PixelFormatClass, PlaneFormat, PlaneSampling,
    SampleKind, Swizzle, classify_pixel_format,
};
use jxl_gpu_protocol::Extent2d;

use crate::{Error, Result};

/// A GPU decode profile negotiated before any frame is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeProfile {
    /// A sequence of independently negotiated physical frames. Per-frame coding modes and
    /// dependencies are represented by the common frame execution plan.
    FrameSequence {
        physical_frames: usize,
        presentation_frames: usize,
    },
    /// Modular working words reconstructed by a GPU entropy/MA pipeline, with optional resampling.
    Modular {
        sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth,
        channels: ModularChannelCounts,
        prediction: ModularPredictionProfile,
        grouping: ModularGrouping,
        /// Progressive pass count declared by the frame (`1..=11`).
        passes: u32,
    },
    /// Standard XYB VarDCT decoded into a GPU-resident presentation buffer. Transform strategy is
    /// selected independently for every first block and remains GPU-resident.
    VarDct {
        sample_bit_depth: jxl_gpu_bitstream::SampleBitDepth,
    },
}

/// Channel arrangement of a native unsigned Modular output pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularChannels {
    Gray,
    Rgb,
    Rgba,
}

impl ModularChannels {
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Codestream channel topology, independent of the requested output pixel format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModularChannelCounts {
    color: u32,
    total: NonZeroU32,
}

impl ModularChannelCounts {
    /// Creates a Gray or RGB source topology with a checked number of extra channels.
    pub fn new(grayscale: bool, extra: u32) -> Result<Self> {
        let color: u32 = if grayscale { 1 } else { 3 };
        let total = color.checked_add(extra).and_then(NonZeroU32::new).ok_or(
            Error::ModularChannelCountOverflow {
                color_channels: color,
                extra_channels: extra,
            },
        )?;
        Ok(Self { color, total })
    }

    #[must_use]
    pub const fn count(self) -> u32 {
        self.total.get()
    }

    #[must_use]
    pub const fn color_count(self) -> u32 {
        self.color
    }

    #[must_use]
    pub const fn extra_count(self) -> u32 {
        self.count() - self.color
    }

    pub(crate) const fn conventional(self) -> Option<ModularChannels> {
        match (self.color, self.extra_count()) {
            (1, 0) => Some(ModularChannels::Gray),
            (3, 0) => Some(ModularChannels::Rgb),
            (3, 1) => Some(ModularChannels::Rgba),
            _ => None,
        }
    }
}

impl From<ModularChannels> for ModularChannelCounts {
    fn from(channels: ModularChannels) -> Self {
        Self {
            color: if channels == ModularChannels::Gray {
                1
            } else {
                3
            },
            total: NonZeroU32::new(channels.count()).expect("native channels are nonempty"),
        }
    }
}

/// JPEG XL Modular predictor selected by a fixed synthetic frontend or an MA-tree leaf.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ModularPredictor {
    #[default]
    Zero = 0,
    West,
    North,
    AvgWestAndNorth,
    Select,
    Gradient,
    SelfCorrecting,
    NorthEast,
    NorthWest,
    WestWest,
    AvgWestAndNorthWest,
    AvgNorthAndNorthWest,
    AvgNorthAndNorthEast,
    AvgAll,
}

impl ModularPredictor {
    #[must_use]
    pub const fn index(self) -> u8 {
        self as u8
    }
}

/// Predictor metadata represented by a negotiated lossless Modular profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularPredictionProfile {
    /// A synthetic/custom engine applies one predictor without a standard MA-tree descriptor.
    Fixed { predictor: ModularPredictor },
    /// A standards-compliant MA tree and its entropy contexts were lowered to GPU metadata.
    MetaAdaptive {
        node_count: u32,
        decision_node_count: u32,
        leaf_context_count: u32,
        max_depth: u32,
        uses_self_correcting: bool,
    },
}

/// Pass-group layout represented by the GPU entropy/group frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModularGrouping {
    SingleGroup,
    /// Row-major 256x256 pass groups covering one canvas.
    MultipleGroups {
        columns: u32,
        rows: u32,
    },
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
    /// Extra-channel declarations in codestream order, including exact names and sample metadata.
    pub extra_channels: Vec<jxl_gpu_bitstream::ExtraChannelInventory>,
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
            extra_channels: Vec::new(),
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
            extra_channels: Vec::new(),
        }
    }

    #[must_use]
    pub const fn is_animation(&self) -> bool {
        self.timebase.is_some()
    }
}

/// Numeric interpretation applied while writing one decoded color or extra-channel sample.
///
/// This mapping is explicit because a [`PixelFormat`] with `ColorModel::NonColor` carries storage
/// shape, not normalization semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericSampleMapping {
    /// Deliver floating samples in the original image encoding as binary32 values. Lossless
    /// Modular samples are exact binary32 widenings of the declared source precision. Unresampled,
    /// uncomposed lossless samples preserve signed zeros, subnormals, infinities and NaN payloads.
    /// VarDCT, filtering and composition return reconstructed floating values. No presentation
    /// color conversion, spot rendering or alpha-association change is applied.
    NativeFloat,
    /// Deliver unsigned integer codes in the low valid bits of the canonical Gray
    /// `u8`/`u16`/`u32` storage descriptor. The requested valid depth and the codestream depth must
    /// match. Lossless Modular preserves source codes exactly; uncomposed Modular working samples
    /// outside that unsigned range return a typed error. Composed presentation clamps to the range.
    /// Resampled planes are reconstructed at presentation resolution first, then rounded to the
    /// nearest code at the declared depth; exact preservation applies before filtering/composition.
    /// VarDCT color samples are reconstructed in the original image encoding, clamped to the
    /// unsigned range and rounded once at the declared depth; the original lossy input is not
    /// recoverable.
    NativeUnsigned,
    /// Divide a 1–31-bit unsigned source by its own maximum code into scalar F32 storage.
    /// Color samples use the original image encoding after codec reconstruction; no presentation
    /// color conversion, spot rendering or alpha-association change is applied. Extra channels
    /// never pass through a color transform.
    /// Signed working samples outside the declared unsigned range remain outside `[0, 1]`;
    /// normalization does not wrap or clamp them.
    NormalizedUnsigned,
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

/// Source of a scalar numeric output. Color indices follow the original image encoding:
/// `0` for grayscale, or `0`, `1`, `2` for red, green and blue. Extra indices follow stream metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericChannel {
    Color(u32),
    Extra(u32),
}

/// Generic GPU output request. No CPU-readable fallback representation exists.
///
/// Construction is deliberately split between [`GpuOutputRequest::color`] and
/// [`GpuOutputRequest::numeric`]. There is no implicit numeric interpretation and no compatibility
/// constructor which guesses one from the pixel format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOutputRequest {
    image: crate::ImageSelection,
    progressive_output: bool,
    format: PixelFormat,
    mapping: GpuOutputMapping,
    max_frame_slots: NonZeroUsize,
    orientation: OrientationPolicy,
    numeric_channel: Option<NumericChannel>,
    spot_colors: SpotColorPolicy,
    alpha: AlphaOutputPolicy,
    frame_surface: Option<crate::frame_surface::FrameSurfaceEncoding>,
    lf_presentation: bool,
}

/// Association of the first alpha channel and color output. Numeric requests, including selected
/// extra channels, always preserve their sample semantics and ignore this policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaOutputPolicy {
    /// Return straight color. Associated sources are divided by alpha after the requested color
    /// conversion and before quantization, using a finite `2^-26` denominator floor.
    /// This also applies when the output format omits alpha.
    #[default]
    Unassociated,
    /// Keep the association declared in stream metadata, including invisible color values.
    Preserve,
    /// Return associated color. Unassociated sources are multiplied by alpha after the requested
    /// color conversion, with the same finite floor used by unpremultiplication.
    Associated,
}

impl AlphaOutputPolicy {
    pub(crate) fn conversion(self, associated: Option<bool>) -> jxl_wgpu::AlphaConversion {
        use jxl_wgpu::AlphaConversion;
        match (self, associated) {
            (Self::Unassociated, Some(true)) => AlphaConversion::Unpremultiply,
            (Self::Associated, Some(false)) => AlphaConversion::Premultiply,
            _ => AlphaConversion::Preserve,
        }
    }
}

/// Whether spot inks are rendered into color output or preserved as independent channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpotColorPolicy {
    /// Mix declared spot inks in order at presentation, after reference storage and before
    /// target color conversion, alpha association, and output quantization.
    #[default]
    Render,
    /// Return the base color and keep spot declarations available in stream metadata.
    Preserve,
}

/// Whether image orientation is applied to the returned pixels and extent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrientationPolicy {
    /// Return presentation coordinates, applying the orientation in the image header.
    #[default]
    Apply,
    /// Keep pixels in codestream coordinates with the unrotated width and height.
    Keep,
}

impl OrientationPolicy {
    pub(crate) const fn resolve(
        self,
        orientation: jxl_gpu_protocol::OutputOrientation,
    ) -> jxl_gpu_protocol::OutputOrientation {
        match self {
            Self::Apply => orientation,
            Self::Keep => jxl_gpu_protocol::OutputOrientation::Identity,
        }
    }
}

impl GpuOutputRequest {
    /// Creates a color-bearing output request after semantic classification or recognition of the
    /// canonical valid-bit-padded RGB/RGBA lossless-Modular descriptor.
    pub fn color(format: PixelFormat) -> Result<Self> {
        if native_modular_format(&format).is_some_and(|native| {
            matches!(
                native.channels,
                ModularChannels::Rgb | ModularChannels::Rgba
            )
        }) {
            return Ok(Self::from_parts(format, GpuOutputMapping::Color));
        }
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Color(_) => Ok(Self::from_parts(format, GpuOutputMapping::Color)),
            PixelFormatClass::Numeric(_) => Err(Error::NumericMappingRequired),
        }
    }

    /// Creates a non-color output request with an explicit sample mapping. `NativeUnsigned`
    /// recognizes the canonical valid-bit-padded unsigned Gray descriptor directly.
    pub fn numeric(format: PixelFormat, mapping: NumericSampleMapping) -> Result<Self> {
        if mapping == NumericSampleMapping::NativeUnsigned {
            return match native_modular_format(&format) {
                Some(NativeModularFormat {
                    channels: ModularChannels::Gray,
                    ..
                }) => Ok(Self::from_parts(format, GpuOutputMapping::Numeric(mapping))),
                Some(_) => Err(Error::NumericMappingForColorOutput),
                None => Err(Error::UnsupportedOutputFormat(
                    "native unsigned output requires the canonical unsigned Gray descriptor".into(),
                )),
            };
        }
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Numeric(numeric) => {
                if mapping == NumericSampleMapping::NativeFloat
                    && !(numeric.sample_kind == SampleKind::Float
                        && numeric.bits_per_component == 32
                        && numeric.components == 1)
                {
                    return Err(Error::UnsupportedOutputFormat(
                        "native floating samples require scalar F32 storage".into(),
                    ));
                }
                let is_f64 = numeric.sample_kind == jxl_gpu_formats::SampleKind::Float
                    && numeric.bits_per_component == 64;
                match (is_f64, mapping) {
                    (_, NumericSampleMapping::NativeUnsigned) => unreachable!(
                        "native unsigned requests return before generic numeric classification"
                    ),
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
            image: crate::ImageSelection::Main,
            progressive_output: false,
            format,
            mapping,
            max_frame_slots: NonZeroUsize::new(2).expect("two is nonzero"),
            orientation: OrientationPolicy::Apply,
            numeric_channel: None,
            spot_colors: SpotColorPolicy::Render,
            alpha: AlphaOutputPolicy::default(),
            frame_surface: None,
            lf_presentation: false,
        }
    }

    #[must_use]
    pub const fn format(&self) -> &PixelFormat {
        &self.format
    }

    /// Select the main image/animation or the independent embedded preview.
    #[must_use]
    pub const fn with_image_selection(mut self, selection: crate::ImageSelection) -> Self {
        self.image = selection;
        self
    }

    #[must_use]
    pub const fn image_selection(&self) -> crate::ImageSelection {
        self.image
    }

    /// Requests intermediate images at validated LF-frame, coefficient-pass and Modular residual
    /// boundaries. Consume them with `GpuDecodeSession::next_update` or its async counterpart.
    /// Composed updates retain presentation timing and committed references, including numeric
    /// output. LF images expand pre-color XYB and independent normalized extras before packing
    /// or composition. Frames without an available intermediate image still return their final image.
    #[must_use]
    pub const fn with_progressive_output(mut self, enabled: bool) -> Self {
        self.progressive_output = enabled;
        self
    }

    #[must_use]
    pub const fn progressive_output(&self) -> bool {
        self.progressive_output
    }

    #[must_use]
    pub const fn mapping(&self) -> GpuOutputMapping {
        self.mapping
    }

    #[must_use]
    pub const fn alpha_output_policy(&self) -> AlphaOutputPolicy {
        self.alpha
    }

    pub(crate) fn for_frame_surface(
        mut self,
        encoding: crate::frame_surface::FrameSurfaceEncoding,
    ) -> Self {
        self.frame_surface = Some(encoding);
        self.format = encoding.format();
        self.alpha = AlphaOutputPolicy::Preserve;
        self.spot_colors = SpotColorPolicy::Preserve;
        self.orientation = OrientationPolicy::Keep;
        self
    }

    /// Retain normalized LF extras only while an intermediate presentation can use them.
    pub(crate) const fn with_lf_presentation(mut self, enabled: bool) -> Self {
        self.lf_presentation = enabled;
        self
    }

    pub(crate) const fn retains_lf_presentation(&self) -> bool {
        self.lf_presentation
    }

    pub(crate) const fn retains_frame_surface(&self) -> bool {
        self.frame_surface.is_some()
    }

    pub(crate) fn frame_surface_encoding(&self) -> crate::frame_surface::FrameSurfaceEncoding {
        self.frame_surface.expect("private frame surface request")
    }

    pub(crate) fn renders_spot_colors(
        &self,
        extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
    ) -> bool {
        self.mapping == GpuOutputMapping::Color
            && self.spot_colors == SpotColorPolicy::Render
            && extras.iter().any(|extra| {
                matches!(
                    extra.channel_type,
                    jxl_gpu_bitstream::ExtraChannelTypeInventory::SpotColour { .. }
                )
            })
    }

    #[must_use]
    pub const fn with_alpha_output_policy(mut self, policy: AlphaOutputPolicy) -> Self {
        self.alpha = policy;
        self
    }

    pub(crate) fn alpha_conversion(
        &self,
        extras: &[jxl_gpu_bitstream::ExtraChannelInventory],
    ) -> jxl_wgpu::AlphaConversion {
        if self.mapping != GpuOutputMapping::Color {
            return jxl_wgpu::AlphaConversion::Preserve;
        }
        let associated = extras.iter().find_map(|extra| match extra.channel_type {
            jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated } => Some(associated),
            _ => None,
        });
        self.alpha.conversion(associated)
    }

    /// Selects one decoded sample channel, replacing any previous selection. RGB numeric output
    /// requires an explicit color index; grayscale defaults to its only color channel. Indices
    /// are checked against the selected image when opening the decoder.
    ///
    /// Color values preserve the original image encoding and alpha association, with spot inks
    /// excluded. VarDCT color output uses the common [`crate::WgpuDecodeEngine`] presentation path.
    pub fn with_numeric_channel(mut self, channel: NumericChannel) -> Result<Self> {
        let scalar = self.format.planes.len() == 1
            && match classify_pixel_format(&self.format) {
                Ok(PixelFormatClass::Numeric(numeric)) => numeric.components == 1,
                _ => native_modular_format(&self.format)
                    .is_some_and(|native| native.channels == ModularChannels::Gray),
            };
        if !scalar
            || !matches!(
                self.mapping,
                GpuOutputMapping::Numeric(
                    NumericSampleMapping::NativeUnsigned
                        | NumericSampleMapping::NormalizedUnsigned
                        | NumericSampleMapping::NativeFloat
                )
            )
        {
            return Err(Error::UnsupportedOutputFormat(
                "channel selection requires scalar native unsigned, normalized integer or floating F32 samples"
                    .into(),
            ));
        }
        self.numeric_channel = Some(channel);
        Ok(self)
    }

    /// Selects gray (`0`) or one original RGB component (`0` red, `1` green, `2` blue).
    pub fn with_color_channel(self, index: u32) -> Result<Self> {
        self.with_numeric_channel(NumericChannel::Color(index))
    }

    /// Selects one extra channel by its index in the stream metadata. The result is a scalar
    /// numeric image; its values never pass through a color transfer function.
    pub fn with_extra_channel(self, index: u32) -> Result<Self> {
        self.with_numeric_channel(NumericChannel::Extra(index))
    }

    #[must_use]
    pub const fn numeric_channel(&self) -> Option<NumericChannel> {
        self.numeric_channel
    }

    #[must_use]
    pub const fn color_channel(&self) -> Option<u32> {
        match self.numeric_channel {
            Some(NumericChannel::Color(index)) => Some(index),
            _ => None,
        }
    }

    #[must_use]
    pub const fn extra_channel(&self) -> Option<u32> {
        match self.numeric_channel {
            Some(NumericChannel::Extra(index)) => Some(index),
            _ => None,
        }
    }

    pub(crate) fn numeric_color_channel(&self, grayscale: bool) -> Result<Option<u32>> {
        if self.mapping == GpuOutputMapping::Color || self.extra_channel().is_some() {
            return Ok(None);
        }
        let count = if grayscale { 1 } else { 3 };
        let index = match self.color_channel() {
            Some(index) => index,
            None if grayscale => 0,
            None => return Err(Error::NumericColorChannelRequired),
        };
        if index >= count {
            return Err(Error::ColorChannelIndex { index, count });
        }
        Ok(Some(index))
    }

    #[must_use]
    pub const fn with_spot_color_policy(mut self, policy: SpotColorPolicy) -> Self {
        self.spot_colors = policy;
        self
    }

    #[must_use]
    pub const fn spot_color_policy(&self) -> SpotColorPolicy {
        self.spot_colors
    }

    #[must_use]
    pub const fn orientation_policy(&self) -> OrientationPolicy {
        self.orientation
    }

    /// Selects presentation or codestream coordinates for every output frame.
    #[must_use]
    pub const fn with_orientation_policy(mut self, orientation: OrientationPolicy) -> Self {
        self.orientation = orientation;
        self
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeModularFormat {
    pub channels: ModularChannels,
    pub bits_per_sample: u8,
    pub storage_bits: u8,
}

/// Creates the canonical unsigned Gray/RGB/RGBA delivery layout for 1–31 valid bits.
/// Samples occupy native-endian 8-, 16-, or 32-bit words with zero high padding bits.
/// Unfiltered original-color Modular planes retain their exact codes; XYB reconstruction, rendering or composition uses
/// F32 working values and quantizes at presentation. Integer alpha with an independent depth
/// is rescaled to the requested color depth.
pub fn native_modular_pixel_format(
    channels: ModularChannels,
    bits_per_sample: u8,
) -> Result<PixelFormat> {
    if !(1..=31).contains(&bits_per_sample) {
        return Err(Error::UnsupportedOutputFormat(
            "native lossless-Modular depth must be in 1..=31".into(),
        ));
    }
    let storage_bits = native_integer_storage_bits(bits_per_sample);
    let (model, color_spec, swizzle, components): (_, _, _, &[Channel]) = match channels {
        ModularChannels::Gray => (
            ColorModel::NonColor,
            ColorSpecification::Undefined,
            Swizzle::X000,
            &[Channel::X],
        ),
        ModularChannels::Rgb => (
            ColorModel::Rgb,
            ColorSpecification::Default,
            Swizzle::XYZ1,
            &[Channel::X, Channel::Y, Channel::Z],
        ),
        ModularChannels::Rgba => (
            ColorModel::Rgb,
            ColorSpecification::Default,
            Swizzle::XYZW,
            &[Channel::X, Channel::Y, Channel::Z, Channel::W],
        ),
    };
    let words = components
        .iter()
        .copied()
        .map(|channel| {
            let mut fields = Vec::with_capacity(2);
            if bits_per_sample < storage_bits {
                fields.push(PackingField::padding(storage_bits - bits_per_sample));
            }
            fields.push(PackingField::channel(channel, bits_per_sample));
            PackingWord { fields }
        })
        .collect();
    Ok(PixelFormat {
        model,
        color_spec,
        chroma_subsampling: ChromaSubsampling::None,
        sample_kind: SampleKind::Unsigned,
        byte_order: ByteOrder::Native,
        swizzle,
        planes: vec![PlaneFormat {
            sampling: PlaneSampling::FULL,
            pixels_per_element: 1,
            words,
        }],
    })
}

const fn native_integer_storage_bits(bits: u8) -> u8 {
    match bits {
        0..=8 => 8,
        9..=16 => 16,
        _ => 32,
    }
}

/// Recognizes the canonical valid-bit-padded integer delivery descriptor.
pub(crate) fn native_modular_format(format: &PixelFormat) -> Option<NativeModularFormat> {
    if format.validate().is_err()
        || format.sample_kind != SampleKind::Unsigned
        || format.byte_order != ByteOrder::Native
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() != 1
    {
        return None;
    }
    let native_rgb_color = matches!(
        format.color_spec,
        ColorSpecification::Default | ColorSpecification::Undefined
    ) || matches!(format.color_spec, ColorSpecification::Defined(spec)
            if spec.space == jxl_gpu_formats::ColorSpace::Bt709
                && spec.encoding == jxl_gpu_formats::YcbcrEncoding::Undefined
                && spec.transfer == jxl_gpu_formats::TransferFunction::Srgb
                && spec.range == jxl_gpu_formats::ColorRange::Full);
    let channels = match (format.model, format.swizzle, format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined) => {
            ModularChannels::Gray
        }
        (ColorModel::Rgb, Swizzle::XYZ1, _) if native_rgb_color => ModularChannels::Rgb,
        (ColorModel::Rgb, Swizzle::XYZW, _) if native_rgb_color => ModularChannels::Rgba,
        _ => return None,
    };
    let plane = &format.planes[0];
    if plane.sampling != PlaneSampling::FULL
        || plane.pixels_per_element != 1
        || plane.words.len() != channels.count() as usize
    {
        return None;
    }
    let expected_channels = [Channel::X, Channel::Y, Channel::Z, Channel::W];
    let mut bits_per_sample = None;
    let mut storage_bits = None;
    for (word, expected_channel) in plane
        .words
        .iter()
        .zip(&expected_channels[..plane.words.len()])
    {
        let (padding, bits, channel) = match word.fields.as_slice() {
            [sample] => match sample.kind {
                PackingFieldKind::Channel(channel) => (0, sample.bits, channel),
                PackingFieldKind::Padding => return None,
            },
            [padding, sample] => match (padding.kind, sample.kind) {
                (PackingFieldKind::Padding, PackingFieldKind::Channel(channel)) => {
                    (padding.bits, sample.bits, channel)
                }
                _ => return None,
            },
            _ => return None,
        };
        let word_bits = padding.checked_add(bits)?;
        let expected_storage_bits = native_integer_storage_bits(bits);
        if channel != *expected_channel
            || !(1..=31).contains(&bits)
            || word_bits != expected_storage_bits
            || bits_per_sample.is_some_and(|value| value != bits)
            || storage_bits.is_some_and(|value| value != word_bits)
        {
            return None;
        }
        bits_per_sample = Some(bits);
        storage_bits = Some(word_bits);
    }
    Some(NativeModularFormat {
        channels,
        bits_per_sample: bits_per_sample?,
        storage_bits: storage_bits?,
    })
}
