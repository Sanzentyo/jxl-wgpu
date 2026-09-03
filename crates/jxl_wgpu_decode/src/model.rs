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
    /// Lossless Modular data reconstructed by a GPU entropy/MA pipeline.
    ModularLossless {
        bits_per_sample: u8,
        channels: ModularChannels,
        prediction: ModularPredictionProfile,
        grouping: ModularGrouping,
        /// Progressive pass count declared by the frame (`1..=3` for the negotiated profile).
        passes: u32,
    },
    /// Standard XYB VarDCT decoded into a GPU-resident presentation buffer. Transform strategy is
    /// selected independently for every first block and remains GPU-resident.
    VarDct { bits_per_sample: u8 },
}

/// Logical channels reconstructed by a lossless Modular profile.
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

/// Numeric interpretation applied while writing a decoded non-color Modular sample.
///
/// This mapping is explicit because a [`PixelFormat`] with `ColorModel::NonColor` carries storage
/// shape, not normalization semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericSampleMapping {
    /// Preserve the decoded unsigned integer code exactly in the low valid bits of the canonical
    /// lossless-Modular Gray `u8`/`u16` storage descriptor. The requested valid depth and the
    /// codestream depth must match.
    NativeUnsigned,
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

/// Semantic side of a [`GpuOutputItem`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuOutputMapping {
    /// The format's explicit color specification determines conversion and packing.
    Color,
    /// A non-color numeric image uses the supplied, explicit sample mapping.
    Numeric(NumericSampleMapping),
    /// An extra channel (e.g. alpha, depth, spot color) identified by 0-based index.
    ExtraChannel {
        index: u32,
    },
}

/// One targeted output buffer description within a [`GpuOutputRequest`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOutputItem {
    pub format: PixelFormat,
    pub mapping: GpuOutputMapping,
}

impl GpuOutputItem {
    /// Creates a color-bearing output target.
    pub fn color(format: PixelFormat) -> Result<Self> {
        if native_modular_format(&format).is_some_and(|native| {
            matches!(
                native.channels,
                ModularChannels::Rgb | ModularChannels::Rgba
            )
        }) {
            return Ok(Self {
                format,
                mapping: GpuOutputMapping::Color,
            });
        }
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Color(_) => Ok(Self {
                format,
                mapping: GpuOutputMapping::Color,
            }),
            PixelFormatClass::Numeric(_) => Err(Error::NumericMappingRequired),
        }
    }

    /// Creates a numeric sample output target.
    pub fn numeric(format: PixelFormat, mapping: NumericSampleMapping) -> Result<Self> {
        if mapping == NumericSampleMapping::NativeUnsigned {
            return match native_modular_format(&format) {
                Some(NativeModularFormat {
                    channels: ModularChannels::Gray,
                    ..
                }) => Ok(Self {
                    format,
                    mapping: GpuOutputMapping::Numeric(mapping),
                }),
                Some(_) => Err(Error::NumericMappingForColorOutput),
                None => Err(Error::UnsupportedOutputFormat(
                    "native lossless-Modular output requires the canonical unsigned Gray descriptor"
                        .into(),
                )),
            };
        }
        match classify_pixel_format(&format)
            .map_err(|error| Error::UnsupportedOutputFormat(format!("{format:?}: {error}")))?
        {
            PixelFormatClass::Numeric(numeric) => {
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
                Ok(Self {
                    format,
                    mapping: GpuOutputMapping::Numeric(mapping),
                })
            }
            PixelFormatClass::Color(_) => Err(Error::NumericMappingForColorOutput),
        }
    }

    /// Creates an extra channel output target.
    pub fn extra_channel(format: PixelFormat, index: u32) -> Result<Self> {
        Ok(Self {
            format,
            mapping: GpuOutputMapping::ExtraChannel { index },
        })
    }
}

/// Generic GPU output request supporting one or more simultaneous outputs.
///
/// Supports color planes, numeric images, and extra channels simultaneously.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOutputRequest {
    outputs: Vec<GpuOutputItem>,
    max_frame_slots: NonZeroUsize,
}

impl GpuOutputRequest {
    /// Creates a single color output request.
    pub fn color(format: PixelFormat) -> Result<Self> {
        let item = GpuOutputItem::color(format)?;
        Ok(Self::from_items(vec![item]))
    }

    /// Creates a single numeric output request.
    pub fn numeric(format: PixelFormat, mapping: NumericSampleMapping) -> Result<Self> {
        let item = GpuOutputItem::numeric(format, mapping)?;
        Ok(Self::from_items(vec![item]))
    }

    /// Creates a single extra channel output request.
    pub fn extra_channel(format: PixelFormat, index: u32) -> Result<Self> {
        let item = GpuOutputItem::extra_channel(format, index)?;
        Ok(Self::from_items(vec![item]))
    }

    /// Creates a request with multiple output targets.
    pub fn multi(outputs: Vec<GpuOutputItem>) -> Result<Self> {
        if outputs.is_empty() {
            return Err(Error::UnsupportedOutputFormat(
                "at least one output target is required".into(),
            ));
        }
        Ok(Self::from_items(outputs))
    }

    fn from_items(outputs: Vec<GpuOutputItem>) -> Self {
        Self {
            outputs,
            max_frame_slots: NonZeroUsize::new(2).expect("two is nonzero"),
        }
    }

    /// Appends an additional output target to this request.
    #[must_use]
    pub fn with_output(mut self, item: GpuOutputItem) -> Self {
        self.outputs.push(item);
        self
    }

    /// All configured output targets.
    #[must_use]
    pub fn outputs(&self) -> &[GpuOutputItem] {
        &self.outputs
    }

    /// The primary (first) output target.
    #[must_use]
    pub fn primary_output(&self) -> &GpuOutputItem {
        &self.outputs[0]
    }

    /// Canonical format of the primary output (for backwards compatibility).
    #[must_use]
    pub fn format(&self) -> &PixelFormat {
        &self.outputs[0].format
    }

    /// Mapping of the primary output (for backwards compatibility).
    #[must_use]
    pub fn mapping(&self) -> GpuOutputMapping {
        self.outputs[0].mapping
    }

    /// Number of output targets in this request.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
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

pub(crate) fn native_modular_pixel_format(
    channels: ModularChannels,
    bits_per_sample: u8,
) -> Result<PixelFormat> {
    if !(1..=16).contains(&bits_per_sample) {
        return Err(Error::UnsupportedOutputFormat(
            "native lossless-Modular depth must be in 1..=16".into(),
        ));
    }
    let storage_bits = if bits_per_sample <= 8 { 8 } else { 16 };
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

/// Recognizes the exact pitch-linear descriptor shared with the GPU lossless Modular encoder.
pub(crate) fn native_modular_format(format: &PixelFormat) -> Option<NativeModularFormat> {
    if format.validate().is_err()
        || format.sample_kind != SampleKind::Unsigned
        || format.byte_order != ByteOrder::Native
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() != 1
    {
        return None;
    }
    let channels = match (format.model, format.swizzle, format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined) => {
            ModularChannels::Gray
        }
        (
            ColorModel::Rgb,
            Swizzle::XYZ1,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => ModularChannels::Rgb,
        (
            ColorModel::Rgb,
            Swizzle::XYZW,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => ModularChannels::Rgba,
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
        let expected_storage_bits = if bits <= 8 { 8 } else { 16 };
        if channel != *expected_channel
            || !(1..=16).contains(&bits)
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

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_formats::{
        ChromaLocation2d, ColorRange, ColorSpace, ColorSpec, ColorSpecification, RgbChannelOrder,
        TransferFunction, YcbcrEncoding,
    };

    fn test_rgb8() -> PixelFormat {
        PixelFormat::rgb8(
            RgbChannelOrder::Rgb,
            false,
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Undefined,
                transfer: TransferFunction::Srgb,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d::BOTH,
            }),
        )
    }

    fn test_gray8() -> PixelFormat {
        PixelFormat::non_color(
            jxl_gpu_formats::SampleKind::Unsigned,
            8,
            &[jxl_gpu_formats::Channel::X],
        )
    }

    #[test]
    fn single_output_request_backwards_compatible() {
        let rgb = test_rgb8();
        let req = GpuOutputRequest::color(rgb.clone()).expect("color request");
        assert_eq!(req.output_count(), 1);
        assert_eq!(req.format(), &rgb);
        assert_eq!(req.mapping(), GpuOutputMapping::Color);
        assert_eq!(req.primary_output().format, rgb);
        assert_eq!(req.primary_output().mapping, GpuOutputMapping::Color);

        let gray = test_gray8();
        let num_req = GpuOutputRequest::numeric(gray.clone(), NumericSampleMapping::NormalizedGray8)
            .expect("numeric request");
        assert_eq!(num_req.output_count(), 1);
        assert_eq!(num_req.format(), &gray);
        assert_eq!(
            num_req.mapping(),
            GpuOutputMapping::Numeric(NumericSampleMapping::NormalizedGray8)
        );
    }

    #[test]
    fn multi_output_request_construction_and_access() {
        let rgb = test_rgb8();
        let gray = test_gray8();

        let item0 = GpuOutputItem::color(rgb.clone()).expect("color item");
        let item1 = GpuOutputItem::extra_channel(gray.clone(), 0).expect("extra channel item");

        let multi = GpuOutputRequest::multi(vec![item0.clone(), item1.clone()]).expect("multi request");
        assert_eq!(multi.output_count(), 2);
        assert_eq!(multi.primary_output(), &item0);
        assert_eq!(multi.format(), &rgb);
        assert_eq!(multi.mapping(), GpuOutputMapping::Color);
        assert_eq!(&multi.outputs()[0], &item0);
        assert_eq!(&multi.outputs()[1], &item1);

        // Test with_output chaining
        let item2 = GpuOutputItem::extra_channel(gray.clone(), 1).expect("second extra channel");
        let chained = multi.with_output(item2.clone());
        assert_eq!(chained.output_count(), 3);
        assert_eq!(&chained.outputs()[2], &item2);
    }

    #[test]
    fn multi_output_request_rejects_empty() {
        let empty_result = GpuOutputRequest::multi(Vec::new());
        assert!(empty_result.is_err());
    }
}
