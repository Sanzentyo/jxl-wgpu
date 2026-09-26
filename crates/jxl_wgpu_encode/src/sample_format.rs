//! Checked source precision and the shared JPEG XL bit-depth writer.

use jxl_gpu_bitstream::{BitWriter, SampleBitDepth};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, FloatPrecision,
    PackingField, PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::EncodeError;

/// Interpretation of source color relative to alpha, shared by both encoders.
/// No association conversion or invisible-color replacement is performed. Modular
/// preserves color words; VarDCT applies its selected lossy color coding normally.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AlphaAssociation {
    /// Color samples are independent of alpha.
    #[default]
    Unassociated,
    /// Color samples are already multiplied by alpha.
    Associated,
}

impl AlphaAssociation {
    pub(crate) fn validate(self, format: crate::source::SourceChannels) -> Result<(), EncodeError> {
        if self == Self::Associated && !format.has_alpha() {
            return Err(EncodeError::InvalidConfiguration(
                "associated source color requires an alpha channel",
            ));
        }
        Ok(())
    }
}

/// Logical image samples, independent of physical storage or three-plane VarDCT work.
/// Packed alpha shares color precision; independent scalar inputs have separate declarations.
/// This owns the component mapping and resolved image-header order for both frontends.
#[derive(Clone, Debug)]
pub(crate) struct ImageSamplePlan {
    pub(crate) color: ColorSampleFormat,
    pub(crate) alpha: Option<AlphaAssociation>,
    pub(crate) extra_channels: std::sync::Arc<[crate::ExtraChannel]>,
    pub(crate) cmyk: bool,
}

impl ImageSamplePlan {
    pub(crate) fn new(color: ColorSampleFormat, alpha: Option<AlphaAssociation>) -> Self {
        let extra_channels = alpha
            .into_iter()
            .map(|association| crate::ExtraChannel::packed_alpha(color.precision(), association))
            .collect();
        Self {
            color,
            alpha,
            extra_channels,
            cmyk: false,
        }
    }

    /// The embedded CMYK profile owns one full-resolution Black in the primary buffer.
    /// It follows packed alpha and precedes independently attached scalar inputs.
    pub(crate) fn with_cmyk(mut self, cmyk: bool) -> Result<Self, EncodeError> {
        if cmyk {
            if self.color.channels() != ColorChannels::Rgb {
                return Err(EncodeError::InvalidConfiguration(
                    "CMYK requires three coded color components",
                ));
            }
            let black = crate::ExtraChannel::new(
                crate::ExtraChannelKind::Black,
                self.color.precision(),
                0,
                Vec::new(),
            )?;
            self.extra_channels = self.extra_channels.iter().cloned().chain([black]).collect();
            self.cmyk = true;
        }
        Ok(self)
    }

    pub(crate) fn with_extra_channels(
        mut self,
        channels: &[crate::ExtraChannel],
        limit: u64,
    ) -> Result<Self, EncodeError> {
        if self.cmyk
            && channels
                .iter()
                .any(|channel| channel.kind() == crate::ExtraChannelKind::Black)
        {
            return Err(EncodeError::InvalidConfiguration(
                "CMYK has one primary Black channel",
            ));
        }
        if self.extra_channels.len() + channels.len() > crate::extra_channel::MAX_EXTRA_CHANNELS {
            return Err(EncodeError::InvalidConfiguration(
                "extra-channel count exceeds the JPEG XL profile limit",
            ));
        }
        let metadata_bytes = self
            .extra_channels
            .iter()
            .chain(channels)
            .try_fold(0u64, |total, channel| {
                total.checked_add(channel.name().len() as u64 + 32)
            })
            .ok_or(EncodeError::InvalidConfiguration(
                "extra-channel metadata size overflow",
            ))?;
        if metadata_bytes > limit {
            return Err(EncodeError::InvalidConfiguration(
                "extra-channel metadata exceeds its configured byte limit",
            ));
        }
        self.extra_channels = self
            .extra_channels
            .iter()
            .chain(channels)
            .cloned()
            .collect();
        Ok(self)
    }

    pub(crate) const fn channels(&self) -> crate::source::SourceChannels {
        use crate::source::SourceChannels;
        match (self.color.channels(), self.alpha) {
            (ColorChannels::Gray, None) => SourceChannels::Gray,
            (ColorChannels::Gray, Some(_)) => SourceChannels::GrayAlpha,
            (ColorChannels::Rgb, None) => SourceChannels::Rgb,
            (ColorChannels::Rgb, Some(_)) => SourceChannels::Rgba,
        }
    }

    pub(crate) fn alpha_component(&self) -> Option<usize> {
        self.alpha.map(|_| self.color.channels().count() as usize)
    }

    pub(crate) fn pixel_format(&self) -> PixelFormat {
        let mut format = self.color.pixel_format();
        if self.alpha.is_some() {
            let mut word = format.planes[0].words[0].clone();
            word.fields.last_mut().expect("canonical sample field").kind =
                jxl_gpu_formats::PackingFieldKind::Channel(Channel::W);
            format.planes[0].words.push(word);
            format.swizzle = match self.color.channels() {
                ColorChannels::Gray => Swizzle::X00W,
                ColorChannels::Rgb => Swizzle::XYZW,
            };
        }
        format
    }

    pub(crate) fn pixel_format_with_color(&self, color: ColorSpecification) -> PixelFormat {
        let mut format = self.pixel_format();
        if matches!(&color, ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
        {
            format.model = ColorModel::IccDevice;
            format.swizzle = Swizzle::Device;
            let words = &mut format.planes[0].words;
            for (index, word) in words.iter_mut().enumerate() {
                word.fields.last_mut().expect("canonical sample").kind =
                    jxl_gpu_formats::PackingFieldKind::Channel(if index == 3 {
                        Channel::Alpha
                    } else {
                        Channel::Device(index as u8)
                    });
            }
            let mut black = words[0].clone();
            black.fields.last_mut().expect("canonical sample").kind =
                jxl_gpu_formats::PackingFieldKind::Channel(Channel::Device(3));
            words.insert(3.min(words.len()), black);
        }
        format.color_spec = color;
        format
    }

    pub(crate) fn matches_format(&self, format: &PixelFormat) -> bool {
        let Ok(spec) = crate::source::source_spec(format) else {
            return false;
        };
        (format.model
            == match self.color.channels() {
                ColorChannels::Gray => ColorModel::Gray,
                ColorChannels::Rgb => ColorModel::Rgb,
            }
            || matches!(
                (&format.model, &format.color_spec),
                (ColorModel::IccDevice, ColorSpecification::Icc(_))
            ))
            && spec.format == self.channels()
            && spec.is_cmyk() == self.cmyk
            && spec.bits_per_sample == self.color.bits_per_sample()
            && spec.exponent_bits_per_sample == self.color.exponent_bits()
    }

    pub(crate) fn write_extra_channels(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        crate::extra_channel::write_count(output, self.extra_channels.len())?;
        for channel in self.extra_channels.iter() {
            channel.write(output)?;
        }
        Ok(())
    }
}

/// Coded color channels, separate from physical packing and VarDCT's working planes.
/// A CMYK ICC source uses `Rgb` for its three coded CMY channels and adds a primary Black extra.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorChannels {
    Gray,
    Rgb,
}

impl ColorChannels {
    #[must_use]
    pub const fn count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
        }
    }

    pub(crate) const fn working_components(self) -> [usize; 3] {
        match self {
            Self::Gray => [0; 3],
            Self::Rgb => [0, 1, 2],
        }
    }
}

/// Stream-wide Gray/RGB channels and precision, independent of color and physical source layout.
///
/// Integers support 1–31 bits; floating samples support all checked [`FloatPrecision`]
/// combinations. Gray selects one stored component; RGB may be packed, planar or split,
/// with bijective swizzles, shared or separate 8/16/24/32-bit words, declared byte order and sample bit positions.
/// Every plane's extent, byte offset, row pitch and bounded GPU binding is checked.
/// [`Self::pixel_format`] constructs canonical native-endian Gray or interleaved RGB with one
/// 1/2/4-byte word per component and valid bits right aligned. Other valid layouts retain
/// the same logical precision. VarDCT converts finite floating samples on GPU and rejects
/// NaN/infinity before publishing a frame; Modular preserves raw words.
///
/// ```rust
/// use jxl_wgpu_encode::{ColorChannels, ColorSampleFormat, VarDctConfig};
/// let config = VarDctConfig {
///     sample_format: ColorSampleFormat::integer(ColorChannels::Rgb, 12)?,
///     ..Default::default()
/// };
/// config.sample_format.pixel_format().validate()?;
/// assert_eq!(config.sample_format.word_bytes(), 2);
/// let gray = ColorSampleFormat::float(ColorChannels::Gray, 16, 5)?;
/// assert_eq!(gray.channels().count(), 1);
/// gray.pixel_format().validate()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorSampleFormat {
    channels: ColorChannels,
    precision: SamplePrecision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PrecisionKind {
    Integer(u8),
    Float(FloatPrecision),
}

/// Checked sample precision shared by color and independent extra channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplePrecision(PrecisionKind);

impl SamplePrecision {
    pub fn integer(bits: u8) -> Result<Self, EncodeError> {
        Ok(ColorSampleFormat::integer(ColorChannels::Gray, bits)?.precision)
    }

    pub fn float(bits: u8, exponent_bits: u8) -> Result<Self, EncodeError> {
        Ok(ColorSampleFormat::float(ColorChannels::Gray, bits, exponent_bits)?.precision)
    }

    #[must_use]
    pub const fn color(self, channels: ColorChannels) -> ColorSampleFormat {
        ColorSampleFormat {
            channels,
            precision: self,
        }
    }

    /// Canonical scalar storage; layout changes do not alter the logical precision.
    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        let mut format = self.color(ColorChannels::Gray).pixel_format();
        format.model = ColorModel::NonColor;
        format.color_spec = ColorSpecification::Undefined;
        format.swizzle = Swizzle::X000;
        format
    }

    #[must_use]
    pub const fn bit_depth(self) -> SampleBitDepth {
        self.color(ColorChannels::Gray).bit_depth()
    }

    pub(crate) const fn mask(self) -> u32 {
        self.color(ColorChannels::Gray).sample_mask()
    }
}

impl ColorSampleFormat {
    #[must_use]
    pub const fn precision(self) -> SamplePrecision {
        self.precision
    }

    pub const RGB8: Self = Self {
        channels: ColorChannels::Rgb,
        precision: SamplePrecision(PrecisionKind::Integer(8)),
    };

    pub const GRAY8: Self = Self {
        channels: ColorChannels::Gray,
        precision: SamplePrecision(PrecisionKind::Integer(8)),
    };

    #[must_use]
    pub const fn channels(self) -> ColorChannels {
        self.channels
    }

    /// Checks the complete JPEG XL unsigned integer precision range.
    pub fn integer(channels: ColorChannels, bits: u8) -> Result<Self, EncodeError> {
        if !(1..=31).contains(&bits) {
            return Err(EncodeError::InvalidConfiguration(
                "color integer depth must be in 1..=31",
            ));
        }
        Ok(Self {
            channels,
            precision: SamplePrecision(PrecisionKind::Integer(bits)),
        })
    }

    /// Checks all JPEG XL floating precisions (2–8 exponent and 2–23 fraction bits).
    pub fn float(
        channels: ColorChannels,
        bits: u8,
        exponent_bits: u8,
    ) -> Result<Self, EncodeError> {
        let precision = FloatPrecision::new(bits, exponent_bits)
            .map_err(|_| EncodeError::InvalidConfiguration("invalid color floating precision"))?;
        Ok(Self {
            channels,
            precision: SamplePrecision(PrecisionKind::Float(precision)),
        })
    }

    #[must_use]
    pub const fn float_precision(self) -> Option<FloatPrecision> {
        match self.precision.0 {
            PrecisionKind::Float(p) => Some(p),
            PrecisionKind::Integer(_) => None,
        }
    }

    #[must_use]
    pub const fn exponent_bits(self) -> u8 {
        match self.float_precision() {
            Some(p) => p.exponent_bits(),
            None => 0,
        }
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> u8 {
        match self.precision.0 {
            PrecisionKind::Integer(bits) => bits,
            PrecisionKind::Float(p) => p.bits(),
        }
    }

    /// Component word width in the canonical layout returned by `pixel_format`.
    #[must_use]
    pub const fn word_bytes(self) -> u8 {
        match self.bits_per_sample() {
            1..=8 => 1,
            9..=16 => 2,
            _ => 4,
        }
    }

    #[must_use]
    pub const fn bit_depth(self) -> SampleBitDepth {
        match self.precision.0 {
            PrecisionKind::Integer(bits) => SampleBitDepth::Integer {
                bits_per_sample: bits as u32,
            },
            PrecisionKind::Float(p) => SampleBitDepth::Float {
                bits_per_sample: p.bits() as u32,
                exponent_bits_per_sample: p.exponent_bits() as u32,
            },
        }
    }

    /// Construct canonical native-endian interleaved storage for this logical precision.
    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        PixelFormat {
            model: match self.channels {
                ColorChannels::Gray => ColorModel::Gray,
                ColorChannels::Rgb => ColorModel::Rgb,
            },
            color_spec: ColorSpecification::Default,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: match self.precision.0 {
                PrecisionKind::Integer(_) => SampleKind::Unsigned,
                PrecisionKind::Float(p)
                    if p == FloatPrecision::BINARY16 || p == FloatPrecision::BINARY32 =>
                {
                    SampleKind::Float
                }
                PrecisionKind::Float(p) => SampleKind::CustomFloat(p),
            },
            byte_order: ByteOrder::Native,
            swizzle: match self.channels {
                ColorChannels::Gray => Swizzle::Xyzw([
                    jxl_gpu_formats::SwizzleComponent::X,
                    jxl_gpu_formats::SwizzleComponent::Zero,
                    jxl_gpu_formats::SwizzleComponent::Zero,
                    jxl_gpu_formats::SwizzleComponent::One,
                ]),
                ColorChannels::Rgb => Swizzle::XYZ1,
            },
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words: [Channel::X, Channel::Y, Channel::Z]
                    .into_iter()
                    .take(self.channels.count() as usize)
                    .map(|channel| {
                        let mut fields = Vec::with_capacity(2);
                        let padding = 8 * self.word_bytes() - self.bits_per_sample();
                        if padding != 0 {
                            fields.push(PackingField::padding(padding));
                        }
                        fields.push(PackingField::channel(channel, self.bits_per_sample()));
                        PackingWord { fields }
                    })
                    .collect(),
            }],
        }
    }

    pub(crate) const fn sample_mask(self) -> u32 {
        u32::MAX >> (32 - self.bits_per_sample())
    }
}

impl Default for ColorSampleFormat {
    fn default() -> Self {
        Self::RGB8
    }
}

pub(crate) fn write_sample_bit_depth(
    output: &mut BitWriter,
    bits_per_sample: u8,
    exponent_bits_per_sample: u8,
) -> Result<(), EncodeError> {
    if exponent_bits_per_sample != 0 {
        jxl_gpu_formats::FloatPrecision::new(bits_per_sample, exponent_bits_per_sample)
            .map_err(|_| EncodeError::InvalidConfiguration("invalid JPEG XL floating precision"))?;
        output.write_bits(1, 1)?;
        match bits_per_sample {
            32 => output.write_bits(0, 2)?,
            16 => output.write_bits(1, 2)?,
            24 => output.write_bits(2, 2)?,
            bits => {
                output.write_bits(3, 2)?;
                output.write_bits(u64::from(bits - 1), 6)?;
            }
        }
        output.write_bits(u64::from(exponent_bits_per_sample - 1), 4)?;
        return Ok(());
    }
    if !(1..=31).contains(&bits_per_sample) {
        return Err(EncodeError::InvalidConfiguration(
            "JPEG XL integer depth must be in 1..=31",
        ));
    }
    output.write_bits(0, 1)?;
    match bits_per_sample {
        8 => output.write_bits(0, 2)?,
        10 => output.write_bits(1, 2)?,
        12 => output.write_bits(2, 2)?,
        bits => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(bits - 1), 6)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_channels_are_independent_of_precision_and_never_interchangeable() {
        for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
            for bits in 0..=u8::MAX {
                assert_eq!(
                    ColorSampleFormat::integer(channels, bits).is_ok(),
                    (1..=31).contains(&bits)
                );
            }
            let formats = (1..=31)
                .map(|bits| ColorSampleFormat::integer(channels, bits).unwrap())
                .chain((2..=8).flat_map(|exponent| {
                    (2..=23).map(move |fraction| {
                        ColorSampleFormat::float(channels, 1 + exponent + fraction, exponent)
                            .unwrap()
                    })
                }));
            for format in formats {
                let pixel_format = format.pixel_format();
                pixel_format.validate().unwrap();
                assert!(ImageSamplePlan::new(format, None).matches_format(&pixel_format));
                assert_eq!(
                    pixel_format.planes[0].words.len(),
                    channels.count() as usize
                );
                let other = ColorSampleFormat {
                    channels: match channels {
                        ColorChannels::Gray => ColorChannels::Rgb,
                        ColorChannels::Rgb => ColorChannels::Gray,
                    },
                    ..format
                };
                assert!(!ImageSamplePlan::new(other, None).matches_format(&pixel_format));
            }
        }
    }
}
