//! Checked source precision and the shared JPEG XL bit-depth writer.

use jxl_gpu_bitstream::{BitWriter, SampleBitDepth};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, FloatPrecision,
    PackingField, PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::EncodeError;

/// Logical color channels, separate from physical packing and VarDCT's working planes.
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

    pub(crate) const fn source_channels(self) -> crate::source::SourceChannels {
        match self {
            Self::Gray => crate::source::SourceChannels::Gray,
            Self::Rgb => crate::source::SourceChannels::Rgb,
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
enum SamplePrecision {
    Integer(u8),
    Float(FloatPrecision),
}

impl ColorSampleFormat {
    pub const RGB8: Self = Self {
        channels: ColorChannels::Rgb,
        precision: SamplePrecision::Integer(8),
    };

    pub const GRAY8: Self = Self {
        channels: ColorChannels::Gray,
        precision: SamplePrecision::Integer(8),
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
            precision: SamplePrecision::Integer(bits),
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
            precision: SamplePrecision::Float(precision),
        })
    }

    #[must_use]
    pub const fn float_precision(self) -> Option<FloatPrecision> {
        match self.precision {
            SamplePrecision::Float(p) => Some(p),
            SamplePrecision::Integer(_) => None,
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
        match self.precision {
            SamplePrecision::Integer(bits) => bits,
            SamplePrecision::Float(p) => p.bits(),
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
        match self.precision {
            SamplePrecision::Integer(bits) => SampleBitDepth::Integer {
                bits_per_sample: bits as u32,
            },
            SamplePrecision::Float(p) => SampleBitDepth::Float {
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
            sample_kind: match self.precision {
                SamplePrecision::Integer(_) => SampleKind::Unsigned,
                SamplePrecision::Float(p)
                    if p == FloatPrecision::BINARY16 || p == FloatPrecision::BINARY32 =>
                {
                    SampleKind::Float
                }
                SamplePrecision::Float(p) => SampleKind::CustomFloat(p),
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

    pub(crate) fn matches_format(self, format: &PixelFormat) -> bool {
        let Ok(spec) = crate::source::source_spec(format) else {
            return false;
        };
        (format.model
            == match self.channels {
                ColorChannels::Gray => ColorModel::Gray,
                ColorChannels::Rgb => ColorModel::Rgb,
            }
            || matches!(
                (&format.model, &format.color_spec),
                (ColorModel::IccDevice, ColorSpecification::Icc(_))
            ))
            && spec.format == self.channels.source_channels()
            && spec.bits_per_sample == self.bits_per_sample()
            && spec.exponent_bits_per_sample == self.exponent_bits()
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
                assert!(format.matches_format(&pixel_format));
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
                assert!(!other.matches_format(&pixel_format));
            }
        }
    }
}
