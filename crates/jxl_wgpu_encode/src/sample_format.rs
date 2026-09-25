//! Checked source precision and the shared JPEG XL bit-depth writer.

use jxl_gpu_bitstream::{BitWriter, SampleBitDepth};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, FloatPrecision,
    PackingField, PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::EncodeError;

/// Stream-wide precision for sRGB/D65 components, independent of physical source layout.
///
/// Integers support 1–31 bits; floating samples support all checked [`FloatPrecision`]
/// combinations. Sources may be packed, planar or split, with bijective RGB swizzles,
/// shared or separate 8/16/24/32-bit words, declared byte order and sample bit positions.
/// Every plane's extent, byte offset, row pitch and bounded GPU binding is checked.
/// [`Self::pixel_format`] constructs canonical native-endian interleaved RGB with one
/// 1/2/4-byte word per component and valid bits right aligned. Other valid layouts retain
/// the same logical precision. VarDCT converts finite floating samples on GPU and rejects
/// NaN/infinity before publishing a frame; Modular preserves raw words.
///
/// ```rust
/// use jxl_wgpu_encode::{RgbSampleFormat, VarDctConfig};
/// let config = VarDctConfig {
///     sample_format: RgbSampleFormat::integer(12)?,
///     ..Default::default()
/// };
/// config.sample_format.pixel_format().validate()?;
/// assert_eq!(config.sample_format.word_bytes(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbSampleFormat {
    precision: RgbPrecision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RgbPrecision {
    Integer(u8),
    Float(FloatPrecision),
}

impl RgbSampleFormat {
    pub const RGB8: Self = Self {
        precision: RgbPrecision::Integer(8),
    };

    /// Checks the complete JPEG XL unsigned integer precision range.
    pub fn integer(bits: u8) -> Result<Self, EncodeError> {
        if !(1..=31).contains(&bits) {
            return Err(EncodeError::InvalidConfiguration(
                "RGB integer depth must be in 1..=31",
            ));
        }
        Ok(Self {
            precision: RgbPrecision::Integer(bits),
        })
    }

    /// Checks all JPEG XL floating precisions (2–8 exponent and 2–23 fraction bits).
    pub fn float(bits: u8, exponent_bits: u8) -> Result<Self, EncodeError> {
        let precision = FloatPrecision::new(bits, exponent_bits)
            .map_err(|_| EncodeError::InvalidConfiguration("invalid RGB floating precision"))?;
        Ok(Self {
            precision: RgbPrecision::Float(precision),
        })
    }

    #[must_use]
    pub const fn float_precision(self) -> Option<FloatPrecision> {
        match self.precision {
            RgbPrecision::Float(p) => Some(p),
            RgbPrecision::Integer(_) => None,
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
            RgbPrecision::Integer(bits) => bits,
            RgbPrecision::Float(p) => p.bits(),
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
            RgbPrecision::Integer(bits) => SampleBitDepth::Integer {
                bits_per_sample: bits as u32,
            },
            RgbPrecision::Float(p) => SampleBitDepth::Float {
                bits_per_sample: p.bits() as u32,
                exponent_bits_per_sample: p.exponent_bits() as u32,
            },
        }
    }

    /// Construct canonical native-endian interleaved storage for this logical precision.
    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        PixelFormat {
            model: ColorModel::Rgb,
            color_spec: ColorSpecification::Default,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: match self.precision {
                RgbPrecision::Integer(_) => SampleKind::Unsigned,
                RgbPrecision::Float(p)
                    if p == FloatPrecision::BINARY16 || p == FloatPrecision::BINARY32 =>
                {
                    SampleKind::Float
                }
                RgbPrecision::Float(p) => SampleKind::CustomFloat(p),
            },
            byte_order: ByteOrder::Native,
            swizzle: Swizzle::XYZ1,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words: [Channel::X, Channel::Y, Channel::Z]
                    .map(|channel| {
                        let mut fields = Vec::with_capacity(2);
                        let padding = 8 * self.word_bytes() - self.bits_per_sample();
                        if padding != 0 {
                            fields.push(PackingField::padding(padding));
                        }
                        fields.push(PackingField::channel(channel, self.bits_per_sample()));
                        PackingWord { fields }
                    })
                    .into(),
            }],
        }
    }

    pub(crate) fn matches_format(self, format: &PixelFormat) -> bool {
        let Ok(spec) = crate::source::source_spec(format) else {
            return false;
        };
        format.model == ColorModel::Rgb
            && format.color_spec == ColorSpecification::Default
            && spec.format == crate::source::SourceChannels::Rgb
            && spec.bits_per_sample == self.bits_per_sample()
            && spec.exponent_bits_per_sample == self.exponent_bits()
    }

    pub(crate) const fn sample_mask(self) -> u32 {
        u32::MAX >> (32 - self.bits_per_sample())
    }
}

impl Default for RgbSampleFormat {
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
