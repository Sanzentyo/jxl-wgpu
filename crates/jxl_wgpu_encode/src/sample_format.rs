//! Checked source precision and the shared JPEG XL bit-depth writer.

use jxl_gpu_bitstream::{BitWriter, SampleBitDepth};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::EncodeError;

/// Stream-wide precision and canonical storage for integer sRGB/D65 components.
///
/// Each component occupies a native-endian word: one byte for 1–8 bits, two for
/// 9–16, and four for 17–31. Valid bits are right aligned; unused high bits are
/// ignored. RGB components are interleaved, with independently checked row padding
/// and byte offsets. Floating-point sources are not supported by this contract.
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
    bits: u8,
}

impl RgbSampleFormat {
    pub const RGB8: Self = Self { bits: 8 };

    /// Checks the complete JPEG XL unsigned integer precision range.
    pub fn integer(bits: u8) -> Result<Self, EncodeError> {
        if !(1..=31).contains(&bits) {
            return Err(EncodeError::InvalidConfiguration(
                "RGB integer depth must be in 1..=31",
            ));
        }
        Ok(Self { bits })
    }

    #[must_use]
    pub const fn bits_per_sample(self) -> u8 {
        self.bits
    }

    #[must_use]
    pub const fn word_bytes(self) -> u8 {
        match self.bits {
            1..=8 => 1,
            9..=16 => 2,
            _ => 4,
        }
    }

    #[must_use]
    pub const fn bit_depth(self) -> SampleBitDepth {
        SampleBitDepth::Integer {
            bits_per_sample: self.bits as u32,
        }
    }

    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        PixelFormat {
            model: ColorModel::Rgb,
            color_spec: ColorSpecification::Default,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle: Swizzle::XYZ1,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words: [Channel::X, Channel::Y, Channel::Z]
                    .map(|channel| {
                        let mut fields = Vec::with_capacity(2);
                        let padding = 8 * self.word_bytes() - self.bits;
                        if padding != 0 {
                            fields.push(PackingField::padding(padding));
                        }
                        fields.push(PackingField::channel(channel, self.bits));
                        PackingWord { fields }
                    })
                    .into(),
            }],
        }
    }

    pub(crate) const fn sample_mask(self) -> u32 {
        (1u32 << self.bits) - 1
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
