//! Scalar reference conversion used by tests and correctness oracles.

use jxl_gpu_protocol::Extent2d;
use thiserror::Error;

use crate::{
    ByteOrder, Channel, ChromaLocation, ColorModel, ColorRange, ColorSpecification, ImageLayout,
    LayoutError, PackingFieldKind, PixelFormat, PlaneFormat, SampleKind, SwizzleComponent,
    YcbcrEncoding,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvertedImage {
    pub layout: ImageLayout,
    pub bytes: Vec<u8>,
}

/// Converts normalized nonlinear R'G'B' planes into a packed pitch-linear
/// allocation. It is deliberately scalar and intended as an oracle, not as a
/// production codec fallback.
pub fn convert_rgb_f32(
    rgb: [&[f32]; 3],
    extent: Extent2d,
    format: &PixelFormat,
) -> Result<ConvertedImage, ConversionError> {
    let area = extent.area().ok_or(ConversionError::SizeOverflow)?;
    for (channel, samples) in rgb.iter().enumerate() {
        if samples.len() < area {
            return Err(ConversionError::ShortInput {
                channel,
                expected: area,
                actual: samples.len(),
            });
        }
        if let Some(index) = samples[..area].iter().position(|value| !value.is_finite()) {
            return Err(ConversionError::NonFiniteInput { channel, index });
        }
    }
    if format.sample_kind != SampleKind::Unsigned {
        return Err(ConversionError::UnsupportedSampleKind(format.sample_kind));
    }
    if !matches!(format.model, ColorModel::Ycbcr | ColorModel::Rgb) {
        return Err(ConversionError::UnsupportedColorModel(format.model));
    }

    let layout = ImageLayout::packed(extent, format.clone())?;
    let byte_len =
        usize::try_from(layout.logical_size).map_err(|_| ConversionError::SizeOverflow)?;
    let mut bytes = vec![0_u8; byte_len];
    let color = match format.model {
        ColorModel::Ycbcr => Some(YcbcrTransform::from_format(format)?),
        ColorModel::Rgb => None,
        _ => unreachable!("model checked above"),
    };

    for (plane_index, (plane_format, plane_layout)) in
        format.planes.iter().zip(&layout.planes).enumerate()
    {
        let word_bytes: Vec<_> = plane_format
            .words
            .iter()
            .map(|word| {
                let bits = word.bits();
                if bits % 8 != 0 {
                    return Err(ConversionError::UnsupportedPacking(
                        "scalar conversion requires byte-aligned packing words",
                    ));
                }
                Ok(u64::from(bits / 8))
            })
            .collect::<Result<_, _>>()?;
        let bytes_per_element = word_bytes.iter().try_fold(0_u64, |sum, value| {
            sum.checked_add(*value).ok_or(ConversionError::SizeOverflow)
        })?;

        for sample_y in 0..plane_layout.sample_extent.height {
            let row_start = plane_layout
                .offset
                .checked_add(
                    u64::from(sample_y)
                        .checked_mul(plane_layout.row_stride)
                        .ok_or(ConversionError::SizeOverflow)?,
                )
                .ok_or(ConversionError::SizeOverflow)?;
            let element_count = plane_layout
                .sample_extent
                .width
                .div_ceil(u32::from(plane_format.pixels_per_element));
            for element_x in 0..element_count {
                let element_start = row_start
                    .checked_add(
                        u64::from(element_x)
                            .checked_mul(bytes_per_element)
                            .ok_or(ConversionError::SizeOverflow)?,
                    )
                    .ok_or(ConversionError::SizeOverflow)?;
                let mut word_offset = element_start;
                let mut luma_occurrence = 0_u32;
                for (word, word_len) in plane_format.words.iter().zip(&word_bytes) {
                    let mut value = 0_u64;
                    for field in &word.fields {
                        value = value.checked_shl(u32::from(field.bits)).ok_or(
                            ConversionError::UnsupportedPacking("packing word exceeds 64 bits"),
                        )?;
                        if let PackingFieldKind::Channel(channel) = field.kind {
                            let sample = sample_for_channel(
                                rgb,
                                extent,
                                format,
                                plane_format,
                                plane_index,
                                element_x,
                                sample_y,
                                channel,
                                luma_occurrence,
                                color.as_ref(),
                            )?;
                            let quantized = quantize(
                                sample,
                                field.bits,
                                channel,
                                format.model,
                                color.as_ref(),
                            )?;
                            value |= quantized;
                            if channel == Channel::X
                                && format.model == ColorModel::Ycbcr
                                && plane_format.pixels_per_element > 1
                            {
                                luma_occurrence = luma_occurrence.saturating_add(1);
                            }
                        }
                    }
                    write_word(&mut bytes, word_offset, *word_len, value, format.byte_order)?;
                    word_offset = word_offset
                        .checked_add(*word_len)
                        .ok_or(ConversionError::SizeOverflow)?;
                }
            }
        }
    }

    Ok(ConvertedImage { layout, bytes })
}

#[allow(clippy::too_many_arguments)]
fn sample_for_channel(
    rgb: [&[f32]; 3],
    extent: Extent2d,
    format: &PixelFormat,
    plane: &PlaneFormat,
    _plane_index: usize,
    element_x: u32,
    sample_y: u32,
    channel: Channel,
    luma_occurrence: u32,
    color: Option<&YcbcrTransform>,
) -> Result<f32, ConversionError> {
    match format.model {
        ColorModel::Rgb => {
            let source_x = element_x
                .checked_mul(u32::from(plane.pixels_per_element))
                .ok_or(ConversionError::SizeOverflow)?
                .min(extent.width - 1);
            let source_y = sample_y.min(extent.height - 1);
            let canonical = stored_channel_source(format, channel)?;
            Ok(match canonical {
                0..=2 => rgb[canonical][pixel_index(extent, source_x, source_y)?],
                3 => 1.0,
                _ => unreachable!(),
            })
        }
        ColorModel::Ycbcr => {
            let transform = color.ok_or(ConversionError::UnsupportedColorSpecification)?;
            if channel == Channel::X {
                let base_x = element_x
                    .checked_mul(u32::from(plane.pixels_per_element))
                    .ok_or(ConversionError::SizeOverflow)?;
                let source_x = base_x.saturating_add(luma_occurrence).min(extent.width - 1);
                let source_y = sample_y
                    .saturating_mul(u32::from(plane.sampling.vertical_divisor))
                    .min(extent.height - 1);
                Ok(transform.rgb_to_ycbcr(rgb_at(rgb, extent, source_x, source_y)?)[0])
            } else {
                let (horizontal, vertical) = format.chroma_subsampling.chroma_divisors().ok_or(
                    ConversionError::UnsupportedPacking(
                        "chroma channel is present without chroma subsampling",
                    ),
                )?;
                let chroma = transform.chroma_at(
                    rgb,
                    extent,
                    element_x,
                    sample_y,
                    u32::from(horizontal),
                    u32::from(vertical),
                )?;
                Ok(match channel {
                    Channel::Y => chroma[0],
                    Channel::Z => chroma[1],
                    Channel::W => 1.0,
                    Channel::X => unreachable!(),
                })
            }
        }
        _ => Err(ConversionError::UnsupportedColorModel(format.model)),
    }
}

fn stored_channel_source(format: &PixelFormat, stored: Channel) -> Result<usize, ConversionError> {
    let component = match stored {
        Channel::X => SwizzleComponent::X,
        Channel::Y => SwizzleComponent::Y,
        Channel::Z => SwizzleComponent::Z,
        Channel::W => SwizzleComponent::W,
    };
    format
        .swizzle
        .0
        .iter()
        .position(|candidate| *candidate == component)
        .ok_or(ConversionError::UnsupportedPacking(
            "stored RGB channel is absent from the swizzle",
        ))
}

fn quantize(
    value: f32,
    bits: u8,
    channel: Channel,
    model: ColorModel,
    color: Option<&YcbcrTransform>,
) -> Result<u64, ConversionError> {
    if bits == 0 || bits > 16 {
        return Err(ConversionError::UnsupportedBitDepth(bits));
    }
    let maximum = ((1_u32 << bits) - 1) as f32;
    let code = if model == ColorModel::Ycbcr {
        let range = color
            .ok_or(ConversionError::UnsupportedColorSpecification)?
            .range;
        match range {
            ColorRange::Full => maximum * value,
            ColorRange::Limited if bits >= 8 => {
                let scale = (1_u32 << (bits - 8)) as f32;
                if channel == Channel::X {
                    scale * (16.0 + 219.0 * value)
                } else {
                    scale * (128.0 + 224.0 * (value - 0.5))
                }
            }
            ColorRange::Limited => return Err(ConversionError::UnsupportedBitDepth(bits)),
        }
    } else {
        maximum * value
    };
    Ok(code.clamp(0.0, maximum).round() as u64)
}

fn write_word(
    output: &mut [u8],
    offset: u64,
    word_bytes: u64,
    value: u64,
    byte_order: ByteOrder,
) -> Result<(), ConversionError> {
    let start = usize::try_from(offset).map_err(|_| ConversionError::SizeOverflow)?;
    let len = usize::try_from(word_bytes).map_err(|_| ConversionError::SizeOverflow)?;
    let end = start
        .checked_add(len)
        .ok_or(ConversionError::SizeOverflow)?;
    let destination = output
        .get_mut(start..end)
        .ok_or(ConversionError::SizeOverflow)?;
    let little_endian = match byte_order {
        ByteOrder::Little => true,
        ByteOrder::Big => false,
        ByteOrder::Native => cfg!(target_endian = "little"),
    };
    for (index, byte) in destination.iter_mut().enumerate() {
        let shift_index = if little_endian {
            index
        } else {
            len - 1 - index
        };
        *byte = (value >> (shift_index * 8)) as u8;
    }
    Ok(())
}

fn rgb_at(rgb: [&[f32]; 3], extent: Extent2d, x: u32, y: u32) -> Result<[f32; 3], ConversionError> {
    let index = pixel_index(extent, x, y)?;
    Ok([rgb[0][index], rgb[1][index], rgb[2][index]])
}

fn pixel_index(extent: Extent2d, x: u32, y: u32) -> Result<usize, ConversionError> {
    let index = u64::from(y)
        .checked_mul(u64::from(extent.width))
        .and_then(|index| index.checked_add(u64::from(x)))
        .ok_or(ConversionError::SizeOverflow)?;
    usize::try_from(index).map_err(|_| ConversionError::SizeOverflow)
}

struct YcbcrTransform {
    kr: f32,
    kb: f32,
    range: ColorRange,
    horizontal_location: ChromaLocation,
    vertical_location: ChromaLocation,
}

impl YcbcrTransform {
    fn from_format(format: &PixelFormat) -> Result<Self, ConversionError> {
        let ColorSpecification::Defined(spec) = format.color_spec else {
            return Err(ConversionError::UnsupportedColorSpecification);
        };
        let (kr, kb) = match spec.encoding {
            YcbcrEncoding::Bt601 => (0.299, 0.114),
            YcbcrEncoding::Bt709 => (0.2126, 0.0722),
            YcbcrEncoding::Bt2020 => (0.2627, 0.0593),
            encoding => return Err(ConversionError::UnsupportedEncoding(encoding)),
        };
        Ok(Self {
            kr,
            kb,
            range: spec.range,
            horizontal_location: spec.chroma_location.horizontal,
            vertical_location: spec.chroma_location.vertical,
        })
    }

    fn rgb_to_ycbcr(&self, rgb: [f32; 3]) -> [f32; 3] {
        let kg = 1.0 - self.kr - self.kb;
        let y = self.kr * rgb[0] + kg * rgb[1] + self.kb * rgb[2];
        let cb = (rgb[2] - y) / (2.0 * (1.0 - self.kb)) + 0.5;
        let cr = (rgb[0] - y) / (2.0 * (1.0 - self.kr)) + 0.5;
        [y, cb, cr]
    }

    fn chroma_at(
        &self,
        rgb: [&[f32]; 3],
        extent: Extent2d,
        chroma_x: u32,
        chroma_y: u32,
        horizontal: u32,
        vertical: u32,
    ) -> Result<[f32; 2], ConversionError> {
        let origin_x = chroma_x
            .checked_mul(horizontal)
            .ok_or(ConversionError::SizeOverflow)?;
        let origin_y = chroma_y
            .checked_mul(vertical)
            .ok_or(ConversionError::SizeOverflow)?;
        let x_range = sampling_offsets(self.horizontal_location, horizontal);
        let y_range = sampling_offsets(self.vertical_location, vertical);
        let mut sum = [0.0_f32; 2];
        let mut count = 0_u32;
        for dy in &y_range {
            for dx in &x_range {
                let x = origin_x.saturating_add(*dx);
                let y = origin_y.saturating_add(*dy);
                if x < extent.width && y < extent.height {
                    let ycbcr = self.rgb_to_ycbcr(rgb_at(rgb, extent, x, y)?);
                    sum[0] += ycbcr[1];
                    sum[1] += ycbcr[2];
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Err(ConversionError::SizeOverflow);
        }
        Ok([sum[0] / count as f32, sum[1] / count as f32])
    }
}

fn sampling_offsets(location: ChromaLocation, divisor: u32) -> Vec<u32> {
    match location {
        ChromaLocation::Center => (0..divisor).collect(),
        ChromaLocation::Odd => vec![divisor.saturating_sub(1)],
        ChromaLocation::Even | ChromaLocation::Both => vec![0],
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ConversionError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("RGB input channel {channel} has {actual} samples; expected at least {expected}")]
    ShortInput {
        channel: usize,
        expected: usize,
        actual: usize,
    },
    #[error("RGB input channel {channel} sample {index} is not finite")]
    NonFiniteInput { channel: usize, index: usize },
    #[error("conversion size overflow")]
    SizeOverflow,
    #[error("scalar conversion does not support color model {0:?}")]
    UnsupportedColorModel(ColorModel),
    #[error("scalar conversion does not support sample kind {0:?}")]
    UnsupportedSampleKind(SampleKind),
    #[error("scalar YCbCr conversion requires a defined color specification")]
    UnsupportedColorSpecification,
    #[error("scalar conversion does not support YCbCr encoding {0:?}")]
    UnsupportedEncoding(YcbcrEncoding),
    #[error("scalar conversion does not support {0}-bit channels")]
    UnsupportedBitDepth(u8),
    #[error("scalar conversion does not support this packing: {0}")]
    UnsupportedPacking(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaLocation2d, ChromaOrder, ChromaSubsampling, ColorSpec, PixelFormat};

    fn spec(encoding: YcbcrEncoding, range: ColorRange) -> ColorSpecification {
        let mut spec = match encoding {
            YcbcrEncoding::Bt601 => ColorSpec::bt601(range, ChromaLocation2d::CENTER),
            YcbcrEncoding::Bt709 => ColorSpec::bt709(range, ChromaLocation2d::CENTER),
            YcbcrEncoding::Bt2020 => ColorSpec::bt2020_ncl(range, ChromaLocation2d::CENTER),
            _ => unreachable!(),
        };
        spec.chroma_location = ChromaLocation2d::CENTER;
        ColorSpecification::Defined(spec)
    }

    #[test]
    fn grey_maps_to_neutral_limited_nv12() {
        let grey = [0.5; 15];
        let converted = convert_rgb_f32(
            [&grey, &grey, &grey],
            Extent2d::new(5, 3),
            &PixelFormat::nv12(spec(YcbcrEncoding::Bt709, ColorRange::Limited)),
        )
        .unwrap();
        let y = converted.layout.plane(0).unwrap();
        let uv = converted.layout.plane(1).unwrap();
        assert_eq!(converted.bytes[y.offset as usize], 126);
        assert_eq!(converted.bytes[uv.offset as usize], 128);
        assert_eq!(converted.bytes[uv.offset as usize + 1], 128);
    }

    #[test]
    fn nv21_reverses_the_interleaved_chroma_words() {
        let red = [1.0; 4];
        let zero = [0.0; 4];
        let color_spec = spec(YcbcrEncoding::Bt601, ColorRange::Full);
        let nv12 = convert_rgb_f32(
            [&red, &zero, &zero],
            Extent2d::new(2, 2),
            &PixelFormat::nv12(color_spec),
        )
        .unwrap();
        let nv21 = convert_rgb_f32(
            [&red, &zero, &zero],
            Extent2d::new(2, 2),
            &PixelFormat::nv21(color_spec),
        )
        .unwrap();
        let a = nv12.layout.planes[1].offset as usize;
        let b = nv21.layout.planes[1].offset as usize;
        assert_eq!(nv12.bytes[a], nv21.bytes[b + 1]);
        assert_eq!(nv12.bytes[a + 1], nv21.bytes[b]);
    }

    #[test]
    fn p010_writes_codes_in_the_high_ten_bits() {
        let black = [0.0; 4];
        let converted = convert_rgb_f32(
            [&black, &black, &black],
            Extent2d::new(2, 2),
            &PixelFormat::p010(spec(YcbcrEncoding::Bt709, ColorRange::Limited)),
        )
        .unwrap();
        let y = converted.layout.planes[0].offset as usize;
        let word = u16::from_ne_bytes([converted.bytes[y], converted.bytes[y + 1]]);
        assert_eq!(word, 64 << 6);
    }

    #[test]
    fn common_planar_shapes_are_scalar_convertible() {
        let r = [
            0.0, 1.0, 0.5, 0.25, 0.75, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 1.0, 0.0,
        ];
        let g = [0.5; 15];
        let b = [1.0; 15];
        for subsampling in [
            ChromaSubsampling::Cs444,
            ChromaSubsampling::Cs422,
            ChromaSubsampling::Cs420,
        ] {
            let format = PixelFormat::yuv_planar(
                subsampling,
                8,
                8,
                spec(YcbcrEncoding::Bt2020, ColorRange::Full),
            )
            .unwrap();
            let converted = convert_rgb_f32([&r, &g, &b], Extent2d::new(5, 3), &format).unwrap();
            assert_eq!(converted.bytes.len() as u64, converted.layout.logical_size);
        }

        let semi = PixelFormat::yuv_semiplanar(
            ChromaSubsampling::Cs444,
            8,
            8,
            ChromaOrder::CbCr,
            spec(YcbcrEncoding::Bt709, ColorRange::Full),
        )
        .unwrap();
        convert_rgb_f32([&r, &g, &b], Extent2d::new(5, 3), &semi).unwrap();
    }
}
