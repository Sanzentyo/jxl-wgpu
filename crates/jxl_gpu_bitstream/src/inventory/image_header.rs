//! One bounded image-metadata walk, retaining normative preview dimensions and extensions.
//!
//! The public primitive bundles still come from jxl-image. The outer grammar is owned here so
//! conditional preview widths and unknown extensions cannot be lost by an opaque bundle parser.

use jxl_bitstream::{Bitstream, U};
use jxl_image::{
    AnimationHeader, BitDepth, ExtraChannelInfo, ImageMetadata, SizeHeader,
    color::{ColourEncoding, OpsinInverseMatrix, ToneMapping},
};
use jxl_oxide_common::{Bundle, BundleDefault};

use super::InventoryError;

pub(super) struct HeaderFields {
    pub size: SizeHeader,
    /// Public sample/color fields only; preview dimensions are retained separately below.
    pub metadata: ImageMetadata,
    pub preview_size: Option<(u32, u32)>,
}

#[derive(Debug)]
pub(super) enum HeaderError {
    Syntax(jxl_bitstream::Error),
    Inventory(InventoryError),
}

impl From<jxl_bitstream::Error> for HeaderError {
    fn from(error: jxl_bitstream::Error) -> Self {
        Self::Syntax(error)
    }
}

pub(super) fn parse(reader: &mut Bitstream<'_>) -> Result<HeaderFields, HeaderError> {
    if reader.read_bits(16)? != 0x0aff {
        return Err(jxl_bitstream::Error::ValidationFailed("Invalid JPEG XL signature").into());
    }
    let size = SizeHeader::parse(reader, ())?;
    let mut metadata = ImageMetadata::default_with_context(());
    let mut preview_size = None;
    if !reader.read_bool()? {
        let extra_fields = reader.read_bool()?;
        if extra_fields {
            metadata.orientation = 1 + reader.read_bits(3)?;
            if reader.read_bool()? {
                metadata.intrinsic_size = Some(SizeHeader::parse(reader, ())?);
            }
            if reader.read_bool()? {
                preview_size = Some(parse_preview_size(reader)?);
            }
            if reader.read_bool()? {
                metadata.animation = Some(AnimationHeader::parse(reader, ())?);
            }
        }
        metadata.bit_depth = BitDepth::parse(reader, ())?;
        metadata.modular_16bit_buffers = reader.read_bool()?;
        let extra_channels = reader.read_u32(0, 1, 2 + U(4), 1 + U(12))?;
        if extra_channels > super::MAX_EXTRA_CHANNELS {
            return Err(HeaderError::Inventory(InventoryError::ResourceLimit(
                "extra channels",
            )));
        }
        metadata
            .ec_info
            .try_reserve_exact(extra_channels as usize)
            .map_err(|_| {
                HeaderError::Inventory(InventoryError::AllocationFailed("extra-channel metadata"))
            })?;
        for _ in 0..extra_channels {
            metadata.ec_info.push(ExtraChannelInfo::parse(reader, ())?);
        }
        metadata.xyb_encoded = reader.read_bool()?;
        metadata.colour_encoding = ColourEncoding::parse(reader, ())?;
        if extra_fields {
            metadata.tone_mapping = ToneMapping::parse(reader, ())?;
        }
        let selector = reader.read_u64()?;
        if selector != 0 {
            return Err(HeaderError::Inventory(
                InventoryError::UnsupportedExtensions {
                    scope: "image",
                    selector,
                },
            ));
        }
    }
    let default_transform = reader.read_bool()?;
    if !default_transform {
        if metadata.xyb_encoded {
            metadata.opsin_inverse_matrix = OpsinInverseMatrix::parse(reader, ())?;
        }
        let weights = reader.read_bits(3)?;
        for (mask, values) in [
            (1, metadata.up2_weight.as_mut_slice()),
            (2, metadata.up4_weight.as_mut_slice()),
            (4, metadata.up8_weight.as_mut_slice()),
        ] {
            if weights & mask != 0 {
                for value in values {
                    *value = reader.read_f16_as_f32()?;
                }
            }
        }
    }
    let tone = &metadata.tone_mapping;
    if tone.intensity_target <= 0.0
        || tone.min_nits < 0.0
        || tone.min_nits > tone.intensity_target
        || tone.linear_below < 0.0
        || (tone.relative_to_max_display && tone.linear_below > 1.0)
    {
        return Err(jxl_bitstream::Error::ValidationFailed("Invalid tone mapping").into());
    }
    Ok(HeaderFields {
        size,
        metadata,
        preview_size,
    })
}

fn parse_preview_size(reader: &mut Bitstream<'_>) -> Result<(u32, u32), HeaderError> {
    let div8 = reader.read_bool()?;
    let height = if div8 {
        8 * reader.read_u32(16, 32, 1 + U(5), 33 + U(9))?
    } else {
        reader.read_u32(1 + U(6), 65 + U(8), 321 + U(10), 1345 + U(12))?
    };
    let ratio = reader.read_bits(3)?;
    let width = if ratio != 0 {
        let (numerator, denominator) = [
            (1, 1),
            (1, 1),
            (12, 10),
            (4, 3),
            (3, 2),
            (16, 9),
            (5, 4),
            (2, 1),
        ][ratio as usize];
        height * numerator / denominator
    } else if div8 {
        8 * reader.read_u32(16, 32, 1 + U(5), 33 + U(9))?
    } else {
        reader.read_u32(1 + U(6), 65 + U(8), 321 + U(10), 1345 + U(12))?
    };
    if width > 4096 || height > 4096 {
        return Err(HeaderError::Inventory(
            InventoryError::InvalidPreviewDimensions { width, height },
        ));
    }
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{parse_extensions, parse_image_header};
    use crate::{BitReader, BitWriter, InventoryLimits};

    fn preview_dimension(bits: &mut BitWriter, size: u32, div8: bool) {
        if div8 {
            assert!(size.is_multiple_of(8));
            let value = size / 8;
            let (selector, base, width) = match value {
                16 => (0, 16, 0),
                32 => (1, 32, 0),
                1..=32 => (2, 1, 5),
                33..=544 => (3, 33, 9),
                _ => panic!("preview extent cannot be encoded"),
            };
            bits.write_bits(selector, 2).unwrap();
            bits.write_bits(u64::from(value - base), width).unwrap();
        } else {
            let (selector, base, width) = match size {
                1..=64 => (0, 1, 6),
                65..=320 => (1, 65, 8),
                321..=1344 => (2, 321, 10),
                1345..=5440 => (3, 1345, 12),
                _ => panic!("preview extent cannot be encoded"),
            };
            bits.write_bits(selector, 2).unwrap();
            bits.write_bits(u64::from(size - base), width).unwrap();
        }
    }

    fn preview_prefix(div8: bool, height: u32, ratio: u32, width: u32) -> BitWriter {
        let mut bits = BitWriter::new();
        bits.write_bits(u64::from(div8), 1).unwrap();
        preview_dimension(&mut bits, height, div8);
        bits.write_bits(u64::from(ratio), 3).unwrap();
        if ratio == 0 {
            preview_dimension(&mut bits, width, div8);
        }
        bits
    }

    #[test]
    fn preview_ratio_controls_width_presence_without_consuming_the_next_field() {
        for div8 in [false, true] {
            let height = if div8 { 256 } else { 27 };
            for (ratio, expected_width) in [
                (0, if div8 { 128 } else { 15 }),
                (1, height),
                (2, height * 12 / 10),
                (3, height * 4 / 3),
                (4, height * 3 / 2),
                (5, height * 16 / 9),
                (6, height * 5 / 4),
                (7, height * 2),
            ] {
                let mut bits = preview_prefix(div8, height, ratio, expected_width);
                let end = bits.bit_len();
                bits.write_bits(0x6d, 7).unwrap();
                let mut reader = Bitstream::new(bits.as_bytes());
                assert_eq!(
                    parse_preview_size(&mut reader).unwrap(),
                    (expected_width, height)
                );
                assert_eq!(reader.num_read_bits(), end);
                assert_eq!(reader.read_bits(7).unwrap(), 0x6d);
            }
        }
    }

    #[test]
    fn preview_dimensions_enforce_the_normative_limit_for_explicit_and_derived_widths() {
        for div8 in [false, true] {
            let bits = preview_prefix(div8, 4096, 0, 4096);
            assert_eq!(
                parse_preview_size(&mut Bitstream::new(bits.as_bytes())).unwrap(),
                (4096, 4096)
            );
            for (height, ratio, width) in [(4104, 0, 8), (8, 0, 4104), (4096, 7, 8192)] {
                let bits = preview_prefix(div8, height, ratio, width);
                assert!(
                    matches!(parse_preview_size(&mut Bitstream::new(bits.as_bytes())),
                    Err(HeaderError::Inventory(InventoryError::InvalidPreviewDimensions { width: w, height: h }))
                    if w == width && h == height)
                );
            }
        }
    }

    fn image_prefix(extension: bool) -> Vec<u8> {
        let mut bits = BitWriter::new();
        for (value, width) in [
            (0x0aff, 16), // signature
            (1, 1),
            (0, 5),
            (1, 3), // small 8x8 size, square aspect
            (0, 1),
            (0, 1), // explicit metadata, no extra fields
            (0, 1),
            (0, 2),
            (1, 1), // integer eight-bit samples and 16-bit buffers
            (0, 2),
            (1, 1),
            (1, 1), // no extra channels, XYB, default sRGB
        ] {
            bits.write_bits(value, width).unwrap();
        }
        if extension {
            bits.write_bits(1, 2).unwrap();
            bits.write_bits(0, 4).unwrap(); // U64 selector 1
            bits.write_bits(0, 2).unwrap(); // zero-length extension payload
        } else {
            bits.write_bits(0, 2).unwrap();
        }
        bits.write_bits(1, 1).unwrap(); // default transform data
        bits.into_bytes()
    }

    #[test]
    fn unknown_image_extensions_are_rejected_even_with_empty_payloads() {
        let known = image_prefix(false);
        let image = parse_image_header(&known, InventoryLimits::default()).unwrap();
        assert_eq!((image.inventory.width, image.inventory.height), (8, 8));
        let unknown = image_prefix(true);
        assert!(matches!(
            parse_image_header(&unknown, InventoryLimits::default()),
            Err(InventoryError::UnsupportedExtensions {
                scope: "image",
                selector: 1
            })
        ));
    }

    #[test]
    fn frame_and_restoration_extensions_validate_limits_before_rejection() {
        for scope in ["frame", "restoration"] {
            let mut bits = BitWriter::new();
            bits.write_bits(1, 2).unwrap();
            bits.write_bits(0, 4).unwrap(); // selector 1
            bits.write_bits(0, 2).unwrap(); // empty payload
            assert_eq!(
                parse_extensions(&mut BitReader::new(bits.as_bytes()), 16, scope),
                Err(InventoryError::UnsupportedExtensions { scope, selector: 1 })
            );
            let mut oversized = BitWriter::new();
            oversized.write_bits(1, 2).unwrap();
            oversized.write_bits(0, 4).unwrap();
            oversized.write_bits(2, 2).unwrap();
            oversized.write_bits(0, 8).unwrap(); // payload length 17
            assert_eq!(
                parse_extensions(&mut BitReader::new(oversized.as_bytes()), 16, scope),
                Err(InventoryError::ResourceLimit("extension payload bits"))
            );
            assert_eq!(
                parse_extensions(&mut BitReader::new(&[0]), 16, scope),
                Ok(())
            );
        }
    }
}
