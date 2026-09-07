//! Observes the extension selector hidden by jxl-image's opaque Extensions bundle.
//!
//! The caller first validates the complete header using jxl-image. This bounded metadata walk
//! reuses the same public field parsers up to the selector; it never decodes image-domain data.

use jxl_bitstream::{Bitstream, BitstreamResult, U};
use jxl_image::{
    AnimationHeader, BitDepth, ExtraChannelInfo, PreviewHeader, SizeHeader,
    color::{ColourEncoding, ToneMapping},
};
use jxl_oxide_common::Bundle;

use super::InventoryError;

pub(super) fn read_selector(bytes: &[u8]) -> Result<Option<(usize, u64)>, InventoryError> {
    read_prefix(&mut Bitstream::new(bytes))
        .map_err(|error| InventoryError::ImageHeader(error.to_string()))
}

fn read_prefix(reader: &mut Bitstream<'_>) -> BitstreamResult<Option<(usize, u64)>> {
    reader.read_bits(16)?;
    SizeHeader::parse(reader, ())?;
    let all_default = reader.read_bits(1)? != 0;
    if all_default {
        return Ok(None);
    }
    let extra_fields = reader.read_bits(1)? != 0;
    if extra_fields {
        reader.read_bits(3)?; // orientation
        if reader.read_bits(1)? != 0 {
            SizeHeader::parse(reader, ())?;
        }
        if reader.read_bits(1)? != 0 {
            PreviewHeader::parse(reader, ())?;
        }
        if reader.read_bits(1)? != 0 {
            AnimationHeader::parse(reader, ())?;
        }
    }
    BitDepth::parse(reader, ())?;
    reader.read_bits(1)?; // modular_16bit_buffers
    let extra_channels = reader.read_u32(0, 1, 2 + U(4), 1 + U(12))?;
    for _ in 0..extra_channels {
        ExtraChannelInfo::parse(reader, ())?;
    }
    reader.read_bits(1)?; // xyb_encoded
    ColourEncoding::parse(reader, ())?;
    if extra_fields {
        ToneMapping::parse(reader, ())?;
    }
    let offset = reader.num_read_bits();
    Ok(Some((offset, reader.read_u64()?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::{parse_extensions, parse_image_header};
    use crate::{BitReader, BitWriter, InventoryLimits};

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
        assert_eq!(read_selector(&unknown).unwrap().unwrap().1, 1);
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
