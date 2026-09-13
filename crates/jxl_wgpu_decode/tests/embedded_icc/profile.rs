//! Change only the color declaration; preserve every entropy byte and sample representation.
use jxl_bitstream::{Bitstream, U};
use jxl_gpu_bitstream::{BitReader, BitWriter};
use jxl_image::{AnimationHeader, BitDepth, ExtraChannelInfo, SizeHeader, color::ColourEncoding};
use jxl_oxide_common::Bundle;

fn copy(writer: &mut BitWriter, data: &[u8], start: u64, end: u64) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    while reader.bit_offset() < end {
        let count = (end - reader.bit_offset()).min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
    }
}

pub(super) fn replace(data: &[u8], donor: &[u8]) -> Vec<u8> {
    let data_file = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let donor_file = jxl_gpu_bitstream::parse(donor, Default::default()).unwrap();
    let data = data_file.codestream();
    let donor = donor_file.codestream();
    let original = super::inventory(data);
    let source = super::inventory(donor);
    let image = &original.image_header;
    assert!(image.embedded_icc.is_none());
    assert_eq!(image.grayscale, source.image_header.grayscale);
    let icc = source.image_header.embedded_icc.as_ref().unwrap();
    let mut reader = Bitstream::new(data);
    assert_eq!(reader.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut reader, ()).unwrap();
    assert!(
        !reader.read_bool().unwrap(),
        "fixture metadata must be explicit"
    );
    if reader.read_bool().unwrap() {
        reader.read_bits(3).unwrap();
        if reader.read_bool().unwrap() {
            SizeHeader::parse(&mut reader, ()).unwrap();
        }
        assert!(!reader.read_bool().unwrap(), "no preview in source fixture");
        if reader.read_bool().unwrap() {
            AnimationHeader::parse(&mut reader, ()).unwrap();
        }
    }
    BitDepth::parse(&mut reader, ()).unwrap();
    reader.read_bool().unwrap();
    let extras = reader.read_u32(0, 1, 2 + U(4), 1 + U(12)).unwrap();
    for _ in 0..extras {
        ExtraChannelInfo::parse(&mut reader, ()).unwrap();
    }
    assert_eq!(reader.read_bool().unwrap(), image.xyb_encoded);
    let color_start = reader.num_read_bits() as u64;
    ColourEncoding::parse(&mut reader, ()).unwrap();
    let color_end = reader.num_read_bits() as u64;
    let mut writer = BitWriter::new();
    copy(&mut writer, data, 0, color_start);
    writer.write_bits(0, 1).unwrap(); // Explicit color.
    writer.write_bits(1, 1).unwrap(); // ICC follows the image header.
    writer.write_bits(u64::from(image.grayscale), 2).unwrap();
    copy(&mut writer, data, color_end, image.bit_range.end().unwrap());
    copy(
        &mut writer,
        donor,
        icc.bit_range.offset,
        icc.bit_range.end().unwrap(),
    );
    writer.align_to_byte().unwrap();
    let mut encoded = writer.into_bytes();
    let frames = &data[original.frames[0].header_bits.offset as usize / 8..];
    encoded.extend_from_slice(frames);
    let changed = super::inventory(&encoded);
    assert_eq!(
        &encoded[changed.frames[0].header_bits.offset as usize / 8..],
        frames
    );
    assert_eq!(
        changed.image_header.embedded_icc.as_ref().unwrap().profile,
        icc.profile
    );
    assert_eq!(
        changed.image_header.colour_encoding,
        source.image_header.colour_encoding
    );
    let mut normalized = changed.image_header;
    normalized.bit_range = image.bit_range;
    normalized.colour_encoding = image.colour_encoding;
    normalized.embedded_icc = None;
    assert_eq!(&normalized, image);
    encoded
}
