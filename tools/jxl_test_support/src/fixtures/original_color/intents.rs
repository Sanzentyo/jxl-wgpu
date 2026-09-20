//! Rewrite only an explicit RGB/Gray intent, preserving every physical frame byte.

use jxl_bitstream::{Bitstream, U};
use jxl_gpu_bitstream::{BitReader, BitWriter, ColourEncodingInventory, RenderingIntentInventory};
use jxl_image::{
    AnimationHeader, BitDepth, ExtraChannelInfo, SizeHeader,
    color::{ColourEncoding, ColourSpace},
};
use jxl_oxide_common::Bundle;

pub const ALL: [RenderingIntentInventory; 4] = [
    RenderingIntentInventory::Perceptual,
    RenderingIntentInventory::Relative,
    RenderingIntentInventory::Saturation,
    RenderingIntentInventory::Absolute,
];

fn copy(writer: &mut BitWriter, data: &[u8], start: u64, end: u64) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    while reader.bit_offset() < end {
        let bits = (end - reader.bit_offset()).min(56) as u8;
        writer
            .write_bits(reader.read_bits(bits).unwrap(), bits)
            .unwrap();
    }
}

pub fn replace(data: &[u8], intent: RenderingIntentInventory) -> Vec<u8> {
    let file = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let data = file.codestream();
    let original = file.codestream_inventory(Default::default()).unwrap();
    let image = &original.image_header;
    assert!(image.embedded_icc.is_none());
    let mut reader = Bitstream::new(data);
    assert_eq!(reader.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut reader, ()).unwrap();
    assert!(!reader.read_bool().unwrap(), "explicit image metadata");
    if reader.read_bool().unwrap() {
        reader.read_bits(3).unwrap();
        if reader.read_bool().unwrap() {
            SizeHeader::parse(&mut reader, ()).unwrap();
        }
        assert!(
            !reader.read_bool().unwrap(),
            "intent fixtures have no preview"
        );
        if reader.read_bool().unwrap() {
            AnimationHeader::parse(&mut reader, ()).unwrap();
        }
    }
    BitDepth::parse(&mut reader, ()).unwrap();
    reader.read_bool().unwrap();
    let count = reader.read_u32(0, 1, 2 + U(4), 1 + U(12)).unwrap();
    for _ in 0..count {
        ExtraChannelInfo::parse(&mut reader, ()).unwrap();
    }
    reader.read_bool().unwrap();
    let color_start = reader.num_read_bits() as u64;
    let ColourEncoding::Enum(color) = ColourEncoding::parse(&mut reader, ()).unwrap() else {
        panic!("enumerated fixture");
    };
    assert!(matches!(
        color.colour_space,
        ColourSpace::Rgb | ColourSpace::Grey
    ));
    assert_eq!(
        color.rendering_intent,
        jxl_image::color::RenderingIntent::Relative
    );
    let end = reader.num_read_bits() as u64;
    // All these fixtures declare explicit color. Relative is the final two-bit enum 1.
    let mut bits = BitReader::new(data);
    bits.skip_bits(color_start).unwrap();
    assert_eq!(bits.read_bits(1).unwrap(), 0);
    bits.skip_bits(end - 2 - bits.bit_offset()).unwrap();
    assert_eq!(bits.read_bits(2).unwrap(), 1);
    let mut writer = BitWriter::new();
    copy(&mut writer, data, 0, end - 2);
    let value = intent as u64;
    writer.write_bits(value.min(2), 2).unwrap();
    if value >= 2 {
        writer.write_bits(value - 2, 4).unwrap();
    }
    copy(&mut writer, data, end, image.bit_range.end().unwrap());
    writer.align_to_byte().unwrap();
    let mut encoded = writer.into_bytes();
    let frames = &data[original.frames[0].header_bits.offset as usize / 8..];
    encoded.extend_from_slice(frames);
    let changed = jxl_gpu_bitstream::parse(&encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        &encoded[changed.frames[0].header_bits.offset as usize / 8..],
        frames
    );
    let mut normalized = changed.image_header;
    let ColourEncodingInventory::Enumerated {
        rendering_intent, ..
    } = &mut normalized.colour_encoding
    else {
        unreachable!();
    };
    assert_eq!(*rendering_intent, intent);
    *rendering_intent = RenderingIntentInventory::Relative;
    normalized.bit_range = image.bit_range;
    assert_eq!(&normalized, image);
    let mut independent = jxl_oxide::JxlImage::builder().build_uninit();
    independent
        .feed_bytes(&encoded[..changed.frames[0].header_bits.offset as usize / 8])
        .unwrap();
    let jxl_oxide::InitializeResult::Initialized(independent) = independent.try_init().unwrap()
    else {
        panic!("complete rewritten header");
    };
    let ColourEncoding::Enum(color) = &independent.image_header().metadata.colour_encoding else {
        unreachable!();
    };
    assert_eq!(color.rendering_intent as u8, intent as u8);
    encoded
}
