//! Replace only image intensity, retaining compressed ICC and every physical frame byte.
use jxl_bitstream::{Bitstream, U};
use jxl_gpu_bitstream::{BitReader, BitWriter, CodestreamInventory};
use jxl_image::{
    AnimationHeader, BitDepth, ExtraChannelInfo, SizeHeader,
    color::{ColourEncoding, ToneMapping},
};
use jxl_oxide_common::Bundle;

fn inventory(data: &[u8]) -> CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

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

/// The fixture intensity must be a positive integer exactly representable as normal binary16.
pub fn replace(data: &[u8], nits: u16) -> Vec<u8> {
    assert!(nits > 0);
    let bits = f32::from(nits).to_bits();
    let exponent = ((bits >> 23) & 255) - 127 + 15;
    assert!((1..31).contains(&exponent) && bits & 0x1fff == 0);
    let half = (exponent << 10) | ((bits >> 13) & 0x3ff);
    let file = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let data = file.codestream();
    let original = inventory(data);
    let image = &original.image_header;
    let mut reader = Bitstream::new(data);
    assert_eq!(reader.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut reader, ()).unwrap();
    assert!(!reader.read_bool().unwrap(), "explicit fixture metadata");
    let extra_flag = reader.num_read_bits() as u64;
    let extras = reader.read_bool().unwrap();
    if extras {
        reader.read_bits(3).unwrap();
        if reader.read_bool().unwrap() {
            SizeHeader::parse(&mut reader, ()).unwrap();
        }
        assert!(
            !reader.read_bool().unwrap(),
            "intensity fixture has no preview"
        );
        if reader.read_bool().unwrap() {
            AnimationHeader::parse(&mut reader, ()).unwrap();
        }
    }
    let representation = reader.num_read_bits() as u64;
    BitDepth::parse(&mut reader, ()).unwrap();
    reader.read_bool().unwrap();
    let count = reader.read_u32(0, 1, 2 + U(4), 1 + U(12)).unwrap();
    for _ in 0..count {
        ExtraChannelInfo::parse(&mut reader, ()).unwrap();
    }
    reader.read_bool().unwrap();
    ColourEncoding::parse(&mut reader, ()).unwrap();
    let tone_start = reader.num_read_bits() as u64;
    if extras {
        ToneMapping::parse(&mut reader, ()).unwrap();
    }
    let tone_end = reader.num_read_bits() as u64;
    assert_eq!(image.tone_mapping.min_nits.to_f32(), 0.0);
    assert!(!image.tone_mapping.relative_to_max_display);
    assert_eq!(image.tone_mapping.linear_below.to_f32(), 0.0);
    let mut writer = BitWriter::new();
    copy(&mut writer, data, 0, extra_flag);
    writer.write_bits(1, 1).unwrap();
    if extras {
        copy(&mut writer, data, extra_flag + 1, representation);
    } else {
        writer.write_bits(0, 6).unwrap(); // Identity orientation; no intrinsic size, preview or animation.
    }
    copy(&mut writer, data, representation, tone_start);
    writer.write_bits(0, 1).unwrap(); // Explicit tone mapping.
    writer.write_bits(u64::from(half), 16).unwrap();
    writer.write_bits(0, 33).unwrap(); // Zero min_nits and linear_below; absolute threshold.
    copy(&mut writer, data, tone_end, image.bit_range.end().unwrap());
    if let Some(icc) = &image.embedded_icc {
        copy(
            &mut writer,
            data,
            icc.bit_range.offset,
            icc.bit_range.end().unwrap(),
        );
    }
    writer.align_to_byte().unwrap();
    let mut encoded = writer.into_bytes();
    let frame_bytes = &data[original.frames[0].header_bits.offset as usize / 8..];
    encoded.extend_from_slice(frame_bytes);
    let changed = inventory(&encoded);
    assert_eq!(
        &encoded[changed.frames[0].header_bits.offset as usize / 8..],
        frame_bytes
    );
    assert_eq!(
        changed.image_header.tone_mapping.intensity_target.to_f32(),
        f32::from(nits)
    );
    let mut normalized = changed.image_header;
    normalized.bit_range = image.bit_range;
    normalized.tone_mapping = image.tone_mapping;
    // ICC's bit range moves with the header; its exact profile and all remaining metadata stay fixed.
    if let Some(icc) = &mut normalized.embedded_icc {
        assert_eq!(icc.profile, image.embedded_icc.as_ref().unwrap().profile);
        icc.bit_range = image.embedded_icc.as_ref().unwrap().bit_range;
    }
    assert_eq!(&normalized, image);
    encoded
}
