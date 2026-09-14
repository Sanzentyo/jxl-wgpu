//! Change presentation sampling while preserving every coded grid and entropy packet.
use jxl_bitstream::Bitstream;
use jxl_gpu_bitstream::BitWriter;
use jxl_image::SizeHeader;
use jxl_oxide_common::Bundle;

use super::frame_features::{copy_bits, header, packet_frame_prefix};

/// Scale a full-canvas still's nominal size and all frame sampling factors together.
/// The caller must independently verify the resulting reconstruction.
pub fn scale_presentation(bytes: &[u8], scale: u32) -> Vec<u8> {
    assert!(matches!(scale, 1 | 2 | 4 | 8));
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let bytes = parsed.codestream();
    assert_eq!(inventory.frames.len(), 1);
    let original = &inventory.frames[0];
    let mut image = inventory.image_header.clone();
    assert!(
        image.animation.is_none() && image.preview_size.is_none() && image.embedded_icc.is_none()
    );
    assert!(original.is_last && !original.have_crop && original.lf_level == 0);
    assert_eq!(original.frame_type, jxl_gpu_bitstream::FrameType::Regular);
    assert_eq!(original.color_blend, Default::default());
    assert!(
        original
            .extra_channel_blends
            .iter()
            .all(|blend| *blend == Default::default())
    );
    assert!(original.name_bytes.is_empty() && original.save_as_reference == 0);
    assert_eq!(original.flags & (1 | 2 | 16), 0, "no frame features");
    image.width = image.width.checked_mul(scale).unwrap();
    image.height = image.height.checked_mul(scale).unwrap();
    let mut frame = original.clone();
    frame.upsampling *= scale;
    assert!(frame.upsampling <= 8);
    for (factor, extra) in frame
        .extra_channel_upsampling
        .iter_mut()
        .zip(&image.extra_channels)
    {
        *factor *= scale;
        assert!(*factor <= 64 && (*factor >> extra.dimension_shift) <= 8);
    }
    let mut reader = Bitstream::new(bytes);
    assert_eq!(reader.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut reader, ()).unwrap();
    let metadata_start = reader.num_read_bits() as u64;
    let mut writer = BitWriter::new();
    writer.write_bits(0x0aff, 16).unwrap();
    writer.write_bits(0, 1).unwrap(); // Explicit dimensions, not divided by eight.
    for (index, dimension) in [image.height, image.width].into_iter().enumerate() {
        writer.write_bits(3, 2).unwrap();
        writer.write_bits(u64::from(dimension - 1), 30).unwrap();
        if index == 0 {
            writer.write_bits(0, 3).unwrap(); // Explicit width, no aspect-ratio shortcut.
        }
    }
    copy_bits(
        &mut writer,
        bytes,
        metadata_start,
        inventory.image_header.bit_range.end().unwrap(),
    );
    writer.align_to_byte().unwrap();
    let mut output = writer.into_bytes();
    output.extend(packet_frame_prefix(
        bytes,
        original,
        header(&image, &frame, None, 0),
        None,
    ));
    let checked = jxl_gpu_bitstream::parse(&output, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let after = &checked.frames[0];
    assert_eq!(after.color_sample_extent(), original.color_sample_extent());
    assert_eq!(
        (after.group_count, after.low_frequency_group_count),
        (original.group_count, original.low_frequency_group_count)
    );
    assert_eq!(
        after.extra_channel_upsampling,
        frame.extra_channel_upsampling
    );
    assert_eq!(after.sections.len(), original.sections.len());
    for (old, new) in original.sections.iter().zip(&after.sections) {
        assert_eq!(old.kind, new.kind);
        assert_eq!(
            &bytes[old.bytes.offset as usize..old.bytes.end().unwrap() as usize],
            &output[new.bytes.offset as usize..new.bytes.end().unwrap() as usize]
        );
    }
    output
}
