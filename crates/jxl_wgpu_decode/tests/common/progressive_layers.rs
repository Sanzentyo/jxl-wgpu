//! Offline/test-only metadata reframing. Every entropy byte stays in the original fixture.
use jxl_gpu_bitstream::{BitWriter, FrameInventory, FrameSectionKind};
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

#[allow(dead_code)]
#[path = "../../examples/support/offline/hex.rs"]
pub mod hex;

pub fn headers(data: &[u8]) -> String {
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let end = inventory.frames[0].header_bits.offset as usize / 8;
    let mut result = format!("image {}\n", hex::hex(&data[..end]).replace('\n', ""));
    for frame in &inventory.frames {
        let range = frame.header_bits;
        let mut bits = BitWriter::new();
        for bit in range.offset..range.offset + range.length {
            bits.write_bits(u64::from((data[bit as usize / 8] >> (bit % 8)) & 1), 1)
                .unwrap();
        }
        result.push_str(&format!(
            "frame {} {}\n",
            range.length,
            hex::hex(&bits.into_bytes()).replace('\n', "")
        ));
    }
    result
}

pub fn reframe(data: &[u8], frames: &[FrameInventory], headers: &str) -> Vec<u8> {
    let mut lines = headers.lines();
    let mut result = hex::unhex(lines.next().unwrap().strip_prefix("image ").unwrap());
    for source in frames {
        assert!(!source.toc_permuted);
        // Noise and patches depend on the original frame identity/reference slots.
        assert_eq!(source.flags & 3, 0);
        let fields: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(fields[0], "frame");
        let header = BitFragment::new(hex::unhex(fields[2]), fields[1].parse().unwrap()).unwrap();
        let packets = source.sections.iter().map(|section| {
            let kind = match section.kind {
                FrameSectionKind::Single => GroupPacketKind::Single,
                FrameSectionKind::LowFrequencyGlobal => GroupPacketKind::DcGlobal,
                FrameSectionKind::LowFrequencyGroup { group_index } => {
                    GroupPacketKind::DcGroup(group_index.try_into().unwrap())
                }
                FrameSectionKind::HighFrequencyGlobal => GroupPacketKind::AcGlobal,
                FrameSectionKind::PassGroup {
                    pass_index,
                    group_index,
                } => GroupPacketKind::AcGroup {
                    pass: pass_index.try_into().unwrap(),
                    group: group_index.try_into().unwrap(),
                },
            };
            let start = section.bytes.offset as usize;
            GroupPacket::new(
                kind,
                data[start..start + section.bytes.length as usize].to_vec(),
            )
        });
        result.extend(
            assemble_frame(
                FramePacketSet::new(
                    header,
                    FrameGroupLayout::new(
                        source.low_frequency_group_count.try_into().unwrap(),
                        source.group_count.try_into().unwrap(),
                        source.num_passes.try_into().unwrap(),
                    )
                    .unwrap(),
                    packets,
                )
                .unwrap(),
            )
            .unwrap()
            .into_bytes(),
        );
    }
    assert!(lines.next().is_none());
    let original = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let standalone = jxl_gpu_bitstream::parse(&result, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let mut image = standalone.image_header.clone();
    image.bit_range = original.image_header.bit_range;
    image.width = original.image_header.width;
    image.height = original.image_header.height;
    image.orientation = original.image_header.orientation;
    image.animation = original.image_header.animation;
    assert_eq!(
        image, original.image_header,
        "reframing preserves color and restoration metadata"
    );
    assert_eq!(standalone.frames.len(), frames.len());
    for (actual, source) in standalone.frames.iter().zip(frames) {
        assert_eq!(
            actual
                .lf_source_frame
                .map(|id| frames[id as usize].frame_index),
            source.lf_source_frame
        );
        let mut actual = actual.clone();
        actual.frame_index = source.frame_index;
        actual.noise_seed = source.noise_seed;
        actual.header_bits = source.header_bits;
        actual.toc_bits = source.toc_bits;
        actual.lf_source_frame = source.lf_source_frame;
        actual.x0 = source.x0;
        actual.y0 = source.y0;
        actual.have_crop = source.have_crop;
        actual.color_blend = source.color_blend;
        actual.duration_ticks = source.duration_ticks;
        actual.timecode = source.timecode;
        actual.is_last = source.is_last;
        actual.save_as_reference = source.save_as_reference;
        actual.save_before_color_transform = source.save_before_color_transform;
        actual.name_bytes.clone_from(&source.name_bytes);
        assert_eq!(actual.sections.len(), source.sections.len());
        for (section, source) in actual.sections.iter_mut().zip(&source.sections) {
            assert_eq!(section.bytes.length, source.bytes.length);
            assert_eq!(section.bits.length, source.bits.length);
            section.bytes.offset = source.bytes.offset;
            section.bits.offset = source.bits.offset;
        }
        assert_eq!(
            &actual, source,
            "all entropy interpretation fields must be unchanged"
        );
    }
    result
}
