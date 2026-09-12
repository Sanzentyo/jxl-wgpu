//! Offline/test-only pass relocation; every entropy payload is retained byte for byte.
use jxl_gpu_bitstream::{BitWriter, FrameEncoding, FrameSectionKind, FrameType};
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

#[derive(Clone, Debug)]
pub struct Schedule {
    pub passes: u8,
    pub boundaries: Vec<(u32, u32)>,
    /// Target pass index for each original pass. New passes have empty entropy sections.
    pub source_passes: Vec<u8>,
}

pub fn schedules(squeeze: bool) -> Vec<Schedule> {
    let mut output = Vec::new();
    for passes in 1u8..=11 {
        if squeeze {
            if passes < 2 {
                continue;
            }
            let first = (passes - 2).min(7);
            output.push(Schedule {
                passes,
                boundaries: vec![(2, u32::from(first))],
                source_passes: vec![first, passes - 1],
            });
            if passes >= 3 {
                let full = (passes - 2).min(7);
                output.push(Schedule {
                    passes,
                    boundaries: vec![(2, 0), (1, u32::from(full))],
                    source_passes: vec![0, full],
                });
            }
        } else {
            output.push(Schedule {
                passes,
                boundaries: vec![],
                source_passes: vec![passes - 1],
            });
            if (2..=4).contains(&passes) {
                output.push(Schedule {
                    passes,
                    boundaries: (0..passes)
                        .map(|pass| (1 << (passes - 1 - pass), u32::from(pass)))
                        .collect(),
                    source_passes: vec![passes - 1],
                });
            }
            if passes > 4 {
                output.push(Schedule {
                    passes,
                    boundaries: vec![(8, 0), (4, 1), (2, 2), (1, 3)],
                    source_passes: vec![3],
                });
            }
        }
    }
    output
}

pub fn reframe(data: &[u8], schedule: &Schedule) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    let image = &inventory.image_header;
    assert_eq!(frame.encoding, FrameEncoding::Modular);
    assert_eq!(frame.frame_type, FrameType::Regular);
    assert_eq!(frame.flags, 0);
    assert_eq!(frame.upsampling, 1);
    assert!(!frame.do_ycbcr && !image.xyb_encoded && image.animation.is_none());
    assert!(image.extra_channels.is_empty());
    assert!(frame.group_count > 1);
    assert_eq!(schedule.source_passes.len(), frame.num_passes as usize);
    assert!(
        schedule
            .source_passes
            .iter()
            .all(|&pass| pass < schedule.passes)
    );
    assert!(
        schedule
            .source_passes
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    );

    let mut header = BitWriter::new();
    header.write_bits(0, 1).unwrap(); // explicit frame
    header.write_bits(0, 2).unwrap(); // Regular
    header.write_bits(1, 1).unwrap(); // Modular
    header.write_bits(0, 2).unwrap(); // flags
    header.write_bits(0, 1).unwrap(); // no YCbCr
    header.write_bits(0, 2).unwrap(); // 1x upsampling
    header
        .write_bits(u64::from(frame.group_size_shift), 2)
        .unwrap();
    let passes = schedule.passes;
    assert!((1..=11).contains(&passes));
    header.write_bits(u64::from(passes.min(4) - 1), 2).unwrap();
    if passes >= 4 {
        header.write_bits(u64::from(passes - 4), 3).unwrap();
    }
    if passes > 1 {
        let count = schedule.boundaries.len();
        assert!(count <= 4);
        header.write_bits(count.min(3) as u64, 2).unwrap();
        if count >= 3 {
            header.write_bits((count - 3) as u64, 1).unwrap();
        }
        for pass in 0..passes - 1 {
            // Modular does not use the coefficient shifts; exercise every serialized value.
            header.write_bits(u64::from(pass % 4), 2).unwrap();
        }
        for &(factor, _) in &schedule.boundaries {
            header
                .write_bits(u64::from(factor.trailing_zeros()), 2)
                .unwrap();
        }
        for &(_, pass) in &schedule.boundaries {
            header.write_bits(u64::from(pass.min(3)), 2).unwrap();
            if pass >= 3 {
                header.write_bits(u64::from(pass), 3).unwrap();
            }
        }
    } else {
        assert!(schedule.boundaries.is_empty());
    }
    header.write_bits(0, 1).unwrap(); // no crop
    header.write_bits(0, 2).unwrap(); // Replace
    header.write_bits(1, 1).unwrap(); // final frame
    header.write_bits(0, 2).unwrap(); // empty name
    header.write_bits(0, 1).unwrap(); // explicit restoration
    header.write_bits(0, 1).unwrap(); // no Gaborish
    header.write_bits(0, 2).unwrap(); // no EPF
    header.write_bits(0, 4).unwrap(); // empty restoration/frame extensions
    let header_bits = header.bit_len();

    let mut packets = Vec::new();
    for section in &frame.sections {
        let kind = match section.kind {
            FrameSectionKind::LowFrequencyGlobal => GroupPacketKind::DcGlobal,
            FrameSectionKind::LowFrequencyGroup { group_index } => {
                GroupPacketKind::DcGroup(group_index as u32)
            }
            FrameSectionKind::HighFrequencyGlobal => GroupPacketKind::AcGlobal,
            FrameSectionKind::PassGroup { .. } => continue,
            FrameSectionKind::Single => unreachable!(),
        };
        packets.push(GroupPacket::new(
            kind,
            data[section.bytes.offset as usize..section.bytes.end().unwrap() as usize].to_vec(),
        ));
    }
    for pass in 0..passes {
        for group in 0..frame.group_count {
            let payload = schedule
                .source_passes
                .iter()
                .position(|&target| target == pass)
                .map_or_else(Vec::new, |source| {
                    let section = frame
                        .sections
                        .iter()
                        .find(|section| {
                            section.kind
                                == FrameSectionKind::PassGroup {
                                    pass_index: source as u32,
                                    group_index: group,
                                }
                        })
                        .unwrap();
                    data[section.bytes.offset as usize..section.bytes.end().unwrap() as usize]
                        .to_vec()
                });
            packets.push(GroupPacket::new(
                GroupPacketKind::AcGroup {
                    pass,
                    group: group as u32,
                },
                payload,
            ));
        }
    }
    let encoded = assemble_frame(
        FramePacketSet::new(
            BitFragment::new(header.into_bytes(), header_bits).unwrap(),
            FrameGroupLayout::new(
                frame.low_frequency_group_count as u32,
                frame.group_count as u32,
                passes,
            )
            .unwrap(),
            packets,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(frame.header_bits.offset % 8, 0);
    let mut output = data[..frame.header_bits.offset as usize / 8].to_vec();
    output.extend_from_slice(encoded.bytes());
    output
}

pub fn samples(width: u32, height: u32) -> Vec<u8> {
    (0..width * height)
        .map(|i| {
            let (x, y) = (i % width, i / width);
            (x * 37 + y * 73 + (x * y) % 251) as u8
        })
        .collect()
}
