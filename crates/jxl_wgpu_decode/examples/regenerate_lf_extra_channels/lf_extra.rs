//! Offline LF-group surgery. Entropy is copied from independent libjxl output, never re-encoded.
use super::*;

pub fn packets(data: &[u8], inventory: &FrameInventory, uses_lf: bool) -> Vec<GroupPacket> {
    assert!(!inventory.toc_permuted);
    if !uses_lf {
        return inventory
            .sections
            .iter()
            .map(|section| {
                GroupPacket::new(
                    kind(section.kind),
                    data[section.bytes.offset as usize
                        ..(section.bytes.offset + section.bytes.length) as usize]
                        .to_vec(),
                )
            })
            .collect();
    }
    if inventory.sections.len() == 1 {
        let (lf, end) = boundaries(data, inventory);
        let mut packet = BitWriter::new();
        copy_bits(
            &mut packet,
            data,
            inventory.sections[0].bits.offset..lf.start,
        );
        copy_bits(&mut packet, data, lf.end..end);
        packet.align_to_byte().unwrap();
        return vec![GroupPacket::new(
            GroupPacketKind::Single,
            packet.into_bytes(),
        )];
    }
    let mut bits = Bitstream::new(data);
    let image = Arc::new(jxl_image::ImageHeader::parse(&mut bits, ()).unwrap());
    let pool = jxl_threadpool::JxlThreadPool::none();
    let mut frame = jxl_frame::Frame::parse(
        &mut bits,
        jxl_frame::FrameContext {
            image_header: image,
            tracker: None,
            pool: pool.clone(),
        },
    )
    .unwrap();
    frame.feed_bytes(&data[bits.num_read_bits() / 8..]).unwrap();
    assert!(frame.is_loading_done());
    let global = frame.try_parse_lf_global::<i32>().unwrap().unwrap();
    let vardct = global.vardct.as_ref().unwrap();
    assert!(vardct.hf_block_ctx.lf_thresholds.iter().all(Vec::is_empty));
    let mut gmodular = global.gmodular.try_clone().unwrap();
    let groups = gmodular
        .modular
        .image_mut()
        .unwrap()
        .prepare_groups(frame.pass_shifts())
        .unwrap();
    let nonempty = groups
        .lf_groups
        .iter()
        .filter(|group| !group.is_empty())
        .count();
    eprintln!(
        "sectioned consumer: {} LF groups with extra entropy",
        nonempty
    );
    assert!(
        nonempty > 0,
        "distributed fixture must contain LF extra entropy"
    );
    let mut subimages: Vec<_> = groups.lf_groups.into_iter().map(Some).collect();
    inventory
        .sections
        .iter()
        .map(|section| {
            let FrameSectionKind::LowFrequencyGroup { group_index } = section.kind else {
                return GroupPacket::new(
                    kind(section.kind),
                    data[section.bytes.offset as usize
                        ..(section.bytes.offset + section.bytes.length) as usize]
                        .to_vec(),
                );
            };
            let mut bits = Bitstream::new(data);
            bits.skip_bits(section.bits.offset as usize).unwrap();
            let (lf_width, lf_height) = frame.header().lf_group_size_for(group_index as u32);
            jxl_vardct::LfCoeff::<i32>::parse(
                &mut bits,
                jxl_vardct::LfCoeffParams {
                    lf_group_idx: group_index as u32,
                    lf_width,
                    lf_height,
                    jpeg_upsampling: inventory.jpeg_upsampling,
                    bits_per_sample: frame.header().bit_depth.bits_per_sample(),
                    global_ma_config: global.gmodular.ma_config(),
                    allow_partial: false,
                    tracker: None,
                    pool: &pool,
                },
            )
            .unwrap();
            let coefficient_end = bits.num_read_bits() as u64;
            let mut bits = Bitstream::new(data);
            bits.skip_bits(section.bits.offset as usize).unwrap();
            jxl_frame::data::LfGroup::<i32>::parse(
                &mut bits,
                jxl_frame::data::LfGroupParams {
                    frame_header: frame.header(),
                    quantizer: Some(&vardct.quantizer),
                    global_ma_config: global.gmodular.ma_config(),
                    mlf_group: subimages
                        .get_mut(group_index as usize)
                        .and_then(Option::take),
                    lf_group_idx: group_index as u32,
                    allow_partial: false,
                    tracker: None,
                    pool: &pool,
                },
            )
            .unwrap();
            let end = bits.num_read_bits() as u64;
            assert_eq!(end.div_ceil(8) * 8, section.bits.end().unwrap());
            let mut packet = BitWriter::new();
            copy_bits(&mut packet, data, coefficient_end..end);
            packet.align_to_byte().unwrap();
            GroupPacket::new(kind(section.kind), packet.into_bytes())
        })
        .collect()
}

fn kind(kind: FrameSectionKind) -> GroupPacketKind {
    match kind {
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
    }
}
