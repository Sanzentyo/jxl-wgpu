//! Offline conformance assembly. Patch entropy is independently written as flat ANS; all image
//! entropy and image metadata are preserved from native encoder fixtures.
use jxl_gpu_bitstream::{
    BitReader, BitWriter, FrameEncoding, FrameInventory, FrameSectionKind, ImageHeaderInventory,
    RestorationFilterInventory,
};
use jxl_wgpu_encode::{
    BitFragment, FrameGroupLayout, FramePacketSet, GroupPacket, GroupPacketKind, assemble_frame,
};

pub fn copy_bits(writer: &mut BitWriter, bytes: &[u8], start: u64, end: u64) {
    let mut reader = BitReader::new(bytes);
    reader.skip_bits(start).unwrap();
    let mut remaining = end - start;
    while remaining != 0 {
        let count = remaining.min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
        remaining -= u64::from(count);
    }
}

pub fn dictionary(values: &[u32]) -> BitWriter {
    // Keep the existing packet's padding unchanged. Select a legal hybrid configuration whose
    // entropy prefix ends at a byte boundary; there is no invented padding between substreams.
    for split in 0..=8 {
        for msb in 0..=split {
            for lsb in 0..=split - msb {
                if let Some(writer) = dictionary_config(values, split, msb, lsb)
                    && writer.bit_len().is_multiple_of(8)
                {
                    return writer;
                }
            }
        }
    }
    panic!("no byte-aligned test dictionary configuration");
}

fn dictionary_config(values: &[u32], split: u32, msb: u32, lsb: u32) -> Option<BitWriter> {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap(); // No LZ77.
    writer.write_bits(1, 1).unwrap(); // Simple context map.
    writer.write_bits(0, 2).unwrap(); // Every context uses cluster zero.
    writer.write_bits(0, 1).unwrap(); // ANS.
    writer.write_bits(3, 2).unwrap(); // 256-symbol alphabet.
    writer.write_bits(u64::from(split), 4).unwrap();
    if split != 8 {
        writer
            .write_bits(u64::from(msb), (32 - split.leading_zeros()) as u8)
            .unwrap();
        writer
            .write_bits(u64::from(lsb), (32 - (split - msb).leading_zeros()) as u8)
            .unwrap();
    } else if msb != 0 || lsb != 0 {
        return None;
    }
    writer.write_bits(0, 1).unwrap(); // Non-simple histogram.
    writer.write_bits(1, 1).unwrap(); // Flat.
    writer.write_bits(1, 1).unwrap();
    writer.write_bits(7, 3).unwrap();
    writer.write_bits(127, 7).unwrap(); // 256 symbols.
    let mut state = 0x13_0000u32;
    let mut parts = Vec::new();
    for &value in values.iter().rev() {
        let (token, payload, extra_bits) = if value < (1 << split) {
            (value, 0, 0)
        } else {
            let exponent = 31 - value.leading_zeros();
            let bits = exponent - msb - lsb;
            (
                (1 << split)
                    + ((exponent - split) << (msb + lsb))
                    + (((value >> (exponent - msb)) & ((1 << msb) - 1)) << lsb)
                    + (value & ((1 << lsb) - 1)),
                (value >> lsb) & ((1u32 << bits) - 1),
                bits,
            )
        };
        if token >= 256 {
            return None;
        }
        let refill = (state >= (16 << 20)).then_some(state & 0xffff);
        if refill.is_some() {
            state >>= 16;
        }
        state = (state / 16) * 4096 + token * 16 + state % 16;
        parts.push((refill, payload, extra_bits));
    }
    writer.write_bits(u64::from(state), 32).unwrap();
    for (refill, payload, bits) in parts.into_iter().rev() {
        if let Some(refill) = refill {
            writer.write_bits(u64::from(refill), 16).unwrap();
        }
        writer.write_bits(u64::from(payload), bits as u8).unwrap();
    }
    Some(writer)
}

pub fn values(frame: &FrameInventory, extras: usize, count: u32) -> Vec<u32> {
    if count == 0 {
        return vec![0];
    }
    let width = frame.width.min(7);
    let height = frame.height.min(5);
    let mut result = vec![1, 3, 0, 0, width - 1, height - 1, count - 1];
    let mut previous = [0i32; 2];
    for index in 0..count {
        let position = [
            (index * 3 % (frame.width - width + 1)) as i32,
            (index * 2 % (frame.height - height + 1)) as i32,
        ];
        for axis in 0..2 {
            let value = if index == 0 {
                position[axis] as u32
            } else {
                let delta = position[axis] - previous[axis];
                ((delta << 1) ^ (delta >> 31)) as u32
            };
            result.push(value);
        }
        previous = position;
        for channel in 0..=extras {
            let mode = (index + channel as u32) % 8;
            result.push(mode);
            if mode >= 4 && extras > 1 {
                result.push((index as usize + channel) as u32 % extras as u32);
            }
            if mode >= 3 {
                result.push((index / 8) % 2);
            }
        }
    }
    result
}

fn flags(writer: &mut BitWriter, value: u64) {
    match value {
        0 => writer.write_bits(0, 2).unwrap(),
        1..=16 => {
            writer.write_bits(1, 2).unwrap();
            writer.write_bits(value - 1, 4).unwrap();
        }
        17..=272 => {
            writer.write_bits(2, 2).unwrap();
            writer.write_bits(value - 17, 8).unwrap();
        }
        _ => panic!("unexpected fixture flags"),
    }
}

fn header(
    image: &ImageHeaderInventory,
    source: &FrameInventory,
    reference: Option<(u32, bool)>,
    patches: bool,
) -> BitFragment {
    assert!(!source.have_crop && !source.toc_permuted && source.num_passes == 1);
    assert!(source.flags & !0x80 == 0 && source.upsampling == 1);
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    writer
        .write_bits(if reference.is_some() { 2 } else { 0 }, 2)
        .unwrap();
    writer
        .write_bits(u64::from(source.encoding == FrameEncoding::Modular), 1)
        .unwrap();
    flags(&mut writer, source.flags | if patches { 2 } else { 0 });
    if !image.xyb_encoded {
        writer.write_bits(u64::from(source.do_ycbcr), 1).unwrap();
    }
    if source.do_ycbcr {
        for factor in source.jpeg_upsampling {
            writer.write_bits(u64::from(factor), 2).unwrap();
        }
    }
    writer.write_bits(0, 2).unwrap();
    for (extra, factor) in image
        .extra_channels
        .iter()
        .zip(&source.extra_channel_upsampling)
    {
        let encoded = factor >> extra.dimension_shift;
        writer.write_bits(u64::from(encoded.ilog2()), 2).unwrap();
    }
    if source.encoding == FrameEncoding::Modular {
        writer
            .write_bits(u64::from(source.group_size_shift), 2)
            .unwrap();
    }
    if image.xyb_encoded && source.encoding == FrameEncoding::VarDct {
        writer.write_bits(u64::from(source.x_qm_scale), 3).unwrap();
        writer.write_bits(u64::from(source.b_qm_scale), 3).unwrap();
    }
    if reference.is_none() {
        writer.write_bits(0, 2).unwrap();
    }
    writer.write_bits(0, 1).unwrap(); // Full canvas.
    if let Some((slot, before_color)) = reference {
        writer.write_bits(u64::from(slot), 2).unwrap();
        writer.write_bits(u64::from(before_color), 1).unwrap();
    } else {
        for _ in 0..=image.extra_channels.len() {
            writer.write_bits(0, 2).unwrap();
        }
        assert!(image.animation.is_none());
        writer.write_bits(1, 1).unwrap();
    }
    writer.write_bits(0, 2).unwrap(); // Empty frame name.
    match source.restoration_filter {
        RestorationFilterInventory::Default => writer.write_bits(1, 1).unwrap(),
        RestorationFilterInventory::Custom {
            gaborish: jxl_gpu_bitstream::GaborishInventory::Disabled,
            epf: jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled,
        } => writer.write_bits(0, 6).unwrap(),
        RestorationFilterInventory::Custom {
            gaborish,
            epf:
                jxl_gpu_bitstream::EdgePreservingFilterInventory::Enabled {
                    iterations,
                    sharp_lut: None,
                    weights: None,
                    sigma: None,
                    sigma_for_modular,
                },
        } => {
            writer.write_bits(0, 1).unwrap();
            match gaborish {
                jxl_gpu_bitstream::GaborishInventory::Disabled => writer.write_bits(0, 1).unwrap(),
                jxl_gpu_bitstream::GaborishInventory::Default => writer.write_bits(1, 2).unwrap(),
                _ => panic!("custom fixture Gaborish weights"),
            }
            writer.write_bits(u64::from(iterations), 2).unwrap();
            if source.encoding == FrameEncoding::VarDct {
                writer.write_bits(0, 1).unwrap();
            }
            writer.write_bits(0, 2).unwrap();
            if let Some(sigma) = sigma_for_modular {
                writer.write_bits(u64::from(sigma.to_bits()), 16).unwrap();
            }
            writer.write_bits(0, 2).unwrap();
        }
        other => panic!("unhandled fixture restoration {other:?}"),
    }
    writer.write_bits(0, 2).unwrap();
    BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
}

pub fn assemble(bytes: &[u8], patch_values: &[u32]) -> Vec<u8> {
    assemble_frames(
        bytes,
        &[(Some((3, true)), None), (None, Some(patch_values))],
    )
}

pub type FixtureFrame<'a> = (Option<(u32, bool)>, Option<&'a [u32]>);

/// Reuse native image entropy while exercising successive reference-slot versions. Every frame
/// except the final presentation is reference-only; an optional dictionary precedes its body.
pub fn assemble_frames(bytes: &[u8], frames: &[FixtureFrame<'_>]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let image = &inventory.image_header;
    let frame = &inventory.frames[0];
    let bytes = parsed.codestream();
    let mut result = bytes[..frame.header_bits.offset as usize / 8].to_vec();
    for &(reference, patch_values) in frames {
        let packets = frame.sections.iter().map(|section| {
            let kind = match section.kind {
                FrameSectionKind::Single => GroupPacketKind::Single,
                FrameSectionKind::LowFrequencyGlobal => GroupPacketKind::DcGlobal,
                FrameSectionKind::LowFrequencyGroup { group_index } => {
                    GroupPacketKind::DcGroup(group_index as u32)
                }
                FrameSectionKind::HighFrequencyGlobal => GroupPacketKind::AcGlobal,
                FrameSectionKind::PassGroup {
                    pass_index,
                    group_index,
                } => GroupPacketKind::AcGroup {
                    pass: pass_index as u8,
                    group: group_index as u32,
                },
            };
            let payload = if let Some(patch_values) = patch_values
                && matches!(
                    section.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                ) {
                let mut prefix = dictionary(patch_values);
                // Preserve the enclosing section's byte padding when shifting its entropy.
                copy_bits(
                    &mut prefix,
                    bytes,
                    section.bits.offset,
                    section.bits.end().unwrap(),
                );
                prefix.align_to_byte().unwrap();
                prefix.into_bytes()
            } else {
                bytes[section.bytes.offset as usize..section.bytes.end().unwrap() as usize].to_vec()
            };
            GroupPacket::new(kind, payload)
        });
        result.extend(
            assemble_frame(
                FramePacketSet::new(
                    header(image, frame, reference, patch_values.is_some()),
                    FrameGroupLayout::new(
                        frame.low_frequency_group_count as u32,
                        frame.group_count as u32,
                        frame.num_passes as u8,
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
    result
}
