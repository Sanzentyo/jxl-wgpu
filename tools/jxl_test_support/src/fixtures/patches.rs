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
    values_using_modes(frame, extras, count, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Equivalent RGB operations for an image without any alpha or extra channels. Source-over
/// reduces to replacement and multiply-add reduces to addition when alpha is implicitly one.
pub fn arithmetic_values(frame: &FrameInventory, count: u32) -> Vec<u32> {
    values_using_modes(frame, 0, count, [0, 1, 2, 3, 1, 1, 2, 2])
}

fn values_using_modes(
    frame: &FrameInventory,
    extras: usize,
    count: u32,
    modes: [u32; 8],
) -> Vec<u32> {
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
            let mode = modes[(index as usize + channel) % 8];
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

fn feature_header(bytes: &[u8], frame: &FrameInventory, new_flags: u64) -> BitFragment {
    let start = frame.header_bits.offset;
    let mut reader = BitReader::new(bytes);
    reader.skip_bits(start).unwrap();
    assert_eq!(reader.read_bits(1).unwrap(), 0); // explicit header
    reader.skip_bits(3).unwrap(); // frame type, encoding
    let mut old = BitWriter::new();
    flags(&mut old, frame.flags);
    let mut expected = BitReader::new(old.as_bytes());
    assert_eq!(
        reader.read_bits(old.bit_len() as u8).unwrap(),
        expected.read_bits(old.bit_len() as u8).unwrap()
    );
    let mut writer = BitWriter::new();
    copy_bits(&mut writer, bytes, start, start + 4);
    flags(&mut writer, new_flags);
    copy_bits(
        &mut writer,
        bytes,
        start + 4 + old.bit_len() as u64,
        frame.header_bits.end().unwrap(),
    );
    BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
}

/// Insert a fixed, nonzero model without changing any image entropy or LF dependencies.
pub fn with_noise(bytes: &[u8], lf_only: bool) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let bytes = parsed.codestream();
    let mut model = BitWriter::new();
    for value in [16, 24, 32, 48, 64, 80, 96, 112] {
        model.write_bits(value, 10).unwrap();
    }
    let mut output = bytes[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for frame in &inventory.frames {
        if lf_only && frame.lf_level == 0 {
            output.extend_from_slice(
                &bytes[frame.header_bits.offset as usize / 8
                    ..frame.sections.last().unwrap().bytes.end().unwrap() as usize],
            );
        } else {
            assert_eq!(frame.flags & (1 | 2 | 16), 0);
            output.extend(packet_frame_prefix(
                bytes,
                frame,
                feature_header(bytes, frame, frame.flags | 1),
                Some(&model),
            ));
        }
    }
    let checked = jxl_gpu_bitstream::parse(&output, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(checked.image_header, inventory.image_header);
    for (before, after) in inventory.frames.iter().zip(&checked.frames) {
        assert_eq!(before.noise_seed, after.noise_seed);
        assert_eq!(before.lf_source_frame, after.lf_source_frame);
        assert_eq!(before.color_sample_extent(), after.color_sample_extent());
        for (old, new) in before.sections.iter().zip(&after.sections) {
            let skip = if after.flags & 1 != 0
                && matches!(
                    old.kind,
                    FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
                ) {
                10
            } else {
                0
            };
            assert_eq!(
                &output[new.bytes.offset as usize + skip..new.bytes.end().unwrap() as usize],
                &bytes[old.bytes.offset as usize..old.bytes.end().unwrap() as usize]
            );
        }
    }
    output
}

fn header(
    image: &ImageHeaderInventory,
    source: &FrameInventory,
    reference: Option<(u32, bool)>,
    patches: bool,
) -> BitFragment {
    assert!(!source.have_crop && source.lf_level == 0);
    assert!(source.flags & !0xa1 == 0 && matches!(source.upsampling, 1 | 2 | 4 | 8));
    assert!(image.animation.is_none());
    // Reference-only headers imply one pass. Multipass producers remain hidden regular frames
    // with duration zero, preserving the complete native pass schedule and entropy packets.
    let reference_only = reference.is_some() && source.num_passes == 1;
    let mut writer = BitWriter::new();
    writer.write_bits(0, 1).unwrap();
    writer
        .write_bits(if reference_only { 2 } else { 0 }, 2)
        .unwrap();
    writer
        .write_bits(u64::from(source.encoding == FrameEncoding::Modular), 1)
        .unwrap();
    flags(&mut writer, source.flags | if patches { 2 } else { 0 });
    if !image.xyb_encoded {
        writer.write_bits(u64::from(source.do_ycbcr), 1).unwrap();
    }
    if source.do_ycbcr && !source.uses_lf_frame() {
        for factor in source.jpeg_upsampling {
            writer.write_bits(u64::from(factor), 2).unwrap();
        }
    }
    if !source.uses_lf_frame() {
        writer
            .write_bits(u64::from(source.upsampling.ilog2()), 2)
            .unwrap();
        for (extra, factor) in image
            .extra_channels
            .iter()
            .zip(&source.extra_channel_upsampling)
        {
            let encoded = factor >> extra.dimension_shift;
            writer.write_bits(u64::from(encoded.ilog2()), 2).unwrap();
        }
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
    if !reference_only {
        passes(&mut writer, source);
    }
    writer.write_bits(0, 1).unwrap(); // Full canvas.
    if !reference_only {
        for _ in 0..=image.extra_channels.len() {
            writer.write_bits(0, 2).unwrap();
        }
        writer
            .write_bits(u64::from(reference.is_none()), 1)
            .unwrap(); // Is last.
    }
    if let Some((slot, before_color)) = reference {
        writer.write_bits(u64::from(slot), 2).unwrap();
        writer.write_bits(u64::from(before_color), 1).unwrap();
    }
    writer.write_bits(0, 2).unwrap(); // Empty frame name.
    restoration(&mut writer, source.restoration_filter, source.encoding);
    writer.write_bits(0, 2).unwrap();
    BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
}

fn restoration(
    writer: &mut BitWriter,
    filter: RestorationFilterInventory,
    encoding: FrameEncoding,
) {
    use jxl_gpu_bitstream::{EdgePreservingFilterInventory as Epf, GaborishInventory as Gaborish};
    let RestorationFilterInventory::Custom { gaborish, epf } = filter else {
        writer.write_bits(1, 1).unwrap();
        return;
    };
    writer.write_bits(0, 1).unwrap();
    match gaborish {
        Gaborish::Disabled => writer.write_bits(0, 1).unwrap(),
        Gaborish::Default => writer.write_bits(1, 2).unwrap(),
        Gaborish::Custom { weights } => {
            writer.write_bits(3, 2).unwrap();
            for value in weights.into_iter().flatten() {
                writer.write_bits(u64::from(value.to_bits()), 16).unwrap();
            }
        }
    }
    match epf {
        Epf::Disabled => writer.write_bits(0, 2).unwrap(),
        Epf::Enabled {
            iterations,
            sharp_lut,
            weights,
            sigma,
            sigma_for_modular,
        } => {
            writer.write_bits(u64::from(iterations), 2).unwrap();
            if encoding == FrameEncoding::VarDct {
                writer
                    .write_bits(u64::from(sharp_lut.is_some()), 1)
                    .unwrap();
                if let Some(lut) = sharp_lut {
                    for value in lut {
                        writer.write_bits(u64::from(value.to_bits()), 16).unwrap();
                    }
                }
            } else {
                assert!(sharp_lut.is_none());
            }
            writer.write_bits(u64::from(weights.is_some()), 1).unwrap();
            if let Some(weights) = weights {
                for value in weights
                    .channel_scale
                    .into_iter()
                    .chain([weights.pass1_zeroflush, weights.pass2_zeroflush])
                {
                    writer.write_bits(u64::from(value.to_bits()), 16).unwrap();
                }
            }
            writer.write_bits(u64::from(sigma.is_some()), 1).unwrap();
            if let Some(sigma) = sigma {
                assert_eq!(sigma.quant_mul.is_some(), encoding == FrameEncoding::VarDct);
                for value in sigma.quant_mul.into_iter().chain([
                    sigma.pass0_sigma_scale,
                    sigma.pass2_sigma_scale,
                    sigma.border_sad_mul,
                ]) {
                    writer.write_bits(u64::from(value.to_bits()), 16).unwrap();
                }
            }
            assert_eq!(
                sigma_for_modular.is_some(),
                encoding == FrameEncoding::Modular
            );
            if let Some(value) = sigma_for_modular {
                writer.write_bits(u64::from(value.to_bits()), 16).unwrap();
            }
        }
    }
    writer.write_bits(0, 2).unwrap(); // No restoration extensions.
}

fn passes(writer: &mut BitWriter, frame: &FrameInventory) {
    let count = frame.num_passes;
    writer.write_bits(u64::from((count - 1).min(3)), 2).unwrap();
    if count >= 4 {
        writer.write_bits(u64::from(count - 4), 3).unwrap();
    }
    if count == 1 {
        return;
    }
    let passes = &frame.progressive_passes;
    assert_eq!(passes.shifts.len(), count as usize - 1);
    let downsample = passes.downsampling.len();
    writer.write_bits(downsample.min(3) as u64, 2).unwrap();
    if downsample >= 3 {
        writer.write_bits((downsample - 3) as u64, 1).unwrap();
    }
    for &shift in &passes.shifts {
        writer.write_bits(u64::from(shift), 2).unwrap();
    }
    for &divisor in &passes.downsampling {
        writer.write_bits(u64::from(divisor.ilog2()), 2).unwrap();
    }
    for &last in &passes.last_pass {
        writer.write_bits(u64::from(last.min(3)), 2).unwrap();
        if last >= 3 {
            writer.write_bits(u64::from(last), 3).unwrap();
        }
    }
}

pub fn assemble(bytes: &[u8], patch_values: &[u32]) -> Vec<u8> {
    assemble_frames(&[
        Frame {
            codestream: bytes,
            reference: Some((3, true)),
            patches: None,
        },
        Frame {
            codestream: bytes,
            reference: None,
            patches: Some(patch_values),
        },
    ])
}

/// LF dependencies precede both the hidden patch reference and its visible consumer.
pub fn assemble_shared_lf(bytes: &[u8], patch_values: &[u32]) -> Vec<u8> {
    assemble_frame_sequence(
        &[
            Frame {
                codestream: bytes,
                reference: Some((3, true)),
                patches: None,
            },
            Frame {
                codestream: bytes,
                reference: None,
                patches: Some(patch_values),
            },
        ],
        false,
    )
}

pub struct Frame<'a> {
    pub codestream: &'a [u8],
    pub reference: Option<(u32, bool)>,
    pub patches: Option<&'a [u32]>,
}

/// Reuse native image entropy while exercising successive reference-slot versions. Every frame
/// except the final presentation is hidden; an optional dictionary precedes its body.
pub fn assemble_frames(frames: &[Frame<'_>]) -> Vec<u8> {
    assemble_frame_sequence(frames, true)
}

fn assemble_frame_sequence(frames: &[Frame<'_>], repeat_lf: bool) -> Vec<u8> {
    assert!(!frames.is_empty());
    let mut common_image: Option<ImageHeaderInventory> = None;
    let mut result = Vec::new();
    for (index, input) in frames.iter().enumerate() {
        let parsed = jxl_gpu_bitstream::parse(input.codestream, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let frame = inventory.frames.last().unwrap();
        assert!(
            inventory.frames[..inventory.frames.len() - 1]
                .iter()
                .all(|frame| frame.frame_type == jxl_gpu_bitstream::FrameType::LowFrequency)
        );
        let bytes = parsed.codestream();
        let image_end = inventory.frames[0].header_bits.offset as usize / 8;
        let mut image = inventory.image_header;
        if let Some(common) = &common_image {
            // Header offsets and redundant bit encodings may differ; every decoded image
            // parameter, including transforms, extra declarations and ICC, must agree.
            image.bit_range = common.bit_range;
            assert_eq!(&image, common, "incompatible patch frame image metadata");
        } else {
            result.extend_from_slice(&bytes[..image_end]);
            common_image = Some(image.clone());
        }
        // LF slot identities are implicit in the stream. Copying the dependency chain before
        // each consumer establishes a new validated version without changing its entropy.
        if index == 0 || repeat_lf {
            result.extend_from_slice(&bytes[image_end..frame.header_bits.offset as usize / 8]);
        } else {
            assert_eq!(
                input.codestream, frames[0].codestream,
                "shared LF sequence changed source"
            );
        }
        result.extend(packet_frame(
            bytes,
            frame,
            header(&image, frame, input.reference, input.patches.is_some()),
            input.patches,
        ));
    }
    result
}

/// First decode an unchanged LF chain and save its main frame as a component reference. Then
/// replace the LF slots with patch-bearing producers and decode the unchanged main consumer.
pub fn assemble_lf_producers(bytes: &[u8], dictionaries: &[Option<&[u32]>]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let bytes = parsed.codestream();
    let main = inventory.frames.last().unwrap();
    let dependencies = &inventory.frames[..inventory.frames.len() - 1];
    assert_eq!(dependencies.len(), dictionaries.len());
    assert!(dependencies.iter().all(|frame| frame.lf_level != 0));
    let mut result = bytes[..main.header_bits.offset as usize / 8].to_vec();
    result.extend(packet_frame(
        bytes,
        main,
        header(&inventory.image_header, main, Some((3, true)), false),
        None,
    ));
    for (frame, &values) in dependencies.iter().zip(dictionaries) {
        let header = if values.is_some() {
            let mut old_flags = BitWriter::new();
            flags(&mut old_flags, frame.flags);
            let mut writer = BitWriter::new();
            // Non-default, frame type and encoding precede the U64 flags. Preserve every
            // remaining header bit, including the LF level and native filter parameters.
            let start = frame.header_bits.offset;
            let mut reader = BitReader::new(bytes);
            reader.skip_bits(start).unwrap();
            assert_eq!(reader.read_bits(1).unwrap(), 0);
            reader.skip_bits(3).unwrap();
            let mut expected = BitReader::new(old_flags.as_bytes());
            assert_eq!(
                reader.read_bits(old_flags.bit_len() as u8).unwrap(),
                expected.read_bits(old_flags.bit_len() as u8).unwrap()
            );
            copy_bits(&mut writer, bytes, start, start + 4);
            flags(&mut writer, frame.flags | 2);
            copy_bits(
                &mut writer,
                bytes,
                start + 4 + old_flags.bit_len() as u64,
                frame.header_bits.end().unwrap(),
            );
            BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
        } else {
            let mut writer = BitWriter::new();
            copy_bits(
                &mut writer,
                bytes,
                frame.header_bits.offset,
                frame.header_bits.end().unwrap(),
            );
            BitFragment::new(writer.as_bytes().to_vec(), writer.bit_len()).unwrap()
        };
        result.extend(packet_frame(bytes, frame, header, values));
    }
    result.extend_from_slice(&bytes[main.header_bits.offset as usize / 8..]);
    result
}

fn packet_frame(
    bytes: &[u8],
    frame: &FrameInventory,
    header: BitFragment,
    patch_values: Option<&[u32]>,
) -> Vec<u8> {
    packet_frame_prefix(bytes, frame, header, patch_values.map(dictionary).as_ref())
}

fn packet_frame_prefix(
    bytes: &[u8],
    frame: &FrameInventory,
    header: BitFragment,
    prefix: Option<&BitWriter>,
) -> Vec<u8> {
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
        let payload = if let Some(prefix) = prefix
            && matches!(
                section.kind,
                FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
            ) {
            let mut prefix = prefix.clone();
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
    assemble_frame(
        FramePacketSet::new(
            header,
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
    .into_bytes()
}
