//! Add rendering syntax to independent entropy, without decoding or re-encoding image samples.
use std::path::Path;

use jxl_gpu_bitstream::{
    BitReader, BitWriter, FrameInventory, FrameSectionKind, FrameType, RestorationFilterInventory,
};

use super::{copy_bits, offline, toc_size};

fn data(source: &Path, name: &str) -> Vec<u8> {
    offline::unhex(&std::fs::read_to_string(source.join(format!("{name}.jxl.hex"))).unwrap())
}

fn section<'a>(data: &'a [u8], frame: &FrameInventory, index: usize) -> &'a [u8] {
    let range = frame.sections[index].bytes;
    &data[range.offset as usize..range.end().unwrap() as usize]
}

fn global(kind: FrameSectionKind) -> bool {
    matches!(
        kind,
        FrameSectionKind::Single | FrameSectionKind::LowFrequencyGlobal
    )
}

fn append_sections(
    mut header: BitWriter,
    data: &[u8],
    frame: &FrameInventory,
    noise: Option<&[u8]>,
) -> Vec<u8> {
    assert!(!frame.toc_permuted);
    header.write_bits(0, 1).unwrap();
    header.align_to_byte().unwrap();
    for entry in &frame.sections {
        let additional = if global(entry.kind) {
            noise.map_or(0, <[u8]>::len)
        } else {
            0
        };
        toc_size(&mut header, entry.bytes.length + additional as u64);
    }
    header.align_to_byte().unwrap();
    let mut output = header.into_bytes();
    for (index, entry) in frame.sections.iter().enumerate() {
        if global(entry.kind) {
            output.extend_from_slice(noise.unwrap_or_default());
        }
        output.extend_from_slice(section(data, frame, index));
    }
    output
}

fn restoration(data: &[u8], gaborish: bool, epf: u64) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert!(frame.do_ycbcr && !inventory.image_header.xyb_encoded);
    assert_eq!(
        frame.restoration_filter,
        RestorationFilterInventory::Custom {
            gaborish: jxl_gpu_bitstream::GaborishInventory::Disabled,
            epf: jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled,
        }
    );
    let data = parsed.codestream();
    // Disabled restoration followed by the two empty extension selectors ends this header.
    let start = frame.header_bits.end().unwrap() - 8;
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    assert_eq!(reader.read_bits(8).unwrap(), 0);
    let mut header = BitWriter::new();
    copy_bits(&mut header, data, 0, start);
    header.write_bits(0, 1).unwrap();
    header.write_bits(u64::from(gaborish), 1).unwrap();
    if gaborish {
        header.write_bits(0, 1).unwrap();
    }
    header.write_bits(epf, 2).unwrap();
    if epf != 0 {
        // JPEG recompression uses sharpness index zero. Default LUT[0] would make EPF
        // an identity. Set every entry to one and quant_mul to eight for effective filtering.
        header.write_bits(1, 1).unwrap();
        for _ in 0..8 {
            header.write_bits(0x3c00, 16).unwrap();
        }
        header.write_bits(0, 1).unwrap(); // default channel weights
        header.write_bits(1, 1).unwrap(); // custom sigma parameters
        for value in [0x4800, 0x3b33, 0x4680, 0x3955] {
            header.write_bits(value, 16).unwrap(); // 8, ~0.9, 6.5, ~2/3
        }
    }
    header.write_bits(0, 4).unwrap();
    let output = append_sections(header, data, frame, None);
    let checked = jxl_gpu_bitstream::parse(&output, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(checked.image_header, inventory.image_header);
    assert_eq!(checked.frames[0].jpeg_upsampling, frame.jpeg_upsampling);
    assert_eq!(checked.frames[0].sections.len(), frame.sections.len());
    for index in 0..frame.sections.len() {
        assert_eq!(
            section(&output, &checked.frames[0], index),
            section(data, frame, index)
        );
    }
    output
}

fn write_flags(writer: &mut BitWriter, value: u64) {
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
        _ => panic!("unexpected source frame flags"),
    }
}

fn lf_noise(data: &[u8]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let mut model = BitWriter::new();
    for value in [16, 24, 32, 48, 64, 80, 96, 112] {
        model.write_bits(value, 10).unwrap();
    }
    let model = model.into_bytes();
    let mut output = data[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
    for frame in &inventory.frames {
        let start = frame.header_bits.offset;
        assert_eq!(start % 8, 0);
        if frame.frame_type != FrameType::LowFrequency {
            let end = frame.sections.last().unwrap().bytes.end().unwrap();
            output.extend_from_slice(&data[start as usize / 8..end as usize]);
            continue;
        }
        assert_eq!(frame.flags & (1 | 2 | 16), 0);
        let mut reader = BitReader::new(data);
        reader.skip_bits(start).unwrap();
        assert_eq!(reader.read_bits(1).unwrap(), 0); // explicit frame header
        let flags_bits = match frame.flags {
            0 => 2,
            1..=16 => 6,
            17..=272 => 10,
            _ => panic!("flags"),
        };
        let mut header = BitWriter::new();
        copy_bits(&mut header, data, start, start + 4);
        write_flags(&mut header, frame.flags | 1);
        copy_bits(
            &mut header,
            data,
            start + 4 + flags_bits,
            frame.header_bits.end().unwrap(),
        );
        output.extend(append_sections(header, data, frame, Some(&model)));
    }
    let checked = jxl_gpu_bitstream::parse(&output, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(checked.image_header, inventory.image_header);
    assert_eq!(checked.frames.len(), inventory.frames.len());
    for (old, new) in inventory.frames.iter().zip(&checked.frames) {
        let lf = old.frame_type == FrameType::LowFrequency;
        assert_eq!(new.flags, old.flags | u64::from(lf));
        assert_eq!(new.noise_seed, old.noise_seed);
        assert_eq!(new.lf_source_frame, old.lf_source_frame);
        assert_eq!(new.color_sample_extent(), old.color_sample_extent());
        assert_eq!(new.sections.len(), old.sections.len());
        for index in 0..old.sections.len() {
            let skip = if lf && global(old.sections[index].kind) {
                model.len()
            } else {
                0
            };
            assert_eq!(&section(&output, new, index)[..skip], &model[..skip]);
            assert_eq!(
                &section(&output, new, index)[skip..],
                section(data, old, index)
            );
        }
    }
    output
}

pub fn generate(source: &Path, output: &Path) {
    for sampling in ["444", "422", "440", "420", "gray"] {
        let seed = data(output, &format!("jpeg_{sampling}"));
        for (name, gab, epf) in [
            ("gab", true, 0),
            ("epf1", false, 1),
            ("gab_epf2", true, 2),
            ("gab_epf3", true, 3),
        ] {
            std::fs::write(
                output.join(format!("jpeg_{sampling}_{name}.jxl.hex")),
                offline::hex(&restoration(&seed, gab, epf)),
            )
            .unwrap();
        }
    }
    for name in [
        "modular_gab0",
        "modular_gab1",
        "vardct_gab0",
        "vardct_gab1",
        "nested_modular_gab1",
        "nested_vardct_gab1",
    ] {
        let seed = data(&source.join("lf_extra_channels"), name);
        std::fs::write(
            output.join(format!("lf_{name}.jxl.hex")),
            offline::hex(&lf_noise(&seed)),
        )
        .unwrap();
    }
    let seed = data(source, "testsrc_vardct_progressive_dc_ac");
    std::fs::write(
        output.join("lf_progressive_ac.jxl.hex"),
        offline::hex(&lf_noise(&seed)),
    )
    .unwrap();
}
