//! Reproduce local-tree raw matrices from the checked cjpeg-to-cjxl fixture.
//! `cargo run -p jxl_wgpu_decode --example regenerate_raw_matrix -- [output-directory]`
//!
//! One variant embeds the MA descriptor in the raw image; another gives LF and HF metadata their
//! own local descriptors and removes the global tree. Entropy tokens stay unchanged. The offline
//! reference decoder locates LF/HF boundaries; production never performs this CPU image work.

use std::ops::Range;
use std::path::{Path, PathBuf};

use jxl_bitstream::Bitstream;
use jxl_gpu_bitstream::{BitRange, BitReader, BitWriter, FrameInventory, FrameSectionKind};
use jxl_modular::{MaConfig, MaConfigParams};
use jxl_oxide_common::Bundle;
use jxl_vardct::{HfMetadata, HfMetadataParams, LfCoeff, LfCoeffParams};
use jxl_wgpu_decode::vardct::frontend::LfGlobalPrefix;

#[path = "support/offline/hex.rs"]
mod offline;

fn copy_bits(writer: &mut BitWriter, bytes: &[u8], range: Range<u64>) {
    let mut reader = BitReader::new(bytes);
    reader.skip_bits(range.start).unwrap();
    let mut remaining = range.end - range.start;
    while remaining != 0 {
        let count = remaining.min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
        remaining -= u64::from(count);
    }
}

fn toc_entry(writer: &mut BitWriter, length: u64) {
    for (selector, (base, count)) in [(0, 10), (1024, 14), (17408, 22), (4211712, 30)]
        .into_iter()
        .enumerate()
    {
        if (base..base + (1 << count)).contains(&length) {
            writer.write_bits(selector as u64, 2).unwrap();
            writer.write_bits(length - base, count).unwrap();
            return;
        }
    }
    panic!("section exceeds TOC capacity");
}

fn local_packet(
    data: &[u8],
    group: BitRange,
    frame: &FrameInventory,
    prefix: &LfGlobalPrefix,
    ma: &MaConfig,
    tree: Range<u64>,
) -> Vec<u8> {
    let pool = jxl_threadpool::JxlThreadPool::none();
    let mut bits = Bitstream::new(data);
    bits.skip_bits(group.offset as usize).unwrap();
    LfCoeff::<i32>::parse(
        &mut bits,
        LfCoeffParams {
            lf_group_idx: 0,
            lf_width: frame.width,
            lf_height: frame.height,
            jpeg_upsampling: frame.jpeg_upsampling,
            bits_per_sample: 8,
            global_ma_config: Some(ma),
            allow_partial: false,
            tracker: None,
            pool: &pool,
        },
    )
    .unwrap();
    let lf_end = bits.num_read_bits() as u64;
    HfMetadata::parse(
        &mut bits,
        HfMetadataParams {
            num_lf_groups: 1,
            lf_group_idx: 0,
            lf_width: frame.width,
            lf_height: frame.height,
            jpeg_upsampling: frame.jpeg_upsampling,
            bits_per_sample: 8,
            global_ma_config: Some(ma),
            epf: None,
            quantizer_global_scale: prefix.global_scale,
            tracker: None,
            pool: &pool,
        },
    )
    .unwrap();
    let hf_end = bits.num_read_bits() as u64;
    assert!(hf_end <= group.end().unwrap());
    let padded_blocks = frame.width.div_ceil(16) * 2 * frame.height.div_ceil(16) * 2;
    let count_bits = padded_blocks.next_power_of_two().trailing_zeros();
    let mut writer = BitWriter::new();
    for (start, header, end) in [
        (group.offset, group.offset + 2, lf_end),
        (lf_end, lf_end + u64::from(count_bits), hf_end),
    ] {
        copy_bits(&mut writer, data, start..header);
        let mut check = BitReader::new(data);
        check.skip_bits(header).unwrap();
        assert_eq!(check.read_bits(1).unwrap(), 1); // global tree
        assert_eq!(check.read_bits(1).unwrap(), 1); // default WP
        assert_eq!(check.read_bits(2).unwrap(), 0); // no transforms
        writer.write_bits(0, 1).unwrap();
        copy_bits(&mut writer, data, header + 1..header + 4);
        copy_bits(&mut writer, data, tree.clone());
        copy_bits(&mut writer, data, header + 4..end);
    }
    writer.align_to_byte().unwrap();
    writer.into_bytes()
}

fn local_matrix(container: &[u8], local_packets: bool) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(container, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let data = parsed.codestream();
    let frame = &inventory.frames[0];
    assert!(!frame.toc_permuted);
    let section = |kind| frame.sections.iter().find(|s| s.kind == kind).unwrap().bits;
    let lf = section(FrameSectionKind::LowFrequencyGlobal);
    let hf = section(FrameSectionKind::HighFrequencyGlobal);
    let prefix = LfGlobalPrefix::parse(data, lf).unwrap();
    let tree_start = prefix.global_ma_tree_bit_offset.unwrap();
    let mut bits = Bitstream::new(data);
    bits.skip_bits(tree_start as usize).unwrap();
    let ma = MaConfig::parse(
        &mut bits,
        MaConfigParams {
            tracker: None,
            node_limit: 1 << 20,
            depth_limit: 2048,
        },
    )
    .unwrap();
    let tree_end = bits.num_read_bits() as u64;
    assert!(tree_end <= lf.end().unwrap());

    let mut reader = BitReader::new(data);
    reader.skip_bits(hf.offset).unwrap();
    assert_eq!(reader.read_bits(1).unwrap(), 0); // custom matrix set
    assert_eq!(reader.read_bits(3).unwrap(), 7); // mode 7, DCT8
    reader.skip_bits(16).unwrap(); // denominator
    let global_bit = reader.bit_offset();
    assert_eq!(reader.read_bits(1).unwrap(), 1); // global MA tree
    assert_eq!(reader.read_bits(1).unwrap(), 1); // default weighted predictor
    assert_eq!(reader.read_bits(2).unwrap(), 0); // zero transforms
    let token_start = reader.bit_offset();
    let mut replacement = BitWriter::new();
    copy_bits(&mut replacement, data, hf.offset..global_bit);
    replacement.write_bits(0, 1).unwrap(); // local MA tree
    copy_bits(&mut replacement, data, global_bit + 1..token_start);
    copy_bits(&mut replacement, data, tree_start..tree_end);
    // This checked source's HF-global syntax occupies 582 bits followed by two zero pad bits.
    // Strip its old padding before aligning the longer local descriptor to avoid an extra byte.
    assert_eq!(hf.length, 584);
    let syntax_end = hf.end().unwrap() - 2;
    let mut padding = BitReader::new(data);
    padding.skip_bits(syntax_end).unwrap();
    assert_eq!(padding.read_bits(2).unwrap(), 0);
    copy_bits(&mut replacement, data, token_start..syntax_end);
    replacement.align_to_byte().unwrap();
    let replacement = replacement.into_bytes();

    let mut replacements = vec![(FrameSectionKind::HighFrequencyGlobal, replacement)];
    if local_packets {
        assert_eq!(frame.low_frequency_group_count, 1);
        assert!(inventory.image_header.extra_channels.is_empty());
        let mut global = BitWriter::new();
        copy_bits(&mut global, data, lf.offset..tree_start - 1);
        global.write_bits(0, 1).unwrap(); // no global MA tree
        global.align_to_byte().unwrap();
        replacements.push((FrameSectionKind::LowFrequencyGlobal, global.into_bytes()));
        let kind = FrameSectionKind::LowFrequencyGroup { group_index: 0 };
        replacements.push((
            kind,
            local_packet(
                data,
                section(kind),
                frame,
                &prefix,
                &ma,
                tree_start..tree_end,
            ),
        ));
    }
    let replaced = |kind| {
        replacements
            .iter()
            .find(|(key, _)| *key == kind)
            .map(|(_, data)| data)
    };

    let mut writer = BitWriter::new();
    copy_bits(&mut writer, data, 0..frame.toc_bits.offset);
    writer.write_bits(0, 1).unwrap(); // no permutation
    writer.align_to_byte().unwrap();
    for section in &frame.sections {
        toc_entry(
            &mut writer,
            replaced(section.kind).map_or(section.bytes.length, |bytes| bytes.len() as u64),
        );
    }
    writer.align_to_byte().unwrap();
    let mut output = writer.into_bytes();
    for section in &frame.sections {
        if let Some(replacement) = replaced(section.kind) {
            output.extend_from_slice(replacement);
        } else {
            output.extend_from_slice(
                &data[section.bytes.offset as usize..section.bytes.end().unwrap() as usize],
            );
        }
    }
    let checked = jxl_gpu_bitstream::parse(&output, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(checked.image_header, inventory.image_header);
    assert_eq!(checked.frames[0].sections.len(), frame.sections.len());
    eprintln!(
        "local raw matrix (local LF/HF: {local_packets}): {} descriptor bits; {} -> {} HF-global bytes",
        tree_end - tree_start,
        hf.length / 8,
        replaced(FrameSectionKind::HighFrequencyGlobal)
            .unwrap()
            .len()
    );
    output
}

fn main() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| source.clone(), PathBuf::from);
    std::fs::create_dir_all(&output).unwrap();
    let original = offline::unhex(
        &std::fs::read_to_string(source.join("jpeg_transcode_raw_matrix.jxl.hex")).unwrap(),
    );
    for (suffix, local_packets) in [("local", false), ("local_packets", true)] {
        let generated = local_matrix(&original, local_packets);
        std::fs::write(
            output.join(format!("jpeg_transcode_raw_matrix_{suffix}.jxl.hex")),
            offline::hex(&generated),
        )
        .unwrap();
    }
}
