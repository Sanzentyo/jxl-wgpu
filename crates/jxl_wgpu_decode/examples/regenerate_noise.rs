//! Reproduce noise fixtures and custom-correlation linear F32 references with libjxl offline.
use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_bitstream::Bitstream;
use jxl_gpu_bitstream::{BitReader, BitWriter, FrameSectionKind};
use jxl_modular::{MaConfig, MaConfigParams};
use jxl_oxide_common::Bundle;
use jxl_wgpu_decode::vardct::frontend::LfGlobalPrefix;

#[allow(dead_code)]
#[path = "support/offline.rs"]
mod offline;

fn copy_bits(writer: &mut BitWriter, data: &[u8], start: u64, end: u64) {
    let mut reader = BitReader::new(data);
    reader.skip_bits(start).unwrap();
    let mut remaining = end.checked_sub(start).unwrap();
    while remaining != 0 {
        let count = remaining.min(56) as u8;
        writer
            .write_bits(reader.read_bits(count).unwrap(), count)
            .unwrap();
        remaining -= u64::from(count);
    }
}

fn toc_size(writer: &mut BitWriter, length: u64) {
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
    panic!("fixture section exceeds the TOC domain");
}

fn custom_correlation(bytes: &[u8], bases: [u16; 2], lf_factors: [u8; 2]) -> Vec<u8> {
    let parsed = jxl_gpu_bitstream::parse(bytes, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    assert_eq!(inventory.frames.len(), 1);
    let frame = &inventory.frames[0];
    assert!(!frame.toc_permuted);
    assert_eq!(frame.flags & (1 | 2 | 16), 1);
    let section = frame
        .sections
        .iter()
        .find(|section| section.kind == FrameSectionKind::LowFrequencyGlobal)
        .unwrap();
    let data = parsed.codestream();
    let mut range = section.bits;
    range.offset += 80;
    range.length -= 80;
    let prefix = LfGlobalPrefix::parse(data, range).unwrap();
    assert_eq!(prefix.lf_correlation, Default::default());
    // This fixture has the default correlation bit followed by the global MA-tree flag.
    let correlation_bit = prefix.suffix_bit_offset - 2;
    let mut reader = BitReader::new(data);
    reader.skip_bits(correlation_bit).unwrap();
    assert_eq!(reader.read_bits(1).unwrap(), 1);

    // Find the syntax boundary before zero padding so changing correlation length does not
    // shift the entropy payload. This CPU parser is used only by the offline generator.
    let mut ma_bits = Bitstream::new(data);
    ma_bits
        .skip_bits(prefix.global_ma_tree_bit_offset.unwrap() as usize)
        .unwrap();
    MaConfig::parse(
        &mut ma_bits,
        MaConfigParams {
            tracker: None,
            node_limit: 1 << 20,
            depth_limit: 2048,
        },
    )
    .unwrap();
    let end = ma_bits.num_read_bits() as u64;
    assert!(end <= section.bits.end().unwrap());
    assert!(section.bits.end().unwrap() - end < 8);
    let mut replacement = BitWriter::new();
    copy_bits(&mut replacement, data, section.bits.offset, correlation_bit);
    replacement.write_bits(0, 1).unwrap(); // custom correlation
    replacement.write_bits(0, 2).unwrap(); // color factor 84
    for value in bases {
        replacement.write_bits(u64::from(value), 16).unwrap();
    }
    for value in lf_factors {
        replacement.write_bits(u64::from(value), 8).unwrap();
    }
    copy_bits(&mut replacement, data, correlation_bit + 1, end);
    replacement.align_to_byte().unwrap();
    let replacement = replacement.into_bytes();

    let mut output = BitWriter::new();
    copy_bits(&mut output, data, 0, frame.toc_bits.offset);
    output.write_bits(0, 1).unwrap();
    output.align_to_byte().unwrap();
    for entry in &frame.sections {
        toc_size(
            &mut output,
            if entry.kind == section.kind {
                replacement.len() as u64
            } else {
                entry.bytes.length
            },
        );
    }
    output.align_to_byte().unwrap();
    let mut output = output.into_bytes();
    for entry in &frame.sections {
        if entry.kind == section.kind {
            output.extend_from_slice(&replacement);
        } else {
            output.extend_from_slice(
                &data[entry.bytes.offset as usize..entry.bytes.end().unwrap() as usize],
            );
        }
    }
    output
}

fn references(output: &Path, temporary: &Path, name: &str, data: &[u8]) {
    std::fs::write(output.join(format!("{name}.jxl.hex")), offline::hex(data)).unwrap();
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
    for zero in [false, true] {
        let mut data = data.to_vec();
        if zero {
            let start = inventory.frames[0].sections[0].bytes.offset as usize;
            data[start..start + 10].fill(0);
        }
        let input = temporary.join("input.jxl");
        std::fs::write(&input, data).unwrap();
        let reference = offline::run(
            Command::new(temporary.join("oracle"))
                .arg(input)
                .arg("--linear"),
        )
        .stdout;
        assert_eq!(reference.len(), pixels * 16);
        let suffix = if zero { "zero.linear" } else { "linear" };
        std::fs::write(
            output.join(format!("{name}.{suffix}.f32.hex")),
            offline::float_hex(&reference),
        )
        .unwrap();
    }
}

fn main() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("noise"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!("jxl-wgpu-noise-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    let generator = temporary.join("generate");
    offline::compile(&source.join("generate_noise.c"), &generator, &["libjxl"]);
    offline::run(Command::new(generator).arg(&output));
    offline::compile(
        &source.join("decode_extra_channels.c"),
        &temporary.join("oracle"),
        &["libjxl", "libjxl_cms"],
    );
    let base =
        offline::unhex(&std::fs::read_to_string(output.join("vardct_257x17.jxl.hex")).unwrap());
    for (name, bases, factors) in [
        ("vardct_lf_correlation", [0, 0x3c00], [140, 109]),
        // Half-precision base correlations [0.125, 0.875]; LF slopes remain zero.
        ("vardct_base_correlation", [0x3000, 0x3b00], [128, 128]),
    ] {
        references(
            &output,
            &temporary,
            name,
            &custom_correlation(&base, bases, factors),
        );
    }
    std::fs::remove_dir_all(temporary).unwrap();
    eprintln!("Regenerated noise fixtures in {}", output.display());
}
