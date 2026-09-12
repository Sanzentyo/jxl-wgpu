//! Reproduce every JPEG XL component sampling selector combination using offline JPEG entropy.
use std::path::{Path, PathBuf};
use std::process::Command;

use jxl_gpu_bitstream::{BitReader, BitWriter};
use jxl_oxide_common::Bundle;

use jxl_test_support::fixtures::jpeg_sampling as sampling;
use jxl_test_support::offline;

fn crop_aligned(data: &[u8]) -> Vec<u8> {
    let original = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        (original.image_header.width, original.image_header.height),
        (272, 32)
    );
    assert_eq!(original.frames[0].jpeg_upsampling, [1; 3]);
    assert!(!original.frames[0].have_crop);
    let mut bits = jxl_bitstream::Bitstream::new(data);
    assert_eq!(bits.read_bits(16).unwrap(), 0x0aff);
    jxl_image::SizeHeader::parse(&mut bits, ()).unwrap();
    let mut writer = BitWriter::new();
    writer.write_bits(0x0aff, 16).unwrap();
    writer.write_bits(0, 1).unwrap(); // explicit dimensions
    writer.write_bits(0, 2).unwrap();
    writer.write_bits(16, 9).unwrap(); // height = 17
    writer.write_bits(0, 3).unwrap(); // explicit width
    writer.write_bits(0, 2).unwrap();
    writer.write_bits(256, 9).unwrap(); // width = 257
    let mut reader = BitReader::new(data);
    reader.skip_bits(bits.num_read_bits() as u64).unwrap();
    while reader.bit_offset() < original.image_header.bit_range.end().unwrap() {
        writer.write_bits(reader.read_bits(1).unwrap(), 1).unwrap();
    }
    writer.align_to_byte().unwrap();
    let mut output = writer.into_bytes();
    output.extend_from_slice(&data[original.frames[0].header_bits.offset as usize / 8..]);
    output
}

fn generate(output: &Path, temporary: &Path, name: &str, width: u32, height: u32) {
    let mut pixels = format!("P6\n{width} {height}\n255\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                pixels.push((32 + (x * (channel + 2) / 8 + y * (7 - channel) / 2) % 192) as u8);
            }
        }
    }
    let input = temporary.join("source.ppm");
    let jpeg = temporary.join("source.jpg");
    let jxl = temporary.join("source.jxl");
    std::fs::write(&input, pixels).unwrap();
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                let selectors = [cb, y, cr];
                // JPEG limits an MCU to ten blocks. Equal 2x2 factors require twelve, but
                // JPEG XL accepts them. Both extents have a padded 34x4 block grid for
                // these factors; copy the aligned 1x1 stream's entropy, preserving every
                // block, and change the size header for the odd presentation rectangle.
                let seed = if selectors == [1; 3] {
                    offline::unhex(
                        &std::fs::read_to_string(output.join("sampling_000.jxl.hex")).unwrap(),
                    )
                } else {
                    let factors = [selectors[1], selectors[0], selectors[2]]
                        .map(|value| ["1x1", "2x2", "2x1", "1x2"][value as usize])
                        .join(",");
                    offline::run(
                        Command::new("cjpeg")
                            .args(["-quality", "90", "-sample", &factors, "-outfile"])
                            .arg(&jpeg)
                            .arg(&input),
                    );
                    offline::run(Command::new("cjxl").arg(&jpeg).arg(&jxl).args([
                        "--lossless_jpeg=1",
                        "--photon_noise_iso=800",
                        "--allow_jpeg_reconstruction=0",
                        "--gaborish=0",
                        "--epf=0",
                        "--quiet",
                    ]));
                    std::fs::read(&jxl).unwrap()
                };
                let mut data = sampling::rewrite(&seed, selectors, false);
                if selectors == [1; 3] && width == 257 {
                    data = crop_aligned(&data);
                }
                let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let original = jxl_gpu_bitstream::parse(&seed, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                assert_eq!(
                    (inventory.image_header.width, inventory.image_header.height),
                    (width, height)
                );
                let mut preserved = inventory.image_header.clone();
                preserved.width = original.image_header.width;
                preserved.height = original.image_header.height;
                preserved.bit_range = original.image_header.bit_range;
                assert_eq!(preserved, original.image_header);
                let frame = &inventory.frames[0];
                assert_eq!(frame.jpeg_upsampling, selectors);
                assert_eq!(frame.flags, 129);
                assert_eq!(frame.group_count, 2);
                assert_eq!(frame.sections.len(), original.frames[0].sections.len());
                for (new, old) in frame.sections.iter().zip(&original.frames[0].sections) {
                    assert_eq!(
                        &data[new.bytes.offset as usize..new.bytes.end().unwrap() as usize],
                        &seed[old.bytes.offset as usize..old.bytes.end().unwrap() as usize]
                    );
                }
                let file = format!("{name}_{cb}{y}{cr}.jxl.hex");
                std::fs::write(output.join(&file), offline::hex(&data)).unwrap();
                eprintln!("{file}: {} bytes", data.len());
                if selectors == [cb; 3] {
                    let data = sampling::correlated(&data);
                    let file = format!("correlation_{name}_{cb}.jxl.hex");
                    std::fs::write(output.join(&file), offline::hex(&data)).unwrap();
                    eprintln!("{file}: {} bytes", data.len());
                }
            }
        }
    }
}

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/jpeg_sampling")
        });
    std::fs::create_dir_all(&output).unwrap();
    let temporary =
        std::env::temp_dir().join(format!("jxl-wgpu-jpeg-sampling-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    generate(&output, &temporary, "sampling", 272, 32);
    generate(&output, &temporary, "odd", 257, 17);
    std::fs::remove_dir_all(temporary).unwrap();
}
