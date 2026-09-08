//! Reproducible offline libjxl integer precision fixtures. No production CPU codec is linked.
//! `cargo run -p jxl_wgpu_decode --example regenerate_integer -- [output-directory]`

use jxl_gpu_bitstream::{BitWriter, SampleBitDepth};
#[path = "support/offline.rs"]
mod offline;
use offline::{compile, extra_references, float_hex, generate_extras, hex, run};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy)]
struct Case {
    bits: u8,
    colors: u8,
    alpha: u8,
    width: u32,
    height: u32,
    predictor: u8,
    rct: u8,
}

impl Case {
    fn name(self) -> String {
        format!(
            "{}-{}-{}-{}x{}-p{}-r{}",
            self.bits, self.colors, self.alpha, self.width, self.height, self.predictor, self.rct
        )
    }

    /// Explicit sRGB metadata for the generated words. Image precision is independent of the
    /// integer frame coding. No Palette is emitted, so implicit tables cannot depend on it.
    fn header(self) -> Vec<u8> {
        let mut writer = BitWriter::new();
        writer.write_bits(0x0aff, 16).unwrap();
        writer.write_bits(0, 1).unwrap(); // small size = false
        for (size, ratio) in [(self.height, true), (self.width, false)] {
            assert!((1..=512).contains(&size));
            writer.write_bits(0, 2).unwrap();
            writer.write_bits(u64::from(size - 1), 9).unwrap();
            if ratio {
                writer.write_bits(0, 3).unwrap();
            }
        }
        writer.write_bits(0, 1).unwrap(); // explicit image metadata
        writer.write_bits(0, 1).unwrap(); // no extra fields
        depth(&mut writer, self.bits);
        writer.write_bits(0, 1).unwrap(); // 32-bit Modular working buffers
        writer.write_bits(u64::from(self.alpha != 0), 2).unwrap();
        if self.alpha != 0 {
            writer.write_bits(0, 1).unwrap(); // explicit extra channel
            writer.write_bits(0, 2).unwrap(); // Alpha
            depth(&mut writer, self.alpha);
            writer.write_bits(0, 2).unwrap(); // dim_shift
            writer.write_bits(0, 2).unwrap(); // name length
            writer.write_bits(0, 1).unwrap(); // unassociated
        }
        writer.write_bits(0, 1).unwrap(); // original color, not XYB
        if self.colors == 3 {
            writer.write_bits(1, 1).unwrap(); // default sRGB
        } else {
            writer.write_bits(0, 1).unwrap(); // explicit grayscale encoding
            writer.write_bits(0, 1).unwrap(); // no ICC
            writer.write_bits(1, 2).unwrap(); // gray
            writer.write_bits(1, 2).unwrap(); // D65
            writer.write_bits(0, 1).unwrap(); // enumerated transfer
            writer.write_bits(2, 2).unwrap();
            writer.write_bits(11, 4).unwrap(); // sRGB = 13
            writer.write_bits(1, 2).unwrap(); // relative rendering intent
        }
        writer.write_bits(0, 2).unwrap(); // no extensions
        writer.write_bits(1, 1).unwrap(); // default transform data
        writer.align_to_byte().unwrap();
        writer.into_bytes()
    }
}

fn depth(writer: &mut BitWriter, bits: u8) {
    assert!((1..=31).contains(&bits));
    writer.write_bits(0, 1).unwrap();
    writer.write_bits(3, 2).unwrap();
    writer.write_bits(u64::from(bits - 1), 6).unwrap();
}

fn cases() -> Vec<Case> {
    let base = Case {
        bits: 31,
        colors: 1,
        alpha: 0,
        width: 33,
        height: 5,
        predictor: 0,
        rct: 0,
    };
    let mut cases: Vec<_> = (1..=31).map(|bits| Case { bits, ..base }).collect();
    for (bits, alpha) in [(17, 31), (24, 17), (31, 5), (31, 17), (31, 24), (31, 31)] {
        cases.push(Case {
            bits,
            alpha,
            colors: 3,
            ..base
        });
    }
    for predictor in [5, 6, 13] {
        cases.push(Case {
            predictor,
            width: 257,
            height: 9,
            ..base
        });
    }
    for rct in [6, 41] {
        cases.push(Case {
            bits: 29,
            rct,
            colors: 3,
            width: 129,
            height: 5,
            predictor: 5,
            ..base
        });
    }
    cases
}

/// Locate explicit precision fields by parsing the image grammar. Their fixed-width values and
/// the working-buffer hint change; every frame byte, transform, crop, blend and entropy stream stays.
fn extend_header(data: &[u8], primary_bits: u32) -> Vec<u8> {
    use jxl_bitstream::{Bitstream, U};
    use jxl_image::{AnimationHeader, ExtraChannelInfo, PreviewHeader, SizeHeader};
    use jxl_oxide_common::Bundle;

    fn precision(bits: &mut Bitstream<'_>, replacement: Option<u32>) -> (usize, u32) {
        assert!(!bits.read_bool().unwrap());
        assert_eq!(bits.read_bits(2).unwrap(), 3);
        let offset = bits.num_read_bits();
        let old = bits.read_bits(6).unwrap() + 1;
        assert!((17..=24).contains(&old));
        let value = replacement.unwrap_or(old + 7);
        assert!((17..=31).contains(&value));
        (offset, value - 1)
    }

    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let original = parsed.codestream_inventory(Default::default()).unwrap();
    let data = parsed.codestream();
    let mut bits = Bitstream::new(data);
    assert_eq!(bits.read_bits(16).unwrap(), 0x0aff);
    SizeHeader::parse(&mut bits, ()).unwrap();
    assert!(!bits.read_bool().unwrap()); // explicit metadata
    if bits.read_bool().unwrap() {
        bits.read_bits(3).unwrap(); // orientation
        if bits.read_bool().unwrap() {
            SizeHeader::parse(&mut bits, ()).unwrap();
        }
        if bits.read_bool().unwrap() {
            PreviewHeader::parse(&mut bits, ()).unwrap();
        }
        if bits.read_bool().unwrap() {
            AnimationHeader::parse(&mut bits, ()).unwrap();
        }
    }
    let mut changes = vec![precision(&mut bits, Some(primary_bits))];
    let buffer_hint = bits.num_read_bits();
    bits.read_bool().unwrap();
    let extras = bits.read_u32(0, 1, 2 + U(4), 1 + U(12)).unwrap();
    for _ in 0..extras {
        let mut prefix = bits.clone();
        assert!(!prefix.read_bool().unwrap()); // explicit extra-channel metadata
        prefix.read_u32(0, 1, 2 + U(4), 18 + U(6)).unwrap(); // type enum
        changes.push(precision(&mut prefix, None));
        ExtraChannelInfo::parse(&mut bits, ()).unwrap();
    }
    let mut extended = data.to_vec();
    extended[buffer_hint / 8] &= !(1 << (buffer_hint % 8)); // permit full-width working buffers
    for (offset, value) in changes {
        for bit in 0..6 {
            let index = offset + bit;
            extended[index / 8] = (extended[index / 8] & !(1 << (index % 8)))
                | (((value >> bit) as u8 & 1) << (index % 8));
        }
    }
    let checked = jxl_gpu_bitstream::parse(&extended, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        checked.image_header.bit_depth,
        SampleBitDepth::Integer {
            bits_per_sample: primary_bits
        }
    );
    assert_eq!(checked.frames, original.frames);
    for (extra, previous) in checked
        .image_header
        .extra_channels
        .iter()
        .zip(&original.image_header.extra_channels)
    {
        let mut expected = previous.clone();
        let SampleBitDepth::Integer { bits_per_sample } = previous.bit_depth else {
            unreachable!()
        };
        expected.bit_depth = SampleBitDepth::Integer {
            bits_per_sample: bits_per_sample + 7,
        };
        assert_eq!(extra, &expected);
    }
    extended
}

fn extend_precisions(output: &Path) {
    let extend = |name: &str, bits: u32| {
        let input = offline::unhex(
            &std::fs::read_to_string(output.join(format!("{name}.jxl.hex"))).unwrap(),
        );
        let data = extend_header(&input, bits);
        std::fs::write(
            output.join(format!("{name}_extended{bits}.jxl.hex")),
            hex(&data),
        )
        .unwrap();
    };
    for bits in 18..=31 {
        extend("vardct_extras_integer_rgb", bits);
    }
    for name in [
        "extras_integer_resampled",
        "extras_integer_distributed",
        "vardct_extras_integer_progressive_dc",
        "vardct_extras_integer_resampled",
        "composition_extras_integer_resampled",
        "composition_extras_integer_vardct_resampled",
    ] {
        extend(name, 31);
    }
}

fn main() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| source.join("integer"), PathBuf::from);
    std::fs::create_dir_all(&output).unwrap();
    let temporary =
        std::env::temp_dir().join(format!("jxl-wgpu-integer-generator-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    let binary = temporary.join("precision");
    compile(
        &source.join("generate_integer_samples.c"),
        &binary,
        &["libjxl"],
    );
    let oracle = temporary.join("oracle");
    compile(
        &source.join("decode_extra_channels.c"),
        &oracle,
        &["libjxl", "libjxl_cms"],
    );
    let encoded = temporary.join("encoded.jxl");
    let raw = temporary.join("source.u32.hex");
    for case in cases() {
        let name = case.name();
        run(Command::new(&binary)
            .args([
                case.bits.to_string(),
                case.colors.to_string(),
                case.alpha.to_string(),
                case.width.to_string(),
                case.height.to_string(),
                case.predictor.to_string(),
                case.rct.to_string(),
            ])
            .arg(&encoded)
            .arg(&raw));
        let data = std::fs::read(&encoded).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&data, Default::default()).unwrap();
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        assert_eq!(inventory.frames.len(), 1);
        let offset = inventory.frames[0].header_bits.offset;
        assert_eq!(offset % 8, 0);
        let mut data = case.header();
        data.extend_from_slice(&parsed.codestream()[offset as usize / 8..]);
        let checked = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(
            checked.image_header.bit_depth,
            SampleBitDepth::Integer {
                bits_per_sample: u32::from(case.bits)
            }
        );
        assert_eq!(
            (checked.image_header.width, checked.image_header.height),
            (case.width, case.height)
        );
        assert_eq!(
            checked.image_header.extra_channels.len(),
            usize::from(case.alpha != 0)
        );
        std::fs::write(&encoded, &data).unwrap();
        let reference = run(Command::new(&oracle).arg(&encoded)).stdout;
        assert_eq!(
            reference.len(),
            case.width as usize * case.height as usize * (4 + usize::from(case.alpha != 0)) * 4
        );
        std::fs::write(output.join(format!("{name}.jxl.hex")), hex(&data)).unwrap();
        std::fs::write(
            output.join(format!("{name}.f32.hex")),
            float_hex(&reference),
        )
        .unwrap();
        std::fs::copy(&raw, output.join(format!("{name}.u32.hex"))).unwrap();
        eprintln!("{name}: {} bytes", data.len());
    }
    generate_extras(&source, &output, &temporary, "integer");
    extend_precisions(&output);
    extra_references(&source, &output, &temporary, "integer");
    std::fs::remove_dir_all(temporary).unwrap();
}
