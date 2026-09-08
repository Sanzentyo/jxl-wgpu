//! Offline libjxl fixture generation; this example never invokes the production GPU decoder.
//! Run `cargo run -p jxl_wgpu_decode --example regenerate_floating -- [output-directory]`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn run(command: &mut Command) -> Output {
    let output = command.output().expect("run offline fixture tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn compile(source: &Path, binary: &Path, libraries: &[&str]) {
    let flags = run(Command::new("pkg-config")
        .args(["--cflags", "--libs"])
        .args(libraries));
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(source)
        .args(
            std::str::from_utf8(&flags.stdout)
                .unwrap()
                .split_whitespace(),
        )
        .arg("-o")
        .arg(binary));
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::new();
    for line in bytes.chunks(32) {
        for byte in line {
            write!(text, "{byte:02x}").unwrap();
        }
        text.push('\n');
    }
    text
}

fn unhex(text: &str) -> Vec<u8> {
    let hex = text.split_whitespace().collect::<String>();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn float_hex(bytes: &[u8]) -> String {
    assert!(bytes.len().is_multiple_of(4));
    let mut text = String::new();
    for line in bytes.chunks(32) {
        for (index, word) in line.chunks_exact(4).enumerate() {
            if index != 0 {
                text.push(' ');
            }
            write!(text, "{:08x}", u32::from_le_bytes(word.try_into().unwrap())).unwrap();
        }
        text.push('\n');
    }
    text
}

fn extra_fixtures(source: &Path, output: &Path, temporary: &Path) {
    for name in ["generate_extra_channels", "generate_extra_composition"] {
        let binary = temporary.join(name);
        compile(&source.join(format!("{name}.c")), &binary, &["libjxl"]);
        run(Command::new(binary).arg(output).arg("--floating"));
    }
    let oracle = temporary.join("oracle");
    compile(
        &source.join("decode_extra_channels.c"),
        &oracle,
        &["libjxl", "libjxl_cms"],
    );
    let encoded = temporary.join("oracle-input.jxl");
    let mut paths: Vec<_> = std::fs::read_dir(output)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_str()
                .is_some_and(|name| name.contains("_float_") && name.ends_with(".jxl.hex"))
        })
        .collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix(".jxl.hex")
            .unwrap();
        let data = unhex(&std::fs::read_to_string(&path).unwrap());
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let frame_bytes = pixels * (4 + image.extra_channels.len()) * 4;
        std::fs::write(&encoded, data).unwrap();
        for (suffix, options) in [
            ("f32.hex", &[][..]),
            ("spots.f32.hex", &["--render-spots"][..]),
            ("associated.f32.hex", &["--preserve-alpha"][..]),
        ] {
            let reference = run(Command::new(&oracle).arg(&encoded).args(options)).stdout;
            assert!(!reference.is_empty() && reference.len().is_multiple_of(frame_bytes));
            let values = if options.is_empty() {
                reference
            } else {
                reference
                    .chunks_exact(frame_bytes)
                    .flat_map(|frame| frame[..pixels * 16].iter().copied())
                    .collect()
            };
            std::fs::write(output.join(format!("{name}.{suffix}")), float_hex(&values)).unwrap();
        }
    }
}

/// Frame boundaries come from the codestream parser, never hard-coded header offsets.
fn transplant_frame(header: &[u8], frame: &[u8]) -> Vec<u8> {
    let header = jxl_gpu_bitstream::parse(header, Default::default()).unwrap();
    let frame = jxl_gpu_bitstream::parse(frame, Default::default()).unwrap();
    let hi = header.codestream_inventory(Default::default()).unwrap();
    let fi = frame.codestream_inventory(Default::default()).unwrap();
    assert_eq!(
        (hi.image_header.width, hi.image_header.height),
        (fi.image_header.width, fi.image_header.height)
    );
    assert_eq!(hi.frames.len(), 1);
    assert_eq!(fi.frames.len(), 1);
    let h = hi.frames[0].header_bits.offset;
    let f = fi.frames[0].header_bits.offset;
    assert_eq!((h % 8, f % 8), (0, 0));
    let mut data = header.codestream()[..(h / 8) as usize].to_vec();
    data.extend_from_slice(&frame.codestream()[(f / 8) as usize..]);
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.image_header.bit_depth, hi.image_header.bit_depth);
    data
}

fn main() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map_or_else(|| source.join("floating"), PathBuf::from);
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!(
        "jxl-wgpu-floating-generator-{}",
        std::process::id()
    ));
    std::fs::create_dir(&temporary).unwrap();
    let binary = temporary.join("precision");
    compile(
        &source.join("generate_floating_samples.c"),
        &binary,
        &["libjxl"],
    );
    let encoded = temporary.join("encoded.jxl");
    for exponent in 2..=8 {
        for mantissa in 2..=23 {
            let bits = exponent + mantissa + 1;
            let encode = |mode: &str, path: &Path| {
                run(Command::new(&binary)
                    .arg("encode")
                    .arg(bits.to_string())
                    .arg(exponent.to_string())
                    .arg(mode)
                    .arg(path));
            };
            if exponent == 8 && bits != 32 {
                let header = temporary.join("header.jxl");
                let frame = temporary.join("frame.jxl");
                encode("header", &header);
                encode("words", &frame);
                let data = transplant_frame(
                    &std::fs::read(header).unwrap(),
                    &std::fs::read(frame).unwrap(),
                );
                std::fs::write(&encoded, data).unwrap();
            } else {
                encode("samples", &encoded);
            }
            let expected = run(Command::new(&binary).arg("decode").arg(&encoded));
            assert_eq!(
                std::str::from_utf8(&expected.stdout)
                    .unwrap()
                    .lines()
                    .count(),
                120
            );
            std::fs::write(
                output.join(format!("{bits}-{exponent}.jxl.hex")),
                hex(&std::fs::read(&encoded).unwrap()),
            )
            .unwrap();
            std::fs::write(
                output.join(format!("{bits}-{exponent}.f32.hex")),
                expected.stdout,
            )
            .unwrap();
        }
    }
    extra_fixtures(&source, &output, &temporary);
    std::fs::remove_dir_all(temporary).unwrap();
    eprintln!(
        "Regenerated 154 precision and 27 rendering fixtures with independent binary32 references in {}",
        output.display()
    );
}
