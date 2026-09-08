use std::fmt::Write as _;
use std::path::Path;
use std::process::{Command, Output};
pub fn run(command: &mut Command) -> Output {
    let output = command.output().expect("run offline fixture tool");
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub fn compile(source: &Path, binary: &Path, libraries: &[&str]) {
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

pub fn hex(bytes: &[u8]) -> String {
    let mut text = String::new();
    for line in bytes.chunks(32) {
        for byte in line {
            write!(text, "{byte:02x}").unwrap();
        }
        text.push('\n');
    }
    text
}

pub fn unhex(text: &str) -> Vec<u8> {
    let hex = text.split_whitespace().collect::<String>();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

pub fn float_hex(bytes: &[u8]) -> String {
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

pub fn generate_extras(source: &Path, output: &Path, temporary: &Path, mode: &str) {
    for name in ["generate_extra_channels", "generate_extra_composition"] {
        let binary = temporary.join(name);
        compile(&source.join(format!("{name}.c")), &binary, &["libjxl"]);
        run(Command::new(binary).arg(output).arg(format!("--{mode}")));
    }
}

pub fn extra_references(source: &Path, output: &Path, temporary: &Path, mode: &str) {
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
            path.file_name().unwrap().to_str().is_some_and(|name| {
                name.contains(if mode == "floating" {
                    "_float_"
                } else {
                    "_integer_"
                }) && name.ends_with(".jxl.hex")
            })
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
