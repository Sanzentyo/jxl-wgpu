//! Offline libjxl fixture generation; this example never invokes the production GPU decoder.
//! Run `cargo run -p jxl_wgpu_decode --example regenerate_floating -- [output-directory]`.

#[path = "support/offline.rs"]
mod offline;
use offline::{compile, extra_references, generate_extras, hex, run};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    generate_extras(&source, &output, &temporary, "floating");
    extra_references(&source, &output, &temporary, "floating");
    std::fs::remove_dir_all(temporary).unwrap();
    eprintln!(
        "Regenerated 154 precision and 27 rendering fixtures with independent binary32 references in {}",
        output.display()
    );
}
