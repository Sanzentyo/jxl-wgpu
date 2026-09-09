//! Generate compact libjxl standalone-layer headers, without duplicating fixture entropy.
//! `cargo run -p jxl_wgpu_decode --example regenerate_progressive_composition`
use std::path::Path;
use std::process::Command;

use progressive_layers::hex;
#[path = "../tests/common/progressive_layers.rs"]
mod progressive_layers;

fn run(command: &mut Command) -> Vec<u8> {
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("progressive_composition");
    std::fs::create_dir_all(&output).unwrap();
    let temporary =
        std::env::temp_dir().join(format!("jxl-progressive-headers-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    let binary = temporary.join("generate");
    let flags = run(Command::new("pkg-config").args(["--cflags", "--libs", "libjxl"]));
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-O2"])
        .arg(data.join("generate_frame_composition.c"))
        .args(std::str::from_utf8(&flags).unwrap().split_whitespace())
        .arg("-o")
        .arg(&binary));
    run(Command::new(&binary)
        .arg(&temporary)
        .arg("--progressive-layers"));
    for name in ["vardct", "vardct_gray", "vardct_dc"] {
        let source = hex::unhex(
            &std::fs::read_to_string(data.join(format!("composition_{name}.jxl.hex"))).unwrap(),
        );
        let inventory = jxl_gpu_bitstream::parse(&source, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let mut first = 0;
        let mut layer = 0;
        for (index, frame) in inventory.frames.iter().enumerate() {
            if frame.frame_type == jxl_gpu_bitstream::FrameType::LowFrequency {
                continue;
            }
            let standalone = hex::unhex(
                &std::fs::read_to_string(
                    temporary.join(format!("composition_{name}_layer{layer}.jxl.hex")),
                )
                .unwrap(),
            );
            let headers = progressive_layers::headers(&standalone);
            progressive_layers::reframe(&source, &inventory.frames[first..=index], &headers);
            std::fs::write(output.join(format!("{name}_layer{layer}.headers")), headers).unwrap();
            first = index + 1;
            layer += 1;
        }
        assert_eq!(layer, 9);
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
