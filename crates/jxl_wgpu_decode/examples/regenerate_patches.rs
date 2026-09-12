//! Reframe native image entropy into pre-transform references and independently coded patches.
use std::path::Path;
use std::process::Command;

#[allow(dead_code)]
#[path = "support/offline.rs"]
mod offline;
#[allow(dead_code)]
#[path = "../tests/support/patches.rs"]
mod patches;

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = data.join("patches");
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!("jxl-patches-{}", std::process::id()));
    std::fs::create_dir_all(&temporary).unwrap();
    let oracle = temporary.join("oracle");
    offline::compile(
        &data.join("decode_extra_channels.c"),
        &oracle,
        &["libjxl", "libjxl_cms"],
    );
    for (name, source) in [
        ("modular", "testsrc_modular_orientation_rgb_1"),
        ("gray", "testsrc_modular_orientation_gray_1"),
        ("alpha", "testsrc_modular_orientation_rgba_16"),
        ("xyb_modular", "lossy_modular/extras_lossy_epf0"),
        ("xyb_vardct", "vardct_extras_transformed"),
        ("float", "floating/extras_float_rgb"),
        ("float_vardct", "floating/vardct_extras_float_rgb"),
        ("associated", "floating/extras_float_associated"),
        (
            "associated_vardct",
            "floating/vardct_extras_float_associated",
        ),
    ] {
        let source = offline::unhex(
            &std::fs::read_to_string(data.join(format!("{source}.jxl.hex"))).unwrap(),
        );
        let inventory = jxl_gpu_bitstream::parse(&source, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        eprintln!(
            "{name}: {:?}, {} passes, {:?}",
            inventory.frames[0].encoding,
            inventory.frames[0].num_passes,
            inventory.frames[0].restoration_filter
        );
        for (suffix, count) in [("", 16), ("_empty", 0)] {
            let name = format!("{name}{suffix}");
            let values = patches::values(
                &inventory.frames[0],
                inventory.image_header.extra_channels.len(),
                count,
            );
            let encoded = patches::assemble(&source, &values);
            let path = temporary.join("input.jxl");
            std::fs::write(&path, &encoded).unwrap();
            let reference =
                offline::run(Command::new(&oracle).arg(&path).arg("--preserve-alpha")).stdout;
            std::fs::write(
                output.join(format!("{name}.jxl.hex")),
                offline::hex(&encoded),
            )
            .unwrap();
            std::fs::write(
                output.join(format!("{name}.f32.hex")),
                offline::float_hex(&reference),
            )
            .unwrap();
            let linear = offline::run(
                Command::new(&oracle)
                    .arg(&path)
                    .args(["--preserve-alpha", "--linear"]),
            )
            .stdout;
            std::fs::write(
                output.join(format!("{name}.linear.f32.hex")),
                offline::float_hex(&linear),
            )
            .unwrap();
        }
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
