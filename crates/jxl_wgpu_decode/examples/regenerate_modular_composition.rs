//! Native two-pass Modular animations and independent standalone-layer header oracles.
use std::path::PathBuf;
use std::process::Command;

use jxl_test_support::fixtures::progressive_layers as layers;
use jxl_test_support::offline::process;

fn main() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("modular_composition"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary =
        std::env::temp_dir().join(format!("jxl-modular-composition-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    let binary = temporary.join("generate");
    process::compile(
        &source.join("generate_frame_composition.c"),
        &binary,
        &["libjxl"],
    );
    process::run(
        Command::new(binary)
            .arg(&temporary)
            .arg("--modular-progressive"),
    );
    for name in [
        "modular_pass_rgb",
        "modular_pass_gray_alpha",
        "modular_pass_float",
    ] {
        let hex =
            std::fs::read_to_string(temporary.join(format!("composition_{name}.jxl.hex"))).unwrap();
        let data = layers::hex::unhex(&hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(inventory.frames.len(), 9);
        for (index, frame) in inventory.frames.iter().enumerate() {
            assert_eq!(frame.num_passes, 2);
            let standalone = layers::hex::unhex(
                &std::fs::read_to_string(
                    temporary.join(format!("composition_{name}_layer{index}.jxl.hex")),
                )
                .unwrap(),
            );
            let headers = layers::headers(&standalone);
            let reframed = layers::reframe(&data, std::slice::from_ref(frame), &headers);
            assert_eq!(
                reframed,
                jxl_gpu_bitstream::parse(&standalone, Default::default())
                    .unwrap()
                    .codestream(),
                "standalone metadata preserves every native entropy byte"
            );
            std::fs::write(output.join(format!("{name}_layer{index}.headers")), headers).unwrap();
        }
        std::fs::write(
            output.join(format!("{name}.jxl.hex")),
            layers::hex::hex(&data),
        )
        .unwrap();
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
