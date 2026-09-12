//! Generate independently encoded base payloads for all normative Modular pass schedules.
use std::path::PathBuf;
use std::process::Command;

use jxl_test_support::fixtures::modular_passes;
use jxl_test_support::offline;

fn main() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data");
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| source.join("modular_passes"));
    std::fs::create_dir_all(&output).unwrap();
    let temporary = std::env::temp_dir().join(format!("jxl-modular-passes-{}", std::process::id()));
    std::fs::create_dir(&temporary).unwrap();
    for (name, width, height, responsive) in
        [("plain", 1025, 3, false), ("squeeze", 2051, 17, true)]
    {
        let input = temporary.join("input.pgm");
        let encoded = temporary.join("encoded.jxl");
        let mut bytes = format!("P5\n{width} {height}\n255\n").into_bytes();
        bytes.extend(modular_passes::samples(width, height));
        std::fs::write(&input, bytes).unwrap();
        let mut command = Command::new("cjxl");
        command.arg(&input).arg(&encoded).args([
            "-d",
            "0",
            "-m",
            "1",
            "-e",
            "9",
            "-R",
            if responsive { "1" } else { "0" },
            "-x",
            "color_space=Gra_D65_Rel_SRG",
            "--container=0",
        ]);
        if responsive {
            command.arg("-p");
        }
        offline::run(&mut command);
        let data = std::fs::read(encoded).unwrap();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(
            inventory.frames[0].num_passes,
            if responsive { 2 } else { 1 }
        );
        eprintln!(
            "{name}: {} bytes, passes {:?}, flags {}",
            data.len(),
            inventory.frames[0].progressive_passes,
            inventory.frames[0].flags
        );
        std::fs::write(output.join(format!("{name}.jxl.hex")), offline::hex(&data)).unwrap();
    }
    std::fs::remove_dir_all(temporary).unwrap();
}
