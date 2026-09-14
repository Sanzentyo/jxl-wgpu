use super::*;
use std::process::Command;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn libjxl_and_libultrahdr_read_rust_serialized_bundles_and_exact_fractions() {
    let Some(executable) = std::env::var_os("JXL_GAIN_MAP_ORACLE") else {
        assert!(
            std::env::var_os("JXL_REQUIRE_NATIVE_ORACLES").is_none(),
            "JXL_GAIN_MAP_ORACLE is required"
        );
        eprintln!(
            "skipping native gain-map interoperability; set JXL_GAIN_MAP_ORACLE to the pinned helper"
        );
        return;
    };
    let scratch = Scratch(std::env::temp_dir().join(
        format!("jxl-gain-map-{}-{}", std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
    ));
    std::fs::create_dir(&scratch.0).unwrap();
    let roundtrip = |operation: &str, bytes: &[u8]| {
        let input = scratch.0.join("input");
        let output = scratch.0.join("output");
        std::fs::write(&input, bytes).unwrap();
        let result = Command::new(&executable)
            .arg(operation)
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::read(&output).unwrap()
    };
    let cases = std::fs::read_to_string(directory().join("cases.txt")).unwrap();
    let mut count = 0;
    for line in cases.lines() {
        let name = line.split_whitespace().next().unwrap();
        let bytes = std::fs::read(directory().join(format!("{name}.jxl"))).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let native = parsed
            .auxiliary_boxes()
            .iter()
            .find(|b| b.box_type == JHGM)
            .unwrap();
        let bundle = GainMapBundle::parse(native.payload, Default::default()).unwrap();
        let rust = bundle.encode(Default::default()).unwrap();
        assert_eq!(roundtrip("bundle", &rust), rust, "{name}");
        for backward in [false, true] {
            let metadata = GainMapMetadata {
                backward_direction: backward,
                ..*bundle.metadata()
            };
            let bytes = metadata.encode().unwrap();
            let native = roundtrip("iso", &bytes);
            assert_eq!(GainMapMetadata::parse(&native).unwrap(), metadata);
            assert_eq!(native, bytes);
        }
        count += 1;
    }
    assert_eq!(count, 64);
    eprintln!(
        "native gain-map: {count} Rust/libjxl bundle roundtrips and {} Rust/libultrahdr ISO roundtrips",
        count * 2
    );
}
