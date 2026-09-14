use super::*;
use std::process::Command;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn libjxl_and_libavif_read_rust_serialized_bundles_and_exact_fractions() {
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
    let native = |operation: &str, bytes: &[u8]| {
        let input = scratch.0.join("input");
        let output = scratch.0.join("output");
        std::fs::write(&input, bytes).unwrap();
        Command::new(&executable)
            .arg(operation)
            .arg(&input)
            .arg(&output)
            .output()
            .unwrap()
    };
    let roundtrip = |operation: &str, bytes: &[u8]| {
        let result = native(operation, bytes);
        assert!(
            result.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        std::fs::read(scratch.0.join("output")).unwrap()
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
        for descending_headroom in [false, true] {
            let mut metadata = bundle.metadata().clone();
            if descending_headroom {
                std::mem::swap(
                    &mut metadata.base_hdr_headroom,
                    &mut metadata.alternate_hdr_headroom,
                );
            }
            let bytes = metadata.encode().unwrap();
            let native = roundtrip("iso", &bytes);
            assert_eq!(GainMapMetadata::parse(&native).unwrap(), metadata);
            assert_eq!(native, bytes);
        }
        count += 1;
    }
    assert_eq!(count, 64);
    let version_zero = GainMapMetadata::default().encode().unwrap();
    for writer in [1, 255, u16::MAX] {
        for extension in [&[][..], &[1, 0, 255, 17][..]] {
            let mut bytes = version_zero.clone();
            bytes[2..4].copy_from_slice(&writer.to_be_bytes());
            bytes.extend_from_slice(extension);
            assert_eq!(
                GainMapMetadata::parse(&bytes).unwrap().encode().unwrap(),
                bytes
            );
            // libavif consumes compatible extensions but writes a fresh version-zero record.
            assert_eq!(roundtrip("iso", &bytes), version_zero);
        }
    }
    let mut invalid = Vec::new();
    for index in [0, 1] {
        let mut bytes = version_zero.clone();
        bytes[index] = 1;
        invalid.push(bytes);
    }
    for offset in [9, 17, 25, 33, 37, 41, 49, 57] {
        let mut bytes = version_zero.clone();
        bytes[offset..offset + 4].fill(0);
        invalid.push(bytes);
    }
    let mut trailing = version_zero.clone();
    trailing.push(0);
    invalid.push(trailing);
    for bytes in invalid {
        assert!(GainMapMetadata::parse(&bytes).is_err());
        assert!(!native("iso", &bytes).status.success());
    }
    eprintln!(
        "native gain-map: {count} Rust/libjxl bundle roundtrips, {} Rust/libavif ISO roundtrips, 6 compatible-writer reads and 11 shared invalid-record rejections",
        count * 2
    );
}
