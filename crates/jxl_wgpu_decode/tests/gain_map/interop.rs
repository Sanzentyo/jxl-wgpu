use super::*;

#[test]
fn libjxl_and_libavif_read_rust_serialized_bundles_and_exact_fractions() {
    let Some(oracle) = super::native::Oracle::new() else {
        return;
    };
    let roundtrip = |operation, bytes: &[u8]| oracle.roundtrip(operation, bytes);
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
        assert!(!oracle.run("iso", &bytes).status.success());
    }
    eprintln!(
        "native gain-map: {count} Rust/libjxl bundle roundtrips, {} Rust/libavif ISO roundtrips, 6 compatible-writer reads and 11 shared invalid-record rejections",
        count * 2
    );
}
