use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "jxl-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn available(tool: &str) -> bool {
    let output = Command::new(tool).arg("--version").output();
    if let Ok(output) = output
        && output.status.success()
    {
        eprintln!(
            "metadata native oracle: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return true;
    }
    assert!(
        std::env::var_os("JXL_REQUIRE_NATIVE_ORACLES").is_none(),
        "required native oracle {tool} is unavailable"
    );
    eprintln!("skipping metadata native oracle: {tool} is unavailable");
    false
}

fn command(tool: &str, args: &[&str], input: &Path) -> Vec<u8> {
    let result = Command::new(tool).args(args).arg(input).output().unwrap();
    assert!(
        result.status.success(),
        "{tool}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

#[test]
fn google_brotli_interoperates_in_both_directions_at_every_quality_and_window() {
    if !available("brotli") {
        return;
    }
    let directory = TemporaryDirectory::new();
    let input_path = directory.0.join("metadata.bin");
    let compressed_path = directory.0.join("metadata.br");
    let limits = MetadataLimits {
        max_expansion_ratio: u32::MAX,
        ..MetadataLimits::default()
    };
    let mut pairs = 0;
    for quality in 0..=11 {
        for window in 10..=24 {
            let count = [0, 1, 257, 16_385, 65_537][(quality as usize + window as usize) % 5];
            let input = (0..count)
                .map(|n| ((n * 131 + n / 19 + (n >> 9)) % 251) as u8)
                .collect::<Vec<_>>();
            std::fs::write(&input_path, &input).unwrap();
            let native = command(
                "brotli",
                &["-c", "-q", &quality.to_string(), "-w", &window.to_string()],
                &input_path,
            );
            let mut wire = JUMBF.to_vec();
            wire.extend_from_slice(&native);
            let decoded = MetadataBox::from_encoded(
                ContainerBoxRef {
                    box_type: BROB,
                    payload: &wire,
                },
                limits,
            )
            .unwrap();
            assert_eq!(
                decoded.decode(limits).unwrap().as_ref(),
                input,
                "native q{quality} w{window}"
            );
            let encoded = MetadataBox::new(
                JUMBF,
                &input,
                MetadataCompression::Brotli(BrotliOptions::new(quality, window).unwrap()),
                limits,
            )
            .unwrap();
            std::fs::write(&compressed_path, &encoded.payload[4..]).unwrap();
            assert_eq!(
                command("brotli", &["-d", "-c"], &compressed_path),
                input,
                "Rust q{quality} w{window}"
            );
            pairs += 1;
        }
    }
    assert_eq!(pairs, 180);
    eprintln!("metadata native Brotli: {pairs} encode/decode pairs passed");
}

#[test]
fn libjxl_extracts_rewritten_exif_xmp_and_jumbf_payloads() {
    if !available("djxl") || !available("clang++") || !available("pkg-config") {
        return;
    }
    let directory = TemporaryDirectory::new();
    let flags = Command::new("pkg-config")
        .args(["--cflags", "--libs", "libjxl"])
        .output()
        .unwrap();
    assert!(
        flags.status.success(),
        "libjxl development package is required"
    );
    let oracle = directory.0.join("raw-box-oracle");
    let compiled = Command::new("clang++")
        .args(["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/metadata_oracle/main.cpp"))
        .args(
            std::str::from_utf8(&flags.stdout)
                .unwrap()
                .split_whitespace(),
        )
        .arg("-o")
        .arg(&oracle)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let source = crate::test_fixtures::basic();
    let source = crate::parse(&source, crate::ParseLimits::default()).unwrap();
    // A complete little-endian TIFF IFD declares orientation 8; rendering still uses codestream.
    let exif = [
        0, 0, 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0,
        0, 0, 0,
    ];
    let xmp = b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><opaque>JPEG XL metadata</opaque></x:xmpmeta>";
    let jumbf = b"\0\0\0\x0cjson{\"v\":1}";
    for compressed in [false, true] {
        let mut metadata = Metadata::default();
        for (kind, payload) in [(EXIF, &exif[..]), (XMP, &xmp[..]), (JUMBF, &jumbf[..])] {
            metadata
                .push(item(kind, payload, compressed), MetadataLimits::default())
                .unwrap();
        }
        let encoded = metadata.write_container(source.codestream()).unwrap();
        let path = directory.0.join("image.jxl");
        std::fs::write(&path, &encoded).unwrap();
        let status = Command::new(&oracle)
            .arg(&path)
            .arg(&directory.0)
            .output()
            .unwrap();
        assert!(
            status.status.success(),
            "{}",
            String::from_utf8_lossy(&status.stderr)
        );
        for (extension, expected) in [
            ("exif", &exif[..]),
            ("xmp", &xmp[..]),
            ("jumbf", &jumbf[..]),
        ] {
            let output = directory.0.join(format!("{extension}.bin"));
            assert_eq!(
                std::fs::read(&output).unwrap(),
                expected,
                "{extension} compressed={compressed}"
            );
        }
        let parsed = crate::parse(&encoded, crate::ParseLimits::default()).unwrap();
        assert_eq!(
            parsed
                .codestream_inventory(crate::InventoryLimits::default())
                .unwrap(),
            source
                .codestream_inventory(crate::InventoryLimits::default())
                .unwrap()
        );
    }
    eprintln!("metadata native libjxl: 6 extracted payloads and unchanged inventories passed");
}
