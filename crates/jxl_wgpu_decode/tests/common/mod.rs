#![allow(dead_code)]

use std::sync::LazyLock;

pub fn cjxl_local_tree_codestream() -> Option<Vec<u8>> {
    if std::process::Command::new("cjxl")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping local-tree VarDCT oracle: cjxl is not installed");
        return None;
    }
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let ppm_path = std::env::temp_dir().join(format!("jxl-wgpu-local-tree-{nonce}.ppm"));
    let jxl_path = std::env::temp_dir().join(format!("jxl-wgpu-local-tree-{nonce}.jxl"));
    let width = 2056_u32;
    let height = 256_u32;
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.reserve((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            ppm.extend_from_slice(&[
                (x.wrapping_mul(13) + y.wrapping_mul(7)) as u8,
                (x.wrapping_mul(3) ^ y.wrapping_mul(11)) as u8,
                (x.wrapping_mul(5) + y.wrapping_mul(17) + (x ^ y)) as u8,
            ]);
        }
    }
    std::fs::write(&ppm_path, ppm).unwrap();
    let output = std::process::Command::new("cjxl")
        .args(["-d", "2", "-e", "7", "--container=0"])
        .arg(&ppm_path)
        .arg(&jxl_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&ppm_path);
    if !output.status.success() {
        let _ = std::fs::remove_file(&jxl_path);
        panic!("cjxl failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let codestream = std::fs::read(&jxl_path).unwrap();
    let _ = std::fs::remove_file(&jxl_path);
    Some(codestream)
}

pub fn cjxl_progressive_dc_codestream(level: u8) -> Option<Vec<u8>> {
    assert!((1..=2).contains(&level));
    if std::process::Command::new("cjxl")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping progressive-DC oracle: cjxl is not installed");
        return None;
    }
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let ppm_path = std::env::temp_dir().join(format!("jxl-wgpu-progressive-dc-{nonce}.ppm"));
    let jxl_path = std::env::temp_dir().join(format!("jxl-wgpu-progressive-dc-{nonce}.jxl"));
    let (width, height) = (1_024_u32, 128_u32);
    let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.reserve((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            ppm.extend_from_slice(&[
                (x.wrapping_mul(13) + y.wrapping_mul(7)) as u8,
                (x.wrapping_mul(3) ^ y.wrapping_mul(11)) as u8,
                (x.wrapping_mul(5) + y.wrapping_mul(17) + (x ^ y)) as u8,
            ]);
        }
    }
    std::fs::write(&ppm_path, ppm).unwrap();
    let progressive_dc = format!("--progressive_dc={level}");
    let output = std::process::Command::new("cjxl")
        .args(["-d", "2", "-e", "7", "-m", "0", "--container=0"])
        .arg(progressive_dc)
        .arg(&ppm_path)
        .arg(&jxl_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&ppm_path);
    if !output.status.success() {
        let _ = std::fs::remove_file(&jxl_path);
        panic!("cjxl failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let codestream = std::fs::read(&jxl_path).unwrap();
    let _ = std::fs::remove_file(&jxl_path);
    Some(codestream)
}

fn nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid checked-in fixture hex digit"),
    }
}

fn decode_hex(input: &str) -> Vec<u8> {
    let digits = input
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect::<Vec<_>>();
    assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
    digits
        .chunks_exact(2)
        .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
        .collect()
}

pub fn basic() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/basic.jxl.hex")));
    BYTES.as_slice()
}

pub fn gpu_gray8_lossless() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/gpu_gray8_lossless.jxl.hex")));
    BYTES.as_slice()
}

pub fn testsrc_modular_weighted() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_modular_weighted.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn fragmented_animation() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/fragmented_animation.jxl.hex")));
    BYTES.as_slice()
}

pub fn green_queen_vardct_nonzero_ac() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_vardct_nonzero_ac.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn green_queen_vardct_permuted() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_vardct_permuted.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn green_queen_vardct_mixed() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_vardct_mixed.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn green_queen_vardct_gaborish() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_vardct_gaborish.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn green_queen_crop_vardct_epf2() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_crop_vardct_epf2.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn green_queen_crop_vardct_epf3() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_crop_vardct_epf3.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn testsrc_vardct_multi_lf() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_multi_lf.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn testsrc_vardct_multi_lf_skip_smoothing() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_multi_lf_skip_smoothing.jxl.hex"
        ))
    });
    BYTES.as_slice()
}
