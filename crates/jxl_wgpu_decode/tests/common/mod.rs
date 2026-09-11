#![allow(dead_code)]

use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temp_file_nonce() -> String {
    format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    )
}

pub fn cjxl_local_tree_codestream() -> Option<Vec<u8>> {
    if std::process::Command::new("cjxl")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("skipping local-tree VarDCT oracle: cjxl is not installed");
        return None;
    }
    let nonce = temp_file_nonce();
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
    let nonce = temp_file_nonce();
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

pub fn jpeg_transcode_raw_matrix() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/jpeg_transcode_raw_matrix.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn jpeg_transcode_raw_matrix_local() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/jpeg_transcode_raw_matrix_local.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn jpeg_transcode_raw_matrix_local_packets() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/jpeg_transcode_raw_matrix_local_packets.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn jpeg_transcode_444() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/jpeg_transcode_444.jxl.hex")));
    BYTES.as_slice()
}

pub fn jpeg_transcode_422() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/jpeg_transcode_422.jxl.hex")));
    BYTES.as_slice()
}

pub fn jpeg_transcode_440() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> =
        LazyLock::new(|| decode_hex(include_str!("../../test-data/jpeg_transcode_440.jxl.hex")));
    BYTES.as_slice()
}

pub fn vardct_progressive_spectral() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_progressive_spectral.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn vardct_progressive_quantized() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_progressive_quantized.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn vardct_progressive_multilf() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_progressive_multilf.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn vardct_progressive_dc_ac() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/testsrc_vardct_progressive_dc_ac.jxl.hex"
        ))
    });
    BYTES.as_slice()
}

pub fn vardct_upsampling(name: &str) -> Vec<u8> {
    decode_hex(match name {
        "2" => include_str!("../../test-data/testsrc_vardct_upsample_2.jxl.hex"),
        "4" => include_str!("../../test-data/testsrc_vardct_upsample_4.jxl.hex"),
        "8" => include_str!("../../test-data/testsrc_vardct_upsample_8.jxl.hex"),
        "8_custom" => include_str!("../../test-data/testsrc_vardct_upsample_8_custom.jxl.hex"),
        "2_multilf" => include_str!("../../test-data/testsrc_vardct_upsample_2_multilf.jxl.hex"),
        "4_thin" => include_str!("../../test-data/testsrc_vardct_upsample_4_thin.jxl.hex"),
        "8_single" => include_str!("../../test-data/testsrc_vardct_upsample_8_single.jxl.hex"),
        _ => panic!("unknown upsampling fixture: {name}"),
    })
}

pub fn vardct_orientation(value: u32) -> Vec<u8> {
    decode_hex(match value {
        1 => include_str!("../../test-data/testsrc_vardct_orientation_1.jxl.hex"),
        2 => include_str!("../../test-data/testsrc_vardct_orientation_2.jxl.hex"),
        3 => include_str!("../../test-data/testsrc_vardct_orientation_3.jxl.hex"),
        4 => include_str!("../../test-data/testsrc_vardct_orientation_4.jxl.hex"),
        5 => include_str!("../../test-data/testsrc_vardct_orientation_5.jxl.hex"),
        6 => include_str!("../../test-data/testsrc_vardct_orientation_6.jxl.hex"),
        7 => include_str!("../../test-data/testsrc_vardct_orientation_7.jxl.hex"),
        8 => include_str!("../../test-data/testsrc_vardct_orientation_8.jxl.hex"),
        _ => panic!("unknown orientation fixture: {value}"),
    })
}

pub fn vardct_gray(name: &str) -> Vec<u8> {
    decode_hex(match name {
        "single" => include_str!("../../test-data/testsrc_vardct_gray_single.jxl.hex"),
        "progressive" => include_str!("../../test-data/testsrc_vardct_gray_progressive.jxl.hex"),
        "upsample" => include_str!("../../test-data/testsrc_vardct_gray_upsample.jxl.hex"),
        "multilf" => include_str!("../../test-data/testsrc_vardct_gray_multilf.jxl.hex"),
        "dc_ac" => include_str!("../../test-data/testsrc_vardct_gray_dc_ac.jxl.hex"),
        "jpeg" => include_str!("../../test-data/testsrc_vardct_gray_jpeg.jxl.hex"),
        _ => panic!("unknown grayscale fixture: {name}"),
    })
}

pub fn vardct_oriented_jpeg() -> Vec<u8> {
    decode_hex(include_str!(
        "../../test-data/testsrc_vardct_jpeg_orientation_6.jxl.hex"
    ))
}

pub fn vardct_depth_rgb(bits: u32) -> Vec<u8> {
    const FIXTURES: [&str; 16] = [
        include_str!("../../test-data/testsrc_vardct_depth_rgb_1.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_2.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_3.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_4.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_5.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_6.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_7.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_8.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_9.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_10.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_11.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_12.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_13.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_14.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_15.jxl.hex"),
        include_str!("../../test-data/testsrc_vardct_depth_rgb_16.jxl.hex"),
    ];
    assert!((1..=16).contains(&bits));
    decode_hex(FIXTURES[(bits - 1) as usize])
}

pub fn vardct_depth_combined(name: &str) -> Vec<u8> {
    decode_hex(match name {
        "gray_12_upsample" => {
            include_str!("../../test-data/testsrc_vardct_depth_gray_12_upsample.jxl.hex")
        }
        "gray_16_dc" => include_str!("../../test-data/testsrc_vardct_depth_gray_16_dc.jxl.hex"),
        "rgb_16_multilf" => {
            include_str!("../../test-data/testsrc_vardct_depth_rgb_16_multilf.jxl.hex")
        }
        "rgb_16_single" => {
            include_str!("../../test-data/testsrc_vardct_depth_rgb_16_single.jxl.hex")
        }
        _ => panic!("unknown integer-depth fixture: {name}"),
    })
}

pub fn with_custom_upsampling_weights(data: &[u8]) -> Vec<u8> {
    use jxl_gpu_bitstream::BitWriter;
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let codestream = parsed.codestream();
    let data: &[u8] = codestream;
    let end = inventory.image_header.bit_range.end().unwrap() as usize;
    let default_transform = data[(end - 1) / 8] & (1 << ((end - 1) % 8)) != 0;
    let prefix_end = end - if default_transform { 1 } else { 3 };
    if !default_transform {
        for bit in prefix_end..end {
            assert_eq!(
                (data[bit / 8] >> (bit % 8)) & 1,
                0,
                "fixture has no custom weights"
            );
        }
    }
    let mut writer = BitWriter::new();
    for bit in 0..prefix_end {
        writer
            .write_bits(u64::from((data[bit / 8] >> (bit % 8)) & 1), 1)
            .unwrap();
    }
    if default_transform {
        writer.write_bits(0, 1).unwrap(); // Explicit transform data.
        if inventory.image_header.xyb_encoded {
            writer.write_bits(1, 1).unwrap();
        } // Default opsin matrix.
    }
    writer.write_bits(7, 3).unwrap(); // All three custom kernels.
    for _ in 0..15 + 55 + 210 {
        writer.write_bits(0x2800, 16).unwrap();
    } // Exact binary16 1/32.
    writer.align_to_byte().unwrap();
    let mut result = writer.into_bytes();
    result.extend_from_slice(&data[end.div_ceil(8)..]);
    result
}

pub fn vardct_progressive_raw_matrix() -> Vec<u8> {
    decode_hex(include_str!(
        "../../test-data/testsrc_vardct_progressive_raw_matrix.jxl.hex"
    ))
}

pub fn jpeg_dc_edge_case(selectors: &str) -> Vec<u8> {
    decode_hex(match selectors {
        "003" => include_str!("../../test-data/jpeg_sampling/odd_003.jxl.hex"),
        "321" => include_str!("../../test-data/jpeg_sampling/odd_321.jxl.hex"),
        "111" => include_str!("../../test-data/jpeg_sampling/odd_111.jxl.hex"),
        _ => panic!("unknown DC sampling edge fixture: {selectors}"),
    })
}

pub mod modular_passes;
pub mod progressive_oracle;
