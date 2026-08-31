#![allow(dead_code)]

use std::sync::LazyLock;

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

pub fn green_queen_vardct_gaborish() -> &'static [u8] {
    static BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
        decode_hex(include_str!(
            "../../test-data/green_queen_vardct_gaborish.jxl.hex"
        ))
    });
    BYTES.as_slice()
}
