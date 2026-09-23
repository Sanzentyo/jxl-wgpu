//! Raw original component words from the pinned scalar libjxl Modular decoder.
//! The native helper exports physical frames before float conversion or composition.

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct FrameWords {
    pub width: u32,
    pub height: u32,
    pub bits: u32,
    pub exponent_bits: u32,
    pub planes: Vec<Vec<i32>>,
}

pub fn original_frames(encoded: &[u8]) -> Vec<FrameWords> {
    static INPUT: AtomicUsize = AtomicUsize::new(0);
    let binary = std::env::var_os("JXL_MODULAR_WORD_ORACLE")
        .expect("required pinned scalar libjxl oracle: set JXL_MODULAR_WORD_ORACLE (see tools/jxl_test_support/README.md)");
    let path = std::env::temp_dir().join(format!(
        "jxl-wgpu-modular-words-{}-{}.jxl",
        std::process::id(),
        INPUT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, encoded).unwrap();
    let result = Command::new(binary).arg(&path).output();
    std::fs::remove_file(path).unwrap();
    let result = result.expect("run required native Modular word oracle");
    assert!(
        result.status.success(),
        "native Modular word oracle: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let (magic, bytes) = result
        .stdout
        .split_at_checked(8)
        .expect("native word header");
    assert_eq!(magic, b"JXLRAW12");
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(tail.is_empty());
    let mut words = words.iter().map(|word| u32::from_le_bytes(*word));
    assert_eq!(words.next(), Some(12000), "libjxl runtime identity");
    let count = words.next().expect("frame count");
    assert!((1..=64).contains(&count));
    let frames = (0..count)
        .map(|_| {
            let width = words.next().expect("width");
            let height = words.next().expect("height");
            let channels = words.next().expect("channels");
            let bits = words.next().expect("bits");
            let exponent_bits = words.next().expect("exponent bits");
            assert!((1..=4).contains(&channels));
            let pixels = usize::try_from(u64::from(width) * u64::from(height)).unwrap();
            assert!(pixels > 0 && pixels <= 1 << 24);
            let planes = (0..channels)
                .map(|_| {
                    (0..pixels)
                        .map(|_| words.next().expect("original sample word") as i32)
                        .collect()
                })
                .collect();
            FrameWords {
                width,
                height,
                bits,
                exponent_bits,
                planes,
            }
        })
        .collect();
    assert!(words.next().is_none(), "trailing native word data");
    frames
}
