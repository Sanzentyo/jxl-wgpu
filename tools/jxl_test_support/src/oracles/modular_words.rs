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

fn run_input(encoded: &[u8], option: Option<&str>) -> Vec<u8> {
    static INPUT: AtomicUsize = AtomicUsize::new(0);
    let binary = std::env::var_os("JXL_MODULAR_WORD_ORACLE")
        .expect("required pinned scalar libjxl oracle: set JXL_MODULAR_WORD_ORACLE (see tools/jxl_test_support/README.md)");
    let path = std::env::temp_dir().join(format!(
        "jxl-wgpu-modular-words-{}-{}.jxl",
        std::process::id(),
        INPUT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, encoded).unwrap();
    let result = Command::new(binary).args(option).arg(&path).output();
    std::fs::remove_file(path).unwrap();
    let result = result.expect("run required native Modular word oracle");
    assert!(
        result.status.success(),
        "native Modular word oracle: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

fn words<'a>(bytes: &'a [u8], signature: &[u8; 8]) -> impl Iterator<Item = u32> + 'a {
    let (magic, bytes) = bytes.split_at_checked(8).expect("native word header");
    assert_eq!(magic, signature);
    let (words, tail) = bytes.as_chunks::<4>();
    assert!(tail.is_empty());
    let mut words = words.iter().map(|word| u32::from_le_bytes(*word));
    assert_eq!(words.next(), Some(12000), "libjxl runtime identity");
    words
}

/// Native implicit entries for indices -143 through 188, with the native inverse's depth cap.
pub fn implicit_entries(bits: u8) -> Vec<[i32; 4]> {
    assert!((1..=32).contains(&bits));
    let binary = std::env::var_os("JXL_MODULAR_WORD_ORACLE").expect("native Modular word oracle");
    let result = Command::new(binary)
        .arg("--implicit-entries")
        .arg(bits.to_string())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let words: Vec<_> = words(&result.stdout, b"JXLIMP12").collect();
    assert_eq!(words.len(), 332 * 4);
    words
        .as_chunks::<4>()
        .0
        .iter()
        .map(|values| std::array::from_fn(|c| values[c] as i32))
        .collect()
}

/// Counts negative implicit, cube implicit and explicit indices before the native inverse.
/// Requires a final Palette transform, without following Squeeze, in each group.
pub fn palette_index_counts(encoded: &[u8]) -> [u32; 3] {
    let output = run_input(encoded, Some("--palette-audit"));
    let mut words = words(&output, b"JXLPAL12");
    let counts = std::array::from_fn(|_| words.next().expect("palette index count"));
    assert!(words.next().is_none(), "trailing palette index counts");
    counts
}

pub fn original_frames(encoded: &[u8]) -> Vec<FrameWords> {
    read_frames(run_input(encoded, None))
}

/// Exact physical Modular grids before upsampling, composition or float conversion.
/// The pinned native parser/decoder determines both channel routing and dimensions.
pub fn channel_frames(encoded: &[u8]) -> Vec<Vec<super::modular_integer::ExtraWords>> {
    let output = run_input(encoded, Some("--channel-words"));
    let mut words = words(&output, b"JXLCHN12");
    let count = words.next().expect("frame count");
    assert!((1..=64).contains(&count));
    let frames = (0..count)
        .map(|_| {
            let width = words.next().expect("color width");
            let height = words.next().expect("color height");
            let channels = words.next().expect("channels");
            let bits = words.next().expect("color bits");
            let exponent = words.next().expect("color exponent");
            assert!(width > 0 && height > 0 && (1..=259).contains(&channels));
            assert!((1..=32).contains(&bits) && exponent <= 8);
            let extents: Vec<_> = (0..channels)
                .map(|_| {
                    (
                        words.next().expect("plane width"),
                        words.next().expect("plane height"),
                    )
                })
                .collect();
            extents
                .into_iter()
                .map(|(width, height)| {
                    let pixels = u64::from(width) * u64::from(height);
                    assert!((1..=1 << 26).contains(&pixels));
                    super::modular_integer::ExtraWords {
                        width,
                        height,
                        words: (0..pixels)
                            .map(|_| words.next().expect("physical sample word"))
                            .collect(),
                    }
                })
                .collect()
        })
        .collect();
    assert!(words.next().is_none(), "trailing native channel words");
    frames
}

/// Reads only an original-color, final Modular preview with the native preview frame context.
/// Main bytes remain present and are not rewritten; main entropy is outside this word check.
pub fn original_preview(encoded: &[u8]) -> FrameWords {
    let mut frames = read_frames(run_input(encoded, Some("--preview-words")));
    assert_eq!(frames.len(), 1);
    frames.remove(0)
}

fn read_frames(output: Vec<u8>) -> Vec<FrameWords> {
    let mut words = words(&output, b"JXLRAW12");
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

/// Independent physical-frame sampling metadata, parsed by pinned libjxl without rendering.
/// The helper accepts raw/plain-container, no-preview enumerated-color encoder sequences.
#[derive(Debug)]
pub struct FrameSampling {
    pub presented: [u32; 2],
    pub coded: [u32; 2],
    pub factor: u32,
    pub passes: u32,
    pub extras: Vec<u32>,
}

pub fn sampling_headers(encoded: &[u8]) -> Vec<FrameSampling> {
    let output = run_input(encoded, Some("--sampling-headers"));
    let mut words = words(&output, b"JXLSMP12");
    let count = words.next().expect("physical frame count");
    assert!((1..=64).contains(&count));
    let frames = (0..count)
        .map(|_| {
            let presented = std::array::from_fn(|_| words.next().expect("presented dimension"));
            let coded = std::array::from_fn(|_| words.next().expect("coded dimension"));
            let factor = words.next().expect("color factor");
            let passes = words.next().expect("coefficient passes");
            let extras = words.next().expect("extra count");
            assert!(extras <= 256);
            FrameSampling {
                presented,
                coded,
                factor,
                passes,
                extras: (0..extras)
                    .map(|_| words.next().expect("extra factor"))
                    .collect(),
            }
        })
        .collect();
    assert!(words.next().is_none(), "trailing sampling headers");
    frames
}

/// Original orientation and raw names of every physical frame, including hidden/reference
/// frames. Uses the same pinned header/TOC walk and transport limits as sampling inspection.
pub fn presentation_headers(encoded: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let output = run_input(encoded, Some("--presentation-headers"));
    let mut words = words(&output, b"JXLMET12");
    let count = words.next().expect("physical frame count");
    assert!((1..=64).contains(&count));
    let frames = (0..count)
        .map(|_| {
            let orientation = words.next().expect("orientation");
            assert!((1..=8).contains(&orientation));
            let length = words.next().expect("name length");
            assert!(length <= 1071);
            let name = (0..length)
                .map(|_| u8::try_from(words.next().expect("name byte")).unwrap())
                .collect();
            (orientation, name)
        })
        .collect();
    assert!(words.next().is_none(), "trailing presentation headers");
    frames
}
