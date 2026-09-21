//! Pinned libjxl 0.12.0 outputs after decoding mode-1/2 binary16 wire parameters.
//! No production matrix expansion is used to load or construct the expected values.

use std::sync::OnceLock;

use sha2::{Digest, Sha256};

pub struct ParametricMatrixOracle {
    pub mode: u8,
    pub variant: u8,
    pub parameters: Vec<u16>,
    pub scales: [[f32; 3]; 64],
}

/// Four native records: modes 1/2, each with all-one and channel-varying parameters.
/// Entry zero is unused LLF and includes libjxl's DCT2 sentinel; compare AC entries.
#[must_use]
pub fn records() -> &'static [ParametricMatrixOracle] {
    static RECORDS: OnceLock<Vec<ParametricMatrixOracle>> = OnceLock::new();
    RECORDS.get_or_init(|| {
        let bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../crates/jxl_gpu_protocol/test-data/parametric_matrices.bin"
        ));
        assert_eq!(
            &Sha256::digest(bytes)[..],
            &[
                0xf9, 0x00, 0xca, 0x5c, 0xf7, 0xf2, 0x70, 0x86, 0x6a, 0xe8, 0x80, 0x14, 0x54, 0x70,
                0x06, 0xaf, 0xf8, 0x04, 0x7b, 0x77, 0x0e, 0x67, 0xb3, 0x48, 0xd8, 0x31, 0x5b, 0x34,
                0x60, 0xdb, 0x5d, 0xeb
            ]
        );
        assert_eq!(&bytes[..8], b"JXLPQM01");
        let mut words = bytes[8..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word));
        assert_eq!(words.next(), Some(4));
        let records = (0..4)
            .map(|index| {
                let mode = words.next().unwrap();
                let variant = words.next().unwrap();
                let count = words.next().unwrap();
                assert_eq!(
                    (mode, variant, count),
                    (1 + index / 2, index % 2, if index < 2 { 3 } else { 6 })
                );
                let parameters = words
                    .by_ref()
                    .take(count as usize * 3)
                    .map(|v| u16::try_from(v).unwrap())
                    .collect();
                let mut scales = [[0.0; 3]; 64];
                for channel in 0..3 {
                    for entry in &mut scales {
                        entry[channel] = f32::from_bits(words.next().unwrap());
                    }
                }
                ParametricMatrixOracle {
                    mode: mode as u8,
                    variant: variant as u8,
                    parameters,
                    scales,
                }
            })
            .collect();
        assert_eq!(words.next(), None);
        records
    })
}
