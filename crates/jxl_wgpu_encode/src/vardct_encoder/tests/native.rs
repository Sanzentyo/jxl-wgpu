//! Independent f64 calculations using committed libjxl matrix/order/basis fixtures.

use super::*;
use jxl_gpu_bitstream::BitReader;

pub(super) struct Oracle {
    order: Vec<usize>,
    dequant: [Vec<f64>; 3],
    basis: Vec<f64>,
}

pub(super) fn native_oracles() -> Vec<Oracle> {
    let bytes = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../jxl_gpu_protocol/test-data/vardct_metadata.bin"
    ))
    .unwrap();
    assert_eq!(&bytes[..8], b"JXLQNT01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    assert_eq!(words.next(), Some(27));
    let mut oracles = (0..27)
        .map(|id| {
            assert_eq!(words.next(), Some(id));
            let area = words.next().unwrap() as usize;
            let order = words
                .by_ref()
                .take(area)
                .map(|word| word as usize)
                .collect();
            let dequant = std::array::from_fn(|_| {
                words
                    .by_ref()
                    .take(area)
                    .map(|word| f64::from(f32::from_bits(word)))
                    .collect()
            });
            Oracle {
                order,
                dequant,
                basis: if area == 64 {
                    vec![0.0; 4096]
                } else {
                    Vec::new()
                },
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(words.next(), None);
    let bytes = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../jxl_wgpu/test-data/forward_vardct.bin"
    ))
    .unwrap();
    assert_eq!(&bytes[..8], b"JXLFWD01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    let records = words.next().unwrap();
    assert_eq!(records, 667);
    let mut impulses = [0; 27];
    for _ in 0..records {
        let id = words.next().unwrap() as usize;
        let test = words.next().unwrap();
        let width = words.next().unwrap() as usize;
        let height = words.next().unwrap() as usize;
        let area = width * height;
        for coefficient in 0..3 * (area + area / 64) {
            let value = f64::from(f32::from_bits(words.next().unwrap()));
            if test != 0 && coefficient < area {
                oracles[id].basis[coefficient * 64 + test as usize - 1] = value;
            }
        }
        impulses[id] += usize::from(test != 0);
    }
    assert_eq!(words.next(), None);
    for (oracle, impulses) in oracles.iter().zip(impulses) {
        assert_eq!(impulses, if oracle.basis.is_empty() { 0 } else { 64 });
    }
    oracles
}

pub(super) fn forward(
    pixels: &[[u8; 3]],
    width: usize,
    height: usize,
    oracle: &Oracle,
) -> Vec<[f64; 3]> {
    let xyb = pixels
        .iter()
        .map(|&pixel| reference::xyb(pixel))
        .collect::<Vec<_>>();
    let area = width * height;
    if !oracle.basis.is_empty() {
        return (0..area)
            .map(|coefficient| {
                std::array::from_fn(|channel| {
                    xyb.iter()
                        .enumerate()
                        .map(|(pixel, value)| {
                            value[channel] * oracle.basis[coefficient * 64 + pixel]
                        })
                        .sum()
                })
            })
            .collect();
    }
    let cosines = |size: usize| {
        (0..size * size)
            .map(|index| {
                let frequency = index / size;
                let position = index % size;
                if frequency == 0 {
                    1.0 / size as f64
                } else {
                    std::f64::consts::SQRT_2
                        * (std::f64::consts::PI * frequency as f64 * (position as f64 + 0.5)
                            / size as f64)
                            .cos()
                        / size as f64
                }
            })
            .collect::<Vec<_>>()
    };
    let horizontal_basis = cosines(width);
    let vertical_basis = cosines(height);
    let mut horizontal = vec![[0.0; 3]; area];
    for y in 0..height {
        for fx in 0..width {
            horizontal[y * width + fx] = std::array::from_fn(|channel| {
                (0..width)
                    .map(|x| xyb[y * width + x][channel] * horizontal_basis[fx * width + x])
                    .sum()
            });
        }
    }
    let mut result = vec![[0.0; 3]; area];
    for fy in 0..height {
        for fx in 0..width {
            let wire = if height < width {
                fy * width + fx
            } else {
                fx * height + fy
            };
            result[wire] = std::array::from_fn(|channel| {
                (0..height)
                    .map(|y| horizontal[y * width + fx][channel] * vertical_basis[fy * height + y])
                    .sum()
            });
        }
    }
    result
}

pub(super) fn check_ac(
    words: &[u32],
    bit_len: u32,
    coefficients: &[[f64; 3]],
    oracle: &Oracle,
    config: VarDctConfig,
) -> usize {
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let mut reader = BitReader::new(&bytes);
    let slopes = config
        .lf_metadata
        .base_correlation()
        .map(|value| f64::from(value.to_f32()));
    let skip = coefficients.len() / 64;
    let mut nonzero = 0;
    for channel in [1, 0, 2] {
        let mut remaining = ac::read_unsigned(&mut reader, &entropy);
        assert!(remaining as usize <= coefficients.len() - skip);
        for &index in &oracle.order[skip..] {
            let packed = if remaining == 0 {
                0
            } else {
                ac::read_unsigned(&mut reader, &entropy)
            };
            remaining -= u32::from(packed != 0);
            nonzero += usize::from(packed != 0);
            let actual = if packed.is_multiple_of(2) {
                (packed / 2) as i32
            } else {
                -((packed / 2) as i32) - 1
            };
            let coefficient = coefficients[index];
            let decorrelated = match channel {
                0 => coefficient[0] - slopes[0] * coefficient[1],
                1 => coefficient[1],
                _ => coefficient[2] - slopes[1] * coefficient[1],
            };
            let expected = (decorrelated
                * (f64::from(config.quantization.global_scale())
                    * f64::from(config.quantization.hf_multiplier().get())
                    / 65536.0)
                * [1.25, 1.0, 1.0][channel]
                / oracle.dequant[channel][index])
                .round() as i32;
            // Fixed f64/native oracle regression bound: at most one quantizer
            // step across f32 rounding boundaries, not a distance-quality claim.
            assert!(
                (actual - expected).abs() <= 1,
                "channel {channel}, coefficient {index}: GPU={actual}, reference={expected}"
            );
        }
        assert_eq!(remaining, 0);
    }
    assert_eq!(reader.bit_offset(), u64::from(bit_len));
    nonzero
}
