//! Independent f64 equations for the fixed DCT8 quantization policy.
//!
//! This reference evaluates the two-dimensional cosine sum and log-interpolated
//! matrix bands directly. It neither loads WGSL nor calls production transform,
//! quantization or coefficient-order helpers. A one-integer quantizer-step bound
//! permits a different side of a rounding boundary in f32; this is a regression
//! oracle, not an ISO 18181-3 decoder precision or distance-quality claim.

use super::VarDctLfMetadata;

pub(super) fn natural_order() -> Vec<usize> {
    let mut order = Vec::new();
    for diagonal in 0..15usize {
        let mut entries = (0..8)
            .filter_map(|row| {
                let column = diagonal.checked_sub(row)?;
                (column < 8).then_some(row * 8 + column)
            })
            .collect::<Vec<_>>();
        if diagonal % 2 == 0 {
            entries.reverse();
        }
        order.extend(entries);
    }
    order
}

pub(super) fn pattern(width: usize, height: usize) -> Vec<[u8; 3]> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                let sx = x % 8;
                let sy = y % 8;
                match (x / 8 + 3 * (y / 8)) % 7 {
                    0 => [if (sx + sy) % 2 == 0 { 32 } else { 224 }; 3],
                    1 => [if sy % 2 == 0 { 48 } else { 208 }; 3],
                    2 => [if sx % 2 == 0 { 211 } else { 23 }, 65, 157],
                    3 => {
                        if sx == 2 && sy == 5 {
                            [228, 170, 78]
                        } else {
                            [21, 32, 53]
                        }
                    }
                    4 => [
                        (sx * 29 + sy * 5) as u8,
                        (sy * 29 + sx * 3) as u8,
                        ((sx + sy) * 17) as u8,
                    ],
                    5 => [
                        ((sx * 29 + sy * 101) % 256) as u8,
                        ((sx * 131 + sy * 47) % 256) as u8,
                        ((sx * 41 + sy * 17) % 256) as u8,
                    ],
                    _ => [41, 85, 129],
                }
            })
        })
        .collect()
}

pub(super) fn block(
    pixels: &[[u8; 3]],
    width: usize,
    height: usize,
    bx: usize,
    by: usize,
) -> [[u8; 3]; 64] {
    std::array::from_fn(|index| {
        let x = (bx * 8 + index % 8).min(width - 1);
        let y = (by * 8 + index / 8).min(height - 1);
        pixels[y * width + x]
    })
}

pub(super) fn quantized_ac(pixels: &[[u8; 3]; 64], metadata: VarDctLfMetadata) -> [[i32; 64]; 3] {
    let xyb = pixels.map(|pixel| {
        let rgb = pixel.map(|value| {
            let value = f64::from(value) / 255.0;
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        });
        let bias = 0.0037930732552754493f64;
        let absorbance = [
            [0.3, 0.622, 0.078],
            [0.23, 0.692, 0.078],
            [0.2434226892, 0.2047674442, 0.5518098665],
        ]
        .map(|row| {
            (bias + row.into_iter().zip(rgb).map(|(a, b)| a * b).sum::<f64>()).cbrt() - bias.cbrt()
        });
        [
            (absorbance[0] - absorbance[1]) * 0.5,
            (absorbance[0] + absorbance[1]) * 0.5,
            absorbance[2],
        ]
    });
    let basis: [[f64; 8]; 8] = std::array::from_fn(|frequency| {
        std::array::from_fn(|position| {
            if frequency == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
                    * (std::f64::consts::PI * (position as f64 + 0.5) * frequency as f64 / 8.0)
                        .cos()
            }
        })
    });
    let slopes = metadata
        .base_correlation
        .map(|value| f64::from(value.to_f32()));
    let bands: [[f64; 6]; 3] = [
        [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
        [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
        [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
    ]
    .map(|parameters| {
        let mut values = parameters;
        for i in 1..6 {
            values[i] = values[i - 1]
                * if parameters[i] > 0.0 {
                    1.0 + parameters[i]
                } else {
                    1.0 / (1.0 - parameters[i])
                };
        }
        values
    });
    let mut result = [[0; 64]; 3];
    for fy in 0..8 {
        for fx in 0..8 {
            if fx == 0 && fy == 0 {
                continue;
            }
            let coefficient: [f64; 3] = std::array::from_fn(|channel| {
                (0..64)
                    .map(|pixel| xyb[pixel][channel] * basis[fx][pixel % 8] * basis[fy][pixel / 8])
                    .sum::<f64>()
                    / 64.0
            });
            let decorrelated = [
                coefficient[0] - slopes[0] * coefficient[1],
                coefficient[1],
                coefficient[2] - slopes[1] * coefficient[1],
            ];
            let distance =
                ((fx * fx + fy * fy) as f64).sqrt() / 7.0 * 5.0 / (std::f64::consts::SQRT_2 + 1e-6);
            let band = (distance as usize).min(4);
            let fraction = distance - band as f64;
            for channel in 0..3 {
                let weight = ((1.0 - fraction) * bands[channel][band].ln()
                    + fraction * bands[channel][band + 1].ln())
                .exp();
                let quantized = decorrelated[channel]
                    * (8813.0 * 6.0 / 65536.0)
                    * [1.25, 1.0, 1.0][channel]
                    * weight;
                result[channel][fx * 8 + fy] = quantized.round() as i32;
            }
        }
    }
    result
}
