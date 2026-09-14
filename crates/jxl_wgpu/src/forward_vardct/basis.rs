//! Fixed transform bases. Only strategy and coordinate indices enter
//! this module: it never receives image samples or encoded coefficients.
//!
//! The equations follow libjxl TransformFromPixels (enc_transforms-inl.h).
//! The AFV basis is shared with the resident inverse transform.

use jxl_gpu_protocol::TransformKind;

use crate::VAR_DCT_AFV_BASIS;

fn cosine(frequency: usize, position: usize, size: usize) -> f64 {
    if frequency == 0 {
        1.0
    } else {
        std::f64::consts::SQRT_2
            * (std::f64::consts::PI * frequency as f64 * (position as f64 + 0.5) / size as f64)
                .cos()
    }
}

fn dct_weight(fx: usize, fy: usize, x: usize, y: usize, width: usize, height: usize) -> f64 {
    cosine(fx, x, width) * cosine(fy, y, height) / (width * height) as f64
}

fn quadrant_low(row: usize, col: usize, x: usize, y: usize) -> f64 {
    let negative = (row == 1 && x >= 4) ^ (col == 1 && y >= 4);
    if negative { -1.0 / 64.0 } else { 1.0 / 64.0 }
}

fn weight(kind: TransformKind, coefficient: usize, pixel: usize) -> f64 {
    let row = coefficient / 8;
    let col = coefficient % 8;
    let x = pixel % 8;
    let y = pixel / 8;
    match kind {
        TransformKind::Hornuss => {
            if row < 2 && col < 2 {
                return quadrant_low(row, col, x, y);
            }
            if x / 4 != col % 2 || y / 4 != row % 2 {
                return 0.0;
            }
            let (px, py) = if row / 2 == 1 && col / 2 == 1 {
                (0, 0)
            } else {
                (col / 2, row / 2)
            };
            f64::from(u8::from(x % 4 == px && y % 4 == py))
                - f64::from(u8::from(x % 4 == 1 && y % 4 == 1))
        }
        TransformKind::Dct2x2 => {
            let (side, cells) = if row >= 4 || col >= 4 {
                (2, 4)
            } else if row >= 2 || col >= 2 {
                (4, 2)
            } else {
                (8, 1)
            };
            let x0 = col % cells * side;
            let y0 = row % cells * side;
            if !(x0..x0 + side).contains(&x) || !(y0..y0 + side).contains(&y) {
                return 0.0;
            }
            let negative =
                (row >= cells && x - x0 >= side / 2) ^ (col >= cells && y - y0 >= side / 2);
            (if negative { -1.0 } else { 1.0 }) / (side * side) as f64
        }
        TransformKind::Dct4x4 => {
            if row < 2 && col < 2 {
                return quadrant_low(row, col, x, y);
            }
            if x / 4 != col % 2 || y / 4 != row % 2 {
                return 0.0;
            }
            dct_weight(row / 2, col / 2, x % 4, y % 4, 4, 4)
        }
        TransformKind::Dct4x8 | TransformKind::Dct8x4 => {
            let horizontal_halves = kind == TransformKind::Dct8x4;
            let half = if horizontal_halves { x / 4 } else { y / 4 };
            if col == 0 && row < 2 {
                return if row == 1 && half == 1 {
                    -1.0 / 64.0
                } else {
                    1.0 / 64.0
                };
            }
            if half != row % 2 {
                return 0.0;
            }
            if horizontal_halves {
                dct_weight(row / 2, col, x % 4, y, 4, 8)
            } else {
                dct_weight(col, row / 2, x, y % 4, 8, 4)
            }
        }
        TransformKind::Afv0 | TransformKind::Afv1 | TransformKind::Afv2 | TransformKind::Afv3 => {
            let (ax, ay) = match kind {
                TransformKind::Afv0 => (0, 0),
                TransformKind::Afv1 => (1, 0),
                TransformKind::Afv2 => (0, 1),
                TransformKind::Afv3 => (1, 1),
                _ => unreachable!(),
            };
            if coefficient == 0 {
                return 1.0 / 64.0;
            }
            if coefficient == 1 {
                return if y / 4 != ay {
                    0.0
                } else if x / 4 == ax {
                    1.0 / 32.0
                } else {
                    -1.0 / 32.0
                };
            }
            if coefficient == 8 {
                return if y / 4 == ay { 1.0 / 64.0 } else { -1.0 / 64.0 };
            }
            if row % 2 == 1 {
                return if y / 4 == ay {
                    0.0
                } else {
                    dct_weight(col, row / 2, x, y % 4, 8, 4)
                };
            }
            if y / 4 != ay {
                return 0.0;
            }
            if col % 2 == 1 {
                return if x / 4 == ax {
                    0.0
                } else {
                    dct_weight(row / 2, col / 2, x % 4, y % 4, 4, 4)
                };
            }
            if x / 4 != ax {
                return 0.0;
            }
            let px = if ax == 1 { 3 - x % 4 } else { x % 4 };
            let py = if ay == 1 { 3 - y % 4 } else { y % 4 };
            f64::from(VAR_DCT_AFV_BASIS[(row / 2 * 4 + col / 2) * 16 + py * 4 + px])
        }
        _ => unreachable!("regular DCT uses the separable GPU passes"),
    }
}

pub(super) struct TransformBasis {
    pub(super) weights: Vec<f32>,
    pub(super) offsets: [u32; 4],
}

pub(super) fn matrix(kind: TransformKind) -> TransformBasis {
    if kind.is_special() {
        return TransformBasis {
            weights: (0..64 * 64)
                .map(|index| weight(kind, index / 64, index % 64) as f32)
                .collect(),
            offsets: [0; 4],
        };
    }
    let extent = kind.pixel_extent();
    let lf = kind.lf_extent();
    let mut weights = Vec::new();
    let mut offsets = [0; 4];
    for (table, size) in [extent.width, extent.height, lf.width, lf.height]
        .into_iter()
        .enumerate()
    {
        offsets[table] = weights.len() as u32;
        for frequency in 0..size as usize {
            for position in 0..size as usize {
                let mut value = cosine(frequency, position, size as usize);
                if table < 2 {
                    value /= f64::from(size);
                } else {
                    let phase = std::f64::consts::PI * frequency as f64 / f64::from(size);
                    value *= (phase / 16.0).cos() * (phase / 8.0).cos() * (phase / 4.0).cos();
                }
                weights.push(value as f32);
            }
        }
    }
    TransformBasis { weights, offsets }
}
