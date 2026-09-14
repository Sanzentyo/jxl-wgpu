//! Default JPEG XL quantization library, expanded in canonical coefficient order.
//! Constants follow libjxl 0.12.0 quant_weights.cc (BSD-3-Clause).

use crate::{TransformKind, VarDctDequantMatrix};

const SEQUENCES: [[f32; 7]; 3] = [
    [
        -1.025,
        -0.78,
        -0.65012,
        -0.19041574,
        -0.20819396,
        -0.421064,
        -0.32733846,
    ],
    [
        -0.30419582,
        -0.36330363,
        -0.3566038,
        -0.34430745,
        -0.33699593,
        -0.30180866,
        -0.27321684,
    ],
    [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5],
];
const DCT4: [[f32; 4]; 3] = [
    [2200.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5],
];
const DCT4X8: [[f32; 4]; 3] = [
    [2198.0505, -0.96269625, -0.7619425, -0.65511405],
    [764.36554, -0.926302, -0.967523, -0.2784529],
    [527.10754, -1.4594386, -1.4500821, -1.5843723],
];

fn multiplier(value: f32) -> f32 {
    if value > 0.0 {
        1.0 + value
    } else {
        1.0 / (1.0 - value)
    }
}

fn bands(parameters: &[f32]) -> Vec<f32> {
    let mut bands = vec![parameters[0]];
    for &value in &parameters[1..] {
        bands.push(bands.last().unwrap() * multiplier(value));
    }
    bands
}

fn interpolate(bands: &[f32], position: f32, maximum: f32) -> f32 {
    let position = position * (bands.len() - 1) as f32 / maximum;
    let index = (position as usize).min(bands.len() - 2);
    bands[index] * (bands[index + 1] / bands[index]).powf(position - index as f32)
}

fn dct(parameters: &[f32], width: usize, height: usize) -> Vec<f32> {
    let bands = bands(parameters);
    (0..width * height)
        .map(|index| {
            let x = (index % width) as f32 / (width - 1) as f32;
            let y = (index / width) as f32 / (height - 1) as f32;
            interpolate(
                &bands,
                (x * x + y * y).sqrt(),
                std::f32::consts::SQRT_2 + 1e-6,
            )
        })
        .collect()
}

fn regular_parameters(transform: TransformKind, channel: usize) -> Vec<f32> {
    use TransformKind::*;
    let parameters: &[f32] = match transform {
        Dct8 => &[
            [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
            [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
            [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
        ][channel],
        Dct16x16 => &[
            [
                8996.873,
                -1.3000778,
                -0.4942453,
                -0.43909377,
                -0.6350102,
                -0.9017726,
                -1.6162099,
            ],
            [
                3191.4836,
                -0.67424583,
                -0.80745816,
                -0.4492584,
                -0.3586544,
                -0.3132239,
                -0.37615025,
            ],
            [
                1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914,
            ],
        ][channel],
        Dct32x32 => &[
            [
                15718.408,
                -1.025,
                -0.98,
                -0.9012,
                -0.4,
                -0.48819396,
                -0.421064,
                -0.27,
            ],
            [
                7305.7637,
                -0.8041958,
                -0.76330364,
                -0.5566038,
                -0.49785304,
                -0.43699592,
                -0.40180868,
                -0.27321684,
            ],
            [
                3803.5317,
                -3.0607336,
                -2.041327,
                -2.023565,
                -0.54953897,
                -0.4,
                -0.4,
                -0.3,
            ],
        ][channel],
        Dct16x8 | Dct8x16 => &[
            [7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
            [1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
            [506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
        ][channel],
        Dct32x8 | Dct8x32 => &[
            [
                16283.249, -1.7812846, -1.6309059, -1.0382179, -0.85, -0.7, -0.9, -1.2360638,
            ],
            [
                5089.1577, -0.3200494, -0.3536285, -0.3034, -0.61, -0.5, -0.5, -0.6,
            ],
            [
                3397.7761,
                -0.32132736,
                -0.3450762,
                -0.7034,
                -0.9,
                -1.0,
                -1.0,
                -1.1754606,
            ],
        ][channel],
        Dct32x16 | Dct16x32 => &[
            [
                13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6,
            ],
            [
                4798.964,
                -0.6112531,
                -0.8377079,
                -0.7901486,
                -0.26927274,
                -0.38272768,
                -0.22924222,
                -0.20719099,
            ],
            [1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5],
        ][channel],
        _ => {
            let first = match transform {
                Dct64x64 => [23966.166, 8380.191, 4493.024],
                Dct64x32 | Dct32x64 => [15358.898, 5597.3604, 2919.9617],
                Dct128x128 => [47932.332, 16760.383, 8986.048],
                Dct128x64 | Dct64x128 => [30717.797, 11194.721, 5839.9233],
                Dct256x256 => [95864.664, 33520.766, 17972.096],
                Dct256x128 | Dct128x256 => [61435.594, 22389.441, 11679.847],
                _ => unreachable!("special transform has its own quantization matrix"),
            };
            return std::iter::once(first[channel])
                .chain(SEQUENCES[channel])
                .collect();
        }
    };
    parameters.to_vec()
}

fn special(transform: TransformKind, channel: usize) -> Vec<f32> {
    use TransformKind::*;
    match transform {
        Hornuss => {
            let [base, edge, corner] = [
                [280.0, 3160.0, 3160.0],
                [60.0, 864.0, 864.0],
                [18.0, 200.0, 200.0],
            ][channel];
            let mut weights = vec![base; 64];
            weights[0] = 1.0;
            weights[1] = edge;
            weights[8] = edge;
            weights[9] = corner;
            weights
        }
        Dct2x2 => {
            let parameters = [
                [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
                [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
                [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
            ][channel];
            let mut weights = vec![1.0; 64];
            for (index, &value) in parameters.iter().enumerate() {
                let side = 1 << (index / 2);
                for y in 0..2 * side {
                    for x in 0..2 * side {
                        let high_x = x >= side;
                        let high_y = y >= side;
                        if (index.is_multiple_of(2) && high_x != high_y)
                            || (!index.is_multiple_of(2) && high_x && high_y)
                        {
                            weights[y * 8 + x] = value;
                        }
                    }
                }
            }
            weights
        }
        Dct4x4 => {
            let small = dct(&DCT4[channel], 4, 4);
            (0..64)
                .map(|i| small[(i / 8 / 2) * 4 + (i % 8 / 2)])
                .collect()
        }
        Dct4x8 | Dct8x4 => {
            let small = dct(&DCT4X8[channel], 8, 4);
            (0..64).map(|i| small[(i / 8 / 2) * 8 + i % 8]).collect()
        }
        Afv0 | Afv1 | Afv2 | Afv3 => {
            let parameters = [
                [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
                [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
                [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
            ][channel];
            let frequencies = [
                0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0,
                7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
            ];
            let afv_bands = bands(&parameters[5..]);
            let four = dct(&DCT4[channel], 4, 4);
            let half = dct(&DCT4X8[channel], 8, 4);
            (0usize..64)
                .map(|index| {
                    let x = index % 8;
                    let y = index / 8;
                    match (x, y) {
                        (0, 0) => 1.0,
                        (0, 1) => parameters[0],
                        (1, 0) => parameters[1],
                        (0, 2) => parameters[2],
                        (2, 0) => parameters[3],
                        (2, 2) => parameters[4],
                        _ if !y.is_multiple_of(2) => half[y / 2 * 8 + x],
                        _ if !x.is_multiple_of(2) => four[y / 2 * 4 + x / 2],
                        _ => interpolate(
                            &afv_bands,
                            frequencies[y / 2 * 4 + x / 2] - frequencies[2],
                            frequencies[15] - frequencies[2] + 1e-6,
                        ),
                    }
                })
                .collect()
        }
        _ => unreachable!("regular DCT has a separable frequency grid"),
    }
}

impl TransformKind {
    /// Expands the standard quantization library into X/Y/B dequantization
    /// multipliers in canonical transform-buffer order. This is bounded
    /// strategy metadata; it does not inspect image samples or coefficients.
    #[must_use]
    pub fn default_dequant_matrix(self) -> VarDctDequantMatrix {
        let extent = self.pixel_extent();
        let width = extent.width.max(extent.height) as usize;
        let height = extent.width.min(extent.height) as usize;
        let channels: [Vec<f32>; 3] = std::array::from_fn(|channel| {
            if self.is_special() {
                special(self, channel)
            } else {
                dct(&regular_parameters(self, channel), width, height)
            }
        });
        VarDctDequantMatrix {
            transform: self,
            scales: (0..width * height)
                .map(|index| std::array::from_fn(|channel| 1.0 / channels[channel][index]))
                .collect(),
        }
    }
}
