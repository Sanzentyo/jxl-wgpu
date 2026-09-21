//! Parametric HF dequantization metadata; no image samples are processed here.

use crate::{TransformKind, VarDctDequantMatrix};

/// JPEG XL parametric HF matrix modes 0–6, in exact serialized parameter units.
///
/// Each DCT band vector has the same length in X/Y/B, from 1 through 16. The first
/// band, all Hornuss/DCT2 parameters, and the first six AFV parameters are multiplied
/// by 64 during expansion.
/// Modes 1–5 are restricted to 8×8 matrix families. Mode 7 requires a separately
/// decoded Modular side image and is not represented by this metadata type.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarDctMatrixEncoding<T = f32> {
    #[default]
    Default,
    Hornuss([[T; 3]; 3]),
    Dct2([[T; 6]; 3]),
    Dct4 {
        params: [[T; 2]; 3],
        dct_params: [Vec<T>; 3],
    },
    Dct4x8 {
        params: [[T; 1]; 3],
        dct_params: [Vec<T>; 3],
    },
    Afv {
        params: [[T; 9]; 3],
        dct_params: [Vec<T>; 3],
        dct4x4_params: [Vec<T>; 3],
    },
    Dct([Vec<T>; 3]),
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VarDctMatrixError {
    #[error("HF matrix {matrix} does not support encoding {encoding}")]
    Encoding { matrix: usize, encoding: u8 },
    #[error("invalid HF matrix {matrix}: {reason}")]
    Value { matrix: usize, reason: &'static str },
}

impl<T> VarDctMatrixEncoding<T> {
    #[must_use]
    pub const fn encoding_id(&self) -> u8 {
        match self {
            Self::Default => 0,
            Self::Hornuss(_) => 1,
            Self::Dct2(_) => 2,
            Self::Dct4 { .. } => 3,
            Self::Dct4x8 { .. } => 4,
            Self::Afv { .. } => 5,
            Self::Dct(_) => 6,
        }
    }

    /// Validates compatibility and bounded vector lengths before cloning or expansion.
    pub fn validate_shape(&self, transform: TransformKind) -> Result<(), VarDctMatrixError> {
        let matrix = transform.dequant_matrix_index();
        let encoding = self.encoding_id();
        if (1..=5).contains(&encoding) && !matches!(matrix, 0 | 1 | 2 | 3 | 9 | 10) {
            return Err(VarDctMatrixError::Encoding { matrix, encoding });
        }
        let bands = |params: &[Vec<T>; 3]| {
            let len = params[0].len();
            if !(1..=16).contains(&len) || params.iter().any(|channel| channel.len() != len) {
                return Err(VarDctMatrixError::Value {
                    matrix,
                    reason: "DCT bands must have equal X/Y/B lengths in 1..=16",
                });
            }
            Ok(())
        };
        match self {
            Self::Dct4 { dct_params, .. }
            | Self::Dct4x8 { dct_params, .. }
            | Self::Dct(dct_params) => bands(dct_params),
            Self::Afv {
                dct_params,
                dct4x4_params,
                ..
            } => {
                bands(dct_params)?;
                bands(dct4x4_params)
            }
            _ => Ok(()),
        }
    }
}

impl<T: Copy> VarDctMatrixEncoding<T> {
    /// Converts metadata scalars while retaining channel and band ordering.
    #[must_use]
    pub fn map<U>(&self, mut map: impl FnMut(T) -> U) -> VarDctMatrixEncoding<U> {
        fn fixed<T: Copy, U, const N: usize>(
            p: &[[T; N]; 3],
            f: &mut impl FnMut(T) -> U,
        ) -> [[U; N]; 3] {
            std::array::from_fn(|c| std::array::from_fn(|i| f(p[c][i])))
        }
        fn bands<T: Copy, U>(p: &[Vec<T>; 3], f: &mut impl FnMut(T) -> U) -> [Vec<U>; 3] {
            std::array::from_fn(|c| p[c].iter().copied().map(&mut *f).collect())
        }
        match self {
            Self::Default => VarDctMatrixEncoding::Default,
            Self::Hornuss(p) => VarDctMatrixEncoding::Hornuss(fixed(p, &mut map)),
            Self::Dct2(p) => VarDctMatrixEncoding::Dct2(fixed(p, &mut map)),
            Self::Dct4 { params, dct_params } => VarDctMatrixEncoding::Dct4 {
                params: fixed(params, &mut map),
                dct_params: bands(dct_params, &mut map),
            },
            Self::Dct4x8 { params, dct_params } => VarDctMatrixEncoding::Dct4x8 {
                params: fixed(params, &mut map),
                dct_params: bands(dct_params, &mut map),
            },
            Self::Afv {
                params,
                dct_params,
                dct4x4_params,
            } => VarDctMatrixEncoding::Afv {
                params: fixed(params, &mut map),
                dct_params: bands(dct_params, &mut map),
                dct4x4_params: bands(dct4x4_params, &mut map),
            },
            Self::Dct(p) => VarDctMatrixEncoding::Dct(bands(p, &mut map)),
        }
    }
}

impl VarDctMatrixEncoding {
    /// Expands bounded control metadata into canonical transform-buffer order.
    pub fn expand(
        &self,
        transform: TransformKind,
    ) -> Result<VarDctDequantMatrix, VarDctMatrixError> {
        self.validate_shape(transform)?;
        if matches!(self, Self::Default) {
            return Ok(transform.default_dequant_matrix());
        }
        let matrix = transform.dequant_matrix_index();
        let mut finite = true;
        let _ = self.map(|value| {
            finite &= value.is_finite();
        });
        if !finite {
            return Err(VarDctMatrixError::Value {
                matrix,
                reason: "parameter is not finite",
            });
        }
        let representative = representative(matrix);
        let mut channels = expand_channels(self, representative, matrix)?;
        if transform.needs_transpose() {
            let extent = representative.pixel_extent();
            channels = transpose(&channels, extent.width, extent.height);
        }
        let extent = transform.pixel_extent();
        let mut scales = vec![[0.0; 3]; (extent.width * extent.height) as usize];
        for y in 0..extent.height {
            for x in 0..extent.width {
                let raster = (y * extent.width + x) as usize;
                let wire = if transform.is_special() || extent.height < extent.width {
                    raster
                } else {
                    (x * extent.height + y) as usize
                };
                scales[wire] = std::array::from_fn(|c| channels[c][raster]);
            }
        }
        Ok(VarDctDequantMatrix { transform, scales })
    }
}

fn representative(index: usize) -> TransformKind {
    [
        TransformKind::Dct8,
        TransformKind::Hornuss,
        TransformKind::Dct2x2,
        TransformKind::Dct4x4,
        TransformKind::Dct16x16,
        TransformKind::Dct32x32,
        TransformKind::Dct8x16,
        TransformKind::Dct8x32,
        TransformKind::Dct16x32,
        TransformKind::Dct4x8,
        TransformKind::Afv0,
        TransformKind::Dct64x64,
        TransformKind::Dct32x64,
        TransformKind::Dct128x128,
        TransformKind::Dct64x128,
        TransformKind::Dct256x256,
        TransformKind::Dct128x256,
    ][index]
}

fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
    if let [value] = bands {
        return *value;
    }
    let scaled = pos * (bands.len() - 1) as f32 / max;
    let index = (scaled as usize).min(bands.len() - 2);
    let fraction = scaled - index as f32;
    let left = bands[index];
    let right = bands[index + 1];
    left * (right / left).powf(fraction)
}

fn multiplier(value: f32) -> f32 {
    if value > 0.0 {
        1.0 + value
    } else {
        1.0 / (1.0 - value)
    }
}

fn dct_weights(
    params: &[f32],
    width: u32,
    height: u32,
    matrix: usize,
) -> Result<Vec<f32>, VarDctMatrixError> {
    let mut bands = Vec::with_capacity(params.len());
    let mut last = *params.first().ok_or(VarDctMatrixError::Value {
        matrix,
        reason: "DCT matrix has no bands",
    })?;
    last *= 64.0;
    bands.push(last);
    for &value in &params[1..] {
        last *= multiplier(value);
        if !last.is_finite() || last <= 0.0 {
            return Err(VarDctMatrixError::Value {
                matrix,
                reason: "DCT band is non-positive or non-finite",
            });
        }
        bands.push(last);
    }
    let mut output = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            output.push(interpolate(
                (dx * dx + dy * dy).sqrt(),
                std::f32::consts::SQRT_2 + 1e-6,
                &bands,
            ));
        }
    }
    Ok(output)
}

fn expand_channels(
    encoding: &VarDctMatrixEncoding,
    transform: TransformKind,
    matrix: usize,
) -> Result<[Vec<f32>; 3], VarDctMatrixError> {
    let output = match encoding {
        VarDctMatrixEncoding::Default => {
            unreachable!("default matrices are expanded separately")
        }
        VarDctMatrixEncoding::Dct(params) => {
            let extent = transform.pixel_extent();
            [
                dct_weights(&params[0], extent.width, extent.height, matrix)?,
                dct_weights(&params[1], extent.width, extent.height, matrix)?,
                dct_weights(&params[2], extent.width, extent.height, matrix)?,
            ]
        }
        VarDctMatrixEncoding::Hornuss(params) => params.map(|params| {
            let params = params.map(|value| value * 64.0);
            let mut values = vec![params[0]; 64];
            values[0] = 1.0;
            values[1] = params[1];
            values[8] = params[1];
            values[9] = params[2];
            values
        }),
        VarDctMatrixEncoding::Dct2(params) => params.map(|params| {
            let params = params.map(|value| value * 64.0);
            let mut values = vec![0.0; 64];
            values[0] = 1.0;
            for (index, value) in params.into_iter().enumerate() {
                let shift = index / 2;
                let dimension = 1_usize << shift;
                if index % 2 == 0 {
                    for y in 0..dimension {
                        for x in dimension..dimension * 2 {
                            values[y * 8 + x] = value;
                            values[x * 8 + y] = value;
                        }
                    }
                } else {
                    for y in dimension..dimension * 2 {
                        for x in dimension..dimension * 2 {
                            values[y * 8 + x] = value;
                        }
                    }
                }
            }
            values
        }),
        VarDctMatrixEncoding::Dct4 { params, dct_params } => {
            let mut output = [Vec::new(), Vec::new(), Vec::new()];
            for (output, (params, dct)) in output.iter_mut().zip(params.iter().zip(dct_params)) {
                let matrix = dct_weights(dct, 4, 4, matrix)?;
                *output = vec![0.0; 64];
                for y in 0..4 {
                    for x in 0..4 {
                        output[y * 16 + x * 2] = matrix[y * 4 + x];
                        output[y * 16 + x * 2 + 1] = matrix[y * 4 + x];
                        output[(y * 2 + 1) * 8 + x * 2] = matrix[y * 4 + x];
                        output[(y * 2 + 1) * 8 + x * 2 + 1] = matrix[y * 4 + x];
                    }
                }
                output[1] /= params[0];
                output[8] /= params[0];
                output[9] /= params[1];
            }
            output
        }
        VarDctMatrixEncoding::Dct4x8 { params, dct_params } => {
            let mut output = [Vec::new(), Vec::new(), Vec::new()];
            for (output, (params, dct)) in output.iter_mut().zip(params.iter().zip(dct_params)) {
                let matrix = dct_weights(dct, 8, 4, matrix)?;
                *output = matrix
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .flat_map(|row| [row, row])
                    .flatten()
                    .copied()
                    .collect();
                output[8] /= params[0];
            }
            output
        }
        VarDctMatrixEncoding::Afv {
            params,
            dct_params,
            dct4x4_params,
        } => {
            const FREQUENCIES: [f32; 16] = [
                0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0,
                7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
            ];
            let mut output = [Vec::new(), Vec::new(), Vec::new()];
            for (output, ((params, dct), dct4)) in output
                .iter_mut()
                .zip(params.iter().zip(dct_params).zip(dct4x4_params))
            {
                let mut params = *params;
                for value in &mut params[..6] {
                    *value *= 64.0;
                }
                let weights_4x8 = dct_weights(dct, 8, 4, matrix)?;
                let weights_4x4 = dct_weights(dct4, 4, 4, matrix)?;
                let mut bands = [params[5], 0.0, 0.0, 0.0];
                for index in 1..4 {
                    bands[index] = bands[index - 1] * multiplier(params[index + 5]);
                }
                *output = vec![0.0; 64];
                for y in 0..4 {
                    for x in 0..4 {
                        output[16 * y + 2 * x] = match (x, y) {
                            (0, 0) => 1.0,
                            (0, 1) => params[2],
                            (1, 0) => params[3],
                            (1, 1) => params[4],
                            _ => interpolate(
                                FREQUENCIES[y * 4 + x] - FREQUENCIES[2],
                                FREQUENCIES[15] - FREQUENCIES[2] + 1e-6,
                                &bands,
                            ),
                        };
                    }
                }
                for (y, ((rows, weights_8), weights_4)) in output
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(weights_4x8.as_chunks::<8>().0.iter())
                    .zip(weights_4x4.as_chunks::<4>().0.iter())
                    .enumerate()
                {
                    let (row0, row1) = rows.split_at_mut(8);
                    for (x, (value, &weight)) in row1.iter_mut().zip(weights_8).enumerate() {
                        *value = if y == 0 && x == 0 { params[0] } else { weight };
                    }
                    for (x, (pair, &weight)) in row0
                        .as_chunks_mut::<2>()
                        .0
                        .iter_mut()
                        .zip(weights_4)
                        .enumerate()
                    {
                        pair[1] = if y == 0 && x == 0 { params[1] } else { weight };
                    }
                }
            }
            output
        }
    };
    let mut output = output;
    for value in output.iter_mut().flatten() {
        *value = 1.0 / *value;
        if !value.is_finite() || *value <= 0.0 || *value >= 1e8 {
            return Err(VarDctMatrixError::Value {
                matrix,
                reason: "expanded value is non-positive, non-finite, or too large",
            });
        }
    }
    Ok(output)
}

fn transpose(channels: &[Vec<f32>; 3], width: u32, height: u32) -> [Vec<f32>; 3] {
    std::array::from_fn(|channel| {
        let mut output = vec![0.0; channels[channel].len()];
        for y in 0..height {
            for x in 0..width {
                output[(x * height + y) as usize] = channels[channel][(y * width + x) as usize];
            }
        }
        output
    })
}

impl TransformKind {
    /// Index of this strategy's shared JPEG XL dequantization matrix (0..17).
    #[must_use]
    pub const fn dequant_matrix_index(self) -> usize {
        match self {
            TransformKind::Dct8 => 0,
            TransformKind::Hornuss => 1,
            TransformKind::Dct2x2 => 2,
            TransformKind::Dct4x4 => 3,
            TransformKind::Dct16x16 => 4,
            TransformKind::Dct32x32 => 5,
            TransformKind::Dct16x8 | TransformKind::Dct8x16 => 6,
            TransformKind::Dct32x8 | TransformKind::Dct8x32 => 7,
            TransformKind::Dct32x16 | TransformKind::Dct16x32 => 8,
            TransformKind::Dct4x8 | TransformKind::Dct8x4 => 9,
            TransformKind::Afv0
            | TransformKind::Afv1
            | TransformKind::Afv2
            | TransformKind::Afv3 => 10,
            TransformKind::Dct64x64 => 11,
            TransformKind::Dct64x32 | TransformKind::Dct32x64 => 12,
            TransformKind::Dct128x128 => 13,
            TransformKind::Dct128x64 | TransformKind::Dct64x128 => 14,
            TransformKind::Dct256x256 => 15,
            TransformKind::Dct256x128 | TransformKind::Dct128x256 => 16,
        }
    }
}
