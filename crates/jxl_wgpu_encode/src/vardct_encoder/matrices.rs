//! Validated parametric and raw HF matrices shared by all VarDCT encoder layouts.

use std::sync::Arc;

use jxl_gpu_bitstream::{BitWriter, FiniteF16};

use super::{VarDctCoefficientOrders, VarDctStrategy};
use crate::{BackendError, EncodeError};

/// A validated raw matrix in JPEG XL's wire raster: the shorter axis is horizontal.
///
/// X/Y/B contain positive signed 32-bit weights. Their products with the binary16
/// denominator are the dequantization scales. Transposed strategies share this raster.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarDctRawMatrix {
    pub(super) denominator: FiniteF16,
    pub(super) channels: [Vec<i32>; 3],
    pub(super) width: u32,
}

impl VarDctRawMatrix {
    #[must_use]
    pub const fn denominator(&self) -> FiniteF16 {
        self.denominator
    }

    #[must_use]
    pub fn channels(&self) -> [&[i32]; 3] {
        self.channels.each_ref().map(Vec::as_slice)
    }

    #[must_use]
    pub fn extent(&self) -> jxl_gpu_protocol::Extent2d {
        jxl_gpu_protocol::Extent2d {
            width: self.width,
            height: self.channels[0].len() as u32 / self.width,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum MatrixSelection {
    Parametric(Arc<VarDctMatrixEncoding>),
    Raw(Arc<VarDctRawMatrix>),
}

impl MatrixSelection {
    fn encoding_id(&self) -> u8 {
        match self {
            Self::Parametric(encoding) => encoding.encoding_id(),
            Self::Raw(_) => 7,
        }
    }
}

/// Parametric JPEG XL HF matrix modes with exact finite binary16 wire parameters.
pub type VarDctMatrixEncoding = jxl_gpu_protocol::VarDctMatrixEncoding<FiniteF16>;

/// Caller-selected dequantization matrices for the 17 JPEG XL matrix families.
///
/// Transposed strategies share parameters; all AFV orientations share one family.
/// Unspecified families use the standard defaults. Parametric modes 0–6 and raw
/// Modular side images (mode 7) are supported; selection from image content is separate.
///
/// ```
/// use jxl_wgpu_encode::{VarDctConfig, VarDctDequantMatrices, VarDctMatrixEncoding, VarDctStrategy};
/// use jxl_gpu_bitstream::FiniteF16;
/// let one = FiniteF16::from_bits(0x3c00).unwrap();
/// let config = VarDctConfig {
///     dequant_matrices: VarDctDequantMatrices::default().with_matrix(
///         VarDctStrategy::Dct8,
///         VarDctMatrixEncoding::Dct(std::array::from_fn(|_| vec![one])),
///     )?,
///     ..Default::default()
/// };
/// # Ok::<(), jxl_wgpu_encode::EncodeError>(())
/// ```
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct VarDctDequantMatrices {
    families: [Option<MatrixSelection>; 17],
}

impl std::fmt::Debug for VarDctDequantMatrices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarDctDequantMatrices")
            .field(
                "modes",
                &self.families.each_ref().map(|encoding| {
                    encoding
                        .as_ref()
                        .map_or(0, |encoding| encoding.encoding_id())
                }),
            )
            .finish()
    }
}

impl VarDctDequantMatrices {
    /// Replaces one shared matrix family after bounded shape and numerical validation.
    ///
    /// DCT band vectors must have equal X/Y/B lengths in 1..=16; modes 1–5 require
    /// an 8×8 family. Expanded dequantization values must be finite and in (0, 1e8).
    /// `Default` restores that family's standard matrix and its default wire encoding.
    pub fn with_matrix(
        mut self,
        strategy: VarDctStrategy,
        encoding: VarDctMatrixEncoding,
    ) -> Result<Self, EncodeError> {
        encoding.validate_shape(strategy)?;
        encoding.map(FiniteF16::to_f32).expand(strategy)?;
        self.families[strategy.dequant_matrix_index()] =
            (!matches!(encoding, VarDctMatrixEncoding::Default))
                .then(|| MatrixSelection::Parametric(Arc::new(encoding)));
        Ok(self)
    }

    /// Returns parametric metadata, or `None` for default or raw matrices.
    #[must_use]
    pub fn encoding(&self, strategy: VarDctStrategy) -> Option<&VarDctMatrixEncoding> {
        match &self.families[strategy.dequant_matrix_index()] {
            Some(MatrixSelection::Parametric(encoding)) => Some(encoding),
            _ => None,
        }
    }

    /// Replaces one family with a raw matrix. Each X/Y/B vector is a row-major
    /// raster with width `min(transform_width, transform_height)` and exactly
    /// the transform's sample count. All scales must be finite and in `(0, 1e8)`.
    /// Prediction, signed tokenization and prefix packing execute on GPU at submission.
    pub fn with_raw_matrix(
        mut self,
        strategy: VarDctStrategy,
        denominator: FiniteF16,
        channels: [Vec<i32>; 3],
    ) -> Result<Self, EncodeError> {
        let extent = strategy.pixel_extent();
        let area = (extent.width * extent.height) as usize;
        if denominator.to_f32() <= 0.0
            || channels.iter().any(|channel| channel.len() != area)
            || channels.iter().flatten().any(|&value| {
                let scale = value as f32 * denominator.to_f32();
                value <= 0 || !scale.is_finite() || scale <= 0.0 || scale >= 1e8
            })
        {
            return Err(jxl_gpu_protocol::VarDctMatrixError::Value {
                matrix: strategy.dequant_matrix_index(),
                reason: "raw matrix requires exact dimensions, positive samples and scales in (0, 1e8)",
            }.into());
        }
        self.families[strategy.dequant_matrix_index()] =
            Some(MatrixSelection::Raw(Arc::new(VarDctRawMatrix {
                denominator,
                channels,
                width: extent.width.min(extent.height),
            })));
        Ok(self)
    }

    #[must_use]
    pub fn raw_matrix(&self, strategy: VarDctStrategy) -> Option<&VarDctRawMatrix> {
        self.raw_family(strategy.dequant_matrix_index())
    }

    pub(super) fn raw_family(&self, index: usize) -> Option<&VarDctRawMatrix> {
        match &self.families[index] {
            Some(MatrixSelection::Raw(raw)) => Some(raw),
            _ => None,
        }
    }

    pub(super) fn metadata(
        &self,
        strategy: VarDctStrategy,
        orders: &VarDctCoefficientOrders,
    ) -> Result<Vec<[u32; 6]>, EncodeError> {
        let scales = if let Some(raw) = self.raw_matrix(strategy) {
            (0..raw.channels[0].len())
                .map(|index| {
                    std::array::from_fn(|c| {
                        raw.channels[c][index] as f32 * raw.denominator.to_f32()
                    })
                })
                .collect()
        } else {
            match self.encoding(strategy) {
                Some(encoding) => encoding.map(FiniteF16::to_f32).expand(strategy)?,
                None => strategy.default_dequant_matrix(),
            }
            .scales
        };
        Ok(scales
            .into_iter()
            .zip(orders.indices(strategy))
            .map(|([x, y, b], [ox, oy, ob])| [x.to_bits(), y.to_bits(), b.to_bits(), ox, oy, ob])
            .collect())
    }

    #[cfg(test)]
    pub(super) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        self.write_with_raw(output, super::raw_matrices::Fragments::default())
    }

    pub(super) fn write_with_raw(
        &self,
        output: &mut BitWriter,
        raw_fragments: super::raw_matrices::Fragments<'_>,
    ) -> Result<(), EncodeError> {
        let all_default = self.families.iter().all(Option::is_none);
        output.write_bits(u64::from(all_default), 1)?;
        if all_default {
            return Ok(());
        }
        for (index, selection) in self.families.iter().enumerate() {
            if let Some(MatrixSelection::Raw(raw)) = selection {
                output.write_bits(7, 3)?;
                output.write_bits(u64::from(raw.denominator.to_bits()), 16)?;
                output.write_bits(3, 4)?; // global MA tree, default WP, no transforms
                raw_fragments.append(output, index)?;
                continue;
            }
            let encoding = match selection {
                Some(MatrixSelection::Parametric(encoding)) => encoding.as_ref(),
                None => &VarDctMatrixEncoding::Default,
                _ => return Err(BackendError::Invariant("raw matrix not handled").into()),
            };
            output.write_bits(u64::from(encoding.encoding_id()), 3)?;
            match encoding {
                VarDctMatrixEncoding::Default => {}
                VarDctMatrixEncoding::Hornuss(params) => write_fixed(output, params)?,
                VarDctMatrixEncoding::Dct2(params) => write_fixed(output, params)?,
                VarDctMatrixEncoding::Dct4 { params, dct_params } => {
                    write_fixed(output, params)?;
                    write_bands(output, dct_params)?;
                }
                VarDctMatrixEncoding::Dct4x8 { params, dct_params } => {
                    write_fixed(output, params)?;
                    write_bands(output, dct_params)?;
                }
                VarDctMatrixEncoding::Afv {
                    params,
                    dct_params,
                    dct4x4_params,
                } => {
                    write_fixed(output, params)?;
                    write_bands(output, dct_params)?;
                    write_bands(output, dct4x4_params)?;
                }
                VarDctMatrixEncoding::Dct(params) => write_bands(output, params)?,
            }
        }
        Ok(())
    }
}

fn write_fixed<const N: usize>(
    output: &mut BitWriter,
    params: &[[FiniteF16; N]; 3],
) -> Result<(), EncodeError> {
    for value in params.iter().flatten() {
        output.write_bits(u64::from(value.to_bits()), 16)?;
    }
    Ok(())
}

fn write_bands(output: &mut BitWriter, params: &[Vec<FiniteF16>; 3]) -> Result<(), EncodeError> {
    output.write_bits((params[0].len() - 1) as u64, 4)?;
    for value in params.iter().flatten() {
        output.write_bits(u64::from(value.to_bits()), 16)?;
    }
    Ok(())
}
