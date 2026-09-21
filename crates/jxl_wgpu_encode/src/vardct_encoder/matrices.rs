//! Validated parametric HF matrix metadata shared by all VarDCT encoder layouts.

use std::sync::Arc;

use jxl_gpu_bitstream::{BitWriter, FiniteF16};

use super::{VarDctCoefficientOrders, VarDctStrategy};
use crate::EncodeError;

/// Parametric JPEG XL HF matrix modes with exact finite binary16 wire parameters.
pub type VarDctMatrixEncoding = jxl_gpu_protocol::VarDctMatrixEncoding<FiniteF16>;

/// Caller-selected dequantization matrices for the 17 JPEG XL matrix families.
///
/// Transposed strategies share parameters; all AFV orientations share one family.
/// Unspecified families use the standard defaults. This does not select matrices
/// from image content and does not encode raw Modular matrix side images (mode 7).
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
    families: [Option<Arc<VarDctMatrixEncoding>>; 17],
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
            (!matches!(encoding, VarDctMatrixEncoding::Default)).then(|| Arc::new(encoding));
        Ok(self)
    }

    /// Returns this strategy's parameters, or `None` for the standard matrix.
    #[must_use]
    pub fn encoding(&self, strategy: VarDctStrategy) -> Option<&VarDctMatrixEncoding> {
        self.families[strategy.dequant_matrix_index()].as_deref()
    }

    pub(super) fn metadata(
        &self,
        strategy: VarDctStrategy,
        orders: &VarDctCoefficientOrders,
    ) -> Result<Vec<[u32; 6]>, EncodeError> {
        let matrix = match self.encoding(strategy) {
            Some(encoding) => encoding.map(FiniteF16::to_f32).expand(strategy)?,
            None => strategy.default_dequant_matrix(),
        };
        Ok(matrix
            .scales
            .into_iter()
            .zip(orders.indices(strategy))
            .map(|([x, y, b], [ox, oy, ob])| [x.to_bits(), y.to_bits(), b.to_bits(), ox, oy, ob])
            .collect())
    }

    pub(super) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        let all_default = self.families.iter().all(Option::is_none);
        output.write_bits(u64::from(all_default), 1)?;
        if all_default {
            return Ok(());
        }
        for encoding in &self.families {
            let encoding = encoding
                .as_deref()
                .unwrap_or(&VarDctMatrixEncoding::Default);
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
