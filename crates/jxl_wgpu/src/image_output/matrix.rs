//! Backend lowering of the shared f64 colorimetric metadata matrix.

use crate::{Error, Result};
use jxl_gpu_protocol::{ColorMatrix, ColorMatrixError, RgbColorSpace, WhitePointAdaptation};

pub(super) const IDENTITY: [[f64; 3]; 3] = *ColorMatrix::IDENTITY.rows();

/// Lower declared RGB chromaticities and an explicit white-point policy to GPU F32 rows.
/// Rejects singular geometry and non-finite coefficients before submission.
pub fn rgb_color_matrix(
    source: RgbColorSpace,
    target: RgbColorSpace,
    adaptation: WhitePointAdaptation,
) -> Result<[[f32; 4]; 3]> {
    let matrix =
        ColorMatrix::between_rgb(source, target, adaptation).map_err(|error| match error {
            ColorMatrixError::UndefinedSource | ColorMatrixError::UndefinedTarget => {
                Error::Unsupported(error.to_string())
            }
            _ => Error::InvalidPayload(error.to_string()),
        })?;
    let lowered = matrix
        .rows()
        .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32, 0.0]);
    if lowered.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::InvalidPayload(
            "RGB conversion coefficients exceed finite GPU F32 storage".into(),
        ));
    }
    Ok(lowered)
}

#[cfg(test)]
mod tests;
