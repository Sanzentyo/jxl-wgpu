//! Metadata-only lowering of display luminance for the common RGB transfer helpers.

use crate::{Error, Result};
use jxl_gpu_protocol::{ColorMatrix, RgbColorSpace, WhitePointAdaptation};

/// RGB luminance coefficients and the HLG OOTF exponent in one GPU vector.
/// `to_scene` selects the inverse OOTF used before HLG encoding; otherwise it
/// selects the forward OOTF applied after decoding HLG scene values.
pub fn display_luminance(
    space: RgbColorSpace,
    intensity_target: f32,
    to_scene: bool,
) -> Result<[f32; 4]> {
    if !intensity_target.is_finite() || intensity_target <= 0.0 {
        return Err(Error::InvalidPayload(
            "display intensity must be finite and positive".into(),
        ));
    }
    let chromaticities = space.chromaticities().ok_or_else(|| {
        Error::InvalidPayload("display luminance requires RGB chromaticities".into())
    })?;
    let matrix =
        ColorMatrix::rgb_to_xyz(space, chromaticities.white, WhitePointAdaptation::Bradford)
            .map_err(|error| Error::InvalidPayload(error.to_string()))?;
    let [r, g, b] = matrix.rows()[1].map(|v| v as f32);
    let gamma = 1.2 * 1.111_f64.powf((f64::from(intensity_target) / 1000.0).log2());
    let exponent = if to_scene { gamma.recip() } else { gamma } - 1.0;
    let parameters = [r, g, b, exponent as f32];
    if parameters.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidPayload(
            "display luminance exceeds GPU F32 storage".into(),
        ));
    }
    Ok(parameters)
}
