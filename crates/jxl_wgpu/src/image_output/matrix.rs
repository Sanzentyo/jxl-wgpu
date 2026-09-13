//! RGB-to-RGB matrices derived from declared CIE chromaticities and white points.

use crate::{Error, Result};
use jxl_gpu_protocol::{Chromaticity, RgbChromaticities, RgbColorSpace, WhitePointAdaptation};

type Matrix = [[f64; 3]; 3];
pub(super) const IDENTITY: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Lower declared RGB chromaticities and an explicit white-point policy to GPU F32 rows.
/// Rejects singular geometry and non-finite coefficients before submission.
pub fn rgb_color_matrix(
    source: RgbColorSpace,
    target: RgbColorSpace,
    adaptation: WhitePointAdaptation,
) -> Result<[[f32; 4]; 3]> {
    let source = source.chromaticities().ok_or_else(|| {
        Error::Unsupported("RGB conversion requires a defined source color space".into())
    })?;
    let target = target.chromaticities().ok_or_else(|| {
        Error::Unsupported("RGB conversion requires a defined target color space".into())
    })?;
    let source_to_xyz = to_xyz(source)?;
    let xyz_to_target = inverse(to_xyz(target)?)?;
    let matrix = if source == target {
        IDENTITY
    } else {
        let adaptation = match adaptation {
            WhitePointAdaptation::Bradford => adapt(source.white, target.white)?,
            WhitePointAdaptation::None => IDENTITY,
        };
        multiply(xyz_to_target, multiply(adaptation, source_to_xyz))
    };
    let lowered = matrix.map(|row| [row[0] as f32, row[1] as f32, row[2] as f32, 0.0]);
    if lowered.iter().flatten().any(|value| !value.is_finite()) {
        return invalid("RGB conversion coefficients exceed finite GPU F32 storage");
    }
    Ok(lowered)
}

fn to_xyz(color: RgbChromaticities) -> Result<Matrix> {
    // Homogeneous XYZ columns also represent valid primaries on y=0. Only the
    // reference white needs Y=1 normalization; dividing each primary by y loses these axes.
    let primaries = [color.red, color.green, color.blue]
        .map(|point| [point.x(), point.y(), 1.0 - point.x() - point.y()]);
    let columns = std::array::from_fn(|row| std::array::from_fn(|column| primaries[column][row]));
    let scale = vector(inverse(columns)?, xyz(color.white)?);
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| columns[row][column] * scale[column])
    }))
}

fn xyz(point: Chromaticity) -> Result<[f64; 3]> {
    let value = [
        point.x() / point.y(),
        1.0,
        (1.0 - point.x() - point.y()) / point.y(),
    ];
    if value.iter().any(|value| !value.is_finite()) {
        return invalid("CIE xy coordinates do not define finite XYZ values");
    }
    Ok(value)
}

fn adapt(source: Chromaticity, target: Chromaticity) -> Result<Matrix> {
    if source == target {
        return Ok(IDENTITY);
    }
    const BRADFORD: Matrix = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let source = vector(BRADFORD, xyz(source)?);
    let target = vector(BRADFORD, xyz(target)?);
    let ratio = std::array::from_fn::<_, 3, _>(|index| target[index] / source[index]);
    if ratio.iter().any(|value| !value.is_finite()) {
        return invalid("white points do not define finite Bradford adaptation");
    }
    let scaled = std::array::from_fn(|row| BRADFORD[row].map(|value| value * ratio[row]));
    Ok(multiply(inverse(BRADFORD)?, scaled))
}

fn inverse(matrix: Matrix) -> Result<Matrix> {
    let [[a, b, c], [d, e, f], [g, h, i]] = matrix;
    let adjugate = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let determinant = a * adjugate[0][0] + b * adjugate[1][0] + c * adjugate[2][0];
    if !determinant.is_finite() || determinant == 0.0 {
        return invalid("RGB chromaticities define a singular or non-finite determinant");
    }
    let inverse = adjugate.map(|row| row.map(|value| value / determinant));
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return invalid("RGB chromaticities define a singular or non-finite matrix");
    }
    Ok(inverse)
}

fn vector(matrix: Matrix, values: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * values[0] + row[1] * values[1] + row[2] * values[2])
}

fn multiply(lhs: Matrix, rhs: Matrix) -> Matrix {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            lhs[row][0] * rhs[0][column]
                + lhs[row][1] * rhs[1][column]
                + lhs[row][2] * rhs[2][column]
        })
    })
}

fn invalid<T>(message: &'static str) -> Result<T> {
    Err(Error::InvalidPayload(message.into()))
}

#[cfg(test)]
mod tests;
