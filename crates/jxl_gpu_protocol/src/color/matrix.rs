//! Host-side lowering of declared CIE geometry. No image samples enter this module.

use crate::{Chromaticity, RgbChromaticities, RgbColorSpace, WhitePointAdaptation};

type Matrix = [[f64; 3]; 3];
pub(crate) const IDENTITY: Matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A finite colorimetric matrix, calculated in f64 before backend-specific lowering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix(Matrix);

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum ColorMatrixError {
    #[error("RGB conversion requires a defined source color space")]
    UndefinedSource,
    #[error("RGB conversion requires a defined target color space")]
    UndefinedTarget,
    #[error("CIE xy coordinates do not define finite XYZ values")]
    WhitePoint,
    #[error("white points do not define finite Bradford adaptation")]
    Adaptation,
    #[error("color geometry defines a singular or non-finite matrix")]
    Singular,
    #[error("color matrix coefficients are not finite")]
    NonFinite,
}

impl ColorMatrix {
    pub const IDENTITY: Self = Self(IDENTITY);

    /// Converts linear RGB with an explicit reference-white policy. Both endpoints are
    /// validated even when their declarations match and the resulting matrix is identity.
    pub fn between_rgb(
        source: RgbColorSpace,
        target: RgbColorSpace,
        adaptation: WhitePointAdaptation,
    ) -> Result<Self, ColorMatrixError> {
        let source = source
            .chromaticities()
            .ok_or(ColorMatrixError::UndefinedSource)?;
        let target = target
            .chromaticities()
            .ok_or(ColorMatrixError::UndefinedTarget)?;
        let source_to_xyz = to_xyz(source)?;
        let xyz_to_target = inverse(to_xyz(target)?)?;
        Self::checked(if source == target {
            IDENTITY
        } else {
            multiply(
                xyz_to_target,
                multiply(
                    adapt(source.white, target.white, adaptation)?,
                    source_to_xyz,
                ),
            )
        })
    }

    /// Converts linear RGB to XYZ whose unit Y is relative to the supplied white point.
    /// This preserves the f64 coefficients needed to connect an exact ICC colorant matrix.
    pub fn rgb_to_xyz(
        source: RgbColorSpace,
        white: Chromaticity,
        adaptation: WhitePointAdaptation,
    ) -> Result<Self, ColorMatrixError> {
        let source = source
            .chromaticities()
            .ok_or(ColorMatrixError::UndefinedSource)?;
        let matrix = to_xyz(source)?;
        inverse(matrix)?;
        xyz(white)?;
        Self::checked(multiply(adapt(source.white, white, adaptation)?, matrix))
    }

    /// Converts XYZ relative to the supplied white point to linear RGB. Values outside
    /// the RGB unit cube remain meaningful; this metadata matrix specifies no clipping.
    pub fn xyz_to_rgb(
        white: Chromaticity,
        target: RgbColorSpace,
        adaptation: WhitePointAdaptation,
    ) -> Result<Self, ColorMatrixError> {
        let target = target
            .chromaticities()
            .ok_or(ColorMatrixError::UndefinedTarget)?;
        let matrix = inverse(to_xyz(target)?)?;
        xyz(white)?;
        Self::checked(multiply(matrix, adapt(white, target.white, adaptation)?))
    }

    #[must_use]
    pub const fn rows(&self) -> &[[f64; 3]; 3] {
        &self.0
    }

    fn checked(matrix: Matrix) -> Result<Self, ColorMatrixError> {
        if matrix.iter().flatten().any(|value| !value.is_finite()) {
            Err(ColorMatrixError::NonFinite)
        } else {
            Ok(Self(matrix))
        }
    }
}

fn to_xyz(color: RgbChromaticities) -> Result<Matrix, ColorMatrixError> {
    // Homogeneous columns retain valid primaries on y=0; only the white needs Y=1.
    let primaries = [color.red, color.green, color.blue]
        .map(|point| [point.x(), point.y(), 1.0 - point.x() - point.y()]);
    let columns = std::array::from_fn(|row| std::array::from_fn(|column| primaries[column][row]));
    let scale = vector(inverse(columns)?, xyz(color.white)?);
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| columns[row][column] * scale[column])
    }))
}

fn xyz(point: Chromaticity) -> Result<[f64; 3], ColorMatrixError> {
    let value = [
        point.x() / point.y(),
        1.0,
        (1.0 - point.x() - point.y()) / point.y(),
    ];
    if value.iter().any(|value| !value.is_finite()) {
        return Err(ColorMatrixError::WhitePoint);
    }
    Ok(value)
}

fn adapt(
    source: Chromaticity,
    target: Chromaticity,
    policy: WhitePointAdaptation,
) -> Result<Matrix, ColorMatrixError> {
    if source == target || policy == WhitePointAdaptation::None {
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
        return Err(ColorMatrixError::Adaptation);
    }
    let scaled = std::array::from_fn(|row| BRADFORD[row].map(|value| value * ratio[row]));
    Ok(multiply(inverse(BRADFORD)?, scaled))
}

pub(crate) fn inverse(matrix: Matrix) -> Result<Matrix, ColorMatrixError> {
    let [[a, b, c], [d, e, f], [g, h, i]] = matrix;
    let adjugate = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let determinant = a * adjugate[0][0] + b * adjugate[1][0] + c * adjugate[2][0];
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(ColorMatrixError::Singular);
    }
    let inverse = adjugate.map(|row| row.map(|value| value / determinant));
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ColorMatrixError::Singular);
    }
    Ok(inverse)
}

fn vector(matrix: Matrix, values: [f64; 3]) -> [f64; 3] {
    matrix.map(|row| row[0] * values[0] + row[1] * values[1] + row[2] * values[2])
}

pub(crate) fn multiply(lhs: Matrix, rhs: Matrix) -> Matrix {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            lhs[row][0] * rhs[0][column]
                + lhs[row][1] * rhs[1][column]
                + lhs[row][2] * rhs[2][column]
        })
    })
}

#[cfg(test)]
mod tests;
