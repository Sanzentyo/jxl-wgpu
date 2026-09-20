// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// libjxl 0.12.0 metadata matrix arithmetic, BSD-3-Clause; see THIRD_PARTY.md.
use super::{IccProfileError as Error, Result};

pub(super) type Matrix = [[f32; 3]; 3];

const BRADFORD: Matrix = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const BRADFORD_INV: Matrix = [
    [0.986_992_9, -0.147_054_3, 0.159_962_7],
    [0.432_305_3, 0.518_360_3, 0.049_291_2],
    [-0.008_528_7, 0.040_042_8, 0.968_486_7],
];

pub(super) fn vector(a: Matrix, b: [f32; 3]) -> [f32; 3] {
    a.map(|row| {
        row.into_iter()
            .zip(b)
            .fold(0.0, |sum, (x, y)| sum + f64::from(x) * f64::from(y)) as f32
    })
}

pub(super) fn product(a: Matrix, b: Matrix) -> Matrix {
    a.map(|row| {
        std::array::from_fn(|c| {
            (f64::from(row[0]) * f64::from(b[0][c])
                + f64::from(row[1]) * f64::from(b[1][c])
                + f64::from(row[2]) * f64::from(b[2][c])) as f32
        })
    })
}

fn inverse(m: Matrix) -> Result<Matrix> {
    let m = m.map(|row| row.map(f64::from));
    let mut adjugate = [[0.0; 3]; 3];
    for (r, row) in adjugate.iter_mut().enumerate() {
        for (c, value) in row.iter_mut().enumerate() {
            *value = m[(c + 1) % 3][(r + 1) % 3] * m[(c + 2) % 3][(r + 2) % 3]
                - m[(c + 1) % 3][(r + 2) % 3] * m[(c + 2) % 3][(r + 1) % 3];
        }
    }
    let det = m[0][0] * adjugate[0][0] + m[0][1] * adjugate[1][0] + m[0][2] * adjugate[2][0];
    if !det.is_finite() || det.abs() < 1e-10 {
        return Err(Error::Invalid("singular primaries"));
    }
    let reciprocal = 1.0 / det;
    Ok(adjugate.map(|row| row.map(|v| (v * reciprocal) as f32)))
}

fn white([x, y]: [f32; 2]) -> Result<[f32; 3]> {
    if !(0.0..=1.0).contains(&x) || !(0.0 < y && y <= 1.0) {
        return Err(Error::Invalid("white point"));
    }
    let w = [x / y, 1.0, (1.0 - x - y) / y];
    if !w.into_iter().all(f32::is_finite) {
        return Err(Error::Invalid("white point"));
    }
    Ok(w)
}

pub(super) fn adaptation(xy: [f64; 2]) -> Result<Matrix> {
    let lms = vector(BRADFORD, white(xy.map(|x| x as f32))?);
    let lms50 = vector(BRADFORD, [0.96422, 1.0, 0.82521]);
    let mut diagonal = [[0.0; 3]; 3];
    for c in 0..3 {
        let ratio = lms50[c] / lms[c];
        if !ratio.is_finite() {
            return Err(Error::Invalid("chromatic adaptation"));
        }
        diagonal[c][c] = ratio;
    }
    Ok(product(BRADFORD_INV, product(diagonal, BRADFORD)))
}

pub(super) fn primaries(points: [[f64; 2]; 3], xy: [f64; 2]) -> Result<Matrix> {
    let points = points.map(|p| p.map(|v| v as f32));
    let p = [
        points.map(|v| v[0]),
        points.map(|v| v[1]),
        points.map(|v| 1.0 - v[0] - v[1]),
    ];
    let scale = vector(inverse(p)?, white(xy.map(|v| v as f32))?);
    let diagonal = [
        [scale[0], 0.0, 0.0],
        [0.0, scale[1], 0.0],
        [0.0, 0.0, scale[2]],
    ];
    Ok(product(p, diagonal))
}

pub(super) fn grey_white([x, y]: [f64; 2]) -> Result<[f32; 3]> {
    if y.abs() < 1e-12 {
        return Err(Error::Invalid("white point"));
    }
    let factor = (1.0 / y) as f32;
    Ok([
        (x * f64::from(factor)) as f32,
        1.0,
        ((1.0 - x - y) * f64::from(factor)) as f32,
    ])
}
