//! Independent F64 transfer equations and tabulated CIE luminances; no production GPU helpers.
use super::color;
use jxl_gpu_formats::{ColorSpace, TransferFunction};

pub fn luminance(space: ColorSpace) -> [f64; 3] {
    match space {
        ColorSpace::Bt709 => [0.212639005871510, 0.715168678767756, 0.072192315360734],
        ColorSpace::Bt2020 => [0.262700212011267, 0.677998071518871, 0.059301716469862],
        ColorSpace::DisplayP3 => [0.228974564069749, 0.691738521836506, 0.079286914093745],
        ColorSpace::CustomRgb(_) => color::luminance(space),
        _ => panic!("HDR reference primaries"),
    }
}

fn ootf_factor(y: f64, exponent: f64) -> f64 {
    if y > 0.0 {
        y.powf(exponent)
    } else {
        // Independently express the native signed-bit logarithm extension in
        // real arithmetic. Its signed exponent wraps at the 2/3 reduction pivot.
        let log = if y == 0.0 {
            -127.0
        } else {
            y.abs().log2()
                + if y.abs() < f64::from(f32::from_bits(0x3f2aaaab)) {
                    256.0
                } else {
                    -256.0
                }
        };
        (log * exponent).exp2()
    }
    .min(1e9)
}

fn ootf(rgb: [f64; 3], space: ColorSpace, nits: f64, inverse: bool) -> [f64; 3] {
    let gamma = 1.2 * 1.111_f64.powf((nits / 1000.0).log2());
    let exponent = if inverse { 1.0 / gamma } else { gamma } - 1.0;
    if exponent.abs() <= 0.01 {
        return rgb;
    }
    let y = rgb
        .into_iter()
        .zip(luminance(space))
        .map(|(v, weight)| v * weight)
        .sum();
    let factor = ootf_factor(y, exponent);
    rgb.map(|v| v * factor)
}

pub fn to_linear(
    rgb: [f64; 3],
    transfer: TransferFunction,
    space: ColorSpace,
    nits: f64,
) -> [f64; 3] {
    let rgb = rgb.map(|v| color::to_linear(v, transfer));
    match transfer {
        TransferFunction::Pq => rgb.map(|v| v * 10000.0 / nits),
        TransferFunction::Hlg => ootf(rgb, space, nits, false),
        _ => rgb,
    }
}

pub fn from_linear(
    rgb: [f64; 3],
    transfer: TransferFunction,
    space: ColorSpace,
    nits: f64,
) -> [f64; 3] {
    let rgb = match transfer {
        TransferFunction::Pq => rgb.map(|v| v * nits / 10000.0),
        TransferFunction::Hlg => ootf(rgb, space, nits, true),
        _ => rgb,
    };
    rgb.map(|v| color::from_linear(v, transfer))
}

pub fn convert(
    rgb: [f64; 3],
    source: TransferFunction,
    target: TransferFunction,
    source_space: ColorSpace,
    target_space: ColorSpace,
    nits: f64,
) -> [f64; 3] {
    let linear = to_linear(rgb, source, source_space, nits);
    let matrix = color::matrix(source_space, target_space);
    let converted = matrix.map(|row| (0..3).map(|c| row[c] * linear[c]).sum());
    from_linear(converted, target, target_space, nits)
}

type Interval = [f64; 2];

fn product(a: Interval, b: Interval) -> Interval {
    let values = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
    [
        values.into_iter().fold(f64::INFINITY, f64::min),
        values.into_iter().fold(f64::NEG_INFINITY, f64::max),
    ]
}
fn ootf_interval(rgb: [Interval; 3], space: ColorSpace, nits: f64, inverse: bool) -> [Interval; 3] {
    let gamma = 1.2 * 1.111_f64.powf((nits / 1000.0).log2());
    let exponent = if inverse { 1.0 / gamma } else { gamma } - 1.0;
    if exponent.abs() <= 0.01 {
        return rgb;
    }
    let y: Interval = std::array::from_fn(|edge| {
        (0..3)
            .map(|c| rgb[c][edge] * luminance(space)[c])
            .sum::<f64>()
    });
    let mut candidates: Vec<_> = y.map(|y| ootf_factor(y, exponent)).into();
    if y[0] <= 0.0 && y[1] >= 0.0 {
        candidates.extend([
            ootf_factor(0.0, exponent),
            if exponent > 0.0 { 0.0 } else { 1e9 },
        ]);
    }
    let pivot = f64::from(f32::from_bits(0x3f2aaaab));
    if y[0] <= -pivot && y[1] >= -pivot {
        candidates.extend(
            [-256.0, 256.0].map(|shift| ((pivot.log2() + shift) * exponent).exp2().min(1e9)),
        );
    }
    let factors = [
        candidates.iter().copied().fold(f64::INFINITY, f64::min),
        candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ];
    rgb.map(|channel| product(channel, factors))
}

/// Propagate the predeclared codec error through EOTF, coupled OOTF, signed
/// primary matrix and OETF, retaining channel/luminance dependence conservatively.
pub fn interval(
    rgb: [f64; 3],
    source: TransferFunction,
    target: TransferFunction,
    source_space: ColorSpace,
    target_space: ColorSpace,
    nits: f64,
    error: f64,
) -> [Interval; 3] {
    let linear = linear_interval(rgb, source, source_space, nits, error);
    let target_linear = color::matrix(source_space, target_space).map(|row| {
        std::array::from_fn(|edge| {
            (0..3)
                .map(|c| row[c] * linear[c][if row[c] >= 0.0 { edge } else { 1 - edge }])
                .sum()
        })
    });
    from_linear_interval(target_linear, target, target_space, nits)
}

/// Decode a fixed source-error interval into display-linear light.
pub fn linear_interval(
    rgb: [f64; 3],
    source: TransferFunction,
    source_space: ColorSpace,
    nits: f64,
    error: f64,
) -> [Interval; 3] {
    let mut linear = rgb.map(|v| {
        [v - error * (1.0 + v.abs()), v + error * (1.0 + v.abs())]
            .map(|v| color::to_linear(v, source))
    });
    if source == TransferFunction::Pq {
        linear = linear.map(|channel| channel.map(|v| v * 10000.0 / nits));
    }
    if source == TransferFunction::Hlg {
        linear = ootf_interval(linear, source_space, nits, false);
    }
    linear
}

/// Encode predeclared linear-light uncertainty through the coupled OOTF and OETF.
pub fn from_linear_interval(
    mut target_linear: [Interval; 3],
    target: TransferFunction,
    target_space: ColorSpace,
    nits: f64,
) -> [Interval; 3] {
    if target == TransferFunction::Pq {
        target_linear = target_linear.map(|channel| channel.map(|v| v * nits / 10000.0));
    }
    if target == TransferFunction::Hlg {
        target_linear = ootf_interval(target_linear, target_space, nits, true);
    }
    target_linear.map(|channel| channel.map(|v| color::from_linear(v, target)))
}
