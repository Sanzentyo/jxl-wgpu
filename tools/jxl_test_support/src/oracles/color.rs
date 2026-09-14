//! Independent f64 RGB colorimetry and jxl-oxide reconstruction, for tests and offline fixtures.
use crate::fixtures::original_color as corpus;
use jxl_gpu_formats::{ColorSpace, TransferFunction};

/// Unbounded, pre-OETF samples for unreferenced XYB stills. Gamma/DCI original pixels are
/// non-invertible after the codec black floor, so they cannot serve as a linear-output oracle.
pub fn linear_original_still(case: &corpus::Case) -> Vec<[f64; 4]> {
    assert!(case.mode.xyb() && !case.sequence);
    let mut image = jxl_oxide::JxlImage::read_with_defaults(case.bytes().as_slice()).unwrap();
    image.set_render_spot_color(false);
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).unwrap();
    let pixels = render.image_all_channels();
    assert_eq!(pixels.channels(), 4);
    pixels
        .buf()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| {
            let linear = xyb_original_linear([p[0], p[1], p[2]].map(f64::from), case);
            [linear[0], linear[1], linear[2], f64::from(p[3])]
        })
        .collect()
}

pub fn xyb_original_linear(mut rgb: [f64; 3], case: &corpus::Case) -> [f64; 3] {
    use jxl_gpu_protocol::{Chromaticity as Xy, RgbChromaticities};
    let space = if case.profile.grayscale || case.profile.space() == ColorSpace::Bt709 {
        ColorSpace::Bt709
    } else {
        // libjxl v0.12.0 ColorEncoding::GetPrimaries(kSRGB), used by OutputEncodingInfo
        // when an XYB presentation targets an original non-sRGB RGB profile.
        ColorSpace::CustomRgb(RgbChromaticities {
            red: Xy::new(0.639998686, 0.330010138).unwrap(),
            green: Xy::new(0.300003784, 0.600003357).unwrap(),
            blue: Xy::new(0.150002046, 0.059997204).unwrap(),
            white: Xy::D65,
        })
    };
    if case.profile.grayscale {
        rgb = [(0..3).map(|c| rgb[c] * [0.2126, 0.7152, 0.0722][c]).sum(); 3];
    }
    matrix(space, case.profile.space()).map(|row| (0..3).map(|c| row[c] * rgb[c]).sum())
}

pub fn to_linear(value: f64, transfer: TransferFunction) -> f64 {
    match transfer {
        TransferFunction::Linear => value,
        TransferFunction::Srgb | TransferFunction::Sycc => {
            let a = value.abs();
            if a <= 0.04045 {
                value / 12.92
            } else {
                ((a + 0.055) / 1.055).powf(2.4).copysign(value)
            }
        }
        // libjxl v0.12.0 TF_709 and jxl-oxide 0.12.6 extend the linear toe below zero.
        // Rust jxl 0.6.0 instead reflects the positive power curve for negative inputs.
        TransferFunction::Bt709 => {
            if value <= 0.081 {
                value / 4.5
            } else {
                ((value + 0.099) / 1.099).powf(1.0 / 0.45)
            }
        }
        TransferFunction::Gamma(exponent) => value.max(0.0).powf(1.0 / f64::from(exponent.value())),
        TransferFunction::Dci => {
            if value <= 0.0 {
                value
            } else {
                value.powf(2.6)
            }
        }
        TransferFunction::Bt2020 => {
            let a = 1.09929682680944;
            let b = 0.018053968510807;
            if value.abs() < 4.5 * b {
                value / 4.5
            } else {
                ((value.abs() + a - 1.0) / a)
                    .powf(1.0 / 0.45)
                    .copysign(value)
            }
        }
        TransferFunction::Pq => {
            let p = value.abs().powf(32.0 / 2523.0);
            ((p - 3424.0 / 4096.0).max(0.0) / (2413.0 / 128.0 - 2392.0 / 128.0 * p).max(1e-10))
                .powf(16384.0 / 2610.0)
                .copysign(value)
        }
        TransferFunction::Hlg => {
            let a = 0.17883277;
            let magnitude = value.abs();
            let linear = if magnitude <= 0.5 {
                magnitude * magnitude / 3.0
            } else {
                (((magnitude - 0.5599107295) / a).exp() + 1.0 - 4.0 * a) / 12.0
            };
            linear.copysign(value)
        }
        TransferFunction::Undefined | TransferFunction::Smpte240M => {
            panic!("unsupported reference transfer")
        }
    }
}

pub fn from_linear(value: f64, transfer: TransferFunction) -> f64 {
    match transfer {
        TransferFunction::Linear => value,
        TransferFunction::Srgb | TransferFunction::Sycc => {
            let a = value.abs();
            if a <= 0.0031308 {
                value * 12.92
            } else {
                (1.055 * a.powf(1.0 / 2.4) - 0.055).copysign(value)
            }
        }
        TransferFunction::Bt709 => {
            if value <= 0.018 {
                value * 4.5
            } else {
                1.099 * value.powf(0.45) - 0.099
            }
        }
        TransferFunction::Gamma(exponent) => value.max(0.0).powf(f64::from(exponent.value())),
        TransferFunction::Dci => {
            if value <= 0.0 {
                value
            } else {
                value.powf(1.0 / 2.6)
            }
        }
        TransferFunction::Bt2020 => {
            if value.abs() < 0.018053968510807 {
                value * 4.5
            } else {
                (1.09929682680944 * value.abs().powf(0.45) - 0.09929682680944).copysign(value)
            }
        }
        TransferFunction::Pq => {
            let p = value.abs().powf(2610.0 / 16384.0);
            ((3424.0 / 4096.0 + 2413.0 / 128.0 * p) / (1.0 + 2392.0 / 128.0 * p))
                .powf(2523.0 / 32.0)
                .copysign(value)
        }
        TransferFunction::Hlg => {
            let a = 0.17883277;
            let magnitude = value.abs();
            let encoded = if magnitude <= 1.0 / 12.0 {
                (3.0 * magnitude).sqrt()
            } else {
                a * (12.0 * magnitude - (1.0 - 4.0 * a)).ln() + 0.5599107295
            };
            encoded.copysign(value)
        }
        TransferFunction::Undefined | TransferFunction::Smpte240M => {
            panic!("unsupported reference transfer")
        }
    }
}

type Matrix = [[f64; 3]; 3];

fn inverse(matrix: Matrix) -> Matrix {
    let mut augmented: [[f64; 6]; 3] = std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            if c < 3 {
                matrix[r][c]
            } else if c - 3 == r {
                1.0
            } else {
                0.0
            }
        })
    });
    for c in 0..3 {
        let pivot = (c..3)
            .max_by(|&a, &b| augmented[a][c].abs().total_cmp(&augmented[b][c].abs()))
            .unwrap();
        augmented.swap(c, pivot);
        let scale = augmented[c][c];
        assert!(scale.abs() > 1e-10);
        for value in &mut augmented[c] {
            *value /= scale;
        }
        for r in 0..3 {
            if r == c {
                continue;
            }
            let scale = augmented[r][c];
            let pivot = augmented[c];
            for (value, pivot) in augmented[r].iter_mut().zip(pivot) {
                *value -= scale * pivot;
            }
        }
    }
    augmented.map(|row| row[3..].try_into().unwrap())
}

fn xyz(space: ColorSpace) -> Matrix {
    // CIE xy from BT.709, BT.2020 and Display-P3; common D65 is (0.3127, 0.3290).
    let xy = match space {
        ColorSpace::Bt709 => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
        ColorSpace::Bt2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        ColorSpace::DisplayP3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
        ColorSpace::CustomRgb(color) => {
            [color.red, color.green, color.blue].map(|p| [p.x(), p.y()])
        }
        _ => unreachable!(),
    };
    let matrix: Matrix = std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let [x, y] = xy[c];
            [x / y, 1.0, (1.0 - x - y) / y][r]
        })
    });
    let [x, y] = white(space);
    let white = [x / y, 1.0, (1.0 - x - y) / y];
    let scale = inverse(matrix).map(|row| (0..3).map(|c| row[c] * white[c]).sum::<f64>());
    matrix.map(|row| std::array::from_fn(|c| row[c] * scale[c]))
}

fn white(space: ColorSpace) -> [f64; 2] {
    if let ColorSpace::CustomRgb(color) = space {
        [color.white.x(), color.white.y()]
    } else {
        [0.3127, 0.3290]
    }
}

pub fn matrix(source: ColorSpace, target: ColorSpace) -> Matrix {
    matrix_with_adaptation(source, target, true)
}

pub fn matrix_with_adaptation(source: ColorSpace, target: ColorSpace, adapt: bool) -> Matrix {
    multiply(
        inverse(xyz(target)),
        adapted_xyz(source, white(target), adapt),
    )
}

/// Physical PCS XYZ, adapted to ICC's exact encoded D50 rather than a surrogate RGB profile.
pub fn pcs_matrix(source: ColorSpace) -> Matrix {
    let white = [0xf6d6 as f64 / 65536.0, 1.0, 0xd32d as f64 / 65536.0];
    let sum: f64 = white.iter().sum();
    adapted_xyz(source, [white[0] / sum, white[1] / sum], true)
}

pub fn inverse_pcs_matrix(target: ColorSpace) -> Matrix {
    inverse(pcs_matrix(target))
}

fn multiply(a: Matrix, b: Matrix) -> Matrix {
    a.map(|row| std::array::from_fn(|c| (0..3).map(|k| row[k] * b[k][c]).sum()))
}

fn adapted_xyz(source: ColorSpace, target_white: [f64; 2], adapt: bool) -> Matrix {
    let mut source_xyz = xyz(source);
    if adapt && white(source) != target_white {
        let cone = [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ];
        let response = |[x, y]: [f64; 2]| {
            let xyz = [x / y, 1.0, (1.0 - x - y) / y];
            cone.map(|row| (0..3).map(|c| row[c] * xyz[c]).sum::<f64>())
        };
        let from = response(white(source));
        let to = response(target_white);
        let scale: Matrix = std::array::from_fn(|r| {
            std::array::from_fn(|c| if r == c { to[r] / from[r] } else { 0.0 })
        });
        source_xyz = multiply(multiply(multiply(inverse(cone), scale), cone), source_xyz);
    }
    source_xyz
}

pub fn convert(
    rgb: [f64; 3],
    source: TransferFunction,
    target: TransferFunction,
    matrix: Matrix,
) -> [f64; 3] {
    let linear = rgb.map(|v| to_linear(v, source));
    matrix.map(|row| from_linear((0..3).map(|c| row[c] * linear[c]).sum(), target))
}

/// Propagate the original reconstruction bound through piecewise EOTFs and a signed matrix.
/// The bound is fixed before conversion and cannot be widened by observed GPU errors.
pub fn interval(
    rgb: [f64; 3],
    source: TransferFunction,
    target: TransferFunction,
    matrix: Matrix,
    tolerance: f64,
) -> [[f64; 2]; 3] {
    let range = rgb.map(|v| {
        let error = tolerance * (1.0 + v.abs());
        let mut range = [to_linear(v - error, source), to_linear(v + error, source)];
        // BT.709's rounded constants leave a small inverse discontinuity at 0.081.
        if source == TransferFunction::Bt709 && v - error <= 0.081 && v + error > 0.081 {
            range[0] = range[0].min(((0.081_f64 + 0.099) / 1.099).powf(1.0 / 0.45));
            range[1] = range[1].max(0.018);
        }
        range
    });
    matrix.map(|row| {
        std::array::from_fn(|end| {
            from_linear(
                (0..3)
                    .map(|c| row[c] * range[c][if row[c] < 0.0 { 1 - end } else { end }])
                    .sum(),
                target,
            )
        })
    })
}
