//! Test-only references: jxl-oxide and f64 colorimetry derived from primary chromaticities.
use super::{corpus, tolerance};
use jxl_gpu_formats::{ColorSpace, TransferFunction};

#[test]
fn native_still_references_agree_with_independent_decoder() {
    check_still_references(corpus::cases());
}

#[test]
fn analytic_native_still_references_agree_with_independent_decoder() {
    check_still_references(corpus::analytic_cases());
}

fn check_still_references(cases: Vec<corpus::Case>) {
    use jxl_oxide::color::{ColourSpace, Customxy, Primaries, TransferFunction as Tf, WhitePoint};
    let xy = |p: jxl_gpu_bitstream::ChromaticityInventory| Customxy { x: p.x, y: p.y };
    // jxl-frame 0.13.3 reads an extra-channel source selector using the color blend mode.
    // These sequences use full-canvas color Mul/Blend with alpha Replace, where that selector
    // is absent. libjxl and our parser use the extra's own mode; jxl-oxide loses bit alignment.
    // All 148 originals still have native references and GPU coverage; this second oracle
    // therefore covers the 74 stills. See the generator README for the precise upstream sites.
    for case in cases.into_iter().filter(|case| !case.sequence) {
        let mut image = jxl_oxide::JxlImage::read_with_defaults(case.bytes().as_slice()).unwrap();
        image.set_render_spot_color(false);
        image.request_color_encoding(jxl_oxide::EnumColourEncoding {
            colour_space: if case.profile.grayscale {
                ColourSpace::Grey
            } else {
                ColourSpace::Rgb
            },
            white_point: match case.profile.white {
                jxl_gpu_bitstream::WhitePointInventory::D65 => WhitePoint::D65,
                jxl_gpu_bitstream::WhitePointInventory::E => WhitePoint::E,
                jxl_gpu_bitstream::WhitePointInventory::Dci => WhitePoint::Dci,
                jxl_gpu_bitstream::WhitePointInventory::Custom(p) => WhitePoint::Custom(xy(p)),
            },
            primaries: match case.profile.primaries {
                jxl_gpu_bitstream::PrimariesInventory::Srgb => Primaries::Srgb,
                jxl_gpu_bitstream::PrimariesInventory::Bt2100 => Primaries::Bt2100,
                jxl_gpu_bitstream::PrimariesInventory::P3 => Primaries::P3,
                jxl_gpu_bitstream::PrimariesInventory::Custom { red, green, blue } => {
                    Primaries::Custom {
                        red: xy(red),
                        green: xy(green),
                        blue: xy(blue),
                    }
                }
            },
            tf: match case.transfer.transfer {
                jxl_gpu_bitstream::TransferFunctionInventory::Linear => Tf::Linear,
                jxl_gpu_bitstream::TransferFunctionInventory::Srgb => Tf::Srgb,
                jxl_gpu_bitstream::TransferFunctionInventory::Bt709 => Tf::Bt709,
                jxl_gpu_bitstream::TransferFunctionInventory::Dci => Tf::Dci,
                jxl_gpu_bitstream::TransferFunctionInventory::Gamma {
                    scaled_gamma,
                    inverted,
                } => Tf::Gamma {
                    g: scaled_gamma,
                    inverted,
                },
                _ => unreachable!(),
            },
            rendering_intent: jxl_oxide::RenderingIntent::Relative,
        });
        // jxl-color clips or gamut maps before a primary/Gray conversion. Ask it for the
        // unbounded XYB intermediate, then perform the reference's requested conversion in f64.
        if case.mode.xyb() {
            image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
                jxl_oxide::RenderingIntent::Relative,
            ));
        }
        let expected = case.reference();
        assert_eq!(
            image.num_loaded_keyframes(),
            if case.sequence { 4 } else { 1 }
        );
        for frame in 0..image.num_loaded_keyframes() {
            let render = image.render_frame(frame).unwrap();
            let pixels = render.image_all_channels();
            assert!(pixels.channels() == 4 || (case.profile.grayscale && pixels.channels() == 2));
            let mut words: Vec<_> = if pixels.channels() == 2 {
                pixels
                    .buf()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|p| [p[0], p[0], p[0], p[1]])
                    .map(f32::to_bits)
                    .collect()
            } else {
                pixels.buf().iter().map(|v| v.to_bits()).collect()
            };
            if case.mode.xyb() {
                let jxl_gpu_formats::ColorSpecification::Defined(target) = case.format().color_spec
                else {
                    unreachable!()
                };
                for pixel in words.as_chunks_mut::<4>().0 {
                    let rgb = [pixel[0], pixel[1], pixel[2]].map(|v| f64::from(f32::from_bits(v)));
                    let linear = xyb_original_linear(rgb, &case);
                    for (channel, value) in linear.into_iter().enumerate() {
                        // The native JPEG XL original OETF has a black floor, unlike a general CMS conversion.
                        let value = if matches!(
                            target.transfer,
                            TransferFunction::Gamma(_) | TransferFunction::Dci
                        ) && value <= 1e-5
                        {
                            0.0
                        } else {
                            value
                        };
                        pixel[channel] = (from_linear(value, target.transfer) as f32).to_bits();
                    }
                }
            }
            super::compare(
                &words,
                &expected[frame * 37 * 19 * 4..(frame + 1) * 37 * 19 * 4],
                tolerance(&case),
                &format!("oxide {}", case.name),
            );
        }
    }
}

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

fn xyb_original_linear(mut rgb: [f64; 3], case: &corpus::Case) -> [f64; 3] {
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
        TransferFunction::Srgb => {
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
        _ => unreachable!(),
    }
}

pub fn from_linear(value: f64, transfer: TransferFunction) -> f64 {
    match transfer {
        TransferFunction::Linear => value,
        TransferFunction::Srgb => {
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
        _ => unreachable!(),
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
    let multiply = |a: Matrix, b: Matrix| {
        a.map(|row| std::array::from_fn(|c| (0..3).map(|k| row[k] * b[k][c]).sum()))
    };
    let mut source_xyz = xyz(source);
    if adapt && white(source) != white(target) {
        let cone = [
            [0.8951, 0.2664, -0.1614],
            [-0.7502, 1.7135, 0.0367],
            [0.0389, -0.0685, 1.0296],
        ];
        let response = |space| {
            let [x, y] = white(space);
            let xyz = [x / y, 1.0, (1.0 - x - y) / y];
            cone.map(|row| (0..3).map(|c| row[c] * xyz[c]).sum::<f64>())
        };
        let from = response(source);
        let to = response(target);
        let scale: Matrix = std::array::from_fn(|r| {
            std::array::from_fn(|c| if r == c { to[r] / from[r] } else { 0.0 })
        });
        source_xyz = multiply(multiply(multiply(inverse(cone), scale), cone), source_xyz);
    }
    multiply(inverse(xyz(target)), source_xyz)
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
