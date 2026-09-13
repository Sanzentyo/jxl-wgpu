use super::*;
use jxl_gpu_protocol::{Chromaticity, GammaExponent, RgbChromaticities, WhitePointAdaptation};

#[test]
fn gamma_and_dci_match_native_extended_curve_contracts_on_gpu() {
    let backend = backend().expect("color conformance requires an adapter");
    let gamma = GammaExponent::new(0.4545455).unwrap();
    for (source_tf, target_tf, inverse, dci) in [
        (
            SourceTransferFunction::Gamma(gamma),
            TransferFunction::Linear,
            true,
            false,
        ),
        (
            SourceTransferFunction::Linear,
            TransferFunction::Gamma(gamma),
            false,
            false,
        ),
        (
            SourceTransferFunction::Dci,
            TransferFunction::Linear,
            true,
            true,
        ),
        (
            SourceTransferFunction::Linear,
            TransferFunction::Dci,
            false,
            true,
        ),
    ] {
        let source = RgbColorEncoding {
            space: RgbColorSpace::Bt709,
            transfer: source_tf,
        };
        let format = PixelFormat::rgb_f32(
            RgbChannelOrder::Rgb,
            false,
            rgb_color(ColorSpace::Bt709, target_tf),
        );
        for input in [-0.25f32, -1e-8, 0.0, 0.125, 1.0, 1.25] {
            let (_, bytes) =
                submit_format(&backend, source, source, format.clone(), [input; 3]).unwrap();
            // Native libjxl 0.12.0 + LCMS: Gamma para-0 clamps negatives; DCI para-3
            // has a unit-slope negative branch. Positive samples follow the declared exponent.
            let exponent = if dci {
                1.0 / 2.6
            } else {
                f64::from(gamma.value())
            };
            let expected = if dci && input <= 0.0 {
                f64::from(input)
            } else {
                f64::from(input)
                    .max(0.0)
                    .powf(if inverse { 1.0 / exponent } else { exponent })
            };
            for word in bytes.as_chunks::<4>().0 {
                let actual = f64::from(f32::from_le_bytes(*word));
                assert!(
                    actual.is_finite() && (actual - expected).abs() < 5e-6 * (1.0 + expected.abs()),
                    "{source_tf:?} -> {target_tf:?} {input}: {actual} != {expected}"
                );
            }
        }
    }
    // Equal selectors with unequal parameters must not take the identity shortcut.
    let source = RgbColorEncoding {
        space: RgbColorSpace::Bt709,
        transfer: SourceTransferFunction::Gamma(GammaExponent::new(0.5).unwrap()),
    };
    let format = PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        false,
        rgb_color(ColorSpace::Bt709, TransferFunction::Gamma(gamma)),
    );
    let (_, bytes) = submit_format(&backend, source, source, format, [0.25; 3]).unwrap();
    let expected = 0.0625f32.powf(gamma.value());
    assert!((f32::from_le_bytes(bytes[..4].try_into().unwrap()) - expected).abs() < 2e-6);
}

#[test]
fn custom_white_points_follow_the_requested_adaptation_on_gpu() {
    let backend = backend().expect("color conformance requires an adapter");
    let extent = Extent2d::new(1, 1);
    let source = RgbColorEncoding {
        space: RgbColorSpace::Custom(RgbChromaticities {
            white: Chromaticity::E,
            ..RgbChromaticities::BT709
        }),
        transfer: SourceTransferFunction::Linear,
    };
    let format = PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        false,
        rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
    );
    for (policy, expected) in [
        (WhitePointAdaptation::Bradford, [1.0; 3]),
        (
            WhitePointAdaptation::None,
            [1.204_976, 0.948_279, 0.908_625],
        ),
    ] {
        let mut session = backend
            .create_session(&frame_desc(extent), plan(extent, source))
            .unwrap();
        enqueue(&mut session, extent, &[vec![1.0], vec![1.0], vec![1.0]]);
        let token = session
            .submit_image(
                RenderIntent::Final,
                ImageOutputRequest::new(source, format.clone()).with_white_point_adaptation(policy),
            )
            .unwrap();
        let output = session.wait_image(token).unwrap().outputs.remove(0);
        for (word, expected) in output.bytes.as_chunks::<4>().0.iter().zip(expected) {
            assert!((f32::from_le_bytes(*word) - expected).abs() < 2e-6);
        }
    }
}

#[test]
fn zero_y_primaries_convert_absolute_xyz_on_gpu() {
    let backend = backend().expect("color conformance requires an adapter");
    let extent = Extent2d::new(1, 1);
    let source = RgbColorEncoding {
        space: RgbColorSpace::Custom(RgbChromaticities {
            red: Chromaticity::new(1.0, 0.0).unwrap(),
            green: Chromaticity::new(0.0, 1.0).unwrap(),
            blue: Chromaticity::new(0.0, 0.0).unwrap(),
            white: Chromaticity::E,
        }),
        transfer: SourceTransferFunction::Linear,
    };
    let format = PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        false,
        rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
    );
    // Independent XYZ -> linear sRGB coefficients, with absolute white preservation.
    let matrix = [
        [3.240_969_9_f64, -1.537_383_2, -0.498_610_76],
        [-0.969_243_65, 1.875_967_5, 0.041_555_06],
        [0.055_630_08, -0.203_976_96, 1.056_971_5],
    ];
    for input in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.125, 0.5, 1.25],
    ] {
        let mut session = backend
            .create_session(&frame_desc(extent), plan(extent, source))
            .unwrap();
        enqueue(&mut session, extent, &input.map(|value| vec![value]));
        let token = session
            .submit_image(
                RenderIntent::Final,
                ImageOutputRequest::new(source, format.clone())
                    .with_white_point_adaptation(WhitePointAdaptation::None),
            )
            .unwrap();
        let output = session.wait_image(token).unwrap().outputs.remove(0);
        for (word, row) in output.bytes.as_chunks::<4>().0.iter().zip(matrix) {
            let expected: f64 = row
                .into_iter()
                .zip(input)
                .map(|(a, b)| a * f64::from(b))
                .sum();
            let actual = f64::from(f32::from_le_bytes(*word));
            assert!((actual - expected).abs() < 2e-6 * (1.0 + expected.abs()));
        }
    }
}
