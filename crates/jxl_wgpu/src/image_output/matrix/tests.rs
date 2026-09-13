use super::*;
use jxl_gpu_protocol::{Chromaticity, RgbChromaticities};

#[test]
fn white_adaptation_and_absolute_xyz_have_distinct_colorimetric_meanings() {
    let source = RgbColorSpace::Custom(RgbChromaticities {
        white: Chromaticity::E,
        ..RgbChromaticities::BT709
    });
    let adapted =
        rgb_color_matrix(source, RgbColorSpace::Bt709, WhitePointAdaptation::Bradford).unwrap();
    for row in adapted {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 2e-7);
    }
    let absolute =
        rgb_color_matrix(source, RgbColorSpace::Bt709, WhitePointAdaptation::None).unwrap();
    // Independently tabulated CIE XYZ -> linear sRGB, applied to equal-energy white XYZ=(1,1,1).
    for (row, expected) in absolute
        .into_iter()
        .zip([1.204_976_f32, 0.948_279, 0.908_625])
    {
        assert!((row.iter().sum::<f32>() - expected).abs() < 2e-6);
    }
    let inverse =
        rgb_color_matrix(RgbColorSpace::Bt709, source, WhitePointAdaptation::Bradford).unwrap();
    for (r, row) in inverse.iter().enumerate() {
        for (c, _) in adapted[0][..3].iter().enumerate() {
            let actual = (0..3).map(|k| row[k] * adapted[k][c]).sum::<f32>();
            assert!((actual - if r == c { 1.0 } else { 0.0 }).abs() < 2e-7);
        }
    }
}

#[test]
fn zero_y_primaries_define_a_valid_xyz_basis() {
    let color = RgbChromaticities {
        red: Chromaticity::new(1.0, 0.0).unwrap(),
        green: Chromaticity::new(0.0, 1.0).unwrap(),
        blue: Chromaticity::new(0.0, 0.0).unwrap(),
        white: Chromaticity::E,
    };
    for (row, expected) in ColorMatrix::rgb_to_xyz(
        RgbColorSpace::Custom(color),
        Chromaticity::E,
        WhitePointAdaptation::None,
    )
    .unwrap()
    .rows()
    .iter()
    .copied()
    .zip(IDENTITY)
    {
        for (actual, expected) in row.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }
    let converted = rgb_color_matrix(
        RgbColorSpace::Custom(color),
        RgbColorSpace::Bt709,
        WhitePointAdaptation::None,
    )
    .unwrap();
    // Independently tabulated CIE XYZ -> D65 linear sRGB.
    let expected = [
        [3.240_969_9_f64, -1.537_383_2, -0.498_610_76],
        [-0.969_243_65, 1.875_967_5, 0.041_555_06],
        [0.055_630_08, -0.203_976_96, 1.056_971_5],
    ];
    for (row, expected) in converted.into_iter().zip(expected) {
        for (actual, expected) in row[..3].iter().zip(expected) {
            assert!((f64::from(*actual) - expected).abs() < 3e-7);
        }
    }
}

#[test]
fn invalid_chromaticities_are_rejected_even_for_identity_conversion() {
    let mut singular = RgbChromaticities::BT709;
    singular.green = singular.red;
    let mut zero_y = RgbChromaticities::BT709;
    zero_y.white = Chromaticity::new(0.3, 0.0).unwrap();
    let mut overflowing = RgbChromaticities::BT709;
    overflowing.red = Chromaticity::new(f64::MAX, f64::MAX).unwrap();
    for color in [singular, zero_y, overflowing] {
        let space = RgbColorSpace::Custom(color);
        assert!(
            rgb_color_matrix(space, space, WhitePointAdaptation::Bradford).is_err(),
            "{color:?}"
        );
    }
    assert!(
        rgb_color_matrix(
            RgbColorSpace::Undefined,
            RgbColorSpace::Bt709,
            WhitePointAdaptation::None
        )
        .is_err()
    );
}
