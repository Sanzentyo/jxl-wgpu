use super::*;

#[test]
fn pcs_connections_preserve_the_declared_white_and_cancel_in_f64() {
    for space in [
        RgbColorSpace::Bt709,
        RgbColorSpace::Bt2020,
        RgbColorSpace::DisplayP3,
        RgbColorSpace::Custom(RgbChromaticities {
            white: Chromaticity::E,
            ..RgbChromaticities::BT709
        }),
    ] {
        let forward =
            ColorMatrix::rgb_to_xyz(space, Chromaticity::ICC_D50, WhitePointAdaptation::Bradford)
                .unwrap();
        let reverse =
            ColorMatrix::xyz_to_rgb(Chromaticity::ICC_D50, space, WhitePointAdaptation::Bradford)
                .unwrap();
        for (row, expected) in forward.rows().iter().zip([0xf6d6, 0x10000, 0xd32d]) {
            assert!((row.iter().sum::<f64>() - f64::from(expected) / 65536.0).abs() < 1e-14);
        }
        for (row, expected) in multiply(*reverse.rows(), *forward.rows())
            .into_iter()
            .zip(IDENTITY)
        {
            for (actual, expected) in row.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn connections_validate_geometry_even_without_adaptation() {
    let invalid_white = Chromaticity::new(0.3, 0.0).unwrap();
    let singular = RgbColorSpace::Custom(RgbChromaticities {
        green: RgbChromaticities::BT709.red,
        ..RgbChromaticities::BT709
    });
    for policy in [WhitePointAdaptation::None, WhitePointAdaptation::Bradford] {
        assert!(ColorMatrix::rgb_to_xyz(singular, Chromaticity::ICC_D50, policy).is_err());
        assert!(ColorMatrix::xyz_to_rgb(Chromaticity::ICC_D50, singular, policy).is_err());
        assert_eq!(
            ColorMatrix::rgb_to_xyz(RgbColorSpace::Bt709, invalid_white, policy),
            Err(ColorMatrixError::WhitePoint)
        );
        assert_eq!(
            ColorMatrix::xyz_to_rgb(invalid_white, RgbColorSpace::Bt709, policy),
            Err(ColorMatrixError::WhitePoint)
        );
        assert_eq!(
            ColorMatrix::rgb_to_xyz(RgbColorSpace::Undefined, Chromaticity::ICC_D50, policy),
            Err(ColorMatrixError::UndefinedSource)
        );
        assert_eq!(
            ColorMatrix::xyz_to_rgb(Chromaticity::ICC_D50, RgbColorSpace::Undefined, policy),
            Err(ColorMatrixError::UndefinedTarget)
        );
    }
}
