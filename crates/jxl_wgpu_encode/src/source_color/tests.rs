use super::*;
use jxl_bitstream::{Bitstream, U};
use jxl_gpu_formats::{ChromaLocation2d, ColorSpace, ColorSpec, RgbChannelOrder};

fn format(space: ColorSpace, transfer: TransferFunction) -> PixelFormat {
    PixelFormat::rgb8(
        RgbChannelOrder::Rgb,
        false,
        ColorSpecification::Defined(ColorSpec {
            space,
            transfer,
            encoding: YcbcrEncoding::Undefined,
            range: ColorRange::Full,
            chroma_location: ChromaLocation2d::CENTER,
        }),
    )
}

#[test]
fn custom_coordinates_keep_signed_bucket_boundaries() {
    for value in [i32::MIN, -2_097_153, 2_097_152, i32::MAX] {
        assert!(quantize_xy(Chromaticity::new(f64::from(value) / 1e6, 0.3).unwrap()).is_err());
    }
    for x in [
        -2_097_152, -1_048_576, -524_288, -262_144, -1, 0, 1, 262_143, 262_144, 524_287, 524_288,
        1_048_575, 1_048_576, 2_097_151,
    ] {
        let value = ChromaticityInventory { x, y: -x - 1 };
        assert_eq!(quantize_xy(expand_xy(value)).unwrap(), value);
        let mut output = BitWriter::new();
        write_xy(&mut output, value).unwrap();
        let bytes = output.into_bytes();
        let mut reader = Bitstream::new(&bytes);
        for expected in [value.x, value.y] {
            let packed = reader
                .read_u32(U(19), 524_288 + U(19), 1_048_576 + U(20), 2_097_152 + U(21))
                .unwrap();
            let actual = (packed >> 1) as i32 ^ -((packed & 1) as i32);
            assert_eq!(actual, expected);
        }
    }
    let rounded = quantize_xy(Chromaticity::new(0.312_345_7, -0.012_345_7).unwrap()).unwrap();
    assert_eq!(
        rounded,
        ChromaticityInventory {
            x: 312_346,
            y: -12_346
        }
    );
}

#[test]
fn unrepresentable_or_singular_source_metadata_is_rejected() {
    for transfer in [
        TransferFunction::Undefined,
        TransferFunction::Smpte240M,
        TransferFunction::Bt2020,
    ] {
        assert!(SourceColorEncoding::from_format(&format(ColorSpace::Bt709, transfer)).is_err());
    }
    for gamma in [f32::MIN_POSITIVE, 0.0001, 1.00001, 2.2] {
        let transfer =
            TransferFunction::Gamma(jxl_gpu_protocol::GammaExponent::new(gamma).unwrap());
        assert!(SourceColorEncoding::from_format(&format(ColorSpace::Bt709, transfer)).is_err());
    }
    for gamma in [1.0 / 8192.0, 1.0 / 2.2, 1.0] {
        let transfer =
            TransferFunction::Gamma(jxl_gpu_protocol::GammaExponent::new(gamma).unwrap());
        let encoded =
            SourceColorEncoding::from_format(&format(ColorSpace::Bt709, transfer)).unwrap();
        let SourceColorEncoding::Enumerated(encoded) = encoded else {
            panic!("enumerated source color");
        };
        let TransferFunctionInventory::Gamma {
            scaled_gamma,
            inverted: true,
        } = encoded.transfer
        else {
            panic!("quantized source OETF");
        };
        assert!((1221..=10_000_000).contains(&scaled_gamma));
        assert!((f64::from(scaled_gamma) / 1e7 - f64::from(gamma)).abs() <= 0.5e-7);
    }
    for coordinates in [
        RgbChromaticities {
            green: RgbChromaticities::BT709.red,
            ..RgbChromaticities::BT709
        },
        RgbChromaticities {
            white: Chromaticity::new(0.3, 0.0).unwrap(),
            ..RgbChromaticities::BT709
        },
        RgbChromaticities {
            red: Chromaticity::new(2.1, 0.3).unwrap(),
            ..RgbChromaticities::BT709
        },
        // Nonsingular before serialization, collapsed by the mandatory xy quantization.
        RgbChromaticities {
            green: Chromaticity::new(0.640_000_1, 0.330_000_1).unwrap(),
            ..RgbChromaticities::BT709
        },
    ] {
        assert!(
            SourceColorEncoding::from_format(&format(
                ColorSpace::CustomRgb(coordinates),
                TransferFunction::Srgb
            ))
            .is_err()
        );
    }
}

#[test]
fn source_aliases_are_explicit() {
    let default = SourceColorEncoding::default();
    for transfer in [TransferFunction::Srgb, TransferFunction::Sycc] {
        assert_eq!(
            SourceColorEncoding::from_format(&format(ColorSpace::Bt709, transfer)).unwrap(),
            default
        );
    }
}
