//! Resolve the original image encoding once, without assigning a color meaning to unknown data.

use jxl_gpu_bitstream::{
    ColourEncodingInventory, ColourSpaceInventory, ImageHeaderInventory, PrimariesInventory,
    TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_protocol::{RgbColorEncoding, RgbPrimaries, TransferFunction};

pub(crate) fn original_encoding(image: &ImageHeaderInventory) -> Option<RgbColorEncoding> {
    if image.embedded_icc.is_some() {
        return None;
    }
    let ColourEncodingInventory::Enumerated {
        colour_space,
        white_point: WhitePointInventory::D65,
        primaries,
        transfer_function,
        ..
    } = image.colour_encoding
    else {
        return None;
    };
    let primaries = match (colour_space, image.grayscale, primaries) {
        // Grayscale has no primary declaration; replicated D65 luminance is neutral RGB.
        (ColourSpaceInventory::Grey, true, _) => RgbPrimaries::Bt709,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::Srgb) => RgbPrimaries::Bt709,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::Bt2100) => RgbPrimaries::Bt2020,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::P3) => RgbPrimaries::DisplayP3,
        _ => return None,
    };
    let transfer = match transfer_function {
        TransferFunctionInventory::Linear => TransferFunction::Linear,
        TransferFunctionInventory::Srgb => TransferFunction::Srgb,
        TransferFunctionInventory::Bt709 => TransferFunction::Bt709,
        _ => return None,
    };
    Some(RgbColorEncoding {
        primaries,
        transfer,
    })
}

pub(crate) fn require_original_encoding(
    image: &ImageHeaderInventory,
) -> Result<RgbColorEncoding, crate::UnsupportedProfile> {
    original_encoding(image).ok_or_else(|| {
        crate::UnsupportedProfile::new(
            crate::UnsupportedCodestreamFeature::ColorEncoding,
            "image color requires enumerated D65 BT.709/BT.2020/Display-P3 primaries and Linear/sRGB/BT.709 transfer",
        )
    })
}

pub(crate) const fn linear_encoding(original: RgbColorEncoding) -> RgbColorEncoding {
    RgbColorEncoding {
        primaries: original.primaries,
        transfer: TransferFunction::Linear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_bitstream::{ChromaticityInventory, EmbeddedIccInventory};

    fn header() -> ImageHeaderInventory {
        let data = jxl_test_support::fixtures::original_color::cases()[0].bytes();
        jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header
    }

    #[test]
    fn original_profiles_keep_primaries_and_gray_neutrality_explicit() {
        let mut image = header();
        for (declared, expected) in [
            (PrimariesInventory::Srgb, RgbPrimaries::Bt709),
            (PrimariesInventory::Bt2100, RgbPrimaries::Bt2020),
            (PrimariesInventory::P3, RgbPrimaries::DisplayP3),
        ] {
            for (declared_tf, expected_tf) in [
                (TransferFunctionInventory::Linear, TransferFunction::Linear),
                (TransferFunctionInventory::Srgb, TransferFunction::Srgb),
                (TransferFunctionInventory::Bt709, TransferFunction::Bt709),
            ] {
                for gray in [false, true] {
                    image.grayscale = gray;
                    let ColourEncodingInventory::Enumerated {
                        colour_space,
                        primaries,
                        transfer_function,
                        ..
                    } = &mut image.colour_encoding
                    else {
                        unreachable!()
                    };
                    *colour_space = if gray {
                        ColourSpaceInventory::Grey
                    } else {
                        ColourSpaceInventory::Rgb
                    };
                    *primaries = declared;
                    *transfer_function = declared_tf;
                    let expected = RgbColorEncoding {
                        primaries: if gray { RgbPrimaries::Bt709 } else { expected },
                        transfer: expected_tf,
                    };
                    assert_eq!(require_original_encoding(&image).unwrap(), expected);
                    assert_eq!(
                        linear_encoding(expected),
                        RgbColorEncoding {
                            primaries: expected.primaries,
                            transfer: TransferFunction::Linear
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn incomplete_or_inconsistent_profiles_are_rejected_without_relabeling() {
        let original = header();
        let custom = ChromaticityInventory {
            x: 312700,
            y: 329000,
        };
        let mut unsupported = Vec::new();
        for white in [
            WhitePointInventory::E,
            WhitePointInventory::Dci,
            WhitePointInventory::Custom(custom),
        ] {
            let mut image = original.clone();
            let ColourEncodingInventory::Enumerated { white_point, .. } =
                &mut image.colour_encoding
            else {
                unreachable!()
            };
            *white_point = white;
            unsupported.push(image);
        }
        for tf in [
            TransferFunctionInventory::Pq,
            TransferFunctionInventory::Hlg,
            TransferFunctionInventory::Dci,
            TransferFunctionInventory::Unknown,
            TransferFunctionInventory::Gamma {
                scaled_gamma: 10_000_000,
                inverted: false,
            },
        ] {
            let mut image = original.clone();
            let ColourEncodingInventory::Enumerated {
                transfer_function, ..
            } = &mut image.colour_encoding
            else {
                unreachable!()
            };
            *transfer_function = tf;
            unsupported.push(image);
        }
        let mut image = original.clone();
        let ColourEncodingInventory::Enumerated { primaries, .. } = &mut image.colour_encoding
        else {
            unreachable!()
        };
        *primaries = PrimariesInventory::Custom {
            red: custom,
            green: custom,
            blue: custom,
        };
        unsupported.push(image);
        let mut image = original.clone();
        image.grayscale = true;
        unsupported.push(image);
        let mut image = original.clone();
        image.colour_encoding = ColourEncodingInventory::IccProfile {
            colour_space: ColourSpaceInventory::Rgb,
        };
        unsupported.push(image);
        let mut image = original;
        image.embedded_icc = Some(EmbeddedIccInventory {
            bit_range: image.bit_range,
            encoded_byte_count: 0,
            profile: vec![],
        });
        unsupported.push(image);
        for image in unsupported {
            assert_eq!(original_encoding(&image), None);
            assert_eq!(
                require_original_encoding(&image).unwrap_err().feature,
                crate::UnsupportedCodestreamFeature::ColorEncoding
            );
        }
    }
}
