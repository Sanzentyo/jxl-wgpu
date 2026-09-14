//! Resolve the original image encoding once, without assigning a color meaning to unknown data.

use jxl_gpu_bitstream::{
    ChromaticityInventory, ColourEncodingInventory, ColourSpaceInventory, ImageHeaderInventory,
    PrimariesInventory, RenderingIntentInventory, TransferFunctionInventory, WhitePointInventory,
};
use jxl_gpu_protocol::{
    Chromaticity, GammaExponent, RgbChromaticities, RgbColorEncoding, RgbColorSpace,
    TransferFunction, WhitePointAdaptation,
};

/// Check the declaration's channel meaning independently of the requested color conversion.
/// Numeric original samples do not require a supported transfer function, matrix, or ICC LUT.
pub(crate) fn validate_declaration(
    image: &ImageHeaderInventory,
) -> Result<(), crate::UnsupportedProfile> {
    let (colour_space, has_icc) = match image.colour_encoding {
        ColourEncodingInventory::Enumerated { colour_space, .. } => (colour_space, false),
        ColourEncodingInventory::IccProfile { colour_space } => (colour_space, true),
    };
    if has_icc != image.embedded_icc.is_some()
        || image.grayscale != (colour_space == ColourSpaceInventory::Grey)
    {
        return Err(crate::UnsupportedProfile::new(
            crate::UnsupportedCodestreamFeature::ColorEncoding,
            "image color declaration disagrees with its channels or embedded ICC metadata",
        ));
    }
    Ok(())
}

pub(crate) fn original_encoding(image: &ImageHeaderInventory) -> Option<RgbColorEncoding> {
    if image.embedded_icc.is_some() {
        return None;
    }
    let ColourEncodingInventory::Enumerated {
        colour_space,
        white_point,
        primaries,
        transfer_function,
        rendering_intent,
    } = image.colour_encoding
    else {
        return None;
    };
    let white = match white_point {
        WhitePointInventory::D65 => Chromaticity::D65,
        WhitePointInventory::E => Chromaticity::E,
        WhitePointInventory::Dci => Chromaticity::DCI,
        WhitePointInventory::Custom(value) => chromaticity(value),
    };
    // Non-D65 intent policies need their own reference-white and gamut conformance.
    if white != Chromaticity::D65 && rendering_intent != RenderingIntentInventory::Relative {
        return None;
    }
    let mut coordinates = match (colour_space, image.grayscale, primaries) {
        // Replicated luminance represents the declared white, not an assumed D65 gray.
        (ColourSpaceInventory::Grey, true, _) => RgbChromaticities::BT709,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::Srgb) => RgbChromaticities::BT709,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::Bt2100) => RgbChromaticities::BT2020,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::P3) => RgbChromaticities::DISPLAY_P3,
        (ColourSpaceInventory::Rgb, false, PrimariesInventory::Custom { red, green, blue }) => {
            RgbChromaticities {
                red: chromaticity(red),
                green: chromaticity(green),
                blue: chromaticity(blue),
                white,
            }
        }
        _ => return None,
    };
    coordinates.white = white;
    let space = match coordinates {
        RgbChromaticities::BT709 => RgbColorSpace::Bt709,
        RgbChromaticities::BT2020 => RgbColorSpace::Bt2020,
        RgbChromaticities::DISPLAY_P3 => RgbColorSpace::DisplayP3,
        value => RgbColorSpace::Custom(value),
    };
    jxl_wgpu::rgb_color_matrix(space, space, WhitePointAdaptation::Bradford).ok()?;
    let transfer = match transfer_function {
        TransferFunctionInventory::Linear => TransferFunction::Linear,
        TransferFunctionInventory::Srgb => TransferFunction::Srgb,
        TransferFunctionInventory::Bt709 => TransferFunction::Bt709,
        TransferFunctionInventory::Pq => TransferFunction::Pq,
        TransferFunctionInventory::Hlg => TransferFunction::Hlg,
        TransferFunctionInventory::Dci => TransferFunction::Dci,
        TransferFunctionInventory::Gamma {
            scaled_gamma,
            inverted,
        } => {
            let value = f64::from(scaled_gamma) / 10_000_000.0;
            let exponent = if inverted { value } else { 1.0 / value };
            // JPEG XL encodes an OETF exponent in [1/8192, 1].
            if !(1.0 / 8192.0..=1.0).contains(&exponent) {
                return None;
            }
            TransferFunction::Gamma(GammaExponent::new(exponent as f32)?)
        }
        _ => return None,
    };
    Some(RgbColorEncoding { space, transfer })
}

fn chromaticity(value: ChromaticityInventory) -> Chromaticity {
    Chromaticity::new(
        f64::from(value.x) / 1_000_000.0,
        f64::from(value.y) / 1_000_000.0,
    )
    .expect("finite scaled i32 chromaticities")
}

pub(crate) fn require_original_encoding(
    image: &ImageHeaderInventory,
) -> Result<RgbColorEncoding, crate::UnsupportedProfile> {
    original_encoding(image).ok_or_else(|| {
        crate::UnsupportedProfile::new(
            crate::UnsupportedCodestreamFeature::ColorEncoding,
            "image color requires a nonsingular enumerated RGB/gray profile; non-D65 whites currently require relative intent",
        )
    })
}

/// Resolve owned ICC metadata once for the selected image. Raw numeric requests can avoid this
/// entirely; retaining device values only requires a structurally valid profile, not a CMS method.
pub(crate) fn original_domain(
    image: &ImageHeaderInventory,
) -> crate::Result<crate::frame_surface::FrameSurfaceEncoding> {
    validate_declaration(image)?;
    if let Some(icc) = &image.embedded_icc {
        use jxl_gpu_protocol::icc::{IccLimits, IccProfile, IccSignature};
        let profile = IccProfile::parse(icc.profile.clone(), IccLimits::default())?;
        if !image.grayscale && profile.header().device_space == IccSignature(*b"CMYK") {
            let mut black = image
                .extra_channels
                .iter()
                .enumerate()
                .filter(|(_, extra)| {
                    matches!(
                        extra.channel_type,
                        jxl_gpu_bitstream::ExtraChannelTypeInventory::Black
                    )
                });
            let Some((black_extra, _)) = black.next() else {
                return Err(crate::UnsupportedProfile::new(
                    crate::UnsupportedCodestreamFeature::ColorEncoding,
                    "CMYK ICC requires a Black extra channel",
                )
                .into());
            };
            if black.next().is_some() {
                return Err(crate::UnsupportedProfile::new(
                    crate::UnsupportedCodestreamFeature::ColorEncoding,
                    "CMYK ICC has ambiguous Black extra channels",
                )
                .into());
            }
            return Ok(crate::frame_surface::FrameSurfaceEncoding::Cmyk {
                profile,
                black_extra,
            });
        }
        let expected = IccSignature(if image.grayscale { *b"GRAY" } else { *b"RGB " });
        if profile.header().device_space != expected {
            return Err(crate::UnsupportedProfile::new(
                crate::UnsupportedCodestreamFeature::ColorEncoding,
                "embedded ICC device channels disagree with the JPEG XL image declaration",
            )
            .into());
        }
        Ok(crate::frame_surface::FrameSurfaceEncoding::Icc(profile))
    } else {
        Ok(crate::frame_surface::FrameSurfaceEncoding::Rgb(
            require_original_encoding(image)?,
        ))
    }
}

pub(crate) const fn linear_encoding(original: RgbColorEncoding) -> RgbColorEncoding {
    RgbColorEncoding {
        space: original.space,
        transfer: TransferFunction::Linear,
    }
}

/// Native JPEG XL's original gamma/DCI reconstruction uses a 1e-5 linear black floor
/// (libjxl v0.12.0 stage_from_linear::OpGamma), before blending or reference storage.
/// General RGB color conversion continues to use the profile curve itself.
pub(crate) fn reconstruction_black_threshold(
    original: RgbColorEncoding,
    target: &jxl_gpu_formats::ColorSpecification,
) -> Option<f32> {
    let jxl_gpu_formats::ColorSpecification::Defined(target) = target else {
        return None;
    };
    (matches!(
        original.transfer,
        TransferFunction::Gamma(_) | TransferFunction::Dci
    ) && target.space.rgb_space() == Some(original.space)
        && target.transfer.rgb_transfer() == Some(original.transfer))
    .then_some(1e-5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_bitstream::{ChromaticityInventory, EmbeddedIccInventory};

    #[test]
    fn cmyk_domain_owns_its_profile_and_requires_an_unambiguous_black_plane() {
        use crate::frame_surface::FrameSurfaceEncoding;
        let bytes = include_bytes!("../test-data/cmyk/generated/lut8_xyz_4_0.jxl");
        let mut image = jxl_gpu_bitstream::parse(bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        for black in [0, 2] {
            if black == 2 {
                image.extra_channels.swap(0, 2);
            }
            let domain = original_domain(&image).unwrap();
            assert!(
                matches!(&domain, FrameSurfaceEncoding::Cmyk { profile, black_extra }
                if *black_extra == black && profile.bytes().as_ref() == image.embedded_icc.as_ref().unwrap().profile.as_ref())
            );
            assert!(FrameSurfaceEncoding::from_format(&domain.format()).is_none());
            assert_eq!(domain.rgb_encoding(), None);
        }
        let mut missing = image.clone();
        missing.extra_channels.clear();
        assert!(original_domain(&missing).is_err());
        image.extra_channels.push(image.extra_channels[2].clone());
        assert!(original_domain(&image).is_err());
    }

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
            (PrimariesInventory::Srgb, RgbColorSpace::Bt709),
            (PrimariesInventory::Bt2100, RgbColorSpace::Bt2020),
            (PrimariesInventory::P3, RgbColorSpace::DisplayP3),
        ] {
            for (declared_tf, expected_tf) in [
                (TransferFunctionInventory::Linear, TransferFunction::Linear),
                (TransferFunctionInventory::Srgb, TransferFunction::Srgb),
                (TransferFunctionInventory::Bt709, TransferFunction::Bt709),
                (TransferFunctionInventory::Pq, TransferFunction::Pq),
                (TransferFunctionInventory::Hlg, TransferFunction::Hlg),
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
                        space: if gray { RgbColorSpace::Bt709 } else { expected },
                        transfer: expected_tf,
                    };
                    assert_eq!(require_original_encoding(&image).unwrap(), expected);
                    assert_eq!(
                        linear_encoding(expected),
                        RgbColorEncoding {
                            space: expected.space,
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
        let mut image = original.clone();
        let ColourEncodingInventory::Enumerated { white_point, .. } = &mut image.colour_encoding
        else {
            unreachable!()
        };
        *white_point = WhitePointInventory::Custom(ChromaticityInventory { x: 0, y: 0 });
        unsupported.push(image);
        for tf in [
            TransferFunctionInventory::Unknown,
            TransferFunctionInventory::Gamma {
                scaled_gamma: 0,
                inverted: true,
            },
            TransferFunctionInventory::Gamma {
                scaled_gamma: 1_220,
                inverted: true,
            },
            TransferFunctionInventory::Gamma {
                scaled_gamma: 10_000_001,
                inverted: true,
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
            profile: [].into(),
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

    #[test]
    fn analytic_declarations_resolve_without_losing_original_color_parameters() {
        for case in jxl_test_support::fixtures::original_color::analytic_cases()
            .into_iter()
            .filter(|case| !case.mode.ycbcr())
        {
            let bytes = case.bytes();
            let image = jxl_gpu_bitstream::parse(&bytes, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap()
                .image_header;
            let encoding = require_original_encoding(&image).unwrap();
            let jxl_gpu_formats::ColorSpecification::Defined(expected) = case.format().color_spec
            else {
                unreachable!()
            };
            assert_eq!(
                encoding.space,
                expected.space.rgb_space().unwrap(),
                "{}",
                case.name
            );
            assert_eq!(
                encoding.transfer,
                expected.transfer.rgb_transfer().unwrap(),
                "{}",
                case.name
            );
        }
        let mut image = header();
        let ColourEncodingInventory::Enumerated {
            white_point,
            rendering_intent,
            ..
        } = &mut image.colour_encoding
        else {
            unreachable!()
        };
        *white_point = WhitePointInventory::E;
        *rendering_intent = RenderingIntentInventory::Absolute;
        assert!(original_encoding(&image).is_none());
        image.colour_encoding = ColourEncodingInventory::Enumerated {
            colour_space: ColourSpaceInventory::Rgb,
            white_point: WhitePointInventory::E,
            primaries: PrimariesInventory::Custom {
                red: ChromaticityInventory { x: 1_000_000, y: 0 },
                green: ChromaticityInventory { x: 0, y: 1_000_000 },
                blue: ChromaticityInventory { x: 0, y: 0 },
            },
            transfer_function: TransferFunctionInventory::Linear,
            rendering_intent: RenderingIntentInventory::Relative,
        };
        let coordinates = require_original_encoding(&image)
            .unwrap()
            .space
            .chromaticities()
            .unwrap();
        assert_eq!(coordinates.red, Chromaticity::new(1.0, 0.0).unwrap());
        assert_eq!(coordinates.blue, Chromaticity::new(0.0, 0.0).unwrap());
        assert_eq!(coordinates.white, Chromaticity::E);
    }
}
