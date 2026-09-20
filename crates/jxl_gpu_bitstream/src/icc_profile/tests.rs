use super::*;
use crate::{
    ColourSpaceInventory, PrimariesInventory, RenderingIntentInventory, TransferFunctionInventory,
    WhitePointInventory,
};

fn srgb() -> ColourEncodingInventory {
    ColourEncodingInventory::Enumerated {
        colour_space: ColourSpaceInventory::Rgb,
        white_point: WhitePointInventory::D65,
        primaries: PrimariesInventory::Srgb,
        transfer_function: TransferFunctionInventory::Srgb,
        rendering_intent: RenderingIntentInventory::Relative,
    }
}

#[test]
fn original_icc_srgb_has_exact_size_admission() {
    let profile = srgb().generate_icc_profile(Default::default()).unwrap();
    let length = profile.len() as u64;
    assert_eq!(
        srgb()
            .generate_icc_profile(IccProfileLimits {
                max_profile_bytes: length
            })
            .unwrap(),
        profile
    );
    assert_eq!(
        srgb()
            .generate_icc_profile(IccProfileLimits {
                max_profile_bytes: length - 1
            })
            .unwrap_err(),
        IccProfileError::Limit {
            required: length,
            limit: length - 1
        }
    );
    assert_eq!(
        u32::from_be_bytes(profile[..4].try_into().unwrap()) as usize,
        profile.len()
    );
}

#[test]
fn embedded_export_borrows_exact_bytes_and_validates_binding_and_limits() {
    let raw = crate::test_fixtures::with_icc();
    let mut image = crate::parse(&raw, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let retained = image.clone();
    let original = retained.embedded_icc.as_ref().unwrap();
    let profile = retained.original_icc_profile(Default::default()).unwrap();
    assert!(matches!(profile, Cow::Borrowed(_)));
    assert_eq!(profile.as_ptr(), original.profile.as_ptr());
    let size = profile.len() as u64;
    assert_eq!(
        image
            .original_icc_profile(IccProfileLimits {
                max_profile_bytes: size
            })
            .unwrap(),
        profile
    );
    assert_eq!(
        image
            .original_icc_profile(IccProfileLimits {
                max_profile_bytes: size - 1
            })
            .unwrap_err(),
        IccProfileError::Limit {
            required: size,
            limit: size - 1
        }
    );
    // Export is independent of CMS profile parsing/admission and does not rewrite opaque bytes.
    image.embedded_icc.as_mut().unwrap().profile =
        std::sync::Arc::from(&b"opaque unsupported ICC"[..]);
    assert_eq!(
        image
            .original_icc_profile(Default::default())
            .unwrap()
            .as_ref(),
        b"opaque unsupported ICC"
    );
    image.embedded_icc = None;
    assert!(matches!(
        image.original_icc_profile(Default::default()),
        Err(IccProfileError::Invalid(_))
    ));
    image.colour_encoding = srgb();
    image.embedded_icc = Some(original.clone());
    assert!(matches!(
        image.original_icc_profile(Default::default()),
        Err(IccProfileError::Invalid(_))
    ));
    drop(image);
    drop(raw);
    assert_eq!(profile.as_ref(), &*original.profile);
}

#[test]
fn invalid_and_unrepresentable_color_metadata_never_yields_a_profile() {
    use crate::ChromaticityInventory as Point;
    use ColourEncodingInventory::{Enumerated, IccProfile};
    let Enumerated {
        white_point,
        primaries,
        transfer_function,
        rendering_intent,
        ..
    } = srgb()
    else {
        unreachable!()
    };
    let rgb = |white_point, primaries, transfer_function| Enumerated {
        colour_space: ColourSpaceInventory::Rgb,
        white_point,
        primaries,
        transfer_function,
        rendering_intent,
    };
    for encoding in [
        IccProfile {
            colour_space: ColourSpaceInventory::Rgb,
        },
        Enumerated {
            colour_space: ColourSpaceInventory::Unknown,
            white_point,
            primaries,
            transfer_function,
            rendering_intent,
        },
        rgb(white_point, primaries, TransferFunctionInventory::Unknown),
        Enumerated {
            colour_space: ColourSpaceInventory::Xyb,
            white_point,
            primaries,
            transfer_function,
            rendering_intent,
        },
    ] {
        assert!(matches!(
            encoding.generate_icc_profile(Default::default()),
            Err(IccProfileError::Unsupported(_))
        ));
    }
    for (scaled_gamma, inverted) in [
        (0, true),
        (1220, true),
        (10_000_001, true),
        (u32::MAX, true),
        (9_999_999, false),
    ] {
        let encoding = rgb(
            white_point,
            primaries,
            TransferFunctionInventory::Gamma {
                scaled_gamma,
                inverted,
            },
        );
        assert!(matches!(
            encoding.generate_icc_profile(Default::default()),
            Err(IccProfileError::Invalid(_))
        ));
    }
    for p in [
        Point { x: 312700, y: 0 },
        Point {
            x: 1_000_001,
            y: 329000,
        },
        Point {
            x: i32::MAX,
            y: 329000,
        },
    ] {
        assert!(matches!(
            rgb(WhitePointInventory::Custom(p), primaries, transfer_function)
                .generate_icc_profile(Default::default()),
            Err(IccProfileError::Invalid(_))
        ));
    }
    let singular = Point {
        x: 312700,
        y: 329000,
    };
    let primaries = PrimariesInventory::Custom {
        red: singular,
        green: singular,
        blue: singular,
    };
    assert!(matches!(
        rgb(white_point, primaries, transfer_function).generate_icc_profile(Default::default()),
        Err(IccProfileError::Invalid(_))
    ));
    let excessive = Enumerated {
        colour_space: ColourSpaceInventory::Grey,
        white_point: WhitePointInventory::Custom(Point { x: 2_000_000, y: 1 }),
        primaries,
        transfer_function,
        rendering_intent,
    };
    assert!(matches!(
        excessive.generate_icc_profile(Default::default()),
        Err(IccProfileError::Invalid(_))
    ));
}
