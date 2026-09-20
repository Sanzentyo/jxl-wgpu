#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::{
    InventoryLimits, ParseLimits,
    icc_profile::{IccProfileError, IccProfileLimits},
};
use jxl_test_support::oracles::icc_profile::IccProfileOracle;

fn compare(oracle: &IccProfileOracle, declaration: &str) {
    let native = oracle.create(declaration);
    let parsed = jxl_gpu_bitstream::parse(&native.input, ParseLimits::default()).unwrap();
    let image = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap_or_else(|error| panic!("{declaration}: {error:?}"))
        .image_header;
    let actual = image
        .original_icc_profile(IccProfileLimits::default())
        .unwrap();
    assert_eq!(
        actual.len(),
        native.profile.len(),
        "{declaration}; {:?}",
        image.colour_encoding
    );
    if actual.as_ref() != native.profile {
        let differences: Vec<_> = actual
            .iter()
            .zip(&native.profile)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .take(30)
            .collect();
        panic!(
            "{declaration}; {:?}: {differences:?}",
            image.colour_encoding
        );
    }
    let length = actual.len() as u64;
    assert_eq!(
        image
            .original_icc_profile(IccProfileLimits {
                max_profile_bytes: length
            })
            .unwrap(),
        actual
    );
    assert_eq!(
        image
            .original_icc_profile(IccProfileLimits {
                max_profile_bytes: length - 1
            })
            .unwrap_err(),
        IccProfileError::Limit {
            required: length,
            limit: length - 1
        }
    );
}

#[test]
fn original_profiles_match_native_generated_color_metadata() {
    let oracle = IccProfileOracle::compile();
    let whites = [
        (1, "0.3127 0.329"),
        (10, "0.333333 0.333333"),
        (11, "0.314 0.351"),
        (2, "0.345669 0.358496"),
        (2, "0.299999 0.315678"),
    ];
    let mut count = 0;
    for space in [0, 1] {
        for (white, xy) in whites {
            for primaries in [1, 9, 11, 2] {
                if space == 1 && primaries != 1 {
                    continue;
                }
                for transfer in [1, 8, 13, 16, 17, 18, 65535] {
                    for intent in 0..4 {
                        compare(
                            &oracle,
                            &format!(
                                "{space} {white} {primaries} {transfer} {intent} {xy} 0.64 0.33 0.21 0.71 0.15 0.06 0.4545455"
                            ),
                        );
                        count += 1;
                    }
                }
            }
        }
    }
    for space in [0, 1] {
        for gamma in [0.0001221, 0.0001234, 0.0033, 0.3333333, 0.9999, 1.0] {
            for intent in 0..4 {
                compare(
                    &oracle,
                    &format!(
                        "{space} 1 1 65535 {intent} 0.3127 0.329 0.64 0.33 0.30 0.60 0.15 0.06 {gamma}"
                    ),
                );
                count += 1;
            }
        }
    }
    for intent in 0..4 {
        // Adobe RGB and ProPhoto description aliases and a signed/scientific custom primary.
        for declaration in [
            format!("0 1 2 65535 {intent} 0.3127 0.329 0.64 0.33 0.21 0.71 0.15 0.06 0.4547069"),
            format!(
                "0 2 2 65535 {intent} 0.345669 0.358496 0.734699 0.265301 0.159597 0.840403 0.036598 0.000105 0.5555556"
            ),
            format!("0 1 2 13 {intent} 0.3127 0.329 0.70 0.30 -0.01 0.85 0.10 0.000009 0.4545455"),
        ] {
            compare(&oracle, &declaration);
            count += 1;
        }
    }
    compare(
        &oracle,
        "2 1 1 8 0 0.3127 0.329 0.64 0.33 0.30 0.60 0.15 0.06 0.4545455",
    );
    count += 1;
    assert_eq!(count, 761);
    eprintln!(
        "{count} native original ICC profiles matched exactly, including exact/one-short byte limits"
    );
}
