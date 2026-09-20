//! Native public original-profile export, independent of pixel-output profile selection.
use std::path::Path;

use jxl_gpu_bitstream::icc_profile::{IccProfileError, IccProfileLimits};
use jxl_test_support::oracles::icc_profile::IccProfileOracle;
use sha2::{Digest, Sha256};

#[test]
fn original_profiles_of_all_official_inputs_match_native_bytes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut inputs: Vec<_> = super::cases::CASES
        .iter()
        .map(|case| {
            (
                root.join("test-data/official_conformance")
                    .join(case.name)
                    .join("input.jxl"),
                case.input_sha256,
            )
        })
        .collect();
    inputs.extend([
        (
            root.join("../../fixtures/animation_spline.jxl"),
            "87793cac33d05eaa380011e3b0754ff6f228967431a126fd0a0ace2106940c79",
        ),
        (
            root.join("test-data/cmyk/layers.jxl"),
            "d732c8836bf1abeadf310d2e07387a32813ed4690d32650c1c25e541b80eed4a",
        ),
    ]);
    assert_eq!(inputs.len(), 27);
    let oracle = IccProfileOracle::compile();
    for (path, digest) in inputs {
        let input = std::fs::read(&path).unwrap();
        assert_eq!(
            Sha256::digest(&input).as_slice(),
            jxl_test_support::offline::hex::unhex(digest)
        );
        let native = oracle.read(&input);
        assert_eq!(native.input, input);
        let image = jxl_gpu_bitstream::parse(&input, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap()
            .image_header;
        let actual = image.original_icc_profile(Default::default()).unwrap();
        assert_eq!(actual.as_ref(), native.profile, "{path:?}");
        let size = actual.len() as u64;
        assert_eq!(
            image
                .original_icc_profile(IccProfileLimits {
                    max_profile_bytes: size
                })
                .unwrap(),
            actual
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
    }
}
