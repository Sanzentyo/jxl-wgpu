use super::{Manifest, backend, directory, floats, profile, references, run};
use jxl_gpu_protocol::{
    Extent2d,
    icc::{IccRenderingIntent, IccTransform},
};
use jxl_wgpu::ResidentIccPipeline;

#[test]
fn every_matrix_intent_preserves_white_black_and_parametric_curve_boundaries() {
    let backend = backend().expect("ICC intent conformance requires a GPU adapter");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("intents/manifest.json")).unwrap())
            .unwrap();
    assert_eq!((manifest.width, manifest.height), (17, 9));
    assert_eq!(manifest.profiles.len(), 26);
    let profiles: Vec<_> = manifest
        .profiles
        .iter()
        .map(|record| {
            let name = format!("intents/{}", record.name);
            let parsed = profile(&name);
            let input = floats(&format!("{name}_input.f32le"));
            assert_eq!(input.len(), 153 * record.channels);
            (record, parsed, input)
        })
        .collect();
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let mut components = 0;
    let mut masks = [0usize; 32];
    let mut maximum_error = 0.0_f32;
    let mut nonzero_offsets = 0;
    for (source, source_profile, input) in &profiles {
        for (target, target_profile, _) in &profiles {
            for intent in [
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Relative,
                IccRenderingIntent::Saturation,
                IccRenderingIntent::Absolute,
            ] {
                let transform = IccTransform::new(source_profile, target_profile, intent).unwrap();
                assert_eq!(transform.source().channels(), source.channels);
                assert_eq!(transform.target().channels(), target.channels);
                nonzero_offsets += usize::from(transform.offset().iter().any(|v| *v != 0.0));
                let actual = run(
                    &backend,
                    &pipeline,
                    &transform,
                    Extent2d::new(manifest.width, manifest.height),
                    input,
                    9,
                );
                let name = format!("{}_to_{}_{}", source.name, target.name, intent as u32);
                let expected = references(&format!("intents/{name}.reference"));
                assert_eq!(actual.len(), expected.len());
                for (index, (value, reference)) in actual.into_iter().zip(expected).enumerate() {
                    assert!(
                        reference.lower <= reference.exact && reference.exact <= reference.upper
                    );
                    assert!(
                        value.is_finite() && value >= reference.lower && value <= reference.upper,
                        "{name} component {index}: GPU {value}, independent {reference:?}"
                    );
                    assert!((0.0..=1.0).contains(&value));
                    assert!(reference.native_semantics < masks.len() as u32);
                    masks[reference.native_semantics as usize] += 1;
                    if reference.native_semantics == 0 {
                        assert!(
                            reference.native >= reference.native_lower
                                && reference.native <= reference.native_upper,
                            "{name} component {index}: native outside independent interval: {reference:?}"
                        );
                    }
                    components += 1;
                    maximum_error = maximum_error.max((value - reference.exact).abs());
                }
            }
        }
    }
    assert_eq!(components, 827424);
    let nonempty: Vec<_> = masks
        .into_iter()
        .enumerate()
        .filter(|(_, count)| *count != 0)
        .collect();
    assert_eq!(
        nonempty,
        [
            (0, 767812),
            (1, 360),
            (4, 548),
            (8, 2304),
            (12, 96),
            (16, 55918),
            (17, 264),
            (20, 26),
            (24, 96)
        ]
    );
    assert!(nonzero_offsets > 500);
    eprintln!(
        "ICC intents: 2704 connections, {components} components, {nonzero_offsets} affine programs, maximum error {maximum_error}"
    );
}
