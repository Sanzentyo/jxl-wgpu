use super::*;

#[test]
fn full_range_mpe_curves_keep_finite_results_across_large_intermediates() {
    check_scalar_corpus("mpe/range", 13);
}

#[test]
fn affine_powers_preserve_small_terms_through_exponents_and_offset_cancellation() {
    check_scalar_corpus("mpe/power", 22);
}

fn check_scalar_corpus(corpus: &str, profile_count: usize) {
    let Some(backend) = backend() else {
        return;
    };
    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(directory().join(corpus).join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest.profiles.len(), profile_count);
    let input = floats(&format!("{corpus}/input.f32le"));
    let extent = Extent2d::new(manifest.width, manifest.height);
    let identity = profile("mpe/identity");
    let mut components = 0;
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for record in &manifest.profiles {
            let selected = profile(&format!("{corpus}/{}", record.name));
            let bytes = std::fs::read(
                directory()
                    .join(corpus)
                    .join(format!("{}.reference", record.name)),
            )
            .unwrap();
            let (references, rest) = bytes.as_chunks::<16>();
            assert!(rest.is_empty());
            assert_eq!(references.len(), input.len());
            for reverse in [false, true] {
                let (source, target) = if reverse {
                    (&identity, &selected)
                } else {
                    (&selected, &identity)
                };
                let transform =
                    IccTransform::new(source, target, IccRenderingIntent::Relative).unwrap();
                let actual = run(&backend, &pipeline, &transform, extent, &input, 13);
                for (i, (&value, reference)) in actual.iter().zip(references).enumerate() {
                    let expected = f64::from_le_bytes(reference[..8].try_into().unwrap());
                    let radius = f64::from_le_bytes(reference[8..].try_into().unwrap());
                    assert!(expected.is_finite() && radius.is_finite() && radius >= 0.0);
                    if record.name == "half" && value != 0.0 {
                        assert_eq!(
                            value.to_bits(),
                            (expected as f32).to_bits(),
                            "round-to-even subnormal {i}"
                        );
                    }
                    assert!(
                        value.is_finite() && (f64::from(value) - expected).abs() <= radius,
                        "{} {variant:?} reverse={reverse}, sample {i}, input {}: GPU {value}, expected {expected} +/- {radius}",
                        record.name,
                        input[i]
                    );
                    components += 1;
                }
            }
        }
    }
    assert_eq!(components, input.len() * profile_count * 6);
    eprintln!(
        "{corpus}: {components} checked components across both directions and all kernel variants"
    );
}

#[test]
fn native_mpe_programs_match_ordered_scalar_references_in_both_directions() {
    let Some(backend) = backend() else {
        return;
    };
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("mpe/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.profiles.len(), 9);
    let identity = profile("mpe/identity");
    let input = floats("mpe/input.f32le");
    let extent = Extent2d::new(manifest.width, manifest.height);
    let pipeline = ResidentIccPipeline::new(backend.device()).unwrap();
    let mut components = 0;
    let mut signed = [0usize; 2];
    let mut max_channels = 0;
    let mut worst = 0_f32;
    for record in manifest.profiles {
        let selected = profile(&format!("mpe/{}", record.name));
        for (code, intent) in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ]
        .into_iter()
        .enumerate()
        {
            for reverse in [false, true] {
                let (source, target, direction) = if reverse {
                    (&identity, &selected, "reverse")
                } else {
                    (&selected, &identity, "forward")
                };
                let transform = IccTransform::new(source, target, intent).unwrap();
                max_channels = max_channels.max(transform.program().max_channels());
                let expected =
                    references(&format!("mpe/{}_{direction}_{code}.reference", record.name));
                let actual = run(
                    &backend,
                    &pipeline,
                    &transform,
                    extent,
                    &input,
                    7 + code as u32,
                );
                assert_eq!(actual.len(), expected.len());
                for (i, (value, reference)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(reference.native_semantics, 0);
                    assert!(
                        reference.lower <= reference.exact && reference.exact <= reference.upper
                    );
                    assert!(
                        reference.native >= reference.native_lower
                            && reference.native <= reference.native_upper
                    );
                    assert!(
                        value.is_finite() && *value >= reference.lower && *value <= reference.upper,
                        "{} {direction} intent {code}, sample {i}: GPU {value}, independent {reference:?}",
                        record.name
                    );
                    signed[0] += usize::from(*value < 0.0);
                    signed[1] += usize::from(*value > 1.0);
                    worst = worst.max((value - reference.exact).abs());
                    components += 1;
                }
            }
        }
    }
    assert_eq!(max_channels, 15);
    assert_eq!(components, 135864);
    assert!(signed[0] > 1000 && signed[1] > 1000);
    eprintln!("MPE {components} components, signed/above-one {signed:?}, max error {worst}");
}
