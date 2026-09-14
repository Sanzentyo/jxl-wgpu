use super::*;
use jxl_gpu_protocol::icc::IccStage;

#[test]
fn v2_lut_black_connections_match_native_and_independent_references() {
    let backend = backend().expect("source-black conformance requires a GPU adapter");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("black/manifest.json")).unwrap())
            .unwrap();
    let spaces: super::linear::Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("linear/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.profiles.len(), 20);
    assert_eq!(spaces.spaces.len(), 5);
    let identity = profile("mpe/identity");
    let targets = std::iter::once(("identity", None))
        .chain(
            spaces
                .spaces
                .iter()
                .map(|space| (space.name.as_str(), Some(space.encoding()))),
        )
        .collect::<Vec<_>>();
    let extent = Extent2d::new(manifest.width, manifest.height);
    let mut components = 0;
    let mut prepared = 0;
    let mut distinct = [0; 4];
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for record in &manifest.profiles {
            let source = profile(&format!("black/{}", record.name));
            let input = floats(&format!("black/{}.f32le", record.name));
            assert_eq!(input.len(), extent.area().unwrap() * record.channels);
            for &(target, space) in &targets {
                let relative =
                    references(&format!("black/{}_to_{target}_1.reference", record.name));
                for intent in [
                    IccRenderingIntent::Perceptual,
                    IccRenderingIntent::Relative,
                    IccRenderingIntent::Saturation,
                    IccRenderingIntent::Absolute,
                ] {
                    let transform = match space {
                        Some(space) => IccTransform::to_linear_rgb(&source, space, intent),
                        None => IccTransform::new(&source, &identity, intent),
                    }
                    .unwrap();
                    let has_probe = record.channels != 5
                        && matches!(
                            intent,
                            IccRenderingIntent::Perceptual | IccRenderingIntent::Saturation
                        );
                    assert_eq!(
                        transform
                            .program()
                            .stages()
                            .iter()
                            .filter(|stage| matches!(stage, IccStage::BlackPointConnection(_)))
                            .count(),
                        usize::from(has_probe)
                    );
                    let memory =
                        ResidentIccMemoryPlan::new(&transform, &backend.device().limits()).unwrap();
                    assert_eq!(memory.validation_bytes, if has_probe { 4 } else { 0 });
                    prepared += usize::from(has_probe);
                    let expected = references(&format!(
                        "black/{}_to_{target}_{}.reference",
                        record.name, intent as u32
                    ));
                    let actual = run(&backend, &pipeline, &transform, extent, &input, 5);
                    assert_eq!(actual.len(), expected.len());
                    for (index, ((value, reference), relative)) in
                        actual.iter().zip(&expected).zip(&relative).enumerate()
                    {
                        assert!(matches!(reference.native_semantics, 0 | 32));
                        assert!(
                            reference.lower <= reference.exact
                                && reference.exact <= reference.upper
                        );
                        assert!(
                            reference.native_lower <= reference.native
                                && reference.native <= reference.native_upper
                        );
                        assert!(
                            value.is_finite()
                                && reference.lower <= *value
                                && *value <= reference.upper,
                            "{} -> {target} {intent:?} {variant:?} sample {index}: GPU {value}, independent {reference:?}",
                            record.name
                        );
                        distinct[intent as usize] += usize::from(
                            reference.upper < relative.lower || relative.upper < reference.lower,
                        );
                        components += 1;
                    }
                }
            }
        }
    }
    assert_eq!(components, 954720);
    assert_eq!(prepared, 576);
    assert!(distinct[0] > 10000 && distinct[2] > 10000);
    assert_eq!(distinct[1], 0);
    assert_eq!(distinct[3], 0);
    eprintln!(
        "v2 LUT black: {components} components, {prepared} validated GPU preparations, distinct from relative {distinct:?}"
    );
}
