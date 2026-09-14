use super::*;
use jxl_gpu_protocol::icc::{IccClutInterpolation, IccError, IccStage};

#[test]
fn legacy_luts_match_independent_stages_in_both_directions_and_all_kernels() {
    let Some(backend) = backend() else {
        return;
    };
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory().join("lut/manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.profiles.len(), 41);
    let extent = Extent2d::new(manifest.width, manifest.height);
    let identity = profile("mpe/identity");
    let mut components = 0;
    let mut native_extensions = 0;
    let mut interpolations = [0; 2];
    let mut max_channels = 0;
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for record in &manifest.profiles {
            let selected = profile(&format!("lut/{}", record.name));
            for reverse in [false, true] {
                let (source, target, direction) = if reverse {
                    (&identity, &selected, "reverse")
                } else {
                    (&selected, &identity, "forward")
                };
                let input = floats(&format!("lut/{}_{direction}.f32le", record.name));
                assert_eq!(
                    input.len(),
                    extent.area().unwrap() * if reverse { 3 } else { record.channels }
                );
                for intent in [
                    IccRenderingIntent::Perceptual,
                    IccRenderingIntent::Relative,
                    IccRenderingIntent::Saturation,
                    IccRenderingIntent::Absolute,
                ] {
                    if !reverse
                        && selected.header().version >> 24 == 2
                        && matches!(
                            intent,
                            IccRenderingIntent::Perceptual | IccRenderingIntent::Saturation
                        )
                    {
                        assert!(matches!(
                            IccTransform::new(source, target, intent),
                            Err(IccError::LutBlackPoint { .. })
                        ));
                        continue;
                    }
                    let transform = IccTransform::new(source, target, intent).unwrap();
                    max_channels = max_channels.max(transform.program().max_channels());
                    for stage in transform.program().stages() {
                        if let IccStage::Clut(clut) = stage {
                            let index = match clut.interpolation() {
                                IccClutInterpolation::Tetrahedral => 0,
                                IccClutInterpolation::Multilinear => 1,
                            };
                            interpolations[index] += 1;
                        }
                    }
                    let expected = references(&format!(
                        "lut/{}_{direction}_{}.reference",
                        record.name, intent as u32
                    ));
                    let actual = run(&backend, &pipeline, &transform, extent, &input, 7);
                    assert_eq!(actual.len(), expected.len());
                    assert_eq!(
                        actual.len(),
                        extent.area().unwrap() * if reverse { record.channels } else { 3 }
                    );
                    for (i, (&value, reference)) in actual.iter().zip(&expected).enumerate() {
                        assert!(matches!(reference.native_semantics, 0 | 32));
                        native_extensions += usize::from(reference.native_semantics != 0);
                        assert!(
                            reference.lower <= reference.exact
                                && reference.exact <= reference.upper
                        );
                        // Native extensions have their own independently evaluated center.
                        // Every component still checks both intervals; no mask skips a check.
                        assert!(
                            reference.native_lower <= reference.native
                                && reference.native <= reference.native_upper
                        );
                        assert!(
                            value.is_finite()
                                && reference.lower <= value
                                && value <= reference.upper,
                            "{} {direction} {intent:?} {variant:?}, sample {i}: GPU {value}, independent {reference:?}",
                            record.name
                        );
                        components += 1;
                    }
                }
            }
        }
    }
    assert_eq!(max_channels, 15);
    assert!(interpolations.into_iter().all(|count| count > 0));
    assert_eq!(components, 607308);
    assert!(native_extensions > 1000);
    eprintln!(
        "legacy LUTs: {components} components, {native_extensions} native extension intervals, interpolation programs {interpolations:?}"
    );
}
