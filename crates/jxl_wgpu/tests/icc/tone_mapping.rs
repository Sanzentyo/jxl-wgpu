use super::*;
use jxl_gpu_protocol::{LuminanceRange, ToneMapping};
use jxl_test_support::oracles::tone_mapping::Mapping;

fn transform(profile: &IccProfile, mapping: Mapping) -> IccTransform {
    IccTransform::new(profile, profile, IccRenderingIntent::Relative)
        .unwrap()
        .with_tone_mapping(
            ToneMapping::new(
                LuminanceRange::new(mapping.source[0] as f32, mapping.source[1] as f32).unwrap(),
                LuminanceRange::new(mapping.target[0] as f32, mapping.target[1] as f32).unwrap(),
                mapping.protected,
            )
            .unwrap(),
        )
        .unwrap()
}

const NEUTRAL: [f64; 3] = [0.9642, 1.0, 0.8249];

#[test]
fn bt2408_matches_native_libjxl_and_independent_bernstein_reference() {
    let backend = backend().expect("tone mapping requires an actual GPU");
    eprintln!("Tone mapping adapter: {:?}", backend.adapter_info());
    let identity = profile("mpe/identity");
    let records: Vec<[f32; 10]> =
        std::fs::read_to_string(directory().join("../tone_mapping/native.txt"))
            .unwrap()
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();
    assert_eq!(records.len(), 18 * 277);
    let mut components = 0;
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for records in records.as_chunks::<277>().0 {
            let first = records[0];
            let mapping = Mapping {
                source: [first[0], first[1]].map(f64::from),
                target: [first[2], first[3]].map(f64::from),
                protected: 0.0,
            };
            let input: Vec<_> = records
                .iter()
                .flat_map(|v| v[4..7].iter().copied())
                .collect();
            let actual = run(
                &backend,
                &pipeline,
                &transform(&identity, mapping),
                Extent2d::new(277, 1),
                &input,
                7,
            );
            for (index, (record, actual)) in
                records.iter().zip(actual.as_chunks::<3>().0).enumerate()
            {
                let rgb = [record[4], record[5], record[6]].map(f64::from);
                let expected = mapping.apply(rgb, [0.0, 1.0, 0.0], NEUTRAL);
                for c in 0..3 {
                    // Two ST 2084 evaluations plus F32 coefficients. This fixed allowance
                    // predates GPU execution; the native F32 implementation has its own check.
                    let allowance = 8e-5 * (1.0 + expected[c].abs());
                    let gpu = f64::from(actual[c]);
                    let native = f64::from(record[c + 7]);
                    assert!(
                        native.is_finite() && (native - expected[c]).abs() <= allowance,
                        "native {mapping:?}/{index}/{c}: {native}, F64 {}",
                        expected[c]
                    );
                    assert!(
                        gpu.is_finite() && (gpu - expected[c]).abs() <= allowance,
                        "GPU {variant:?} {mapping:?}/{index}/{c}: {gpu}, F64 {}",
                        expected[c]
                    );
                    components += 1;
                }
            }
        }
    }
    assert_eq!(components, 44_874);
}

#[test]
fn protected_shadows_and_degenerate_ranges_keep_defined_luminance() {
    let backend = backend().expect("tone mapping requires an actual GPU");
    let identity = profile("mpe/identity");
    let mut mappings: Vec<_> = [0.0, 0.01, 20.0, 95.0, 99.0, 100.0, 200.0, 4000.0]
        .into_iter()
        .map(|protected| Mapping {
            source: [0.0, 4000.0],
            target: [0.0, 100.0],
            protected,
        })
        .collect();
    mappings.extend([
        Mapping {
            source: [0.0, 1000.0],
            target: [0.125, 80.0],
            protected: 0.0,
        },
        Mapping {
            source: [0.0, 1000.0],
            target: [0.0, f64::from(80.12345_f32)],
            protected: f64::from(80.12345_f32) * 0.375,
        },
        Mapping {
            source: [20.0, 4000.0],
            target: [0.5, 100.0],
            protected: 10.0,
        },
        Mapping {
            source: [100.0, 100.0],
            target: [100.0, 100.0],
            protected: 0.0,
        },
        Mapping {
            source: [0.0, 100.0],
            target: [0.0, 4000.0],
            protected: 0.0,
        },
        Mapping {
            source: [0.0, 100.0],
            target: [0.0, 100.0],
            protected: 0.0,
        },
        Mapping {
            source: [1000.0, 1000.0],
            target: [0.0, 100.0],
            protected: 0.0,
        },
        Mapping {
            source: [0.0, 1000.0],
            target: [100.0, 100.0],
            protected: 0.0,
        },
    ]);
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(backend.device(), variant).unwrap();
        for mapping in &mappings {
            // Dense neutral ramps and exact threshold neighbors exercise continuity and monotonicity.
            let mut ys: Vec<_> = (0..=1024).map(|i| i as f32 / 1024.0).collect();
            let threshold = (mapping.protected / mapping.source[1]) as f32;
            ys.extend([
                threshold.next_down(),
                threshold,
                threshold.next_up(),
                -0.1,
                2.0,
            ]);
            ys.sort_by(f32::total_cmp);
            let input: Vec<_> = ys
                .iter()
                .flat_map(|y| NEUTRAL.map(|v| *y * v as f32))
                .collect();
            let actual = run(
                &backend,
                &pipeline,
                &transform(&identity, *mapping),
                Extent2d::new(ys.len() as u32, 1),
                &input,
                5,
            );
            let mut previous = f64::NEG_INFINITY;
            for (index, (rgb, actual)) in input
                .as_chunks::<3>()
                .0
                .iter()
                .zip(actual.as_chunks::<3>().0)
                .enumerate()
            {
                let expected = mapping.apply(rgb.map(f64::from), [0.0, 1.0, 0.0], NEUTRAL);
                for c in 0..3 {
                    let value = f64::from(actual[c]);
                    assert!(
                        value.is_finite()
                            && (value - expected[c]).abs() <= 8e-5 * (1.0 + expected[c].abs()),
                        "{variant:?} {mapping:?}/{index}/{c}: {value}, {}",
                        expected[c]
                    );
                }
                let y = f64::from(actual[1]);
                // Negative-light extension is checked against the oracle above. Display
                // monotonicity applies to nonnegative input luminance.
                if rgb[1] >= 0.0 {
                    assert!(
                        y >= previous - 1.6e-4,
                        "decreasing shoulder {mapping:?}/{index}: {previous} -> {y}"
                    );
                    previous = y;
                }
                if mapping.protected > 0.0
                    && f64::from(rgb[1]) * mapping.source[1] < mapping.protected
                {
                    let preserved = f64::from(rgb[1]) * mapping.source[1] / mapping.target[1];
                    assert!(
                        (y - preserved).abs() <= 2e-6 * (1.0 + preserved.abs()),
                        "protected {variant:?} {mapping:?}/{index}: {y}, preserved {preserved}"
                    );
                }
            }
        }
    }
}
