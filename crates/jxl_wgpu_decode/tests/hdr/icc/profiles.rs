use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    target: String,
    width: usize,
    height: usize,
    channels: usize,
    frames: usize,
}

fn references(path: &Path) -> Vec<[f32; 6]> {
    let bytes = std::fs::read(path).unwrap();
    let (records, tail) = bytes.as_chunks::<28>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|record| {
            let values: [f32; 6] = std::array::from_fn(|c| {
                f32::from_le_bytes(record[c * 4..c * 4 + 4].try_into().unwrap())
            });
            assert!(values.iter().all(|v| v.is_finite()));
            let [native, exact, low, high, native_low, native_high] = values;
            assert!(low <= exact && exact <= high);
            assert!(native_low <= native && native <= native_high);
            assert!(matches!(
                u32::from_le_bytes(record[24..].try_into().unwrap()),
                0 | 32 | 64 | 96
            ));
            values
        })
        .collect()
}

#[test]
fn hdr_rgb_xyb_stills_and_sequences_match_native_and_scalar_icc_profiles() {
    let backend = backend();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = root.join("test-data/hdr_icc");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.cases.len(), 56);
    let mut components = 0;
    let mut presentations = 0;
    let mut targets = std::collections::BTreeSet::new();
    for (case, source) in manifest.cases.into_iter().zip(corpus::cases()) {
        assert_eq!(case.name, source.name);
        assert_eq!(
            (case.width, case.height, case.frames),
            (source.width, source.height, source.frame_count())
        );
        let data = source.bytes();
        let profile = IccProfile::parse(
            std::fs::read(
                root.join("../jxl_wgpu/test-data/icc")
                    .join(format!("{}.icc", case.target)),
            )
            .unwrap()
            .into(),
            Default::default(),
        )
        .unwrap();
        targets.insert(case.target.clone());
        let original = frames(
            &backend,
            &data,
            GpuOutputRequest::color(source.format(source.transfer, source.space))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve),
            false,
            4,
            false,
        );
        assert_eq!(original.len(), case.frames);
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            let reference =
                references(&directory.join(format!("{}_{}.reference", case.name, intent as u32)));
            assert_eq!(
                reference.len(),
                case.frames * case.width * case.height * case.channels
            );
            let mut baseline = None;
            for planar in [false, true] {
                let color = ColorSpecification::Icc(profile.clone());
                let format = if case.channels == 1 {
                    PixelFormat::gray_f32(true, planar, color)
                } else {
                    assert_eq!(case.channels, 3);
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                };
                for bounded in [false, true] {
                    eprintln!(
                        "{} -> {} {intent:?} planar {planar} bounded {bounded}",
                        case.name, case.target
                    );
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_icc_rendering_intent(intent)
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
                    let actual =
                        frames(&backend, &data, request, planar, case.channels + 1, bounded);
                    assert_eq!(actual.len(), case.frames);
                    presentations += actual.len();
                    for (frame, pixels) in actual.iter().enumerate() {
                        assert_eq!(pixels.len(), case.width * case.height * (case.channels + 1));
                        for (p, pixel) in pixels.chunks_exact(case.channels + 1).enumerate() {
                            for (c, &word) in pixel[..case.channels].iter().enumerate() {
                                let index =
                                    (frame * case.width * case.height + p) * case.channels + c;
                                let [_, exact, low, high, _, _] = reference[index];
                                let value = f32::from_bits(word);
                                assert!(
                                    value.is_finite() && value >= low && value <= high,
                                    "{} -> {} {intent:?} {frame}/{p}/{c}: {value}, exact {exact}, [{low}, {high}]",
                                    case.name,
                                    case.target
                                );
                                components += 1;
                            }
                            assert_eq!(
                                pixel[case.channels],
                                original[frame][p * 4 + 3],
                                "{} alpha",
                                case.name
                            );
                        }
                    }
                    if let Some(baseline) = &baseline {
                        assert_eq!(&actual, baseline, "{} ICC packing", case.name);
                    }
                    baseline = Some(actual);
                }
            }
        }
    }
    assert_eq!(targets.len(), 9);
    assert_eq!(components, 1_958_144);
    assert_eq!(presentations, 1280);
}
