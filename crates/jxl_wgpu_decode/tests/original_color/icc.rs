use super::*;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Manifest {
    width: usize,
    height: usize,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    target: String,
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

fn frames(
    backend: &WgpuBackend,
    data: &[u8],
    request: GpuOutputRequest,
    limit: Option<NonZeroU64>,
    channels: usize,
    planar: bool,
    progressive: bool,
) -> Vec<Vec<u32>> {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if let Some(limit) = limit {
        engine = engine.with_stream_window_limit(limit);
    }
    let decoder = GpuDecoder::new(engine);
    let format = request.format().clone();
    let request = request
        .with_progressive_output(progressive)
        .with_max_frame_slots(NonZeroUsize::new(64).unwrap());
    let mut session = if limit.is_some() {
        planes::open_fragmented(&decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
        let output = &update.output().outputs[0];
        assert_eq!(output.layout.format, format);
        let words = planes::read(backend, output);
        assert_eq!(words.len(), 37 * 19 * channels);
        held.push((update, words));
    }
    drop(session);
    let mut frames = Vec::new();
    for (update, words) in held {
        assert_eq!(planes::read(backend, &update.output().outputs[0]), words);
        if update.progression().is_none() {
            frames.push(if planar {
                (0..37 * 19)
                    .flat_map(|p| (0..channels).map(move |c| (p, c)))
                    .map(|(p, c)| words[c * 37 * 19 + p])
                    .collect()
            } else {
                words
            });
        }
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    frames
}

#[test]
fn enumerated_color_converts_to_requested_icc_through_stills_and_reference_composition() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let directory = root.join("test-data/rgb_icc");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        (manifest.width, manifest.height, manifest.cases.len()),
        (37, 19, 228)
    );
    let corpus: Vec<_> = corpus::cases()
        .into_iter()
        .chain(corpus::analytic_cases())
        .collect();
    let mut components = 0;
    let mut presentations = 0;
    let mut targets = std::collections::BTreeSet::new();
    for (case, original) in manifest.cases.into_iter().zip(corpus) {
        assert_eq!(case.name, original.name);
        targets.insert(case.target.clone());
        let data = original.bytes();
        let baseline = frames(
            &backend,
            &data,
            GpuOutputRequest::color(original.format())
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve),
            None,
            4,
            false,
            false,
        );
        assert_eq!(baseline.len(), case.frames);
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
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            let reference =
                references(&directory.join(format!("{}_{}.reference", case.name, intent as u32)));
            assert_eq!(reference.len(), case.frames * 37 * 19 * case.channels);
            let mut final_baseline = None;
            for planar in [false, true] {
                let color = ColorSpecification::Icc(profile.clone());
                let format = if case.channels == 1 {
                    PixelFormat::gray_f32(true, planar, color)
                } else {
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                };
                for limit in [None, NonZeroU64::new(256)] {
                    eprintln!(
                        "{} -> {} {intent:?}, planar {planar}, window {limit:?}",
                        case.name, case.target
                    );
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                        .with_icc_rendering_intent(intent);
                    let actual = frames(
                        &backend,
                        &data,
                        request,
                        limit,
                        case.channels + 1,
                        planar,
                        !planar,
                    );
                    assert_eq!(actual.len(), case.frames);
                    for (frame, pixels) in actual.iter().enumerate() {
                        for (p, pixel) in pixels.chunks_exact(case.channels + 1).enumerate() {
                            assert_eq!(
                                pixel[case.channels],
                                baseline[frame][p * 4 + 3],
                                "alpha {} frame {frame} pixel {p}",
                                case.name
                            );
                            for c in 0..case.channels {
                                let actual = f32::from_bits(pixel[c]);
                                let expected = reference[(frame * 37 * 19 + p) * case.channels + c];
                                assert!(
                                    actual.is_finite()
                                        && actual >= expected[2]
                                        && actual <= expected[3],
                                    "{} -> {} {intent:?}, frame {frame}, pixel {p}, channel {c}: GPU {actual}, independent {expected:?}",
                                    case.name,
                                    case.target
                                );
                                components += 1;
                            }
                        }
                    }
                    presentations += actual.len();
                    if let Some(baseline) = &final_baseline {
                        assert_eq!(
                            &actual, baseline,
                            "planar/fragmented/final-only {}",
                            case.name
                        );
                    } else {
                        final_baseline = Some(actual);
                    }
                }
            }
        }
    }
    assert_eq!(targets.len(), 9);
    assert_eq!(presentations, 9120);
    assert_eq!(components, 16_197_120);
    eprintln!(
        "enumerated to ICC: {presentations} presentations, {components} independent components"
    );
}
