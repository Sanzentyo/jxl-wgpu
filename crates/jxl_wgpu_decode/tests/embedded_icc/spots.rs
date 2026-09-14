use super::inventory;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
use jxl_gpu_protocol::icc::IccProfile;
use jxl_test_support::{fixtures::icc_spots as corpus, gpu::planes};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, OrientationPolicy, SpotColorPolicy,
    WgpuDecodeEngine,
};
use std::num::{NonZeroU64, NonZeroUsize};

const PIXELS: usize = 17 * 9;

#[derive(Clone, Copy)]
enum Target {
    Rgb,
    Gray,
    Linear,
}
impl Target {
    fn name(self) -> &'static str {
        match self {
            Self::Rgb => "rgb",
            Self::Gray => "gray",
            Self::Linear => "linear",
        }
    }
    fn colors(self) -> usize {
        if matches!(self, Self::Gray) { 1 } else { 3 }
    }
    fn format(self, planar: bool) -> PixelFormat {
        let color = if matches!(self, Self::Linear) {
            let ColorSpecification::Defined(mut color) =
                jxl_wgpu_decode::vardct_rgb8_format().color_spec
            else {
                unreachable!()
            };
            color.transfer = TransferFunction::Linear;
            ColorSpecification::Defined(color)
        } else {
            ColorSpecification::Icc(
                IccProfile::parse(
                    std::fs::read(
                        corpus::directory()
                            .parent()
                            .unwrap()
                            .join("embedded_icc")
                            .join(format!("{}.icc", self.name())),
                    )
                    .unwrap()
                    .into(),
                    Default::default(),
                )
                .unwrap(),
            )
        };
        if self.colors() == 1 {
            PixelFormat::gray_f32(true, planar, color)
        } else {
            PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
        }
    }
}

#[derive(Clone, Copy)]
struct Delivery {
    planar: bool,
    keep: bool,
    progressive: bool,
    window: Option<NonZeroU64>,
}

fn frames(
    backend: &WgpuBackend,
    case: &corpus::Case,
    request: GpuOutputRequest,
    delivery: Delivery,
    colors: usize,
) -> (Vec<Vec<u32>>, usize) {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if let Some(limit) = delivery.window {
        engine = engine.with_stream_window_limit(limit);
    }
    let decoder = GpuDecoder::new(engine);
    let format = request.format().clone();
    let request = request
        .with_progressive_output(delivery.progressive)
        .with_orientation_policy(if delivery.keep {
            OrientationPolicy::Keep
        } else {
            OrientationPolicy::Apply
        })
        .with_max_frame_slots(NonZeroUsize::new(64).unwrap());
    let data = case.bytes();
    let mut session = if delivery.window.is_some() {
        planes::open_fragmented(&decoder, &data, request)
    } else {
        decoder
            .open(&data, request)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name))
    };
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async())
        .unwrap_or_else(|e| panic!("{}: {e}", case.name))
    {
        let output = &update.output().outputs[0];
        assert_eq!(output.layout.format, format);
        assert_eq!(
            (output.layout.extent.width, output.layout.extent.height),
            if delivery.keep { (17, 9) } else { (9, 17) }
        );
        let words = planes::read(backend, output);
        assert_eq!(words.len(), PIXELS * (colors + 1));
        held.push((update, words));
    }
    drop(session);
    let updates = held.len();
    let mut frames = Vec::new();
    for (update, words) in held {
        assert_eq!(planes::read(backend, &update.output().outputs[0]), words);
        if update.progression().is_some() {
            continue;
        }
        let mut canonical = Vec::new();
        for p in 0..PIXELS {
            let (x, y) = (p % 17, p / 17);
            // Independent clockwise/counterclockwise placement in the transposed output.
            let position = if delivery.keep {
                p
            } else if case.gray {
                (16 - x) * 9 + y
            } else {
                x * 9 + 8 - y
            };
            for c in 0..colors + 1 {
                canonical.push(
                    words[if delivery.planar {
                        c * PIXELS + position
                    } else {
                        position * (colors + 1) + c
                    }],
                );
            }
        }
        frames.push(canonical);
    }
    assert_eq!(frames.len(), if case.sequence { 3 } else { 1 });
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    (frames, updates)
}

fn bounds(case: &corpus::Case, spots: bool, target: Target) -> Vec<[f64; 2]> {
    let bytes = std::fs::read(corpus::directory().join(format!(
        "{}.{}.{}.bounds",
        case.name,
        if spots { "render" } else { "preserve" },
        target.name()
    )))
    .unwrap();
    let (records, tail) = bytes.as_chunks::<16>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|record| {
            let pair = [
                f64::from_le_bytes(record[..8].try_into().unwrap()),
                f64::from_le_bytes(record[8..].try_into().unwrap()),
            ];
            assert!(pair[0].is_finite() && pair[1].is_finite() && pair[0] <= pair[1]);
            pair
        })
        .collect()
}

#[test]
fn spots_precede_icc_connections_and_leave_references_extras_and_output_lifetimes_intact() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let cases = corpus::cases();
    assert_eq!(cases.len(), 32);
    let mut components = 0;
    let mut presentations = 0;
    let mut updates = 0;
    let mut invalid_references = 0;
    for case in cases {
        let data = case.bytes();
        if !case.valid_reference_color {
            super::xyb::reject_post_transform_reference(&backend, &data, &case.name);
            invalid_references += 1;
            continue;
        }
        case.validate(&inventory(&data));
        let source =
            std::fs::read(corpus::directory().join(format!("{}.source.f32", case.name))).unwrap();
        let (words, tail) = source.as_chunks::<4>();
        assert!(tail.is_empty());
        let native: Vec<_> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
        let mut alpha_words = None;
        for target in [Target::Rgb, Target::Gray, Target::Linear] {
            for spots in [false, true] {
                let expected = bounds(&case, spots, target);
                for alpha in [
                    AlphaOutputPolicy::Preserve,
                    AlphaOutputPolicy::Unassociated,
                    AlphaOutputPolicy::Associated,
                ] {
                    let mut baseline = None;
                    for delivery in [
                        Delivery {
                            planar: false,
                            keep: true,
                            progressive: false,
                            window: None,
                        },
                        Delivery {
                            planar: true,
                            keep: false,
                            progressive: true,
                            window: None,
                        },
                        Delivery {
                            planar: false,
                            keep: false,
                            progressive: true,
                            window: NonZeroU64::new(256),
                        },
                    ] {
                        eprintln!(
                            "{} {} spots={spots}, alpha={alpha:?}, planar={}, window={:?}",
                            case.name,
                            target.name(),
                            delivery.planar,
                            delivery.window
                        );
                        let request = GpuOutputRequest::color(target.format(delivery.planar))
                            .unwrap()
                            .with_alpha_output_policy(alpha)
                            .with_spot_color_policy(if spots {
                                SpotColorPolicy::Render
                            } else {
                                SpotColorPolicy::Preserve
                            });
                        let (actual, count) =
                            frames(&backend, &case, request, delivery, target.colors());
                        updates += count;
                        presentations += actual.len();
                        let mut this_alpha = Vec::new();
                        assert_eq!(expected.len(), actual.len() * PIXELS * target.colors());
                        for (frame, words) in actual.iter().enumerate() {
                            for (p, pixel) in words.chunks_exact(target.colors() + 1).enumerate() {
                                let alpha_word = pixel[target.colors()];
                                this_alpha.push(alpha_word);
                                let a = f32::from_bits(alpha_word);
                                let native_alpha = native[frame * PIXELS * 13 + p * 4 + 3];
                                assert!((a - native_alpha).abs() <= 2e-6, "{} alpha", case.name);
                                let factor = match (case.sequence, alpha) {
                                    (true, AlphaOutputPolicy::Unassociated) => {
                                        f64::from(a).max(1.0 / 67108864.0)
                                    }
                                    (false, AlphaOutputPolicy::Associated) => {
                                        1.0 / f64::from(a).max(1.0 / 67108864.0)
                                    }
                                    _ => 1.0,
                                };
                                for c in 0..target.colors() {
                                    let value = f64::from(f32::from_bits(pixel[c])) * factor;
                                    let [low, high] =
                                        expected[(frame * PIXELS + p) * target.colors() + c];
                                    assert!(
                                        value.is_finite() && value >= low && value <= high,
                                        "{} {} spots={spots} {alpha:?}, frame {frame} pixel {p} channel {c}: {value} outside [{low}, {high}]",
                                        case.name,
                                        target.name()
                                    );
                                    components += 1;
                                }
                            }
                        }
                        if let Some(expected) = &alpha_words {
                            assert_eq!(&this_alpha, expected, "{} alpha changed", case.name);
                        } else {
                            alpha_words = Some(this_alpha);
                        }
                        if let Some(expected) = &baseline {
                            assert_eq!(
                                &actual, expected,
                                "{} layout/transport/progression",
                                case.name
                            );
                        } else {
                            baseline = Some(actual);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(invalid_references, 4);
    assert_eq!(components, 1_002_456);
    assert_eq!(presentations, 2808);
    assert!(updates > presentations);
    eprintln!(
        "ICC spots: {presentations} final presentations, {updates} retained updates, {components} independent components"
    );
}

#[test]
fn numeric_spot_and_other_extra_samples_ignore_presentation_policy() {
    use jxl_gpu_bitstream::SampleBitDepth;
    use jxl_gpu_formats::{Channel, SampleKind};
    use jxl_wgpu_decode::NumericSampleMapping;
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut compared = 0;
    for case in corpus::cases()
        .into_iter()
        .filter(|case| case.valid_reference_color)
    {
        let data = case.bytes();
        let image = inventory(&data).image_header;
        let bytes =
            std::fs::read(corpus::directory().join(format!("{}.source.f32", case.name))).unwrap();
        let native: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        for (index, extra) in image.extra_channels.iter().enumerate() {
            eprintln!("{} numeric extra {index}", case.name);
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                if matches!(extra.bit_depth, SampleBitDepth::Float { .. }) {
                    NumericSampleMapping::NativeFloat
                } else {
                    NumericSampleMapping::NormalizedUnsigned
                },
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Keep)
            .with_max_frame_slots(NonZeroUsize::new(3).unwrap());
            let rendered = super::output::frames(&backend, &data, request.clone(), None);
            let preserved = super::output::frames(
                &backend,
                &data,
                request.with_spot_color_policy(SpotColorPolicy::Preserve),
                NonZeroU64::new(256),
            );
            assert_eq!(rendered, preserved, "{} extra {index}", case.name);
            assert_eq!(rendered.len(), if case.sequence { 3 } else { 1 });
            for (frame, values) in rendered.iter().enumerate() {
                assert_eq!(values.len(), PIXELS);
                for (pixel, &word) in values.iter().enumerate() {
                    let expected = native[frame * PIXELS * 13 + (4 + index) * PIXELS + pixel];
                    let actual = f32::from_bits(word);
                    assert!(
                        (actual - expected).abs() <= 2e-6 * (1.0 + expected.abs()),
                        "{} extra {index}, frame {frame}, pixel {pixel}: {actual} vs {expected}",
                        case.name
                    );
                    compared += 1;
                }
            }
        }
    }
    assert_eq!(compared, 71_604);
    eprintln!("ICC spot numeric bypass: {compared} native extra-channel comparisons");
}
