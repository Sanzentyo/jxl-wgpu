use super::{WgpuBackend, corpus, oracle, planes, tolerance};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::WhitePointAdaptation;
use jxl_test_support::oracles::extra_channels;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest};

fn native(data: &[u8], case: &corpus::Case) -> Vec<f32> {
    let values = extra_channels::libjxl_output(
        data,
        &["--original", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("libjxl 0.12.0 original-color oracle is required");
    let pixels = 37 * 19;
    assert_eq!(values.len(), pixels * 5 * if case.sequence { 4 } else { 1 });
    let mut rgba = Vec::new();
    for frame in values.chunks_exact(pixels * 5) {
        for pixel in 0..pixels {
            assert_eq!(
                frame[pixel * 4 + 3].to_bits(),
                frame[pixels * 4 + pixel].to_bits()
            );
        }
        rgba.extend_from_slice(&frame[..pixels * 4]);
    }
    rgba
}

#[test]
fn all_original_intents_preserve_native_reconstruction_and_composed_progression() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let cases = corpus::analytic_cases();
    assert_eq!(cases.len(), 80);
    for case in cases {
        let original = case.bytes();
        let frozen: Vec<_> = case.reference().into_iter().map(f32::to_bits).collect();
        for intent in corpus::intents::ALL {
            let data = corpus::intents::replace(&original, intent);
            let reference = native(&data, &case);
            assert_eq!(
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                frozen,
                "independent native reconstruction {}/{intent:?}",
                case.name
            );
            let mut named = case.clone();
            named.name = format!("{}_{intent:?}", case.name);
            eprintln!("intent profile {}", named.name);
            super::check_original_profile(&backend, &named, &data, &reference);
        }
    }
}

#[test]
fn original_intents_keep_requested_bradford_and_absolute_output_independent() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in corpus::analytic_cases() {
        let original = case.bytes();
        let reference = case.reference();
        let linear_original = (case.mode.xyb()
            && !case.sequence
            && matches!(
                case.transfer.transfer,
                jxl_gpu_bitstream::TransferFunctionInventory::Gamma { .. }
                    | jxl_gpu_bitstream::TransferFunctionInventory::Dci
            ))
        .then(|| oracle::linear_original_still(&case));
        let ColorSpecification::Defined(source) = case.format().color_spec else {
            unreachable!()
        };
        for adaptation in [WhitePointAdaptation::Bradford, WhitePointAdaptation::None] {
            let matrix = oracle::matrix_with_adaptation(
                source.space,
                ColorSpace::Bt709,
                adaptation == WhitePointAdaptation::Bradford,
            );
            let mut target = source;
            target.space = ColorSpace::Bt709;
            target.transfer = TransferFunction::Linear;
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                ColorSpecification::Defined(target),
            ))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_white_point_adaptation(adaptation);
            let mut baseline = None;
            for intent in corpus::intents::ALL {
                let data = corpus::intents::replace(&original, intent);
                let mut session = decoder.open(&data, request.clone()).unwrap();
                let mut snapshots = Vec::new();
                while let Some(frame) = session.next_frame().unwrap() {
                    let words = planes::read(&backend, &frame.output().outputs[0]);
                    assert_eq!(words.len(), 37 * 19 * 4);
                    for (pixel, actual) in words.as_chunks::<4>().0.iter().enumerate() {
                        let values = &reference[(snapshots.len() * 37 * 19 + pixel) * 4..][..4];
                        let (rgb, transfer) = if let Some(linear) = &linear_original {
                            (
                                [linear[pixel][0], linear[pixel][1], linear[pixel][2]],
                                TransferFunction::Linear,
                            )
                        } else {
                            (
                                [values[0], values[1], values[2]].map(f64::from),
                                source.transfer,
                            )
                        };
                        let expected =
                            oracle::convert(rgb, transfer, TransferFunction::Linear, matrix);
                        let bounds = oracle::interval(
                            rgb,
                            transfer,
                            TransferFunction::Linear,
                            matrix,
                            f64::from(tolerance(&case)),
                        );
                        for channel in 0..4 {
                            let actual = f64::from(f32::from_bits(actual[channel]));
                            let (low, high, packing) = if channel == 3 {
                                let alpha = f64::from(values[3]);
                                (alpha - 2e-6, alpha + 2e-6, 0.0)
                            } else {
                                (
                                    bounds[channel][0],
                                    bounds[channel][1],
                                    5e-6 * (1.0 + expected[channel].abs()),
                                )
                            };
                            assert!(
                                actual.is_finite()
                                    && actual >= low - packing
                                    && actual <= high + packing,
                                "{}/{intent:?}/{adaptation:?}/{}/{pixel}/{channel}: {actual}, [{low}, {high}] + {packing}",
                                case.name,
                                snapshots.len()
                            );
                        }
                    }
                    snapshots.push(words);
                }
                assert_eq!(snapshots.len(), if case.sequence { 4 } else { 1 });
                if let Some(baseline) = &baseline {
                    assert_eq!(
                        &snapshots, baseline,
                        "{} {adaptation:?}: original intent changed requested output",
                        case.name
                    );
                } else {
                    baseline = Some(snapshots);
                }
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
