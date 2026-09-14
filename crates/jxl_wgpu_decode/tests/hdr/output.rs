use super::{backend, compare, corpus, oracle, planes, reference};
use jxl_gpu_formats::{Channel, ColorSpace, PixelFormat, SampleKind, TransferFunction};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, ModularChannels, NumericSampleMapping,
    native_modular_pixel_format,
};

#[test]
fn hdr_linear_conversion_and_numeric_original_output_match_independent_references() {
    let backend = backend();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in corpus::cases() {
        let data = case.bytes();
        let original = case.reference(false);
        let native_linear = (!case.sequence).then(|| case.reference(true));
        let cross_transfer = if case.transfer == TransferFunction::Pq {
            TransferFunction::Hlg
        } else {
            TransferFunction::Pq
        };
        for target in [TransferFunction::Linear, cross_transfer] {
            // The direct native linear reference uses the source primaries. Cross-HDR
            // output also changes primaries, exercising both sets of HLG coefficients.
            let target_space = if target == TransferFunction::Linear {
                case.space
            } else {
                ColorSpace::DisplayP3
            };
            let request = GpuOutputRequest::color(case.format(target, target_space))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
            let mut session = decoder.open(&data, request).unwrap();
            let mut frames = 0;
            while let Some(frame) = session.next_frame().unwrap() {
                let words = planes::read(&backend, &frame.output().outputs[0]);
                if target == TransferFunction::Linear
                    && case.xyb
                    && let Some(native) = &native_linear
                {
                    compare(
                        &words,
                        native,
                        case.tolerance(),
                        &format!("{} native linear", case.name),
                    );
                } else {
                    for (pixel, actual) in words.as_chunks::<4>().0.iter().enumerate() {
                        let source = &original[frames * case.frame_words() + pixel * 4..][..4];
                        let source_rgb = [source[0], source[1], source[2]].map(f64::from);
                        let (source_rgb, source_transfer) = if let Some(native) = &native_linear
                            && case.xyb
                        {
                            (
                                [
                                    native[pixel * 4],
                                    native[pixel * 4 + 1],
                                    native[pixel * 4 + 2],
                                ]
                                .map(f64::from),
                                TransferFunction::Linear,
                            )
                        } else {
                            (source_rgb, case.transfer)
                        };
                        let expected = oracle::convert(
                            source_rgb,
                            source_transfer,
                            target,
                            case.space,
                            target_space,
                            case.nits,
                        );
                        let interval = oracle::interval(
                            source_rgb,
                            source_transfer,
                            target,
                            case.space,
                            target_space,
                            case.nits,
                            f64::from(case.tolerance()),
                        );
                        for c in 0..4 {
                            let actual = f64::from(f32::from_bits(actual[c]));
                            let ([low, high], packing) = if c == 3 {
                                (
                                    [f64::from(source[3]) - 2e-6, f64::from(source[3]) + 2e-6],
                                    0.0,
                                )
                            } else {
                                (interval[c], 5e-5 * (1.0 + expected[c].abs()))
                            };
                            assert!(
                                actual.is_finite()
                                    && actual >= low - packing
                                    && actual <= high + packing,
                                "{}/{target:?}/{frames}/{pixel}/{c}: {actual}, interval [{low}, {high}], packing {packing}",
                                case.name
                            );
                        }
                    }
                }
                frames += 1;
            }
            assert_eq!(frames, case.frame_count());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        for alpha in [false, true] {
            let mapping = if !alpha && case.floating {
                NumericSampleMapping::NativeFloat
            } else {
                NumericSampleMapping::NormalizedUnsigned
            };
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                mapping,
            )
            .unwrap();
            let channel = if alpha { 3 } else { usize::from(!case.gray) };
            let request = if alpha {
                request.with_extra_channel(0).unwrap()
            } else {
                request.with_color_channel(channel as u32).unwrap()
            };
            let mut session = planes::open_fragmented(&decoder, &data, request);
            let mut frames = 0;
            while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
                for (pixel, actual) in planes::read(&backend, &frame.output().outputs[0])
                    .into_iter()
                    .enumerate()
                {
                    let expected = original[frames * case.frame_words() + pixel * 4 + channel];
                    let actual = f32::from_bits(actual);
                    let index = frames * case.width * case.height + pixel;
                    let [low, high] = reference::original_bounds(
                        &case,
                        &original,
                        native_linear.as_deref(),
                        index,
                    )[channel];
                    assert!(
                        actual.is_finite() && f64::from(actual) >= low && f64::from(actual) <= high,
                        "{} numeric {channel}/{frames}/{pixel}: {actual}, native {expected}",
                        case.name
                    );
                }
                frames += 1;
            }
            assert_eq!(frames, case.frame_count());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        if !case.floating {
            let channel = usize::from(!case.gray);
            let request = GpuOutputRequest::numeric(
                native_modular_pixel_format(ModularChannels::Gray, 16).unwrap(),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap()
            .with_color_channel(channel as u32)
            .unwrap();
            let mut session = decoder.open(&data, request).unwrap();
            let mut frames = 0;
            while let Some(frame) = session.next_frame().unwrap() {
                for (pixel, actual) in planes::read_bytes(&backend, &frame.output().outputs[0])
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .enumerate()
                {
                    let actual = f32::from(u16::from_le_bytes(*actual)) / 65535.0;
                    let expected = original[frames * case.frame_words() + pixel * 4 + channel];
                    let index = frames * case.width * case.height + pixel;
                    let [low, high] = reference::original_bounds(
                        &case,
                        &original,
                        native_linear.as_deref(),
                        index,
                    )[channel];
                    assert!(
                        f64::from(actual) >= low.clamp(0.0, 1.0) - 1.0 / 65535.0
                            && f64::from(actual) <= high.clamp(0.0, 1.0) + 1.0 / 65535.0,
                        "{} native U16: {actual}, native {expected}",
                        case.name
                    );
                }
                frames += 1;
            }
            assert_eq!(frames, case.frame_count());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
