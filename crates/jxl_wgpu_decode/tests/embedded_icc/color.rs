use super::corpus;
use jxl_gpu_bitstream::FrameEncoding;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccLimits, IccProfile};
use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use std::num::NonZeroU64;

fn reference(case: corpus::Case, kind: &str) -> Vec<u32> {
    jxl_test_support::offline::unhex(
        &std::fs::read_to_string(
            corpus::directory().join(format!("{}.{kind}.f32.hex", case.name())),
        )
        .unwrap(),
    )
    .as_chunks::<4>()
    .0
    .iter()
    .copied()
    .map(u32::from_le_bytes)
    .collect()
}

fn assert_color(actual: &[u8], expected: &[u8], tolerance: f32, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let mut maximum = 0.0_f32;
    for (actual, expected) in actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
    {
        let a = f32::from_le_bytes(*actual);
        let e = f32::from_le_bytes(*expected);
        assert!(a.is_finite() && e.is_finite(), "{context}: {a} vs {e}");
        maximum = maximum.max((a - e).abs());
    }
    println!("{context}: max absolute error {maximum}");
    assert!(
        maximum <= tolerance,
        "{context}: {maximum} exceeds {tolerance}"
    );
}

#[test]
fn original_icc_device_color_preserves_gray_rgb_and_alpha_words() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for limit in [None, NonZeroU64::new(256)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for case in corpus::cases().filter(|case| !case.xyb) {
            let data = case.bytes();
            let profile = IccProfile::parse(case.profile().into(), IccLimits::default()).unwrap();
            for planar in [false, true] {
                for alpha in [false, true] {
                    let color = ColorSpecification::Icc(profile.clone());
                    let format = if case.gray {
                        PixelFormat::gray_f32(alpha, planar, color)
                    } else {
                        PixelFormat::rgb_f32(
                            if alpha {
                                RgbChannelOrder::Rgba
                            } else {
                                RgbChannelOrder::Rgb
                            },
                            planar,
                            color,
                        )
                    };
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
                    let mut session = if limit.is_some() {
                        planes::open_fragmented(&decoder, &data, request)
                    } else {
                        decoder.open(&data, request).unwrap()
                    };
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    let output = &frame.output().outputs[0];
                    assert_eq!(output.layout.format, format);
                    let actual = planes::read_bytes(&backend, output);
                    let input = reference(case, "native");
                    let source_channels = if case.gray { 2 } else { 4 };
                    let channels = source_channels - usize::from(!alpha);
                    let mut expected = Vec::new();
                    if planar {
                        for channel in 0..channels {
                            for pixel in input.chunks_exact(source_channels) {
                                expected.extend_from_slice(&pixel[channel].to_le_bytes());
                            }
                        }
                    } else {
                        for pixel in input.chunks_exact(source_channels) {
                            for word in &pixel[..channels] {
                                expected.extend_from_slice(&word.to_le_bytes());
                            }
                        }
                    }
                    let context = format!("{} planar={planar} alpha={alpha}", case.name());
                    if case.encoding == FrameEncoding::Modular {
                        assert_eq!(actual, expected, "{context}");
                    } else {
                        assert_color(&actual, &expected, 2e-5, &context);
                    }
                    assert!(
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .is_none()
                    );
                }
            }
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn original_icc_color_matches_independent_native_cms_and_preserves_alpha() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for limit in [None, NonZeroU64::new(256)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for case in corpus::cases().filter(|case| !case.xyb) {
            let data = case.bytes();
            for kind in ["linear", "srgb", "other"] {
                let (specification, gray) = if kind == "other" {
                    let target = corpus::cases()
                        .find(|target| target.gray != case.gray)
                        .unwrap();
                    (
                        ColorSpecification::Icc(
                            IccProfile::parse(target.profile().into(), IccLimits::default())
                                .unwrap(),
                        ),
                        target.gray,
                    )
                } else {
                    let mut spec = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
                    if kind == "linear"
                        && let ColorSpecification::Defined(ref mut spec) = spec
                    {
                        spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
                    }
                    (spec, false)
                };
                for planar in [false, true] {
                    let format = if gray {
                        PixelFormat::gray_f32(true, planar, specification.clone())
                    } else {
                        PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, specification.clone())
                    };
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
                    let mut session = if limit.is_some() {
                        planes::open_fragmented(&decoder, &data, request)
                    } else {
                        decoder.open(&data, request).unwrap()
                    };
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    let output = &frame.output().outputs[0];
                    assert_eq!(output.layout.format, format);
                    let actual = planes::read_bytes(&backend, output);
                    let expected = reference(case, &format!("{kind}.scalar"));
                    let channels = if gray { 2 } else { 4 };
                    let expected: Vec<_> = if planar {
                        (0..channels)
                            .flat_map(|channel| {
                                expected
                                    .chunks_exact(channels)
                                    .flat_map(move |pixel| pixel[channel].to_le_bytes())
                            })
                            .collect()
                    } else {
                        expected
                            .iter()
                            .flat_map(|word| word.to_le_bytes())
                            .collect()
                    };
                    assert_color(
                        &actual,
                        &expected,
                        2e-4,
                        &format!("{} {kind} planar={planar}", case.name()),
                    );
                    let source_channels = if case.gray { 2 } else { 4 };
                    for (index, pixel) in case.input().chunks_exact(source_channels).enumerate() {
                        let alpha_index = if planar {
                            (channels - 1) * 153 + index
                        } else {
                            channels * index + channels - 1
                        };
                        assert_eq!(
                            &actual[alpha_index * 4..alpha_index * 4 + 4],
                            &pixel[source_channels - 1].to_le_bytes()
                        );
                    }
                    assert!(
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .is_none()
                    );
                }
            }
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn same_icc_device_output_preserves_ieee_words_without_selecting_a_cms_method() {
    use jxl_gpu_protocol::{WhitePointAdaptation, icc::IccRenderingIntent};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let donor = corpus::cases().find(|case| case.gray && !case.xyb).unwrap();
    let profile = IccProfile::parse(donor.profile().into(), IccLimits::default()).unwrap();
    for name in ["5-2", "16-5", "24-7", "32-8"] {
        let (data, expected) = super::numeric::sample_fixture("floating", name);
        let data = super::profile::replace(&data, &donor.bytes());
        let format = PixelFormat::gray_f32(false, true, ColorSpecification::Icc(profile.clone()));
        let request = GpuOutputRequest::color(format.clone())
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_icc_rendering_intent(IccRenderingIntent::Perceptual)
            .with_white_point_adaptation(WhitePointAdaptation::None);
        let mut session = decoder.open(&data, request).unwrap();
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.format, format);
        assert_eq!(
            planes::read_bytes(&backend, output),
            expected
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
            "{name}"
        );
        assert!(
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn icc_conversion_rejects_unimplemented_intent_and_adaptation_before_submission() {
    use jxl_gpu_protocol::{
        WhitePointAdaptation,
        icc::{IccError, IccRenderingIntent},
    };
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in corpus::cases().filter(|case| !case.xyb) {
        let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap();
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            assert!(
                matches!(decoder.open(&case.bytes(), request.clone().with_icc_rendering_intent(intent)), Err(jxl_wgpu_decode::Error::Icc(IccError::RenderingIntent { intent: selected })) if selected == intent)
            );
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
        assert!(matches!(
            decoder.open(
                &case.bytes(),
                request.with_white_point_adaptation(WhitePointAdaptation::None)
            ),
            Err(jxl_wgpu_decode::Error::UnsupportedOutputFormat(_))
        ));
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn icc_gray_composition_keeps_original_values_through_blends_and_orientation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
    );
    let donor = corpus::cases().find(|case| case.gray && !case.xyb).unwrap();
    let original = jxl_test_support::offline::unhex(
        &std::fs::read_to_string(
            corpus::directory()
                .parent()
                .unwrap()
                .join("composition_gray_clamp.jxl.hex"),
        )
        .unwrap(),
    );
    let data = super::profile::replace(&original, &donor.bytes());
    let profile = IccProfile::parse(donor.profile().into(), IccLimits::default()).unwrap();
    let format = PixelFormat::gray_f32(false, true, ColorSpecification::Icc(profile));
    let request = GpuOutputRequest::color(format.clone())
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
    let mut session = planes::open_fragmented(&decoder, &data, request);
    let mut values = Vec::new();
    while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.format, format);
        let bytes = planes::read_bytes(&backend, output);
        // Native fixture algebra: (253,6) maps to (10,253) under Exif orientation six.
        let index = 253 * 17 + 10;
        values.push(f32::from_le_bytes(
            bytes[index * 4..index * 4 + 4].try_into().unwrap(),
        ));
    }
    assert_eq!(values.len(), 6);
    assert!((values[1] - 350.0 / 255.0).abs() < 2e-7);
    assert!((values[2] - 350.0 * 191.0 / (255.0 * 255.0)).abs() < 3e-7);
    drop(session);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn icc_device_u8_packing_orders_channels_and_associates_alpha_before_quantization() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in corpus::cases().filter(|case| !case.xyb && case.encoding == FrameEncoding::Modular)
    {
        let data = case.bytes();
        let profile = IccProfile::parse(case.profile().into(), IccLimits::default()).unwrap();
        let source_channels = if case.gray { 2 } else { 4 };
        for order in [
            RgbChannelOrder::Rgb,
            RgbChannelOrder::Rgba,
            RgbChannelOrder::Bgr,
            RgbChannelOrder::Bgra,
        ] {
            if case.gray && matches!(order, RgbChannelOrder::Bgr | RgbChannelOrder::Bgra) {
                continue;
            }
            let alpha = matches!(order, RgbChannelOrder::Rgba | RgbChannelOrder::Bgra);
            let channels = source_channels - usize::from(!alpha);
            for planar in [false, true] {
                for policy in [AlphaOutputPolicy::Preserve, AlphaOutputPolicy::Associated] {
                    let color = ColorSpecification::Icc(profile.clone());
                    let format = if case.gray {
                        PixelFormat::gray8(alpha, planar, color)
                    } else {
                        PixelFormat::rgb8(order, planar, color)
                    };
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(policy);
                    let mut session = decoder.open(&data, request).unwrap();
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    let output = &frame.output().outputs[0];
                    assert_eq!(output.layout.format, format);
                    let actual = planes::read_bytes(&backend, output);
                    let mut expected = vec![0u8; output.layout.logical_size as usize];
                    for (pixel_index, pixel) in
                        case.input().chunks_exact(source_channels).enumerate()
                    {
                        let source_alpha = f32::from_bits(pixel[source_channels - 1]);
                        for stored in 0..channels {
                            let channel = if stored < 3
                                && matches!(order, RgbChannelOrder::Bgr | RgbChannelOrder::Bgra)
                            {
                                2 - stored
                            } else {
                                stored
                            };
                            let mut value = f32::from_bits(pixel[channel]);
                            if channel != source_channels - 1
                                && policy == AlphaOutputPolicy::Associated
                            {
                                value *= source_alpha;
                            }
                            let plane = &output.layout.planes[if planar { stored } else { 0 }];
                            let index = plane.offset as usize
                                + (pixel_index / 17) * plane.row_stride as usize
                                + (pixel_index % 17) * if planar { 1 } else { channels }
                                + if planar { 0 } else { stored };
                            expected[index] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                        }
                    }
                    assert_eq!(
                        actual,
                        expected,
                        "{} {order:?} planar={planar} {policy:?}",
                        case.name()
                    );
                    assert!(
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .is_none()
                    );
                }
            }
        }
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}
