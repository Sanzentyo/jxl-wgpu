use super::*;
use jxl_test_support::{fixtures::splines, offline::hex};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, NumericSampleMapping, OrientationPolicy, SpotColorPolicy,
};

fn encoded(name: &str) -> Vec<u8> {
    hex::unhex(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("test-data/splines/features/{name}.jxl.hex")),
        )
        .unwrap(),
    )
}

fn reference(name: &str) -> Vec<f32> {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("test-data/splines/features/{name}.f32.hex")),
    )
    .unwrap()
    .split_whitespace()
    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
    .collect()
}

#[test]
fn splines_compose_with_patches_noise_upsampling_and_lf_in_both_coding_modes() {
    let backend = backend();
    let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
    if let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = color {
        color.transfer = jxl_gpu_formats::TransferFunction::Linear;
    }
    for case in splines::cases() {
        let bytes = encoded(&case.name);
        let expected = reference(&case.name);
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let pixels = inventory.image_header.width as usize * inventory.image_header.height as usize;
        let request =
            GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_spot_color_policy(SpotColorPolicy::Preserve)
                .with_orientation_policy(OrientationPolicy::Keep);
        let mut prior = None;
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let mut session = if limit.is_some() {
                planes::open_fragmented(&decoder, &bytes, request.clone())
            } else {
                decoder.open(&bytes, request.clone()).unwrap()
            };
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            let actual = planes::read(&backend, &frame.output().outputs[0]);
            assert_eq!(actual.len(), pixels * 4);
            let mut worst = 0.0f32;
            for (index, (&actual, &expected)) in
                actual.iter().zip(&expected[..pixels * 4]).enumerate()
            {
                let value = f32::from_bits(actual);
                let error = (value - expected).abs() / (1.0 + expected.abs());
                worst = worst.max(error);
                assert!(
                    value.is_finite()
                        && error
                            <= if index % 4 == 3 {
                                0.000002
                            } else {
                                1.0 / 1024.0
                            },
                    "{} at {index}: {value} != {expected}, error {error}",
                    case.name
                );
            }
            eprintln!(
                "{} window {limit:?}, max normalized error {worst}",
                case.name
            );
            if let Some(prior) = &prior {
                assert_eq!(&actual, prior, "bounded {}", case.name);
            }
            prior = Some(actual);
            assert!(session.next_frame().unwrap().is_none());
            drop((frame, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        for (channel, extra) in inventory.image_header.extra_channels.iter().enumerate() {
            let mapping = match extra.bit_depth {
                jxl_gpu_bitstream::SampleBitDepth::Float { .. } => {
                    NumericSampleMapping::NativeFloat
                }
                jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
                    NumericSampleMapping::NormalizedUnsigned
                }
            };
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(
                    jxl_gpu_formats::SampleKind::Float,
                    32,
                    &[jxl_gpu_formats::Channel::X],
                ),
                mapping,
            )
            .unwrap()
            .with_extra_channel(channel as u32)
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Keep);
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            );
            let mut session = planes::open_fragmented(&decoder, &bytes, request);
            let frame = session.next_frame().unwrap().unwrap();
            let actual = planes::read(&backend, &frame.output().outputs[0]);
            assert_eq!(actual.len(), pixels);
            for (&actual, &expected) in actual
                .iter()
                .zip(&expected[(4 + channel) * pixels..][..pixels])
            {
                let value = f32::from_bits(actual);
                assert!(
                    value.is_finite()
                        && (value - expected).abs() <= 0.000002 * (1.0 + expected.abs()),
                    "{} extra {channel}: {value} != {expected}",
                    case.name
                );
            }
            drop((frame, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn low_frequency_spline_updates_wait_for_features_and_keep_completed_images_immutable() {
    let backend = backend();
    for case in splines::cases().into_iter().filter(|case| {
        case.scenario == splines::Scenario::LowFrequency
            || case.name == "lf_consumer"
            || case.name == "lf_consumer_chain"
    }) {
        let data = encoded(&case.name);
        let info = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let mut source = info.frames.last().unwrap().lf_source_frame;
        let mut levels = Vec::new();
        while let Some(index) = source {
            let frame = info
                .frames
                .iter()
                .find(|frame| frame.frame_index == index)
                .unwrap();
            levels.push(frame.lf_level as u8);
            source = frame.lf_source_frame;
        }
        levels.reverse();
        assert!(!levels.is_empty());
        let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_spot_color_policy(SpotColorPolicy::Preserve)
        .with_orientation_policy(OrientationPolicy::Keep)
        .with_progressive_output(true)
        .with_max_frame_slots(std::num::NonZeroUsize::new(1).unwrap());
        let mut prior = None;
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let mut session = planes::open_fragmented(&decoder, &data, request.clone());
            let mut held = Vec::new();
            let mut outputs = Vec::new();
            let mut actual_levels = Vec::new();
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                if let Some(jxl_wgpu_decode::FrameProgression::LowFrequency { level, .. }) =
                    update.progression()
                {
                    actual_levels.push(level);
                }
                outputs.push(planes::read(&backend, &update.output().outputs[0]));
                held.push(update);
            }
            assert_eq!(actual_levels, levels, "{} LF order", case.name);
            assert!(held.last().unwrap().is_complete());
            for (output, expected) in held.iter().zip(&outputs) {
                assert_eq!(
                    &planes::read(&backend, &output.output().outputs[0]),
                    expected
                );
            }
            if let Some(prior) = &prior {
                assert_eq!(&outputs, prior, "bounded {} LF", case.name);
            }
            prior = Some(outputs);
            drop((held, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            let mut session = planes::open_fragmented(&decoder, &data, request.clone());
            let first = session.next_update().unwrap().unwrap();
            assert!(!first.is_complete());
            let final_image = session.next_frame().unwrap().unwrap();
            assert_eq!(
                planes::read(&backend, &final_image.output().outputs[0]),
                *prior.as_ref().unwrap().last().unwrap()
            );
            drop((first, final_image, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
