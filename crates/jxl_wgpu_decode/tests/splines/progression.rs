use super::*;
use jxl_test_support::{fixtures::splines, offline::hex};
use jxl_wgpu_decode::{
    AlphaOutputPolicy, NumericSampleMapping, OrientationPolicy, SpotColorPolicy,
};

#[test]
fn spline_and_patch_passes_match_native_prefixes_and_keep_prior_images_immutable() {
    let backend = backend();
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/splines/progressive");
    for &(name, _) in splines::PROGRESSIVE_SOURCES {
        let data =
            hex::unhex(&std::fs::read_to_string(root.join(format!("{name}.jxl.hex"))).unwrap());
        let info = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let main = info.frames.last().unwrap();
        let expected: Vec<_> = std::fs::read_to_string(root.join(format!("{name}.f32.hex")))
            .unwrap()
            .split_whitespace()
            .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
            .collect();
        let pixels = info.image_header.width as usize * info.image_header.height as usize;
        let frame_words = pixels * (4 + info.image_header.extra_channels.len());
        assert_eq!(expected.len(), frame_words * (main.num_passes as usize + 1));
        let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
        if info.image_header.xyb_encoded
            && let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = color
        {
            color.transfer = jxl_gpu_formats::TransferFunction::Linear;
        }
        let request =
            GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_spot_color_policy(SpotColorPolicy::Preserve)
                .with_orientation_policy(OrientationPolicy::Keep)
                .with_progressive_output(true)
                .with_max_frame_slots(std::num::NonZeroUsize::new(1).unwrap());
        let mut requests = vec![(request, 0, pixels * 4, 1.0 / 1024.0)];
        for (channel, extra) in info.image_header.extra_channels.iter().enumerate() {
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
            .with_orientation_policy(OrientationPolicy::Keep)
            .with_progressive_output(true)
            .with_max_frame_slots(std::num::NonZeroUsize::new(1).unwrap());
            requests.push((request, (4 + channel) * pixels, pixels, 0.000002));
        }
        for (request, offset, output_words, tolerance) in requests {
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
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let index = held.len();
                    if let Some(progression) = update.progression() {
                        assert_eq!(progression.physical_frame_index(), main.frame_index);
                        assert_eq!(progression.completed_passes(), Some(index as u8));
                    } else {
                        assert_eq!(index, main.num_passes as usize);
                    }
                    let actual = planes::read(&backend, &update.output().outputs[0]);
                    assert_eq!(actual.len(), output_words);
                    let reference = &expected[index * frame_words + offset..][..output_words];
                    let mut worst = 0.0f32;
                    for (&word, &expected) in actual.iter().zip(reference) {
                        let value = f32::from_bits(word);
                        let error = (value - expected).abs() / (1.0 + expected.abs());
                        worst = worst.max(error);
                        assert!(
                            value.is_finite() && error <= tolerance,
                            "{name} offset {offset} pass {index}: {value} != {expected}"
                        );
                    }
                    eprintln!(
                        "{name} offset {offset} pass {index}, window {limit:?}, max normalized error {worst}"
                    );
                    outputs.push(actual);
                    held.push(update);
                }
                assert_eq!(held.len(), main.num_passes as usize + 1);
                for (update, expected) in held.iter().zip(&outputs) {
                    assert_eq!(
                        &planes::read(&backend, &update.output().outputs[0]),
                        expected
                    );
                }
                if let Some(prior) = &prior {
                    assert_eq!(&outputs, prior, "bounded {name}");
                }
                prior = Some(outputs);
                drop((held, session));
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                // Switching to final-only consumption drains pending features without changing the result.
                let mut session = planes::open_fragmented(&decoder, &data, request.clone());
                let first = session.next_update().unwrap().unwrap();
                assert!(!first.is_complete());
                let last = session.next_frame().unwrap().unwrap();
                assert_eq!(
                    planes::read(&backend, &last.output().outputs[0]),
                    *prior.as_ref().unwrap().last().unwrap()
                );
                drop((first, last, session));
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
