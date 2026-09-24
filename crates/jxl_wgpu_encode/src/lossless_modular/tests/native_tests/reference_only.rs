use super::*;
use crate::{
    FrameKind, FrameTiming, LosslessModularConfig, LosslessModularEntropyCoding, ReferenceSlot,
};
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::{modular_words::original_frames, progressive::native_updates};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

fn pixels(width: u32, height: u32, channels: usize, seed: u8) -> Vec<u8> {
    (0..width * height)
        .flat_map(|p| {
            (0..channels).map(move |c| {
                if c == 3 {
                    255
                } else {
                    seed + (p % 7) as u8 + c as u8 * 11
                }
            })
        })
        .collect()
}

fn source(
    context: &WgpuContext,
    width: u32,
    height: u32,
    channels: usize,
    pixels: Vec<u8>,
) -> crate::BufferImageSource {
    if channels == 1 {
        packed_gray8_source(context, width, height, pixels)
    } else {
        packed_rgba8_source(context, width, height, pixels)
    }
}

#[test]
fn reference_only_modular_words_composition_and_streaming_keep_all_slots() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required");
    let context = WgpuContext::from_backend(&backend);
    for format in [LosslessModularFormat::Gray, LosslessModularFormat::Rgba] {
        let channels = format.channel_count() as usize;
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    entropy,
                    ..Default::default()
                },
            );
            let animation = AnimationHeader::Animation {
                ticks_per_second_numerator: NonZeroU32::new(24_000).unwrap(),
                ticks_per_second_denominator: NonZeroU32::new(1001).unwrap(),
                num_loops: 2,
                have_timecodes: true,
            };
            let mut session = encoder
                .begin_animation(
                    LosslessModularAnimationDescriptor::new(257, 3, format, 8, animation).unwrap(),
                )
                .unwrap();
            let mut inputs = Vec::new();
            let mut submissions = Vec::new();
            // A bounded pre-transform source is validated even though its slot is later replaced.
            let tiny = pixels(3, 2, channels, 3);
            submissions.push(
                session
                    .submit_frame(
                        source(&context, 3, 2, channels, tiny.clone()),
                        FrameOptions {
                            kind: FrameKind::ReferenceOnly,
                            crop: Some(crate::FrameCrop::new(0, 0, 3, 2).unwrap()),
                            save_as_reference: ReferenceSlot::new(3).unwrap(),
                            save_before_color_transform: true,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
            );
            inputs.push((3, 2, tiny));
            let mut expected = Vec::new();
            for slot in 0..4 {
                let reference = pixels(257, 3, channels, 20 + slot);
                let input = source(&context, 257, 3, channels, reference.clone());
                assert_eq!(
                    encoder.memory_plan(&input).unwrap().streaming,
                    entropy == LosslessModularEntropyCoding::Ans
                );
                submissions.push(
                    session
                        .submit_frame(
                            input,
                            FrameOptions {
                                kind: FrameKind::ReferenceOnly,
                                save_as_reference: ReferenceSlot::new(slot).unwrap(),
                                ..Default::default()
                            },
                        )
                        .unwrap(),
                );
                inputs.push((257, 3, reference.clone()));
                let foreground = pixels(257, 3, channels, 1);
                expected.push(
                    reference
                        .iter()
                        .zip(&foreground)
                        .enumerate()
                        .map(|(i, (a, b))| {
                            if channels == 4 && i % 4 == 3 {
                                255
                            } else {
                                a + b
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                let input = source(&context, 257, 3, channels, foreground.clone());
                let options = FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: u32::from(slot) + 2,
                        timecode: Some(100 + u32::from(slot)),
                    },
                    color_blend: FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: ReferenceSlot::new(slot).unwrap(),
                        clamp: false,
                    },
                    ..Default::default()
                };
                submissions.push(
                    if slot == 3 {
                        session.submit_last_frame(input, options)
                    } else {
                        session.submit_frame(input, options)
                    }
                    .unwrap(),
                );
                inputs.push((257, 3, foreground));
            }
            for submission in submissions.into_iter().rev() {
                session.insert(submission.wait().unwrap()).unwrap();
            }
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            let raw = session.finish_raw().unwrap();
            let inventory = jxl_gpu_bitstream::parse(&raw, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap();
            assert_eq!(inventory.frames.len(), 9);
            assert_eq!(
                inventory.frames[0].frame_type,
                jxl_gpu_bitstream::FrameType::ReferenceOnly
            );
            assert!(inventory.frames[0].save_before_color_transform);
            for slot in 0..4 {
                let frame = &inventory.frames[1 + slot * 2];
                assert_eq!(
                    frame.frame_type,
                    jxl_gpu_bitstream::FrameType::ReferenceOnly
                );
                assert_eq!(
                    (frame.duration_ticks, frame.timecode, frame.num_passes),
                    (0, None, 1)
                );
                assert_eq!(frame.save_as_reference, slot as u32);
            }
            let original = original_frames(&raw);
            assert_eq!(original.len(), inputs.len());
            for (frame, (width, height, samples)) in original.iter().zip(inputs) {
                assert_eq!((frame.width, frame.height), (width, height));
                assert_eq!(frame.planes.len(), channels);
                for (channel, words) in frame.planes.iter().enumerate() {
                    assert_eq!(
                        words,
                        &samples
                            .iter()
                            .skip(channel)
                            .step_by(channels)
                            .map(|&v| i32::from(v))
                            .collect::<Vec<_>>()
                    );
                }
            }
            let (_, rust) = decode_animation8(&raw, format).unwrap();
            assert_eq!(rust.len(), 4);
            for (frame, expected) in rust.iter().zip(&expected) {
                assert_eq!(&frame.1, expected);
            }
            let native: Vec<_> = native_updates(&raw, false)
                .expect("required native oracle")
                .into_iter()
                .filter(|u| u.complete)
                .collect();
            assert_eq!(native.len(), 4);
            let expected: Vec<Vec<u8>> = expected
                .into_iter()
                .map(|samples| {
                    if channels == 1 {
                        samples.into_iter().flat_map(|p| [p, p, p, 255]).collect()
                    } else {
                        samples
                    }
                })
                .collect();
            for (i, frame) in native.iter().enumerate() {
                assert_eq!(
                    (frame.duration, frame.timecode),
                    (i as u32 + 2, i as u32 + 100)
                );
                let words: Vec<_> = frame
                    .pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| (f32::from_le_bytes(*v) * 255.0).round() as u8)
                    .collect();
                assert_eq!(words, expected[i]);
            }
            for window in [u64::MAX, 256] {
                let decoder = GpuDecoder::new(
                    WgpuDecodeEngine::new(backend.clone())
                        .unwrap()
                        .with_stream_window_limit(NonZeroU64::new(window).unwrap()),
                );
                let request = GpuOutputRequest::color(PixelFormat::rgb8(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap();
                let mut decode = if window == u64::MAX {
                    decoder.open(&raw, request).unwrap()
                } else {
                    open_fragmented(&decoder, &raw, request)
                };
                for (index, pixels) in expected.iter().enumerate() {
                    let frame = pollster::block_on(decode.next_frame_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(frame.metadata.index, index);
                    assert_eq!(frame.metadata.timecode, Some(100 + index as u32));
                    assert_eq!(read_bytes(&backend, &frame.output().outputs[0]), *pixels);
                }
                assert!(decode.next_frame().unwrap().is_none());
                drop(decode);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
