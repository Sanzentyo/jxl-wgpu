use super::*;
use crate::{
    AlphaAssociation, FrameCrop, FrameKind, LosslessModularConfig, LosslessModularEntropyCoding,
    LosslessModularSequenceDescriptor, ReferenceSlot,
};
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::extra_channels::{floats, libjxl_output, rust_frame_planes};
use jxl_test_support::oracles::modular_words::original_frames;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

fn input(
    context: &WgpuContext,
    format: LosslessModularFormat,
    float: bool,
    associated: bool,
    extent: Extent2d,
    frame: usize,
) -> (crate::BufferImageSource, Vec<u32>) {
    let channels = format.channel_count() as usize;
    let mut words = Vec::new();
    for p in 0..extent.area().unwrap() {
        let alpha = [0.25, 0.5, 0.75, 1.0][(p + frame) % 4];
        for c in 0..channels {
            let mut sample = 0.0625 * (1 + (p + c + frame) % 7) as f32;
            if format.has_alpha() && c == channels - 1 {
                sample = alpha;
            } else if associated {
                sample *= alpha;
            }
            words.push(if float {
                sample.to_bits()
            } else {
                (sample * 255.0).round() as u32
            });
        }
    }
    let bytes: Vec<_> = words
        .iter()
        .flat_map(|&word| {
            word.to_le_bytes()
                .into_iter()
                .take(if float { 4 } else { 1 })
        })
        .collect();
    let pixel_format = if float {
        format.float_pixel_format(32)
    } else {
        format.pixel_format(8)
    }
    .unwrap();
    let layout = ImageLayout::packed(extent, pixel_format).unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("layered still source words"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    (
        crate::BufferImageSource::new(buffer, layout).unwrap(),
        words,
    )
}

fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        let bound = if i % 4 == 3 { 4e-7 } else { 3e-6 };
        assert!(
            (a - b).abs() / b.abs().max(1.0) < bound,
            "component {i}: {a} vs {b}"
        );
    }
}

#[test]
fn layered_still_modular_preserves_physical_words_alpha_blends_and_one_presentation() {
    let backend = pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default()))
        .expect("actual GPU required");
    let context = WgpuContext::from_backend(&backend);
    let canvas = Extent2d::new(257, 3);
    for (format, float, associated) in [
        (LosslessModularFormat::Gray, false, false),
        (LosslessModularFormat::Rgba, false, false),
        (LosslessModularFormat::Rgba, false, true),
        (LosslessModularFormat::GrayAlpha, true, false),
        (LosslessModularFormat::Rgba, true, true),
    ] {
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
            )
            .with_alpha_association(if associated {
                AlphaAssociation::Associated
            } else {
                AlphaAssociation::Unassociated
            });
            let pixel_format = if float {
                format.float_pixel_format(32)
            } else {
                format.pixel_format(8)
            }
            .unwrap();
            let desc = LosslessModularSequenceDescriptor::from_pixel_format(
                canvas.width,
                canvas.height,
                &pixel_format,
                AnimationHeader::Still,
            )
            .unwrap();
            assert!(matches!(
                encoder.begin_animation(desc.clone()),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            let mut sequence = encoder.begin_sequence(desc).unwrap();
            let contracts = [
                (BlendMode::Replace, 0, BlendMode::Replace, 0, None, 3),
                (
                    BlendMode::Add,
                    3,
                    BlendMode::Replace,
                    3,
                    Some(FrameCrop::new(-2, 1, 9, 3).unwrap()),
                    0,
                ),
                (
                    if format.has_alpha() {
                        BlendMode::Blend
                    } else {
                        BlendMode::Multiply
                    },
                    0,
                    BlendMode::Blend,
                    3,
                    None,
                    1,
                ),
                (
                    if format.has_alpha() {
                        BlendMode::MultiplyAdd
                    } else {
                        BlendMode::Add
                    },
                    1,
                    BlendMode::Add,
                    0,
                    Some(FrameCrop::new(2, -1, 11, 3).unwrap()),
                    2,
                ),
                (BlendMode::Multiply, 2, BlendMode::Replace, 1, None, 3),
                (
                    BlendMode::Replace,
                    3,
                    BlendMode::Replace,
                    3,
                    Some(FrameCrop::new(270, 0, 7, 2).unwrap()),
                    1,
                ),
                (
                    BlendMode::Replace,
                    1,
                    BlendMode::Multiply,
                    2,
                    Some(FrameCrop::new(-3, 1, 11, 2).unwrap()),
                    0,
                ),
            ];
            let mut expected = Vec::new();
            let mut pending = Vec::new();
            for (index, &(mode, source, alpha_mode, alpha_source, crop, save)) in
                contracts.iter().enumerate()
            {
                let extent = crop.map_or(canvas, |c| Extent2d::new(c.width(), c.height()));
                let (input, words) = input(&context, format, float, associated, extent, index);
                expected.push((extent, words));
                let options = FrameOptions {
                    kind: if index == 0 {
                        FrameKind::ReferenceOnly
                    } else {
                        FrameKind::Regular
                    },
                    crop,
                    color_blend: FrameBlend {
                        mode,
                        source_reference: ReferenceSlot::new(source).unwrap(),
                        clamp: mode == BlendMode::Multiply,
                    },
                    extra_channel_blends: if format.has_alpha() && index != 0 {
                        vec![FrameBlend {
                            mode: alpha_mode,
                            source_reference: ReferenceSlot::new(alpha_source).unwrap(),
                            clamp: false,
                        }]
                    } else {
                        Vec::new()
                    },
                    save_as_reference: ReferenceSlot::new(save).unwrap(),
                    ..Default::default()
                };
                if index == 0 {
                    for timing in [
                        crate::FrameTiming {
                            duration_ticks: 1,
                            timecode: None,
                        },
                        crate::FrameTiming {
                            duration_ticks: 0,
                            timecode: Some(0),
                        },
                    ] {
                        assert!(matches!(
                            sequence.submit_last_frame(
                                input.clone(),
                                FrameOptions {
                                    timing,
                                    ..Default::default()
                                }
                            ),
                            Err(EncodeError::InvalidConfiguration(_))
                        ));
                    }
                    assert!(matches!(
                        sequence.submit_last_frame(input.clone(), options.clone()),
                        Err(EncodeError::InvalidConfiguration(_))
                    ));
                    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
                    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                }
                pending.push(
                    if index + 1 == contracts.len() {
                        sequence.submit_last_frame(input, options)
                    } else {
                        sequence.submit_frame(input, options)
                    }
                    .unwrap(),
                );
            }
            for (i, job) in pending.into_iter().rev().enumerate() {
                sequence
                    .insert(
                        if i % 2 == 0 {
                            job.wait()
                        } else {
                            pollster::block_on(job)
                        }
                        .unwrap(),
                    )
                    .unwrap();
            }
            let encoded = sequence
                .finish_indexed_container(Default::default(), Default::default())
                .unwrap();
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
            let inventory = parsed.codestream_inventory(Default::default()).unwrap();
            assert!(inventory.image_header.animation.is_none());
            assert_eq!(inventory.frames.len(), contracts.len());
            assert_eq!(
                inventory.frames[0].frame_type,
                jxl_gpu_bitstream::FrameType::ReferenceOnly
            );
            assert!(
                inventory
                    .frames
                    .iter()
                    .all(|f| f.duration_ticks == 0 && f.timecode.is_none())
            );
            let index = jxl_gpu_bitstream::FrameIndex::from_container(&parsed, Default::default())
                .unwrap()
                .unwrap();
            assert_eq!(
                (index.tick_numerator(), index.tick_denominator().get()),
                (1, 1)
            );
            assert_eq!(
                index.entries(),
                [jxl_gpu_bitstream::FrameIndexEntry {
                    codestream_offset: inventory.frames[0].header_bits.offset / 8,
                    duration_ticks: 0,
                    frames: 1,
                }]
            );
            let words = original_frames(parsed.codestream());
            assert_eq!(words.len(), expected.len());
            for (actual, (extent, samples)) in words.iter().zip(expected) {
                assert_eq!((actual.width, actual.height), (extent.width, extent.height));
                assert_eq!(actual.planes.len(), format.channel_count() as usize);
                for (c, plane) in actual.planes.iter().enumerate() {
                    assert_eq!(
                        plane,
                        &samples
                            .iter()
                            .skip(c)
                            .step_by(format.channel_count() as usize)
                            .map(|&x| x as i32)
                            .collect::<Vec<_>>()
                    );
                }
            }
            let native = libjxl_output(
                &encoded,
                &["--original", "--preserve-alpha", "--keep-orientation"],
            )
            .expect("required native oracle");
            let pixels = canvas.area().unwrap();
            assert_eq!(native.len(), pixels * (4 + usize::from(format.has_alpha())));
            let native_color = &native[..pixels * 4];
            let rust = rust_frame_planes(&encoded);
            assert_eq!(rust.len(), 1);
            compare(&rust[0].0, native_color);
            let mut whole = None;
            for window in [u64::MAX, 256] {
                let decoder = GpuDecoder::new(
                    WgpuDecodeEngine::new(backend.clone())
                        .unwrap()
                        .with_stream_window_limit(NonZeroU64::new(window).unwrap()),
                );
                let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
                let mut session = if window == u64::MAX {
                    decoder.open(&encoded, request.clone()).unwrap()
                } else {
                    open_fragmented(&decoder, &encoded, request.clone())
                };
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    (
                        frame.metadata.index,
                        frame.metadata.duration.ticks,
                        frame.metadata.timecode,
                        frame.metadata.is_last
                    ),
                    (0, 0, None, true)
                );
                assert!(session.next_frame().unwrap().is_none());
                drop(session);
                let output = read_bytes(&backend, &frame.output().outputs[0]);
                compare(&floats(&output), native_color);
                compare(&floats(&output), &rust[0].0);
                drop(frame);
                let mut seek = decoder
                    .open_seek(&encoded, request, 0, Default::default(), Default::default())
                    .unwrap();
                let frame = pollster::block_on(seek.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert!(seek.next_frame().unwrap().is_none());
                drop(seek);
                assert_eq!(read_bytes(&backend, &frame.output().outputs[0]), output);
                drop(frame);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
                if let Some(whole) = &whole {
                    assert_eq!(whole, &output);
                } else {
                    whole = Some(output);
                }
            }
        }
    }
}
