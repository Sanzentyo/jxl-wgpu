use super::*;
use std::num::{NonZeroU32, NonZeroUsize};

use jxl_wgpu_encode::{
    AnimationHeader, BlendMode, EncodeError, FrameBlend, FrameCrop, FrameOptions, FrameTiming,
    LosslessModularAnimationDescriptor, ReferenceSlot,
};

fn solid(context: &WgpuContext, extent: Extent2d, bits: u8, pixel: [f32; 4]) -> BufferImageSource {
    let words = pixel.map(|value| {
        let word = value.to_bits();
        if bits == 32 {
            return word;
        }
        // These dyadic test inputs are exact, normal binary16 values.
        let exponent = (word >> 23) & 255;
        assert!((113..=142).contains(&exponent) && word & 0x1fff == 0);
        ((word >> 16) & 0x8000) | ((exponent - 112) << 10) | ((word >> 13) & 1023)
    });
    let bytes: Vec<_> = (0..extent.width * extent.height)
        .flat_map(|_| words)
        .flat_map(|word| word.to_le_bytes().into_iter().take(usize::from(bits / 8)))
        .collect();
    let row_bytes = u64::from(extent.width) * 4 * u64::from(bits / 8);
    let layout = ImageLayout::from_planes(
        extent,
        LosslessModularFormat::Rgba
            .float_pixel_format(bits)
            .unwrap(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: row_bytes,
            sample_extent: extent,
            row_bytes,
        }],
    )
    .unwrap();
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exact dyadic floating animation input"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

#[test]
fn floating_crops_and_reference_arithmetic_match_both_cpu_decoders() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let encoder = LosslessModularEncoder::new(context.clone());
    let decoders = decoders(&backend);
    let extent = Extent2d::new(17, 3);
    let slot1 = ReferenceSlot::new(1).unwrap();
    let slot2 = ReferenceSlot::new(2).unwrap();
    let first = [0.25f32, -0.5, 1.5, 0.5];
    let added = [0.5f32, 0.25, -0.5, 0.25];
    let multiplied = [0.5f32, -0.5, 2.0, 0.75];
    let mut expected = vec![vec![first; 51]];
    expected.push(
        (0..51)
            .map(|index| {
                if index / 17 == 1 && (3..12).contains(&(index % 17)) {
                    [
                        first[0] + added[0],
                        first[1] + added[1],
                        first[2] + added[2],
                        added[3],
                    ]
                } else {
                    first
                }
            })
            .collect(),
    );
    expected.push(
        expected[1]
            .iter()
            .map(|pixel| {
                [
                    pixel[0] * multiplied[0],
                    pixel[1] * multiplied[1],
                    pixel[2] * multiplied[2],
                    multiplied[3],
                ]
            })
            .collect(),
    );
    expected.push(vec![
        [first[0], first[1], first[2], multiplied[3] + first[3]];
        51
    ]);
    let expected: Vec<Vec<u32>> = expected
        .into_iter()
        .map(|frame| frame.into_iter().flatten().map(f32::to_bits).collect())
        .collect();
    for bits in [16, 32] {
        let descriptor = LosslessModularAnimationDescriptor::new_float(
            extent.width,
            extent.height,
            LosslessModularFormat::Rgba,
            bits,
            AnimationHeader::Animation {
                ticks_per_second_numerator: NonZeroU32::new(24).unwrap(),
                ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                num_loops: 0,
                have_timecodes: false,
            },
        )
        .unwrap();
        let mut assembly = encoder.begin_animation(descriptor).unwrap();
        let first = assembly
            .submit_frame(
                solid(&context, extent, bits, first),
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: 1,
                        timecode: None,
                    },
                    save_as_reference: slot1,
                    ..Default::default()
                },
            )
            .unwrap()
            .wait()
            .unwrap();
        assembly.insert(first).unwrap();
        let second = assembly
            .submit_frame(
                solid(&context, Extent2d::new(9, 1), bits, added),
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: 2,
                        timecode: None,
                    },
                    crop: Some(FrameCrop::new(3, 1, 9, 1).unwrap()),
                    color_blend: FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: slot1,
                        ..Default::default()
                    },
                    extra_channel_blends: vec![FrameBlend {
                        source_reference: slot1,
                        ..Default::default()
                    }],
                    save_as_reference: slot2,
                    ..Default::default()
                },
            )
            .unwrap();
        assembly
            .insert(pollster::block_on(second).unwrap())
            .unwrap();
        let third = assembly
            .submit_frame(
                solid(&context, extent, bits, multiplied),
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: 3,
                        timecode: None,
                    },
                    color_blend: FrameBlend {
                        mode: BlendMode::Multiply,
                        source_reference: slot2,
                        ..Default::default()
                    },
                    extra_channel_blends: vec![FrameBlend {
                        source_reference: slot2,
                        ..Default::default()
                    }],
                    save_as_reference: slot1,
                    ..Default::default()
                },
            )
            .unwrap()
            .wait()
            .unwrap();
        assembly.insert(third).unwrap();
        let fourth = assembly
            .submit_last_frame(
                solid(&context, extent, bits, [0.25, -0.5, 1.5, 0.5]),
                FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: 4,
                        timecode: None,
                    },
                    extra_channel_blends: vec![FrameBlend {
                        mode: BlendMode::Add,
                        source_reference: slot1,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )
            .unwrap()
            .wait()
            .unwrap();
        assembly.insert(fourth).unwrap();
        let encoded = assembly.finish_container().unwrap();
        let native = extra_channels::libjxl_output(&encoded, &["--original", "--preserve-alpha"])
            .expect("required libjxl 0.12.0 floating composition oracle");
        assert_eq!(native.len(), expected.len() * 51 * 5);
        let rust = extra_channels::rust_frame_planes(&encoded);
        assert_eq!(rust.len(), expected.len());
        for (index, expected) in expected.iter().enumerate() {
            let native = &native[index * 51 * 5..(index + 1) * 51 * 5];
            assert_eq!(
                native[..51 * 4]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                *expected
            );
            assert_eq!(
                rust[index]
                    .0
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                *expected
            );
            let alpha: Vec<_> = expected.iter().skip(3).step_by(4).copied().collect();
            assert_eq!(
                native[51 * 4..]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                alpha
            );
            assert_eq!(
                rust[index].1[0]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                alpha
            );
        }
        for (decoder, fragmented) in [(&decoders[0], false), (&decoders[1], true)] {
            for (selected, request) in requests(LosslessModularFormat::Rgba) {
                let mut session = if fragmented {
                    open_fragmented(decoder, &encoded, request)
                } else {
                    decoder.open(&encoded, request).unwrap()
                };
                for expected in &expected {
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    let actual = read(&backend, &frame.output().outputs[0]);
                    let expected =
                        expected_output(expected, LosslessModularFormat::Rgba, 32, selected);
                    assert_eq!(
                        actual, expected,
                        "GPU floating composition {bits}/{fragmented}/{selected:?}"
                    );
                }
                assert!(
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .is_none()
                );
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn floating_animation_preserves_replace_words_timing_and_retained_outputs() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let encoder = LosslessModularEncoder::new(context.clone());
    let decoders = decoders(&backend);
    let extent = Extent2d::new(257, 3);
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        for bits in [16, 32] {
            let descriptor = LosslessModularAnimationDescriptor::new_float(
                extent.width,
                extent.height,
                format,
                bits,
                AnimationHeader::Animation {
                    ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
                    ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                    num_loops: 2,
                    have_timecodes: true,
                },
            )
            .unwrap();
            assert_eq!(descriptor.sample_bit_depth(), depth(bits));
            let mut assembly = encoder.begin_animation(descriptor).unwrap();
            if bits == 16 {
                // Identical storage width does not permit changing the stream's numeric type.
                let (mut integer, _) = source(&context, extent, format, bits, 0);
                integer.layout.format = format.pixel_format(bits).unwrap();
                assert!(matches!(
                    assembly.submit_frame(integer, FrameOptions::default()),
                    Err(EncodeError::InvalidConfiguration(_))
                ));
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            }
            let mut expected = Vec::new();
            let mut pending = Vec::new();
            for index in 0..3 {
                let (source, samples) = source(&context, extent, format, bits, index);
                expected.push(samples);
                let options = FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: index + 2,
                        timecode: Some(100 + index),
                    },
                    ..Default::default()
                };
                pending.push(if index == 2 {
                    assembly.submit_last_frame(source, options).unwrap()
                } else {
                    assembly.submit_frame(source, options).unwrap()
                });
            }
            for (index, job) in pending.into_iter().rev().enumerate() {
                let frame = if index == 1 {
                    job.wait().unwrap()
                } else {
                    pollster::block_on(job).unwrap()
                };
                assembly.insert(frame).unwrap();
            }
            let encoded = assembly.finish_container().unwrap();
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            let references: Vec<_> = expected.iter().map(Vec::as_slice).collect();
            check_frame_oracles(&encoded, &references, format, bits);
            for (decoder, fragmented) in [(&decoders[0], false), (&decoders[1], true)] {
                for (channel, request) in requests(format) {
                    let request = request.with_max_frame_slots(NonZeroUsize::new(4).unwrap());
                    let mut session = if fragmented {
                        open_fragmented(decoder, &encoded, request)
                    } else {
                        decoder.open(&encoded, request).unwrap()
                    };
                    assert_eq!(session.metadata().loop_count, Some(2));
                    let mut retained = Vec::new();
                    let mut ticks = 0;
                    for index in 0..3 {
                        let frame = pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap();
                        assert_eq!(frame.metadata.index, index);
                        assert_eq!(frame.metadata.duration.ticks, index as u32 + 2);
                        assert_eq!(frame.metadata.timecode, Some(100 + index as u32));
                        assert_eq!(frame.metadata.presentation_ticks, ticks);
                        assert_eq!(frame.metadata.is_last, index == 2);
                        ticks += index as u64 + 2;
                        retained.push(frame);
                    }
                    assert!(
                        pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .is_none()
                    );
                    drop(session);
                    for (frame, expected) in retained.iter().zip(&expected) {
                        let actual = read(&backend, &frame.output().outputs[0]);
                        let expected = expected_output(expected, format, bits, channel);
                        assert_eq!(
                            actual, expected,
                            "retained {format:?}/{bits}/{channel:?}/{fragmented}"
                        );
                    }
                    drop(retained);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
