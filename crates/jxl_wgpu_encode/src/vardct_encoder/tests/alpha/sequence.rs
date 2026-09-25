use super::*;
use crate::{
    AnimationHeader, BlendMode, FrameBlend, FrameCrop, FrameKind, FrameOptions, FrameTiming,
    ImageSequenceDescriptor, MixedModeConfig, MixedModeEncoder, MixedModeFrameEncoding,
    ReferenceSlot,
};

fn compare(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "component {i}: {a} vs {b}"
        );
    }
}

fn blend(mode: BlendMode, source: u8) -> FrameBlend {
    FrameBlend {
        alpha_channel: 0,
        mode,
        source_reference: ReferenceSlot::new(source).unwrap(),
        clamp: false,
    }
}

#[test]
fn alpha_input_mixed_and_vardct_sequences_blend_across_independent_references() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&gpu);
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
            let config = VarDctConfig {
                alpha: Some(association),
                sample_format: ColorSampleFormat::float(channels, 32, 8).unwrap(),
                color_transform: VarDctColorTransform::Original,
                progressive: progressive::combined(),
                ..Default::default()
            };
            let mixed = MixedModeEncoder::new(
                context.clone(),
                MixedModeConfig {
                    vardct: config.clone(),
                    ..Default::default()
                },
            )
            .unwrap();
            let vardct =
                TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
            for mode in [
                BlendMode::Replace,
                BlendMode::Add,
                BlendMode::Blend,
                BlendMode::MultiplyAdd,
                BlendMode::Multiply,
            ] {
                for mixed_mode in [false, true] {
                    eprintln!("sequence {channels:?}/{association:?}/{mode:?}/mixed={mixed_mode}");
                    let descriptor = ImageSequenceDescriptor::new(
                        9,
                        9,
                        AnimationHeader::Animation {
                            ticks_per_second_numerator: 1000.try_into().unwrap(),
                            ticks_per_second_denominator: 1.try_into().unwrap(),
                            num_loops: 2,
                            have_timecodes: true,
                        },
                    )
                    .unwrap();
                    let mut mixed_session =
                        mixed_mode.then(|| mixed.begin_sequence(descriptor.clone()).unwrap());
                    let mut vardct_session =
                        (!mixed_mode).then(|| vardct.begin_sequence(descriptor).unwrap());
                    for index in 0..3 {
                        let extent = if index == 2 {
                            Extent2d::new(8, 8)
                        } else {
                            Extent2d::new(9, 9)
                        };
                        let n = channels.count() as usize + 1;
                        let mut words = vec![0; extent.area().unwrap() * n];
                        for (i, pixel) in words.chunks_exact_mut(n).enumerate() {
                            let alpha = 0.25 + ((i + index * 3) % 4) as f32 * 0.125;
                            for (c, value) in pixel[..n - 1].iter_mut().enumerate() {
                                let color = 0.125 + ((i + c + index * 7) % 8) as f32 / 32.0;
                                *value = (color
                                    * if association == AlphaAssociation::Associated {
                                        alpha
                                    } else {
                                        1.0
                                    })
                                .to_bits();
                            }
                            pixel[n - 1] = alpha.to_bits();
                        }
                        let options = if index == 0 {
                            FrameOptions {
                                kind: FrameKind::ReferenceOnly,
                                save_as_reference: ReferenceSlot::new(1).unwrap(),
                                ..Default::default()
                            }
                        } else {
                            FrameOptions {
                                timing: FrameTiming {
                                    duration_ticks: index as u32 + 2,
                                    timecode: Some(100 + index as u32),
                                },
                                crop: (index == 2).then(|| FrameCrop::new(-1, 1, 8, 8).unwrap()),
                                color_blend: blend(
                                    if index == 1 { BlendMode::Add } else { mode },
                                    1,
                                ),
                                extra_channel_blends: vec![blend(
                                    if index == 1 { BlendMode::Add } else { mode },
                                    if index == 1 { 1 } else { 3 },
                                )],
                                save_as_reference: ReferenceSlot::new(if index == 1 {
                                    3
                                } else {
                                    0
                                })
                                .unwrap(),
                                ..Default::default()
                            }
                        };
                        let source =
                            upload(&context, extent, &config, &words, Storage::Split, true);
                        if let Some(session) = &mut mixed_session {
                            let encoding = if index == 1 {
                                MixedModeFrameEncoding::Modular
                            } else {
                                MixedModeFrameEncoding::VarDct
                            };
                            let job = if index == 2 {
                                session.submit_last_frame(source, encoding, options)
                            } else {
                                session.submit_frame(source, encoding, options)
                            }
                            .unwrap();
                            session.insert(job.wait().unwrap()).unwrap();
                        } else {
                            let session = vardct_session.as_mut().unwrap();
                            let job = if index == 2 {
                                session.submit_last_frame(source, options)
                            } else {
                                session.submit_frame(source, options)
                            }
                            .unwrap();
                            session.insert(pollster::block_on(job).unwrap()).unwrap();
                        }
                    }
                    let bytes = if let Some(session) = mixed_session {
                        session.finish_raw().unwrap()
                    } else {
                        vardct_session.unwrap().finish_raw().unwrap()
                    };
                    check_header(&bytes, &config);
                    let native = extra_channels::libjxl_output(&bytes, &[])
                        .expect("native alpha sequence oracle");
                    // Rust jxl retains the source association. Compare like-for-like native
                    // output before separately checking the GPU's default unassociated output.
                    let preserved = (association == AlphaAssociation::Associated).then(|| {
                        extra_channels::libjxl_output(&bytes, &["--preserve-alpha"])
                            .expect("native associated alpha oracle")
                    });
                    let rust_reference = preserved.as_deref().unwrap_or(&native);
                    let rust = extra_channels::rust_frame_planes(&bytes);
                    assert_eq!(rust.len(), 2);
                    let per_frame = 9 * 9 * 5;
                    assert_eq!(native.len(), 2 * per_frame);
                    for (i, (color, extras)) in rust.iter().enumerate() {
                        let reference = &rust_reference[i * per_frame..(i + 1) * per_frame];
                        eprintln!("Rust comparison {i}");
                        compare(color, &reference[..9 * 9 * 4]);
                        compare(&extras[0], &reference[9 * 9 * 4..]);
                    }
                    let format = jxl_gpu_formats::PixelFormat::rgb_f32(
                        jxl_gpu_formats::RgbChannelOrder::Rgba,
                        false,
                        vardct_rgb8_format().color_spec,
                    );
                    let mut session = decoder
                        .open(&bytes, GpuOutputRequest::color(format).unwrap())
                        .unwrap();
                    for index in 0..2 {
                        let frame = session.next_frame().unwrap().unwrap();
                        assert_eq!(frame.metadata.duration.ticks, index as u32 + 3);
                        assert_eq!(frame.metadata.timecode, Some(index as u32 + 101));
                        let actual = readback.submit(frame.output()).unwrap().wait().unwrap();
                        eprintln!("GPU comparison {index}");
                        compare(
                            &extra_channels::floats(&actual.frame.outputs[0].bytes),
                            &native[index * per_frame..index * per_frame + 9 * 9 * 4],
                        );
                    }
                    assert!(session.next_frame().unwrap().is_none());
                    drop(session);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn alpha_input_sequence_admission_rejects_wrong_channels_without_consuming_finality() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(8, 8);
    let config = VarDctConfig {
        alpha: Some(AlphaAssociation::Associated),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let mut session = encoder
        .begin_sequence(ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still).unwrap())
        .unwrap();
    let words = input(extent, config.sample_format);
    let no_alpha = VarDctConfig {
        alpha: None,
        ..config.clone()
    };
    let colors: Vec<_> = words
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..3].iter().copied())
        .collect();
    assert!(matches!(
        session.submit_last_frame(
            upload(&context, extent, &no_alpha, &colors, Storage::Packed, false),
            Default::default()
        ),
        Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
    ));
    let bad = FrameOptions {
        extra_channel_blends: vec![blend(BlendMode::Blend, 0); 2],
        ..Default::default()
    };
    assert!(matches!(
        session.submit_last_frame(
            upload(&context, extent, &config, &words, Storage::Packed, false),
            bad
        ),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert_eq!(context.memory_stats().reserved_bytes, 0);
    let job = session
        .submit_last_frame(
            upload(&context, extent, &config, &words, Storage::Packed, false),
            Default::default(),
        )
        .unwrap();
    let artifacts = job.wait().unwrap();
    assert_eq!(artifacts.frame_index.get(), 0);
    session.insert(artifacts).unwrap();
    check_alpha(&session.finish_raw().unwrap(), &words, config.sample_format);
}
