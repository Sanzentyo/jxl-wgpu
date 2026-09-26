use super::*;

#[test]
fn sampling_changes_before_mixed_codec_references_crops_and_all_blends() {
    let rig = Rig::new();
    let canvas = Extent2d::new(65, 37);
    let animation = AnimationHeader::Animation {
        ticks_per_second_numerator: 1000.try_into().unwrap(),
        ticks_per_second_denominator: 1.try_into().unwrap(),
        num_loops: 2,
        have_timecodes: true,
    };
    for samples in [
        ColorSampleFormat::integer(ColorChannels::Gray, 13).unwrap(),
        ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap(),
    ] {
        for association in [AlphaAssociation::Associated, AlphaAssociation::Unassociated] {
            let encoder = MixedModeEncoder::new(
                rig.context.clone(),
                MixedModeConfig {
                    vardct: VarDctConfig {
                        sample_format: samples,
                        color_transform: VarDctColorTransform::Original,
                        alpha: Some(association),
                        progressive: progression(),
                        ..Default::default()
                    },
                    modular: LosslessModularConfig {
                        entropy: LosslessModularEntropyCoding::Ans,
                        local_transforms: LosslessModularSqueeze::VerticalThenHorizontal.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            for first_modular in [true, false] {
                for mode in [
                    BlendMode::Replace,
                    BlendMode::Add,
                    BlendMode::Blend,
                    BlendMode::MultiplyAdd,
                    BlendMode::Multiply,
                ] {
                    let mut session = encoder
                        .begin_sequence(
                            ImageSequenceDescriptor::new(canvas.width, canvas.height, animation)
                                .unwrap(),
                        )
                        .unwrap();
                    let mut pending = Vec::new();
                    for (index, factor) in FACTORS.into_iter().enumerate() {
                        let frame_extent = if index == 2 {
                            Extent2d::new(61, 31)
                        } else {
                            canvas
                        };
                        let coded = factor.source_extent(frame_extent);
                        let input = words(coded, samples, true, index as u32);
                        let source = upload(
                            &rig.context,
                            coded,
                            packed_format(samples, true),
                            &input,
                            index,
                        );
                        let encoding = if (index % 2 == 0) == first_modular {
                            MixedModeFrameEncoding::Modular
                        } else {
                            MixedModeFrameEncoding::VarDct
                        };
                        let mut options = if index == 0 {
                            FrameOptions {
                                kind: FrameKind::ReferenceOnly,
                                save_as_reference: ReferenceSlot::new(1).unwrap(),
                                ..Default::default()
                            }
                        } else {
                            FrameOptions {
                                timing: FrameTiming {
                                    duration_ticks: index as u32 + 2,
                                    timecode: Some(501 + index as u32),
                                },
                                crop: (index == 2).then(|| {
                                    FrameCrop::new(-3, 9, frame_extent.width, frame_extent.height)
                                        .unwrap()
                                }),
                                color_blend: FrameBlend {
                                    mode: if index == 1 { BlendMode::Add } else { mode },
                                    source_reference: ReferenceSlot::new(if index == 1 {
                                        1
                                    } else {
                                        3
                                    })
                                    .unwrap(),
                                    ..Default::default()
                                },
                                extra_channel_blends: vec![FrameBlend {
                                    mode: if index == 1 { BlendMode::Replace } else { mode },
                                    source_reference: ReferenceSlot::new(1).unwrap(),
                                    ..Default::default()
                                }],
                                save_as_reference: ReferenceSlot::new(if index == 1 {
                                    3
                                } else {
                                    0
                                })
                                .unwrap(),
                                ..Default::default()
                            }
                        };
                        options.upsampling = factor;
                        if index == 1 {
                            options.extra_channel_upsampling = vec![factor];
                        } // explicit and default agree
                        let memory = session
                            .memory_plan(&source, encoding, options.clone(), index == 2)
                            .unwrap();
                        assert!(memory.owned_bytes_per_job() > 0);
                        pending.push(
                            if index == 2 {
                                session.submit_last_frame(source, encoding, options)
                            } else {
                                session.submit_frame(source, encoding, options)
                            }
                            .unwrap(),
                        );
                    }
                    // Retain out-of-order completed packets, with one finality owner.
                    for job in pending.into_iter().rev() {
                        session.insert(job.wait().unwrap()).unwrap();
                    }
                    let bytes = session.finish_raw().unwrap();
                    // jxl-oxide 0.12's reader does not implement the per-extra blend-field
                    // presence rule. Native headers and both native/Rust pixels independently
                    // cover this mixed full-frame Add/Replace contract without rewriting it.
                    let headers = modular_words::sampling_headers(&bytes);
                    assert_eq!(headers.len(), 3);
                    for (index, (factor, header)) in FACTORS.into_iter().zip(headers).enumerate() {
                        let extent = if index == 2 {
                            Extent2d::new(61, 31)
                        } else {
                            canvas
                        };
                        let coded = factor.source_extent(extent);
                        assert_eq!(header.presented, [extent.width, extent.height]);
                        assert_eq!(header.coded, [coded.width, coded.height]);
                        assert_eq!(header.factor, factor.factor());
                        assert_eq!(header.extras, [factor.factor()]);
                        let modular = (index % 2 == 0) == first_modular;
                        assert_eq!(header.passes, if index == 0 || modular { 1 } else { 3 });
                    }
                    let expected = native(&bytes);
                    let rust = extra_channels::rust_frame_planes(&bytes);
                    let pixels = canvas.width as usize * canvas.height as usize;
                    assert_eq!(rust.len(), 2);
                    assert_eq!(expected.len(), 2 * 5 * pixels);
                    let mut whole = Vec::new();
                    for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
                        let mut session = if fragmented {
                            open_fragmented(decoder, &bytes, color_request())
                        } else {
                            decoder.open(&bytes, color_request()).unwrap()
                        };
                        let mut held = Vec::new();
                        for index in 0..2 {
                            let frame = session.next_frame().unwrap().unwrap();
                            assert_eq!(frame.metadata.timecode, Some(502 + index as u32));
                            assert_eq!(frame.metadata.duration.ticks, index as u32 + 3);
                            let actual = read_bytes(&rig.gpu, &frame.output().outputs[0]);
                            let label = format!(
                                "{samples:?}/{association:?}/{first_modular}/{mode:?}/{index}"
                            );
                            check_color(&actual, &rust[index].0, &format!("Rust {label}"));
                            check_color(
                                &actual,
                                &expected[index * 5 * pixels..][..4 * pixels],
                                &label,
                            );
                            if fragmented {
                                assert_eq!(actual, whole[index]);
                            } else {
                                whole.push(actual);
                            }
                            held.push(frame);
                        }
                        assert!(session.next_frame().unwrap().is_none());
                        drop(session);
                        for (frame, expected) in held.iter().zip(&whole) {
                            assert_eq!(&read_bytes(&rig.gpu, &frame.output().outputs[0]), expected);
                        }
                        drop(held);
                        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    }
                    assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
