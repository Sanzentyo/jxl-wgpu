use super::*;
use crate::{
    AnimationHeader, FrameIndex, FrameOptions, FrameTiming, ImageSequenceDescriptor,
    MixedModeConfig, MixedModeEncoder, MixedModeFrameEncoding, VarDctTransformSelection,
};

#[test]
fn icc_sequences_keep_one_profile_across_codecs_intents_and_out_of_order_completion() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = Pixels::new(&gpu);
    let native_profiles = IccProfileOracle::compile();
    for gray in [false, true] {
        for intent in 0_u32..4 {
            let mut bytes = profile(gray).bytes().to_vec();
            bytes[64..68].copy_from_slice(&intent.to_be_bytes());
            bytes[84..100].fill(0);
            let profile = IccProfile::parse(bytes.into(), Default::default()).unwrap();
            for (topology, extent) in [
                (
                    VarDctTransformSelection::Single(VarDctStrategy::Dct8),
                    Extent2d::new(8, 8),
                ),
                (
                    VarDctTransformSelection::Map(mixed::packed_map(25, 17, false)),
                    Extent2d::new(25, 17),
                ),
                (VarDctTransformSelection::TiledDct8, Extent2d::new(259, 3)),
            ] {
                for (mixed, transform) in [
                    (true, VarDctColorTransform::Original),
                    (false, VarDctColorTransform::Original),
                    (false, VarDctColorTransform::Xyb),
                ] {
                    let mut config = config(&profile, transform);
                    config.progressive = progressive::combined();
                    if intent % 2 == 1 {
                        config.sample_format =
                            ColorSampleFormat::integer(config.sample_format.channels(), 12)
                                .unwrap();
                    }
                    let descriptor = ImageSequenceDescriptor::new(
                        extent.width,
                        extent.height,
                        AnimationHeader::Animation {
                            ticks_per_second_numerator: 60.try_into().unwrap(),
                            ticks_per_second_denominator: 1.try_into().unwrap(),
                            num_loops: 2,
                            have_timecodes: false,
                        },
                    )
                    .unwrap();
                    let header_bytes = VarDctColorPlan::new(&config)
                        .unwrap()
                        .image_header(&descriptor)
                        .unwrap()
                        .icc_storage_bytes;
                    let sources: Vec<_> = (0..3)
                        .map(|i| {
                            let mut input = words(extent, config.sample_format.channels());
                            input.rotate_left(i * 3);
                            if intent % 2 == 1 {
                                input.iter_mut().for_each(|p| {
                                    *p = p.map(|v| (f32::from_bits(v) * 4095.0).round() as u32)
                                });
                            }
                            let source = upload(&context, extent, &config, &input, i % 2 == 1);
                            let mut wrong = source.clone();
                            let mut altered = profile.bytes().to_vec();
                            altered[80..84].copy_from_slice(b"test");
                            wrong.layout.format.color_spec = ColorSpecification::Icc(
                                IccProfile::parse(altered.into(), Default::default()).unwrap(),
                            );
                            let mut options = FrameOptions {
                                timing: FrameTiming {
                                    duration_ticks: 7 + i as u32,
                                    timecode: None,
                                },
                                ..Default::default()
                            };
                            if !gray && i > 0 && transform == VarDctColorTransform::Original {
                                options.color_blend = crate::FrameBlend {
                                    mode: crate::BlendMode::Multiply,
                                    source_reference: crate::ReferenceSlot::new(1).unwrap(),
                                    clamp: true,
                                };
                                options.crop = Some(
                                    crate::FrameCrop::new(-1, 2, extent.width, extent.height)
                                        .unwrap(),
                                );
                            }
                            options.save_as_reference = crate::ReferenceSlot::new(u8::from(
                                i != 2 && transform == VarDctColorTransform::Original,
                            ))
                            .unwrap();
                            (source, wrong, options)
                        })
                        .collect();
                    let bytes = if mixed {
                        let encoder = MixedModeEncoder::new(
                            context.clone(),
                            MixedModeConfig {
                                vardct: config.clone(),
                                vardct_transform: topology.clone(),
                                ..Default::default()
                            },
                        )
                        .unwrap();
                        let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
                        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
                        let mut jobs = Vec::new();
                        for (i, (source, wrong, options)) in sources.into_iter().enumerate() {
                            let mode = if i == 1 {
                                MixedModeFrameEncoding::VarDct
                            } else {
                                MixedModeFrameEncoding::Modular
                            };
                            let reserved = context.memory_stats().reserved_bytes;
                            assert!(
                                session
                                    .submit_last_frame(wrong, mode, options.clone())
                                    .is_err()
                            );
                            assert_eq!(session.next_frame_index(), FrameIndex::new(i as u32));
                            assert_eq!(context.memory_stats().reserved_bytes, reserved);
                            jobs.push(
                                if i == 2 {
                                    session.submit_last_frame(source, mode, options)
                                } else {
                                    session.submit_frame(source, mode, options)
                                }
                                .unwrap(),
                            );
                        }
                        for job in jobs.into_iter().rev() {
                            session.insert(job.wait().unwrap()).unwrap();
                        }
                        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
                        session
                            .finish_indexed_container(Default::default(), Default::default())
                            .unwrap()
                    } else {
                        let mut session = match &topology {
                            VarDctTransformSelection::Single(strategy) => {
                                VarDctEncoder::new_with_config(
                                    context.clone(),
                                    *strategy,
                                    config.clone(),
                                )
                                .unwrap()
                                .begin_sequence(descriptor.clone())
                                .unwrap()
                            }
                            VarDctTransformSelection::Map(map) => {
                                VarDctEncoder::new_with_strategy_map(
                                    context.clone(),
                                    map.clone(),
                                    config.clone(),
                                )
                                .unwrap()
                                .begin_sequence(descriptor.clone())
                                .unwrap()
                            }
                            VarDctTransformSelection::TiledDct8 => {
                                TiledVarDctEncoder::new_with_config(context.clone(), config.clone())
                                    .unwrap()
                                    .begin_sequence(descriptor.clone())
                                    .unwrap()
                            }
                        };
                        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
                        let mut jobs = Vec::new();
                        for (i, (source, wrong, options)) in sources.into_iter().enumerate() {
                            let reserved = context.memory_stats().reserved_bytes;
                            if transform == VarDctColorTransform::Xyb {
                                for (slot, duration, kind) in [
                                    (0, 0, crate::FrameKind::Regular),
                                    (1, 7, crate::FrameKind::Regular),
                                    (3, 0, crate::FrameKind::Regular),
                                    (2, 0, crate::FrameKind::ReferenceOnly),
                                ] {
                                    let forbidden = FrameOptions {
                                        kind,
                                        timing: FrameTiming {
                                            duration_ticks: duration,
                                            timecode: None,
                                        },
                                        save_as_reference: crate::ReferenceSlot::new(slot).unwrap(),
                                        ..Default::default()
                                    };
                                    assert!(matches!(
                                        session.submit_frame(source.clone(), forbidden),
                                        Err(EncodeError::InvalidConfiguration(
                                            "XYB with an embedded ICC profile cannot save post-color-transform references"
                                        ))
                                    ));
                                    assert_eq!(
                                        session.next_frame_index(),
                                        FrameIndex::new(i as u32)
                                    );
                                    assert_eq!(context.memory_stats().reserved_bytes, reserved);
                                }
                            }
                            assert!(session.submit_last_frame(wrong, options.clone()).is_err());
                            assert_eq!(session.next_frame_index(), FrameIndex::new(i as u32));
                            assert_eq!(context.memory_stats().reserved_bytes, reserved);
                            jobs.push(
                                if i == 2 {
                                    session.submit_last_frame(source, options)
                                } else {
                                    session.submit_frame(source, options)
                                }
                                .unwrap(),
                            );
                        }
                        for job in jobs.into_iter().rev() {
                            session.insert(job.wait().unwrap()).unwrap();
                        }
                        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
                        session
                            .finish_indexed_container(Default::default(), Default::default())
                            .unwrap()
                    };
                    assert_eq!(context.memory_stats().reserved_bytes, 0);
                    assert_eq!(
                        native_profiles.read(&bytes).profile,
                        profile.bytes().as_ref()
                    );
                    eprintln!(
                        "ICC sequence gray={gray} intent={intent} mixed={mixed} transform={transform:?} extent={extent:?}"
                    );
                    pixels.check_frames(&bytes, &config, 3);
                }
            }
        }
    }
}
