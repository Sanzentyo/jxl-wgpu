use super::*;

fn drain(context: &WgpuContext, source: &std::sync::Weak<wgpu::Buffer>) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while source.upgrade().is_some() || context.memory_stats().reserved_bytes != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "cancelled sampling job did not release ownership"
        );
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn sampling_admission_counts_only_coded_pixels_and_retains_retry_and_cancellation() {
    let rig = Rig::new();
    let coded = Extent2d::new(259, 9);
    let samples = ColorSampleFormat::RGB8;
    let source = upload(
        &rig.context,
        coded,
        packed_format(samples, true),
        &words(coded, samples, true, 4),
        0,
    );
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        let config = MixedModeConfig {
            vardct: VarDctConfig {
                alpha: Some(AlphaAssociation::Unassociated),
                color_transform: VarDctColorTransform::Original,
                ..Default::default()
            },
            modular: LosslessModularConfig {
                entropy,
                ..Default::default()
            },
            ..Default::default()
        };
        let encoder = MixedModeEncoder::new(rig.context.clone(), config.clone()).unwrap();
        for encoding in [
            MixedModeFrameEncoding::Modular,
            MixedModeFrameEncoding::VarDct,
        ] {
            let baseline = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(coded.width, coded.height, AnimationHeader::Still)
                        .unwrap(),
                )
                .unwrap()
                .memory_plan(&source, encoding, FrameOptions::default(), true)
                .unwrap();
            for factor in FACTORS {
                let extent = presented(coded, factor);
                let descriptor = ImageSequenceDescriptor::new(
                    extent.width,
                    extent.height,
                    AnimationHeader::Still,
                )
                .unwrap();
                let options = FrameOptions {
                    upsampling: factor,
                    ..Default::default()
                };
                let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
                let plan = session
                    .memory_plan(&source, encoding, options.clone(), true)
                    .unwrap();
                assert_eq!(plan, baseline);
                for bad in [
                    FrameOptions::default(),
                    FrameOptions {
                        upsampling: factor,
                        extra_channel_upsampling: vec![UpsamplingFactor::One],
                        ..Default::default()
                    },
                    FrameOptions {
                        upsampling: factor,
                        crop: Some(FrameCrop::new(0, 0, coded.width, coded.height).unwrap()),
                        ..Default::default()
                    },
                ] {
                    assert!(matches!(
                        session.memory_plan(&source, encoding, bad.clone(), true),
                        Err(EncodeError::InvalidConfiguration(_))
                    ));
                    assert!(matches!(
                        session.submit_last_frame(source.clone(), encoding, bad),
                        Err(EncodeError::InvalidConfiguration(_))
                    ));
                    assert_eq!(session.next_frame_index().get(), 0);
                    assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
                }
                let frame = session
                    .submit_last_frame(source.clone(), encoding, options.clone())
                    .unwrap()
                    .wait()
                    .unwrap();
                session.insert(frame).unwrap();
                let bytes = session.finish_raw().unwrap();
                check_header(&bytes, 0, extent, coded, factor);
                for limit in [plan.owned_bytes_per_job() - 1, plan.owned_bytes_per_job()] {
                    let bounded = WgpuContext::with_memory_budget(
                        Arc::new(rig.context.device().clone()),
                        Arc::new(rig.context.queue().clone()),
                        NonZeroU64::new(limit).unwrap(),
                    )
                    .unwrap();
                    let encoder = MixedModeEncoder::new(bounded.clone(), config.clone()).unwrap();
                    let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
                    let mut owned = source.clone();
                    owned.buffer = Arc::new(source.buffer.as_ref().clone());
                    let weak = Arc::downgrade(&owned.buffer);
                    let result = session.submit_last_frame(owned, encoding, options.clone());
                    if limit < plan.owned_bytes_per_job() {
                        assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
                        assert_eq!(session.next_frame_index().get(), 0);
                    } else {
                        drop(result.unwrap());
                    }
                    drop(session);
                    drain(&bounded, &weak);
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    if limit == plan.owned_bytes_per_job() {
                        let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
                        let result = session
                            .submit_last_frame(source.clone(), encoding, options.clone())
                            .unwrap()
                            .wait()
                            .unwrap();
                        session.insert(result).unwrap();
                        assert_eq!(session.finish_raw().unwrap(), bytes);
                    }
                }
            }
        }
    }
}

#[test]
fn singleton_sources_validate_declared_factors_even_when_extents_are_identical() {
    let rig = Rig::new();
    for shift in 0..=3 {
        let definition = ExtraChannel::new(
            ExtraChannelKind::Depth,
            SamplePrecision::integer(13).unwrap(),
            shift,
            Vec::new(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(
            rig.context.clone(),
            VarDctConfig {
                extra_channels: vec![definition.clone()],
                ..Default::default()
            },
        )
        .unwrap();
        let extent = Extent2d::new(1, 1);
        let scalar = upload(
            &rig.context,
            extent,
            definition.precision().pixel_format(),
            &[719],
            0,
        );
        let source = upload(
            &rig.context,
            extent,
            ColorSampleFormat::RGB8.pixel_format(),
            &[21, 57, 89],
            0,
        )
        .with_extra_channels(vec![scalar])
        .unwrap();
        for color in FACTORS {
            for extra in [
                UpsamplingFactor::One,
                UpsamplingFactor::Two,
                UpsamplingFactor::Four,
                UpsamplingFactor::Eight,
            ] {
                let mut session = encoder
                    .begin_sequence(
                        ImageSequenceDescriptor::new(1, 1, AnimationHeader::Still).unwrap(),
                    )
                    .unwrap();
                let header_bytes = rig.context.memory_stats().reserved_bytes;
                let options = FrameOptions {
                    upsampling: color,
                    extra_channel_upsampling: vec![extra],
                    ..Default::default()
                };
                let result = session.submit_last_frame(source.clone(), options);
                if extra.factor() << shift < color.factor() {
                    assert!(matches!(result, Err(EncodeError::InvalidConfiguration(_))));
                    assert_eq!(session.next_frame_index().get(), 0);
                    assert_eq!(rig.context.memory_stats().reserved_bytes, header_bytes);
                    // Retry with a legal default factor must retain the final-frame slot.
                    let result = session
                        .submit_last_frame(
                            source.clone(),
                            FrameOptions {
                                upsampling: color,
                                ..Default::default()
                            },
                        )
                        .unwrap()
                        .wait()
                        .unwrap();
                    session.insert(result).unwrap();
                } else {
                    session.insert(result.unwrap().wait().unwrap()).unwrap();
                }
                let bytes = session.finish_raw().unwrap();
                check_header(&bytes, 0, extent, extent, color);
                assert_eq!(
                    modular_integer::vardct_extra_words(&bytes, 0)[0].words,
                    [719]
                );
                assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
