use super::*;

#[test]
fn invalid_preview_dimensions_are_typed_before_any_gpu_input() {
    for (width, height) in [(0, 1), (1, 0), (4097, 1), (1, 4097), (u32::MAX, u32::MAX)] {
        assert!(matches!(
            PreviewSize::new(width, height),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
    assert_eq!(
        PreviewSize::new(4096, 4096).unwrap().extent(),
        Extent2d::new(4096, 4096)
    );
}

fn descriptor() -> ImageSequenceDescriptor {
    ImageSequenceDescriptor::new(17, 9, AnimationHeader::Still)
        .unwrap()
        .with_preview(PreviewSize::new(17, 9).unwrap())
}

fn config() -> MixedModeConfig {
    MixedModeConfig {
        vardct: VarDctConfig {
            color_transform: VarDctColorTransform::Original,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn retire(context: &WgpuContext, source: &std::sync::Weak<wgpu::Buffer>) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while source.upgrade().is_some() || context.memory_stats().reserved_bytes != 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "preview cancellation retained resources"
        );
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn preview_admission_is_retryable_and_retained_storage_shares_the_job_budget() {
    let rig = Rig::new();
    let input = source(
        &rig,
        Extent2d::new(17, 9),
        ColorSampleFormat::RGB8,
        false,
        1,
    );
    let encoder = MixedModeEncoder::new(rig.context.clone(), config()).unwrap();
    for encoding in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let plan = encoder
            .begin_sequence(descriptor())
            .unwrap()
            .preview_memory_plan(&input, encoding, FrameOptions::default())
            .unwrap();
        for limit in [plan.owned_bytes_per_job() - 1, plan.owned_bytes_per_job()] {
            let context = WgpuContext::with_memory_budget(
                Arc::new(rig.context.device().clone()),
                Arc::new(rig.context.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let bounded = MixedModeEncoder::new(context.clone(), config()).unwrap();
            let mut session = bounded.begin_sequence(descriptor()).unwrap();
            let mut owned = input.clone();
            owned.buffer = Arc::new(input.buffer.as_ref().clone());
            let weak = Arc::downgrade(&owned.buffer);
            let result = session.submit_preview(owned, encoding, FrameOptions::default());
            assert_eq!(session.next_frame_index().get(), 0);
            if limit < plan.owned_bytes_per_job() {
                assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
            } else {
                drop(result.unwrap());
            }
            drop(session);
            retire(&context, &weak);
            if limit < plan.owned_bytes_per_job() {
                continue;
            }

            let mut blocker = bounded.begin_sequence(descriptor()).unwrap();
            let blocked = blocker
                .submit_preview(input.clone(), encoding, FrameOptions::default())
                .unwrap();
            let mut session = bounded.begin_sequence(descriptor()).unwrap();
            assert!(matches!(
                session.submit_preview(input.clone(), encoding, FrameOptions::default()),
                Err(EncodeError::MemoryBackpressure(_))
            ));
            assert_eq!(session.next_frame_index().get(), 0);
            drop((blocked, blocker));
            retire(&context, &std::sync::Weak::new());
            let mut owned = input.clone();
            owned.buffer = Arc::new(input.buffer.as_ref().clone());
            let weak = Arc::downgrade(&owned.buffer);
            let mut job = session
                .submit_preview(owned, encoding, FrameOptions::default())
                .unwrap();
            let preview = pollster::block_on(&mut job).unwrap();
            assert!(weak.upgrade().is_none());
            assert!(preview.encoded_bytes() > 0);
            assert!(preview.reserved_bytes() >= preview.encoded_bytes() as u64);
            assert_eq!(
                context.memory_stats().reserved_bytes,
                preview.reserved_bytes()
            );
            assert!(matches!(
                session.submit_preview(input.clone(), encoding, FrameOptions::default()),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            // A completed preview still consumes the same aggregate budget as a main job.
            assert!(matches!(
                session.submit_last_frame(input.clone(), encoding, FrameOptions::default()),
                Err(EncodeError::MemoryBackpressure(_))
            ));
            drop((job, session));
            assert_eq!(
                context.memory_stats().reserved_bytes,
                preview.reserved_bytes()
            );
            drop(preview);
            assert_eq!(context.memory_stats().reserved_bytes, 0);

            let mut session = bounded.begin_sequence(descriptor()).unwrap();
            let preview = session
                .submit_preview(input.clone(), encoding, FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            drop(preview);
            let frame = session
                .submit_last_frame(input.clone(), encoding, FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            assert!(matches!(
                session.finish_raw(),
                Err(EncodeError::Packet(PacketError::MissingPreview))
            ));
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn invalid_preview_requests_and_cross_session_outputs_cannot_authorize_assembly() {
    let rig = Rig::new();
    let encoder = MixedModeEncoder::new(rig.context.clone(), config()).unwrap();
    let input = source(
        &rig,
        Extent2d::new(17, 9),
        ColorSampleFormat::RGB8,
        false,
        1,
    );
    for encoding in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let mut absent = encoder
            .begin_sequence(ImageSequenceDescriptor::new(17, 9, AnimationHeader::Still).unwrap())
            .unwrap();
        assert!(matches!(
            absent.submit_preview(input.clone(), encoding, FrameOptions::default()),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        let mut session = encoder.begin_sequence(descriptor()).unwrap();
        let blend = FrameBlend {
            mode: BlendMode::Add,
            ..Default::default()
        };
        for options in [
            FrameOptions {
                kind: FrameKind::ReferenceOnly,
                ..Default::default()
            },
            FrameOptions {
                crop: Some(FrameCrop::new(0, 0, 17, 9).unwrap()),
                ..Default::default()
            },
            FrameOptions {
                color_blend: blend,
                ..Default::default()
            },
            FrameOptions {
                extra_channel_blends: vec![blend],
                ..Default::default()
            },
            FrameOptions {
                save_as_reference: ReferenceSlot::new(1).unwrap(),
                ..Default::default()
            },
            FrameOptions {
                save_before_color_transform: true,
                ..Default::default()
            },
            FrameOptions {
                timing: FrameTiming {
                    duration_ticks: 1,
                    timecode: None,
                },
                ..Default::default()
            },
        ] {
            assert!(
                session
                    .preview_memory_plan(&input, encoding, options.clone())
                    .is_err()
            );
            assert!(
                session
                    .submit_preview(input.clone(), encoding, options)
                    .is_err()
            );
            assert_eq!(session.next_frame_index().get(), 0);
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
        let wrong = source(
            &rig,
            Extent2d::new(16, 9),
            ColorSampleFormat::RGB8,
            false,
            1,
        );
        assert!(
            session
                .submit_preview(wrong, encoding, FrameOptions::default())
                .is_err()
        );
        let mut foreign = encoder.begin_sequence(descriptor()).unwrap();
        let foreign_output = foreign
            .submit_preview(input.clone(), encoding, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        let own = session
            .submit_preview(input.clone(), encoding, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        assert!(matches!(
            session.insert_preview(foreign_output),
            Err(EncodeError::Packet(PacketError::UnexpectedPreview))
        ));
        assert_eq!(
            rig.context.memory_stats().reserved_bytes,
            own.reserved_bytes()
        );
        session.insert_preview(own).unwrap();
        let main = session
            .submit_last_frame(input.clone(), encoding, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        session.insert(main).unwrap();
        assert!(session.finish_raw().unwrap().starts_with(&[0xff, 0x0a]));
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        let main = foreign
            .submit_last_frame(input.clone(), encoding, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        foreign.insert(main).unwrap();
        assert!(matches!(
            foreign.finish_raw(),
            Err(EncodeError::Packet(PacketError::MissingPreview))
        ));

        let mut late = encoder.begin_sequence(descriptor()).unwrap();
        let main = late
            .submit_frame(input.clone(), encoding, FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        late.insert(main).unwrap();
        assert!(matches!(
            late.submit_preview(input.clone(), encoding, FrameOptions::default()),
            Err(EncodeError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn invalid_gpu_samples_do_not_publish_a_preview_or_retain_completed_jobs() {
    let rig = Rig::new();
    let samples = ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap();
    let encoder = TiledVarDctEncoder::new_with_config(
        rig.context.clone(),
        VarDctConfig {
            sample_format: samples,
            color_transform: VarDctColorTransform::Original,
            ..Default::default()
        },
    )
    .unwrap();
    let extent = Extent2d::new(17, 9);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut words = sample_words(extent, samples, false, 1);
        words[17] = value.to_bits();
        let input = packed_source(&rig, extent, samples, false, &words);
        let weak = Arc::downgrade(&input.buffer);
        let mut session = encoder.begin_sequence(descriptor()).unwrap();
        let mut job = session
            .submit_preview(input, FrameOptions::default())
            .unwrap();
        assert!(matches!(
            pollster::block_on(&mut job),
            Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
        ));
        assert!(weak.upgrade().is_none());
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        let main = session
            .submit_last_frame(
                source(&rig, extent, samples, false, 1),
                FrameOptions::default(),
            )
            .unwrap()
            .wait()
            .unwrap();
        session.insert(main).unwrap();
        assert!(matches!(
            session.finish_raw(),
            Err(EncodeError::Packet(PacketError::MissingPreview))
        ));
    }
}
