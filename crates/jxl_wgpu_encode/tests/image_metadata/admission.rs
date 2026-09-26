use super::*;

#[test]
fn bounded_metadata_keeps_gpu_admission_retry_and_cancellation_contracts() {
    let rig = Rig::new();
    let extent = Extent2d::new(17, 9);
    let (source, _) = source(&rig, extent, ColorSampleFormat::RGB8, false, 4);
    let config = MixedModeConfig {
        vardct: VarDctConfig {
            color_transform: VarDctColorTransform::Original,
            image_options: ImageOptions {
                orientation: OutputOrientation::from_exif_value(6).unwrap(),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let descriptor =
        ImageSequenceDescriptor::new(extent.width, extent.height, AnimationHeader::Still).unwrap();
    let encoder = MixedModeEncoder::new(rig.context.clone(), config.clone()).unwrap();
    for encoding in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let options = FrameOptions {
            name: name(7),
            ..Default::default()
        };
        let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
        let plan = session
            .memory_plan(&source, encoding, options.clone(), true)
            .unwrap();
        assert_eq!(
            plan,
            session
                .memory_plan(&source, encoding, FrameOptions::default(), true)
                .unwrap()
        );
        let frame = session
            .submit_last_frame(source.clone(), encoding, options.clone())
            .unwrap()
            .wait()
            .unwrap();
        session.insert(frame).unwrap();
        let bytes = session.finish_raw().unwrap();
        for limit in [plan.owned_bytes_per_job() - 1, plan.owned_bytes_per_job()] {
            let context = WgpuContext::with_memory_budget(
                Arc::new(rig.context.device().clone()),
                Arc::new(rig.context.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
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
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while weak.upgrade().is_some() || context.memory_stats().reserved_bytes != 0 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "metadata-bearing cancellation retained source or budget"
                );
                context.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if limit == plan.owned_bytes_per_job() {
                let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
                let frame = session
                    .submit_last_frame(source.clone(), encoding, options.clone())
                    .unwrap()
                    .wait()
                    .unwrap();
                session.insert(frame).unwrap();
                assert_eq!(session.finish_raw().unwrap(), bytes);
            }
        }
    }
}
