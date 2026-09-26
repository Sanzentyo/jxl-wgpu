use super::*;
use std::num::NonZeroU64;

fn request(extent: Extent2d) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::ModularLossless {
            sample_bit_depth: ColorSampleFormat::RGB8.bit_depth(),
        },
        progressive: ProgressivePlan::single(),
        minimum_determinism: Determinism::CrossDevice,
        animation: AnimationHeader::Still,
        canvas_width: extent.width,
        canvas_height: extent.height,
        options: FrameOptions::default(),
    }
}

fn drain(context: &WgpuContext) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn scalar_aliases_admission_cancellation_and_header_memory_are_exact() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    for (extent, entropy) in [
        (Extent2d::new(17, 9), LosslessModularEntropyCoding::Prefix),
        (Extent2d::new(2051, 9), LosslessModularEntropyCoding::Ans),
    ] {
        let definitions = vec![definition(13, 0); 2];
        let (original, _) = inputs(
            &context,
            extent,
            &definitions,
            extent,
            &[UpsamplingFactor::One; 2],
        );
        let scalar = original.extra_channels()[0].clone();
        let aliased = original
            .clone()
            .with_extra_channels(vec![scalar.clone(), scalar.clone()])
            .unwrap();
        let config = LosslessModularConfig {
            extra_channels: definitions,
            entropy,
            group_size: LosslessModularGroupSize::Pixels128,
            ..Default::default()
        };
        let backend = LosslessModularBackend::with_config(&context, config.clone());
        let memory = backend.memory_plan(&aliased).unwrap();
        let one = LosslessModularBackend::with_config(
            &context,
            LosslessModularConfig {
                extra_channels: vec![definition(13, 0)],
                ..config.clone()
            },
        )
        .memory_plan(&original.clone().with_extra_channels(vec![scalar]).unwrap())
        .unwrap();
        assert_eq!(memory.source_binding_bytes, one.source_binding_bytes);
        assert!(memory.transform_scratch_bytes > 0);
        let encoder = LosslessModularEncoder::with_config(context.clone(), config.clone());
        let still = encoder.memory_plan(&aliased).unwrap();
        assert!(still.extra_channel_metadata_bytes > 0);
        assert_eq!(
            still.owned_bytes_per_job,
            memory.owned_bytes_per_job + still.extra_channel_metadata_bytes
        );
        for limit in [memory.owned_bytes_per_job - 1, memory.owned_bytes_per_job] {
            let bounded = WgpuContext::with_memory_budget(
                Arc::new(context.device().clone()),
                Arc::new(context.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let backend = LosslessModularBackend::with_config(&bounded, config.clone());
            let result = backend.submit(
                &bounded,
                GpuFrameSource::Buffer(aliased.clone()),
                &request(extent),
            );
            if limit < memory.owned_bytes_per_job {
                assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
            } else {
                drop(result.unwrap());
                drain(&bounded);
                backend
                    .submit(
                        &bounded,
                        GpuFrameSource::Buffer(aliased.clone()),
                        &request(extent),
                    )
                    .unwrap()
                    .wait()
                    .unwrap();
            }
            drain(&bounded);
        }
        for limit in [still.owned_bytes_per_job - 1, still.owned_bytes_per_job] {
            let bounded = WgpuContext::with_memory_budget(
                Arc::new(context.device().clone()),
                Arc::new(context.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let encoder = LosslessModularEncoder::with_config(bounded.clone(), config.clone());
            let result = encoder.encode(aliased.clone());
            if limit < still.owned_bytes_per_job {
                assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
            } else {
                result.unwrap();
            }
            drain(&bounded);
        }
        let (owned, _) = inputs(
            &context,
            extent,
            &config.extra_channels,
            extent,
            &[UpsamplingFactor::One; 2],
        );
        let weak: Vec<_> = owned
            .extra_channels()
            .iter()
            .map(|input| Arc::downgrade(&input.buffer))
            .collect();
        drop(
            backend
                .submit(&context, GpuFrameSource::Buffer(owned), &request(extent))
                .unwrap(),
        );
        drain(&context);
        // Cancellation may reach a newly spawned streaming worker before its first
        // reservation. Zero active bytes alone does not establish worker completion.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while weak.iter().any(|buffer| buffer.upgrade().is_some())
            && std::time::Instant::now() < deadline
        {
            context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert!(weak.iter().all(|buffer| buffer.upgrade().is_none()));
    }
}

#[test]
fn invalid_scalar_inputs_and_transform_overflow_never_acquire_output_authority() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(5, 3);
    let config = LosslessModularConfig {
        extra_channels: vec![definition(13, 1)],
        ..Default::default()
    };
    let (valid, _) = inputs(
        &context,
        extent,
        &config.extra_channels,
        extent,
        &[UpsamplingFactor::One],
    );
    let backend = LosslessModularBackend::with_config(&context, config.clone());
    for invalid in [
        valid.clone().with_extra_channels(Vec::new()).unwrap(),
        valid
            .clone()
            .with_extra_channels(vec![valid.extra_channels()[0].clone(); 2])
            .unwrap(),
        valid
            .clone()
            .with_extra_channels(vec![source(
                &context,
                Extent2d::new(3, 2),
                SamplePrecision::integer(12).unwrap().pixel_format(),
                &[0; 6],
            )])
            .unwrap(),
        valid
            .clone()
            .with_extra_channels(vec![source(
                &context,
                Extent2d::new(2, 2),
                SamplePrecision::integer(13).unwrap().pixel_format(),
                &[0; 4],
            )])
            .unwrap(),
    ] {
        assert!(
            backend
                .submit(&context, GpuFrameSource::Buffer(invalid), &request(extent))
                .is_err()
        );
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
    let limited = LosslessModularBackend::with_config(
        &context,
        LosslessModularConfig {
            max_extra_channel_metadata_bytes: 1,
            ..config
        },
    );
    assert!(limited.memory_plan(&valid).is_err());
    let extent = Extent2d::new(2, 1);
    let float = ExtraChannel::new(
        ExtraChannelKind::Optional,
        SamplePrecision::float(32, 8).unwrap(),
        0,
        Vec::new(),
    )
    .unwrap();
    let main = source(
        &context,
        extent,
        ColorSampleFormat::RGB8.pixel_format(),
        &[0; 6],
    );
    let scalar = source(
        &context,
        extent,
        float.precision().pixel_format(),
        &[0x8000_0000, 0x7fff_ffff],
    );
    let encoder = LosslessModularEncoder::with_config(
        context.clone(),
        LosslessModularConfig {
            extra_channels: vec![float],
            local_transforms: LosslessModularSqueeze::Horizontal.into(),
            ..Default::default()
        },
    );
    assert!(matches!(
        encoder.encode(main.with_extra_channels(vec![scalar]).unwrap()),
        Err(EncodeError::Backend(BackendError::ModularSqueezeOverflow))
    ));
    drain(&context);
}
