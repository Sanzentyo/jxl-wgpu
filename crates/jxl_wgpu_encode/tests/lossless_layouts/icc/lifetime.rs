use super::*;
use jxl_wgpu_encode::{AnimationHeader, LosslessModularAnimationDescriptor};
use std::num::NonZeroU32;

fn animation_descriptor(source: &BufferImageSource) -> LosslessModularAnimationDescriptor {
    LosslessModularAnimationDescriptor::from_pixel_format(
        source.layout.extent.width,
        source.layout.extent.height,
        &source.layout.format,
        AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 0,
            have_timecodes: false,
        },
    )
    .unwrap()
}

#[test]
fn icc_headers_share_resident_and_streamed_budgets_and_retire_on_cancel_or_ready() {
    let rig = Rig::new();
    let profile = profile(true);
    let encoder =
        encoder(&rig, &profile, TREES[0]).with_alpha_association(AlphaAssociation::Associated);
    let case = Case {
        format: LosslessModularFormat::GrayAlpha,
        bits: 31,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    for extent in [Extent2d::new(257, 9), Extent2d::new(16_384, 1)] {
        let expected = case.samples(extent);
        let mut input = upload(&rig.context, &case, extent, &expected, 65_539);
        attach(&mut input, &profile, true);
        let plan = encoder.memory_plan(&input).unwrap();
        let backend = jxl_wgpu_encode::LosslessModularBackend::new(&rig.context);
        let gpu_plan = backend.memory_plan(&input).unwrap();
        assert_eq!(plan.streaming, extent.width == 16_384);
        assert_eq!(plan.icc_profile_bytes, profile.bytes().len() as u64);
        assert_eq!(gpu_plan.icc_storage_bytes, 0);
        assert_eq!(
            plan.owned_bytes_per_job,
            gpu_plan.owned_bytes_per_job + plan.icc_storage_bytes
        );
        assert_eq!(
            plan.addressed_bytes_per_job,
            gpu_plan.addressed_bytes_per_job + plan.icc_storage_bytes
        );
        assert_eq!(
            plan.addressed_bytes_per_job,
            plan.owned_bytes_per_job + plan.peak_source_binding_bytes + plan.icc_profile_bytes
        );
        assert_eq!(plan.gpu_submission_count, gpu_plan.gpu_submission_count);
        let limited = |bytes| {
            LosslessModularEncoder::new(
                WgpuContext::with_memory_budget(
                    Arc::new(rig.context.device().clone()),
                    Arc::new(rig.context.queue().clone()),
                    NonZeroU64::new(bytes).unwrap(),
                )
                .unwrap(),
            )
            .with_image_options(ImageOptions {
                rendering_intent: profile.header().rendering_intent,
                ..Default::default()
            })
            .unwrap()
            .with_alpha_association(AlphaAssociation::Associated)
        };
        for bytes in [plan.icc_storage_bytes - 1, plan.owned_bytes_per_job - 1] {
            let short = limited(bytes);
            let failure = match short.submit(input.clone()) {
                Ok(job) => pollster::block_on(job).unwrap_err(),
                Err(error) => error,
            };
            assert!(
                matches!(failure, EncodeError::MemoryBackpressure(_)),
                "{failure:?}"
            );
            assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
        }
        let exact = limited(plan.owned_bytes_per_job);
        let mut cancelled = input.clone();
        cancelled.buffer = Arc::new(input.buffer.as_ref().clone());
        let source = Arc::downgrade(&cancelled.buffer);
        let unique_profile =
            IccProfile::parse(profile.bytes().to_vec().into(), Default::default()).unwrap();
        let profile_bytes = Arc::downgrade(unique_profile.bytes());
        cancelled.layout.format.color_spec = ColorSpecification::Icc(unique_profile);
        let abandoned = exact.submit(cancelled).unwrap();
        if !plan.streaming {
            assert_eq!(
                exact.in_flight_memory_stats().reserved_bytes,
                plan.owned_bytes_per_job
            );
            assert!(matches!(
                exact.submit(input.clone()),
                Err(EncodeError::MemoryBackpressure(_))
            ));
        }
        drop(abandoned);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while source.upgrade().is_some()
            || profile_bytes.upgrade().is_some()
            || exact.in_flight_memory_stats().reserved_bytes != 0
            || exact.buffer_pool_stats().leased_buffer_sets != 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "cancelled ICC job retained source, profile, or budget"
            );
            rig.context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let mut job = exact.submit(input.clone()).unwrap();
        let encoded = pollster::block_on(&mut job).unwrap();
        // The completed future itself remains alive; it must no longer retain its header permit.
        assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoded, exact.encode(input).unwrap());
        drop(job);
        assert!(exact.buffer_pool_stats().reuse_hits > 0);
        check_frame_samples(&encoded, &[&expected], &case, &original(&encoded));
        color::check_numeric(&rig, &encoded, &[expected], &case);
    }
}

#[test]
fn icc_limits_intents_and_device_components_reject_before_gpu_admission() {
    let rig = Rig::new();
    let profile = profile(false);
    let encoder = encoder(&rig, &profile, TREES[0]);
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Packed,
        reversed: false,
        byte_order: ByteOrder::Little,
        shifted: false,
    };
    let extent = Extent2d::new(17, 3);
    let mut input = upload(&rig.context, &case, extent, &case.samples(extent), 259);
    attach(&mut input, &profile, true);
    let profile_size = profile.bytes().len() as u64;
    for limit in [0, profile_size - 1] {
        let limited = super::encoder(&rig, &profile, TREES[0]).with_max_icc_profile_bytes(limit);
        let is_limit = |error| matches!(error, EncodeError::IccLimit { resource:"profile bytes", required, limit:actual } if required == profile_size && actual == limit);
        assert!(is_limit(limited.memory_plan(&input).unwrap_err()));
        assert!(is_limit(limited.submit(input.clone()).err().unwrap()));
        assert!(is_limit(
            limited
                .begin_animation(animation_descriptor(&input))
                .err()
                .unwrap()
        ));
        assert_eq!(limited.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(limited.buffer_pool_stats().allocation_misses, 0);
    }
    assert!(
        super::encoder(&rig, &profile, TREES[0])
            .with_max_icc_profile_bytes(profile_size)
            .memory_plan(&input)
            .is_ok()
    );
    let mismatch = if profile.header().rendering_intent
        == jxl_gpu_protocol::icc::IccRenderingIntent::Relative
    {
        jxl_gpu_protocol::icc::IccRenderingIntent::Perceptual
    } else {
        jxl_gpu_protocol::icc::IccRenderingIntent::Relative
    };
    let conflicting = super::encoder(&rig, &profile, TREES[0])
        .with_image_options(ImageOptions {
            rendering_intent: mismatch,
            ..Default::default()
        })
        .unwrap();
    assert!(matches!(
        conflicting.memory_plan(&input),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        conflicting.submit(input.clone()),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        conflicting.begin_animation(animation_descriptor(&input)),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert_eq!(conflicting.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(conflicting.buffer_pool_stats().allocation_misses, 0);
    for invalid_channel in [
        Channel::Device(3),
        Channel::Device(1),
        Channel::X,
        Channel::Alpha,
    ] {
        let mut invalid = input.clone();
        invalid.layout.format.planes[0].words[0].fields[0].kind =
            PackingFieldKind::Channel(invalid_channel);
        assert!(matches!(
            encoder.submit(invalid),
            Err(EncodeError::Unsupported(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
    let mut wrong_space = input.clone();
    wrong_space.layout.format.color_spec = ColorSpecification::Icc(super::profile(true));
    assert!(matches!(
        encoder.submit(wrong_space),
        Err(EncodeError::Unsupported(_))
    ));
    let mut wrong_model = input;
    wrong_model.layout.format.model = ColorModel::Gray;
    assert!(matches!(
        encoder.submit(wrong_model),
        Err(EncodeError::Unsupported(_))
    ));
    let mut cmyk = upload(&rig.context, &case, extent, &case.samples(extent), 0);
    attach(&mut cmyk, &profile, true);
    cmyk.layout.format.planes[0].words[3].fields[0].kind =
        PackingFieldKind::Channel(Channel::Device(4));
    cmyk.layout.format.color_spec = ColorSpecification::Icc(
        IccProfile::parse(
            std::fs::read(
                jxl_test_support::fixtures::embedded_icc::directory()
                    .parent()
                    .unwrap()
                    .join("cmyk/layers.icc"),
            )
            .unwrap()
            .into(),
            Default::default(),
        )
        .unwrap(),
    );
    assert!(matches!(
        encoder.submit(cmyk),
        Err(EncodeError::Unsupported(_))
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
}

#[test]
fn icc_animation_header_admission_and_drop_are_independent_of_live_frames() {
    let rig = Rig::new();
    let profile = profile(false);
    let encoder = encoder(&rig, &profile, TREES[0]);
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Packed,
        reversed: false,
        byte_order: ByteOrder::Little,
        shifted: false,
    };
    let extent = Extent2d::new(17, 3);
    let mut input = upload(&rig.context, &case, extent, &case.samples(extent), 259);
    attach(&mut input, &profile, true);
    let descriptor = animation_descriptor(&input);
    let animation = encoder.begin_animation(descriptor.clone()).unwrap();
    let header_bytes = encoder.in_flight_memory_stats().reserved_bytes;
    let frame_bytes = jxl_wgpu_encode::LosslessModularBackend::new(&rig.context)
        .memory_plan(&input)
        .unwrap()
        .owned_bytes_per_job;
    drop(animation);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    let limited = |bytes| {
        LosslessModularEncoder::new(
            WgpuContext::with_memory_budget(
                Arc::new(rig.context.device().clone()),
                Arc::new(rig.context.queue().clone()),
                NonZeroU64::new(bytes).unwrap(),
            )
            .unwrap(),
        )
        .with_image_options(ImageOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        })
        .unwrap()
    };
    let short = limited(header_bytes - 1);
    assert!(matches!(
        short.begin_animation(descriptor.clone()),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
    let short = limited(header_bytes + frame_bytes - 1);
    let mut animation = short.begin_animation(descriptor.clone()).unwrap();
    assert!(matches!(
        animation.submit_last_frame(input.clone(), Default::default()),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    assert_eq!(animation.next_frame_index().get(), 0);
    assert_eq!(short.in_flight_memory_stats().reserved_bytes, header_bytes);
    assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
    assert!(matches!(
        animation.finish_container(),
        Err(EncodeError::MissingFinalFrame)
    ));
    assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
    let exact = limited(header_bytes + frame_bytes);
    let mut animation = exact.begin_animation(descriptor).unwrap();
    let frame = animation
        .submit_last_frame(input, Default::default())
        .unwrap();
    assert_eq!(
        exact.in_flight_memory_stats().reserved_bytes,
        header_bytes + frame_bytes
    );
    drop(animation);
    assert_eq!(exact.in_flight_memory_stats().reserved_bytes, frame_bytes);
    drop(frame);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while exact.in_flight_memory_stats().reserved_bytes != 0
        || exact.buffer_pool_stats().leased_buffer_sets != 0
    {
        assert!(
            std::time::Instant::now() < deadline,
            "abandoned ICC frame retained GPU resources"
        );
        rig.context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}
