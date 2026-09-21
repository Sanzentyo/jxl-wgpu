use super::*;

#[test]
fn associated_gray_alpha_keeps_exact_resident_and_streamed_admission() {
    let rig = Rig::new();
    let association = AlphaAssociation::Associated;
    let case = Case {
        format: LosslessModularFormat::GrayAlpha,
        bits: 31,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let encoder =
        LosslessModularEncoder::new(rig.context.clone()).with_alpha_association(association);
    for extent in [Extent2d::new(257, 9), Extent2d::new(16_384, 1)] {
        let expected = case.samples(extent);
        let input = upload(&rig.context, &case, extent, &expected, 65_539);
        let plan = encoder.memory_plan(&input).unwrap();
        assert_eq!(plan.streaming, extent.width == 16_384);
        assert_eq!(plan.channel_count, 2);
        assert!(plan.source_binding_bytes < input.layout.logical_size);
        let limited = |bytes| {
            LosslessModularEncoder::new(
                WgpuContext::with_memory_budget(
                    Arc::new(rig.context.device().clone()),
                    Arc::new(rig.context.queue().clone()),
                    NonZeroU64::new(bytes).unwrap(),
                )
                .unwrap(),
            )
            .with_alpha_association(association)
        };
        let short = limited(plan.owned_bytes_per_job - 1);
        let failure = match short.submit(input.clone()) {
            Ok(job) => pollster::block_on(job).unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(failure, EncodeError::MemoryBackpressure(_)));
        assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
        let exact = limited(plan.owned_bytes_per_job);
        let mut cancelled = input.clone();
        cancelled.buffer = Arc::new(input.buffer.as_ref().clone());
        let source = Arc::downgrade(&cancelled.buffer);
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
            || exact.in_flight_memory_stats().reserved_bytes != 0
            || exact.buffer_pool_stats().leased_buffer_sets != 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "cancelled GrayAlpha job retained source or budget"
            );
            rig.context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let encoded = exact.encode(input.clone()).unwrap();
        assert_eq!(
            encoded,
            pollster::block_on(exact.submit(input).unwrap()).unwrap()
        );
        assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
        assert!(exact.buffer_pool_stats().reuse_hits > 0);
        check_oracles(&encoded, &expected, &case);
        color::check_numeric(&rig, &encoded, &[expected], &case);
    }
}

#[test]
fn alpha_declarations_and_bijective_gray_swizzles_are_checked_before_admission() {
    use jxl_gpu_formats::{Swizzle, SwizzleComponent};
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::GrayAlpha,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: false,
        byte_order: ByteOrder::Native,
        shifted: false,
    };
    let extent = Extent2d::new(17, 3);
    let source = upload(&rig.context, &case, extent, &case.samples(extent), 259);
    let encoder = LosslessModularEncoder::new(rig.context.clone())
        .with_alpha_association(AlphaAssociation::Associated);
    for swizzle in [
        Swizzle::X001,
        Swizzle::Xyzw([
            SwizzleComponent::X,
            SwizzleComponent::Zero,
            SwizzleComponent::Zero,
            SwizzleComponent::X,
        ]),
        Swizzle::XYZW,
        Swizzle::X000,
    ] {
        let mut source = source.clone();
        source.layout.format.swizzle = swizzle;
        assert!(matches!(
            encoder.submit(source),
            Err(EncodeError::Unsupported(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
    for format in [LosslessModularFormat::Gray, LosslessModularFormat::Rgb] {
        let case = Case { format, ..case };
        let source = upload(&rig.context, &case, extent, &case.samples(extent), 0);
        assert!(matches!(
            encoder.memory_plan(&source),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert!(matches!(
            encoder.submit(source),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        let descriptor = jxl_wgpu_encode::LosslessModularAnimationDescriptor::new(
            extent.width,
            extent.height,
            format,
            8,
            jxl_wgpu_encode::AnimationHeader::Animation {
                ticks_per_second_numerator: std::num::NonZeroU32::new(100).unwrap(),
                ticks_per_second_denominator: std::num::NonZeroU32::new(1).unwrap(),
                num_loops: 0,
                have_timecodes: false,
            },
        )
        .unwrap();
        assert!(matches!(
            encoder.begin_animation(descriptor),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
}

#[test]
fn gray_alpha_shared_words_and_mixed_word_widths_preserve_canonical_bytes() {
    let rig = Rig::new();
    for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
        let encoder =
            LosslessModularEncoder::new(rig.context.clone()).with_alpha_association(association);
        for (bits, storage) in [
            (7, Storage::SharedWord),
            (24, Storage::ThreeBytes),
            (8, Storage::MixedWords),
        ] {
            let case = Case {
                format: LosslessModularFormat::GrayAlpha,
                bits,
                kind: SampleKind::Unsigned,
                storage,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            let extent = Extent2d::new(1, 257);
            let expected = case.samples(extent);
            let input = upload(&rig.context, &case, extent, &expected, 4099);
            let encoded = encoder.encode_container(input).unwrap();
            assert_eq!(
                encoded,
                encoder
                    .encode_container(upload(
                        &rig.context,
                        &case.canonical(),
                        extent,
                        &expected,
                        0
                    ))
                    .unwrap()
            );
            check_oracles(&encoded, &expected, &case);
            color::check_numeric(&rig, &encoded, &[expected], &case);
        }
    }
}
