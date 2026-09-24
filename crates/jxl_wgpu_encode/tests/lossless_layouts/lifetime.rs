use super::*;

fn release(encoder: &LosslessModularEncoder, context: &WgpuContext) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while encoder.in_flight_memory_stats().reserved_bytes != 0
        || encoder.buffer_pool_stats().leased_buffer_sets != 0
    {
        assert!(
            std::time::Instant::now() < deadline,
            "abandoned layout job did not release memory"
        );
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn separated_sources_keep_exact_admission_cancellation_and_streaming_contracts() {
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 31,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    check_case(
        &rig,
        case,
        jxl_wgpu_encode::LosslessModularConfig {
            tree_mode: LosslessModularTreeMode::LocalPerGroup,
            ..Default::default()
        },
        Extent2d::new(257, 9),
    );
}

pub(super) fn check_case(
    rig: &Rig,
    case: Case,
    config: jxl_wgpu_encode::LosslessModularConfig,
    resident_extent: Extent2d,
) {
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config.clone());
    let limited = |bytes| {
        LosslessModularEncoder::with_config(
            WgpuContext::with_memory_budget(
                Arc::new(rig.context.device().clone()),
                Arc::new(rig.context.queue().clone()),
                NonZeroU64::new(bytes).unwrap(),
            )
            .unwrap(),
            config.clone(),
        )
    };
    for (extent, streaming) in [(resident_extent, false), (Extent2d::new(16_384, 1), true)] {
        let expected = case.samples(extent);
        let input = upload(&rig.context, &case, extent, &expected, 65_539);
        let plan = encoder.memory_plan(&input).unwrap();
        assert!(plan.source_binding_bytes < input.layout.logical_size);
        assert_eq!(
            plan.addressed_bytes_per_job,
            plan.owned_bytes_per_job + plan.peak_source_binding_bytes
        );
        assert_eq!(plan.streaming, streaming, "{case:?}, {extent:?}");
        if plan.streaming {
            assert_eq!(plan.gpu_submission_count, plan.batch_count * 2);
            assert!(plan.peak_source_binding_bytes < plan.source_binding_bytes);
        }
        let short = limited(plan.owned_bytes_per_job - 1);
        let failure = match short.submit(input.clone()) {
            Ok(job) => pollster::block_on(job).unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(failure, EncodeError::MemoryBackpressure(_)));
        assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
        let exact = limited(plan.owned_bytes_per_job);
        let mut cancelled_input = input.clone();
        cancelled_input.buffer = Arc::new(input.buffer.as_ref().clone());
        let cancelled_source = Arc::downgrade(&cancelled_input.buffer);
        let abandoned = exact.submit(cancelled_input).unwrap();
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
        // Zero reserved bytes alone can be observed before a native worker's first batch.
        // Its unique source owner must disappear before another exact-budget job starts.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while cancelled_source.upgrade().is_some() {
            assert!(
                std::time::Instant::now() < deadline,
                "cancelled worker retained its source"
            );
            rig.context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        release(&exact, &rig.context);
        let encoded = exact.encode(input.clone()).unwrap();
        assert_eq!(
            encoded,
            pollster::block_on(exact.submit(input).unwrap()).unwrap()
        );
        release(&exact, &rig.context);
        assert!(exact.buffer_pool_stats().reuse_hits > 0);
        let native = check_oracles(&encoded, &expected, &case);
        rig.check_gpu(&encoded, &expected, &case, &native);
    }
}

#[test]
fn public_layout_mutations_and_invalid_channels_fail_before_admission() {
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 12,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let extent = Extent2d::new(17, 3);
    let input = upload(&rig.context, &case, extent, &case.samples(extent), 259);
    let encoder = LosslessModularEncoder::new(rig.context.clone());
    let mut invalid = Vec::new();
    let mut changed = input.clone();
    changed.layout.format.planes.pop();
    changed.layout.planes.pop();
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[2].row_stride = 1;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[1].offset = changed.layout.planes[0].offset;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[3].row_bytes += 1;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[2].sample_extent.width += 1;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.logical_size -= 1;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[0].offset = u64::MAX;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.format.swizzle =
        jxl_gpu_formats::Swizzle::Xyzw([jxl_gpu_formats::SwizzleComponent::X; 4]);
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.format.planes[1] = changed.layout.format.planes[0].clone();
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.format.planes[0].words[0].fields[0].bits = 0;
    invalid.push(changed);
    let mut changed = input.clone();
    changed.layout.planes[0].row_stride = u64::MAX;
    invalid.push(changed);
    for source in invalid {
        assert!(matches!(
            encoder.submit(source),
            Err(EncodeError::InvalidSource(_)
                | EncodeError::SourceLayout(_)
                | EncodeError::Unsupported(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
}
