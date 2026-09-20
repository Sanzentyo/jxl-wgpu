use super::*;
use jxl_wgpu_encode::EncodeError;

#[test]
fn wide_artifacts_preserve_streamed_and_resident_memory_contracts() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let format = LosslessModularFormat::Rgba;
    let (streamed_source, streamed_expected) =
        source(&context, Extent2d::new(16_384, 1), format, 31, 0);
    let encoder = LosslessModularEncoder::with_tree_mode(
        context.clone(),
        LosslessModularTreeMode::LocalPerGroup,
    );
    let plan = encoder.memory_plan(&streamed_source).unwrap();
    assert!(plan.streaming && plan.batch_count > 1);
    assert_eq!(plan.gpu_submission_count, plan.batch_count * 2);
    assert!(plan.artifact_storage_bytes < plan.total_artifact_bytes);
    assert!(plan.peak_source_binding_bytes < plan.source_binding_bytes);
    let limited = |bytes| {
        let context = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(bytes).unwrap(),
        )
        .unwrap();
        LosslessModularEncoder::with_tree_mode(context, LosslessModularTreeMode::LocalPerGroup)
    };
    let short = limited(plan.owned_bytes_per_job - 1);
    // Streamed native jobs acquire each batch on their worker. Their admission
    // failure is returned by completion, before a GPU buffer is leased.
    assert!(matches!(
        pollster::block_on(short.submit(streamed_source.clone()).unwrap()),
        Err(EncodeError::MemoryBackpressure(
            jxl_wgpu::MemoryBudgetError::Exhausted { .. }
        ))
    ));
    assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(short.buffer_pool_stats().allocation_misses, 0);

    let exact = limited(plan.owned_bytes_per_job);
    let blocking = exact.encode(streamed_source.clone()).unwrap();
    assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
    let asynchronous = pollster::block_on(exact.submit(streamed_source).unwrap()).unwrap();
    assert_eq!(blocking, asynchronous);
    assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(exact.buffer_pool_stats().leased_buffer_sets, 0);
    assert!(exact.buffer_pool_stats().reuse_hits > 0);
    check_oracles(&blocking, &streamed_expected, format, 31);

    // Resident submissions reserve synchronously and keep their mapped batch
    // until consumption or abandoned-job cleanup, so concurrent pressure is deterministic.
    let (source, expected) = source(&context, Extent2d::new(257, 9), format, 31, 0);
    let plan = encoder.memory_plan(&source).unwrap();
    assert!(!plan.streaming);
    let short = limited(plan.owned_bytes_per_job - 1);
    assert!(matches!(
        short.submit(source.clone()),
        Err(EncodeError::MemoryBackpressure(
            jxl_wgpu::MemoryBudgetError::Exhausted { .. }
        ))
    ));
    assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
    let exact = limited(plan.owned_bytes_per_job);
    let abandoned = exact.submit(source.clone()).unwrap();
    assert_eq!(
        exact.in_flight_memory_stats().reserved_bytes,
        plan.owned_bytes_per_job
    );
    assert!(matches!(
        exact.submit(source.clone()),
        Err(EncodeError::MemoryBackpressure(
            jxl_wgpu::MemoryBudgetError::Exhausted { .. }
        ))
    ));
    drop(abandoned);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while exact.in_flight_memory_stats().reserved_bytes != 0
        || exact.buffer_pool_stats().leased_buffer_sets != 0
    {
        assert!(
            std::time::Instant::now() < deadline,
            "abandoned encode did not release its reservation"
        );
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let blocking = exact.encode(source.clone()).unwrap();
    assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
    let asynchronous = pollster::block_on(exact.submit(source).unwrap()).unwrap();
    assert_eq!(blocking, asynchronous);
    assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(exact.buffer_pool_stats().leased_buffer_sets, 0);
    check_oracles(&blocking, &expected, format, 31);
}
