use super::*;
use jxl_wgpu_encode::EncodeError;

#[test]
fn floating_resident_and_streamed_jobs_keep_exact_admission_and_cancellation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let format = LosslessModularFormat::Rgba;
    let ordinary = LosslessModularEncoder::with_tree_mode(
        context.clone(),
        LosslessModularTreeMode::LocalPerGroup,
    );
    let decoders = decoders(&backend);
    for extent in [Extent2d::new(257, 9), Extent2d::new(16_384, 1)] {
        let (source, expected) = source(&context, extent, format, 32, 0);
        let plan = ordinary.memory_plan(&source).unwrap();
        assert_eq!(plan.streaming, extent.width == 16_384);
        assert_eq!(plan.sample_bit_depth(), depth(32));
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
        let result = short.submit(source.clone()).and_then(pollster::block_on);
        assert!(matches!(
            result,
            Err(EncodeError::MemoryBackpressure(
                jxl_wgpu::MemoryBudgetError::Exhausted { .. }
            ))
        ));
        assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(short.buffer_pool_stats().allocation_misses, 0);

        let exact = limited(plan.owned_bytes_per_job);
        if !plan.streaming {
            let abandoned = exact.submit(source.clone()).unwrap();
            assert_eq!(
                exact.in_flight_memory_stats().reserved_bytes,
                plan.owned_bytes_per_job
            );
            assert!(matches!(
                exact.submit(source.clone()),
                Err(EncodeError::MemoryBackpressure(_))
            ));
            drop(abandoned);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while exact.in_flight_memory_stats().reserved_bytes != 0
                || exact.buffer_pool_stats().leased_buffer_sets != 0
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "floating encode cancellation retained memory"
                );
                context.device().poll(wgpu::PollType::Poll).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        } else {
            assert_eq!(plan.gpu_submission_count, plan.batch_count * 2);
            assert!(plan.artifact_storage_bytes < plan.total_artifact_bytes);
            assert!(plan.peak_source_binding_bytes < plan.source_binding_bytes);
        }
        let blocking = exact.encode(source.clone()).unwrap();
        let asynchronous = pollster::block_on(exact.submit(source).unwrap()).unwrap();
        assert_eq!(blocking, asynchronous);
        assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(exact.buffer_pool_stats().leased_buffer_sets, 0);
        assert!(exact.buffer_pool_stats().reuse_hits > 0);
        check_oracles(&blocking, &expected, format, 32);
        check_gpu(&backend, &decoders, &blocking, &expected, format, 32);
    }
}
