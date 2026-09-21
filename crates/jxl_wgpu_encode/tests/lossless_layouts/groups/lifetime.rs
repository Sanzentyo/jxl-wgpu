use super::*;

#[test]
fn each_group_size_keeps_exact_admission_cancellation_and_streamed_reuse() {
    let rig = Rig::new();
    for size in LosslessModularGroupSize::ALL {
        let encoder = encoder(&rig, size, TREES[1]);
        let config = encoder.config();
        let case = Case {
            format: LosslessModularFormat::Rgba,
            bits: 31,
            kind: SampleKind::Unsigned,
            storage: Storage::Planar,
            reversed: true,
            byte_order: ByteOrder::Big,
            shifted: true,
        };
        for extent in [
            Extent2d::new(size.dimension() + 1, 9),
            Extent2d::new(size.dimension() * 17, 1),
        ] {
            let samples = case.samples(extent);
            let input = upload(&rig.context, &case, extent, &samples, 65_539);
            let plan = encoder.memory_plan(&input).unwrap();
            assert_eq!(plan.streaming, extent.height == 1);
            assert_eq!(
                plan.gpu_submission_count,
                if plan.streaming {
                    2 * plan.batch_count
                } else {
                    1
                }
            );
            let limited = |bytes| {
                LosslessModularEncoder::with_config(
                    WgpuContext::with_memory_budget(
                        Arc::new(rig.context.device().clone()),
                        Arc::new(rig.context.queue().clone()),
                        NonZeroU64::new(bytes).unwrap(),
                    )
                    .unwrap(),
                    config,
                )
            };
            let short = limited(plan.owned_bytes_per_job - 1);
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
            let exact = limited(plan.owned_bytes_per_job);
            let mut abandoned = input.clone();
            abandoned.buffer = Arc::new(input.buffer.as_ref().clone());
            let source = Arc::downgrade(&abandoned.buffer);
            drop(exact.submit(abandoned).unwrap());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while source.upgrade().is_some()
                || exact.in_flight_memory_stats().reserved_bytes != 0
                || exact.buffer_pool_stats().leased_buffer_sets != 0
            {
                assert!(
                    std::time::Instant::now() < deadline,
                    "cancelled {size:?} source or GPU memory retained"
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
            check_header(&encoded, size, &[extent]);
            check_oracles(&encoded, &samples, &case);
            color::check_numeric(&rig, &encoded, &[samples], &case);
        }
    }
}
