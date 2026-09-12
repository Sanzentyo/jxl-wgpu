use super::*;

#[test]
fn extra_pass_failures_and_cancellation_preserve_validated_images_and_release_budgets() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for name in [
        "vardct_extras_rgba_progressive",
        "vardct_extras_associated_squeeze",
    ] {
        let data = fixture(name);
        let parsed = jxl_gpu_bitstream::parse(&data, Default::default()).unwrap();
        assert_eq!(parsed.codestream(), data);
        let inventory = parsed.codestream_inventory(Default::default()).unwrap();
        let frame = &inventory.frames[0];
        for cap in [u64::MAX, 40] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = request(true, AlphaOutputPolicy::Preserve);
            let mut reference = decoder.open(&data, request.clone()).unwrap();
            let mut expected = Vec::new();
            while let Some(update) = reference.next_update().unwrap() {
                expected.push(read(&backend, update.output()));
            }
            let memory = reference
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .unwrap();
            drop(reference);
            drain_gpu(&backend, 0);
            assert_eq!(expected.len(), frame.num_passes as usize + 1);
            assert_eq!(
                memory.intermediate_output_bytes,
                u64::from(frame.num_passes) * memory.output_lease_bytes
            );
            assert!(
                memory.intermediate_transient_bytes
                    >= u64::from(frame.num_passes)
                        * (memory.extra_arena_bytes
                            + memory.extra_inverse_uniform_bytes
                            + memory.extra_render_bytes)
            );

            for completed in 0..=frame.num_passes as usize {
                let mut session = open_incremental(&decoder, &data, request.clone());
                session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                let mut held = None;
                for _ in 0..completed {
                    held = session.next_update().unwrap();
                }
                drop(session);
                let retained = if held.is_some() {
                    memory.output_lease_bytes
                } else {
                    0
                };
                drain_gpu(&backend, retained);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    retained,
                    "{name} cancel{completed}"
                );
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
                if let Some(update) = &held {
                    assert_eq!(read(&backend, update.output()), expected[completed - 1]);
                }
                drop(held);
            }

            for damaged_pass in 0..frame.num_passes {
                let packet = frame.sections.iter().filter(|section| matches!(section.kind,
                    jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index == damaged_pass))
                    .max_by_key(|section| section.bytes.length).unwrap();
                let mut damaged = data.clone();
                let end = (packet.bytes.offset + packet.bytes.length) as usize;
                let len = (packet.bytes.length as usize).min(16);
                damaged[end - len..end].fill(0xff);
                let mut session = open_incremental(&decoder, &damaged, request.clone());
                let mut held = Vec::new();
                for expected in &expected[..=damaged_pass as usize] {
                    let update = session.next_update().unwrap().unwrap();
                    assert!(!update.is_complete());
                    assert_eq!(read(&backend, update.output()), *expected);
                    held.push(update);
                }
                let error = session.next_update().unwrap_err();
                eprintln!("{name} cap{cap} damaged pass{damaged_pass}: {error:?}");
                if name.ends_with("squeeze") {
                    assert!(
                        matches!(
                            error,
                            DecodeError::VarDct(VarDctDecodeError::ExtraModularStatus { .. })
                        ),
                        "{error:?}"
                    );
                } else {
                    assert!(
                        matches!(
                            error,
                            DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_))
                        ),
                        "{error:?}"
                    );
                }
                assert!(matches!(
                    session.next_frame(),
                    Err(DecodeError::SessionPoisoned)
                ));
                drop(session);
                let retained = memory.output_lease_bytes * held.len() as u64;
                drain_gpu(&backend, retained);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    retained
                );
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
                for (update, expected) in held.iter().zip(&expected) {
                    assert_eq!(read(&backend, update.output()), *expected);
                }
                drop(held);
            }

            // Initial allocation failure leaves admission retryable.
            let mut session = decoder.open(&data, request.clone()).unwrap();
            let budget = backend.transient_memory_budget();
            let blocker = budget
                .try_reserve(budget.snapshot().available_bytes)
                .unwrap();
            let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            assert!(matches!(
                progress.backpressure,
                Some(PrefetchBackpressure::Memory(_))
            ));
            assert_eq!(session.frames_submitted(), 0);
            drop(blocker);
            let dc = session.next_update().unwrap().unwrap();
            assert_eq!(read(&backend, dc.output()), expected[0]);
            drop(dc);
            // Final-only continuation drives all remaining extra subimages without publishing them.
            let final_frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            assert_eq!(
                read(&backend, final_frame.output()),
                *expected.last().unwrap()
            );
            assert!(session.next_update().unwrap().is_none());
            drop(final_frame);
            drop(session);

            if name.ends_with("squeeze") {
                if cap == 40 {
                    // Stop inside a bounded AC extra subimage, while its arena is separately live.
                    use std::future::Future;
                    let mut session = open_incremental(&decoder, &data, request.clone());
                    let dc = session.next_update().unwrap().unwrap();
                    let before = budget.snapshot().reserved_bytes;
                    let mut future = Box::pin(session.next_update_async());
                    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                    loop {
                        assert!(future.as_mut().poll(&mut context).is_pending());
                        if budget.snapshot().reserved_bytes > before {
                            break;
                        }
                        assert!(
                            std::time::Instant::now() < deadline,
                            "extra subimage was not admitted"
                        );
                        std::thread::yield_now();
                    }
                    drop(future);
                    drop(session);
                    drain_gpu(&backend, memory.output_lease_bytes);
                    assert_eq!(budget.snapshot().reserved_bytes, memory.output_lease_bytes);
                    assert_eq!(read(&backend, dc.output()), expected[0]);
                    drop(dc);
                }
                // The next extra subimage has not been allocated merely to publish DC.
                let mut session = open_incremental(&decoder, &data, request.clone());
                let dc = session.next_update().unwrap().unwrap();
                let blocker = budget
                    .try_reserve(budget.snapshot().available_bytes)
                    .unwrap();
                assert!(matches!(
                    session.next_update(),
                    Err(DecodeError::MemoryBackpressure(_))
                ));
                assert!(matches!(
                    session.next_update(),
                    Err(DecodeError::SessionPoisoned)
                ));
                drop(blocker);
                drop(session);
                drain_gpu(&backend, memory.output_lease_bytes);
                assert_eq!(budget.snapshot().reserved_bytes, memory.output_lease_bytes);
                assert_eq!(read(&backend, dc.output()), expected[0]);
                drop(dc);
            }
            drain_gpu(&backend, 0);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}
