use super::*;
use jxl_wgpu_decode::Error;

pub(super) fn drain(backend: &WgpuBackend) {
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    // The worker may already own a map callback when the caller's Device::poll returns.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while backend.submission_poller().in_flight() != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(backend.submission_poller().in_flight(), 0);
}

#[test]
fn modular_pass_cancellation_corruption_and_budget_admission_preserve_prior_images() {
    let Some(backend) = backend() else {
        return;
    };
    let Some(encoded) = fixture(2051, 17) else {
        return;
    };
    let parsed = jxl_gpu_bitstream::parse(&encoded, Default::default()).unwrap();
    let data = parsed.codestream().to_vec();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let frame = &inventory.frames[0];
    for cap in [40, 1 << 20] {
        let decoder = GpuDecoder::new(
            WgpuSubmissionEngine::new(backend.clone())
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let request = rgba_request();
        let mut reference = decoder.open(&data, request.clone()).unwrap();
        let stats = reference.submission_session().memory_stats();
        let mut expected = Vec::new();
        while let Some(update) = reference.next_update().unwrap() {
            expected.push(read_output(&backend, &update.output().outputs[0]));
        }
        drop(reference);
        assert_eq!(expected.len(), 3);
        for completed in 0..=3 {
            let mut session = decoder.open(&data, request.clone()).unwrap();
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            assert!(
                session
                    .front_pending_frame()
                    .unwrap()
                    .unvalidated_gpu_frame()
                    .is_err(),
                "the final pass has not been queued"
            );
            let mut held = None;
            for _ in 0..completed {
                held = session.next_update().unwrap();
            }
            if completed < 3 {
                assert!(
                    session
                        .front_pending_frame()
                        .unwrap()
                        .unvalidated_gpu_frame()
                        .is_err()
                );
            }
            drop(session);
            drain(&backend);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                if held.is_some() {
                    stats.output_lease_bytes
                } else {
                    0
                }
            );
            if let Some(update) = &held {
                assert_eq!(
                    read_output(&backend, &update.output().outputs[0]),
                    expected[completed - 1]
                );
            }
            drop(held);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        for damaged_pass in 0..2 {
            let packet = frame.sections.iter().filter(|section| matches!(section.kind,
                jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. } if pass_index == damaged_pass))
                .max_by_key(|section| section.bytes.length).unwrap();
            assert!(packet.bytes.length > 32);
            let mut damaged = data.clone();
            let end = (packet.bytes.offset + packet.bytes.length) as usize;
            damaged[end - 16..end].fill(0xff);
            let mut session = decoder.open(&damaged, request.clone()).unwrap();
            let mut held = Vec::new();
            for bytes in &expected[..=damaged_pass as usize] {
                let update = session.next_update().unwrap().unwrap();
                assert_eq!(read_output(&backend, &update.output().outputs[0]), *bytes);
                held.push(update);
            }
            assert!(matches!(
                session.next_update(),
                Err(Error::ModularEntropyRejected { .. })
            ));
            assert!(matches!(session.next_update(), Err(Error::SessionPoisoned)));
            drop(session);
            drain(&backend);
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                stats.output_lease_bytes * held.len() as u64
            );
            for (update, bytes) in held.iter().zip(&expected) {
                assert_eq!(read_output(&backend, &update.output().outputs[0]), *bytes);
            }
            drop(held);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        let budget = MemoryBudget::new(NonZeroU64::new(stats.per_frame_bytes).unwrap());
        let bounded = GpuDecoder::new(
            WgpuSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let mut session = bounded.open(&data, request.clone()).unwrap();
        // Occupying one admitted byte proves all snapshots are reserved before any source is
        // consumed; dropping the blocker permits an exact-budget retry of that same session.
        let blocker = budget.try_reserve(1).unwrap();
        let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(matches!(
            progress.backpressure,
            Some(PrefetchBackpressure::Memory(_))
        ));
        assert_eq!(session.frames_submitted(), 0);
        assert_eq!(budget.snapshot().reserved_bytes, 1);
        drop(blocker);
        session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        let dc = session.next_update().unwrap().unwrap();
        let final_frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        assert_eq!(
            read_output(&backend, &final_frame.output().outputs[0]),
            expected[2]
        );
        assert_eq!(read_output(&backend, &dc.output().outputs[0]), expected[0]);
        drop((dc, final_frame, session));
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn modular_numeric_passes_share_native_codes_and_exact_f64_output() {
    let Some(backend) = backend() else {
        return;
    };
    let Some(data) = fixture(259, 35) else {
        return;
    };
    let decoder = GpuDecoder::new(
        WgpuSubmissionEngine::new(backend.clone())
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let native_request =
        GpuOutputRequest::numeric(Vpi::U8.pixel_format(), NumericSampleMapping::NativeUnsigned)
            .unwrap()
            .with_progressive_output(true);
    let mut native = decoder.open(&data, native_request).unwrap();
    let mut expected = Vec::new();
    while let Some(update) = native.next_update().unwrap() {
        expected.push((
            update.progression(),
            read_output(&backend, &update.output().outputs[0]),
        ));
    }
    drop(native);
    for (format, mapping) in [
        (
            Vpi::F64.pixel_format(),
            NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening),
        ),
        (
            Vpi::F32.pixel_format(),
            NumericSampleMapping::NormalizedGray8,
        ),
        (
            Vpi::TwoS16.pixel_format(),
            NumericSampleMapping::NormalizedGray8,
        ),
    ] {
        let request = GpuOutputRequest::numeric(format.clone(), mapping)
            .unwrap()
            .with_progressive_output(true);
        let mut session = decoder.open(&data, request).unwrap();
        for (progression, pixels) in &expected {
            let update = pollster::block_on(session.next_update_async())
                .unwrap()
                .unwrap();
            assert_eq!(update.progression(), *progression);
            let output = &update.output().outputs[0];
            assert_eq!(
                read_output(&backend, output),
                expected_numeric_bytes(&format, &output.layout, pixels, false)
            );
        }
        assert!(session.next_update().unwrap().is_none());
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
