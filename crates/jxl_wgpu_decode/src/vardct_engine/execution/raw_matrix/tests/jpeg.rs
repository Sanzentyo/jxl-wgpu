use super::*;
use crate::{JpegCoefficientError, JpegCoefficientLimits, VarDctSubmissionEngine};

fn engine() -> VarDctSubmissionEngine {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU adapter is required");
    VarDctSubmissionEngine::new(backend)
        .unwrap()
        .with_stream_window_limit(NonZeroU64::new(40).unwrap())
}

fn input() -> &'static [u8] {
    jxl_test_support::corpus::jpeg_reconstruction::CASES
        .iter()
        .find(|case| case.name == "rgb_sequential")
        .unwrap()
        .input
}

fn drain(engine: &VarDctSubmissionEngine) {
    engine
        .backend
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while engine.memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        engine.backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(engine.memory.snapshot().reserved_bytes, 0);
}

#[test]
fn jpeg_reconstruction_raw_capture_is_admitted_exactly_and_cancelled_at_each_stage() {
    let engine = engine();
    for stage in 0..3 {
        let mut session = engine
            .open_jpeg_coefficients(input(), Default::default())
            .unwrap();
        let mut pending = session.submit_next().unwrap().unwrap();
        let weak = loop {
            if let VarDctPendingStage::RawHfDequant { work, lifetime, .. } = &pending.inner.stage {
                let plan = work
                    .source
                    .packet
                    .pending_raw_hf_dequant_side_image()
                    .unwrap();
                if plan.matrix_index == 0 {
                    let without_capture = engine
                        .pipelines
                        .raw_hf_dequant
                        .plan_source_with_capture(
                            &work.source.codestream,
                            plan,
                            work.source
                                .packet
                                .pending_raw_hf_dequant_packet_end()
                                .unwrap(),
                            40,
                            false,
                        )
                        .unwrap();
                    assert_eq!(work.stream.memory_bytes, without_capture.memory_bytes + 48);
                    assert_eq!(lifetime._permit.bytes(), work.stream.memory_bytes);
                    let matches = match stage {
                        0 => true,
                        1 => work.next_window > 1 && lifetime.job.has_finalization_commands(),
                        2 => !lifetime.job.has_finalization_commands(),
                        _ => unreachable!(),
                    };
                    if matches {
                        break Arc::downgrade(&lifetime._frame);
                    }
                }
            }
            assert!(
                !pending.inner.dependency_submission_ready(),
                "missed capture cancellation stage {stage}"
            );
            let completion = pending.inner.stage_completion().unwrap();
            pending
                .inner
                .advance_staged_packet(completion.wait())
                .unwrap();
        };
        let runtime = Arc::clone(&session.inner.runtime_stats);
        let submissions = session
            .inner
            .runtime_stats
            .submissions_per_frame
            .load(Ordering::Acquire);
        drop(pending);
        drop(session);
        drain(&engine);
        assert!(weak.upgrade().is_none());
        // No cancellation callback may submit the next entropy window.
        assert_eq!(
            runtime.submissions_per_frame.load(Ordering::Acquire),
            submissions
        );
    }
}

#[test]
fn jpeg_reconstruction_capture_admission_is_exact_and_one_byte_short_is_typed() {
    let engine = engine();
    let mut discovery = engine
        .open_jpeg_coefficients(input(), Default::default())
        .unwrap();
    let mut pending = discovery.submit_next().unwrap().unwrap();
    while !matches!(pending.inner.stage, VarDctPendingStage::RawHfDequant { .. }) {
        let completion = pending.inner.stage_completion().unwrap();
        pending
            .inner
            .advance_staged_packet(completion.wait())
            .unwrap();
    }
    // Includes all descriptor admissions between the initial packet and this raw image.
    let exact = engine.memory.snapshot().reserved_bytes;
    drop(pending);
    drop(discovery);
    drain(&engine);
    for shortage in [0, 1] {
        let mut session = engine
            .open_jpeg_coefficients(input(), Default::default())
            .unwrap();
        session.inner.memory = MemoryBudget::new(NonZeroU64::new(exact - shortage).unwrap());
        let memory = session.inner.memory.clone();
        let mut pending = session.submit_next().unwrap().unwrap();
        if shortage == 0 {
            while !matches!(pending.inner.stage, VarDctPendingStage::RawHfDequant { .. }) {
                let completion = pending.inner.stage_completion().unwrap();
                pending
                    .inner
                    .advance_staged_packet(completion.wait())
                    .unwrap();
            }
            assert_eq!(memory.snapshot().available_bytes, 0);
            // This is exact admission for capture. Later AC descriptors have their own permit.
            drop(pending);
        } else {
            let error = pending.wait().unwrap_err();
            assert!(
                matches!(
                    error,
                    crate::Error::VarDct(VarDctDecodeError::MemoryBackpressure(_))
                ),
                "{error:?}"
            );
        }
        engine
            .backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            engine.backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn jpeg_reconstruction_full_decode_budget_and_final_cancellation() {
    let engine = engine();
    let mut session = engine
        .open_jpeg_coefficients(input(), Default::default())
        .unwrap();
    let mut pending = session.submit_next().unwrap().unwrap();
    let mut peak = engine.memory.snapshot().reserved_bytes;
    while !pending.inner.dependency_submission_ready() {
        let completion = pending.inner.stage_completion().unwrap();
        pending
            .inner
            .advance_staged_packet(completion.wait())
            .unwrap();
        peak = peak.max(engine.memory.snapshot().reserved_bytes);
    }
    // The final submission includes restoration and status mapping, but grants no authority yet.
    let weak = Arc::downgrade(pending.inner.lifetime.as_ref().unwrap());
    drop(pending);
    drop(session);
    drain(&engine);
    assert!(weak.upgrade().is_none());
    for shortage in [1, 0] {
        let mut session = engine
            .open_jpeg_coefficients(input(), Default::default())
            .unwrap();
        let memory = MemoryBudget::new(NonZeroU64::new(peak - shortage).unwrap());
        session.inner.memory = memory.clone();
        let result = session.submit_next().unwrap().unwrap().wait();
        if shortage == 0 {
            let frame = result.unwrap();
            assert_eq!(
                memory.snapshot().reserved_bytes,
                frame.output.buffer().size()
            );
            drop(session);
            assert_eq!(
                memory.snapshot().reserved_bytes,
                frame.output.buffer().size()
            );
            drop(frame);
        } else {
            assert!(
                matches!(
                    result,
                    Err(crate::Error::MemoryBackpressure(_))
                        | Err(crate::Error::VarDct(VarDctDecodeError::MemoryBackpressure(
                            _
                        )))
                ),
                "{result:?}"
            );
        }
        engine
            .backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}

#[test]
fn jpeg_reconstruction_retries_initial_backpressure_and_rejects_corrupt_captured_quantizers() {
    let engine = engine();
    for invalid in [0i32, 65536] {
        let mut session = engine
            .open_jpeg_coefficients(input(), JpegCoefficientLimits::default())
            .unwrap();
        let available = engine.memory.snapshot().available_bytes;
        let blocker = engine
            .memory
            .try_reserve(available - session.memory_stats().total_frame_bytes + 1)
            .unwrap();
        assert!(matches!(
            session.submit_next(),
            Err(crate::Error::MemoryBackpressure(_))
        ));
        assert!(
            session.inner.source.is_some(),
            "pre-submission backpressure preserves retry"
        );
        drop(blocker);
        let mut pending = session.submit_next().unwrap().unwrap();
        loop {
            let completed_capture = matches!(&pending.inner.stage, VarDctPendingStage::RawHfDequant { work, lifetime, .. }
                if work.source.packet.pending_raw_hf_dequant_side_image().unwrap().matrix_index == 0 && !lifetime.job.has_finalization_commands());
            assert!(!pending.inner.dependency_submission_ready());
            let completion = pending.inner.stage_completion().unwrap();
            let mapping = completion.wait();
            if completed_capture {
                let life = pending.inner.lifetime.as_ref().unwrap();
                // Fault injection after the exact raw image completes, before JPEG restoration.
                engine.backend.queue().write_buffer(
                    life.output.as_wgpu_buffer(),
                    0,
                    &invalid.to_le_bytes(),
                );
            }
            pending.inner.advance_staged_packet(mapping).unwrap();
            if completed_capture {
                break;
            }
        }
        let error = pending.wait().unwrap_err();
        assert!(
            matches!(
                error,
                crate::Error::VarDct(VarDctDecodeError::Jpeg(
                    JpegCoefficientError::GpuStatus { .. }
                ))
            ),
            "{error:?}"
        );
        drop(session);
        drain(&engine);
        let mut valid = engine
            .open_jpeg_coefficients(input(), Default::default())
            .unwrap();
        let frame = valid.submit_next().unwrap().unwrap().wait().unwrap();
        assert_eq!(
            engine.memory.snapshot().reserved_bytes,
            frame.output.buffer().size()
        );
        drop(frame);
        drain(&engine);
    }
}
