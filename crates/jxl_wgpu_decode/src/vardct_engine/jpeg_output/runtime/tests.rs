use super::*;
use jxl_test_support::corpus::jpeg_reconstruction::{CASES, Case};
use jxl_wgpu::{MemoryBudget, WgpuBackend};
use std::num::NonZeroU64;

fn case(name: &str) -> &'static Case {
    CASES.iter().find(|case| case.name == name).unwrap()
}
fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default())).expect("real GPU required")
}
fn stage(pending: &JpegReconstructionPending) -> u32 {
    match pending.state.as_ref().unwrap() {
        State::Coefficients(_) => 0,
        State::Scan(state) => match state.scan.phase {
            gpu::ScanPhase::Count => 1,
            gpu::ScanPhase::Emit => 2,
            gpu::ScanPhase::Pack => 3,
        },
        State::Assembly(_) => 4,
    }
}
fn reach(pending: &mut JpegReconstructionPending, target: u32) {
    while stage(pending) != target {
        if stage(pending) == 0 {
            let Some(State::Coefficients(coefficients)) = pending.state.take() else {
                unreachable!()
            };
            pending
                .coefficients_ready(coefficients.wait().unwrap())
                .unwrap();
        } else {
            let mapping = pending.operation().unwrap().wait();
            assert!(pending.advance(mapping).unwrap().is_none());
        }
    }
}
fn drained(backend: &WgpuBackend, memory: &MemoryBudget) {
    backend
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    // A concurrent native poll worker can still be returning from the completion callback.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(memory.snapshot().reserved_bytes, 0);
}

#[test]
fn jpeg_stage_cancellation_and_late_admission_release_every_reservation() {
    let backend = backend();
    let memory = MemoryBudget::new(NonZeroU64::new(256 << 20).unwrap());
    let engine =
        VarDctSubmissionEngine::with_memory_budget(backend.clone(), memory.clone()).unwrap();
    let input = case("gray_restart").input;
    // Initial admission is retryable. No metadata/session authority is consumed on failure.
    let mut session = engine
        .open_jpeg_reconstruction(input, Default::default())
        .unwrap();
    let blocker = memory
        .try_reserve(memory.snapshot().limit_bytes - 1)
        .unwrap();
    assert!(matches!(
        session.submit_next(),
        Err(crate::Error::MemoryBackpressure(_))
    ));
    drop(blocker);
    drop(session.submit_next().unwrap().unwrap());
    drop(session);
    drained(&backend, &memory);
    for target in 0..=4 {
        let mut session = engine
            .open_jpeg_reconstruction(input, Default::default())
            .unwrap();
        let mut pending = session.submit_next().unwrap().unwrap();
        reach(&mut pending, target);
        drop(session);
        assert!(memory.snapshot().reserved_bytes > 0);
        drop(pending);
        drained(&backend, &memory);
    }
    // A successfully completed current stage cannot allocate its successor while the shared
    // budget is exhausted. Failure drops all intermediates; no final byte lease is published.
    for target in 1..=3 {
        let mut session = engine
            .open_jpeg_reconstruction(input, Default::default())
            .unwrap();
        if target == 3 {
            // A legal opaque JPEG tail can make final assembly larger than the released scan
            // scratch. Model that allocation in the private plan without replacing a fixture.
            let plan = Arc::get_mut(session.plan.as_mut().unwrap()).unwrap();
            plan.framing.bytes.resize(1 << 20, 0);
            plan.framing.owned_bytes += 1 << 20;
        }
        let mut pending = session.submit_next().unwrap().unwrap();
        reach(&mut pending, target);
        let mapping = pending.operation().unwrap().wait();
        let free = memory.snapshot().limit_bytes - memory.snapshot().reserved_bytes;
        let blocker = memory.try_reserve(free).unwrap();
        let result = pending.advance(mapping);
        assert!(
            matches!(result, Err(crate::Error::MemoryBackpressure(_))),
            "stage {target}: {result:?}"
        );
        assert!(pending.state.is_none());
        drop(pending);
        drop(session);
        drop(blocker);
        drained(&backend, &memory);
    }
    let mut session = engine
        .open_jpeg_reconstruction(input, Default::default())
        .unwrap();
    let mut pending = session.submit_next().unwrap().unwrap();
    reach(&mut pending, 1);
    let mapping = pending.operation().unwrap().wait();
    let mut slots = Vec::new();
    while let Ok(slot) = backend.submission_poller().try_reserve() {
        slots.push(slot);
    }
    assert!(matches!(
        pending.advance(mapping),
        Err(crate::Error::PollBackpressure(_))
    ));
    drop(slots);
    drop(pending);
    drop(session);
    drained(&backend, &memory);
}

#[test]
fn jpeg_gpu_faults_do_not_publish_byte_authority() {
    let backend = backend();
    let engine = VarDctSubmissionEngine::new(backend.clone()).unwrap();
    for (name, fault, expected_stage, expected_code) in [
        ("gray_restart", "code", "scan entropy", 4),
        ("gray_restart", "previous", "scan entropy", 5),
        ("gray_restart", "extra", "scan entropy", 7),
        ("gray_restart", "marker", "scan entropy", 8),
        ("gray_restart", "scan", "scan entropy", 13),
        ("gray_restart", "reset", "scan entropy", 5),
        ("gray_restart", "coverage", "assembly", 3),
        ("gray_zero_padding", "padding", "scan entropy", 9),
        ("gray_zero_padding", "trailing", "scan entropy", 9),
        (
            "gray_progressive_restart",
            "refinement_extra",
            "scan entropy",
            14,
        ),
        ("ycbcr444_sequential", "quantizer", "assembly", 2),
        ("quant16", "precision", "assembly", 1),
    ] {
        let mut session = engine
            .open_jpeg_reconstruction(case(name).input, Default::default())
            .unwrap();
        let plan = Arc::get_mut(session.plan.as_mut().unwrap()).unwrap();
        match fault {
            "code" => plan.scans.scans[0].tables.fill(0),
            "previous" => plan.scans.scans[0].tasks[0][1] = u32::MAX - 1,
            "extra" => plan.scans.scans[0].tasks[0][4] = 4,
            "marker" => plan.scans.scans[0].tasks[0][5] = 201,
            "scan" => plan.scans.scans[0].parameters = 64,
            "reset" => plan.scans.scans[0].tasks[0][8] = 2,
            "coverage" => plan.scans.masks[0].fill(0),
            "padding" => plan.padding_bits = 1,
            "trailing" => {
                plan.padding_bits += 1;
                plan.padding.push(0);
            }
            "refinement_extra" => {
                let scan = &mut plan.scans.scans[1];
                let low = scan.parameters >> 24;
                scan.parameters = (scan.parameters & 0xff00ffff) | ((low + 1) << 16);
                scan.tasks[0][4] = 1;
            }
            "quantizer" => plan.framing.quantizers[0][2] = 7,
            "precision" => plan.framing.quantizers[0][3] = 0,
            _ => unreachable!(),
        }
        let error = session
            .submit_next()
            .unwrap()
            .unwrap()
            .wait()
            .expect_err(&format!("{name} {fault}"));
        assert!(
            matches!(error, crate::Error::JpegReconstruction(Error::GpuStatus { stage, status })
            if stage == expected_stage && status[0] == expected_code),
            "{name} {fault}: {error:?}"
        );
        drop(session);
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn jpeg_exact_size_limits_and_two_dimensional_dispatch() {
    let backend = backend();
    let engine = VarDctSubmissionEngine::new(backend.clone()).unwrap();
    let sample = case("gray_restart");
    let mut session = engine
        .open_jpeg_reconstruction(sample.input, Default::default())
        .unwrap();
    let mut pending = session.submit_next().unwrap().unwrap();
    reach(&mut pending, 1);
    let operation = pending.operation().unwrap();
    let status = operation.status(operation.wait(), "count").unwrap();
    let raw_bytes = u64::from(status[2]);
    drop(pending);
    drop(session);
    for (raw_limit, output_limit, valid) in [
        (raw_bytes, sample.jpeg.len() as u64, true),
        (raw_bytes - 1, sample.jpeg.len() as u64, false),
        (raw_bytes, sample.jpeg.len() as u64 - 1, false),
    ] {
        let limits = JpegReconstructionLimits {
            max_raw_scan_bytes: raw_limit,
            max_output_bytes: output_limit,
            ..Default::default()
        };
        let mut session = engine
            .open_jpeg_reconstruction(sample.input, limits)
            .unwrap();
        session.resources.dispatch_width = 1; // Exercise rectangular dispatch and its incomplete final workgroup.
        let result = session.submit_next().unwrap().unwrap().wait();
        assert_eq!(
            result.is_ok(),
            valid,
            "raw {raw_limit}, output {output_limit}: {result:?}"
        );
        drop(result);
        drop(session);
        backend
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert_eq!(engine.in_flight_memory_stats().reserved_bytes, 0);
    }
    let mut session = engine
        .open_jpeg_reconstruction(case("gray_long_eob").input, Default::default())
        .unwrap();
    session.resources.dispatch_width = 7; // Over-dispatched groups must not corrupt hierarchical prefix scratch.
    let output = session.submit_next().unwrap().unwrap().wait().unwrap();
    let expected = case("gray_long_eob").jpeg;
    assert_eq!(output.output.byte_len(), expected.len() as u64);
    let bytes = jxl_test_support::gpu::buffer::read_bytes(&backend, output.output.buffer());
    assert_eq!(&bytes[..expected.len()], expected);
    assert!(bytes[expected.len()..].iter().all(|byte| *byte == 0));
}
