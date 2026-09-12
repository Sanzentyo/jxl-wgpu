use super::*;
use jxl_wgpu::{MemoryBudget, MemoryBudgetError};
use jxl_wgpu_decode::{PrefetchBackpressure, WgpuSubmissionEngine};
use std::num::NonZeroUsize;

pub(super) fn drain(backend: &WgpuBackend, expected: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (backend.transient_memory_budget().snapshot().reserved_bytes != expected
        || backend.submission_poller().in_flight() != 0)
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        expected
    );
    assert_eq!(backend.submission_poller().in_flight(), 0);
}

#[test]
fn component_expansion_is_fully_admitted_and_survives_retry_and_cancellation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let mut equal_grid_render_bytes = None;
    for name in [
        "sampling_000",
        "sampling_010",
        "sampling_123",
        "restoration_1",
        "restoration_2",
        "restoration_3",
        "resampling_2",
        "resampling_4",
        "resampling_8",
        "global_prefix",
        "lf_extras",
    ] {
        let (bytes, expected) = reference(name);
        let decoder = GpuDecoder::new(
            WgpuSubmissionEngine::new(backend.clone())
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        let session = decoder.open(&bytes, color_request()).unwrap();
        let stats = session.submission_session().memory_stats();
        if name == "global_prefix" {
            assert_eq!(stats.global_reconstruction_sample_words, 129 * 10);
        }
        if name == "lf_extras" {
            assert_eq!(stats.global_reconstruction_sample_words, 0);
            // A 128-pixel group has a 1024-pixel LF span; width 2051 crosses two boundaries.
            assert_eq!(stats.low_frequency_group_stream_count, 3);
        }
        if name == "sampling_000" {
            equal_grid_render_bytes = Some(stats.modular_render_bytes);
        }
        if let Some(delta) = match name {
            "sampling_010" => Some(1584),
            "sampling_123" => Some(2988),
            _ => None,
        } {
            // Relative to 37x19 4:4:4: actual normalized component grids, two full-size
            // expanded destinations and two 32-byte uniforms. Packing is identical.
            assert_eq!(
                stats.modular_render_bytes - equal_grid_render_bytes.unwrap(),
                delta
            );
        }
        drop(session);
        let budget = MemoryBudget::new(NonZeroU64::new(stats.per_frame_bytes).unwrap());
        let decoder = GpuDecoder::new(
            WgpuSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
                .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
        );
        let mut session = decoder.open(&bytes, color_request()).unwrap();
        let blocker = budget.try_reserve(1).unwrap();
        let pressure = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(pressure.submitted, 0);
        assert!(matches!(
            pressure.backpressure,
            Some(PrefetchBackpressure::Memory(
                MemoryBudgetError::Exhausted { .. }
            ))
        ));
        assert_eq!(budget.snapshot().reserved_bytes, 1);
        drop(blocker);
        let frame = session.next_frame().unwrap().unwrap();
        let actual = planes::read(&backend, &frame.output().outputs[0]);
        require_samples(name, &actual, expected[..actual.len()].iter().copied());
        drop(session);
        drain(&backend, 0);
        assert_eq!(budget.snapshot().reserved_bytes, stats.output_lease_bytes);
        assert_eq!(planes::read(&backend, &frame.output().outputs[0]), actual);
        drop(frame);
        assert_eq!(budget.snapshot().reserved_bytes, 0);

        let mut session = planes::open_fragmented(&decoder, &bytes, color_request());
        assert_eq!(
            session
                .prefetch(NonZeroUsize::new(1).unwrap())
                .unwrap()
                .submitted,
            1
        );
        drop(session);
        drain(&backend, 0);
        assert_eq!(budget.snapshot().reserved_bytes, 0);
    }
}
