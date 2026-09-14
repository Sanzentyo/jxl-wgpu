use super::*;
use jxl_wgpu::{MemoryBudget, MemoryBudgetError};
use jxl_wgpu_decode::{PrefetchBackpressure, WgpuSubmissionEngine};
use std::num::NonZeroUsize;

fn drain(backend: &WgpuBackend, budget: &MemoryBudget) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (budget.snapshot().reserved_bytes != 0 || backend.submission_poller().in_flight() != 0)
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(budget.snapshot().reserved_bytes, 0);
    assert_eq!(backend.submission_poller().in_flight(), 0);
}

#[test]
fn intermediate_storage_obeys_exact_budget_retry_retention_and_cancellation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in corpus::cases()
        .into_iter()
        .filter(|case| case.width == 129 && case.color_factor == 1)
    {
        let (data, _, expected) = case.load(Arithmetic::Wgsl, false);
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NativeFloat,
        )
        .unwrap()
        .with_extra_channel(0)
        .unwrap()
        .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
        let window = NonZeroU64::new(40).unwrap();
        let decoder = GpuDecoder::new(
            WgpuSubmissionEngine::new(backend.clone()).with_stream_window_limit(window),
        );
        let session = decoder.open(&data, request.clone()).unwrap();
        let stats = session.submission_session().memory_stats();
        let output = (case.width * case.height * 4) as u64;
        let alignment = u64::from(
            backend
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        );
        let source = (case.width.div_ceil(case.extra_factor as usize)
            * case.height.div_ceil(case.extra_factor as usize)
            * 4) as u64;
        let steps = if case.extra_factor == 8 { 1 } else { 2 };
        let intermediate = if steps == 1 { 0 } else { source * 64 };
        let weights = 6400
            + match case.extra_factor {
                16 => 400,
                32 => 1600,
                _ => 0,
            };
        assert_eq!(
            stats.modular_render_bytes,
            output.div_ceil(alignment) * alignment
                + source
                + intermediate
                + weights
                + 32
                + steps * 48
        );
        drop(session);
        let budget = MemoryBudget::new(NonZeroU64::new(stats.per_frame_bytes).unwrap());
        let decoder = GpuDecoder::new(
            WgpuSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
                .with_stream_window_limit(window),
        );
        let mut session = decoder.open(&data, request.clone()).unwrap();
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
        let pixels = planes::read(&backend, &frame.output().outputs[0]);
        for (index, &word) in pixels.iter().enumerate() {
            expected[3].samples[index].check(f32::from_bits(word), &case.name, index);
        }
        drop(session);
        assert_eq!(budget.snapshot().reserved_bytes, stats.output_lease_bytes);
        assert_eq!(planes::read(&backend, &frame.output().outputs[0]), pixels);
        drop(frame);
        drain(&backend, &budget);

        let mut session = planes::open_fragmented(&decoder, &data, request);
        assert_eq!(
            session
                .prefetch(NonZeroUsize::new(1).unwrap())
                .unwrap()
                .submitted,
            1
        );
        drop(session);
        drain(&backend, &budget);
    }
}
