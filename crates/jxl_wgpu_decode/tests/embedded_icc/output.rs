use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use std::num::NonZeroU64;

pub(super) fn frames(
    backend: &WgpuBackend,
    data: &[u8],
    request: GpuOutputRequest,
    limit: Option<NonZeroU64>,
) -> Vec<Vec<u32>> {
    frames_bytes(backend, data, request, limit)
        .into_iter()
        .map(|bytes| {
            let (words, tail) = bytes.as_chunks::<4>();
            assert!(tail.is_empty(), "word output must contain complete words");
            words.iter().map(|word| u32::from_le_bytes(*word)).collect()
        })
        .collect()
}

pub(super) fn frames_bytes(
    backend: &WgpuBackend,
    data: &[u8],
    request: GpuOutputRequest,
    limit: Option<NonZeroU64>,
) -> Vec<Vec<u8>> {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if let Some(limit) = limit {
        engine = engine.with_stream_window_limit(limit);
    }
    let decoder = GpuDecoder::new(engine);
    let format = request.format().clone();
    let mut session = if limit.is_some() {
        planes::open_fragmented(&decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let mut held = Vec::new();
    // Every returned frame is retained. Fail explicitly if a producer's local capacity
    // accidentally narrows a sequence's public window, rather than parking on our own leases.
    assert!(
        session
            .metadata()
            .frame_count_hint
            .is_none_or(|count| count <= session.resolved_frame_slots().get())
    );
    while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.format, format);
        let pixels = planes::read_bytes(backend, output);
        held.push((frame, pixels));
    }
    drop(session);
    let frames = held
        .into_iter()
        .map(|(frame, pixels)| {
            assert_eq!(
                planes::read_bytes(backend, &frame.output().outputs[0]),
                pixels
            );
            pixels
        })
        .collect();
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    frames
}
