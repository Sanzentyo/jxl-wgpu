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
    while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.format, format);
        let pixels = planes::read(backend, output);
        held.push((frame, pixels));
    }
    drop(session);
    let frames = held
        .into_iter()
        .map(|(frame, pixels)| {
            assert_eq!(planes::read(backend, &frame.output().outputs[0]), pixels);
            pixels
        })
        .collect();
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    frames
}
