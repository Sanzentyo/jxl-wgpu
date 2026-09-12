//! Shared transport helpers for private frame-executor tests.
use jxl_wgpu::{GpuImageFrame, ImageReadbackPipeline, WgpuBackend};
use std::sync::Arc;

pub(super) fn fixture(name: &str) -> Arc<[u8]> {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("test-data/{name}.jxl.hex")),
    )
    .unwrap()
    .split_whitespace()
    .collect::<String>()
    .as_bytes()
    .as_chunks::<2>()
    .0
    .iter()
    .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
    .collect::<Vec<_>>()
    .into()
}

pub(super) fn drain(backend: &WgpuBackend) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while backend.transient_memory_budget().snapshot().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}

pub(super) fn read(backend: &WgpuBackend, frame: &GpuImageFrame) -> Vec<u8> {
    ImageReadbackPipeline::new(backend)
        .submit(frame)
        .unwrap()
        .wait()
        .unwrap()
        .frame
        .outputs[0]
        .bytes
        .clone()
}
