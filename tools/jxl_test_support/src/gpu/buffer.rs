//! Explicit readback of a retained byte/word lease for independent test comparisons.
use jxl_wgpu::{GpuBufferLease, WgpuBackend};

pub fn read_bytes(backend: &WgpuBackend, source: &GpuBufferLease) -> Vec<u8> {
    let buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("Original JPEG comparison"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = backend.device().create_command_encoder(&Default::default());
    let guard = source.try_acquire_gpu_submission().unwrap();
    commands.copy_buffer_to_buffer(source.as_wgpu_buffer(), 0, &buffer, 0, buffer.size());
    backend.queue().submit([commands.finish()]);
    drop(guard);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap()
    });
    backend
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let bytes = mapped.to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}
