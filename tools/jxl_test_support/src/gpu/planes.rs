//! Whole/fragmented decode and byte-exact GPU readback shared by precision conformance tests.
use jxl_wgpu::{GpuImageOutput, WgpuBackend};
use jxl_wgpu_decode::{
    GpuDecodeSession, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine, WgpuDecodeSubmissionSession,
};
use std::sync::Arc;
pub fn open_fragmented(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSubmissionSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport =
        jxl_gpu_bitstream::ContainerStreamScanner::new(decoder.container_stream_limits());
    for chunk in data.chunks(43) {
        for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    stream.finish().unwrap()
}

pub fn read_bytes(backend: &WgpuBackend, output: &GpuImageOutput) -> Vec<u8> {
    let device = backend.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample precision test readback"),
        size: output.buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = device.create_command_encoder(&Default::default());
    commands.copy_buffer_to_buffer(output.buffer.as_wgpu_buffer(), 0, &buffer, 0, buffer.size());
    backend.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    let bytes = mapped[..output.layout.logical_size as usize].to_vec();
    drop(mapped);
    buffer.unmap();
    bytes
}

pub fn read(backend: &WgpuBackend, output: &GpuImageOutput) -> Vec<u32> {
    read_bytes(backend, output)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}
