#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;
use std::sync::mpsc;

use jxl_gpu_formats::{
    Channel, ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, PixelFormat, SampleKind,
    TransferFunction,
};
use jxl_wgpu::{WgpuAccelerator, WgpuAcceleratorConfig};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuSubmissionEngine};

const INDEXED_GRAY8: &[u8] = include_bytes!("../../../fixtures/gpu_gray8_lossless.jxl");

fn accelerator() -> Option<WgpuAccelerator> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu indexed Gray8 decode test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    let config = WgpuAcceleratorConfig {
        enable_timestamps: false,
        enable_direct_readback: false,
        ..WgpuAcceleratorConfig::default()
    };
    WgpuAccelerator::from_device(device, queue, info, config).ok()
}

fn read_output(accelerator: &WgpuAccelerator, output: &jxl_wgpu::GpuImageOutput) -> Vec<u8> {
    let staging = accelerator.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed Gray8 test output readback"),
        size: output.buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands =
        accelerator
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("indexed Gray8 test readback commands"),
            });
    commands.copy_buffer_to_buffer(&output.buffer, 0, &staging, 0, output.buffer.size());
    let (sender, receiver) = mpsc::sync_channel(1);
    commands.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    let submission = accelerator.queue().submit([commands.finish()]);
    accelerator
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU output readback completes");
    receiver
        .recv()
        .expect("mapping callback runs")
        .expect("output mapping succeeds");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped range is valid");
    let bytes = mapped[..output.layout.logical_size as usize].to_vec();
    drop(mapped);
    staging.unmap();
    bytes
}

fn expected_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(17 * 13);
    for y in 0..13u32 {
        for x in 0..17u32 {
            pixels.push(if y < 3 {
                0
            } else {
                ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
            });
        }
    }
    pixels
}

#[test]
fn indexed_jxl_entropy_and_gradient_reconstruct_exact_gray8_on_gpu() {
    let Some(accelerator) = accelerator() else {
        eprintln!("skipping indexed Gray8 decode test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(accelerator.clone());
    let request = GpuOutputRequest::new(PixelFormat::non_color(
        SampleKind::Unsigned,
        8,
        &[Channel::X],
    ));
    let mut session = decoder
        .open(INDEXED_GRAY8, request)
        .expect("indexed profile opens without CPU fallback");
    let frame = session
        .next_frame()
        .expect("GPU decode succeeds")
        .expect("one frame is returned");
    let output = &frame.output().outputs[0];
    assert_eq!(
        (output.layout.extent.width, output.layout.extent.height),
        (17, 13)
    );
    assert_eq!(read_output(&accelerator, output), expected_pixels());
}

#[test]
fn indexed_gray8_writes_native_limited_nv12_without_rgb_readback() {
    let Some(accelerator) = accelerator() else {
        eprintln!("skipping indexed Gray8 NV12 test: no wgpu adapter");
        return;
    };
    let mut spec = ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER);
    // Keep the source codestream's nonlinear sRGB transfer so this test isolates native YUV
    // packing/range conversion from transfer-function conversion.
    spec.transfer = TransferFunction::Srgb;
    let format = PixelFormat::nv12(ColorSpecification::Defined(spec));
    let decoder = GpuDecoder::wgpu(accelerator.clone());
    let mut session = decoder
        .open(INDEXED_GRAY8, GpuOutputRequest::new(format))
        .expect("native NV12 request is supported");
    let frame = session
        .next_frame()
        .expect("GPU NV12 decode succeeds")
        .expect("one frame is returned");
    let output = &frame.output().outputs[0];
    let bytes = read_output(&accelerator, output);
    let y_plane = &output.layout.planes[0];
    for (index, sample) in expected_pixels().into_iter().enumerate() {
        let expected = (16.0 + 219.0 * f32::from(sample) / 255.0).round() as u8;
        assert_eq!(bytes[y_plane.offset as usize + index], expected);
    }
    let uv_plane = &output.layout.planes[1];
    let uv = &bytes[uv_plane.byte_range().unwrap().start as usize
        ..uv_plane.byte_range().unwrap().end as usize];
    assert!(uv.iter().all(|&sample| sample == 128));
}

#[test]
fn indexed_gpu_future_reports_and_releases_bounded_memory() {
    let Some(accelerator) = accelerator() else {
        eprintln!("skipping indexed Gray8 async test: no wgpu adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::with_memory_budget(
        accelerator.clone(),
        NonZeroU64::new(1024 * 1024).unwrap(),
    );
    let decoder = GpuDecoder::new(engine);
    let request = GpuOutputRequest::new(PixelFormat::non_color(
        SampleKind::Unsigned,
        8,
        &[Channel::X],
    ));
    let mut session = decoder.open(INDEXED_GRAY8, request).unwrap();
    let stats = session.submission_session().memory_stats();
    assert!(stats.per_frame_bytes > 128 * 1024);
    assert_eq!(stats.max_in_flight, 2);
    assert_eq!(stats.reserved_bytes, stats.per_frame_bytes * 2);
    assert_eq!(
        decoder.engine().reserved_session_bytes(),
        stats.reserved_bytes
    );

    let frame = pollster::block_on(session.next_frame_async())
        .expect("runtime-neutral GPU future succeeds")
        .expect("one frame is returned");
    assert_eq!(
        read_output(&accelerator, &frame.output().outputs[0]),
        expected_pixels()
    );
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().reserved_session_bytes(), 0);
}
