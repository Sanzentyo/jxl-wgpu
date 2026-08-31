#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use jxl_wgpu_decode::vardct::resource::{
    VarDctResourceBuffers, VarDctResourceConfig, VarDctResourceParams, VarDctResourcePipeline,
};
use wgpu::util::DeviceExt;

fn request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("VarDCT LF resource test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

#[test]
fn custom_lf_dequantization_and_correlation_execute_on_gpu() {
    let Some((device, queue)) = request_device() else {
        eprintln!("skipping VarDCT LF resource GPU test: no adapter is available");
        return;
    };
    let params = VarDctResourceParams::new(VarDctResourceConfig {
        block_extent: [2, 1],
        output_stride: 2,
        output_origin: [0, 0],
        global_scale: 4,
        quant_lf: 2,
        lf_dequantization: [0.0625, 0.125, 0.25],
        lf_correlation: [0.2, 0.8],
        extra_precision: 1,
    })
    .unwrap();
    assert_eq!(params.scales, [2.0, 4.0, 8.0, 2.0]);
    assert_eq!(params.smoothing_thresholds(), [4.0, 8.0, 16.0]);

    let quantized = [4i32, -2, 3, 5, -1, 6];
    let quantized_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom quantized LF samples"),
        contents: bytemuck::cast_slice(&quantized),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("custom dequantized LF samples"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("custom LF readback"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let pipeline = VarDctResourcePipeline::new(&device).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("custom LF resource commands"),
    });
    let _uniform = pipeline.encode(
        &device,
        &mut encoder,
        VarDctResourceBuffers {
            quantized_lf: &quantized_buffer,
            dequantized_lf: &output,
        },
        params,
    );
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 32);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("poll custom LF readback");
    receiver
        .recv()
        .expect("custom LF map callback")
        .expect("map custom LF output");
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let actual = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped);
    let expected = [[9.2, 16.0, 4.8, 0.0], [8.4, -8.0, 41.6, 0.0]];
    for (actual, expected) in actual.iter().zip(expected) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-5);
        }
    }
}
