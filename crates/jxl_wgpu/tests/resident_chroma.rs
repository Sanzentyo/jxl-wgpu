#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use jxl_wgpu::{
    DirectReadbackPolicy, ResidentChromaShift, ResidentChromaUpsampleInputs,
    ResidentChromaUpsamplePipeline, ResidentF32Plane, ResidentStorageBinding, WgpuBackend,
    WgpuBackendConfig,
};
use wgpu::util::DeviceExt;

fn test_backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        ..WgpuBackendConfig::default()
    })) {
        Ok(backend) => Some(backend),
        Err(jxl_wgpu::Error::NoAdapter) => {
            eprintln!("skipping resident chroma test: no compatible adapter");
            None
        }
        Err(error) => panic!("request resident chroma test backend: {error}"),
    }
}

fn source_value(width: u32, x: u32, y: u32) -> f32 {
    (y * width + x) as f32
}

fn interpolate_axis(position: u32, size: u32) -> (u32, u32, f32) {
    let current = position / 2;
    if position.is_multiple_of(2) {
        (current.saturating_sub(1), current, 0.75)
    } else {
        (current, (current + 1).min(size - 1), 0.25)
    }
}

fn expected(
    input_width: u32,
    input_height: u32,
    x: u32,
    y: u32,
    shift: ResidentChromaShift,
) -> f32 {
    let (x0, x1, x_weight) = if shift.horizontal {
        interpolate_axis(x, input_width)
    } else {
        (x, x, 0.0)
    };
    let (y0, y1, y_weight) = if shift.vertical {
        interpolate_axis(y, input_height)
    } else {
        (y, y, 0.0)
    };
    let top = source_value(input_width, x0, y0) * (1.0 - x_weight)
        + source_value(input_width, x1, y0) * x_weight;
    let bottom = source_value(input_width, x0, y1) * (1.0 - x_weight)
        + source_value(input_width, x1, y1) * x_weight;
    top * (1.0 - y_weight) + bottom * y_weight
}

#[test]
fn resident_chroma_upsample_matches_codec_edges_on_gpu() {
    let Some(backend) = test_backend() else {
        return;
    };
    let pipeline = ResidentChromaUpsamplePipeline::new(backend.device()).unwrap();
    for shift in [
        ResidentChromaShift {
            horizontal: true,
            vertical: false,
        },
        ResidentChromaShift {
            horizontal: false,
            vertical: true,
        },
        ResidentChromaShift {
            horizontal: true,
            vertical: true,
        },
    ] {
        let output_width = 5;
        let output_height = 3;
        let input_width = if shift.horizontal { 3 } else { output_width };
        let input_height = if shift.vertical { 2 } else { output_height };
        let input = (0..input_width * input_height)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let input_buffer = backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resident chroma test input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_bytes = u64::from(output_width * output_height) * 4;
        let output_buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident chroma test output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident chroma test staging"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            backend
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("resident chroma test commands"),
                });
        let uniform = pipeline
            .encode(
                backend.device(),
                &mut encoder,
                ResidentChromaUpsampleInputs {
                    input: ResidentF32Plane {
                        storage: ResidentStorageBinding::entire(&input_buffer).unwrap(),
                        width: input_width,
                        height: input_height,
                        stride: input_width,
                    },
                    output: ResidentF32Plane {
                        storage: ResidentStorageBinding::entire(&output_buffer).unwrap(),
                        width: output_width,
                        height: output_height,
                        stride: output_width,
                    },
                    shift,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_bytes);
        let submission = backend.queue().submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        let actual: &[f32] = bytemuck::cast_slice(&mapped);
        for y in 0..output_height {
            for x in 0..output_width {
                let index = (y * output_width + x) as usize;
                let expected = expected(input_width, input_height, x, y, shift);
                assert!((actual[index] - expected).abs() <= 1.0e-6);
            }
        }
        drop(mapped);
        staging.unmap();
        drop(uniform);
    }
}
