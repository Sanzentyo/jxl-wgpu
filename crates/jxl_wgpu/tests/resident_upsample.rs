#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use jxl_wgpu::{
    DirectReadbackPolicy, ResidentChannelUpsampleInputs, ResidentChannelUpsamplePipeline,
    ResidentF32Plane, ResidentImageUpsampleInputs, ResidentImageUpsamplePipeline,
    ResidentImageUpsampleWeights, ResidentStorageBinding, WgpuBackend, WgpuBackendConfig,
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
            eprintln!("skipping resident upsample test: no compatible adapter");
            None
        }
        Err(error) => panic!("request resident upsample test backend: {error}"),
    }
}

fn readback_plane(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test readback encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    let submission = queue.submit([encoder.finish()]);

    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..size)
        .map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();

    let view = staging.slice(..size).get_mapped_range().unwrap();
    let floats: &[f32] = bytemuck::cast_slice(&view);
    let result = floats.to_vec();
    drop(view);
    staging.unmap();
    result
}

#[test]
fn resident_channel_upsample_matches_image_upsample_on_gpu() {
    let Some(backend) = test_backend() else {
        return;
    };
    let device = backend.device();
    let queue = backend.queue();

    let channel_pipeline = ResidentChannelUpsamplePipeline::new(device).unwrap();
    let image_pipeline = ResidentImageUpsamplePipeline::new(device).unwrap();

    let in_w = 8_u32;
    let in_h = 8_u32;
    let input_scalars = (in_w * in_h) as usize;
    let input_bytes = (input_scalars * 4) as u64;

    let input_data: Vec<f32> = (0..input_scalars)
        .map(|i| ((i * 17 + 3) % 256) as f32 / 255.0)
        .collect();

    for (factor, compact_count) in [(2, 15), (4, 55), (8, 210)] {
        let out_w = in_w * factor;
        let out_h = in_h * factor;
        let output_scalars = (out_w * out_h) as usize;
        let output_bytes = (output_scalars * 4) as u64;

        // Construct normalized triangle weights
        let mut compact: Vec<f32> = (0..compact_count).map(|i| 1.0 / ((i + 1) as f32)).collect();
        compact[0] = 1.0;

        let weights = ResidentImageUpsampleWeights::new(factor, &compact).unwrap();

        // Create GPU buffers
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("input buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let channel_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("channel out buffer"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let image_out_buf0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image out buffer 0"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let image_out_buf1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image out buffer 1"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let image_out_buf2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image out buffer 2"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let channel_input_plane = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &input_buf,
                offset: 0,
                size: std::num::NonZeroU64::new(input_bytes).unwrap(),
            },
            width: in_w,
            height: in_h,
            stride: in_w,
        };

        let channel_output_plane = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &channel_out_buf,
                offset: 0,
                size: std::num::NonZeroU64::new(output_bytes).unwrap(),
            },
            width: out_w,
            height: out_h,
            stride: out_w,
        };

        let image_output_planes = [
            ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer: &image_out_buf0,
                    offset: 0,
                    size: std::num::NonZeroU64::new(output_bytes).unwrap(),
                },
                width: out_w,
                height: out_h,
                stride: out_w,
            },
            ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer: &image_out_buf1,
                    offset: 0,
                    size: std::num::NonZeroU64::new(output_bytes).unwrap(),
                },
                width: out_w,
                height: out_h,
                stride: out_w,
            },
            ResidentF32Plane {
                storage: ResidentStorageBinding {
                    buffer: &image_out_buf2,
                    offset: 0,
                    size: std::num::NonZeroU64::new(output_bytes).unwrap(),
                },
                width: out_w,
                height: out_h,
                stride: out_w,
            },
        ];

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("upsample test encoder"),
        });

        // 1. Dispatch single-channel upsample
        let _channel_res = channel_pipeline
            .encode(
                device,
                &mut encoder,
                ResidentChannelUpsampleInputs {
                    input: channel_input_plane,
                    output: channel_output_plane,
                    weights: &weights,
                },
            )
            .unwrap();

        // 2. Dispatch 3-plane upsample with same input in all 3 planes
        let _image_res = image_pipeline
            .encode(
                device,
                &mut encoder,
                ResidentImageUpsampleInputs {
                    inputs: [
                        channel_input_plane,
                        channel_input_plane,
                        channel_input_plane,
                    ],
                    outputs: image_output_planes,
                    weights: &weights,
                },
            )
            .unwrap();

        queue.submit([encoder.finish()]);

        let channel_out = readback_plane(device, queue, &channel_out_buf, output_bytes);
        let image_out0 = readback_plane(device, queue, &image_out_buf0, output_bytes);

        assert_eq!(channel_out.len(), output_scalars);
        assert_eq!(image_out0.len(), output_scalars);

        // Bit-exact agreement between single channel upsample and 3-plane upsample plane 0
        for i in 0..output_scalars {
            assert_eq!(
                channel_out[i], image_out0[i],
                "mismatch at index {i} for factor {factor}"
            );
            assert!(channel_out[i].is_finite());
            assert!(channel_out[i] >= 0.0 && channel_out[i] <= 1.0);
        }
    }
}
