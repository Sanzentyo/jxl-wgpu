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

#[test]
fn resident_upsample_rejects_aliased_buffers() {
    let Some(backend) = test_backend() else {
        return;
    };
    let device = backend.device();

    let channel_pipeline = ResidentChannelUpsamplePipeline::new(device).unwrap();
    let image_pipeline = ResidentImageUpsamplePipeline::new(device).unwrap();

    let weights = ResidentImageUpsampleWeights::new(2, &[1.0; 15]).unwrap();

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shared test buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let other_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("other test buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let plane_in = ResidentF32Plane {
        storage: ResidentStorageBinding {
            buffer: &buf,
            offset: 0,
            size: std::num::NonZeroU64::new(256).unwrap(),
        },
        width: 8,
        height: 8,
        stride: 8,
    };
    let plane_out_alias = ResidentF32Plane {
        storage: ResidentStorageBinding {
            buffer: &buf,
            offset: 0,
            size: std::num::NonZeroU64::new(1024).unwrap(),
        },
        width: 16,
        height: 16,
        stride: 16,
    };
    let plane_out_other = ResidentF32Plane {
        storage: ResidentStorageBinding {
            buffer: &other_buf,
            offset: 0,
            size: std::num::NonZeroU64::new(1024).unwrap(),
        },
        width: 16,
        height: 16,
        stride: 16,
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // 1. Channel upsample: input and output alias
    let err = channel_pipeline.encode(
        device,
        &mut encoder,
        ResidentChannelUpsampleInputs {
            input: plane_in,
            output: plane_out_alias,
            weights: &weights,
        },
    );
    assert!(matches!(
        err,
        Err(jxl_wgpu::ResidentImageUpsampleError::Aliasing { .. })
    ));

    // 2. Image upsample: output planes alias each other
    let err = image_pipeline.encode(
        device,
        &mut encoder,
        ResidentImageUpsampleInputs {
            inputs: [plane_in, plane_in, plane_in],
            outputs: [plane_out_alias, plane_out_alias, plane_out_other],
            weights: &weights,
        },
    );
    assert!(matches!(
        err,
        Err(jxl_wgpu::ResidentImageUpsampleError::Aliasing { .. })
    ));

    // 3. Image upsample: output aliases input
    let err = image_pipeline.encode(
        device,
        &mut encoder,
        ResidentImageUpsampleInputs {
            inputs: [plane_in, plane_in, plane_in],
            outputs: [plane_out_alias, plane_out_other, plane_out_other],
            weights: &weights,
        },
    );
    assert!(matches!(
        err,
        Err(jxl_wgpu::ResidentImageUpsampleError::Aliasing { .. })
    ));
}

#[test]
fn resident_upsample_supports_arbitrary_and_odd_extents_on_gpu() {
    let Some(backend) = test_backend() else {
        return;
    };
    let device = backend.device();
    let queue = backend.queue();

    let channel_pipeline = ResidentChannelUpsamplePipeline::new(device).unwrap();
    let image_pipeline = ResidentImageUpsamplePipeline::new(device).unwrap();

    let test_geometries = [
        // Tiny and edge dimensions
        (1_u32, 1_u32, 2_u32, 1_u32, 1_u32),
        (1_u32, 17_u32, 2_u32, 2_u32, 33_u32),
        (17_u32, 1_u32, 4_u32, 65_u32, 3_u32),
        // Odd / non-multiple dimensions
        (25_u32, 19_u32, 2_u32, 49_u32, 37_u32),
        (33_u32, 21_u32, 4_u32, 131_u32, 81_u32),
        (19_u32, 23_u32, 8_u32, 151_u32, 183_u32),
    ];

    for (in_w, in_h, factor, out_w, out_h) in test_geometries {
        let compact_count = match factor {
            2 => 15,
            4 => 55,
            8 => 210,
            _ => unreachable!(),
        };
        let compact: Vec<f32> = (0..compact_count).map(|i| 1.0 / ((i + 1) as f32)).collect();
        let weights = ResidentImageUpsampleWeights::new(factor, &compact).unwrap();

        let in_stride = in_w + 3;
        let out_stride = out_w + 5;
        let in_scalars = (in_stride * in_h) as usize;
        let out_scalars = (out_stride * out_h) as usize;

        let in_data: Vec<f32> = (0..in_scalars)
            .map(|i| ((i * 13 + 7) % 256) as f32 / 255.0)
            .collect();

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("odd extent in"),
            contents: bytemuck::cast_slice(&in_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf_channel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("odd extent out channel"),
            size: (out_scalars * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf_image0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("odd extent out image 0"),
            size: (out_scalars * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_buf_image1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("odd extent out image 1"),
            size: (out_scalars * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let out_buf_image2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("odd extent out image 2"),
            size: (out_scalars * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let plane_in = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &in_buf,
                offset: 0,
                size: std::num::NonZeroU64::new((in_scalars * 4) as u64).unwrap(),
            },
            width: in_w,
            height: in_h,
            stride: in_stride,
        };
        let plane_out_chan = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &out_buf_channel,
                offset: 0,
                size: std::num::NonZeroU64::new((out_scalars * 4) as u64).unwrap(),
            },
            width: out_w,
            height: out_h,
            stride: out_stride,
        };
        let plane_out_img0 = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &out_buf_image0,
                offset: 0,
                size: std::num::NonZeroU64::new((out_scalars * 4) as u64).unwrap(),
            },
            width: out_w,
            height: out_h,
            stride: out_stride,
        };
        let plane_out_img1 = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &out_buf_image1,
                offset: 0,
                size: std::num::NonZeroU64::new((out_scalars * 4) as u64).unwrap(),
            },
            width: out_w,
            height: out_h,
            stride: out_stride,
        };
        let plane_out_img2 = ResidentF32Plane {
            storage: ResidentStorageBinding {
                buffer: &out_buf_image2,
                offset: 0,
                size: std::num::NonZeroU64::new((out_scalars * 4) as u64).unwrap(),
            },
            width: out_w,
            height: out_h,
            stride: out_stride,
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        channel_pipeline
            .encode(
                device,
                &mut encoder,
                ResidentChannelUpsampleInputs {
                    input: plane_in,
                    output: plane_out_chan,
                    weights: &weights,
                },
            )
            .unwrap();
        image_pipeline
            .encode(
                device,
                &mut encoder,
                ResidentImageUpsampleInputs {
                    inputs: [plane_in, plane_in, plane_in],
                    outputs: [plane_out_img0, plane_out_img1, plane_out_img2],
                    weights: &weights,
                },
            )
            .unwrap();
        queue.submit([encoder.finish()]);

        let channel_floats =
            readback_plane(device, queue, &out_buf_channel, (out_scalars * 4) as u64);
        let image_floats = readback_plane(device, queue, &out_buf_image0, (out_scalars * 4) as u64);

        for y in 0..out_h {
            for x in 0..out_w {
                let idx = (y * out_stride + x) as usize;
                assert_eq!(
                    channel_floats[idx], image_floats[idx],
                    "mismatch at ({x}, {y}) for geom in={in_w}x{in_h} out={out_w}x{out_h} factor={factor}"
                );
            }
        }
    }
}
