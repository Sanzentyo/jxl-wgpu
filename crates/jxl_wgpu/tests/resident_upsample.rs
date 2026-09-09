#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;
use std::sync::mpsc;

use jxl_wgpu::{
    ResidentF32Plane, ResidentStorageBinding, ResidentUpsampleError, ResidentUpsampleInputs,
    ResidentUpsampleKernel, ResidentUpsamplePipeline, ResidentUpsampleSource, WgpuBackend,
};
use wgpu::util::DeviceExt;

fn backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(Default::default())) {
        Ok(backend) => Some(backend),
        Err(jxl_wgpu::Error::NoAdapter) => None,
        Err(error) => panic!("resident upsampling adapter: {error}"),
    }
}

#[test]
fn strided_scalar_views_match_planar_filtering_for_all_factors_and_mirrored_edges() {
    let Some(backend) = backend() else { return };
    let device = backend.device();
    let pipeline = ResidentUpsamplePipeline::new(device).unwrap();
    for factor in [2, 4, 8] {
        let side = factor * 5 / 2;
        let compact = (0..side * (side + 1) / 2)
            .map(|i| 0.035 + (i % 11) as f32 * 0.001)
            .collect::<Vec<_>>();
        let weights = ResidentUpsampleKernel::from_compact(factor, &compact)
            .unwrap()
            .upload(device)
            .unwrap();
        for (width, height) in [(1, 1), (1, 5), (7, 1), (5, 3)] {
            let planar = (0..width * height)
                .map(|i| ((i * 17 + 3) % 29) as f32 / 29.0 - 0.4)
                .collect::<Vec<_>>();
            let binding_offset = u64::from(device.limits().min_storage_buffer_offset_alignment);
            let prefix = (binding_offset / 4) as usize;
            let row_stride = width * 4 + 5;
            let offset = 3;
            let mut interleaved = vec![f32::NAN; prefix + (offset + row_stride * height) as usize];
            for y in 0..height {
                for x in 0..width {
                    interleaved[prefix + (offset + y * row_stride + x * 4) as usize] =
                        planar[(y * width + x) as usize];
                }
            }
            let upload = |label, values: &[f32]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let tight = upload("upsampling planar oracle", &planar);
            let strided = upload("upsampling poisoned strided view", &interleaved);
            let output_width = width * factor - 1;
            let output_height = height * factor - 1;
            let output_stride = output_width + 3;
            let output_bytes = u64::from(output_stride * output_height) * 4;
            let allocate_output = || {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("upsampling output with row padding"),
                    size: output_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            let outputs = [allocate_output(), allocate_output()];
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upsampling comparison"),
                size: output_bytes * 2,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let sources = [
                ResidentF32Plane {
                    storage: ResidentStorageBinding::entire(&tight).unwrap(),
                    width,
                    height,
                    stride: width,
                }
                .into(),
                ResidentUpsampleSource {
                    storage: ResidentStorageBinding {
                        buffer: &strided,
                        offset: binding_offset,
                        size: NonZeroU64::new(strided.size() - binding_offset).unwrap(),
                    },
                    width,
                    height,
                    offset,
                    row_stride,
                    sample_stride: 4,
                },
            ];
            let mut commands = device.create_command_encoder(&Default::default());
            let mut uniforms = Vec::new();
            for (index, input) in sources.into_iter().enumerate() {
                uniforms.push(
                    pipeline
                        .encode(
                            device,
                            &mut commands,
                            ResidentUpsampleInputs {
                                input,
                                weights: &weights,
                                output: ResidentF32Plane {
                                    storage: ResidentStorageBinding::entire(&outputs[index])
                                        .unwrap(),
                                    width: output_width,
                                    height: output_height,
                                    stride: output_stride,
                                },
                            },
                        )
                        .unwrap(),
                );
                commands.copy_buffer_to_buffer(
                    &outputs[index],
                    0,
                    &staging,
                    index as u64 * output_bytes,
                    output_bytes,
                );
            }
            let submission = backend.queue().submit([commands.finish()]);
            let (sender, receiver) = mpsc::sync_channel(1);
            staging.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let (expected, actual) = mapped.split_at(output_bytes as usize);
            assert_eq!(actual, expected, "{factor}x, {width}x{height}");
            for value in actual.chunks_exact(4) {
                assert!(f32::from_le_bytes(value.try_into().unwrap()).is_finite());
            }
            drop(mapped);
            staging.unmap();
            drop(uniforms);
        }
    }
}

#[test]
fn invalid_scalar_addressing_is_rejected_before_recording_gpu_work() {
    let Some(backend) = backend() else { return };
    let device = backend.device();
    let pipeline = ResidentUpsamplePipeline::new(device).unwrap();
    let weights = ResidentUpsampleKernel::from_compact(2, &[0.04; 15])
        .unwrap()
        .upload(device)
        .unwrap();
    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 128,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let valid = ResidentUpsampleSource {
        storage: ResidentStorageBinding::entire(&input).unwrap(),
        width: 2,
        height: 2,
        offset: 3,
        row_stride: 8,
        sample_stride: 4,
    };
    for source in [
        ResidentUpsampleSource { width: 0, ..valid },
        ResidentUpsampleSource { height: 0, ..valid },
        ResidentUpsampleSource {
            sample_stride: 0,
            ..valid
        },
        ResidentUpsampleSource {
            row_stride: 4,
            ..valid
        },
        ResidentUpsampleSource {
            offset: u32::MAX,
            ..valid
        },
        ResidentUpsampleSource {
            row_stride: u32::MAX,
            ..valid
        },
        ResidentUpsampleSource {
            sample_stride: u32::MAX,
            ..valid
        },
        ResidentUpsampleSource {
            offset: 32,
            ..valid
        },
        ResidentUpsampleSource {
            storage: ResidentStorageBinding {
                size: NonZeroU64::new(12).unwrap(),
                ..valid.storage
            },
            ..valid
        },
        ResidentUpsampleSource {
            storage: ResidentStorageBinding {
                offset: 4,
                ..valid.storage
            },
            ..valid
        },
    ] {
        let mut commands = device.create_command_encoder(&Default::default());
        assert!(matches!(
            pipeline.encode(
                device,
                &mut commands,
                ResidentUpsampleInputs {
                    input: source,
                    weights: &weights,
                    output: ResidentF32Plane {
                        storage: ResidentStorageBinding::entire(&output).unwrap(),
                        width: 4,
                        height: 4,
                        stride: 4
                    },
                }
            ),
            Err(ResidentUpsampleError::PlaneGeometry { role: "input" }
                | ResidentUpsampleError::Binding { role: "input" })
        ));
    }
}
