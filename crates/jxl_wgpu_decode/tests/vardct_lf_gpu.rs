#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use jxl_wgpu::{DirectReadbackPolicy, WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::vardct::lf::{ADAPTIVE_LF_SHADER, AdaptiveLfParams};
use wgpu::util::DeviceExt;

fn backend() -> Option<WgpuBackend> {
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
        label: Some("jxl-wgpu adaptive LF smoothing test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            direct_readback_policy: DirectReadbackPolicy::Disabled,
            ..WgpuBackendConfig::default()
        },
    )
    .ok()
}

fn scalar_smoothing(
    width: usize,
    height: usize,
    input: &[[f32; 4]],
    scale: [f32; 3],
) -> Vec<[f32; 4]> {
    const SCALE_SELF: f32 = 0.052262735;
    const SCALE_SIDE: f32 = 0.2034514;
    const SCALE_DIAG: f32 = 0.03348292;
    let mut output = input.to_vec();
    if width <= 2 || height <= 2 {
        return output;
    }
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y * width + x;
            let mut weighted = [0.0; 3];
            for channel in 0..3 {
                let side = input[index - 1][channel]
                    + input[index + 1][channel]
                    + input[index - width][channel]
                    + input[index + width][channel];
                let diagonal = input[index - width - 1][channel]
                    + input[index - width + 1][channel]
                    + input[index + width - 1][channel]
                    + input[index + width + 1][channel];
                weighted[channel] =
                    input[index][channel] * SCALE_SELF + side * SCALE_SIDE + diagonal * SCALE_DIAG;
            }
            let gap = (0..3)
                .map(|channel| (weighted[channel] - input[index][channel]).abs() / scale[channel])
                .fold(0.5f32, f32::max);
            let gap_scale = (3.0 - 4.0 * gap).max(0.0);
            for channel in 0..3 {
                output[index][channel] =
                    (weighted[channel] - input[index][channel]) * gap_scale + input[index][channel];
            }
        }
    }
    output
}

#[test]
fn adaptive_lf_shader_is_portable_wgsl() {
    naga::front::wgsl::parse_str(ADAPTIVE_LF_SHADER).expect("adaptive LF shader parses");
    assert_eq!(std::mem::size_of::<AdaptiveLfParams>(), 32);
    assert_eq!(std::mem::align_of::<AdaptiveLfParams>(), 16);
}

#[test]
fn adaptive_lf_odd_tail_matches_scalar_oracle_on_gpu() {
    let Some(backend) = backend() else {
        return;
    };
    let (width, height) = (19u32, 7u32);
    let input = (0..width * height)
        .map(|index| {
            let x = (index % width) as f32;
            let y = (index / width) as f32;
            [
                (x * 0.03125 - y * 0.015625).sin(),
                x * 0.00390625 + y * 0.0078125,
                (x * y + 3.0) * 0.001953125,
                123.0,
            ]
        })
        .collect::<Vec<_>>();
    let lf_scale = [0.25, 0.5, 1.0];
    let expected = scalar_smoothing(width as usize, height as usize, &input, lf_scale);
    let params = AdaptiveLfParams::new(width, height, 0, 0, lf_scale);
    let input_buffer = backend
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adaptive LF input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output_size = u64::try_from(input.len() * std::mem::size_of::<[f32; 4]>()).unwrap();
    let output_buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("adaptive LF output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniform = backend
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("adaptive LF params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let shader = backend
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("adaptive LF shader"),
            source: wgpu::ShaderSource::Wgsl(ADAPTIVE_LF_SHADER.into()),
        });
    let pipeline = backend
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adaptive LF pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("adaptive_lf_smoothing"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let bind_group = backend
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adaptive LF bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("adaptive LF readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("adaptive LF commands"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("adaptive LF pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let [dispatch_x, dispatch_y] = params.dispatch();
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
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
        .expect("poll adaptive LF readback");
    receiver
        .recv()
        .expect("adaptive LF callback")
        .expect("map adaptive LF output");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped LF output");
    let actual = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped);
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        for channel in 0..4 {
            let tolerance = 2.0e-6f32.max(expected[channel].abs() * 2.0e-6);
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "sample {index} channel {channel}: GPU {}, scalar {}, tolerance {tolerance}",
                actual[channel],
                expected[channel],
            );
        }
    }
}

#[test]
fn subsampled_adaptive_lf_smoothing_matches_scalar_on_gpu() {
    use jxl_wgpu_decode::vardct::frontend::VarDctChannelShift;
    use jxl_wgpu_decode::vardct::lf::{AdaptiveLfPipeline, AdaptiveLfBuffers, SubsampledAdaptiveLfLayout};

    let Some(backend) = backend() else {
        return;
    };
    let device = backend.device();
    let queue = backend.queue();

    // 4:2:0 Subsampling: ch0(X) = 8x8, ch1(Y) = 16x16, ch2(B) = 8x8
    let shifts = [
        VarDctChannelShift { horizontal: 1, vertical: 1 },
        VarDctChannelShift { horizontal: 0, vertical: 0 },
        VarDctChannelShift { horizontal: 1, vertical: 1 },
    ];
    let layout = SubsampledAdaptiveLfLayout::new(16, 16, shifts);

    let lf_scale = [0.2f32, 0.3f32, 0.4f32];
    let pipeline = AdaptiveLfPipeline::new(device);

    for channel in [0, 1, 2] {
        let extent = layout.channel_extent(channel);
        let width = extent.width as usize;
        let height = extent.height as usize;
        let count = width * height;

        let mut input = Vec::with_capacity(count);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 17 + y * 23) % 256) as f32 / 255.0;
                input.push([v, v * 0.9, v * 0.8, 0.0]);
            }
        }

        let expected = scalar_smoothing(width, height, &input, lf_scale);

        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("subsampled LF input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("subsampled LF output"),
            size: (count * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("subsampled LF staging"),
            size: (count * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = layout.channel_params(channel, 0, 0, lf_scale);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("subsampled LF encoder"),
        });
        let _uniform = pipeline.encode(
            device,
            &mut encoder,
            AdaptiveLfBuffers {
                input: &in_buf,
                output: &out_buf,
            },
            params,
        );
        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            (count * std::mem::size_of::<[f32; 4]>()) as u64,
        );

        let submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::sync_channel(1);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        backend
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .expect("poll subsampled adaptive LF readback");
        receiver
            .recv()
            .expect("subsampled adaptive LF callback")
            .expect("map subsampled adaptive LF output");

        let mapped = staging.slice(..).get_mapped_range().expect("mapped range");
        let actual = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped);

        for (index, (expected_val, actual_val)) in expected.iter().zip(actual).enumerate() {
            let ch = channel;
            let tolerance = 2.0e-6f32.max(expected_val[ch].abs() * 2.0e-6);
            assert!(
                (actual_val[ch] - expected_val[ch]).abs() <= tolerance,
                "ch {channel} sample {index}: GPU {}, scalar {}, tolerance {tolerance}",
                actual_val[ch],
                expected_val[ch],
            );
        }
    }
}
