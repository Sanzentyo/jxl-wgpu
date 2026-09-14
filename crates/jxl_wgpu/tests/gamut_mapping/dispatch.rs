use jxl_wgpu::{GAMUT_MAPPING_SHADER, GamutMappingParams, WgpuBackend};
use wgpu::util::DeviceExt;

pub(super) fn pipeline(device: &wgpu::Device, workgroup: u32) -> wgpu::ComputePipeline {
    let shader = format!(
        "{GAMUT_MAPPING_SHADER}\n{}",
        r#"
        override workgroup: u32 = 32u;
        @group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
        @group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
        @group(0) @binding(2) var<uniform> params: GamutMappingParams;
        @compute @workgroup_size(workgroup)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            if id.x >= arrayLength(&input) { return; }
            let pixel = input[id.x];
            output[id.x] = vec4<f32>(gamut_map_rgb(pixel.xyz, params), pixel.w);
        }
    "#
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("independent gamut dispatch"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gamut variant"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[("workgroup", f64::from(workgroup))],
            ..Default::default()
        },
        cache: None,
    })
}

pub(super) fn run(
    backend: &WgpuBackend,
    pipeline: &wgpu::ComputePipeline,
    workgroup: u32,
    input: &[[f32; 4]],
    params: GamutMappingParams,
) -> Vec<[f32; 4]> {
    let device = backend.device();
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamut input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let words = input.len() * 4;
    let guard = [0x7fc0_4321_u32; 64];
    let initial: Vec<_> = guard
        .into_iter()
        .chain(std::iter::repeat_n(0, words))
        .chain(guard)
        .collect();
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("guarded gamut output"),
        contents: bytemuck::cast_slice(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamut parameters"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gamut bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &output,
                    offset: 256,
                    size: std::num::NonZeroU64::new(words as u64 * 4),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gamut readback"),
        size: output.size(),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((input.len() as u32).div_ceil(workgroup), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output.size());
    backend.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let data: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(&data[..64], &guard);
    assert_eq!(&data[64 + words..], &guard);
    let pixels = data[64..64 + words]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| p.map(f32::from_bits))
        .collect();
    drop(mapped);
    staging.unmap();
    pixels
}
