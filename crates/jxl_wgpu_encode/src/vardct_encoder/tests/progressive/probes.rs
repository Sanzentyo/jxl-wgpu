use super::*;

#[test]
fn progressive_signed_endpoints_and_spectral_rectangles_match_i64() {
    let context = test_context().expect("actual GPU required for signed progressive splitting");
    let source = shader_source(
        r#"
@group(0) @binding(3) var<storage, read> probe_input: array<u32>;
@group(0) @binding(4) var<storage, read_write> probe_output: array<i32>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 5u;
    if base >= arrayLength(&probe_input) { return; }
    probe_output[id.x] = progressive_value(bitcast<i32>(probe_input[base]),
        probe_input[base + 1u], probe_input[base + 2u], probe_input[base + 3u], probe_input[base + 4u]);
}

"#,
    );
    let module = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("signed progressive split oracle"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = context
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("signed progressive split oracle"),
            layout: None,
            module: &module,
            entry_point: Some("probe"),
            compilation_options: Default::default(),
            cache: None,
        });
    for progressive in [
        plan(&[(1, 3), (8, 0)]),
        plan(&[(2, 0), (4, 0), (8, 0)]),
        plan(&[(8, 3), (8, 1), (8, 0)]),
        combined(),
        maximum(),
    ] {
        let mut params: VarDctKernelParams = bytemuck::Zeroable::zeroed();
        params.ac_pass_count = progressive.passes().len() as u32;
        for (word, spec) in params.progressive.iter_mut().zip(progressive.passes()) {
            *word = u32::from(spec.coefficient_square.get()) | u32::from(spec.shift) << 8;
        }
        let mut input = Vec::<u32>::new();
        let mut expected = Vec::<i32>::new();
        for strategy in VarDctStrategy::ALL {
            let extent = strategy.pixel_extent();
            let (w, h) = (
                extent.width.max(extent.height),
                extent.width.min(extent.height),
            );
            for index in [
                0,
                1,
                w / 4 - 1,
                w / 4,
                w / 2,
                w - 1,
                w,
                w * h / 4 - 1,
                w * h / 4,
                w * h / 2,
                w * h - 1,
            ] {
                for value in [
                    i32::MIN,
                    i32::MIN + 1,
                    -65537,
                    -9,
                    -1,
                    0,
                    1,
                    7,
                    9,
                    65537,
                    i32::MAX,
                ] {
                    let mut emitted = 0i64;
                    let mut best_shift = None::<u8>;
                    for (pass, spec) in progressive.passes().iter().enumerate() {
                        if index % w < w / 8 * u32::from(spec.coefficient_square.get())
                            && index / w < h / 8 * u32::from(spec.coefficient_square.get())
                        {
                            best_shift =
                                Some(best_shift.map_or(spec.shift, |old| old.min(spec.shift)));
                        }
                        let target = best_shift.map_or(0, |shift| {
                            let step = 1i64 << shift;
                            i64::from(value) / step * step
                        });
                        expected.push(((target - emitted) / (1i64 << spec.shift)) as i32);
                        emitted = target;
                        input.extend([
                            value as u32,
                            index,
                            extent.width,
                            extent.height,
                            pass as u32,
                        ]);
                    }
                    assert_eq!(emitted, i64::from(value));
                }
            }
        }
        let device = context.device();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("progressive probe parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("progressive signed endpoints"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bytes = expected.len() as u64 * 4;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("progressive probe results"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("progressive probe staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("progressive probe bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut commands = device.create_command_encoder(&Default::default());
        {
            let mut pass = commands.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((expected.len() as u32).div_ceil(64), 1, 1);
        }
        commands.copy_buffer_to_buffer(&output, 0, &staging, 0, bytes);
        let submission = context.queue().submit([commands.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |result| {
            sender.send(result).unwrap()
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = staging.get_mapped_range(..).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, i32>(&mapped), expected);
        drop(mapped);
        staging.unmap();
    }
}
