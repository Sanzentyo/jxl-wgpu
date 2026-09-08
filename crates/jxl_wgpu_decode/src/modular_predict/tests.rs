use wgpu::util::DeviceExt;

mod reference;

fn execute(shader: &str, input: &[u32], output_words: usize, invocations: u32) -> Option<Vec<u32>> {
    let backend =
        pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default())).ok()?;
    let device = backend.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Modular wide arithmetic oracle"),
        source: wgpu::ShaderSource::Wgsl(super::shader(shader).into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = output_words as u64 * 4;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(invocations.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &read, 0, bytes);
    backend.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    read.slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let data = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
    read.unmap();
    Some(data)
}

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn words(value: i64) -> [u32; 2] {
    [value as u32, (value as u64 >> 32) as u32]
}

#[test]
fn portable_wide_arithmetic_matches_signed_integer_boundaries() {
    let mut input = Vec::new();
    let mut expected = Vec::new();
    let mut seed = 0x2b6a_5c43_2399_835du64;
    let values = [
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        -1,
        0,
        1,
        i32::MIN as i64,
        i32::MAX as i64,
        0xffff_ffff,
        0x1_0000_0000,
    ];
    for (a, b) in values
        .into_iter()
        .flat_map(|a| values.map(|b| (a, b)))
        .chain((0..128).map(|_| (random(&mut seed) as i64, random(&mut seed) as i64)))
    {
        for shift in [0, 1, 3, 4, 24, 31, 32, 33, 63, 64] {
            let multiplier = (a as u32).wrapping_add(b as u32);
            input.extend(words(a));
            input.extend(words(b));
            input.extend([multiplier, shift, 0, 0]);
            let sar = if shift >= 64 { a >> 63 } else { a >> shift };
            let shl = if shift >= 64 {
                0
            } else {
                a.wrapping_shl(shift)
            };
            let div = ((a as i128) / (1_i128 << shift)) as i64;
            for value in [
                a.wrapping_add(b),
                a.wrapping_sub(b),
                a.wrapping_mul(i64::from(multiplier)),
                sar,
                shl,
                div,
                a.min(b),
                a.max(b),
                a.wrapping_neg(),
                a.wrapping_abs(),
            ] {
                expected.extend(words(value));
            }
        }
    }
    let shader = r#"
        /*__JXL_MODULAR_INTEGER__*/
        @group(0) @binding(0) var<storage, read> input: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<u32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let index = id.x;
            if index >= arrayLength(&input) / 8u { return; }
            let base = index * 8u;
            let a = vec2<u32>(input[base], input[base + 1u]);
            let b = vec2<u32>(input[base + 2u], input[base + 3u]);
            let shift = input[base + 5u];
            let results = array<ModularI64, 10>(mi_add(a,b), mi_sub(a,b), mi_mul_u32(a,input[base+4u]),
                mi_sar(a,shift), mi_shl(a,shift), mi_div_pow2(a,shift), mi_min(a,b), mi_max(a,b), mi_neg(a), mi_abs(a));
            for (var operation = 0u; operation < 10u; operation++) {
                output[index * 20u + operation * 2u] = results[operation].x;
                output[index * 20u + operation * 2u + 1u] = results[operation].y;
            }
        }
    "#;
    if let Some(actual) = execute(shader, &input, expected.len(), (input.len() / 8) as u32) {
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a,
                b,
                "case {} operation {} word {}",
                index / 20,
                index % 20 / 2,
                index % 2
            );
        }
    }
}

#[test]
fn every_predictor_and_weighted_error_matches_a_wide_cpu_oracle() {
    let patterns: [u32; 16] = [
        0,
        1,
        u32::MAX,
        i32::MAX as u32,
        i32::MIN as u32,
        0x007fffff,
        0x00800000,
        0x3f800000,
        0xbf800000,
        0x7f7fffff,
        0xff7fffff,
        0x7f800000,
        0xff800000,
        0x7fc00001,
        0xffc00001,
        65535,
    ];
    let mut seed = 0x6492_d3ec_9860_91bd;
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for index in 0..512 {
        let mut data = [0u32; 40];
        for (c, slot) in data[..24].iter_mut().enumerate() {
            *slot = if index < 32 {
                patterns[(index + c * 3) % patterns.len()]
            } else {
                random(&mut seed) as u32
            };
        }
        if index == 0 {
            data[12] = u32::MAX;
            data[16] = 0;
            data[20] = 0;
        }
        for (c, slot) in data[24..31].iter_mut().enumerate() {
            *slot = match index % 3 {
                0 => [16, 10, 7, 7, 7, 0, 0][c],
                1 => 31,
                _ => 0,
            };
        }
        data[31..35].copy_from_slice(if index % 2 == 0 {
            &[13, 12, 12, 12]
        } else {
            &[0, 15, 1, 7]
        });
        expected.extend(reference::predict(&data));
        input.extend(data);
    }
    if let Some(actual) = execute(
        include_str!("tests/predict.wgsl"),
        &input,
        expected.len(),
        512,
    ) {
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a, b, "case {} field {}", index / 30, index % 30);
        }
    }
}
