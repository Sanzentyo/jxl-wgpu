use super::*;
#[test]
fn jpeg_reconstruction_restoration_abi_and_shader_validate() {
    assert_eq!(std::mem::offset_of!(RestoreParams, outputs), 48);
    assert_eq!(std::mem::offset_of!(RestoreParams, shifts), 96);
    assert_eq!(std::mem::offset_of!(RestoreParams, group), 144);
    assert_eq!(std::mem::offset_of!(RestoreParams, artifact), 160);
    assert_eq!(std::mem::offset_of!(RestoreParams, config), 176);
    let module = naga::front::wgsl::parse_str(include_str!("../jpeg.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

struct Probe {
    name: &'static str,
    params: RestoreParams,
    lf: [i32; 3],
    ac: [i32; 192],
    artifact: [u32; 22],
    correlation: [i32; 2],
    quantization: [i32; 192],
    capture: [u32; 4],
    expected_dc: Option<[i32; 3]>,
    expected_ac: Option<i32>,
    error: bool,
}

impl Probe {
    fn valid(name: &'static str) -> Self {
        let mut artifact = [0; 22];
        artifact[..12].copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 0, 0, 192, 0, 0x700]);
        artifact[16] = 1; // One lowered task, following the 12-word metadata record.
        Self {
            name,
            params: RestoreParams {
                lf: [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 2, 0]],
                outputs: [[192, 1, 1, 0], [256, 1, 1, 64], [320, 1, 1, 128]],
                shifts: [[0; 4]; 3],
                group: [0, 0, 1, 1],
                artifact: [0, 12, 1, 192],
                config: [0, 0, 0, 1],
            },
            lf: [0; 3],
            ac: [0; 192],
            artifact,
            correlation: [0; 2],
            quantization: [1; 192],
            capture: [192, 0, 0, 0],
            expected_dc: None,
            expected_ac: None,
            error: false,
        }
    }
}

#[test]
fn jpeg_reconstruction_integer_boundaries_and_invalid_bindings_execute_on_gpu() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("actual GPU adapter is required");
    let device = backend.device();
    let pipeline = JpegRestorePipeline::new(device);
    let mut probes = Vec::new();
    // Independent high-precision evaluation of the native decoder's float DC rule.
    for rgb in [false, true] {
        for precision in 0..=3 {
            for quant in [1, 3, 255, 65535] {
                for raw in [i32::MIN, -16377, -1, 1, 16377, i32::MAX] {
                    let mut probe = Probe::valid("DC precision and clamping");
                    probe.params.config[0] = u32::from(rgb);
                    probe.params.config[2] = precision;
                    probe.quantization.fill(quant);
                    probe.lf.fill(raw);
                    let offset = if rgb { 1024 / quant } else { 0 };
                    let expected = (f64::from(raw) / f64::from(1u32 << precision)
                        - f64::from(offset))
                    .clamp(-2047.0, 2047.0) as i32;
                    probe.expected_dc = Some([expected; 3]);
                    probes.push(probe);
                }
            }
        }
    }
    for (luma, correlation, q_y, q_c) in [
        (-2047, -128, 65535, 256),
        (2047, 127, 65535, 256),
        (-2047, 127, 1, 65535),
        (2047, -128, 1, 65535),
        (-1, -1, 3, 1),
        (1, 1, 3, 1),
    ] {
        for expected in [-2047, 0, 2047] {
            let mut probe = Probe::valid("fixed-point CfL boundary");
            probe.params.config[1] = 1;
            probe.quantization.fill(q_c);
            probe.quantization[64..128].fill(q_y);
            probe.correlation.fill(correlation);
            // Wider signed arithmetic and Euclidean division express the specified rounding
            // independently of WGSL's bounded i32 multiply/arithmetic-shift operations.
            let ratio = i64::from(q_y) * 2048 / i64::from(q_c);
            let scale = i64::from(correlation) * 2048 / 84;
            let coefficient_scale = (ratio * scale + 1024).div_euclid(2048);
            let correction = (i64::from(luma) * coefficient_scale + 1024).div_euclid(2048);
            let raw = i32::try_from(i64::from(expected) - correction).unwrap();
            probe.ac.fill(raw);
            probe.ac[64..128].fill(luma);
            probe.expected_ac = Some(expected);
            probes.push(probe);
        }
    }
    for kind in 0..13 {
        let mut probe = Probe::valid("invalid capture, values or artifact");
        probe.error = true;
        match kind {
            0 => probe.capture[0] = 0,
            1 => probe.capture[1] = 1,
            2 => probe.params.config[2] = 4,
            3 => probe.artifact[12] = 1,
            4 => probe.artifact[0] = 1,
            5 => probe.artifact[16] = 2,
            6 => probe.artifact[8] = 193,
            7 => probe.quantization[1] = 0,
            8 => probe.quantization[1] = 65536,
            9 => probe.ac[1] = i32::MAX,
            10 => probe.ac[1] = i32::MIN,
            11 => {
                probe.params.config[1] = 1;
                probe.correlation[0] = 128;
            }
            12 => {
                probe.params.config[1] = 1;
                probe.quantization[65] = 256;
            }
            _ => unreachable!(),
        }
        probes.push(probe);
    }
    let record_bytes = (384 + 4) * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("JPEG integer probe readback"),
        size: probes.len() as u64 * record_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = device.create_command_encoder(&Default::default());
    // All probes share one ordered GPU submission and one explicit development-only readback.
    for (index, probe) in probes.iter().enumerate() {
        let storage = |bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(probe.name),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let lf = storage(bytemuck::cast_slice(&probe.lf));
        let ac = storage(bytemuck::cast_slice(&probe.ac));
        let artifact = storage(bytemuck::cast_slice(&probe.artifact));
        let correlation = storage(bytemuck::cast_slice(&probe.correlation));
        let mut words = [0i32; 384];
        words[..192].copy_from_slice(&probe.quantization);
        let output = storage(bytemuck::cast_slice(&words));
        let status = storage(bytemuck::cast_slice(&probe.capture));
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(probe.name),
            contents: bytemuck::bytes_of(&probe.params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &lf,
            &ac,
            &artifact,
            &correlation,
            &output,
            &status,
            &uniform,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(probe.name),
            layout: &pipeline.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        commands.copy_buffer_to_buffer(
            &output,
            0,
            &staging,
            index as u64 * record_bytes,
            output.size(),
        );
        commands.copy_buffer_to_buffer(
            &status,
            0,
            &staging,
            index as u64 * record_bytes + output.size(),
            status.size(),
        );
    }
    backend.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    for (probe, bytes) in probes
        .iter()
        .zip(mapped.chunks_exact(record_bytes as usize))
    {
        let words: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        if probe.error {
            assert_ne!(words[387], 0, "{}", probe.name);
            continue;
        }
        assert_eq!(&words[384..], &[192, 0, 192, 0], "{}", probe.name);
        if let Some(expected) = probe.expected_dc {
            for channel in 0..3 {
                assert_eq!(
                    words[192 + channel * 64],
                    expected[channel],
                    "{}",
                    probe.name
                );
            }
        }
        if let Some(expected) = probe.expected_ac {
            for channel in [0, 2] {
                for k in 1..64 {
                    assert_eq!(words[192 + channel * 64 + k], expected, "{}", probe.name);
                }
            }
        }
    }
    drop(mapped);
    staging.unmap();
}
