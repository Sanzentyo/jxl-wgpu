use jxl_wgpu_decode::vardct::packet::{VarDctPacketControl, vardct_packet_shader_source};
use serde::Deserialize;
use wgpu::util::DeviceExt;

#[derive(Deserialize)]
struct Case {
    kind: u32,
    width: u32,
    height: u32,
    first_blocks: u32,
    lf: [[u32; 2]; 3],
    channels: Vec<Channel>,
}

#[derive(Deserialize)]
struct Channel {
    width: u32,
    height: u32,
    samples: Vec<i32>,
    properties: Vec<i32>,
}

const PROBE: &str = r"
@compute @workgroup_size(1)
fn probe_previous_channels() {
    target_kind = control.streams.x;
    current_channel = control.streams.y;
    packet_first_blocks = control.quantization.w;
    for (var y = 0u; y < control.streams.w; y += 1u) {
        for (var x = 0u; x < control.streams.z; x += 1u) {
            for (var p = 0u; p < 16u; p += 1u) {
                let value = ma_property(p + 16u, x, y, 0i, 0i, 0i, 0i, 0i, 0i, 0i);
                status[(y * control.streams.z + x) * 16u + p] = bitcast<u32>(value);
            }
        }
    }
}
";

#[test]
fn packet_properties_match_native_references_for_every_lf_geometry_and_hf_layout() {
    let Some(backend) = super::backend() else {
        return;
    };
    let device = backend.device();
    let queue = backend.queue();
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(super::directory().join("references.json")).unwrap())
            .unwrap();
    assert_eq!(cases.len(), 70);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("native VarDCT MA reference probe"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}{PROBE}", vardct_packet_shader_source()).into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("native VarDCT MA reference probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe_previous_channels"),
        compilation_options: Default::default(),
        cache: None,
    });
    let storage = |label, words: &[i32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mut comparisons = 0;
    for (case_index, case) in cases.iter().enumerate() {
        let bx = case.width.div_ceil(8);
        let by = case.height.div_ceil(8);
        let blocks = bx * by;
        let correlation = case.width.div_ceil(64) * case.height.div_ceil(64);
        let offsets = [0, correlation, 2 * correlation, 2 * correlation + blocks];
        let sharpness = 2 * correlation + 2 * blocks;
        let mut lf = vec![0x1234_5678_i32; (3 * blocks) as usize];
        let mut hf = vec![0x7654_3210_i32; (sharpness + blocks) as usize];
        for (index, channel) in case.channels.iter().enumerate() {
            let (destination, offset, stride) = if case.kind == 0 {
                (&mut lf, index as u32 * blocks, channel.width)
            } else if index == 2 {
                (&mut hf, offsets[2], blocks)
            } else if index == 3 {
                (&mut hf, sharpness, bx)
            } else {
                (&mut hf, offsets[index], channel.width)
            };
            assert_eq!(
                channel.samples.len(),
                (channel.width * channel.height) as usize
            );
            for y in 0..channel.height {
                let start = (offset + y * stride) as usize;
                let row = (y * channel.width) as usize;
                destination[start..start + channel.width as usize]
                    .copy_from_slice(&channel.samples[row..row + channel.width as usize]);
            }
        }
        let lf = storage("native LF references", &lf);
        let hf = storage("native HF references", &hf);
        for (index, channel) in case.channels.iter().enumerate() {
            let size = channel.properties.len() as u64 * 4;
            assert_eq!(channel.properties.len(), channel.samples.len() * 16);
            let status = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MA property results"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MA property readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut control = VarDctPacketControl {
                section_bits: [0; 4],
                geometry: [case.width, case.height, bx, by],
                offsets,
                capacities: [0, 0, 0, blocks],
                expected: [0, 0, 0, sharpness],
                quantization: [0, 0, 0, case.first_blocks],
                streams: [case.kind, index as u32, channel.width, channel.height],
                scratch: [0; 4],
            };
            for (word, extent) in control.scratch[1..].iter_mut().zip(case.lf) {
                *word = extent[0] | extent[1] << 16;
            }
            let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MA geometry"),
                contents: bytemuck::bytes_of(&control),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MA reference probe bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: hf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: status.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: control.as_entire_binding(),
                    },
                ],
            });
            let mut commands = device.create_command_encoder(&Default::default());
            {
                let mut pass = commands.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            commands.copy_buffer_to_buffer(&status, 0, &staging, 0, size);
            queue.submit([commands.finish()]);
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            staging.map_async(wgpu::MapMode::Read, .., move |result| {
                sender.send(result).unwrap()
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            receiver.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            assert_eq!(
                bytemuck::cast_slice::<u8, i32>(&mapped),
                channel.properties,
                "native geometry case {case_index}, channel {index}"
            );
            comparisons += channel.properties.len();
            drop(mapped);
            staging.unmap();
        }
    }
    eprintln!("{comparisons} MA property values exactly match native PrecomputeReferences");
}
