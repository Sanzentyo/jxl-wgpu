use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{BLOCK_CONTEXT, BLOCK_CONTEXT_MARKER, HfBlockContextTables};

const PROBE_TEMPLATE: &str = include_str!("../vardct_block_context_probe.wgsl");

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ProbeParams {
    tables: HfBlockContextTables,
    order_channel: u32,
    order_id: u32,
    qf: u32,
    lf_x: u32,
    lf_y: u32,
    lf_b: u32,
    _reserved: [u32; 2],
}

#[derive(Clone, Copy)]
struct Case {
    order_channel: u32,
    order_id: u32,
    qf: u32,
    lf: [i32; 3],
}

fn segment_i32(value: i32, thresholds: &[i32]) -> usize {
    thresholds
        .iter()
        .filter(|&&threshold| value > threshold)
        .count()
}

fn segment_u32(value: u32, thresholds: &[u32]) -> usize {
    thresholds
        .iter()
        .filter(|&&threshold| value > threshold)
        .count()
}

fn scalar_index(case: Case, qf_thresholds: &[u32], lf_thresholds: &[Vec<i32>; 3]) -> u32 {
    let mut lf_index = segment_i32(case.lf[0], &lf_thresholds[0]);
    lf_index = lf_index * (lf_thresholds[2].len() + 1) + segment_i32(case.lf[2], &lf_thresholds[2]);
    lf_index = lf_index * (lf_thresholds[1].len() + 1) + segment_i32(case.lf[1], &lf_thresholds[1]);
    let lf_contexts = lf_thresholds
        .iter()
        .map(|thresholds| thresholds.len() + 1)
        .product::<usize>();
    let qf_index = segment_u32(case.qf, qf_thresholds);
    u32::try_from(
        (((case.order_channel as usize * 13 + case.order_id as usize) * (qf_thresholds.len() + 1)
            + qf_index)
            * lf_contexts)
            + lf_index,
    )
    .unwrap()
}

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu HF block-context differential test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

#[test]
fn nondefault_hf_block_context_matches_scalar_on_gpu() {
    let source = PROBE_TEMPLATE.replace(BLOCK_CONTEXT_MARKER, BLOCK_CONTEXT);
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let qf_thresholds = vec![4u32, 10];
    let lf_thresholds = [vec![-2, 5], vec![0], vec![-10, 3]];
    let lf_contexts = lf_thresholds
        .iter()
        .map(|thresholds| thresholds.len() + 1)
        .product::<usize>();
    let map_len = 3 * 13 * (qf_thresholds.len() + 1) * lf_contexts;
    let mut metadata = (0..u32::try_from(map_len).unwrap()).collect::<Vec<_>>();
    let qf_threshold_offset_words = u32::try_from(metadata.len()).unwrap();
    metadata.extend_from_slice(&qf_thresholds);
    let lf0_threshold_offset_words = u32::try_from(metadata.len()).unwrap();
    metadata.extend(lf_thresholds[0].iter().map(|&value| value as u32));
    let lf1_threshold_offset_words = u32::try_from(metadata.len()).unwrap();
    metadata.extend(lf_thresholds[1].iter().map(|&value| value as u32));
    let lf2_threshold_offset_words = u32::try_from(metadata.len()).unwrap();
    metadata.extend(lf_thresholds[2].iter().map(|&value| value as u32));
    let tables = HfBlockContextTables {
        block_context_map_offset_words: 0,
        qf_threshold_offset_words,
        qf_threshold_count: u32::try_from(qf_thresholds.len()).unwrap(),
        lf0_threshold_offset_words,
        lf0_threshold_count: u32::try_from(lf_thresholds[0].len()).unwrap(),
        lf1_threshold_offset_words,
        lf1_threshold_count: u32::try_from(lf_thresholds[1].len()).unwrap(),
        lf2_threshold_offset_words,
        lf2_threshold_count: u32::try_from(lf_thresholds[2].len()).unwrap(),
        _reserved: [0; 3],
    };
    let cases = [
        Case {
            order_channel: 0,
            order_id: 0,
            qf: 4,
            lf: [-2, 0, -10],
        },
        Case {
            order_channel: 0,
            order_id: 0,
            qf: 5,
            lf: [-1, 1, -9],
        },
        Case {
            order_channel: 1,
            order_id: 7,
            qf: 10,
            lf: [5, 1, 3],
        },
        Case {
            order_channel: 2,
            order_id: 12,
            qf: 99,
            lf: [6, -5, 4],
        },
    ];
    let expected = cases
        .iter()
        .copied()
        .map(|case| scalar_index(case, &qf_thresholds, &lf_thresholds))
        .collect::<Vec<_>>();
    let params = cases
        .iter()
        .map(|case| ProbeParams {
            tables,
            order_channel: case.order_channel,
            order_id: case.order_id,
            qf: case.qf,
            lf_x: case.lf[0] as u32,
            lf_y: case.lf[1] as u32,
            lf_b: case.lf[2] as u32,
            _reserved: [0; 2],
        })
        .collect::<Vec<_>>();

    let Some((device, queue)) = device() else {
        eprintln!("skipping HF block-context GPU differential test: no adapter");
        return;
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jxl-wgpu HF block-context probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("jxl-wgpu HF block-context probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe_block_context"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HF block-context probe tables"),
        contents: bytemuck::cast_slice(&metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HF block-context probe params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_bytes = u64::try_from(expected.len() * std::mem::size_of::<u32>()).unwrap();
    let result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HF block-context probe result"),
        size: result_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HF block-context probe readback"),
        size: result_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HF block-context probe bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: metadata.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: result.as_entire_binding(),
            },
        ],
    });
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("HF block-context probe"),
    });
    {
        let mut pass = commands.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HF block-context probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(u32::try_from(expected.len()).unwrap(), 1, 1);
    }
    commands.copy_buffer_to_buffer(&result, 0, &staging, 0, result_bytes);
    let submission = queue.submit([commands.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), expected);
    drop(mapped);
    staging.unmap();
}

const _: () = {
    assert!(std::mem::size_of::<ProbeParams>() == 80);
    assert!(std::mem::align_of::<ProbeParams>() == 16);
    assert!(std::mem::offset_of!(ProbeParams, tables) == 0);
    assert!(std::mem::offset_of!(ProbeParams, order_channel) == 48);
};
