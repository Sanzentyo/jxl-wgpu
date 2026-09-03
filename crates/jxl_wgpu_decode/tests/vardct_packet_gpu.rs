use std::sync::Arc;

use jxl_gpu_bitstream::{InventoryLimits, ParseLimits};
use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::{Extent2d, TransformKind};
use jxl_wgpu_decode::vardct::frontend::StandardVarDctProfile;
use jxl_wgpu_decode::vardct::packet::{
    BoundedVarDctPacketPlan, GpuVarDctPacketError, GpuVarDctPacketStatus, VarDctModularParams,
    VarDctPacketBuffers, VarDctPacketControl, VarDctPacketPipeline, VarDctPacketValidation,
    vardct_packet_shader_source,
};
use jxl_wgpu_encode::{
    BufferImageSource, VarDctColorEncoding, VarDctEncoder, VarDctStrategy, WgpuContext,
};
use wgpu::util::DeviceExt;

mod common;

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
        label: Some("jxl-wgpu VarDCT packet oracle"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

fn black_source(context: &WgpuContext) -> BufferImageSource {
    let extent = Extent2d::new(8, 8);
    let bytes = vec![0u8; 8 * 8 * 3];
    let layout = ImageLayout::from_planes(
        extent,
        VarDctColorEncoding::SrgbD65.pixel_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: 24,
            sample_extent: extent,
            row_bytes: 24,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("black VarDCT packet oracle source"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

fn map_statuses(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
) -> Vec<GpuVarDctPacketStatus> {
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
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
    let statuses = bytemuck::cast_slice::<u8, GpuVarDctPacketStatus>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    statuses
}

#[test]
fn gpu_decodes_fixed_standard_packet_entropy_and_validates_zero_ac() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping VarDCT packet GPU oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let codestream = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8)
        .unwrap()
        .encode(black_source(&context))
        .unwrap();
    let parsed = jxl_gpu_bitstream::parse(&codestream, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: codestream.len() as u64,
            ..InventoryLimits::default()
        })
        .unwrap();
    let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
    let plan = BoundedVarDctPacketPlan::parse(&codestream, profile).unwrap();
    let group = plan.groups.first().unwrap();
    let control = group.packet_control(&plan).unwrap();

    let mut stream_bytes = codestream.clone();
    stream_bytes.resize((stream_bytes.len() + 7) & !3, 0);
    let stream = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet codestream"),
        contents: &stream_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet MA metadata"),
        contents: bytemuck::cast_slice(&plan.modular_metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet LF samples"),
        size: u64::from(
            group
                .reconstructed_words(plan.needs_self_correcting)
                .unwrap(),
        ) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let raw = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet raw HF metadata"),
        size: u64::from(control.capacities[1]) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let coefficients = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet zero AC coefficients"),
        size: u64::from(group.coefficient_words()) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet status"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let control_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet control"),
        contents: bytemuck::bytes_of(&control),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT packet Modular params"),
        contents: bytemuck::bytes_of(
            &VarDctModularParams::default().with_lz77_window(group.lz77_window_words),
        ),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT packet status readback"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let pipeline = VarDctPacketPipeline::new(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VarDCT packet oracle"),
    });
    pipeline.encode(
        &device,
        &mut encoder,
        VarDctPacketBuffers {
            codestream: &stream,
            modular_metadata: &metadata,
            reconstructed_lf: &lf,
            raw_hf_metadata: &raw,
            coefficients: &coefficients,
            status: &status,
            control: &control_buffer,
            modular_params: &params,
        },
    );
    encoder.copy_buffer_to_buffer(
        &status,
        0,
        &staging,
        0,
        std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
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
    let status = bytemuck::from_bytes::<GpuVarDctPacketStatus>(&mapped).to_owned();
    drop(mapped);
    staging.unmap();
    let block_count = control.geometry[2] * control.geometry[3];
    status
        .validate(VarDctPacketValidation {
            expected_strategy: plan.uniform_transform,
            expected_lf_samples: block_count * 3,
            block_count,
            correlation_samples: plan.profile.width().div_ceil(64)
                * plan.profile.height().div_ceil(64),
            task_capacity: group.task_capacity,
            expected_global_scale: plan.global_scale,
            expected_quant_lf: plan.quant_lf,
            expected_extra_precision: group.extra_precision,
        })
        .unwrap();
    assert_eq!(status.coefficient_words, group.coefficient_words());
}

#[test]
fn gpu_stages_cjxl_local_ma_trees_without_host_image_entropy() {
    let Some(codestream) = common::cjxl_local_tree_codestream() else {
        return;
    };
    let Some((device, queue)) = device() else {
        eprintln!("skipping local-tree VarDCT GPU oracle: no adapter");
        return;
    };
    let parsed = jxl_gpu_bitstream::parse(&codestream, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: codestream.len() as u64,
            ..InventoryLimits::default()
        })
        .unwrap();
    let profile = StandardVarDctProfile::negotiate(&inventory).unwrap();
    let plan = BoundedVarDctPacketPlan::parse(&codestream, profile).unwrap();
    assert!(plan.requires_local_tree_staging());
    assert!(plan.groups.len() > 1);

    let mut stream_bytes = codestream;
    stream_bytes.resize((stream_bytes.len() + 7) & !3, 0);
    let stream = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("local-tree VarDCT codestream"),
        contents: &stream_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    struct GroupBuffers {
        reconstructed: wgpu::Buffer,
        raw: wgpu::Buffer,
        coefficients: wgpu::Buffer,
        status: wgpu::Buffer,
    }
    let mut groups = Vec::with_capacity(plan.groups.len());
    for group in &plan.groups {
        let control = group.lf_stage_control(&plan).unwrap();
        groups.push(GroupBuffers {
            reconstructed: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("local-tree LF reconstruction"),
                size: u64::from(group.reconstructed_words(true).unwrap()) * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            raw: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("local-tree raw HF metadata"),
                size: u64::from(control.capacities[1]) * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            coefficients: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("local-tree coefficients"),
                size: u64::from(group.coefficient_words()) * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            status: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("local-tree packet status"),
                size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        });
    }
    let status_bytes = (groups.len() * std::mem::size_of::<GpuVarDctPacketStatus>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("local-tree aggregate status staging"),
        size: status_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let pipeline = VarDctPacketPipeline::new(&device);
    let mut first = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("local-tree LF stage"),
    });
    for (index, (group, buffers)) in plan.groups.iter().zip(&groups).enumerate() {
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree LF MA metadata"),
            contents: bytemuck::cast_slice(&group.lf_modular.metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree LF control"),
            contents: bytemuck::bytes_of(&group.lf_stage_control(&plan).unwrap()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree LF Modular params"),
            contents: bytemuck::bytes_of(
                &VarDctModularParams::default()
                    .with_lz77_window(group.lf_modular.lz77_window_words)
                    .with_self_correcting(group.lf_modular.needs_self_correcting),
            ),
            usage: wgpu::BufferUsages::STORAGE,
        });
        pipeline.encode_lf(
            &device,
            &mut first,
            VarDctPacketBuffers {
                codestream: &stream,
                modular_metadata: &metadata,
                reconstructed_lf: &buffers.reconstructed,
                raw_hf_metadata: &buffers.raw,
                coefficients: &buffers.coefficients,
                status: &buffers.status,
                control: &control,
                modular_params: &params,
            },
        );
        first.copy_buffer_to_buffer(
            &buffers.status,
            0,
            &staging,
            (index * std::mem::size_of::<GpuVarDctPacketStatus>()) as u64,
            std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        );
    }
    let first_statuses = map_statuses(&device, &staging, queue.submit([first.finish()]));
    let continuations = plan
        .groups
        .iter()
        .zip(first_statuses)
        .map(|(group, status)| {
            let block_count = group.block_extent()[0] * group.block_extent()[1];
            let cursor = status
                .validate_lf_stage(
                    block_count * 3,
                    plan.global_scale,
                    plan.quant_lf,
                    group.extra_precision,
                )
                .unwrap();
            plan.parse_hf_continuation(&stream_bytes, group, cursor)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mut second = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("local-tree HF stage"),
    });
    for (index, ((group, buffers), continuation)) in plan
        .groups
        .iter()
        .zip(&groups)
        .zip(&continuations)
        .enumerate()
    {
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree HF MA metadata"),
            contents: bytemuck::cast_slice(&continuation.modular.metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let control_value = group.hf_stage_control(&plan, continuation).unwrap();
        let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree HF control"),
            contents: bytemuck::bytes_of(&control_value),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("local-tree HF Modular params"),
            contents: bytemuck::bytes_of(
                &VarDctModularParams::default()
                    .with_lz77_window(continuation.modular.lz77_window_words)
                    .with_self_correcting(continuation.modular.needs_self_correcting),
            ),
            usage: wgpu::BufferUsages::STORAGE,
        });
        pipeline.encode_hf(
            &device,
            &mut second,
            VarDctPacketBuffers {
                codestream: &stream,
                modular_metadata: &metadata,
                reconstructed_lf: &buffers.reconstructed,
                raw_hf_metadata: &buffers.raw,
                coefficients: &buffers.coefficients,
                status: &buffers.status,
                control: &control,
                modular_params: &params,
            },
        );
        second.copy_buffer_to_buffer(
            &buffers.status,
            0,
            &staging,
            (index * std::mem::size_of::<GpuVarDctPacketStatus>()) as u64,
            std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        );
    }
    let final_statuses = map_statuses(&device, &staging, queue.submit([second.finish()]));
    for ((group, continuation), status) in
        plan.groups.iter().zip(&continuations).zip(final_statuses)
    {
        let [blocks_x, blocks_y] = group.block_extent();
        status
            .validate(VarDctPacketValidation {
                expected_strategy: plan.uniform_transform,
                expected_lf_samples: blocks_x * blocks_y * 3,
                block_count: blocks_x * blocks_y,
                correlation_samples: group.rect.width.div_ceil(64) * group.rect.height.div_ceil(64),
                task_capacity: group.task_capacity,
                expected_global_scale: plan.global_scale,
                expected_quant_lf: plan.quant_lf,
                expected_extra_precision: group.extra_precision,
            })
            .unwrap();
        assert_eq!(status.first_blocks, continuation.block_count);
    }
}

#[test]
fn gpu_sharpness_validation_rejects_out_of_range_metadata() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping VarDCT sharpness GPU validation: no adapter");
        return;
    };
    let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("malformed VarDCT sharpness metadata"),
        contents: bytemuck::cast_slice(&[8_u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let status = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT sharpness status"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let control = VarDctPacketControl {
        section_bits: [0; 4],
        geometry: [0, 0, 1, 1],
        offsets: [0; 4],
        capacities: [0; 4],
        expected: [0; 4],
        quantization: [0; 4],
        streams: [0; 4],
        scratch: [0; 4],
    };
    let control = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("malformed VarDCT sharpness control"),
        contents: bytemuck::bytes_of(&control),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT sharpness status readback"),
        size: std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("VarDCT sharpness validation oracle"),
        source: wgpu::ShaderSource::Wgsl(vardct_packet_shader_source().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("VarDCT sharpness validation oracle"),
        layout: None,
        module: &module,
        entry_point: Some("validate_vardct_sharpness"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("VarDCT sharpness validation oracle bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 3,
                resource: raw.as_entire_binding(),
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
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VarDCT sharpness validation oracle"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VarDCT sharpness validation oracle"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &status,
        0,
        &staging,
        0,
        std::mem::size_of::<GpuVarDctPacketStatus>() as u64,
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
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
    let status = *bytemuck::from_bytes::<GpuVarDctPacketStatus>(&mapped);
    assert_eq!(
        status.validate(VarDctPacketValidation {
            expected_strategy: Some(TransformKind::Dct8),
            ..VarDctPacketValidation::default()
        }),
        Err(GpuVarDctPacketError::Sharpness { value: 8 })
    );
}
