#![cfg(not(target_arch = "wasm32"))]

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::TransformKind;
use jxl_wgpu_decode::vardct::artifact::{
    BACKEND_REQUIREMENT_FREQUENCY_CFL_GRID, GpuDispatchIndirectArgs, GpuGeneralVarDctTask,
    GpuHfTaskMetadata, GpuVarDctArtifactStatus, GpuVarDctBucket, GpuVarDctLoweringError,
    HF_COEFFICIENT_SINK_SHADER, HfCoefficientSinkParams, HfDispatchStage, HfMetadataArtifactConfig,
    HfMetadataLoweringBuffers, HfMetadataLoweringParams, HfMetadataLoweringPipeline,
    HfOrderTableLayout, VAR_DCT_ARTIFACT_SHADER, VAR_DCT_STRATEGY_COUNT,
    VarDctArtifactDeviceLimits, VarDctArtifactError, VarDctArtifactLayout,
};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TestCoefficientToken {
    task_index: u32,
    channel: u32,
    order_index: u32,
    value: i32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TestTokenParams {
    values: [u32; 4],
}

fn request_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("VarDCT artifact test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

fn config(raw_metadata_words: u64) -> HfMetadataArtifactConfig {
    HfMetadataArtifactConfig {
        blocks_width: 5,
        blocks_height: 1,
        block_info_entries: 5,
        strategy_offset_words: 0,
        hf_mul_offset_words: 5,
        raw_metadata_words,
        pass_group_dim_blocks: 32,
        lf_stride: 13,
        correlation_stride: 2,
        correlation_width: 1,
        correlation_height: 1,
        destination_origin: [64, 64],
        afv_basis_offset: 9_999,
        quant_offset: 0,
        correlation_offset: 0,
        global_scale: 8_813,
        matrix_offsets: std::array::from_fn(|strategy| 1_000 + strategy as u32 * 64),
    }
}

fn unbounded_limits() -> VarDctArtifactDeviceLimits {
    VarDctArtifactDeviceLimits {
        max_buffer_size: u64::MAX,
        max_storage_buffer_binding_size: u64::MAX,
        storage_binding_alignment: 256,
    }
}

fn align_256(value: u64) -> u64 {
    value.div_ceil(256) * 256
}

fn cast_one<T: Pod>(bytes: &[u8], offset: usize) -> T {
    *bytemuck::from_bytes(&bytes[offset..offset + std::mem::size_of::<T>()])
}

fn lower_topology(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    blocks: [u32; 2],
    strategies: &[i32],
    pass_group_dim_blocks: u32,
) -> GpuVarDctArtifactStatus {
    let entries = strategies.len() as u32;
    let mut raw = strategies.to_vec();
    raw.extend(std::iter::repeat_n(5, strategies.len()));
    let config = HfMetadataArtifactConfig {
        blocks_width: blocks[0],
        blocks_height: blocks[1],
        block_info_entries: entries,
        strategy_offset_words: 0,
        hf_mul_offset_words: entries,
        raw_metadata_words: raw.len() as u64,
        pass_group_dim_blocks,
        lf_stride: blocks[0],
        correlation_stride: blocks[0].div_ceil(8),
        correlation_width: blocks[0].div_ceil(8),
        correlation_height: blocks[1].div_ceil(8),
        destination_origin: [0, 0],
        afv_basis_offset: 0,
        quant_offset: 0,
        correlation_offset: 0,
        global_scale: 8_813,
        matrix_offsets: [0; VAR_DCT_STRATEGY_COUNT],
    };
    let layout = VarDctArtifactLayout::plan(
        &config,
        VarDctArtifactDeviceLimits::from_wgpu(&device.limits()),
    )
    .unwrap();
    let params = HfMetadataLoweringParams::new(&config, layout).unwrap();
    let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("malformed VarDCT topology"),
        contents: bytemuck::cast_slice(&raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let artifact = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT topology artifact"),
        size: layout.artifact_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let occupancy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT topology occupancy"),
        size: layout.occupancy_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("malformed VarDCT topology params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let resources = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT quantization resources"),
        size: u64::from(entries.max(1)) * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("malformed VarDCT topology status"),
        size: std::mem::size_of::<GpuVarDctArtifactStatus>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let lowering = HfMetadataLoweringPipeline::new(device);
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("malformed VarDCT topology validation"),
    });
    lowering.encode(
        device,
        &mut commands,
        HfMetadataLoweringBuffers {
            raw_metadata: &raw,
            artifact: &artifact,
            occupancy: &occupancy,
            resources: &resources,
            params: &params,
        },
    );
    commands.copy_buffer_to_buffer(
        &artifact,
        u64::from(layout.status_offset_words) * 4,
        &staging,
        0,
        std::mem::size_of::<GpuVarDctArtifactStatus>() as u64,
    );
    let submission = queue.submit([commands.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
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
    bytemuck::pod_read_unaligned(&mapped)
}

#[test]
fn artifact_abi_and_composable_sink_are_valid_wgsl() {
    let module = naga::front::wgsl::parse_str(VAR_DCT_ARTIFACT_SHADER)
        .expect("metadata artifact shader parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("metadata artifact shader validates");

    let test_wrapper = format!(
        r#"{HF_COEFFICIENT_SINK_SHADER}
struct TestCoefficientToken {{
    task_index: u32,
    channel: u32,
    order_index: u32,
    value: i32,
}};
struct TestTokenParams {{
    values: vec4<u32>,
}};
@group(0) @binding(0) var<storage, read> test_tokens: array<TestCoefficientToken>;
@group(0) @binding(1) var<uniform> test_params: TestTokenParams;
@compute @workgroup_size(64, 1, 1)
fn scatter_test(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    if (invocation.x >= test_params.values.x) {{
        return;
    }}
    let token = test_tokens[invocation.x];
    _ = hf_store_quantized_coefficient(
        token.task_index,
        token.channel,
        token.order_index,
        token.value,
    );
}}
"#
    );
    let module = naga::front::wgsl::parse_str(&test_wrapper).expect("sink wrapper parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("sink wrapper validates");

    assert_eq!(std::mem::size_of::<GpuGeneralVarDctTask>(), 64);
    assert_eq!(std::mem::size_of::<GpuHfTaskMetadata>(), 48);
    assert_eq!(std::mem::size_of::<GpuDispatchIndirectArgs>(), 12);
    assert_eq!(std::mem::size_of::<HfMetadataLoweringParams>(), 208);
}

#[test]
fn artifact_layout_is_aligned_bounded_and_typed() {
    let config = config(10);
    let layout = VarDctArtifactLayout::plan(&config, unbounded_limits()).unwrap();
    for offset in [
        layout.buckets_offset_words,
        layout.tasks_offset_words,
        layout.indirect_offset_words,
        layout.task_metadata_offset_words,
        layout.block_task_map_offset_words,
    ] {
        assert_eq!(u64::from(offset) * 4 % 256, 0);
    }
    assert_eq!(layout.task_capacity, 5);
    assert_eq!(layout.block_count, 5);
    assert_eq!(layout.coefficient_capacity_words, 5 * 192);
    assert_eq!(layout.task_binding().1, 5 * 64);
    assert_eq!(layout.binding_alignment(), 256);
    assert_eq!(
        layout.persistent_bytes(),
        layout.artifact_bytes + layout.coefficient_bytes
    );
    assert_eq!(
        layout.transient_bytes(),
        layout.occupancy_bytes + std::mem::size_of::<HfMetadataLoweringParams>() as u64
    );
    assert_eq!(
        layout.indirect_offset(7, HfDispatchStage::Horizontal),
        Some(u64::from(layout.indirect_offset_words) * 4 + (7 * 3 + 1) * 12)
    );
    assert_eq!(
        layout.indirect_offset(7, HfDispatchStage::Vertical),
        Some(u64::from(layout.indirect_offset_words) * 4 + (7 * 3 + 2) * 12)
    );

    let mut outside = config;
    outside.raw_metadata_words = 9;
    assert_eq!(
        VarDctArtifactLayout::plan(&outside, unbounded_limits()),
        Err(VarDctArtifactError::RawMetadataRange {
            end: 10,
            available: 9,
        })
    );
    let mut zero = config;
    zero.blocks_width = 0;
    assert_eq!(
        VarDctArtifactLayout::plan(&zero, unbounded_limits()),
        Err(VarDctArtifactError::ZeroDimension { axis: "horizontal" })
    );
    let mut short_correlation = config;
    short_correlation.correlation_width = 0;
    assert_eq!(
        VarDctArtifactLayout::plan(&short_correlation, unbounded_limits()),
        Err(VarDctArtifactError::InvalidGeometry {
            field: "correlation grid does not cover the LF group",
        })
    );
    let mut short_lf_stride = config;
    short_lf_stride.lf_stride = 12;
    assert_eq!(
        VarDctArtifactLayout::plan(&short_lf_stride, unbounded_limits()),
        Err(VarDctArtifactError::InvalidGeometry {
            field: "global LF horizontal extent exceeds its stride",
        })
    );
    let mut short_correlation_stride = config;
    short_correlation_stride.correlation_stride = 1;
    assert_eq!(
        VarDctArtifactLayout::plan(&short_correlation_stride, unbounded_limits()),
        Err(VarDctArtifactError::InvalidGeometry {
            field: "global correlation horizontal extent exceeds its stride",
        })
    );

    let limited = VarDctArtifactDeviceLimits {
        max_buffer_size: u64::MAX,
        max_storage_buffer_binding_size: layout.coefficient_bytes - 1,
        storage_binding_alignment: 256,
    };
    assert!(matches!(
        VarDctArtifactLayout::plan(&config, limited),
        Err(VarDctArtifactError::StorageBindingLimit {
            resource: "VarDCT coefficients",
            ..
        })
    ));
}

#[test]
fn custom_order_layout_covers_every_channel_without_aliasing() {
    let used_orders = (1 << 0) | (1 << 4) | (1 << 12);
    let layout = HfOrderTableLayout::new(used_orders).unwrap();
    assert!(layout.custom(0));
    assert!(layout.custom(4));
    assert!(layout.custom(12));
    assert!(!layout.custom(3));
    let mut cursor = 0u32;
    for order_id in 0..13 {
        for channel in 0..3 {
            let descriptor = layout.descriptor(order_id, channel).unwrap();
            assert_eq!(descriptor.offset, cursor);
            assert_eq!(descriptor.len, descriptor.width * descriptor.height);
            cursor += descriptor.len;
        }
    }
    assert_eq!(cursor, layout.coordinate_words);
    assert_eq!(layout.descriptor(13, 0), None);
    assert_eq!(layout.descriptor(0, 3), None);
}

#[test]
fn malformed_varblock_overlap_and_pass_group_crossing_are_gpu_rejected() {
    let Some((device, queue)) = request_device() else {
        eprintln!("skipping malformed VarDCT topology GPU test: no adapter is available");
        return;
    };

    // DCT8 at (0,0), tall DCT16x8 at (1,0), then wide DCT8x16 at (0,1).
    // The final transform covers the already occupied (1,1) block.
    let overlap = lower_topology(&device, &queue, [2, 2], &[0, 6, 7], 32);
    assert_eq!(
        overlap.validate(),
        Err(GpuVarDctLoweringError::TransformOverlap {
            x: 1,
            y: 1,
            strategy: 7,
        })
    );

    // Thirty-one DCT8 blocks leave x=31 as the next anchor. A two-block-wide transform fits the
    // 33-block LF group but crosses the standard 32-block pass-group boundary.
    let mut crossing = vec![0; 32];
    crossing[31] = 7;
    let crossing = lower_topology(&device, &queue, [33, 1], &crossing, 32);
    assert_eq!(
        crossing.validate(),
        Err(GpuVarDctLoweringError::PassGroupCrossing {
            x: 31,
            y: 0,
            strategy: 7,
        })
    );
}

#[test]
fn mixed_odd_tail_lowers_and_custom_order_scatters_without_cpu_readback_boundary() {
    let Some((device, queue)) = request_device() else {
        eprintln!("skipping VarDCT artifact GPU test: no adapter is available");
        return;
    };
    // Raster varblocks: DCT8 at x=0, DCT8x16 at x=1..2, AFV0 at x=3, DCT4x8 at x=4.
    // The fifth raw entry is deliberately unused, matching libjxl's bounded list semantics.
    let raw_metadata = [0i32, 7, 14, 12, 26, 2, 4, 6, 8, 10];
    let config = config(raw_metadata.len() as u64);
    let layout = VarDctArtifactLayout::plan(
        &config,
        VarDctArtifactDeviceLimits::from_wgpu(&device.limits()),
    )
    .expect("artifact fits the test adapter");
    let params = HfMetadataLoweringParams::new(&config, layout).unwrap();

    let raw_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("raw HF metadata"),
        contents: bytemuck::cast_slice(&raw_metadata),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let artifact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU VarDCT artifact"),
        size: layout.artifact_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let occupancy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT occupancy workspace"),
        size: layout.occupancy_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let coefficient_zeros = vec![0i32; layout.coefficient_capacity_words as usize];
    let coefficient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU VarDCT coefficients"),
        contents: bytemuck::cast_slice(&coefficient_zeros),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VarDCT artifact params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let resource_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VarDCT quantization resources"),
        size: u64::from(layout.task_capacity) * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let order_layout = HfOrderTableLayout::new((1 << 0) | (1 << 4)).unwrap();
    let mut order_coordinates = vec![0u32; order_layout.coordinate_words as usize];
    for descriptor in order_layout.descriptors {
        for y in 0..descriptor.height {
            for x in 0..descriptor.width {
                order_coordinates[(descriptor.offset + y * descriptor.width + x) as usize] =
                    x | (y << 16);
            }
        }
    }
    // A custom DCT8/X order sends order slot 1 to the bottom-right frequency.
    let dct8_x = order_layout.descriptor(0, 0).unwrap();
    order_coordinates[(dct8_x.offset + 1) as usize] = 7 | (7 << 16);
    let descriptor_words = bytemuck::cast_slice::<_, u32>(&order_layout.descriptors);
    let order_coordinate_offset_words = descriptor_words.len() as u32;
    let mut order_words = Vec::with_capacity(descriptor_words.len() + order_coordinates.len());
    order_words.extend_from_slice(descriptor_words);
    order_words.extend_from_slice(&order_coordinates);
    let order_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HF packed order table"),
        contents: bytemuck::cast_slice(&order_words),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Compact task order is strategy order: DCT8=0, DCT8x16=1, DCT4x8=2, AFV0=3.
    let tokens = [
        TestCoefficientToken {
            task_index: 0,
            channel: 0,
            order_index: 1,
            value: 11,
        },
        TestCoefficientToken {
            task_index: 0,
            channel: 0,
            order_index: 1,
            value: 2,
        },
        TestCoefficientToken {
            task_index: 1,
            channel: 1,
            order_index: 127,
            value: 19,
        },
        TestCoefficientToken {
            task_index: 3,
            channel: 2,
            order_index: 5,
            value: -7,
        },
    ];
    let token_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test entropy tokens"),
        contents: bytemuck::cast_slice(&tokens),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let token_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test entropy token params"),
        contents: bytemuck::bytes_of(&TestTokenParams {
            values: [tokens.len() as u32, 0, 0, 0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let sink_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HF coefficient sink params"),
        contents: bytemuck::bytes_of(&HfCoefficientSinkParams {
            task_metadata_offset_words: layout.task_metadata_offset_words,
            task_count: 4,
            coefficient_words: layout.coefficient_capacity_words,
            order_descriptor_count: (13 * 3) as u32,
            order_coordinate_offset_words,
            _reserved: [0; 3],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let sink_source = format!(
        r#"{HF_COEFFICIENT_SINK_SHADER}
struct TestCoefficientToken {{
    task_index: u32,
    channel: u32,
    order_index: u32,
    value: i32,
}};
struct TestTokenParams {{
    values: vec4<u32>,
}};
@group(0) @binding(0) var<storage, read> test_tokens: array<TestCoefficientToken>;
@group(0) @binding(1) var<uniform> test_params: TestTokenParams;
@compute @workgroup_size(64, 1, 1)
fn scatter_test(@builtin(global_invocation_id) invocation: vec3<u32>) {{
    if (invocation.x >= test_params.values.x) {{ return; }}
    let token = test_tokens[invocation.x];
    _ = hf_store_quantized_coefficient(
        token.task_index, token.channel, token.order_index, token.value
    );
}}
"#
    );
    let sink_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("HF coefficient sink test"),
        source: wgpu::ShaderSource::Wgsl(sink_source.into()),
    });
    let sink_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("HF coefficient sink test"),
        layout: None,
        module: &sink_module,
        entry_point: Some("scatter_test"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let token_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HF coefficient token bindings"),
        layout: &sink_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: token_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: token_params.as_entire_binding(),
            },
        ],
    });
    let sink_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("HF coefficient sink bindings"),
        layout: &sink_pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: artifact_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: order_table.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: coefficient_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sink_params.as_entire_binding(),
            },
        ],
    });

    let artifact_readback_offset = 0;
    let coefficient_readback_offset = align_256(layout.artifact_bytes);
    let staging_bytes = coefficient_readback_offset + layout.coefficient_bytes;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aggregate VarDCT artifact readback"),
        size: staging_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let lowering = HfMetadataLoweringPipeline::new(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("VarDCT artifact and coefficient commands"),
    });
    lowering.encode(
        &device,
        &mut encoder,
        HfMetadataLoweringBuffers {
            raw_metadata: &raw_buffer,
            artifact: &artifact_buffer,
            occupancy: &occupancy_buffer,
            resources: &resource_buffer,
            params: &params_buffer,
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("HF coefficient sink test"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&sink_pipeline);
        pass.set_bind_group(0, &token_bind_group, &[]);
        pass.set_bind_group(1, &sink_bind_group, &[]);
        pass.dispatch_workgroups((tokens.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &artifact_buffer,
        0,
        &staging,
        artifact_readback_offset,
        layout.artifact_bytes,
    );
    encoder.copy_buffer_to_buffer(
        &coefficient_buffer,
        0,
        &staging,
        coefficient_readback_offset,
        layout.coefficient_bytes,
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("poll VarDCT aggregate readback");
    receiver
        .recv()
        .expect("VarDCT map callback")
        .expect("map VarDCT aggregate output");
    let mapped = staging.slice(..).get_mapped_range().expect("mapped output");
    let bytes = &*mapped;

    let status: GpuVarDctArtifactStatus = cast_one(bytes, 0);
    status.validate().expect("valid GPU varblock tiling");
    assert_eq!(status.task_count, 4);
    assert_eq!(status.coefficient_words, 5 * 192);
    assert_eq!(status.covered_blocks, 5);
    assert_eq!(status.consumed_block_info_entries, 4);
    assert_eq!(
        status.backend_requirements & BACKEND_REQUIREMENT_FREQUENCY_CFL_GRID,
        0
    );

    let buckets_offset = layout.buckets_offset_words as usize * 4;
    let buckets = bytemuck::cast_slice::<u8, GpuVarDctBucket>(
        &bytes[buckets_offset
            ..buckets_offset + VAR_DCT_STRATEGY_COUNT * std::mem::size_of::<GpuVarDctBucket>()],
    );
    assert_eq!((buckets[0].task_offset, buckets[0].task_count), (0, 1));
    assert_eq!((buckets[7].task_offset, buckets[7].task_count), (1, 1));
    assert_eq!((buckets[12].task_offset, buckets[12].task_count), (2, 1));
    assert_eq!((buckets[14].task_offset, buckets[14].task_count), (3, 1));
    let dct8x16_dispatch = layout.bucket_dispatch(buckets[7]).unwrap();
    assert_eq!(dct8x16_dispatch.transform, TransformKind::Dct8x16);
    assert_eq!(dct8x16_dispatch.task_offset, 1);
    assert_eq!(dct8x16_dispatch.task_count, 1);
    assert_eq!(
        dct8x16_dispatch.dequantize_indirect_offset,
        layout
            .indirect_offset(7, HfDispatchStage::Dequantize)
            .unwrap()
    );
    assert_eq!(
        dct8x16_dispatch.horizontal_indirect_offset,
        layout
            .indirect_offset(7, HfDispatchStage::Horizontal)
            .unwrap()
    );
    assert_eq!(
        dct8x16_dispatch.vertical_indirect_offset,
        layout
            .indirect_offset(7, HfDispatchStage::Vertical)
            .unwrap()
    );

    let tasks_offset = layout.tasks_offset_words as usize * 4;
    let tasks = bytemuck::cast_slice::<u8, GpuGeneralVarDctTask>(
        &bytes[tasks_offset..tasks_offset + 4 * std::mem::size_of::<GpuGeneralVarDctTask>()],
    );
    assert_eq!(tasks[0].coefficient_offset, 0);
    assert_eq!(tasks[1].coefficient_offset, 192);
    assert_eq!(tasks[2].coefficient_offset, 768);
    assert_eq!(tasks[3].coefficient_offset, 576);
    assert_eq!(tasks[3].scratch_or_basis_offset, 9_999);
    assert_eq!(
        (tasks[1].coefficient_origin_x, tasks[1].coefficient_origin_y),
        (72, 64)
    );
    assert_eq!(
        (tasks[1].destination_x_x, tasks[1].destination_y_x),
        (72, 64)
    );
    assert_eq!(tasks[1].matrix_offset, 1_000 + 7 * 64);

    let metadata_offset = layout.task_metadata_offset_words as usize * 4;
    let metadata = bytemuck::cast_slice::<u8, GpuHfTaskMetadata>(
        &bytes[metadata_offset..metadata_offset + 4 * std::mem::size_of::<GpuHfTaskMetadata>()],
    );
    assert_eq!((metadata[1].block_width, metadata[1].block_height), (2, 1));
    assert_eq!(metadata[1].order_id, 4);
    assert_eq!(metadata[1].flags & 1, 0);
    assert_eq!(metadata[3].flags & 2, 2);
    assert_eq!(metadata[3].hf_mul, 7);

    let map_offset = layout.block_task_map_offset_words as usize * 4;
    let block_task_map = bytemuck::cast_slice::<u8, u32>(&bytes[map_offset..map_offset + 5 * 4]);
    assert_eq!(block_task_map, [1, 2, 0, 4, 3]);

    let indirect_offset = layout
        .indirect_offset(7, HfDispatchStage::Dequantize)
        .unwrap() as usize;
    let dequant: GpuDispatchIndirectArgs = cast_one(bytes, indirect_offset);
    let horizontal: GpuDispatchIndirectArgs = cast_one(bytes, indirect_offset + 12);
    assert_eq!(dequant, GpuDispatchIndirectArgs { x: 2, y: 1, z: 1 });
    assert_eq!(horizontal, GpuDispatchIndirectArgs { x: 6, y: 1, z: 1 });
    let afv_offset = layout
        .indirect_offset(14, HfDispatchStage::Dequantize)
        .unwrap() as usize;
    assert_eq!(
        cast_one::<GpuDispatchIndirectArgs>(bytes, afv_offset),
        GpuDispatchIndirectArgs { x: 1, y: 1, z: 1 }
    );
    assert_eq!(
        cast_one::<GpuDispatchIndirectArgs>(bytes, afv_offset + 12),
        GpuDispatchIndirectArgs { x: 0, y: 0, z: 0 }
    );

    let coefficient_start = coefficient_readback_offset as usize;
    let coefficients = bytemuck::cast_slice::<u8, i32>(
        &bytes[coefficient_start..coefficient_start + layout.coefficient_bytes as usize],
    );
    assert_eq!(coefficients[63], 13);
    assert_eq!(coefficients[192 + 128 + 127], 19);
    // Special-transform coefficient buffers use raster order, so coordinate (5, 0) is slot 5.
    assert_eq!(coefficients[576 + 2 * 64 + 5], -7);
    assert_eq!(coefficients.iter().filter(|&&value| value != 0).count(), 3);
}
