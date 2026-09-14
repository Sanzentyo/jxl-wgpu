use super::*;

#[test]
fn hf_metadata_clamps_signed_samples_before_lowering_quantization_resources() {
    let (device, queue) =
        request_device().expect("actual GPU required for HF multiplier boundaries");
    // Independent expected effective values from libjxl's DecodeAcMetadata:
    // lower and upper saturation also accept the signed i32 endpoints.
    let samples = [i32::MIN, -1, 0, 1, 254, 255, 256, 65_535, i32::MAX];
    let expected = [1, 1, 1, 2, 255, 256, 256, 256, 256];
    let count = samples.len() as u32;
    let mut raw = vec![0i32; samples.len()]; // one DCT8 per block
    raw.extend(samples);
    let mut config = config(raw.len() as u64);
    config.blocks_width = count;
    config.destination_origin = [0, 0];
    config.block_info_entries = count;
    config.hf_mul_offset_words = count;
    config.lf_stride = count;
    config.lf_strides = [count; 3];
    config.correlation_width = count.div_ceil(8);
    config.quant_offset = config.correlation_width;
    let layout = VarDctArtifactLayout::plan(
        &config,
        VarDctArtifactDeviceLimits::from_wgpu(&device.limits()),
    )
    .unwrap();
    let params =
        HfMetadataLoweringParams::new(&config, layout, [0.8, 1.0, 1.0], [0.0, 1.0, 1.0 / 84.0])
            .unwrap();
    let raw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("signed HF metadata endpoints"),
        contents: bytemuck::cast_slice(&raw),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("HF metadata endpoint parameters"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let allocate = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let artifact = allocate(
        "HF metadata endpoint artifact",
        layout.artifact_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let occupancy = allocate(
        "HF metadata endpoint occupancy",
        layout.occupancy_bytes,
        wgpu::BufferUsages::STORAGE,
    );
    let resource_bytes = u64::from(config.quant_offset + count) * 16;
    let resources = allocate(
        "HF metadata endpoint scales",
        resource_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = allocate(
        "HF metadata endpoint readback",
        layout.artifact_bytes + resource_bytes,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut commands = device.create_command_encoder(&Default::default());
    HfMetadataLoweringPipeline::new(&device).encode(
        &device,
        &mut commands,
        HfMetadataLoweringBuffers {
            raw_metadata: &raw,
            artifact: &artifact,
            occupancy: &occupancy,
            resources: &resources,
            params: &params,
        },
    );
    commands.copy_buffer_to_buffer(&artifact, 0, &staging, 0, layout.artifact_bytes);
    commands.copy_buffer_to_buffer(
        &resources,
        0,
        &staging,
        layout.artifact_bytes,
        resource_bytes,
    );
    let submission = queue.submit([commands.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
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
    let bytes = staging.get_mapped_range(..).unwrap();
    let status: GpuVarDctArtifactStatus = cast_one(&bytes, layout.status_offset_words as usize * 4);
    status.validate().unwrap();
    assert_eq!(status.task_count, count);
    for (index, &multiplier) in expected.iter().enumerate() {
        let metadata: GpuHfTaskMetadata = cast_one(
            &bytes,
            layout.task_metadata_offset_words as usize * 4 + index * 48,
        );
        assert_eq!(metadata.hf_mul, multiplier, "raw sample {}", samples[index]);
        let actual: [f32; 4] = cast_one(
            &bytes,
            layout.artifact_bytes as usize + (config.quant_offset as usize + index) * 16,
        );
        let scale = 65536.0 / (8813.0 * multiplier as f32);
        for (actual, factor) in actual[..3].iter().zip([0.8, 1.0, 1.0]) {
            assert!((actual - scale * factor).abs() <= 1.0e-6 * scale);
        }
        assert_eq!(actual[3], 0.0);
    }
    drop(bytes);
    staging.unmap();
}
