@group(0) @binding(3) var<storage, read_write> icc_device_samples: array<f32>;
override icc_channels: u32 = 3u;

@compute @workgroup_size(wg_x)
fn normalize_icc_input(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group_index = group.y * params.workgroups_x + group.x;
    let pixel = group_index * wg_x + lane;
    let width = params.blocks_x * 8u;
    let area = width * params.blocks_y * 8u;
    if pixel < area {
        let x = min(pixel % width, params.width - 1u);
        let y = min(pixel / width, params.height - 1u);
        for (var channel = 0u; channel < icc_channels; channel += 1u) {
            icc_device_samples[channel * area + pixel] = normalize_source_sample(source_sample(x, y, channel));
        }
    }
    workgroupBarrier();
    if lane == 0u && group_index < params.source_validation_groups {
        artifact_words[params.source_validation_offset + group_index] =
            SOURCE_VALIDATED | atomicLoad(&quantization_error);
    }
}
