// Image normalization and transform-parallel quantization/entropy coding.
struct QuantizationEntry { dequant: vec3<f32>, order: u32 }
struct TransformTask {
    block_x: u32, block_y: u32, coefficient_offset: u32, lf_offset: u32,
    width: u32, height: u32, metadata_offset: u32, ac_word_offset: u32,
    ac_word_capacity: u32, strategy: u32,
}
@group(0) @binding(3) var<storage, read> forward_coefficients: array<f32>;
@group(0) @binding(4) var<storage, read> forward_lf: array<f32>;
@group(0) @binding(5) var<storage, read_write> forward_xyb: array<f32>;
@group(0) @binding(6) var<storage, read_write> quantized_coefficients: array<i32>;
@group(0) @binding(7) var<storage, read> quantization: array<QuantizationEntry>;
@group(0) @binding(8) var<storage, read> tasks: array<TransformTask>;

fn transform_index(group: vec3<u32>) -> u32 {
    return group.y * params.workgroups_x + group.x;
}

@compute @workgroup_size(wg_x)
fn normalize_image(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let pixel = transform_index(group) * wg_x + lane;
    let width = params.blocks_x * 8u;
    let area = width * params.blocks_y * 8u;
    if pixel >= area { return; }
    let x = min(pixel % width, params.width - 1u);
    let y = min(pixel / width, params.height - 1u);
    let address = params.byte_offset + y * params.row_stride + x * 3u;
    let linear = vec3<f32>(srgb_to_linear(f32(load_u8(address)) / 255.0),
        srgb_to_linear(f32(load_u8(address + 1u)) / 255.0),
        srgb_to_linear(f32(load_u8(address + 2u)) / 255.0));
    let xyb = linear_rgb_to_xyb(linear);
    forward_xyb[pixel] = xyb.x;
    forward_xyb[area + pixel] = xyb.y;
    forward_xyb[2u * area + pixel] = xyb.z;
}

@compute @workgroup_size(wg_x)
fn quantize_transforms_ac(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let task_index = transform_index(group);
    if task_index >= params.ac_descriptor_len { return; }
    let task = tasks[task_index];
    let area = task.width * task.height;
    for (var index = lane; index < area; index += wg_x) {
        let offset = task.coefficient_offset + index;
        let coefficient = vec3<f32>(forward_coefficients[offset],
            forward_coefficients[offset + area], forward_coefficients[offset + 2u * area]);
        let decorrelated = vec3<f32>(fma(-coefficient.y, params.hf_correlation[0], coefficient.x),
            coefficient.y, fma(-coefficient.y, params.hf_correlation[1], coefficient.z));
        let scale = f32(params.global_scale) * 6.0 / 65536.0;
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let value = decorrelated[channel] * scale * params.hf_quantization[channel]
                / quantization[task.metadata_offset + index].dequant[channel];
            quantized_coefficients[offset + channel * area] = clamp(i32(round(value)),
                -MAX_HF_QUANTIZED_MAGNITUDE, MAX_HF_QUANTIZED_MAGNITUDE);
        }
    }
}

@compute @workgroup_size(wg_x)
fn quantize_transforms_lf(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let task_index = transform_index(group);
    if task_index >= params.ac_descriptor_len { return; }
    let task = tasks[task_index];
    let width = task.width / 8u;
    let count = width * task.height / 8u;
    let canvas_count = params.blocks_x * params.blocks_y;
    for (var block = lane; block < count; block += wg_x) {
        let offset = task.lf_offset + block;
        let lf = vec3<f32>(forward_lf[offset], forward_lf[offset + count], forward_lf[offset + 2u * count]);
        let scale = f32(params.global_scale * params.quant_lf);
        let qy = i32(round(lf.y * scale * params.lf_quantization[1]));
        let qx = i32(round(fma(-lf.y, params.lf_correlation[0], lf.x) * scale * params.lf_quantization[0]));
        let qb = i32(round(fma(-lf.y, params.lf_correlation[1], lf.z) * scale * params.lf_quantization[2]));
        let destination = (task.block_y + block / width) * params.blocks_x + task.block_x + block % width;
        artifact_words[params.dc_offset + destination] = bitcast<u32>(qy);
        artifact_words[params.dc_offset + canvas_count + destination] = bitcast<u32>(qx);
        artifact_words[params.dc_offset + 2u * canvas_count + destination] = bitcast<u32>(qb);
        artifact_words[params.strategy_offset + destination] = task.strategy | (u32(block == 0u) << 8u);
    }
}

@compute @workgroup_size(1)
fn serialize_transforms_ac(@builtin(workgroup_id) group: vec3<u32>) {
    let task_index = transform_index(group);
    if task_index >= params.ac_descriptor_len { return; }
    let task = tasks[task_index];
    let area = task.width * task.height;
    let llf = area / 64u;
    let base = params.ac_fragment_offset + task.ac_word_offset;
    var bit_offset = 0u;
    for (var channel_index = 0u; channel_index < 3u; channel_index += 1u) {
        let channel = array<u32, 3>(1u, 0u, 2u)[channel_index];
        let offset = task.coefficient_offset + channel * area;
        var nonzero = 0u;
        for (var order = llf; order < area; order += 1u) {
            nonzero += u32(quantized_coefficients[offset + quantization[task.metadata_offset + order].order] != 0);
        }
        bit_offset = encode_ac_unsigned(base, task.ac_word_capacity, nonzero, bit_offset);
        for (var order = llf; order < area && nonzero != 0u; order += 1u) {
            let value = quantized_coefficients[offset + quantization[task.metadata_offset + order].order];
            bit_offset = encode_ac_unsigned(base, task.ac_word_capacity, zigzag_signed(value), bit_offset);
            nonzero -= u32(value != 0);
        }
    }
    artifact_words[params.ac_descriptor_offset + task_index] = bit_offset;
}
