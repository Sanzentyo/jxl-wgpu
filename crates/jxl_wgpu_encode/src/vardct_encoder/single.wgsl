// General single-transform frontend. Shared Params, color conversion, prefix
// writer and LF control serializer come from common.wgsl and control.wgsl.

struct QuantizationEntry {
    dequant: vec3<f32>,
    order: u32,
}

@group(0) @binding(3) var<storage, read> forward_coefficients: array<f32>;
@group(0) @binding(4) var<storage, read> forward_lf: array<f32>;
@group(0) @binding(5) var<storage, read_write> forward_xyb: array<f32>;
@group(0) @binding(6) var<storage, read_write> quantized_coefficients: array<i32>;
@group(0) @binding(7) var<storage, read> quantization: array<QuantizationEntry>;

fn single_index(group: vec3<u32>, lane: u32) -> u32 {
    return (group.y * params.workgroups_x + group.x) * wg_x + lane;
}

@compute @workgroup_size(wg_x)
fn normalize_single(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let pixel = single_index(group, lane);
    let area = params.width * params.height;
    if pixel >= area { return; }
    let address = params.byte_offset + pixel / params.width * params.row_stride + pixel % params.width * 3u;
    let linear = vec3<f32>(srgb_to_linear(f32(load_u8(address)) / 255.0),
        srgb_to_linear(f32(load_u8(address + 1u)) / 255.0),
        srgb_to_linear(f32(load_u8(address + 2u)) / 255.0));
    let xyb = linear_rgb_to_xyb(linear);
    forward_xyb[pixel] = xyb.x;
    forward_xyb[area + pixel] = xyb.y;
    forward_xyb[2u * area + pixel] = xyb.z;
}

@compute @workgroup_size(wg_x)
fn quantize_single_ac(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = single_index(group, lane);
    let area = params.width * params.height;
    if index >= area { return; }
    let coefficient = vec3<f32>(forward_coefficients[index],
        forward_coefficients[area + index], forward_coefficients[2u * area + index]);
    let decorrelated = vec3<f32>(fma(-coefficient.y, params.hf_correlation[0], coefficient.x),
        coefficient.y, fma(-coefficient.y, params.hf_correlation[1], coefficient.z));
    let scale = f32(params.global_scale) * 6.0 / 65536.0;
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let value = decorrelated[channel] * scale * params.hf_quantization[channel]
            / quantization[index].dequant[channel];
        quantized_coefficients[channel * area + index] = clamp(i32(round(value)),
            -MAX_HF_QUANTIZED_MAGNITUDE, MAX_HF_QUANTIZED_MAGNITUDE);
    }
}

@compute @workgroup_size(wg_x)
fn quantize_single_lf(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let block = single_index(group, lane);
    let count = params.blocks_x * params.blocks_y;
    if block >= count { return; }
    let lf = vec3<f32>(forward_lf[block], forward_lf[count + block], forward_lf[2u * count + block]);
    let scale = f32(params.global_scale * params.quant_lf);
    let qy = i32(round(lf.y * scale * params.lf_quantization[1]));
    let qx = i32(round(fma(-lf.y, params.lf_correlation[0], lf.x) * scale * params.lf_quantization[0]));
    let qb = i32(round(fma(-lf.y, params.lf_correlation[1], lf.z) * scale * params.lf_quantization[2]));
    artifact_words[params.dc_offset + block] = bitcast<u32>(qy);
    artifact_words[params.dc_offset + count + block] = bitcast<u32>(qx);
    artifact_words[params.dc_offset + 2u * count + block] = bitcast<u32>(qb);
}

@compute @workgroup_size(1)
fn serialize_single_ac() {
    let area = params.width * params.height;
    let llf = params.blocks_x * params.blocks_y;
    var bit_offset = 0u;
    for (var channel_index = 0u; channel_index < 3u; channel_index += 1u) {
        let channel = array<u32, 3>(1u, 0u, 2u)[channel_index];
        var nonzero = 0u;
        for (var order = llf; order < area; order += 1u) {
            nonzero += u32(quantized_coefficients[channel * area + quantization[order].order] != 0);
        }
        bit_offset = encode_ac_unsigned(params.ac_fragment_offset, nonzero, bit_offset);
        for (var order = llf; order < area && nonzero != 0u; order += 1u) {
            let value = quantized_coefficients[channel * area + quantization[order].order];
            bit_offset = encode_ac_unsigned(params.ac_fragment_offset, zigzag_signed(value), bit_offset);
            nonzero -= u32(value != 0);
        }
    }
    artifact_words[params.ac_descriptor_offset] = bit_offset;
}
