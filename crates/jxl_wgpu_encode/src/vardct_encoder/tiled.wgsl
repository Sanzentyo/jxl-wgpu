// Tiled DCT8: one workgroup owns each replicated 8x8 input block and AC fragment.
@group(0) @binding(3) var<storage, read> quantization: array<QuantizationEntry, 64>;

// Both vec3 arrays have a 16-byte stride: exactly 2,048 workgroup bytes.
// AC coefficients live only here, never in storage or mapped readback buffers.
var<workgroup> block_components: array<vec3<f32>, 64>;
var<workgroup> block_ac: array<vec3<i32>, 64>;

fn serialize_block_ac(block: u32) {
    let error = atomicLoad(&quantization_error);
    if error != 0u {
        artifact_words[params.ac_descriptor_offset + block] = error;
        return;
    }
    // Fixed word-sized slots have disjoint writes even when adjacent blocks
    // finish mid-word. All 495 contexts use one prefix distribution, so the
    // complete block token sequences can be joined without entropy state.
    for (var pass_index = 0u; pass_index < params.ac_pass_count; pass_index += 1u) {
        let base = params.ac_fragment_offset + pass_index * params.ac_pass_words + block * params.ac_words_per_block;
        var bit_offset = 0u;
        for (var channel_index = 0u; channel_index < 3u; channel_index += 1u) {
            let channel = array<u32, 3>(1u, 0u, 2u)[channel_index];
            var nonzero = 0u;
            for (var order = 1u; order < 64u; order += 1u) {
                let index = quantization[order].order[channel];
                nonzero += u32(progressive_value(block_ac[index][channel], index, 8u, 8u, pass_index) != 0);
            }
            bit_offset = encode_ac_unsigned(base, params.ac_words_per_block, nonzero, bit_offset);
            for (var order = 1u; order < 64u && nonzero != 0u; order += 1u) {
                let index = quantization[order].order[channel];
                let value = progressive_value(block_ac[index][channel], index, 8u, 8u, pass_index);
                bit_offset = encode_ac_unsigned(base, params.ac_words_per_block, zigzag_signed(value), bit_offset);
                nonzero -= u32(value != 0);
            }
        }
        artifact_words[params.ac_descriptor_offset + pass_index * params.ac_descriptor_len + block] = bit_offset;
    }
}

@compute @workgroup_size(wg_x, 1, 1)
fn quantize_blocks(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let block_x = workgroup_id.x;
    let block_y = workgroup_id.y;
    if block_x >= params.blocks_x || block_y >= params.blocks_y {
        return;
    }
    let block_count = params.blocks_x * params.blocks_y;
    let block = block_y * params.blocks_x + block_x;
    for (var sample = local_index; sample < 64u; sample += wg_x) {
        let local_x = sample & 7u;
        let local_y = sample >> 3u;
        // JPEG XL pads a partial edge block by replicating the final source row
        // or column. Keeping the clamped coordinates in the GPU kernel avoids a
        // CPU-side staging/padding fallback for odd and asymmetric dimensions.
        let pixel_x = min(block_x * 8u + local_x, params.width - 1u);
        let pixel_y = min(block_y * 8u + local_y, params.height - 1u);
        block_components[sample] = normalize_rgb(pixel_x, pixel_y);
    }
    workgroupBarrier();

    for (var index = local_index; index < 64u; index += wg_x) {
        if index == 0u {
            continue;
        }
        let fx = index & 7u;
        let fy = index >> 3u;
        var coefficient = vec3<f32>(0.0);
        for (var pixel = 0u; pixel < 64u; pixel += 1u) {
            let basis = dct_basis(fx, pixel & 7u, 8u)
                * dct_basis(fy, pixel >> 3u, 8u) / 64.0;
            coefficient += block_components[pixel] * basis;
        }
        // ComputeScaledDCT's wire layout is transposed (row = horizontal
        // frequency). DC belongs to the separate LF stream.
        block_ac[fx * 8u + fy] = quantize_dct8_ac(coefficient, fx, fy);
    }
    workgroupBarrier();

    if local_index == 0u {
        var sum = vec3<f32>(0.0);
        for (var index = 0u; index < 64u; index += 1u) {
            sum += block_components[index];
        }
        let mean = sum / 64.0;
        let dc_scale = f32(params.global_scale) * f32(params.quant_lf);
        let decorrelated_x = fma(-mean.y, params.lf_correlation[0], mean.x);
        let decorrelated_b = fma(-mean.y, params.lf_correlation[1], mean.z);
        let quantized_y = quantize_checked(mean.y * dc_scale * params.lf_quantization[1], LF_QUANTIZATION_OVERFLOW);
        let quantized_x = quantize_checked(decorrelated_x * dc_scale * params.lf_quantization[0], LF_QUANTIZATION_OVERFLOW);
        let quantized_b = quantize_checked(decorrelated_b * dc_scale * params.lf_quantization[2], LF_QUANTIZATION_OVERFLOW);
        artifact_words[params.dc_offset + block] = bitcast<u32>(quantized_y);
        artifact_words[params.dc_offset + block_count + block] = bitcast<u32>(quantized_x);
        artifact_words[params.dc_offset + 2u * block_count + block] = bitcast<u32>(quantized_b);
        serialize_block_ac(block);
    }
}

fn quantize_dct8_ac(coefficient: vec3<f32>, frequency_x: u32, frequency_y: u32) -> vec3<i32> {
    let decorrelated = vec3<f32>(
        fma(-coefficient.y, params.hf_correlation[0], coefficient.x),
        coefficient.y,
        fma(-coefficient.y, params.hf_correlation[1], coefficient.z),
    );
    let scale = f32(params.global_scale) * f32(params.hf_multiplier) / 65536.0;
    var quantized = vec3<i32>(0);
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let value = decorrelated[channel]
            * scale
            * params.hf_quantization[channel]
            / quantization[frequency_x * 8u + frequency_y].dequant[channel];
        quantized[channel] = quantize_checked(value, HF_QUANTIZATION_OVERFLOW);
    }
    return quantized;
}
