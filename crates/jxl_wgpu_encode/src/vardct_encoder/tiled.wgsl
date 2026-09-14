// Tiled DCT8: one workgroup owns each replicated 8x8 input block and AC fragment.

// Both vec3 arrays have a 16-byte stride: exactly 2,048 workgroup bytes.
// AC coefficients live only here, never in storage or mapped readback buffers.
var<workgroup> block_xyb: array<vec3<f32>, 64>;
var<workgroup> block_ac: array<vec3<i32>, 64>;

fn serialize_block_ac(block: u32) {
    // Fixed word-sized slots have disjoint writes even when adjacent blocks
    // finish mid-word. All 495 contexts use one prefix distribution, so the
    // complete block token sequences can be joined without entropy state.
    let base = params.ac_fragment_offset + block * params.ac_words_per_block;
    var bit_offset = 0u;
    for (var channel_index = 0u; channel_index < 3u; channel_index += 1u) {
        let channel = array<u32, 3>(1u, 0u, 2u)[channel_index];
        var nonzero = 0u;
        for (var order = 1u; order < 64u; order += 1u) {
            nonzero += u32(block_ac[DCT8_NATURAL_ORDER[order]][channel] != 0);
        }
        bit_offset = encode_ac_unsigned(base, params.ac_words_per_block, nonzero, bit_offset);
        for (var order = 1u; order < 64u && nonzero != 0u; order += 1u) {
            let value = block_ac[DCT8_NATURAL_ORDER[order]][channel];
            bit_offset = encode_ac_unsigned(base, params.ac_words_per_block, zigzag_signed(value), bit_offset);
            nonzero -= u32(value != 0);
        }
    }
    artifact_words[params.ac_descriptor_offset + block] = bit_offset;
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
        let pixel_address = params.byte_offset + pixel_y * params.row_stride + pixel_x * 3u;
        let encoded = vec3<f32>(
            f32(load_u8(pixel_address)) / 255.0,
            f32(load_u8(pixel_address + 1u)) / 255.0,
            f32(load_u8(pixel_address + 2u)) / 255.0,
        );
        let linear = vec3<f32>(
            srgb_to_linear(encoded.x),
            srgb_to_linear(encoded.y),
            srgb_to_linear(encoded.z),
        );
        block_xyb[sample] = linear_rgb_to_xyb(linear);
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
            coefficient += block_xyb[pixel] * basis;
        }
        // ComputeScaledDCT's wire layout is transposed (row = horizontal
        // frequency). DC belongs to the separate LF stream.
        block_ac[fx * 8u + fy] = quantize_dct8_ac(coefficient, fx, fy);
    }
    workgroupBarrier();

    if local_index == 0u {
        var sum = vec3<f32>(0.0);
        for (var index = 0u; index < 64u; index += 1u) {
            sum += block_xyb[index];
        }
        let mean = sum / 64.0;
        let dc_scale = f32(params.global_scale * params.quant_lf);
        let decorrelated_x = fma(-mean.y, params.lf_correlation[0], mean.x);
        let decorrelated_b = fma(-mean.y, params.lf_correlation[1], mean.z);
        let quantized_y = i32(round(mean.y * dc_scale * params.lf_quantization[1]));
        let quantized_x = i32(round(
            decorrelated_x * dc_scale * params.lf_quantization[0],
        ));
        let quantized_b = i32(round(
            decorrelated_b * dc_scale * params.lf_quantization[2],
        ));
        artifact_words[params.dc_offset + block] = bitcast<u32>(quantized_y);
        artifact_words[params.dc_offset + block_count + block] = bitcast<u32>(quantized_x);
        artifact_words[params.dc_offset + 2u * block_count + block] = bitcast<u32>(quantized_b);
        serialize_block_ac(block);
    }
}

