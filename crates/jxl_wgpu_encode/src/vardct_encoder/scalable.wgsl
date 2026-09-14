// Scalable tiled DCT8 and LF-first large-transform frontend. The first dispatch
// owns one 8x8 block per workgroup, including its AC bits for tiled DCT8; the
// second dispatch owns deterministic LF prediction and serialization. Ending
// the first WebGPU compute pass before beginning the second is the global
// storage-visibility boundary between these entry points.


// Exactly 512 bytes. All artifact offsets and lengths are expressed in u32
// words and are independently checked by the host before dispatch.
struct Params {
    row_stride: u32,
    byte_offset: u32,
    width: u32,
    height: u32,
    blocks_x: u32,
    blocks_y: u32,
    strategy: u32,
    global_scale: u32,
    quant_lf: u32,
    raw_prefix: array<PrefixEntry, 19>,
    strategy_offset: u32,
    dc_offset: u32,
    token_offset: u32,
    extra_offset: u32,
    fragment_offset: u32,
    fragment_word_capacity: u32,
    artifact_words: u32,
    topology: u32,
    fragment_descriptor_offset: u32,
    fragment_descriptor_len: u32,
    lf_groups_x: u32,
    lf_groups_y: u32,
    lf_quantization: array<f32, 3>,
    lf_correlation: array<f32, 2>,
    hf_prefix: array<PrefixEntry, 19>,
    hf_correlation: array<f32, 2>,
    hf_quantization: array<f32, 3>,
    ac_descriptor_offset: u32,
    ac_descriptor_len: u32,
    ac_fragment_offset: u32,
    ac_words_per_block: u32,
    ac_fragment_words: u32,
    padding: array<u32, 16>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(1)
var<storage, read> params: Params;

@group(0) @binding(2)
var<storage, read_write> artifact_words: array<u32>;

// Both vec3 arrays have a 16-byte stride: exactly 2,048 workgroup bytes.
// AC coefficients live only here, never in storage or mapped readback buffers.
var<workgroup> block_xyb: array<vec3<f32>, 64>;
var<workgroup> block_ac: array<vec3<i32>, 64>;

override wg_x: u32 = 64u;

const ARTIFACT_READY: u32 = 0x56444354u;
const HEADER_HISTOGRAM_OFFSET: u32 = 22u;

fn append_fragment_bits(value: u32, count: u32, start: u32) -> u32 {
    let capacity_bits = params.fragment_word_capacity * 32u;
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < capacity_bits {
            let word_index = params.fragment_offset + (bit_offset >> 5u);
            let bit = (value >> index) & 1u;
            artifact_words[word_index] |= bit << (bit_offset & 31u);
        }
    }
    return start + count;
}

fn encode_dc_token(slot: u32, signed_value: i32, start: u32) -> u32 {
    let value = zigzag_signed(signed_value);
    var token = 0u;
    var extra_bit_count = 0u;
    var extra = 0u;
    if value != 0u {
        extra_bit_count = 31u - countLeadingZeros(value);
        token = extra_bit_count + 1u;
        extra = value - (1u << extra_bit_count);
    }
    artifact_words[params.token_offset + slot] = token;
    artifact_words[params.extra_offset + slot] = extra;
    if token < 19u {
        artifact_words[HEADER_HISTOGRAM_OFFSET + token] += 1u;
        let prefix = params.raw_prefix[token];
        let after_prefix = append_fragment_bits(prefix.bits, prefix.bit_len, start);
        return append_fragment_bits(extra, extra_bit_count, after_prefix);
    }
    return params.fragment_word_capacity * 32u + 1u;
}

fn append_ac_bits(base: u32, value: u32, count: u32, start: u32) -> u32 {
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        if bit_offset < params.ac_words_per_block * 32u {
            artifact_words[base + (bit_offset >> 5u)] |=
                ((value >> index) & 1u) << (bit_offset & 31u);
        }
    }
    return start + count;
}

fn encode_ac_unsigned(base: u32, value: u32, start: u32) -> u32 {
    var token = 0u;
    var extra_count = 0u;
    var extra = 0u;
    if value != 0u {
        extra_count = 31u - countLeadingZeros(value);
        token = extra_count + 1u;
        extra = value - (1u << extra_count);
    }
    if token >= 19u {
        return params.ac_words_per_block * 32u + 1u;
    }
    let prefix = params.hf_prefix[token];
    let after_prefix = append_ac_bits(base, prefix.bits, prefix.bit_len, start);
    return append_ac_bits(base, extra, extra_count, after_prefix);
}

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
        bit_offset = encode_ac_unsigned(base, nonzero, bit_offset);
        for (var order = 1u; order < 64u && nonzero != 0u; order += 1u) {
            let value = block_ac[DCT8_NATURAL_ORDER[order]][channel];
            bit_offset = encode_ac_unsigned(base, zigzag_signed(value), bit_offset);
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

    if params.topology == 1u {
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
        if params.topology == 1u {
            serialize_block_ac(block);
        }
    }
}

@compute @workgroup_size(1)
fn serialize_control() {
    let block_count = params.blocks_x * params.blocks_y;
    let sample_count = block_count * 3u;
    var bit_offset = 0u;
    let lf_group_count = params.lf_groups_x * params.lf_groups_y;

    for (var block = 0u; block < block_count; block += 1u) {
        let is_first = block == 0u || params.topology == 1u;
        artifact_words[params.strategy_offset + block] =
            params.strategy | select(0u, 1u << 8u, is_first);
    }
    for (var group = 0u; group < lf_group_count; group += 1u) {
        let group_x = group % params.lf_groups_x;
        let group_y = group / params.lf_groups_x;
        let origin_x = group_x * 256u;
        let origin_y = group_y * 256u;
        let group_width = min(256u, params.blocks_x - origin_x);
        let group_height = min(256u, params.blocks_y - origin_y);
        let group_bit_offset = bit_offset;
        for (var channel = 0u; channel < 3u; channel += 1u) {
            let base = channel * block_count;
            for (var local_y = 0u; local_y < group_height; local_y += 1u) {
                for (var local_x = 0u; local_x < group_width; local_x += 1u) {
                    let block =
                        (origin_y + local_y) * params.blocks_x + origin_x + local_x;
                    var left = 0;
                    if local_x > 0u {
                        left = bitcast<i32>(
                            artifact_words[params.dc_offset + base + block - 1u],
                        );
                    } else if local_y > 0u {
                        left = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x
                            ],
                        );
                    }
                    var top = left;
                    if local_y > 0u {
                        top = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x
                            ],
                        );
                    }
                    var top_left = left;
                    if local_x > 0u && local_y > 0u {
                        top_left = bitcast<i32>(
                            artifact_words[
                                params.dc_offset + base + block - params.blocks_x - 1u
                            ],
                        );
                    }
                    let actual = bitcast<i32>(
                        artifact_words[params.dc_offset + base + block],
                    );
                    bit_offset = encode_dc_token(
                        base + block,
                        actual - clamped_gradient(top, left, top_left),
                        bit_offset,
                    );
                }
            }
        }
        artifact_words[params.fragment_descriptor_offset + 2u * group] = group_bit_offset;
        artifact_words[params.fragment_descriptor_offset + 2u * group + 1u] =
            bit_offset - group_bit_offset;
    }

    // Header ABI. Status is written last so a mapped ready record cannot
    // expose partially initialized live counts or layout metadata.
    artifact_words[1] = block_count;
    artifact_words[2] = sample_count;
    artifact_words[3] = params.strategy;
    artifact_words[4] = u32(params.topology == 1u);
    artifact_words[5] = params.strategy_offset;
    artifact_words[6] = block_count;
    artifact_words[7] = params.dc_offset;
    artifact_words[8] = sample_count;
    artifact_words[9] = params.token_offset;
    artifact_words[10] = sample_count;
    artifact_words[11] = params.extra_offset;
    artifact_words[12] = sample_count;
    artifact_words[13] = params.fragment_offset;
    artifact_words[14] = params.fragment_word_capacity;
    artifact_words[15] = bit_offset;
    artifact_words[16] = params.artifact_words;
    artifact_words[17] = params.width;
    artifact_words[18] = params.height;
    artifact_words[19] = params.blocks_x;
    artifact_words[20] = params.blocks_y;
    artifact_words[21] = params.topology;
    artifact_words[41] = params.fragment_descriptor_offset;
    artifact_words[42] = params.fragment_descriptor_len;
    artifact_words[43] = params.lf_groups_x;
    artifact_words[44] = params.lf_groups_y;
    artifact_words[45] = lf_group_count;
    artifact_words[46] = params.ac_descriptor_offset;
    artifact_words[47] = params.ac_descriptor_len;
    artifact_words[48] = params.ac_fragment_offset;
    artifact_words[49] = params.ac_words_per_block;
    artifact_words[50] = params.ac_fragment_words;
    artifact_words[0] = ARTIFACT_READY;
}
