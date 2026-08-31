// Scalable LF-first VarDCT frontend for the nine regular strategies larger
// than 32x32. The first dispatch owns one 8x8 block per workgroup; the second
// dispatch owns deterministic prediction and entropy serialization. Ending
// the first WebGPU compute pass before beginning the second is the global
// storage-visibility boundary between these entry points.

struct PrefixEntry {
    bits: u32,
    bit_len: u32,
}

// Exactly 256 bytes. All artifact offsets and lengths are expressed in u32
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
    padding: array<u32, 9>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(1)
var<storage, read> params: Params;

@group(0) @binding(2)
var<storage, read_write> artifact_words: array<u32>;

// vec3<f32> has a 16-byte storage stride, so the complete reduction consumes
// 1,024 bytes: far below WebGPU's portable 16 KiB workgroup-storage floor.
var<workgroup> block_xyb: array<vec3<f32>, 64>;

override wg_x: u32 = 64u;

const ARTIFACT_READY: u32 = 0x56444354u;
const HEADER_HISTOGRAM_OFFSET: u32 = 22u;
const OPSIN_BIAS: f32 = 0.0037930732552754493;
const NEG_OPSIN_BIAS_CBRT: f32 = -0.15595420054924863;

fn load_u8(byte_address: u32) -> u32 {
    let word = source_words[byte_address >> 2u];
    return (word >> ((byte_address & 3u) * 8u)) & 255u;
}

fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        return encoded / 12.92;
    }
    return pow((encoded + 0.055) / 1.055, 2.4);
}

fn linear_rgb_to_xyb(rgb: vec3<f32>) -> vec3<f32> {
    let mixed = max(
        vec3<f32>(
            0.3000000000 * rgb.x + 0.6220000000 * rgb.y + 0.0780000000 * rgb.z,
            0.2300000000 * rgb.x + 0.6920000000 * rgb.y + 0.0780000000 * rgb.z,
            0.2434226892 * rgb.x + 0.2047674442 * rgb.y + 0.5518098665 * rgb.z,
        ) + vec3<f32>(OPSIN_BIAS),
        vec3<f32>(0.0),
    );
    let absorbance = vec3<f32>(
        pow(mixed.x, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
        pow(mixed.y, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
        pow(mixed.z, 1.0 / 3.0) + NEG_OPSIN_BIAS_CBRT,
    );
    return vec3<f32>(
        0.5 * (absorbance.x - absorbance.y),
        0.5 * (absorbance.x + absorbance.y),
        absorbance.z,
    );
}

fn zigzag_signed(value: i32) -> u32 {
    if value < 0 {
        return u32(-value) * 2u - 1u;
    }
    return u32(value) * 2u;
}

fn clamped_gradient(top: i32, left: i32, top_left: i32) -> i32 {
    return clamp(top + left - top_left, min(top, left), max(top, left));
}

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

@compute @workgroup_size(wg_x, 1, 1)
fn quantize_blocks(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let block_count = params.blocks_x * params.blocks_y;
    let block = workgroup_id.x;
    if block >= block_count {
        return;
    }
    let block_x = block % params.blocks_x;
    let block_y = block / params.blocks_x;
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

    if local_index == 0u {
        var sum = vec3<f32>(0.0);
        for (var index = 0u; index < 64u; index += 1u) {
            sum += block_xyb[index];
        }
        let mean = sum / 64.0;
        let dc_scale = f32(params.global_scale * params.quant_lf);
        let quantized_y = i32(round(mean.y * dc_scale / 128.0));
        let quantized_x = i32(round(mean.x * dc_scale / 16.0));
        let quantized_b = i32(round((mean.z - mean.y) * dc_scale / 256.0));
        artifact_words[params.dc_offset + block] = bitcast<u32>(quantized_y);
        artifact_words[params.dc_offset + block_count + block] = bitcast<u32>(quantized_x);
        artifact_words[params.dc_offset + 2u * block_count + block] = bitcast<u32>(quantized_b);
    }
}

@compute @workgroup_size(1)
fn serialize_control() {
    let block_count = params.blocks_x * params.blocks_y;
    let sample_count = block_count * 3u;
    var bit_offset = 0u;

    for (var block = 0u; block < block_count; block += 1u) {
        let is_first = block == 0u || params.topology == 1u;
        artifact_words[params.strategy_offset + block] =
            params.strategy | select(0u, 1u << 8u, is_first);
    }
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let base = channel * block_count;
        for (var block = 0u; block < block_count; block += 1u) {
            let block_x = block % params.blocks_x;
            let block_y = block / params.blocks_x;
            var left = 0;
            if block_x > 0u {
                left = bitcast<i32>(artifact_words[params.dc_offset + base + block - 1u]);
            } else if block_y > 0u {
                left = bitcast<i32>(
                    artifact_words[params.dc_offset + base + block - params.blocks_x],
                );
            }
            var top = left;
            if block_y > 0u {
                top = bitcast<i32>(
                    artifact_words[params.dc_offset + base + block - params.blocks_x],
                );
            }
            var top_left = left;
            if block_x > 0u && block_y > 0u {
                top_left = bitcast<i32>(
                    artifact_words[
                        params.dc_offset + base + block - params.blocks_x - 1u
                    ],
                );
            }
            let actual = bitcast<i32>(artifact_words[params.dc_offset + base + block]);
            bit_offset = encode_dc_token(
                base + block,
                actual - clamped_gradient(top, left, top_left),
                bit_offset,
            );
        }
    }

    // Header ABI. Status is written last so a mapped ready record cannot
    // expose partially initialized live counts or layout metadata.
    artifact_words[1] = block_count;
    artifact_words[2] = sample_count;
    artifact_words[3] = params.strategy;
    artifact_words[4] = 1u; // AC coefficients are deliberately quantized to zero.
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
    artifact_words[0] = ARTIFACT_READY;
}
