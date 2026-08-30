// One standards-profile DCT8 block. The shader owns every data-dependent
// operation: input normalization, transfer function, XYB, forward transform,
// quantization, signed tokenization, histogramming, and prefix bit packing.

struct PrefixEntry {
    bits: u32,
    bit_len: u32,
}

struct Params {
    row_stride: u32,
    byte_offset: u32,
    global_scale: u32,
    quant_lf: u32,
    raw_prefix: array<PrefixEntry, 19>,
}

struct Artifact {
    quantized_dc_yxb: array<i32, 3>,
    dc_raw_tokens: array<u32, 3>,
    dc_extra_bits: array<u32, 3>,
    dc_fragment_words: array<u32, 5>,
    dc_fragment_bit_len: u32,
    raw_histogram: array<u32, 19>,
    forward_xyb_bits: array<u32, 192>,
    quantized_xyb: array<i32, 192>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(1)
var<storage, read> params: Params;

@group(0) @binding(2)
var<storage, read_write> artifact: Artifact;

var<workgroup> block_xyb: array<vec3<f32>, 64>;

const PI: f32 = 3.14159265358979323846;
const SQRT_TWO: f32 = 1.41421356237309504880;
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

fn dct_basis(frequency: u32, position: u32) -> f32 {
    if frequency == 0u {
        return 1.0;
    }
    return SQRT_TWO * cos(f32((2u * position + 1u) * frequency) * PI / 16.0);
}

fn zigzag_signed(value: i32) -> u32 {
    if value < 0 {
        return u32(-value) * 2u - 1u;
    }
    return u32(value) * 2u;
}

fn append_fragment_bits(value: u32, count: u32, start: u32) -> u32 {
    for (var index = 0u; index < count; index += 1u) {
        let bit_offset = start + index;
        let bit = (value >> index) & 1u;
        artifact.dc_fragment_words[bit_offset >> 5u] |= bit << (bit_offset & 31u);
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
    artifact.dc_raw_tokens[slot] = token;
    artifact.dc_extra_bits[slot] = extra;
    artifact.raw_histogram[token] += 1u;
    let prefix = params.raw_prefix[token];
    let after_prefix = append_fragment_bits(prefix.bits, prefix.bit_len, start);
    return append_fragment_bits(extra, extra_bit_count, after_prefix);
}

@compute @workgroup_size(64)
fn encode(@builtin(local_invocation_index) local_index: u32) {
    let pixel_x = local_index & 7u;
    let pixel_y = local_index >> 3u;
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
    block_xyb[local_index] = linear_rgb_to_xyb(linear);
    workgroupBarrier();

    // Coefficient index is frequency_x-major, matching the decoder's DCT8 ABI.
    let frequency_x = local_index >> 3u;
    let frequency_y = local_index & 7u;
    var coefficient = vec3<f32>(0.0);
    for (var pixel = 0u; pixel < 64u; pixel += 1u) {
        let x = pixel & 7u;
        let y = pixel >> 3u;
        let basis = dct_basis(frequency_x, x) * dct_basis(frequency_y, y) / 64.0;
        coefficient += block_xyb[pixel] * basis;
    }
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let offset = channel * 64u + local_index;
        artifact.forward_xyb_bits[offset] = bitcast<u32>(coefficient[channel]);
        // This first profile intentionally spends its full rate on LF. The
        // forward AC values remain observable in the typed artifact, while
        // quantization maps them to the all-zero HF entropy distribution.
        artifact.quantized_xyb[offset] = 0;
    }

    if local_index == 0u {
        let dc_scale = f32(params.global_scale * params.quant_lf);
        let q_x = i32(round(coefficient.x * dc_scale / 16.0));
        let q_y = i32(round(coefficient.y * dc_scale / 128.0));
        let q_b = i32(round((coefficient.z - coefficient.y) * dc_scale / 256.0));
        artifact.quantized_dc_yxb[0] = q_y;
        artifact.quantized_dc_yxb[1] = q_x;
        artifact.quantized_dc_yxb[2] = q_b;
        artifact.quantized_xyb[0] = q_x;
        artifact.quantized_xyb[64] = q_y;
        artifact.quantized_xyb[128] = q_b;

        for (var index = 0u; index < 5u; index += 1u) {
            artifact.dc_fragment_words[index] = 0u;
        }
        for (var index = 0u; index < 19u; index += 1u) {
            artifact.raw_histogram[index] = 0u;
        }
        var bit_offset = 0u;
        bit_offset = encode_dc_token(0u, q_y, bit_offset);
        bit_offset = encode_dc_token(1u, q_x, bit_offset);
        bit_offset = encode_dc_token(2u, q_b, bit_offset);
        artifact.dc_fragment_bit_len = bit_offset;
    }
}
