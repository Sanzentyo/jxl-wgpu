// Shared RGB/XYB conversion, DCT8 quantization, and entropy primitives.
// Both entry-point modules supply source_words and the matching Params fields.

struct QuantizationEntry { dequant: array<f32, 3>, order: array<u32, 3> }

struct PrefixEntry {
    bits: u32,
    bit_len: u32,
}

const PI: f32 = 3.14159265358979323846;
const SQRT_TWO: f32 = 1.41421356237309504880;
const OPSIN_BIAS: f32 = 0.0037930732552754493;
const NEG_OPSIN_BIAS_CBRT: f32 = -0.15595420054924863;
const LF_QUANTIZATION_OVERFLOW: u32 = 0x40000000u;
const HF_QUANTIZATION_OVERFLOW: u32 = 0x80000000u;
const NON_FINITE_SOURCE: u32 = 0x20000000u;
const SOURCE_VALIDATION_INCOMPLETE: u32 = 0x10000000u;
const SOURCE_VALIDATED: u32 = 0x00524345u;
var<workgroup> quantization_error: atomic<u32>;

fn quantize_checked(value: f32, error: u32) -> i32 {
    let rounded = round(value);
    // The upper endpoint is exclusive: f32(i32::MAX) rounds to 2^31.
    if !(rounded >= -2147483648.0 && rounded < 2147483648.0) {
        atomicOr(&quantization_error, error);
        return 0;
    }
    return i32(rounded);
}

fn load_u8(byte_address: u32) -> u32 {
    let word = source_words[byte_address >> 2u];
    return (word >> ((byte_address & 3u) * 8u)) & 255u;
}

// Byte loads preserve arbitrary validated source offsets/strides, including words
// crossing a storage-buffer u32 boundary. Padding bits never enter normalization.
fn load_source_sample(address: u32) -> u32 {
    var value = load_u8(address);
    for (var byte = 1u; byte < params.source_word_bytes; byte += 1u) {
        value |= load_u8(address + byte) << (8u * byte);
    }
    return value & params.source_sample_mask;
}

// All admitted floating precisions fit binary32 exactly. Rebase their fields instead
// of evaluating an exponential, preserving subnormal/sign bits before GPU arithmetic.
fn normalize_source_sample(address: u32) -> f32 {
    let word = load_source_sample(address);
    let exponent_bits = params.source_exponent_bits;
    if exponent_bits == 0u { return f32(word) / f32(params.source_sample_mask); }
    let bits = 32u - countLeadingZeros(params.source_sample_mask);
    let fraction_bits = bits - exponent_bits - 1u;
    let exponent_mask = (1u << exponent_bits) - 1u;
    let exponent = (word >> fraction_bits) & exponent_mask;
    let fraction = word & ((1u << fraction_bits) - 1u);
    let sign = (word >> (bits - 1u)) << 31u;
    if exponent == exponent_mask {
        atomicOr(&quantization_error, NON_FINITE_SOURCE);
        return 0.0;
    }
    let bias = (1u << (exponent_bits - 1u)) - 1u;
    if exponent != 0u {
        return bitcast<f32>(sign | ((exponent + 127u - bias) << 23u)
            | (fraction << (23u - fraction_bits)));
    }
    if fraction == 0u { return bitcast<f32>(sign); }
    if exponent_bits == 8u {
        return bitcast<f32>(sign | (fraction << (23u - fraction_bits)));
    }
    let high_bit = 31u - countLeadingZeros(fraction);
    let rebased_exponent = 128u - bias - fraction_bits + high_bit;
    let rebased_fraction = (fraction ^ (1u << high_bit)) << (23u - high_bit);
    return bitcast<f32>(sign | (rebased_exponent << 23u) | rebased_fraction);
}

fn srgb_to_linear(encoded: f32) -> f32 {
    if abs(encoded) <= 0.04045 {
        return encoded / 12.92;
    }
    return sign(encoded) * pow((abs(encoded) + 0.055) / 1.055, 2.4);
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

// The checked color plan selects the same domain for image/frame headers and quantization.
// 0 = XYB, 1 = original sRGB. No source samples cross the host boundary.
fn normalize_rgb(address: u32) -> vec3<f32> {
    let stride = params.source_word_bytes;
    let encoded = vec3<f32>(
        normalize_source_sample(address),
        normalize_source_sample(address + stride),
        normalize_source_sample(address + 2u * stride),
    );
    if params.color_normalization == 1u { return encoded; }
    return linear_rgb_to_xyb(vec3<f32>(
        srgb_to_linear(encoded.x), srgb_to_linear(encoded.y), srgb_to_linear(encoded.z),
    ));
}

fn dct_basis(frequency: u32, position: u32, size: u32) -> f32 {
    if frequency == 0u {
        return 1.0;
    }
    return SQRT_TWO * cos(
        f32((2u * position + 1u) * frequency) * PI / (2.0 * f32(size)),
    );
}

fn zigzag_signed(value: i32) -> u32 {
    return (bitcast<u32>(value) << 1u) ^ bitcast<u32>(value >> 31u);
}

fn clamped_gradient(top: i32, left: i32, top_left: i32) -> i32 {
    let lower = min(top, left);
    let upper = max(top, left);
    if top_left >= upper { return lower; }
    if top_left <= lower { return upper; }
    // The mathematical result lies between top and left; intermediate i32
    // operations wrap, matching the Modular integer predictor.
    return top + (left - top_left);
}
