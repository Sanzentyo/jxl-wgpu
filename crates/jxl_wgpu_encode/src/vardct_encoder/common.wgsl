// Shared RGB/XYB conversion, DCT8 quantization, and entropy primitives.
// Both entry-point modules supply source_words and the matching Params fields.

struct PrefixEntry {
    bits: u32,
    bit_len: u32,
}

const PI: f32 = 3.14159265358979323846;
const SQRT_TWO: f32 = 1.41421356237309504880;
const OPSIN_BIAS: f32 = 0.0037930732552754493;
const NEG_OPSIN_BIAS_CBRT: f32 = -0.15595420054924863;
const MAX_HF_QUANTIZED_MAGNITUDE: i32 = 131071;
const DCT8_NATURAL_ORDER: array<u32, 64> = array<u32, 64>(
    0u, 1u, 8u, 16u, 9u, 2u, 3u, 10u, 17u, 24u, 32u, 25u, 18u, 11u, 4u, 5u,
    12u, 19u, 26u, 33u, 40u, 48u, 41u, 34u, 27u, 20u, 13u, 6u, 7u, 14u, 21u, 28u,
    35u, 42u, 49u, 56u, 57u, 50u, 43u, 36u, 29u, 22u, 15u, 23u, 30u, 37u, 44u, 51u,
    58u, 59u, 52u, 45u, 38u, 31u, 39u, 46u, 53u, 60u, 61u, 54u, 47u, 55u, 62u, 63u,
);

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

fn dct_basis(frequency: u32, position: u32, size: u32) -> f32 {
    if frequency == 0u {
        return 1.0;
    }
    return SQRT_TWO * cos(
        f32((2u * position + 1u) * frequency) * PI / (2.0 * f32(size)),
    );
}

fn dct8_weight_parameter(channel: u32, index: u32) -> f32 {
    if channel == 0u {
        return array<f32, 6>(3150.0, 0.0, -0.4, -0.4, -0.4, -2.0)[index];
    }
    if channel == 1u {
        return array<f32, 6>(560.0, 0.0, -0.3, -0.3, -0.3, -0.3)[index];
    }
    return array<f32, 6>(512.0, -2.0, -1.0, 0.0, -1.0, -2.0)[index];
}

fn dct8_quant_weight(channel: u32, frequency_x: u32, frequency_y: u32) -> f32 {
    var bands: array<f32, 6>;
    bands[0] = dct8_weight_parameter(channel, 0u);
    for (var index = 1u; index < 6u; index += 1u) {
        let parameter = dct8_weight_parameter(channel, index);
        let multiplier = select(1.0 / (1.0 - parameter), 1.0 + parameter, parameter > 0.0);
        bands[index] = bands[index - 1u] * multiplier;
    }
    let dx = f32(frequency_x) / 7.0;
    let dy = f32(frequency_y) / 7.0;
    let scaled_position = sqrt(dx * dx + dy * dy) * 5.0 / (SQRT_TWO + 1e-6);
    let band_index = min(u32(scaled_position), 4u);
    let fraction = scaled_position - f32(band_index);
    let lower = bands[band_index];
    let upper = bands[band_index + 1u];
    return lower * pow(upper / lower, fraction);
}

fn quantize_dct8_ac(coefficient: vec3<f32>, frequency_x: u32, frequency_y: u32) -> vec3<i32> {
    let decorrelated = vec3<f32>(
        fma(-coefficient.y, params.hf_correlation[0], coefficient.x),
        coefficient.y,
        fma(-coefficient.y, params.hf_correlation[1], coefficient.z),
    );
    let scale = f32(params.global_scale) * 6.0 / 65536.0;
    var quantized = vec3<i32>(0);
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let value = decorrelated[channel]
            * scale
            * params.hf_quantization[channel]
            * dct8_quant_weight(channel, frequency_x, frequency_y);
        quantized[channel] = clamp(
            i32(round(value)),
            -MAX_HF_QUANTIZED_MAGNITUDE,
            MAX_HF_QUANTIZED_MAGNITUDE,
        );
    }
    return quantized;
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

