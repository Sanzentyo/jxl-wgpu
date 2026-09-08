//! Decode the original JPEG XL sample representation after all integer inverse transforms.
//! Precision is total bits in the low byte, exponent bits in the next byte (zero for integers).

fn modular_sample_is_float(encoding: u32) -> bool { return (encoding >> 8u) != 0u; }
fn modular_sample_maximum(encoding: u32) -> u32 { return 0xffffffffu >> (32u - (encoding & 255u)); }

// Returning bits permits exact storage of zeros, subnormals, infinities and NaN payloads without
// entering the shader's floating arithmetic domain. Filtering/color/blending opt into arithmetic.
fn modular_sample_f32_bits(word: u32, encoding: u32) -> u32 {
    let bits = encoding & 255u;
    let exponent_bits = encoding >> 8u;
    if exponent_bits == 0u {
        return bitcast<u32>(f32(bitcast<i32>(word)) / f32(modular_sample_maximum(encoding)));
    }
    if bits == 32u { return word; }
    let sign_shift = bits - 1u;
    let sign = select(0u, 0x80000000u, (word >> sign_shift) != 0u);
    let magnitude = word & ((1u << sign_shift) - 1u);
    if magnitude == 0u { return sign; }
    let mantissa_bits = bits - exponent_bits - 1u;
    var exponent = i32(magnitude >> mantissa_bits);
    var mantissa = (magnitude & ((1u << mantissa_bits) - 1u)) << (23u - mantissa_bits);
    if exponent == i32((1u << exponent_bits) - 1u) {
        return sign | 0x7f800000u | mantissa;
    }
    if exponent == 0i && exponent_bits < 8u {
        while (mantissa & 0x800000u) == 0u {
            mantissa <<= 1u;
            exponent -= 1i;
        }
        exponent += 1i;
        mantissa &= 0x7fffffu;
    }
    exponent += 127i - i32((1u << (exponent_bits - 1u)) - 1u);
    return sign | (u32(exponent) << 23u) | mantissa;
}
