//! Decode the original JPEG XL sample representation after all integer inverse transforms.
//! Precision is total bits in the low byte, exponent bits in the next byte (zero for integers).

fn modular_sample_is_float(encoding: u32) -> bool { return (encoding >> 8u) != 0u; }
fn modular_sample_maximum(encoding: u32) -> u32 { return 0xffffffffu >> (32u - (encoding & 255u)); }

fn modular_unsigned_product(a: u32, b: u32) -> vec2<u32> {
    let p0 = (a & 65535u) * (b & 65535u);
    let p1 = (a & 65535u) * (b >> 16u);
    let p2 = (a >> 16u) * (b & 65535u);
    let low1 = p0 + (p1 << 16u);
    let low = low1 + (p2 << 16u);
    return vec2<u32>(low, (a >> 16u) * (b >> 16u) + (p1 >> 16u) + (p2 >> 16u)
        + u32(low1 < p0) + u32(low < low1));
}

// Round the binary32 value times the exact integer maximum. F32 multiplication cannot retain
// the requested low bits at wide depths and rounds 25–31-bit maxima to a power of two.
fn modular_quantize_unsigned(value: f32, maximum: u32) -> u32 {
    if !(value > 0.0) { return 0u; }
    if value >= 1.0 { return maximum; }
    let word = bitcast<u32>(value);
    let exponent = word >> 23u;
    if exponent < 95u { return 0u; }
    let shift = 150u - exponent;
    var product = modular_unsigned_product((word & 0x7fffffu) | 0x800000u, maximum);
    if shift > 32u {
        return (product.y + (1u << (shift - 33u))) >> (shift - 32u);
    }
    let low = product.x + (1u << (shift - 1u));
    product.y += u32(low < product.x);
    if shift == 32u { return product.y; }
    return (low >> shift) | (product.y << (32u - shift));
}

// Exact round-to-nearest rescaling of independently declared integer alpha, up to 31 bits.
// The two-limb numerator avoids overflowing when either precision exceeds 16 bits.
fn modular_rescale_unsigned(value: u32, source_maximum: u32, target_maximum: u32) -> u32 {
    if source_maximum == target_maximum { return value; }
    if source_maximum <= 65535u && target_maximum <= 65535u {
        return (value * target_maximum + source_maximum / 2u) / source_maximum;
    }
    let product = modular_unsigned_product(value, target_maximum);
    let low = product.x + source_maximum / 2u;
    var remainder = product.y + u32(low < product.x);
    var quotient = 0u;
    for (var bit = 32u; bit > 0u; bit -= 1u) {
        remainder = (remainder << 1u) | ((low >> (bit - 1u)) & 1u);
        if remainder >= source_maximum {
            remainder -= source_maximum;
            quotient |= 1u << (bit - 1u);
        }
    }
    return quotient;
}

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
