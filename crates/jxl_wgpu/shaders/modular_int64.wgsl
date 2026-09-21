//! Two-word two's-complement arithmetic for JPEG XL's signed 64-bit intermediates.
//! Storage and committed predictor error state remain 32-bit words.
alias ModularI64 = vec2<u32>; // low word, high word

fn mi_from_i32(value: i32) -> ModularI64 {
    return vec2<u32>(bitcast<u32>(value), bitcast<u32>(value >> 31u));
}
fn mi_add(a: ModularI64, b: ModularI64) -> ModularI64 {
    let low = a.x + b.x;
    return vec2<u32>(low, a.y + b.y + u32(low < a.x));
}
fn mi_neg(value: ModularI64) -> ModularI64 {
    return mi_add(~value, vec2<u32>(1u, 0u));
}
fn mi_sub(a: ModularI64, b: ModularI64) -> ModularI64 {
    return mi_add(a, mi_neg(b));
}
fn mi_negative(value: ModularI64) -> bool { return (value.y & 0x80000000u) != 0u; }
fn mi_abs(value: ModularI64) -> ModularI64 {
    return select(value, mi_neg(value), mi_negative(value));
}
fn mi_less(a: ModularI64, b: ModularI64) -> bool {
    return bitcast<i32>(a.y) < bitcast<i32>(b.y) || (a.y == b.y && a.x < b.x);
}
fn mi_min(a: ModularI64, b: ModularI64) -> ModularI64 { return select(b, a, mi_less(a, b)); }
fn mi_max(a: ModularI64, b: ModularI64) -> ModularI64 { return select(a, b, mi_less(a, b)); }
fn mi_shl(value: ModularI64, shift: u32) -> ModularI64 {
    if shift == 0u { return value; }
    if shift >= 64u { return vec2<u32>(0u); }
    if shift >= 32u { return vec2<u32>(0u, value.x << (shift - 32u)); }
    return vec2<u32>(value.x << shift, (value.y << shift) | (value.x >> (32u - shift)));
}
fn mi_sar(value: ModularI64, shift: u32) -> ModularI64 {
    if shift == 0u { return value; }
    let high = bitcast<i32>(value.y);
    if shift >= 64u { return vec2<u32>(bitcast<u32>(high >> 31u)); }
    if shift >= 32u { return vec2<u32>(bitcast<u32>(high >> (shift - 32u)), bitcast<u32>(high >> 31u)); }
    return vec2<u32>((value.x >> shift) | (value.y << (32u - shift)), bitcast<u32>(high >> shift));
}
fn mi_shr(value: ModularI64, shift: u32) -> ModularI64 {
    if shift == 0u { return value; }
    if shift >= 64u { return vec2<u32>(0u); }
    if shift >= 32u { return vec2<u32>(value.y >> (shift - 32u), 0u); }
    return vec2<u32>((value.x >> shift) | (value.y << (32u - shift)), value.y >> shift);
}
// Division truncates toward zero, unlike arithmetic right shift for negative values.
fn mi_div_pow2(value: ModularI64, shift: u32) -> ModularI64 {
    if mi_negative(value) { return mi_neg(mi_shr(mi_abs(value), shift)); }
    return mi_sar(value, shift);
}
fn mi_mul_u32(value: ModularI64, multiplier: u32) -> ModularI64 {
    let a0 = value.x & 0xffffu;
    let a1 = value.x >> 16u;
    let b0 = multiplier & 0xffffu;
    let b1 = multiplier >> 16u;
    let p0 = a0 * b0;
    let p1 = a0 * b1;
    let p2 = a1 * b0;
    let low1 = p0 + (p1 << 16u);
    let low2 = low1 + (p2 << 16u);
    let high = value.y * multiplier + a1 * b1 + (p1 >> 16u) + (p2 >> 16u)
        + u32(low1 < p0) + u32(low2 < low1);
    return vec2<u32>(low2, high);
}
fn mi_average(a: i32, b: i32) -> i32 {
    return bitcast<i32>(mi_div_pow2(mi_add(mi_from_i32(a), mi_from_i32(b)), 1u).x);
}
fn mi_gradient(north: i32, west: i32, northwest: i32) -> i32 {
    let low = min(north, west);
    let high = max(north, west);
    if northwest < low { return high; }
    if northwest > high { return low; }
    // With northwest inside the interval, the mathematical result is representable in i32.
    return bitcast<i32>(bitcast<u32>(north) + bitcast<u32>(west) - bitcast<u32>(northwest));
}
