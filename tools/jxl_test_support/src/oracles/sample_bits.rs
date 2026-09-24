/// Derive binary16's exact binary32 representation from its integer significand/exponent.
/// Neither the encoder nor either decoder supplies these expected source words.
pub fn binary32(word: u32, bits: u8) -> u32 {
    if bits == 32 {
        return word;
    }
    let sign = (word & 0x8000) << 16;
    let exponent = (word >> 10) & 31;
    let fraction = word & 1023;
    match (exponent, fraction) {
        (0, 0) => sign,
        (0, fraction) => {
            let leading = 31 - fraction.leading_zeros();
            sign | ((103 + leading) << 23) | ((fraction << (23 - leading)) & 0x7f_ffff)
        }
        (31, _) => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    }
}

/// Independent exact conversion for JPEG XL's binary floating domain.
/// Finite values are evaluated as dyadic rationals in F64, not with the GPU decoder's
/// bit-normalization algorithm. Specials are assembled explicitly to retain every NaN payload.
pub fn custom_binary32(word: u32, bits: u8, exponent_bits: u8) -> u32 {
    let fraction_bits = bits - exponent_bits - 1;
    assert!((2..=8).contains(&exponent_bits) && (2..=23).contains(&fraction_bits));
    let sign = (word >> (bits - 1)) << 31;
    let fraction = word & ((1 << fraction_bits) - 1);
    let exponent_mask = (1 << exponent_bits) - 1;
    let exponent = (word >> fraction_bits) & exponent_mask;
    if exponent == exponent_mask {
        return sign | 0x7f80_0000 | (fraction << (23 - fraction_bits));
    }
    let bias = (1 << (exponent_bits - 1)) - 1;
    let (significand, power) = if exponent == 0 {
        (fraction, 1 - bias - i32::from(fraction_bits))
    } else {
        (
            (1 << fraction_bits) | fraction,
            exponent as i32 - bias - i32::from(fraction_bits),
        )
    };
    sign | ((f64::from(significand) * 2.0f64.powi(power)) as f32).to_bits()
}
