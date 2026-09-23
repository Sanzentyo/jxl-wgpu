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
