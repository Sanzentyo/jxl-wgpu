// JPEG XL output association is adjusted after color conversion, before rounding. The finite
// 2^-26 floor preserves invisible colors and avoids division by zero, matching libjxl output.
fn image_alpha_multiplier(alpha: f32, conversion: u32) -> f32 {
    if conversion == 1u { return 1.0 / max(1.0 / 67108864.0, alpha); }
    if conversion == 2u { return max(1.0 / 67108864.0, alpha); }
    return 1.0;
}
