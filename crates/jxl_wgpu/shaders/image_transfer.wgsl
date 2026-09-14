fn transfer_to_linear(value: f32, transfer: u32, gamma: f32) -> f32 {
    // BT.709 extends its linear toe below zero, as in libjxl and jxl-oxide.
    // Test the signed input before pow so negative bases are never evaluated.
    if transfer == 2u {
        if value <= 0.081 { return value / 4.5; }
        return pow((value + 0.099) / 1.099, 1.0 / 0.45);
    }
    let magnitude = abs(value);
    var linear = magnitude;
    if transfer == 6u {
        return pow(max(value, 0.0), 1.0 / gamma);
    }
    if transfer == 7u {
        if value <= 0.0 { return value; }
        return pow(value, 2.6);
    }
    if transfer == 1u {
        linear = select(pow((magnitude + 0.055) / 1.055, 2.4), magnitude / 12.92, magnitude <= 0.04045);
    } else if transfer == 3u {
        let m1 = 2610.0 / 16384.0;
        let m2 = (2523.0 / 4096.0) * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = (2413.0 / 4096.0) * 32.0;
        let c3 = (2392.0 / 4096.0) * 32.0;
        let powered = pow(magnitude, 1.0 / m2);
        let numerator = max(powered - c1, 0.0);
        linear = pow(numerator / max(c2 - c3 * powered, 1e-10), 1.0 / m1);
    } else if transfer == 4u {
        let hlg_a = 0.17883277;
        let hlg_b = 1.0 - 4.0 * hlg_a;
        let hlg_c = 0.5599107295;
        linear = select(
            (exp((magnitude - hlg_c) / hlg_a) + hlg_b) / 12.0,
            magnitude * magnitude / 3.0,
            magnitude <= 0.5,
        );
    } else if transfer == 5u {
        let alpha = 1.09929682680944;
        let beta = 0.018053968510807;
        linear = select(pow((magnitude + alpha - 1.0) / alpha, 1.0 / 0.45), magnitude / 4.5,
            magnitude < 4.5 * beta);
    }
    return select(linear, -linear, value < 0.0);
}

fn transfer_from_linear(value: f32, transfer: u32, gamma: f32) -> f32 {
    if transfer == 2u {
        if value <= 0.018 { return 4.5 * value; }
        return 1.099 * pow(value, 0.45) - 0.099;
    }
    let magnitude = abs(value);
    var encoded = magnitude;
    if transfer == 6u {
        return pow(max(value, 0.0), gamma);
    }
    if transfer == 7u {
        if value <= 0.0 { return value; }
        return pow(value, 1.0 / 2.6);
    }
    if transfer == 1u {
        encoded = select(1.055 * pow(magnitude, 1.0 / 2.4) - 0.055, 12.92 * magnitude, magnitude <= 0.0031308);
    } else if transfer == 3u {
        let m1 = 2610.0 / 16384.0;
        let m2 = (2523.0 / 4096.0) * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = (2413.0 / 4096.0) * 32.0;
        let c3 = (2392.0 / 4096.0) * 32.0;
        let powered = pow(magnitude, m1);
        encoded = pow((c1 + c2 * powered) / (1.0 + c3 * powered), m2);
    } else if transfer == 4u {
        let hlg_a = 0.17883277;
        let hlg_b = 1.0 - 4.0 * hlg_a;
        let hlg_c = 0.5599107295;
        encoded = select(
            hlg_a * log(12.0 * magnitude - hlg_b) + hlg_c,
            sqrt(3.0 * magnitude),
            magnitude <= 1.0 / 12.0,
        );
    } else if transfer == 5u {
        let alpha = 1.09929682680944;
        let beta = 0.018053968510807;
        encoded = select(alpha * pow(magnitude, 0.45) - (alpha - 1.0), 4.5 * magnitude, magnitude < beta);
    }
    return select(encoded, -encoded, value < 0.0);
}
// Display-relative RGB uses unit white at the declared intensity target. Generic
// callers pass zero nits to retain absolute-normalized PQ / scene-linear HLG.
fn display_ootf(rgb: vec3<f32>, luminance: vec4<f32>) -> vec3<f32> {
    let exponent = luminance.w;
    if abs(exponent) <= 0.01 { return rgb; }
    let y = dot(rgb, luminance.xyz);
    var ratio: f32;
    if y > 0.0 {
        ratio = min(pow(y, exponent), 1e9);
    } else {
        // XYB ringing can produce nonpositive luminance. Preserve libjxl's
        // reconstruction extension (base/fast_math-inl.h::FastLog2f) with
        // explicit wrapping integer range reduction; never pow a negative base.
        let bits = bitcast<i32>(y);
        let shift = (bits - 0x3f2aaaab) >> 23u;
        let mantissa = bitcast<f32>(bits - (shift << 23u));
        ratio = min(exp2((log2(mantissa) + f32(shift)) * exponent), 1e9);
    }
    return rgb * ratio;
}

fn display_to_linear(
    rgb: vec3<f32>, transfer: u32, gamma: f32, intensity: f32, luminance: vec4<f32>,
) -> vec3<f32> {
    var linear = vec3<f32>(
        transfer_to_linear(rgb.r, transfer, gamma),
        transfer_to_linear(rgb.g, transfer, gamma),
        transfer_to_linear(rgb.b, transfer, gamma),
    );
    if intensity > 0.0 {
        if transfer == 3u { linear *= 10000.0 / intensity; }
        if transfer == 4u { linear = display_ootf(linear, luminance); }
    }
    return linear;
}

fn display_from_linear(
    rgb: vec3<f32>, transfer: u32, gamma: f32, intensity: f32, luminance: vec4<f32>,
) -> vec3<f32> {
    var linear = rgb;
    if intensity > 0.0 {
        if transfer == 3u { linear *= intensity / 10000.0; }
        if transfer == 4u { linear = display_ootf(linear, luminance); }
    }
    return vec3<f32>(
        transfer_from_linear(linear.r, transfer, gamma),
        transfer_from_linear(linear.g, transfer, gamma),
        transfer_from_linear(linear.b, transfer, gamma),
    );
}
