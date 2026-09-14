struct GamutMappingParams {
    luminance: vec4<f32>, // target-primary Y weights, saturation preference (-1 disables)
};

fn gamut_map_rgb(color: vec3<f32>, gamut: GamutMappingParams) -> vec3<f32> {
    let preference = gamut.luminance.w;
    // Float comparisons may flush subnormals. A negative nonzero word must not pass the
    // in-gamut shortcut, even when the adapter compares that value equal to zero.
    let words = bitcast<vec3<u32>>(color);
    let negative = ((words & vec3<u32>(0x80000000u)) != vec3<u32>(0u))
        & ((words & vec3<u32>(0x7fffffffu)) != vec3<u32>(0u));
    if preference < 0.0 || (!any(negative) && all(color <= vec3<f32>(1.0))) {
        return color;
    }
    // Scaling before subtraction keeps finite, extended RGB away from F32 overflow.
    let scale = max(1.0, max(max(abs(color.x), abs(color.y)), abs(color.z)));
    let rgb = color / scale;
    let white = 1.0 / scale;
    let gray = dot(rgb, gamut.luminance.xyz);
    // Outside the cube, nonpositive luminance maps to black.
    if gray <= 0.0 { return vec3<f32>(0.0); }
    var saturation_mix = 0.0;
    var luminance_mix = 0.0;
    for (var c = 0u; c < 3u; c += 1u) {
        let distance = rgb[c] - gray;
        if rgb[c] < 0.0 {
            saturation_mix = max(saturation_mix, rgb[c] / distance);
        }
        if preference < 1.0 && rgb[c] > white && distance > 0.0 {
            luminance_mix = max(luminance_mix, (rgb[c] - white) / distance);
        }
    }
    luminance_mix = max(luminance_mix, saturation_mix);
    var gray_mix = saturation_mix;
    if preference < 1.0 && luminance_mix > saturation_mix {
        gray_mix = clamp(preference * saturation_mix + (1.0 - preference) * luminance_mix, 0.0, 1.0);
    }
    var mixed = rgb * (1.0 - gray_mix) + vec3<f32>(gray * gray_mix);
    let minimum = min(min(rgb.x, rgb.y), rgb.z);
    if minimum < 0.0 && gray_mix == saturation_mix {
        // The active lower cube face is exactly zero. Express other components relative
        // to that face, avoiding cancellation residue amplified by PQ/HLG near black.
        mixed = (rgb - vec3<f32>(minimum)) * (1.0 - gray_mix);
    }
    let peak = max(white, max(max(mixed.x, mixed.y), mixed.z));
    let mapped = clamp(mixed / peak, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(mapped, vec3<f32>(0.0),
        (bitcast<vec3<u32>>(mapped) & vec3<u32>(0x80000000u)) != vec3<u32>(0u));
}
