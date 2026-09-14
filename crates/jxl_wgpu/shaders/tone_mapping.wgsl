struct ToneMappingParams {
    range: vec4<f32>, // source/target white nits, exclusive protected unit-luminance bound, mode
    curve: vec4<f32>, // source PQ minimum/span, normalized target minimum/maximum
    knee: vec4<f32>, // knee start, inverse shoulder span, white ratio, inverse source PQ span
};

fn tone_map_light(
    color: vec3<f32>, luminance: vec3<f32>, neutral: vec3<f32>, tone: ToneMappingParams,
) -> vec3<f32> {
    let mode = u32(tone.range.w);
    if mode == 0u { return color; }
    let relative_y = dot(color, luminance);
    let nits = tone.range.x * relative_y;
    if mode == 1u || mode == 5u || (tone.range.z > 0.0 && relative_y < tone.range.z) {
        return color * tone.knee.z;
    }
    if mode == 3u { return neutral; }
    if mode == 4u {
        if relative_y <= 1e-6 { return neutral; }
        return color / relative_y;
    }
    let encoded = select(transfer_from_linear(nits / 10000.0, 3u, 1.0), 0.0, nits == 0.0);
    let normalized = min(1.0, (encoded - tone.curve.x) * tone.knee.w);
    var shoulder = normalized;
    if normalized >= tone.knee.x {
        let t = (normalized - tone.knee.x) * tone.knee.y;
        let squared = t * t;
        let cubed = squared * t;
        // Limit the start slope to three times the secant. A protected linear interval
        // can leave too little highlight headroom for the unconstrained unit slope.
        let slope = clamp(3.0 * (tone.curve.w - tone.knee.x) * tone.knee.y, 0.0, 1.0);
        shoulder = (2.0 * cubed - 3.0 * squared + 1.0) * tone.knee.x
            + (cubed - 2.0 * squared + t) * (1.0 - tone.knee.x) * slope
            + (-2.0 * cubed + 3.0 * squared) * tone.curve.w;
    }
    let shadow = 1.0 - shoulder;
    let shadow_squared = shadow * shadow;
    let mapped = shoulder + tone.curve.z * shadow_squared * shadow_squared;
    // Clip in monotonic PQ space before its inverse; extreme out-of-range colors must not
    // evaluate ST 2084 beyond its asymptote merely to be clipped afterwards.
    let output_pq = clamp(mapped * tone.curve.y + tone.curve.x,
        0.0, tone.curve.w * tone.curve.y + tone.curve.x);
    let output_nits = clamp(transfer_to_linear(output_pq, 3u, 1.0) * 10000.0, 0.0, tone.range.y);
    if nits <= 1e-6 { return neutral * (output_nits / tone.range.y); }
    return color * ((output_nits / max(nits, 1e-6)) * tone.knee.z);
}
