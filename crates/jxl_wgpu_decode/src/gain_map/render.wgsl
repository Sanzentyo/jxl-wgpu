struct GainApplication {
    weight: f32,
    base_to_reference: f32,
    reference_to_base: f32,
    padding: u32,
};

struct GainParams {
    base_offsets: vec4<u32>,
    base_strides: vec4<u32>,
    map_planes: array<vec4<u32>, 3>,
    minimum: vec4<f32>,
    maximum: vec4<f32>,
    inverse_gamma: vec4<f32>,
    base_offset: vec4<f32>,
    alternate_offset: vec4<f32>,
    application: GainApplication,
};

@group(0) @binding(5) var<uniform> gain_params: GainParams;
@group(0) @binding(6) var<storage, read> gain_samples: array<f32>;

fn gain_sample(channel: u32, coordinate: vec2<u32>) -> f32 {
    let plane = gain_params.map_planes[channel];
    if all(plane.zw == vec2<u32>(params.source_width, params.source_height)) {
        return clamp(gain_samples[plane.x + coordinate.y * plane.y + coordinate.x], 0.0, 1.0);
    }
    let position = clamp((vec2<f32>(coordinate) + 0.5) * vec2<f32>(plane.zw)
        / vec2<f32>(f32(params.source_width), f32(params.source_height)) - 0.5,
        vec2<f32>(0.0), vec2<f32>(plane.zw - vec2<u32>(1u)));
    let p0 = vec2<u32>(floor(position));
    let p1 = min(p0 + vec2<u32>(1u), plane.zw - vec2<u32>(1u));
    let fraction = position - vec2<f32>(p0);
    let top = mix(gain_samples[plane.x + p0.y * plane.y + p0.x],
        gain_samples[plane.x + p0.y * plane.y + p1.x], fraction.x);
    let bottom = mix(gain_samples[plane.x + p1.y * plane.y + p0.x],
        gain_samples[plane.x + p1.y * plane.y + p1.x], fraction.x);
    return clamp(mix(top, bottom, fraction.y), 0.0, 1.0);
}

fn source_rgb_words_at(x: u32, y: u32) -> vec3<u32> {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    let base = vec3<f32>(
        bitcast<f32>(source_r[gain_params.base_offsets.x + p.y * gain_params.base_strides.x + p.x]),
        bitcast<f32>(source_g[gain_params.base_offsets.y + p.y * gain_params.base_strides.y + p.x]),
        bitcast<f32>(source_b[gain_params.base_offsets.z + p.y * gain_params.base_strides.z + p.x]));
    var mapped: vec3<f32>;
    for (var c = 0u; c < 3u; c += 1u) {
        let value = pow(gain_sample(c, p), gain_params.inverse_gamma[c]);
        let log_gain = mix(gain_params.minimum[c], gain_params.maximum[c], value);
        let application = gain_params.application;
        let reference = base[c] * application.base_to_reference;
        mapped[c] = ((reference + gain_params.base_offset[c]) * exp2(log_gain * application.weight)
            - gain_params.alternate_offset[c]) * application.reference_to_base;
    }
    return bitcast<vec3<u32>>(mapped);
}

fn source_alpha_word_at(x: u32, y: u32) -> u32 {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return source_r[gain_params.base_offsets.w + p.y * gain_params.base_strides.w + p.x];
}
