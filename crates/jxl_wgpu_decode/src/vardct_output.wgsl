struct VarDctSourceParams {
    plane_geometry: array<vec4<u32>, 3>,
    matrix_r: vec4<f32>,
    matrix_g: vec4<f32>,
    matrix_b: vec4<f32>,
    bias_cbrt: vec4<f32>,
    scaled_bias: vec4<f32>,
    intensity_scale: f32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(5) var<uniform> source_params: VarDctSourceParams;

fn plane_value(channel: u32, x: u32, y: u32) -> f32 {
    let index = y * source_params.plane_geometry[channel].x + x;
    if (channel == 0u) { return bitcast<f32>(source_r[index]); }
    if (channel == 1u) { return bitcast<f32>(source_g[index]); }
    return bitcast<f32>(source_b[index]);
}

fn jpeg_sample(channel: u32, x: u32, y: u32) -> f32 {
    let geometry = source_params.plane_geometry[channel];
    let horizontal_shift = geometry.w & 1u;
    let vertical_shift = (geometry.w >> 1u) & 1u;
    var x0 = x;
    var x1 = x;
    var x_weight = 0.0;
    if (horizontal_shift != 0u) {
        let center = x >> 1u;
        if ((x & 1u) == 0u) {
            x0 = select(0u, center - 1u, center != 0u);
            x1 = center;
            x_weight = 0.75;
        } else {
            x0 = center;
            x1 = min(center + 1u, geometry.y - 1u);
            x_weight = 0.25;
        }
    }
    var y0 = y;
    var y1 = y;
    var y_weight = 0.0;
    if (vertical_shift != 0u) {
        let center = y >> 1u;
        if ((y & 1u) == 0u) {
            y0 = select(0u, center - 1u, center != 0u);
            y1 = center;
            y_weight = 0.75;
        } else {
            y0 = center;
            y1 = min(center + 1u, geometry.z - 1u);
            y_weight = 0.25;
        }
    }
    let top = mix(plane_value(channel, x0, y0), plane_value(channel, x1, y0), x_weight);
    let bottom = mix(plane_value(channel, x0, y1), plane_value(channel, x1, y1), x_weight);
    return mix(top, bottom, y_weight);
}

fn source_rgb_at(output_x: u32, output_y: u32) -> vec3<f32> {
    let coordinate = source_coordinate(vec2<u32>(min(output_x, params.width - 1u), min(output_y, params.height - 1u)));
    let column = coordinate.x;
    let row = coordinate.y;
    if source_params.mode == 1u {
        let cb = jpeg_sample(0u, column, row);
        let y = jpeg_sample(1u, column, row) + 128.0 / 255.0;
        let cr = jpeg_sample(2u, column, row);
        return vec3<f32>(
            y + 1.402 * cr,
            y - (0.114 * 1.772 / 0.587) * cb - (0.299 * 1.402 / 0.587) * cr,
            y + 1.772 * cb,
        );
    }
    let x = plane_value(0u, column, row);
    let y = plane_value(1u, column, row);
    let b = plane_value(2u, column, row);

    // This is deliberately identical to jxl_wgpu's XYB inverse contract:
    // reconstruct biased LMS, apply the sign-preserving cube, then the
    // codestream-selected inverse opsin matrix.
    let mixed = vec3<f32>(
        y + x - source_params.bias_cbrt.x,
        y - x - source_params.bias_cbrt.y,
        b - source_params.bias_cbrt.z,
    );
    let lms = mixed * mixed * (mixed * source_params.intensity_scale)
        + source_params.scaled_bias.xyz;
    let linear_rgb = vec3<f32>(
        dot(source_params.matrix_r.xyz, lms),
        dot(source_params.matrix_g.xyz, lms),
        dot(source_params.matrix_b.xyz, lms),
    );
    return linear_rgb;
}
