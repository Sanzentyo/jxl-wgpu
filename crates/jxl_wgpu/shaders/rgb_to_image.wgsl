fn source_rgb_at(x: u32, y: u32) -> vec3<f32> {
    let coordinate = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return vec3<f32>(
        bitcast<f32>(source_r[coordinate.y * params.r_stride + coordinate.x]),
        bitcast<f32>(source_g[coordinate.y * params.g_stride + coordinate.x]),
        bitcast<f32>(source_b[coordinate.y * params.b_stride + coordinate.x]),
    );
}

fn source_alpha_at(x: u32, y: u32) -> f32 {
    return 1.0;
}
