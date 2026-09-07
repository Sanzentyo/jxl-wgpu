fn source_rgba_at(x: u32, y: u32) -> vec4<f32> {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    let offset = (p.y * params.source_width + p.x) * 4u;
    return vec4<f32>(bitcast<f32>(source_r[offset]), bitcast<f32>(source_r[offset + 1u]),
        bitcast<f32>(source_r[offset + 2u]), bitcast<f32>(source_r[offset + 3u]));
}
fn source_rgb_at(x: u32, y: u32) -> vec3<f32> { return source_rgba_at(x, y).rgb; }
fn source_alpha_at(x: u32, y: u32) -> f32 { return source_rgba_at(x, y).a; }
