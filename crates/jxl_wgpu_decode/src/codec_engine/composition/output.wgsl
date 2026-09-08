override surface_plane_words: u32;
override surface_alpha_channel: u32;
fn surface_position(x: u32, y: u32) -> u32 {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return p.y * params.source_width + p.x;
}
fn source_rgb_at(x: u32, y: u32) -> vec3<f32> {
    let position = surface_position(x, y);
    return vec3<f32>(bitcast<f32>(source_r[position]), bitcast<f32>(source_r[surface_plane_words + position]),
        bitcast<f32>(source_r[2u * surface_plane_words + position]));
}
fn source_alpha_at(x: u32, y: u32) -> f32 {
    if surface_alpha_channel == 0xffffffffu { return 1.0; }
    return bitcast<f32>(source_r[surface_alpha_channel * surface_plane_words + surface_position(x, y)]);
}
