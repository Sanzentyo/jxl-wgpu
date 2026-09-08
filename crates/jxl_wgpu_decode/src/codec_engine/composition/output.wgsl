override surface_plane_words: u32;
override surface_alpha_channel: u32;
fn surface_value(word: u32) -> f32 { return bitcast<f32>(source_r[word]); }
fn surface_position(x: u32, y: u32) -> u32 {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return p.y * params.source_width + p.x;
}
fn source_rgb_at(x: u32, y: u32) -> vec3<f32> {
    let position = surface_position(x, y);
    return present_rgb(vec3<f32>(surface_value(position), surface_value(surface_plane_words + position),
        surface_value(2u * surface_plane_words + position)), position);
}
fn source_alpha_at(x: u32, y: u32) -> f32 {
    if surface_alpha_channel == 0xffffffffu { return 1.0; }
    return bitcast<f32>(source_r[surface_alpha_channel * surface_plane_words + surface_position(x, y)]);
}
