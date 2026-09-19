override surface_plane_words: u32;
override surface_alpha_channel: u32;
override surface_color_channels: u32 = 3u;
fn surface_value(word: u32) -> f32 { return bitcast<f32>(source_r[word]); }
fn surface_position(x: u32, y: u32) -> u32 {
    let p = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return p.y * params.source_width + p.x;
}
fn source_rgb_words_at(x: u32, y: u32) -> vec3<u32> {
    let position = surface_position(x, y);
    if surface_color_channels == 1u { return vec3<u32>(source_r[position]); }
    return present_rgb_words(vec3<u32>(source_r[position], source_r[surface_plane_words + position],
        source_r[2u * surface_plane_words + position]), position);
}
fn source_alpha_word_at(x: u32, y: u32) -> u32 {
    if surface_alpha_channel == 0xffffffffu { return 0x3f800000u; }
    return source_r[surface_alpha_channel * surface_plane_words + surface_position(x, y)];
}
