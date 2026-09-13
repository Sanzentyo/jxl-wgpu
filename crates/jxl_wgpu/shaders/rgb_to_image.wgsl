fn source_rgb_words_at(x: u32, y: u32) -> vec3<u32> {
    let coordinate = source_coordinate(vec2<u32>(min(x, params.width - 1u), min(y, params.height - 1u)));
    return vec3<u32>(
        source_r[coordinate.y * params.r_stride + coordinate.x],
        source_g[coordinate.y * params.g_stride + coordinate.x],
        source_b[coordinate.y * params.b_stride + coordinate.x],
    );
}

fn source_alpha_word_at(x: u32, y: u32) -> u32 {
    return 0x3f800000u;
}
