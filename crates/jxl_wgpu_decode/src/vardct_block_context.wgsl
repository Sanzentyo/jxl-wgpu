struct HfBlockContextTables {
    block_context_map_offset_words: u32,
    qf_threshold_offset_words: u32,
    qf_threshold_count: u32,
    lf0_threshold_offset_words: u32,
    lf0_threshold_count: u32,
    lf1_threshold_offset_words: u32,
    lf1_threshold_count: u32,
    lf2_threshold_offset_words: u32,
    lf2_threshold_count: u32,
    _reserved0: u32,
    _reserved1: u32,
    _reserved2: u32,
};

fn hf_signed_threshold_segment(value: i32, offset: u32, count: u32) -> u32 {
    var segment = 0u;
    for (var index = 0u; index < count; index += 1u) {
        if value > bitcast<i32>(modular_metadata[offset + index]) {
            segment += 1u;
        }
    }
    return segment;
}

fn hf_quant_threshold_segment(value: u32, offset: u32, count: u32) -> u32 {
    var segment = 0u;
    for (var index = 0u; index < count; index += 1u) {
        if value > modular_metadata[offset + index] {
            segment += 1u;
        }
    }
    return segment;
}

fn hf_block_context(
    tables: HfBlockContextTables,
    order_channel: u32,
    order_id: u32,
    qf: u32,
    lf: vec3<i32>,
) -> u32 {
    // JPEG XL folds LF threshold segments in X, B, Y order.
    var lf_index = hf_signed_threshold_segment(
        lf.x,
        tables.lf0_threshold_offset_words,
        tables.lf0_threshold_count,
    );
    lf_index = lf_index * (tables.lf2_threshold_count + 1u)
        + hf_signed_threshold_segment(
            lf.z,
            tables.lf2_threshold_offset_words,
            tables.lf2_threshold_count,
        );
    lf_index = lf_index * (tables.lf1_threshold_count + 1u)
        + hf_signed_threshold_segment(
            lf.y,
            tables.lf1_threshold_offset_words,
            tables.lf1_threshold_count,
        );
    let lf_context_count = (tables.lf0_threshold_count + 1u)
        * (tables.lf1_threshold_count + 1u)
        * (tables.lf2_threshold_count + 1u);
    let qf_index = hf_quant_threshold_segment(
        qf,
        tables.qf_threshold_offset_words,
        tables.qf_threshold_count,
    );
    let map_index = (((order_channel * 13u + order_id)
        * (tables.qf_threshold_count + 1u) + qf_index)
        * lf_context_count) + lf_index;
    return modular_metadata[tables.block_context_map_offset_words + map_index];
}
