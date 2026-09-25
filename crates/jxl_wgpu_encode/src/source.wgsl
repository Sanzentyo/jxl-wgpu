// Checked component records and bit-exact byte/word loading shared by both encoders.
struct Source {
    row_stride: u32,
    byte_offset: u32,
    pixel_stride: u32,
    word_bytes: u32,
    bit_shift: u32,
    plane: u32,
}

fn source_byte(plane: u32, byte_index: u32) -> u32 {
    var word: u32;
    switch plane {
        case 0u: { word = source_words[byte_index >> 2u]; }
        case 1u: { word = source_words_1[byte_index >> 2u]; }
        case 2u: { word = source_words_2[byte_index >> 2u]; }
        default: { word = source_words_3[byte_index >> 2u]; }
    }
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 255u;
}

fn load_source_component(source: Source, x: u32, y: u32, big_endian: u32, sample_mask: u32) -> u32 {
    let byte_index = source.byte_offset + y * source.row_stride + x * source.pixel_stride;
    var value = 0u;
    for (var byte = 0u; byte < source.word_bytes; byte += 1u) {
        let shift = select(byte, source.word_bytes - 1u - byte, big_endian != 0u) * 8u;
        value |= source_byte(source.plane, byte_index + byte) << shift;
    }
    return (value >> source.bit_shift) & sample_mask;
}
