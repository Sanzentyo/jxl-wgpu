// Scalar side plane: one row owns one bounded, word-aligned entropy fragment.
/*__JXL_SOURCE__*/
struct Prefix { bits: u32, bit_len: u32, }
struct Params {
    source: Source,
    width: u32, height: u32, groups_x: u32, group_dim: u32, row_words: u32,
    big_endian: u32, sample_mask: u32, channel_index: u32,
    prefix: array<Prefix, 33>,
}
@group(0) @binding(0) var<storage, read> source_words: array<u32>;
@group(0) @binding(12) var<storage, read> source_words_1: array<u32>;
@group(0) @binding(13) var<storage, read> source_words_2: array<u32>;
@group(0) @binding(14) var<storage, read> source_words_3: array<u32>;
@group(0) @binding(1) var<storage, read> params: Params;
@group(0) @binding(2) var<storage, read_write> output_words: array<u32>;

fn sample(x: u32, y: u32) -> i32 {
    return bitcast<i32>(load_source_component(params.source, x, y, params.big_endian, params.sample_mask));
}

fn append_bits(base: u32, value: u32, count: u32, cursor: u32) -> u32 {
    if count == 0u { return cursor; }
    let capacity = (params.row_words - 4u) * 32u;
    if cursor + count > capacity { return capacity + 1u; }
    let word = cursor >> 5u;
    let shift = cursor & 31u;
    output_words[base + word] |= value << shift;
    if shift + count > 32u { output_words[base + word + 1u] |= value >> (32u - shift); }
    return cursor + count;
}

@compute @workgroup_size(1)
fn encode(@builtin(workgroup_id) id: vec3<u32>) {
    if id.x >= params.groups_x || id.y >= params.height { return; }
    let row = id.x * params.height + id.y;
    let base = row * params.row_words;
    let x0 = id.x * params.group_dim;
    let width = min(params.group_dim, params.width - x0);
    let y = id.y;
    let first_row = (y % params.group_dim) == 0u;
    var cursor = 0u;
    for (var x = 0u; x < width; x += 1u) {
        var left = 0i;
        if x != 0u { left = sample(x0 + x - 1u, y); }
        else if !first_row { left = sample(x0 + x, y - 1u); }
        var top = left;
        var top_left = left;
        if !first_row { top = sample(x0 + x, y - 1u); }
        if x != 0u && !first_row { top_left = sample(x0 + x - 1u, y - 1u); }
        let low = min(top, left);
        let high = max(top, left);
        var prediction = top + (left - top_left);
        if top_left >= high { prediction = low; }
        else if top_left <= low { prediction = high; }
        let residual = bitcast<i32>(bitcast<u32>(sample(x0 + x, y)) - bitcast<u32>(prediction));
        let value = (bitcast<u32>(residual) << 1u) ^ bitcast<u32>(residual >> 31u);
        var symbol = 0u;
        var extra_count = 0u;
        var extra = 0u;
        if value != 0u {
            extra_count = 31u - countLeadingZeros(value);
            symbol = extra_count + 1u;
            extra = value - (1u << extra_count);
        }
        cursor = append_bits(base + 4u, params.prefix[symbol].bits, params.prefix[symbol].bit_len, cursor);
        cursor = append_bits(base + 4u, extra, extra_count, cursor);
    }
    if cursor > (params.row_words - 4u) * 32u { return; }
    output_words[base] = 0x4d504c4eu ^ params.channel_index;
    output_words[base + 1u] = row;
    output_words[base + 2u] = width;
    output_words[base + 3u] = cursor;
}
