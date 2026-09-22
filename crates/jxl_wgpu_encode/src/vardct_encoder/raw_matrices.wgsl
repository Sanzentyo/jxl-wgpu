// One invocation owns each word-aligned matrix fragment. No image samples leave GPU.
@group(0) @binding(0) var<storage, read> input_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words: array<u32>;

fn append_bits(base: u32, capacity: u32, value: u32, count: u32, cursor: u32) -> u32 {
    if count == 0u { return cursor; }
    if cursor + count > capacity * 32u { return capacity * 32u + 1u; }
    let word = cursor >> 5u;
    let shift = cursor & 31u;
    output_words[base + word] |= value << shift;
    if shift + count > 32u { output_words[base + word + 1u] |= value >> (32u - shift); }
    return cursor + count;
}

@compute @workgroup_size(1)
fn encode(@builtin(workgroup_id) id: vec3<u32>) {
    let matrix = id.x;
    if matrix >= input_words[0] { return; }
    let task = 67u + 5u * matrix;
    let width = input_words[task];
    let area = input_words[task + 1u];
    let source = input_words[task + 2u];
    let destination = input_words[task + 3u];
    let capacity = input_words[task + 4u];
    var cursor = 0u;
    for (var c = 0u; c < 3u; c += 1u) {
        let plane = source + c * area;
        for (var i = 0u; i < area; i += 1u) {
            let x = i % width;
            let y = i / width;
            var left = 0i;
            if x != 0u { left = bitcast<i32>(input_words[plane + i - 1u]); }
            else if y != 0u { left = bitcast<i32>(input_words[plane + i - width]); }
            var top = left;
            var top_left = left;
            if y != 0u { top = bitcast<i32>(input_words[plane + i - width]); }
            if x != 0u && y != 0u { top_left = bitcast<i32>(input_words[plane + i - width - 1u]); }
            let low = min(top, left);
            let high = max(top, left);
            var prediction = top + (left - top_left);
            if top_left >= high { prediction = low; }
            else if top_left <= low { prediction = high; }
            let residual = bitcast<i32>(input_words[plane + i]) - prediction;
            let value = (bitcast<u32>(residual) << 1u) ^ bitcast<u32>(residual >> 31u);
            var symbol = 0u;
            var extra_count = 0u;
            var extra = 0u;
            if value != 0u {
                extra_count = 31u - countLeadingZeros(value);
                symbol = extra_count + 1u;
                extra = value - (1u << extra_count);
            }
            cursor = append_bits(destination, capacity, input_words[1u + 2u * symbol], input_words[2u + 2u * symbol], cursor);
            cursor = append_bits(destination, capacity, extra, extra_count, cursor);
        }
    }
    output_words[4u * matrix] = 0x524d4154u;
    output_words[4u * matrix + 1u] = matrix;
    output_words[4u * matrix + 2u] = 3u * area;
    output_words[4u * matrix + 3u] = cursor;
}
