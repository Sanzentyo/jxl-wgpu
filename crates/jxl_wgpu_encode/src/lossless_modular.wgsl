struct Params {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
    output_word_offset: u32,
    channel: u32,
    channels: u32,
    bytes_per_sample: u32,
    sample_mask: u32,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

// Word 0 is the event count, words 1..20 are raw-token counts, words
// 20..53 are LZ77-token counts, and the remaining words are four-word events
// (kind, token, extra-bit count, extra bits).
@group(0) @binding(1)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(2)
var<storage, read> group_params: array<Params>;

const OUTPUT_HEADER_WORDS: u32 = 53u;
const EVENT_WORDS: u32 = 4u;
const EVENT_OVERFLOW: u32 = 0xffffffffu;

fn source_byte(byte_index: u32) -> u32 {
    let word = source_words[byte_index >> 2u];
    let shift = (byte_index & 3u) * 8u;
    return (word >> shift) & 255u;
}

fn source_component(params: Params, x: u32, y: u32, component: u32) -> i32 {
    let sample_index = x * params.channels + component;
    let byte_index = params.byte_offset + y * params.row_stride
        + sample_index * params.bytes_per_sample;
    var value = source_byte(byte_index);
    if params.bytes_per_sample == 2u {
        value |= source_byte(byte_index + 1u) << 8u;
    }
    return i32(value & params.sample_mask);
}

// JPEG XL's reversible color transform type 0 maps RGB to YCoCg. Computing it
// in the token kernel avoids both an intermediate image and a CPU color path.
fn sample_at(params: Params, x: u32, y: u32) -> i32 {
    if params.channels == 1u {
        return source_component(params, x, y, 0u);
    }
    if params.channel == 3u {
        return source_component(params, x, y, 3u);
    }
    let red = source_component(params, x, y, 0u);
    let green = source_component(params, x, y, 1u);
    let blue = source_component(params, x, y, 2u);
    let co = red - blue;
    let temporary = blue + (co >> 1u);
    let cg = green - temporary;
    let luma = temporary + (cg >> 1u);
    if params.channel == 0u {
        return luma;
    }
    if params.channel == 1u {
        return co;
    }
    return cg;
}

fn append_event(params: Params, kind: u32, token: u32, nbits: u32, bits: u32) {
    let output_base = params.output_word_offset;
    let event = output_words[output_base];
    let pixel_count = params.width * params.height;
    let capacity = pixel_count + (pixel_count + 7u) / 8u + 1u;
    if event >= capacity {
        // Host validation treats this sentinel as a bounded backend failure.
        output_words[output_base] = EVENT_OVERFLOW;
        return;
    }
    let base = output_base + OUTPUT_HEADER_WORDS + event * EVENT_WORDS;
    output_words[base] = kind;
    output_words[base + 1u] = token;
    output_words[base + 2u] = nbits;
    output_words[base + 3u] = bits;
    output_words[output_base] = event + 1u;
}

fn emit_raw(params: Params, value: u32) {
    var token = 0u;
    var nbits = 0u;
    var bits = 0u;
    if value != 0u {
        let n = 31u - countLeadingZeros(value | 1u);
        token = n + 1u;
        nbits = n;
        bits = value - (1u << n);
    }
    let count_index = params.output_word_offset + 1u + token;
    output_words[count_index] = output_words[count_index] + 1u;
    append_event(params, 0u, token, nbits, bits);
}

fn emit_run(params: Params, count: u32) {
    if count == 0u {
        return;
    }
    // The prefix stream's raw symbol zero is the LZ77 escape. JPEG XL's
    // configured minimum length is seven, hence the encoded value is count-8.
    let output_base = params.output_word_offset;
    output_words[output_base + 1u] = output_words[output_base + 1u] + 1u;
    let value = count - 8u;
    var token = value;
    var nbits = 0u;
    var bits = 0u;
    if value >= 16u {
        let n = 31u - countLeadingZeros(value | 1u);
        token = 16u + n - 4u;
        nbits = n;
        bits = value - (1u << n);
    }
    output_words[output_base + 20u + token] = output_words[output_base + 20u + token] + 1u;
    append_event(params, 1u, token, nbits, bits);
}

fn packed_residual(params: Params, x: u32, y: u32) -> u32 {
    let pixel = sample_at(params, x, y);
    var left = 0i;
    var top = 0i;
    var top_left = 0i;
    if y == 0u {
        if x != 0u {
            left = sample_at(params, x - 1u, y);
        }
        top = left;
        top_left = left;
    } else {
        top = sample_at(params, x, y - 1u);
        if x == 0u {
            left = top;
            top_left = top;
        } else {
            left = sample_at(params, x - 1u, y);
            top_left = sample_at(params, x - 1u, y - 1u);
        }
    }

    let ac = left - top_left;
    let ab = left - top;
    let bc = top - top_left;
    let gradient = ac + top;
    let clamped = select(left, top, (ab ^ bc) < 0i);
    let prediction = select(clamped, gradient, (ac ^ bc) < 0i);
    let residual = pixel - prediction;
    if residual < 0i {
        return u32(-residual * 2i - 1i);
    }
    return u32(residual * 2i);
}

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.y != 0u || global_id.z != 0u || global_id.x >= arrayLength(&group_params) {
        return;
    }
    let params = group_params[global_id.x];

    var run = 0u;
    for (var y = 0u; y < params.height; y += 1u) {
        for (var chunk_x = 0u; chunk_x < params.width; chunk_x += 8u) {
            let count = min(8u, params.width - chunk_x);
            var residuals: array<u32, 8>;
            var prefix = 0u;
            var prefix_open = true;
            for (var index = 0u; index < count; index += 1u) {
                let residual = packed_residual(params, chunk_x + index, y);
                residuals[index] = residual;
                if prefix_open && residual == 0u {
                    prefix += 1u;
                } else {
                    prefix_open = false;
                }
            }

            if prefix == count && (run > 0u || prefix > 7u) {
                run += prefix;
            } else if prefix + run > 7u {
                emit_run(params, run + prefix);
                for (var index = prefix; index < count; index += 1u) {
                    emit_raw(params, residuals[index]);
                }
                run = 0u;
            } else {
                for (var index = 0u; index < count; index += 1u) {
                    emit_raw(params, residuals[index]);
                }
            }
        }
    }
    emit_run(params, run);
}
