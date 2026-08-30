struct Params {
    width: u32,
    height: u32,
    row_stride: u32,
    byte_offset: u32,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

// Word 0 is the event count, words 1..20 are raw-token counts, words
// 20..53 are LZ77-token counts, and the remaining words are four-word events
// (kind, token, extra-bit count, extra bits).
@group(0) @binding(1)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(2)
var<uniform> params: Params;

const OUTPUT_HEADER_WORDS: u32 = 53u;
const EVENT_WORDS: u32 = 4u;
const EVENT_OVERFLOW: u32 = 0xffffffffu;

fn sample_at(x: u32, y: u32) -> i32 {
    let byte_index = params.byte_offset + y * params.row_stride + x;
    let word = source_words[byte_index >> 2u];
    let shift = (byte_index & 3u) * 8u;
    return i32((word >> shift) & 255u);
}

fn append_event(kind: u32, token: u32, nbits: u32, bits: u32) {
    let event = output_words[0];
    let word_count = arrayLength(&output_words);
    let capacity = (word_count - OUTPUT_HEADER_WORDS) / EVENT_WORDS;
    if event >= capacity {
        // Host validation treats this sentinel as a bounded backend failure.
        output_words[0] = EVENT_OVERFLOW;
        return;
    }
    let base = OUTPUT_HEADER_WORDS + event * EVENT_WORDS;
    output_words[base] = kind;
    output_words[base + 1u] = token;
    output_words[base + 2u] = nbits;
    output_words[base + 3u] = bits;
    output_words[0] = event + 1u;
}

fn emit_raw(value: u32) {
    var token = 0u;
    var nbits = 0u;
    var bits = 0u;
    if value != 0u {
        let n = 31u - countLeadingZeros(value | 1u);
        token = n + 1u;
        nbits = n;
        bits = value - (1u << n);
    }
    output_words[1u + token] = output_words[1u + token] + 1u;
    append_event(0u, token, nbits, bits);
}

fn emit_run(count: u32) {
    if count == 0u {
        return;
    }
    // The prefix stream's raw symbol zero is the LZ77 escape. JPEG XL's
    // configured minimum length is seven, hence the encoded value is count-8.
    output_words[1] = output_words[1] + 1u;
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
    output_words[20u + token] = output_words[20u + token] + 1u;
    append_event(1u, token, nbits, bits);
}

fn packed_residual(x: u32, y: u32) -> u32 {
    let pixel = sample_at(x, y);
    var left = 0i;
    var top = 0i;
    var top_left = 0i;
    if y == 0u {
        if x != 0u {
            left = sample_at(x - 1u, y);
        }
        top = left;
        top_left = left;
    } else {
        top = sample_at(x, y - 1u);
        if x == 0u {
            left = top;
            top_left = top;
        } else {
            left = sample_at(x - 1u, y);
            top_left = sample_at(x - 1u, y - 1u);
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
    if any(global_id != vec3<u32>(0u)) {
        return;
    }

    var run = 0u;
    for (var y = 0u; y < params.height; y += 1u) {
        for (var chunk_x = 0u; chunk_x < params.width; chunk_x += 8u) {
            let count = min(8u, params.width - chunk_x);
            var residuals: array<u32, 8>;
            var prefix = 0u;
            var prefix_open = true;
            for (var index = 0u; index < count; index += 1u) {
                let residual = packed_residual(chunk_x + index, y);
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
                emit_run(run + prefix);
                for (var index = prefix; index < count; index += 1u) {
                    emit_raw(residuals[index]);
                }
                run = 0u;
            } else {
                for (var index = 0u; index < count; index += 1u) {
                    emit_raw(residuals[index]);
                }
            }
        }
    }
    emit_run(run);
}
