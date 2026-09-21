struct Source {
    row_stride: u32,
    byte_offset: u32,
    pixel_stride: u32,
    word_bytes: u32,
    bit_shift: u32,
    plane: u32,
}

struct Params {
    width: u32,
    height: u32,
    output_word_offset: u32,
    channel: u32,
    channels: u32,
    sample_mask: u32,
    rct_type: u32,
    big_endian: u32,
    sources: array<Source, 4>,
    predictor: u32,
    wp_scratch_word_offset: u32,
    wp_coefficients: array<u32, 7>,
    wp_max_weights: array<u32, 4>,
    _padding: array<u32, 19>,
}

@group(0) @binding(0)
var<storage, read> source_words: array<u32>;

@group(0) @binding(3)
var<storage, read> source_words_1: array<u32>;

@group(0) @binding(4)
var<storage, read> source_words_2: array<u32>;

@group(0) @binding(5)
var<storage, read> source_words_3: array<u32>;

// Word 0 is the event count, words 1..34 are raw-token counts, words
// 34..67 are LZ77-token counts, and the remaining words are four-word events
// (kind, token, extra-bit count, extra bits). Weighted row state follows the
// maximum event range and is private to each group/channel invocation.
@group(0) @binding(1)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(2)
var<storage, read> group_params: array<Params>;

const OUTPUT_HEADER_WORDS: u32 = 67u;
const EVENT_WORDS: u32 = 4u;
const EVENT_OVERFLOW: u32 = 0xffffffffu;

/*__JXL_MODULAR_PREDICT__*/

var<private> active_params: Params;

fn wp_current_width() -> u32 { return active_params.width; }
fn wp_coefficient(index: u32) -> u32 { return active_params.wp_coefficients[index]; }
fn wp_max_weight(index: u32) -> u32 { return active_params.wp_max_weights[index]; }
fn wp_true_error(index: u32) -> i32 {
    return bitcast<i32>(output_words[active_params.wp_scratch_word_offset + index]);
}
fn wp_subpred_error(index: u32, component: u32) -> u32 {
    return output_words[active_params.wp_scratch_word_offset + active_params.width + index * 4u + component];
}
fn wp_store_row(index: u32, true_error: i32, errors: array<u32, 4>) {
    output_words[active_params.wp_scratch_word_offset + index] = bitcast<u32>(true_error);
    for (var component = 0u; component < 4u; component += 1u) {
        output_words[active_params.wp_scratch_word_offset + active_params.width + index * 4u + component] = errors[component];
    }
}
fn predictor_error() { output_words[active_params.output_word_offset] = EVENT_OVERFLOW; }

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

fn source_component(params: Params, x: u32, y: u32, component: u32) -> i32 {
    let source = params.sources[component];
    let byte_index = source.byte_offset + y * source.row_stride + x * source.pixel_stride;
    var value = 0u;
    for (var byte = 0u; byte < source.word_bytes; byte += 1u) {
        let shift = select(byte, source.word_bytes - 1u - byte, params.big_endian != 0u) * 8u;
        value |= source_byte(source.plane, byte_index + byte) << shift;
    }
    return bitcast<i32>((value >> source.bit_shift) & params.sample_mask);
}

fn add_wrap(a: i32, b: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(a) + bitcast<u32>(b));
}

fn sub_wrap(a: i32, b: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(a) - bitcast<u32>(b));
}

// The forward RCT uses wrapping integer words even for raw IEEE input. Computing it
// in the token kernel avoids both an intermediate image and a CPU color path.
fn sample_at(params: Params, x: u32, y: u32) -> i32 {
    if params.rct_type == 42u || params.channel >= 3u {
        return source_component(params, x, y, params.channel);
    }
    let permutation = params.rct_type / 7u;
    let first = source_component(params, x, y, permutation % 3u);
    let second = source_component(params, x, y, (permutation + 1u + permutation / 3u) % 3u);
    let third = source_component(params, x, y, (permutation + 2u - permutation / 3u) % 3u);
    let operation = params.rct_type % 7u;
    if operation == 6u {
        let co = sub_wrap(first, third);
        let temporary = add_wrap(third, co >> 1u);
        let cg = sub_wrap(second, temporary);
        let luma = add_wrap(temporary, cg >> 1u);
        return vec3<i32>(luma, co, cg)[params.channel];
    }
    var transformed = vec3<i32>(first, second, third);
    if operation >= 4u {
        transformed.y = sub_wrap(second, add_wrap(first, third) >> 1u);
    } else if operation >= 2u {
        transformed.y = sub_wrap(second, first);
    }
    if (operation & 1u) != 0u {
        transformed.z = sub_wrap(third, first);
    }
    return transformed[params.channel];
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
    output_words[output_base + 34u + token] = output_words[output_base + 34u + token] + 1u;
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

    // Keep the default predictor's source reads independent of unused neighbours.
    if params.predictor == 5u {
        let prediction = gradient_i32(top, left, top_left);
        let residual = bitcast<u32>(pixel) - bitcast<u32>(prediction);
        return (residual << 1u) ^ (0u - (residual >> 31u));
    }
    var top_right = top;
    if y != 0u && x + 1u < params.width { top_right = sample_at(params, x + 1u, y - 1u); }
    var top_right_right = top_right;
    if y != 0u && x + 2u < params.width { top_right_right = sample_at(params, x + 2u, y - 1u); }
    var top_top = top;
    if y >= 2u { top_top = sample_at(params, x, y - 2u); }
    var left_left = left;
    if x >= 2u { left_left = sample_at(params, x - 2u, y); }
    var weighted = WeightedPrediction();
    if params.predictor == 6u {
        weighted = weighted_predict(top, top_left, top_right, left, top_top);
    }
    let prediction = predictor_value(params.predictor, weighted, top, left, top_left, top_right, top_top, left_left, top_right_right);
    if params.predictor == 6u { weighted_record(weighted, pixel); }
    let residual = bitcast<u32>(pixel) - bitcast<u32>(prediction);
    return (residual << 1u) ^ (0u - (residual >> 31u));
}

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.y != 0u || global_id.z != 0u || global_id.x >= arrayLength(&group_params) {
        return;
    }
    let params = group_params[global_id.x];
    active_params = params;
    if params.predictor == 6u {
        wp_reset();
        for (var index = 0u; index < params.width * 5u; index += 1u) {
            output_words[params.wp_scratch_word_offset + index] = 0u;
        }
    }

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
