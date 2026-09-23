override squeeze_enabled: bool = false;
override palette_enabled: bool = false;

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
    lz77_mode: u32,
    lz77_scratch_word_offset: u32,
    lz77_hash_mask: u32,
    squeeze: u32,
    source_width: u32,
    source_height: u32,
    palette_capacity: u32,
    palette_scratch_word_offset: u32,
    palette_hash_mask: u32,
    palette_channels: u32,
    palette_delta_predictor: u32,
    palette_delta_capacity: u32,
    _padding: array<u32, 7>,
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
// 34..67 are LZ77-token counts, words 67..100 are distance-token counts,
// followed by four-word events
// (kind, token, extra-bit count, extra bits). Weighted row state follows the
// maximum event range and is private to each group/channel invocation.
@group(0) @binding(1)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(2)
var<storage, read> group_params: array<Params>;

const OUTPUT_HEADER_WORDS: u32 = 100u;
const EVENT_WORDS: u32 = 4u;
const EVENT_OVERFLOW: u32 = 0xffffffffu;
const SQUEEZE_OVERFLOW: u32 = 0xfffffffeu;
const PALETTE_OVERFLOW: u32 = 0xfffffffdu;
const PALETTE_INVALID: u32 = 0xfffffffcu;

/*__JXL_MODULAR_PREDICT__*/

var<private> active_params: Params;
var<private> squeeze_failed: bool;
var<private> palette_failed: bool;

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
fn transformed_component(params: Params, x: u32, y: u32, component: u32) -> i32 {
    if params.rct_type == 42u || component >= 3u {
        return source_component(params, x, y, component);
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
        return vec3<i32>(luma, co, cg)[component];
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
    return transformed[component];
}

// Divide a signed wide value by twelve with truncation toward zero. Four base-65536
// digits keep each division operand in u32; Squeeze's largest numerator is below 2^35.
fn squeeze_div12(value: ModularI64) -> ModularI64 {
    let magnitude = mi_abs(value);
    var result = vec2<u32>(0u);
    var remainder = 0u;
    for (var digit = 4u; digit != 0u; digit -= 1u) {
        let shift = (digit - 1u) * 16u;
        let part = mi_shr(magnitude, shift).x & 65535u;
        let input = remainder * 65536u + part;
        result |= mi_shl(vec2<u32>(input / 12u, 0u), shift);
        remainder = input % 12u;
    }
    return select(result, mi_neg(result), mi_negative(value));
}

fn squeeze_average(a: i32, b: i32) -> i32 {
    let sum = mi_add(mi_add(mi_from_i32(a), mi_from_i32(b)), vec2<u32>(u32(a > b), 0u));
    return bitcast<i32>(mi_sar(sum, 1u).x);
}

fn squeeze_tendency(previous: i32, average: i32, next: i32) -> ModularI64 {
    let p = mi_from_i32(previous);
    let a = mi_from_i32(average);
    let n = mi_from_i32(next);
    let numerator = mi_sub(mi_sub(mi_mul_u32(p, 4u), mi_mul_u32(n, 3u)), a);
    let left_limit = mi_mul_u32(mi_sub(p, a), 2u);
    let right_limit = mi_mul_u32(mi_sub(a, n), 2u);
    var diff = vec2<u32>(0u);
    if previous >= average && average >= next {
        diff = squeeze_div12(mi_add(numerator, vec2<u32>(6u, 0u)));
        if mi_less(left_limit, mi_sub(diff, vec2<u32>(diff.x & 1u, 0u))) {
            diff = mi_add(left_limit, vec2<u32>(1u, 0u));
        }
        if mi_less(right_limit, mi_add(diff, vec2<u32>(diff.x & 1u, 0u))) { diff = right_limit; }
    } else if previous <= average && average <= next {
        diff = squeeze_div12(mi_sub(numerator, vec2<u32>(6u, 0u)));
        if mi_less(mi_add(diff, vec2<u32>(diff.x & 1u, 0u)), left_limit) {
            diff = mi_sub(left_limit, vec2<u32>(1u, 0u));
        }
        if mi_less(mi_sub(diff, vec2<u32>(diff.x & 1u, 0u)), right_limit) { diff = right_limit; }
    }
    return diff;
}

fn squeeze_residual(a: i32, b: i32, previous: i32, average: i32, next: i32) -> i32 {
    let residual = mi_sub(mi_sub(mi_from_i32(a), mi_from_i32(b)), squeeze_tendency(previous, average, next));
    let result = bitcast<i32>(residual.x);
    // Narrowing an unrepresentable residual would destroy reversibility. Publish only status.
    if any(residual != mi_from_i32(result)) { squeeze_failed = true; }
    return result;
}

fn palette_color(params: Params, x: u32, y: u32, delta: bool) -> vec4<u32> {
    var color = vec4<u32>(0u);
    for (var component = 0u; component < params.channels; component += 1u) {
        if delta {
            color[component] = output_words[palette_residual_base(params) + (component * params.source_height + y) * params.source_width + x];
        } else {
            color[component] = bitcast<u32>(transformed_component(params, x, y, component));
        }
    }
    return color;
}

fn palette_hash_base(params: Params) -> u32 {
    return params.palette_scratch_word_offset + 2u + params.channels * params.palette_capacity;
}

fn palette_residual_base(params: Params) -> u32 {
    return palette_hash_base(params) + params.palette_hash_mask + 1u;
}

fn palette_slot(params: Params, color: vec4<u32>, delta: bool) -> u32 {
    var hash = 2166136261u ^ select(0u, 0x80000000u, delta);
    for (var component = 0u; component < params.channels; component += 1u) {
        hash = (hash ^ color[component]) * 16777619u;
        hash ^= hash >> 16u;
    }
    let table = params.palette_scratch_word_offset + 2u;
    let heads = palette_hash_base(params);
    for (var probe = 0u; probe <= params.palette_hash_mask; probe += 1u) {
        let slot = (hash + probe) & params.palette_hash_mask;
        let entry = output_words[heads + slot];
        if entry == 0u { return slot; }
        if (entry - 1u < params.palette_delta_capacity) != delta { continue; }
        var matches = true;
        for (var component = 0u; component < params.channels; component += 1u) {
            if output_words[table + component * params.palette_capacity + entry - 1u] != color[component] {
                matches = false;
            }
        }
        if matches { return slot; }
    }
    return params.palette_hash_mask + 1u;
}

fn build_palette(params: Params) -> bool {
    // The artifact allocation, including hash heads, was cleared before this dispatch.
    // Only this group's first invocation owns its dictionary and all token channels.
    let count_offset = params.palette_scratch_word_offset;
    let table = count_offset + 2u;
    let heads = palette_hash_base(params);
    var colors = 0u;
    var deltas = 0u;
    let color_capacity = params.palette_capacity - params.palette_delta_capacity;
    for (var y = 0u; y < params.source_height; y += 1u) {
        for (var x = 0u; x < params.source_width; x += 1u) {
            // Absolute matches take priority. The first distinct tuples fill the color
            // partition; subsequent new colors use the independent residual partition.
            var delta = color_capacity == 0u;
            var color = palette_color(params, x, y, delta);
            var slot = palette_slot(params, color, delta);
            if slot > params.palette_hash_mask { return false; }
            if output_words[heads + slot] != 0u { continue; }
            if !delta && colors == color_capacity {
                if params.palette_delta_capacity == 0u { return false; }
                delta = true;
                color = palette_color(params, x, y, true);
                slot = palette_slot(params, color, true);
                if slot > params.palette_hash_mask { return false; }
                if output_words[heads + slot] != 0u { continue; }
            }
            var entry = params.palette_delta_capacity + colors;
            if delta {
                if deltas == params.palette_delta_capacity { return false; }
                entry = deltas;
                deltas += 1u;
            } else {
                colors += 1u;
            }
            for (var component = 0u; component < params.channels; component += 1u) {
                output_words[table + component * params.palette_capacity + entry] = color[component];
            }
            output_words[heads + slot] = entry + 1u;
        }
    }
    output_words[count_offset] = colors + deltas;
    output_words[count_offset + 1u] = deltas;
    return true;
}

fn palette_index(params: Params, x: u32, y: u32) -> i32 {
    let heads = palette_hash_base(params);
    if params.palette_capacity != params.palette_delta_capacity {
        let slot = palette_slot(params, palette_color(params, x, y, false), false);
        if slot <= params.palette_hash_mask {
            let entry = output_words[heads + slot];
            if entry != 0u {
                // Wire tables contain actual deltas followed immediately by actual colors.
                return i32(entry - 1u - params.palette_delta_capacity + output_words[params.palette_scratch_word_offset + 1u]);
            }
        }
    }
    if params.palette_delta_capacity != 0u {
        let slot = palette_slot(params, palette_color(params, x, y, true), true);
        if slot <= params.palette_hash_mask {
            let entry = output_words[heads + slot];
            if entry != 0u { return i32(entry - 1u); }
        }
    }
    palette_failed = true;
    return 0;
}

fn working_component(params: Params, x: u32, y: u32, component: u32) -> i32 {
    if !palette_enabled || params.palette_capacity == 0u {
        return transformed_component(params, x, y, component);
    }
    return palette_index(params, x, y);
}

fn squeeze_first(params: Params, component: u32, band: u32, point: vec2<u32>) -> i32 {
    let horizontal = params.squeeze == 1u || params.squeeze == 3u;
    let axis = select(1u, 0u, horizontal);
    let size = vec2<u32>(params.source_width, params.source_height)[axis];
    var step = vec2<u32>(0u);
    step[axis] = 1u;
    var source = point;
    source[axis] *= 2u;
    let a = working_component(params, source.x, source.y, component);
    if source[axis] + 1u == size { return a; } // unpaired average tail
    let b = working_component(params, source.x + step.x, source.y + step.y, component);
    let average = squeeze_average(a, b);
    if band == 0u { return average; }
    var next = average;
    if source[axis] + 2u < size {
        let c = working_component(params, source.x + 2u * step.x, source.y + 2u * step.y, component);
        next = c;
        if source[axis] + 3u < size {
            let d = working_component(params, source.x + 3u * step.x, source.y + 3u * step.y, component);
            next = squeeze_average(c, d);
        }
    }
    var previous = average;
    if source[axis] != 0u { previous = working_component(params, source.x - step.x, source.y - step.y, component); }
    return squeeze_residual(a, b, previous, average, next);
}

fn sample_at(params: Params, x: u32, y: u32) -> i32 {
    var channel = params.channel;
    var channels = params.channels;
    if palette_enabled && params.palette_capacity != 0u {
        if channel == 0u {
            let deltas = output_words[params.palette_scratch_word_offset + 1u];
            let entry = select(params.palette_delta_capacity + x - deltas, x, x < deltas);
            return bitcast<i32>(output_words[params.palette_scratch_word_offset + 2u + y * params.palette_capacity + entry]);
        }
        channel -= 1u;
        channels = 1u;
    }
    if !squeeze_enabled || params.squeeze == 0u { return working_component(params, x, y, channel); }
    let component = channel % channels;
    let first_band = (channel / channels) & 1u;
    let point = vec2<u32>(x, y);
    if params.squeeze < 3u { return squeeze_first(params, component, first_band, point); }
    // The second axis is perpendicular, so its input axis length is the original group length.
    let axis = select(1u, 0u, params.squeeze == 4u);
    let size = vec2<u32>(params.source_width, params.source_height)[axis];
    var step = vec2<u32>(0u);
    step[axis] = 1u;
    var source = point;
    source[axis] *= 2u;
    let a = squeeze_first(params, component, first_band, source);
    if source[axis] + 1u == size { return a; }
    let b = squeeze_first(params, component, first_band, source + step);
    let average = squeeze_average(a, b);
    if channel / (2u * channels) == 0u { return average; }
    var next = average;
    if source[axis] + 2u < size {
        let c = squeeze_first(params, component, first_band, source + 2u * step);
        next = c;
        if source[axis] + 3u < size {
            next = squeeze_average(c, squeeze_first(params, component, first_band, source + 3u * step));
        }
    }
    var previous = average;
    if source[axis] != 0u { previous = squeeze_first(params, component, first_band, source - step); }
    return squeeze_residual(a, b, previous, average, next);
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
    // One literal zero seeds the legacy distance-one match. JPEG XL's configured
    // minimum match length is seven, hence the encoded value is count-8.
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

fn residual_hash(params: Params, position: u32) -> u32 {
    let base = params.lz77_scratch_word_offset + position;
    var hash = output_words[base] * 0x9e3779b9u;
    hash = (hash ^ output_words[base + 1u]) * 0x85ebca6bu;
    hash = (hash ^ output_words[base + 2u]) * 0xc2b2ae35u;
    return (hash ^ (hash >> 16u)) & params.lz77_hash_mask;
}

fn emit_match(params: Params, length: u32, distance: u32) {
    let value = length - 7u;
    var token = value;
    var nbits = 0u;
    var bits = 0u;
    if value >= 16u {
        nbits = 31u - countLeadingZeros(value);
        token = 12u + nbits;
        bits = value - (1u << nbits);
    }
    output_words[params.output_word_offset + 34u + token] += 1u;
    append_event(params, 2u, token, nbits, bits);
    // JPEG XL's regular distance range follows 120 two-dimensional short codes.
    // Its decoder adds one after subtracting 120, so this is an exact linear distance.
    let coded_distance = distance + 119u;
    nbits = 31u - countLeadingZeros(coded_distance);
    token = nbits + 1u;
    bits = coded_distance - (1u << nbits);
    output_words[params.output_word_offset + 67u + token] += 1u;
    append_event(params, 3u, token, nbits, bits);
}

fn encode_greedy(params: Params) {
    let pixels = params.width * params.height;
    let residuals = params.lz77_scratch_word_offset;
    let previous = residuals + pixels;
    let heads = previous + pixels;
    for (var bucket = 0u; bucket <= params.lz77_hash_mask; bucket += 1u) {
        output_words[heads + bucket] = 0u;
    }
    // Prediction always visits every sample, including pixels later covered by a match.
    for (var y = 0u; y < params.height; y += 1u) {
        for (var x = 0u; x < params.width; x += 1u) {
            output_words[residuals + y * params.width + x] = packed_residual(params, x, y);
        }
    }
    var position = 0u;
    while position < pixels {
        var best_length = 0u;
        var best_distance = 0u;
        if position + 7u <= pixels {
            var link = output_words[heads + residual_hash(params, position)];
            for (var attempt = 0u; attempt < 32u && link != 0u; attempt += 1u) {
                let candidate = link - 1u;
                var length = 0u;
                while position + length < pixels {
                    if output_words[residuals + candidate + length] != output_words[residuals + position + length] { break; }
                    length += 1u;
                }
                if length > best_length {
                    best_length = length;
                    best_distance = position - candidate;
                }
                if position + best_length == pixels { break; }
                link = output_words[previous + candidate];
            }
        }
        var consumed = 1u;
        if best_length >= 7u {
            emit_match(params, best_length, best_distance);
            consumed = best_length;
        } else {
            emit_raw(params, output_words[residuals + position]);
        }
        // Inserting skipped positions preserves overlap and future hash-chain matches.
        for (var index = position; index < position + consumed && index + 2u < pixels; index += 1u) {
            let bucket = heads + residual_hash(params, index);
            output_words[previous + index] = output_words[bucket];
            output_words[bucket] = index + 1u;
        }
        position += consumed;
    }
}

fn encode_tokens(params: Params) {
    squeeze_failed = false;
    palette_failed = false;
    reset_prediction(params);

    if params.lz77_mode == 1u {
        encode_greedy(params);
        if squeeze_failed { output_words[params.output_word_offset] = SQUEEZE_OVERFLOW; }
        if palette_failed { output_words[params.output_word_offset] = PALETTE_INVALID; }
        return;
    }

    encode_zero_runs(params);
    if squeeze_failed { output_words[params.output_word_offset] = SQUEEZE_OVERFLOW; }
    if palette_failed { output_words[params.output_word_offset] = PALETTE_INVALID; }
}

fn reset_prediction(params: Params) {
    active_params = params;
    if params.predictor == 6u {
        wp_reset();
        for (var index = 0u; index < params.width * 5u; index += 1u) {
            output_words[params.wp_scratch_word_offset + index] = 0u;
        }
    }
}

fn encode_zero_runs(params: Params) {
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

fn build_palette_residuals(params: Params) {
    if params.palette_delta_predictor >= 14u { return; }
    var source_params = params;
    source_params.width = params.source_width;
    source_params.height = params.source_height;
    source_params.squeeze = 0u;
    source_params.palette_capacity = 0u; // read post-RCT source words, before Palette or Squeeze
    source_params.predictor = params.palette_delta_predictor;
    let residual_base = palette_residual_base(params);
    let pixels = params.source_width * params.source_height;
    source_params.wp_scratch_word_offset = residual_base + pixels * params.channels;
    for (var component = 0u; component < params.channels; component += 1u) {
        source_params.channel = component;
        reset_prediction(source_params);
        for (var y = 0u; y < params.source_height; y += 1u) {
            for (var x = 0u; x < params.source_width; x += 1u) {
                let packed = packed_residual(source_params, x, y);
                // Undo the entropy zigzag; Palette stores raw signed residual words.
                output_words[residual_base + component * pixels + y * params.source_width + x] = (packed >> 1u) ^ (0u - (packed & 1u));
            }
        }
    }
}

@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.y != 0u || global_id.z != 0u || global_id.x >= arrayLength(&group_params) { return; }
    let params = group_params[global_id.x];
    if palette_enabled && params.palette_capacity != 0u {
        if params.channel != 0u { return; }
        build_palette_residuals(params);
        if !build_palette(params) {
            output_words[params.output_word_offset] = PALETTE_OVERFLOW;
            return;
        }
        for (var channel = 0u; channel < params.palette_channels; channel += 1u) {
            var token_params = group_params[global_id.x + channel];
            if channel == 0u { token_params.width = output_words[params.palette_scratch_word_offset]; }
            encode_tokens(token_params);
        }
    } else {
        encode_tokens(params);
    }
}
