//! Reusable JPEG XL Modular MA-tree reconstruction and predictor kernel fragment.
//!
//! The caller supplies packed `modular_metadata`, entropy helpers, one raw-i32 `reconstructed`
//! storage view, `Params`, and the shared decode error constants. `decode_adaptive_channel()`
//! reconstructs exactly one channel and deliberately performs no color conversion or output IO.

struct WeightedPrediction {
    prediction: i32,
    max_error: i32,
    subpred: array<i32, 4>,
};

var<private> predictor_prev_grad: i32;
var<private> wp_x: u32;
var<private> wp_y: u32;
var<private> wp_true_err_w: i32;
var<private> wp_true_err_nw: i32;
var<private> wp_true_err_n: i32;
var<private> wp_true_err_ne: i32;
var<private> wp_subpred_nw_ww: array<u32, 4>;
var<private> wp_subpred_n_w: array<u32, 4>;
var<private> wp_subpred_ne: array<u32, 4>;

fn abs_diff_i32(a: i32, b: i32) -> u32 {
    if a >= b {
        return bitcast<u32>(a) - bitcast<u32>(b);
    }
    return bitcast<u32>(b) - bitcast<u32>(a);
}

fn unsigned_abs_i32(value: i32) -> i32 {
    if value < 0i {
        return bitcast<i32>(0u - bitcast<u32>(value));
    }
    return value;
}

fn gradient_i32(north: i32, west: i32, north_west: i32) -> i32 {
    let gradient = north + west - north_west;
    return clamp(gradient, min(north, west), max(north, west));
}

fn sample_at(channel: u32, index: u32, x: u32, y: u32) -> i32 {
    if modular_descriptor_mode() {
        return bitcast<i32>(modular_descriptor_sample_load(channel, x, y));
    }
    return bitcast<i32>(reconstruction_load(channel * params.sample_count + index));
}

fn wp_row_base() -> u32 {
    return modular_arena_words(params.sample_count * params.source_channels);
}

fn wp_scratch_width() -> u32 {
    return modular_entropy_max_width(params.width);
}

fn wp_current_width() -> u32 {
    return modular_current_channel_width(params.width);
}

fn wp_true_error(index: u32) -> i32 {
    return bitcast<i32>(reconstruction_load(wp_row_base() + index));
}

fn wp_subpred_error(index: u32, component: u32) -> u32 {
    return reconstruction_load(wp_row_base() + wp_scratch_width() + index * 4u + component);
}

fn wp_store_row(index: u32, true_error: i32, errors: array<u32, 4>) {
    reconstruction_store(wp_row_base() + index, bitcast<u32>(true_error));
    for (var component = 0u; component < 4u; component += 1u) {
        reconstruction_store(
            wp_row_base() + wp_scratch_width() + index * 4u + component,
            errors[component],
        );
    }
}

fn wp_reset() {
    wp_x = 0u;
    wp_y = 0u;
    wp_true_err_w = 0i;
    wp_true_err_nw = 0i;
    wp_true_err_n = 0i;
    wp_true_err_ne = 0i;
    for (var component = 0u; component < 4u; component += 1u) {
        wp_subpred_nw_ww[component] = 0u;
        wp_subpred_n_w[component] = 0u;
        wp_subpred_ne[component] = 0u;
    }
}

fn floor_log2(value: u32) -> u32 {
    var remaining = value;
    var out = 0u;
    while remaining > 1u {
        remaining >>= 1u;
        out += 1u;
    }
    return out;
}

fn wp_max_weight(component: u32) -> u32 {
    switch component {
        case 0u: { return params.wp_w0; }
        case 1u: { return params.wp_w1; }
        case 2u: { return params.wp_w2; }
        default: { return params.wp_w3; }
    }
}

// Computes the exact low 32 bits of `(value * multiplier) >> 24` with an arithmetic shift.
// JPEG XL's normalized self-correcting weights keep the mathematical result in i32, while the
// intermediate product requires the i64 precision that portable WGSL intentionally omits.
fn signed_mul_shift24(value: i32, multiplier: u32) -> i32 {
    let negative = value < 0i;
    var magnitude = bitcast<u32>(value);
    if negative {
        magnitude = 0u - magnitude;
    }
    let a0 = magnitude & 0xffffu;
    let a1 = magnitude >> 16u;
    let b0 = multiplier & 0xffffu;
    let b1 = multiplier >> 16u;
    var low = a0 * b0;
    var high = a1 * b1 + (a0 * b1 >> 16u) + (a1 * b0 >> 16u);
    let add1 = (a0 * b1 & 0xffffu) << 16u;
    let before1 = low;
    low += add1;
    if low < before1 {
        high += 1u;
    }
    let add2 = (a1 * b0 & 0xffffu) << 16u;
    let before2 = low;
    low += add2;
    if low < before2 {
        high += 1u;
    }
    var shifted = (low >> 24u) | (high << 8u);
    if negative {
        if (low & 0x00ffffffu) != 0u {
            shifted += 1u;
        }
        return bitcast<i32>(0u - shifted);
    }
    return bitcast<i32>(shifted);
}

fn weighted_predict(n: i32, nw: i32, ne: i32, w: i32, nn: i32) -> WeightedPrediction {
    let n3 = n << 3u;
    let nw3 = nw << 3u;
    let ne3 = ne << 3u;
    let w3 = w << 3u;
    let nn3 = nn << 3u;
    var subpred = array<i32, 4>(
        w3 + ne3 - n3,
        n3 - ((wp_true_err_w + wp_true_err_n + wp_true_err_ne) * i32(params.wp_p1) >> 5u),
        w3 - ((wp_true_err_w + wp_true_err_n + wp_true_err_nw) * i32(params.wp_p2) >> 5u),
        n3 - ((wp_true_err_nw * i32(params.wp_p3a)
            + wp_true_err_n * i32(params.wp_p3b)
            + wp_true_err_ne * i32(params.wp_p3c)
            + (nn3 - n3) * i32(params.wp_p3d)
            + (nw3 - w3) * i32(params.wp_p3e)) >> 5u),
    );
    var weights: array<u32, 4>;
    var sum_weights = 0u;
    for (var component = 0u; component < 4u; component += 1u) {
        let error_sum = wp_subpred_nw_ww[component]
            + wp_subpred_n_w[component]
            + wp_subpred_ne[component];
        let shifted_error = (error_sum + 1u) >> 5u;
        let shift = floor_log2(shifted_error);
        let divisor_index = (error_sum >> shift) + 1u;
        let reciprocal = (1u << 24u) / divisor_index;
        weights[component] = 4u + ((wp_max_weight(component) * reciprocal) >> shift);
        sum_weights += weights[component];
    }
    let log_weight = floor_log2(sum_weights >> 4u);
    sum_weights = 0u;
    for (var component = 0u; component < 4u; component += 1u) {
        weights[component] >>= log_weight;
        sum_weights += weights[component];
    }
    var weighted_sum = i32(sum_weights >> 1u) - 1i;
    for (var component = 0u; component < 4u; component += 1u) {
        weighted_sum += subpred[component] * i32(weights[component]);
    }
    var prediction = signed_mul_shift24(weighted_sum, (1u << 24u) / sum_weights);
    if ((wp_true_err_n ^ wp_true_err_w) | (wp_true_err_n ^ wp_true_err_nw)) <= 0i {
        prediction = clamp(prediction, min(n3, min(w3, ne3)), max(n3, max(w3, ne3)));
    }
    var max_error = wp_true_err_w;
    if abs(wp_true_err_n) > abs(max_error) {
        max_error = wp_true_err_n;
    }
    if abs(wp_true_err_nw) > abs(max_error) {
        max_error = wp_true_err_nw;
    }
    if abs(wp_true_err_ne) > abs(max_error) {
        max_error = wp_true_err_ne;
    }
    return WeightedPrediction(prediction, max_error, subpred);
}

fn weighted_record(prediction: WeightedPrediction, sample: i32) {
    let width = wp_current_width();
    let sample3 = sample << 3u;
    let true_error = prediction.prediction - sample3;
    var errors: array<u32, 4>;
    for (var component = 0u; component < 4u; component += 1u) {
        errors[component] = (abs_diff_i32(prediction.subpred[component], sample3) + 3u) >> 3u;
    }
    wp_store_row(wp_x, true_error, errors);
    wp_x += 1u;
    if wp_x >= width {
        wp_y += 1u;
        wp_x = 0u;
        wp_true_err_w = 0i;
        wp_true_err_n = wp_true_error(0u);
        wp_true_err_nw = wp_true_err_n;
        for (var component = 0u; component < 4u; component += 1u) {
            wp_subpred_n_w[component] = wp_subpred_error(0u, component);
            wp_subpred_nw_ww[component] = wp_subpred_n_w[component];
        }
        if width <= 1u {
            wp_true_err_ne = wp_true_err_n;
            for (var component = 0u; component < 4u; component += 1u) {
                wp_subpred_ne[component] = wp_subpred_n_w[component];
            }
        } else {
            wp_true_err_ne = wp_true_error(1u);
            for (var component = 0u; component < 4u; component += 1u) {
                wp_subpred_ne[component] = wp_subpred_error(1u, component);
            }
        }
        return;
    }
    wp_true_err_w = true_error;
    wp_true_err_nw = wp_true_err_n;
    wp_true_err_n = wp_true_err_ne;
    for (var component = 0u; component < 4u; component += 1u) {
        wp_subpred_nw_ww[component] = wp_subpred_n_w[component];
        wp_subpred_n_w[component] = wp_subpred_ne[component] + errors[component];
    }
    if wp_x + 1u >= width {
        wp_true_err_ne = wp_true_err_n;
        for (var component = 0u; component < 4u; component += 1u) {
            wp_subpred_ne[component] = wp_subpred_n_w[component];
        }
    } else if wp_y != 0u {
        wp_true_err_ne = wp_true_error(wp_x + 1u);
        for (var component = 0u; component < 4u; component += 1u) {
            wp_subpred_ne[component] = wp_subpred_error(wp_x + 1u, component);
        }
    }
}

fn ma_property(
    property: u32,
    index: u32,
    x: u32,
    y: u32,
    n: i32,
    w: i32,
    nw: i32,
    ne: i32,
    nn: i32,
    ww: i32,
    max_error: i32,
) -> i32 {
    switch property {
        case 0u: { return i32(current_channel); }
        case 1u: { return i32(params.stream_index); }
        case 2u: { return i32(y); }
        case 3u: { return i32(x); }
        case 4u: { return unsigned_abs_i32(n); }
        case 5u: { return unsigned_abs_i32(w); }
        case 6u: { return n; }
        case 7u: { return w; }
        case 8u: { return w - predictor_prev_grad; }
        case 9u: { return w - nw + n; }
        case 10u: { return w - nw; }
        case 11u: { return nw - n; }
        case 12u: { return n - ne; }
        case 13u: { return n - nn; }
        case 14u: { return w - ww; }
        case 15u: { return max_error; }
        default: {}
    }
    let previous_index = (property - 16u) / 4u;
    var previous_channel = 0u;
    if modular_descriptor_mode() {
        if previous_index >= modular_channel_reference_count(current_channel) {
            return 0i;
        }
        previous_channel = modular_metadata[
            modular_channel_reference_offset(current_channel) + previous_index
        ];
    } else {
        if previous_index >= current_channel {
            return 0i;
        }
        previous_channel = current_channel - previous_index - 1u;
    }
    let center = sample_at(previous_channel, index, x, y);
    let kind = (property - 16u) & 3u;
    if kind == 0u {
        return unsigned_abs_i32(center);
    }
    if kind == 1u {
        return center;
    }
    var previous_gradient = 0i;
    let width = modular_current_channel_width(params.width);
    if x == 0u && y != 0u {
        previous_gradient = sample_at(previous_channel, index - width, x, y - 1u);
    } else if y == 0u && x != 0u {
        previous_gradient = sample_at(previous_channel, index - 1u, x - 1u, y);
    } else if x != 0u && y != 0u {
        previous_gradient = gradient_i32(
            sample_at(previous_channel, index - width, x, y - 1u),
            sample_at(previous_channel, index - 1u, x - 1u, y),
            sample_at(previous_channel, index - width - 1u, x - 1u, y - 1u),
        );
    }
    if kind == 2u {
        return bitcast<i32>(abs_diff_i32(center, previous_gradient));
    }
    return center - previous_gradient;
}

fn ma_leaf(
    index: u32,
    x: u32,
    y: u32,
    n: i32,
    w: i32,
    nw: i32,
    ne: i32,
    nn: i32,
    ww: i32,
    max_error: i32,
) -> u32 {
    var node_index = 0u;
    var depth = 0u;
    loop {
        if node_index >= modular_metadata[META_NODE_COUNT]
            || depth > modular_metadata[META_MAX_DEPTH] {
            decode_error = ERROR_MA_TREE;
            return modular_metadata[META_TREE_OFFSET];
        }
        let node = modular_metadata[META_TREE_OFFSET] + node_index * 8u;
        let kind = modular_metadata[node];
        if kind == 1u {
            return node;
        }
        if kind != 0u {
            decode_error = ERROR_MA_TREE;
            return node;
        }
        let property = modular_metadata[node + 1u];
        let threshold = bitcast<i32>(modular_metadata[node + 2u]);
        let value = ma_property(property, index, x, y, n, w, nw, ne, nn, ww, max_error);
        // MA trees encode the preorder left subtree for values greater than the threshold.
        if value > threshold {
            node_index = modular_metadata[node + 3u];
        } else {
            node_index = modular_metadata[node + 4u];
        }
        depth += 1u;
    }
    return modular_metadata[META_TREE_OFFSET];
}

fn predictor_value(
    predictor: u32,
    weighted: WeightedPrediction,
    n: i32,
    w: i32,
    nw: i32,
    ne: i32,
    nn: i32,
    ww: i32,
    nee: i32,
) -> i32 {
    switch predictor {
        case 0u: { return 0i; }
        case 1u: { return w; }
        case 2u: { return n; }
        case 3u: { return (w + n) / 2i; }
        case 4u: {
            if abs_diff_i32(n, nw) < abs_diff_i32(w, nw) {
                return w;
            }
            return n;
        }
        case 5u: { return gradient_i32(n, w, nw); }
        case 6u: { return (weighted.prediction + 3i) >> 3u; }
        case 7u: { return ne; }
        case 8u: { return nw; }
        case 9u: { return ww; }
        case 10u: { return (w + nw) / 2i; }
        case 11u: { return (n + nw) / 2i; }
        case 12u: { return (n + ne) / 2i; }
        case 13u: { return (6i * n - 2i * nn + 7i * w + ww + nee + 3i * ne + 8i) / 16i; }
        default: {
            decode_error = ERROR_PREDICTOR;
            return 0i;
        }
    }
}

fn decode_adaptive_channel(start: u32, may_pause: bool, pause_cursor: u32) -> u32 {
    modular_select_channel(current_channel);
    let width = modular_current_channel_width(params.width);
    let height = modular_current_channel_height(params.height);
    let channel_samples = width * height;
    if start == 0u {
        predictor_prev_grad = 0i;
        if params.needs_self_correcting != 0u {
            wp_reset();
        }
    }
    var decoded = start;
    while decoded < channel_samples && decode_error == 0u
        && (!may_pause || bit_cursor < pause_cursor) {
        let x = decoded % width;
        let y = decoded / width;
        var w = 0i;
        if x != 0u {
            w = sample_at(current_channel, decoded - 1u, x - 1u, y);
        } else if y != 0u {
            w = sample_at(current_channel, decoded - width, x, y - 1u);
        }
        var n = w;
        var nw = w;
        if y != 0u {
            n = sample_at(current_channel, decoded - width, x, y - 1u);
            nw = n;
            if x != 0u {
                nw = sample_at(current_channel, decoded - width - 1u, x - 1u, y - 1u);
            }
        }
        var ne = n;
        if y != 0u && x + 1u < width {
            ne = sample_at(current_channel, decoded - width + 1u, x + 1u, y - 1u);
        }
        var nee = ne;
        if y != 0u && x + 2u < width {
            nee = sample_at(current_channel, decoded - width + 2u, x + 2u, y - 1u);
        }
        var nn = n;
        if y >= 2u {
            nn = sample_at(current_channel, decoded - 2u * width, x, y - 2u);
        }
        var ww = w;
        if x >= 2u {
            ww = sample_at(current_channel, decoded - 2u, x - 2u, y);
        }
        var weighted = WeightedPrediction(0i, 0i, array<i32, 4>(0i, 0i, 0i, 0i));
        if params.needs_self_correcting != 0u {
            weighted = weighted_predict(n, nw, ne, w, nn);
        }
        let leaf = ma_leaf(decoded, x, y, n, w, nw, ne, nn, ww, weighted.max_error);
        if decode_error != 0u {
            break;
        }
        let predictor = modular_metadata[leaf + 1u];
        let leaf_offset = modular_metadata[leaf + 2u];
        let cluster = modular_metadata[leaf + 3u];
        let multiplier = modular_metadata[leaf + 4u];
        let packed = entropy_read_varint(cluster, width);
        let difference = unpack_signed(packed);
        let residual = bitcast<i32>(
            bitcast<u32>(difference) * multiplier + leaf_offset
        );
        let prediction = predictor_value(predictor, weighted, n, w, nw, ne, nn, ww, nee);
        let sample = bitcast<i32>(bitcast<u32>(residual) + bitcast<u32>(prediction));
        if !modular_descriptor_mode() {
            let maximum = i32(params.source_mask);
            let signed_transform_channel = params.source_channels >= 3u
                && (current_channel == 1u || current_channel == 2u);
            if (!signed_transform_channel && (sample < 0i || sample > maximum))
                || (signed_transform_channel && (sample < -maximum || sample > maximum)) {
                decode_error = ERROR_RAW_TOKEN;
                break;
            }
            reconstruction_store(
                current_channel * params.sample_count + decoded,
                bitcast<u32>(sample),
            );
        } else {
            modular_descriptor_sample_store(current_channel, x, y, bitcast<u32>(sample));
        }
        if params.needs_self_correcting != 0u {
            weighted_record(weighted, sample);
        }
        if x + 1u == width {
            predictor_prev_grad = 0i;
        } else {
            predictor_prev_grad = w - nw + n;
        }
        decoded += 1u;
    }
    return decoded - start;
}
