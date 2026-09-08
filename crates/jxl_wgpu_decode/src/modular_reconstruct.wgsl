//! Reusable JPEG XL Modular MA-tree reconstruction and predictor kernel fragment.
//!
//! The caller supplies packed `modular_metadata`, entropy helpers, one raw-i32 `reconstructed`
//! storage view, `Params`, and the shared decode error constants. `decode_adaptive_channel()`
//! reconstructs exactly one channel and deliberately performs no color conversion or output IO.

/*__JXL_MODULAR_PREDICT__*/

var<private> predictor_prev_grad: i32;

fn unsigned_abs_i32(value: i32) -> i32 {
    if value < 0i {
        return bitcast<i32>(0u - bitcast<u32>(value));
    }
    return value;
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

fn wp_max_weight(component: u32) -> u32 {
    switch component {
        case 0u: { return params.wp_w0; }
        case 1u: { return params.wp_w1; }
        case 2u: { return params.wp_w2; }
        default: { return params.wp_w3; }
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
        if node_index >= entropy_metadata(META_NODE_COUNT)
            || depth > entropy_metadata(META_MAX_DEPTH) {
            decode_error = ERROR_MA_TREE;
            return entropy_metadata(META_TREE_OFFSET);
        }
        let node = entropy_metadata(META_TREE_OFFSET) + node_index * 8u;
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
    return entropy_metadata(META_TREE_OFFSET);
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
        var weighted = WeightedPrediction();
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

fn predictor_error() { decode_error = ERROR_PREDICTOR; }
fn wp_coefficient(index: u32) -> u32 {
    switch index {
        case 0u: { return params.wp_p1; }
        case 1u: { return params.wp_p2; }
        case 2u: { return params.wp_p3a; }
        case 3u: { return params.wp_p3b; }
        case 4u: { return params.wp_p3c; }
        case 5u: { return params.wp_p3d; }
        case 6u: { return params.wp_p3e; }
        default: { return 0u; }
    }
}
