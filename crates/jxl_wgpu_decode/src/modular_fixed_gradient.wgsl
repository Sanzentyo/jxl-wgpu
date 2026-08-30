//! JPEG XL Modular channel-fixed Gradient reconstruction fast path.
//!
//! The host proves that every MA decision tests only the channel property and that channels 0..3
//! all terminate at Gradient leaves with zero offset and unit multiplier. The four channel-ordered
//! entropy clusters and the common leaf fields are carried by `Params`. This fragment deliberately
//! omits MA traversal, Weighted prediction, unused neighbors, predictor dispatch, and per-sample
//! division/modulo. It provides the same `decode_adaptive_channel()` entry as the generic fragment.

fn fixed_gradient_i32(north: i32, west: i32, north_west: i32) -> i32 {
    let gradient = north + west - north_west;
    return clamp(gradient, min(north, west), max(north, west));
}

fn fixed_leaf_cluster(channel: u32) -> u32 {
    switch channel {
        case 0u: { return params.fixed_leaf_cluster0; }
        case 1u: { return params.fixed_leaf_cluster1; }
        case 2u: { return params.fixed_leaf_cluster2; }
        case 3u: { return params.fixed_leaf_cluster3; }
        default: {
            decode_error = ERROR_ENTROPY_CLUSTER;
            return 0u;
        }
    }
}

fn decode_adaptive_channel() -> u32 {
    if params.fixed_leaf_predictor != 5u
        || params.fixed_leaf_offset != 0u
        || params.fixed_leaf_multiplier != 1u {
        decode_error = ERROR_PREDICTOR;
        return 0u;
    }
    let cluster = fixed_leaf_cluster(current_channel);
    let channel_base = current_channel * params.sample_count;
    let maximum = i32(params.source_mask);
    let signed_transform_channel = params.source_channels >= 3u
        && (current_channel == 1u || current_channel == 2u);
    var decoded = 0u;
    for (var y = 0u; y < params.height && decode_error == 0u; y += 1u) {
        let row_base = y * params.width;
        var west = 0i;
        for (var x = 0u; x < params.width && decode_error == 0u; x += 1u) {
            let index = row_base + x;
            var north = west;
            var north_west = west;
            if y != 0u {
                north = bitcast<i32>(reconstruction_load(channel_base + index - params.width));
                north_west = north;
                if x != 0u {
                    north_west = bitcast<i32>(
                        reconstruction_load(channel_base + index - params.width - 1u)
                    );
                }
            }
            if x == 0u {
                west = north;
            }
            let packed = entropy_read_varint(cluster, params.width);
            if decode_error != 0u {
                break;
            }
            let residual = unpack_signed(packed);
            let prediction = fixed_gradient_i32(north, west, north_west);
            let sample = bitcast<i32>(bitcast<u32>(residual) + bitcast<u32>(prediction));
            if (!signed_transform_channel && (sample < 0i || sample > maximum))
                || (signed_transform_channel && (sample < -maximum || sample > maximum)) {
                decode_error = ERROR_RAW_TOKEN;
                break;
            }
            reconstruction_store(channel_base + index, bitcast<u32>(sample));
            west = sample;
            decoded += 1u;
        }
    }
    return decoded;
}
