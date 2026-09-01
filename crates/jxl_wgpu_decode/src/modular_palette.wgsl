//! GPU-resident inverse JPEG XL Modular Palette.

override wg_x: u32 = 64u;

struct Params {
    // width, height, row stride, word offset
    palette: vec4<u32>,
    indices: vec4<u32>,
    output: vec4<u32>,
    // palette channel, color count, delta count, predictor
    info: vec4<u32>,
    // first sample, exclusive end, bit depth, predictor scratch word offset
    range: vec4<u32>,
    // p1, p2, p3a, p3b
    wp_first: vec4<u32>,
    // p3c, p3d, p3e, w0
    wp_second: vec4<u32>,
    // w1, w2, w3, reserved
    wp_third: vec4<u32>,
};

struct WeightedPrediction {
    prediction: i32,
    max_error: i32,
    subpred: array<i32, 4>,
};

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

const STATE_WORDS: u32 = 20u;

var<private> wp_x: u32;
var<private> wp_y: u32;
var<private> wp_true_err_w: i32;
var<private> wp_true_err_nw: i32;
var<private> wp_true_err_n: i32;
var<private> wp_true_err_ne: i32;
var<private> wp_subpred_nw_ww: array<u32, 4>;
var<private> wp_subpred_n_w: array<u32, 4>;
var<private> wp_subpred_ne: array<u32, 4>;

const DELTA_PALETTE: array<vec3<i32>, 72> = array<vec3<i32>, 72>(
    vec3<i32>(0, 0, 0), vec3<i32>(4, 4, 4), vec3<i32>(11, 0, 0),
    vec3<i32>(0, 0, -13), vec3<i32>(0, -12, 0), vec3<i32>(-10, -10, -10),
    vec3<i32>(-18, -18, -18), vec3<i32>(-27, -27, -27), vec3<i32>(-18, -18, 0),
    vec3<i32>(0, 0, -32), vec3<i32>(-32, 0, 0), vec3<i32>(-37, -37, -37),
    vec3<i32>(0, -32, -32), vec3<i32>(24, 24, 45), vec3<i32>(50, 50, 50),
    vec3<i32>(-45, -24, -24), vec3<i32>(-24, -45, -45), vec3<i32>(0, -24, -24),
    vec3<i32>(-34, -34, 0), vec3<i32>(-24, 0, -24), vec3<i32>(-45, -45, -24),
    vec3<i32>(64, 64, 64), vec3<i32>(-32, 0, -32), vec3<i32>(0, -32, 0),
    vec3<i32>(-32, 0, 32), vec3<i32>(-24, -45, -24), vec3<i32>(45, 24, 45),
    vec3<i32>(24, -24, -45), vec3<i32>(-45, -24, 24), vec3<i32>(80, 80, 80),
    vec3<i32>(64, 0, 0), vec3<i32>(0, 0, -64), vec3<i32>(0, -64, -64),
    vec3<i32>(-24, -24, 45), vec3<i32>(96, 96, 96), vec3<i32>(64, 64, 0),
    vec3<i32>(45, -24, -24), vec3<i32>(34, -34, 0), vec3<i32>(112, 112, 112),
    vec3<i32>(24, -45, -45), vec3<i32>(45, 45, -24), vec3<i32>(0, -32, 32),
    vec3<i32>(24, -24, 45), vec3<i32>(0, 96, 96), vec3<i32>(45, -24, 24),
    vec3<i32>(24, -45, -24), vec3<i32>(-24, -45, 24), vec3<i32>(0, -64, 0),
    vec3<i32>(96, 0, 0), vec3<i32>(128, 128, 128), vec3<i32>(64, 0, 64),
    vec3<i32>(144, 144, 144), vec3<i32>(96, 96, 0), vec3<i32>(-36, -36, 36),
    vec3<i32>(45, -24, -45), vec3<i32>(45, -45, -24), vec3<i32>(0, 0, -96),
    vec3<i32>(0, 128, 128), vec3<i32>(0, 96, 0), vec3<i32>(45, 24, -45),
    vec3<i32>(-128, 0, 0), vec3<i32>(24, -45, 24), vec3<i32>(-45, 24, -45),
    vec3<i32>(64, 0, -64), vec3<i32>(64, -64, -64), vec3<i32>(96, 0, 96),
    vec3<i32>(45, -45, 24), vec3<i32>(24, 45, -45), vec3<i32>(64, 64, -64),
    vec3<i32>(128, 128, 0), vec3<i32>(0, 0, -128), vec3<i32>(-24, 45, -45),
);

fn load_plane(plane: vec4<u32>, x: u32, y: u32) -> i32 {
    return bitcast<i32>(arena[plane.w + y * plane.z + x]);
}

fn store_output(x: u32, y: u32, value: i32) {
    arena[params.output.w + y * params.output.z + x] = bitcast<u32>(value);
}

fn output_at(x: u32, y: u32) -> i32 {
    return bitcast<i32>(arena[params.output.w + y * params.output.z + x]);
}

fn add_wrap(left: i32, right: i32) -> i32 {
    return bitcast<i32>(bitcast<u32>(left) + bitcast<u32>(right));
}

fn palette_value(index: i32) -> i32 {
    let channel = params.info.x;
    let palette_size = params.info.y + params.info.z;
    if index >= 0i && u32(index) < palette_size {
        return load_plane(params.palette, u32(index), channel);
    }
    if index < 0i {
        if channel >= 3u {
            return 0i;
        }
        let normalized = (0u - (bitcast<u32>(index) + 1u)) % 143u;
        var value = DELTA_PALETTE[(normalized + 1u) >> 1u][channel];
        if (normalized & 1u) == 0u {
            value = bitcast<i32>(0u - bitcast<u32>(value));
        }
        if params.range.z > 8u {
            value = bitcast<i32>(bitcast<u32>(value) << min(params.range.z, 24u) - 8u);
        }
        return value;
    }
    if channel >= 3u {
        return 0i;
    }
    let maximum = (1u << params.range.z) - 1u;
    var implicit_index = u32(index) - palette_size;
    if implicit_index < 64u {
        let digit = (implicit_index >> (2u * channel)) % 4u;
        return i32(digit * maximum / 4u + (1u << (max(params.range.z, 3u) - 3u)));
    }
    implicit_index -= 64u;
    if channel == 1u {
        implicit_index /= 5u;
    } else if channel == 2u {
        implicit_index /= 25u;
    }
    return i32((implicit_index % 5u) * maximum / 4u);
}

fn abs_diff_i32(a: i32, b: i32) -> u32 {
    if a >= b {
        return bitcast<u32>(a) - bitcast<u32>(b);
    }
    return bitcast<u32>(b) - bitcast<u32>(a);
}

fn gradient_i32(north: i32, west: i32, north_west: i32) -> i32 {
    let gradient = north + west - north_west;
    return clamp(gradient, min(north, west), max(north, west));
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
        case 0u: { return params.wp_second.w; }
        case 1u: { return params.wp_third.x; }
        case 2u: { return params.wp_third.y; }
        default: { return params.wp_third.z; }
    }
}

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

fn wp_row_base() -> u32 {
    return params.range.w + STATE_WORDS;
}

fn wp_true_error(index: u32) -> i32 {
    return bitcast<i32>(arena[wp_row_base() + index]);
}

fn wp_subpred_error(index: u32, component: u32) -> u32 {
    return arena[wp_row_base() + params.output.x + index * 4u + component];
}

fn wp_store_row(index: u32, true_error: i32, errors: array<u32, 4>) {
    arena[wp_row_base() + index] = bitcast<u32>(true_error);
    for (var component = 0u; component < 4u; component += 1u) {
        arena[wp_row_base() + params.output.x + index * 4u + component] = errors[component];
    }
}

fn predictor_reset() {
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

fn predictor_load_state(start: u32) {
    if params.info.w != 6u {
        return;
    }
    wp_x = start % params.output.x;
    wp_y = start / params.output.x;
    wp_true_err_w = bitcast<i32>(arena[params.range.w + 1u]);
    wp_true_err_nw = bitcast<i32>(arena[params.range.w + 2u]);
    wp_true_err_n = bitcast<i32>(arena[params.range.w + 3u]);
    wp_true_err_ne = bitcast<i32>(arena[params.range.w + 4u]);
    for (var component = 0u; component < 4u; component += 1u) {
        wp_subpred_nw_ww[component] = arena[params.range.w + 5u + component];
        wp_subpred_n_w[component] = arena[params.range.w + 9u + component];
        wp_subpred_ne[component] = arena[params.range.w + 13u + component];
    }
}

fn predictor_store_state() {
    if params.info.w != 6u {
        return;
    }
    arena[params.range.w + 1u] = bitcast<u32>(wp_true_err_w);
    arena[params.range.w + 2u] = bitcast<u32>(wp_true_err_nw);
    arena[params.range.w + 3u] = bitcast<u32>(wp_true_err_n);
    arena[params.range.w + 4u] = bitcast<u32>(wp_true_err_ne);
    for (var component = 0u; component < 4u; component += 1u) {
        arena[params.range.w + 5u + component] = wp_subpred_nw_ww[component];
        arena[params.range.w + 9u + component] = wp_subpred_n_w[component];
        arena[params.range.w + 13u + component] = wp_subpred_ne[component];
    }
}

fn weighted_predict(n: i32, nw: i32, ne: i32, w: i32, nn: i32) -> WeightedPrediction {
    let n3 = n << 3u;
    let nw3 = nw << 3u;
    let ne3 = ne << 3u;
    let w3 = w << 3u;
    let nn3 = nn << 3u;
    var subpred = array<i32, 4>(
        w3 + ne3 - n3,
        n3 - ((wp_true_err_w + wp_true_err_n + wp_true_err_ne) * i32(params.wp_first.x) >> 5u),
        w3 - ((wp_true_err_w + wp_true_err_n + wp_true_err_nw) * i32(params.wp_first.y) >> 5u),
        n3 - ((wp_true_err_nw * i32(params.wp_first.z)
            + wp_true_err_n * i32(params.wp_first.w)
            + wp_true_err_ne * i32(params.wp_second.x)
            + (nn3 - n3) * i32(params.wp_second.y)
            + (nw3 - w3) * i32(params.wp_second.z)) >> 5u),
    );
    var weights: array<u32, 4>;
    var sum_weights = 0u;
    for (var component = 0u; component < 4u; component += 1u) {
        let error_sum = wp_subpred_nw_ww[component]
            + wp_subpred_n_w[component]
            + wp_subpred_ne[component];
        let shift = floor_log2((error_sum + 1u) >> 5u);
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
    let width = params.output.x;
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
        default: { return (6i * n - 2i * nn + 7i * w + ww + nee + 3i * ne + 8i) / 16i; }
    }
}

fn inverse_serial() {
    if params.range.x == 0u {
        predictor_reset();
    } else {
        predictor_load_state(params.range.x);
    }
    var cursor = params.range.x;
    while cursor < params.range.y {
        let x = cursor % params.output.x;
        let y = cursor / params.output.x;
        var w = 0i;
        if x != 0u {
            w = output_at(x - 1u, y);
        } else if y != 0u {
            w = output_at(x, y - 1u);
        }
        var n = w;
        var nw = w;
        if y != 0u {
            n = output_at(x, y - 1u);
            nw = n;
            if x != 0u {
                nw = output_at(x - 1u, y - 1u);
            }
        }
        var ne = n;
        if y != 0u && x + 1u < params.output.x {
            ne = output_at(x + 1u, y - 1u);
        }
        var nee = ne;
        if y != 0u && x + 2u < params.output.x {
            nee = output_at(x + 2u, y - 1u);
        }
        var nn = n;
        if y >= 2u {
            nn = output_at(x, y - 2u);
        }
        var ww = w;
        if x >= 2u {
            ww = output_at(x - 2u, y);
        }
        var weighted = WeightedPrediction(0i, 0i, array<i32, 4>(0i, 0i, 0i, 0i));
        if params.info.w == 6u {
            weighted = weighted_predict(n, nw, ne, w, nn);
        }
        let index = load_plane(params.indices, x, y);
        var value = palette_value(index);
        if index < i32(params.info.z) {
            value = add_wrap(
                value,
                predictor_value(params.info.w, weighted, n, w, nw, ne, nn, ww, nee),
            );
        }
        store_output(x, y, value);
        if params.info.w == 6u {
            weighted_record(weighted, value);
        }
        cursor += 1u;
    }
    predictor_store_state();
}

@compute @workgroup_size(wg_x, 1, 1)
fn inverse_palette(@builtin(global_invocation_id) id: vec3<u32>) {
    if params.info.w != 0u {
        if id.x == 0u && id.y == 0u {
            inverse_serial();
        }
        return;
    }
    if id.x >= params.output.x || id.y >= params.output.y {
        return;
    }
    let index = load_plane(params.indices, id.x, id.y);
    store_output(id.x, id.y, palette_value(index));
}
