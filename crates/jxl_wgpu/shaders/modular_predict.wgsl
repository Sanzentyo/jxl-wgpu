//! Shared JPEG XL predictors; callers supply coefficients and 32-bit row/state access.
/*__JXL_MODULAR_INTEGER__*/

struct WeightedPrediction {
    prediction: ModularI64,
    max_error: i32,
    subpred: array<ModularI64, 4>,
};

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

fn gradient_i32(north: i32, west: i32, north_west: i32) -> i32 {
    return mi_gradient(north, west, north_west);
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

fn weighted_predict(n: i32, nw: i32, ne: i32, w: i32, nn: i32) -> WeightedPrediction {
    let n3 = mi_shl(mi_from_i32(n), 3u);
    let nw3 = mi_shl(mi_from_i32(nw), 3u);
    let ne3 = mi_shl(mi_from_i32(ne), 3u);
    let w3 = mi_shl(mi_from_i32(w), 3u);
    let nn3 = mi_shl(mi_from_i32(nn), 3u);
    let error_n = mi_from_i32(wp_true_err_n);
    let error_nw = mi_from_i32(wp_true_err_nw);
    let error_ne = mi_from_i32(wp_true_err_ne);
    let error_wn = mi_add(mi_from_i32(wp_true_err_w), error_n);
    var correction = mi_mul_u32(error_nw, wp_coefficient(2u));
    correction = mi_add(correction, mi_mul_u32(error_n, wp_coefficient(3u)));
    correction = mi_add(correction, mi_mul_u32(error_ne, wp_coefficient(4u)));
    correction = mi_add(correction, mi_mul_u32(mi_sub(nn3, n3), wp_coefficient(5u)));
    correction = mi_add(correction, mi_mul_u32(mi_sub(nw3, w3), wp_coefficient(6u)));
    var subpred = array<ModularI64, 4>(
        mi_sub(mi_add(w3, ne3), n3),
        mi_sub(n3, mi_sar(mi_mul_u32(mi_add(error_wn, error_ne), wp_coefficient(0u)), 5u)),
        mi_sub(w3, mi_sar(mi_mul_u32(mi_add(error_wn, error_nw), wp_coefficient(1u)), 5u)),
        mi_sub(n3, mi_sar(correction, 5u)),
    );
    var weights: array<u32, 4>;
    var sum_weights = 0u;
    for (var component = 0u; component < 4u; component += 1u) {
        let error_sum = wp_subpred_nw_ww[component]
            + wp_subpred_n_w[component]
            + wp_subpred_ne[component];
        let shifted_error = (error_sum + 1u) >> 5u;
        let shift = select(floor_log2(shifted_error), 27u, error_sum == 0xffffffffu);
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
    var weighted_sum = mi_from_i32(i32(sum_weights >> 1u) - 1i);
    for (var component = 0u; component < 4u; component += 1u) {
        weighted_sum = mi_add(weighted_sum, mi_mul_u32(subpred[component], weights[component]));
    }
    var prediction = mi_sar(mi_mul_u32(weighted_sum, (1u << 24u) / sum_weights), 24u);
    if ((wp_true_err_n ^ wp_true_err_w) | (wp_true_err_n ^ wp_true_err_nw)) <= 0i {
        prediction = mi_max(mi_min(prediction, mi_max(n3, mi_max(w3, ne3))), mi_min(n3, mi_min(w3, ne3)));
    }
    var max_error = wp_true_err_w;
    if abs_diff_i32(wp_true_err_n, 0i) > abs_diff_i32(max_error, 0i) {
        max_error = wp_true_err_n;
    }
    if abs_diff_i32(wp_true_err_nw, 0i) > abs_diff_i32(max_error, 0i) {
        max_error = wp_true_err_nw;
    }
    if abs_diff_i32(wp_true_err_ne, 0i) > abs_diff_i32(max_error, 0i) {
        max_error = wp_true_err_ne;
    }
    return WeightedPrediction(prediction, max_error, subpred);
}

fn weighted_record(prediction: WeightedPrediction, sample: i32) {
    let width = wp_current_width();
    let sample3 = mi_shl(mi_from_i32(sample), 3u);
    // The reference keeps wide predictions, but commits true errors and error sums as 32 bits.
    let true_error = bitcast<i32>(mi_sub(prediction.prediction, sample3).x);
    var errors: array<u32, 4>;
    for (var component = 0u; component < 4u; component += 1u) {
        errors[component] = mi_sar(mi_add(mi_abs(mi_sub(prediction.subpred[component], sample3)), mi_from_i32(3i)), 3u).x;
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
        case 3u: { return mi_average(w, n); }
        case 4u: {
            if abs_diff_i32(n, nw) < abs_diff_i32(w, nw) {
                return w;
            }
            return n;
        }
        case 5u: { return gradient_i32(n, w, nw); }
        case 6u: { return bitcast<i32>(mi_sar(mi_add(weighted.prediction, mi_from_i32(3i)), 3u).x); }
        case 7u: { return ne; }
        case 8u: { return nw; }
        case 9u: { return ww; }
        case 10u: { return mi_average(w, nw); }
        case 11u: { return mi_average(n, nw); }
        case 12u: { return mi_average(n, ne); }
        case 13u: {
            var sum = mi_sub(mi_mul_u32(mi_from_i32(n), 6u), mi_mul_u32(mi_from_i32(nn), 2u));
            sum = mi_add(sum, mi_mul_u32(mi_from_i32(w), 7u));
            sum = mi_add(sum, mi_add(mi_from_i32(ww), mi_from_i32(nee)));
            sum = mi_add(sum, mi_mul_u32(mi_from_i32(ne), 3u));
            return bitcast<i32>(mi_div_pow2(mi_add(sum, mi_from_i32(8i)), 4u).x);
        }
        default: {
            predictor_error();
            return 0i;
        }
    }
}
