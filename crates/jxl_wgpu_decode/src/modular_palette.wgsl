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

/*__JXL_MODULAR_PREDICT__*/

@group(0) @binding(0) var<storage, read_write> arena: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

const STATE_WORDS: u32 = 20u;

/*__JXL_MODULAR_PALETTE__*/

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
    let implicit_index = select(index - i32(palette_size), index, index < 0i);
    return mp_implicit_palette_value(implicit_index, channel, params.range.z);
}

fn wp_max_weight(component: u32) -> u32 {
    switch component {
        case 0u: { return params.wp_second.w; }
        case 1u: { return params.wp_third.x; }
        case 2u: { return params.wp_third.y; }
        default: { return params.wp_third.z; }
    }
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

fn inverse_serial() {
    if params.range.x == 0u {
        wp_reset();
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
        var weighted = WeightedPrediction();
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

fn wp_current_width() -> u32 { return params.output.x; }
fn predictor_error() {}
fn wp_coefficient(index: u32) -> u32 {
    if index < 4u { return params.wp_first[index]; }
    return params.wp_second[index - 4u];
}
