/*__JXL_MODULAR_PREDICT__*/
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
var<private> case_index: u32;
fn wp_coefficient(index: u32) -> u32 { return input[case_index * 40u + 24u + index]; }
fn wp_max_weight(index: u32) -> u32 { return input[case_index * 40u + 31u + index]; }
fn wp_current_width() -> u32 { return 2u; }
fn wp_true_error(index: u32) -> i32 { return 0i; }
fn wp_subpred_error(index: u32, component: u32) -> u32 { return 0u; }
fn wp_store_row(index: u32, true_error: i32, errors: array<u32, 4>) {
    output[case_index * 30u + 25u] = bitcast<u32>(true_error);
    for (var c=0u; c<4u; c++) { output[case_index * 30u + 26u + c] = errors[c]; }
}
fn predictor_error() {}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    case_index = id.x;
    if case_index >= arrayLength(&input) / 40u { return; }
    let base = case_index * 40u;
    wp_reset();
    wp_true_err_w = bitcast<i32>(input[base + 8u]);
    wp_true_err_nw = bitcast<i32>(input[base + 9u]);
    wp_true_err_n = bitcast<i32>(input[base + 10u]);
    wp_true_err_ne = bitcast<i32>(input[base + 11u]);
    for (var c=0u; c<4u; c++) {
        wp_subpred_nw_ww[c] = input[base + 12u + c];
        wp_subpred_n_w[c] = input[base + 16u + c];
        wp_subpred_ne[c] = input[base + 20u + c];
    }
    let n = bitcast<i32>(input[base]); let w = bitcast<i32>(input[base + 1u]);
    let nw = bitcast<i32>(input[base + 2u]); let ne = bitcast<i32>(input[base + 3u]);
    let nn = bitcast<i32>(input[base + 4u]); let ww = bitcast<i32>(input[base + 5u]);
    let nee = bitcast<i32>(input[base + 6u]);
    let prediction = weighted_predict(n, nw, ne, w, nn);
    output[case_index * 30u] = prediction.prediction.x;
    output[case_index * 30u + 1u] = prediction.prediction.y;
    output[case_index * 30u + 2u] = bitcast<u32>(prediction.max_error);
    for (var c=0u; c<4u; c++) {
        output[case_index * 30u + 3u + c * 2u] = prediction.subpred[c].x;
        output[case_index * 30u + 4u + c * 2u] = prediction.subpred[c].y;
    }
    for (var predictor=0u; predictor<14u; predictor++) {
        output[case_index * 30u + 11u + predictor] = bitcast<u32>(predictor_value(predictor, prediction, n,w,nw,ne,nn,ww,nee));
    }
    weighted_record(prediction, bitcast<i32>(input[base + 7u]));
}
