//! Persistent generic Modular reconstruction state for bounded stream windows.
//!
//! This fragment is composed only into `lossless_gray8.wgsl`. The shared
//! `modular_reconstruct.wgsl` remains usable by whole-range VarDCT packet consumers.

fn load_modular_execution_state() {
    current_channel = modular_channel_for_decoded(consumer_decoded);
    if current_channel >= modular_entropy_channel_count() {
        predictor_prev_grad = 0i;
        if params.needs_self_correcting != 0u {
            wp_reset();
        }
        return;
    }
    modular_select_channel(current_channel);
    let channel_decoded = consumer_decoded - modular_current_channel_decoded_start();
    if channel_decoded == 0u {
        predictor_prev_grad = 0i;
        if params.needs_self_correcting != 0u {
            wp_reset();
        }
        return;
    }

    let base = params.entropy_state_offset;
    predictor_prev_grad = bitcast<i32>(reconstruction_load(base + 8u));
    if params.needs_self_correcting == 0u {
        return;
    }
    let width = modular_current_channel_width(params.width);
    wp_x = channel_decoded % width;
    wp_y = channel_decoded / width;
    wp_true_err_w = bitcast<i32>(reconstruction_load(base + 9u));
    wp_true_err_nw = bitcast<i32>(reconstruction_load(base + 10u));
    wp_true_err_n = bitcast<i32>(reconstruction_load(base + 11u));
    wp_true_err_ne = bitcast<i32>(reconstruction_load(base + 12u));
    for (var component = 0u; component < 4u; component += 1u) {
        wp_subpred_nw_ww[component] = reconstruction_load(base + 13u + component);
        wp_subpred_n_w[component] = reconstruction_load(base + 17u + component);
        wp_subpred_ne[component] = reconstruction_load(base + 21u + component);
    }
}

fn save_modular_execution_state() {
    let base = params.entropy_state_offset;
    reconstruction_store(base + 8u, bitcast<u32>(predictor_prev_grad));
    if params.needs_self_correcting == 0u {
        return;
    }
    reconstruction_store(base + 9u, bitcast<u32>(wp_true_err_w));
    reconstruction_store(base + 10u, bitcast<u32>(wp_true_err_nw));
    reconstruction_store(base + 11u, bitcast<u32>(wp_true_err_n));
    reconstruction_store(base + 12u, bitcast<u32>(wp_true_err_ne));
    for (var component = 0u; component < 4u; component += 1u) {
        reconstruction_store(base + 13u + component, wp_subpred_nw_ww[component]);
        reconstruction_store(base + 17u + component, wp_subpred_n_w[component]);
        reconstruction_store(base + 21u + component, wp_subpred_ne[component]);
    }
}
