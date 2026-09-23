/// Expands shared integer, predictor and implicit Palette WGSL fragments for codec consumers.
///
/// The caller supplies `wp_current_width`, `wp_coefficient`, `wp_max_weight`, `wp_true_error`,
/// `wp_subpred_error`, `wp_store_row`, and `predictor_error`. All sample-dependent arithmetic
/// and committed row/error state remain on GPU. The fragments introduce no bindings.
#[must_use]
pub fn modular_prediction_shader(source: &str) -> String {
    source
        .replace(
            "/*__JXL_MODULAR_PALETTE__*/",
            include_str!("../shaders/modular_palette.wgsl"),
        )
        .replace(
            "/*__JXL_MODULAR_PREDICT__*/",
            include_str!("../shaders/modular_predict.wgsl"),
        )
        .replace(
            "/*__JXL_MODULAR_INTEGER__*/",
            include_str!("../shaders/modular_int64.wgsl"),
        )
}
