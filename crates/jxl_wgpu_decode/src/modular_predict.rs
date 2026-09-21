//! Shared predictor arithmetic for entropy reconstruction and inverse Palette.

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(crate) fn shader(source: &str) -> String {
    jxl_wgpu::modular_prediction_shader(source)
}
