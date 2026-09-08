//! Shared predictor arithmetic for entropy reconstruction and inverse Palette.

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(crate) fn shader(source: &str) -> String {
    source
        .replace(
            "/*__JXL_MODULAR_PREDICT__*/",
            include_str!("modular_predict.wgsl"),
        )
        .replace(
            "/*__JXL_MODULAR_INTEGER__*/",
            include_str!("modular_int64.wgsl"),
        )
}
