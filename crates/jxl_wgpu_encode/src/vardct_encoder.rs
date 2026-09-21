//! Standard VarDCT still-image encoder frontend.
//!
//! [`VarDctEncoder`] encodes one transform or a validated [`VarDctStrategyMap`] with resident coefficients, while
//! [`TiledVarDctEncoder`] uses regular DCT8 blocks across checked AC- and LF-group grids. Their
//! control syntax shares the deterministic frame assembler with the lossless Modular encoder.

mod ac;
mod bitstream;
mod dispatch;
mod entropy;
mod orders;
mod quantization;
mod strategy_map;
mod transforms;
mod types;

#[cfg(test)]
mod tests;

pub use dispatch::{TiledVarDctEncoder, VarDctBackend, VarDctEncoder, VarDctJob, VarDctSubmission};
pub use orders::VarDctCoefficientOrders;
pub use quantization::{VarDctConfig, VarDctHfMultiplier, VarDctQuantization};
pub use strategy_map::{VarDctStrategyMap, VarDctTransform};
pub use types::{
    TiledVarDctGrid, VarDctColorEncoding, VarDctKernelLayout, VarDctLfMetadata, VarDctMemoryPlan,
    VarDctStrategy, VarDctTransformMemoryPlan,
};
