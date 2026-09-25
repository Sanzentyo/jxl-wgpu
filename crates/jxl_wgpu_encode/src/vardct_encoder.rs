//! Standard VarDCT still-image and animation encoder frontends.
//!
//! [`VarDctEncoder`] encodes one transform or a validated [`VarDctStrategyMap`] with resident coefficients, while
//! [`TiledVarDctEncoder`] uses regular DCT8 blocks across checked AC- and LF-group grids. Their
//! control syntax shares the deterministic frame assembler with the lossless Modular encoder.

mod ac;
mod bitstream;
mod color;
mod dispatch;
mod entropy;
mod group_order;
mod icc_input;
mod matrices;
mod modular_plane;
mod orders;
mod quantization;
mod raw_matrices;
mod saliency;
mod sequence;
mod strategy_map;
mod transforms;
mod types;

#[cfg(test)]
mod tests;

pub use color::VarDctColorTransform;
pub use dispatch::{TiledVarDctEncoder, VarDctBackend, VarDctEncoder, VarDctJob, VarDctSubmission};
pub use group_order::VarDctGroupOrder;
pub use icc_input::VarDctIccMemoryPlan;
pub use matrices::{VarDctDequantMatrices, VarDctMatrixEncoding, VarDctRawMatrix};
pub use modular_plane::{VarDctAlphaMemoryPlan, VarDctExtraChannelMemoryPlan};
pub use orders::VarDctCoefficientOrders;
pub use quantization::{VarDctConfig, VarDctHfMultiplier, VarDctQuantization};
pub use sequence::{
    VarDctAnimationDescriptor, VarDctAnimationSession, VarDctSequenceDescriptor,
    VarDctSequenceSession,
};
pub use strategy_map::{VarDctStrategyMap, VarDctTransform};
pub use types::{
    TiledVarDctGrid, VarDctKernelLayout, VarDctLfMetadata, VarDctMemoryPlan, VarDctStrategy,
    VarDctTransformMemoryPlan,
};
