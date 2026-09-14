//! Standard VarDCT still-image encoder frontend.
//!
//! [`VarDctEncoder`] encodes one complete transform with GPU-resident coefficients, while
//! [`TiledVarDctEncoder`] uses regular DCT8 blocks across checked AC- and LF-group grids. Their
//! control syntax shares the deterministic frame assembler with the lossless Modular encoder.

mod ac;
mod bitstream;
mod dispatch;
mod entropy;
mod single;
mod types;

#[cfg(test)]
mod tests;

pub use dispatch::{TiledVarDctEncoder, VarDctBackend, VarDctEncoder, VarDctJob, VarDctSubmission};
pub use types::{
    TiledVarDctGrid, VarDctColorEncoding, VarDctKernelLayout, VarDctLfMetadata, VarDctMemoryPlan,
    VarDctStrategy, VarDctTransformMemoryPlan,
};
