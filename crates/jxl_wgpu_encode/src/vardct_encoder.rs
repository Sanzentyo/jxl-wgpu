//! Standard VarDCT still-image encoder frontend.
//!
//! The bounded frontend encodes one strategy over a small image extent, while
//! [`TiledVarDctEncoder`] uses regular DCT8 blocks across checked AC- and LF-group grids. Their
//! control-plane syntax is kept separate from the lossless Modular encoder so neither coding mode
//! becomes a compatibility layer for the other.

mod bitstream;
mod dispatch;
mod entropy;
mod types;

#[cfg(test)]
mod tests;

pub use dispatch::{TiledVarDctEncoder, VarDctBackend, VarDctEncoder, VarDctJob, VarDctSubmission};
pub use types::{
    TiledVarDctGrid, VarDctColorEncoding, VarDctKernelLayout, VarDctLfMetadata, VarDctMemoryPlan,
    VarDctStrategy,
};
