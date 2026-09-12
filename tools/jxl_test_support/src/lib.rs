//! Development-only support shared by integration tests, private GPU tests, and fixture tools.
//! Production codec dependencies never select this crate or its CPU reference decoders.

pub mod corpus;
pub mod fixtures;
pub mod gpu;
pub mod offline;
pub mod oracles;

/// Resolve the checked-in decoder corpus and native oracle sources from this workspace tool.
pub fn decoder_directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/jxl_wgpu_decode")
}
