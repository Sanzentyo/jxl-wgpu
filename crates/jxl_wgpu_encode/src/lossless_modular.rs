// The JPEG XL header and fast-lossless control-plane construction in this module is derived
// from the permissively licensed zune-jpegxl 0.5.2 encoder. See `THIRD_PARTY.md` and
// `LICENSES/zune-jpegxl-MIT.txt` in this crate.

mod color;
mod dispatch;
mod grid;
mod icc;
mod memory;
mod serializer;
mod source;
mod streaming;
#[cfg(test)]
mod tests;
mod types;

pub use color::LosslessModularColorOptions;
pub use dispatch::LosslessModularBackend;
pub use grid::{LosslessModularGroup, LosslessModularGroupGrid};
pub use memory::{
    LosslessModularInFlightMemory, LosslessModularMemoryLimits, LosslessModularMemoryPlan,
};
pub use serializer::{
    LosslessModularAnimationDescriptor, LosslessModularAnimationSession, LosslessModularEncoder,
    LosslessModularSubmission,
};
pub use streaming::LosslessModularJob;
pub use types::{
    AlphaAssociation, LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularConfig,
    LosslessModularFormat, LosslessModularGroupSize, LosslessModularTreeMode,
};
