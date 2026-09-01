// The JPEG XL header and fast-lossless control-plane construction in this module is derived
// from the permissively licensed zune-jpegxl 0.5.2 encoder. See `THIRD_PARTY.md` and
// `LICENSES/zune-jpegxl-MIT.txt` in this crate.

mod dispatch;
mod grid;
mod memory;
mod serializer;
mod streaming;
#[cfg(test)]
mod tests;
mod types;

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
pub use types::{LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularFormat, LosslessModularTreeMode};
