// The JPEG XL header and fast-lossless control-plane construction in this module is derived
// from the permissively licensed zune-jpegxl 0.5.2 encoder. See `THIRD_PARTY.md` and
// `LICENSES/zune-jpegxl-MIT.txt` in this crate.

mod color;
mod dispatch;
mod entropy;
mod grid;
mod icc;
mod local_transforms;
mod lz77;
mod memory;
mod palette;
mod predictor;
mod rct;
mod serializer;
mod source;
mod squeeze;
mod streaming;
#[cfg(test)]
mod tests;
mod transform;
mod types;
mod upload;

pub use dispatch::LosslessModularBackend;
pub use entropy::LosslessModularEntropyCoding;
pub use grid::{LosslessModularGroup, LosslessModularGroupGrid};
pub use local_transforms::{LosslessModularLocalTransforms, LosslessModularTransform};
pub use lz77::LosslessModularLz77;
pub use memory::{
    LosslessModularInFlightMemory, LosslessModularMemoryLimits, LosslessModularMemoryPlan,
};
pub use palette::LosslessModularPalette;
pub use predictor::{LosslessModularPredictor, LosslessModularWeightedPredictor};
pub use rct::{LosslessModularColorTransform, LosslessModularRctType};
pub use serializer::{
    LosslessModularAnimationDescriptor, LosslessModularAnimationSession, LosslessModularEncoder,
    LosslessModularSequenceDescriptor, LosslessModularSequenceSession, LosslessModularSubmission,
};
pub use squeeze::{LosslessModularSqueeze, LosslessModularSqueezeStep};
pub use streaming::LosslessModularJob;
pub use types::{
    AlphaAssociation, LOSSLESS_MODULAR_GROUP_DIMENSION, LosslessModularConfig,
    LosslessModularFormat, LosslessModularGroupSize, LosslessModularTreeMode,
};
