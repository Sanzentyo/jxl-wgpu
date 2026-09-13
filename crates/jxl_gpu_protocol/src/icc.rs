//! Bounded ICC metadata and backend-neutral matrix/TRC transforms.
//!
//! Profiles keep their original bytes, signed fixed-point colorants, and independent channel
//! curves. This module never reads image samples. GPU backends lower the resulting transform
//! once and execute the curves and matrix on resident pixels.
//!
//! The initial execution model covers ICC v2/v4 RGB matrix and XYZ monochrome profiles with
//! media-relative colorimetric intent. LUT, Lab, absolute/perceptual/saturation policies and
//! other device spaces return structured unsupported errors, without substituting a profile.

mod curve;
mod profile;
mod transform;

pub use curve::{IccCurve, IccCurveKind, IccInverseDirection};
pub use profile::{IccHeader, IccProfile, IccTag};
pub use transform::{IccMatrixTrc, IccTransform};

/// An ICC four-byte signature, preserved even for unknown tags and profile classes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IccSignature(pub [u8; 4]);

impl std::fmt::Debug for IccSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", String::from_utf8_lossy(&self.0))
    }
}

impl std::fmt::Display for IccSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", String::from_utf8_lossy(&self.0))
    }
}

/// ICC header rendering-intent values. Transform selection always takes an explicit intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum IccRenderingIntent {
    Perceptual = 0,
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

/// Direction used when selecting profile transform tags (ICC.1:2022, 8.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IccDirection {
    DeviceToPcs,
    PcsToDevice,
}

/// Bounds applied before metadata allocation or iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IccLimits {
    pub max_profile_bytes: u64,
    pub max_tags: u32,
    /// Maximum entries in each selected sampled curve; parameterized curves need no table.
    pub max_curve_samples: u32,
}

impl Default for IccLimits {
    fn default() -> Self {
        Self {
            max_profile_bytes: 16 << 20,
            max_tags: 4096,
            max_curve_samples: 1 << 20,
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum IccError {
    #[error("ICC {field} needs bytes through {required}, available {available}")]
    Truncated {
        field: &'static str,
        required: u64,
        available: u64,
    },
    #[error("ICC {resource} requires {required}, limit {limit}")]
    Limit {
        resource: &'static str,
        required: u64,
        limit: u64,
    },
    #[error("invalid ICC {field} at byte {offset}")]
    Invalid { field: &'static str, offset: u64 },
    #[error("ICC profile version {version:#010x} is unsupported")]
    Version { version: u32 },
    #[error("duplicate ICC tag {tag}")]
    DuplicateTag { tag: IccSignature },
    #[error("ICC tags {first} and {second} overlap without sharing one complete element")]
    TagOverlap {
        first: IccSignature,
        second: IccSignature,
    },
    #[error("ICC transform needs missing tag {tag}")]
    MissingTag { tag: IccSignature },
    #[error("ICC tag {tag} has unsupported type {kind}")]
    TagType {
        tag: IccSignature,
        kind: IccSignature,
    },
    #[error("ICC {field} {signature} is unsupported for matrix/TRC execution")]
    Unsupported {
        field: &'static str,
        signature: IccSignature,
    },
    #[error("ICC rendering intent {intent:?} is not implemented")]
    RenderingIntent { intent: IccRenderingIntent },
    #[error("ICC transform selects {tag}; its execution is not implemented")]
    TransformTag { tag: IccSignature },
    #[error("ICC tag {tag} has invalid curve parameters")]
    CurveParameters { tag: IccSignature },
    #[error("ICC tag {tag} has unsupported parametric function {function}")]
    CurveFunction { tag: IccSignature, function: u16 },
    #[error("ICC curve cannot be inverted: {reason}")]
    CurveInverse { reason: &'static str },
    #[error("ICC colorant matrix is singular or non-finite")]
    Matrix,
}

#[cfg(test)]
mod tests;
