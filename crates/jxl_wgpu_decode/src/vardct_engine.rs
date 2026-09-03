//! Runtime-neutral GPU submission engine for the bounded standard VarDCT profile.
//!
//! The accepted codestream profile is intentionally bounded and authoritative: one still XYB or
//! JPEG-reconstruction YCbCr frame, independently bounded LF groups, GPU-decoded mixed
//! strategy/quantization/correlation metadata, and GPU-decoded single-pass AC coefficients for
//! every JPEG XL VarDCT strategy. No pixel, coefficient, transform, quantization, residual, or
//! entropy fallback runs on the CPU.

mod execution;
mod pipeline;
mod restoration;
mod source;
#[cfg(test)]
mod tests;
mod types;
mod window_plan;

pub use execution::{VarDctDecodeSession, VarDctPendingFrame};
pub use pipeline::VarDctSubmissionEngine;
pub use types::{VarDctDecodeError, VarDctDecodeMemoryStats};
