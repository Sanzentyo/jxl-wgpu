//! Runtime-neutral GPU submission engine for the bounded standard VarDCT profile.
//!
//! The accepted codestream profile is intentionally bounded and authoritative: one still XYB or
//! JPEG-reconstruction YCbCr frame, independently bounded LF groups, GPU-decoded mixed
//! strategy/quantization/correlation metadata, and GPU-accumulated spectral/refinement AC coefficients for
//! every JPEG XL VarDCT strategy. No pixel, coefficient, transform, quantization, residual, or
//! entropy fallback runs on the CPU.

mod execution;
mod output;
mod pipeline;
mod restoration;
mod source;
mod staging;
#[cfg(test)]
mod tests;
mod types;
mod window_plan;

pub use pipeline::VarDctSubmissionEngine;
pub use staging::{VarDctDecodeSession, VarDctGlobalModularMemoryStats, VarDctPendingFrame};
pub use types::{VarDctDecodeError, VarDctDecodeMemoryStats, vardct_rgb8_format};
