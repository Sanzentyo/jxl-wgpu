//! Original JPEG bytes from validated resident coefficients, without a CPU entropy codec.

use jxl_wgpu::GpuBufferLease;

use super::jpeg::JpegCoefficientLimits;

mod framing;
pub(super) mod gpu;
mod runtime;
mod scan;

pub use runtime::{JpegReconstructionPending, JpegReconstructionSession};

/// Independent host planning, GPU counting and final JPEG byte limits.
#[derive(Clone, Copy, Debug)]
pub struct JpegReconstructionLimits {
    pub coefficients: JpegCoefficientLimits,
    pub metadata: jxl_gpu_bitstream::metadata::MetadataLimits,
    pub max_tasks: u64,
    pub max_plan_bytes: u64,
    pub max_raw_scan_bytes: u64,
    pub max_output_bytes: u64,
}

impl Default for JpegReconstructionLimits {
    fn default() -> Self {
        Self {
            coefficients: JpegCoefficientLimits::default(),
            metadata: Default::default(),
            max_tasks: 1 << 20,
            max_plan_bytes: 64 << 20,
            max_raw_scan_bytes: 16 << 20,
            max_output_bytes: 64 << 20,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JpegReconstructionError {
    #[error("invalid JPEG reconstruction binding: {0}")]
    Invalid(&'static str),
    #[error("unsupported JPEG reconstruction: {0}")]
    Unsupported(&'static str),
    #[error("JPEG reconstruction {resource} needs {required}, limit {limit}")]
    Limit {
        resource: &'static str,
        required: u64,
        limit: u64,
    },
    #[error("JPEG reconstruction host allocation failed")]
    Allocation,
    #[error("JPEG reconstruction GPU {stage} status: {status:?}")]
    GpuStatus {
        stage: &'static str,
        status: [u32; 4],
    },
}

/// Validated original JPEG bytes. Storage is rounded to four bytes; `byte_len` excludes padding.
#[derive(Clone, Debug)]
pub struct GpuJpegFrame {
    buffer: GpuBufferLease,
    byte_len: u64,
}

impl GpuJpegFrame {
    /// Retain this lease through any GPU use or explicit readback.
    #[must_use]
    pub const fn buffer(&self) -> &GpuBufferLease {
        &self.buffer
    }

    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

type Result<T> = std::result::Result<T, JpegReconstructionError>;
use JpegReconstructionError as Error;

fn check(resource: &'static str, required: u64, limit: u64) -> Result<()> {
    if required > limit {
        Err(Error::Limit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn allocate<T>(count: usize) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error::Allocation)?;
    Ok(result)
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(Error::Invalid("size overflow"))
}

fn storage_bytes(bytes: u64) -> Result<u64> {
    Ok(add(bytes, 3)? & !3)
}
