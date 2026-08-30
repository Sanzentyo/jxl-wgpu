// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_gpu_protocol::{BackendError, PlaneId};

use crate::session::SubmissionMode;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no compatible wgpu adapter was found")]
    NoAdapter,
    #[error("failed to request a wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("render plan is invalid: {0}")]
    InvalidPlan(#[from] jxl_gpu_protocol::PlanError),
    #[error("operation is not implemented by the portable backend: {0}")]
    Unsupported(String),
    #[error("missing source payload for plane {0:?}")]
    MissingPlane(PlaneId),
    #[error("invalid plane payload: {0}")]
    InvalidPayload(String),
    #[error("GPU buffer size overflow")]
    BufferSizeOverflow,
    #[error("generic image readback requires at least one output")]
    ImageReadbackEmpty,
    #[error("generic image readback output {output} does not have COPY_SRC usage")]
    ImageReadbackSourceUsage { output: usize },
    #[error(
        "generic image readback output {output} buffer is too small: requires {required} bytes, has {actual}"
    )]
    ImageReadbackSourceSize {
        output: usize,
        required: u64,
        actual: u64,
    },
    #[error(
        "generic image readback staging requires {required} bytes, exceeding transient limit {limit}"
    )]
    ImageReadbackTransientLimit { required: u64, limit: u64 },
    #[error(
        "generic image readback staging requires {required} bytes, exceeding device buffer limit {limit}"
    )]
    ImageReadbackDeviceLimit { required: u64, limit: u64 },
    #[error("display texture extent {width}x{height} exceeds the device 2D texture limit {limit}")]
    DisplayTextureExtent { width: u32, height: u32, limit: u32 },
    #[error(
        "texture copy row pitch {bytes_per_row} is not a multiple of the required alignment {required_alignment}"
    )]
    TextureCopyRowAlignment {
        bytes_per_row: u32,
        required_alignment: u32,
    },
    #[error("pitch-linear image layout is invalid: {0}")]
    ImageLayout(#[from] jxl_gpu_formats::LayoutError),
    #[error("GPU resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    #[error("GPU polling failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("GPU execution failed: {0}")]
    Execution(String),
    #[error("submission token {0} is not pending")]
    UnknownSubmission(u64),
    #[error(
        "submission token {token} belongs to {actual:?} mode, but {expected:?} mode was requested"
    )]
    SubmissionModeMismatch {
        token: u64,
        expected: SubmissionMode,
        actual: SubmissionMode,
    },
    #[error("autotune profile I/O failed: {0}")]
    ProfileIo(#[from] std::io::Error),
    #[error("autotune profile is invalid: {0}")]
    ProfileFormat(#[from] serde_json::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for BackendError {
    fn from(error: Error) -> Self {
        match error {
            Error::Unsupported(message) => Self::Unsupported(message),
            Error::InvalidPayload(message) => Self::InvalidPayload(message),
            Error::MissingPlane(plane) => {
                Self::InvalidPayload(format!("missing source payload for plane {plane:?}"))
            }
            Error::ResourceLimit(message) => Self::ResourceLimit(message),
            Error::BufferSizeOverflow => Self::ResourceLimit("GPU buffer size overflow".into()),
            Error::ImageReadbackEmpty => {
                Self::InvalidPayload("generic image readback frame has no outputs".into())
            }
            Error::ImageReadbackSourceUsage { output } => Self::InvalidPayload(format!(
                "generic image readback output {output} does not have COPY_SRC usage"
            )),
            Error::ImageReadbackSourceSize {
                output,
                required,
                actual,
            } => Self::InvalidPayload(format!(
                "generic image readback output {output} requires {required} bytes, has {actual}"
            )),
            Error::ImageReadbackTransientLimit { required, limit }
            | Error::ImageReadbackDeviceLimit { required, limit } => Self::ResourceLimit(format!(
                "generic image readback requires {required} bytes, limit is {limit}"
            )),
            Error::ImageLayout(error) => Self::InvalidPayload(error.to_string()),
            other => Self::Execution(other.to_string()),
        }
    }
}
