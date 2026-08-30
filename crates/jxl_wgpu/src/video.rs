// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Generic pitch-linear image output contracts for video and image consumers.
//!
//! These contracts are selected on a concrete [`crate::WgpuFrameSession`]. The backend-neutral
//! `JxlAccelerator` trait currently asks for the `RenderPlan`'s ordinary output and therefore a
//! stock `JxlDecoder` call cannot request NV12 directly. A decoder-facing wrapper must expose its
//! concrete wgpu session (or the core trait must gain an output-request hook) before this API can
//! provide end-to-end decoder zero-copy.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use jxl_gpu_formats::{ImageLayout, PixelFormat};
use jxl_gpu_protocol::{ChangedRegions, OutputId, SubmissionToken};

use crate::upload::aligned_buffer_size;
use crate::{Error, Result};

/// Requested portable pitch-linear output format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageOutputRequest {
    pub format: PixelFormat,
}

impl ImageOutputRequest {
    pub const fn new(format: PixelFormat) -> Self {
        Self { format }
    }
}

/// Generic pitch-linear image output stored entirely on the GPU.
#[derive(Clone, Debug)]
pub struct GpuImageOutput {
    pub id: OutputId,
    pub layout: ImageLayout,
    pub buffer: Arc<wgpu::Buffer>,
}

/// Non-blocking generic image result.
#[derive(Clone, Debug)]
pub struct GpuImageFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<GpuImageOutput>,
    pub changed: ChangedRegions,
}

/// CPU-visible generic image output. Consult `layout` rather than assuming contiguous planes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuImageOutput {
    pub id: OutputId,
    pub layout: ImageLayout,
    pub bytes: Vec<u8>,
}

/// Completed CPU-readback result for a generic image submission.
#[derive(Clone, Debug)]
pub struct CpuImageFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<CpuImageOutput>,
    pub changed: ChangedRegions,
}

#[derive(Debug)]
pub(crate) struct PackedImageOutput {
    pub id: OutputId,
    pub layout: ImageLayout,
    pub buffer: Arc<wgpu::Buffer>,
}

#[derive(Debug)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) struct ImageReadbackRequest {
    id: OutputId,
    layout: ImageLayout,
    buffer: Arc<wgpu::Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    mapping: mpsc::Receiver<std::result::Result<(), wgpu::BufferAsyncError>>,
}

pub(crate) fn stage_image_output(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    output: &PackedImageOutput,
    direct: bool,
) -> Result<ImageReadbackRequest> {
    let buffer = if direct {
        Arc::clone(&output.buffer)
    } else {
        let padded_size = aligned_buffer_size(output.layout.logical_size)?;
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu generic image readback"),
            size: padded_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &buffer, 0, padded_size);
        buffer
    };
    #[cfg(not(target_arch = "wasm32"))]
    let mapping = {
        let (sender, receiver) = mpsc::sync_channel(1);
        encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        receiver
    };
    Ok(ImageReadbackRequest {
        id: output.id,
        layout: output.layout.clone(),
        buffer,
        #[cfg(not(target_arch = "wasm32"))]
        mapping,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn resolve_image_outputs(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    requests: Vec<ImageReadbackRequest>,
) -> Result<Vec<CpuImageOutput>> {
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;

    requests
        .into_iter()
        .map(|request| {
            request.mapping.recv().map_err(|_| {
                Error::Execution(format!(
                    "mapping callback was dropped for generic image output {:?}",
                    request.id
                ))
            })??;
            let byte_len = usize::try_from(request.layout.logical_size)
                .map_err(|_| Error::BufferSizeOverflow)?;
            let slice = request.buffer.slice(..);
            let mapped = slice.get_mapped_range().map_err(|error| {
                Error::Execution(format!("mapped generic image range is invalid: {error}"))
            })?;
            let bytes = mapped
                .get(..byte_len)
                .ok_or_else(|| {
                    Error::Execution(
                        "mapped generic image output was shorter than requested".into(),
                    )
                })?
                .to_vec();
            drop(mapped);
            request.buffer.unmap();
            Ok(CpuImageOutput {
                id: request.id,
                layout: request.layout,
                bytes,
            })
        })
        .collect()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn resolve_image_outputs(
    _device: &wgpu::Device,
    _submission: wgpu::SubmissionIndex,
    _requests: Vec<ImageReadbackRequest>,
) -> Result<Vec<CpuImageOutput>> {
    Err(Error::Unsupported(
        "generic image CPU readback cannot synchronously block browser WebGPU".into(),
    ))
}
