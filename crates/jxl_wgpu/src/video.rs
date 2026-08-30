// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Generic pitch-linear image output contracts for video and image consumers.
//!
//! These contracts are selected on a concrete [`crate::WgpuFrameSession`]. The backend-neutral
//! `RenderBackend` trait currently asks for the `RenderPlan`'s ordinary output and therefore a
//! stock `JxlDecoder` call cannot request NV12 directly. A decoder-facing wrapper must expose its
//! concrete wgpu session (or the core trait must gain an output-request hook) before this API can
//! provide end-to-end decoder zero-copy.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;

use jxl_gpu_formats::{ImageLayout, PixelFormat};
use jxl_gpu_protocol::{ChangedRegions, OutputId, RgbColorEncoding, SubmissionToken};

use crate::upload::aligned_buffer_size;
use crate::{Error, MemoryPermit, Result};

/// Requested portable pitch-linear output format.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageOutputRequest {
    /// Encoding of the three F32 RGB planes consumed by the conversion.
    ///
    /// This must agree with the corresponding render-plan output descriptor. Requiring both
    /// sides to state the contract prevents an output request from silently reinterpreting the
    /// same source samples with different color metadata.
    pub source_encoding: RgbColorEncoding,
    pub format: PixelFormat,
}

impl ImageOutputRequest {
    pub const fn new(source_encoding: RgbColorEncoding, format: PixelFormat) -> Self {
        Self {
            source_encoding,
            format,
        }
    }
}

/// Cloneable GPU buffer ownership that keeps an optional byte reservation alive.
///
/// Clone this lease, rather than the raw [`wgpu::Buffer`], when ownership must remain covered by
/// the crate's byte budget. [`Self::as_wgpu_buffer`] deliberately makes raw access explicit:
/// `wgpu::Buffer` is itself cloneable, and a raw handle cloned through that borrow is outside this
/// lease's accounting. Safe Rust cannot keep a reservation coupled to a handle after callers clone
/// the raw wgpu value.
///
/// ```compile_fail
/// fn implicit_raw(lease: &jxl_wgpu::GpuBufferLease) -> &wgpu::Buffer {
///     lease
/// }
/// ```
#[derive(Clone)]
pub struct GpuBufferLease {
    buffer: Arc<wgpu::Buffer>,
    memory_permit: Option<MemoryPermit>,
}

impl GpuBufferLease {
    #[must_use]
    pub fn new(buffer: Arc<wgpu::Buffer>) -> Self {
        Self {
            buffer,
            memory_permit: None,
        }
    }

    #[must_use]
    pub fn with_memory_permit(buffer: Arc<wgpu::Buffer>, memory_permit: MemoryPermit) -> Self {
        Self {
            buffer,
            memory_permit: Some(memory_permit),
        }
    }

    /// Borrows the underlying buffer for wgpu interoperability.
    ///
    /// Cloning the returned raw handle does not clone this lease or its byte reservation. Keep a
    /// [`GpuBufferLease`] clone alive whenever the raw handle must remain covered by the budget.
    #[must_use]
    pub fn as_wgpu_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Size of the underlying wgpu allocation.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.buffer.size()
    }

    /// Usage flags of the underlying wgpu allocation.
    #[must_use]
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.buffer.usage()
    }

    /// Size of the shared reservation retained by this lease, or zero for externally managed
    /// buffers.
    ///
    /// Sibling outputs and clones from one submission report the same reservation rather than
    /// independent charges. Do not add their values together.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self.memory_permit.as_ref().map_or(0, MemoryPermit::bytes)
    }
}

impl std::fmt::Debug for GpuBufferLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuBufferLease")
            .field("size", &self.size())
            .field("usage", &self.usage())
            .field("reserved_bytes", &self.reserved_bytes())
            .finish_non_exhaustive()
    }
}

/// Generic pitch-linear image output stored entirely on the GPU.
///
/// This container is intentionally not cloneable. Clone [`GpuBufferLease`] explicitly when a
/// second owner must retain the tracked allocation.
///
/// ```compile_fail
/// fn require_clone<T: Clone>() {}
/// require_clone::<jxl_wgpu::GpuImageOutput>();
/// ```
#[derive(Debug)]
pub struct GpuImageOutput {
    pub id: OutputId,
    pub layout: ImageLayout,
    pub buffer: GpuBufferLease,
}

/// Non-blocking generic image result.
///
/// The frame is intentionally not cloneable; individual buffer leases express retained GPU
/// ownership without implying that a decoder frame slot was duplicated.
///
/// ```compile_fail
/// fn require_clone<T: Clone>() {}
/// require_clone::<jxl_wgpu::GpuImageFrame>();
/// ```
#[derive(Debug)]
pub struct GpuImageFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<GpuImageOutput>,
    pub changed: ChangedRegions,
}

/// One queue-submitted GPU image output whose codec validation has not completed.
///
/// This type is intentionally distinct from [`GpuImageOutput`]. Its layout and allocation are
/// usable immediately by commands submitted to the producer's device and queue, but its bytes are
/// not authoritative until the codec returns the corresponding validated frame. Cloning
/// [`Self::buffer`] retains the producer's byte-budget reservation; cloning the raw wgpu handle
/// obtained from [`GpuBufferLease::as_wgpu_buffer`] does not.
#[derive(Debug)]
pub struct UnvalidatedGpuImageOutput {
    pub id: OutputId,
    pub layout: ImageLayout,
    pub buffer: GpuBufferLease,
}

/// Queue-submitted GPU image made available before codec validation completes.
///
/// The absence of frame metadata and changed regions is deliberate: neither is authoritative at
/// this point. Same-queue display, readback, or custom GPU work can consume the outputs without a
/// host wait. If later validation fails, already-submitted consumer work cannot be rolled back and
/// every value derived from this frame must be discarded by the application.
///
/// This container is intentionally not cloneable. Clone individual [`GpuBufferLease`] values when
/// a consumer must retain an allocation under the shared memory budget.
#[derive(Debug)]
pub struct UnvalidatedGpuImageFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<UnvalidatedGpuImageOutput>,
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
