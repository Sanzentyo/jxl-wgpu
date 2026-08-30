// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Non-blocking conversion of decoder-owned GPU buffers into display textures.
//!
//! [`DisplayPipeline::encode_rgb`], [`DisplayPipeline::encode_image`], and
//! [`DisplayPipeline::encode_unvalidated_image`] only append work to a caller-owned command
//! encoder. Their `submit_*` counterparts submit that work to the exact queue from which the
//! pipeline was constructed. None of these display paths waits for the host. Once a caller owns a
//! source buffer lease, queue ordering makes an earlier producer submission a valid dependency, and
//! the returned texture can be sampled, rendered, or copied by later commands on that queue.
//!
//! This module consumes portable pitch-linear storage buffers. Native multi-plane textures and
//! platform-specific external-memory layouts are deliberately outside this API contract.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard};

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification, ImageLayout,
    NumericFormatClass, Packed422Order, PixelFormat, PixelFormatClass, RgbChannelOrder, RgbStorage,
    TransferFunction, WgslNumericCapability, YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{Extent2d, OutputLayout, SampleType};
use wgpu::util::DeviceExt;

use crate::context::WgpuBackend;
use crate::session::GpuOutputBuffer;
use crate::video::{GpuImageOutput, UnvalidatedGpuImageOutput};
use crate::{Error, Result};

const WORKGROUP_SIZE: u32 = 16;
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

/// Texture properties used by a display conversion.
///
/// `Rgba8Unorm` is the portable storage-texture format used by this backend. The pipeline always
/// adds `STORAGE_BINDING`, `TEXTURE_BINDING`, `RENDER_ATTACHMENT`, `COPY_SRC`, and `COPY_DST`; use
/// `additional_usage` only for an application-specific usage bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayTextureDescriptor {
    pub format: wgpu::TextureFormat,
    pub additional_usage: wgpu::TextureUsages,
}

impl Default for DisplayTextureDescriptor {
    fn default() -> Self {
        Self {
            format: wgpu::TextureFormat::Rgba8Unorm,
            additional_usage: wgpu::TextureUsages::empty(),
        }
    }
}

/// Color encoding of a texture returned by [`DisplayPipeline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplayColorEncoding {
    /// Full-range linear-light RGB using BT.709/sRGB primaries.
    LinearBt709,
}

/// A GPU-resident texture ready for sampling, rendering, or copying.
///
/// The handles are `Arc`-owned, so this value can outlive both the frame session and the
/// [`DisplayPipeline`] that created it. Dropping the source buffer after submission is safe: wgpu
/// retains every resource referenced by an in-flight command buffer.
#[derive(Clone, Debug)]
pub struct DisplayTexture {
    pub extent: Extent2d,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    /// The explicit interpretation of the normalized RGB texels.
    pub color_encoding: DisplayColorEncoding,
    texture: Arc<wgpu::Texture>,
    view: Arc<wgpu::TextureView>,
}

impl DisplayTexture {
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn texture_arc(&self) -> Arc<wgpu::Texture> {
        Arc::clone(&self.texture)
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn view_arc(&self) -> Arc<wgpu::TextureView> {
        Arc::clone(&self.view)
    }
}

/// The result of a convenience submission. No host wait has occurred.
#[derive(Clone, Debug)]
pub struct DisplaySubmission {
    pub texture: DisplayTexture,
    pub submission: wgpu::SubmissionIndex,
}

/// Observable pipeline-cache occupancy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisplayPipelineCacheStats {
    pub pipelines: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DisplayPipelineKey {
    Rgb {
        sample_type: u8,
        channels: u8,
        layout: u8,
        destination: wgpu::TextureFormat,
    },
    Image {
        format: DisplayImageFormat,
        destination: wgpu::TextureFormat,
    },
}

struct DisplayPipelineInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<DisplayPipelineKey, Arc<wgpu::ComputePipeline>>>,
}

/// Reusable display conversion state associated with one backend device and queue.
#[derive(Clone)]
pub struct DisplayPipeline {
    inner: Arc<DisplayPipelineInner>,
}

impl fmt::Debug for DisplayPipeline {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DisplayPipeline")
            .field("cache", &self.cache_stats())
            .finish_non_exhaustive()
    }
}

impl DisplayPipeline {
    /// Creates a display pipeline that submits convenience operations to `backend.queue()`.
    pub fn new(backend: &WgpuBackend) -> Self {
        Self::from_device_queue(backend.device().clone(), backend.queue().clone())
    }

    /// Creates a display pipeline for an application-owned device and its queue.
    ///
    /// Inputs passed to this pipeline must originate from `device`. To depend on an already
    /// exposed producer submission without an additional host wait, `queue` must be the same
    /// queue used for that submission.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            inner: Arc::new(DisplayPipelineInner {
                device,
                queue,
                pipelines: Mutex::new(HashMap::new()),
            }),
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub fn cache_stats(&self) -> DisplayPipelineCacheStats {
        DisplayPipelineCacheStats {
            pipelines: self.pipelines().len(),
        }
    }

    /// Drops compiled display pipelines. Existing textures and in-flight submissions are
    /// unaffected.
    pub fn clear_cache(&self) {
        self.pipelines().clear();
    }

    /// Encodes RGB or RGBA buffer conversion into a sampleable/renderable RGBA texture.
    ///
    /// F32, F16, U16, and U8 samples are accepted in planar or interleaved form. Floating-point
    /// values are interpreted as normalized linear-light BT.709 RGB and clamped by `Rgba8Unorm`.
    /// Integer samples are normalized over their complete unsigned range. Three-channel input gets
    /// opaque alpha. Signed I32 output has no bit-depth/range contract and returns
    /// [`Error::Unsupported`].
    pub fn encode_rgb(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuOutputBuffer,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        validate_display_descriptor(descriptor)?;
        validate_extent(self.device(), source.extent)?;
        require_buffer_usage(
            source.buffer.as_wgpu_buffer(),
            wgpu::BufferUsages::STORAGE,
            "RGB display source",
        )?;
        let validated = validate_rgb(source)?;
        validate_storage_binding(self.device(), validated.binding_size)?;

        let texture = self.create_texture(source.extent, descriptor, "jxl-wgpu RGB display");
        let key = DisplayPipelineKey::Rgb {
            sample_type: validated.sample_type,
            channels: source.channels,
            layout: validated.layout,
            destination: descriptor.format,
        };
        let pipeline = self.compute_pipeline(
            key,
            "jxl-wgpu RGB display",
            wgpu::include_wgsl!("../shaders/display_rgb.wgsl"),
        );
        let params = DisplayRgbParams {
            width: source.extent.width,
            height: source.extent.height,
            channels: u32::from(source.channels),
            sample_type: u32::from(validated.sample_type),
            layout: u32::from(validated.layout),
            logical_samples: validated.logical_samples,
            _padding: [0; 2],
        };
        self.record_dispatch(
            encoder,
            &pipeline,
            source.buffer.as_wgpu_buffer(),
            validated.binding_size,
            texture.view(),
            &params,
            source.extent,
            "jxl-wgpu RGB display bindings",
        );
        Ok(texture)
    }

    /// Submits [`Self::encode_rgb`] to this pipeline's queue without waiting for completion.
    pub fn submit_rgb(
        &self,
        source: &GpuOutputBuffer,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        self.submit_encoded("jxl-wgpu RGB display submission", |encoder| {
            self.encode_rgb(encoder, source, descriptor)
        })
    }

    /// Encodes a pitch-linear color buffer into a linear-light BT.709 RGBA display texture.
    ///
    /// Matrix, transfer, range, chroma siting, and pixel format are taken from
    /// `source.layout.format`. The source must use BT.709/sRGB primaries and a supported transfer;
    /// unsupported HDR/wide-gamut contracts reject instead of being silently mis-presented. Rows
    /// may be tightly packed; unlike a buffer-to-texture copy, this compute path has no 256-byte
    /// row-pitch rule.
    pub fn encode_image(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            descriptor,
            "jxl-wgpu image display",
        )
    }

    /// Encodes an explicitly unvalidated image output without waiting for codec validation.
    ///
    /// The pipeline must use the same device and queue as the producer. Queue ordering then makes
    /// the decode submission a dependency of this display dispatch. A later validation failure
    /// cannot retract the dispatch or its texture; applications must discard every derived result
    /// when validation fails.
    pub fn encode_unvalidated_image(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &UnvalidatedGpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            descriptor,
            "jxl-wgpu unvalidated image display",
        )
    }

    fn encode_image_source(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layout: &ImageLayout,
        buffer: &crate::GpuBufferLease,
        descriptor: DisplayTextureDescriptor,
        label: &str,
    ) -> Result<DisplayTexture> {
        validate_display_descriptor(descriptor)?;
        validate_extent(self.device(), layout.extent)?;
        require_buffer_usage(
            buffer.as_wgpu_buffer(),
            wgpu::BufferUsages::STORAGE,
            "image display source",
        )?;
        let validated = validate_image(layout, buffer.size())?;
        validate_storage_binding(self.device(), validated.binding_size)?;

        let texture = self.create_texture(layout.extent, descriptor, label);
        let pipeline = self.compute_pipeline(
            DisplayPipelineKey::Image {
                format: validated.format,
                destination: descriptor.format,
            },
            label,
            wgpu::include_wgsl!("../shaders/display_image.wgsl"),
        );
        let params = validated.params;
        self.record_dispatch(
            encoder,
            &pipeline,
            buffer.as_wgpu_buffer(),
            validated.binding_size,
            texture.view(),
            &params,
            layout.extent,
            "jxl-wgpu YUV display bindings",
        );
        Ok(texture)
    }

    /// Submits [`Self::encode_image`] to this pipeline's queue without waiting for completion.
    pub fn submit_image(
        &self,
        source: &GpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        self.submit_encoded("jxl-wgpu YUV display submission", |encoder| {
            self.encode_image(encoder, source, descriptor)
        })
    }

    /// Submits [`Self::encode_unvalidated_image`] without waiting for codec validation.
    pub fn submit_unvalidated_image(
        &self,
        source: &UnvalidatedGpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        self.submit_encoded("jxl-wgpu unvalidated image display submission", |encoder| {
            self.encode_unvalidated_image(encoder, source, descriptor)
        })
    }

    /// Encodes a direct buffer-to-texture copy for tightly packed linear BT.709 RGBA8 input.
    ///
    /// This avoids a compute dispatch. Multi-row copies require `width * 4` to be a multiple of
    /// [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`]. Use [`Self::encode_rgb`] when a tightly packed row is
    /// not copy-aligned; that conversion has no row-pitch restriction.
    pub fn encode_rgba8_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuOutputBuffer,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        validate_display_descriptor(descriptor)?;
        validate_extent(self.device(), source.extent)?;
        require_buffer_usage(
            source.buffer.as_wgpu_buffer(),
            wgpu::BufferUsages::COPY_SRC,
            "RGBA8 copy source",
        )?;
        validate_rgba8_copy(source)?;
        let bytes_per_row = source
            .extent
            .width
            .checked_mul(4)
            .ok_or(Error::BufferSizeOverflow)?;
        if source.extent.height > 1 && bytes_per_row % COPY_BYTES_PER_ROW_ALIGNMENT != 0 {
            return Err(Error::TextureCopyRowAlignment {
                bytes_per_row,
                required_alignment: COPY_BYTES_PER_ROW_ALIGNMENT,
            });
        }

        let texture = self.create_texture(source.extent, descriptor, "jxl-wgpu RGBA8 copy");
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: source.buffer.as_wgpu_buffer(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: (source.extent.height > 1).then_some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: texture.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            extent_3d(source.extent),
        );
        Ok(texture)
    }

    /// Submits [`Self::encode_rgba8_copy`] to this pipeline's queue without waiting.
    pub fn submit_rgba8_copy(
        &self,
        source: &GpuOutputBuffer,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        self.submit_encoded("jxl-wgpu RGBA8 copy submission", |encoder| {
            self.encode_rgba8_copy(encoder, source, descriptor)
        })
    }

    fn submit_encoded(
        &self,
        label: &str,
        encode: impl FnOnce(&mut wgpu::CommandEncoder) -> Result<DisplayTexture>,
    ) -> Result<DisplaySubmission> {
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let texture = encode(&mut encoder)?;
        let submission = self.queue().submit([encoder.finish()]);
        Ok(DisplaySubmission {
            texture,
            submission,
        })
    }

    fn create_texture(
        &self,
        extent: Extent2d,
        descriptor: DisplayTextureDescriptor,
        label: &str,
    ) -> DisplayTexture {
        let usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | descriptor.additional_usage;
        let texture = Arc::new(self.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent_3d(extent),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: descriptor.format,
            usage,
            view_formats: &[],
        }));
        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        DisplayTexture {
            extent,
            format: descriptor.format,
            usage,
            color_encoding: DisplayColorEncoding::LinearBt709,
            texture,
            view,
        }
    }

    fn compute_pipeline(
        &self,
        key: DisplayPipelineKey,
        label: &str,
        shader: wgpu::ShaderModuleDescriptor<'static>,
    ) -> Arc<wgpu::ComputePipeline> {
        let mut pipelines = self.pipelines();
        if let Some(pipeline) = pipelines.get(&key) {
            return Arc::clone(pipeline);
        }
        let module = self.device().create_shader_module(shader);
        let pipeline = Arc::new(self.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));
        pipelines.insert(key, Arc::clone(&pipeline));
        pipeline
    }

    #[allow(clippy::too_many_arguments)]
    fn record_dispatch<T: Pod>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        source: &wgpu::Buffer,
        source_binding_size: u64,
        destination: &wgpu::TextureView,
        params: &T,
        extent: Extent2d,
        label: &str,
    ) {
        let uniform = self
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: source,
                        offset: 0,
                        size: NonZeroU64::new(source_binding_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            extent.width.div_ceil(WORKGROUP_SIZE),
            extent.height.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    fn pipelines(&self) -> MutexGuard<'_, HashMap<DisplayPipelineKey, Arc<wgpu::ComputePipeline>>> {
        self.inner
            .pipelines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Clone, Copy, Debug)]
struct ValidatedRgb {
    sample_type: u8,
    layout: u8,
    logical_samples: u32,
    binding_size: u64,
}

fn validate_rgb(source: &GpuOutputBuffer) -> Result<ValidatedRgb> {
    if !matches!(source.channels, 3 | 4) {
        return Err(Error::InvalidPayload(format!(
            "RGB display input has {} channels; expected RGB or RGBA",
            source.channels
        )));
    }
    let sample_type = match source.sample_type {
        SampleType::F32 => 0,
        SampleType::F16 => 1,
        SampleType::U16 => 2,
        SampleType::U8 => 3,
        SampleType::I32 => {
            return Err(Error::Unsupported(
                "signed I32 RGB display input requires an explicit normalization contract".into(),
            ));
        }
    };
    let layout = match source.layout {
        OutputLayout::Planar => 0,
        OutputLayout::Interleaved => 1,
    };
    let logical_samples = source
        .extent
        .area()
        .and_then(|area| area.checked_mul(usize::from(source.channels)))
        .and_then(|samples| u32::try_from(samples).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    let required_size = u64::from(logical_samples)
        .checked_mul(
            u64::try_from(source.sample_type.bytes_per_sample())
                .map_err(|_| Error::BufferSizeOverflow)?,
        )
        .ok_or(Error::BufferSizeOverflow)?;
    validate_buffer_size(
        source.logical_size,
        source.buffer.size(),
        required_size,
        "RGB",
    )?;
    Ok(ValidatedRgb {
        sample_type,
        layout,
        logical_samples,
        binding_size: align_to_word(required_size)?,
    })
}

fn validate_rgba8_copy(source: &GpuOutputBuffer) -> Result<()> {
    if source.sample_type != SampleType::U8
        || source.channels != 4
        || source.layout != OutputLayout::Interleaved
    {
        return Err(Error::Unsupported(
            "direct display copy requires tightly packed interleaved RGBA8 input".into(),
        ));
    }
    let required_size = source
        .extent
        .area()
        .and_then(|area| area.checked_mul(4))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    validate_buffer_size(
        source.logical_size,
        source.buffer.size(),
        required_size,
        "RGBA8 copy",
    )
}

struct ValidatedImage {
    format: DisplayImageFormat,
    params: DisplayImageParams,
    binding_size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DisplayImageFormat {
    kind: u8,
    channels: u8,
    order: u8,
    matrix: u8,
    range: u8,
    siting_x: u8,
    siting_y: u8,
    chroma_order: u8,
    subsample_x: u8,
    subsample_y: u8,
    bits: u8,
    storage_bits: u8,
    transfer: u8,
}

fn validate_image(layout: &ImageLayout, buffer_size: u64) -> Result<ValidatedImage> {
    if layout.extent.is_empty() {
        return Err(Error::InvalidPayload(
            "display image extent must not be empty".into(),
        ));
    }
    if layout.logical_size == 0 || layout.logical_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "image logical size {} exceeds buffer size {buffer_size}",
            layout.logical_size
        )));
    }
    let canonical =
        ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
    if canonical.logical_size != layout.logical_size {
        return Err(Error::InvalidPayload(
            "image layout logical size is not canonical for its planes".into(),
        ));
    }
    let class = classify_display_format(&layout.format)?;
    let color = match layout.format.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Default | ColorSpecification::Undefined => {
            return Err(Error::Unsupported(
                "display conversion requires an explicit color specification".into(),
            ));
        }
    };
    if color.space != jxl_gpu_formats::ColorSpace::Bt709 {
        return Err(Error::Unsupported(format!(
            "display conversion to linear BT.709 does not implement {:?} primaries",
            color.space
        )));
    }
    let transfer = match color.transfer {
        TransferFunction::Linear => 0,
        TransferFunction::Srgb | TransferFunction::Sycc => 1,
        TransferFunction::Bt709 => 2,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "display transfer {unsupported:?} is unsupported"
            )));
        }
    };
    let (subsample_x, subsample_y) = layout
        .format
        .chroma_subsampling
        .chroma_divisors()
        .unwrap_or((1, 1));
    let chroma_extent = Extent2d::new(
        layout.extent.width.div_ceil(u32::from(subsample_x)),
        layout.extent.height.div_ceil(u32::from(subsample_y)),
    );
    let (kind, channels, order, bits, storage_bits, matrix, range, siting_x, siting_y) = match class
    {
        ColorFormatClass::Rgb8 { storage, order } => {
            let kind = match storage {
                RgbStorage::Interleaved => 0,
                RgbStorage::Planar => 1,
            };
            let (channels, order) = match order {
                RgbChannelOrder::Rgb => (3, 0),
                RgbChannelOrder::Bgr => (3, 1),
                RgbChannelOrder::Rgba => (4, 2),
                RgbChannelOrder::Bgra => (4, 3),
            };
            (kind, channels, order, 8, 8, 1, 0, 1, 1)
        }
        color_class => {
            let matrix = match color.encoding {
                YcbcrEncoding::Bt601 => 0,
                YcbcrEncoding::Bt709 => 1,
                YcbcrEncoding::Bt2020 => 2,
                unsupported => {
                    return Err(Error::Unsupported(format!(
                        "display YCbCr matrix {unsupported:?} is unsupported"
                    )));
                }
            };
            let range = u8::from(color.range == ColorRange::Limited);
            let siting_x = display_chroma_siting(color.chroma_location.horizontal, subsample_x)?;
            let siting_y = display_chroma_siting(color.chroma_location.vertical, subsample_y)?;
            match color_class {
                ColorFormatClass::Luma { bits, storage_bits } => (
                    if bits == 8 { 2 } else { 3 },
                    1,
                    0,
                    bits,
                    storage_bits,
                    matrix,
                    range,
                    siting_x,
                    siting_y,
                ),
                ColorFormatClass::YuvPlanar {
                    bits, storage_bits, ..
                } => (
                    4,
                    3,
                    0,
                    bits,
                    storage_bits,
                    matrix,
                    range,
                    siting_x,
                    siting_y,
                ),
                ColorFormatClass::YuvSemiplanar {
                    bits,
                    storage_bits,
                    chroma_order,
                    ..
                } => {
                    let order = u8::from(chroma_order == ChromaOrder::CrCb);
                    (
                        5,
                        3,
                        order,
                        bits,
                        storage_bits,
                        matrix,
                        range,
                        siting_x,
                        siting_y,
                    )
                }
                ColorFormatClass::Yuv422Packed { order } => (
                    6,
                    3,
                    u8::from(order == Packed422Order::Uyvy),
                    8,
                    8,
                    matrix,
                    range,
                    siting_x,
                    siting_y,
                ),
                ColorFormatClass::Rgb8 { .. } => {
                    unreachable!("RGB color classes were handled before YCbCr lowering")
                }
            }
        }
    };
    let display_format = DisplayImageFormat {
        kind,
        channels,
        order,
        matrix,
        range,
        siting_x,
        siting_y,
        chroma_order: order,
        subsample_x,
        subsample_y,
        bits,
        storage_bits,
        transfer,
    };
    for (index, plane) in layout.planes.iter().enumerate() {
        if plane.plane_index != index {
            return Err(Error::InvalidPayload(format!(
                "image plane slot {index} declares plane index {}",
                plane.plane_index
            )));
        }
        validate_image_plane_range(plane, layout.logical_size, buffer_size)?;
    }
    let binding_size = align_to_word(layout.logical_size)?;
    if binding_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "image buffer size {buffer_size} does not include word padding through {binding_size}"
        )));
    }
    let mut offsets = [0; 4];
    let mut strides = [0; 4];
    for (index, plane) in layout.planes.iter().enumerate() {
        if index >= 4 {
            return Err(Error::Unsupported(
                "display conversion supports at most four image planes".into(),
            ));
        }
        offsets[index] = shader_u32(plane.offset, "image plane offset")?;
        strides[index] = shader_u32(plane.row_stride, "image plane row stride")?;
    }
    Ok(ValidatedImage {
        format: display_format,
        params: DisplayImageParams {
            width: layout.extent.width,
            height: layout.extent.height,
            kind: u32::from(kind),
            channels: u32::from(channels),
            order: u32::from(order),
            matrix: u32::from(matrix),
            range: u32::from(range),
            siting_x: u32::from(siting_x),
            siting_y: u32::from(siting_y),
            subsample_x: u32::from(subsample_x),
            subsample_y: u32::from(subsample_y),
            bits: u32::from(bits),
            storage_bits: u32::from(storage_bits),
            plane0_offset: offsets[0],
            plane0_stride: strides[0],
            plane1_offset: offsets[1],
            plane1_stride: strides[1],
            plane2_offset: offsets[2],
            plane2_stride: strides[2],
            plane3_offset: offsets[3],
            plane3_stride: strides[3],
            chroma_width: chroma_extent.width,
            chroma_height: chroma_extent.height,
            transfer: u32::from(transfer),
        },
        binding_size,
    })
}

fn classify_display_format(format: &PixelFormat) -> Result<ColorFormatClass> {
    match classify_pixel_format(format) {
        Ok(PixelFormatClass::Color(color)) => Ok(color),
        Ok(PixelFormatClass::Numeric(numeric)) => Err(numeric_display_error(numeric)),
        Err(error) => Err(Error::Unsupported(format!(
            "display image format is unsupported: {error}"
        ))),
    }
}

fn numeric_display_error(numeric: NumericFormatClass) -> Error {
    if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
        Error::Unsupported(
            "display conversion has no implicit color semantics for numeric F64; portable WGSL also has no native F64 arithmetic"
                .into(),
        )
    } else {
        Error::Unsupported(format!(
            "display conversion has no implicit color semantics for numeric format {numeric:?}"
        ))
    }
}

fn validate_image_plane_range(
    plane: &jxl_gpu_formats::PitchLinearPlaneLayout,
    logical_size: u64,
    buffer_size: u64,
) -> Result<()> {
    let end = plane
        .offset
        .checked_add(
            u64::from(plane.sample_extent.height.saturating_sub(1))
                .checked_mul(plane.row_stride)
                .ok_or(Error::BufferSizeOverflow)?,
        )
        .and_then(|offset| offset.checked_add(plane.row_bytes))
        .ok_or(Error::BufferSizeOverflow)?;
    if end > logical_size || end > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "image plane {} ends at byte {end}, outside logical/buffer sizes {logical_size}/{buffer_size}",
            plane.plane_index
        )));
    }
    let last_byte = end.checked_sub(1).ok_or_else(|| {
        Error::InvalidPayload(format!(
            "image plane {} has an empty byte range",
            plane.plane_index
        ))
    })?;
    let _ = shader_u32(last_byte, "image plane final byte address")?;
    Ok(())
}

fn require_buffer_usage(
    buffer: &wgpu::Buffer,
    required: wgpu::BufferUsages,
    label: &str,
) -> Result<()> {
    if buffer.usage().contains(required) {
        Ok(())
    } else {
        Err(Error::InvalidPayload(format!(
            "{label} requires buffer usage {required:?}, actual usage is {:?}",
            buffer.usage()
        )))
    }
}

fn display_chroma_siting(location: ChromaLocation, divisor: u8) -> Result<u8> {
    if divisor == 1 {
        return Ok(1);
    }
    match location {
        ChromaLocation::Center => Ok(0),
        ChromaLocation::Even => Ok(1),
        unsupported => Err(Error::Unsupported(format!(
            "display chroma location {unsupported:?} is not supported for {divisor}:1 subsampling"
        ))),
    }
}

fn validate_display_descriptor(descriptor: DisplayTextureDescriptor) -> Result<()> {
    if descriptor.format != wgpu::TextureFormat::Rgba8Unorm {
        return Err(Error::Unsupported(format!(
            "display texture format {:?}; portable conversion supports only Rgba8Unorm",
            descriptor.format
        )));
    }
    Ok(())
}

fn validate_extent(device: &wgpu::Device, extent: Extent2d) -> Result<()> {
    if extent.is_empty() {
        return Err(Error::InvalidPayload(
            "display texture extent must not be empty".into(),
        ));
    }
    let limit = device.limits().max_texture_dimension_2d;
    if extent.width > limit || extent.height > limit {
        return Err(Error::DisplayTextureExtent {
            width: extent.width,
            height: extent.height,
            limit,
        });
    }
    Ok(())
}

fn validate_storage_binding(device: &wgpu::Device, binding_size: u64) -> Result<()> {
    let limit = device.limits().max_storage_buffer_binding_size;
    if binding_size > limit {
        return Err(Error::ResourceLimit(format!(
            "display input binding requires {binding_size} bytes, device permits {limit}"
        )));
    }
    Ok(())
}

fn validate_buffer_size(
    logical_size: u64,
    buffer_size: u64,
    required_size: u64,
    label: &str,
) -> Result<()> {
    if logical_size < required_size || buffer_size < required_size || logical_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "{label} requires {required_size} bytes, but logical/buffer sizes are {logical_size}/{buffer_size}"
        )));
    }
    let binding_size = align_to_word(required_size)?;
    if binding_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "{label} buffer size {buffer_size} does not include word padding through {binding_size}"
        )));
    }
    Ok(())
}

fn align_to_word(size: u64) -> Result<u64> {
    size.checked_add(3)
        .map(|size| size & !3)
        .ok_or(Error::BufferSizeOverflow)
}

fn shader_u32(value: u64, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::ResourceLimit(format!(
            "{label} {value} is not addressable by the portable display shader"
        ))
    })
}

fn extent_3d(extent: Extent2d) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: 1,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DisplayRgbParams {
    width: u32,
    height: u32,
    channels: u32,
    sample_type: u32,
    layout: u32,
    logical_samples: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DisplayImageParams {
    width: u32,
    height: u32,
    kind: u32,
    channels: u32,
    order: u32,
    matrix: u32,
    range: u32,
    siting_x: u32,
    siting_y: u32,
    subsample_x: u32,
    subsample_y: u32,
    bits: u32,
    storage_bits: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    plane3_offset: u32,
    plane3_stride: u32,
    chroma_width: u32,
    chroma_height: u32,
    transfer: u32,
}

const _: () = {
    assert!(std::mem::size_of::<DisplayRgbParams>() == 32);
    assert!(std::mem::align_of::<DisplayRgbParams>() == 4);
    assert!(std::mem::size_of::<DisplayImageParams>() == 96);
    assert!(std::mem::align_of::<DisplayImageParams>() == 4);
};

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use jxl_gpu_formats::ColorSpec;

    use super::*;

    fn abi_words<T: Pod>(value: &T) -> &[u32] {
        bytemuck::cast_slice(std::slice::from_ref(value))
    }

    fn assert_wgsl_fields(shader: &str, name: &str, expected: &[&str]) {
        let marker = format!("struct {name} {{");
        let (_, after_marker) = shader
            .split_once(&marker)
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is missing"));
        let (body, _) = after_marker
            .split_once("};")
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is not terminated"));
        let actual = body
            .lines()
            .filter_map(|line| line.split_once(':').map(|(field, _)| field.trim()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "WGSL field-order drift for {name}");
    }

    #[test]
    fn canonical_classifier_drives_vpi_display_support() {
        let mut color_count = 0;
        let mut numeric_count = 0;
        for predefined in jxl_gpu_formats::vpi::VpiPitchLinearFormat::ALL {
            let format = predefined.pixel_format();
            let layout = ImageLayout::packed(Extent2d::new(3, 3), format.clone()).unwrap();
            let buffer_size = align_to_word(layout.logical_size).unwrap();
            match classify_pixel_format(&format).unwrap() {
                PixelFormatClass::Color(_) => {
                    color_count += 1;
                    validate_image(&layout, buffer_size)
                        .unwrap_or_else(|error| panic!("{}: {error}", predefined.name()));
                }
                PixelFormatClass::Numeric(numeric) => {
                    numeric_count += 1;
                    let Err(Error::Unsupported(message)) = validate_image(&layout, buffer_size)
                    else {
                        panic!("{} did not return typed Unsupported", predefined.name());
                    };
                    assert!(message.contains("no implicit color semantics"));
                    if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
                        assert!(message.contains("no native F64 arithmetic"));
                    }
                }
            }
        }
        assert_eq!(color_count, 20);
        assert_eq!(numeric_count, 10);
    }

    #[test]
    fn display_uniform_abi_sizes_are_explicit_and_aligned() {
        assert_eq!(size_of::<DisplayRgbParams>(), 32);
        assert_eq!(size_of::<DisplayImageParams>(), 96);
        assert_eq!(std::mem::align_of::<DisplayRgbParams>(), 4);
        assert_eq!(std::mem::align_of::<DisplayImageParams>(), 4);
        assert_eq!(size_of::<DisplayRgbParams>() % 16, 0);
        assert_eq!(size_of::<DisplayImageParams>() % 16, 0);
    }

    #[test]
    fn display_uniform_rust_word_order_matches_wgsl_field_order() {
        let rgb = DisplayRgbParams {
            width: 1,
            height: 2,
            channels: 3,
            sample_type: 4,
            layout: 5,
            logical_samples: 6,
            _padding: [7, 8],
        };
        assert_eq!(abi_words(&rgb), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_wgsl_fields(
            include_str!("../shaders/display_rgb.wgsl"),
            "DisplayRgbParams",
            &[
                "width",
                "height",
                "channels",
                "sample_type",
                "storage_layout",
                "logical_samples",
                "_padding0",
                "_padding1",
            ],
        );

        let image = DisplayImageParams {
            width: 1,
            height: 2,
            kind: 3,
            channels: 4,
            order: 5,
            matrix: 6,
            range: 7,
            siting_x: 8,
            siting_y: 9,
            subsample_x: 10,
            subsample_y: 11,
            bits: 12,
            storage_bits: 13,
            plane0_offset: 14,
            plane0_stride: 15,
            plane1_offset: 16,
            plane1_stride: 17,
            plane2_offset: 18,
            plane2_stride: 19,
            plane3_offset: 20,
            plane3_stride: 21,
            chroma_width: 22,
            chroma_height: 23,
            transfer: 24,
        };
        let expected = (1..=24).collect::<Vec<_>>();
        assert_eq!(abi_words(&image), expected);
        assert_wgsl_fields(
            include_str!("../shaders/display_image.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "kind",
                "channels",
                "order",
                "matrix",
                "range",
                "siting_x",
                "siting_y",
                "subsample_x",
                "subsample_y",
                "bits",
                "storage_bits",
                "plane0_offset",
                "plane0_stride",
                "plane1_offset",
                "plane1_stride",
                "plane2_offset",
                "plane2_stride",
                "plane3_offset",
                "plane3_stride",
                "chroma_width",
                "chroma_height",
                "transfer",
            ],
        );
    }

    #[test]
    fn cache_key_includes_yuv_format_matrix_and_range() {
        let format = DisplayImageFormat {
            kind: 5,
            channels: 3,
            order: 0,
            matrix: 1,
            range: 1,
            siting_x: 0,
            siting_y: 0,
            chroma_order: 0,
            subsample_x: 2,
            subsample_y: 2,
            bits: 8,
            storage_bits: 8,
            transfer: 2,
        };
        let base = DisplayPipelineKey::Image {
            format,
            destination: wgpu::TextureFormat::Rgba8Unorm,
        };
        let matrix = DisplayPipelineKey::Image {
            format: DisplayImageFormat {
                matrix: 2,
                ..format
            },
            destination: wgpu::TextureFormat::Rgba8Unorm,
        };
        let range = DisplayPipelineKey::Image {
            format: DisplayImageFormat { range: 0, ..format },
            destination: wgpu::TextureFormat::Rgba8Unorm,
        };
        let planar = DisplayPipelineKey::Image {
            format: DisplayImageFormat { kind: 4, ..format },
            destination: wgpu::TextureFormat::Rgba8Unorm,
        };
        let transfer = DisplayPipelineKey::Image {
            format: DisplayImageFormat {
                transfer: 1,
                ..format
            },
            destination: wgpu::TextureFormat::Rgba8Unorm,
        };
        assert_ne!(base, matrix);
        assert_ne!(base, range);
        assert_ne!(base, planar);
        assert_ne!(base, transfer);
    }

    #[test]
    fn validates_pitch_linear_yuv_rows_and_extents() {
        let color = ColorSpecification::Defined(jxl_gpu_formats::ColorSpec {
            space: jxl_gpu_formats::ColorSpace::Bt709,
            encoding: YcbcrEncoding::Bt709,
            transfer: jxl_gpu_formats::TransferFunction::Bt709,
            range: ColorRange::Limited,
            chroma_location: jxl_gpu_formats::ChromaLocation2d::CENTER,
        });
        let format = jxl_gpu_formats::PixelFormat::yuv_semiplanar(
            jxl_gpu_formats::ChromaSubsampling::Cs420,
            8,
            8,
            jxl_gpu_formats::ChromaOrder::CbCr,
            color,
        )
        .unwrap();
        let mut layout = ImageLayout::packed(Extent2d::new(5, 3), format).unwrap();
        let padded = align_to_word(layout.logical_size).unwrap();
        let validated = validate_image(&layout, padded).unwrap();
        assert_eq!(validated.params.chroma_width, 3);
        assert_eq!(validated.params.chroma_height, 2);
        assert_eq!(validated.params.plane1_stride, 6);

        layout.planes[1].row_stride = 5;
        assert!(matches!(
            validate_image(&layout, padded),
            Err(Error::ImageLayout(
                jxl_gpu_formats::LayoutError::ShortRowStride {
                    plane: 1,
                    minimum: 6,
                    actual: 5,
                }
            ))
        ));
    }

    #[test]
    fn direct_copy_alignment_rule_is_explicit() {
        let row = 63 * 4;
        assert_ne!(row % COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        assert_eq!((64 * 4) % COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    }

    #[test]
    fn display_rejects_unimplemented_primaries_and_hdr_transfer() {
        let extent = Extent2d::new(2, 2);
        let unsupported = [
            ColorSpec {
                space: jxl_gpu_formats::ColorSpace::Bt2020,
                encoding: YcbcrEncoding::Bt709,
                transfer: TransferFunction::Bt709,
                range: ColorRange::Full,
                chroma_location: jxl_gpu_formats::ChromaLocation2d::CENTER,
            },
            ColorSpec {
                space: jxl_gpu_formats::ColorSpace::Bt709,
                encoding: YcbcrEncoding::Bt709,
                transfer: TransferFunction::Pq,
                range: ColorRange::Full,
                chroma_location: jxl_gpu_formats::ChromaLocation2d::CENTER,
            },
        ];
        for color in unsupported {
            let format = PixelFormat::rgb8(
                RgbChannelOrder::Rgba,
                false,
                ColorSpecification::Defined(color),
            );
            let layout = ImageLayout::packed(extent, format).unwrap();
            let buffer_size = align_to_word(layout.logical_size).unwrap();
            assert!(matches!(
                validate_image(&layout, buffer_size),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn image_plane_final_address_must_fit_wgsl_u32() {
        let plane = jxl_gpu_formats::PitchLinearPlaneLayout {
            plane_index: 0,
            offset: u64::from(u32::MAX),
            row_stride: 4,
            sample_extent: Extent2d::new(1, 1),
            row_bytes: 2,
        };
        assert!(matches!(
            validate_image_plane_range(&plane, u64::from(u32::MAX) + 2, u64::from(u32::MAX) + 2),
            Err(Error::ResourceLimit(_))
        ));
    }
}
