// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Non-blocking conversion of decoder-owned GPU buffers into display textures.
//!
//! [`DisplayPipeline::encode_rgb`] and [`DisplayPipeline::encode_image`] only append work to a
//! caller-owned command encoder. [`DisplayPipeline::submit_rgb`] and
//! [`DisplayPipeline::submit_image`] submit that work to the exact queue from which the pipeline was
//! constructed. Neither path waits for the host. A decoder output submitted earlier to the same
//! queue is therefore a valid input immediately, and the returned texture can be sampled, rendered,
//! or copied by later commands on that queue.
//!
//! This module consumes portable pitch-linear storage buffers. Native multi-plane textures and
//! platform-specific external-memory layouts are deliberately outside this API contract.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard};

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaLocation, ColorModel, ColorRange, ColorSpecification, ImageLayout,
    PackingFieldKind, SampleKind, Swizzle, YcbcrEncoding,
};
use jxl_gpu_protocol::{Extent2d, OutputLayout, SampleType};
use wgpu::util::DeviceExt;

use crate::context::WgpuBackend;
use crate::session::GpuOutputBuffer;
use crate::video::GpuImageOutput;
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
    /// Inputs passed to this pipeline must originate from `device`. To depend on a decoder
    /// submission without a host wait, `queue` must be the same queue used for that submission.
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
    /// values are interpreted as normalized nonlinear display RGB and clamped by `Rgba8Unorm`.
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
            &source.buffer,
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
            &source.buffer,
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

    /// Encodes a pitch-linear I444, I422, I420, or NV12 buffer into an RGBA display texture.
    ///
    /// Matrix, range, chroma siting, and pixel format are taken from `source.layout.format`. The
    /// compiled pipeline cache is keyed by that complete format contract. Rows may be tightly
    /// packed; unlike a buffer-to-texture copy, this compute path has no 256-byte row-pitch rule.
    pub fn encode_image(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        validate_display_descriptor(descriptor)?;
        validate_extent(self.device(), source.layout.extent)?;
        require_buffer_usage(
            &source.buffer,
            wgpu::BufferUsages::STORAGE,
            "image display source",
        )?;
        let validated = validate_image(&source.layout, source.buffer.size())?;
        validate_storage_binding(self.device(), validated.binding_size)?;

        let texture = self.create_texture(source.layout.extent, descriptor, "jxl-wgpu YUV display");
        let pipeline = self.compute_pipeline(
            DisplayPipelineKey::Image {
                format: validated.format,
                destination: descriptor.format,
            },
            "jxl-wgpu YUV display",
            wgpu::include_wgsl!("../shaders/display_image.wgsl"),
        );
        let params = validated.params;
        self.record_dispatch(
            encoder,
            &pipeline,
            &source.buffer,
            validated.binding_size,
            texture.view(),
            &params,
            source.layout.extent,
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

    /// Encodes a direct buffer-to-texture copy for tightly packed interleaved RGBA8 input.
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
            &source.buffer,
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
                buffer: &source.buffer,
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
    if layout.format.sample_kind != SampleKind::Unsigned
        || layout.format.byte_order == ByteOrder::Big
    {
        return Err(Error::Unsupported(
            "display conversion requires unsigned native/little-endian pitch-linear storage".into(),
        ));
    }
    let storage = layout
        .format
        .planes
        .iter()
        .map(stored_plane_channels)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::Unsupported(
                "display conversion requires uniform byte-aligned channel words".into(),
            )
        })?;
    let (subsample_x, subsample_y) = layout
        .format
        .chroma_subsampling
        .chroma_divisors()
        .unwrap_or((1, 1));
    let chroma_extent = Extent2d::new(
        layout.extent.width.div_ceil(u32::from(subsample_x)),
        layout.extent.height.div_ceil(u32::from(subsample_y)),
    );
    let (kind, channels, order, bits, storage_bits, matrix, range, siting_x, siting_y) =
        match layout.format.model {
            ColorModel::Rgb => {
                let (kind, channels) = match storage.as_slice() {
                    [(channels, 8, 8)] if matches!(channels.len(), 3 | 4) => {
                        (0, channels.len() as u8)
                    }
                    planar
                        if matches!(planar.len(), 3 | 4)
                            && planar.iter().all(|(channels, bits, storage_bits)| {
                                channels.len() == 1 && *bits == 8 && *storage_bits == 8
                            }) =>
                    {
                        (1, planar.len() as u8)
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "display RGB image requires planar or interleaved RGB(A)/BGR(A)8"
                                .into(),
                        ));
                    }
                };
                let order = if layout.format.swizzle == Swizzle::XYZ1 {
                    0
                } else if layout.format.swizzle == Swizzle::ZYX1 {
                    1
                } else if layout.format.swizzle == Swizzle::XYZW {
                    2
                } else if layout.format.swizzle == Swizzle::ZYXW {
                    3
                } else {
                    return Err(Error::Unsupported(format!(
                        "display RGB swizzle {:?} is unsupported",
                        layout.format.swizzle
                    )));
                };
                (kind, channels, order, 8, 8, 1, 0, 1, 1)
            }
            ColorModel::Ycbcr => {
                let color = match layout.format.color_spec {
                    ColorSpecification::Defined(color) => color,
                    ColorSpecification::Default | ColorSpecification::Undefined => {
                        return Err(Error::Unsupported(
                            "display YCbCr conversion requires an explicit matrix and range".into(),
                        ));
                    }
                };
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
                let siting_x =
                    display_chroma_siting(color.chroma_location.horizontal, subsample_x)?;
                let siting_y = display_chroma_siting(color.chroma_location.vertical, subsample_y)?;
                match storage.as_slice() {
                    [(y, bits, storage_bits)] if y.as_slice() == [Channel::X] => {
                        if !matches!((*bits, *storage_bits), (8, 8) | (16, 16)) {
                            return Err(Error::Unsupported(
                                "display luma supports only Y8 and Y16".into(),
                            ));
                        }
                        (
                            if *bits == 8 { 2 } else { 3 },
                            1,
                            0,
                            *bits,
                            *storage_bits,
                            matrix,
                            range,
                            siting_x,
                            siting_y,
                        )
                    }
                    [(packed, 8, 8)]
                        if packed.as_slice()
                            == [Channel::X, Channel::Y, Channel::X, Channel::Z] =>
                    {
                        (6, 3, 0, 8, 8, matrix, range, siting_x, siting_y)
                    }
                    [(packed, 8, 8)]
                        if packed.as_slice()
                            == [Channel::Y, Channel::X, Channel::Z, Channel::X] =>
                    {
                        (6, 3, 1, 8, 8, matrix, range, siting_x, siting_y)
                    }
                    [
                        (y, bits, stored),
                        (cb, cb_bits, cb_stored),
                        (cr, cr_bits, cr_stored),
                    ] if y.as_slice() == [Channel::X]
                        && cb.as_slice() == [Channel::Y]
                        && cr.as_slice() == [Channel::Z]
                        && bits == cb_bits
                        && bits == cr_bits
                        && stored == cb_stored
                        && stored == cr_stored =>
                    {
                        validate_display_yuv_depth(*bits, *stored)?;
                        (4, 3, 0, *bits, *stored, matrix, range, siting_x, siting_y)
                    }
                    [(y, bits, stored), (uv, uv_bits, uv_stored)]
                        if y.as_slice() == [Channel::X]
                            && matches!(
                                uv.as_slice(),
                                [Channel::Y, Channel::Z] | [Channel::Z, Channel::Y]
                            )
                            && bits == uv_bits
                            && stored == uv_stored =>
                    {
                        validate_display_yuv_depth(*bits, *stored)?;
                        let order = u8::from(uv.as_slice() == [Channel::Z, Channel::Y]);
                        (
                            5, 3, order, *bits, *stored, matrix, range, siting_x, siting_y,
                        )
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "display YCbCr format is not a supported luma, planar/semi-planar, or packed 4:2:2 layout"
                                .into(),
                        ));
                    }
                }
            }
            unsupported => {
                return Err(Error::Unsupported(format!(
                    "display semantics for color model {unsupported:?} are undefined"
                )));
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
            _padding: 0,
        },
        binding_size,
    })
}

fn validate_display_yuv_depth(bits: u8, storage_bits: u8) -> Result<()> {
    if matches!((bits, storage_bits), (8, 8) | (10 | 12 | 16, 16)) {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "display YCbCr depth {bits} in {storage_bits}-bit storage is unsupported"
        )))
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

fn stored_plane_channels(plane: &jxl_gpu_formats::PlaneFormat) -> Option<(Vec<Channel>, u8, u8)> {
    let mut channels = Vec::with_capacity(plane.words.len());
    let mut channel_bits = None;
    let mut storage_bits = None;
    for word in &plane.words {
        let field = word.fields.first()?;
        let PackingFieldKind::Channel(channel) = field.kind else {
            return None;
        };
        if word
            .fields
            .iter()
            .skip(1)
            .any(|field| !matches!(field.kind, PackingFieldKind::Padding) || field.bits == 0)
        {
            return None;
        }
        let word_bits = u8::try_from(word.bits()).ok()?;
        if channel_bits
            .replace(field.bits)
            .is_some_and(|bits| bits != field.bits)
            || storage_bits
                .replace(word_bits)
                .is_some_and(|bits| bits != word_bits)
        {
            return None;
        }
        channels.push(channel);
    }
    Some((channels, channel_bits?, storage_bits?))
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
    _padding: u32,
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

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
        assert_ne!(base, matrix);
        assert_ne!(base, range);
        assert_ne!(base, planar);
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
