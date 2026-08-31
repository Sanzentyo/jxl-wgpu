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
//! Caller-owned encoder paths also take a [`GpuBufferSubmissionGuard`], which must remain alive
//! through queue submission; convenience `submit_*` methods manage that guard automatically.
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
    SampleKind, TransferFunction, WgslNumericCapability, YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{Extent2d, OutputLayout, SampleType};
use wgpu::util::DeviceExt;

use crate::context::WgpuBackend;
use crate::session::GpuOutputBuffer;
use crate::video::{GpuImageOutput, UnvalidatedGpuImageOutput};
use crate::{Error, GpuBufferSubmissionGuard, KernelPolicy, KernelVariant, Result};

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
const NUMERIC_F64_MARKER: &str = "/*__JXL_NUMERIC_F64__*/";

/// Declared interpretation of the stored scalar type for numeric visualization.
///
/// This must match the [`jxl_gpu_formats::SampleKind`] in the image's [`PixelFormat`]. Keeping it
/// in the contract makes a numeric display request self-contained instead of silently deriving a
/// normalization rule from storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericDisplaySource {
    Unsigned,
    Signed,
    Floating { non_finite: NumericNonFinitePolicy },
}

/// Handling applied when a floating-point source, or its affine normalization, is not finite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericNonFinitePolicy {
    /// NaN and both infinities become zero.
    Zero,
    /// NaN and negative infinity become zero; positive infinity becomes one.
    Saturate,
}

/// Mapping from one- or two-component VPI numeric storage to display RGBA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericDisplayChannels {
    /// Component X is replicated to RGB; alpha is one.
    Luma,
    /// Component X is replicated to RGB and component Y supplies alpha. Requires two components.
    LumaAlpha,
    /// Components X/Y supply red/green; blue is zero and alpha is one. Requires two components.
    RedGreen,
}

/// Transfer interpretation of normalized numeric color components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericDisplayTransfer {
    /// Normalized values already represent linear-light BT.709/sRGB-primary intensities.
    Linear,
    /// Normalized RGB values use the sRGB transfer and are decoded to linear light. Alpha is never
    /// transfer-decoded.
    Srgb,
}

/// Destination clamp applied after affine normalization and non-finite handling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericDisplayClamp {
    /// Clamp every normalized component to `[0, 1]` before writing `Rgba8Unorm`.
    Unit,
}

/// Complete, explicit numeric-to-display contract.
///
/// For every finite stored scalar `x`, normalization first computes `x * scale + bias`. Floating
/// non-finite results follow [`NumericNonFinitePolicy`]; integer affine overflow saturates by sign
/// (`+Inf` to one and `-Inf` to zero). [`NumericDisplayClamp::Unit`] is then applied.
/// [`NumericDisplayTransfer`] affects RGB only, after channel visualization; alpha stays linear.
/// The output texture is linear-light BT.709 RGBA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericDisplayContract {
    pub source: NumericDisplaySource,
    pub scale: f32,
    pub bias: f32,
    pub channels: NumericDisplayChannels,
    pub transfer: NumericDisplayTransfer,
    pub clamp: NumericDisplayClamp,
}

/// Arithmetic selected for one numeric display conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumericDisplayPrecision {
    /// Integer and F32 storage are converted and normalized with portable f32 arithmetic.
    PortableF32,
    /// F64 storage is rounded to f32 before the affine mapping because native shader f64 is not
    /// enabled on this device, or because the supplied plane offset/row pitch is not naturally
    /// eight-byte aligned.
    F64RoundedToF32,
    /// F64 storage and the affine mapping use native shader f64 before the display result is
    /// rounded to f32.
    NativeF64,
}

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
pub enum NumericDisplayError {
    #[error("numeric display scale must be finite, got {0}")]
    NonFiniteScale(f32),
    #[error("numeric display bias must be finite, got {0}")]
    NonFiniteBias(f32),
    #[error("numeric display requires a non-color numeric pixel format")]
    NonNumericFormat,
    #[error("numeric display source contract {declared:?} does not match stored kind {actual:?}")]
    SourceKindMismatch {
        declared: NumericDisplaySource,
        actual: jxl_gpu_formats::SampleKind,
    },
    #[error("numeric channel mapping {mapping:?} requires two components, format has {components}")]
    TwoComponentsRequired {
        mapping: NumericDisplayChannels,
        components: u8,
    },
}

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
    /// Arithmetic used for a numeric visualization, or `None` for color image/RGB conversion.
    pub numeric_precision: Option<NumericDisplayPrecision>,
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
        variant: KernelVariant,
    },
    Image {
        format: DisplayImageFormat,
        destination: wgpu::TextureFormat,
        variant: KernelVariant,
    },
    Numeric {
        native_f64: bool,
        destination: wgpu::TextureFormat,
        variant: KernelVariant,
    },
}

struct DisplayPipelineInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    native_f64_enabled: bool,
    kernel_policy: KernelPolicy,
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
        Self::from_device_queue_with_f64(
            backend.device().clone(),
            backend.queue().clone(),
            backend.native_f64_enabled(),
            backend.kernel_policy().clone(),
        )
    }

    /// Creates a display pipeline for an application-owned device and its queue.
    ///
    /// Inputs passed to this pipeline must originate from `device`. To depend on an already
    /// exposed producer submission without an additional host wait, `queue` must be the same
    /// queue used for that submission. Native f64 display arithmetic is enabled exactly when the
    /// supplied logical device exposes `wgpu::Features::SHADER_F64`.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let native_f64_enabled = device.features().contains(wgpu::Features::SHADER_F64);
        Self::from_device_queue_with_f64(device, queue, native_f64_enabled, KernelPolicy::Default)
    }

    fn from_device_queue_with_f64(
        device: wgpu::Device,
        queue: wgpu::Queue,
        native_f64_enabled: bool,
        kernel_policy: KernelPolicy,
    ) -> Self {
        Self {
            inner: Arc::new(DisplayPipelineInner {
                device,
                queue,
                native_f64_enabled,
                kernel_policy,
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

    /// Whether F64 numeric display input uses native shader f64 arithmetic.
    #[must_use]
    pub fn native_f64_enabled(&self) -> bool {
        self.inner.native_f64_enabled
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
        submission_guard: &GpuBufferSubmissionGuard,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        require_submission_guard(submission_guard, &source.buffer)?;
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
        let variant = self.kernel_variant("display_rgb", KernelVariant::Tile16x16)?;
        let key = DisplayPipelineKey::Rgb {
            sample_type: validated.sample_type,
            channels: source.channels,
            layout: validated.layout,
            destination: descriptor.format,
            variant,
        };
        let pipeline = self.compute_pipeline(
            key,
            "jxl-wgpu RGB display",
            wgpu::include_wgsl!("../shaders/display_rgb.wgsl"),
            variant,
        )?;
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
            variant,
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
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded("jxl-wgpu RGB display submission", |encoder| {
            self.encode_rgb(encoder, source, &submission_guard, descriptor)
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
        submission_guard: &GpuBufferSubmissionGuard,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            submission_guard,
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
        submission_guard: &GpuBufferSubmissionGuard,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            submission_guard,
            descriptor,
            "jxl-wgpu unvalidated image display",
        )
    }

    fn encode_image_source(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layout: &ImageLayout,
        buffer: &crate::GpuBufferLease,
        submission_guard: &GpuBufferSubmissionGuard,
        descriptor: DisplayTextureDescriptor,
        label: &str,
    ) -> Result<DisplayTexture> {
        require_submission_guard(submission_guard, buffer)?;
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
        let variant = self.kernel_variant("display_image", KernelVariant::Tile16x16)?;
        let pipeline = self.compute_pipeline(
            DisplayPipelineKey::Image {
                format: validated.format,
                destination: descriptor.format,
                variant,
            },
            label,
            wgpu::include_wgsl!("../shaders/display_image.wgsl"),
            variant,
        )?;
        let params = validated.params;
        self.record_dispatch(
            encoder,
            &pipeline,
            buffer.as_wgpu_buffer(),
            validated.binding_size,
            texture.view(),
            &params,
            layout.extent,
            variant,
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
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded("jxl-wgpu YUV display submission", |encoder| {
            self.encode_image(encoder, source, &submission_guard, descriptor)
        })
    }

    /// Submits [`Self::encode_unvalidated_image`] without waiting for codec validation.
    pub fn submit_unvalidated_image(
        &self,
        source: &UnvalidatedGpuImageOutput,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded("jxl-wgpu unvalidated image display submission", |encoder| {
            self.encode_unvalidated_image(encoder, source, &submission_guard, descriptor)
        })
    }

    /// Encodes a non-color numeric pitch-linear image under an explicit affine/display contract.
    ///
    /// All ten numeric VPI 4.1 pitch-linear formats are accepted. No sample kind, scale, bias,
    /// non-finite policy, clamp, transfer, or channel visualization is inferred. F64 uses native
    /// shader f64 when the supplied device has `wgpu::Features::SHADER_F64` enabled and the F64
    /// plane offset/row pitch are naturally eight-byte aligned; otherwise it follows the
    /// documented [`NumericDisplayPrecision::F64RoundedToF32`] path.
    pub fn encode_numeric_image(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &GpuImageOutput,
        submission_guard: &GpuBufferSubmissionGuard,
        contract: NumericDisplayContract,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_numeric_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            submission_guard,
            contract,
            descriptor,
            "jxl-wgpu numeric image display",
        )
    }

    /// Encodes an unvalidated numeric decoder output under an explicit display contract.
    pub fn encode_unvalidated_numeric_image(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source: &UnvalidatedGpuImageOutput,
        submission_guard: &GpuBufferSubmissionGuard,
        contract: NumericDisplayContract,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        self.encode_numeric_image_source(
            encoder,
            &source.layout,
            &source.buffer,
            submission_guard,
            contract,
            descriptor,
            "jxl-wgpu unvalidated numeric image display",
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_numeric_image_source(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layout: &ImageLayout,
        buffer: &crate::GpuBufferLease,
        submission_guard: &GpuBufferSubmissionGuard,
        contract: NumericDisplayContract,
        descriptor: DisplayTextureDescriptor,
        label: &str,
    ) -> Result<DisplayTexture> {
        require_submission_guard(submission_guard, buffer)?;
        validate_display_descriptor(descriptor)?;
        validate_extent(self.device(), layout.extent)?;
        require_buffer_usage(
            buffer.as_wgpu_buffer(),
            wgpu::BufferUsages::STORAGE,
            "numeric display source",
        )?;
        let validated = validate_numeric_image(layout, buffer.size(), contract)?;
        validate_storage_binding(self.device(), validated.binding_size)?;

        let native_f64 = validated.native_f64_compatible && self.native_f64_enabled();
        let precision = if native_f64 {
            NumericDisplayPrecision::NativeF64
        } else if validated.is_f64 {
            NumericDisplayPrecision::F64RoundedToF32
        } else {
            NumericDisplayPrecision::PortableF32
        };
        let mut texture = self.create_texture(layout.extent, descriptor, label);
        texture.numeric_precision = Some(precision);
        let variant = self.kernel_variant("display_numeric", KernelVariant::Tile16x16)?;
        let pipeline = self.numeric_compute_pipeline(
            DisplayPipelineKey::Numeric {
                native_f64,
                destination: descriptor.format,
                variant,
            },
            label,
            native_f64,
            variant,
        )?;
        self.record_numeric_dispatch(
            encoder,
            &pipeline,
            buffer.as_wgpu_buffer(),
            validated.binding_size,
            texture.view(),
            &validated.params,
            layout.extent,
            native_f64,
            variant,
            "jxl-wgpu numeric display bindings",
        );
        Ok(texture)
    }

    /// Submits [`Self::encode_numeric_image`] without waiting for completion.
    pub fn submit_numeric_image(
        &self,
        source: &GpuImageOutput,
        contract: NumericDisplayContract,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded("jxl-wgpu numeric image display submission", |encoder| {
            self.encode_numeric_image(encoder, source, &submission_guard, contract, descriptor)
        })
    }

    /// Submits [`Self::encode_unvalidated_numeric_image`] without waiting for codec validation.
    pub fn submit_unvalidated_numeric_image(
        &self,
        source: &UnvalidatedGpuImageOutput,
        contract: NumericDisplayContract,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplaySubmission> {
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded(
            "jxl-wgpu unvalidated numeric image display submission",
            |encoder| {
                self.encode_unvalidated_numeric_image(
                    encoder,
                    source,
                    &submission_guard,
                    contract,
                    descriptor,
                )
            },
        )
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
        submission_guard: &GpuBufferSubmissionGuard,
        descriptor: DisplayTextureDescriptor,
    ) -> Result<DisplayTexture> {
        require_submission_guard(submission_guard, &source.buffer)?;
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
        let submission_guard = source.buffer.try_acquire_gpu_submission()?;
        self.submit_encoded("jxl-wgpu RGBA8 copy submission", |encoder| {
            self.encode_rgba8_copy(encoder, source, &submission_guard, descriptor)
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
            numeric_precision: None,
            texture,
            view,
        }
    }

    fn compute_pipeline(
        &self,
        key: DisplayPipelineKey,
        label: &str,
        shader: wgpu::ShaderModuleDescriptor<'static>,
        variant: KernelVariant,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        let mut pipelines = self.pipelines();
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(Arc::clone(pipeline));
        }
        let module = self.device().create_shader_module(shader);
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = Arc::new(self.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            },
        ));
        pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    fn numeric_compute_pipeline(
        &self,
        key: DisplayPipelineKey,
        label: &str,
        native_f64: bool,
        variant: KernelVariant,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        let mut pipelines = self.pipelines();
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(Arc::clone(pipeline));
        }
        let source = numeric_shader_source(native_f64);
        let module = self
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        let constants = [
            ("wg_x", f64::from(workgroup_x)),
            ("wg_y", f64::from(workgroup_y)),
        ];
        let pipeline = Arc::new(self.device().create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            },
        ));
        pipelines.insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    fn kernel_variant(&self, key: &str, default: KernelVariant) -> Result<KernelVariant> {
        let variant = self.inner.kernel_policy.variant_for(key, default)?;
        variant.validate_for(key, &self.device().limits(), 0)?;
        Ok(variant)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_numeric_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        source: &wgpu::Buffer,
        source_binding_size: u64,
        destination: &wgpu::TextureView,
        params: &DisplayNumericParams,
        extent: Extent2d,
        native_f64: bool,
        variant: KernelVariant,
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
        let mut entries = vec![
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
        ];
        if native_f64 {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: source,
                    offset: 0,
                    size: NonZeroU64::new(source_binding_size),
                }),
            });
        }
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        pass.dispatch_workgroups(
            extent.width.div_ceil(workgroup_x),
            extent.height.div_ceil(workgroup_y),
            1,
        );
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
        variant: KernelVariant,
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
        let (workgroup_x, workgroup_y) = variant.workgroup_size();
        pass.dispatch_workgroups(
            extent.width.div_ceil(workgroup_x),
            extent.height.div_ceil(workgroup_y),
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

struct ValidatedNumeric {
    params: DisplayNumericParams,
    binding_size: u64,
    is_f64: bool,
    native_f64_compatible: bool,
}

fn validate_numeric_image(
    layout: &ImageLayout,
    buffer_size: u64,
    contract: NumericDisplayContract,
) -> Result<ValidatedNumeric> {
    if !contract.scale.is_finite() {
        return Err(NumericDisplayError::NonFiniteScale(contract.scale).into());
    }
    if !contract.bias.is_finite() {
        return Err(NumericDisplayError::NonFiniteBias(contract.bias).into());
    }
    if layout.extent.is_empty() {
        return Err(Error::InvalidPayload(
            "numeric display image extent must not be empty".into(),
        ));
    }
    if layout.logical_size == 0 || layout.logical_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "numeric image logical size {} exceeds buffer size {buffer_size}",
            layout.logical_size
        )));
    }
    let canonical =
        ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
    if canonical.logical_size != layout.logical_size {
        return Err(Error::InvalidPayload(
            "numeric image layout logical size is not canonical for its planes".into(),
        ));
    }
    let numeric = match classify_pixel_format(&layout.format) {
        Ok(PixelFormatClass::Numeric(numeric)) => numeric,
        Ok(PixelFormatClass::Color(_)) => return Err(NumericDisplayError::NonNumericFormat.into()),
        Err(error) => {
            return Err(Error::Unsupported(format!(
                "numeric display image format is unsupported: {error}"
            )));
        }
    };
    let source_matches = matches!(
        (contract.source, numeric.sample_kind),
        (NumericDisplaySource::Unsigned, SampleKind::Unsigned)
            | (NumericDisplaySource::Signed, SampleKind::Signed)
            | (NumericDisplaySource::Floating { .. }, SampleKind::Float)
    );
    if !source_matches {
        return Err(NumericDisplayError::SourceKindMismatch {
            declared: contract.source,
            actual: numeric.sample_kind,
        }
        .into());
    }
    if numeric.components != 2
        && matches!(
            contract.channels,
            NumericDisplayChannels::LumaAlpha | NumericDisplayChannels::RedGreen
        )
    {
        return Err(NumericDisplayError::TwoComponentsRequired {
            mapping: contract.channels,
            components: numeric.components,
        }
        .into());
    }
    let plane = layout
        .planes
        .first()
        .ok_or(NumericDisplayError::NonNumericFormat)?;
    validate_image_plane_range(plane, layout.logical_size, buffer_size)?;
    let binding_size = align_to_word(layout.logical_size)?;
    if binding_size > buffer_size {
        return Err(Error::InvalidPayload(format!(
            "numeric image buffer size {buffer_size} does not include word padding through {binding_size}"
        )));
    }
    let sample_kind = match numeric.sample_kind {
        SampleKind::Unsigned => 0,
        SampleKind::Signed => 1,
        SampleKind::Float => 2,
    };
    let visualization = match contract.channels {
        NumericDisplayChannels::Luma => 0,
        NumericDisplayChannels::LumaAlpha => 1,
        NumericDisplayChannels::RedGreen => 2,
    };
    let non_finite = match contract.source {
        NumericDisplaySource::Floating {
            non_finite: NumericNonFinitePolicy::Saturate,
        }
        | NumericDisplaySource::Unsigned
        | NumericDisplaySource::Signed => 1,
        NumericDisplaySource::Floating {
            non_finite: NumericNonFinitePolicy::Zero,
        } => 0,
    };
    let is_f64 = numeric.sample_kind == SampleKind::Float && numeric.bits_per_component == 64;
    Ok(ValidatedNumeric {
        params: DisplayNumericParams {
            width: layout.extent.width,
            height: layout.extent.height,
            sample_kind,
            bits: u32::from(numeric.bits_per_component),
            components: u32::from(numeric.components),
            plane_offset: shader_u32(plane.offset, "numeric plane offset")?,
            plane_stride: shader_u32(plane.row_stride, "numeric plane row stride")?,
            visualization,
            non_finite,
            transfer: u32::from(contract.transfer == NumericDisplayTransfer::Srgb),
            clamp: 0,
            _reserved: 0,
            scale: contract.scale,
            bias: contract.bias,
            _padding: [0; 2],
        },
        binding_size,
        is_f64,
        native_f64_compatible: is_f64 && plane.offset % 8 == 0 && plane.row_stride % 8 == 0,
    })
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

fn require_submission_guard(
    guard: &GpuBufferSubmissionGuard,
    buffer: &crate::GpuBufferLease,
) -> Result<()> {
    if guard.protects(buffer) {
        Ok(())
    } else {
        Err(Error::InvalidPayload(
            "display submission guard belongs to a different GPU buffer".into(),
        ))
    }
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

const PORTABLE_F64_NORMALIZATION: &str = r#"
fn f64_storage_to_f32(offset: u32) -> f32 {
    let low = read_u32(offset);
    let high = read_u32(offset + 4u);
    let negative = (high & 0x80000000u) != 0u;
    let exponent = (high >> 20u) & 0x7ffu;
    let fraction_high = high & 0x000fffffu;
    if exponent == 0x7ffu {
        if fraction_high != 0u || low != 0u { return bitcast<f32>(0x7fc00000u); }
        return bitcast<f32>(select(0x7f800000u, 0xff800000u, negative));
    }
    if exponent == 0u {
        return bitcast<f32>(select(0u, 0x80000000u, negative));
    }
    let unbiased = i32(exponent) - 1023i;
    if unbiased > 127i {
        return bitcast<f32>(select(0x7f800000u, 0xff800000u, negative));
    }
    if unbiased < -149i {
        return bitcast<f32>(select(0u, 0x80000000u, negative));
    }
    let fraction = f32(fraction_high) * 9.5367431640625e-7
        + f32(low) * 2.220446049250313e-16;
    let magnitude = ldexp(1.0 + fraction, unbiased);
    return select(magnitude, -magnitude, negative);
}

fn normalized_f64(offset: u32) -> f32 {
    return f64_storage_to_f32(offset) * params.scale + params.bias;
}
"#;

const NATIVE_F64_NORMALIZATION: &str = r#"
@group(0) @binding(3) var<storage, read> source_f64: array<f64>;

fn normalized_f64(offset: u32) -> f32 {
    let value = source_f64[offset >> 3u];
    return f32(value * f64(params.scale) + f64(params.bias));
}
"#;

fn numeric_shader_source(native_f64: bool) -> String {
    let implementation = if native_f64 {
        NATIVE_F64_NORMALIZATION
    } else {
        PORTABLE_F64_NORMALIZATION
    };
    include_str!("../shaders/display_numeric.wgsl").replace(NUMERIC_F64_MARKER, implementation)
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
struct DisplayNumericParams {
    width: u32,
    height: u32,
    sample_kind: u32,
    bits: u32,
    components: u32,
    plane_offset: u32,
    plane_stride: u32,
    visualization: u32,
    non_finite: u32,
    transfer: u32,
    clamp: u32,
    _reserved: u32,
    scale: f32,
    bias: f32,
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
    assert!(std::mem::size_of::<DisplayNumericParams>() == 64);
    assert!(std::mem::align_of::<DisplayNumericParams>() == 4);
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
        let module = naga::front::wgsl::parse_str(shader).expect("WGSL parses");
        let ty = module
            .types
            .iter()
            .map(|(_, ty)| ty)
            .find(|ty| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is missing"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("WGSL type '{name}' is not a struct");
        };
        let actual = members
            .iter()
            .map(|member| member.name.as_deref().expect("WGSL struct member is named"))
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
    fn explicit_numeric_contract_validates_all_ten_vpi_numeric_formats() {
        let numeric = [
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::U8,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::S8,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::U16,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::U32,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::S32,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::S16,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::TwoS16,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::F32,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::F64,
            jxl_gpu_formats::vpi::VpiPitchLinearFormat::TwoF32,
        ];
        for predefined in numeric {
            let format = predefined.pixel_format();
            let class = classify_pixel_format(&format).unwrap().numeric().unwrap();
            let contract = NumericDisplayContract {
                source: match class.sample_kind {
                    SampleKind::Unsigned => NumericDisplaySource::Unsigned,
                    SampleKind::Signed => NumericDisplaySource::Signed,
                    SampleKind::Float => NumericDisplaySource::Floating {
                        non_finite: NumericNonFinitePolicy::Saturate,
                    },
                },
                scale: 1.0,
                bias: 0.0,
                channels: NumericDisplayChannels::Luma,
                transfer: NumericDisplayTransfer::Linear,
                clamp: NumericDisplayClamp::Unit,
            };
            let layout = ImageLayout::packed(Extent2d::new(3, 3), format).unwrap();
            let validated = validate_numeric_image(
                &layout,
                align_to_word(layout.logical_size).unwrap(),
                contract,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", predefined.name()));
            assert_eq!(validated.params.bits, u32::from(class.bits_per_component));
            assert_eq!(validated.params.components, u32::from(class.components));
            assert_eq!(
                validated.is_f64,
                predefined == jxl_gpu_formats::vpi::VpiPitchLinearFormat::F64
            );
            assert_eq!(validated.native_f64_compatible, validated.is_f64);
        }
    }

    #[test]
    fn numeric_contract_mismatches_and_non_finite_affine_terms_are_typed() {
        let format = jxl_gpu_formats::vpi::VpiPitchLinearFormat::S16.pixel_format();
        let layout = ImageLayout::packed(Extent2d::new(2, 2), format).unwrap();
        let base = NumericDisplayContract {
            source: NumericDisplaySource::Signed,
            scale: 1.0,
            bias: 0.0,
            channels: NumericDisplayChannels::Luma,
            transfer: NumericDisplayTransfer::Linear,
            clamp: NumericDisplayClamp::Unit,
        };
        let buffer_size = align_to_word(layout.logical_size).unwrap();
        assert_eq!(
            validate_numeric_image(&layout, buffer_size, base)
                .unwrap()
                .params
                .non_finite,
            1,
            "integer affine overflow follows unit-clamp saturation"
        );
        assert!(matches!(
            validate_numeric_image(
                &layout,
                buffer_size,
                NumericDisplayContract {
                    source: NumericDisplaySource::Unsigned,
                    ..base
                }
            ),
            Err(Error::NumericDisplay(
                NumericDisplayError::SourceKindMismatch { .. }
            ))
        ));
        assert!(matches!(
            validate_numeric_image(
                &layout,
                buffer_size,
                NumericDisplayContract {
                    scale: f32::INFINITY,
                    ..base
                }
            ),
            Err(Error::NumericDisplay(
                NumericDisplayError::NonFiniteScale(value)
            )) if value == f32::INFINITY
        ));
    }

    #[test]
    fn unaligned_f64_plane_uses_the_reported_portable_path() {
        let extent = Extent2d::new(2, 2);
        let format = jxl_gpu_formats::vpi::VpiPitchLinearFormat::F64.pixel_format();
        let layout = ImageLayout::from_planes(
            extent,
            format,
            vec![jxl_gpu_formats::PitchLinearPlaneLayout {
                plane_index: 0,
                offset: 4,
                row_stride: 20,
                sample_extent: extent,
                row_bytes: 16,
            }],
        )
        .unwrap();
        let validated = validate_numeric_image(
            &layout,
            layout.logical_size,
            NumericDisplayContract {
                source: NumericDisplaySource::Floating {
                    non_finite: NumericNonFinitePolicy::Saturate,
                },
                scale: 1.0,
                bias: 0.0,
                channels: NumericDisplayChannels::Luma,
                transfer: NumericDisplayTransfer::Linear,
                clamp: NumericDisplayClamp::Unit,
            },
        )
        .unwrap();
        assert!(validated.is_f64);
        assert!(!validated.native_f64_compatible);
    }

    #[test]
    fn numeric_shader_variants_validate_with_exact_naga_capabilities() {
        for (native_f64, capabilities) in [
            (false, naga::valid::Capabilities::empty()),
            (true, naga::valid::Capabilities::FLOAT64),
        ] {
            let module = naga::front::wgsl::parse_str(&numeric_shader_source(native_f64))
                .unwrap_or_else(|error| panic!("numeric WGSL parse failed: {error}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
                .validate(&module)
                .unwrap_or_else(|error| panic!("numeric WGSL validation failed: {error}"));
        }
    }

    #[test]
    fn display_uniform_abi_sizes_are_explicit_and_aligned() {
        assert_eq!(size_of::<DisplayRgbParams>(), 32);
        assert_eq!(size_of::<DisplayNumericParams>(), 64);
        assert_eq!(size_of::<DisplayImageParams>(), 96);
        assert_eq!(std::mem::align_of::<DisplayRgbParams>(), 4);
        assert_eq!(std::mem::align_of::<DisplayNumericParams>(), 4);
        assert_eq!(std::mem::align_of::<DisplayImageParams>(), 4);
        assert_eq!(size_of::<DisplayRgbParams>() % 16, 0);
        assert_eq!(size_of::<DisplayNumericParams>() % 16, 0);
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

        let numeric = DisplayNumericParams {
            width: 1,
            height: 2,
            sample_kind: 3,
            bits: 4,
            components: 5,
            plane_offset: 6,
            plane_stride: 7,
            visualization: 8,
            non_finite: 9,
            transfer: 10,
            clamp: 11,
            _reserved: 12,
            scale: f32::from_bits(13),
            bias: f32::from_bits(14),
            _padding: [15, 16],
        };
        assert_eq!(abi_words(&numeric), (1..=16).collect::<Vec<_>>());
        assert_wgsl_fields(
            &numeric_shader_source(false),
            "NumericParams",
            &[
                "width",
                "height",
                "sample_kind",
                "bits",
                "components",
                "plane_offset",
                "plane_stride",
                "visualization",
                "non_finite",
                "transfer",
                "clamp",
                "_reserved",
                "scale",
                "bias",
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
            variant: KernelVariant::Tile16x16,
        };
        let matrix = DisplayPipelineKey::Image {
            format: DisplayImageFormat {
                matrix: 2,
                ..format
            },
            destination: wgpu::TextureFormat::Rgba8Unorm,
            variant: KernelVariant::Tile16x16,
        };
        let range = DisplayPipelineKey::Image {
            format: DisplayImageFormat { range: 0, ..format },
            destination: wgpu::TextureFormat::Rgba8Unorm,
            variant: KernelVariant::Tile16x16,
        };
        let planar = DisplayPipelineKey::Image {
            format: DisplayImageFormat { kind: 4, ..format },
            destination: wgpu::TextureFormat::Rgba8Unorm,
            variant: KernelVariant::Tile16x16,
        };
        let transfer = DisplayPipelineKey::Image {
            format: DisplayImageFormat {
                transfer: 1,
                ..format
            },
            destination: wgpu::TextureFormat::Rgba8Unorm,
            variant: KernelVariant::Tile16x16,
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
