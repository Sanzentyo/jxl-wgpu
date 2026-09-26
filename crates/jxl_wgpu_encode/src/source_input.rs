//! Checked physical input preparation, independent of codec and frame sampling.
use std::ops::Deref;
use std::sync::Arc;

use jxl_gpu_formats::{ImageLayout, PlaneSampling};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::MemoryPermit;

use crate::{
    BufferImageSource, CmykSampleEncoding, EncodeError, GpuFrameSource, UnsupportedFeature,
};

/// Plans storage without allocation or submission. All consumers use this checked layout;
/// only admission may materialize it. Scalar attachments remain caller-owned buffers.
pub(crate) struct FrameInputPlan {
    source: GpuFrameSource,
    pub(crate) layout: ImageLayout,
    pub(crate) copy_bytes: u64,
    pub(crate) texture_bytes: u64,
}

impl FrameInputPlan {
    pub(crate) fn new(source: GpuFrameSource) -> Result<Self, EncodeError> {
        let (layout, copy_bytes, texture_bytes) = match &source {
            GpuFrameSource::Buffer(buffer) => (buffer.layout.clone(), 0, 0),
            GpuFrameSource::Texture(input) => {
                let texture = &input.texture;
                let format = input.texture_format;
                if texture.format() != format
                    || texture.dimension() != wgpu::TextureDimension::D2
                    || texture.sample_count() != 1
                    || !texture.usage().contains(wgpu::TextureUsages::COPY_SRC)
                    || input.mip_level >= texture.mip_level_count()
                    || input.array_layer >= texture.depth_or_array_layers()
                    || format.is_depth_stencil_format()
                    || format.is_multi_planar_format()
                    || format.block_dimensions() != (1, 1)
                {
                    return Err(EncodeError::InvalidSource(
                        "texture input requires a matching copyable single-sample 2D color mip/layer",
                    ));
                }
                let texel_bytes =
                    format
                        .block_copy_size(None)
                        .ok_or(EncodeError::InvalidSource(
                            "texture has no portable color copy layout",
                        ))?;
                let pixel = &input.pixel_format;
                pixel
                    .validate()
                    .map_err(jxl_gpu_formats::LayoutError::from)?;
                if pixel.planes.len() != 1
                    || pixel.planes[0].sampling != PlaneSampling::FULL
                    || pixel.planes[0].pixels_per_element != 1
                    || pixel.planes[0].bits_per_element() != u64::from(texel_bytes) * 8
                {
                    return Err(EncodeError::InvalidSource(
                        "pixel format must describe exactly one copied texel",
                    ));
                }
                let extent = Extent2d::new(
                    (texture.width() >> input.mip_level).max(1),
                    (texture.height() >> input.mip_level).max(1),
                );
                let mut layout = ImageLayout::packed(extent, pixel.clone())?;
                let row_bytes = layout.planes[0].row_bytes;
                let row_stride = row_bytes.div_ceil(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
                    * u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
                u32::try_from(row_stride).map_err(|_| {
                    EncodeError::InvalidSource("texture copy row pitch exceeds u32")
                })?;
                layout.planes[0].row_stride = row_stride;
                let layout = ImageLayout::from_planes(extent, pixel.clone(), layout.planes)?;
                let copy_bytes = layout
                    .logical_size
                    .checked_add(3)
                    .ok_or(EncodeError::InvalidSource("texture copy size overflow"))?
                    & !3;
                let texture_bytes = row_bytes
                    .checked_mul(u64::from(extent.height))
                    .ok_or(EncodeError::InvalidSource("texture input size overflow"))?;
                (layout, copy_bytes, texture_bytes)
            }
        };
        Ok(Self {
            source,
            layout,
            copy_bytes,
            texture_bytes,
        })
    }

    pub(crate) fn validate_limits(&self, max_buffer_size: u64) -> Result<(), EncodeError> {
        if self.copy_bytes > max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: self.copy_bytes,
                available: max_buffer_size,
            }
            .into());
        }
        Ok(())
    }

    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.caller_buffer()
            .map_or(self.copy_bytes, wgpu::Buffer::size)
    }

    pub(crate) fn buffer_usage(&self) -> wgpu::BufferUsages {
        self.caller_buffer()
            .map_or(wgpu::BufferUsages::STORAGE, wgpu::Buffer::usage)
    }

    /// None denotes the encoder-owned copy, excluded from caller-buffer accounting.
    pub(crate) fn caller_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.source {
            GpuFrameSource::Buffer(source) => Some(&source.buffer),
            GpuFrameSource::Texture(_) => None,
        }
    }

    pub(crate) fn extra_channels(&self) -> &[BufferImageSource] {
        match &self.source {
            GpuFrameSource::Buffer(source) => source.extra_channels(),
            GpuFrameSource::Texture(source) => source.extra_channels(),
        }
    }

    pub(crate) fn cmyk_encoding(&self) -> CmykSampleEncoding {
        match &self.source {
            GpuFrameSource::Buffer(source) => source.cmyk_encoding(),
            GpuFrameSource::Texture(source) => source.cmyk_encoding(),
        }
    }

    pub(crate) fn into_source(self) -> GpuFrameSource {
        self.source
    }

    /// The caller has reserved the complete resident peak before entering here. Streaming
    /// separately retains a persistent copy permit while individual batches reserve scratch.
    pub(crate) fn materialize(
        self,
        device: &wgpu::Device,
        permit: Option<MemoryPermit>,
    ) -> Arc<PreparedInput> {
        let (source, texture) = match self.source {
            GpuFrameSource::Buffer(source) => (source, None),
            GpuFrameSource::Texture(texture) => {
                let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jxl-wgpu texture input copy"),
                    size: self.copy_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }));
                let mut source =
                    BufferImageSource::new(buffer, self.layout).expect("checked input allocation");
                source.extra_channels = texture.extra_channels.clone();
                source.cmyk_encoding = texture.cmyk_encoding;
                (source, Some(texture))
            }
        };
        Arc::new(PreparedInput {
            source,
            texture,
            _permit: permit,
        })
    }
}

impl From<&FrameInputPlan> for GpuFrameSource {
    fn from(plan: &FrameInputPlan) -> Self {
        plan.source.clone()
    }
}

/// Shared with completion callbacks, including the last pending batch after cancellation.
pub(crate) struct PreparedInput {
    source: BufferImageSource,
    texture: Option<crate::TextureImageSource>,
    _permit: Option<MemoryPermit>,
}

impl Deref for PreparedInput {
    type Target = BufferImageSource;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl PreparedInput {
    /// Record once, before the first compute pass, on the job's ordinary submission.
    pub(crate) fn record_copy(&self, commands: &mut wgpu::CommandEncoder) {
        let Some(source) = &self.texture else {
            return;
        };
        commands.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &source.texture,
                mip_level: source.mip_level,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: source.array_layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.layout.planes[0].row_stride as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.layout.extent.width,
                height: self.layout.extent.height,
                depth_or_array_layers: 1,
            },
        );
    }
}
