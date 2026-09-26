//! Checked physical storage, before optional image-domain conversion.
use std::sync::Arc;

use jxl_gpu_formats::{ImageLayout, PixelFormat, PlaneSampling};
use jxl_gpu_protocol::Extent2d;

use crate::{
    BufferImageSource, CmykSampleEncoding, EncodeError, GpuFrameSource, TextureImageSource,
    TexturePlaneSource, TexturePlanesSource,
};

/// Caller-supplied storage before any explicit color conversion.
#[derive(Clone, Debug)]
pub enum ImageSourceStorage {
    Buffer(BufferImageSource),
    Texture(TextureImageSource),
    TexturePlanes(TexturePlanesSource),
}

impl ImageSourceStorage {
    #[must_use]
    pub fn as_buffer(&self) -> Option<&BufferImageSource> {
        match self {
            Self::Buffer(source) => Some(source),
            _ => None,
        }
    }

    #[must_use]
    pub fn pixel_format(&self) -> &PixelFormat {
        match self {
            Self::Buffer(source) => &source.layout.format,
            Self::Texture(source) => &source.pixel_format,
            Self::TexturePlanes(source) => &source.pixel_format,
        }
    }

    #[must_use]
    pub fn extra_channels(&self) -> &[BufferImageSource] {
        match self {
            Self::Buffer(source) => source.extra_channels(),
            Self::Texture(source) => source.extra_channels(),
            Self::TexturePlanes(source) => source.extra_channels(),
        }
    }

    #[must_use]
    pub fn cmyk_encoding(&self) -> CmykSampleEncoding {
        match self {
            Self::Buffer(source) => source.cmyk_encoding(),
            Self::Texture(source) => source.cmyk_encoding(),
            Self::TexturePlanes(source) => source.cmyk_encoding(),
        }
    }

    pub fn with_extra_channels(
        self,
        channels: Vec<BufferImageSource>,
    ) -> Result<Self, EncodeError> {
        Ok(match self {
            Self::Buffer(source) => source.with_extra_channels(channels)?.into(),
            Self::Texture(source) => source.with_extra_channels(channels)?.into(),
            Self::TexturePlanes(source) => source.with_extra_channels(channels)?.into(),
        })
    }
}

impl From<BufferImageSource> for ImageSourceStorage {
    fn from(source: BufferImageSource) -> Self {
        Self::Buffer(source)
    }
}
impl From<TextureImageSource> for ImageSourceStorage {
    fn from(source: TextureImageSource) -> Self {
        Self::Texture(source)
    }
}
impl From<TexturePlanesSource> for ImageSourceStorage {
    fn from(source: TexturePlanesSource) -> Self {
        Self::TexturePlanes(source)
    }
}
impl From<&BufferImageSource> for ImageSourceStorage {
    fn from(source: &BufferImageSource) -> Self {
        source.clone().into()
    }
}
impl From<&TextureImageSource> for ImageSourceStorage {
    fn from(source: &TextureImageSource) -> Self {
        source.clone().into()
    }
}
impl From<&TexturePlanesSource> for ImageSourceStorage {
    fn from(source: &TexturePlanesSource) -> Self {
        source.clone().into()
    }
}
impl From<ImageSourceStorage> for GpuFrameSource {
    fn from(source: ImageSourceStorage) -> Self {
        match source {
            ImageSourceStorage::Buffer(source) => Self::Buffer(source),
            ImageSourceStorage::Texture(source) => Self::Texture(source),
            ImageSourceStorage::TexturePlanes(source) => Self::TexturePlanes(source),
        }
    }
}

pub(crate) fn validate_extra_channels(channels: &[BufferImageSource]) -> Result<(), EncodeError> {
    if channels.len() > crate::extra_channel::MAX_EXTRA_CHANNELS
        || channels
            .iter()
            .any(|channel| !channel.extra_channels().is_empty())
    {
        return Err(EncodeError::InvalidSource(
            "extra sources must be flat and within the JPEG XL channel count",
        ));
    }
    Ok(())
}

pub(crate) fn validate_cmyk_format(format: &PixelFormat) -> Result<(), EncodeError> {
    if !matches!(&format.color_spec, jxl_gpu_formats::ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
        || format.model != jxl_gpu_formats::ColorModel::IccDevice
    {
        return Err(EncodeError::InvalidSource(
            "CMYK sample convention requires a CMYK ICC input",
        ));
    }
    Ok(())
}

pub(crate) struct StoragePlan {
    source: ImageSourceStorage,
    pub(crate) layout: ImageLayout,
    pub(crate) copy_bytes: u64,
    pub(crate) texture_bytes: u64,
    copies: Vec<TextureCopy>,
}

impl StoragePlan {
    pub(crate) fn new(source: ImageSourceStorage) -> Result<Self, EncodeError> {
        let (layout, copies, texture_bytes) = match &source {
            ImageSourceStorage::Buffer(source) => (source.layout.clone(), Vec::new(), 0),
            ImageSourceStorage::Texture(source) => {
                let plane = TexturePlaneSource {
                    texture: source.texture.clone(),
                    texture_format: source.texture_format,
                    mip_level: source.mip_level,
                    array_layer: source.array_layer,
                };
                let (extent, _) = plane.geometry()?;
                let format = &source.pixel_format;
                if format.planes.len() != 1
                    || format.planes[0].sampling != PlaneSampling::FULL
                    || format.planes[0].pixels_per_element != 1
                {
                    return Err(EncodeError::InvalidSource(
                        "pixel format must describe exactly one copied texel",
                    ));
                }
                texture_layout(extent, format, std::slice::from_ref(&plane))?
            }
            ImageSourceStorage::TexturePlanes(source) => {
                texture_layout(source.extent, &source.pixel_format, &source.planes)?
            }
        };
        let copy_bytes = if copies.is_empty() {
            0
        } else {
            align(layout.logical_size, 4)?
        };
        Ok(Self {
            source,
            layout,
            copy_bytes,
            texture_bytes,
            copies,
        })
    }

    pub(crate) fn source(&self) -> &ImageSourceStorage {
        &self.source
    }

    pub(crate) fn caller_buffer(&self) -> Option<&wgpu::Buffer> {
        self.source.as_buffer().map(|source| source.buffer.as_ref())
    }

    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.caller_buffer()
            .map_or(self.copy_bytes, wgpu::Buffer::size)
    }

    pub(crate) fn buffer_usage(&self) -> wgpu::BufferUsages {
        self.caller_buffer()
            .map_or(wgpu::BufferUsages::STORAGE, wgpu::Buffer::usage)
    }

    pub(crate) fn materialize(
        self,
        device: &wgpu::Device,
    ) -> (BufferImageSource, Option<PreparedTextureCopies>) {
        if let ImageSourceStorage::Buffer(source) = self.source {
            return (source, None);
        }
        let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jxl-wgpu texture plane copies"),
            size: self.copy_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let mut source =
            BufferImageSource::new(buffer.clone(), self.layout).expect("checked input allocation");
        source.extra_channels = self.source.extra_channels().to_vec();
        source.cmyk_encoding = self.source.cmyk_encoding();
        (
            source,
            Some(PreparedTextureCopies {
                buffer,
                copies: self.copies,
            }),
        )
    }
}

fn align(value: u64, alignment: u64) -> Result<u64, EncodeError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(EncodeError::InvalidSource("texture copy size overflow"))
}

fn texture_layout(
    extent: Extent2d,
    format: &PixelFormat,
    sources: &[TexturePlaneSource],
) -> Result<(ImageLayout, Vec<TextureCopy>, u64), EncodeError> {
    let mut layout = ImageLayout::packed(extent, format.clone())?;
    if sources.len() != layout.planes.len() {
        return Err(EncodeError::InvalidSource(
            "texture plane count differs from the pixel format",
        ));
    }
    let mut copies = Vec::with_capacity(sources.len());
    let mut end = 0;
    let mut texture_bytes = 0u64;
    for (index, ((plane, packing), source)) in layout
        .planes
        .iter_mut()
        .zip(&format.planes)
        .zip(sources)
        .enumerate()
    {
        let (actual, texel_bytes) = source.geometry()?;
        let required = Extent2d::new(
            plane
                .sample_extent
                .width
                .div_ceil(u32::from(packing.pixels_per_element)),
            plane.sample_extent.height,
        );
        if actual != required || packing.bits_per_element() != u64::from(texel_bytes) * 8 {
            return Err(EncodeError::InvalidSource(
                "texture texel geometry or width differs from its logical plane packing",
            ));
        }
        plane.offset = align(end, u64::from(texel_bytes).max(4))?;
        plane.row_stride = align(
            plane.row_bytes,
            u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
        )?;
        let pitch = u32::try_from(plane.row_stride)
            .map_err(|_| EncodeError::InvalidSource("texture copy row pitch exceeds u32"))?;
        end = plane.end_offset()?;
        if !sources[..index].iter().any(|known| {
            known.texture == source.texture
                && known.mip_level == source.mip_level
                && known.array_layer == source.array_layer
        }) {
            texture_bytes = plane
                .row_bytes
                .checked_mul(u64::from(actual.height))
                .and_then(|bytes| texture_bytes.checked_add(bytes))
                .ok_or(EncodeError::InvalidSource("texture input size overflow"))?;
        }
        copies.push(TextureCopy {
            source: source.clone(),
            extent: actual,
            offset: plane.offset,
            pitch,
        });
    }
    let layout = ImageLayout::from_planes(extent, format.clone(), layout.planes)?;
    Ok((layout, copies, texture_bytes))
}

struct TextureCopy {
    source: TexturePlaneSource,
    extent: Extent2d,
    offset: u64,
    pitch: u32,
}

pub(crate) struct PreparedTextureCopies {
    buffer: Arc<wgpu::Buffer>,
    copies: Vec<TextureCopy>,
}

impl PreparedTextureCopies {
    pub(crate) fn record(&self, commands: &mut wgpu::CommandEncoder) {
        for copy in &self.copies {
            commands.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &copy.source.texture,
                    mip_level: copy.source.mip_level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: copy.source.array_layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: copy.offset,
                        bytes_per_row: Some(copy.pitch),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: copy.extent.width,
                    height: copy.extent.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}
