//! Portable image planes carried by independent uncompressed GPU textures.
use std::sync::Arc;

use jxl_gpu_formats::PixelFormat;
use jxl_gpu_protocol::Extent2d;

use crate::{BufferImageSource, CmykSampleEncoding, EncodeError};

/// One complete mip/layer supplying the raw texels of one logical image plane.
/// The texture must be single-sample 2D color storage with `COPY_SRC` usage.
#[derive(Clone, Debug)]
pub struct TexturePlaneSource {
    pub texture: Arc<wgpu::Texture>,
    pub texture_format: wgpu::TextureFormat,
    pub mip_level: u32,
    pub array_layer: u32,
}

impl TexturePlaneSource {
    pub fn new(
        texture: Arc<wgpu::Texture>,
        texture_format: wgpu::TextureFormat,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<Self, EncodeError> {
        let source = Self {
            texture,
            texture_format,
            mip_level,
            array_layer,
        };
        source.geometry()?;
        Ok(source)
    }

    pub(crate) fn geometry(&self) -> Result<(Extent2d, u32), EncodeError> {
        let texture = &self.texture;
        let format = self.texture_format;
        if texture.format() != format
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.sample_count() != 1
            || !texture.usage().contains(wgpu::TextureUsages::COPY_SRC)
            || self.mip_level >= texture.mip_level_count()
            || self.array_layer >= texture.depth_or_array_layers()
            || format.is_depth_stencil_format()
            || format.is_multi_planar_format()
            || format.block_dimensions() != (1, 1)
        {
            return Err(EncodeError::InvalidSource(
                "texture input requires a matching copyable single-sample 2D color mip/layer",
            ));
        }
        let bytes = format
            .block_copy_size(None)
            .ok_or(EncodeError::InvalidSource(
                "texture has no portable color copy layout",
            ))?;
        Ok((
            Extent2d::new(
                (texture.width() >> self.mip_level).max(1),
                (texture.height() >> self.mip_level).max(1),
            ),
            bytes,
        ))
    }
}

/// One texture subresource for each plane in `pixel_format`, in format order.
///
/// `extent` is the logical full image size. Each selected mip must equal its plane's
/// sampled extent, with width rounded up in units of `pixels_per_element`. One texel
/// carries exactly one packing element. No sampling or texture color conversion occurs.
/// Native multi-planar texture formats are not portable plane carriers; use separate textures.
#[derive(Clone, Debug)]
pub struct TexturePlanesSource {
    pub extent: Extent2d,
    pub pixel_format: PixelFormat,
    pub planes: Vec<TexturePlaneSource>,
    pub(crate) extra_channels: Vec<BufferImageSource>,
    pub(crate) cmyk_encoding: CmykSampleEncoding,
}

impl TexturePlanesSource {
    pub fn new(
        extent: Extent2d,
        pixel_format: PixelFormat,
        planes: Vec<TexturePlaneSource>,
    ) -> Result<Self, EncodeError> {
        let source = Self {
            extent,
            pixel_format,
            planes,
            extra_channels: Vec::new(),
            cmyk_encoding: Default::default(),
        };
        crate::source_storage::StoragePlan::new(source.clone().into())?;
        Ok(source)
    }

    pub fn with_extra_channels(
        mut self,
        channels: Vec<BufferImageSource>,
    ) -> Result<Self, EncodeError> {
        crate::source_storage::validate_extra_channels(&channels)?;
        self.extra_channels = channels;
        Ok(self)
    }

    pub fn with_cmyk_encoding(mut self, encoding: CmykSampleEncoding) -> Result<Self, EncodeError> {
        crate::source_storage::validate_cmyk_format(&self.pixel_format)?;
        self.cmyk_encoding = encoding;
        Ok(self)
    }

    #[must_use]
    pub fn extra_channels(&self) -> &[BufferImageSource] {
        &self.extra_channels
    }

    #[must_use]
    pub fn cmyk_encoding(&self) -> CmykSampleEncoding {
        self.cmyk_encoding
    }
}
