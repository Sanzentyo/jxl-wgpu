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
    conversion: Option<crate::yuv_input::YuvPlan>,
}

impl FrameInputPlan {
    pub(crate) fn determinism(&self) -> crate::Determinism {
        if self.conversion.is_some() {
            crate::Determinism::SameDevice
        } else {
            crate::Determinism::CrossDevice
        }
    }

    pub(crate) fn validate_request(
        &self,
        request: &crate::FrameEncodeRequest,
    ) -> Result<(), EncodeError> {
        if request.minimum_determinism > self.determinism() {
            return Err(UnsupportedFeature::InputDeterminism {
                requested: request.minimum_determinism,
                supported: self.determinism(),
            }
            .into());
        }
        Ok(())
    }
    pub(crate) fn new(source: GpuFrameSource) -> Result<Self, EncodeError> {
        let mut conversion = None;
        let (layout, copy_bytes, texture_bytes) = match &source {
            GpuFrameSource::Buffer(buffer) => (buffer.layout.clone(), 0, 0),
            GpuFrameSource::Yuv(input) => {
                let plan = crate::yuv_input::YuvPlan::new(input.source(), input.rgb_transfer())?;
                let layout = plan.layout.clone();
                conversion = Some(plan);
                (layout, 0, 0)
            }
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
            conversion,
        })
    }

    pub(crate) fn validate_limits(
        &self,
        max_buffer_size: u64,
        limits: &wgpu::Limits,
    ) -> Result<(), EncodeError> {
        let allocation = self.copy_bytes.max(
            self.conversion
                .as_ref()
                .map_or(0, crate::yuv_input::YuvPlan::largest_allocation),
        );
        if allocation > max_buffer_size {
            return Err(UnsupportedFeature::DeviceLimit {
                name: "max_buffer_size",
                required: allocation,
                available: max_buffer_size,
            }
            .into());
        }
        if let Some(plan) = &self.conversion {
            plan.validate_limits(limits)?;
        }
        Ok(())
    }

    pub(crate) fn conversion_bytes(&self) -> u64 {
        self.conversion
            .as_ref()
            .map_or(0, crate::yuv_input::YuvPlan::owned_bytes)
    }

    pub(crate) fn owned_bytes(&self) -> u64 {
        self.copy_bytes + self.conversion_bytes()
    }

    pub(crate) fn preparation_binding(
        &self,
    ) -> Option<(Option<&wgpu::Buffer>, crate::source::SourceWindows)> {
        let GpuFrameSource::Yuv(source) = &self.source else {
            return None;
        };
        let plan = self.conversion.as_ref().expect("YUV preparation plan");
        Some((
            Some(&source.source().buffer),
            crate::source::SourceWindows::prefix(plan.source_bytes),
        ))
    }

    pub(crate) fn validate_alpha_association(
        &self,
        association: crate::AlphaAssociation,
    ) -> Result<(), EncodeError> {
        if association == crate::AlphaAssociation::Associated
            && self
                .conversion
                .as_ref()
                .is_some_and(crate::yuv_input::YuvPlan::requires_unassociated_alpha)
        {
            return Err(EncodeError::InvalidSource(
                "nonlinear YUV conversion requires unassociated alpha",
            ));
        }
        if association == crate::AlphaAssociation::Associated
            && self.cmyk_encoding() == CmykSampleEncoding::InkAmounts
            && matches!(&self.layout.format.color_spec, jxl_gpu_formats::ColorSpecification::Icc(profile) if profile.header().device_space.0 == *b"CMYK")
        {
            return Err(EncodeError::InvalidSource(
                "associated CMYK input requires explicit complemented samples",
            ));
        }
        Ok(())
    }

    pub(crate) fn buffer_bytes(&self) -> u64 {
        self.caller_buffer().map_or_else(
            || {
                self.conversion
                    .as_ref()
                    .map_or(self.copy_bytes, |plan| plan.layout.logical_size)
            },
            wgpu::Buffer::size,
        )
    }

    pub(crate) fn buffer_usage(&self) -> wgpu::BufferUsages {
        self.caller_buffer()
            .map_or(wgpu::BufferUsages::STORAGE, wgpu::Buffer::usage)
    }

    /// None denotes encoder-owned prepared RGB/texels, excluded from caller-buffer accounting.
    pub(crate) fn caller_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.source {
            GpuFrameSource::Buffer(source) => Some(&source.buffer),
            GpuFrameSource::Texture(_) | GpuFrameSource::Yuv(_) => None,
        }
    }

    pub(crate) fn extra_channels(&self) -> &[BufferImageSource] {
        match &self.source {
            GpuFrameSource::Buffer(source) => source.extra_channels(),
            GpuFrameSource::Texture(source) => source.extra_channels(),
            GpuFrameSource::Yuv(source) => source.source().extra_channels(),
        }
    }

    pub(crate) fn cmyk_encoding(&self) -> CmykSampleEncoding {
        match &self.source {
            GpuFrameSource::Buffer(source) => source.cmyk_encoding(),
            GpuFrameSource::Texture(source) => source.cmyk_encoding(),
            GpuFrameSource::Yuv(_) => CmykSampleEncoding::default(),
        }
    }

    pub(crate) fn into_source(self) -> GpuFrameSource {
        self.source
    }

    /// The caller has reserved the complete resident peak before entering here. Streaming
    /// separately retains a persistent preparation permit while individual batches reserve scratch.
    pub(crate) fn materialize(
        self,
        context: &crate::WgpuContext,
        permit: Option<MemoryPermit>,
    ) -> Arc<PreparedInput> {
        let device = context.device();
        let (source, preparation) = match self.source {
            GpuFrameSource::Buffer(source) => (source, InputPreparation::None),
            GpuFrameSource::Yuv(input) => {
                let plan = self.conversion.expect("checked YUV input plan");
                let buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jxl-wgpu converted RGB input"),
                    size: plan.layout.logical_size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }));
                let mut source =
                    BufferImageSource::new(buffer, self.layout).expect("checked input allocation");
                source.extra_channels = input.source().extra_channels.clone();
                let preparation = plan.materialize(context, input, &source.buffer);
                (source, InputPreparation::Yuv(Box::new(preparation)))
            }
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
                (source, InputPreparation::Texture(texture))
            }
        };
        Arc::new(PreparedInput {
            source,
            preparation,
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
    preparation: InputPreparation,
    _permit: Option<MemoryPermit>,
}

enum InputPreparation {
    None,
    Texture(crate::TextureImageSource),
    Yuv(Box<crate::yuv_input::PreparedYuv>),
}

impl Deref for PreparedInput {
    type Target = BufferImageSource;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl PreparedInput {
    /// Record once, before the first compute pass, on the job's ordinary submission.
    pub(crate) fn record_preparation(&self, commands: &mut wgpu::CommandEncoder) {
        let source = match &self.preparation {
            InputPreparation::None => return,
            InputPreparation::Yuv(source) => return source.record(commands),
            InputPreparation::Texture(source) => source,
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
