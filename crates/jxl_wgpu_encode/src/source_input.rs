//! Checked physical input preparation, independent of codec and frame sampling.
use std::ops::Deref;
use std::sync::Arc;

use jxl_gpu_formats::ImageLayout;
use jxl_wgpu::MemoryPermit;

use crate::{
    BufferImageSource, CmykSampleEncoding, EncodeError, GpuFrameSource, UnsupportedFeature,
};

/// Plans storage without allocation or submission. All consumers use this checked layout;
/// only admission may materialize it. Scalar attachments remain caller-owned buffers.
pub(crate) struct FrameInputPlan {
    source: GpuFrameSource,
    storage: crate::source_storage::StoragePlan,
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
        let (storage, transfer) = match &source {
            GpuFrameSource::Buffer(input) => (input.clone().into(), None),
            GpuFrameSource::Texture(input) => (input.clone().into(), None),
            GpuFrameSource::TexturePlanes(input) => (input.clone().into(), None),
            GpuFrameSource::Yuv(input) => (input.source().clone(), Some(input.rgb_transfer())),
        };
        let storage = crate::source_storage::StoragePlan::new(storage)?;
        let conversion = transfer
            .map(|transfer| {
                crate::yuv_input::YuvPlan::new(
                    &storage.layout,
                    storage.buffer_bytes(),
                    storage.buffer_usage(),
                    transfer,
                )
            })
            .transpose()?;
        let layout = conversion
            .as_ref()
            .map_or(&storage.layout, |plan| &plan.layout)
            .clone();
        Ok(Self {
            source,
            layout,
            copy_bytes: storage.copy_bytes,
            texture_bytes: storage.texture_bytes,
            storage,
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
        let plan = self.conversion.as_ref()?;
        let buffer = self.storage.caller_buffer()?;
        Some((
            Some(buffer),
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
        self.conversion.as_ref().map_or_else(
            || self.storage.buffer_bytes(),
            |plan| plan.layout.logical_size,
        )
    }

    pub(crate) fn buffer_usage(&self) -> wgpu::BufferUsages {
        if self.conversion.is_some() {
            wgpu::BufferUsages::STORAGE
        } else {
            self.storage.buffer_usage()
        }
    }

    /// None denotes encoder-owned prepared RGB/texels, excluded from caller-buffer accounting.
    pub(crate) fn caller_buffer(&self) -> Option<&wgpu::Buffer> {
        if self.conversion.is_some() {
            None
        } else {
            self.storage.caller_buffer()
        }
    }

    pub(crate) fn extra_channels(&self) -> &[BufferImageSource] {
        self.storage.source().extra_channels()
    }

    pub(crate) fn cmyk_encoding(&self) -> CmykSampleEncoding {
        self.storage.source().cmyk_encoding()
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
        let (mut source, copies) = self.storage.materialize(context.device());
        let conversion = self.conversion.map(|plan| {
            let buffer = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("jxl-wgpu converted RGB input"),
                size: plan.layout.logical_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            let mut rgb =
                BufferImageSource::new(buffer, self.layout).expect("checked input allocation");
            rgb.extra_channels = source.extra_channels.clone();
            let input = std::mem::replace(&mut source, rgb);
            plan.materialize(context, input, &source.buffer)
        });
        Arc::new(PreparedInput {
            source,
            copies,
            conversion,
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
    copies: Option<crate::source_storage::PreparedTextureCopies>,
    conversion: Option<crate::yuv_input::PreparedYuv>,
    _permit: Option<MemoryPermit>,
}

impl Deref for PreparedInput {
    type Target = BufferImageSource;
    fn deref(&self) -> &Self::Target {
        &self.source
    }
}

impl PreparedInput {
    /// Record each preparation stage once, before the first codec compute pass.
    pub(crate) fn record_preparation(&self, commands: &mut wgpu::CommandEncoder) {
        if let Some(copies) = &self.copies {
            copies.record(commands);
        }
        if let Some(conversion) = &self.conversion {
            conversion.record(commands);
        }
    }
}
