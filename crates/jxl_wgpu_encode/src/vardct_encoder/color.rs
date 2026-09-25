//! One lowering of the source-to-codestream color contract, shared by headers and GPU work.

use std::sync::Arc;

use crate::source_color::{SourceColorEncoding, icc::PreparedImageHeader};
use crate::{EncodeError, ImageColorOptions, UnsupportedFeature};
use jxl_gpu_formats::{ColorSpecification, PixelFormat};
use jxl_gpu_protocol::icc::{IccRenderingIntent, IccTransform};
use jxl_gpu_protocol::{ColorMatrix, RgbColorSpace, WhitePointAdaptation};

/// Coding-domain selection for integer or floating Gray/RGB sources.
///
/// This is independent of source storage and the declared presentation encoding.
/// Every physical frame in a sequence uses the encoder's selected domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarDctColorTransform {
    /// Linearize the declared source color and transform to XYB on the GPU (the default).
    #[default]
    Xyb,
    /// Transform the original, normalized components without a color conversion.
    Original,
}

/// Only the typed policies above can construct this plan. In particular, original components
/// omits the XYB-only matrix-scale fields and must use their implicit neutral scales.
#[derive(Clone, Debug)]
pub(super) struct VarDctColorPlan {
    pub(super) samples: crate::sample_format::ImageSamplePlan,
    encoding: SourceColorEncoding,
    options: ImageColorOptions,
    max_icc_profile_bytes: u64,
    pub(super) icc_transform: Option<Arc<IccTransform>>,
    pub(super) gpu: SourceColorParams,
    source_components: [usize; 3],
    normalization: u32,
    qm_scales: Option<[u8; 2]>,
    hf_quantization: [f32; 3],
}

impl VarDctColorPlan {
    pub(super) fn new(config: &super::VarDctConfig) -> Result<Self, EncodeError> {
        config.color_options.validate()?;
        let encoding = source_encoding(&config.pixel_format())?;
        let xyb = config.color_transform == VarDctColorTransform::Xyb;
        let icc_transform = if let Some(profile) = encoding.icc_profile() {
            crate::source_color::icc::IccStreamPlan::new(
                profile.bytes().len() as u64,
                config.max_icc_profile_bytes,
            )?;
            if profile.header().rendering_intent != config.color_options.rendering_intent {
                return Err(EncodeError::InvalidConfiguration(
                    "encoder intent must match the unchanged embedded ICC profile",
                ));
            }
            // XYB's working connection is relative to linear BT.709, independently of the
            // unchanged profile's presentation intent. Pixel evaluation stays on the GPU.
            xyb.then(|| {
                if config.sample_format.channels() == crate::ColorChannels::Gray {
                    IccTransform::to_linear_gray(profile, IccRenderingIntent::Relative)
                } else {
                    IccTransform::to_linear_rgb(
                        profile,
                        RgbColorSpace::Bt709,
                        IccRenderingIntent::Relative,
                    )
                }
                .map(Arc::new)
            })
            .transpose()?
        } else {
            None
        };
        let rgb = match &encoding {
            SourceColorEncoding::Enumerated(encoding) => encoding.rgb_encoding(),
            SourceColorEncoding::Icc(_) => jxl_gpu_protocol::RgbColorEncoding::LINEAR_BT709,
        };
        let (transfer, gamma) = jxl_wgpu::transfer_parameters(rgb.transfer);
        let intensity = config.color_options.intensity_target.to_f32();
        let luminance = jxl_wgpu::display_luminance(rgb.space, intensity, false)?;
        let matrix = ColorMatrix::between_rgb(
            rgb.space,
            RgbColorSpace::Bt709,
            WhitePointAdaptation::Bradford,
        )
        .map_err(|_| UnsupportedFeature::InputFormat)?
        .rows()
        .map(|row| row.map(|v| v as f32));
        if matrix.iter().flatten().any(|v| !v.is_finite()) {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        Ok(Self {
            samples: crate::sample_format::ImageSamplePlan::new(config.sample_format, config.alpha)
                .with_extra_channels(
                    &config.extra_channels,
                    config.max_extra_channel_metadata_bytes,
                )?,
            encoding,
            options: config.color_options,
            max_icc_profile_bytes: config.max_icc_profile_bytes,
            icc_transform,
            gpu: SourceColorParams {
                transfer,
                gamma,
                intensity,
                luminance,
                matrix,
            },
            source_components: config.sample_format.channels().working_components(),
            normalization: if !xyb {
                1
            } else if matches!(config.source_color, ColorSpecification::Icc(_)) {
                2
            } else {
                0
            },
            qm_scales: xyb.then_some([3, 2]),
            hf_quantization: if xyb { [1.25, 1.0, 1.0] } else { [1.0; 3] },
        })
    }

    pub(super) fn matches_format(&self, format: &PixelFormat) -> bool {
        self.samples.matches_format(format)
            && source_encoding(format).is_ok_and(|encoding| encoding == self.encoding)
    }

    pub(super) fn validate_frame(
        &self,
        frame: &crate::frame_header::FrameHeaderPlan,
    ) -> Result<(), EncodeError> {
        // ISO/IEC 18181-1 F.2 forbids this image/frame combination, regardless of the
        // selected ICC method or whether a subsequent frame actually reads the slot.
        if self.xyb_encoded()
            && self.encoding.icc_profile().is_some()
            && frame.requires_post_color_reference()
        {
            return Err(EncodeError::InvalidConfiguration(
                "XYB with an embedded ICC profile cannot save post-color-transform references",
            ));
        }
        Ok(())
    }

    pub(super) fn image_header(
        &self,
        descriptor: &crate::ImageSequenceDescriptor,
    ) -> Result<PreparedImageHeader, EncodeError> {
        descriptor.image_header(
            self.xyb_encoded(),
            &self.samples,
            &self.encoding,
            self.options,
            self.max_icc_profile_bytes,
        )
    }

    pub(super) fn icc_profile_bytes(&self) -> u64 {
        self.encoding
            .icc_profile()
            .map_or(0, |profile| profile.bytes().len() as u64)
    }

    /// Lower logical Gray/RGB samples to the standard three VarDCT working components.
    /// Repeated Gray records alias the same checked source bytes; expansion remains on GPU.
    pub(super) fn bind_sources(
        &self,
        region: &crate::source::SourceRegion,
    ) -> ([crate::source::SourceParams; 3], [u64; 4]) {
        let components = self.source_components.map(|index| region.components[index]);
        let mut offsets = [0; 4];
        for (destination, source) in offsets.iter_mut().zip(self.source_components) {
            *destination = region.offsets[source];
        }
        (components, offsets)
    }

    pub(super) const fn samples(&self) -> crate::ColorSampleFormat {
        self.samples.color
    }

    pub(super) const fn xyb_encoded(&self) -> bool {
        self.qm_scales.is_some()
    }

    pub(super) const fn normalization(&self) -> u32 {
        self.normalization
    }

    pub(super) const fn qm_scales(&self) -> Option<[u8; 2]> {
        self.qm_scales
    }

    pub(super) const fn hf_quantization(&self) -> [f32; 3] {
        self.hf_quantization
    }
}

fn source_encoding(format: &PixelFormat) -> Result<SourceColorEncoding, EncodeError> {
    if matches!(format.color_spec, ColorSpecification::Undefined) {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    SourceColorEncoding::from_format(format)
}

/// One bounded color conversion record; all fields have four-byte storage alignment.
/// Original-component coding retains the declaration but bypasses this conversion.
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(super) struct SourceColorParams {
    pub(super) transfer: u32,
    pub(super) gamma: f32,
    pub(super) intensity: f32,
    pub(super) luminance: [f32; 4],
    pub(super) matrix: [[f32; 3]; 3],
}

const _: () = assert!(std::mem::size_of::<SourceColorParams>() == 64);
