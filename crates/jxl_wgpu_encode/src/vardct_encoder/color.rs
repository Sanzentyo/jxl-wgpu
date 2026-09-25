//! One lowering of the source-to-codestream color contract, shared by headers and GPU work.

use crate::source_color::{EnumeratedColorEncoding, SourceColorEncoding};
use crate::{EncodeError, ImageColorOptions, UnsupportedFeature};
use jxl_gpu_formats::{ColorSpecification, PixelFormat};
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
#[derive(Clone, Copy, Debug)]
pub(super) struct VarDctColorPlan {
    samples: crate::ColorSampleFormat,
    encoding: EnumeratedColorEncoding,
    options: ImageColorOptions,
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
        let rgb = encoding.rgb_encoding();
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
        let xyb = config.color_transform == VarDctColorTransform::Xyb;
        Ok(Self {
            samples: config.sample_format,
            encoding,
            options: config.color_options,
            gpu: SourceColorParams {
                transfer,
                gamma,
                intensity,
                luminance,
                matrix,
            },
            source_components: config.sample_format.channels().working_components(),
            normalization: u32::from(!xyb),
            qm_scales: xyb.then_some([3, 2]),
            hf_quantization: if xyb { [1.25, 1.0, 1.0] } else { [1.0; 3] },
        })
    }

    pub(super) fn matches_format(self, format: &PixelFormat) -> bool {
        self.samples.matches_format(format)
            && source_encoding(format).is_ok_and(|encoding| encoding == self.encoding)
    }

    pub(super) fn image_header(
        self,
        descriptor: &crate::ImageSequenceDescriptor,
    ) -> Result<crate::BitFragment, EncodeError> {
        descriptor.image_header(
            self.xyb_encoded(),
            self.samples,
            self.encoding,
            self.options,
        )
    }

    /// Lower logical Gray/RGB samples to the standard three VarDCT working components.
    /// Repeated Gray records alias the same checked source bytes; expansion remains on GPU.
    pub(super) fn bind_sources(
        self,
        region: &crate::source::SourceRegion,
    ) -> ([crate::source::SourceParams; 3], [u64; 4]) {
        let components = self.source_components.map(|index| region.components[index]);
        let mut offsets = [0; 4];
        for (destination, source) in offsets.iter_mut().zip(self.source_components) {
            *destination = region.offsets[source];
        }
        (components, offsets)
    }

    pub(super) const fn samples(self) -> crate::ColorSampleFormat {
        self.samples
    }

    pub(super) const fn xyb_encoded(self) -> bool {
        self.qm_scales.is_some()
    }

    pub(super) const fn normalization(self) -> u32 {
        self.normalization
    }

    pub(super) const fn qm_scales(self) -> Option<[u8; 2]> {
        self.qm_scales
    }

    pub(super) const fn hf_quantization(self) -> [f32; 3] {
        self.hf_quantization
    }
}

fn source_encoding(format: &PixelFormat) -> Result<EnumeratedColorEncoding, EncodeError> {
    if matches!(format.color_spec, ColorSpecification::Undefined) {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    match SourceColorEncoding::from_format(format)? {
        SourceColorEncoding::Enumerated(encoding) => Ok(encoding),
        SourceColorEncoding::Icc(_) => Err(UnsupportedFeature::InputFormat.into()),
    }
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
