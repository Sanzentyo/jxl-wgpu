//! One lowering of the source-to-codestream color contract, shared by headers and GPU work.

/// Coding-domain selection for integer or floating Gray/RGB sRGB/D65 sources.
///
/// This is independent of source storage and the declared presentation encoding.
/// Every physical frame in a sequence uses the encoder's selected domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarDctColorTransform {
    /// Linearize sRGB and transform to XYB on the GPU (the default).
    #[default]
    Xyb,
    /// Transform the original, normalized sRGB components without a color conversion.
    Original,
}

/// Only the typed policies above can construct this plan. In particular, original components
/// omits the XYB-only matrix-scale fields and must use their implicit neutral scales.
#[derive(Clone, Copy, Debug)]
pub(super) struct VarDctColorPlan {
    samples: crate::ColorSampleFormat,
    source_components: [usize; 3],
    normalization: u32,
    qm_scales: Option<[u8; 2]>,
    hf_quantization: [f32; 3],
}

impl VarDctColorPlan {
    pub(super) const fn new(
        transform: VarDctColorTransform,
        samples: crate::ColorSampleFormat,
    ) -> Self {
        match transform {
            VarDctColorTransform::Xyb => Self {
                samples,
                source_components: samples.channels().working_components(),
                normalization: 0,
                qm_scales: Some([3, 2]),
                hf_quantization: [1.25, 1.0, 1.0],
            },
            VarDctColorTransform::Original => Self {
                samples,
                source_components: samples.channels().working_components(),
                normalization: 1,
                qm_scales: None,
                hf_quantization: [1.0; 3],
            },
        }
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
