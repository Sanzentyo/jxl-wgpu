//! Exact quantizer controls, independent of a perceptual quality search.

use crate::EncodeError;

use super::{VarDctCoefficientOrders, VarDctDequantMatrices, VarDctLfMetadata};

/// An effective JPEG XL HF multiplier in `1..=256`. Larger values retain finer AC detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarDctHfMultiplier(u32);

impl VarDctHfMultiplier {
    /// Validates an effective multiplier. Decoders clamp the raw metadata to
    /// `0..=255` before adding one, so larger encoder settings cannot add precision.
    pub fn new(value: u32) -> Result<Self, EncodeError> {
        if !(1..=256).contains(&value) {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT HF multiplier must be in 1..=256",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Default for VarDctHfMultiplier {
    fn default() -> Self {
        Self(6)
    }
}

/// Exact JPEG XL frame quantizer values.
///
/// `global_scale` scales both LF and HF precision. `quant_lf` additionally scales
/// LF precision; `hf_multiplier` scales AC precision unless a transform overrides
/// it. These controls do not imply a perceptual distance or a rate guarantee.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VarDctQuantization {
    global_scale: u32,
    quant_lf: u32,
    hf_multiplier: VarDctHfMultiplier,
}

impl VarDctQuantization {
    /// Validates the full JPEG XL global-scale and LF-quantizer U32 ranges.
    ///
    /// A legal configuration can still overflow signed coefficients for a
    /// particular image. GPU submission reports that error instead of clipping.
    pub fn new(
        global_scale: u32,
        quant_lf: u32,
        hf_multiplier: VarDctHfMultiplier,
    ) -> Result<Self, EncodeError> {
        if !(1..=73_728).contains(&global_scale) {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT global scale must be in 1..=73728",
            ));
        }
        if !(1..=65_536).contains(&quant_lf) {
            return Err(EncodeError::InvalidConfiguration(
                "VarDCT LF quantizer must be in 1..=65536",
            ));
        }
        Ok(Self {
            global_scale,
            quant_lf,
            hf_multiplier,
        })
    }

    #[must_use]
    pub const fn global_scale(self) -> u32 {
        self.global_scale
    }

    #[must_use]
    pub const fn quant_lf(self) -> u32 {
        self.quant_lf
    }

    #[must_use]
    pub const fn hf_multiplier(self) -> VarDctHfMultiplier {
        self.hf_multiplier
    }
}

impl Default for VarDctQuantization {
    fn default() -> Self {
        Self {
            global_scale: 8_813,
            quant_lf: 10,
            hf_multiplier: VarDctHfMultiplier::default(),
        }
    }
}

/// Source color, coding domain, quantizers, matrices, LF metadata, coefficient orders and AC passes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VarDctConfig {
    /// Stream-wide Gray/RGB source channels and precision; defaults to interleaved RGB8.
    pub sample_format: crate::ColorSampleFormat,
    /// Optional full-resolution alpha at the same precision, compressed losslessly on GPU.
    /// Color association is declared unchanged; no premultiplication or division is performed.
    pub alpha: Option<crate::AlphaAssociation>,
    /// Independently stored scalar planes, following any packed alpha in codestream order.
    pub extra_channels: Vec<crate::ExtraChannel>,
    /// Bound for extra declarations and names, checked before image-header allocation.
    pub max_extra_channel_metadata_bytes: u64,
    /// Stream-wide enumerated or embedded RGB/Gray ICC source color; defaults to sRGB/D65.
    pub source_color: jxl_gpu_formats::ColorSpecification,
    /// Image-wide orientation, rendering intent and positive exact binary16 image white.
    pub image_options: crate::ImageOptions,
    /// Maximum original ICC profile bytes, checked before serialization or GPU lowering (default 16 MiB).
    pub max_icc_profile_bytes: u64,
    /// Stream-wide coding domain for integer or floating Gray/RGB sources; defaults to XYB.
    pub color_transform: super::VarDctColorTransform,
    /// Spectral/quantized AC progression; defaults to one complete pass.
    pub progressive: crate::ProgressivePlan,
    /// Physical AC group order, repeated per pass; defaults to raster order.
    pub group_order: super::VarDctGroupOrder,
    pub quantization: VarDctQuantization,
    pub lf_metadata: VarDctLfMetadata,
    /// Immutable caller-selected permutations; defaults to natural order for every size class.
    pub coefficient_orders: VarDctCoefficientOrders,
    /// Validated parametric/raw HF matrices; defaults to the standard matrices for all 17 families.
    pub dequant_matrices: VarDctDequantMatrices,
}

impl Default for VarDctConfig {
    fn default() -> Self {
        Self {
            sample_format: Default::default(),
            alpha: None,
            extra_channels: Vec::new(),
            max_extra_channel_metadata_bytes: 1 << 20,
            source_color: jxl_gpu_formats::ColorSpecification::Default,
            image_options: Default::default(),
            max_icc_profile_bytes: crate::source_color::icc::DEFAULT_PROFILE_LIMIT,
            color_transform: Default::default(),
            progressive: Default::default(),
            group_order: Default::default(),
            quantization: Default::default(),
            lf_metadata: Default::default(),
            coefficient_orders: Default::default(),
            dequant_matrices: Default::default(),
        }
    }
}

impl VarDctConfig {
    /// Canonical source storage with this configuration's channels, precision and color.
    #[must_use]
    pub fn pixel_format(&self) -> jxl_gpu_formats::PixelFormat {
        jxl_gpu_formats::PixelFormat {
            color_spec: self.source_color.clone(),
            ..crate::sample_format::ImageSamplePlan::new(self.sample_format, self.alpha)
                .pixel_format()
        }
    }
}
