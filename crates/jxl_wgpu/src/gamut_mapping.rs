//! Metadata for linear RGB gamut mapping in the shared GPU output stage.
use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::{ColorMatrix, GamutMapping, RgbColorSpace, WhitePointAdaptation};

use crate::{Error, Result};

/// Standalone WGSL fragment defining `gamut_map_rgb` and its 16-byte parameter record.
pub const GAMUT_MAPPING_SHADER: &str = include_str!("../shaders/gamut_mapping.wgsl");

/// Target linear-primary luminances and the saturation preference. A negative preference
/// disables mapping. This record adds no image allocation, dispatch, or storage binding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GamutMappingParams {
    pub(crate) luminance: [f32; 4],
}

impl Default for GamutMappingParams {
    fn default() -> Self {
        Self {
            luminance: [0.0, 0.0, 0.0, -1.0],
        }
    }
}

impl GamutMappingParams {
    pub fn new(space: RgbColorSpace, mapping: GamutMapping) -> Result<Self> {
        Self::for_space(space)?.with_mapping(Some(mapping))
    }

    pub(crate) fn for_space(space: RgbColorSpace) -> Result<Self> {
        let chromaticities = space.chromaticities().ok_or_else(|| {
            Error::InvalidPayload("gamut mapping requires target RGB chromaticities".into())
        })?;
        let matrix =
            ColorMatrix::rgb_to_xyz(space, chromaticities.white, WhitePointAdaptation::Bradford)
                .map_err(|error| Error::InvalidPayload(error.to_string()))?;
        let row = matrix.rows()[1];
        let sum: f64 = row.iter().sum();
        let row = row.map(|v| (v / sum) as f32);
        if !sum.is_finite() || sum <= 0.0 || row.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidPayload(
                "target luminance exceeds GPU F32 precision".into(),
            ));
        }
        Ok(Self {
            luminance: [row[0], row[1], row[2], -1.0],
        })
    }

    pub(crate) fn with_mapping(mut self, mapping: Option<GamutMapping>) -> Result<Self> {
        if let Some(mapping) = mapping {
            let row = &self.luminance[..3];
            if row.iter().any(|v| *v < 0.0) || row.iter().sum::<f32>() <= 0.0 {
                return Err(Error::InvalidPayload(
                    "gamut mapping requires a target white inside its RGB primary triangle".into(),
                ));
            }
            self.luminance[3] = mapping.preserve_saturation();
        } else {
            self.luminance[3] = -1.0;
        }
        Ok(self)
    }
}

const _: () = assert!(std::mem::size_of::<GamutMappingParams>() == 16);
