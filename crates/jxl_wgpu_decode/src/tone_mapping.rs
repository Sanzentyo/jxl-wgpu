//! Resolve JPEG XL image metadata against the requested display, without reading pixels.
use jxl_gpu_bitstream::ImageHeaderInventory;
use jxl_gpu_protocol::{LuminanceRange, ToneMapping};

use crate::color_output::ColorOutputError;
use crate::{GpuOutputRequest, Result};

#[cfg(test)]
mod tests;

pub(crate) fn for_image(
    image: &ImageHeaderInventory,
    request: &GpuOutputRequest,
) -> Result<Option<ToneMapping>> {
    request
        .tone_mapping_target()
        .map(|target| {
            let metadata = image.tone_mapping;
            let source = LuminanceRange::new(
                metadata.min_nits.to_f32(),
                metadata.intensity_target.to_f32(),
            )
            .ok_or(ColorOutputError::InvalidToneMapping)?;
            let threshold = metadata.linear_below.to_f32();
            let threshold = if metadata.relative_to_max_display {
                if !(0.0..=1.0).contains(&threshold) {
                    return Err(ColorOutputError::InvalidToneMapping.into());
                }
                f64::from(threshold) * f64::from(target.white().nits())
            } else {
                f64::from(threshold)
            };
            ToneMapping::new(source, target, threshold)
                .ok_or_else(|| ColorOutputError::InvalidToneMapping.into())
        })
        .transpose()
}
