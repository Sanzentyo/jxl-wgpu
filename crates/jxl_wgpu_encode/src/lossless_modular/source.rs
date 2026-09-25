//! Modular color metadata layered onto the common checked source packing.
use crate::EncodeError;
use crate::source::{SourceSpec, source_spec};
use crate::source_color::SourceColorEncoding;
use jxl_gpu_formats::PixelFormat;

pub(super) struct LosslessModularSourceSpec {
    pub(super) packing: SourceSpec,
    pub(super) color: SourceColorEncoding,
}

pub(super) fn lossless_modular_source_spec(
    format: &PixelFormat,
) -> Result<LosslessModularSourceSpec, EncodeError> {
    Ok(LosslessModularSourceSpec {
        packing: source_spec(format)?,
        color: SourceColorEncoding::from_format(format)?,
    })
}
