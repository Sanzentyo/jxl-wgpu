//! Portable logical pixel formats, pitch-linear buffer layouts, and scalar
//! reference conversion.
//!
//! The model deliberately separates [`PixelFormat`] (meaning and packing) from
//! [`ImageLayout`] (offsets, pitches, and extents). Only directly addressable
//! pitch-linear storage is represented. NVIDIA VPI block-linear formats and
//! opaque CUDA/EGL/NvBuffer/NvSciBuf storage require vendor interop and are
//! intentionally outside this portable `wgpu` contract.
//!
//! See the crate README and [`vpi`] for the NVIDIA VPI 4.1.3 inventory.

mod classify;
mod format;
mod layout;
pub mod vpi;

#[cfg(any(feature = "cpu-reference", test))]
mod convert;

pub use classify::{
    ColorFormatClass, NumericFormatClass, PixelFormatClass, PixelFormatClassificationError,
    RgbStorage, WgslNumericCapability, classify_pixel_format,
};
#[cfg(any(feature = "cpu-reference", test))]
pub use convert::{ConversionError, ConvertedImage, convert_rgb_f32};
pub use format::{
    ByteOrder, Channel, ChromaLocation, ChromaLocation2d, ChromaOrder, ChromaSubsampling,
    ColorModel, ColorRange, ColorSpace, ColorSpec, ColorSpecification, Packed422Order,
    PackingField, PackingFieldKind, PackingWord, PixelFormat, PixelFormatError, PlaneFormat,
    PlaneSampling, RawPattern, RgbChannelOrder, SampleKind, Swizzle, SwizzleComponent,
    TransferFunction, YcbcrEncoding,
};
pub use layout::{ImageLayout, LayoutError, PitchLinearPlaneLayout};
