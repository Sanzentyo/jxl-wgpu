//! NVIDIA VPI 4.1.3 logical-format inventory.
//!
//! Only VPI's directly addressable pitch-linear (`PL`) formats are modeled.
//! `_BL`, `_BL16`, CUDA array, EGLImage, NvBuffer, and NvSciBuf storage are
//! deliberately excluded because their memory contracts are not portable
//! `wgpu` pitch-linear buffers.

use crate::{
    Channel, ChromaLocation, ChromaLocation2d, ColorRange, ColorSpace, ColorSpec,
    ColorSpecification, Packed422Order, PixelFormat, RgbChannelOrder, SampleKind, TransferFunction,
    YcbcrEncoding,
};

pub const VERSION: &str = "4.1.3";
pub const RELEASE_NOTES_URL: &str = "https://docs.nvidia.com/vpi/release_notes.html";
pub const IMAGE_FORMAT_URL: &str = "https://docs.nvidia.com/vpi/group__VPI__ImageFormat.html";
pub const IMAGE_FORMAT_SOURCE_URL: &str = "https://docs.nvidia.com/vpi/ImageFormat_8h_source.html";
pub const COLOR_SPEC_URL: &str = "https://docs.nvidia.com/vpi/group__VPI__ColorSpec.html";
pub const COLOR_SPEC_SOURCE_URL: &str = "https://docs.nvidia.com/vpi/ColorSpec_8h_source.html";
pub const DATA_LAYOUT_URL: &str = "https://docs.nvidia.com/vpi/group__VPI__DataLayout.html";
pub const IMAGE_BUFFER_URL: &str = "https://docs.nvidia.com/vpi/group__VPI__Image.html";

/// Named color specifications in VPI 4.1, excluding the invalid sentinel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VpiColorSpec {
    Default,
    Undefined,
    Bt601,
    Bt601Er,
    Bt709,
    Bt709Er,
    Bt709Linear,
    Bt2020,
    Bt2020Er,
    Bt2020Linear,
    Bt2020Pq,
    Bt2020PqEr,
    Bt2020ConstantLuminance,
    Bt2020ConstantLuminanceEr,
    Mpeg2Bt601,
    Mpeg2Bt709,
    Mpeg2Smpte240M,
    Srgb,
    Sycc,
    Smpte240M,
    DisplayP3,
    DisplayP3Linear,
    Sensor,
}

impl VpiColorSpec {
    pub const ALL: [Self; 23] = [
        Self::Default,
        Self::Undefined,
        Self::Bt601,
        Self::Bt601Er,
        Self::Bt709,
        Self::Bt709Er,
        Self::Bt709Linear,
        Self::Bt2020,
        Self::Bt2020Er,
        Self::Bt2020Linear,
        Self::Bt2020Pq,
        Self::Bt2020PqEr,
        Self::Bt2020ConstantLuminance,
        Self::Bt2020ConstantLuminanceEr,
        Self::Mpeg2Bt601,
        Self::Mpeg2Bt709,
        Self::Mpeg2Smpte240M,
        Self::Srgb,
        Self::Sycc,
        Self::Smpte240M,
        Self::DisplayP3,
        Self::DisplayP3Linear,
        Self::Sensor,
    ];

    #[must_use]
    pub const fn specification(self) -> ColorSpecification {
        use ChromaLocation::{Both, Center, Even};
        use ColorRange::{Full, Limited};
        use ColorSpace::{Bt709, Bt2020, DciP3, Sensor};
        use TransferFunction::{Bt709 as Bt709Xfer, Bt2020 as Bt2020Xfer};
        use TransferFunction::{Linear, Pq, Smpte240M as Smpte240MXfer, Srgb, Sycc};
        use YcbcrEncoding::{Bt601, Bt709 as Bt709Encoding, Smpte240M, Undefined as NoEncoding};
        use YcbcrEncoding::{Bt2020 as Bt2020Encoding, Bt2020ConstantLuminance};

        let (space, encoding, transfer, range, horizontal, vertical) = match self {
            Self::Default => return ColorSpecification::Default,
            Self::Undefined => (Bt709, NoEncoding, Linear, Full, Both, Both),
            Self::Bt601 => (Bt709, Bt601, Bt709Xfer, Limited, Even, Even),
            Self::Bt601Er => (Bt709, Bt601, Bt709Xfer, Full, Even, Even),
            Self::Bt709 => (Bt709, Bt709Encoding, Bt709Xfer, Limited, Even, Even),
            Self::Bt709Er => (Bt709, Bt709Encoding, Bt709Xfer, Full, Even, Even),
            Self::Bt709Linear => (Bt709, Bt709Encoding, Linear, Limited, Even, Even),
            Self::Bt2020 => (Bt2020, Bt2020Encoding, Bt2020Xfer, Limited, Even, Even),
            Self::Bt2020Er => (Bt2020, Bt2020Encoding, Bt2020Xfer, Full, Even, Even),
            Self::Bt2020Linear => (Bt2020, Bt2020Encoding, Linear, Limited, Even, Even),
            Self::Bt2020Pq => (Bt2020, Bt2020Encoding, Pq, Limited, Even, Even),
            Self::Bt2020PqEr => (Bt2020, Bt2020Encoding, Pq, Full, Even, Even),
            Self::Bt2020ConstantLuminance => (
                Bt2020,
                Bt2020ConstantLuminance,
                Bt2020Xfer,
                Limited,
                Even,
                Even,
            ),
            Self::Bt2020ConstantLuminanceEr => (
                Bt2020,
                Bt2020ConstantLuminance,
                Bt2020Xfer,
                Full,
                Even,
                Even,
            ),
            Self::Mpeg2Bt601 => (Bt709, Bt601, Bt709Xfer, Full, Even, Center),
            Self::Mpeg2Bt709 => (Bt709, Bt709Encoding, Bt709Xfer, Full, Even, Center),
            Self::Mpeg2Smpte240M => (Bt709, Smpte240M, Smpte240MXfer, Full, Even, Center),
            Self::Srgb => (Bt709, NoEncoding, Srgb, Full, Both, Both),
            Self::Sycc => (Bt709, Bt601, Sycc, Full, Center, Center),
            Self::Smpte240M => (Bt709, Smpte240M, Smpte240MXfer, Limited, Even, Even),
            Self::DisplayP3 => (DciP3, NoEncoding, Srgb, Full, Both, Both),
            Self::DisplayP3Linear => (DciP3, NoEncoding, Linear, Full, Both, Both),
            Self::Sensor => (Sensor, NoEncoding, Linear, Full, Both, Both),
        };
        ColorSpecification::Defined(ColorSpec {
            space,
            encoding,
            transfer,
            range,
            chroma_location: ChromaLocation2d {
                horizontal,
                vertical,
            },
        })
    }
}

/// Complete set of VPI 4.1 predefined pitch-linear image formats. The enum
/// intentionally has no block-linear variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VpiPitchLinearFormat {
    U8,
    S8,
    U16,
    U32,
    S32,
    S16,
    TwoS16,
    F32,
    F64,
    TwoF32,
    Y8,
    Y8Er,
    Y16,
    Y16Er,
    Nv12,
    Nv12Er,
    Nv24,
    Nv24Er,
    Uyvy,
    UyvyEr,
    Yuyv,
    YuyvEr,
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Rgb8Planar,
    Bgr8Planar,
    Rgba8Planar,
    Bgra8Planar,
}

impl VpiPitchLinearFormat {
    pub const ALL: [Self; 30] = [
        Self::U8,
        Self::S8,
        Self::U16,
        Self::U32,
        Self::S32,
        Self::S16,
        Self::TwoS16,
        Self::F32,
        Self::F64,
        Self::TwoF32,
        Self::Y8,
        Self::Y8Er,
        Self::Y16,
        Self::Y16Er,
        Self::Nv12,
        Self::Nv12Er,
        Self::Nv24,
        Self::Nv24Er,
        Self::Uyvy,
        Self::UyvyEr,
        Self::Yuyv,
        Self::YuyvEr,
        Self::Rgb8,
        Self::Bgr8,
        Self::Rgba8,
        Self::Bgra8,
        Self::Rgb8Planar,
        Self::Bgr8Planar,
        Self::Rgba8Planar,
        Self::Bgra8Planar,
    ];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "VPI_IMAGE_FORMAT_U8",
            Self::S8 => "VPI_IMAGE_FORMAT_S8",
            Self::U16 => "VPI_IMAGE_FORMAT_U16",
            Self::U32 => "VPI_IMAGE_FORMAT_U32",
            Self::S32 => "VPI_IMAGE_FORMAT_S32",
            Self::S16 => "VPI_IMAGE_FORMAT_S16",
            Self::TwoS16 => "VPI_IMAGE_FORMAT_2S16",
            Self::F32 => "VPI_IMAGE_FORMAT_F32",
            Self::F64 => "VPI_IMAGE_FORMAT_F64",
            Self::TwoF32 => "VPI_IMAGE_FORMAT_2F32",
            Self::Y8 => "VPI_IMAGE_FORMAT_Y8",
            Self::Y8Er => "VPI_IMAGE_FORMAT_Y8_ER",
            Self::Y16 => "VPI_IMAGE_FORMAT_Y16",
            Self::Y16Er => "VPI_IMAGE_FORMAT_Y16_ER",
            Self::Nv12 => "VPI_IMAGE_FORMAT_NV12",
            Self::Nv12Er => "VPI_IMAGE_FORMAT_NV12_ER",
            Self::Nv24 => "VPI_IMAGE_FORMAT_NV24",
            Self::Nv24Er => "VPI_IMAGE_FORMAT_NV24_ER",
            Self::Uyvy => "VPI_IMAGE_FORMAT_UYVY",
            Self::UyvyEr => "VPI_IMAGE_FORMAT_UYVY_ER",
            Self::Yuyv => "VPI_IMAGE_FORMAT_YUYV",
            Self::YuyvEr => "VPI_IMAGE_FORMAT_YUYV_ER",
            Self::Rgb8 => "VPI_IMAGE_FORMAT_RGB8",
            Self::Bgr8 => "VPI_IMAGE_FORMAT_BGR8",
            Self::Rgba8 => "VPI_IMAGE_FORMAT_RGBA8",
            Self::Bgra8 => "VPI_IMAGE_FORMAT_BGRA8",
            Self::Rgb8Planar => "VPI_IMAGE_FORMAT_RGB8p",
            Self::Bgr8Planar => "VPI_IMAGE_FORMAT_BGR8p",
            Self::Rgba8Planar => "VPI_IMAGE_FORMAT_RGBA8p",
            Self::Bgra8Planar => "VPI_IMAGE_FORMAT_BGRA8p",
        }
    }

    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        let limited = VpiColorSpec::Bt601.specification();
        let full = VpiColorSpec::Bt601Er.specification();
        match self {
            Self::U8 => PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
            Self::S8 => PixelFormat::non_color(SampleKind::Signed, 8, &[Channel::X]),
            Self::U16 => PixelFormat::non_color(SampleKind::Unsigned, 16, &[Channel::X]),
            Self::U32 => PixelFormat::non_color(SampleKind::Unsigned, 32, &[Channel::X]),
            Self::S32 => PixelFormat::non_color(SampleKind::Signed, 32, &[Channel::X]),
            Self::S16 => PixelFormat::non_color(SampleKind::Signed, 16, &[Channel::X]),
            Self::TwoS16 => {
                PixelFormat::non_color(SampleKind::Signed, 16, &[Channel::X, Channel::Y])
            }
            Self::F32 => PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            Self::F64 => PixelFormat::non_color(SampleKind::Float, 64, &[Channel::X]),
            Self::TwoF32 => {
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X, Channel::Y])
            }
            Self::Y8 => PixelFormat::luma(8, limited),
            Self::Y8Er => PixelFormat::luma(8, full),
            Self::Y16 => PixelFormat::luma(16, limited),
            Self::Y16Er => PixelFormat::luma(16, full),
            Self::Nv12 => PixelFormat::nv12(limited),
            Self::Nv12Er => PixelFormat::nv12(full),
            Self::Nv24 => PixelFormat::nv24(limited),
            Self::Nv24Er => PixelFormat::nv24(full),
            Self::Uyvy => PixelFormat::packed_yuv4228(Packed422Order::Uyvy, limited),
            Self::UyvyEr => PixelFormat::packed_yuv4228(Packed422Order::Uyvy, full),
            Self::Yuyv => PixelFormat::packed_yuv4228(Packed422Order::Yuyv, limited),
            Self::YuyvEr => PixelFormat::packed_yuv4228(Packed422Order::Yuyv, full),
            Self::Rgb8 => PixelFormat::rgb8(
                RgbChannelOrder::Rgb,
                false,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Bgr8 => PixelFormat::rgb8(
                RgbChannelOrder::Bgr,
                false,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Rgba8 => PixelFormat::rgb8(
                RgbChannelOrder::Rgba,
                false,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Bgra8 => PixelFormat::rgb8(
                RgbChannelOrder::Bgra,
                false,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Rgb8Planar => PixelFormat::rgb8(
                RgbChannelOrder::Rgb,
                true,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Bgr8Planar => PixelFormat::rgb8(
                RgbChannelOrder::Bgr,
                true,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Rgba8Planar => PixelFormat::rgb8(
                RgbChannelOrder::Rgba,
                true,
                VpiColorSpec::Undefined.specification(),
            ),
            Self::Bgra8Planar => PixelFormat::rgb8(
                RgbChannelOrder::Bgra,
                true,
                VpiColorSpec::Undefined.specification(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaSubsampling, ColorModel, Swizzle};

    #[test]
    fn every_pitch_linear_predefined_format_is_valid() {
        assert_eq!(VpiPitchLinearFormat::ALL.len(), 30);
        for predefined in VpiPitchLinearFormat::ALL {
            predefined
                .pixel_format()
                .validate()
                .unwrap_or_else(|error| {
                    panic!(
                        "{} did not map to a valid descriptor: {error}",
                        predefined.name()
                    )
                });
        }
    }

    #[test]
    fn vpi_nv12_matches_the_header_components() {
        let format = VpiPitchLinearFormat::Nv12.pixel_format();
        assert_eq!(format.model, ColorModel::Ycbcr);
        assert_eq!(format.chroma_subsampling, ChromaSubsampling::Cs420);
        assert_eq!(format.swizzle, Swizzle::XYZ0);
        assert_eq!(format.planes.len(), 2);
        assert_eq!(format.planes[0].bits_per_element(), 8);
        assert_eq!(format.planes[1].bits_per_element(), 16);
    }

    #[test]
    fn all_named_color_specs_are_distinctly_available() {
        assert_eq!(VpiColorSpec::ALL.len(), 23);
        assert_eq!(
            VpiColorSpec::Mpeg2Bt709.specification(),
            ColorSpecification::Defined(ColorSpec {
                space: ColorSpace::Bt709,
                encoding: YcbcrEncoding::Bt709,
                transfer: TransferFunction::Bt709,
                range: ColorRange::Full,
                chroma_location: ChromaLocation2d {
                    horizontal: ChromaLocation::Even,
                    vertical: ChromaLocation::Center,
                },
            })
        );
    }
}
