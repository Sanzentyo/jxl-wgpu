//! Shared color conversion, orientation, and pitch-linear packing for GPU image producers.
//!
//! The shader fragment owns the output words. A producer supplies `source_rgb_at(x, y)` in
//! oriented coordinates, returning unclipped RGB in the configured source encoding. This allows
//! codec reconstruction to fuse with the same output conversion used by the render graph.

use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpace, ColorSpecification,
    ImageLayout, NumericFormatClass, Packed422Order, PixelFormat, PixelFormatClass,
    RgbChannelOrder, RgbStorage, TransferFunction as ImageTransferFunction, WgslNumericCapability,
    YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{
    Extent2d, OutputOrientation, RgbColorEncoding, RgbPrimaries,
    TransferFunction as SourceTransferFunction,
};

/// Shared WGSL declarations, color conversion, and word-owned output entry point `main`.
/// Append a source fragment defining `source_rgb_at(x: u32, y: u32) -> vec3<f32>`.
pub const IMAGE_OUTPUT_SHADER: &str = concat!(
    include_str!("../shaders/image_orientation.wgsl"),
    include_str!("../shaders/image_output.wgsl"),
);

/// WGSL forward/inverse image-coordinate helpers using zero-based Exif orientation codes.
/// Callers validate nonempty extents and coordinate bounds before invoking either helper.
pub const IMAGE_ORIENTATION_SHADER: &str = include_str!("../shaders/image_orientation.wgsl");

pub(crate) const RGB_TO_IMAGE_SHADER: &str = concat!(
    include_str!("../shaders/image_orientation.wgsl"),
    include_str!("../shaders/image_output.wgsl"),
    include_str!("../shaders/rgb_to_image.wgsl"),
);

/// Source coordinates and color encoding for the shared output shader.
#[derive(Clone, Copy, Debug)]
pub struct ImageOutputSource {
    pub extent: Extent2d,
    pub orientation: OutputOrientation,
    /// Producer-specific scalar strides. The source fragment owns plane-bound validation.
    pub strides: [u32; 3],
    pub encoding: RgbColorEncoding,
}

/// Fixed 176-byte uniform for the shared output shader.
/// Construct it with [`Self::new`] to validate geometry, color, and packed addressing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ImageOutputParams {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) source_width: u32,
    pub(crate) source_height: u32,
    pub(crate) r_stride: u32,
    pub(crate) g_stride: u32,
    pub(crate) b_stride: u32,
    pub(crate) kind: u32,
    pub(crate) channels: u32,
    pub(crate) order: u32,
    pub(crate) matrix: u32,
    pub(crate) range: u32,
    pub(crate) siting_x: u32,
    pub(crate) siting_y: u32,
    pub(crate) subsample_x: u32,
    pub(crate) subsample_y: u32,
    pub(crate) bits: u32,
    pub(crate) storage_bits: u32,
    pub(crate) plane0_offset: u32,
    pub(crate) plane0_stride: u32,
    pub(crate) plane1_offset: u32,
    pub(crate) plane1_stride: u32,
    pub(crate) plane2_offset: u32,
    pub(crate) plane2_stride: u32,
    pub(crate) plane3_offset: u32,
    pub(crate) plane3_stride: u32,
    pub(crate) logical_size: u32,
    pub(crate) dispatch_width: u32,
    pub(crate) orientation: u32,
    pub(crate) source_transfer: u32,
    pub(crate) target_transfer: u32,
    pub(crate) _padding: u32,
    pub(crate) primaries_r: [f32; 4],
    pub(crate) primaries_g: [f32; 4],
    pub(crate) primaries_b: [f32; 4],
}
impl ImageOutputParams {
    /// Lowers the requested layout and source metadata before any GPU submission.
    pub fn new(
        layout: &ImageLayout,
        source: ImageOutputSource,
        dispatch_width: u32,
    ) -> Result<Self> {
        let prepared = prepare_image_output(layout)?;
        if source.extent.is_empty()
            || source.orientation.map_extent(source.extent) != layout.extent
            || dispatch_width == 0
        {
            return Err(Error::InvalidPayload(
                "image output source geometry or dispatch is invalid".into(),
            ));
        }
        let color = image_color_transform(source.encoding, &layout.format)?;
        Ok(Self {
            width: layout.extent.width,
            height: layout.extent.height,
            source_width: source.extent.width,
            source_height: source.extent.height,
            r_stride: source.strides[0],
            g_stride: source.strides[1],
            b_stride: source.strides[2],
            kind: prepared.kind,
            channels: prepared.channels,
            order: prepared.order,
            matrix: prepared.matrix,
            range: prepared.range,
            siting_x: prepared.siting_x,
            siting_y: prepared.siting_y,
            subsample_x: prepared.subsample_x,
            subsample_y: prepared.subsample_y,
            bits: prepared.bits,
            storage_bits: prepared.storage_bits,
            plane0_offset: prepared.plane_offsets[0],
            plane0_stride: prepared.plane_strides[0],
            plane1_offset: prepared.plane_offsets[1],
            plane1_stride: prepared.plane_strides[1],
            plane2_offset: prepared.plane_offsets[2],
            plane2_stride: prepared.plane_strides[2],
            plane3_offset: prepared.plane_offsets[3],
            plane3_stride: prepared.plane_strides[3],
            logical_size: to_shader_u32(layout.logical_size)?,
            dispatch_width,
            orientation: source.orientation.to_exif_value() - 1,
            source_transfer: color.source_transfer,
            target_transfer: color.target_transfer,
            _padding: 0,
            primaries_r: color.primaries[0],
            primaries_g: color.primaries[1],
            primaries_b: color.primaries[2],
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedImageOutput {
    kind: u32,
    channels: u32,
    order: u32,
    matrix: u32,
    range: u32,
    siting_x: u32,
    siting_y: u32,
    subsample_x: u32,
    subsample_y: u32,
    bits: u32,
    storage_bits: u32,
    plane_offsets: [u32; 4],
    plane_strides: [u32; 4],
}

pub(crate) fn prepare_image_output(layout: &ImageLayout) -> Result<PreparedImageOutput> {
    if layout.planes.len() != layout.format.planes.len() || layout.planes.len() > 4 {
        return Err(Error::Unsupported(format!(
            "generic GPU output supports 1..=4 planes, layout has {}",
            layout.planes.len()
        )));
    }
    let validated =
        ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
    if validated.logical_size != layout.logical_size {
        return Err(Error::InvalidPayload(
            "image output logical size disagrees with its planes".into(),
        ));
    }
    to_shader_u32(layout.logical_size)?;
    let mut plane_offsets = [0; 4];
    let mut plane_strides = [0; 4];
    for (index, plane) in layout.planes.iter().enumerate() {
        plane_offsets[index] = to_shader_u32(plane.offset)?;
        plane_strides[index] = to_shader_u32(plane.row_stride)?;
    }

    let class = classify_image_output_format(&layout.format)?;
    match class {
        ColorFormatClass::Rgb8 { storage, order } => {
            if matches!(layout.format.color_spec, ColorSpecification::Defined(color) if color.range != ColorRange::Full)
            {
                return Err(Error::Unsupported(
                    "RGB image output requires full range".into(),
                ));
            }
            let kind = match storage {
                RgbStorage::Interleaved => 0,
                RgbStorage::Planar => 1,
            };
            let (channels, order) = match order {
                RgbChannelOrder::Rgb => (3, 0),
                RgbChannelOrder::Bgr => (3, 1),
                RgbChannelOrder::Rgba => (4, 2),
                RgbChannelOrder::Bgra => (4, 3),
            };
            Ok(PreparedImageOutput {
                kind,
                channels,
                order,
                matrix: 1,
                range: 0,
                siting_x: 1,
                siting_y: 1,
                subsample_x: 1,
                subsample_y: 1,
                bits: 8,
                storage_bits: 8,
                plane_offsets,
                plane_strides,
            })
        }
        color => {
            let (matrix, range, siting_x, siting_y) = image_color_params(layout)?;
            let (subsample_x, subsample_y) = layout
                .format
                .chroma_subsampling
                .chroma_divisors()
                .unwrap_or((1, 1));
            let (kind, channels, order, bits, storage_bits) = match color {
                ColorFormatClass::Luma { bits, storage_bits } => {
                    (if bits == 8 { 2 } else { 3 }, 1, 0, bits, storage_bits)
                }
                ColorFormatClass::YuvPlanar {
                    bits, storage_bits, ..
                } => (4, 3, 0, bits, storage_bits),
                ColorFormatClass::YuvSemiplanar {
                    bits,
                    storage_bits,
                    chroma_order,
                    ..
                } => (
                    5,
                    3,
                    u32::from(chroma_order == ChromaOrder::CrCb),
                    bits,
                    storage_bits,
                ),
                ColorFormatClass::Yuv422Packed { order } => {
                    (6, 3, u32::from(order == Packed422Order::Uyvy), 8, 8)
                }
                ColorFormatClass::Rgb8 { .. } => {
                    unreachable!("RGB color classes were handled before YCbCr lowering")
                }
            };
            Ok(PreparedImageOutput {
                kind,
                channels,
                order,
                matrix,
                range,
                siting_x: u32::from(siting_x),
                siting_y: u32::from(siting_y),
                subsample_x: u32::from(subsample_x),
                subsample_y: u32::from(subsample_y),
                bits: u32::from(bits),
                storage_bits: u32::from(storage_bits),
                plane_offsets,
                plane_strides,
            })
        }
    }
}

fn classify_image_output_format(format: &PixelFormat) -> Result<ColorFormatClass> {
    match classify_pixel_format(format) {
        Ok(PixelFormatClass::Color(color)) => Ok(color),
        Ok(PixelFormatClass::Numeric(numeric)) => Err(numeric_image_output_error(numeric)),
        Err(error) => Err(Error::Unsupported(format!(
            "generic GPU output format is unsupported: {error}"
        ))),
    }
}

fn numeric_image_output_error(numeric: NumericFormatClass) -> Error {
    if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
        Error::Unsupported(
            "generic GPU output does not assign color semantics to numeric F64; portable WGSL also has no native F64 arithmetic"
                .into(),
        )
    } else {
        Error::Unsupported(format!(
            "generic GPU output does not assign color semantics to numeric format {numeric:?}"
        ))
    }
}
fn image_color_params(layout: &ImageLayout) -> Result<(u32, u32, u8, u8)> {
    let color = match layout.format.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Default | ColorSpecification::Undefined => {
            return Err(Error::Unsupported(
                "YCbCr GPU output requires an explicit matrix, range, and chroma location".into(),
            ));
        }
    };
    let matrix = match color.encoding {
        YcbcrEncoding::Bt601 => 0,
        YcbcrEncoding::Bt709 => 1,
        YcbcrEncoding::Bt2020 => 2,
        YcbcrEncoding::Bt2020ConstantLuminance => 3,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "YCbCr GPU output matrix {unsupported:?} is unsupported"
            )));
        }
    };
    let range = match color.range {
        ColorRange::Full => 0,
        ColorRange::Limited => 1,
    };
    let (subsample_x, subsample_y) = layout
        .format
        .chroma_subsampling
        .chroma_divisors()
        .unwrap_or((1, 1));
    Ok((
        matrix,
        range,
        image_siting(color.chroma_location.horizontal, subsample_x)?,
        image_siting(color.chroma_location.vertical, subsample_y)?,
    ))
}

fn image_siting(location: ChromaLocation, divisor: u8) -> Result<u8> {
    if divisor == 1 {
        return Ok(1);
    }
    match location {
        ChromaLocation::Center => Ok(0),
        ChromaLocation::Even => Ok(1),
        unsupported => Err(Error::Unsupported(format!(
            "chroma location {unsupported:?} is unsupported for {divisor}:1 output"
        ))),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ImageColorTransform {
    pub(crate) source_transfer: u32,
    pub(crate) target_transfer: u32,
    pub(crate) primaries: [[f32; 4]; 3],
}

pub(crate) fn image_color_transform(
    source: RgbColorEncoding,
    target: &PixelFormat,
) -> Result<ImageColorTransform> {
    let source_transfer = match source.transfer {
        SourceTransferFunction::Linear => 0,
        SourceTransferFunction::Srgb => 1,
        SourceTransferFunction::Bt709 => 2,
        SourceTransferFunction::Pq => 3,
        SourceTransferFunction::Hlg => 4,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output source transfer {unsupported:?} has no complete numeric contract"
            )));
        }
    };
    let target_color = match target.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Default | ColorSpecification::Undefined => {
            return Err(Error::Unsupported(
                "generic GPU output requires an explicit target color specification".into(),
            ));
        }
    };
    let target_transfer = match target_color.transfer {
        ImageTransferFunction::Linear => 0,
        ImageTransferFunction::Srgb | ImageTransferFunction::Sycc => 1,
        ImageTransferFunction::Bt709 => 2,
        ImageTransferFunction::Pq => 3,
        ImageTransferFunction::Hlg => 4,
        ImageTransferFunction::Bt2020 => 5,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target transfer {unsupported:?} is unsupported"
            )));
        }
    };
    let primaries = primaries_transform(source.primaries, target_color.space)?;
    Ok(ImageColorTransform {
        source_transfer,
        target_transfer,
        primaries,
    })
}

type Matrix3 = [[f32; 3]; 3];

const IDENTITY_3: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const BT709_TO_XYZ: Matrix3 = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175],
    [0.019_333_9, 0.119_192, 0.950_304_1],
];
const BT2020_TO_XYZ: Matrix3 = [
    [0.636_958, 0.144_616_9, 0.168_881],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.0, 0.028_072_7, 1.060_985_1],
];
const DISPLAY_P3_TO_XYZ: Matrix3 = [
    [0.486_570_95, 0.265_667_7, 0.198_217_29],
    [0.228_974_57, 0.691_738_55, 0.079_286_91],
    [0.0, 0.045_113_38, 1.043_944_4],
];
const XYZ_TO_BT709: Matrix3 = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];
const XYZ_TO_BT2020: Matrix3 = [
    [1.716_651_2, -0.355_670_8, -0.253_366_3],
    [-0.666_684_4, 1.616_481_2, 0.015_768_5],
    [0.017_639_9, -0.042_770_6, 0.942_103_1],
];
const XYZ_TO_DISPLAY_P3: Matrix3 = [
    [2.493_497, -0.931_383_6, -0.402_710_8],
    [-0.829_489, 1.762_664, 0.023_624_7],
    [0.035_845_8, -0.076_172_4, 0.956_884_5],
];

fn primaries_transform(source: RgbPrimaries, target: ColorSpace) -> Result<[[f32; 4]; 3]> {
    let source_index = match source {
        RgbPrimaries::Bt709 => 0,
        RgbPrimaries::Bt2020 => 1,
        RgbPrimaries::DisplayP3 => 2,
        RgbPrimaries::Undefined => {
            return Err(Error::Unsupported(
                "generic GPU output requires defined source RGB primaries".into(),
            ));
        }
    };
    let target_index = match target {
        ColorSpace::Bt709 => 0,
        ColorSpace::Bt2020 => 1,
        ColorSpace::DisplayP3 => 2,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target primaries {unsupported:?} are unsupported"
            )));
        }
    };
    let matrix = if source_index == target_index {
        IDENTITY_3
    } else {
        let source_to_xyz = [BT709_TO_XYZ, BT2020_TO_XYZ, DISPLAY_P3_TO_XYZ][source_index];
        let xyz_to_target = [XYZ_TO_BT709, XYZ_TO_BT2020, XYZ_TO_DISPLAY_P3][target_index];
        multiply_matrix3(xyz_to_target, source_to_xyz)
    };
    Ok(matrix.map(|row| [row[0], row[1], row[2], 0.0]))
}

fn multiply_matrix3(lhs: Matrix3, rhs: Matrix3) -> Matrix3 {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = lhs[row][0] * rhs[0][column]
                + lhs[row][1] * rhs[1][column]
                + lhs[row][2] * rhs[2][column];
        }
    }
    product
}
fn to_shader_u32(value: u64) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Unsupported("image output addressing exceeds WGSL u32".into()))
}

const _: () = {
    assert!(std::mem::size_of::<ImageOutputParams>() == 176);
    assert!(std::mem::align_of::<ImageOutputParams>() == 4);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_r) == 128);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_g) == 144);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_b) == 160);
};
