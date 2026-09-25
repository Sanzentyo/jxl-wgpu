//! Shared color conversion, orientation, and pitch-linear packing for GPU image producers.
//!
//! The shader fragment owns the output words. A producer supplies RGB and alpha as binary32
//! words in oriented coordinates. Unchanged F32 output preserves those words; color conversion
//! evaluates them as floats. Codec reconstruction can fuse with the render graph's output path.

use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification, ColorStorage,
    ImageLayout, NumericFormatClass, Packed422Order, PixelFormat, PixelFormatClass,
    RgbChannelOrder, TransferFunction as ImageTransferFunction, WgslNumericCapability,
    YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{
    Extent2d, OutputOrientation, RgbColorEncoding, TransferFunction as SourceTransferFunction,
    WhitePointAdaptation,
};

/// Shared WGSL declarations, color conversion, and word-owned output entry point `main`.
/// Append a source fragment defining `source_rgb_words_at(x: u32, y: u32) -> vec3<u32>`
/// and `source_alpha_word_at(x: u32, y: u32) -> u32`, both in oriented coordinates. These are
/// binary32 representations of unclipped source RGB and linear alpha. A source that needs no
/// reconstruction must return its stored words directly, preserving nonfinite and subnormal
/// representations without a float round trip.
pub const IMAGE_OUTPUT_SHADER: &str = concat!(
    include_str!("../shaders/image_orientation.wgsl"),
    include_str!("../shaders/alpha_output.wgsl"),
    include_str!("../shaders/image_transfer.wgsl"),
    include_str!("../shaders/tone_mapping.wgsl"),
    include_str!("../shaders/gamut_mapping.wgsl"),
    include_str!("../shaders/image_output.wgsl"),
);

/// Shared unbounded EOTF/OETF helpers with the same transfer selectors as image output.
/// BT.709 uses a linear negative extension; sRGB, PQ, HLG and BT.2020 reflect by sign.
pub const IMAGE_TRANSFER_SHADER: &str = include_str!("../shaders/image_transfer.wgsl");

/// WGSL forward/inverse image-coordinate helpers using zero-based Exif orientation codes.
/// Callers validate nonempty extents and coordinate bounds before invoking either helper.
pub const IMAGE_ORIENTATION_SHADER: &str = include_str!("../shaders/image_orientation.wgsl");

pub(crate) const RGB_TO_IMAGE_SHADER: &str = concat!(
    include_str!("../shaders/image_orientation.wgsl"),
    include_str!("../shaders/alpha_output.wgsl"),
    include_str!("../shaders/image_transfer.wgsl"),
    include_str!("../shaders/tone_mapping.wgsl"),
    include_str!("../shaders/gamut_mapping.wgsl"),
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

/// Coordinates for packing values whose color transform has already completed.
#[derive(Clone, Copy, Debug)]
pub struct ImageOutputGeometry {
    pub extent: Extent2d,
    pub orientation: OutputOrientation,
    /// Producer-specific scalar strides. The source fragment validates its own plane bounds.
    pub strides: [u32; 3],
}

/// Shared output association helper. Conversion follows the requested color transfer and
/// precedes quantization; the alpha value itself is unchanged.
pub const ALPHA_OUTPUT_SHADER: &str = include_str!("../shaders/alpha_output.wgsl");

/// Resolved conversion between source and output alpha association.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaConversion {
    #[default]
    Preserve = 0,
    Unpremultiply = 1,
    Premultiply = 2,
}

/// Fixed 304-byte uniform for the shared output shader.
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
    pub(crate) identity_color_transform: u32,
    pub(crate) primaries_r: [f32; 4],
    pub(crate) primaries_g: [f32; 4],
    pub(crate) primaries_b: [f32; 4],
    pub(crate) alpha: [u32; 4],
    pub(crate) transfer_parameters: [f32; 4],
    pub(crate) source_luminance: [f32; 4],
    pub(crate) target_luminance: [f32; 4],
    pub(crate) tone_mapping: crate::ToneMappingParams,
    pub(crate) gamut_mapping: crate::GamutMappingParams,
}
impl ImageOutputParams {
    /// Pack one or three F32 components without assigning RGB or ICC meaning to their storage.
    pub fn for_components(
        layout: &ImageLayout,
        source: ImageOutputGeometry,
        dispatch_width: u32,
    ) -> Result<Self> {
        use jxl_gpu_formats::{Channel, PixelFormat, PlaneFormat, PlaneSampling, SampleKind};
        let channels: &[Channel] = if layout.planes.len() == 1 {
            &[Channel::X]
        } else {
            &[Channel::X, Channel::Y, Channel::Z]
        };
        let mut expected = PixelFormat::non_color(SampleKind::Float, 32, channels);
        expected.planes = channels
            .iter()
            .map(|&channel| PlaneFormat::separate_words(PlaneSampling::FULL, 1, &[channel], 32))
            .collect();
        if layout.format != expected {
            return Err(Error::InvalidPayload(
                "component packing requires one or three planar native F32 channels".into(),
            ));
        }
        let validated =
            ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
        if validated.logical_size != layout.logical_size {
            return Err(Error::InvalidPayload(
                "component layout logical size disagrees with its planes".into(),
            ));
        }
        let mut plane_offsets = [0; 4];
        let mut plane_strides = [0; 4];
        for (index, plane) in layout.planes.iter().enumerate() {
            plane_offsets[index] = to_shader_u32(plane.offset)?;
            plane_strides[index] = to_shader_u32(plane.row_stride)?;
        }
        Self::lower(
            layout,
            source,
            dispatch_width,
            PreparedImageOutput {
                kind: 1,
                channels: channels.len() as u32,
                order: 0,
                matrix: 1,
                range: 0,
                siting_x: 1,
                siting_y: 1,
                subsample_x: 1,
                subsample_y: 1,
                bits: 32,
                storage_bits: 32,
                plane_offsets,
                plane_strides,
            },
            ImageColorTransform {
                source_transfer: 0,
                target_transfer: 0,
                source_gamma: 1.0,
                target_gamma: 1.0,
                primaries: IDENTITY_3.map(|row| [row[0] as f32, row[1] as f32, row[2] as f32, 0.0]),
            },
        )
    }

    /// Clamp target-linear components at or below a codec reconstruction threshold to zero.
    /// This happens before the target transfer; it does not clamp encoded reference samples.
    pub fn with_linear_black_threshold(mut self, threshold: f32) -> Result<Self> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(Error::InvalidPayload(
                "linear black threshold must be finite and nonnegative".into(),
            ));
        }
        self.transfer_parameters[2] = threshold;
        self.identity_color_transform = 0;
        Ok(self)
    }

    /// Lowers the requested layout and source metadata before any GPU submission.
    pub fn new(
        layout: &ImageLayout,
        source: ImageOutputSource,
        dispatch_width: u32,
        adaptation: WhitePointAdaptation,
    ) -> Result<Self> {
        let prepared = prepare_image_output(layout)?;
        let color = image_color_transform(source.encoding, &layout.format, adaptation)?;
        let mut params = Self::lower(
            layout,
            ImageOutputGeometry {
                extent: source.extent,
                orientation: source.orientation,
                strides: source.strides,
            },
            dispatch_width,
            prepared,
            color,
        )?;
        let ColorSpecification::Defined(target) = layout.format.color_spec else {
            unreachable!("image color transform validates enumerated output")
        };
        let space = target.space.rgb_space().ok_or_else(|| {
            Error::InvalidPayload("image output requires target RGB chromaticities".into())
        })?;
        params.gamut_mapping = crate::GamutMappingParams::for_space(space)?;
        Ok(params)
    }

    /// Map the already declared target-primary linear RGB after tone mapping and before its
    /// transfer function. Protected tone-map samples retain their absolute light unchanged.
    /// ICC/device and numeric parameter records have no linear RGB gamut and reject this option.
    pub fn with_gamut_mapping(
        mut self,
        mapping: Option<jxl_gpu_protocol::GamutMapping>,
    ) -> Result<Self> {
        self.gamut_mapping = self.gamut_mapping.with_mapping(mapping)?;
        Ok(self)
    }

    /// Convert display-relative RGB using an explicit unit-white intensity in nits.
    /// PQ is absolute light; HLG includes the display OOTF using each encoding's own
    /// luminance coefficients. SDR and linear values remain relative to this intensity.
    /// This conversion does not tone-map, change the display peak, or clip RGB.
    pub fn new_with_intensity_target(
        layout: &ImageLayout,
        source: ImageOutputSource,
        dispatch_width: u32,
        adaptation: WhitePointAdaptation,
        intensity_target: f32,
    ) -> Result<Self> {
        let mut params = Self::new(layout, source, dispatch_width, adaptation)?;
        let ColorSpecification::Defined(target) = layout.format.color_spec else {
            return Err(Error::InvalidPayload(
                "display luminance requires enumerated RGB".into(),
            ));
        };
        let target_space = target.space.rgb_space().ok_or_else(|| {
            Error::InvalidPayload("display luminance requires target chromaticities".into())
        })?;
        params.source_luminance =
            display_luminance(source.encoding.space, intensity_target, false)?;
        params.target_luminance = display_luminance(target_space, intensity_target, true)?;
        params.transfer_parameters[3] = intensity_target;
        Ok(params)
    }

    /// Map explicit source luminance to a display range in target-linear RGB before its OETF.
    /// The same word-owned dispatch performs conversion, mapping, association and quantization.
    pub fn new_with_tone_mapping(
        layout: &ImageLayout,
        source: ImageOutputSource,
        dispatch_width: u32,
        adaptation: WhitePointAdaptation,
        mapping: jxl_gpu_protocol::ToneMapping,
    ) -> Result<Self> {
        let mut params = Self::new_with_intensity_target(
            layout,
            source,
            dispatch_width,
            adaptation,
            mapping.source().white().nits(),
        )?;
        let ColorSpecification::Defined(target) = layout.format.color_spec else {
            unreachable!("validated enumerated color target")
        };
        params.target_luminance = display_luminance(
            target.space.rgb_space().expect("validated RGB geometry"),
            mapping.target().white().nits(),
            true,
        )?;
        params.tone_mapping = crate::ToneMappingParams::new(mapping)?;
        if !matches!(params.tone_mapping.range[3], 1.0 | 5.0) || params.tone_mapping.knee[2] != 1.0
        {
            params.identity_color_transform = 0;
        }
        Ok(params)
    }

    /// Pack RGB or gray device values already in the exact target ICC profile. This performs
    /// no curve or matrix evaluation, preserving F32 device samples when alpha is preserved.
    /// A gray source supplies its one color word in `source_rgb_words_at(...).x`; alpha is separate.
    pub fn for_icc_device(
        layout: &ImageLayout,
        source: ImageOutputGeometry,
        profile: &jxl_gpu_protocol::icc::IccProfile,
        dispatch_width: u32,
    ) -> Result<Self> {
        if !matches!(&layout.format.color_spec, ColorSpecification::Icc(target) if target == profile)
        {
            return Err(Error::InvalidPayload(
                "device packing requires the exact source ICC profile".into(),
            ));
        }
        let prepared = prepare_output(layout, true)?;
        if prepared.kind > 1 {
            return Err(Error::Unsupported(
                "ICC device packing requires RGB or gray storage".into(),
            ));
        }
        Self::lower(
            layout,
            source,
            dispatch_width,
            prepared,
            ImageColorTransform {
                source_transfer: 0,
                target_transfer: 0,
                source_gamma: 1.0,
                target_gamma: 1.0,
                primaries: IDENTITY_3.map(|row| [row[0] as f32, row[1] as f32, row[2] as f32, 0.0]),
            },
        )
    }

    fn lower(
        layout: &ImageLayout,
        source: ImageOutputGeometry,
        dispatch_width: u32,
        prepared: PreparedImageOutput,
        color: ImageColorTransform,
    ) -> Result<Self> {
        if source.extent.is_empty()
            || source.orientation.map_extent(source.extent) != layout.extent
            || dispatch_width == 0
        {
            return Err(Error::InvalidPayload(
                "image output source geometry or dispatch is invalid".into(),
            ));
        }
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
            identity_color_transform: u32::from(
                color.source_transfer == color.target_transfer
                    && color.source_gamma == color.target_gamma
                    && color.primaries
                        == IDENTITY_3.map(|row| [row[0] as f32, row[1] as f32, row[2] as f32, 0.0]),
            ),
            primaries_r: color.primaries[0],
            primaries_g: color.primaries[1],
            primaries_b: color.primaries[2],
            alpha: [0; 4],
            transfer_parameters: [color.source_gamma, color.target_gamma, -1.0, 0.0],
            source_luminance: [0.0; 4],
            target_luminance: [0.0; 4],
            tone_mapping: crate::ToneMappingParams::default(),
            gamut_mapping: crate::GamutMappingParams::default(),
        })
    }

    /// Adjusts RGB association after color conversion, including when the output omits alpha.
    /// The producer must supply the matching alpha plane through `source_alpha_word_at`.
    #[must_use]
    pub const fn with_alpha_conversion(mut self, conversion: AlphaConversion) -> Self {
        self.alpha[0] = conversion as u32;
        self
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
    prepare_output(layout, false)
}

fn prepare_output(layout: &ImageLayout, gray_device: bool) -> Result<PreparedImageOutput> {
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
        ColorFormatClass::IccDevice { .. } => Err(Error::Unsupported(
            "ICC device output requires the device component packer".into(),
        )),
        ColorFormatClass::Gray {
            sample,
            storage,
            alpha,
        } => {
            if !gray_device
                && matches!(layout.format.color_spec, ColorSpecification::Defined(color) if color.range != ColorRange::Full)
            {
                return Err(Error::Unsupported(
                    "Gray image output requires full range".into(),
                ));
            }
            Ok(PreparedImageOutput {
                kind: match storage {
                    ColorStorage::Interleaved => 0,
                    ColorStorage::Planar => 1,
                },
                channels: if alpha { 2 } else { 1 },
                // ICC gray copies its device sample; enumerated gray projects target-linear
                // RGB to luminance before the target transfer. Alpha occupies position one.
                order: if gray_device { 4 } else { 5 },
                matrix: 1,
                range: 0,
                siting_x: 1,
                siting_y: 1,
                subsample_x: 1,
                subsample_y: 1,
                bits: u32::from(sample.bits()),
                storage_bits: u32::from(sample.bits()),
                plane_offsets,
                plane_strides,
            })
        }
        ColorFormatClass::Rgb {
            sample,
            storage,
            order,
        } => {
            if matches!(layout.format.color_spec, ColorSpecification::Defined(color) if color.range != ColorRange::Full)
            {
                return Err(Error::Unsupported(
                    "RGB image output requires full range".into(),
                ));
            }
            let kind = match storage {
                ColorStorage::Interleaved => 0,
                ColorStorage::Planar => 1,
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
                bits: u32::from(sample.bits()),
                storage_bits: u32::from(sample.bits()),
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
                ColorFormatClass::Rgb { .. }
                | ColorFormatClass::Gray { .. }
                | ColorFormatClass::IccDevice { .. } => {
                    unreachable!("RGB and gray color classes were handled before YCbCr lowering")
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
        ColorSpecification::Icc(_) => return Err(Error::Unsupported("ICC color must be converted by the resident ICC pipeline before enumerated RGB packing or display".into())),
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
    pub(crate) source_gamma: f32,
    pub(crate) target_gamma: f32,
    pub(crate) primaries: [[f32; 4]; 3],
}

/// Selector and gamma for [`IMAGE_TRANSFER_SHADER`], shared by image packing,
/// ordered ICC RGB stages and encoder source normalization.
pub fn transfer_parameters(transfer: SourceTransferFunction) -> (u32, f32) {
    let code = match transfer {
        SourceTransferFunction::Linear => 0,
        SourceTransferFunction::Srgb => 1,
        SourceTransferFunction::Bt709 => 2,
        SourceTransferFunction::Bt2020 => 5,
        SourceTransferFunction::Pq => 3,
        SourceTransferFunction::Hlg => 4,
        SourceTransferFunction::Gamma(_) => 6,
        SourceTransferFunction::Dci => 7,
    };
    let gamma = match transfer {
        SourceTransferFunction::Gamma(exponent) => exponent.value(),
        _ => 1.0,
    };
    (code, gamma)
}

pub(crate) fn image_color_transform(
    source: RgbColorEncoding,
    target: &PixelFormat,
    adaptation: WhitePointAdaptation,
) -> Result<ImageColorTransform> {
    let (source_transfer, source_gamma) = transfer_parameters(source.transfer);
    let target_color = match target.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Icc(_) => return Err(Error::Unsupported("ICC color must be converted by the resident ICC pipeline before enumerated RGB packing or display".into())),
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
        ImageTransferFunction::Gamma(_) => 6,
        ImageTransferFunction::Dci => 7,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target transfer {unsupported:?} is unsupported"
            )));
        }
    };
    let target_space = target_color.space.rgb_space().ok_or_else(|| {
        Error::Unsupported("RGB output requires defined target chromaticities".into())
    })?;
    let primaries = rgb_color_matrix(source.space, target_space, adaptation)?;
    Ok(ImageColorTransform {
        source_transfer,
        target_transfer,
        source_gamma,
        target_gamma: match target_color.transfer {
            ImageTransferFunction::Gamma(exponent) => exponent.value(),
            _ => 1.0,
        },
        primaries,
    })
}

mod luminance;
mod matrix;
pub use luminance::display_luminance;
use matrix::IDENTITY as IDENTITY_3;
pub use matrix::rgb_color_matrix;

fn to_shader_u32(value: u64) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Unsupported("image output addressing exceeds WGSL u32".into()))
}

const _: () = {
    assert!(std::mem::size_of::<ImageOutputParams>() == 304);
    assert!(std::mem::align_of::<ImageOutputParams>() == 4);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_r) == 128);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_g) == 144);
    assert!(std::mem::offset_of!(ImageOutputParams, primaries_b) == 160);
    assert!(std::mem::offset_of!(ImageOutputParams, alpha) == 176);
    assert!(std::mem::offset_of!(ImageOutputParams, transfer_parameters) == 192);
    assert!(std::mem::offset_of!(ImageOutputParams, source_luminance) == 208);
    assert!(std::mem::offset_of!(ImageOutputParams, target_luminance) == 224);
    assert!(std::mem::offset_of!(ImageOutputParams, tone_mapping) == 240);
    assert!(std::mem::offset_of!(ImageOutputParams, gamut_mapping) == 288);
};
