use thiserror::Error;

use crate::{
    ByteOrder, Channel, ChromaOrder, ChromaSubsampling, ColorModel, Packed422Order,
    PackingFieldKind, PixelFormat, PixelFormatError, PlaneFormat, PlaneSampling, RgbChannelOrder,
    SampleKind, Swizzle,
};

/// Storage shape of a color-bearing pitch-linear [`PixelFormat`].
///
/// This classification describes channel packing only. Matrix, transfer, range, and chroma
/// location remain in [`PixelFormat::color_spec`] and must still be negotiated by the consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorFormatClass {
    Rgb8 {
        storage: RgbStorage,
        order: RgbChannelOrder,
    },
    Luma {
        bits: u8,
        storage_bits: u8,
    },
    YuvPlanar {
        subsampling: ChromaSubsampling,
        bits: u8,
        storage_bits: u8,
    },
    YuvSemiplanar {
        subsampling: ChromaSubsampling,
        bits: u8,
        storage_bits: u8,
        chroma_order: ChromaOrder,
    },
    Yuv422Packed {
        order: Packed422Order,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbStorage {
    Interleaved,
    Planar,
}

/// Portable WGSL arithmetic available for one numeric storage class.
///
/// This is deliberately not a display capability. A caller must supply its own numeric-to-color
/// semantics before a [`NumericFormatClass`] can be visualized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgslNumericCapability {
    /// Components are directly representable as WGSL `u32`, `i32`, or `f32` values.
    Native32,
    /// Sub-32-bit components require word extraction and, for signed values, sign extension.
    Packed32,
    /// Portable WGSL has no native 64-bit floating-point arithmetic.
    UnavailableFloat64,
}

impl WgslNumericCapability {
    #[must_use]
    pub const fn supports_arithmetic(self) -> bool {
        !matches!(self, Self::UnavailableFloat64)
    }
}

/// Storage metadata for a non-color numeric image.
///
/// No range, normalization, or channel-to-color mapping is implied by this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumericFormatClass {
    pub sample_kind: SampleKind,
    pub bits_per_component: u8,
    pub components: u8,
    pub wgsl: WgslNumericCapability,
}

/// Semantic classification of the canonical pitch-linear formats understood by the portable GPU
/// pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormatClass {
    Color(ColorFormatClass),
    Numeric(NumericFormatClass),
}

impl PixelFormatClass {
    #[must_use]
    pub const fn color(self) -> Option<ColorFormatClass> {
        match self {
            Self::Color(color) => Some(color),
            Self::Numeric(_) => None,
        }
    }

    #[must_use]
    pub const fn numeric(self) -> Option<NumericFormatClass> {
        match self {
            Self::Color(_) => None,
            Self::Numeric(numeric) => Some(numeric),
        }
    }
}

/// Classifies a generic [`PixelFormat`] without assigning color semantics to numeric data.
pub fn classify_pixel_format(
    format: &PixelFormat,
) -> Result<PixelFormatClass, PixelFormatClassificationError> {
    format.validate()?;
    if format.byte_order == ByteOrder::Big {
        return Err(PixelFormatClassificationError::BigEndian);
    }
    match format.model {
        ColorModel::NonColor => classify_numeric(format).map(PixelFormatClass::Numeric),
        ColorModel::Rgb => classify_rgb(format).map(PixelFormatClass::Color),
        ColorModel::Ycbcr => classify_ycbcr(format).map(PixelFormatClass::Color),
        unsupported => Err(PixelFormatClassificationError::UnsupportedColorModel(
            unsupported,
        )),
    }
}

fn classify_numeric(
    format: &PixelFormat,
) -> Result<NumericFormatClass, PixelFormatClassificationError> {
    if format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() != 1
        || !is_full_sample_plane(&format.planes[0])
    {
        return Err(PixelFormatClassificationError::UnsupportedNumericPacking);
    }
    let stored = stored_plane(&format.planes[0])
        .ok_or(PixelFormatClassificationError::UnsupportedNumericPacking)?;
    let expected_channels = match stored.channels.as_slice() {
        [Channel::X] => 1,
        [Channel::X, Channel::Y] => 2,
        _ => return Err(PixelFormatClassificationError::UnsupportedNumericPacking),
    };
    if stored.bits != stored.storage_bits
        || !matches!(stored.bits, 8 | 16 | 32 | 64)
        || format.swizzle
            != if expected_channels == 1 {
                Swizzle::X000
            } else {
                Swizzle::XY00
            }
    {
        return Err(PixelFormatClassificationError::UnsupportedNumericPacking);
    }
    let wgsl = match (format.sample_kind, stored.bits) {
        (SampleKind::Unsigned | SampleKind::Signed, 8 | 16) => WgslNumericCapability::Packed32,
        (SampleKind::Unsigned | SampleKind::Signed | SampleKind::Float, 32) => {
            WgslNumericCapability::Native32
        }
        (SampleKind::Float, 64) => WgslNumericCapability::UnavailableFloat64,
        _ => return Err(PixelFormatClassificationError::UnsupportedNumericPacking),
    };
    Ok(NumericFormatClass {
        sample_kind: format.sample_kind,
        bits_per_component: stored.bits,
        components: expected_channels,
        wgsl,
    })
}

fn classify_rgb(format: &PixelFormat) -> Result<ColorFormatClass, PixelFormatClassificationError> {
    require_unsigned_color(format)?;
    if format.chroma_subsampling != ChromaSubsampling::None {
        return Err(PixelFormatClassificationError::UnsupportedColorPacking);
    }
    let stored = format
        .planes
        .iter()
        .map(stored_plane)
        .collect::<Option<Vec<_>>>()
        .ok_or(PixelFormatClassificationError::UnsupportedColorPacking)?;
    let storage = match stored.as_slice() {
        [plane]
            if is_full_sample_plane(&format.planes[0])
                && matches!(plane.channels.len(), 3 | 4)
                && canonical_rgb_channels(&plane.channels)
                && plane.bits == 8
                && plane.storage_bits == 8 =>
        {
            RgbStorage::Interleaved
        }
        planes
            if matches!(planes.len(), 3 | 4)
                && format.planes.iter().all(is_full_sample_plane)
                && planes.iter().all(|plane| {
                    plane.channels.len() == 1 && plane.bits == 8 && plane.storage_bits == 8
                })
                && canonical_rgb_channels(
                    &planes
                        .iter()
                        .flat_map(|plane| plane.channels.iter().copied())
                        .collect::<Vec<_>>(),
                ) =>
        {
            RgbStorage::Planar
        }
        _ => return Err(PixelFormatClassificationError::UnsupportedColorPacking),
    };
    let channels = stored
        .iter()
        .map(|plane| plane.channels.len())
        .sum::<usize>();
    let order = match (channels, format.swizzle) {
        (3, Swizzle::XYZ1) => RgbChannelOrder::Rgb,
        (3, Swizzle::ZYX1) => RgbChannelOrder::Bgr,
        (4, Swizzle::XYZW) => RgbChannelOrder::Rgba,
        (4, Swizzle::ZYXW) => RgbChannelOrder::Bgra,
        _ => return Err(PixelFormatClassificationError::UnsupportedColorPacking),
    };
    Ok(ColorFormatClass::Rgb8 { storage, order })
}

fn classify_ycbcr(
    format: &PixelFormat,
) -> Result<ColorFormatClass, PixelFormatClassificationError> {
    require_unsigned_color(format)?;
    let stored = format
        .planes
        .iter()
        .map(stored_plane)
        .collect::<Option<Vec<_>>>()
        .ok_or(PixelFormatClassificationError::UnsupportedColorPacking)?;
    match stored.as_slice() {
        [y] if y.channels.as_slice() == [Channel::X]
            && format.chroma_subsampling == ChromaSubsampling::None
            && is_full_sample_plane(&format.planes[0])
            && matches!((y.bits, y.storage_bits), (8, 8) | (16, 16)) =>
        {
            Ok(ColorFormatClass::Luma {
                bits: y.bits,
                storage_bits: y.storage_bits,
            })
        }
        [packed]
            if format.chroma_subsampling == ChromaSubsampling::Cs422
                && format.planes[0].sampling == PlaneSampling::FULL
                && format.planes[0].pixels_per_element == 2
                && packed.bits == 8
                && packed.storage_bits == 8
                && packed.channels.as_slice()
                    == [Channel::X, Channel::Y, Channel::X, Channel::Z] =>
        {
            Ok(ColorFormatClass::Yuv422Packed {
                order: Packed422Order::Yuyv,
            })
        }
        [packed]
            if format.chroma_subsampling == ChromaSubsampling::Cs422
                && format.planes[0].sampling == PlaneSampling::FULL
                && format.planes[0].pixels_per_element == 2
                && packed.bits == 8
                && packed.storage_bits == 8
                && packed.channels.as_slice()
                    == [Channel::Y, Channel::X, Channel::Z, Channel::X] =>
        {
            Ok(ColorFormatClass::Yuv422Packed {
                order: Packed422Order::Uyvy,
            })
        }
        [y, cb, cr]
            if y.channels.as_slice() == [Channel::X]
                && cb.channels.as_slice() == [Channel::Y]
                && cr.channels.as_slice() == [Channel::Z]
                && equal_depths(&[y, cb, cr])
                && valid_yuv_depth(y.bits, y.storage_bits)
                && valid_yuv_planes(format, &[0, 1, 2]) =>
        {
            Ok(ColorFormatClass::YuvPlanar {
                subsampling: format.chroma_subsampling,
                bits: y.bits,
                storage_bits: y.storage_bits,
            })
        }
        [y, chroma]
            if y.channels.as_slice() == [Channel::X]
                && matches!(
                    chroma.channels.as_slice(),
                    [Channel::Y, Channel::Z] | [Channel::Z, Channel::Y]
                )
                && equal_depths(&[y, chroma])
                && valid_yuv_depth(y.bits, y.storage_bits)
                && valid_yuv_planes(format, &[0, 1]) =>
        {
            Ok(ColorFormatClass::YuvSemiplanar {
                subsampling: format.chroma_subsampling,
                bits: y.bits,
                storage_bits: y.storage_bits,
                chroma_order: if chroma.channels.as_slice() == [Channel::Y, Channel::Z] {
                    ChromaOrder::CbCr
                } else {
                    ChromaOrder::CrCb
                },
            })
        }
        _ => Err(PixelFormatClassificationError::UnsupportedColorPacking),
    }
}

fn require_unsigned_color(format: &PixelFormat) -> Result<(), PixelFormatClassificationError> {
    if format.sample_kind == SampleKind::Unsigned {
        Ok(())
    } else {
        Err(PixelFormatClassificationError::UnsupportedColorSampleKind(
            format.sample_kind,
        ))
    }
}

fn is_full_sample_plane(plane: &PlaneFormat) -> bool {
    plane.sampling == PlaneSampling::FULL && plane.pixels_per_element == 1
}

fn canonical_rgb_channels(channels: &[Channel]) -> bool {
    matches!(
        channels,
        [Channel::X, Channel::Y, Channel::Z] | [Channel::X, Channel::Y, Channel::Z, Channel::W]
    )
}

fn valid_yuv_planes(format: &PixelFormat, indices: &[usize]) -> bool {
    let Some((horizontal, vertical)) = format.chroma_subsampling.chroma_divisors() else {
        return false;
    };
    indices.iter().copied().all(|index| {
        let expected = if index == 0 {
            PlaneSampling::FULL
        } else {
            PlaneSampling::new(horizontal, vertical)
        };
        format.planes[index].sampling == expected && format.planes[index].pixels_per_element == 1
    })
}

fn valid_yuv_depth(bits: u8, storage_bits: u8) -> bool {
    matches!((bits, storage_bits), (8, 8) | (10 | 12 | 16, 16))
}

fn equal_depths(planes: &[&StoredPlane]) -> bool {
    planes
        .windows(2)
        .all(|pair| pair[0].bits == pair[1].bits && pair[0].storage_bits == pair[1].storage_bits)
}

struct StoredPlane {
    channels: Vec<Channel>,
    bits: u8,
    storage_bits: u8,
}

fn stored_plane(plane: &PlaneFormat) -> Option<StoredPlane> {
    let mut channels = Vec::with_capacity(plane.words.len());
    let mut bits = None;
    let mut storage_bits = None;
    for word in &plane.words {
        let field = word.fields.first()?;
        let PackingFieldKind::Channel(channel) = field.kind else {
            return None;
        };
        if word
            .fields
            .iter()
            .skip(1)
            .any(|field| !matches!(field.kind, PackingFieldKind::Padding))
        {
            return None;
        }
        let word_bits = u8::try_from(word.bits()).ok()?;
        if bits
            .replace(field.bits)
            .is_some_and(|old| old != field.bits)
            || storage_bits
                .replace(word_bits)
                .is_some_and(|old| old != word_bits)
        {
            return None;
        }
        channels.push(channel);
    }
    Some(StoredPlane {
        channels,
        bits: bits?,
        storage_bits: storage_bits?,
    })
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PixelFormatClassificationError {
    #[error(transparent)]
    Invalid(#[from] PixelFormatError),
    #[error("big-endian packing is not supported by portable word-addressed GPU shaders")]
    BigEndian,
    #[error("color model {0:?} has no portable color or numeric processing class")]
    UnsupportedColorModel(ColorModel),
    #[error("color model uses unsupported sample kind {0:?}")]
    UnsupportedColorSampleKind(SampleKind),
    #[error("color channel packing is not supported by the portable processing class")]
    UnsupportedColorPacking,
    #[error("numeric channel packing is not supported by the portable processing class")]
    UnsupportedNumericPacking,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, PixelFormat,
        vpi::VpiPitchLinearFormat as Vpi,
    };

    fn numeric(
        sample_kind: SampleKind,
        bits: u8,
        components: u8,
        wgsl: WgslNumericCapability,
    ) -> PixelFormatClass {
        PixelFormatClass::Numeric(NumericFormatClass {
            sample_kind,
            bits_per_component: bits,
            components,
            wgsl,
        })
    }

    fn color(class: ColorFormatClass) -> PixelFormatClass {
        PixelFormatClass::Color(class)
    }

    #[test]
    fn vpi_pitch_linear_inventory_has_explicit_semantic_classification() {
        use WgslNumericCapability::{Native32, Packed32, UnavailableFloat64};

        let expected = [
            (Vpi::U8, numeric(SampleKind::Unsigned, 8, 1, Packed32)),
            (Vpi::S8, numeric(SampleKind::Signed, 8, 1, Packed32)),
            (Vpi::U16, numeric(SampleKind::Unsigned, 16, 1, Packed32)),
            (Vpi::U32, numeric(SampleKind::Unsigned, 32, 1, Native32)),
            (Vpi::S32, numeric(SampleKind::Signed, 32, 1, Native32)),
            (Vpi::S16, numeric(SampleKind::Signed, 16, 1, Packed32)),
            (Vpi::TwoS16, numeric(SampleKind::Signed, 16, 2, Packed32)),
            (Vpi::F32, numeric(SampleKind::Float, 32, 1, Native32)),
            (
                Vpi::F64,
                numeric(SampleKind::Float, 64, 1, UnavailableFloat64),
            ),
            (Vpi::TwoF32, numeric(SampleKind::Float, 32, 2, Native32)),
            (
                Vpi::Y8,
                color(ColorFormatClass::Luma {
                    bits: 8,
                    storage_bits: 8,
                }),
            ),
            (
                Vpi::Y8Er,
                color(ColorFormatClass::Luma {
                    bits: 8,
                    storage_bits: 8,
                }),
            ),
            (
                Vpi::Y16,
                color(ColorFormatClass::Luma {
                    bits: 16,
                    storage_bits: 16,
                }),
            ),
            (
                Vpi::Y16Er,
                color(ColorFormatClass::Luma {
                    bits: 16,
                    storage_bits: 16,
                }),
            ),
            (
                Vpi::Nv12,
                color(ColorFormatClass::YuvSemiplanar {
                    subsampling: ChromaSubsampling::Cs420,
                    bits: 8,
                    storage_bits: 8,
                    chroma_order: ChromaOrder::CbCr,
                }),
            ),
            (
                Vpi::Nv12Er,
                color(ColorFormatClass::YuvSemiplanar {
                    subsampling: ChromaSubsampling::Cs420,
                    bits: 8,
                    storage_bits: 8,
                    chroma_order: ChromaOrder::CbCr,
                }),
            ),
            (
                Vpi::Nv24,
                color(ColorFormatClass::YuvSemiplanar {
                    subsampling: ChromaSubsampling::Cs444,
                    bits: 8,
                    storage_bits: 8,
                    chroma_order: ChromaOrder::CbCr,
                }),
            ),
            (
                Vpi::Nv24Er,
                color(ColorFormatClass::YuvSemiplanar {
                    subsampling: ChromaSubsampling::Cs444,
                    bits: 8,
                    storage_bits: 8,
                    chroma_order: ChromaOrder::CbCr,
                }),
            ),
            (
                Vpi::Uyvy,
                color(ColorFormatClass::Yuv422Packed {
                    order: Packed422Order::Uyvy,
                }),
            ),
            (
                Vpi::UyvyEr,
                color(ColorFormatClass::Yuv422Packed {
                    order: Packed422Order::Uyvy,
                }),
            ),
            (
                Vpi::Yuyv,
                color(ColorFormatClass::Yuv422Packed {
                    order: Packed422Order::Yuyv,
                }),
            ),
            (
                Vpi::YuyvEr,
                color(ColorFormatClass::Yuv422Packed {
                    order: Packed422Order::Yuyv,
                }),
            ),
            (
                Vpi::Rgb8,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Interleaved,
                    order: RgbChannelOrder::Rgb,
                }),
            ),
            (
                Vpi::Bgr8,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Interleaved,
                    order: RgbChannelOrder::Bgr,
                }),
            ),
            (
                Vpi::Rgba8,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Interleaved,
                    order: RgbChannelOrder::Rgba,
                }),
            ),
            (
                Vpi::Bgra8,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Interleaved,
                    order: RgbChannelOrder::Bgra,
                }),
            ),
            (
                Vpi::Rgb8Planar,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Planar,
                    order: RgbChannelOrder::Rgb,
                }),
            ),
            (
                Vpi::Bgr8Planar,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Planar,
                    order: RgbChannelOrder::Bgr,
                }),
            ),
            (
                Vpi::Rgba8Planar,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Planar,
                    order: RgbChannelOrder::Rgba,
                }),
            ),
            (
                Vpi::Bgra8Planar,
                color(ColorFormatClass::Rgb8 {
                    storage: RgbStorage::Planar,
                    order: RgbChannelOrder::Bgra,
                }),
            ),
        ];

        assert_eq!(expected.len(), Vpi::ALL.len());
        for ((predefined, expected_class), inventory) in expected.into_iter().zip(Vpi::ALL) {
            assert_eq!(
                predefined, inventory,
                "the explicit table must cover ALL in order"
            );
            assert_eq!(
                classify_pixel_format(&predefined.pixel_format()).unwrap(),
                expected_class,
                "{}",
                predefined.name()
            );
        }
        assert_eq!(
            Vpi::ALL
                .into_iter()
                .filter(|format| classify_pixel_format(&format.pixel_format())
                    .unwrap()
                    .color()
                    .is_some())
                .count(),
            20
        );
        assert_eq!(
            Vpi::ALL
                .into_iter()
                .filter(|format| classify_pixel_format(&format.pixel_format())
                    .unwrap()
                    .numeric()
                    .is_some())
                .count(),
            10
        );
    }

    #[test]
    fn generic_high_depth_and_reverse_chroma_formats_share_the_color_classifier() {
        let spec = ColorSpecification::Defined(ColorSpec::bt709(
            ColorRange::Limited,
            ChromaLocation2d::CENTER,
        ));
        assert_eq!(
            classify_pixel_format(&PixelFormat::p010(spec)).unwrap(),
            color(ColorFormatClass::YuvSemiplanar {
                subsampling: ChromaSubsampling::Cs420,
                bits: 10,
                storage_bits: 16,
                chroma_order: ChromaOrder::CbCr,
            })
        );
        assert_eq!(
            classify_pixel_format(&PixelFormat::nv21(spec)).unwrap(),
            color(ColorFormatClass::YuvSemiplanar {
                subsampling: ChromaSubsampling::Cs420,
                bits: 8,
                storage_bits: 8,
                chroma_order: ChromaOrder::CrCb,
            })
        );
        assert!(matches!(
            classify_pixel_format(&PixelFormat::i444(12, 16, spec).unwrap()),
            Ok(PixelFormatClass::Color(ColorFormatClass::YuvPlanar {
                bits: 12,
                storage_bits: 16,
                ..
            }))
        ));
    }

    #[test]
    fn numeric_formats_never_gain_implicit_color_semantics() {
        let class = classify_pixel_format(&Vpi::U8.pixel_format()).unwrap();
        assert!(class.color().is_none());
        assert_eq!(class.numeric().unwrap().components, 1);

        let f64 = classify_pixel_format(&Vpi::F64.pixel_format())
            .unwrap()
            .numeric()
            .unwrap();
        assert_eq!(f64.wgsl, WgslNumericCapability::UnavailableFloat64);
        assert!(!f64.wgsl.supports_arithmetic());
    }
}
