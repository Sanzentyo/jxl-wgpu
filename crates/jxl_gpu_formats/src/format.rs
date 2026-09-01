use thiserror::Error;

/// Interpretation of the canonical X/Y/Z/W channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorModel {
    NonColor,
    Ycbcr,
    Rgb,
    Raw(RawPattern),
    Xyz,
}

/// RAW mosaics predefined by NVIDIA VPI 4.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RawPattern {
    Rggb,
    Bggr,
    Grbg,
    Gbrg,
    Rccb,
    Bccr,
    Crbc,
    Cbrc,
    Rccc,
    Crcc,
    Ccrc,
    Cccr,
    Cccc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Sensor,
    Bt601,
    Bt709,
    Bt2020,
    DisplayP3,
    Undefined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YcbcrEncoding {
    Undefined,
    Bt601,
    Bt709,
    Bt2020,
    Bt2020ConstantLuminance,
    Smpte240M,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    Undefined,
    Linear,
    Srgb,
    Sycc,
    Pq,
    Hlg,
    Bt709,
    Bt2020,
    Smpte240M,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorRange {
    Full,
    Limited,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaLocation {
    Even,
    Center,
    Odd,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChromaLocation2d {
    pub horizontal: ChromaLocation,
    pub vertical: ChromaLocation,
}

impl ChromaLocation2d {
    pub const BOTH: Self = Self {
        horizontal: ChromaLocation::Both,
        vertical: ChromaLocation::Both,
    };
    pub const EVEN: Self = Self {
        horizontal: ChromaLocation::Even,
        vertical: ChromaLocation::Even,
    };
    pub const CENTER: Self = Self {
        horizontal: ChromaLocation::Center,
        vertical: ChromaLocation::Center,
    };
}

/// Fully specified color interpretation. The color model and chroma sampling
/// are properties of [`PixelFormat`], not this value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorSpec {
    pub space: ColorSpace,
    pub encoding: YcbcrEncoding,
    pub transfer: TransferFunction,
    pub range: ColorRange,
    pub chroma_location: ChromaLocation2d,
}

impl ColorSpec {
    #[must_use]
    pub const fn bt601(range: ColorRange, chroma_location: ChromaLocation2d) -> Self {
        // VPI's named BT.601 specifications use BT.709 primaries with a
        // BT.601 YCbCr encoding.
        Self {
            space: ColorSpace::Bt709,
            encoding: YcbcrEncoding::Bt601,
            transfer: TransferFunction::Bt709,
            range,
            chroma_location,
        }
    }

    #[must_use]
    pub const fn bt709(range: ColorRange, chroma_location: ChromaLocation2d) -> Self {
        Self {
            space: ColorSpace::Bt709,
            encoding: YcbcrEncoding::Bt709,
            transfer: TransferFunction::Bt709,
            range,
            chroma_location,
        }
    }

    #[must_use]
    pub const fn bt2020_ncl(range: ColorRange, chroma_location: ChromaLocation2d) -> Self {
        Self {
            space: ColorSpace::Bt2020,
            encoding: YcbcrEncoding::Bt2020,
            transfer: TransferFunction::Bt2020,
            range,
            chroma_location,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpecification {
    /// An external API may infer its default interpretation.
    Default,
    /// Color interpretation is absent or irrelevant.
    Undefined,
    Defined(ColorSpec),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaSubsampling {
    None,
    Cs444,
    Cs422,
    Cs422R,
    Cs411,
    Cs411R,
    Cs420,
}

impl ChromaSubsampling {
    #[must_use]
    pub const fn chroma_divisors(self) -> Option<(u8, u8)> {
        match self {
            Self::None => None,
            Self::Cs444 => Some((1, 1)),
            Self::Cs422 => Some((2, 1)),
            Self::Cs422R => Some((1, 2)),
            Self::Cs411 => Some((4, 1)),
            Self::Cs411R => Some((1, 4)),
            Self::Cs420 => Some((2, 2)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleKind {
    Unsigned,
    Signed,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    Native,
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    X,
    Y,
    Z,
    W,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleComponent {
    Zero,
    X,
    Y,
    Z,
    W,
    One,
}

/// Selects stored X/Y/Z/W values for the canonical output components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle(pub [SwizzleComponent; 4]);

impl Swizzle {
    pub const X000: Self = Self([
        SwizzleComponent::X,
        SwizzleComponent::Zero,
        SwizzleComponent::Zero,
        SwizzleComponent::Zero,
    ]);
    pub const XY00: Self = Self([
        SwizzleComponent::X,
        SwizzleComponent::Y,
        SwizzleComponent::Zero,
        SwizzleComponent::Zero,
    ]);
    pub const XYZ0: Self = Self([
        SwizzleComponent::X,
        SwizzleComponent::Y,
        SwizzleComponent::Z,
        SwizzleComponent::Zero,
    ]);
    pub const XYZ1: Self = Self([
        SwizzleComponent::X,
        SwizzleComponent::Y,
        SwizzleComponent::Z,
        SwizzleComponent::One,
    ]);
    pub const XYZW: Self = Self([
        SwizzleComponent::X,
        SwizzleComponent::Y,
        SwizzleComponent::Z,
        SwizzleComponent::W,
    ]);
    pub const ZYX1: Self = Self([
        SwizzleComponent::Z,
        SwizzleComponent::Y,
        SwizzleComponent::X,
        SwizzleComponent::One,
    ]);
    pub const ZYXW: Self = Self([
        SwizzleComponent::Z,
        SwizzleComponent::Y,
        SwizzleComponent::X,
        SwizzleComponent::W,
    ]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackingFieldKind {
    Channel(Channel),
    Padding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackingField {
    pub kind: PackingFieldKind,
    pub bits: u8,
}

impl PackingField {
    #[must_use]
    pub const fn channel(channel: Channel, bits: u8) -> Self {
        Self {
            kind: PackingFieldKind::Channel(channel),
            bits,
        }
    }

    #[must_use]
    pub const fn padding(bits: u8) -> Self {
        Self {
            kind: PackingFieldKind::Padding,
            bits,
        }
    }
}

/// One independently endian-addressed word. Fields are ordered from most to
/// least significant bit. This makes both MSB-aligned P010 and LSB-aligned
/// custom formats representable without a storage modifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PackingWord {
    pub fields: Vec<PackingField>,
}

impl PackingWord {
    #[must_use]
    pub fn channel(channel: Channel, bits: u8) -> Self {
        Self {
            fields: vec![PackingField::channel(channel, bits)],
        }
    }

    #[must_use]
    pub fn msb_aligned(channel: Channel, bits: u8, storage_bits: u8) -> Option<Self> {
        (bits != 0 && bits <= storage_bits).then(|| {
            let mut fields = vec![PackingField::channel(channel, bits)];
            if bits < storage_bits {
                fields.push(PackingField::padding(storage_bits - bits));
            }
            Self { fields }
        })
    }

    #[must_use]
    pub fn bits(&self) -> u32 {
        self.fields.iter().map(|field| u32::from(field.bits)).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneSampling {
    pub horizontal_divisor: u8,
    pub vertical_divisor: u8,
}

impl PlaneSampling {
    pub const FULL: Self = Self {
        horizontal_divisor: 1,
        vertical_divisor: 1,
    };

    #[must_use]
    pub const fn new(horizontal_divisor: u8, vertical_divisor: u8) -> Self {
        Self {
            horizontal_divisor,
            vertical_divisor,
        }
    }
}

/// Packing and sampling of one directly addressable pitch-linear plane.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaneFormat {
    pub sampling: PlaneSampling,
    /// Number of horizontally adjacent image samples described by one element.
    pub pixels_per_element: u8,
    /// Independently endian-addressed words comprising one element.
    pub words: Vec<PackingWord>,
}

impl PlaneFormat {
    #[must_use]
    pub fn separate_words(
        sampling: PlaneSampling,
        pixels_per_element: u8,
        channels: &[Channel],
        bits: u8,
    ) -> Self {
        Self {
            sampling,
            pixels_per_element,
            words: channels
                .iter()
                .map(|&channel| PackingWord::channel(channel, bits))
                .collect(),
        }
    }

    #[must_use]
    pub fn bits_per_element(&self) -> u64 {
        self.words.iter().map(|word| u64::from(word.bits())).sum()
    }
}

/// A storage-complete logical pixel format independent of concrete offsets and
/// row pitches.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PixelFormat {
    pub model: ColorModel,
    pub color_spec: ColorSpecification,
    pub chroma_subsampling: ChromaSubsampling,
    pub sample_kind: SampleKind,
    pub byte_order: ByteOrder,
    pub swizzle: Swizzle,
    pub planes: Vec<PlaneFormat>,
}

impl PixelFormat {
    pub const MAX_PLANES: usize = 6;

    pub fn new(
        model: ColorModel,
        color_spec: ColorSpecification,
        chroma_subsampling: ChromaSubsampling,
        sample_kind: SampleKind,
        byte_order: ByteOrder,
        swizzle: Swizzle,
        planes: Vec<PlaneFormat>,
    ) -> Result<Self, PixelFormatError> {
        let format = Self {
            model,
            color_spec,
            chroma_subsampling,
            sample_kind,
            byte_order,
            swizzle,
            planes,
        };
        format.validate()?;
        Ok(format)
    }

    pub fn validate(&self) -> Result<(), PixelFormatError> {
        if self.planes.is_empty() || self.planes.len() > Self::MAX_PLANES {
            return Err(PixelFormatError::PlaneCount(self.planes.len()));
        }
        for (plane_index, plane) in self.planes.iter().enumerate() {
            if plane.sampling.horizontal_divisor == 0 || plane.sampling.vertical_divisor == 0 {
                return Err(PixelFormatError::ZeroSamplingDivisor { plane: plane_index });
            }
            if plane.pixels_per_element == 0 {
                return Err(PixelFormatError::ZeroPixelsPerElement { plane: plane_index });
            }
            if plane.words.is_empty() {
                return Err(PixelFormatError::EmptyPacking { plane: plane_index });
            }
            for (word_index, word) in plane.words.iter().enumerate() {
                if word.fields.is_empty() || word.fields.iter().any(|field| field.bits == 0) {
                    return Err(PixelFormatError::EmptyPackingWord {
                        plane: plane_index,
                        word: word_index,
                    });
                }
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn non_color(sample_kind: SampleKind, bits: u8, channels: &[Channel]) -> Self {
        let swizzle = match channels.len() {
            1 => Swizzle::X000,
            2 => Swizzle::XY00,
            3 => Swizzle::XYZ0,
            _ => Swizzle::XYZW,
        };
        Self {
            model: ColorModel::NonColor,
            color_spec: ColorSpecification::Undefined,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind,
            byte_order: ByteOrder::Native,
            swizzle,
            planes: vec![PlaneFormat::separate_words(
                PlaneSampling::FULL,
                1,
                channels,
                bits,
            )],
        }
    }

    #[must_use]
    pub fn luma(bits: u8, color_spec: ColorSpecification) -> Self {
        Self {
            model: ColorModel::Ycbcr,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle: Swizzle::X000,
            planes: vec![PlaneFormat::separate_words(
                PlaneSampling::FULL,
                1,
                &[Channel::X],
                bits,
            )],
        }
    }

    /// Constructs planar Y/Cb/Cr with samples MSB-aligned in their storage words.
    pub fn yuv_planar(
        subsampling: ChromaSubsampling,
        bits: u8,
        storage_bits: u8,
        color_spec: ColorSpecification,
    ) -> Result<Self, PixelFormatError> {
        validate_sample_storage(bits, storage_bits)?;
        let (horizontal, vertical) = require_chroma_sampling(subsampling)?;
        let word = |channel| {
            PackingWord::msb_aligned(channel, bits, storage_bits)
                .expect("validated sample and storage widths")
        };
        let chroma_sampling = PlaneSampling::new(horizontal, vertical);
        Self::new(
            ColorModel::Ycbcr,
            color_spec,
            subsampling,
            SampleKind::Unsigned,
            ByteOrder::Native,
            Swizzle::XYZ0,
            vec![
                PlaneFormat {
                    sampling: PlaneSampling::FULL,
                    pixels_per_element: 1,
                    words: vec![word(Channel::X)],
                },
                PlaneFormat {
                    sampling: chroma_sampling,
                    pixels_per_element: 1,
                    words: vec![word(Channel::Y)],
                },
                PlaneFormat {
                    sampling: chroma_sampling,
                    pixels_per_element: 1,
                    words: vec![word(Channel::Z)],
                },
            ],
        )
    }

    /// Constructs semi-planar Y plus interleaved chroma with samples
    /// MSB-aligned in their storage words. This represents NV12/NV21,
    /// NV24/NV42, and P010/P012/P016-style storage.
    pub fn yuv_semiplanar(
        subsampling: ChromaSubsampling,
        bits: u8,
        storage_bits: u8,
        chroma_order: ChromaOrder,
        color_spec: ColorSpecification,
    ) -> Result<Self, PixelFormatError> {
        validate_sample_storage(bits, storage_bits)?;
        let (horizontal, vertical) = require_chroma_sampling(subsampling)?;
        let channels = match chroma_order {
            ChromaOrder::CbCr => [Channel::Y, Channel::Z],
            ChromaOrder::CrCb => [Channel::Z, Channel::Y],
        };
        let word = |channel| {
            PackingWord::msb_aligned(channel, bits, storage_bits)
                .expect("validated sample and storage widths")
        };
        Self::new(
            ColorModel::Ycbcr,
            color_spec,
            subsampling,
            SampleKind::Unsigned,
            ByteOrder::Native,
            Swizzle::XYZ0,
            vec![
                PlaneFormat {
                    sampling: PlaneSampling::FULL,
                    pixels_per_element: 1,
                    words: vec![word(Channel::X)],
                },
                PlaneFormat {
                    sampling: PlaneSampling::new(horizontal, vertical),
                    pixels_per_element: 1,
                    words: channels.into_iter().map(word).collect(),
                },
            ],
        )
    }

    #[must_use]
    pub fn packed_yuv4228(order: Packed422Order, color_spec: ColorSpecification) -> Self {
        let channels = match order {
            Packed422Order::Yuyv => [Channel::X, Channel::Y, Channel::X, Channel::Z],
            Packed422Order::Uyvy => [Channel::Y, Channel::X, Channel::Z, Channel::X],
        };
        Self {
            model: ColorModel::Ycbcr,
            color_spec,
            chroma_subsampling: ChromaSubsampling::Cs422,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle: Swizzle::XYZ1,
            planes: vec![PlaneFormat::separate_words(
                PlaneSampling::FULL,
                2,
                &channels,
                8,
            )],
        }
    }

    #[must_use]
    pub fn rgb8(order: RgbChannelOrder, planar: bool, color_spec: ColorSpecification) -> Self {
        let (channel_count, swizzle) = match order {
            RgbChannelOrder::Rgb => (3, Swizzle::XYZ1),
            RgbChannelOrder::Bgr => (3, Swizzle::ZYX1),
            RgbChannelOrder::Rgba => (4, Swizzle::XYZW),
            RgbChannelOrder::Bgra => (4, Swizzle::ZYXW),
        };
        let channels = [Channel::X, Channel::Y, Channel::Z, Channel::W];
        let channels = &channels[..channel_count];
        let planes = if planar {
            channels
                .iter()
                .map(|&channel| PlaneFormat::separate_words(PlaneSampling::FULL, 1, &[channel], 8))
                .collect()
        } else {
            vec![PlaneFormat::separate_words(
                PlaneSampling::FULL,
                1,
                channels,
                8,
            )]
        };
        Self {
            model: ColorModel::Rgb,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle,
            planes,
        }
    }

    pub fn i444(
        bits: u8,
        storage_bits: u8,
        color_spec: ColorSpecification,
    ) -> Result<Self, PixelFormatError> {
        Self::yuv_planar(ChromaSubsampling::Cs444, bits, storage_bits, color_spec)
    }

    pub fn i422(
        bits: u8,
        storage_bits: u8,
        color_spec: ColorSpecification,
    ) -> Result<Self, PixelFormatError> {
        Self::yuv_planar(ChromaSubsampling::Cs422, bits, storage_bits, color_spec)
    }

    pub fn i420(
        bits: u8,
        storage_bits: u8,
        color_spec: ColorSpecification,
    ) -> Result<Self, PixelFormatError> {
        Self::yuv_planar(ChromaSubsampling::Cs420, bits, storage_bits, color_spec)
    }

    #[must_use]
    pub fn nv12(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs420,
            8,
            8,
            ChromaOrder::CbCr,
            color_spec,
        )
        .expect("NV12 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn nv21(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs420,
            8,
            8,
            ChromaOrder::CrCb,
            color_spec,
        )
        .expect("NV21 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn nv24(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs444,
            8,
            8,
            ChromaOrder::CbCr,
            color_spec,
        )
        .expect("NV24 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn nv42(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs444,
            8,
            8,
            ChromaOrder::CrCb,
            color_spec,
        )
        .expect("NV42 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn nv16(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs422,
            8,
            8,
            ChromaOrder::CbCr,
            color_spec,
        )
        .expect("NV16 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn nv61(color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(
            ChromaSubsampling::Cs422,
            8,
            8,
            ChromaOrder::CrCb,
            color_spec,
        )
        .expect("NV61 has a valid fixed descriptor")
    }

    #[must_use]
    pub fn p010(color_spec: ColorSpecification) -> Self {
        Self::p0xx(10, color_spec)
    }

    #[must_use]
    pub fn p012(color_spec: ColorSpecification) -> Self {
        Self::p0xx(12, color_spec)
    }

    #[must_use]
    pub fn p016(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs420, 16, color_spec)
    }

    #[must_use]
    pub fn p210(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs422, 10, color_spec)
    }

    #[must_use]
    pub fn p212(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs422, 12, color_spec)
    }

    #[must_use]
    pub fn p216(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs422, 16, color_spec)
    }

    #[must_use]
    pub fn p410(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs444, 10, color_spec)
    }

    #[must_use]
    pub fn p412(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs444, 12, color_spec)
    }

    #[must_use]
    pub fn p416(color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs444, 16, color_spec)
    }

    fn p0xx(bits: u8, color_spec: ColorSpecification) -> Self {
        Self::pxxx(ChromaSubsampling::Cs420, bits, color_spec)
    }

    fn pxxx(subsampling: ChromaSubsampling, bits: u8, color_spec: ColorSpecification) -> Self {
        Self::yuv_semiplanar(subsampling, bits, 16, ChromaOrder::CbCr, color_spec)
            .expect("P0xx constructors use valid fixed widths")
    }
}

fn validate_sample_storage(bits: u8, storage_bits: u8) -> Result<(), PixelFormatError> {
    if bits == 0 || storage_bits == 0 || bits > storage_bits {
        return Err(PixelFormatError::InvalidSampleStorage { bits, storage_bits });
    }
    Ok(())
}

fn require_chroma_sampling(subsampling: ChromaSubsampling) -> Result<(u8, u8), PixelFormatError> {
    subsampling
        .chroma_divisors()
        .ok_or(PixelFormatError::MissingChromaSubsampling)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaOrder {
    CbCr,
    CrCb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Packed422Order {
    Yuyv,
    Uyvy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbChannelOrder {
    Rgb,
    Bgr,
    Rgba,
    Bgra,
}

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum PixelFormatError {
    #[error("pixel format has {0} planes; expected 1..=6")]
    PlaneCount(usize),
    #[error("plane {plane} has a zero sampling divisor")]
    ZeroSamplingDivisor { plane: usize },
    #[error("plane {plane} has zero pixels per element")]
    ZeroPixelsPerElement { plane: usize },
    #[error("plane {plane} has no packing words")]
    EmptyPacking { plane: usize },
    #[error("plane {plane} packing word {word} has no non-zero fields")]
    EmptyPackingWord { plane: usize, word: usize },
    #[error("sample width {bits} is incompatible with {storage_bits}-bit storage")]
    InvalidSampleStorage { bits: u8, storage_bits: u8 },
    #[error("a YCbCr constructor requires an explicit chroma subsampling mode")]
    MissingChromaSubsampling,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ColorSpecification {
        ColorSpecification::Defined(ColorSpec::bt709(
            ColorRange::Limited,
            ChromaLocation2d::CENTER,
        ))
    }

    #[test]
    fn p010_is_msb_aligned_420_semiplanar() {
        let format = PixelFormat::p010(spec());
        assert_eq!(format.chroma_subsampling, ChromaSubsampling::Cs420);
        assert_eq!(format.planes.len(), 2);
        assert_eq!(format.planes[0].words[0].bits(), 16);
        assert_eq!(
            format.planes[0].words[0].fields,
            vec![
                PackingField::channel(Channel::X, 10),
                PackingField::padding(6),
            ]
        );
        assert_eq!(format.planes[1].words.len(), 2);
    }

    #[test]
    fn common_planar_depths_are_representable() {
        for bits in [8, 10, 12, 16] {
            let storage_bits = if bits == 8 { 8 } else { 16 };
            for format in [
                PixelFormat::i444(bits, storage_bits, spec()).unwrap(),
                PixelFormat::i422(bits, storage_bits, spec()).unwrap(),
                PixelFormat::i420(bits, storage_bits, spec()).unwrap(),
            ] {
                assert_eq!(format.planes.len(), 3);
                format.validate().unwrap();
            }
        }
    }

    #[test]
    fn invalid_storage_width_is_typed() {
        assert_eq!(
            PixelFormat::i420(12, 8, spec()),
            Err(PixelFormatError::InvalidSampleStorage {
                bits: 12,
                storage_bits: 8,
            })
        );
    }
}
