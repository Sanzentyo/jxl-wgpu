use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingFieldKind, PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::prefix::{LZ77_SYMBOLS, RAW_SYMBOLS};
use crate::{EncodeError, UnsupportedFeature};

/// JPEG XL's default Modular pass-group edge length.
pub const LOSSLESS_MODULAR_GROUP_DIMENSION: u32 = 256;
pub(super) const LOSSLESS_MODULAR_LF_GROUP_DIMENSION: u32 = LOSSLESS_MODULAR_GROUP_DIMENSION * 8;
pub(super) const SHADER: &str = include_str!("../lossless_modular.wgsl");
pub(super) const MAX_DISPATCHES_PER_ARTIFACT_BINDING: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) row_stride: u32,
    pub(super) byte_offset: u32,
    pub(super) output_word_offset: u32,
    pub(super) channel: u32,
    pub(super) channels: u32,
    pub(super) bytes_per_sample: u32,
    pub(super) sample_mask: u32,
    // An explicit 256-byte array stride keeps every batch boundary valid for the portable
    // storage-buffer offset alignment without hidden Rust padding.
    pub(super) _padding: [u32; 55],
}

/// Fixed storage-buffer header written by `lossless_modular.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularArtifactHeader {
    pub(super) event_count: u32,
    pub(super) raw_counts: [u32; RAW_SYMBOLS],
    pub(super) lz77_counts: [u32; LZ77_SYMBOLS],
}

/// Fixed storage-buffer event written after [`ModularArtifactHeader`].
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularEvent {
    pub(super) kind: u32,
    pub(super) token: u32,
    pub(super) extra_bit_count: u32,
    pub(super) extra_bits: u32,
}

pub(super) const OUTPUT_HEADER_WORDS: usize = std::mem::size_of::<ModularArtifactHeader>() / 4;
pub(super) const EVENT_WORDS: usize = std::mem::size_of::<ModularEvent>() / 4;

const _: () = {
    assert!(std::mem::size_of::<ModularParams>() == 256);
    assert!(std::mem::align_of::<ModularParams>() == 4);
    assert!(std::mem::size_of::<ModularArtifactHeader>() == 53 * 4);
    assert!(std::mem::align_of::<ModularArtifactHeader>() == 4);
    assert!(std::mem::size_of::<ModularEvent>() == 16);
    assert!(std::mem::align_of::<ModularEvent>() == 4);
};

/// Standard lossless Modular input profile selected from a pitch-linear source descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessModularFormat {
    Gray,
    Rgb,
    Rgba,
}

/// Selects where a multi-group lossless Modular frame stores its MA tree and entropy tables.
///
/// Both modes keep residual generation on the GPU and only change deterministic bitstream
/// assembly. A single-group frame has no separate pass-group section and therefore uses its
/// DC-global tree in either mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LosslessModularTreeMode {
    /// Every pass group refers to the one DC-global MA configuration.
    #[default]
    SharedGlobal,
    /// Every pass group carries a complete local MA configuration.
    LocalPerGroup,
}

impl LosslessModularFormat {
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba)
    }

    /// Constructs the canonical pitch-linear source format for an unsigned integer depth.
    ///
    /// Depths `1..=8` use one native-endian `u8` word per component. Depths `9..=16` use one
    /// native-endian `u16` word per component. Sub-byte and sub-16-bit values occupy the low bits;
    /// the high padding bits are outside the valid sample and are ignored by the encoder.
    pub fn pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !(1..=16).contains(&bits_per_sample) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular integer depth must be in 1..=16",
            ));
        }
        let storage_bits = if bits_per_sample <= 8 { 8 } else { 16 };
        let (model, color_spec, swizzle, channels): (_, _, _, &[Channel]) = match self {
            Self::Gray => (
                ColorModel::NonColor,
                ColorSpecification::Undefined,
                Swizzle::X000,
                &[Channel::X],
            ),
            Self::Rgb => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZ1,
                &[Channel::X, Channel::Y, Channel::Z],
            ),
            Self::Rgba => (
                ColorModel::Rgb,
                ColorSpecification::Default,
                Swizzle::XYZW,
                &[Channel::X, Channel::Y, Channel::Z, Channel::W],
            ),
        };
        let words = channels
            .iter()
            .copied()
            .map(|channel| {
                let mut fields = Vec::with_capacity(2);
                if bits_per_sample < storage_bits {
                    fields.push(PackingField::padding(storage_bits - bits_per_sample));
                }
                fields.push(PackingField::channel(channel, bits_per_sample));
                PackingWord { fields }
            })
            .collect();
        Ok(PixelFormat {
            model,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind: SampleKind::Unsigned,
            byte_order: ByteOrder::Native,
            swizzle,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words,
            }],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LosslessModularSourceSpec {
    pub(super) format: LosslessModularFormat,
    pub(super) bits_per_sample: u8,
    pub(super) bytes_per_sample: u8,
}

pub(super) fn lossless_modular_source_spec(
    format: &PixelFormat,
) -> Result<LosslessModularSourceSpec, EncodeError> {
    if format.sample_kind != SampleKind::Unsigned
        || format.byte_order != ByteOrder::Native
        || format.chroma_subsampling != ChromaSubsampling::None
        || format.planes.len() != 1
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let logical_format = match (format.model, format.swizzle, format.color_spec) {
        (ColorModel::NonColor, Swizzle::X000, ColorSpecification::Undefined) => {
            LosslessModularFormat::Gray
        }
        (
            ColorModel::Rgb,
            Swizzle::XYZ1,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => LosslessModularFormat::Rgb,
        (
            ColorModel::Rgb,
            Swizzle::XYZW,
            ColorSpecification::Default | ColorSpecification::Undefined,
        ) => LosslessModularFormat::Rgba,
        _ => return Err(UnsupportedFeature::InputFormat.into()),
    };
    let plane = &format.planes[0];
    if plane.sampling != PlaneSampling::FULL
        || plane.pixels_per_element != 1
        || plane.words.len() != logical_format.channel_count() as usize
    {
        return Err(UnsupportedFeature::InputFormat.into());
    }
    let expected_channels = [Channel::X, Channel::Y, Channel::Z, Channel::W];
    let mut bits_per_sample = None;
    let mut storage_bits = None;
    for (word, expected_channel) in plane
        .words
        .iter()
        .zip(&expected_channels[..plane.words.len()])
    {
        let (padding, channel_bits, channel) = match word.fields.as_slice() {
            [field] => match field.kind {
                PackingFieldKind::Channel(channel) => (0, field.bits, channel),
                PackingFieldKind::Padding => return Err(UnsupportedFeature::InputFormat.into()),
            },
            [padding, sample] => match (padding.kind, sample.kind) {
                (PackingFieldKind::Padding, PackingFieldKind::Channel(channel)) => {
                    (padding.bits, sample.bits, channel)
                }
                _ => return Err(UnsupportedFeature::InputFormat.into()),
            },
            _ => return Err(UnsupportedFeature::InputFormat.into()),
        };
        let word_bits = padding
            .checked_add(channel_bits)
            .ok_or(UnsupportedFeature::InputFormat)?;
        let expected_storage_bits = if channel_bits <= 8 { 8 } else { 16 };
        if channel != *expected_channel
            || !(1..=16).contains(&channel_bits)
            || word_bits != expected_storage_bits
            || bits_per_sample.is_some_and(|bits| bits != channel_bits)
            || storage_bits.is_some_and(|bits| bits != word_bits)
        {
            return Err(UnsupportedFeature::InputFormat.into());
        }
        bits_per_sample = Some(channel_bits);
        storage_bits = Some(word_bits);
    }
    let bits_per_sample = bits_per_sample.ok_or(UnsupportedFeature::InputFormat)?;
    let storage_bits = storage_bits.ok_or(UnsupportedFeature::InputFormat)?;
    Ok(LosslessModularSourceSpec {
        format: logical_format,
        bits_per_sample,
        bytes_per_sample: storage_bits / 8,
    })
}
