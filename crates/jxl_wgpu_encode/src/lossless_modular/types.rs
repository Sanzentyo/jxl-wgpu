use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

use crate::EncodeError;
use crate::prefix::{LZ77_SYMBOLS, RAW_SYMBOLS};

/// JPEG XL's default Modular pass-group edge length.
pub const LOSSLESS_MODULAR_GROUP_DIMENSION: u32 = LosslessModularGroupSize::Pixels256.dimension();
pub(super) const SHADER: &str = include_str!("../lossless_modular.wgsl");
pub(super) const MAX_DISPATCHES_PER_ARTIFACT_BINDING: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularSourceParams {
    pub(super) row_stride: u32,
    pub(super) byte_offset: u32,
    pub(super) pixel_stride: u32,
    pub(super) word_bytes: u32,
    pub(super) bit_shift: u32,
    pub(super) plane: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) output_word_offset: u32,
    pub(super) channel: u32,
    pub(super) channels: u32,
    pub(super) sample_mask: u32,
    pub(super) rct_type: u32,
    pub(super) big_endian: u32,
    pub(super) sources: [ModularSourceParams; 4],
    pub(super) predictor: u32,
    pub(super) wp_scratch_word_offset: u32,
    pub(super) wp_coefficients: [u32; 7],
    pub(super) wp_max_weights: [u32; 4],
    pub(super) lz77_mode: u32,
    pub(super) lz77_scratch_word_offset: u32,
    pub(super) lz77_hash_mask: u32,
    pub(super) squeeze: u32,
    pub(super) source_width: u32,
    pub(super) source_height: u32,
    // An explicit 256-byte array stride keeps every batch boundary valid for the portable
    // storage-buffer offset alignment without hidden Rust padding.
    pub(super) _padding: [u32; 13],
}

/// Fixed storage-buffer header written by `lossless_modular.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ModularArtifactHeader {
    pub(super) event_count: u32,
    pub(super) raw_counts: [u32; RAW_SYMBOLS],
    pub(super) lz77_counts: [u32; LZ77_SYMBOLS],
    pub(super) distance_counts: [u32; RAW_SYMBOLS],
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
    assert!(std::mem::size_of::<ModularSourceParams>() == 24);
    assert!(std::mem::align_of::<ModularSourceParams>() == 4);
    assert!(std::mem::size_of::<ModularParams>() == 256);
    assert!(std::mem::align_of::<ModularParams>() == 4);
    assert!(std::mem::size_of::<ModularArtifactHeader>() == 100 * 4);
    assert!(std::mem::align_of::<ModularArtifactHeader>() == 4);
    assert!(std::mem::size_of::<ModularEvent>() == 16);
    assert!(std::mem::align_of::<ModularEvent>() == 4);
};

/// Standard lossless Modular input profile selected from a pitch-linear source descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LosslessModularFormat {
    Gray,
    GrayAlpha,
    Rgb,
    Rgba,
}

/// Interpretation of caller-supplied color samples relative to the alpha plane.
/// Encoding preserves the source words, including color at zero alpha; it never multiplies,
/// divides or discards them. The same declaration applies to every frame in an animation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AlphaAssociation {
    /// Color samples are independent of alpha (the default).
    #[default]
    Unassociated,
    /// Color samples are already multiplied by alpha.
    Associated,
}

impl AlphaAssociation {
    pub(super) fn validate(self, format: LosslessModularFormat) -> Result<(), EncodeError> {
        if self == Self::Associated && !format.has_alpha() {
            return Err(EncodeError::InvalidConfiguration(
                "associated source color requires an alpha channel",
            ));
        }
        Ok(())
    }
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

/// JPEG XL's four legal Modular pass-group edge lengths.
/// A group's LF region is eight times this length on each axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum LosslessModularGroupSize {
    Pixels128 = 0,
    #[default]
    Pixels256 = 1,
    Pixels512 = 2,
    Pixels1024 = 3,
}

impl LosslessModularGroupSize {
    pub const ALL: [Self; 4] = [
        Self::Pixels128,
        Self::Pixels256,
        Self::Pixels512,
        Self::Pixels1024,
    ];

    #[must_use]
    pub const fn dimension(self) -> u32 {
        128 << (self as u32)
    }

    #[must_use]
    pub const fn size_shift(self) -> u8 {
        self as u8
    }
}

/// Stream encoding policy shared by stills and frames in an animation session.
/// Defaults retain 256-pixel groups, the shared global MA tree, Gradient prediction,
/// default Weighted coefficients, zero-run coding and automatic color transforms.
///
/// ```no_run
/// # use jxl_wgpu_encode::{LosslessModularConfig, LosslessModularEncoder,
/// #     LosslessModularColorTransform, LosslessModularGroupSize, LosslessModularRctType,
/// #     LosslessModularTreeMode, WgpuContext};
/// # fn configure(context: WgpuContext) {
/// let encoder = LosslessModularEncoder::with_config(context, LosslessModularConfig {
///     group_size: LosslessModularGroupSize::Pixels512,
///     tree_mode: LosslessModularTreeMode::LocalPerGroup,
///     color_transform: LosslessModularColorTransform::LocalRct(LosslessModularRctType::YCOCG),
///     ..Default::default()
/// });
/// assert_eq!(encoder.config().group_size.dimension(), 512);
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LosslessModularConfig {
    pub group_size: LosslessModularGroupSize,
    pub tree_mode: LosslessModularTreeMode,
    /// GPU source-word transform and its wire placement.
    pub color_transform: super::rct::LosslessModularColorTransform,
    /// One of all fourteen standard predictors, shared by every group and component.
    pub predictor: super::predictor::LosslessModularPredictor,
    /// Serialized in every Modular header; used when `predictor` is `Weighted`.
    pub weighted_predictor: super::predictor::LosslessModularWeightedPredictor,
    /// Zero-run coding or bounded GPU search for arbitrary residual matches.
    pub lz77: super::lz77::LosslessModularLz77,
    /// Group-local separable Squeeze after color transformation. GPU validation rejects an
    /// unrepresentable signed residual instead of emitting a lossy result.
    pub squeeze: super::squeeze::LosslessModularSqueeze,
}

impl LosslessModularFormat {
    #[must_use]
    pub const fn channel_count(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }

    #[must_use]
    pub const fn color_channel_count(self) -> u32 {
        match self {
            Self::Gray | Self::GrayAlpha => 1,
            Self::Rgb | Self::Rgba => 3,
        }
    }

    /// Constructs the canonical pitch-linear source format for an unsigned integer depth.
    ///
    /// Depths `1..=8` use one native-endian `u8` word per component. Depths `9..=16` use one
    /// native-endian `u16` word per component, and `17..=31` use `u32`. Samples occupy the low bits;
    /// the high padding bits are outside the valid sample and are ignored by the encoder.
    pub fn pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !(1..=31).contains(&bits_per_sample) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular integer depth must be in 1..=31",
            ));
        }
        Ok(self.packed_pixel_format(bits_per_sample, SampleKind::Unsigned))
    }

    /// Constructs native IEEE binary16 or binary32 storage, preserving every source bit.
    ///
    /// Components remain in the declared sRGB/gray domain. No floating-point arithmetic,
    /// normalization or alpha association is performed by the lossless encoder.
    pub fn float_pixel_format(self, bits_per_sample: u8) -> Result<PixelFormat, EncodeError> {
        if !matches!(bits_per_sample, 16 | 32) {
            return Err(EncodeError::InvalidConfiguration(
                "lossless Modular floating storage must be binary16 or binary32",
            ));
        }
        Ok(self.packed_pixel_format(bits_per_sample, SampleKind::Float))
    }

    fn packed_pixel_format(self, bits_per_sample: u8, sample_kind: SampleKind) -> PixelFormat {
        let storage_bits = bits_per_sample.next_power_of_two().max(8);
        let (model, color_spec, swizzle, channels): (_, _, _, &[Channel]) = match self {
            Self::Gray => (
                ColorModel::NonColor,
                ColorSpecification::Undefined,
                Swizzle::X000,
                &[Channel::X],
            ),
            Self::GrayAlpha => (
                ColorModel::Gray,
                ColorSpecification::Default,
                Swizzle::X00W,
                &[Channel::X, Channel::W],
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
        PixelFormat {
            model,
            color_spec,
            chroma_subsampling: ChromaSubsampling::None,
            sample_kind,
            byte_order: ByteOrder::Native,
            swizzle,
            planes: vec![PlaneFormat {
                sampling: PlaneSampling::FULL,
                pixels_per_element: 1,
                words,
            }],
        }
    }
}

pub(super) const fn modular_sample_depth(
    bits: u8,
    exponent: u8,
) -> jxl_gpu_bitstream::SampleBitDepth {
    if exponent == 0 {
        jxl_gpu_bitstream::SampleBitDepth::Integer {
            bits_per_sample: bits as u32,
        }
    } else {
        jxl_gpu_bitstream::SampleBitDepth::Float {
            bits_per_sample: bits as u32,
            exponent_bits_per_sample: exponent as u32,
        }
    }
}
