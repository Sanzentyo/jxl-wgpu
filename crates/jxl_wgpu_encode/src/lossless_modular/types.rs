use crate::prefix::{LZ77_SYMBOLS, RAW_SYMBOLS};

/// JPEG XL's default Modular pass-group edge length.
pub const LOSSLESS_MODULAR_GROUP_DIMENSION: u32 = LosslessModularGroupSize::Pixels256.dimension();
pub(super) const SHADER: &str = include_str!("../lossless_modular.wgsl");
pub(super) const MAX_DISPATCHES_PER_ARTIFACT_BINDING: usize = 64;

pub(super) use crate::source::SourceParams as ModularSourceParams;

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
    pub(super) palette_capacity: u32,
    pub(super) palette_scratch_word_offset: u32,
    pub(super) palette_hash_mask: u32,
    pub(super) group_channels: u32,
    pub(super) palette_delta_predictor: u32,
    pub(super) palette_delta_capacity: u32,
    pub(super) palette_implicit_depth: u32,
    pub(super) palette_begin: u32,
    pub(super) palette_components: u32,
    pub(super) sample_source: u32,
    pub(super) squeeze_band: u32,
    // An explicit 256-byte array stride keeps every batch boundary valid for the portable
    // storage-buffer offset alignment without hidden Rust padding.
    pub(super) transform_program_word_offset: u32,
    pub(super) transform_sample_word_offset: u32,
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

pub use crate::source::SourceChannels as LosslessModularFormat;

pub use crate::sample_format::AlphaAssociation;

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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessModularConfig {
    /// Independent scalar inputs, following any packed alpha in the image declaration.
    pub extra_channels: Vec<crate::ExtraChannel>,
    /// Maximum bounded image-wide extra-channel metadata, including packed alpha.
    pub max_extra_channel_metadata_bytes: u64,
    pub group_size: LosslessModularGroupSize,
    pub tree_mode: LosslessModularTreeMode,
    /// GPU source-word transform and its wire placement.
    pub color_transform: super::rct::LosslessModularColorTransform,
    /// One of all fourteen standard predictors, shared by every group and component.
    pub predictor: super::predictor::LosslessModularPredictor,
    /// Serialized in every Modular header; used by Weighted token or palette delta prediction.
    pub weighted_predictor: super::predictor::LosslessModularWeightedPredictor,
    /// Zero-run coding or bounded GPU search for arbitrary residual matches.
    pub lz77: super::lz77::LosslessModularLz77,
    /// Prefix (default) or GPU ANS with deterministic frame-wide histogram normalization.
    pub entropy: super::entropy::LosslessModularEntropyCoding,
    /// Ordered group-local transforms after source RCT and optional Palette.
    /// A Squeeze policy converts with `.into()`; explicit programs apply RCT/Squeeze to the
    /// current image-channel topology. GPU validation rejects unrepresentable signed residuals.
    pub local_transforms: super::local_transforms::LosslessModularLocalTransforms,
    /// Optional exact local palette built on GPU after source RCT and before local transforms.
    pub palette: Option<super::palette::LosslessModularPalette>,
}

impl Default for LosslessModularConfig {
    fn default() -> Self {
        Self {
            extra_channels: Vec::new(),
            max_extra_channel_metadata_bytes: 1 << 20,
            group_size: Default::default(),
            tree_mode: Default::default(),
            color_transform: Default::default(),
            predictor: Default::default(),
            weighted_predictor: Default::default(),
            lz77: Default::default(),
            entropy: Default::default(),
            local_transforms: Default::default(),
            palette: None,
        }
    }
}

impl LosslessModularConfig {
    pub(crate) fn samples(
        &self,
        format: LosslessModularFormat,
        bits: u8,
        exponent: u8,
    ) -> Result<crate::sample_format::ImageSamplePlan, crate::EncodeError> {
        let color = if exponent == 0 {
            crate::ColorSampleFormat::integer(format.color_channels(), bits)?
        } else {
            crate::ColorSampleFormat::float(format.color_channels(), bits, exponent)?
        };
        crate::sample_format::ImageSamplePlan::new(
            color,
            format.has_alpha().then_some(Default::default()),
        )
        .with_extra_channels(&self.extra_channels, self.max_extra_channel_metadata_bytes)
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
