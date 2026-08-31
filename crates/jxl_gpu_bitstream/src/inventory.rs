//! Bounded standard-codestream header and TOC inventory.
//!
//! Image-header grammar is delegated to the header-only `jxl-image` crate. Frame headers are
//! parsed locally, while the published `jxl-coding` metadata decoder is used only for the
//! entropy-coded TOC permutation. No image sample, Modular, or VarDCT data is decoded here.

use jxl_bitstream::Bitstream as ImageBitstream;
use jxl_image::{
    BitDepth as JxlBitDepth, ExtraChannelType as JxlExtraChannelType, ImageHeader,
    color::{
        ColourEncoding as JxlColourEncoding, ColourSpace as JxlColourSpace,
        Primaries as JxlPrimaries, RenderingIntent as JxlRenderingIntent,
        TransferFunction as JxlTransferFunction, WhitePoint as JxlWhitePoint,
    },
};
use jxl_oxide_common::Bundle;
use thiserror::Error;

use crate::{BitReader, Error as BitReaderError};

const FLAG_USE_LF_FRAME: u64 = 0x20;
const GROUP_DIM_LOG2_MINUS_ONE: u32 = 7;
const MAX_EXTRA_CHANNELS: u32 = 256;
const MAX_TOC_ENTRIES: u64 = 65_536;

/// Resource limits for standard codestream inventory construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InventoryLimits {
    /// Maximum bytes exposed to the header-only image-header parser.
    pub max_image_header_bytes: u64,
    /// Maximum entropy-decoded transformed ICC bytes retained while reconstructing a profile.
    pub max_encoded_icc_bytes: u64,
    /// Maximum reconstructed embedded ICC profile bytes retained in the inventory.
    pub max_decoded_icc_bytes: u64,
    /// Maximum preview and main frames retained in one inventory.
    pub max_frames: usize,
    /// Maximum TOC entries in an individual frame.
    pub max_toc_entries_per_frame: usize,
    /// Maximum TOC entries retained across all frames.
    pub max_total_toc_entries: usize,
    /// Maximum serialized bits in an individual frame header.
    pub max_frame_header_bits: u64,
    /// Maximum bytes in an individual frame name.
    pub max_frame_name_bytes: usize,
    /// Maximum skipped payload bits in each extension bundle.
    pub max_extension_bits: u64,
    /// Maximum total bytes named by all frame TOCs.
    pub max_total_section_bytes: u64,
}

impl Default for InventoryLimits {
    fn default() -> Self {
        Self {
            max_image_header_bytes: 1 << 20,
            max_encoded_icc_bytes: 1 << 28,
            max_decoded_icc_bytes: 1 << 28,
            max_frames: 16_384,
            max_toc_entries_per_frame: 65_536,
            max_total_toc_entries: 1 << 20,
            max_frame_header_bits: 1 << 20,
            max_frame_name_bytes: 4_096,
            max_extension_bits: 1 << 20,
            max_total_section_bytes: 1 << 30,
        }
    }
}

/// A range in the contiguous standard codestream, measured in bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitRange {
    pub offset: u64,
    pub length: u64,
}

impl BitRange {
    fn between(start: u64, end: u64) -> Result<Self, InventoryError> {
        Ok(Self {
            offset: start,
            length: end.checked_sub(start).ok_or(InventoryError::SizeOverflow)?,
        })
    }

    #[must_use]
    pub fn end(self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

/// A range in the contiguous standard codestream, measured in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteRange {
    pub offset: u64,
    pub length: u64,
}

impl ByteRange {
    #[must_use]
    pub fn end(self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

/// Image sample representation declared by the JPEG XL image header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleBitDepth {
    Integer {
        bits_per_sample: u32,
    },
    Float {
        bits_per_sample: u32,
        exponent_bits_per_sample: u32,
    },
}

/// Finite `f32` value stored by exact IEEE 754 bits so inventories remain equality-comparable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FiniteF32(u32);

impl FiniteF32 {
    #[must_use]
    pub fn from_f32(value: f32) -> Option<Self> {
        value.is_finite().then_some(Self(value.to_bits()))
    }

    #[must_use]
    pub fn from_bits(bits: u32) -> Option<Self> {
        Self::from_f32(f32::from_bits(bits))
    }

    #[must_use]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// JPEG XL color-space declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ColourSpaceInventory {
    #[default]
    Rgb = 0,
    Grey = 1,
    Xyb = 2,
    Unknown = 3,
}

/// Integer-scaled custom xy chromaticity coordinate (`1_000_000` units per 1.0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChromaticityInventory {
    pub x: i32,
    pub y: i32,
}

/// JPEG XL white-point declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WhitePointInventory {
    #[default]
    D65,
    Custom(ChromaticityInventory),
    E,
    Dci,
}

/// JPEG XL RGB-primary declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrimariesInventory {
    #[default]
    Srgb,
    Custom {
        red: ChromaticityInventory,
        green: ChromaticityInventory,
        blue: ChromaticityInventory,
    },
    Bt2100,
    P3,
}

/// JPEG XL transfer-function declaration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferFunctionInventory {
    Gamma {
        scaled_gamma: u32,
        inverted: bool,
    },
    Bt709,
    Unknown,
    Linear,
    #[default]
    Srgb,
    Pq,
    Dci,
    Hlg,
}

/// ICC rendering intent declared by an enumerated JPEG XL color encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum RenderingIntentInventory {
    Perceptual = 0,
    #[default]
    Relative = 1,
    Saturation = 2,
    Absolute = 3,
}

/// Standard JPEG XL color encoding, either enumerated or supplied by an embedded ICC profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourEncodingInventory {
    Enumerated {
        colour_space: ColourSpaceInventory,
        white_point: WhitePointInventory,
        primaries: PrimariesInventory,
        transfer_function: TransferFunctionInventory,
        rendering_intent: RenderingIntentInventory,
    },
    IccProfile {
        colour_space: ColourSpaceInventory,
    },
}

/// Semantic type and parameters of one JPEG XL extra channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtraChannelTypeInventory {
    Alpha {
        associated: bool,
    },
    Depth,
    SpotColour {
        red: FiniteF32,
        green: FiniteF32,
        blue: FiniteF32,
        solidity: FiniteF32,
    },
    SelectionMask,
    Black,
    Cfa {
        channel: u32,
    },
    Thermal,
    NonOptional,
    Optional,
}

/// Complete image-header declaration for one extra channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtraChannelInventory {
    pub channel_type: ExtraChannelTypeInventory,
    pub bit_depth: SampleBitDepth,
    pub dimension_shift: u32,
    pub name_bytes: Vec<u8>,
}

/// HDR tone-mapping metadata from the JPEG XL image header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ToneMappingInventory {
    pub intensity_target: FiniteF32,
    pub min_nits: FiniteF32,
    pub relative_to_max_display: bool,
    pub linear_below: FiniteF32,
}

/// Resolved XYB opsin inverse parameters from the image header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpsinInverseMatrixInventory {
    pub inverse_matrix: [[FiniteF32; 3]; 3],
    pub opsin_bias: [FiniteF32; 3],
    pub quant_bias: [FiniteF32; 3],
    pub quant_bias_numerator: FiniteF32,
}

/// Resolved standard or custom image upsampling weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpsamplingWeightsInventory {
    pub up2: [FiniteF32; 15],
    pub up4: [FiniteF32; 55],
    pub up8: [FiniteF32; 210],
}

/// Animation timing metadata from the JPEG XL image header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimationInventory {
    pub ticks_per_second_numerator: u32,
    pub ticks_per_second_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
}

/// Reconstructed embedded ICC profile and its compressed codestream range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedIccInventory {
    /// Entropy-coded ICC payload immediately following the base image header.
    pub bit_range: BitRange,
    /// Size of the intermediate transformed ICC byte stream.
    pub encoded_byte_count: u64,
    /// Original ICC profile bytes reconstructed from the transformed stream.
    pub profile: Vec<u8>,
}

/// Bounded, owned subset of the standard JPEG XL image header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageHeaderInventory {
    pub bit_range: BitRange,
    pub width: u32,
    pub height: u32,
    pub orientation: u32,
    pub intrinsic_size: Option<(u32, u32)>,
    pub preview_size: Option<(u32, u32)>,
    pub bit_depth: SampleBitDepth,
    pub modular_16bit_buffers: bool,
    pub extra_channel_count: u32,
    pub extra_channels: Vec<ExtraChannelInventory>,
    pub xyb_encoded: bool,
    pub grayscale: bool,
    pub colour_encoding: ColourEncodingInventory,
    pub tone_mapping: ToneMappingInventory,
    /// Present only when the codestream encodes color in XYB.
    pub opsin_inverse_matrix: Option<OpsinInverseMatrixInventory>,
    pub upsampling_weights: UpsamplingWeightsInventory,
    pub embedded_icc: Option<EmbeddedIccInventory>,
    pub animation: Option<AnimationInventory>,
}

/// Frame type declared by a standard JPEG XL frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    Regular = 0,
    LowFrequency = 1,
    ReferenceOnly = 2,
    SkipProgressive = 3,
}

/// Coding mode declared by a standard JPEG XL frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameEncoding {
    VarDct = 0,
    Modular = 1,
}

/// Standard JPEG XL frame blending operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameBlendMode {
    #[default]
    Replace = 0,
    Add = 1,
    Blend = 2,
    MultiplyAdd = 3,
    Multiply = 4,
}

/// Blending contract for the color image or one extra channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameBlendInfo {
    pub mode: FrameBlendMode,
    /// Extra-channel index providing alpha when the mode uses alpha.
    pub alpha_channel: Option<u32>,
    pub clamp: bool,
    /// Reference-frame slot supplying the destination canvas.
    pub source: u32,
}

/// Progressive-pass schedule declared by a frame header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FramePassesInventory {
    /// Per-pass coefficient shift, with one entry for every pass except the final pass.
    pub shifts: Vec<u32>,
    /// Resolution divisor for each progressive downsampling boundary.
    pub downsampling: Vec<u32>,
    /// Inclusive pass index at which the corresponding downsampling boundary ends.
    pub last_pass: Vec<u32>,
}

/// Finite IEEE 754 binary16 value retained exactly as serialized in a JPEG XL header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct FiniteF16(u16);

impl FiniteF16 {
    /// Construct a finite binary16 value. Infinity and NaN encodings are rejected.
    #[must_use]
    pub const fn from_bits(bits: u16) -> Option<Self> {
        if bits & 0x7c00 == 0x7c00 {
            None
        } else {
            Some(Self(bits))
        }
    }

    #[must_use]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert this value exactly to `f32`.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        let bits = u32::from(self.0);
        let sign = (bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let fraction = bits & 0x03ff;
        let expanded = if exponent == 0 {
            if fraction == 0 {
                sign
            } else {
                let top_bit = 31 - fraction.leading_zeros();
                let biased_exponent = top_bit + 103;
                let mantissa = (fraction - (1 << top_bit)) << (23 - top_bit);
                sign | (biased_exponent << 23) | mantissa
            }
        } else {
            sign | ((exponent + 112) << 23) | (fraction << 13)
        };
        f32::from_bits(expanded)
    }
}

/// Gaborish filter declaration from a frame header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaborishInventory {
    Disabled,
    #[default]
    Default,
    Custom {
        weights: [[FiniteF16; 2]; 3],
    },
}

/// Custom EPF channel weights and the two serialized zero-flush thresholds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpfWeightsInventory {
    pub channel_scale: [FiniteF16; 3],
    pub pass1_zeroflush: FiniteF16,
    pub pass2_zeroflush: FiniteF16,
}

/// Custom EPF sigma values. `quant_mul` is serialized only for VarDCT frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpfSigmaInventory {
    pub quant_mul: Option<FiniteF16>,
    pub pass0_sigma_scale: FiniteF16,
    pub pass2_sigma_scale: FiniteF16,
    pub border_sad_mul: FiniteF16,
}

/// Edge-preserving filter declaration from a frame header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgePreservingFilterInventory {
    Disabled,
    Enabled {
        iterations: u32,
        /// `None` selects the standard eight-entry sharpness LUT.
        sharp_lut: Option<[FiniteF16; 8]>,
        /// `None` selects the standard channel scales and zero-flush thresholds.
        weights: Option<EpfWeightsInventory>,
        /// `None` selects the standard EPF sigma values.
        sigma: Option<EpfSigmaInventory>,
        /// Serialized for Modular frames; VarDCT frames leave this as `None`.
        sigma_for_modular: Option<FiniteF16>,
    },
}

impl Default for EdgePreservingFilterInventory {
    fn default() -> Self {
        Self::Enabled {
            iterations: 2,
            sharp_lut: None,
            weights: None,
            sigma: None,
            sigma_for_modular: None,
        }
    }
}

/// Restoration-filter contract retained from a standard frame header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RestorationFilterInventory {
    /// The complete standard default selected by the filter bundle's `all_default` bit.
    #[default]
    Default,
    Custom {
        gaborish: GaborishInventory,
        epf: EdgePreservingFilterInventory,
    },
}

/// Logical meaning of a physical TOC entry after undoing an optional TOC permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSectionKind {
    /// Single-section special case containing all frame data.
    Single,
    LowFrequencyGlobal,
    HighFrequencyGlobal,
    LowFrequencyGroup {
        group_index: u64,
    },
    PassGroup {
        pass_index: u32,
        group_index: u64,
    },
}

/// One physical frame section and its exact byte/bit range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSection {
    /// Position of this section among the physically serialized frame sections.
    pub bitstream_index: u32,
    /// Logical TOC index after undoing an optional entropy-coded permutation.
    pub toc_index: u32,
    pub kind: FrameSectionKind,
    pub bytes: ByteRange,
    pub bits: BitRange,
}

/// Bounded standard frame-header and TOC inventory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameInventory {
    pub frame_index: u32,
    pub is_preview: bool,
    pub header_bits: BitRange,
    pub toc_bits: BitRange,
    pub toc_permuted: bool,
    pub frame_type: FrameType,
    pub encoding: FrameEncoding,
    pub flags: u64,
    pub do_ycbcr: bool,
    pub jpeg_upsampling: [u32; 3],
    pub upsampling: u32,
    /// One upsampling factor per image-header extra channel, in channel order.
    pub extra_channel_upsampling: Vec<u32>,
    pub group_size_shift: u32,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    pub width: u32,
    pub height: u32,
    pub x0: i32,
    pub y0: i32,
    pub have_crop: bool,
    pub color_blend: FrameBlendInfo,
    /// One blending contract per image-header extra channel, in channel order.
    pub extra_channel_blends: Vec<FrameBlendInfo>,
    pub num_passes: u32,
    pub progressive_passes: FramePassesInventory,
    pub duration_ticks: u32,
    pub timecode: Option<u32>,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_color_transform: bool,
    pub name_bytes: Vec<u8>,
    pub restoration_filter: RestorationFilterInventory,
    pub group_count: u64,
    pub low_frequency_group_count: u64,
    pub sections: Vec<FrameSection>,
}

/// Standard codestream metadata and physical frame section ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodestreamInventory {
    pub codestream_bytes: u64,
    pub image_header: ImageHeaderInventory,
    pub frames: Vec<FrameInventory>,
}

/// Failure while constructing a bounded standard codestream inventory.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum InventoryError {
    #[error("JPEG XL image header is invalid: {0}")]
    ImageHeader(String),
    #[error("unexpected end of codestream at bit {bit_offset}")]
    UnexpectedEndOfBits { bit_offset: u64 },
    #[error("non-zero byte-alignment padding at bit {bit_offset}")]
    NonZeroPadding { bit_offset: u64 },
    #[error("invalid {name} enum value {value}")]
    InvalidEnum { name: &'static str, value: u32 },
    #[error("invalid frame header: {0}")]
    InvalidFrame(&'static str),
    #[error("embedded ICC profile is invalid: {0}")]
    InvalidIcc(String),
    #[error("inventory resource limit exceeded for {0}")]
    ResourceLimit(&'static str),
    #[error("allocation failed while building {0}")]
    AllocationFailed(&'static str),
    #[error("codestream contains {remaining_bits} trailing bits after its final frame")]
    TrailingData { remaining_bits: u64 },
    #[error("inventory size arithmetic overflow")]
    SizeOverflow,
}

struct ParsedImageHeader {
    inventory: ImageHeaderInventory,
    frame_start_bits: u64,
    context: ImageContext,
}

#[derive(Clone, Copy)]
struct ImageContext {
    width: u32,
    height: u32,
    preview_size: Option<(u32, u32)>,
    xyb_encoded: bool,
    num_extra_channels: u32,
    have_animation: bool,
    have_timecodes: bool,
}

pub(crate) fn parse_codestream_inventory(
    codestream: &[u8],
    limits: InventoryLimits,
) -> Result<CodestreamInventory, InventoryError> {
    let codestream_bytes =
        u64::try_from(codestream.len()).map_err(|_| InventoryError::SizeOverflow)?;
    let parsed_image = parse_image_header(codestream, limits)?;
    let mut reader = BitReader::new(codestream);
    reader
        .skip_bits(parsed_image.frame_start_bits)
        .map_err(|error| map_reader_error(error, reader.bit_offset()))?;
    zero_pad(&mut reader)?;

    let mut frames = Vec::new();
    let mut is_preview = parsed_image.context.preview_size.is_some();
    let mut total_toc_entries = 0usize;
    let mut total_section_bytes = 0u64;

    loop {
        if frames.len() >= limits.max_frames {
            return Err(InventoryError::ResourceLimit("frame count"));
        }
        let frame_index = u32::try_from(frames.len()).map_err(|_| InventoryError::SizeOverflow)?;
        let frame_context = if is_preview {
            let (width, height) = parsed_image
                .context
                .preview_size
                .ok_or(InventoryError::InvalidFrame("missing preview size"))?;
            FrameContext {
                width,
                height,
                ..FrameContext::from_image(parsed_image.context)
            }
        } else {
            FrameContext::from_image(parsed_image.context)
        };

        let header_start = reader.bit_offset();
        let header = parse_frame_header(&mut reader, frame_context, is_preview, limits)?;
        let header_end = reader.bit_offset();
        if header_end
            .checked_sub(header_start)
            .ok_or(InventoryError::SizeOverflow)?
            > limits.max_frame_header_bits
        {
            return Err(InventoryError::ResourceLimit("frame header bits"));
        }

        let counts = compute_frame_counts(&header)?;
        if counts.toc_entries > MAX_TOC_ENTRIES {
            return Err(InventoryError::InvalidFrame("too many TOC entries"));
        }
        let toc_entries = usize::try_from(counts.toc_entries)
            .map_err(|_| InventoryError::ResourceLimit("TOC entries per frame"))?;
        if toc_entries > limits.max_toc_entries_per_frame {
            return Err(InventoryError::ResourceLimit("TOC entries per frame"));
        }
        total_toc_entries = total_toc_entries
            .checked_add(toc_entries)
            .ok_or(InventoryError::SizeOverflow)?;
        if total_toc_entries > limits.max_total_toc_entries {
            return Err(InventoryError::ResourceLimit("total TOC entries"));
        }

        let toc_start = reader.bit_offset();
        let toc = parse_toc(&mut reader, toc_entries, codestream)?;
        let toc_end = reader.bit_offset();
        let section_start_byte = toc_end / 8;
        let mut section_cursor = section_start_byte;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(toc_entries)
            .map_err(|_| InventoryError::AllocationFailed("frame sections"))?;

        for (bitstream_index, (&toc_index, &length)) in toc
            .logical_indices_in_bitstream_order
            .iter()
            .zip(&toc.entry_lengths_in_bitstream_order)
            .enumerate()
        {
            let length = u64::from(length);
            total_section_bytes = total_section_bytes
                .checked_add(length)
                .ok_or(InventoryError::SizeOverflow)?;
            if total_section_bytes > limits.max_total_section_bytes {
                return Err(InventoryError::ResourceLimit("total frame section bytes"));
            }
            let section_end = section_cursor
                .checked_add(length)
                .ok_or(InventoryError::SizeOverflow)?;
            if section_end > codestream_bytes {
                return Err(InventoryError::UnexpectedEndOfBits {
                    bit_offset: reader.bit_offset(),
                });
            }
            let bit_offset = section_cursor
                .checked_mul(8)
                .ok_or(InventoryError::SizeOverflow)?;
            let bit_length = length.checked_mul(8).ok_or(InventoryError::SizeOverflow)?;
            sections.push(FrameSection {
                bitstream_index: u32::try_from(bitstream_index)
                    .map_err(|_| InventoryError::SizeOverflow)?,
                toc_index: u32::try_from(toc_index).map_err(|_| InventoryError::SizeOverflow)?,
                kind: section_kind(toc_index as u64, counts, header.num_passes),
                bytes: ByteRange {
                    offset: section_cursor,
                    length,
                },
                bits: BitRange {
                    offset: bit_offset,
                    length: bit_length,
                },
            });
            section_cursor = section_end;
        }

        let section_bits = section_cursor
            .checked_sub(section_start_byte)
            .and_then(|bytes| bytes.checked_mul(8))
            .ok_or(InventoryError::SizeOverflow)?;
        reader
            .skip_bits(section_bits)
            .map_err(|error| map_reader_error(error, reader.bit_offset()))?;

        let is_last = header.is_last;
        frames.push(FrameInventory {
            frame_index,
            is_preview,
            header_bits: BitRange::between(header_start, header_end)?,
            toc_bits: BitRange::between(toc_start, toc_end)?,
            toc_permuted: toc.permuted,
            frame_type: header.frame_type,
            encoding: header.encoding,
            flags: header.flags,
            do_ycbcr: header.do_ycbcr,
            jpeg_upsampling: header.jpeg_upsampling,
            upsampling: header.upsampling,
            extra_channel_upsampling: header.extra_channel_upsampling,
            group_size_shift: header.group_size_shift,
            x_qm_scale: header.x_qm_scale,
            b_qm_scale: header.b_qm_scale,
            width: header.width,
            height: header.height,
            x0: header.x0,
            y0: header.y0,
            have_crop: header.have_crop,
            color_blend: header.color_blend,
            extra_channel_blends: header.extra_channel_blends,
            num_passes: header.num_passes,
            progressive_passes: header.progressive_passes,
            duration_ticks: header.duration,
            timecode: header.timecode,
            is_last,
            save_as_reference: header.save_as_reference,
            save_before_color_transform: header.save_before_color_transform,
            name_bytes: header.name_bytes,
            restoration_filter: header.restoration_filter,
            group_count: counts.groups,
            low_frequency_group_count: counts.low_frequency_groups,
            sections,
        });

        if is_last {
            if is_preview {
                is_preview = false;
            } else {
                break;
            }
        }
    }

    let remaining_bits = reader.remaining_bits();
    if remaining_bits != 0 {
        return Err(InventoryError::TrailingData { remaining_bits });
    }

    Ok(CodestreamInventory {
        codestream_bytes,
        image_header: parsed_image.inventory,
        frames,
    })
}

fn parse_image_header(
    codestream: &[u8],
    limits: InventoryLimits,
) -> Result<ParsedImageHeader, InventoryError> {
    let max_header_bytes = usize::try_from(limits.max_image_header_bytes).unwrap_or(usize::MAX);
    let visible_bytes = codestream.len().min(max_header_bytes);
    let mut bitstream = ImageBitstream::new(&codestream[..visible_bytes]);
    let image = match ImageHeader::parse(&mut bitstream, ()) {
        Ok(image) => image,
        Err(error) if error.unexpected_eof() && visible_bytes < codestream.len() => {
            return Err(InventoryError::ResourceLimit("image header bytes"));
        }
        Err(error) if error.unexpected_eof() => {
            return Err(InventoryError::UnexpectedEndOfBits {
                bit_offset: u64::try_from(bitstream.num_read_bits())
                    .map_err(|_| InventoryError::SizeOverflow)?,
            });
        }
        Err(error) => return Err(InventoryError::ImageHeader(error.to_string())),
    };
    let header_bits =
        u64::try_from(bitstream.num_read_bits()).map_err(|_| InventoryError::SizeOverflow)?;

    let metadata = &image.metadata;
    let extra_channel_count = u32::try_from(metadata.ec_info.len())
        .map_err(|_| InventoryError::ResourceLimit("extra channels"))?;
    if extra_channel_count > MAX_EXTRA_CHANNELS {
        return Err(InventoryError::ResourceLimit("extra channels"));
    }
    let bit_depth = sample_bit_depth(metadata.bit_depth);
    let mut extra_channels = Vec::new();
    extra_channels
        .try_reserve_exact(metadata.ec_info.len())
        .map_err(|_| InventoryError::AllocationFailed("extra-channel inventory"))?;
    for info in &metadata.ec_info {
        extra_channels.push(extra_channel_inventory(info)?);
    }
    let colour_encoding = colour_encoding_inventory(&metadata.colour_encoding);
    let tone_mapping = ToneMappingInventory {
        intensity_target: finite_header_f32(metadata.tone_mapping.intensity_target)?,
        min_nits: finite_header_f32(metadata.tone_mapping.min_nits)?,
        relative_to_max_display: metadata.tone_mapping.relative_to_max_display,
        linear_below: finite_header_f32(metadata.tone_mapping.linear_below)?,
    };
    let opsin_inverse_matrix = if metadata.xyb_encoded {
        let matrix = &metadata.opsin_inverse_matrix;
        Some(OpsinInverseMatrixInventory {
            inverse_matrix: finite_header_matrix(matrix.inv_mat)?,
            opsin_bias: finite_header_array(matrix.opsin_bias)?,
            quant_bias: finite_header_array(matrix.quant_bias)?,
            quant_bias_numerator: finite_header_f32(matrix.quant_bias_numerator)?,
        })
    } else {
        None
    };
    let upsampling_weights = UpsamplingWeightsInventory {
        up2: finite_header_array(metadata.up2_weight)?,
        up4: finite_header_array(metadata.up4_weight)?,
        up8: finite_header_array(metadata.up8_weight)?,
    };
    let animation = metadata
        .animation
        .as_ref()
        .map(|animation| AnimationInventory {
            ticks_per_second_numerator: animation.tps_numerator,
            ticks_per_second_denominator: animation.tps_denominator,
            num_loops: animation.num_loops,
            have_timecodes: animation.have_timecodes,
        });
    let preview_size = metadata
        .preview
        .as_ref()
        .map(|preview| (preview.width, preview.height));

    let embedded_icc = if metadata.colour_encoding.want_icc() {
        Some(parse_embedded_icc(codestream, header_bits, limits)?)
    } else {
        None
    };
    let frame_start_bits = embedded_icc
        .as_ref()
        .and_then(|icc| icc.bit_range.end())
        .unwrap_or(header_bits);

    let inventory = ImageHeaderInventory {
        bit_range: BitRange {
            offset: 0,
            length: header_bits,
        },
        width: image.size.width,
        height: image.size.height,
        orientation: metadata.orientation,
        intrinsic_size: metadata
            .intrinsic_size
            .as_ref()
            .map(|size| (size.width, size.height)),
        preview_size,
        bit_depth,
        modular_16bit_buffers: metadata.modular_16bit_buffers,
        extra_channel_count,
        extra_channels,
        xyb_encoded: metadata.xyb_encoded,
        grayscale: metadata.grayscale(),
        colour_encoding,
        tone_mapping,
        opsin_inverse_matrix,
        upsampling_weights,
        embedded_icc,
        animation,
    };
    Ok(ParsedImageHeader {
        inventory,
        frame_start_bits,
        context: ImageContext {
            width: image.size.width,
            height: image.size.height,
            preview_size,
            xyb_encoded: metadata.xyb_encoded,
            num_extra_channels: extra_channel_count,
            have_animation: metadata.animation.is_some(),
            have_timecodes: animation.is_some_and(|animation| animation.have_timecodes),
        },
    })
}

fn sample_bit_depth(bit_depth: JxlBitDepth) -> SampleBitDepth {
    match bit_depth {
        JxlBitDepth::IntegerSample { bits_per_sample } => {
            SampleBitDepth::Integer { bits_per_sample }
        }
        JxlBitDepth::FloatSample {
            bits_per_sample,
            exp_bits,
        } => SampleBitDepth::Float {
            bits_per_sample,
            exponent_bits_per_sample: exp_bits,
        },
    }
}

fn finite_header_f32(value: f32) -> Result<FiniteF32, InventoryError> {
    FiniteF32::from_f32(value)
        .ok_or_else(|| InventoryError::ImageHeader("non-finite image-header value".into()))
}

fn finite_header_array<const N: usize>(values: [f32; N]) -> Result<[FiniteF32; N], InventoryError> {
    let mut output = [FiniteF32::default(); N];
    for (output, value) in output.iter_mut().zip(values) {
        *output = finite_header_f32(value)?;
    }
    Ok(output)
}

fn finite_header_matrix(values: [[f32; 3]; 3]) -> Result<[[FiniteF32; 3]; 3], InventoryError> {
    Ok([
        finite_header_array(values[0])?,
        finite_header_array(values[1])?,
        finite_header_array(values[2])?,
    ])
}

fn extra_channel_inventory(
    info: &jxl_image::ExtraChannelInfo,
) -> Result<ExtraChannelInventory, InventoryError> {
    let channel_type = match info.ty {
        JxlExtraChannelType::Alpha { alpha_associated } => ExtraChannelTypeInventory::Alpha {
            associated: alpha_associated,
        },
        JxlExtraChannelType::Depth => ExtraChannelTypeInventory::Depth,
        JxlExtraChannelType::SpotColour {
            red,
            green,
            blue,
            solidity,
        } => ExtraChannelTypeInventory::SpotColour {
            red: finite_header_f32(red)?,
            green: finite_header_f32(green)?,
            blue: finite_header_f32(blue)?,
            solidity: finite_header_f32(solidity)?,
        },
        JxlExtraChannelType::SelectionMask => ExtraChannelTypeInventory::SelectionMask,
        JxlExtraChannelType::Black => ExtraChannelTypeInventory::Black,
        JxlExtraChannelType::Cfa { cfa_channel } => ExtraChannelTypeInventory::Cfa {
            channel: cfa_channel,
        },
        JxlExtraChannelType::Thermal => ExtraChannelTypeInventory::Thermal,
        JxlExtraChannelType::NonOptional => ExtraChannelTypeInventory::NonOptional,
        JxlExtraChannelType::Optional => ExtraChannelTypeInventory::Optional,
    };
    let source_name = info.name.as_bytes();
    let mut name_bytes = Vec::new();
    name_bytes
        .try_reserve_exact(source_name.len())
        .map_err(|_| InventoryError::AllocationFailed("extra-channel name"))?;
    name_bytes.extend_from_slice(source_name);
    Ok(ExtraChannelInventory {
        channel_type,
        bit_depth: sample_bit_depth(info.bit_depth),
        dimension_shift: info.dim_shift,
        name_bytes,
    })
}

fn colour_encoding_inventory(encoding: &JxlColourEncoding) -> ColourEncodingInventory {
    match encoding {
        JxlColourEncoding::Enum(enumerated) => ColourEncodingInventory::Enumerated {
            colour_space: colour_space_inventory(enumerated.colour_space),
            white_point: match enumerated.white_point {
                JxlWhitePoint::D65 => WhitePointInventory::D65,
                JxlWhitePoint::Custom(point) => {
                    WhitePointInventory::Custom(ChromaticityInventory {
                        x: point.x,
                        y: point.y,
                    })
                }
                JxlWhitePoint::E => WhitePointInventory::E,
                JxlWhitePoint::Dci => WhitePointInventory::Dci,
            },
            primaries: match enumerated.primaries {
                JxlPrimaries::Srgb => PrimariesInventory::Srgb,
                JxlPrimaries::Custom { red, green, blue } => PrimariesInventory::Custom {
                    red: ChromaticityInventory { x: red.x, y: red.y },
                    green: ChromaticityInventory {
                        x: green.x,
                        y: green.y,
                    },
                    blue: ChromaticityInventory {
                        x: blue.x,
                        y: blue.y,
                    },
                },
                JxlPrimaries::Bt2100 => PrimariesInventory::Bt2100,
                JxlPrimaries::P3 => PrimariesInventory::P3,
            },
            transfer_function: match enumerated.tf {
                JxlTransferFunction::Gamma { g, inverted } => TransferFunctionInventory::Gamma {
                    scaled_gamma: g,
                    inverted,
                },
                JxlTransferFunction::Bt709 => TransferFunctionInventory::Bt709,
                JxlTransferFunction::Unknown => TransferFunctionInventory::Unknown,
                JxlTransferFunction::Linear => TransferFunctionInventory::Linear,
                JxlTransferFunction::Srgb => TransferFunctionInventory::Srgb,
                JxlTransferFunction::Pq => TransferFunctionInventory::Pq,
                JxlTransferFunction::Dci => TransferFunctionInventory::Dci,
                JxlTransferFunction::Hlg => TransferFunctionInventory::Hlg,
            },
            rendering_intent: match enumerated.rendering_intent {
                JxlRenderingIntent::Perceptual => RenderingIntentInventory::Perceptual,
                JxlRenderingIntent::Relative => RenderingIntentInventory::Relative,
                JxlRenderingIntent::Saturation => RenderingIntentInventory::Saturation,
                JxlRenderingIntent::Absolute => RenderingIntentInventory::Absolute,
            },
        },
        JxlColourEncoding::IccProfile(colour_space) => ColourEncodingInventory::IccProfile {
            colour_space: colour_space_inventory(*colour_space),
        },
    }
}

const fn colour_space_inventory(colour_space: JxlColourSpace) -> ColourSpaceInventory {
    match colour_space {
        JxlColourSpace::Rgb => ColourSpaceInventory::Rgb,
        JxlColourSpace::Grey => ColourSpaceInventory::Grey,
        JxlColourSpace::Xyb => ColourSpaceInventory::Xyb,
        JxlColourSpace::Unknown => ColourSpaceInventory::Unknown,
    }
}

fn parse_embedded_icc(
    codestream: &[u8],
    header_bits: u64,
    limits: InventoryLimits,
) -> Result<EmbeddedIccInventory, InventoryError> {
    let header_bits_usize =
        usize::try_from(header_bits).map_err(|_| InventoryError::SizeOverflow)?;
    let mut bitstream = ImageBitstream::new(codestream);
    if let Err(error) = bitstream.skip_bits(header_bits_usize) {
        return Err(map_metadata_bitstream_error(
            error,
            bitstream.num_read_bits(),
        ));
    }

    let mut size_probe = bitstream.clone();
    let encoded_byte_count = match size_probe.read_u64() {
        Ok(size) => size,
        Err(error) => {
            return Err(map_metadata_bitstream_error(
                error,
                size_probe.num_read_bits(),
            ));
        }
    };
    if encoded_byte_count > limits.max_encoded_icc_bytes {
        return Err(InventoryError::ResourceLimit("encoded ICC profile bytes"));
    }

    let encoded = match jxl_color::icc::read_icc(&mut bitstream) {
        Ok(encoded) => encoded,
        Err(error) if error.unexpected_eof() => {
            return Err(InventoryError::UnexpectedEndOfBits {
                bit_offset: u64::try_from(bitstream.num_read_bits()).unwrap_or(u64::MAX),
            });
        }
        Err(error) => return Err(InventoryError::InvalidIcc(error.to_string())),
    };
    let actual_encoded_bytes =
        u64::try_from(encoded.len()).map_err(|_| InventoryError::SizeOverflow)?;
    if actual_encoded_bytes != encoded_byte_count {
        return Err(InventoryError::InvalidIcc(
            "encoded ICC size does not match its declaration".into(),
        ));
    }
    let decoded_size = transformed_icc_output_size(&encoded)?;
    if decoded_size > limits.max_decoded_icc_bytes {
        return Err(InventoryError::ResourceLimit("decoded ICC profile bytes"));
    }
    let profile = jxl_color::icc::decode_icc(&encoded)
        .map_err(|error| InventoryError::InvalidIcc(error.to_string()))?;
    let actual_decoded_bytes =
        u64::try_from(profile.len()).map_err(|_| InventoryError::SizeOverflow)?;
    if actual_decoded_bytes != decoded_size {
        return Err(InventoryError::InvalidIcc(
            "decoded ICC size does not match its declaration".into(),
        ));
    }
    let icc_end =
        u64::try_from(bitstream.num_read_bits()).map_err(|_| InventoryError::SizeOverflow)?;
    Ok(EmbeddedIccInventory {
        bit_range: BitRange::between(header_bits, icc_end)?,
        encoded_byte_count,
        profile,
    })
}

fn transformed_icc_output_size(encoded: &[u8]) -> Result<u64, InventoryError> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for &byte in encoded.iter().take(10) {
        let payload = u64::from(byte & 0x7f);
        if shift == 63 && payload > 1 {
            return Err(InventoryError::InvalidIcc(
                "decoded ICC size varint overflows u64".into(),
            ));
        }
        value |= payload
            .checked_shl(shift)
            .ok_or(InventoryError::SizeOverflow)?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift = shift.checked_add(7).ok_or(InventoryError::SizeOverflow)?;
        if shift >= 64 {
            break;
        }
    }
    Err(InventoryError::InvalidIcc(
        "decoded ICC size varint is truncated".into(),
    ))
}

#[derive(Clone, Copy)]
struct FrameContext {
    width: u32,
    height: u32,
    xyb_encoded: bool,
    num_extra_channels: u32,
    have_animation: bool,
    have_timecodes: bool,
}

impl FrameContext {
    fn from_image(image: ImageContext) -> Self {
        Self {
            width: image.width,
            height: image.height,
            xyb_encoded: image.xyb_encoded,
            num_extra_channels: image.num_extra_channels,
            have_animation: image.have_animation,
            have_timecodes: image.have_timecodes,
        }
    }
}

struct ParsedFrameHeader {
    frame_type: FrameType,
    encoding: FrameEncoding,
    flags: u64,
    do_ycbcr: bool,
    jpeg_upsampling: [u32; 3],
    upsampling: u32,
    extra_channel_upsampling: Vec<u32>,
    group_size_shift: u32,
    x_qm_scale: u32,
    b_qm_scale: u32,
    num_passes: u32,
    progressive_passes: FramePassesInventory,
    lf_level: u32,
    have_crop: bool,
    color_blend: FrameBlendInfo,
    extra_channel_blends: Vec<FrameBlendInfo>,
    x0: i32,
    y0: i32,
    width: u32,
    height: u32,
    duration: u32,
    timecode: Option<u32>,
    is_last: bool,
    save_as_reference: u32,
    save_before_color_transform: bool,
    name_bytes: Vec<u8>,
    restoration_filter: RestorationFilterInventory,
    max_horizontal_shift: u32,
    max_vertical_shift: u32,
}

fn parse_frame_header(
    reader: &mut BitReader<'_>,
    context: FrameContext,
    is_preview: bool,
    limits: InventoryLimits,
) -> Result<ParsedFrameHeader, InventoryError> {
    zero_pad(reader)?;
    let extra_channel_count = usize::try_from(context.num_extra_channels)
        .map_err(|_| InventoryError::ResourceLimit("extra channels"))?;
    let all_default = read_bool(reader)?;
    if all_default {
        return Ok(ParsedFrameHeader {
            frame_type: FrameType::Regular,
            encoding: FrameEncoding::VarDct,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            extra_channel_upsampling: vec![1; extra_channel_count],
            group_size_shift: 1,
            x_qm_scale: if context.xyb_encoded { 3 } else { 2 },
            b_qm_scale: 2,
            num_passes: 1,
            progressive_passes: FramePassesInventory::default(),
            lf_level: 0,
            have_crop: false,
            color_blend: FrameBlendInfo::default(),
            extra_channel_blends: vec![FrameBlendInfo::default(); extra_channel_count],
            x0: 0,
            y0: 0,
            width: context.width,
            height: context.height,
            duration: 0,
            timecode: None,
            is_last: true,
            save_as_reference: 0,
            save_before_color_transform: false,
            name_bytes: Vec::new(),
            restoration_filter: RestorationFilterInventory::default(),
            max_horizontal_shift: 0,
            max_vertical_shift: 0,
        });
    }

    let frame_type = match read_bits(reader, 2)? as u32 {
        0 => FrameType::Regular,
        1 => FrameType::LowFrequency,
        2 => FrameType::ReferenceOnly,
        3 => FrameType::SkipProgressive,
        value => {
            return Err(InventoryError::InvalidEnum {
                name: "FrameType",
                value,
            });
        }
    };
    if is_preview && frame_type != FrameType::Regular {
        return Err(InventoryError::InvalidFrame(
            "preview must use a regular frame",
        ));
    }
    let encoding = if read_bits(reader, 1)? == 0 {
        FrameEncoding::VarDct
    } else {
        FrameEncoding::Modular
    };
    let flags = read_u64(reader)?;
    let do_ycbcr = !context.xyb_encoded && read_bool(reader)?;
    let has_lf_frame = flags & FLAG_USE_LF_FRAME != 0;
    let mut jpeg_upsampling = [0u32; 3];
    if do_ycbcr && !has_lf_frame {
        for value in &mut jpeg_upsampling {
            *value = read_bits(reader, 2)? as u32;
        }
    }
    let upsampling = if has_lf_frame {
        1
    } else {
        read_u32(reader, [c(1), c(2), c(4), c(8)])?
    };
    let mut extra_channel_upsampling = vec![1; extra_channel_count];
    if !has_lf_frame {
        for value in &mut extra_channel_upsampling {
            *value = read_u32(reader, [c(1), c(2), c(4), c(8)])?;
        }
    }
    let group_size_shift = if encoding == FrameEncoding::Modular {
        read_bits(reader, 2)? as u32
    } else {
        1
    };
    let (x_qm_scale, b_qm_scale) = if encoding == FrameEncoding::VarDct && context.xyb_encoded {
        (read_bits(reader, 3)? as u32, read_bits(reader, 3)? as u32)
    } else {
        (2, 2)
    };
    let progressive_passes = if frame_type == FrameType::ReferenceOnly {
        FramePassesInventory::default()
    } else {
        parse_passes(reader)?
    };
    let num_passes = u32::try_from(progressive_passes.shifts.len())
        .map_err(|_| InventoryError::ResourceLimit("frame passes"))?
        .checked_add(1)
        .ok_or(InventoryError::SizeOverflow)?;
    let lf_level = if frame_type == FrameType::LowFrequency {
        read_u32(reader, [c(1), c(2), c(3), c(4)])?
    } else {
        0
    };
    if has_lf_frame && lf_level >= 4 {
        return Err(InventoryError::InvalidFrame("invalid LF level"));
    }
    let have_crop = frame_type != FrameType::LowFrequency && read_bool(reader)?;
    let x0 = if have_crop && frame_type != FrameType::ReferenceOnly {
        unpack_signed(read_u32(reader, frame_dimension_coder())?)
    } else {
        0
    };
    let y0 = if have_crop && frame_type != FrameType::ReferenceOnly {
        unpack_signed(read_u32(reader, frame_dimension_coder())?)
    } else {
        0
    };
    let width = if have_crop {
        read_u32(reader, frame_dimension_coder())?
    } else {
        context.width
    };
    let height = if have_crop {
        read_u32(reader, frame_dimension_coder())?
    } else {
        context.height
    };
    if width == 0 || height == 0 {
        return Err(InventoryError::InvalidFrame("empty frame extent"));
    }
    let completely_covers = i64::from(x0) <= 0
        && i64::from(y0) <= 0
        && i64::from(width) + i64::from(x0) >= i64::from(context.width)
        && i64::from(height) + i64::from(y0) >= i64::from(context.height);
    let full_frame = !have_crop || completely_covers;

    let normal_frame = matches!(frame_type, FrameType::Regular | FrameType::SkipProgressive);
    let color_blend = if normal_frame {
        parse_blending_info(reader, context.num_extra_channels, full_frame, None)?
    } else {
        FrameBlendInfo::default()
    };
    let mut replace_all = color_blend.mode == FrameBlendMode::Replace;
    let mut extra_channel_blends = Vec::new();
    extra_channel_blends
        .try_reserve_exact(extra_channel_count)
        .map_err(|_| InventoryError::AllocationFailed("extra-channel blending"))?;
    if normal_frame {
        for _ in 0..context.num_extra_channels {
            let blend = parse_blending_info(
                reader,
                context.num_extra_channels,
                full_frame,
                Some(color_blend.mode),
            )?;
            replace_all &= blend.mode == FrameBlendMode::Replace;
            extra_channel_blends.push(blend);
        }
    } else {
        extra_channel_blends.resize(extra_channel_count, FrameBlendInfo::default());
    }
    if is_preview && (have_crop || !replace_all) {
        return Err(InventoryError::InvalidFrame("preview cannot crop or blend"));
    }
    let duration = if normal_frame && context.have_animation {
        read_u32(reader, [c(0), c(1), b(0, 8), b(0, 32)])?
    } else {
        0
    };
    let timecode = if normal_frame && context.have_timecodes {
        Some(read_bits(reader, 32)? as u32)
    } else {
        None
    };
    let is_last = if normal_frame {
        read_bool(reader)?
    } else {
        frame_type == FrameType::Regular
    };
    let save_as_reference = if frame_type != FrameType::LowFrequency && !is_last {
        read_bits(reader, 2)? as u32
    } else {
        0
    };
    let can_be_referenced = !is_last
        && frame_type != FrameType::LowFrequency
        && (duration == 0 || save_as_reference != 0);
    let save_before_default_false = can_be_referenced
        && color_blend.mode == FrameBlendMode::Replace
        && full_frame
        && normal_frame;
    let save_before_color_transform =
        if frame_type == FrameType::ReferenceOnly || save_before_default_false {
            read_bool(reader)?
        } else {
            frame_type == FrameType::LowFrequency
        };
    if frame_type == FrameType::ReferenceOnly && !save_before_color_transform && !full_frame {
        return Err(InventoryError::InvalidFrame(
            "post-transform reference frame cannot be cropped",
        ));
    }
    let name_bytes = read_name(reader, limits.max_frame_name_bytes)?;
    let restoration_filter = parse_restoration_filter(reader, encoding, limits.max_extension_bits)?;
    parse_extensions(reader, limits.max_extension_bits)?;

    const H_SHIFT: [u32; 4] = [0, 1, 1, 0];
    const V_SHIFT: [u32; 4] = [0, 1, 0, 1];
    let max_horizontal_shift = jpeg_upsampling
        .iter()
        .map(|&value| H_SHIFT[value as usize])
        .max()
        .unwrap_or(0);
    let max_vertical_shift = jpeg_upsampling
        .iter()
        .map(|&value| V_SHIFT[value as usize])
        .max()
        .unwrap_or(0);

    Ok(ParsedFrameHeader {
        frame_type,
        encoding,
        flags,
        do_ycbcr,
        jpeg_upsampling,
        upsampling,
        extra_channel_upsampling,
        group_size_shift,
        x_qm_scale,
        b_qm_scale,
        num_passes,
        progressive_passes,
        lf_level,
        have_crop,
        color_blend,
        extra_channel_blends,
        x0,
        y0,
        width,
        height,
        duration,
        timecode,
        is_last,
        save_as_reference,
        save_before_color_transform,
        name_bytes,
        restoration_filter,
        max_horizontal_shift,
        max_vertical_shift,
    })
}

fn parse_passes(reader: &mut BitReader<'_>) -> Result<FramePassesInventory, InventoryError> {
    let num_passes = read_u32(reader, [c(1), c(2), c(3), b(4, 3)])?;
    if num_passes == 1 {
        return Ok(FramePassesInventory::default());
    }
    let num_downsampling = read_u32(reader, [c(0), c(1), c(2), b(3, 1)])?;
    let shift_count = usize::try_from(num_passes - 1)
        .map_err(|_| InventoryError::ResourceLimit("frame passes"))?;
    let mut shifts = Vec::new();
    shifts
        .try_reserve_exact(shift_count)
        .map_err(|_| InventoryError::AllocationFailed("frame passes"))?;
    for _ in 0..shift_count {
        shifts.push(read_bits(reader, 2)? as u32);
    }
    let mut downsampling = Vec::new();
    downsampling
        .try_reserve_exact(num_downsampling as usize)
        .map_err(|_| InventoryError::AllocationFailed("frame passes"))?;
    for _ in 0..num_downsampling {
        downsampling.push(read_u32(reader, [c(1), c(2), c(4), c(8)])?);
    }
    let mut last_pass = Vec::new();
    last_pass
        .try_reserve_exact(num_downsampling as usize)
        .map_err(|_| InventoryError::AllocationFailed("frame passes"))?;
    for _ in 0..num_downsampling {
        last_pass.push(read_u32(reader, [c(0), c(1), c(2), b(0, 3)])?);
    }
    if num_downsampling >= num_passes
        || downsampling.windows(2).any(|pair| pair[1] >= pair[0])
        || last_pass.windows(2).any(|pair| pair[1] <= pair[0])
        || last_pass.iter().any(|&pass| pass >= num_passes)
    {
        return Err(InventoryError::InvalidFrame("invalid progressive passes"));
    }
    Ok(FramePassesInventory {
        shifts,
        downsampling,
        last_pass,
    })
}

fn parse_blending_info(
    reader: &mut BitReader<'_>,
    num_extra_channels: u32,
    full_frame: bool,
    canvas_mode: Option<FrameBlendMode>,
) -> Result<FrameBlendInfo, InventoryError> {
    let mode = read_u32(reader, [c(0), c(1), c(2), b(3, 2)])?;
    let mode = match mode {
        0 => FrameBlendMode::Replace,
        1 => FrameBlendMode::Add,
        2 => FrameBlendMode::Blend,
        3 => FrameBlendMode::MultiplyAdd,
        4 => FrameBlendMode::Multiply,
        value => {
            return Err(InventoryError::InvalidEnum {
                name: "BlendingMode",
                value,
            });
        }
    };
    let uses_alpha = matches!(mode, FrameBlendMode::Blend | FrameBlendMode::MultiplyAdd);
    let alpha_channel = if num_extra_channels > 0 && uses_alpha {
        let alpha_channel = read_u32(reader, [c(0), c(1), c(2), b(3, 3)])?;
        if alpha_channel >= num_extra_channels {
            return Err(InventoryError::InvalidFrame(
                "invalid blending alpha channel",
            ));
        }
        Some(alpha_channel)
    } else {
        None
    };
    let clamp = if (num_extra_channels > 0 && uses_alpha) || mode == FrameBlendMode::Multiply {
        read_bool(reader)?
    } else {
        false
    };
    let reset_mode = canvas_mode.unwrap_or(mode);
    let source = if full_frame && reset_mode == FrameBlendMode::Replace {
        0
    } else {
        read_u32(reader, [c(0), c(1), c(2), c(3)])?
    };
    Ok(FrameBlendInfo {
        mode,
        alpha_channel,
        clamp,
        source,
    })
}

fn parse_restoration_filter(
    reader: &mut BitReader<'_>,
    encoding: FrameEncoding,
    max_extension_bits: u64,
) -> Result<RestorationFilterInventory, InventoryError> {
    if read_bool(reader)? {
        return Ok(RestorationFilterInventory::Default);
    }
    let gab = read_bool(reader)?;
    let gab_custom = gab && read_bool(reader)?;
    let gaborish = if !gab {
        GaborishInventory::Disabled
    } else if !gab_custom {
        GaborishInventory::Default
    } else {
        let mut weights = [[FiniteF16::default(); 2]; 3];
        for channel in &mut weights {
            for weight in channel.iter_mut() {
                *weight = read_f16(reader)?;
            }
            if (1.0 + (channel[0].to_f32() + channel[1].to_f32()) * 4.0).abs() < f32::EPSILON {
                return Err(InventoryError::InvalidFrame(
                    "Gaborish weights produce a near-zero kernel",
                ));
            }
        }
        GaborishInventory::Custom { weights }
    };
    let epf_iters = read_bits(reader, 2)? as u32;
    let sharp_custom = epf_iters > 0 && encoding == FrameEncoding::VarDct && read_bool(reader)?;
    let sharp_lut = if sharp_custom {
        let mut values = [FiniteF16::default(); 8];
        for value in &mut values {
            *value = read_f16(reader)?;
        }
        Some(values)
    } else {
        None
    };
    let weight_custom = epf_iters > 0 && read_bool(reader)?;
    let weights = if weight_custom {
        Some(EpfWeightsInventory {
            channel_scale: [read_f16(reader)?, read_f16(reader)?, read_f16(reader)?],
            pass1_zeroflush: read_f16(reader)?,
            pass2_zeroflush: read_f16(reader)?,
        })
    } else {
        None
    };
    let sigma_custom = epf_iters > 0 && read_bool(reader)?;
    let sigma = if sigma_custom {
        Some(EpfSigmaInventory {
            quant_mul: if encoding == FrameEncoding::VarDct {
                Some(read_f16(reader)?)
            } else {
                None
            },
            pass0_sigma_scale: read_f16(reader)?,
            pass2_sigma_scale: read_f16(reader)?,
            border_sad_mul: read_f16(reader)?,
        })
    } else {
        None
    };
    let sigma_for_modular = if epf_iters > 0 && encoding == FrameEncoding::Modular {
        Some(read_f16(reader)?)
    } else {
        None
    };
    let epf = if epf_iters == 0 {
        EdgePreservingFilterInventory::Disabled
    } else {
        EdgePreservingFilterInventory::Enabled {
            iterations: epf_iters,
            sharp_lut,
            weights,
            sigma,
            sigma_for_modular,
        }
    };
    parse_extensions(reader, max_extension_bits)?;
    Ok(RestorationFilterInventory::Custom { gaborish, epf })
}

fn parse_extensions(
    reader: &mut BitReader<'_>,
    max_extension_bits: u64,
) -> Result<(), InventoryError> {
    let selector = read_u64(reader)?;
    let mut total_bits = 0u64;
    for bit in 0..64 {
        if selector & (1u64 << bit) != 0 {
            total_bits = total_bits
                .checked_add(read_u64(reader)?)
                .ok_or(InventoryError::SizeOverflow)?;
            if total_bits > max_extension_bits {
                return Err(InventoryError::ResourceLimit("extension payload bits"));
            }
        }
    }
    reader
        .skip_bits(total_bits)
        .map_err(|error| map_reader_error(error, reader.bit_offset()))
}

struct ParsedToc {
    permuted: bool,
    entry_lengths_in_bitstream_order: Vec<u32>,
    logical_indices_in_bitstream_order: Vec<usize>,
}

fn parse_toc(
    reader: &mut BitReader<'_>,
    num_entries: usize,
    codestream: &[u8],
) -> Result<ParsedToc, InventoryError> {
    let permuted = read_bool(reader)?;
    let original_to_bitstream = if permuted {
        let entropy_start = reader.bit_offset();
        let entropy_start_usize =
            usize::try_from(entropy_start).map_err(|_| InventoryError::SizeOverflow)?;
        let entry_count = u32::try_from(num_entries)
            .map_err(|_| InventoryError::ResourceLimit("TOC entries per frame"))?;
        let mut bitstream = ImageBitstream::new(codestream);
        if let Err(error) = bitstream.skip_bits(entropy_start_usize) {
            return Err(map_metadata_bitstream_error(
                error,
                bitstream.num_read_bits(),
            ));
        }
        let mut decoder = match jxl_coding::Decoder::parse(&mut bitstream, 8) {
            Ok(decoder) => decoder,
            Err(error) => {
                return Err(map_toc_entropy_error(error, bitstream.num_read_bits()));
            }
        };
        if let Err(error) = decoder.begin(&mut bitstream) {
            return Err(map_toc_entropy_error(error, bitstream.num_read_bits()));
        }
        let permutation =
            match jxl_coding::read_permutation(&mut bitstream, &mut decoder, entry_count, 0) {
                Ok(permutation) => permutation,
                Err(error) => {
                    return Err(map_toc_entropy_error(error, bitstream.num_read_bits()));
                }
            };
        if let Err(error) = decoder.finalize() {
            return Err(map_toc_entropy_error(error, bitstream.num_read_bits()));
        }
        let entropy_end =
            u64::try_from(bitstream.num_read_bits()).map_err(|_| InventoryError::SizeOverflow)?;
        let entropy_bits = entropy_end
            .checked_sub(entropy_start)
            .ok_or(InventoryError::SizeOverflow)?;
        reader
            .skip_bits(entropy_bits)
            .map_err(|error| map_reader_error(error, reader.bit_offset()))?;
        Some(permutation)
    } else {
        None
    };
    zero_pad(reader)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(num_entries)
        .map_err(|_| InventoryError::AllocationFailed("TOC entries"))?;
    for _ in 0..num_entries {
        entries.push(read_u32(
            reader,
            [b(0, 10), b(1_024, 14), b(17_408, 22), b(4_211_712, 30)],
        )?);
    }
    zero_pad(reader)?;

    let logical_indices_in_bitstream_order =
        if let Some(original_to_bitstream) = original_to_bitstream {
            if original_to_bitstream.len() != num_entries {
                return Err(InventoryError::InvalidFrame(
                    "invalid TOC permutation length",
                ));
            }
            let mut bitstream_to_original = vec![usize::MAX; num_entries];
            for (original_index, bitstream_index) in original_to_bitstream.into_iter().enumerate() {
                let slot = bitstream_to_original.get_mut(bitstream_index).ok_or(
                    InventoryError::InvalidFrame("invalid TOC permutation index"),
                )?;
                if *slot != usize::MAX {
                    return Err(InventoryError::InvalidFrame(
                        "duplicate TOC permutation index",
                    ));
                }
                *slot = original_index;
            }
            if bitstream_to_original.contains(&usize::MAX) {
                return Err(InventoryError::InvalidFrame("incomplete TOC permutation"));
            }
            bitstream_to_original
        } else {
            (0..num_entries).collect()
        };
    Ok(ParsedToc {
        permuted,
        entry_lengths_in_bitstream_order: entries,
        logical_indices_in_bitstream_order,
    })
}

fn map_toc_entropy_error(error: jxl_coding::Error, bit_offset: usize) -> InventoryError {
    if error.unexpected_eof() {
        InventoryError::UnexpectedEndOfBits {
            bit_offset: u64::try_from(bit_offset).unwrap_or(u64::MAX),
        }
    } else {
        InventoryError::InvalidFrame("invalid entropy-coded TOC permutation")
    }
}

fn map_metadata_bitstream_error(error: jxl_bitstream::Error, bit_offset: usize) -> InventoryError {
    if error.unexpected_eof() {
        InventoryError::UnexpectedEndOfBits {
            bit_offset: u64::try_from(bit_offset).unwrap_or(u64::MAX),
        }
    } else if matches!(error, jxl_bitstream::Error::NonZeroPadding) {
        InventoryError::NonZeroPadding {
            bit_offset: u64::try_from(bit_offset).unwrap_or(u64::MAX),
        }
    } else {
        InventoryError::InvalidFrame("invalid TOC metadata bitstream")
    }
}

#[derive(Clone, Copy)]
struct FrameCounts {
    groups: u64,
    low_frequency_groups: u64,
    toc_entries: u64,
}

fn compute_frame_counts(header: &ParsedFrameHeader) -> Result<FrameCounts, InventoryError> {
    let lf_divisor = 1u64
        .checked_shl(
            header
                .lf_level
                .checked_mul(3)
                .ok_or(InventoryError::SizeOverflow)?,
        )
        .ok_or(InventoryError::SizeOverflow)?;
    let width = div_ceil(u64::from(header.width), lf_divisor)?;
    let height = div_ceil(u64::from(header.height), lf_divisor)?;
    let width = div_ceil(width, u64::from(header.upsampling))?;
    let height = div_ceil(height, u64::from(header.upsampling))?;
    let group_dim = 1u64
        .checked_shl(
            GROUP_DIM_LOG2_MINUS_ONE
                .checked_add(header.group_size_shift)
                .ok_or(InventoryError::SizeOverflow)?,
        )
        .ok_or(InventoryError::SizeOverflow)?;
    let groups = div_ceil(width, group_dim)?
        .checked_mul(div_ceil(height, group_dim)?)
        .ok_or(InventoryError::SizeOverflow)?;

    let horizontal_block_span = 8u64
        .checked_shl(header.max_horizontal_shift)
        .ok_or(InventoryError::SizeOverflow)?;
    let vertical_block_span = 8u64
        .checked_shl(header.max_vertical_shift)
        .ok_or(InventoryError::SizeOverflow)?;
    let block_width = div_ceil(width, horizontal_block_span)?
        .checked_shl(header.max_horizontal_shift)
        .ok_or(InventoryError::SizeOverflow)?;
    let block_height = div_ceil(height, vertical_block_span)?
        .checked_shl(header.max_vertical_shift)
        .ok_or(InventoryError::SizeOverflow)?;
    let low_frequency_groups = div_ceil(block_width, group_dim)?
        .checked_mul(div_ceil(block_height, group_dim)?)
        .ok_or(InventoryError::SizeOverflow)?;
    let toc_entries = if groups == 1 && header.num_passes == 1 {
        1
    } else {
        groups
            .checked_mul(u64::from(header.num_passes))
            .and_then(|value| value.checked_add(low_frequency_groups))
            .and_then(|value| value.checked_add(2))
            .ok_or(InventoryError::SizeOverflow)?
    };
    Ok(FrameCounts {
        groups,
        low_frequency_groups,
        toc_entries,
    })
}

fn section_kind(index: u64, counts: FrameCounts, num_passes: u32) -> FrameSectionKind {
    if counts.toc_entries == 1 {
        return FrameSectionKind::Single;
    }
    if index == 0 {
        return FrameSectionKind::LowFrequencyGlobal;
    }
    let relative = index - 1;
    if relative < counts.low_frequency_groups {
        return FrameSectionKind::LowFrequencyGroup {
            group_index: relative,
        };
    }
    let relative = relative - counts.low_frequency_groups;
    if relative == 0 {
        return FrameSectionKind::HighFrequencyGlobal;
    }
    let relative = relative - 1;
    let pass_index = relative / counts.groups;
    debug_assert!(pass_index < u64::from(num_passes));
    FrameSectionKind::PassGroup {
        pass_index: pass_index as u32,
        group_index: relative % counts.groups,
    }
}

fn read_name(reader: &mut BitReader<'_>, max_name_bytes: usize) -> Result<Vec<u8>, InventoryError> {
    let length = read_u32(reader, [c(0), b(0, 4), b(16, 5), b(48, 10)])?;
    let length = usize::try_from(length).map_err(|_| InventoryError::SizeOverflow)?;
    if length > max_name_bytes {
        return Err(InventoryError::ResourceLimit("frame name bytes"));
    }
    let mut name = Vec::new();
    name.try_reserve_exact(length)
        .map_err(|_| InventoryError::AllocationFailed("frame name"))?;
    for _ in 0..length {
        name.push(read_bits(reader, 8)? as u8);
    }
    Ok(name)
}

fn read_f16(reader: &mut BitReader<'_>) -> Result<FiniteF16, InventoryError> {
    let bits = read_bits(reader, 16)? as u16;
    FiniteF16::from_bits(bits).ok_or(InventoryError::InvalidFrame("non-finite F16 header value"))
}

#[derive(Clone, Copy)]
enum U32Part {
    Constant(u32),
    Bits { offset: u32, count: u8 },
}

const fn c(value: u32) -> U32Part {
    U32Part::Constant(value)
}

const fn b(offset: u32, count: u8) -> U32Part {
    U32Part::Bits { offset, count }
}

fn frame_dimension_coder() -> [U32Part; 4] {
    [b(0, 8), b(256, 11), b(2_304, 14), b(18_688, 30)]
}

fn read_u32(reader: &mut BitReader<'_>, parts: [U32Part; 4]) -> Result<u32, InventoryError> {
    let selector = read_bits(reader, 2)? as usize;
    match parts[selector] {
        U32Part::Constant(value) => Ok(value),
        U32Part::Bits { offset, count } => {
            let value = u32::try_from(read_bits(reader, count)?)
                .map_err(|_| InventoryError::SizeOverflow)?;
            offset
                .checked_add(value)
                .ok_or(InventoryError::SizeOverflow)
        }
    }
}

fn read_u64(reader: &mut BitReader<'_>) -> Result<u64, InventoryError> {
    match read_bits(reader, 2)? {
        0 => Ok(0),
        1 => Ok(1 + read_bits(reader, 4)?),
        2 => Ok(17 + read_bits(reader, 8)?),
        _ => {
            let mut value = read_bits(reader, 12)?;
            let mut shift = 12u32;
            while read_bool(reader)? {
                if shift == 60 {
                    value |= read_bits(reader, 4)? << shift;
                    break;
                }
                value |= read_bits(reader, 8)? << shift;
                shift = shift.checked_add(8).ok_or(InventoryError::SizeOverflow)?;
            }
            Ok(value)
        }
    }
}

fn read_bool(reader: &mut BitReader<'_>) -> Result<bool, InventoryError> {
    Ok(read_bits(reader, 1)? != 0)
}

fn read_bits(reader: &mut BitReader<'_>, count: u8) -> Result<u64, InventoryError> {
    reader
        .read_bits(count)
        .map_err(|error| map_reader_error(error, reader.bit_offset()))
}

fn zero_pad(reader: &mut BitReader<'_>) -> Result<(), InventoryError> {
    let offset = reader.bit_offset();
    reader
        .zero_pad_to_byte()
        .map_err(|error| map_reader_error(error, offset))
}

fn map_reader_error(error: BitReaderError, bit_offset: u64) -> InventoryError {
    match error {
        BitReaderError::UnexpectedEndOfBits => InventoryError::UnexpectedEndOfBits { bit_offset },
        BitReaderError::NonZeroPadding => InventoryError::NonZeroPadding { bit_offset },
        BitReaderError::SizeOverflow | BitReaderError::InvalidBitCount(_) => {
            InventoryError::SizeOverflow
        }
        _ => InventoryError::InvalidFrame("bit reader failure"),
    }
}

fn unpack_signed(value: u32) -> i32 {
    let bit = value & 1;
    let base = value >> 1;
    let flip = 0u32.wrapping_sub(bit);
    (base ^ flip) as i32
}

fn div_ceil(value: u64, divisor: u64) -> Result<u64, InventoryError> {
    if divisor == 0 {
        return Err(InventoryError::InvalidFrame("zero frame divisor"));
    }
    Ok(value.div_ceil(divisor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BitWriter, FragmentedContainerWriter, ParseLimits, parse, write_container};
    use sha2::{Digest, Sha256};

    fn inventory(input: &[u8]) -> CodestreamInventory {
        parse(input, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap()
    }

    #[test]
    fn libjxl_basic_and_upsampling_headers_match_the_oracle() {
        let basic = inventory(&crate::test_fixtures::basic());
        assert_eq!(basic.codestream_bytes, 65);
        assert_eq!(
            basic.image_header.bit_range,
            BitRange {
                offset: 0,
                length: 33
            }
        );
        assert_eq!(
            (basic.image_header.width, basic.image_header.height),
            (1, 1)
        );
        assert_eq!(basic.image_header.orientation, 1);
        assert_eq!(
            basic.image_header.bit_depth,
            SampleBitDepth::Integer { bits_per_sample: 8 }
        );
        assert_eq!(
            basic.image_header.colour_encoding,
            ColourEncodingInventory::Enumerated {
                colour_space: ColourSpaceInventory::Rgb,
                white_point: WhitePointInventory::D65,
                primaries: PrimariesInventory::Srgb,
                transfer_function: TransferFunctionInventory::Srgb,
                rendering_intent: RenderingIntentInventory::Relative,
            }
        );
        assert_eq!(
            basic.image_header.tone_mapping.intensity_target.to_f32(),
            255.0
        );
        assert_eq!(basic.image_header.tone_mapping.min_nits.to_f32(), 0.0);
        assert!(basic.image_header.extra_channels.is_empty());
        assert!(basic.image_header.xyb_encoded);
        let opsin = basic
            .image_header
            .opsin_inverse_matrix
            .expect("XYB fixture retains its resolved opsin matrix");
        assert!((11.0..12.0).contains(&opsin.inverse_matrix[0][0].to_f32()));
        assert_eq!(basic.image_header.upsampling_weights.up2.len(), 15);
        assert_eq!(basic.image_header.upsampling_weights.up4.len(), 55);
        assert_eq!(basic.image_header.upsampling_weights.up8.len(), 210);
        assert_eq!(basic.frames.len(), 1);
        let frame = &basic.frames[0];
        assert_eq!(
            frame.header_bits,
            BitRange {
                offset: 40,
                length: 34
            }
        );
        assert_eq!(
            frame.toc_bits,
            BitRange {
                offset: 74,
                length: 22
            }
        );
        assert_eq!(frame.frame_type, FrameType::Regular);
        assert_eq!(frame.encoding, FrameEncoding::VarDct);
        assert_eq!((frame.x_qm_scale, frame.b_qm_scale), (2, 2));
        assert_eq!(frame.extra_channel_upsampling, Vec::<u32>::new());
        assert_eq!(frame.num_passes, 1);
        assert_eq!(frame.progressive_passes, FramePassesInventory::default());
        assert_eq!((frame.width, frame.height), (1, 1));
        assert_eq!((frame.group_count, frame.low_frequency_group_count), (1, 1));
        assert_eq!(
            frame.sections,
            vec![FrameSection {
                bitstream_index: 0,
                toc_index: 0,
                kind: FrameSectionKind::Single,
                bytes: ByteRange {
                    offset: 12,
                    length: 53
                },
                bits: BitRange {
                    offset: 96,
                    length: 424
                },
            }]
        );

        let odd = inventory(&crate::test_fixtures::oddsize_ups());
        assert_eq!(
            (odd.image_header.width, odd.image_header.height),
            (257, 257)
        );
        assert_eq!(odd.frames.len(), 1);
        assert_eq!(odd.frames[0].upsampling, 2);
        assert_eq!((odd.frames[0].width, odd.frames[0].height), (257, 257));
        assert_eq!(
            odd.frames[0].sections[0].bytes,
            ByteRange {
                offset: 12,
                length: 8_579,
            }
        );
    }

    #[test]
    fn libjxl_vardct_multigroup_toc_has_exact_physical_ranges() {
        let inventory = inventory(&crate::test_fixtures::green_queen_vardct());
        assert_eq!(inventory.codestream_bytes, 88_995);
        assert_eq!(inventory.image_header.bit_range.length, 48);
        assert_eq!(
            (inventory.image_header.width, inventory.image_header.height),
            (438, 589)
        );
        let frame = &inventory.frames[0];
        assert_eq!(
            frame.header_bits,
            BitRange {
                offset: 48,
                length: 33
            }
        );
        assert_eq!(
            frame.toc_bits,
            BitRange {
                offset: 81,
                length: 167
            }
        );
        assert_eq!((frame.group_count, frame.low_frequency_group_count), (6, 1));
        assert_eq!(frame.sections.len(), 9);
        assert_eq!(frame.sections[0].kind, FrameSectionKind::LowFrequencyGlobal);
        assert_eq!(
            frame.sections[1].kind,
            FrameSectionKind::LowFrequencyGroup { group_index: 0 }
        );
        assert_eq!(
            frame.sections[2].kind,
            FrameSectionKind::HighFrequencyGlobal
        );
        assert_eq!(
            frame.sections[8].kind,
            FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index: 5,
            }
        );
        assert_eq!(
            frame
                .sections
                .iter()
                .map(|section| section.bytes.length)
                .collect::<Vec<_>>(),
            vec![
                230, 8_146, 843, 18_405, 15_774, 17_555, 18_396, 4_768, 4_847
            ]
        );
        assert_eq!(frame.sections[0].bytes.offset, 31);
        assert_eq!(frame.sections.last().unwrap().bytes.end(), Some(88_995));
    }

    #[test]
    fn entropy_coded_toc_permutation_maps_physical_sections_to_logical_kinds() {
        let inventory = inventory(&crate::test_fixtures::has_permutation());
        assert_eq!(inventory.codestream_bytes, 2_295);
        assert_eq!(inventory.frames.len(), 1);
        let frame = &inventory.frames[0];
        assert!(frame.toc_permuted);
        assert_eq!(frame.sections.len(), 49);

        let original_to_bitstream = [
            0usize, 1, 42, 48, 2, 3, 4, 5, 6, 7, 8, 9, 43, 10, 11, 12, 13, 14, 15, 16, 17, 44, 18,
            19, 20, 21, 22, 23, 24, 25, 45, 26, 27, 28, 29, 30, 31, 32, 33, 46, 34, 35, 36, 37, 38,
            39, 40, 41, 47,
        ];
        let lengths_in_bitstream_order = [
            155u64, 992, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 697, 5, 5, 5, 5, 5, 60,
        ];
        let mut bitstream_to_original = [usize::MAX; 49];
        for (original, &bitstream) in original_to_bitstream.iter().enumerate() {
            bitstream_to_original[bitstream] = original;
        }
        assert_eq!(
            frame
                .sections
                .iter()
                .map(|section| usize::try_from(section.toc_index).unwrap())
                .collect::<Vec<_>>(),
            bitstream_to_original
        );
        assert_eq!(
            frame
                .sections
                .iter()
                .map(|section| section.bytes.length)
                .collect::<Vec<_>>(),
            lengths_in_bitstream_order
        );
        assert_eq!(
            frame
                .sections
                .iter()
                .map(|section| section.bitstream_index)
                .collect::<Vec<_>>(),
            (0..49).collect::<Vec<_>>()
        );
        assert_eq!(frame.sections.last().unwrap().bytes.end(), Some(2_295));
    }

    #[test]
    fn embedded_icc_is_bounded_reconstructed_and_frame_aligned() {
        let bytes = crate::test_fixtures::with_icc();
        let inventory = inventory(&bytes);
        assert_eq!(inventory.codestream_bytes, 358);
        assert_eq!(
            inventory.image_header.colour_encoding,
            ColourEncodingInventory::IccProfile {
                colour_space: ColourSpaceInventory::Grey,
            }
        );
        let icc = inventory
            .image_header
            .embedded_icc
            .as_ref()
            .expect("fixture declares an embedded ICC profile");
        assert!(icc.bit_range.length > 0);
        assert!(icc.encoded_byte_count > 0);
        assert_eq!(icc.profile.len(), 544);
        assert_eq!(&icc.profile[36..40], b"acsp");
        let profile_hash = Sha256::digest(&icc.profile);
        assert_eq!(
            profile_hash[..],
            [
                0x92, 0xaa, 0xee, 0x40, 0x52, 0x1d, 0x76, 0x71, 0xdb, 0xc9, 0x01, 0xdf, 0xc8, 0x8c,
                0xc3, 0x6c, 0xb8, 0x3d, 0x41, 0x76, 0x6b, 0x2c, 0x46, 0x6f, 0xf9, 0x72, 0x68, 0xee,
                0x5a, 0xb0, 0xc5, 0x77,
            ]
        );
        assert_eq!(
            icc.bit_range.end().unwrap().div_ceil(8) * 8,
            inventory.frames[0].header_bits.offset
        );

        let parsed = parse(&bytes, ParseLimits::default()).unwrap();
        let encoded_limit = InventoryLimits {
            max_encoded_icc_bytes: icc.encoded_byte_count - 1,
            ..InventoryLimits::default()
        };
        assert_eq!(
            parsed.codestream_inventory(encoded_limit).unwrap_err(),
            InventoryError::ResourceLimit("encoded ICC profile bytes")
        );
        let decoded_limit = InventoryLimits {
            max_decoded_icc_bytes: u64::try_from(icc.profile.len()).unwrap() - 1,
            ..InventoryLimits::default()
        };
        assert_eq!(
            parsed.codestream_inventory(decoded_limit).unwrap_err(),
            InventoryError::ResourceLimit("decoded ICC profile bytes")
        );
    }

    #[test]
    fn blending_inventory_preserves_alpha_clamp_and_main_canvas_source_rule() {
        let mut color_bits = BitWriter::new();
        color_bits.write_bits(2, 2).unwrap(); // Blend mode.
        color_bits.write_bits(2, 2).unwrap(); // Alpha extra channel 2.
        color_bits.write_bits(1, 1).unwrap(); // Clamp.
        color_bits.write_bits(3, 2).unwrap(); // Reference source 3.
        let mut color_reader = BitReader::new(color_bits.as_bytes());
        assert_eq!(
            parse_blending_info(&mut color_reader, 3, false, None).unwrap(),
            FrameBlendInfo {
                mode: FrameBlendMode::Blend,
                alpha_channel: Some(2),
                clamp: true,
                source: 3,
            }
        );
        assert_eq!(color_reader.bit_offset(), 7);

        // An extra channel inherits the color blend mode for the source-field condition. A full
        // Replace color frame resets the canvas, so the Add extra channel has no source bits.
        let mut reset_extra_bits = BitWriter::new();
        reset_extra_bits.write_bits(1, 2).unwrap(); // Add mode.
        let mut reset_extra_reader = BitReader::new(reset_extra_bits.as_bytes());
        assert_eq!(
            parse_blending_info(
                &mut reset_extra_reader,
                1,
                true,
                Some(FrameBlendMode::Replace),
            )
            .unwrap(),
            FrameBlendInfo {
                mode: FrameBlendMode::Add,
                alpha_channel: None,
                clamp: false,
                source: 0,
            }
        );
        assert_eq!(reset_extra_reader.bit_offset(), 2);

        // Conversely, a non-resetting color frame requires the source even when this extra
        // channel's own operation is Replace.
        let mut sourced_extra_bits = BitWriter::new();
        sourced_extra_bits.write_bits(0, 2).unwrap(); // Replace mode.
        sourced_extra_bits.write_bits(2, 2).unwrap(); // Reference source 2.
        let mut sourced_extra_reader = BitReader::new(sourced_extra_bits.as_bytes());
        assert_eq!(
            parse_blending_info(
                &mut sourced_extra_reader,
                1,
                true,
                Some(FrameBlendMode::Add),
            )
            .unwrap()
            .source,
            2
        );
        assert_eq!(sourced_extra_reader.bit_offset(), 4);
    }

    #[test]
    fn progressive_pass_inventory_preserves_the_complete_schedule() {
        let mut bits = BitWriter::new();
        bits.write_bits(3, 2).unwrap(); // Extended num_passes selector.
        bits.write_bits(2, 3).unwrap(); // 4 + 2 = 6 passes.
        bits.write_bits(2, 2).unwrap(); // Two downsampling boundaries.
        for shift in [3, 2, 2, 1, 0] {
            bits.write_bits(shift, 2).unwrap();
        }
        bits.write_bits(3, 2).unwrap(); // Downsampling 8.
        bits.write_bits(1, 2).unwrap(); // Downsampling 2.
        bits.write_bits(1, 2).unwrap(); // Last pass 1.
        bits.write_bits(3, 2).unwrap(); // Extended last-pass selector.
        bits.write_bits(4, 3).unwrap(); // Last pass 4.

        let mut reader = BitReader::new(bits.as_bytes());
        assert_eq!(
            parse_passes(&mut reader).unwrap(),
            FramePassesInventory {
                shifts: vec![3, 2, 2, 1, 0],
                downsampling: vec![8, 2],
                last_pass: vec![1, 4],
            }
        );
        assert_eq!(reader.bit_offset(), 28);
    }

    #[test]
    fn finite_f16_expands_normals_subnormals_and_signed_zero_exactly() {
        assert_eq!(FiniteF16::from_bits(0x0000).unwrap().to_f32().to_bits(), 0);
        assert_eq!(
            FiniteF16::from_bits(0x8000).unwrap().to_f32().to_bits(),
            0x8000_0000
        );
        assert_eq!(
            FiniteF16::from_bits(0x0001).unwrap().to_f32().to_bits(),
            0x3380_0000
        );
        assert_eq!(
            FiniteF16::from_bits(0x3c00).unwrap().to_f32().to_bits(),
            1.0_f32.to_bits()
        );
        assert_eq!(
            FiniteF16::from_bits(0x7bff).unwrap().to_f32().to_bits(),
            0x477f_e000
        );
        assert!(FiniteF16::from_bits(0x7c00).is_none());
        assert!(FiniteF16::from_bits(0x7e00).is_none());
    }

    #[test]
    fn restoration_inventory_retains_custom_vardct_and_modular_values() {
        let one = 0x3c00_u64;
        let half = 0x3800_u64;
        let quarter = 0x3400_u64;
        let two = 0x4000_u64;

        let mut vardct_bits = BitWriter::new();
        vardct_bits.write_bits(0, 1).unwrap(); // Non-default restoration bundle.
        vardct_bits.write_bits(1, 1).unwrap(); // Gaborish enabled.
        vardct_bits.write_bits(1, 1).unwrap(); // Custom Gaborish weights.
        for value in [one, half, one, half, one, half] {
            vardct_bits.write_bits(value, 16).unwrap();
        }
        vardct_bits.write_bits(3, 2).unwrap(); // Three EPF iterations.
        vardct_bits.write_bits(1, 1).unwrap(); // Custom sharpness LUT.
        for value in [0, quarter, half, one, 0, quarter, half, one] {
            vardct_bits.write_bits(value, 16).unwrap();
        }
        vardct_bits.write_bits(1, 1).unwrap(); // Custom channel weights.
        for value in [one, half, quarter, half, quarter] {
            vardct_bits.write_bits(value, 16).unwrap();
        }
        vardct_bits.write_bits(1, 1).unwrap(); // Custom sigma values.
        for value in [one, half, quarter, two] {
            vardct_bits.write_bits(value, 16).unwrap();
        }
        vardct_bits.write_bits(0, 2).unwrap(); // No restoration extensions.

        let mut vardct_reader = BitReader::new(vardct_bits.as_bytes());
        let vardct =
            parse_restoration_filter(&mut vardct_reader, FrameEncoding::VarDct, u64::MAX).unwrap();
        let RestorationFilterInventory::Custom { gaborish, epf } = vardct else {
            panic!("custom restoration bundle was not retained");
        };
        assert!(matches!(gaborish, GaborishInventory::Custom { .. }));
        let EdgePreservingFilterInventory::Enabled {
            iterations,
            sharp_lut,
            weights,
            sigma,
            sigma_for_modular,
        } = epf
        else {
            panic!("custom EPF bundle was not retained");
        };
        assert_eq!(iterations, 3);
        assert_eq!(sharp_lut.unwrap()[3].to_bits(), one as u16);
        assert_eq!(weights.unwrap().channel_scale[2].to_bits(), quarter as u16);
        assert_eq!(sigma.unwrap().border_sad_mul.to_bits(), two as u16);
        assert_eq!(sigma_for_modular, None);

        let mut modular_bits = BitWriter::new();
        modular_bits.write_bits(0, 1).unwrap(); // Non-default restoration bundle.
        modular_bits.write_bits(0, 1).unwrap(); // Gaborish disabled.
        modular_bits.write_bits(1, 2).unwrap(); // One EPF iteration.
        modular_bits.write_bits(0, 1).unwrap(); // Default channel weights.
        modular_bits.write_bits(0, 1).unwrap(); // Default sigma values.
        modular_bits.write_bits(one, 16).unwrap(); // Modular sigma.
        modular_bits.write_bits(0, 2).unwrap(); // No restoration extensions.
        let mut modular_reader = BitReader::new(modular_bits.as_bytes());
        let RestorationFilterInventory::Custom { gaborish, epf } =
            parse_restoration_filter(&mut modular_reader, FrameEncoding::Modular, u64::MAX)
                .unwrap()
        else {
            panic!("custom Modular restoration bundle was not retained");
        };
        assert_eq!(gaborish, GaborishInventory::Disabled);
        assert!(matches!(
            epf,
            EdgePreservingFilterInventory::Enabled {
                iterations: 1,
                sigma_for_modular: Some(value),
                ..
            } if value.to_bits() == one as u16
        ));
    }

    #[test]
    fn libjxl_animation_and_fragmented_animation_metadata_match_the_oracle() {
        let spline = inventory(&crate::test_fixtures::animation_spline());
        assert_eq!(
            (spline.image_header.width, spline.image_header.height),
            (320, 320)
        );
        assert_eq!(spline.image_header.bit_range.length, 58);
        assert_eq!(
            spline.image_header.animation,
            Some(AnimationInventory {
                ticks_per_second_numerator: 100,
                ticks_per_second_denominator: 1,
                num_loops: 0,
                have_timecodes: false,
            })
        );
        assert_eq!(spline.frames.len(), 60);
        assert!(spline.frames[..59].iter().all(|frame| !frame.is_last));
        assert!(spline.frames[59].is_last);
        assert!(spline.frames.iter().all(|frame| frame.duration_ticks == 2));
        assert_eq!(
            spline.frames[0].header_bits,
            BitRange {
                offset: 64,
                length: 50
            }
        );
        assert_eq!(
            spline.frames[0].toc_bits,
            BitRange {
                offset: 114,
                length: 94
            }
        );
        assert_eq!(
            spline.frames[59].header_bits,
            BitRange {
                offset: 75_368,
                length: 48
            }
        );
        assert_eq!(
            spline.frames[59].toc_bits,
            BitRange {
                offset: 75_416,
                length: 96
            }
        );
        assert_eq!(
            spline.frames[59].sections.last().unwrap().bytes.end(),
            Some(9_581)
        );

        let fragmented_input = crate::test_fixtures::fragmented_animation();
        let parsed = parse(&fragmented_input, ParseLimits::default()).unwrap();
        assert!(parsed.is_container());
        let fragmented = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert_eq!(fragmented.codestream_bytes, 13_471);
        assert_eq!(
            (
                fragmented.image_header.width,
                fragmented.image_header.height
            ),
            (256, 256)
        );
        assert_eq!(fragmented.image_header.bit_range.length, 69);
        assert_eq!(fragmented.image_header.extra_channel_count, 1);
        assert_eq!(fragmented.image_header.extra_channels.len(), 1);
        assert_eq!(
            fragmented.image_header.extra_channels[0],
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::Alpha { associated: false },
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 8 },
                dimension_shift: 0,
                name_bytes: Vec::new(),
            }
        );
        assert!(!fragmented.image_header.xyb_encoded);
        assert_eq!(fragmented.image_header.opsin_inverse_matrix, None);
        assert_eq!(
            fragmented.image_header.animation,
            Some(AnimationInventory {
                ticks_per_second_numerator: 1_000,
                ticks_per_second_denominator: 1,
                num_loops: 0,
                have_timecodes: false,
            })
        );
        assert_eq!(fragmented.frames.len(), 5);
        assert_eq!(fragmented.frames[0].duration_ticks, 500);
        assert_eq!(
            fragmented.frames[0].header_bits,
            BitRange {
                offset: 72,
                length: 67
            }
        );
        assert_eq!(
            fragmented.frames[0].toc_bits,
            BitRange {
                offset: 139,
                length: 21
            }
        );
        assert_eq!(
            fragmented.frames[0].sections[0].bytes,
            ByteRange {
                offset: 20,
                length: 3_248,
            }
        );
        assert_eq!(fragmented.frames[4].header_bits.offset, 84_632);
        assert!(fragmented.frames[4].is_last);
        assert_eq!(fragmented.frames[4].sections[0].bytes.end(), Some(13_471));

        let raw = inventory(parsed.codestream());
        assert_eq!(fragmented, raw);
    }

    #[test]
    fn raw_jxlc_and_reconstructed_jxlp_share_one_inventory() {
        let raw = crate::test_fixtures::basic();
        let expected = inventory(&raw);
        let container = write_container(&raw).unwrap();
        assert_eq!(inventory(&container), expected);

        let mut writer = FragmentedContainerWriter::new();
        writer.push_fragment(&raw[..1], false).unwrap();
        writer.push_fragment(&raw[1..17], false).unwrap();
        writer.push_fragment(&raw[17..], true).unwrap();
        let fragmented = writer.finish().unwrap();
        assert_eq!(inventory(&fragmented), expected);
    }

    #[test]
    fn every_fixture_prefix_is_rejected_or_yields_the_complete_codestream_inventory() {
        let fixtures = [
            crate::test_fixtures::basic(),
            crate::test_fixtures::oddsize_ups(),
            crate::test_fixtures::green_queen_vardct(),
            crate::test_fixtures::animation_spline(),
            crate::test_fixtures::fragmented_animation(),
            crate::test_fixtures::has_permutation(),
            crate::test_fixtures::with_icc(),
        ];
        for fixture in fixtures {
            let expected = inventory(&fixture);
            for prefix_len in 0..fixture.len() {
                if let Ok(parsed) = parse(&fixture[..prefix_len], ParseLimits::default())
                    && let Ok(actual) = parsed.codestream_inventory(InventoryLimits::default())
                {
                    // A container may finish its codestream before optional trailing boxes.
                    // Such a prefix is itself a valid container, but it must inventory the
                    // exact complete codestream rather than a truncated frame sequence.
                    assert_eq!(
                        actual,
                        expected,
                        "prefix {prefix_len}/{} accepted a partial codestream",
                        fixture.len(),
                    );
                }
            }
            let parsed = parse(&fixture, ParseLimits::default()).unwrap();
            parsed
                .codestream_inventory(InventoryLimits::default())
                .unwrap();
        }
    }

    #[test]
    fn inventory_limits_fail_before_unbounded_retention_or_section_skip() {
        let animation_bytes = crate::test_fixtures::animation_spline();
        let animation = parse(&animation_bytes, ParseLimits::default()).unwrap();
        let limits = InventoryLimits {
            max_frames: 59,
            ..InventoryLimits::default()
        };
        assert_eq!(
            animation.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("frame count")
        );

        let green_bytes = crate::test_fixtures::green_queen_vardct();
        let green = parse(&green_bytes, ParseLimits::default()).unwrap();
        let limits = InventoryLimits {
            max_toc_entries_per_frame: 8,
            ..InventoryLimits::default()
        };
        assert_eq!(
            green.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("TOC entries per frame")
        );
        let limits = InventoryLimits {
            max_total_toc_entries: 8,
            ..InventoryLimits::default()
        };
        assert_eq!(
            green.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("total TOC entries")
        );

        let basic_bytes = crate::test_fixtures::basic();
        let basic = parse(&basic_bytes, ParseLimits::default()).unwrap();
        let limits = InventoryLimits {
            max_image_header_bytes: 2,
            ..InventoryLimits::default()
        };
        assert_eq!(
            basic.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("image header bytes")
        );
        let limits = InventoryLimits {
            max_frame_header_bits: 33,
            ..InventoryLimits::default()
        };
        assert_eq!(
            basic.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("frame header bits")
        );
        let limits = InventoryLimits {
            max_total_section_bytes: 52,
            ..InventoryLimits::default()
        };
        assert_eq!(
            basic.codestream_inventory(limits).unwrap_err(),
            InventoryError::ResourceLimit("total frame section bytes")
        );
    }

    #[test]
    fn toc_padding_and_declared_ranges_are_validated() {
        let basic = crate::test_fixtures::basic();

        let mut non_zero_padding = basic.clone();
        non_zero_padding[9] |= 1 << 3; // Codestream bit 75, after the non-permuted TOC flag.
        assert!(matches!(
            inventory_error(&non_zero_padding),
            InventoryError::NonZeroPadding { .. }
        ));

        let mut oversized = basic;
        for bit in 82..92 {
            oversized[bit / 8] |= 1 << (bit % 8);
        }
        assert!(matches!(
            inventory_error(&oversized),
            InventoryError::UnexpectedEndOfBits { .. }
        ));
    }

    fn inventory_error(codestream: &[u8]) -> InventoryError {
        parse(codestream, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap_err()
    }
}
