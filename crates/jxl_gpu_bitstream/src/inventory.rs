//! Bounded standard-codestream header and TOC inventory.
//!
//! Image-header grammar is delegated to the header-only `jxl-image` crate. Frame headers and
//! non-permuted TOCs are parsed locally because depending on a complete frame decoder would pull
//! CPU entropy, Modular, and VarDCT implementations into the production graph.

use jxl_bitstream::Bitstream as ImageBitstream;
use jxl_image::{BitDepth as JxlBitDepth, ImageHeader};
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

/// Animation timing metadata from the JPEG XL image header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimationInventory {
    pub ticks_per_second_numerator: u32,
    pub ticks_per_second_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
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
    pub xyb_encoded: bool,
    pub grayscale: bool,
    pub has_embedded_icc_profile: bool,
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

/// Logical meaning of a physical TOC entry when the TOC is not permuted.
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
    pub frame_type: FrameType,
    pub encoding: FrameEncoding,
    pub flags: u64,
    pub do_ycbcr: bool,
    pub jpeg_upsampling: [u32; 3],
    pub upsampling: u32,
    pub group_size_shift: u32,
    pub width: u32,
    pub height: u32,
    pub x0: i32,
    pub y0: i32,
    pub have_crop: bool,
    pub num_passes: u32,
    pub duration_ticks: u32,
    pub timecode: Option<u32>,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_color_transform: bool,
    pub name_bytes: Vec<u8>,
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

/// Grammar that deliberately remains outside the no-entropy production boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsupportedCodestreamGrammar {
    /// Locating the first frame requires decoding the embedded ICC entropy stream.
    EmbeddedIccProfile,
    /// Mapping and even locating TOC sizes requires decoding an entropy-coded permutation.
    EntropyCodedTocPermutation,
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
    #[error("unsupported standard codestream grammar: {0:?}")]
    UnsupportedGrammar(UnsupportedCodestreamGrammar),
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
    header_bits: u64,
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
    if parsed_image.inventory.has_embedded_icc_profile {
        return Err(InventoryError::UnsupportedGrammar(
            UnsupportedCodestreamGrammar::EmbeddedIccProfile,
        ));
    }

    let mut reader = BitReader::new(codestream);
    reader
        .skip_bits(parsed_image.header_bits)
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
        let entry_lengths = parse_toc(&mut reader, toc_entries)?;
        let toc_end = reader.bit_offset();
        let section_start_byte = toc_end / 8;
        let mut section_cursor = section_start_byte;
        let mut sections = Vec::new();
        sections
            .try_reserve_exact(toc_entries)
            .map_err(|_| InventoryError::AllocationFailed("frame sections"))?;

        for (toc_index, length) in entry_lengths.into_iter().enumerate() {
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
            frame_type: header.frame_type,
            encoding: header.encoding,
            flags: header.flags,
            do_ycbcr: header.do_ycbcr,
            jpeg_upsampling: header.jpeg_upsampling,
            upsampling: header.upsampling,
            group_size_shift: header.group_size_shift,
            width: header.width,
            height: header.height,
            x0: header.x0,
            y0: header.y0,
            have_crop: header.have_crop,
            num_passes: header.num_passes,
            duration_ticks: header.duration,
            timecode: header.timecode,
            is_last,
            save_as_reference: header.save_as_reference,
            save_before_color_transform: header.save_before_color_transform,
            name_bytes: header.name_bytes,
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
    let bit_depth = match metadata.bit_depth {
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
        xyb_encoded: metadata.xyb_encoded,
        grayscale: metadata.grayscale(),
        has_embedded_icc_profile: metadata.colour_encoding.want_icc(),
        animation,
    };
    Ok(ParsedImageHeader {
        inventory,
        header_bits,
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
    group_size_shift: u32,
    num_passes: u32,
    lf_level: u32,
    have_crop: bool,
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
    let all_default = read_bool(reader)?;
    if all_default {
        return Ok(ParsedFrameHeader {
            frame_type: FrameType::Regular,
            encoding: FrameEncoding::VarDct,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            group_size_shift: 1,
            num_passes: 1,
            lf_level: 0,
            have_crop: false,
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
    if !has_lf_frame {
        for _ in 0..context.num_extra_channels {
            let _ = read_u32(reader, [c(1), c(2), c(4), c(8)])?;
        }
    }
    let group_size_shift = if encoding == FrameEncoding::Modular {
        read_bits(reader, 2)? as u32
    } else {
        1
    };
    if encoding == FrameEncoding::VarDct && context.xyb_encoded {
        let _ = read_bits(reader, 3)?;
        let _ = read_bits(reader, 3)?;
    }
    let num_passes = if frame_type == FrameType::ReferenceOnly {
        1
    } else {
        parse_passes(reader)?
    };
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
    let blending_mode = if normal_frame {
        parse_blending_info(reader, context.num_extra_channels, full_frame)?
    } else {
        0
    };
    let mut replace_all = blending_mode == 0;
    if normal_frame {
        for _ in 0..context.num_extra_channels {
            replace_all &=
                parse_blending_info(reader, context.num_extra_channels, full_frame)? == 0;
        }
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
    let save_before_default_false =
        can_be_referenced && blending_mode == 0 && full_frame && normal_frame;
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
    parse_restoration_filter(reader, encoding, limits.max_extension_bits)?;
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
        group_size_shift,
        num_passes,
        lf_level,
        have_crop,
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
        max_horizontal_shift,
        max_vertical_shift,
    })
}

fn parse_passes(reader: &mut BitReader<'_>) -> Result<u32, InventoryError> {
    let num_passes = read_u32(reader, [c(1), c(2), c(3), b(4, 3)])?;
    if num_passes == 1 {
        return Ok(1);
    }
    let num_downsampling = read_u32(reader, [c(0), c(1), c(2), b(3, 1)])?;
    for _ in 0..num_passes - 1 {
        let _ = read_bits(reader, 2)?;
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
    Ok(num_passes)
}

fn parse_blending_info(
    reader: &mut BitReader<'_>,
    num_extra_channels: u32,
    full_frame: bool,
) -> Result<u32, InventoryError> {
    let mode = read_u32(reader, [c(0), c(1), c(2), b(3, 2)])?;
    if mode > 4 {
        return Err(InventoryError::InvalidEnum {
            name: "BlendingMode",
            value: mode,
        });
    }
    let uses_alpha = matches!(mode, 2 | 3);
    if num_extra_channels > 0 && uses_alpha {
        let alpha_channel = read_u32(reader, [c(0), c(1), c(2), b(3, 3)])?;
        if alpha_channel >= num_extra_channels {
            return Err(InventoryError::InvalidFrame(
                "invalid blending alpha channel",
            ));
        }
    }
    if (num_extra_channels > 0 && uses_alpha) || mode == 4 {
        let _ = read_bool(reader)?;
    }
    if !(full_frame && mode == 0) {
        let _ = read_u32(reader, [c(0), c(1), c(2), c(3)])?;
    }
    Ok(mode)
}

fn parse_restoration_filter(
    reader: &mut BitReader<'_>,
    encoding: FrameEncoding,
    max_extension_bits: u64,
) -> Result<(), InventoryError> {
    if read_bool(reader)? {
        return Ok(());
    }
    let gab = read_bool(reader)?;
    let gab_custom = gab && read_bool(reader)?;
    if gab_custom {
        for _ in 0..6 {
            read_f16(reader)?;
        }
    }
    let epf_iters = read_bits(reader, 2)? as u32;
    let sharp_custom = epf_iters > 0 && encoding == FrameEncoding::VarDct && read_bool(reader)?;
    if sharp_custom {
        for _ in 0..8 {
            read_f16(reader)?;
        }
    }
    let weight_custom = epf_iters > 0 && read_bool(reader)?;
    if weight_custom {
        for _ in 0..5 {
            read_f16(reader)?;
        }
    }
    let sigma_custom = epf_iters > 0 && read_bool(reader)?;
    if sigma_custom && encoding == FrameEncoding::VarDct {
        read_f16(reader)?;
    }
    if sigma_custom {
        for _ in 0..3 {
            read_f16(reader)?;
        }
    }
    if epf_iters > 0 && encoding == FrameEncoding::Modular {
        read_f16(reader)?;
    }
    parse_extensions(reader, max_extension_bits)
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

fn parse_toc(reader: &mut BitReader<'_>, num_entries: usize) -> Result<Vec<u32>, InventoryError> {
    if read_bool(reader)? {
        return Err(InventoryError::UnsupportedGrammar(
            UnsupportedCodestreamGrammar::EntropyCodedTocPermutation,
        ));
    }
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
    Ok(entries)
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

fn read_f16(reader: &mut BitReader<'_>) -> Result<(), InventoryError> {
    let bits = read_bits(reader, 16)? as u16;
    if bits & 0x7c00 == 0x7c00 {
        return Err(InventoryError::InvalidFrame("non-finite F16 header value"));
    }
    Ok(())
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
    use crate::{FragmentedContainerWriter, ParseLimits, parse, write_container};

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
        assert!(basic.image_header.xyb_encoded);
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
        assert_eq!((frame.width, frame.height), (1, 1));
        assert_eq!((frame.group_count, frame.low_frequency_group_count), (1, 1));
        assert_eq!(
            frame.sections,
            vec![FrameSection {
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
        assert!(!fragmented.image_header.xyb_encoded);
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
    fn toc_padding_permutation_and_declared_ranges_are_validated() {
        let basic = crate::test_fixtures::basic();

        let mut non_zero_padding = basic.clone();
        non_zero_padding[9] |= 1 << 3; // Codestream bit 75, after the non-permuted TOC flag.
        assert!(matches!(
            inventory_error(&non_zero_padding),
            InventoryError::NonZeroPadding { .. }
        ));

        let mut permuted = basic.clone();
        permuted[9] |= 1 << 2; // Codestream bit 74.
        assert_eq!(
            inventory_error(&permuted),
            InventoryError::UnsupportedGrammar(
                UnsupportedCodestreamGrammar::EntropyCodedTocPermutation
            )
        );

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
