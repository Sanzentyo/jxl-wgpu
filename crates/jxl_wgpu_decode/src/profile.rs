use jxl_gpu_bitstream::{
    BitReader, CodestreamInventory, FrameBlendInfo, FrameEncoding, FrameSectionKind, FrameType,
    ImageHeaderInventory, PrefixCodeEntry, SampleBitDepth,
};

use crate::{ModularChannels, Result, UnsupportedCodestreamFeature, UnsupportedProfile};

const MAX_CODESTREAM_BYTES: usize = 16 * 1024 * 1024;
const GROUP_DIMENSION: u32 = 256;
const PREFIX_ALPHABET_SIZE: usize = 257;
const RAW_SYMBOLS: usize = 19;
const LZ77_SYMBOLS: usize = 33;
const LZ77_SYMBOL_OFFSET: usize = 224;
const MAX_PREFIX_BITS: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularGroup {
    pub token_bit_offset: u64,
    pub token_bit_end: u64,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ModularGroup {
    pub(crate) fn sample_count(self) -> Result<u32> {
        self.width
            .checked_mul(self.height)
            .ok_or_else(|| unsupported_error("Modular group sample count overflow").into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StandardModularProfile {
    pub width: u32,
    pub height: u32,
    pub bits_per_sample: u8,
    pub channels: ModularChannels,
    pub group_columns: u32,
    pub group_rows: u32,
    pub groups: Vec<ModularGroup>,
    pub raw_prefix: [[PrefixCodeEntry; RAW_SYMBOLS]; 4],
    pub lz77_prefix: [[PrefixCodeEntry; LZ77_SYMBOLS]; 4],
}

fn validate_image_header(
    codestream: &[u8],
    image: &ImageHeaderInventory,
    channels: ModularChannels,
    bits_per_sample: u8,
) -> Result<()> {
    if image.orientation != 1
        || image.intrinsic_size.is_some()
        || image.preview_size.is_some()
        || image.embedded_icc.is_some()
        || image.animation.is_some()
        || image.modular_16bit_buffers != (bits_per_sample <= 14)
    {
        return unsupported(
            "the lossless Modular GPU profile requires canonical still-image metadata",
        );
    }

    let mut reader = BitReader::new(codestream);
    expect(&mut reader, 16, 0x0aff, "JPEG XL codestream signature")?;
    expect(&mut reader, 1, 0, "non-small image header")?;
    let height = read_size(&mut reader, true)?;
    let width = read_size(&mut reader, false)?;
    if width != image.width || height != image.height {
        return unsupported("standard image-header extent does not match its inventory");
    }
    expect(&mut reader, 1, 0, "non-default image metadata")?;
    expect(&mut reader, 1, 0, "no extra metadata fields")?;
    read_integer_bit_depth(&mut reader, bits_per_sample, "main image bit depth")?;
    expect(
        &mut reader,
        1,
        u64::from(bits_per_sample <= 14),
        "canonical Modular buffer depth",
    )?;

    let has_alpha = channels == ModularChannels::Rgba;
    expect(
        &mut reader,
        2,
        u64::from(has_alpha),
        "canonical extra-channel count",
    )?;
    if has_alpha {
        if bits_per_sample == 8 {
            expect(&mut reader, 1, 1, "default unassociated alpha metadata")?;
        } else {
            expect(&mut reader, 1, 0, "explicit alpha metadata")?;
            expect(&mut reader, 2, 0, "alpha extra-channel type")?;
            read_integer_bit_depth(&mut reader, bits_per_sample, "alpha bit depth")?;
            expect(&mut reader, 2, 0, "full-resolution alpha")?;
            expect(&mut reader, 2, 0, "empty alpha name")?;
            expect(&mut reader, 1, 0, "unassociated alpha")?;
        }
    }

    expect(&mut reader, 1, 0, "non-XYB image")?;
    if channels == ModularChannels::Gray {
        for (count, value, name) in [
            (1, 0, "non-default grayscale color encoding"),
            (1, 0, "no ICC profile"),
            (2, 1, "grayscale color space"),
            (2, 1, "D65 white point"),
            (1, 0, "enumerated transfer function"),
            (2, 0b10, "transfer-function selector"),
            (4, 11, "sRGB transfer function"),
            (2, 1, "relative rendering intent"),
        ] {
            expect(&mut reader, count, value, name)?;
        }
    } else {
        expect(&mut reader, 1, 1, "default sRGB color encoding")?;
    }
    expect(&mut reader, 2, 0, "no image extensions")?;
    expect(&mut reader, 1, 1, "default transform data")?;
    let grammar_end = reader.bit_offset();
    if image.bit_range.offset != 0 || image.bit_range.end() != Some(grammar_end) {
        return unsupported(format!(
            "canonical image-header length {} does not match inventory {:?}",
            grammar_end,
            image.bit_range.end()
        ));
    }
    reader.align_to_byte()?;
    Ok(())
}

fn read_integer_bit_depth(reader: &mut BitReader<'_>, expected: u8, field: &str) -> Result<()> {
    expect(reader, 1, 0, field)?;
    let actual = match reader.read_bits(2)? {
        0 => 8,
        1 => 10,
        2 => 12,
        3 => u8::try_from(reader.read_bits(6)?)
            .map_err(|_| unsupported_error("integer bit depth exceeds u8"))?
            .checked_add(1)
            .ok_or_else(|| unsupported_error("integer bit depth overflow"))?,
        _ => unreachable!(),
    };
    if actual != expected {
        return unsupported(format!(
            "the lossless Modular GPU profile requires {field} {expected}, received {actual}"
        ));
    }
    Ok(())
}

fn read_size(reader: &mut BitReader<'_>, has_ratio: bool) -> Result<u32> {
    let selector = usize::try_from(reader.read_bits(2)?)
        .map_err(|_| unsupported_error("image extent selector overflow"))?;
    let widths = [9, 13, 18, 30];
    let value = u32::try_from(reader.read_bits(widths[selector])?)
        .map_err(|_| unsupported_error("image extent overflows u32"))?
        .checked_add(1)
        .ok_or_else(|| unsupported_error("image extent overflows u32"))?;
    if has_ratio {
        expect(reader, 3, 0, "explicit width follows height")?;
    }
    Ok(value)
}

/// Recognizes the standards-compliant lossless Modular grammar emitted by `jxl_wgpu_encode`.
///
/// Only bounded image/frame metadata, the Modular DC-global prefix description, and fixed group
/// headers are inspected here. Entropy symbols, residuals, predictors, and pixels are deliberately
/// left unread for the GPU kernel.
pub(crate) fn parse_standard_modular_profile(
    codestream: &[u8],
    inventory: &CodestreamInventory,
) -> Result<StandardModularProfile> {
    if codestream.len() > MAX_CODESTREAM_BYTES {
        return unsupported("the lossless Modular GPU profile codestream exceeds 16 MiB");
    }
    let image = &inventory.image_header;
    if image.width == 0 || image.height == 0 {
        return unsupported("the lossless Modular GPU profile requires a non-empty image");
    }
    let bits_per_sample = match image.bit_depth {
        SampleBitDepth::Integer {
            bits_per_sample: bits @ 1..=16,
        } => u8::try_from(bits).expect("1..=16 fits u8"),
        _ => {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::ModularBitDepth(match image.bit_depth {
                    SampleBitDepth::Integer { bits_per_sample }
                    | SampleBitDepth::Float {
                        bits_per_sample, ..
                    } => u8::try_from(bits_per_sample).unwrap_or(u8::MAX),
                }),
                "the stock Modular GPU frontend reconstructs 1 through 16-bit integer samples",
            )
            .into());
        }
    };
    let channels = match (image.grayscale, image.extra_channel_count) {
        (true, 0) => ModularChannels::Gray,
        (false, 0) => ModularChannels::Rgb,
        (false, 1) => ModularChannels::Rgba,
        _ => {
            return Err(UnsupportedProfile::new(
                UnsupportedCodestreamFeature::ExtraChannels,
                "the lossless Modular GPU profile supports Gray, RGB, or one RGBA alpha channel",
            )
            .into());
        }
    };
    if image.xyb_encoded {
        return unsupported("the lossless Modular GPU profile does not use XYB metadata");
    }
    validate_image_header(codestream, image, channels, bits_per_sample)?;
    if image.animation.is_some() || image.preview_size.is_some() || inventory.frames.len() != 1 {
        return unsupported(
            "the lossless Modular GPU profile requires exactly one still-image frame",
        );
    }

    let frame = &inventory.frames[0];
    if frame.is_preview
        || frame.frame_type != FrameType::Regular
        || frame.encoding != FrameEncoding::Modular
        || frame.flags != 0
        || frame.do_ycbcr
        || frame.jpeg_upsampling != [0; 3]
        || frame.upsampling != 1
        || frame.group_size_shift != 1
        || frame.num_passes != 1
        || frame.have_crop
        || frame.x0 != 0
        || frame.y0 != 0
        || frame.width != image.width
        || frame.height != image.height
        || frame.duration_ticks != 0
        || !frame.is_last
        || frame.save_as_reference != 0
        || frame.save_before_color_transform
        || !frame.name_bytes.is_empty()
        || frame.color_blend != FrameBlendInfo::default()
        || frame.extra_channel_blends
            != vec![FrameBlendInfo::default(); usize::from(channels == ModularChannels::Rgba)]
    {
        return unsupported(
            "the lossless Modular GPU profile requires one final uncropped regular frame with canonical grouping, replace blending, and no references",
        );
    }

    let group_columns = image.width.div_ceil(GROUP_DIMENSION);
    let group_rows = image.height.div_ceil(GROUP_DIMENSION);
    let expected_group_count = u64::from(group_columns)
        .checked_mul(u64::from(group_rows))
        .ok_or_else(|| unsupported_error("Modular group grid overflow"))?;
    if frame.group_count != expected_group_count {
        return unsupported("frame inventory group count does not match the Modular canvas");
    }

    let dc_global = if expected_group_count == 1 {
        frame
            .sections
            .iter()
            .find(|section| section.kind == FrameSectionKind::Single)
    } else {
        frame
            .sections
            .iter()
            .find(|section| section.kind == FrameSectionKind::LowFrequencyGlobal)
    }
    .ok_or_else(|| unsupported_error("the Modular frame is missing DC-global metadata"))?;
    let mut reader = BitReader::new(codestream);
    reader.skip_bits(dc_global.bits.offset)?;
    let prefix = parse_dc_global(&mut reader, channels)?;
    let dc_end = dc_global
        .bits
        .end()
        .ok_or_else(|| unsupported_error("DC-global bit range overflow"))?;
    if reader.bit_offset() > dc_end {
        return unsupported("Modular DC-global metadata exceeds its TOC section");
    }

    let groups = if expected_group_count == 1 {
        vec![ModularGroup {
            token_bit_offset: reader.bit_offset(),
            token_bit_end: dc_end,
            x: 0,
            y: 0,
            width: image.width,
            height: image.height,
        }]
    } else {
        if !bits_are_zero(codestream, reader.bit_offset(), dc_end) {
            return unsupported("non-zero data follows the bounded DC-global prefix metadata");
        }
        validate_empty_non_pass_sections(codestream, frame)?;
        let group_count = usize::try_from(expected_group_count)
            .map_err(|_| unsupported_error("Modular group count exceeds host address space"))?;
        let mut groups = vec![None; group_count];
        for section in &frame.sections {
            let FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index,
            } = section.kind
            else {
                continue;
            };
            let index = usize::try_from(group_index)
                .map_err(|_| unsupported_error("Modular group index exceeds host address space"))?;
            let slot = groups.get_mut(index).ok_or_else(|| {
                unsupported_error("Modular pass-group index exceeds the frame grid")
            })?;
            if slot.is_some() {
                return unsupported("the Modular frame contains a duplicate pass-group section");
            }
            let mut group_reader = BitReader::new(codestream);
            group_reader.skip_bits(section.bits.offset)?;
            expect(&mut group_reader, 1, 1, "LF-global Modular tree")?;
            expect(&mut group_reader, 1, 1, "default weighted predictor")?;
            expect(&mut group_reader, 2, 0, "no local Modular transforms")?;
            let column = u32::try_from(group_index % u64::from(group_columns))
                .map_err(|_| unsupported_error("Modular group column overflow"))?;
            let row = u32::try_from(group_index / u64::from(group_columns))
                .map_err(|_| unsupported_error("Modular group row overflow"))?;
            let x = column
                .checked_mul(GROUP_DIMENSION)
                .ok_or_else(|| unsupported_error("Modular group x origin overflow"))?;
            let y = row
                .checked_mul(GROUP_DIMENSION)
                .ok_or_else(|| unsupported_error("Modular group y origin overflow"))?;
            let token_bit_end = section
                .bits
                .end()
                .ok_or_else(|| unsupported_error("Modular pass-group bit range overflow"))?;
            *slot = Some(ModularGroup {
                token_bit_offset: group_reader.bit_offset(),
                token_bit_end,
                x,
                y,
                width: image.width.saturating_sub(x).min(GROUP_DIMENSION),
                height: image.height.saturating_sub(y).min(GROUP_DIMENSION),
            });
        }
        groups
            .into_iter()
            .enumerate()
            .map(|(index, group)| {
                group.ok_or_else(|| {
                    unsupported_error(format!("the Modular frame is missing pass-group {index}"))
                        .into()
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    let codestream_bits = u64::try_from(codestream.len())
        .ok()
        .and_then(|bytes| bytes.checked_mul(8))
        .ok_or_else(|| unsupported_error("codestream bit length overflow"))?;
    for group in &groups {
        if group.width == 0
            || group.height == 0
            || group.token_bit_offset >= group.token_bit_end
            || group.token_bit_end > codestream_bits
        {
            return unsupported("the Modular frame contains an invalid group token range");
        }
        group.sample_count()?;
    }

    Ok(StandardModularProfile {
        width: image.width,
        height: image.height,
        bits_per_sample,
        channels,
        group_columns,
        group_rows,
        groups,
        raw_prefix: prefix.map(|code| code.raw),
        lz77_prefix: prefix.map(|code| code.lz77),
    })
}

fn validate_empty_non_pass_sections(
    codestream: &[u8],
    frame: &jxl_gpu_bitstream::FrameInventory,
) -> Result<()> {
    for section in &frame.sections {
        match section.kind {
            FrameSectionKind::LowFrequencyGroup { .. } | FrameSectionKind::HighFrequencyGlobal => {
                let end = section
                    .bits
                    .end()
                    .ok_or_else(|| unsupported_error("Modular section bit range overflow"))?;
                if !bits_are_zero(codestream, section.bits.offset, end) {
                    return unsupported(
                        "the lossless Modular profile requires empty LF-group and HF-global sections",
                    );
                }
            }
            FrameSectionKind::PassGroup { pass_index, .. } if pass_index != 0 => {
                return Err(UnsupportedProfile::new(
                    UnsupportedCodestreamFeature::MultiplePasses,
                    "the lossless Modular GPU frontend accepts exactly one pass",
                )
                .into());
            }
            _ => {}
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ParsedPrefix {
    raw: [PrefixCodeEntry; RAW_SYMBOLS],
    lz77: [PrefixCodeEntry; LZ77_SYMBOLS],
}

fn parse_dc_global(
    reader: &mut BitReader<'_>,
    channels: ModularChannels,
) -> Result<[ParsedPrefix; 4]> {
    for (count, value, name) in [
        (1, 1, "Modular global tree"),
        (1, 1, "Modular WP header"),
        (1, 0, "default Modular WP parameters"),
        (1, 1, "Modular transform tree"),
        (2, 0, "Modular tree size selector"),
        (1, 1, "Modular tree entropy LZ77"),
        (4, 0, "Modular tree LZ77 minimum length"),
        (6, 0b100011, "Modular tree LZ77 hybrid config"),
        (2, 1, "Modular tree context count"),
        (2, 3, "Modular tree context-map selector"),
    ] {
        expect(reader, count, value, name)?;
    }
    for symbol in 0..4 {
        expect(reader, 2, symbol, "Modular tree context-map symbol")?;
    }
    expect(reader, 1, 0, "Modular tree context-map MTF")?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        expect(
            reader,
            SYMBOL_NBITS[index],
            SYMBOL_BITS[index],
            "fixed Modular decision tree",
        )?;
    }

    for (count, value, name) in [
        (1, 1, "Modular global entropy LZ77"),
        (2, 0, "Modular global LZ77 minimum-length selector"),
        (4, 0b1010, "Modular global LZ77 minimum length"),
        (4, 4, "Modular global LZ77 length config split exponent"),
        (3, 0, "Modular global LZ77 length config msb"),
        (3, 0, "Modular global LZ77 length config lsb"),
        (1, 1, "Modular global simple cluster map"),
        (2, 3, "Modular global cluster count"),
    ] {
        expect(reader, count, value, name)?;
    }
    for context in [4, 3, 2, 1, 0] {
        expect(reader, 3, context, "Modular global context cluster")?;
    }
    expect(reader, 1, 1, "Modular global cluster-map MTF")?;
    expect(reader, 4, 0, "Modular global cluster-map MTF prefix")?;
    for _ in 0..4 {
        expect(reader, 4, 0, "Modular global cluster-map MTF symbol")?;
    }
    expect(reader, 5, 1, "Modular global histogram count")?;
    for _ in 0..4 {
        expect(reader, 1, 1, "Modular global hybrid config selector")?;
        expect(reader, 4, 8, "Modular global hybrid config split exponent")?;
        expect(reader, 8, 0, "Modular global hybrid config low bits")?;
    }
    expect(reader, 2, 1, "Modular global entropy coder")?;
    expect(reader, 2, 0, "Modular global entropy selector")?;
    expect(reader, 1, 1, "Modular global prefix histogram mode")?;

    let prefixes = [
        parse_prefix_histogram(reader)?,
        parse_prefix_histogram(reader)?,
        parse_prefix_histogram(reader)?,
        parse_prefix_histogram(reader)?,
    ];
    expect(reader, 1, 1, "Modular global distance multiplier")?;
    expect(reader, 1, 1, "Modular global predictor tree")?;
    if channels == ModularChannels::Gray {
        expect(reader, 2, 0, "no Modular color transforms")?;
    } else {
        expect(reader, 2, 1, "one Modular color transform")?;
        expect(reader, 2, 0, "reversible color transform")?;
        expect(reader, 5, 0, "reversible color transform first channel")?;
        expect(reader, 2, 0, "reversible YCoCg transform type")?;
    }
    Ok(prefixes)
}

fn parse_prefix_histogram(reader: &mut BitReader<'_>) -> Result<ParsedPrefix> {
    expect(reader, 2, 0, "complex prefix histogram")?;
    const CODE_LENGTH_ORDER: [usize; 18] =
        [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut code_length_lengths = [0u8; 18];
    let mut bit_accumulator = 0u32;
    let mut nonzero_count = 0u32;
    let mut nonzero_symbol = 0usize;
    for symbol in CODE_LENGTH_ORDER {
        let length = read_code_length_code(reader)?;
        code_length_lengths[symbol] = length;
        if length != 0 {
            bit_accumulator = bit_accumulator
                .checked_add(32u32 >> length)
                .ok_or_else(|| unsupported_error("prefix code-length histogram overflow"))?;
            nonzero_count += 1;
            nonzero_symbol = symbol;
            if bit_accumulator == 32 {
                break;
            }
            if bit_accumulator > 32 {
                return unsupported("invalid prefix code-length histogram");
            }
        }
    }
    if bit_accumulator != 32 {
        return unsupported("incomplete prefix code-length histogram");
    }

    let code_length_entries = canonical_entries(&code_length_lengths)?;
    let mut code_lengths = [0u8; PREFIX_ALPHABET_SIZE];
    let mut index = 0usize;
    let mut bit_accumulator = 0u32;
    let mut previous_symbol = 8u8;
    let mut last_nonzero_length = 8u8;
    let mut last_repeat_count = 0usize;
    let mut repeat_count = 0usize;
    let mut repeat_length = 0u8;

    while index < PREFIX_ALPHABET_SIZE {
        let length = if repeat_count != 0 {
            repeat_count -= 1;
            repeat_length
        } else {
            let symbol = if nonzero_count == 1 {
                u8::try_from(nonzero_symbol)
                    .map_err(|_| unsupported_error("prefix code-length symbol overflow"))?
            } else {
                u8::try_from(read_prefix_symbol(reader, &code_length_entries)?)
                    .map_err(|_| unsupported_error("prefix code-length symbol overflow"))?
            };
            let length = match symbol {
                0 => 0,
                1..=15 => {
                    last_nonzero_length = symbol;
                    symbol
                }
                16 => {
                    let mut current = usize::try_from(reader.read_bits(2)?)
                        .map_err(|_| unsupported_error("prefix repeat count overflow"))?
                        + 3;
                    if previous_symbol == 16 {
                        current = current
                            .checked_add(
                                last_repeat_count
                                    .checked_mul(3)
                                    .and_then(|value| value.checked_sub(8))
                                    .ok_or_else(|| {
                                        unsupported_error("prefix repeat count overflow")
                                    })?,
                            )
                            .ok_or_else(|| unsupported_error("prefix repeat count overflow"))?;
                        last_repeat_count = last_repeat_count
                            .checked_add(current)
                            .ok_or_else(|| unsupported_error("prefix repeat count overflow"))?;
                    } else {
                        last_repeat_count = current;
                    }
                    repeat_count = current
                        .checked_sub(1)
                        .ok_or_else(|| unsupported_error("zero prefix repeat count"))?;
                    repeat_length = last_nonzero_length;
                    last_nonzero_length
                }
                17 => {
                    let mut current = usize::try_from(reader.read_bits(3)?)
                        .map_err(|_| unsupported_error("prefix zero-repeat count overflow"))?
                        + 3;
                    if previous_symbol == 17 {
                        current = current
                            .checked_add(
                                last_repeat_count
                                    .checked_mul(7)
                                    .and_then(|value| value.checked_sub(16))
                                    .ok_or_else(|| {
                                        unsupported_error("prefix zero-repeat count overflow")
                                    })?,
                            )
                            .ok_or_else(|| {
                                unsupported_error("prefix zero-repeat count overflow")
                            })?;
                        last_repeat_count =
                            last_repeat_count.checked_add(current).ok_or_else(|| {
                                unsupported_error("prefix zero-repeat count overflow")
                            })?;
                    } else {
                        last_repeat_count = current;
                    }
                    repeat_count = current
                        .checked_sub(1)
                        .ok_or_else(|| unsupported_error("zero prefix repeat count"))?;
                    repeat_length = 0;
                    0
                }
                _ => return unsupported("invalid prefix code-length symbol"),
            };
            previous_symbol = symbol;
            length
        };
        code_lengths[index] = length;
        index += 1;
        if length != 0 {
            bit_accumulator = bit_accumulator
                .checked_add(1u32 << (u32::from(MAX_PREFIX_BITS) - u32::from(length)))
                .ok_or_else(|| unsupported_error("prefix histogram accumulator overflow"))?;
            if bit_accumulator > 1u32 << MAX_PREFIX_BITS {
                return unsupported("over-subscribed prefix histogram");
            }
            if bit_accumulator == 1u32 << MAX_PREFIX_BITS && repeat_count == 0 {
                break;
            }
        }
    }
    if bit_accumulator != 1u32 << MAX_PREFIX_BITS || repeat_count != 0 {
        return unsupported("incomplete prefix histogram");
    }

    let entries = canonical_entries(&code_lengths)?;
    Ok(ParsedPrefix {
        raw: std::array::from_fn(|index| entries[index]),
        lz77: std::array::from_fn(|index| entries[LZ77_SYMBOL_OFFSET + index]),
    })
}

fn read_code_length_code(reader: &mut BitReader<'_>) -> Result<u8> {
    Ok(match reader.read_bits(2)? {
        0 => 0,
        1 => 4,
        2 => 3,
        3 => match reader.read_bits(1)? {
            0 => 2,
            1 => {
                if reader.read_bits(1)? == 0 {
                    1
                } else {
                    5
                }
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    })
}

fn read_prefix_symbol(reader: &mut BitReader<'_>, entries: &[PrefixCodeEntry]) -> Result<u32> {
    let mut bits = 0u16;
    for length in 1..=MAX_PREFIX_BITS {
        let bit = u16::try_from(reader.read_bits(1)?)
            .map_err(|_| unsupported_error("prefix bit overflow"))?;
        bits |= bit << (length - 1);
        if let Some(symbol) = entries
            .iter()
            .position(|entry| entry.bit_len == length && entry.bits == bits)
        {
            return u32::try_from(symbol)
                .map_err(|_| unsupported_error("prefix symbol exceeds u32").into());
        }
    }
    unsupported("invalid prefix symbol")
}

fn canonical_entries<const N: usize>(lengths: &[u8; N]) -> Result<[PrefixCodeEntry; N]> {
    let mut counts = [0u16; MAX_PREFIX_BITS as usize + 1];
    for &length in lengths {
        if length > MAX_PREFIX_BITS {
            return unsupported("prefix code length exceeds 15 bits");
        }
        if length != 0 {
            counts[usize::from(length)] = counts[usize::from(length)]
                .checked_add(1)
                .ok_or_else(|| unsupported_error("prefix code-length count overflow"))?;
        }
    }
    let mut next_code = [0u16; MAX_PREFIX_BITS as usize + 1];
    let mut code = 0u16;
    for length in 1..=MAX_PREFIX_BITS as usize {
        code = code
            .checked_add(counts[length - 1])
            .and_then(|value| value.checked_shl(1))
            .ok_or_else(|| unsupported_error("canonical prefix code overflow"))?;
        next_code[length] = code;
    }
    Ok(std::array::from_fn(|symbol| {
        let length = lengths[symbol];
        if length == 0 {
            return PrefixCodeEntry {
                bit_len: 0,
                bits: 0,
            };
        }
        let length_index = usize::from(length);
        let bits = reverse_prefix_bits(length, next_code[length_index]);
        next_code[length_index] = next_code[length_index].wrapping_add(1);
        PrefixCodeEntry {
            bit_len: length,
            bits,
        }
    }))
}

fn reverse_prefix_bits(length: u8, bits: u16) -> u16 {
    bits.reverse_bits() >> (u16::BITS - u32::from(length))
}

fn expect(reader: &mut BitReader<'_>, count: u8, expected: u64, field: &str) -> Result<()> {
    let actual = reader.read_bits(count)?;
    if actual != expected {
        return unsupported(format!(
            "the lossless Modular GPU profile requires {field} (expected {expected}, received {actual})"
        ));
    }
    Ok(())
}

fn bits_are_zero(bytes: &[u8], start: u64, end: u64) -> bool {
    let available_bits = u64::try_from(bytes.len())
        .ok()
        .and_then(|length| length.checked_mul(8));
    if start > end || available_bits.is_none_or(|available| end > available) {
        return false;
    }
    (start..end).all(|cursor| {
        let byte = usize::try_from(cursor / 8).expect("validated bit offset fits usize");
        bytes[byte] & (1u8 << (cursor % 8)) == 0
    })
}

fn unsupported<T>(detail: impl Into<String>) -> Result<T> {
    Err(unsupported_error(detail).into())
}

fn unsupported_error(detail: impl Into<String>) -> UnsupportedProfile {
    UnsupportedProfile::new(
        UnsupportedCodestreamFeature::Other("lossless-modular-standard-profile".into()),
        detail,
    )
}

#[cfg(test)]
mod tests {
    use super::{bits_are_zero, canonical_entries};

    #[test]
    fn checks_unaligned_padding_bits() {
        assert!(bits_are_zero(&[0b0000_0101], 3, 8));
        assert!(!bits_are_zero(&[0b0001_0101], 3, 8));
        assert!(bits_are_zero(&[0xff, 0, 0, 0xff], 8, 24));
        assert!(!bits_are_zero(&[0xff, 0, 1, 0xff], 8, 24));
        assert!(bits_are_zero(&[0b0000_0111, 0, 0b1110_0000], 3, 21));
        assert!(!bits_are_zero(&[0; 1], 0, 9));
        assert!(!bits_are_zero(&[0; 1], 7, 6));
    }

    #[test]
    fn canonical_entries_use_lsb_first_wire_codes() {
        let entries = canonical_entries(&[1, 2, 2]).unwrap();
        assert_eq!((entries[0].bit_len, entries[0].bits), (1, 0));
        assert_eq!((entries[1].bit_len, entries[1].bits), (2, 1));
        assert_eq!((entries[2].bit_len, entries[2].bits), (2, 3));
    }
}
