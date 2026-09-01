use jxl_gpu_bitstream::{
    CodestreamInventory, FrameBlendInfo, FrameEncoding, FrameSectionKind, FrameType,
    ImageHeaderInventory, SampleBitDepth,
};

use crate::codestream_data::CodestreamData;
use crate::modular_tree::BitInput;
use crate::{ModularChannels, Result, UnsupportedCodestreamFeature, UnsupportedProfile};
use crate::{
    ModularTransformFeature,
    modular_tree::{MaConfigIr, MaTreeLimits, WpHeaderIr, parse_ma_config},
};

const GROUP_DIMENSION: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularGroup {
    pub token_bit_offset: u64,
    pub token_bit_end: u64,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub stream_index: u32,
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
    pub ma_config: MaConfigIr,
    pub wp_header: WpHeaderIr,
}

fn validate_image_header(
    codestream: &CodestreamData,
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

    let mut reader = codestream.reader();
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

fn read_integer_bit_depth(reader: &mut impl BitInput, expected: u8, field: &str) -> Result<()> {
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

fn read_size(reader: &mut impl BitInput, has_ratio: bool) -> Result<u32> {
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
    codestream: &CodestreamData,
    inventory: &CodestreamInventory,
) -> Result<StandardModularProfile> {
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
    let mut reader = codestream.reader();
    reader.skip_bits(dc_global.bits.offset)?;
    parse_lf_channel_dequantization(&mut reader)?;
    let (ma_config, wp_header) = parse_dc_global_ir(&mut reader, channels)?;
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
            stream_index: 0,
        }]
    } else {
        if !codestream.bits_are_zero(reader.bit_offset(), dc_end)? {
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
            let mut group_reader = codestream.reader();
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
            *slot =
                Some(ModularGroup {
                    token_bit_offset: group_reader.bit_offset(),
                    token_bit_end,
                    x,
                    y,
                    width: image.width.saturating_sub(x).min(GROUP_DIMENSION),
                    height: image.height.saturating_sub(y).min(GROUP_DIMENSION),
                    stream_index: u32::try_from(
                        18u64
                            .checked_add(
                                frame.low_frequency_group_count.checked_mul(3).ok_or_else(
                                    || unsupported_error("Modular stream index overflow"),
                                )?,
                            )
                            .and_then(|value| value.checked_add(group_index))
                            .ok_or_else(|| unsupported_error("Modular stream index overflow"))?,
                    )
                    .map_err(|_| unsupported_error("Modular stream index exceeds u32"))?,
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

    let codestream_bits = codestream.logical_bits()?;
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
        ma_config,
        wp_header,
    })
}

/// Advances across `LfChannelDequantization`, which precedes `GlobalModular` in LF-global.
///
/// These values only affect VarDCT. A lossless Modular frame must still carry the bundle, so the
/// frontend parses its bounded wire shape before locating the global MA tree.
fn parse_lf_channel_dequantization(reader: &mut impl BitInput) -> Result<()> {
    let all_default = reader.read_bits(1)? != 0;
    if !all_default {
        // m_x_lf, m_y_lf, and m_b_lf are IEEE-754 binary16 values on the wire. Their values are
        // irrelevant to Modular reconstruction, but consuming all three fields is required to
        // locate GlobalModular without depending on a CPU image decoder.
        for _ in 0..3 {
            reader.read_bits(16)?;
        }
    }
    Ok(())
}

fn parse_dc_global_ir(
    reader: &mut impl BitInput,
    channels: ModularChannels,
) -> Result<(MaConfigIr, WpHeaderIr)> {
    expect(reader, 1, 1, "global Modular MA tree")?;
    let ma_config = parse_ma_config(reader, MaTreeLimits::default())?;
    expect(reader, 1, 1, "global Modular tree selection")?;
    let wp_header = WpHeaderIr::parse(reader)?;
    let transform_count = read_u32_selector(reader, [(0, 0), (1, 0), (2, 4), (18, 8)])?;
    match (channels, transform_count) {
        (ModularChannels::Gray, 0) => {}
        (ModularChannels::Rgb | ModularChannels::Rgba, 1) => {
            let transform_id = reader.read_bits(2)?;
            if transform_id != 0 {
                return unsupported_transform(match transform_id {
                    1 => ModularTransformFeature::Palette,
                    2 => ModularTransformFeature::Squeeze,
                    _ => ModularTransformFeature::Invalid,
                });
            }
            let begin_channel = read_u32_selector(reader, [(0, 3), (8, 6), (72, 10), (1096, 13)])?;
            let rct_type = read_u32_selector(reader, [(6, 0), (0, 2), (2, 4), (10, 6)])?;
            if begin_channel != 0 || rct_type != 6 {
                return unsupported_transform(ModularTransformFeature::ReversibleColor {
                    begin_channel,
                    rct_type,
                });
            }
        }
        (_, 0) => {
            return unsupported(
                "the standard Modular color profile requires its reversible color transform",
            );
        }
        _ => return unsupported_transform(ModularTransformFeature::Invalid),
    }
    Ok((ma_config, wp_header))
}

fn read_u32_selector(reader: &mut impl BitInput, variants: [(u32, u8); 4]) -> Result<u32> {
    let selector = usize::try_from(reader.read_bits(2)?)
        .map_err(|_| unsupported_error("U32 selector exceeds host address space"))?;
    let (base, bits) = variants[selector];
    let extra = u32::try_from(reader.read_bits(bits)?)
        .map_err(|_| unsupported_error("U32 field exceeds u32"))?;
    base.checked_add(extra)
        .ok_or_else(|| unsupported_error("U32 field overflows").into())
}

fn unsupported_transform<T>(feature: ModularTransformFeature) -> Result<T> {
    Err(UnsupportedProfile::new(
        UnsupportedCodestreamFeature::ModularTransform(feature),
        "the Modular transform is parsed but has not been lowered to a GPU kernel",
    )
    .into())
}

fn validate_empty_non_pass_sections(
    codestream: &CodestreamData,
    frame: &jxl_gpu_bitstream::FrameInventory,
) -> Result<()> {
    for section in &frame.sections {
        match section.kind {
            FrameSectionKind::LowFrequencyGroup { .. } | FrameSectionKind::HighFrequencyGlobal => {
                let end = section
                    .bits
                    .end()
                    .ok_or_else(|| unsupported_error("Modular section bit range overflow"))?;
                if !codestream.bits_are_zero(section.bits.offset, end)? {
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

fn expect(reader: &mut impl BitInput, count: u8, expected: u64, field: &str) -> Result<()> {
    let actual = reader.read_bits(count)?;
    if actual != expected {
        return unsupported(format!(
            "the lossless Modular GPU profile requires {field} (expected {expected}, received {actual})"
        ));
    }
    Ok(())
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
    use std::sync::Arc;

    use jxl_gpu_bitstream::{InventoryLimits, ParseLimits, StreamSlice, parse};

    use super::*;

    #[test]
    fn modular_profile_is_identical_across_every_shared_chunk_split() {
        let encoded: Arc<[u8]> = Arc::from(fixture(include_str!(
            "../test-data/gpu_gray8_lossless.jxl.hex"
        )));
        let parsed = parse(&encoded, ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let bytes: Arc<[u8]> = Arc::from(parsed.codestream());
        let complete =
            CodestreamData::from_spans([(0, StreamSlice::from_shared(Arc::clone(&bytes)))])
                .unwrap();
        let expected = parse_standard_modular_profile(&complete, &inventory).unwrap();

        for split_offset in 0..=bytes.len() {
            let split_source = CodestreamData::from_spans([
                (
                    0,
                    StreamSlice::from_shared_range(Arc::clone(&bytes), 0..split_offset).unwrap(),
                ),
                (
                    split_offset as u64,
                    StreamSlice::from_shared_range(Arc::clone(&bytes), split_offset..bytes.len())
                        .unwrap(),
                ),
            ])
            .unwrap();
            assert_eq!(
                parse_standard_modular_profile(&split_source, &inventory).unwrap(),
                expected,
                "chunk split {split_offset} changed the parsed profile"
            );
        }
    }

    fn fixture(input: &str) -> Vec<u8> {
        fn nibble(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("invalid checked-in fixture hex digit"),
            }
        }

        let digits = input
            .bytes()
            .filter(|byte| !byte.is_ascii_whitespace())
            .collect::<Vec<_>>();
        assert_eq!(digits.len() % 2, 0, "fixture hex must contain whole bytes");
        digits
            .chunks_exact(2)
            .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
            .collect()
    }
}
