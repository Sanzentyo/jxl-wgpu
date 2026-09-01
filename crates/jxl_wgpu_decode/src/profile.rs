use jxl_gpu_bitstream::{
    CodestreamInventory, FrameBlendInfo, FrameEncoding, FrameSectionKind, FrameType,
    ImageHeaderInventory, SampleBitDepth,
};

use crate::modular_inverse::{ModularInversePlan, plan_modular_inverse};
use crate::modular_transform::{
    ModularChannelTopology, ModularRct, ModularTransformIr, ModularTransformLimits,
    ModularTransformPlan, PackedModularChannelMetadata, parse_modular_transforms,
};
use crate::modular_tree::BitInput;
use crate::{
    GpuCodestream, ModularChannels, Result, UnsupportedCodestreamFeature, UnsupportedProfile,
};
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
    pub transform_plan: ModularTransformPlan,
    pub channel_metadata: PackedModularChannelMetadata,
    pub inverse_plan: ModularInversePlan,
}

fn validate_image_header(
    codestream: &GpuCodestream,
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
    codestream: &GpuCodestream,
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
    let (ma_config, wp_header, transform_plan) = parse_dc_global_ir(
        &mut reader,
        channels,
        image.width,
        image.height,
        u32::from(bits_per_sample),
    )?;
    validate_stock_modular_transform_plan(
        channels,
        image.width,
        image.height,
        u32::from(bits_per_sample),
        &transform_plan,
    )?;
    let inverse_plan = plan_modular_inverse(&transform_plan)?;
    let channel_metadata = transform_plan
        .topology
        .gpu_entropy_channels(ma_config.maximum_tree_property())?;
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
        transform_plan,
        channel_metadata,
        inverse_plan,
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
    width: u32,
    height: u32,
    bit_depth: u32,
) -> Result<(MaConfigIr, WpHeaderIr, ModularTransformPlan)> {
    expect(reader, 1, 1, "global Modular MA tree")?;
    let ma_config = parse_ma_config(reader, MaTreeLimits::default())?;
    expect(reader, 1, 1, "global Modular tree selection")?;
    let wp_header = WpHeaderIr::parse(reader)?;
    let limits = ModularTransformLimits::default();
    let topology = ModularChannelTopology::full_resolution(
        width,
        height,
        bit_depth,
        channels.count(),
        limits,
    )?;
    let transform_plan = parse_modular_transforms(reader, topology, limits)?;
    Ok((ma_config, wp_header, transform_plan))
}

fn validate_stock_modular_transform_plan(
    channels: ModularChannels,
    width: u32,
    height: u32,
    bit_depth: u32,
    transform_plan: &ModularTransformPlan,
) -> Result<()> {
    let expected_sample_count = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|samples| samples.checked_mul(u64::from(channels.count())))
        .ok_or_else(|| unsupported_error("Modular source sample count overflows"))?;
    let topology_is_direct = transform_plan.topology.meta_channel_count() == 0
        && transform_plan.topology == *transform_plan.source_topology()
        && transform_plan.topology.sample_count() == Some(expected_sample_count)
        && transform_plan.topology.channels().len() == channels.count() as usize
        && transform_plan.topology.channels().iter().all(|channel| {
            channel.width == width
                && channel.height == height
                && channel.hshift == 0
                && channel.vshift == 0
                && channel.bit_depth == bit_depth
        });
    // This proves that every entropy-visible channel has a portable u32 WGSL address before any
    // backend allocation. The generalized transformed-channel executor will retain this table.
    let _gpu_channel_layout = transform_plan.topology.gpu_layout()?;
    match (channels, transform_plan.transforms.as_slice()) {
        (ModularChannels::Gray, []) if topology_is_direct => {
            let inverse = plan_modular_inverse(transform_plan)?;
            if u64::from(inverse.entropy_words()) != expected_sample_count
                || inverse.arena_words() != inverse.entropy_words()
                || !inverse.jobs().is_empty()
                || inverse.final_planes().len() != channels.count() as usize
            {
                return Err(crate::ModularInversePlanError::TopologyState {
                    reason: "direct topology produced a non-direct inverse schedule",
                }
                .into());
            }
        }
        (
            ModularChannels::Rgb | ModularChannels::Rgba,
            [
                ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type: 6,
                }),
            ],
        ) if topology_is_direct => {}
        (ModularChannels::Rgb | ModularChannels::Rgba, []) => {
            return unsupported(
                "the standard Modular color profile requires its reversible color transform",
            );
        }
        (_, transforms) => {
            if transforms.iter().all(|transform| {
                matches!(
                    transform,
                    ModularTransformIr::Rct(_) | ModularTransformIr::Squeeze { .. }
                )
            }) {
                let inverse = plan_modular_inverse(transform_plan)?;
                let expected_jobs =
                    transforms
                        .iter()
                        .try_fold(0usize, |total, transform| match transform {
                            ModularTransformIr::Rct(_) => total.checked_add(1),
                            ModularTransformIr::Squeeze { parameters, .. } => {
                                parameters.iter().try_fold(total, |total, parameter| {
                                    total.checked_add(parameter.channel_count as usize)
                                })
                            }
                            ModularTransformIr::Palette(_) => {
                                unreachable!("the RCT/Squeeze predicate was checked above")
                            }
                        });
                if inverse.arena_words() < inverse.entropy_words()
                    || inverse.arena_bytes() != u64::from(inverse.arena_words()) * 4
                    || Some(inverse.jobs().len()) != expected_jobs
                    || inverse.final_planes().len()
                        != transform_plan.source_topology().channels().len()
                {
                    return Err(crate::ModularInversePlanError::TopologyState {
                        reason: "RCT/Squeeze transform produced inconsistent resident requirements",
                    }
                    .into());
                }
            }
            let feature = transforms
                .iter()
                .find_map(|transform| match transform {
                    ModularTransformIr::Rct(rct) if rct.begin_channel == 0 && rct.rct_type == 6 => {
                        None
                    }
                    ModularTransformIr::Rct(rct) => {
                        Some(ModularTransformFeature::ReversibleColor {
                            begin_channel: rct.begin_channel,
                            rct_type: rct.rct_type,
                        })
                    }
                    ModularTransformIr::Palette(_) => Some(ModularTransformFeature::Palette),
                    ModularTransformIr::Squeeze { .. } => Some(ModularTransformFeature::Squeeze),
                })
                .unwrap_or(ModularTransformFeature::Invalid);
            return unsupported_transform(feature);
        }
    }
    transform_plan.visit_inverse(|_, source, destination| {
        source.gpu_layout()?;
        destination.gpu_layout()?;
        Ok(())
    })?;
    Ok(())
}

fn unsupported_transform<T>(feature: ModularTransformFeature) -> Result<T> {
    Err(UnsupportedProfile::new(
        UnsupportedCodestreamFeature::ModularTransform(feature),
        "the Modular transform is parsed but has not been lowered to a GPU kernel",
    )
    .into())
}

fn validate_empty_non_pass_sections(
    codestream: &GpuCodestream,
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

    use jxl_gpu_bitstream::{
        FrameEncoding, FrameType, InventoryLimits, ParseLimits, StreamSlice, parse,
    };

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
            GpuCodestream::from_spans([(0, StreamSlice::from_shared(Arc::clone(&bytes)))]).unwrap();
        let expected = parse_standard_modular_profile(&complete, &inventory).unwrap();

        for split_offset in 0..=bytes.len() {
            let split_source = GpuCodestream::from_spans([
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

    #[test]
    fn libjxl_progressive_dc_root_has_exact_default_squeeze_topology() {
        let Some(encoded) = cjxl_progressive_dc() else {
            return;
        };
        let parsed = parse(&encoded, ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let frame = &inventory.frames[0];
        assert_eq!(frame.frame_type, FrameType::LowFrequency);
        assert_eq!(frame.encoding, FrameEncoding::Modular);
        assert_eq!(frame.lf_level, 2);
        let section = frame.sections.first().unwrap();
        let bytes: Arc<[u8]> = Arc::from(parsed.codestream());
        let codestream = GpuCodestream::from_spans([(0, StreamSlice::from_shared(bytes))]).unwrap();
        let mut reader = codestream.reader();
        reader.skip_bits(section.bits.offset).unwrap();
        parse_lf_channel_dequantization(&mut reader).unwrap();
        let (_, _, plan) = parse_dc_global_ir(
            &mut reader,
            ModularChannels::Rgb,
            frame.width,
            frame.height,
            8,
        )
        .unwrap();
        let [
            ModularTransformIr::Squeeze {
                used_default_parameters,
                parameters,
            },
        ] = plan.transforms.as_slice()
        else {
            panic!(
                "unexpected progressive-DC root transforms: {:?}",
                plan.transforms
            );
        };
        assert!(*used_default_parameters);
        assert_eq!(parameters.len(), 13);
        assert_eq!(plan.source_topology().channels().len(), 3);
        assert_eq!(plan.topology.channels().len(), 40);
        assert_eq!(
            plan.topology.sample_count(),
            Some(u64::from(frame.width) * u64::from(frame.height) * 3)
        );
        assert_eq!(
            plan.topology.channels()[..3]
                .iter()
                .map(|channel| (channel.width, channel.height))
                .collect::<Vec<_>>(),
            vec![(8, 8), (4, 4), (4, 4)]
        );
        let inverse = plan_modular_inverse(&plan).unwrap();
        assert_eq!(inverse.jobs().len(), 37);
        assert_eq!(inverse.final_planes().len(), 3);
        assert_eq!(
            u64::from(inverse.entropy_words()),
            u64::from(frame.width) * u64::from(frame.height) * 3
        );
        assert!(inverse.arena_words() <= inverse.entropy_words() * 2);
        assert_eq!(
            inverse
                .final_planes()
                .iter()
                .map(|plane| (plane.geometry.width, plane.geometry.height))
                .collect::<Vec<_>>(),
            vec![(frame.width, frame.height); 3]
        );
        assert!(reader.bit_offset() <= section.bits.end().unwrap());
    }

    fn cjxl_progressive_dc() -> Option<Vec<u8>> {
        if std::process::Command::new("cjxl")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("skipping progressive-DC transform oracle: cjxl is not installed");
            return None;
        }
        let nonce = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let ppm_path = std::env::temp_dir().join(format!("jxl-wgpu-transform-{nonce}.ppm"));
        let jxl_path = std::env::temp_dir().join(format!("jxl-wgpu-transform-{nonce}.jxl"));
        let (width, height) = (1_024_u32, 128_u32);
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.reserve((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                ppm.extend_from_slice(&[
                    (x.wrapping_mul(13) + y.wrapping_mul(7)) as u8,
                    (x.wrapping_mul(3) ^ y.wrapping_mul(11)) as u8,
                    (x.wrapping_mul(5) + y.wrapping_mul(17) + (x ^ y)) as u8,
                ]);
            }
        }
        std::fs::write(&ppm_path, ppm).unwrap();
        let output = std::process::Command::new("cjxl")
            .args([
                "-d",
                "2",
                "-e",
                "7",
                "-m",
                "0",
                "--progressive_dc=2",
                "--container=0",
            ])
            .arg(&ppm_path)
            .arg(&jxl_path)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&ppm_path);
        if !output.status.success() {
            let _ = std::fs::remove_file(&jxl_path);
            panic!("cjxl failed: {}", String::from_utf8_lossy(&output.stderr));
        }
        let encoded = std::fs::read(&jxl_path).unwrap();
        let _ = std::fs::remove_file(&jxl_path);
        Some(encoded)
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
