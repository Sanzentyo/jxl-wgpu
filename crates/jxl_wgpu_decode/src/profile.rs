use jxl_gpu_bitstream::{
    CodestreamInventory, FiniteF16, FrameBlendInfo, FrameEncoding, FrameSectionKind, FrameType,
    ImageHeaderInventory, SampleBitDepth,
};

use crate::modular_inverse::{ModularInversePlan, plan_modular_inverse};
use crate::modular_transform::{
    GpuModularChannelLayout, ModularChannelGeometry, ModularChannelTopology, ModularRct,
    ModularTransformIr, ModularTransformLimits, ModularTransformPlan, PackedModularChannelMetadata,
    parse_modular_transforms,
};
use crate::modular_tree::BitInput;
use crate::{
    Error, GpuCodestream, ModularChannels, Result, UnsupportedCodestreamFeature, UnsupportedProfile,
};
use crate::{
    ModularTransformFeature,
    modular_tree::{MaConfigIr, MaTreeLimits, WpHeaderIr, parse_ma_config},
};

const MIN_GROUP_DIMENSION: u32 = 128;
const MAX_MODULAR_PASSES: u32 = 3;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlobalModularStream {
    pub token_bit_offset: u64,
    pub token_bit_end: u64,
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
    pub pass_count: u32,
    pub group_columns: u32,
    pub group_rows: u32,
    /// Canvas pass groups exposed by the decoded frame profile.
    pub groups: Vec<ModularGroup>,
    /// Recursive Modular entropy streams in execution order: LF groups, then pass groups.
    pub entropy_groups: Vec<ModularGroup>,
    pub low_frequency_entropy_group_count: usize,
    pub global_stream: Option<GlobalModularStream>,
    pub ma_config: MaConfigIr,
    pub generalized_channels: bool,
    pub resident_entropy_plans: Vec<ResidentModularGroupPlan>,
    pub resident_frame_plan: Option<ResidentModularFramePlan>,
    pub progressive_dc: Option<ProgressiveDcModularProfile>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProgressiveDcModularProfile {
    pub lf_level: u32,
    /// JPEG XL LF multipliers in X/Y/B order, retained exactly as binary16 values.
    pub lf_dequantization: [FiniteF16; 3],
}

impl ProgressiveDcModularProfile {
    pub(crate) fn lf_dequantization(self) -> [f32; 3] {
        self.lf_dequantization.map(FiniteF16::to_f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModularProfilePurpose {
    Presentation,
    ProgressiveDc,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidentModularGroupPlan {
    pub ma_config: ModularMaConfig,
    pub wp_header: WpHeaderIr,
    pub channel_metadata: PackedModularChannelMetadata,
    pub inverse_plan: ModularInversePlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentModularPlaneCopy {
    pub source: GpuModularChannelLayout,
    pub destination: GpuModularChannelLayout,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidentModularFramePlan {
    pub ma_config: ModularMaConfig,
    pub wp_header: WpHeaderIr,
    pub channel_metadata: PackedModularChannelMetadata,
    pub inverse_plan: ModularInversePlan,
    /// Copy plans parallel to `StandardModularProfile::entropy_groups`.
    pub subimage_plane_copies: Vec<Vec<ResidentModularPlaneCopy>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModularMaConfig {
    Global,
    Local(MaConfigIr),
}

impl ModularMaConfig {
    pub(crate) fn resolve<'a>(&'a self, global: &'a MaConfigIr) -> &'a MaConfigIr {
        match self {
            Self::Global => global,
            Self::Local(local) => local,
        }
    }
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
    parse_modular_profile(codestream, inventory, ModularProfilePurpose::Presentation)
}

pub(crate) fn parse_progressive_dc_modular_profile(
    codestream: &GpuCodestream,
    inventory: &CodestreamInventory,
) -> Result<StandardModularProfile> {
    parse_modular_profile(codestream, inventory, ModularProfilePurpose::ProgressiveDc)
}

fn parse_modular_profile(
    codestream: &GpuCodestream,
    inventory: &CodestreamInventory,
    purpose: ModularProfilePurpose,
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
    if image.animation.is_some() || image.preview_size.is_some() || inventory.frames.len() != 1 {
        return unsupported("the Modular GPU profile requires exactly one still-image frame");
    }
    match purpose {
        ModularProfilePurpose::Presentation => {
            if image.xyb_encoded {
                return unsupported("the lossless Modular GPU profile does not use XYB metadata");
            }
            validate_image_header(codestream, image, channels, bits_per_sample)?;
        }
        ModularProfilePurpose::ProgressiveDc => {
            if !image.xyb_encoded || channels != ModularChannels::Rgb {
                return unsupported(
                    "a progressive-DC Modular producer requires three XYB color channels",
                );
            }
        }
    }

    let frame = &inventory.frames[0];
    let shared_frame_is_invalid = frame.is_preview
        || frame.encoding != FrameEncoding::Modular
        || frame.flags != 0
        || frame.do_ycbcr
        || frame.jpeg_upsampling != [0; 3]
        || frame.upsampling != 1
        || frame.group_size_shift > 3
        || frame.have_crop
        || frame.x0 != 0
        || frame.y0 != 0
        || frame.width != image.width
        || frame.height != image.height
        || frame.duration_ticks != 0
        || frame.save_as_reference != 0
        || !frame.name_bytes.is_empty()
        || frame.color_blend != FrameBlendInfo::default()
        || frame.extra_channel_blends
            != vec![FrameBlendInfo::default(); usize::from(channels == ModularChannels::Rgba)];
    let role_is_invalid = match purpose {
        ModularProfilePurpose::Presentation => {
            frame.frame_type != FrameType::Regular
                || frame.lf_level != 0
                || frame.lf_source_frame.is_some()
                || !frame.is_last
                || frame.save_before_color_transform
        }
        ModularProfilePurpose::ProgressiveDc => {
            frame.frame_type != FrameType::LowFrequency
                || frame.lf_level == 0
                || frame.lf_source_frame.is_some()
                || frame.is_last
                || !frame.save_before_color_transform
        }
    };
    if shared_frame_is_invalid || role_is_invalid {
        return unsupported(match purpose {
            ModularProfilePurpose::Presentation => {
                "the lossless Modular GPU profile requires one final uncropped regular frame with canonical grouping, replace blending, and no references"
            }
            ModularProfilePurpose::ProgressiveDc => {
                "the progressive-DC Modular GPU profile requires one uncropped root LF frame with canonical grouping and XYB reference retention"
            }
        });
    }
    if !(1..=MAX_MODULAR_PASSES).contains(&frame.num_passes) {
        return Err(UnsupportedProfile::new(
            UnsupportedCodestreamFeature::MultiplePasses,
            format!(
                "the lossless Modular GPU frontend accepts one through {MAX_MODULAR_PASSES} passes"
            ),
        )
        .into());
    }

    let (frame_width, frame_height) = match purpose {
        ModularProfilePurpose::Presentation => (image.width, image.height),
        ModularProfilePurpose::ProgressiveDc => frame
            .color_sample_extent()
            .ok_or_else(|| unsupported_error("progressive-DC frame extent is invalid"))?,
    };
    if frame_width == 0 || frame_height == 0 {
        return unsupported("the Modular GPU profile requires a non-empty frame extent");
    }

    let group_dimension = MIN_GROUP_DIMENSION
        .checked_shl(frame.group_size_shift)
        .ok_or_else(|| unsupported_error("Modular group dimension overflows u32"))?;
    let group_columns = frame_width.div_ceil(group_dimension);
    let group_rows = frame_height.div_ceil(group_dimension);
    let expected_group_count = u64::from(group_columns)
        .checked_mul(u64::from(group_rows))
        .ok_or_else(|| unsupported_error("Modular group grid overflow"))?;
    if frame.group_count != expected_group_count {
        return unsupported("frame inventory group count does not match the Modular canvas");
    }

    let single_entry = expected_group_count == 1 && frame.num_passes == 1;
    let dc_global = if single_entry {
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
    let lf_dequantization = parse_lf_channel_dequantization(&mut reader)?;
    let (ma_config, has_global_ma_config, dc_ma_config, wp_header, transform_plan) =
        parse_dc_global_ir(
            &mut reader,
            channels,
            frame_width,
            frame_height,
            u32::from(bits_per_sample),
        )?;
    let dc_end = dc_global
        .bits
        .end()
        .ok_or_else(|| unsupported_error("DC-global bit range overflow"))?;
    if reader.bit_offset() > dc_end {
        return unsupported("Modular DC-global metadata exceeds its TOC section");
    }

    let (
        groups,
        entropy_groups,
        low_frequency_entropy_group_count,
        concrete_transform_plans,
        frame_plan_seed,
    ) = if single_entry {
        validate_stock_modular_transform_plan(
            channels,
            frame_width,
            frame_height,
            u32::from(bits_per_sample),
            &transform_plan,
        )?;
        let groups = vec![ModularGroup {
            token_bit_offset: reader.bit_offset(),
            token_bit_end: dc_end,
            x: 0,
            y: 0,
            width: frame_width,
            height: frame_height,
            stream_index: 0,
        }];
        (
            groups.clone(),
            groups,
            0,
            vec![(transform_plan.clone(), wp_header, dc_ma_config.clone())],
            None,
        )
    } else {
        validate_stock_modular_transform_plan(
            channels,
            frame_width,
            frame_height,
            u32::from(bits_per_sample),
            &transform_plan,
        )?;
        validate_modular_section_structure(codestream, frame)?;
        let limits = ModularTransformLimits::default();
        let pass_ranges = build_modular_pass_shift_ranges(
            frame.num_passes,
            &frame.progressive_passes.downsampling,
            &frame.progressive_passes.last_pass,
        )?;
        let requires_frame_arena = transform_plan
            .transforms
            .iter()
            .any(|transform| !matches!(transform, ModularTransformIr::Rct(_)));
        let global_channel_count =
            global_subimage_channel_count(&transform_plan.topology, group_dimension);
        for channel in &transform_plan.topology.channels()[global_channel_count..] {
            let (Ok(hshift), Ok(vshift)) =
                (u32::try_from(channel.hshift), u32::try_from(channel.vshift))
            else {
                return unsupported("a Modular channel shift is negative");
            };
            if (hshift < 3 || vshift < 3)
                && modular_pass_for_channel(hshift, vshift, &pass_ranges).is_none()
            {
                return unsupported(
                    "the Modular progressive-pass ranges do not cover a transformed channel",
                );
            }
        }
        let has_lf_group_channels = transform_plan.topology.channels()[global_channel_count..]
            .iter()
            .any(|channel| channel.hshift >= 3 && channel.vshift >= 3);
        let lf_group_dimension = group_dimension
            .checked_mul(8)
            .ok_or_else(|| unsupported_error("Modular LF-group dimension overflow"))?;
        let lf_group_columns = frame_width.div_ceil(lf_group_dimension);
        let lf_group_rows = frame_height.div_ceil(lf_group_dimension);
        let expected_lf_group_count = u64::from(lf_group_columns)
            .checked_mul(u64::from(lf_group_rows))
            .ok_or_else(|| unsupported_error("Modular LF-group grid overflow"))?;
        if frame.low_frequency_group_count != expected_lf_group_count {
            return unsupported("frame inventory LF-group count does not match the Modular canvas");
        }
        let global_topology = ModularChannelTopology::new(
            transform_plan.topology.channels()[..global_channel_count].to_vec(),
            transform_plan
                .topology
                .meta_channel_count()
                .min(global_channel_count),
            limits,
        )?;
        let global_channel_metadata = global_topology
            .gpu_entropy_channels(dc_ma_config.resolve(&ma_config).maximum_tree_property())?;
        let frame_inverse_plan = plan_modular_inverse(&transform_plan)?;
        let frame_entropy_layout = transform_plan.topology.gpu_layout()?;
        let group_count = usize::try_from(expected_group_count)
            .map_err(|_| unsupported_error("Modular group count exceeds host address space"))?;
        let pass_count = usize::try_from(frame.num_passes)
            .map_err(|_| unsupported_error("Modular pass count exceeds host address space"))?;
        let canvas_groups = (0..group_count)
            .map(|group_index| {
                let group_index_u64 = u64::try_from(group_index)
                    .map_err(|_| unsupported_error("Modular group index exceeds u64"))?;
                let column = u32::try_from(group_index_u64 % u64::from(group_columns))
                    .map_err(|_| unsupported_error("Modular group column overflow"))?;
                let row = u32::try_from(group_index_u64 / u64::from(group_columns))
                    .map_err(|_| unsupported_error("Modular group row overflow"))?;
                let x = column
                    .checked_mul(group_dimension)
                    .ok_or_else(|| unsupported_error("Modular group x origin overflow"))?;
                let y = row
                    .checked_mul(group_dimension)
                    .ok_or_else(|| unsupported_error("Modular group y origin overflow"))?;
                let width = frame_width.saturating_sub(x).min(group_dimension);
                let height = frame_height.saturating_sub(y).min(group_dimension);
                Ok(ModularGroup {
                    token_bit_offset: 0,
                    token_bit_end: 0,
                    x,
                    y,
                    width,
                    height,
                    // Canvas groups have no entropy section; expose the final-pass stream ID
                    // as their stable public stream identity.
                    stream_index: modular_pass_stream_index(
                        frame.low_frequency_group_count,
                        expected_group_count,
                        u64::from(frame.num_passes - 1),
                        group_index_u64,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut pass_sections = (0..pass_count)
            .map(|_| (0..group_count).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<Vec<Option<jxl_gpu_bitstream::FrameSection>>>>();
        for section in &frame.sections {
            let FrameSectionKind::PassGroup {
                pass_index,
                group_index,
            } = section.kind
            else {
                continue;
            };
            let pass_index = usize::try_from(pass_index)
                .map_err(|_| unsupported_error("Modular pass index exceeds host address space"))?;
            let group_index = usize::try_from(group_index)
                .map_err(|_| unsupported_error("Modular group index exceeds host address space"))?;
            let slot = pass_sections
                .get_mut(pass_index)
                .and_then(|groups| groups.get_mut(group_index))
                .ok_or_else(|| {
                    unsupported_error("Modular pass-group index exceeds the frame grid")
                })?;
            if slot.is_some() {
                return unsupported("the Modular frame contains a duplicate pass-group section");
            }
            *slot = Some(*section);
        }

        let mut pass_groups = (0..pass_count)
            .map(|_| (0..group_count).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<Vec<Option<ModularGroup>>>>();
        let mut pass_transform_plans = (0..pass_count)
            .map(|_| (0..group_count).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<Vec<Option<(ModularTransformPlan, WpHeaderIr, ModularMaConfig)>>>>();
        let mut pass_plane_targets = (0..pass_count)
            .map(|_| (0..group_count).map(|_| None).collect::<Vec<_>>())
            .collect::<Vec<Vec<Option<Vec<GpuModularChannelLayout>>>>>();
        for (pass_index, pass_groups_for_pass) in pass_groups.iter_mut().enumerate() {
            let pass_range = pass_ranges[pass_index];
            for (group_index, group_slot) in pass_groups_for_pass.iter_mut().enumerate() {
                let section = pass_sections[pass_index][group_index].ok_or_else(|| {
                    unsupported_error(format!(
                        "the Modular frame is missing pass-group {pass_index}/{group_index}"
                    ))
                })?;
                let canvas_group = canvas_groups[group_index];
                let column = group_index as u32 % group_columns;
                let row = group_index as u32 / group_columns;
                let (assigned_topology, plane_targets) = grouped_subimage_topology(
                    &transform_plan.topology,
                    &frame_entropy_layout,
                    global_channel_count,
                    ModularSubimageRegion {
                        kind: ModularSubimageKind::PassGroup,
                        column,
                        row,
                        group_dimension,
                    },
                    pass_range,
                    limits,
                )?;
                if assigned_topology.channels().is_empty() {
                    validate_empty_pass_group_section(
                        codestream,
                        section,
                        pass_index,
                        group_index,
                    )?;
                    continue;
                }
                let group_width = canvas_group.width;
                let group_height = canvas_group.height;
                let source_topology = if requires_frame_arena {
                    assigned_topology
                } else {
                    ModularChannelTopology::full_resolution(
                        group_width,
                        group_height,
                        u32::from(bits_per_sample),
                        channels.count(),
                        limits,
                    )?
                };
                let mut group_reader = codestream.reader();
                group_reader.skip_bits(section.bits.offset)?;
                let use_global_tree = group_reader.read_bits(1)? != 0;
                let local_wp_header = WpHeaderIr::parse(&mut group_reader)?;
                let local_transform = parse_modular_transforms(
                    &mut group_reader,
                    if requires_frame_arena {
                        source_topology.clone()
                    } else {
                        ModularTransformPlan::from_ir(
                            source_topology.clone(),
                            transform_plan.transforms.clone(),
                            limits,
                        )?
                        .topology
                    },
                    limits,
                )?;
                let concrete_transform = if requires_frame_arena {
                    local_transform
                } else {
                    let mut transforms = transform_plan.transforms.clone();
                    transforms.extend(local_transform.transforms);
                    ModularTransformPlan::from_ir(source_topology, transforms, limits)?
                };
                if !requires_frame_arena {
                    validate_stock_modular_transform_plan(
                        channels,
                        group_width,
                        group_height,
                        u32::from(bits_per_sample),
                        &concrete_transform,
                    )?;
                }
                let group_ma_config = if use_global_tree {
                    if !has_global_ma_config {
                        return unsupported(
                            "a Modular pass-group selects a global MA tree that is absent",
                        );
                    }
                    ModularMaConfig::Global
                } else {
                    ModularMaConfig::Local(parse_ma_config(
                        &mut group_reader,
                        MaTreeLimits::default(),
                    )?)
                };
                let token_bit_end = section
                    .bits
                    .end()
                    .ok_or_else(|| unsupported_error("Modular pass-group bit range overflow"))?;
                *group_slot = Some(ModularGroup {
                    token_bit_offset: group_reader.bit_offset(),
                    token_bit_end,
                    x: canvas_group.x,
                    y: canvas_group.y,
                    width: group_width,
                    height: group_height,
                    stream_index: modular_pass_stream_index(
                        frame.low_frequency_group_count,
                        expected_group_count,
                        u64::try_from(pass_index)
                            .map_err(|_| unsupported_error("Modular pass index exceeds u64"))?,
                        u64::try_from(group_index)
                            .map_err(|_| unsupported_error("Modular group index exceeds u64"))?,
                    )?,
                });
                pass_transform_plans[pass_index][group_index] =
                    Some((concrete_transform, local_wp_header, group_ma_config));
                pass_plane_targets[pass_index][group_index] = Some(plane_targets);
            }
        }

        let (lf_groups, lf_transform_plans, lf_plane_targets) = if has_lf_group_channels {
            let lf_group_count = usize::try_from(expected_lf_group_count).map_err(|_| {
                unsupported_error("Modular LF-group count exceeds host address space")
            })?;
            let mut lf_groups = vec![None; lf_group_count];
            let mut lf_transform_plans = vec![None; lf_group_count];
            let mut lf_plane_targets = vec![None; lf_group_count];
            for section in &frame.sections {
                let FrameSectionKind::LowFrequencyGroup { group_index } = section.kind else {
                    continue;
                };
                let index = usize::try_from(group_index).map_err(|_| {
                    unsupported_error("Modular LF-group index exceeds host address space")
                })?;
                let slot = lf_groups.get_mut(index).ok_or_else(|| {
                    unsupported_error("Modular LF-group index exceeds the frame grid")
                })?;
                if slot.is_some() {
                    return unsupported("the Modular frame contains a duplicate LF-group section");
                }
                let column = u32::try_from(group_index % u64::from(lf_group_columns))
                    .map_err(|_| unsupported_error("Modular LF-group column overflow"))?;
                let row = u32::try_from(group_index / u64::from(lf_group_columns))
                    .map_err(|_| unsupported_error("Modular LF-group row overflow"))?;
                let x = column
                    .checked_mul(lf_group_dimension)
                    .ok_or_else(|| unsupported_error("Modular LF-group x origin overflow"))?;
                let y = row
                    .checked_mul(lf_group_dimension)
                    .ok_or_else(|| unsupported_error("Modular LF-group y origin overflow"))?;
                let width = frame_width.saturating_sub(x).min(lf_group_dimension);
                let height = frame_height.saturating_sub(y).min(lf_group_dimension);
                let (source_topology, plane_targets) = grouped_subimage_topology(
                    &transform_plan.topology,
                    &frame_entropy_layout,
                    global_channel_count,
                    ModularSubimageRegion {
                        kind: ModularSubimageKind::LowFrequencyGroup,
                        column,
                        row,
                        group_dimension,
                    },
                    None,
                    limits,
                )?;
                if source_topology.channels().is_empty() {
                    return Err(Error::EngineContract(
                        "a scheduled Modular LF group has no transformed channels",
                    ));
                }
                let mut group_reader = codestream.reader();
                group_reader.skip_bits(section.bits.offset)?;
                let use_global_tree = group_reader.read_bits(1)? != 0;
                let local_wp_header = WpHeaderIr::parse(&mut group_reader)?;
                let local_transform =
                    parse_modular_transforms(&mut group_reader, source_topology, limits)?;
                let group_ma_config = if use_global_tree {
                    if !has_global_ma_config {
                        return unsupported(
                            "a Modular LF group selects a global MA tree that is absent",
                        );
                    }
                    ModularMaConfig::Global
                } else {
                    ModularMaConfig::Local(parse_ma_config(
                        &mut group_reader,
                        MaTreeLimits::default(),
                    )?)
                };
                let token_bit_end = section
                    .bits
                    .end()
                    .ok_or_else(|| unsupported_error("Modular LF-group bit range overflow"))?;
                *slot = Some(ModularGroup {
                    token_bit_offset: group_reader.bit_offset(),
                    token_bit_end,
                    x,
                    y,
                    width,
                    height,
                    stream_index: u32::try_from(
                        1u64.checked_add(frame.low_frequency_group_count)
                            .and_then(|value| value.checked_add(group_index))
                            .ok_or_else(|| unsupported_error("Modular LF stream index overflow"))?,
                    )
                    .map_err(|_| unsupported_error("Modular LF stream index exceeds u32"))?,
                });
                lf_transform_plans[index] =
                    Some((local_transform, local_wp_header, group_ma_config));
                lf_plane_targets[index] = Some(plane_targets);
            }
            (
                collect_required_entries(lf_groups, "LF-group stream")?,
                collect_required_entries(lf_transform_plans, "LF-group transform plan")?,
                collect_required_entries(lf_plane_targets, "LF-group plane targets")?,
            )
        } else {
            validate_empty_lf_group_sections(codestream, frame)?;
            (Vec::new(), Vec::new(), Vec::new())
        };

        let groups = canvas_groups;
        let mut entropy_groups = lf_groups;
        let low_frequency_entropy_group_count = entropy_groups.len();
        let mut entropy_transform_plans = lf_transform_plans;
        let mut subimage_plane_targets = lf_plane_targets;
        for pass_index in 0..pass_count {
            for group_index in 0..group_count {
                if let Some(group) = pass_groups[pass_index][group_index].take() {
                    entropy_groups.push(group);
                    entropy_transform_plans.push(
                        pass_transform_plans[pass_index][group_index].take().ok_or(
                            Error::EngineContract("Modular pass-group transform plan is missing"),
                        )?,
                    );
                    subimage_plane_targets.push(
                        pass_plane_targets[pass_index][group_index].take().ok_or(
                            Error::EngineContract("Modular pass-group plane targets are missing"),
                        )?,
                    );
                }
            }
        }
        let frame_plan_seed = requires_frame_arena.then_some((
            wp_header,
            dc_ma_config,
            global_channel_metadata,
            frame_inverse_plan,
            subimage_plane_targets,
        ));
        (
            groups,
            entropy_groups,
            low_frequency_entropy_group_count,
            entropy_transform_plans,
            frame_plan_seed,
        )
    };
    let codestream_bits = codestream.logical_bits()?;
    for group in &entropy_groups {
        if group.width == 0
            || group.height == 0
            || group.token_bit_offset > group.token_bit_end
            || group.token_bit_end > codestream_bits
        {
            return unsupported("the Modular frame contains an invalid group token range");
        }
        group.sample_count()?;
    }

    let generalized_channels = frame_plan_seed.is_some()
        || concrete_transform_plans
            .iter()
            .any(|(plan, _, _)| uses_generalized_channel_layout(channels, plan));
    let resident_entropy_plans = concrete_transform_plans
        .into_iter()
        .map(|(group_transform, group_wp_header, group_ma_config)| {
            let concrete_ma_config = group_ma_config.resolve(&ma_config);
            let channel_metadata = group_transform
                .topology
                .gpu_entropy_channels(concrete_ma_config.maximum_tree_property())?;
            let inverse_plan = plan_modular_inverse(&group_transform)?;
            Ok(ResidentModularGroupPlan {
                ma_config: group_ma_config,
                wp_header: group_wp_header,
                channel_metadata,
                inverse_plan,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let resident_frame_plan = frame_plan_seed
        .map(
            |(wp_header, ma_config, channel_metadata, inverse_plan, group_plane_targets)| {
                let subimage_plane_copies = resident_entropy_plans
                    .iter()
                    .zip(group_plane_targets)
                    .map(|(group_plan, targets)| {
                        let sources = group_plan.inverse_plan.final_gpu_layouts();
                        if sources.len() != targets.len() {
                            return Err(Error::EngineContract(
                                "Modular subimage inverse planes do not match frame-arena targets",
                            ));
                        }
                        Ok(sources
                            .into_iter()
                            .zip(targets)
                            .map(|(source, destination)| ResidentModularPlaneCopy {
                                source,
                                destination,
                            })
                            .collect())
                    })
                    .collect::<Result<Vec<Vec<_>>>>()?;
                Ok::<_, Error>(ResidentModularFramePlan {
                    ma_config,
                    wp_header,
                    channel_metadata,
                    inverse_plan,
                    subimage_plane_copies,
                })
            },
        )
        .transpose()?;

    let global_decoded_words = resident_frame_plan
        .as_ref()
        .and_then(|plan| plan.channel_metadata.channels.last())
        .map_or(0, |channel| channel.decoded_end);
    let global_stream = should_schedule_global_modular_stream(
        !single_entry,
        reader.bit_offset(),
        dc_end,
        global_decoded_words,
    )
    .then_some(GlobalModularStream {
        token_bit_offset: reader.bit_offset(),
        token_bit_end: dc_end,
    });

    Ok(StandardModularProfile {
        width: frame_width,
        height: frame_height,
        bits_per_sample,
        channels,
        pass_count: frame.num_passes,
        group_columns,
        group_rows,
        groups,
        entropy_groups,
        low_frequency_entropy_group_count,
        global_stream,
        ma_config,
        generalized_channels,
        resident_entropy_plans,
        resident_frame_plan,
        progressive_dc: (purpose == ModularProfilePurpose::ProgressiveDc).then_some(
            ProgressiveDcModularProfile {
                lf_level: frame.lf_level,
                lf_dequantization,
            },
        ),
    })
}

const fn should_schedule_global_modular_stream(
    multi_group: bool,
    token_bit_offset: u64,
    token_bit_end: u64,
    decoded_words: u32,
) -> bool {
    multi_group && (token_bit_offset < token_bit_end || decoded_words != 0)
}

fn modular_pass_stream_index(
    low_frequency_group_count: u64,
    group_count: u64,
    pass_index: u64,
    group_index: u64,
) -> Result<u32> {
    let stream_index = 18u64
        .checked_add(
            low_frequency_group_count
                .checked_mul(3)
                .ok_or_else(|| unsupported_error("Modular stream index overflow"))?,
        )
        .and_then(|value| {
            pass_index
                .checked_mul(group_count)
                .and_then(|pass_offset| value.checked_add(pass_offset))
        })
        .and_then(|value| value.checked_add(group_index))
        .ok_or_else(|| unsupported_error("Modular stream index overflow"))?;
    u32::try_from(stream_index)
        .map_err(|_| unsupported_error("Modular stream index exceeds u32").into())
}

fn validate_empty_pass_group_section(
    codestream: &GpuCodestream,
    section: jxl_gpu_bitstream::FrameSection,
    pass_index: usize,
    group_index: usize,
) -> Result<()> {
    let end = section
        .bits
        .end()
        .ok_or_else(|| unsupported_error("Modular pass-group bit range overflow"))?;
    if !codestream.bits_are_zero(section.bits.offset, end)? {
        return unsupported(format!(
            "the Modular pass-group {pass_index}/{group_index} has no assigned channels but is nonempty"
        ));
    }
    Ok(())
}

fn collect_required_entries<T>(entries: Vec<Option<T>>, name: &'static str) -> Result<Vec<T>> {
    entries
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            entry.ok_or_else(|| {
                unsupported_error(format!("the Modular frame is missing {name} {index}")).into()
            })
        })
        .collect()
}

pub(crate) fn uses_generalized_channel_layout(
    channels: ModularChannels,
    transform_plan: &ModularTransformPlan,
) -> bool {
    !matches!(
        (channels, transform_plan.transforms.as_slice()),
        (ModularChannels::Gray, [])
            | (
                ModularChannels::Rgb | ModularChannels::Rgba,
                [ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type: 6,
                })],
            )
    )
}

fn global_subimage_channel_count(topology: &ModularChannelTopology, group_dimension: u32) -> usize {
    topology
        .channels()
        .iter()
        .enumerate()
        .take_while(|(index, channel)| {
            *index < topology.meta_channel_count()
                || (channel.width <= group_dimension && channel.height <= group_dimension)
        })
        .count()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModularSubimageKind {
    LowFrequencyGroup,
    PassGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModularSubimageRegion {
    kind: ModularSubimageKind,
    column: u32,
    row: u32,
    group_dimension: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModularPassShiftRange {
    min_shift: u32,
    max_shift: u32,
}

impl ModularPassShiftRange {
    fn contains(self, shift: u32) -> bool {
        (self.min_shift..self.max_shift).contains(&shift)
    }
}

/// Builds the shift brackets used by `jxl-modular`'s `prepare_groups`.
///
/// A missing entry denotes a pass with no shift bracket. Such a pass still has a physical
/// PassGroup section, but the section must be empty. The final pass is always assigned the
/// remaining `[0, max_shift)` bracket, including when that bracket is empty.
fn build_modular_pass_shift_ranges(
    pass_count: u32,
    downsampling: &[u32],
    last_pass: &[u32],
) -> Result<Vec<Option<ModularPassShiftRange>>> {
    if pass_count == 0 {
        return unsupported("the Modular frame declares zero progressive passes");
    }
    if downsampling.len() != last_pass.len()
        || downsampling.len() >= usize::try_from(pass_count).unwrap_or(usize::MAX)
    {
        return unsupported("the Modular frame has inconsistent progressive-pass metadata");
    }

    let pass_count = usize::try_from(pass_count)
        .map_err(|_| unsupported_error("Modular pass count exceeds host address space"))?;
    let mut ranges = vec![None; pass_count];
    let mut max_shift = 3;
    for (&downsample, &pass) in downsampling.iter().zip(last_pass) {
        if downsample == 0 {
            return unsupported("the Modular frame has a zero progressive downsampling factor");
        }
        let pass = usize::try_from(pass).map_err(|_| {
            unsupported_error("Modular progressive-pass index exceeds host address space")
        })?;
        let range = ranges.get_mut(pass).ok_or_else(|| {
            unsupported_error("Modular progressive-pass index exceeds the pass count")
        })?;
        if range.is_some() {
            return unsupported("the Modular frame has duplicate progressive-pass boundaries");
        }
        let min_shift = downsample.trailing_zeros();
        if min_shift > max_shift {
            return unsupported("the Modular frame has non-monotonic progressive downsampling");
        }
        *range = Some(ModularPassShiftRange {
            min_shift,
            max_shift,
        });
        max_shift = min_shift;
    }
    ranges[pass_count - 1] = Some(ModularPassShiftRange {
        min_shift: 0,
        max_shift,
    });
    Ok(ranges)
}

fn modular_pass_for_channel(
    hshift: u32,
    vshift: u32,
    pass_ranges: &[Option<ModularPassShiftRange>],
) -> Option<usize> {
    if hshift >= 3 && vshift >= 3 {
        return None;
    }
    let shift = hshift.min(vshift);
    pass_ranges
        .iter()
        .position(|range| range.is_some_and(|range| range.contains(shift)))
}

fn grouped_subimage_topology(
    frame_topology: &ModularChannelTopology,
    frame_layout: &[GpuModularChannelLayout],
    global_channel_count: usize,
    region: ModularSubimageRegion,
    pass_range: Option<ModularPassShiftRange>,
    limits: ModularTransformLimits,
) -> Result<(ModularChannelTopology, Vec<GpuModularChannelLayout>)> {
    if frame_topology.channels().len() != frame_layout.len()
        || global_channel_count > frame_layout.len()
    {
        return Err(Error::EngineContract(
            "frame Modular topology and packed layout disagree",
        ));
    }

    let mut channels = Vec::new();
    let mut targets = Vec::new();
    for (&channel, &layout) in frame_topology.channels()[global_channel_count..]
        .iter()
        .zip(&frame_layout[global_channel_count..])
    {
        let (Ok(hshift), Ok(vshift)) =
            (u32::try_from(channel.hshift), u32::try_from(channel.vshift))
        else {
            return unsupported(
                "a DC-global meta channel was not retained in the global Modular subimage",
            );
        };
        let low_frequency = hshift >= 3 && vshift >= 3;
        if low_frequency != (region.kind == ModularSubimageKind::LowFrequencyGroup) {
            continue;
        }
        if region.kind == ModularSubimageKind::PassGroup
            && !pass_range.is_some_and(|range| range.contains(hshift.min(vshift)))
        {
            continue;
        }
        let (tile_hshift, tile_vshift) = if low_frequency {
            (hshift - 3, vshift - 3)
        } else {
            (hshift, vshift)
        };
        let tile_width = region
            .group_dimension
            .checked_shr(tile_hshift)
            .filter(|value| *value != 0)
            .ok_or_else(|| unsupported_error("Modular horizontal channel shift is too large"))?;
        let tile_height = region
            .group_dimension
            .checked_shr(tile_vshift)
            .filter(|value| *value != 0)
            .ok_or_else(|| unsupported_error("Modular vertical channel shift is too large"))?;
        let origin_x = region
            .column
            .checked_mul(tile_width)
            .ok_or_else(|| unsupported_error("Modular transformed group x origin overflow"))?;
        let origin_y = region
            .row
            .checked_mul(tile_height)
            .ok_or_else(|| unsupported_error("Modular transformed group y origin overflow"))?;
        let width = channel.width.saturating_sub(origin_x).min(tile_width);
        let height = channel.height.saturating_sub(origin_y).min(tile_height);
        if width == 0 || height == 0 {
            continue;
        }
        channels.push(ModularChannelGeometry::new(
            width,
            height,
            channel.hshift,
            channel.vshift,
            channel.bit_depth,
        ));
        let word_offset = origin_y
            .checked_mul(layout.row_stride_words)
            .and_then(|offset| offset.checked_add(origin_x))
            .and_then(|offset| offset.checked_add(layout.word_offset))
            .ok_or_else(|| unsupported_error("Modular frame-arena plane offset overflow"))?;
        targets.push(GpuModularChannelLayout {
            word_offset,
            row_stride_words: layout.row_stride_words,
            width,
            height,
            hshift: channel.hshift,
            vshift: channel.vshift,
            bit_depth: channel.bit_depth,
            reserved: 0,
        });
    }
    Ok((ModularChannelTopology::new(channels, 0, limits)?, targets))
}

/// Parses `LfChannelDequantization`, which precedes `GlobalModular` in LF-global.
fn parse_lf_channel_dequantization(reader: &mut impl BitInput) -> Result<[FiniteF16; 3]> {
    let all_default = reader.read_bits(1)? != 0;
    if all_default {
        return Ok([0x2800_u16, 0x3400, 0x3800].map(|bits| {
            FiniteF16::from_bits(bits).expect("JPEG XL default LF multipliers are finite")
        }));
    }
    let mut values = [FiniteF16::default(); 3];
    for value in &mut values {
        let bits = u16::try_from(reader.read_bits(16)?)
            .map_err(|_| unsupported_error("LF channel dequantization exceeds binary16"))?;
        *value = FiniteF16::from_bits(bits)
            .ok_or_else(|| unsupported_error("LF channel dequantization is non-finite"))?;
    }
    Ok(values)
}

fn parse_dc_global_ir(
    reader: &mut impl BitInput,
    channels: ModularChannels,
    width: u32,
    height: u32,
    bit_depth: u32,
) -> Result<(
    MaConfigIr,
    bool,
    ModularMaConfig,
    WpHeaderIr,
    ModularTransformPlan,
)> {
    let global_ma_config = (reader.read_bits(1)? != 0)
        .then(|| parse_ma_config(reader, MaTreeLimits::default()))
        .transpose()?;
    let use_global_tree = reader.read_bits(1)? != 0;
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
    let local_ma_config = (!use_global_tree)
        .then(|| parse_ma_config(reader, MaTreeLimits::default()))
        .transpose()?;
    match (global_ma_config, local_ma_config) {
        (Some(global), None) => Ok((
            global,
            true,
            ModularMaConfig::Global,
            wp_header,
            transform_plan,
        )),
        (Some(global), Some(local)) => Ok((
            global,
            true,
            ModularMaConfig::Local(local),
            wp_header,
            transform_plan,
        )),
        (None, Some(local)) => Ok((
            local,
            false,
            ModularMaConfig::Global,
            wp_header,
            transform_plan,
        )),
        (None, None) => {
            unsupported("the DC-global Modular stream selects a global MA tree that is absent")
        }
    }
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
        (_, []) if topology_is_direct => {
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
        (_, transforms) => {
            let gpu_resident = transforms.iter().all(|transform| {
                matches!(
                    transform,
                    ModularTransformIr::Rct(_)
                        | ModularTransformIr::Palette(_)
                        | ModularTransformIr::Squeeze { .. }
                )
            });
            if gpu_resident {
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
                            ModularTransformIr::Palette(palette) => {
                                total.checked_add(palette.channel_count as usize)
                            }
                        });
                if inverse.arena_words() < inverse.entropy_words()
                    || inverse.arena_bytes() != u64::from(inverse.arena_words()) * 4
                    || Some(inverse.jobs().len()) != expected_jobs
                    || inverse.final_planes().len()
                        != transform_plan.source_topology().channels().len()
                {
                    return Err(crate::ModularInversePlanError::TopologyState {
                        reason: "Modular transform produced inconsistent resident requirements",
                    }
                    .into());
                }
            } else {
                let feature = transforms
                    .iter()
                    .find_map(|transform| match transform {
                        ModularTransformIr::Rct(rct)
                            if rct.begin_channel == 0 && rct.rct_type == 6 =>
                        {
                            None
                        }
                        ModularTransformIr::Rct(rct) => {
                            Some(ModularTransformFeature::ReversibleColor {
                                begin_channel: rct.begin_channel,
                                rct_type: rct.rct_type,
                            })
                        }
                        ModularTransformIr::Palette(_) => Some(ModularTransformFeature::Palette),
                        ModularTransformIr::Squeeze { .. } => {
                            Some(ModularTransformFeature::Squeeze)
                        }
                    })
                    .unwrap_or(ModularTransformFeature::Invalid);
                return unsupported_transform(feature);
            }
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

fn validate_modular_section_structure(
    codestream: &GpuCodestream,
    frame: &jxl_gpu_bitstream::FrameInventory,
) -> Result<()> {
    for section in &frame.sections {
        if section.kind == FrameSectionKind::HighFrequencyGlobal {
            let end = section
                .bits
                .end()
                .ok_or_else(|| unsupported_error("Modular section bit range overflow"))?;
            if !codestream.bits_are_zero(section.bits.offset, end)? {
                return unsupported(
                    "the lossless Modular profile requires an empty HF-global section",
                );
            }
        }
    }
    Ok(())
}

fn validate_empty_lf_group_sections(
    codestream: &GpuCodestream,
    frame: &jxl_gpu_bitstream::FrameInventory,
) -> Result<()> {
    for section in &frame.sections {
        if let FrameSectionKind::LowFrequencyGroup { .. } = section.kind {
            let end = section
                .bits
                .end()
                .ok_or_else(|| unsupported_error("Modular LF-group bit range overflow"))?;
            if !codestream.bits_are_zero(section.bits.offset, end)? {
                return unsupported(
                    "the Modular transform topology has no LF-group channels but an LF-group section is nonempty",
                );
            }
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
        BitRange, FrameEncoding, FrameSection, FrameSectionKind, FrameType, InventoryLimits,
        ParseLimits, StreamSlice, parse,
    };

    use super::*;

    #[test]
    fn nonempty_zero_bit_global_prefix_stream_is_still_scheduled() {
        assert!(should_schedule_global_modular_stream(true, 91, 91, 1));
        assert!(!should_schedule_global_modular_stream(true, 91, 91, 0));
        assert!(!should_schedule_global_modular_stream(false, 91, 92, 1));
    }

    #[test]
    fn progressive_pass_ranges_match_jxl_modular_and_assign_asymmetric_shifts() {
        let ranges = build_modular_pass_shift_ranges(3, &[8, 4], &[0, 1]).unwrap();
        assert_eq!(
            ranges,
            vec![
                Some(ModularPassShiftRange {
                    min_shift: 3,
                    max_shift: 3,
                }),
                Some(ModularPassShiftRange {
                    min_shift: 2,
                    max_shift: 3,
                }),
                Some(ModularPassShiftRange {
                    min_shift: 0,
                    max_shift: 2,
                }),
            ]
        );
        assert_eq!(modular_pass_for_channel(4, 0, &ranges), Some(2));
        assert_eq!(modular_pass_for_channel(2, 4, &ranges), Some(1));
        assert_eq!(modular_pass_for_channel(1, 1, &ranges), Some(2));
        assert_eq!(modular_pass_for_channel(3, 3, &ranges), None);
    }

    #[test]
    fn progressive_pass_ranges_keep_declared_empty_passes_empty() {
        let ranges = build_modular_pass_shift_ranges(3, &[8], &[1]).unwrap();
        assert_eq!(ranges[0], None);
        assert_eq!(
            ranges[1],
            Some(ModularPassShiftRange {
                min_shift: 3,
                max_shift: 3,
            })
        );
        assert_eq!(
            ranges[2],
            Some(ModularPassShiftRange {
                min_shift: 0,
                max_shift: 3,
            })
        );
        assert_eq!(modular_pass_for_channel(2, 4, &ranges), Some(2));
        assert_eq!(modular_pass_for_channel(0, 4, &ranges), Some(2));
        assert_eq!(modular_pass_for_channel(3, 3, &ranges), None);

        let limits = ModularTransformLimits::default();
        let topology = ModularChannelTopology::new(
            vec![ModularChannelGeometry::new(64, 16, 2, 4, 8)],
            0,
            limits,
        )
        .unwrap();
        let layout = topology.gpu_layout().unwrap();
        let (empty, targets) = grouped_subimage_topology(
            &topology,
            &layout,
            0,
            ModularSubimageRegion {
                kind: ModularSubimageKind::PassGroup,
                column: 0,
                row: 0,
                group_dimension: 256,
            },
            ranges[0],
            limits,
        )
        .unwrap();
        assert!(empty.channels().is_empty());
        assert!(targets.is_empty());
    }

    #[test]
    fn empty_pass_section_must_contain_only_zero_bits() {
        let section = |bits: u64| FrameSection {
            bitstream_index: 0,
            toc_index: 0,
            kind: FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index: 0,
            },
            bytes: Default::default(),
            bits: BitRange {
                offset: 0,
                length: bits,
            },
        };
        let zero = GpuCodestream::from_spans([(0, StreamSlice::from_shared(Arc::from(vec![0u8])))])
            .unwrap();
        validate_empty_pass_group_section(&zero, section(8), 0, 0).unwrap();

        let nonzero =
            GpuCodestream::from_spans([(0, StreamSlice::from_shared(Arc::from(vec![0x80u8])))])
                .unwrap();
        assert!(validate_empty_pass_group_section(&nonzero, section(8), 0, 0).is_err());
    }

    #[test]
    fn asymmetric_shifts_stay_in_pass_groups_while_double_large_shifts_use_lf_groups() {
        let limits = ModularTransformLimits::default();
        let topology = ModularChannelTopology::new(
            vec![
                ModularChannelGeometry::new(32, 17, 4, 0, 8),
                ModularChannelGeometry::new(300, 20, 3, 3, 8),
            ],
            0,
            limits,
        )
        .unwrap();
        let layout = topology.gpu_layout().unwrap();
        let (pass, pass_targets) = grouped_subimage_topology(
            &topology,
            &layout,
            0,
            ModularSubimageRegion {
                kind: ModularSubimageKind::PassGroup,
                column: 0,
                row: 0,
                group_dimension: 256,
            },
            Some(ModularPassShiftRange {
                min_shift: 0,
                max_shift: 3,
            }),
            limits,
        )
        .unwrap();
        assert_eq!(
            pass.channels(),
            &[ModularChannelGeometry::new(16, 17, 4, 0, 8)]
        );
        assert_eq!(pass_targets.len(), 1);

        let (lf, lf_targets) = grouped_subimage_topology(
            &topology,
            &layout,
            0,
            ModularSubimageRegion {
                kind: ModularSubimageKind::LowFrequencyGroup,
                column: 0,
                row: 0,
                group_dimension: 256,
            },
            None,
            limits,
        )
        .unwrap();
        assert_eq!(
            lf.channels(),
            &[ModularChannelGeometry::new(256, 20, 3, 3, 8)]
        );
        assert_eq!(lf_targets.len(), 1);
    }

    #[test]
    fn stock_profile_admits_gpu_resident_palette_squeeze_and_noncanonical_rct() {
        let limits = ModularTransformLimits::default();
        let gray_source = ModularChannelTopology::full_resolution(9, 5, 8, 1, limits).unwrap();
        let squeeze = ModularTransformPlan::squeeze_only_for_test(
            gray_source,
            vec![crate::modular_transform::ModularSqueezeParameter {
                horizontal: true,
                in_place: true,
                begin_channel: 0,
                channel_count: 1,
            }],
            limits,
        )
        .unwrap();
        validate_stock_modular_transform_plan(ModularChannels::Gray, 9, 5, 8, &squeeze).unwrap();
        let squeeze_inverse = plan_modular_inverse(&squeeze).unwrap();
        assert_eq!(squeeze_inverse.entropy_words(), 45);
        assert_eq!(squeeze_inverse.jobs().len(), 1);
        assert_eq!(squeeze_inverse.final_planes().len(), 1);

        let rgb_source = ModularChannelTopology::full_resolution(7, 3, 8, 3, limits).unwrap();
        let rct = ModularTransformPlan::from_transforms_for_test(
            rgb_source,
            vec![ModularTransformIr::Rct(ModularRct {
                begin_channel: 0,
                rct_type: 41,
            })],
            limits,
        )
        .unwrap();
        validate_stock_modular_transform_plan(ModularChannels::Rgb, 7, 3, 8, &rct).unwrap();
        let rct_inverse = plan_modular_inverse(&rct).unwrap();
        assert_eq!(rct_inverse.entropy_words(), 63);
        assert_eq!(rct_inverse.jobs().len(), 1);
        assert_eq!(rct_inverse.final_planes().len(), 3);

        let palette_source = ModularChannelTopology::full_resolution(11, 7, 8, 3, limits).unwrap();
        let palette = ModularTransformPlan::from_transforms_for_test(
            palette_source,
            vec![ModularTransformIr::Palette(
                crate::modular_transform::ModularPalette {
                    begin_channel: 0,
                    channel_count: 3,
                    color_count: 4,
                    delta_count: 2,
                    predictor: 4,
                },
            )],
            limits,
        )
        .unwrap();
        validate_stock_modular_transform_plan(ModularChannels::Rgb, 11, 7, 8, &palette).unwrap();
        let palette_inverse = plan_modular_inverse(&palette).unwrap();
        assert_eq!(palette_inverse.jobs().len(), 3);
        assert_eq!(palette_inverse.final_planes().len(), 3);
        assert!(palette_inverse.arena_words() > palette_inverse.entropy_words());
    }

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
    fn libjxl_progressive_dc_root_uses_its_effective_sample_extent() {
        let Some(encoded) = cjxl_progressive_dc() else {
            return;
        };
        let parsed = parse(&encoded, ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let frame = inventory.frames[0].clone();
        assert_eq!(frame.frame_type, FrameType::LowFrequency);
        assert_eq!(frame.encoding, FrameEncoding::Modular);
        assert_eq!(frame.lf_level, 2);
        let bytes: Arc<[u8]> = Arc::from(parsed.codestream());
        let codestream = GpuCodestream::from_spans([(0, StreamSlice::from_shared(bytes))]).unwrap();
        let mut projected = inventory.clone();
        projected.frames = vec![frame];
        let profile = parse_progressive_dc_modular_profile(&codestream, &projected).unwrap();
        assert_eq!((profile.width, profile.height), (16, 2));
        assert_eq!(profile.groups.len(), 1);
        assert!(profile.generalized_channels);
        let progressive_dc = profile.progressive_dc.unwrap();
        assert_eq!(progressive_dc.lf_level, 2);
        assert!(
            progressive_dc
                .lf_dequantization()
                .into_iter()
                .map(|multiplier| multiplier / 128.0)
                .all(|multiplier| multiplier.is_finite() && multiplier > 0.0)
        );
        let [resident] = profile.resident_entropy_plans.as_slice() else {
            panic!("progressive-DC root did not produce one resident entropy plan");
        };
        let inverse = &resident.inverse_plan;
        assert_eq!(inverse.final_planes().len(), 3);
        assert_eq!(
            u64::from(inverse.entropy_words()),
            u64::from(profile.width) * u64::from(profile.height) * 3
        );
        assert!(inverse.arena_words() >= inverse.entropy_words());
        assert_eq!(
            inverse
                .final_planes()
                .iter()
                .map(|plane| (plane.geometry.width, plane.geometry.height))
                .collect::<Vec<_>>(),
            vec![(profile.width, profile.height); 3]
        );

        for (frame_index, expected_extent, is_final) in
            [(1_usize, (128, 16), false), (2, (1_024, 128), true)]
        {
            let mut projected = inventory.clone();
            projected.frames = vec![inventory.frames[frame_index].clone()];
            let packet =
                crate::vardct_packet::BoundedVarDctPacketPlan::parse_progressive_dc_source(
                    &codestream,
                    &projected,
                    is_final,
                )
                .unwrap();
            assert_eq!(
                (packet.profile.width, packet.profile.height),
                expected_extent
            );
            assert!(packet.profile.uses_lf_frame);
            assert!(
                packet
                    .groups
                    .iter()
                    .all(|group| group.external_lf_hf.is_some())
            );
        }
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
