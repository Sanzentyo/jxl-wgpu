//! Bounded Modular transform parsing and exact transformed-channel topology.
//!
//! This layer does not reconstruct a pixel on the host. It turns the ordered JPEG XL transform
//! description into the channel geometry consumed by Modular entropy and into the reverse-order
//! operations that later GPU passes must execute.

use bytemuck::{Pod, Zeroable};

use crate::{ModularTransformError, Result, modular_tree::BitInput};

const RCT_CHANNELS: u32 = 3;

/// Resource limits applied before transform metadata can allocate host or GPU planning storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularTransformLimits {
    pub transform_count: usize,
    pub channel_count: usize,
    pub squeeze_parameter_count: usize,
}

impl Default for ModularTransformLimits {
    fn default() -> Self {
        Self {
            transform_count: 512,
            channel_count: 1 << 16,
            squeeze_parameter_count: 1 << 17,
        }
    }
}

/// Host-shareable geometry for one transformed Modular channel.
///
/// Negative shifts mark unshiftable meta channels. Reserved words make this a stable 32-byte
/// storage-buffer element and must remain zero when uploaded.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct ModularChannelGeometry {
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
    pub bit_depth: u32,
    pub reserved: [u32; 3],
}

const _: [(); 32] = [(); std::mem::size_of::<ModularChannelGeometry>()];
const _: [(); 4] = [(); std::mem::align_of::<ModularChannelGeometry>()];

impl ModularChannelGeometry {
    pub(crate) const fn new(
        width: u32,
        height: u32,
        hshift: i32,
        vshift: i32,
        bit_depth: u32,
    ) -> Self {
        Self {
            width,
            height,
            hshift,
            vshift,
            bit_depth,
            reserved: [0; 3],
        }
    }

    const fn meta(width: u32, height: u32, bit_depth: u32) -> Self {
        Self::new(width, height, -1, -1, bit_depth)
    }

    fn sample_count(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Storage-buffer descriptor for one packed transformed channel.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(crate) struct GpuModularChannelLayout {
    pub word_offset: u32,
    pub row_stride_words: u32,
    pub width: u32,
    pub height: u32,
    pub hshift: i32,
    pub vshift: i32,
    pub bit_depth: u32,
    pub reserved: u32,
}

const _: [(); 32] = [(); std::mem::size_of::<GpuModularChannelLayout>()];
const _: [(); 4] = [(); std::mem::align_of::<GpuModularChannelLayout>()];

/// Channel order and meta-channel prefix after applying an ordered transform prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModularChannelTopology {
    channels: Vec<ModularChannelGeometry>,
    meta_channels: usize,
}

impl ModularChannelTopology {
    pub(crate) fn new(
        channels: Vec<ModularChannelGeometry>,
        meta_channels: usize,
        limits: ModularTransformLimits,
    ) -> Result<Self> {
        if meta_channels > channels.len() {
            return Err(ModularTransformError::MixedMetaChannels {
                transform: "initial topology",
            }
            .into());
        }
        check_channel_limit(channels.len(), limits)?;
        Ok(Self {
            channels,
            meta_channels,
        })
    }

    pub(crate) fn full_resolution(
        width: u32,
        height: u32,
        bit_depth: u32,
        channel_count: u32,
        limits: ModularTransformLimits,
    ) -> Result<Self> {
        let channel_count = usize::try_from(channel_count).map_err(|_| {
            ModularTransformError::ChannelLimitExceeded {
                actual: usize::MAX,
                limit: limits.channel_count,
            }
        })?;
        check_channel_limit(channel_count, limits)?;
        Self::new(
            vec![ModularChannelGeometry::new(width, height, 0, 0, bit_depth); channel_count],
            0,
            limits,
        )
    }

    pub(crate) fn channels(&self) -> &[ModularChannelGeometry] {
        &self.channels
    }

    pub(crate) const fn meta_channel_count(&self) -> usize {
        self.meta_channels
    }

    pub(crate) fn sample_count(&self) -> Option<u64> {
        self.channels.iter().try_fold(0u64, |total, channel| {
            total.checked_add(channel.sample_count())
        })
    }

    pub(crate) fn gpu_layout(&self) -> Result<Vec<GpuModularChannelLayout>> {
        let mut word_offset = 0u64;
        let mut layouts = Vec::with_capacity(self.channels.len());
        for channel in &self.channels {
            layouts.push(GpuModularChannelLayout {
                word_offset: u32::try_from(word_offset)
                    .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow)?,
                row_stride_words: channel.width,
                width: channel.width,
                height: channel.height,
                hshift: channel.hshift,
                vshift: channel.vshift,
                bit_depth: channel.bit_depth,
                reserved: 0,
            });
            word_offset = word_offset
                .checked_add(channel.sample_count())
                .ok_or(ModularTransformError::GpuAddressSpaceOverflow)?;
            if word_offset > u64::from(u32::MAX) {
                return Err(ModularTransformError::GpuAddressSpaceOverflow.into());
            }
        }
        Ok(layouts)
    }
}

/// One of the 42 reversible three-channel transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularRct {
    pub begin_channel: u32,
    pub rct_type: u32,
}

/// Palette metadata and its delta-prediction contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularPalette {
    pub begin_channel: u32,
    pub channel_count: u32,
    pub color_count: u32,
    pub delta_count: u32,
    pub predictor: u32,
}

/// One horizontal or vertical channel split from a Squeeze transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularSqueezeParameter {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_channel: u32,
    pub channel_count: u32,
}

/// Parsed transform in codestream order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ModularTransformIr {
    Rct(ModularRct),
    Palette(ModularPalette),
    Squeeze {
        used_default_parameters: bool,
        parameters: Vec<ModularSqueezeParameter>,
    },
}

/// Complete transform sequence plus the entropy-visible topology it produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModularTransformPlan {
    pub transforms: Vec<ModularTransformIr>,
    pub topology: ModularChannelTopology,
}

/// Parses and meta-applies the transform list from one Modular group header.
pub(crate) fn parse_modular_transforms(
    reader: &mut impl BitInput,
    initial_topology: ModularChannelTopology,
    limits: ModularTransformLimits,
) -> Result<ModularTransformPlan> {
    let transform_count = read_u32(reader, [(0, 0), (1, 0), (2, 4), (18, 8)])?;
    parse_modular_transform_count(reader, transform_count, initial_topology, limits)
}

fn parse_modular_transform_count(
    reader: &mut impl BitInput,
    transform_count: u32,
    mut topology: ModularChannelTopology,
    limits: ModularTransformLimits,
) -> Result<ModularTransformPlan> {
    let transform_count = usize::try_from(transform_count).map_err(|_| {
        ModularTransformError::TransformLimitExceeded {
            actual: usize::MAX,
            limit: limits.transform_count,
        }
    })?;
    if transform_count > limits.transform_count {
        return Err(ModularTransformError::TransformLimitExceeded {
            actual: transform_count,
            limit: limits.transform_count,
        }
        .into());
    }

    let mut transforms = Vec::with_capacity(transform_count);
    let mut total_squeeze_parameters = 0usize;
    for _ in 0..transform_count {
        let transform_id = read_u32_bits(reader, 2)?;
        let transform = match transform_id {
            0 => {
                let rct = ModularRct {
                    begin_channel: read_begin_channel(reader)?,
                    rct_type: read_u32(reader, [(6, 0), (0, 2), (2, 4), (10, 6)])?,
                };
                apply_rct(&topology, rct)?;
                ModularTransformIr::Rct(rct)
            }
            1 => {
                let palette = ModularPalette {
                    begin_channel: read_begin_channel(reader)?,
                    channel_count: read_u32(reader, [(1, 0), (3, 0), (4, 0), (1, 13)])?,
                    color_count: read_u32(reader, [(0, 8), (256, 10), (1280, 12), (5376, 16)])?,
                    delta_count: read_u32(reader, [(0, 0), (1, 8), (257, 10), (1281, 16)])?,
                    predictor: read_u32_bits(reader, 4)?,
                };
                apply_palette(&mut topology, palette, limits)?;
                ModularTransformIr::Palette(palette)
            }
            2 => {
                let parameter_count = read_u32(reader, [(0, 0), (1, 4), (9, 6), (41, 8)])?;
                let parameter_count = usize::try_from(parameter_count).map_err(|_| {
                    ModularTransformError::SqueezeLimitExceeded {
                        actual: usize::MAX,
                        limit: limits.squeeze_parameter_count,
                    }
                })?;
                total_squeeze_parameters = total_squeeze_parameters
                    .checked_add(parameter_count)
                    .ok_or(ModularTransformError::SqueezeLimitExceeded {
                        actual: usize::MAX,
                        limit: limits.squeeze_parameter_count,
                    })?;
                if total_squeeze_parameters > limits.squeeze_parameter_count {
                    return Err(ModularTransformError::SqueezeLimitExceeded {
                        actual: total_squeeze_parameters,
                        limit: limits.squeeze_parameter_count,
                    }
                    .into());
                }
                let used_default_parameters = parameter_count == 0;
                let mut parameters = Vec::with_capacity(parameter_count);
                for _ in 0..parameter_count {
                    parameters.push(ModularSqueezeParameter {
                        horizontal: reader.read_bits(1)? != 0,
                        in_place: reader.read_bits(1)? != 0,
                        begin_channel: read_begin_channel(reader)?,
                        channel_count: read_u32(reader, [(1, 0), (2, 0), (3, 0), (4, 4)])?,
                    });
                }
                if used_default_parameters {
                    parameters = default_squeeze_parameters(&topology)?;
                    total_squeeze_parameters = total_squeeze_parameters
                        .checked_add(parameters.len())
                        .ok_or(ModularTransformError::SqueezeLimitExceeded {
                            actual: usize::MAX,
                            limit: limits.squeeze_parameter_count,
                        })?;
                    if total_squeeze_parameters > limits.squeeze_parameter_count {
                        return Err(ModularTransformError::SqueezeLimitExceeded {
                            actual: total_squeeze_parameters,
                            limit: limits.squeeze_parameter_count,
                        }
                        .into());
                    }
                }
                for parameter in &parameters {
                    apply_squeeze(&mut topology, *parameter, limits)?;
                }
                ModularTransformIr::Squeeze {
                    used_default_parameters,
                    parameters,
                }
            }
            id => return Err(ModularTransformError::InvalidTransformId { id }.into()),
        };
        transforms.push(transform);
    }

    Ok(ModularTransformPlan {
        transforms,
        topology,
    })
}

fn apply_rct(topology: &ModularChannelTopology, rct: ModularRct) -> Result<()> {
    if rct.rct_type >= 42 {
        return Err(ModularTransformError::InvalidRctType {
            rct_type: rct.rct_type,
        }
        .into());
    }
    let range = channel_range(
        "RCT",
        rct.begin_channel,
        RCT_CHANNELS,
        topology.channels.len(),
    )?;
    check_meta_boundary("RCT", &range, topology.meta_channels)?;
    check_equal_channels("RCT", &topology.channels[range])
}

fn apply_palette(
    topology: &mut ModularChannelTopology,
    palette: ModularPalette,
    limits: ModularTransformLimits,
) -> Result<()> {
    if palette.predictor >= 14 {
        return Err(ModularTransformError::InvalidPalettePredictor {
            predictor: palette.predictor,
        }
        .into());
    }
    let range = channel_range(
        "Palette",
        palette.begin_channel,
        palette.channel_count,
        topology.channels.len(),
    )?;
    check_meta_boundary("Palette", &range, topology.meta_channels)?;
    check_equal_channels("Palette", &topology.channels[range.clone()])?;
    let selected = topology.channels[range.start];
    let palette_width = palette
        .color_count
        .checked_add(palette.delta_count)
        .ok_or(ModularTransformError::PaletteDimensionOverflow)?;

    let final_channel_count = topology
        .channels
        .len()
        .checked_sub(range.len().saturating_sub(1))
        .and_then(|count| count.checked_add(1))
        .ok_or(ModularTransformError::ChannelLimitExceeded {
            actual: usize::MAX,
            limit: limits.channel_count,
        })?;
    check_channel_limit(final_channel_count, limits)?;

    if range.start < topology.meta_channels {
        topology.meta_channels = topology
            .meta_channels
            .checked_add(2)
            .and_then(|count| count.checked_sub(range.len()))
            .ok_or(ModularTransformError::MixedMetaChannels {
                transform: "Palette",
            })?;
    } else {
        topology.meta_channels = topology.meta_channels.checked_add(1).ok_or(
            ModularTransformError::ChannelLimitExceeded {
                actual: usize::MAX,
                limit: limits.channel_count,
            },
        )?;
    }
    topology.channels.drain((range.start + 1)..range.end);
    topology.channels.insert(
        0,
        ModularChannelGeometry::meta(palette_width, palette.channel_count, selected.bit_depth),
    );
    Ok(())
}

fn apply_squeeze(
    topology: &mut ModularChannelTopology,
    parameter: ModularSqueezeParameter,
    limits: ModularTransformLimits,
) -> Result<()> {
    let range = channel_range(
        "Squeeze",
        parameter.begin_channel,
        parameter.channel_count,
        topology.channels.len(),
    )?;
    check_meta_boundary("Squeeze", &range, topology.meta_channels)?;
    if range.start < topology.meta_channels && !parameter.in_place {
        return Err(ModularTransformError::MetaSqueezeRequiresInPlace.into());
    }
    let final_count = topology.channels.len().checked_add(range.len()).ok_or(
        ModularTransformError::ChannelLimitExceeded {
            actual: usize::MAX,
            limit: limits.channel_count,
        },
    )?;
    check_channel_limit(final_count, limits)?;

    let mut residuals = Vec::with_capacity(range.len());
    for channel_index in range.clone() {
        let channel = &mut topology.channels[channel_index];
        if channel.width == 0 || channel.height == 0 {
            return Err(ModularTransformError::ZeroSizedSqueezeChannel {
                channel: channel_index,
            }
            .into());
        }
        if channel.hshift > 30 || channel.vshift > 30 {
            return Err(ModularTransformError::TooManySqueezes {
                channel: channel_index,
            }
            .into());
        }
        let mut residual = *channel;
        if parameter.horizontal {
            channel.width = channel.width.div_ceil(2);
            residual.width /= 2;
            if channel.hshift >= 0 {
                channel.hshift += 1;
                residual.hshift += 1;
            }
        } else {
            channel.height = channel.height.div_ceil(2);
            residual.height /= 2;
            if channel.vshift >= 0 {
                channel.vshift += 1;
                residual.vshift += 1;
            }
        }
        residuals.push(residual);
    }

    let insertion = if parameter.in_place {
        range.end
    } else {
        topology.channels.len()
    };
    topology.channels.splice(insertion..insertion, residuals);
    if range.start < topology.meta_channels {
        topology.meta_channels = topology.meta_channels.checked_add(range.len()).ok_or(
            ModularTransformError::ChannelLimitExceeded {
                actual: usize::MAX,
                limit: limits.channel_count,
            },
        )?;
    }
    Ok(())
}

fn default_squeeze_parameters(
    topology: &ModularChannelTopology,
) -> Result<Vec<ModularSqueezeParameter>> {
    let first = topology.meta_channels;
    let first_channel = topology
        .channels
        .get(first)
        .ok_or(ModularTransformError::MissingDataChannel)?;
    let mut width = first_channel.width;
    let mut height = first_channel.height;
    let data_channels = topology.channels.len() - first;
    let mut parameters = Vec::new();

    if data_channels > 2
        && topology.channels[first + 1].width == width
        && topology.channels[first + 1].height == height
    {
        let chroma = ModularSqueezeParameter {
            horizontal: true,
            in_place: false,
            begin_channel: u32::try_from(first + 1)
                .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow)?,
            channel_count: 2,
        };
        if width > 1 {
            parameters.push(chroma);
        }
        if height > 1 {
            parameters.push(ModularSqueezeParameter {
                horizontal: false,
                ..chroma
            });
        }
    }

    let all_data = ModularSqueezeParameter {
        horizontal: false,
        in_place: true,
        begin_channel: u32::try_from(first)
            .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow)?,
        channel_count: u32::try_from(data_channels)
            .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow)?,
    };
    if width <= height && height > 8 {
        parameters.push(all_data);
        height = height.div_ceil(2);
    }
    while width > 8 || height > 8 {
        if width > 8 {
            parameters.push(ModularSqueezeParameter {
                horizontal: true,
                ..all_data
            });
            width = width.div_ceil(2);
        }
        if height > 8 {
            parameters.push(all_data);
            height = height.div_ceil(2);
        }
    }
    Ok(parameters)
}

fn channel_range(
    transform: &'static str,
    begin: u32,
    count: u32,
    available: usize,
) -> Result<std::ops::Range<usize>> {
    let end = begin
        .checked_add(count)
        .ok_or(ModularTransformError::ChannelRange {
            transform,
            begin,
            end: u32::MAX,
            available,
        })?;
    let begin_index = usize::try_from(begin).map_err(|_| ModularTransformError::ChannelRange {
        transform,
        begin,
        end,
        available,
    })?;
    let end_index = usize::try_from(end).map_err(|_| ModularTransformError::ChannelRange {
        transform,
        begin,
        end,
        available,
    })?;
    if count == 0 || end_index > available {
        return Err(ModularTransformError::ChannelRange {
            transform,
            begin,
            end,
            available,
        }
        .into());
    }
    Ok(begin_index..end_index)
}

fn check_meta_boundary(
    transform: &'static str,
    range: &std::ops::Range<usize>,
    meta_channels: usize,
) -> Result<()> {
    if range.start < meta_channels && range.end > meta_channels {
        return Err(ModularTransformError::MixedMetaChannels { transform }.into());
    }
    Ok(())
}

fn check_equal_channels(
    transform: &'static str,
    channels: &[ModularChannelGeometry],
) -> Result<()> {
    let Some(first) = channels.first() else {
        return Err(ModularTransformError::ChannelRange {
            transform,
            begin: 0,
            end: 0,
            available: 0,
        }
        .into());
    };
    if channels.iter().skip(1).any(|channel| channel != first) {
        return Err(ModularTransformError::UnequalChannels { transform }.into());
    }
    Ok(())
}

fn check_channel_limit(count: usize, limits: ModularTransformLimits) -> Result<()> {
    if count > limits.channel_count {
        return Err(ModularTransformError::ChannelLimitExceeded {
            actual: count,
            limit: limits.channel_count,
        }
        .into());
    }
    Ok(())
}

fn read_begin_channel(reader: &mut impl BitInput) -> Result<u32> {
    read_u32(reader, [(0, 3), (8, 6), (72, 10), (1096, 13)])
}

fn read_u32(reader: &mut impl BitInput, variants: [(u32, u8); 4]) -> Result<u32> {
    let selector = usize::try_from(reader.read_bits(2)?)
        .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow)?;
    let (base, bits) = variants[selector];
    base.checked_add(read_u32_bits(reader, bits)?)
        .ok_or_else(|| ModularTransformError::GpuAddressSpaceOverflow.into())
}

fn read_u32_bits(reader: &mut impl BitInput, count: u8) -> Result<u32> {
    u32::try_from(reader.read_bits(count)?)
        .map_err(|_| ModularTransformError::GpuAddressSpaceOverflow.into())
}

#[cfg(test)]
mod tests {
    use jxl::{
        bit_reader::BitReader as OracleBitReader,
        headers::{
            encodings::{Empty, UnconditionalCoder},
            modular::{GroupHeader, TransformId},
        },
    };
    use jxl_gpu_bitstream::{BitReader, BitWriter};

    use super::*;
    use crate::Error;

    fn write_u32(writer: &mut BitWriter, value: u32, variants: [(u32, u8); 4]) {
        let (selector, base, bits) = variants
            .into_iter()
            .enumerate()
            .find_map(|(selector, (base, bits))| {
                let span = 1u64 << bits;
                (u64::from(value) >= u64::from(base) && u64::from(value) < u64::from(base) + span)
                    .then_some((selector, base, bits))
            })
            .expect("test value fits selector");
        writer.write_bits(selector as u64, 2).unwrap();
        writer.write_bits(u64::from(value - base), bits).unwrap();
    }

    fn write_begin_channel(writer: &mut BitWriter, value: u32) {
        write_u32(writer, value, [(0, 3), (8, 6), (72, 10), (1096, 13)]);
    }

    fn topology(width: u32, height: u32, channels: u32) -> ModularChannelTopology {
        ModularChannelTopology::full_resolution(
            width,
            height,
            12,
            channels,
            ModularTransformLimits::default(),
        )
        .unwrap()
    }

    fn parse(writer: BitWriter, initial: ModularChannelTopology) -> ModularTransformPlan {
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        parse_modular_transforms(&mut reader, initial, ModularTransformLimits::default()).unwrap()
    }

    fn write_transform_count(writer: &mut BitWriter, count: u32) {
        write_u32(writer, count, [(0, 0), (1, 0), (2, 4), (18, 8)]);
    }

    fn write_rct(writer: &mut BitWriter, begin_channel: u32, rct_type: u32) {
        writer.write_bits(0, 2).unwrap();
        write_begin_channel(writer, begin_channel);
        write_u32(writer, rct_type, [(6, 0), (0, 2), (2, 4), (10, 6)]);
    }

    #[test]
    fn parses_every_normative_rct_type() {
        for rct_type in 0..42 {
            let mut writer = BitWriter::new();
            write_transform_count(&mut writer, 1);
            write_rct(&mut writer, 0, rct_type);
            let plan = parse(writer, topology(17, 9, 3));
            assert_eq!(
                plan.transforms,
                vec![ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type,
                })]
            );
            assert_eq!(plan.topology.channels(), topology(17, 9, 3).channels());
        }
    }

    #[test]
    fn rejects_out_of_range_rct_with_typed_error() {
        let mut writer = BitWriter::new();
        write_transform_count(&mut writer, 1);
        write_rct(&mut writer, 0, 42);
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        assert!(matches!(
            parse_modular_transforms(
                &mut reader,
                topology(1, 1, 3),
                ModularTransformLimits::default()
            ),
            Err(Error::ModularTransform(
                ModularTransformError::InvalidRctType { rct_type: 42 }
            ))
        ));
    }

    #[test]
    fn palette_adds_meta_storage_and_collapses_selected_channels() {
        let mut writer = BitWriter::new();
        write_transform_count(&mut writer, 1);
        writer.write_bits(1, 2).unwrap();
        write_begin_channel(&mut writer, 0);
        write_u32(&mut writer, 3, [(1, 0), (3, 0), (4, 0), (1, 13)]);
        write_u32(
            &mut writer,
            257,
            [(0, 8), (256, 10), (1280, 12), (5376, 16)],
        );
        write_u32(&mut writer, 9, [(0, 0), (1, 8), (257, 10), (1281, 16)]);
        writer.write_bits(13, 4).unwrap();

        let plan = parse(writer, topology(31, 7, 4));
        assert_eq!(plan.topology.meta_channel_count(), 1);
        assert_eq!(plan.topology.channels().len(), 3);
        assert_eq!(
            plan.topology.channels()[0],
            ModularChannelGeometry::meta(266, 3, 12)
        );
        assert_eq!(
            plan.topology.channels()[1],
            ModularChannelGeometry::new(31, 7, 0, 0, 12)
        );
        assert_eq!(plan.topology.sample_count(), Some(266 * 3 + 31 * 7 * 2));
    }

    #[test]
    fn explicit_in_place_squeeze_preserves_residual_and_tail_order() {
        let mut writer = BitWriter::new();
        write_transform_count(&mut writer, 1);
        writer.write_bits(2, 2).unwrap();
        write_u32(&mut writer, 1, [(0, 0), (1, 4), (9, 6), (41, 8)]);
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        write_begin_channel(&mut writer, 0);
        write_u32(&mut writer, 2, [(1, 0), (2, 0), (3, 0), (4, 4)]);

        let plan = parse(writer, topology(5, 3, 3));
        let dimensions = plan
            .topology
            .channels()
            .iter()
            .map(|channel| (channel.width, channel.height, channel.hshift))
            .collect::<Vec<_>>();
        assert_eq!(
            dimensions,
            vec![(3, 3, 1), (3, 3, 1), (2, 3, 1), (2, 3, 1), (5, 3, 0)]
        );
    }

    #[test]
    fn default_squeeze_resolves_odd_pyramid_before_entropy_planning() {
        let mut writer = BitWriter::new();
        write_transform_count(&mut writer, 1);
        writer.write_bits(2, 2).unwrap();
        write_u32(&mut writer, 0, [(0, 0), (1, 4), (9, 6), (41, 8)]);

        let plan = parse(writer, topology(9, 5, 1));
        assert_eq!(
            plan.transforms,
            vec![ModularTransformIr::Squeeze {
                used_default_parameters: true,
                parameters: vec![ModularSqueezeParameter {
                    horizontal: true,
                    in_place: true,
                    begin_channel: 0,
                    channel_count: 1,
                }],
            }]
        );
        assert_eq!(
            plan.topology.channels(),
            &[
                ModularChannelGeometry::new(5, 5, 1, 0, 12),
                ModularChannelGeometry::new(4, 5, 1, 0, 12),
            ]
        );
        let layouts = plan.topology.gpu_layout().unwrap();
        assert_eq!(layouts[0].word_offset, 0);
        assert_eq!(layouts[1].word_offset, 25);
        assert_eq!(bytemuck::bytes_of(&layouts[0]).len(), 32);
    }

    #[test]
    fn rejects_transform_that_crosses_meta_boundary() {
        let limits = ModularTransformLimits::default();
        let initial = ModularChannelTopology::new(
            vec![
                ModularChannelGeometry::meta(4, 1, 8),
                ModularChannelGeometry::new(4, 1, 0, 0, 8),
                ModularChannelGeometry::new(4, 1, 0, 0, 8),
            ],
            1,
            limits,
        )
        .unwrap();
        let mut writer = BitWriter::new();
        write_transform_count(&mut writer, 1);
        write_rct(&mut writer, 0, 6);
        let bytes = writer.into_bytes();
        let mut reader = BitReader::new(&bytes);
        assert!(matches!(
            parse_modular_transforms(&mut reader, initial, limits),
            Err(Error::ModularTransform(
                ModularTransformError::MixedMetaChannels { transform: "RCT" }
            ))
        ));
    }

    #[test]
    fn rejects_gpu_layout_larger_than_u32_words() {
        let topology = ModularChannelTopology::new(
            vec![ModularChannelGeometry::new(u32::MAX, 2, 0, 0, 16)],
            0,
            ModularTransformLimits::default(),
        )
        .unwrap();
        assert!(matches!(
            topology.gpu_layout(),
            Err(Error::ModularTransform(
                ModularTransformError::GpuAddressSpaceOverflow
            ))
        ));
    }

    #[test]
    fn transform_wire_fields_match_the_jxl_metadata_oracle() {
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1).unwrap(); // use global tree
        writer.write_bits(1, 1).unwrap(); // default weighted predictor
        write_transform_count(&mut writer, 3);
        write_rct(&mut writer, 0, 41);

        writer.write_bits(1, 2).unwrap();
        write_begin_channel(&mut writer, 0);
        write_u32(&mut writer, 3, [(1, 0), (3, 0), (4, 0), (1, 13)]);
        write_u32(
            &mut writer,
            257,
            [(0, 8), (256, 10), (1280, 12), (5376, 16)],
        );
        write_u32(&mut writer, 9, [(0, 0), (1, 8), (257, 10), (1281, 16)]);
        writer.write_bits(13, 4).unwrap();

        writer.write_bits(2, 2).unwrap();
        write_u32(&mut writer, 1, [(0, 0), (1, 4), (9, 6), (41, 8)]);
        writer.write_bits(1, 1).unwrap();
        writer.write_bits(1, 1).unwrap();
        write_begin_channel(&mut writer, 1);
        write_u32(&mut writer, 1, [(1, 0), (2, 0), (3, 0), (4, 4)]);

        let bytes = writer.into_bytes();
        let mut oracle_reader = OracleBitReader::new(&bytes);
        let oracle =
            GroupHeader::read_unconditional(&(), &mut oracle_reader, &Empty::default()).unwrap();
        assert!(oracle.use_global_tree);
        assert_eq!(oracle.transforms.len(), 3);
        assert_eq!(oracle.transforms[0].id, TransformId::Rct);
        assert_eq!(oracle.transforms[0].begin_channel, 0);
        assert_eq!(oracle.transforms[0].rct_type, 41);
        assert_eq!(oracle.transforms[1].id, TransformId::Palette);
        assert_eq!(oracle.transforms[1].num_channels, 3);
        assert_eq!(oracle.transforms[1].num_colors, 257);
        assert_eq!(oracle.transforms[1].num_deltas, 9);
        assert_eq!(oracle.transforms[1].predictor_id, 13);
        assert_eq!(oracle.transforms[2].id, TransformId::Squeeze);
        assert_eq!(oracle.transforms[2].squeezes.len(), 1);
        assert!(oracle.transforms[2].squeezes[0].horizontal);
        assert!(oracle.transforms[2].squeezes[0].in_place);
        assert_eq!(oracle.transforms[2].squeezes[0].begin_channel, 1);
        assert_eq!(oracle.transforms[2].squeezes[0].num_channels, 1);

        let mut reader = BitReader::new(&bytes);
        assert_eq!(reader.read_bits(2).unwrap(), 3);
        let plan = parse_modular_transforms(
            &mut reader,
            topology(9, 5, 3),
            ModularTransformLimits::default(),
        )
        .unwrap();
        assert_eq!(
            plan.transforms,
            vec![
                ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type: 41,
                }),
                ModularTransformIr::Palette(ModularPalette {
                    begin_channel: 0,
                    channel_count: 3,
                    color_count: 257,
                    delta_count: 9,
                    predictor: 13,
                }),
                ModularTransformIr::Squeeze {
                    used_default_parameters: false,
                    parameters: vec![ModularSqueezeParameter {
                        horizontal: true,
                        in_place: true,
                        begin_channel: 1,
                        channel_count: 1,
                    }],
                },
            ]
        );
        assert_eq!(plan.topology.meta_channel_count(), 1);
        assert_eq!(
            plan.topology.channels(),
            &[
                ModularChannelGeometry::meta(266, 3, 12),
                ModularChannelGeometry::new(5, 5, 1, 0, 12),
                ModularChannelGeometry::new(4, 5, 1, 0, 12),
            ]
        );
    }
}
