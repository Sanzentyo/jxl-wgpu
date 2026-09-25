//! JPEG XL VarDCT headers, control packets, and GPU fragments.

use jxl_gpu_bitstream::BitWriter;

use super::color::VarDctColorPlan;
use super::entropy::VarDctPrefixCode;
use super::entropy::{HfEntropyPlan, write_prefix_config};
use super::types::{DcFragmentDescriptor, VarDctArtifactData, VarDctFrameLayout};
use super::{VarDctConfig, VarDctQuantization};
use crate::frame_header::FrameHeaderPlan;
use crate::{
    AnimationHeader, BackendError, BitFragment, EncodeError, FrameGroupLayout, FramePacketSet,
    GroupPacket, GroupPacketKind,
};

pub(super) fn image_header(
    width: u32,
    height: u32,
    animation: AnimationHeader,
    color: &VarDctColorPlan,
) -> Result<crate::source_color::icc::PreparedImageHeader, EncodeError> {
    color.image_header(&crate::ImageSequenceDescriptor::new(
        width, height, animation,
    )?)
}

fn frame_header(
    progressive: &crate::ProgressivePlan,
    control: &FrameHeaderPlan,
    color: &VarDctColorPlan,
) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?; // non-default so restoration can be disabled
    control.write_kind(&mut output)?;
    output.write_bits(0, 1)?; // VarDCT
    output.write_bits(0, 2)?; // no frame flags
    if !color.xyb_encoded() {
        output.write_bits(0, 1)?; // original RGB, not YCbCr
    }
    output.write_bits(0, 2)?; // no upsampling
    control.extra_channels().write(&mut output)?;
    if let Some(scales) = color.qm_scales() {
        for scale in scales {
            output.write_bits(u64::from(scale), 3)?;
        }
    }
    if control.has_passes() {
        let passes = progressive.passes();
        write_u32(
            &mut output,
            passes.len() as u32,
            [(1, 0), (2, 0), (3, 0), (4, 3)],
        )?;
        if passes.len() > 1 {
            write_u32(
                &mut output,
                progressive.downsampling().len() as u32,
                [(0, 0), (1, 0), (2, 0), (3, 1)],
            )?;
            for pass in &passes[..passes.len() - 1] {
                output.write_bits(u64::from(pass.shift), 2)?;
            }
            for point in progressive.downsampling() {
                write_u32(
                    &mut output,
                    u32::from(point.factor),
                    [(1, 0), (2, 0), (4, 0), (8, 0)],
                )?;
            }
            for point in progressive.downsampling() {
                write_u32(
                    &mut output,
                    u32::from(point.last_pass),
                    [(0, 0), (1, 0), (2, 0), (0, 3)],
                )?;
            }
        }
    }
    control.append_to(&mut output)?;
    let bit_len = output.bit_len();
    Ok(BitFragment::new(output.into_bytes(), bit_len)?)
}

fn write_u32(
    output: &mut BitWriter,
    value: u32,
    alternatives: [(u32, u8); 4],
) -> Result<(), EncodeError> {
    let Some((selector, offset, bits)) =
        alternatives
            .into_iter()
            .enumerate()
            .find_map(|(selector, (offset, bits))| {
                let encoded = value.checked_sub(offset)?;
                (u64::from(encoded) < (1u64 << bits)).then_some((selector, offset, bits))
            })
    else {
        return Err(EncodeError::InvalidConfiguration(
            "VarDCT integer is outside the JPEG XL U32 representation",
        ));
    };
    output.write_bits(selector as u64, 2)?;
    output.write_bits(u64::from(value - offset), bits)?;
    Ok(())
}

fn write_global_ma_config(
    output: &mut BitWriter,
    code: &VarDctPrefixCode,
) -> Result<(), EncodeError> {
    output.write_bits(1, 1)?; // global MA tree present
    write_prefix_config(output, code, 6)?;
    // One Gradient leaf, offset zero, multiplier one. Both the tree and image
    // stream use the same raw alphabet; no channel routing or LZ77 state is needed.
    for value in [0, 5, 0, 0, 0] {
        write_unsigned_token(output, code, value)?;
    }
    write_prefix_config(output, code, 1)
}

fn write_lf_global(
    output: &mut BitWriter,
    code: &VarDctPrefixCode,
    hf_entropy: &HfEntropyPlan,
    coefficient_payload: bool,
    config: &VarDctConfig,
) -> Result<(), EncodeError> {
    let lf_metadata = config.lf_metadata;
    output.write_bits(u64::from(lf_metadata.has_default_dequantization()), 1)?;
    if !lf_metadata.has_default_dequantization() {
        for value in lf_metadata.lf_dequantization {
            output.write_bits(u64::from(value.to_bits()), 16)?;
        }
    }
    write_u32(
        output,
        config.quantization.global_scale(),
        [(1, 11), (2_049, 11), (4_097, 12), (8_193, 16)],
    )?;
    write_u32(
        output,
        config.quantization.quant_lf(),
        [(16, 0), (1, 5), (1, 8), (1, 16)],
    )?;
    hf_entropy.write_block_context(output, coefficient_payload)?;
    output.write_bits(u64::from(lf_metadata.has_default_correlation()), 1)?;
    if !lf_metadata.has_default_correlation() {
        write_u32(
            output,
            lf_metadata.colour_factor,
            [(84, 0), (256, 0), (2, 8), (258, 16)],
        )?;
        for value in lf_metadata.base_correlation {
            output.write_bits(u64::from(value.to_bits()), 16)?;
        }
        for factor in lf_metadata.lf_factors {
            output.write_bits((i16::from(factor) + 128) as u64, 8)?;
        }
    }
    write_global_ma_config(output, code)
}

pub(super) fn write_local_modular_header(output: &mut BitWriter) -> Result<(), EncodeError> {
    output.write_bits(1, 1)?; // use the LF-global MA tree
    output.write_bits(1, 1)?; // default weighted-predictor header
    output.write_bits(0, 2)?; // zero transforms
    Ok(())
}

pub(super) fn write_unsigned_token(
    output: &mut BitWriter,
    code: &VarDctPrefixCode,
    value: u32,
) -> Result<(), EncodeError> {
    code.write_unsigned(output, value)
}

pub(super) fn pack_signed_control(value: i32) -> u32 {
    ((value as u32) << 1) ^ ((value >> 31) as u32)
}

fn append_gpu_dc_fragment(
    output: &mut BitWriter,
    fragment_words: &[u32],
    descriptor: DcFragmentDescriptor,
) -> Result<(), EncodeError> {
    let bit_offset = usize::try_from(descriptor.bit_offset)
        .map_err(|_| EncodeError::Backend("GPU DC fragment offset overflow".into()))?;
    let bit_len = usize::try_from(descriptor.bit_len)
        .map_err(|_| EncodeError::Backend("GPU DC fragment length overflow".into()))?;
    let bit_end = bit_offset
        .checked_add(bit_len)
        .ok_or_else(|| EncodeError::Backend("GPU DC fragment range overflow".into()))?;
    if bit_end > fragment_words.len() * 32 {
        return Err(EncodeError::Backend(
            "GPU DC fragment exceeds its artifact allocation".into(),
        ));
    }
    for source_bit in bit_offset..bit_end {
        let word = fragment_words[source_bit / 32];
        output.write_bits(u64::from((word >> (source_bit % 32)) & 1), 1)?;
    }
    Ok(())
}

pub(super) fn append_gpu_fragment(
    output: &mut BitWriter,
    fragment_words: &[u32],
    bit_offset: u32,
    bit_len: u32,
) -> Result<(), EncodeError> {
    let bit_offset = usize::try_from(bit_offset)
        .map_err(|_| BackendError::Invariant("GPU entropy fragment offset overflow"))?;
    let bit_len = usize::try_from(bit_len)
        .map_err(|_| BackendError::Invariant("GPU entropy fragment length overflow"))?;
    let bit_end = bit_offset
        .checked_add(bit_len)
        .ok_or(BackendError::Invariant(
            "GPU entropy fragment range overflow",
        ))?;
    if bit_end > fragment_words.len() * 32 {
        return Err(BackendError::Invariant(
            "GPU entropy fragment exceeds its artifact allocation",
        )
        .into());
    }
    let mut source_bit = bit_offset;
    while source_bit < bit_end {
        let shift = source_bit % 32;
        let count = (32 - shift).min(bit_end - source_bit);
        let mask = u32::MAX >> (32 - count);
        let word = (fragment_words[source_bit / 32] >> shift) & mask;
        output.write_bits(u64::from(word), count as u8)?;
        source_bit += count;
    }
    Ok(())
}

fn write_lf_group(
    output: &mut BitWriter,
    code: &VarDctPrefixCode,
    artifact: VarDctArtifactData<'_>,
    frame: VarDctFrameLayout,
    group_index: u32,
    quantization: VarDctQuantization,
) -> Result<(), EncodeError> {
    let group = frame.lf_group_blocks(group_index)?;
    let block_count = group.block_count()?;
    let strategies = if let Some(plan) = artifact.transform_plan {
        plan.lf_groups[group_index as usize]
            .iter()
            .map(|&index| {
                (
                    plan.tasks[index].strategy as i32,
                    plan.tasks[index].hf_multiplier as i32 - 1,
                )
            })
            .collect::<Vec<_>>()
    } else {
        let count = match frame.topology {
            super::types::VarDctTopology::SingleTransform(_) => 1,
            super::types::VarDctTopology::TiledDct8 => block_count as usize,
            super::types::VarDctTopology::StrategyMap => {
                return Err(EncodeError::InvalidConfiguration(
                    "mixed VarDCT metadata requires a strategy map",
                ));
            }
        };
        vec![
            (
                artifact.strategy as i32,
                quantization.hf_multiplier().get() as i32 - 1
            );
            count
        ]
    };
    output.write_bits(0, 2)?; // no extra LF precision
    write_local_modular_header(output)?;
    append_gpu_dc_fragment(
        output,
        artifact.dc_fragment_words,
        artifact.dc_fragment_descriptor(group_index)?,
    )?;

    artifact.extra_channels.write_lf(output, group_index)?;

    // Validated strategy/quantizer metadata, zero local CfL maps,
    // and zero EPF sharpness. All source-dependent entropy is already packed.
    let first_block_bits = block_count.next_power_of_two().trailing_zeros() as u8;
    output.write_bits(
        (strategies
            .len()
            .checked_sub(1)
            .ok_or(EncodeError::InvalidConfiguration(
                "VarDCT frame has no first transform block",
            ))?) as u64,
        first_block_bits,
    )?;
    write_local_modular_header(output)?;
    let correlation_samples = group.width.div_ceil(8) * group.height.div_ceil(8);
    // The two chroma-from-luma maps are tiled on the 8x8-block grid. They are
    // one sample each through DCT64, then scale to 2x2 and 4x4 for the
    // DCT128/DCT256 families.
    for _ in 0..2 * correlation_samples {
        write_unsigned_token(output, code, 0)?;
    }
    // ACS and HF multipliers form the two rows of one Modular channel.
    // The fixed MA tree uses Gradient on both rows: West on row zero, then
    // the clamped gradient of ACS (North), HF (West), and previous ACS (NW).
    for (index, &(strategy, _)) in strategies.iter().enumerate() {
        let west = index
            .checked_sub(1)
            .map_or(0, |previous| strategies[previous].0);
        write_unsigned_token(output, code, pack_signed_control(strategy - west))?;
    }
    for (index, &(north, hf)) in strategies.iter().enumerate() {
        let prediction = if index == 0 {
            north
        } else {
            super::dispatch::clamped_gradient_i32(
                north,
                strategies[index - 1].1,
                strategies[index - 1].0,
            )
        };
        write_unsigned_token(
            output,
            code,
            pack_signed_control(hf.wrapping_sub(prediction)),
        )?;
    }
    for _ in 0..block_count {
        write_unsigned_token(output, code, 0)?;
    }
    Ok(())
}

pub(super) fn build_frame_packet(
    artifact: VarDctArtifactData<'_>,
    code: &VarDctPrefixCode,
    hf_entropy: &HfEntropyPlan,
    frame: VarDctFrameLayout,
    config: &VarDctConfig,
    control: &FrameHeaderPlan,
    color: &VarDctColorPlan,
) -> Result<FramePacketSet, EncodeError> {
    if artifact.extra_channels.len() != color.samples.extra_channels.len() {
        return Err(BackendError::InvalidArtifact(
            "extra-channel artifacts differ from the image sample plan",
        )
        .into());
    }
    config.group_order.validate(frame)?;
    config
        .group_order
        .validate_scores(frame, artifact.saliency)?;
    let ac_groups = frame.ac_group_count()?;
    let lf_groups = frame.lf_group_count()?;
    let coefficient_payload = artifact.has_ac_payload();
    let passes = config.progressive.passes().len() as u8;
    if passes > 1 && !coefficient_payload {
        return Err(BackendError::InvalidArtifact("progressive frame has no AC payload").into());
    }
    if ac_groups == 1 && lf_groups == 1 && passes == 1 {
        let mut group = BitWriter::new();
        write_lf_global(&mut group, code, hf_entropy, coefficient_payload, config)?;
        artifact.extra_channels.write_global(&mut group)?;
        write_lf_group(&mut group, code, artifact, frame, 0, config.quantization)?;
        hf_entropy.write_global(
            &mut group,
            ac_groups,
            coefficient_payload,
            config,
            artifact.raw_matrices,
        )?;
        artifact.ac.append_group(&mut group, frame, 0, 0)?;
        artifact.extra_channels.write_group(&mut group, 0, 0)?;
        group.align_to_byte()?;
        return Ok(FramePacketSet::new(
            frame_header(&config.progressive, control, color)?,
            FrameGroupLayout::new(1, 1, 1)?,
            [GroupPacket::new(
                GroupPacketKind::Single,
                group.into_bytes(),
            )],
        )?);
    }

    let mut dc_global = BitWriter::new();
    write_lf_global(
        &mut dc_global,
        code,
        hf_entropy,
        coefficient_payload,
        config,
    )?;
    artifact.extra_channels.write_global(&mut dc_global)?;
    dc_global.align_to_byte()?;
    let mut ac_global = BitWriter::new();
    hf_entropy.write_global(
        &mut ac_global,
        ac_groups,
        coefficient_payload,
        config,
        artifact.raw_matrices,
    )?;
    ac_global.align_to_byte()?;

    let mut packets = Vec::with_capacity(
        usize::try_from(
            ac_groups
                .checked_mul(u32::from(passes))
                .and_then(|groups| groups.checked_add(lf_groups))
                .ok_or(EncodeError::InvalidConfiguration(
                    "VarDCT packet count overflow",
                ))?,
        )
        .map_err(|_| EncodeError::InvalidConfiguration("VarDCT packet count overflow"))?
            + 2,
    );
    packets.push(GroupPacket::new(
        GroupPacketKind::DcGlobal,
        dc_global.into_bytes(),
    ));
    for group_index in 0..lf_groups {
        let mut dc_group = BitWriter::new();
        write_lf_group(
            &mut dc_group,
            code,
            artifact,
            frame,
            group_index,
            config.quantization,
        )?;
        dc_group.align_to_byte()?;
        packets.push(GroupPacket::new(
            GroupPacketKind::DcGroup(group_index),
            dc_group.into_bytes(),
        ));
    }
    packets.push(GroupPacket::new(
        GroupPacketKind::AcGlobal,
        ac_global.into_bytes(),
    ));
    for pass in 0..passes {
        for group in 0..ac_groups {
            let mut output = BitWriter::new();
            artifact
                .ac
                .append_group(&mut output, frame, group, u32::from(pass))?;
            artifact
                .extra_channels
                .write_group(&mut output, group, u32::from(pass))?;
            output.align_to_byte()?;
            packets.push(GroupPacket::new(
                GroupPacketKind::AcGroup { pass, group },
                output.into_bytes(),
            ));
        }
    }
    config.group_order.apply(
        FramePacketSet::new(
            frame_header(&config.progressive, control, color)?,
            FrameGroupLayout::new(lf_groups, ac_groups, passes)?,
            packets,
        )?,
        frame,
        artifact.saliency,
    )
}
