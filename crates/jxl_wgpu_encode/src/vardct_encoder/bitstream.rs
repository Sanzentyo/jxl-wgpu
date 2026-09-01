//! JPEG XL VarDCT headers, control packets, and GPU fragments.

use jxl_gpu_bitstream::BitWriter;

use super::entropy::HfEntropyPlan;
use super::types::{
    GLOBAL_SCALE, HF_MUL, QUANT_LF, ScalableDcFragmentDescriptor, VarDctArtifactData,
    VarDctFrameLayout, VarDctLfMetadata,
};
use crate::prefix::PrefixCode;
use crate::{
    BackendError, BitFragment, EncodeError, FrameGroupLayout, FramePacketSet, GroupPacket,
    GroupPacketKind,
};

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(1..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "VarDCT dimensions must be in 1..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

pub(super) fn image_header(width: u32, height: u32) -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0x0aff, 16)?;
    output.write_bits(0, 1)?; // dimensions are not encoded as multiples of eight
    write_size(&mut output, height, true)?;
    write_size(&mut output, width, false)?;
    output.write_bits(1, 1)?; // all-default image metadata: 8-bit, XYB, sRGB presentation
    output.write_bits(1, 1)?; // default opsin inverse matrix and upsampling weights
    output.align_to_byte()?;
    Ok(BitFragment::byte_aligned(output.into_bytes())?)
}

fn frame_header() -> Result<BitFragment, EncodeError> {
    let mut output = BitWriter::new();
    output.write_bits(0, 1)?; // non-default so restoration can be disabled
    output.write_bits(0, 2)?; // regular frame
    output.write_bits(0, 1)?; // VarDCT
    output.write_bits(0, 2)?; // no frame flags
    output.write_bits(0, 2)?; // no upsampling
    output.write_bits(3, 3)?; // default X quant-matrix scale
    output.write_bits(2, 3)?; // default B quant-matrix scale
    output.write_bits(0, 2)?; // one pass
    output.write_bits(0, 1)?; // full-canvas frame
    output.write_bits(0, 2)?; // replace blending
    output.write_bits(1, 1)?; // final frame
    output.write_bits(0, 2)?; // empty frame name
    output.write_bits(0, 1)?; // non-default restoration filter
    output.write_bits(0, 1)?; // no Gaborish
    output.write_bits(0, 2)?; // no EPF
    output.write_bits(0, 2)?; // no restoration extensions
    output.write_bits(0, 2)?; // no frame extensions
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
    codes: &[PrefixCode; 4],
) -> Result<(), EncodeError> {
    // A fixed four-cluster MA tree. All four distributions are identical so
    // stream/channel routing cannot change the GPU token bit representation.
    output.write_bits(1, 1)?; // global MA tree present
    output.write_bits(0, 1)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    output.write_bits(0b100011, 6)?;
    output.write_bits(1, 2)?;
    output.write_bits(3, 2)?;
    for symbol in 0..4 {
        output.write_bits(symbol, 2)?;
    }
    output.write_bits(0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        output.write_bits(SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    output.write_bits(1, 1)?;
    output.write_bits(0, 2)?;
    output.write_bits(0b1010, 4)?;
    output.write_bits(4, 4)?;
    output.write_bits(0, 3)?;
    output.write_bits(0, 3)?;
    output.write_bits(1, 1)?;
    output.write_bits(3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        output.write_bits(context, 3)?;
    }
    output.write_bits(1, 1)?;
    output.write_bits(0, 4)?;
    for _ in 0..4 {
        output.write_bits(0, 4)?;
    }
    output.write_bits(1, 5)?;
    for _ in 0..4 {
        output.write_bits(1, 1)?;
        output.write_bits(8, 4)?;
        output.write_bits(0, 8)?;
    }
    output.write_bits(1, 2)?;
    output.write_bits(0, 2)?;
    output.write_bits(1, 1)?;
    for code in codes {
        code.write_tree(output)?;
    }
    Ok(())
}

fn write_lf_global(
    output: &mut BitWriter,
    code: &PrefixCode,
    hf_entropy: &HfEntropyPlan,
    coefficient_payload: bool,
    lf_metadata: VarDctLfMetadata,
) -> Result<(), EncodeError> {
    output.write_bits(u64::from(lf_metadata.has_default_dequantization()), 1)?;
    if !lf_metadata.has_default_dequantization() {
        for value in lf_metadata.lf_dequantization {
            output.write_bits(u64::from(value.to_bits()), 16)?;
        }
    }
    write_u32(
        output,
        GLOBAL_SCALE,
        [(1, 11), (2_049, 11), (4_097, 12), (8_193, 16)],
    )?;
    write_u32(output, QUANT_LF, [(16, 0), (1, 5), (1, 8), (1, 16)])?;
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
    write_global_ma_config(
        output,
        &[code.clone(), code.clone(), code.clone(), code.clone()],
    )
}

fn write_local_modular_header(output: &mut BitWriter) -> Result<(), EncodeError> {
    output.write_bits(1, 1)?; // use the LF-global MA tree
    output.write_bits(1, 1)?; // default weighted-predictor header
    output.write_bits(0, 2)?; // zero transforms
    Ok(())
}

fn write_unsigned_token(
    output: &mut BitWriter,
    code: &PrefixCode,
    value: u32,
) -> Result<(), EncodeError> {
    if value == 0 {
        return code.write_raw(output, 0, 0, 0);
    }
    let nbits = 31 - value.leading_zeros();
    let token = nbits + 1;
    code.write_raw(output, token, nbits, value - (1 << nbits))
}

pub(super) fn pack_signed_control(value: i32) -> u32 {
    if value < 0 {
        value.unsigned_abs() * 2 - 1
    } else {
        value as u32 * 2
    }
}

fn append_gpu_dc_fragment(
    output: &mut BitWriter,
    fragment_words: &[u32],
    descriptor: ScalableDcFragmentDescriptor,
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

fn append_gpu_ac_fragment(
    output: &mut BitWriter,
    artifact: VarDctArtifactData<'_>,
) -> Result<(), EncodeError> {
    append_gpu_fragment(
        output,
        artifact.ac_fragment_words,
        0,
        artifact.ac_fragment_bit_len,
    )
}

fn append_gpu_fragment(
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
    for source_bit in bit_offset..bit_end {
        let word = fragment_words[source_bit / 32];
        output.write_bits(u64::from((word >> (source_bit % 32)) & 1), 1)?;
    }
    Ok(())
}

fn write_lf_group(
    output: &mut BitWriter,
    code: &PrefixCode,
    artifact: VarDctArtifactData<'_>,
    frame: VarDctFrameLayout,
    group_index: u32,
) -> Result<(), EncodeError> {
    let group = frame.lf_group_blocks(group_index)?;
    let block_count = group.block_count()?;
    output.write_bits(0, 2)?; // no extra LF precision
    write_local_modular_header(output)?;
    append_gpu_dc_fragment(
        output,
        artifact.dc_fragment_words,
        artifact.dc_fragment_descriptor(group_index)?,
    )?;

    // GPU-selected regular strategies, no chroma-from-luma correction, fixed
    // HF multiplier, and zero EPF sharpness. Source-dependent DC entropy was
    // already packed by the GPU; these values describe its control map.
    let first_block_bits = block_count.next_power_of_two().trailing_zeros() as u8;
    output.write_bits(
        u64::from(group.first_block_count.checked_sub(1).ok_or(
            EncodeError::InvalidConfiguration("VarDCT frame has no first transform block"),
        )?),
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
    for _ in 0..group.first_block_count {
        write_unsigned_token(output, code, pack_signed_control(artifact.strategy as i32))?;
    }
    let first_quant_residual = (HF_MUL - 1) - artifact.strategy as i32;
    write_unsigned_token(output, code, pack_signed_control(first_quant_residual))?;
    for _ in 1..group.first_block_count {
        write_unsigned_token(output, code, 0)?;
    }
    for _ in 0..block_count {
        write_unsigned_token(output, code, 0)?;
    }
    Ok(())
}

pub(super) fn build_frame_packet(
    artifact: VarDctArtifactData<'_>,
    code: &PrefixCode,
    hf_entropy: &HfEntropyPlan,
    frame: VarDctFrameLayout,
    lf_metadata: VarDctLfMetadata,
) -> Result<FramePacketSet, EncodeError> {
    let ac_groups = frame.ac_group_count()?;
    let lf_groups = frame.lf_group_count()?;
    let coefficient_payload = artifact.has_ac_payload();
    if ac_groups == 1 && lf_groups == 1 {
        let mut group = BitWriter::new();
        write_lf_global(
            &mut group,
            code,
            hf_entropy,
            coefficient_payload,
            lf_metadata,
        )?;
        write_lf_group(&mut group, code, artifact, frame, 0)?;
        hf_entropy.write_global(&mut group, ac_groups, coefficient_payload)?;
        append_gpu_ac_fragment(&mut group, artifact)?;
        group.align_to_byte()?;
        return Ok(FramePacketSet::new(
            frame_header()?,
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
        lf_metadata,
    )?;
    dc_global.align_to_byte()?;
    let mut ac_global = BitWriter::new();
    hf_entropy.write_global(&mut ac_global, ac_groups, coefficient_payload)?;
    ac_global.align_to_byte()?;

    let mut packets = Vec::with_capacity(
        usize::try_from(ac_groups.checked_add(lf_groups).ok_or(
            EncodeError::InvalidConfiguration("VarDCT packet count overflow"),
        )?)
        .map_err(|_| EncodeError::InvalidConfiguration("VarDCT packet count overflow"))?
            + 2,
    );
    packets.push(GroupPacket::new(
        GroupPacketKind::DcGlobal,
        dc_global.into_bytes(),
    ));
    for group_index in 0..lf_groups {
        let mut dc_group = BitWriter::new();
        write_lf_group(&mut dc_group, code, artifact, frame, group_index)?;
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
    packets
        .extend((0..ac_groups).map(|group| {
            GroupPacket::new(GroupPacketKind::AcGroup { pass: 0, group }, Vec::new())
        }));
    Ok(FramePacketSet::new(
        frame_header()?,
        FrameGroupLayout::new(lf_groups, ac_groups, 1)?,
        packets,
    )?)
}
