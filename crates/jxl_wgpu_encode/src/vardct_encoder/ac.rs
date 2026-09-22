//! GPU coefficient fragments and their checked placement in pass groups.

use jxl_gpu_bitstream::BitWriter;

use super::bitstream::append_gpu_fragment;
use super::entropy::{HfEntropyPlan, read_fragment_slice, validate_fragment_padding};
use super::types::{AC_GROUP_DIM_PIXELS, VarDctFrameLayout};
use crate::{BackendError, EncodeError};

#[derive(Clone, Copy)]
pub(super) enum AcFragments<'a> {
    StrategyMap {
        words: &'a [u32],
        bit_lengths: &'a [u32],
        plan: &'a super::strategy_map::TransformPlan,
    },
    Empty,
    Single {
        words: &'a [u32],
        bit_lengths: &'a [u32],
        words_per_pass: u32,
    },
    Dct8Blocks {
        words: &'a [u32],
        bit_lengths: &'a [u32],
        words_per_block: u32,
    },
}

impl AcFragments<'_> {
    pub(super) fn append_group(
        self,
        output: &mut BitWriter,
        frame: VarDctFrameLayout,
        group: u32,
        pass: u32,
    ) -> Result<(), EncodeError> {
        if group >= frame.ac_group_count()? {
            return Err(BackendError::Invariant("VarDCT AC group is out of range").into());
        }
        match self {
            Self::StrategyMap {
                words,
                bit_lengths,
                plan,
            } => {
                let count = plan.tasks.len();
                let passes = bit_lengths.len() / count;
                if pass as usize >= passes {
                    return Err(BackendError::Invariant("VarDCT AC pass is out of range").into());
                }
                let pass_words = words.len() / passes;
                let words = &words[pass as usize * pass_words..(pass as usize + 1) * pass_words];
                let bit_lengths = &bit_lengths[pass as usize * count..(pass as usize + 1) * count];
                for &index in &plan.ac_groups[group as usize] {
                    let task = &plan.tasks[index];
                    let start = task.ac_word_offset as usize;
                    let end = start + task.ac_word_capacity as usize;
                    append_gpu_fragment(output, &words[start..end], 0, bit_lengths[index])?;
                }
                Ok(())
            }
            Self::Empty => Ok(()),
            Self::Single {
                words,
                bit_lengths,
                words_per_pass,
            } => {
                if frame.ac_group_count()? != 1 {
                    return Err(
                        BackendError::Invariant("single AC fragment in a tiled frame").into(),
                    );
                }
                let bit_len = *bit_lengths
                    .get(pass as usize)
                    .ok_or(BackendError::Invariant("VarDCT AC pass is out of range"))?;
                let start = pass as usize * words_per_pass as usize;
                append_gpu_fragment(
                    output,
                    &words[start..start + words_per_pass as usize],
                    0,
                    bit_len,
                )
            }
            Self::Dct8Blocks {
                words,
                bit_lengths,
                words_per_block,
            } => {
                let side = AC_GROUP_DIM_PIXELS / 8;
                let x0 = group % frame.ac_groups_x * side;
                let y0 = group / frame.ac_groups_x * side;
                let x1 = (x0 + side).min(frame.blocks_x);
                let y1 = (y0 + side).min(frame.blocks_y);
                let pass_base = pass
                    .checked_mul(frame.blocks_x * frame.blocks_y)
                    .ok_or(BackendError::Invariant("VarDCT AC pass offset overflow"))?;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let block = (pass_base + y * frame.blocks_x + x) as usize;
                        let bit_len = *bit_lengths
                            .get(block)
                            .ok_or(BackendError::Invariant("missing VarDCT AC block length"))?;
                        let start = block
                            .checked_mul(words_per_block as usize)
                            .ok_or(BackendError::Invariant("VarDCT AC block offset overflow"))?;
                        let end = start
                            .checked_add(words_per_block as usize)
                            .ok_or(BackendError::Invariant("VarDCT AC block end overflow"))?;
                        let fragment = words.get(start..end).ok_or(BackendError::Invariant(
                            "VarDCT AC block exceeds its allocation",
                        ))?;
                        append_gpu_fragment(output, fragment, 0, bit_len)?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Validation only: no decoded coefficients are retained, transformed or re-encoded.
/// A block must contain three Y/X/B counts and exactly the coefficients those counts
/// describe, followed by zero allocation padding. Zero-length or truncated workgroup
/// output must never silently become a valid all-zero block.
pub(super) fn validate_blocks(
    words: &[u32],
    bit_lengths: &[u32],
    words_per_block: u32,
    entropy: &HfEntropyPlan,
) -> Result<(), BackendError> {
    validate_transform_fragments(words, bit_lengths, words_per_block, 63, entropy)
}

pub(super) fn validate_transform_fragments(
    words: &[u32],
    bit_lengths: &[u32],
    words_per_block: u32,
    maximum_nonzero: u32,
    entropy: &HfEntropyPlan,
) -> Result<(), BackendError> {
    let stride = words_per_block as usize;
    if stride == 0 || bit_lengths.len().checked_mul(stride) != Some(words.len()) {
        return Err(BackendError::InvalidArtifact(
            "VarDCT AC block allocation mismatch",
        ));
    }
    let entries = entropy.gpu_entries();
    for (words, &bit_len) in words.chunks_exact(stride).zip(bit_lengths) {
        let mut cursor = 0u32;
        let mut unsigned = || {
            for (symbol, entry) in entries.iter().enumerate() {
                if cursor
                    .checked_add(entry.bit_len)
                    .is_none_or(|end| end > bit_len)
                {
                    continue;
                }
                if read_fragment_slice(words, bit_len, cursor, entry.bit_len)? != entry.bits {
                    continue;
                }
                cursor += entry.bit_len;
                if symbol == 0 {
                    return Ok(0);
                }
                let extra_bits = symbol as u32 - 1;
                let extra = read_fragment_slice(words, bit_len, cursor, extra_bits)?;
                cursor += extra_bits;
                return Ok((1 << extra_bits) + extra);
            }
            Err(BackendError::InvalidArtifact(
                "VarDCT AC prefix is truncated or invalid",
            ))
        };
        for _ in 0..3 {
            let mut remaining = unsigned()?;
            if remaining > maximum_nonzero {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT nonzero count exceeds the transform AC area",
                ));
            }
            for _ in 0..maximum_nonzero {
                if remaining == 0 {
                    break;
                }
                let coefficient = unsigned()?;
                remaining -= u32::from(coefficient != 0);
            }
            if remaining != 0 {
                return Err(BackendError::InvalidArtifact(
                    "VarDCT nonzero count is inconsistent",
                ));
            }
        }
        if cursor != bit_len {
            return Err(BackendError::InvalidArtifact(
                "VarDCT AC block has trailing entropy bits",
            ));
        }
        validate_fragment_padding(words, bit_len)?;
    }
    Ok(())
}
