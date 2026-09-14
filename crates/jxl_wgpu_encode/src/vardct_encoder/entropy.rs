//! Fixed prefix and HF entropy planning for VarDCT.

use jxl_gpu_bitstream::{BitWriter, PrefixCodeEntry};

use super::types::GpuPrefixEntry;
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};
use crate::{BackendError, EncodeError};

pub(super) fn fixed_prefix_code() -> Result<PrefixCode, EncodeError> {
    PrefixCode::from_aggregated_counts(&[0; RAW_SYMBOLS], &[0; LZ77_SYMBOLS], RAW_SYMBOLS - 1, true)
}

/// Entropy policy shared by HF-global metadata and the GPU pass-group serializer.
///
/// Every coefficient context currently maps to one prefix distribution. Tiled DCT8's independent
/// block fragments rely on that stateless policy. A future contextual or ANS policy must preserve
/// its state on the GPU and provide complete group fragments instead.
#[derive(Clone, Debug)]
pub(super) struct HfEntropyPlan {
    pub(super) code: PrefixCode,
}

impl HfEntropyPlan {
    pub(super) fn single_cluster_prefix() -> Result<Self, EncodeError> {
        Ok(Self {
            code: PrefixCode::from_raw_counts(&[1; RAW_SYMBOLS])?,
        })
    }

    pub(super) fn gpu_entries(&self) -> [GpuPrefixEntry; RAW_SYMBOLS] {
        prefix_entries(&self.code)
    }

    pub(super) fn write_block_context(
        &self,
        output: &mut BitWriter,
        coefficient_payload: bool,
    ) -> Result<(), EncodeError> {
        if !coefficient_payload {
            output.write_bits(1, 1)?; // standard 15-cluster shortcut used by zero-HF streams
            return Ok(());
        }

        output.write_bits(0, 1)?; // explicit HF block-context model
        output.write_bits(0, 4)?; // no X LF thresholds
        output.write_bits(0, 4)?; // no Y LF thresholds
        output.write_bits(0, 4)?; // no B LF thresholds
        output.write_bits(0, 4)?; // no HF quant-field thresholds
        output.write_bits(1, 1)?; // simple context clustering
        output.write_bits(0, 2)?; // zero-bit cluster IDs: all 39 block contexts map to cluster 0
        Ok(())
    }

    pub(super) fn write_global(
        &self,
        output: &mut BitWriter,
        ac_groups: u32,
        coefficient_payload: bool,
    ) -> Result<(), EncodeError> {
        if !coefficient_payload {
            // Default matrices, natural order, and the historical single-symbol-zero decoder.
            // Prefix single-symbol distributions consume no pass-group payload bits.
            output.write_bits(1, 1)?;
            let histogram_bits = ac_groups.next_power_of_two().trailing_zeros() as u8;
            output.write_bits(0, histogram_bits)?;
            output.write_bits(0x124a, 17)?;
            return Ok(());
        }

        output.write_bits(1, 1)?; // all default dequantization matrices
        let preset_bits = ac_groups.next_power_of_two().trailing_zeros() as u8;
        output.write_bits(0, preset_bits)?; // one HF preset
        output.write_bits(2, 2)?; // used_orders = 0: natural coefficient order

        output.write_bits(0, 1)?; // LZ77 disabled
        output.write_bits(1, 1)?; // simple distribution clustering
        output.write_bits(0, 2)?; // all 495 coefficient contexts map to cluster 0
        output.write_bits(1, 1)?; // prefix code
        output.write_bits(0, 4)?; // hybrid integer split exponent zero
        output.write_bits(1, 1)?; // explicit alphabet size
        output.write_bits(4, 4)?;
        output.write_bits(2, 4)?; // 1 + 2^4 + 2 = 19 symbols
        self.code.write_raw_tree(output)
    }
}

pub(super) fn prefix_entries(code: &PrefixCode) -> [GpuPrefixEntry; RAW_SYMBOLS] {
    code.raw_entries()
        .map(|PrefixCodeEntry { bit_len, bits }| GpuPrefixEntry {
            bits: u32::from(bits),
            bit_len: u32::from(bit_len),
        })
}

pub(super) fn validate_fragment_padding(words: &[u32], bit_len: u32) -> Result<(), BackendError> {
    let used_words = bit_len
        .checked_add(31)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment word count overflow",
        ))?
        / 32;
    let used_words = usize::try_from(used_words)
        .map_err(|_| BackendError::InvalidArtifact("VarDCT fragment size does not fit usize"))?;
    if let Some(&last_word) = used_words.checked_sub(1).and_then(|index| words.get(index)) {
        let live_bits = bit_len % 32;
        if live_bits != 0 && last_word & !((1u32 << live_bits) - 1) != 0 {
            return Err(BackendError::InvalidArtifact(
                "scalable VarDCT fragment has nonzero high padding bits",
            ));
        }
    }
    if words
        .get(used_words..)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT fragment used-word count is out of bounds",
        ))?
        .iter()
        .any(|&word| word != 0)
    {
        return Err(BackendError::InvalidArtifact(
            "scalable VarDCT fragment word padding is nonzero",
        ));
    }
    Ok(())
}

pub(super) fn read_fragment_slice(
    words: &[u32],
    bit_len: u32,
    start: u32,
    count: u32,
) -> Result<u32, BackendError> {
    let end = start
        .checked_add(count)
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment address overflow",
        ))?;
    let capacity = u32::try_from(words.len())
        .ok()
        .and_then(|len| len.checked_mul(32))
        .ok_or(BackendError::InvalidArtifact(
            "VarDCT GPU fragment capacity overflow",
        ))?;
    if count > 32 || end > bit_len || end > capacity {
        return Err(BackendError::InvalidArtifact(
            "VarDCT GPU fragment is truncated",
        ));
    }
    let mut value = 0u32;
    for index in 0..count {
        let bit = start + index;
        value |= ((words[(bit / 32) as usize] >> (bit % 32)) & 1) << index;
    }
    Ok(value)
}
