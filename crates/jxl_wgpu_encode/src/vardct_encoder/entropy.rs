//! Fixed prefix and HF entropy planning for VarDCT.

use jxl_gpu_bitstream::{BitWriter, PrefixCodeEntry};

use super::types::GpuPrefixEntry;
use crate::prefix::RawPrefixCode;

pub(super) const UINT_SYMBOLS: usize = 33;
pub(super) type VarDctPrefixCode = RawPrefixCode<UINT_SYMBOLS>;
use crate::{BackendError, EncodeError};

pub(super) fn fixed_prefix_code() -> Result<VarDctPrefixCode, EncodeError> {
    VarDctPrefixCode::from_counts(&[1; UINT_SYMBOLS])
}

/// Entropy policy shared by HF-global metadata and the GPU pass-group serializer.
///
/// Every coefficient context currently maps to one prefix distribution. Tiled DCT8's independent
/// block fragments rely on that stateless policy. A future contextual or ANS policy must preserve
/// its state on the GPU and provide complete group fragments instead.
#[derive(Clone, Debug)]
pub(super) struct HfEntropyPlan {
    pub(super) code: VarDctPrefixCode,
}

impl HfEntropyPlan {
    pub(super) fn single_cluster_prefix() -> Result<Self, EncodeError> {
        Ok(Self {
            code: VarDctPrefixCode::from_counts(&[1; UINT_SYMBOLS])?,
        })
    }

    pub(super) fn gpu_entries(&self) -> [GpuPrefixEntry; UINT_SYMBOLS] {
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
        config: &super::VarDctConfig,
    ) -> Result<(), EncodeError> {
        config.dequant_matrices.write(output)?;
        if !coefficient_payload {
            // The historical zero-HF artifact has no coefficient order-dependent payload.
            // Prefix single-symbol distributions consume no pass-group payload bits.
            let histogram_bits = ac_groups.next_power_of_two().trailing_zeros() as u8;
            output.write_bits(0, histogram_bits)?;
            output.write_bits(0x124a, 17)?;
            return Ok(());
        }

        let preset_bits = ac_groups.next_power_of_two().trailing_zeros() as u8;
        output.write_bits(0, preset_bits)?; // one HF preset
        config.coefficient_orders.write(output)?;

        write_prefix_config(output, &self.code, 495)
    }
}

pub(super) fn prefix_entries(code: &VarDctPrefixCode) -> [GpuPrefixEntry; UINT_SYMBOLS] {
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
                "VarDCT fragment has nonzero high padding bits",
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
            "VarDCT fragment word padding is nonzero",
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

/// One raw prefix distribution shared by every context, with split exponent zero.
pub(super) fn write_prefix_config(
    output: &mut BitWriter,
    code: &VarDctPrefixCode,
    contexts: u32,
) -> Result<(), EncodeError> {
    output.write_bits(0, 1)?; // LZ77 disabled
    if contexts > 1 {
        output.write_bits(1, 1)?; // simple clustering
        output.write_bits(0, 2)?; // every context uses distribution zero
    }
    output.write_bits(1, 1)?; // prefix code
    output.write_bits(0, 4)?; // hybrid integer split exponent zero
    output.write_bits(1, 1)?; // explicit alphabet size
    output.write_bits(5, 4)?;
    output.write_bits(0, 5)?; // 1 + 2^5 = 33 symbols, covering every u32
    code.write_raw_tree(output)
}
