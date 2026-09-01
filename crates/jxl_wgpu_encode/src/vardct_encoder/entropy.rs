//! Fixed prefix and HF entropy planning for VarDCT.

use jxl_gpu_bitstream::{BitWriter, PrefixCodeEntry};

use super::types::GpuPrefixEntry;
use crate::EncodeError;
use crate::prefix::{LZ77_SYMBOLS, PrefixCode, RAW_SYMBOLS};

pub(super) fn fixed_prefix_code() -> Result<PrefixCode, EncodeError> {
    PrefixCode::from_aggregated_counts(&[0; RAW_SYMBOLS], &[0; LZ77_SYMBOLS], RAW_SYMBOLS - 1, true)
}

/// Entropy policy shared by HF-global metadata and the GPU pass-group serializer.
///
/// Stage 1 deliberately maps every legal coefficient context to one prefix distribution. The
/// plan boundary is permanent: adaptive context clustering and ANS can add plan variants without
/// changing the GPU fragment contract or moving coefficient scans to the host.
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
