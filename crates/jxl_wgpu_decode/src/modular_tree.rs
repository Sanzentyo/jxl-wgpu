//! Bounded JPEG XL Modular entropy descriptors and meta-adaptive trees.
//!
//! This module deliberately has two different execution domains. Entropy used to describe the
//! MA tree is consumed on the host because the resulting tree is metadata. The image entropy
//! decoder is only parsed into [`EntropyDecoderIr`]; its symbols remain in the codestream for the
//! GPU kernel.

use std::collections::VecDeque;

use jxl_gpu_bitstream::{BitReader, PrefixCodeEntry};

use crate::{ModularTreeError, Result};

pub(crate) trait BitInput {
    fn bit_offset(&self) -> u64;
    fn read_bits(&mut self, count: u8) -> Result<u64>;
}

impl<'input> BitInput for BitReader<'input> {
    fn bit_offset(&self) -> u64 {
        BitReader::bit_offset(self)
    }

    fn read_bits(&mut self, count: u8) -> Result<u64> {
        Ok(BitReader::read_bits(self, count)?)
    }
}

const MAX_PREFIX_BITS: u8 = 15;
const MAX_PREFIX_ALPHABET_SIZE: usize = 1 << MAX_PREFIX_BITS;
const ANS_SIGNATURE: u32 = 0x13_0000;
pub(crate) const GPU_METADATA_HEADER_WORDS: usize = 24;
pub(crate) const GPU_CONFIG_WORDS: usize = 4;
pub(crate) const GPU_TREE_NODE_WORDS: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedModularMetadata {
    pub words: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MaTreeLimits {
    pub node_limit: usize,
    pub depth_limit: usize,
    pub context_limit: usize,
    pub cluster_limit: usize,
    pub metadata_symbol_limit: usize,
}

impl Default for MaTreeLimits {
    fn default() -> Self {
        Self {
            node_limit: 1 << 16,
            depth_limit: 64,
            context_limit: 1 << 16,
            cluster_limit: 256,
            metadata_symbol_limit: 1 << 18,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HybridIntegerConfigIr {
    pub split_exponent: u32,
    pub msb_in_token: u32,
    pub lsb_in_token: u32,
}

impl HybridIntegerConfigIr {
    fn parse(reader: &mut impl BitInput, log_alphabet_size: u32) -> Result<Self> {
        let bit_offset = reader.bit_offset();
        let split_exponent_bits = add_log2_ceil(log_alphabet_size);
        let split_exponent = read_bits_u32(reader, split_exponent_bits)?;
        let (msb_in_token, lsb_in_token) = if split_exponent != log_alphabet_size {
            let msb_in_token = read_bits_u32(reader, add_log2_ceil(split_exponent))?;
            if msb_in_token > split_exponent {
                return invalid_entropy("hybrid integer MSB count exceeds split exponent");
            }
            let lsb_in_token = read_bits_u32(
                reader,
                add_log2_ceil(split_exponent.saturating_sub(msb_in_token)),
            )?;
            (msb_in_token, lsb_in_token)
        } else {
            (0, 0)
        };
        if msb_in_token + lsb_in_token > split_exponent {
            return Err(ModularTreeError::InvalidHybridConfig {
                bit_offset,
                log_alphabet_size,
                split_exponent,
                msb_in_token,
                lsb_in_token,
            }
            .into());
        }
        Ok(Self {
            split_exponent,
            msb_in_token,
            lsb_in_token,
        })
    }

    fn split(self) -> u32 {
        1u32 << self.split_exponent
    }

    /// Inclusive bounds for every integer represented by one entropy token.
    ///
    /// The extra bits form a monotonic range before the JPEG XL `u32` result. If a malformed or
    /// future descriptor could make that range exceed `u32`, using the full upper bound keeps GPU
    /// LZ history sizing conservative without consuming any image entropy on the host.
    fn value_bounds(self, token: u32) -> (u32, u32) {
        if token < self.split() {
            return (token, token);
        }
        let embedded = self.msb_in_token + self.lsb_in_token;
        let bit_count =
            (self.split_exponent - embedded + ((token - self.split()) >> embedded)) & 31;
        let low_mask = (1u32 << self.lsb_in_token).wrapping_sub(1);
        let low = token & low_mask;
        let shifted = token >> self.lsb_in_token;
        let high_mask = (1u32 << self.msb_in_token).wrapping_sub(1);
        let high = (shifted & high_mask) | (1 << self.msb_in_token);
        let assemble = |extra: u64| {
            (((u64::from(high) << bit_count) | extra) << self.lsb_in_token) | u64::from(low)
        };
        let minimum = assemble(0);
        let maximum = assemble((1u64 << bit_count) - 1);
        (
            u32::try_from(minimum).unwrap_or(0),
            u32::try_from(maximum).unwrap_or(u32::MAX),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PrefixHistogramIr {
    pub entries: Vec<PrefixCodeEntry>,
    pub single_symbol: Option<u32>,
}

impl PrefixHistogramIr {
    fn single(symbol: usize) -> Result<Self> {
        let mut entries = vec![PrefixCodeEntry::default(); symbol + 1];
        entries[symbol] = PrefixCodeEntry {
            bit_len: 0,
            bits: 0,
        };
        Ok(Self {
            entries,
            single_symbol: Some(
                u32::try_from(symbol)
                    .map_err(|_| invalid_entropy_error("prefix symbol exceeds u32"))?,
            ),
        })
    }

    fn parse(reader: &mut impl BitInput, alphabet_size: usize) -> Result<Self> {
        if alphabet_size == 0 || alphabet_size > MAX_PREFIX_ALPHABET_SIZE {
            return invalid_entropy("prefix alphabet size is outside 1 through 32768");
        }
        if alphabet_size == 1 {
            return Self::single(0);
        }

        let hskip = read_bits_u32(reader, 2)?;
        if hskip == 1 {
            return Self::parse_simple(reader, alphabet_size);
        }
        Self::parse_complex(reader, alphabet_size, hskip)
    }

    fn parse_simple(reader: &mut impl BitInput, alphabet_size: usize) -> Result<Self> {
        let alphabet_bits = alphabet_size.next_power_of_two().trailing_zeros();
        let symbol_count = read_bits_u32(reader, 2)? + 1;
        let mut lengths = vec![0u8; alphabet_size];
        let mut symbols = [0usize; 4];
        for symbol in symbols.iter_mut().take(symbol_count as usize) {
            *symbol = usize::try_from(read_bits_u32(reader, alphabet_bits)?)
                .map_err(|_| invalid_entropy_error("prefix symbol exceeds host address space"))?;
            if *symbol >= alphabet_size {
                return invalid_entropy("simple prefix symbol exceeds its alphabet");
            }
        }
        let code_lengths: &[u8] = match symbol_count {
            1 => &[0],
            2 => &[1, 1],
            3 => &[1, 2, 2],
            4 if reader.read_bits(1)? != 0 => &[1, 2, 3, 3],
            4 => &[2, 2, 2, 2],
            _ => unreachable!(),
        };
        for (&symbol, &length) in symbols.iter().zip(code_lengths) {
            if lengths[symbol] != 0 || (length == 0 && symbol_count != 1) {
                return invalid_entropy("simple prefix histogram repeats a symbol");
            }
            lengths[symbol] = length;
        }
        if symbol_count == 1 {
            return Self::single(symbols[0]);
        }
        Ok(Self {
            entries: canonical_entries(&lengths)?,
            single_symbol: None,
        })
    }

    fn parse_complex(reader: &mut impl BitInput, alphabet_size: usize, hskip: u32) -> Result<Self> {
        const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let skip = usize::try_from(hskip)
            .map_err(|_| invalid_entropy_error("prefix hskip exceeds host address space"))?;
        if skip > ORDER.len() {
            return invalid_entropy("prefix hskip exceeds the code-length alphabet");
        }
        let mut code_length_lengths = [0u8; 18];
        let mut accumulator = 0u32;
        let mut nonzero_count = 0usize;
        let mut nonzero_symbol = 0usize;
        for &symbol in ORDER.iter().skip(skip) {
            let length = read_code_length_code(reader)?;
            code_length_lengths[symbol] = length;
            if length == 0 {
                continue;
            }
            accumulator = accumulator
                .checked_add(32u32 >> length)
                .ok_or_else(|| invalid_entropy_error("code-length histogram overflow"))?;
            nonzero_count += 1;
            nonzero_symbol = symbol;
            match accumulator.cmp(&32) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Equal => break,
                std::cmp::Ordering::Greater => {
                    return invalid_entropy("over-subscribed code-length histogram");
                }
            }
        }
        if nonzero_count != 1 && accumulator != 32 {
            return invalid_entropy("incomplete code-length histogram");
        }
        let code_length_entries = if nonzero_count == 1 {
            PrefixHistogramIr::single(nonzero_symbol)?.entries
        } else {
            canonical_entries(&code_length_lengths)?
        };

        let mut lengths = vec![0u8; alphabet_size];
        let mut accumulator = 0u32;
        let mut previous_symbol = 8u8;
        let mut last_nonzero_length = 8u8;
        let mut last_repeat_count = 0usize;
        let mut repeat_count = 0usize;
        let mut repeat_length = 0u8;
        for length in &mut lengths {
            if repeat_count != 0 {
                repeat_count -= 1;
                *length = repeat_length;
            } else {
                let symbol = u8::try_from(read_prefix_symbol(reader, &code_length_entries)?)
                    .map_err(|_| invalid_entropy_error("code-length symbol exceeds u8"))?;
                match symbol {
                    0 => *length = 0,
                    1..=15 => {
                        *length = symbol;
                        last_nonzero_length = symbol;
                    }
                    16 => {
                        let mut current = usize::try_from(read_bits_u32(reader, 2)?)
                            .map_err(|_| invalid_entropy_error("prefix repeat count overflow"))?
                            + 3;
                        if previous_symbol == 16 {
                            current = current
                                .checked_add(
                                    last_repeat_count
                                        .checked_mul(3)
                                        .and_then(|value| value.checked_sub(8))
                                        .ok_or_else(|| {
                                            invalid_entropy_error("prefix repeat count overflow")
                                        })?,
                                )
                                .ok_or_else(|| {
                                    invalid_entropy_error("prefix repeat count overflow")
                                })?;
                            last_repeat_count =
                                last_repeat_count.checked_add(current).ok_or_else(|| {
                                    invalid_entropy_error("prefix repeat count overflow")
                                })?;
                        } else {
                            last_repeat_count = current;
                        }
                        repeat_count = current
                            .checked_sub(1)
                            .ok_or_else(|| invalid_entropy_error("zero prefix repeat count"))?;
                        repeat_length = last_nonzero_length;
                        *length = last_nonzero_length;
                    }
                    17 => {
                        let mut current = usize::try_from(read_bits_u32(reader, 3)?)
                            .map_err(|_| invalid_entropy_error("zero repeat count overflow"))?
                            + 3;
                        if previous_symbol == 17 {
                            current = current
                                .checked_add(
                                    last_repeat_count
                                        .checked_mul(7)
                                        .and_then(|value| value.checked_sub(16))
                                        .ok_or_else(|| {
                                            invalid_entropy_error("zero repeat count overflow")
                                        })?,
                                )
                                .ok_or_else(|| {
                                    invalid_entropy_error("zero repeat count overflow")
                                })?;
                            last_repeat_count =
                                last_repeat_count.checked_add(current).ok_or_else(|| {
                                    invalid_entropy_error("zero repeat count overflow")
                                })?;
                        } else {
                            last_repeat_count = current;
                        }
                        repeat_count = current
                            .checked_sub(1)
                            .ok_or_else(|| invalid_entropy_error("zero repeat count"))?;
                        repeat_length = 0;
                        *length = 0;
                    }
                    _ => return invalid_entropy("invalid prefix code-length symbol"),
                }
                previous_symbol = symbol;
            }

            if *length != 0 {
                accumulator = accumulator
                    .checked_add(1u32 << (u32::from(MAX_PREFIX_BITS) - u32::from(*length)))
                    .ok_or_else(|| invalid_entropy_error("prefix histogram overflow"))?;
                if accumulator > 1 << MAX_PREFIX_BITS {
                    return invalid_entropy("over-subscribed prefix histogram");
                }
                if accumulator == 1 << MAX_PREFIX_BITS && repeat_count == 0 {
                    break;
                }
            }
        }
        if accumulator != 1 << MAX_PREFIX_BITS || repeat_count != 0 {
            return invalid_entropy("incomplete prefix histogram");
        }
        Ok(Self {
            entries: canonical_entries(&lengths)?,
            single_symbol: None,
        })
    }

    fn single_symbol(&self) -> Option<u32> {
        self.single_symbol
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub(crate) struct AnsBucketIr {
    /// alias symbol (bits 0..7), alias cutoff (8..15), distribution (16..31)
    pub symbol_cutoff_dist: u32,
    /// alias offset (bits 0..15), distribution xor (16..31)
    pub offset_dist_xor: u32,
}

impl AnsBucketIr {
    fn new(
        alias_symbol: u8,
        alias_cutoff: u8,
        distribution: u16,
        alias_offset: u16,
        alias_dist_xor: u16,
    ) -> Self {
        Self {
            symbol_cutoff_dist: u32::from(alias_symbol)
                | (u32::from(alias_cutoff) << 8)
                | (u32::from(distribution) << 16),
            offset_dist_xor: u32::from(alias_offset) | (u32::from(alias_dist_xor) << 16),
        }
    }

    fn fields(self) -> (u32, u32, u32, u32, u32) {
        (
            self.symbol_cutoff_dist & 0xff,
            (self.symbol_cutoff_dist >> 8) & 0xff,
            self.symbol_cutoff_dist >> 16,
            self.offset_dist_xor & 0xffff,
            self.offset_dist_xor >> 16,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AnsHistogramIr {
    pub buckets: Vec<AnsBucketIr>,
    pub log_bucket_size: u32,
    pub single_symbol: Option<u32>,
}

impl AnsHistogramIr {
    fn parse(reader: &mut impl BitInput, log_alphabet_size: u32) -> Result<Self> {
        #[derive(Clone, Copy)]
        struct WorkingBucket {
            distribution: u16,
            alias_symbol: u16,
            alias_offset: u16,
            alias_cutoff: u16,
        }

        if !(5..=8).contains(&log_alphabet_size) {
            return invalid_entropy("ANS log alphabet size is outside 5 through 8");
        }
        let table_size = 1usize << log_alphabet_size;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1u16 << log_bucket_size;
        let alphabet_size;
        let mut distribution = vec![0u16; table_size];
        if reader.read_bits(1)? != 0 {
            if reader.read_bits(1)? != 0 {
                let first = usize::from(read_ans_u8(reader)?);
                let second = usize::from(read_ans_u8(reader)?);
                if first == second {
                    return invalid_entropy("binary ANS histogram repeats a symbol");
                }
                alphabet_size = first.max(second) + 1;
                if alphabet_size > table_size {
                    return invalid_entropy("binary ANS symbol exceeds its alphabet");
                }
                let probability = u16::try_from(read_bits_u32(reader, 12)?)
                    .map_err(|_| invalid_entropy_error("ANS probability exceeds u16"))?;
                distribution[first] = probability;
                distribution[second] = (1 << 12) - probability;
            } else {
                let symbol = usize::from(read_ans_u8(reader)?);
                alphabet_size = symbol + 1;
                if alphabet_size > table_size {
                    return invalid_entropy("unary ANS symbol exceeds its alphabet");
                }
                distribution[symbol] = 1 << 12;
            }
        } else if reader.read_bits(1)? != 0 {
            alphabet_size = usize::from(read_ans_u8(reader)?) + 1;
            if alphabet_size > table_size {
                return invalid_entropy("flat ANS alphabet exceeds its table");
            }
            let base = (1usize << 12) / alphabet_size;
            let leftover = (1usize << 12) % alphabet_size;
            distribution[..leftover].fill(u16::try_from(base + 1).unwrap());
            distribution[leftover..alphabet_size].fill(u16::try_from(base).unwrap());
        } else {
            let mut length = 0u32;
            while length < 3 && reader.read_bits(1)? != 0 {
                length += 1;
            }
            let shift = i16::try_from(read_bits_u32(reader, length)? + (1 << length) - 1)
                .map_err(|_| invalid_entropy_error("ANS shift exceeds i16"))?;
            if shift > 13 {
                return invalid_entropy("ANS distribution shift exceeds 13");
            }
            alphabet_size = usize::from(read_ans_u8(reader)?) + 3;
            if alphabet_size > table_size {
                return invalid_entropy("compressed ANS alphabet exceeds its table");
            }
            let mut repeat_ranges = Vec::new();
            let mut omitted = None;
            let mut index = 0usize;
            while index < alphabet_size {
                distribution[index] = read_ans_prefix(reader)?;
                if distribution[index] == 13 {
                    let repeat = usize::from(read_ans_u8(reader)?) + 4;
                    let end = index
                        .checked_add(repeat)
                        .ok_or_else(|| invalid_entropy_error("ANS repeat range overflow"))?;
                    if end > alphabet_size {
                        return invalid_entropy("ANS repeat range exceeds its alphabet");
                    }
                    repeat_ranges.push(index..end);
                    index = end;
                    continue;
                }
                match omitted {
                    Some((log, _)) if distribution[index] <= log => {}
                    _ => omitted = Some((distribution[index], index)),
                }
                index += 1;
            }
            let Some((_, omitted_position)) = omitted else {
                return invalid_entropy("ANS histogram does not contain an omitted symbol");
            };
            if distribution.get(omitted_position + 1) == Some(&13) {
                return invalid_entropy("ANS omitted symbol precedes a repeat code");
            }

            let mut repeat_index = 0usize;
            let mut accumulator = 0u16;
            let mut previous = 0u16;
            for (index, code) in distribution.iter_mut().enumerate() {
                if repeat_index < repeat_ranges.len() && repeat_ranges[repeat_index].start <= index
                {
                    if repeat_ranges[repeat_index].end == index {
                        repeat_index += 1;
                    } else {
                        *code = previous;
                        accumulator = accumulator.checked_add(*code).ok_or_else(|| {
                            invalid_entropy_error("ANS probability accumulator overflow")
                        })?;
                        if accumulator > 1 << 12 {
                            return invalid_entropy("ANS probabilities exceed 4096");
                        }
                        continue;
                    }
                }
                if *code == 0 {
                    previous = 0;
                    continue;
                }
                if index == omitted_position {
                    previous = 0;
                    continue;
                }
                if *code > 1 {
                    let zeros = i16::try_from(*code - 1).unwrap();
                    let bit_count = (shift - ((12 - zeros) >> 1)).clamp(0, zeros);
                    let extra = read_bits_u32(reader, u32::try_from(bit_count).unwrap())?;
                    *code = (1u16 << zeros)
                        + (u16::try_from(extra)
                            .map_err(|_| invalid_entropy_error("ANS probability exceeds u16"))?
                            << (zeros - bit_count));
                }
                previous = *code;
                accumulator = accumulator
                    .checked_add(*code)
                    .ok_or_else(|| invalid_entropy_error("ANS probability accumulator overflow"))?;
                if accumulator > 1 << 12 {
                    return invalid_entropy("ANS probabilities exceed 4096");
                }
            }
            distribution[omitted_position] = (1 << 12) - accumulator;
        }

        if let Some(symbol) = distribution.iter().position(|&value| value == 1 << 12) {
            let buckets = distribution
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    AnsBucketIr::new(
                        u8::try_from(symbol).unwrap(),
                        0,
                        value,
                        bucket_size * u16::try_from(index).unwrap(),
                        value ^ (1 << 12),
                    )
                })
                .collect();
            return Ok(Self {
                buckets,
                log_bucket_size,
                single_symbol: Some(u32::try_from(symbol).unwrap()),
            });
        }

        let mut buckets: Vec<_> = distribution
            .into_iter()
            .enumerate()
            .map(|(index, value)| WorkingBucket {
                distribution: value,
                alias_symbol: u16::try_from(index).unwrap(),
                alias_offset: 0,
                alias_cutoff: value,
            })
            .collect();
        let mut underfull = Vec::new();
        let mut overfull = Vec::new();
        for (index, bucket) in buckets.iter().enumerate() {
            match bucket.alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(index),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(index),
            }
        }
        while let (Some(over), Some(under)) = (overfull.pop(), underfull.pop()) {
            let amount = bucket_size - buckets[under].alias_cutoff;
            buckets[over].alias_cutoff -= amount;
            buckets[under].alias_symbol = u16::try_from(over).unwrap();
            buckets[under].alias_offset = buckets[over].alias_cutoff;
            match buckets[over].alias_cutoff.cmp(&bucket_size) {
                std::cmp::Ordering::Less => underfull.push(over),
                std::cmp::Ordering::Equal => {}
                std::cmp::Ordering::Greater => overfull.push(over),
            }
        }
        let finalized = buckets
            .iter()
            .enumerate()
            .map(|(index, bucket)| {
                if bucket.alias_cutoff == bucket_size {
                    AnsBucketIr::new(u8::try_from(index).unwrap(), 0, bucket.distribution, 0, 0)
                } else {
                    let alias = usize::from(bucket.alias_symbol);
                    AnsBucketIr::new(
                        u8::try_from(alias).unwrap(),
                        u8::try_from(bucket.alias_cutoff).unwrap(),
                        bucket.distribution,
                        bucket.alias_offset - bucket.alias_cutoff,
                        bucket.distribution ^ buckets[alias].distribution,
                    )
                }
            })
            .collect();
        Ok(Self {
            buckets: finalized,
            log_bucket_size,
            single_symbol: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EntropyCoderIr {
    Prefix(Vec<PrefixHistogramIr>),
    Ans {
        log_alphabet_size: u32,
        histograms: Vec<AnsHistogramIr>,
    },
}

impl EntropyCoderIr {
    pub(crate) fn cluster_count(&self) -> usize {
        match self {
            Self::Prefix(histograms) => histograms.len(),
            Self::Ans { histograms, .. } => histograms.len(),
        }
    }

    fn single_symbol(&self, cluster: usize) -> Option<u32> {
        match self {
            Self::Prefix(histograms) => histograms.get(cluster)?.single_symbol(),
            Self::Ans { histograms, .. } => histograms.get(cluster)?.single_symbol,
        }
    }

    fn supported_tokens(&self, cluster: usize) -> Result<Vec<u32>> {
        let tokens = match self {
            Self::Prefix(histograms) => {
                let histogram = histograms
                    .get(cluster)
                    .ok_or_else(|| invalid_entropy_error("prefix cluster is missing"))?;
                if let Some(symbol) = histogram.single_symbol {
                    vec![symbol]
                } else {
                    histogram
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, entry)| entry.bit_len != 0)
                        .map(|(symbol, _)| {
                            u32::try_from(symbol)
                                .map_err(|_| invalid_entropy_error("prefix symbol exceeds u32"))
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?
                }
            }
            Self::Ans { histograms, .. } => {
                let histogram = histograms
                    .get(cluster)
                    .ok_or_else(|| invalid_entropy_error("ANS cluster is missing"))?;
                if let Some(symbol) = histogram.single_symbol {
                    vec![symbol]
                } else {
                    let bucket_size = 1u32 << histogram.log_bucket_size;
                    let mut present = vec![false; histogram.buckets.len()];
                    for (index, bucket) in histogram.buckets.iter().copied().enumerate() {
                        let (alias_symbol, cutoff, _, _, _) = bucket.fields();
                        if cutoff > bucket_size {
                            return invalid_entropy("ANS alias cutoff exceeds its bucket");
                        }
                        if cutoff != 0 {
                            present[index] = true;
                        }
                        if cutoff < bucket_size {
                            let alias = usize::try_from(alias_symbol).map_err(|_| {
                                invalid_entropy_error("ANS alias exceeds host address space")
                            })?;
                            let slot = present.get_mut(alias).ok_or_else(|| {
                                invalid_entropy_error("ANS alias exceeds its alphabet")
                            })?;
                            *slot = true;
                        }
                    }
                    present
                        .into_iter()
                        .enumerate()
                        .filter(|(_, present)| *present)
                        .map(|(symbol, _)| {
                            u32::try_from(symbol)
                                .map_err(|_| invalid_entropy_error("ANS symbol exceeds u32"))
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()?
                }
            }
        };
        if tokens.is_empty() {
            return invalid_entropy("entropy histogram has no reachable symbol");
        }
        Ok(tokens)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Lz77Ir {
    pub min_symbol: u32,
    pub min_length: u32,
    pub length_config: HybridIntegerConfigIr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EntropyDecoderIr {
    pub lz77: Option<Lz77Ir>,
    pub context_to_cluster: Vec<u8>,
    pub configs: Vec<HybridIntegerConfigIr>,
    pub coder: EntropyCoderIr,
}

impl EntropyDecoderIr {
    pub(crate) fn parse(
        reader: &mut impl BitInput,
        context_count: usize,
        limits: MaTreeLimits,
    ) -> Result<Self> {
        Self::parse_inner(reader, context_count, limits, false)
    }

    fn parse_assume_no_lz77(
        reader: &mut impl BitInput,
        context_count: usize,
        limits: MaTreeLimits,
    ) -> Result<Self> {
        Self::parse_inner(reader, context_count, limits, true)
    }

    fn parse_inner(
        reader: &mut impl BitInput,
        context_count: usize,
        limits: MaTreeLimits,
        forbid_lz77: bool,
    ) -> Result<Self> {
        if context_count == 0 || context_count > limits.context_limit {
            return Err(ModularTreeError::LimitExceeded {
                resource: "entropy context",
                limit: limits.context_limit,
            }
            .into());
        }
        let lz77_enabled = reader.read_bits(1)? != 0;
        if forbid_lz77 && lz77_enabled {
            return invalid_entropy("LZ77 is forbidden for this metadata context map");
        }
        let lz77 = lz77_enabled
            .then(|| -> Result<_> {
                let min_symbol = match read_bits_u32(reader, 2)? {
                    0 => 224,
                    1 => 512,
                    2 => 4096,
                    3 => 8 + read_bits_u32(reader, 15)?,
                    _ => unreachable!(),
                };
                let min_length = match read_bits_u32(reader, 2)? {
                    0 => 3,
                    1 => 4,
                    2 => 5 + read_bits_u32(reader, 2)?,
                    3 => 9 + read_bits_u32(reader, 8)?,
                    _ => unreachable!(),
                };
                Ok(Lz77Ir {
                    min_symbol,
                    min_length,
                    length_config: HybridIntegerConfigIr::parse(reader, 8)?,
                })
            })
            .transpose()?;
        let effective_contexts = context_count
            .checked_add(usize::from(lz77.is_some()))
            .ok_or_else(|| invalid_entropy_error("entropy context count overflow"))?;
        let context_to_cluster = read_clusters(reader, effective_contexts, limits)?;
        let cluster_count = context_to_cluster
            .iter()
            .copied()
            .max()
            .map_or(0usize, |value| usize::from(value) + 1);
        if cluster_count == 0 || cluster_count > limits.cluster_limit {
            return Err(ModularTreeError::LimitExceeded {
                resource: "entropy cluster",
                limit: limits.cluster_limit,
            }
            .into());
        }
        let use_prefix = reader.read_bits(1)? != 0;
        let log_alphabet_size = if use_prefix {
            15
        } else {
            read_bits_u32(reader, 2)? + 5
        };
        let configs = (0..cluster_count)
            .map(|_| HybridIntegerConfigIr::parse(reader, log_alphabet_size))
            .collect::<Result<Vec<_>>>()?;
        let coder = if use_prefix {
            let counts = (0..cluster_count)
                .map(|_| -> Result<_> {
                    let count = if reader.read_bits(1)? != 0 {
                        let exponent = read_bits_u32(reader, 4)?;
                        let extra =
                            usize::try_from(read_bits_u32(reader, exponent)?).map_err(|_| {
                                invalid_entropy_error("prefix alphabet count exceeds host space")
                            })?;
                        1usize
                            .checked_add(1usize << exponent)
                            .and_then(|base| base.checked_add(extra))
                            .ok_or_else(|| {
                                invalid_entropy_error("prefix alphabet count overflow")
                            })?
                    } else {
                        1
                    };
                    if count > MAX_PREFIX_ALPHABET_SIZE {
                        return invalid_entropy("prefix alphabet exceeds 32768 symbols");
                    }
                    Ok(count)
                })
                .collect::<Result<Vec<_>>>()?;
            EntropyCoderIr::Prefix(
                counts
                    .into_iter()
                    .map(|count| PrefixHistogramIr::parse(reader, count))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            EntropyCoderIr::Ans {
                log_alphabet_size,
                histograms: (0..cluster_count)
                    .map(|_| AnsHistogramIr::parse(reader, log_alphabet_size))
                    .collect::<Result<Vec<_>>>()?,
            }
        };
        Ok(Self {
            lz77,
            context_to_cluster,
            configs,
            coder,
        })
    }

    pub(crate) fn cluster_for_context(&self, context: usize) -> Result<u8> {
        self.context_to_cluster
            .get(context)
            .copied()
            .ok_or_else(|| invalid_entropy_error("entropy context exceeds its cluster map").into())
    }

    fn single_value(&self, context: usize) -> Option<u32> {
        if self.lz77.is_some() {
            return None;
        }
        let cluster = usize::from(*self.context_to_cluster.get(context)?);
        let token = self.coder.single_symbol(cluster)?;
        let config = *self.configs.get(cluster)?;
        (token < config.split()).then_some(token)
    }

    /// Power-of-two LZ history required by a bounded logical stream.
    ///
    /// This examines only the already-parsed distance histogram and hybrid-integer descriptor.
    /// It deliberately does not consume entropy symbols. The returned ring is a conservative
    /// upper bound for every reachable back-reference after the JPEG XL special-distance mapping.
    pub(crate) fn lz77_window_words(
        &self,
        distance_multiplier: u32,
        decoded_symbol_limit: u32,
    ) -> Result<u32> {
        if self.lz77.is_none() || decoded_symbol_limit == 0 {
            return Ok(0);
        }
        let distance_cluster = usize::from(
            *self
                .context_to_cluster
                .last()
                .ok_or_else(|| invalid_entropy_error("LZ77 distance context is missing"))?,
        );
        let config = *self
            .configs
            .get(distance_cluster)
            .ok_or_else(|| invalid_entropy_error("LZ77 distance config is missing"))?;
        let mut maximum_distance = 0u32;
        for token in self.coder.supported_tokens(distance_cluster)? {
            let (minimum, maximum) = config.value_bounds(token);
            if distance_multiplier == 0 {
                maximum_distance =
                    maximum_distance.max(resolve_lz77_distance(maximum, 0, decoded_symbol_limit));
                continue;
            }
            let special_end = maximum.min(119);
            if minimum <= special_end {
                for value in minimum..=special_end {
                    maximum_distance = maximum_distance.max(resolve_lz77_distance(
                        value,
                        distance_multiplier,
                        decoded_symbol_limit,
                    ));
                }
            }
            if maximum >= 120 {
                maximum_distance = maximum_distance.max(resolve_lz77_distance(
                    maximum,
                    distance_multiplier,
                    decoded_symbol_limit,
                ));
            }
        }
        maximum_distance
            .checked_next_power_of_two()
            .ok_or_else(|| invalid_entropy_error("LZ77 history ring size overflow").into())
    }

    /// Packs a descriptor-only GPU ABI for non-Modular consumers such as VarDCT metadata.
    ///
    /// The common 24-word header reports zero MA nodes and a zero maximum depth. Config and
    /// entropy tables use the exact same offsets and representation as [`MaConfigIr`].
    pub(crate) fn pack_gpu_metadata(&self) -> Result<PackedModularMetadata> {
        MaConfigIr {
            nodes: Vec::new(),
            max_depth: 0,
            entropy: self.clone(),
        }
        .pack_gpu_metadata()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MaTreeNodeIr {
    Decision {
        property: u32,
        threshold: i32,
        left: u32,
        right: u32,
    },
    Leaf {
        cluster: u8,
        predictor: u8,
        offset: i32,
        multiplier: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MaConfigIr {
    pub nodes: Vec<MaTreeNodeIr>,
    pub max_depth: usize,
    pub entropy: EntropyDecoderIr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WpHeaderIr {
    pub p1: u32,
    pub p2: u32,
    pub p3a: u32,
    pub p3b: u32,
    pub p3c: u32,
    pub p3d: u32,
    pub p3e: u32,
    pub w0: u32,
    pub w1: u32,
    pub w2: u32,
    pub w3: u32,
}

impl Default for WpHeaderIr {
    fn default() -> Self {
        Self {
            p1: 16,
            p2: 10,
            p3a: 7,
            p3b: 7,
            p3c: 7,
            p3d: 0,
            p3e: 0,
            w0: 13,
            w1: 12,
            w2: 12,
            w3: 12,
        }
    }
}

impl WpHeaderIr {
    pub(crate) fn parse(reader: &mut impl BitInput) -> Result<Self> {
        if reader.read_bits(1)? != 0 {
            return Ok(Self::default());
        }
        Ok(Self {
            p1: read_bits_u32(reader, 5)?,
            p2: read_bits_u32(reader, 5)?,
            p3a: read_bits_u32(reader, 5)?,
            p3b: read_bits_u32(reader, 5)?,
            p3c: read_bits_u32(reader, 5)?,
            p3d: read_bits_u32(reader, 5)?,
            p3e: read_bits_u32(reader, 5)?,
            w0: read_bits_u32(reader, 4)?,
            w1: read_bits_u32(reader, 4)?,
            w2: read_bits_u32(reader, 4)?,
            w3: read_bits_u32(reader, 4)?,
        })
    }
}

/// Parses one standard MA configuration while leaving its image entropy unread.
pub(crate) fn parse_ma_config(
    reader: &mut impl BitInput,
    limits: MaTreeLimits,
) -> Result<MaConfigIr> {
    let tree_entropy = EntropyDecoderIr::parse(reader, 6, limits)?;
    if tree_entropy.single_value(1).is_some_and(|value| value != 0) {
        return invalid_tree("MA tree decision distribution is infinite");
    }
    let mut metadata = MetadataEntropyCursor::new(&tree_entropy, limits.metadata_symbol_limit);
    metadata.begin(reader)?;
    let mut folding = Vec::new();
    let mut contexts = 0usize;
    let mut nodes_left = 1usize;
    while nodes_left != 0 {
        if folding.len() >= limits.node_limit {
            return Err(ModularTreeError::LimitExceeded {
                resource: "MA tree node",
                limit: limits.node_limit,
            }
            .into());
        }
        nodes_left -= 1;
        let property = metadata.read_varint(reader, 1, 0)?;
        if let Some(property) = property.checked_sub(1) {
            let property = validate_tree_property(property)?;
            let threshold = unpack_signed(metadata.read_varint(reader, 0, 0)?);
            folding.push(FoldingNode::Decision {
                property,
                threshold,
            });
            nodes_left = nodes_left
                .checked_add(2)
                .ok_or_else(|| invalid_tree_error("MA tree node count overflow"))?;
        } else {
            let predictor = metadata.read_varint(reader, 2, 0)?;
            if predictor > 13 {
                return Err(ModularTreeError::InvalidPredictor { predictor }.into());
            }
            let offset = unpack_signed(metadata.read_varint(reader, 3, 0)?);
            let multiplier_log = metadata.read_varint(reader, 4, 0)?;
            if multiplier_log > 30 {
                return invalid_tree("MA tree multiplier log exceeds 30");
            }
            let multiplier_bits = metadata.read_varint(reader, 5, 0)?;
            let maximum_bits = (1u32 << (31 - multiplier_log)) - 2;
            if multiplier_bits > maximum_bits {
                return invalid_tree("MA tree multiplier bits overflow i32");
            }
            let multiplier = (multiplier_bits + 1) << multiplier_log;
            folding.push(FoldingNode::Leaf {
                context: contexts,
                predictor: u8::try_from(predictor).unwrap(),
                offset,
                multiplier,
            });
            contexts = contexts
                .checked_add(1)
                .ok_or_else(|| invalid_tree_error("MA tree leaf count overflow"))?;
            if contexts > limits.context_limit {
                return Err(ModularTreeError::LimitExceeded {
                    resource: "MA tree leaf context",
                    limit: limits.context_limit,
                }
                .into());
            }
        }
    }
    metadata.finalize()?;
    let entropy = EntropyDecoderIr::parse(reader, contexts, limits)?;
    let (nodes, max_depth) = lower_tree(&folding, &entropy, limits.depth_limit)?;
    Ok(MaConfigIr {
        nodes,
        max_depth,
        entropy,
    })
}

fn validate_tree_property(property: u32) -> Result<u32> {
    if property > 255 {
        return Err(ModularTreeError::InvalidProperty { property }.into());
    }
    Ok(property)
}

impl MaConfigIr {
    pub(crate) fn maximum_tree_property(&self) -> Option<u32> {
        self.nodes
            .iter()
            .filter_map(|node| match node {
                MaTreeNodeIr::Decision { property, .. } => Some(*property),
                MaTreeNodeIr::Leaf { .. } => None,
            })
            .max()
    }

    pub(crate) fn needs_self_correcting(&self) -> bool {
        self.nodes.iter().any(|node| match node {
            MaTreeNodeIr::Decision { property, .. } => *property == 15,
            MaTreeNodeIr::Leaf { predictor, .. } => *predictor == 6,
        })
    }

    pub(crate) fn pack_gpu_metadata(&self) -> Result<PackedModularMetadata> {
        const HEADER_NODE_COUNT: usize = 0;
        const HEADER_MAX_DEPTH: usize = 1;
        const HEADER_CODER: usize = 2;
        const HEADER_CLUSTER_COUNT: usize = 3;
        const HEADER_CONFIG_OFFSET: usize = 4;
        const HEADER_TREE_OFFSET: usize = 5;
        const HEADER_TABLE_OFFSET: usize = 6;
        const HEADER_TABLE_STRIDE: usize = 7;
        const HEADER_ANS_LOG_BUCKET: usize = 8;
        const HEADER_LZ_ENABLED: usize = 9;
        const HEADER_LZ_MIN_SYMBOL: usize = 10;
        const HEADER_LZ_MIN_LENGTH: usize = 11;
        const HEADER_LZ_LENGTH_SPLIT: usize = 12;
        const HEADER_LZ_LENGTH_MSB: usize = 13;
        const HEADER_LZ_LENGTH_LSB: usize = 14;
        const HEADER_DISTANCE_CLUSTER: usize = 15;

        let cluster_count = self.entropy.coder.cluster_count();
        if cluster_count != self.entropy.configs.len() {
            return invalid_entropy("entropy configs and histograms have different lengths");
        }
        let config_offset = GPU_METADATA_HEADER_WORDS;
        let tree_offset = config_offset
            .checked_add(
                cluster_count
                    .checked_mul(GPU_CONFIG_WORDS)
                    .ok_or_else(|| invalid_entropy_error("GPU config table size overflow"))?,
            )
            .ok_or_else(|| invalid_entropy_error("GPU config table offset overflow"))?;
        let table_offset = tree_offset
            .checked_add(
                self.nodes
                    .len()
                    .checked_mul(GPU_TREE_NODE_WORDS)
                    .ok_or_else(|| invalid_tree_error("GPU MA tree size overflow"))?,
            )
            .ok_or_else(|| invalid_tree_error("GPU MA tree offset overflow"))?;
        let (coder_kind, table_stride, ans_log_bucket) = match &self.entropy.coder {
            EntropyCoderIr::Prefix(_) => (0u32, 1usize << MAX_PREFIX_BITS, 0u32),
            EntropyCoderIr::Ans {
                log_alphabet_size,
                histograms,
            } => {
                let table_size = 1usize << log_alphabet_size;
                let stride = table_size
                    .checked_mul(2)
                    .ok_or_else(|| invalid_entropy_error("GPU ANS table stride overflow"))?;
                let log_bucket = histograms
                    .first()
                    .ok_or_else(|| invalid_entropy_error("ANS histogram set is empty"))?
                    .log_bucket_size;
                if histograms.iter().any(|histogram| {
                    histogram.buckets.len() != table_size || histogram.log_bucket_size != log_bucket
                }) {
                    return invalid_entropy("ANS histograms have inconsistent table shapes");
                }
                (1, stride, log_bucket)
            }
        };
        let total_words = table_offset
            .checked_add(
                cluster_count
                    .checked_mul(table_stride)
                    .ok_or_else(|| invalid_entropy_error("GPU entropy table size overflow"))?,
            )
            .ok_or_else(|| invalid_entropy_error("GPU entropy metadata size overflow"))?;
        let mut words = vec![0u32; total_words];
        words[HEADER_NODE_COUNT] = u32::try_from(self.nodes.len())
            .map_err(|_| invalid_tree_error("GPU MA node count exceeds u32"))?;
        words[HEADER_MAX_DEPTH] = u32::try_from(self.max_depth)
            .map_err(|_| invalid_tree_error("GPU MA depth exceeds u32"))?;
        words[HEADER_CODER] = coder_kind;
        words[HEADER_CLUSTER_COUNT] = u32::try_from(cluster_count)
            .map_err(|_| invalid_entropy_error("GPU cluster count exceeds u32"))?;
        words[HEADER_CONFIG_OFFSET] = u32::try_from(config_offset)
            .map_err(|_| invalid_entropy_error("GPU config offset exceeds u32"))?;
        words[HEADER_TREE_OFFSET] = u32::try_from(tree_offset)
            .map_err(|_| invalid_tree_error("GPU tree offset exceeds u32"))?;
        words[HEADER_TABLE_OFFSET] = u32::try_from(table_offset)
            .map_err(|_| invalid_entropy_error("GPU table offset exceeds u32"))?;
        words[HEADER_TABLE_STRIDE] = u32::try_from(table_stride)
            .map_err(|_| invalid_entropy_error("GPU table stride exceeds u32"))?;
        words[HEADER_ANS_LOG_BUCKET] = ans_log_bucket;
        if let Some(lz77) = self.entropy.lz77 {
            words[HEADER_LZ_ENABLED] = 1;
            words[HEADER_LZ_MIN_SYMBOL] = lz77.min_symbol;
            words[HEADER_LZ_MIN_LENGTH] = lz77.min_length;
            words[HEADER_LZ_LENGTH_SPLIT] = lz77.length_config.split_exponent;
            words[HEADER_LZ_LENGTH_MSB] = lz77.length_config.msb_in_token;
            words[HEADER_LZ_LENGTH_LSB] = lz77.length_config.lsb_in_token;
            words[HEADER_DISTANCE_CLUSTER] = u32::from(
                *self
                    .entropy
                    .context_to_cluster
                    .last()
                    .ok_or_else(|| invalid_entropy_error("LZ77 distance context is missing"))?,
            );
        }

        for (cluster, config) in self.entropy.configs.iter().copied().enumerate() {
            let offset = config_offset + cluster * GPU_CONFIG_WORDS;
            words[offset] = config.split_exponent;
            words[offset + 1] = config.msb_in_token;
            words[offset + 2] = config.lsb_in_token;
            words[offset + 3] = match &self.entropy.coder {
                EntropyCoderIr::Prefix(histograms) => histograms[cluster]
                    .single_symbol
                    .map_or(0, |symbol| symbol.saturating_add(1)),
                EntropyCoderIr::Ans { .. } => 0,
            };
        }
        for (index, node) in self.nodes.iter().copied().enumerate() {
            let offset = tree_offset + index * GPU_TREE_NODE_WORDS;
            match node {
                MaTreeNodeIr::Decision {
                    property,
                    threshold,
                    left,
                    right,
                } => {
                    words[offset] = 0;
                    words[offset + 1] = property;
                    words[offset + 2] = threshold as u32;
                    words[offset + 3] = left;
                    words[offset + 4] = right;
                }
                MaTreeNodeIr::Leaf {
                    cluster,
                    predictor,
                    offset: leaf_offset,
                    multiplier,
                } => {
                    words[offset] = 1;
                    words[offset + 1] = u32::from(predictor);
                    words[offset + 2] = leaf_offset as u32;
                    words[offset + 3] = u32::from(cluster);
                    words[offset + 4] = multiplier;
                }
            }
        }
        match &self.entropy.coder {
            EntropyCoderIr::Prefix(histograms) => {
                for (cluster, histogram) in histograms.iter().enumerate() {
                    let start = table_offset + cluster * table_stride;
                    let table = &mut words[start..start + table_stride];
                    for (symbol, entry) in histogram.entries.iter().copied().enumerate() {
                        insert_gpu_prefix(
                            table,
                            u32::try_from(symbol)
                                .map_err(|_| invalid_entropy_error("prefix symbol exceeds u32"))?,
                            entry,
                        )?;
                    }
                }
            }
            EntropyCoderIr::Ans { histograms, .. } => {
                for (cluster, histogram) in histograms.iter().enumerate() {
                    let start = table_offset + cluster * table_stride;
                    for (index, bucket) in histogram.buckets.iter().copied().enumerate() {
                        words[start + index * 2] = bucket.symbol_cutoff_dist;
                        words[start + index * 2 + 1] = bucket.offset_dist_xor;
                    }
                }
            }
        }
        Ok(PackedModularMetadata { words })
    }
}

fn insert_gpu_prefix(table: &mut [u32], symbol: u32, entry: PrefixCodeEntry) -> Result<()> {
    if entry.bit_len == 0 {
        return Ok(());
    }
    if entry.bit_len > MAX_PREFIX_BITS {
        return invalid_entropy("prefix code exceeds the GPU lookup width");
    }
    let suffix_bits = MAX_PREFIX_BITS - entry.bit_len;
    for suffix in 0..1usize << suffix_bits {
        let index = usize::from(entry.bits) | (suffix << entry.bit_len);
        if table[index] != 0 {
            return invalid_entropy("prefix codes collide in the GPU lookup table");
        }
        table[index] = (symbol << 8) | u32::from(entry.bit_len);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum FoldingNode {
    Decision {
        property: u32,
        threshold: i32,
    },
    Leaf {
        context: usize,
        predictor: u8,
        offset: i32,
        multiplier: u32,
    },
}

fn lower_tree(
    folding: &[FoldingNode],
    entropy: &EntropyDecoderIr,
    depth_limit: usize,
) -> Result<(Vec<MaTreeNodeIr>, usize)> {
    if folding.is_empty() {
        return invalid_tree("MA tree has no root node");
    }

    #[derive(Clone, Copy)]
    enum ArenaNode {
        Decision {
            property: u32,
            threshold: i32,
            left: usize,
            right: usize,
        },
        Leaf {
            context: usize,
            predictor: u8,
            offset: i32,
            multiplier: u32,
        },
    }

    // MA metadata is a folding sequence, not a preorder serialization. Reversing the sequence
    // and folding the two oldest completed subtrees exactly mirrors the normative decoder.
    let mut arena = Vec::with_capacity(folding.len());
    let mut completed: VecDeque<(usize, usize)> = VecDeque::new();
    let mut max_depth = 0usize;
    for node in folding.iter().copied().rev() {
        let (arena_node, depth) = match node {
            FoldingNode::Leaf {
                context,
                predictor,
                offset,
                multiplier,
            } => (
                ArenaNode::Leaf {
                    context,
                    predictor,
                    offset,
                    multiplier,
                },
                0,
            ),
            FoldingNode::Decision {
                property,
                threshold,
            } => {
                let (right, right_depth) = completed
                    .pop_front()
                    .ok_or_else(|| invalid_tree_error("MA tree right child is truncated"))?;
                let (left, left_depth) = completed
                    .pop_front()
                    .ok_or_else(|| invalid_tree_error("MA tree left child is truncated"))?;
                let depth = left_depth
                    .max(right_depth)
                    .checked_add(1)
                    .ok_or_else(|| invalid_tree_error("MA tree depth overflow"))?;
                if depth > depth_limit {
                    return Err(ModularTreeError::TreeDepthExceeded {
                        depth,
                        limit: depth_limit,
                    }
                    .into());
                }
                (
                    ArenaNode::Decision {
                        property,
                        threshold,
                        left,
                        right,
                    },
                    depth,
                )
            }
        };
        max_depth = max_depth.max(depth);
        let index = arena.len();
        arena.push(arena_node);
        completed.push_back((index, depth));
    }
    if completed.len() != 1 {
        return invalid_tree("MA tree contains unreachable folding subtrees");
    }
    let (root, _) = completed.pop_front().unwrap();

    fn emit(
        arena: &[ArenaNode],
        arena_index: usize,
        entropy: &EntropyDecoderIr,
        lowered: &mut Vec<MaTreeNodeIr>,
    ) -> Result<u32> {
        let output_index = u32::try_from(lowered.len())
            .map_err(|_| invalid_tree_error("MA tree index exceeds u32"))?;
        lowered.push(MaTreeNodeIr::Leaf {
            cluster: 0,
            predictor: 0,
            offset: 0,
            multiplier: 1,
        });
        lowered[usize::try_from(output_index).unwrap()] = match *arena
            .get(arena_index)
            .ok_or_else(|| invalid_tree_error("MA tree arena index is truncated"))?
        {
            ArenaNode::Decision {
                property,
                threshold,
                left,
                right,
            } => {
                let left = emit(arena, left, entropy, lowered)?;
                let right = emit(arena, right, entropy, lowered)?;
                MaTreeNodeIr::Decision {
                    property,
                    threshold,
                    left,
                    right,
                }
            }
            ArenaNode::Leaf {
                context,
                predictor,
                offset,
                multiplier,
            } => MaTreeNodeIr::Leaf {
                cluster: entropy.cluster_for_context(context)?,
                predictor,
                offset,
                multiplier,
            },
        };
        Ok(output_index)
    }

    let mut lowered = Vec::with_capacity(folding.len());
    let output_root = emit(&arena, root, entropy, &mut lowered)?;
    debug_assert_eq!(output_root, 0);
    Ok((lowered, max_depth))
}

pub(crate) struct MetadataEntropyCursor<'a> {
    descriptor: &'a EntropyDecoderIr,
    ans_state: Option<u32>,
    lz77: MetadataLz77State,
    decoded_symbols: usize,
    symbol_limit: usize,
}

#[derive(Default)]
struct MetadataLz77State {
    window: Vec<u32>,
    num_to_copy: u32,
    copy_position: u32,
    num_decoded: u32,
}

impl<'a> MetadataEntropyCursor<'a> {
    pub(crate) fn new(descriptor: &'a EntropyDecoderIr, symbol_limit: usize) -> Self {
        Self {
            descriptor,
            ans_state: None,
            lz77: MetadataLz77State::default(),
            decoded_symbols: 0,
            symbol_limit,
        }
    }

    pub(crate) fn begin(&mut self, reader: &mut impl BitInput) -> Result<()> {
        if matches!(self.descriptor.coder, EntropyCoderIr::Ans { .. }) {
            self.ans_state = Some(read_bits_u32(reader, 32)?);
        }
        Ok(())
    }

    pub(crate) fn finalize(&self) -> Result<()> {
        if self.ans_state.is_some_and(|state| state != ANS_SIGNATURE) {
            return invalid_entropy("MA tree ANS stream has an invalid final state");
        }
        Ok(())
    }

    pub(crate) fn read_varint(
        &mut self,
        reader: &mut impl BitInput,
        context: usize,
        distance_multiplier: u32,
    ) -> Result<u32> {
        self.decoded_symbols = self
            .decoded_symbols
            .checked_add(1)
            .ok_or_else(|| invalid_entropy_error("metadata symbol count overflow"))?;
        if self.decoded_symbols > self.symbol_limit {
            return Err(ModularTreeError::LimitExceeded {
                resource: "metadata entropy symbol",
                limit: self.symbol_limit,
            }
            .into());
        }
        let cluster = self.descriptor.cluster_for_context(context)?;
        let Some(lz77) = self.descriptor.lz77 else {
            return self.read_clustered(reader, cluster);
        };
        if self.lz77.num_to_copy != 0 {
            let value = self.copy_lz77_value()?;
            self.record_lz77_value(value);
            return Ok(value);
        }
        let token = self.read_symbol(reader, cluster)?;
        let value = if token >= lz77.min_symbol {
            if self.lz77.num_decoded == 0 {
                return invalid_entropy("LZ77 repeat precedes metadata history");
            }
            let run = self
                .read_hybrid(reader, lz77.length_config, token - lz77.min_symbol)?
                .checked_add(lz77.min_length)
                .ok_or_else(|| invalid_entropy_error("LZ77 repeat length overflow"))?;
            self.lz77.num_to_copy = run;
            let distance_cluster = *self
                .descriptor
                .context_to_cluster
                .last()
                .ok_or_else(|| invalid_entropy_error("LZ77 distance cluster is missing"))?;
            let distance_token = self.read_symbol(reader, distance_cluster)?;
            let distance_config = *self
                .descriptor
                .configs
                .get(usize::from(distance_cluster))
                .ok_or_else(|| invalid_entropy_error("LZ77 distance config is missing"))?;
            let distance = self.read_hybrid(reader, distance_config, distance_token)?;
            let distance =
                resolve_lz77_distance(distance, distance_multiplier, self.lz77.num_decoded);
            self.lz77.copy_position = self.lz77.num_decoded - distance;
            self.copy_lz77_value()?
        } else {
            let config = *self
                .descriptor
                .configs
                .get(usize::from(cluster))
                .ok_or_else(|| invalid_entropy_error("entropy cluster config is missing"))?;
            self.read_hybrid(reader, config, token)?
        };
        self.record_lz77_value(value);
        Ok(value)
    }

    fn record_lz77_value(&mut self, value: u32) {
        let offset = (self.lz77.num_decoded & 0x000f_ffff) as usize;
        if self.lz77.window.len() <= offset {
            self.lz77.window.push(value);
        } else {
            self.lz77.window[offset] = value;
        }
        self.lz77.num_decoded = self.lz77.num_decoded.wrapping_add(1);
    }

    fn copy_lz77_value(&mut self) -> Result<u32> {
        let index = usize::try_from(self.lz77.copy_position & 0x000f_ffff)
            .map_err(|_| invalid_entropy_error("LZ77 copy position exceeds host space"))?;
        let value = *self
            .lz77
            .window
            .get(index)
            .ok_or_else(|| invalid_entropy_error("LZ77 copy position precedes its history"))?;
        self.lz77.copy_position = self.lz77.copy_position.wrapping_add(1);
        self.lz77.num_to_copy -= 1;
        Ok(value)
    }

    fn read_clustered(&mut self, reader: &mut impl BitInput, cluster: u8) -> Result<u32> {
        let token = self.read_symbol(reader, cluster)?;
        let config = *self
            .descriptor
            .configs
            .get(usize::from(cluster))
            .ok_or_else(|| invalid_entropy_error("entropy cluster config is missing"))?;
        self.read_hybrid(reader, config, token)
    }

    fn read_symbol(&mut self, reader: &mut impl BitInput, cluster: u8) -> Result<u32> {
        match &self.descriptor.coder {
            EntropyCoderIr::Prefix(histograms) => {
                let histogram = histograms
                    .get(usize::from(cluster))
                    .ok_or_else(|| invalid_entropy_error("prefix cluster is missing"))?;
                if let Some(symbol) = histogram.single_symbol {
                    return Ok(symbol);
                }
                read_prefix_symbol(reader, &histogram.entries)
            }
            EntropyCoderIr::Ans {
                histograms,
                log_alphabet_size,
            } => {
                let histogram = histograms
                    .get(usize::from(cluster))
                    .ok_or_else(|| invalid_entropy_error("ANS cluster is missing"))?;
                let state = self
                    .ans_state
                    .as_mut()
                    .ok_or_else(|| invalid_entropy_error("ANS stream was not initialized"))?;
                let index = *state & 0xfff;
                let bucket_index = usize::try_from(index >> histogram.log_bucket_size)
                    .map_err(|_| invalid_entropy_error("ANS bucket exceeds host space"))?;
                let position = index & ((1 << histogram.log_bucket_size) - 1);
                let (alias_symbol, cutoff, mut distribution, alias_offset, dist_xor) = histogram
                    .buckets
                    .get(bucket_index)
                    .copied()
                    .ok_or_else(|| invalid_entropy_error("ANS bucket is missing"))?
                    .fields();
                let map_alias = position >= cutoff;
                let symbol = if map_alias {
                    distribution ^= dist_xor;
                    alias_symbol
                } else {
                    u32::try_from(bucket_index)
                        .map_err(|_| invalid_entropy_error("ANS symbol exceeds u32"))?
                };
                let offset = if map_alias { alias_offset } else { 0 } + position;
                let next = (*state >> 12)
                    .checked_mul(distribution)
                    .and_then(|value| value.checked_add(offset))
                    .ok_or_else(|| invalid_entropy_error("ANS state overflow"))?;
                *state = if next < 1 << 16 {
                    (next << 16) | read_bits_u32(reader, 16)?
                } else {
                    next
                };
                let table_size = 1u32 << *log_alphabet_size;
                if symbol >= table_size {
                    return invalid_entropy("ANS symbol exceeds its alphabet");
                }
                Ok(symbol)
            }
        }
    }

    fn read_hybrid(
        &mut self,
        reader: &mut impl BitInput,
        config: HybridIntegerConfigIr,
        token: u32,
    ) -> Result<u32> {
        if token < config.split() {
            return Ok(token);
        }
        let embedded = config.msb_in_token + config.lsb_in_token;
        let bit_count = config
            .split_exponent
            .saturating_sub(embedded)
            .wrapping_add((token - config.split()) >> embedded)
            & 31;
        let extra = read_bits_u32(reader, bit_count)?;
        let low_mask = (1u32 << config.lsb_in_token).wrapping_sub(1);
        let low = token & low_mask;
        let shifted = token >> config.lsb_in_token;
        let high_mask = (1u32 << config.msb_in_token).wrapping_sub(1);
        let high = (shifted & high_mask) | (1 << config.msb_in_token);
        let value = (((u64::from(high) << bit_count) | u64::from(extra)) << config.lsb_in_token)
            | u64::from(low);
        Ok(value as u32)
    }
}

pub(crate) fn read_clusters(
    reader: &mut impl BitInput,
    context_count: usize,
    limits: MaTreeLimits,
) -> Result<Vec<u8>> {
    if context_count == 1 {
        return Ok(vec![0]);
    }
    let mut clusters = if reader.read_bits(1)? != 0 {
        let bits = read_bits_u32(reader, 2)?;
        (0..context_count)
            .map(|_| {
                u8::try_from(read_bits_u32(reader, bits)?)
                    .map_err(|_| invalid_entropy_error("cluster index exceeds u8").into())
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        let use_mtf = reader.read_bits(1)? != 0;
        let descriptor = if context_count <= 2 {
            EntropyDecoderIr::parse_assume_no_lz77(reader, 1, limits)?
        } else {
            EntropyDecoderIr::parse(reader, 1, limits)?
        };
        let mut cursor = MetadataEntropyCursor::new(&descriptor, limits.metadata_symbol_limit);
        cursor.begin(reader)?;
        let mut decoded = (0..context_count)
            .map(|_| {
                let value = cursor.read_varint(reader, 0, 0)?;
                u8::try_from(value)
                    .map_err(|_| invalid_entropy_error("cluster index exceeds u8").into())
            })
            .collect::<Result<Vec<_>>>()?;
        cursor.finalize()?;
        if use_mtf {
            let mut mtf = [0u8; 256];
            for (index, value) in mtf.iter_mut().enumerate() {
                *value = u8::try_from(index).unwrap();
            }
            for cluster in &mut decoded {
                let index = usize::from(*cluster);
                *cluster = mtf[index];
                mtf.copy_within(..index, 1);
                mtf[0] = *cluster;
            }
        }
        decoded
    };
    let maximum = clusters.iter().copied().max().unwrap_or(0);
    let cluster_count = usize::from(maximum) + 1;
    if cluster_count > limits.cluster_limit {
        return Err(ModularTreeError::LimitExceeded {
            resource: "entropy cluster",
            limit: limits.cluster_limit,
        }
        .into());
    }
    let mut seen = vec![false; cluster_count];
    for &cluster in &clusters {
        seen[usize::from(cluster)] = true;
    }
    if seen.iter().any(|&present| !present) {
        return invalid_entropy("entropy cluster map contains a hole");
    }
    clusters.shrink_to_fit();
    Ok(clusters)
}

fn read_code_length_code(reader: &mut impl BitInput) -> Result<u8> {
    Ok(match read_bits_u32(reader, 2)? {
        0 => 0,
        1 => 4,
        2 => 3,
        3 if reader.read_bits(1)? == 0 => 2,
        3 if reader.read_bits(1)? == 0 => 1,
        3 => 5,
        _ => unreachable!(),
    })
}

fn canonical_entries(lengths: &[u8]) -> Result<Vec<PrefixCodeEntry>> {
    let mut counts = [0u16; MAX_PREFIX_BITS as usize + 1];
    for &length in lengths {
        if length > MAX_PREFIX_BITS {
            return invalid_entropy("prefix code length exceeds 15 bits");
        }
        if length != 0 {
            counts[usize::from(length)] = counts[usize::from(length)]
                .checked_add(1)
                .ok_or_else(|| invalid_entropy_error("prefix length count overflow"))?;
        }
    }
    let mut next = [0u16; MAX_PREFIX_BITS as usize + 1];
    let mut code = 0u16;
    for length in 1..=MAX_PREFIX_BITS as usize {
        code = code
            .checked_add(counts[length - 1])
            .and_then(|value| value.checked_shl(1))
            .ok_or_else(|| invalid_entropy_error("canonical prefix code overflow"))?;
        next[length] = code;
    }
    let mut entries = Vec::with_capacity(lengths.len());
    for &length in lengths {
        if length == 0 {
            entries.push(PrefixCodeEntry::default());
            continue;
        }
        let index = usize::from(length);
        let bits = next[index].reverse_bits() >> (u16::BITS - u32::from(length));
        entries.push(PrefixCodeEntry {
            bit_len: length,
            bits,
        });
        next[index] = next[index].wrapping_add(1);
    }
    Ok(entries)
}

fn read_prefix_symbol(reader: &mut impl BitInput, entries: &[PrefixCodeEntry]) -> Result<u32> {
    if entries.len() == 1 && entries[0].bit_len == 0 {
        return Ok(0);
    }
    let mut bits = 0u16;
    for length in 1..=MAX_PREFIX_BITS {
        bits |= u16::try_from(reader.read_bits(1)?)
            .map_err(|_| invalid_entropy_error("prefix bit exceeds u16"))?
            << (length - 1);
        if let Some(symbol) = entries
            .iter()
            .position(|entry| entry.bit_len == length && entry.bits == bits)
        {
            return u32::try_from(symbol)
                .map_err(|_| invalid_entropy_error("prefix symbol exceeds u32").into());
        }
    }
    invalid_entropy("invalid prefix symbol")
}

fn read_ans_u8(reader: &mut impl BitInput) -> Result<u8> {
    if reader.read_bits(1)? == 0 {
        return Ok(0);
    }
    let bits = read_bits_u32(reader, 3)?;
    u8::try_from((1 << bits) + read_bits_u32(reader, bits)?)
        .map_err(|_| invalid_entropy_error("ANS u8 value overflow").into())
}

fn read_ans_prefix(reader: &mut impl BitInput) -> Result<u16> {
    Ok(match read_bits_u32(reader, 3)? {
        0 => 10,
        1 => {
            let mut result = 12;
            for value in [4, 0, 11, 13] {
                if reader.read_bits(1)? != 0 {
                    result = value;
                    break;
                }
            }
            result
        }
        2 => 7,
        3 if reader.read_bits(1)? != 0 => 1,
        3 => 3,
        4 => 6,
        5 => 8,
        6 => 9,
        7 if reader.read_bits(1)? != 0 => 2,
        7 => 5,
        _ => unreachable!(),
    })
}

fn resolve_lz77_distance(value: u32, multiplier: u32, decoded: u32) -> u32 {
    const SPECIAL: [[i8; 2]; 120] = [
        [0, 1],
        [1, 0],
        [1, 1],
        [-1, 1],
        [0, 2],
        [2, 0],
        [1, 2],
        [-1, 2],
        [2, 1],
        [-2, 1],
        [2, 2],
        [-2, 2],
        [0, 3],
        [3, 0],
        [1, 3],
        [-1, 3],
        [3, 1],
        [-3, 1],
        [2, 3],
        [-2, 3],
        [3, 2],
        [-3, 2],
        [0, 4],
        [4, 0],
        [1, 4],
        [-1, 4],
        [4, 1],
        [-4, 1],
        [3, 3],
        [-3, 3],
        [2, 4],
        [-2, 4],
        [4, 2],
        [-4, 2],
        [0, 5],
        [3, 4],
        [-3, 4],
        [4, 3],
        [-4, 3],
        [5, 0],
        [1, 5],
        [-1, 5],
        [5, 1],
        [-5, 1],
        [2, 5],
        [-2, 5],
        [5, 2],
        [-5, 2],
        [4, 4],
        [-4, 4],
        [3, 5],
        [-3, 5],
        [5, 3],
        [-5, 3],
        [0, 6],
        [6, 0],
        [1, 6],
        [-1, 6],
        [6, 1],
        [-6, 1],
        [2, 6],
        [-2, 6],
        [6, 2],
        [-6, 2],
        [4, 5],
        [-4, 5],
        [5, 4],
        [-5, 4],
        [3, 6],
        [-3, 6],
        [6, 3],
        [-6, 3],
        [0, 7],
        [7, 0],
        [1, 7],
        [-1, 7],
        [5, 5],
        [-5, 5],
        [7, 1],
        [-7, 1],
        [4, 6],
        [-4, 6],
        [6, 4],
        [-6, 4],
        [2, 7],
        [-2, 7],
        [7, 2],
        [-7, 2],
        [3, 7],
        [-3, 7],
        [7, 3],
        [-7, 3],
        [5, 6],
        [-5, 6],
        [6, 5],
        [-6, 5],
        [8, 0],
        [4, 7],
        [-4, 7],
        [7, 4],
        [-7, 4],
        [8, 1],
        [8, 2],
        [6, 6],
        [-6, 6],
        [8, 3],
        [5, 7],
        [-5, 7],
        [7, 5],
        [-7, 5],
        [8, 4],
        [6, 7],
        [-6, 7],
        [7, 6],
        [-7, 6],
        [8, 5],
        [7, 7],
        [-7, 7],
        [8, 6],
        [8, 7],
    ];
    let zero_based = if multiplier == 0 {
        value
    } else if value < 120 {
        let [offset, factor] = SPECIAL[value as usize];
        (i64::from(offset) + i64::from(multiplier) * i64::from(factor) - 1).max(0) as u32
    } else {
        value - 120
    };
    (zero_based.min((1 << 20) - 1) + 1).min(decoded)
}

fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        let magnitude = (value >> 1) + 1;
        if magnitude == 1u32 << 31 {
            i32::MIN
        } else {
            -i32::try_from(magnitude).expect("signed mapping magnitude is below i32::MAX")
        }
    }
}

fn read_bits_u32(reader: &mut impl BitInput, count: u32) -> Result<u32> {
    let count = u8::try_from(count)
        .map_err(|_| invalid_entropy_error("bit count exceeds the bounded reader"))?;
    u32::try_from(reader.read_bits(count)?)
        .map_err(|_| invalid_entropy_error("bit value exceeds u32").into())
}

fn add_log2_ceil(value: u32) -> u32 {
    if value >= 0x8000_0000 {
        32
    } else {
        (value + 1).next_power_of_two().trailing_zeros()
    }
}

fn invalid_entropy<T>(reason: &'static str) -> Result<T> {
    Err(invalid_entropy_error(reason).into())
}

fn invalid_entropy_error(reason: &'static str) -> ModularTreeError {
    ModularTreeError::InvalidEntropy { reason }
}

fn invalid_tree<T>(reason: &'static str) -> Result<T> {
    Err(invalid_tree_error(reason).into())
}

fn invalid_tree_error(reason: &'static str) -> ModularTreeError {
    ModularTreeError::InvalidTree { reason }
}

#[cfg(test)]
mod tests {
    use super::{
        AnsBucketIr, EntropyCoderIr, EntropyDecoderIr, HybridIntegerConfigIr,
        MetadataEntropyCursor, PrefixHistogramIr, add_log2_ceil, unpack_signed,
        validate_tree_property,
    };

    #[test]
    fn packed_ans_bucket_round_trips_fields() {
        let bucket = AnsBucketIr::new(7, 31, 4095, 1234, 2048);
        assert_eq!(bucket.fields(), (7, 31, 4095, 1234, 2048));
        assert_eq!(std::mem::size_of::<AnsBucketIr>(), 8);
    }

    #[test]
    fn hybrid_split_and_signed_mapping_match_jpeg_xl() {
        let config = HybridIntegerConfigIr {
            split_exponent: 8,
            msb_in_token: 0,
            lsb_in_token: 0,
        };
        assert_eq!(config.split(), 256);
        assert_eq!(unpack_signed(0), 0);
        assert_eq!(unpack_signed(1), -1);
        assert_eq!(unpack_signed(2), 1);
        assert_eq!(unpack_signed(3), -2);
        assert_eq!(add_log2_ceil(15), 4);
    }

    #[test]
    fn ma_tree_property_rejects_values_outside_the_byte_domain() {
        assert_eq!(validate_tree_property(255).unwrap(), 255);
        assert!(matches!(
            validate_tree_property(256).unwrap_err(),
            crate::Error::ModularTree(crate::ModularTreeError::InvalidProperty { property: 256 })
        ));
    }

    #[test]
    fn hybrid_value_bounds_include_every_extra_bit_pattern() {
        let config = HybridIntegerConfigIr {
            split_exponent: 0,
            msb_in_token: 0,
            lsb_in_token: 0,
        };
        assert_eq!(config.value_bounds(0), (0, 0));
        assert_eq!(config.value_bounds(1), (1, 1));
        assert_eq!(config.value_bounds(2), (2, 3));
    }

    #[test]
    fn prefix_distance_histogram_sizes_exact_power_of_two_lz_ring() {
        let descriptor = EntropyDecoderIr {
            lz77: Some(super::Lz77Ir {
                min_symbol: 224,
                min_length: 3,
                length_config: HybridIntegerConfigIr {
                    split_exponent: 0,
                    msb_in_token: 0,
                    lsb_in_token: 0,
                },
            }),
            context_to_cluster: vec![0, 0],
            configs: vec![HybridIntegerConfigIr {
                split_exponent: 8,
                msb_in_token: 0,
                lsb_in_token: 0,
            }],
            coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr::single(0).unwrap()]),
        };
        assert_eq!(descriptor.lz77_window_words(256, 65_536).unwrap(), 256);

        let mut distance_one = descriptor;
        distance_one.coder = EntropyCoderIr::Prefix(vec![PrefixHistogramIr::single(1).unwrap()]);
        assert_eq!(distance_one.lz77_window_words(256, 65_536).unwrap(), 1);
    }

    #[test]
    fn ans_alias_support_bounds_lz_ring_without_decoding_symbols() {
        let alias_distance_one = AnsBucketIr::new(1, 0, 4096, 0, 0);
        let descriptor = EntropyDecoderIr {
            lz77: Some(super::Lz77Ir {
                min_symbol: 224,
                min_length: 3,
                length_config: HybridIntegerConfigIr {
                    split_exponent: 0,
                    msb_in_token: 0,
                    lsb_in_token: 0,
                },
            }),
            context_to_cluster: vec![0, 0],
            configs: vec![HybridIntegerConfigIr {
                split_exponent: 8,
                msb_in_token: 0,
                lsb_in_token: 0,
            }],
            coder: EntropyCoderIr::Ans {
                log_alphabet_size: 5,
                histograms: vec![super::AnsHistogramIr {
                    buckets: vec![alias_distance_one, alias_distance_one],
                    log_bucket_size: 7,
                    single_symbol: None,
                }],
            },
        };
        assert_eq!(descriptor.lz77_window_words(256, 65_536).unwrap(), 1);
    }

    #[test]
    fn malformed_ans_alias_table_cannot_underallocate_lz_history() {
        let descriptor = EntropyDecoderIr {
            lz77: Some(super::Lz77Ir {
                min_symbol: 224,
                min_length: 3,
                length_config: HybridIntegerConfigIr {
                    split_exponent: 0,
                    msb_in_token: 0,
                    lsb_in_token: 0,
                },
            }),
            context_to_cluster: vec![0, 0],
            configs: vec![HybridIntegerConfigIr {
                split_exponent: 8,
                msb_in_token: 0,
                lsb_in_token: 0,
            }],
            coder: EntropyCoderIr::Ans {
                log_alphabet_size: 5,
                histograms: vec![super::AnsHistogramIr {
                    buckets: vec![AnsBucketIr::new(7, 0, 4096, 0, 0)],
                    log_bucket_size: 7,
                    single_symbol: None,
                }],
            },
        };
        let error = descriptor.lz77_window_words(256, 65_536).unwrap_err();
        assert!(matches!(
            error,
            crate::Error::ModularTree(crate::ModularTreeError::InvalidEntropy {
                reason: "ANS alias exceeds its alphabet"
            })
        ));
    }

    #[test]
    fn standalone_entropy_metadata_uses_common_zero_node_abi() {
        let descriptor = EntropyDecoderIr {
            lz77: None,
            context_to_cluster: vec![0],
            configs: vec![HybridIntegerConfigIr {
                split_exponent: 0,
                msb_in_token: 0,
                lsb_in_token: 0,
            }],
            coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr::single(3).unwrap()]),
        };
        let packed = descriptor.pack_gpu_metadata().unwrap();
        assert_eq!(packed.words[0], 0);
        assert_eq!(packed.words[1], 0);
        assert_eq!(packed.words[4], 24);
        assert_eq!(packed.words[5], 28);
        assert_eq!(packed.words[6], 28);
        assert_eq!(packed.words[27], 4);
    }

    #[test]
    fn nonzero_single_prefix_symbol_consumes_no_bits() {
        let descriptor = EntropyDecoderIr {
            lz77: None,
            context_to_cluster: vec![0],
            configs: vec![HybridIntegerConfigIr {
                split_exponent: 4,
                msb_in_token: 0,
                lsb_in_token: 0,
            }],
            coder: EntropyCoderIr::Prefix(vec![PrefixHistogramIr::single(5).unwrap()]),
        };
        let mut reader = jxl_gpu_bitstream::BitReader::new(&[]);
        let mut cursor = MetadataEntropyCursor::new(&descriptor, 1);
        cursor.begin(&mut reader).unwrap();
        assert_eq!(cursor.read_varint(&mut reader, 0, 0).unwrap(), 5);
        cursor.finalize().unwrap();
        assert_eq!(reader.bit_offset(), 0);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod gpu_tests;
