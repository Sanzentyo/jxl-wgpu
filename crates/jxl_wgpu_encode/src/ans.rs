//! JPEG XL ANS tables built from GPU histogram metadata. Token coding stays on GPU.
// The alias-table construction and histogram grammar follow libjxl's BSD-licensed
// ans_common.cc / enc_ans.cc. See THIRD_PARTY.md and LICENSES/libjxl-BSD.txt.
use jxl_gpu_bitstream::BitWriter;

use crate::{BackendError, EncodeError};

pub(crate) const ALPHABET: usize = 256;
pub(crate) const TABLE_SIZE: usize = 4096;
pub(crate) const TABLE_WORDS: usize = 2 * ALPHABET + TABLE_SIZE;

/// The wire alphabet also determines the decoder's alias bucket width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AnsAlphabet(u8);

impl AnsAlphabet {
    pub(crate) const MAX: Self = Self(8);

    pub(crate) fn for_symbols(symbols: usize) -> Result<Self, EncodeError> {
        if symbols == 0 || symbols > ALPHABET {
            return Err(BackendError::Invariant("ANS alphabet must contain 1..256 symbols").into());
        }
        Ok(Self(
            (usize::BITS - (symbols - 1).leading_zeros()).max(5) as u8
        ))
    }

    pub(crate) fn log_size(self) -> u8 {
        self.0
    }

    pub(crate) fn symbols(self) -> usize {
        1 << self.0
    }
}

/// Normalized selection metadata, without allocating a GPU alias table for each candidate.
#[derive(Clone, Debug)]
pub(crate) struct AnsHistogram {
    frequencies: [u32; ALPHABET],
    alphabet: AnsAlphabet,
}

impl AnsHistogram {
    #[cfg(test)]
    pub(crate) fn frequencies(&self) -> &[u32; ALPHABET] {
        &self.frequencies
    }

    pub(crate) fn from_counts(
        counts: &[u64; ALPHABET],
        alphabet: AnsAlphabet,
    ) -> Result<Self, EncodeError> {
        if counts[alphabet.symbols()..].iter().any(|&count| count != 0) {
            return Err(BackendError::Invariant("ANS counts exceed the selected alphabet").into());
        }
        let total: u128 = counts.iter().map(|&count| u128::from(count)).sum();
        let active = counts.iter().filter(|&&count| count != 0).count();
        let mut frequencies = [0; ALPHABET];
        if active == 0 {
            frequencies[0] = TABLE_SIZE as u32;
        } else {
            let remaining = (TABLE_SIZE - active) as u128;
            let mut remainders = Vec::with_capacity(active);
            for (symbol, &count) in counts.iter().enumerate().filter(|(_, count)| **count != 0) {
                let scaled = u128::from(count) * remaining;
                frequencies[symbol] = 1 + (scaled / total) as u32;
                remainders.push((scaled % total, symbol));
            }
            remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let missing = TABLE_SIZE
                - frequencies
                    .iter()
                    .map(|&count| count as usize)
                    .sum::<usize>();
            for &(_, symbol) in &remainders[..missing] {
                frequencies[symbol] += 1;
            }
        }
        Self::from_frequencies(frequencies, alphabet)
    }

    fn from_frequencies(
        frequencies: [u32; ALPHABET],
        alphabet: AnsAlphabet,
    ) -> Result<Self, EncodeError> {
        if frequencies
            .iter()
            .map(|&value| u64::from(value))
            .sum::<u64>()
            != TABLE_SIZE as u64
            || frequencies[alphabet.symbols()..]
                .iter()
                .any(|&count| count != 0)
        {
            return Err(BackendError::Invariant(
                "ANS frequencies must fit the alphabet and sum to 4096",
            )
            .into());
        }
        Ok(Self {
            frequencies,
            alphabet,
        })
    }

    pub(crate) fn compile(self) -> Result<AnsCode, EncodeError> {
        let frequencies = self.frequencies;
        let mut words = vec![0; TABLE_WORDS];
        words[..ALPHABET].copy_from_slice(&frequencies);
        let mut offset = 0;
        for (symbol, &frequency) in frequencies.iter().enumerate() {
            words[ALPHABET + symbol] = offset;
            offset += frequency;
        }
        if frequencies.contains(&(TABLE_SIZE as u32)) {
            for state in 0..TABLE_SIZE {
                words[2 * ALPHABET + state] = state as u32;
            }
        } else {
            let alphabet = self.alphabet.symbols();
            let bucket_size = (TABLE_SIZE / alphabet) as u32;
            let mut cutoffs = frequencies;
            let mut alias = [0usize; ALPHABET];
            let mut offsets = [0u32; ALPHABET];
            let mut under = Vec::new();
            let mut over = Vec::new();
            for (index, &frequency) in frequencies[..alphabet].iter().enumerate() {
                match frequency.cmp(&bucket_size) {
                    std::cmp::Ordering::Less => under.push(index),
                    std::cmp::Ordering::Greater => over.push(index),
                    std::cmp::Ordering::Equal => {}
                }
            }
            while let Some(large) = over.pop() {
                let small = under
                    .pop()
                    .ok_or(BackendError::Invariant("unbalanced ANS aliases"))?;
                cutoffs[large] -= bucket_size - cutoffs[small];
                alias[small] = large;
                offsets[small] = cutoffs[large];
                match cutoffs[large].cmp(&bucket_size) {
                    std::cmp::Ordering::Less => under.push(large),
                    std::cmp::Ordering::Greater => over.push(large),
                    std::cmp::Ordering::Equal => {}
                }
            }
            for bucket in 0..alphabet {
                for position in 0..bucket_size {
                    let (symbol, rank) = if position < cutoffs[bucket] {
                        (bucket, position)
                    } else {
                        (alias[bucket], offsets[bucket] + position - cutoffs[bucket])
                    };
                    if rank >= frequencies[symbol] {
                        return Err(
                            BackendError::Invariant("ANS alias rank exceeds frequency").into()
                        );
                    }
                    let destination = (words[ALPHABET + symbol] + rank) as usize;
                    words[2 * ALPHABET + destination] = bucket as u32 * bucket_size + position;
                }
            }
        }
        Ok(AnsCode {
            histogram: self,
            words,
        })
    }

    /// Cross entropy of the normalized distribution in Q20 bits. This is a size
    /// estimate, not a simulation of the order-dependent ANS state or extra bits.
    pub(crate) fn estimated_data_bits(
        &self,
        counts: &[u64; ALPHABET],
    ) -> Result<u128, EncodeError> {
        static COSTS: std::sync::OnceLock<[u32; TABLE_SIZE + 1]> = std::sync::OnceLock::new();
        let costs = COSTS.get_or_init(|| {
            std::array::from_fn(|frequency| {
                if frequency == 0 {
                    return 0;
                }
                // Binary logarithm by repeated squaring, with 48 fractional working bits.
                // Integer arithmetic makes codebook selection independent of host libm.
                let exponent = frequency.ilog2();
                let mut value = (frequency as u128) << (48 - exponent);
                let mut fraction = 0;
                for bit in (0..20).rev() {
                    value = (value * value) >> 48;
                    if value >= 1u128 << 49 {
                        value >>= 1;
                        fraction |= 1 << bit;
                    }
                }
                ((12 - exponent) << 20) - fraction
            })
        });
        counts
            .iter()
            .zip(self.frequencies)
            .try_fold(0u128, |sum, (&count, frequency)| {
                if count != 0 && frequency == 0 {
                    return Err(
                        BackendError::Invariant("ANS cost excludes an observed symbol").into(),
                    );
                }
                Ok(sum + u128::from(count) * u128::from(costs[frequency as usize]))
            })
    }

    pub(crate) fn write_histogram(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        let symbols: Vec<_> = self
            .frequencies
            .iter()
            .enumerate()
            .filter_map(|(symbol, &frequency)| (frequency != 0).then_some(symbol))
            .collect();
        if symbols.len() <= 2 {
            writer.write_bits(1, 1)?;
            writer.write_bits((symbols.len() - 1) as u64, 1)?;
            for &symbol in &symbols {
                write_u8(writer, symbol as u32)?;
            }
            if symbols.len() == 2 {
                writer.write_bits(u64::from(self.frequencies[symbols[0]]), 12)?;
            }
            return Ok(());
        }
        writer.write_bits(0, 2)?; // general, non-flat histogram
        writer.write_bits(7, 3)?;
        writer.write_bits(5, 3)?; // exact population precision: shift=12
        let alphabet = symbols.last().copied().unwrap() + 1;
        write_u8(writer, (alphabet - 3) as u32)?;
        let widths = self.frequencies.map(|count| 32 - count.leading_zeros());
        let maximum = widths.iter().copied().max().unwrap();
        let omitted = widths.iter().position(|&width| width == maximum).unwrap();
        const LENGTHS: [u8; 13] = [5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 6, 7];
        const BITS: [u64; 13] = [17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1];
        for &width in &widths[..alphabet] {
            writer.write_bits(BITS[width as usize], LENGTHS[width as usize])?;
        }
        for (symbol, &width) in widths[..alphabet].iter().enumerate() {
            if symbol != omitted && width > 1 {
                writer.write_bits(
                    u64::from(self.frequencies[symbol] - (1 << (width - 1))),
                    (width - 1) as u8,
                )?;
            }
        }
        Ok(())
    }
}

/// GPU lowering of a selected, checked histogram. Wire and GPU tables share its alphabet.
#[derive(Clone, Debug)]
pub(crate) struct AnsCode {
    histogram: AnsHistogram,
    words: Vec<u32>,
}

impl AnsCode {
    pub(crate) fn gpu_words(&self) -> &[u32] {
        &self.words
    }

    pub(crate) fn write_histogram(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        self.histogram.write_histogram(writer)
    }

    pub(crate) fn alphabet(&self) -> AnsAlphabet {
        self.histogram.alphabet
    }

    #[cfg(test)]
    pub(crate) fn from_counts(counts: &[u64; ALPHABET]) -> Result<Self, EncodeError> {
        AnsHistogram::from_counts(counts, AnsAlphabet::MAX)?.compile()
    }

    #[cfg(test)]
    fn from_frequencies(frequencies: [u32; ALPHABET]) -> Result<Self, EncodeError> {
        AnsHistogram::from_frequencies(frequencies, AnsAlphabet::MAX)?.compile()
    }

    #[cfg(test)]
    pub(crate) fn estimated_data_bits(
        &self,
        counts: &[u64; ALPHABET],
    ) -> Result<u128, EncodeError> {
        self.histogram.estimated_data_bits(counts)
    }
}

fn write_u8(writer: &mut BitWriter, value: u32) -> Result<(), EncodeError> {
    writer.write_bits(u64::from(value != 0), 1)?;
    if value != 0 {
        let bits = value.ilog2();
        writer.write_bits(u64::from(bits), 3)?;
        writer.write_bits(u64::from(value - (1 << bits)), bits as u8)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
