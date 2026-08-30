// The prefix-code construction in this module is derived from the permissively
// licensed zune-jpegxl 0.5.2 fast lossless encoder. See this crate's
// `THIRD_PARTY.md` and `LICENSES/zune-jpegxl-MIT.txt` for attribution.

use std::cmp::{max, min};

use jxl_gpu_bitstream::{BitWriter, PrefixCodeEntry};

use crate::EncodeError;

pub(crate) const RAW_SYMBOLS: usize = 19;
pub(crate) const LZ77_SYMBOLS: usize = 33;
const MAX_SYMBOLS: usize = LZ77_SYMBOLS;

const BASE_RAW_COUNTS: [u64; RAW_SYMBOLS] = [
    3843, 852, 1270, 1214, 1014, 727, 481, 300, 159, 51, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const BASE_LZ77_COUNTS: [u64; LZ77_SYMBOLS] = [
    29, 27, 25, 23, 21, 21, 19, 18, 21, 17, 16, 15, 15, 14, 13, 13, 137, 98, 61, 34, 1, 1, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0,
];
const MIN_RAW_LENGTH: [u8; RAW_SYMBOLS + 1] = [0; RAW_SYMBOLS + 1];
const MAX_RAW_LENGTH: [u8; RAW_SYMBOLS + 1] = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10, 15, 15, 15, 15, 15, 15, 15, 15,
];

#[derive(Clone, Debug)]
pub(crate) struct PrefixCode {
    raw_nbits: [u8; RAW_SYMBOLS],
    raw_bits: [u8; RAW_SYMBOLS],
    lz77_nbits: [u8; LZ77_SYMBOLS],
    lz77_bits: [u16; LZ77_SYMBOLS],
}

impl PrefixCode {
    pub(crate) fn from_aggregated_counts(
        raw_gpu: &[u64; RAW_SYMBOLS],
        lz77_gpu: &[u64; LZ77_SYMBOLS],
    ) -> Result<Self, EncodeError> {
        Self::from_aggregated_counts_with_max_token(raw_gpu, lz77_gpu, 9)
    }

    pub(crate) fn from_aggregated_ycocg_counts(
        raw_gpu: &[u64; RAW_SYMBOLS],
        lz77_gpu: &[u64; LZ77_SYMBOLS],
    ) -> Result<Self, EncodeError> {
        Self::from_aggregated_counts_with_max_token(raw_gpu, lz77_gpu, 10)
    }

    fn from_aggregated_counts_with_max_token(
        raw_gpu: &[u64; RAW_SYMBOLS],
        lz77_gpu: &[u64; LZ77_SYMBOLS],
        max_raw_token: usize,
    ) -> Result<Self, EncodeError> {
        if raw_gpu[max_raw_token + 1..].iter().any(|&count| count != 0)
            || lz77_gpu[28..].iter().any(|&count| count != 0)
        {
            return Err(EncodeError::Backend(
                "GPU histogram contains an impossible token".into(),
            ));
        }
        let mut base_raw_counts = BASE_RAW_COUNTS;
        if max_raw_token == 10 {
            // Reversible YCoCg can require one more hybrid-uint symbol than a direct 8-bit plane.
            base_raw_counts[10] = 5;
        }
        let mut raw_counts = [0u64; RAW_SYMBOLS];
        let mut lz77_counts = [0u64; LZ77_SYMBOLS];
        for (index, count) in raw_counts.iter_mut().enumerate() {
            *count = raw_gpu[index]
                .checked_shl(8)
                .and_then(|value| value.checked_add(base_raw_counts[index]))
                .ok_or_else(|| {
                    EncodeError::Backend("aggregate raw histogram scaling overflow".into())
                })?;
        }
        for (index, count) in lz77_counts.iter_mut().enumerate() {
            *count = lz77_gpu[index]
                .checked_shl(8)
                .and_then(|value| value.checked_add(BASE_LZ77_COUNTS[index]))
                .ok_or_else(|| {
                    EncodeError::Backend("aggregate LZ77 histogram scaling overflow".into())
                })?;
        }
        Ok(Self::new(&raw_counts, &lz77_counts))
    }

    pub(crate) fn fixed_unused_channel() -> Self {
        Self::new(&BASE_RAW_COUNTS, &BASE_LZ77_COUNTS)
    }

    pub(crate) fn raw_entries(&self) -> [PrefixCodeEntry; RAW_SYMBOLS] {
        std::array::from_fn(|index| PrefixCodeEntry {
            bit_len: self.raw_nbits[index],
            bits: u16::from(self.raw_bits[index]),
        })
    }

    pub(crate) fn lz77_entries(&self) -> [PrefixCodeEntry; LZ77_SYMBOLS] {
        std::array::from_fn(|index| PrefixCodeEntry {
            bit_len: self.lz77_nbits[index],
            bits: self.lz77_bits[index],
        })
    }

    fn new(raw_counts: &[u64; RAW_SYMBOLS], lz77_counts: &[u64; LZ77_SYMBOLS]) -> Self {
        let mut raw_nbits = [0; RAW_SYMBOLS];
        let mut raw_bits = [0; RAW_SYMBOLS];
        let mut lz77_nbits = [0; LZ77_SYMBOLS];
        let mut lz77_bits = [0_u16; LZ77_SYMBOLS];

        let mut level1_counts = [0; RAW_SYMBOLS + 1];
        level1_counts[..RAW_SYMBOLS].copy_from_slice(raw_counts);
        let mut num_raw = RAW_SYMBOLS;
        while num_raw > 0 && level1_counts[num_raw - 1] == 0 {
            num_raw -= 1;
        }
        level1_counts[num_raw] = lz77_counts.iter().sum();

        let mut level1_nbits = [0; RAW_SYMBOLS + 1];
        compute_code_lengths(
            &level1_counts,
            num_raw + 1,
            &MIN_RAW_LENGTH,
            &MAX_RAW_LENGTH,
            &mut level1_nbits,
        );

        let mut num_lz77 = LZ77_SYMBOLS;
        while num_lz77 > 0 && lz77_counts[num_lz77 - 1] == 0 {
            num_lz77 -= 1;
        }
        let mut level2_nbits = [0; LZ77_SYMBOLS];
        let min_lengths = [0; LZ77_SYMBOLS];
        let max_lengths = [15 - level1_nbits[num_raw]; LZ77_SYMBOLS];
        compute_code_lengths(
            lz77_counts,
            num_lz77,
            &min_lengths,
            &max_lengths,
            &mut level2_nbits,
        );

        raw_nbits[..num_raw].copy_from_slice(&level1_nbits[..num_raw]);
        for index in 0..num_lz77 {
            if level2_nbits[index] != 0 {
                lz77_nbits[index] = level1_nbits[num_raw] + level2_nbits[index];
            }
        }
        compute_canonical_code(
            &raw_nbits[..num_raw],
            &mut raw_bits[..num_raw],
            &lz77_nbits,
            &mut lz77_bits,
        );
        Self {
            raw_nbits,
            raw_bits,
            lz77_nbits,
            lz77_bits,
        }
    }

    pub(crate) fn write_tree(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        let mut code_length_counts = [0u64; 18];
        code_length_counts[17] = 3 + 2 * (LZ77_SYMBOLS - 1) as u64;
        for &length in &self.raw_nbits {
            code_length_counts[usize::from(length)] += 1;
        }
        for &length in &self.lz77_nbits {
            code_length_counts[usize::from(length)] += 1;
        }

        let mut code_length_nbits = [0u8; 18];
        compute_code_lengths(
            &code_length_counts,
            18,
            &[0; 18],
            &[5; 18],
            &mut code_length_nbits,
        );
        writer.write_bits(0, 2)?;

        const ORDER: [u8; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        const LENGTH_NBITS: [u8; 6] = [2, 4, 3, 2, 2, 4];
        const LENGTH_BITS: [u64; 6] = [0, 7, 3, 2, 1, 15];
        let mut num_code_lengths = ORDER.len();
        while code_length_nbits[usize::from(ORDER[num_code_lengths - 1])] == 0 {
            num_code_lengths -= 1;
        }
        for &ordered in &ORDER[..num_code_lengths] {
            let symbol = usize::from(code_length_nbits[usize::from(ordered)]);
            writer.write_bits(LENGTH_BITS[symbol], LENGTH_NBITS[symbol])?;
        }

        let mut code_length_bits = [0u16; 18];
        compute_canonical_code(&[], &mut [], &code_length_nbits, &mut code_length_bits);
        for &length in &self.raw_nbits {
            writer.write_bits(
                u64::from(code_length_bits[usize::from(length)]),
                code_length_nbits[usize::from(length)],
            )?;
        }

        let mut num_lz77 = LZ77_SYMBOLS;
        while self.lz77_nbits[num_lz77 - 1] == 0 {
            num_lz77 -= 1;
        }
        for repeated_bits in [0b010, 0b000, 0b010] {
            writer.write_bits(u64::from(code_length_bits[17]), code_length_nbits[17])?;
            writer.write_bits(repeated_bits, 3)?;
        }
        for &length in &self.lz77_nbits[..num_lz77] {
            writer.write_bits(
                u64::from(code_length_bits[usize::from(length)]),
                code_length_nbits[usize::from(length)],
            )?;
        }
        Ok(())
    }

    pub(crate) fn write_raw(
        &self,
        writer: &mut BitWriter,
        token: u32,
        nbits: u32,
        bits: u32,
    ) -> Result<(), EncodeError> {
        let token = usize::try_from(token)
            .map_err(|_| EncodeError::Backend("GPU raw token overflow".into()))?;
        let expected_nbits = token.saturating_sub(1);
        if token > 10
            || nbits != u32::try_from(expected_nbits).unwrap_or(u32::MAX)
            || !extra_bits_are_canonical(nbits, bits)
        {
            return Err(EncodeError::Backend(
                "GPU emitted an invalid raw token".into(),
            ));
        }
        writer.write_bits(u64::from(self.raw_bits[token]), self.raw_nbits[token])?;
        writer.write_bits(u64::from(bits), nbits as u8)?;
        Ok(())
    }

    pub(crate) fn write_run(
        &self,
        writer: &mut BitWriter,
        token: u32,
        nbits: u32,
        bits: u32,
    ) -> Result<(), EncodeError> {
        let token = usize::try_from(token)
            .map_err(|_| EncodeError::Backend("GPU LZ77 token overflow".into()))?;
        let expected_nbits = if token < 16 { 0 } else { token - 12 };
        if token > 27
            || nbits != u32::try_from(expected_nbits).unwrap_or(u32::MAX)
            || !extra_bits_are_canonical(nbits, bits)
        {
            return Err(EncodeError::Backend(
                "GPU emitted an invalid LZ77 token".into(),
            ));
        }
        writer.write_bits(u64::from(self.raw_bits[0]), self.raw_nbits[0])?;
        writer.write_bits(u64::from(self.lz77_bits[token]), self.lz77_nbits[token])?;
        writer.write_bits(u64::from(bits), nbits as u8)?;
        Ok(())
    }
}

fn extra_bits_are_canonical(nbits: u32, bits: u32) -> bool {
    match nbits {
        0 => bits == 0,
        1..=31 => bits < (1u32 << nbits),
        _ => false,
    }
}

fn bit_reverse(nbits: usize, bits: u16) -> u16 {
    const NIBBLE: [u16; 16] = [
        0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110, 0b0001, 0b1001, 0b0101,
        0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
    ];
    let reversed = (NIBBLE[usize::from(bits & 0xf)] << 12)
        | (NIBBLE[usize::from((bits >> 4) & 0xf)] << 8)
        | (NIBBLE[usize::from((bits >> 8) & 0xf)] << 4)
        | NIBBLE[usize::from((bits >> 12) & 0xf)];
    reversed >> 1 >> (16 - nbits - 1)
}

fn compute_canonical_code(
    first_nbits: &[u8],
    first_bits: &mut [u8],
    second_nbits: &[u8],
    second_bits: &mut [u16],
) {
    const MAX_CODE_LENGTH: usize = 15;
    let mut counts = [0u16; MAX_CODE_LENGTH + 1];
    for &length in first_nbits.iter().chain(second_nbits) {
        counts[usize::from(length)] += 1;
    }
    let mut next_code = [0u16; MAX_CODE_LENGTH + 1];
    let mut code = 0u16;
    for length in 1..=MAX_CODE_LENGTH {
        code = (code + counts[length - 1]) << 1;
        next_code[length] = code;
    }
    for (length, bits) in first_nbits.iter().copied().zip(first_bits) {
        let index = usize::from(length);
        *bits = bit_reverse(index, next_code[index]) as u8;
        next_code[index] = next_code[index].wrapping_add(1);
    }
    for (length, bits) in second_nbits.iter().copied().zip(second_bits) {
        let index = usize::from(length);
        if length != 0 {
            *bits = bit_reverse(index, next_code[index]);
            next_code[index] = next_code[index].wrapping_add(1);
        }
    }
}

fn compute_code_lengths(
    frequencies: &[u64],
    count: usize,
    min_limits: &[u8],
    max_limits: &[u8],
    output: &mut [u8],
) {
    let mut compact_frequencies = [0u64; MAX_SYMBOLS];
    let mut compact_min = [0u8; MAX_SYMBOLS];
    let mut compact_max = [0u8; MAX_SYMBOLS];
    let mut compact_count = 0;
    for index in 0..count {
        if frequencies[index] != 0 {
            compact_frequencies[compact_count] = frequencies[index];
            compact_min[compact_count] = min_limits[index];
            compact_max[compact_count] = max_limits[index];
            compact_count += 1;
        }
    }
    let mut compact_output = [0u8; MAX_SYMBOLS];
    compute_non_zero_lengths(
        &compact_frequencies,
        compact_count,
        &mut compact_min,
        &compact_max,
        &mut compact_output,
    );
    let mut compact_index = 0;
    for index in 0..count {
        output[index] = 0;
        if frequencies[index] != 0 {
            output[index] = compact_output[compact_index];
            compact_index += 1;
        }
    }
}

fn compute_non_zero_lengths(
    frequencies: &[u64],
    count: usize,
    min_limits: &mut [u8],
    max_limits: &[u8],
    output: &mut [u8],
) {
    let mut precision = 0u64;
    let mut shortest = u8::MAX;
    let mut frequency_sum = 0u64;
    for index in 0..count {
        frequency_sum += frequencies[index];
        min_limits[index] = max(min_limits[index], 1);
        precision = max(precision, u64::from(max_limits[index]));
        shortest = min(shortest, min_limits[index]);
    }
    precision -= u64::from(shortest) - 1;
    compute_non_zero_lengths_impl(
        frequencies,
        count,
        precision as usize,
        frequency_sum * precision,
        min_limits,
        max_limits,
        output,
    );
}

fn compute_non_zero_lengths_impl(
    frequencies: &[u64],
    count: usize,
    precision: usize,
    infinity: u64,
    min_limits: &[u8],
    max_limits: &[u8],
    output: &mut [u8],
) {
    let stride = (1usize << precision) + 1;
    let mut dynamic = vec![infinity; stride * (count + 1)];
    let offset = |symbol: usize, value: usize| symbol * stride + value;
    dynamic[offset(0, 0)] = 0;
    for symbol in 0..count {
        for length in min_limits[symbol]..=max_limits[symbol] {
            let delta = 1 << (precision - usize::from(length));
            for used in 0..=((1 << precision) - delta) {
                let target = offset(symbol + 1, used + delta);
                dynamic[target] = min(
                    dynamic[target],
                    dynamic[offset(symbol, used)] + frequencies[symbol] * u64::from(length),
                );
            }
        }
    }
    let mut symbol = count;
    let mut used = 1 << precision;
    while symbol > 0 {
        symbol -= 1;
        for length in min_limits[symbol]..=max_limits[symbol] {
            let delta = 1 << (precision - usize::from(length));
            if delta <= used
                && dynamic[offset(symbol + 1, used)]
                    == dynamic[offset(symbol, used - delta)]
                        + frequencies[symbol] * u64::from(length)
            {
                used -= delta;
                output[symbol] = length;
                break;
            }
        }
    }
}
