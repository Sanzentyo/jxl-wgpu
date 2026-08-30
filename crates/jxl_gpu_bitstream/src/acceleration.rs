//! Versioned GPU acceleration index carried by the private `jwgp` container box.
//!
//! The index does not contain pixels, residuals, or entropy events. It binds a compact set of
//! prefix tables and bit offsets to the exact standard codestream so a GPU frontend can decode the
//! real `jxlc` token bits without repeating the generic JPEG XL header/tree parser.

use std::cmp::{max, min};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::BitWriter;

/// Private ISO BMFF box used for the versioned GPU acceleration index.
pub const ACCELERATION_INDEX_BOX_TYPE: [u8; 4] = *b"jwgp";

const MAGIC: [u8; 4] = *b"JWGP";
const VERSION: u16 = 1;
const HEADER_SIZE: u16 = 84;
const PROFILE_GRAY8: u16 = 1;
const FLAG_LSB_FIRST: u16 = 1;
const PAYLOAD_SIZE: usize = 240;
const RAW_SYMBOLS: usize = 19;
const LZ77_SYMBOLS: usize = 33;
const GRADIENT_PREDICTOR: u8 = 5;

const UNUSED_RAW_PREFIX: [PrefixCodeEntry; RAW_SYMBOLS] = [
    PrefixCodeEntry::new(2, 0),
    PrefixCodeEntry::new(3, 2),
    PrefixCodeEntry::new(3, 6),
    PrefixCodeEntry::new(3, 1),
    PrefixCodeEntry::new(3, 5),
    PrefixCodeEntry::new(4, 3),
    PrefixCodeEntry::new(4, 11),
    PrefixCodeEntry::new(5, 7),
    PrefixCodeEntry::new(6, 23),
    PrefixCodeEntry::new(6, 55),
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
];

const UNUSED_LZ77_PREFIX: [PrefixCodeEntry; LZ77_SYMBOLS] = [
    PrefixCodeEntry::new(9, 159),
    PrefixCodeEntry::new(9, 415),
    PrefixCodeEntry::new(9, 95),
    PrefixCodeEntry::new(9, 351),
    PrefixCodeEntry::new(9, 223),
    PrefixCodeEntry::new(9, 479),
    PrefixCodeEntry::new(9, 63),
    PrefixCodeEntry::new(9, 319),
    PrefixCodeEntry::new(9, 191),
    PrefixCodeEntry::new(9, 447),
    PrefixCodeEntry::new(9, 127),
    PrefixCodeEntry::new(10, 383),
    PrefixCodeEntry::new(10, 895),
    PrefixCodeEntry::new(10, 255),
    PrefixCodeEntry::new(10, 767),
    PrefixCodeEntry::new(10, 511),
    PrefixCodeEntry::new(6, 15),
    PrefixCodeEntry::new(7, 47),
    PrefixCodeEntry::new(7, 111),
    PrefixCodeEntry::new(8, 31),
    PrefixCodeEntry::new(13, 1023),
    PrefixCodeEntry::new(13, 5119),
    PrefixCodeEntry::new(13, 3071),
    PrefixCodeEntry::new(13, 7167),
    PrefixCodeEntry::new(13, 2047),
    PrefixCodeEntry::new(13, 6143),
    PrefixCodeEntry::new(13, 4095),
    PrefixCodeEntry::new(13, 8191),
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
    PrefixCodeEntry::EMPTY,
];

/// One canonical LSB-first prefix-code entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PrefixCodeEntry {
    pub bit_len: u8,
    pub bits: u16,
}

impl PrefixCodeEntry {
    pub const EMPTY: Self = Self::new(0, 0);

    #[must_use]
    pub const fn new(bit_len: u8, bits: u16) -> Self {
        Self { bit_len, bits }
    }
}

/// Acceleration metadata for one 8-bit grayscale, single-group, lossless Modular codestream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray8AccelerationIndex {
    codestream_len: u64,
    codestream_sha256: [u8; 32],
    width: u32,
    height: u32,
    token_bit_offset: u64,
    token_bit_len: u64,
    sample_count: u32,
    raw_prefix: [PrefixCodeEntry; RAW_SYMBOLS],
    lz77_prefix: [PrefixCodeEntry; LZ77_SYMBOLS],
}

impl Gray8AccelerationIndex {
    pub const SERIALIZED_SIZE: usize = PAYLOAD_SIZE;
    pub const RAW_SYMBOLS: usize = RAW_SYMBOLS;
    pub const LZ77_SYMBOLS: usize = LZ77_SYMBOLS;
    pub const PREDICTOR: u8 = GRADIENT_PREDICTOR;

    /// Creates and validates an index bound to `codestream`.
    pub fn new(
        codestream: &[u8],
        width: u32,
        height: u32,
        token_bit_offset: u64,
        token_bit_len: u64,
        raw_prefix: [PrefixCodeEntry; RAW_SYMBOLS],
        lz77_prefix: [PrefixCodeEntry; LZ77_SYMBOLS],
    ) -> Result<Self, AccelerationIndexError> {
        let codestream_len =
            u64::try_from(codestream.len()).map_err(|_| AccelerationIndexError::SizeOverflow)?;
        let sample_count = width
            .checked_mul(height)
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        let index = Self {
            codestream_len,
            codestream_sha256: Sha256::digest(codestream).into(),
            width,
            height,
            token_bit_offset,
            token_bit_len,
            sample_count,
            raw_prefix,
            lz77_prefix,
        };
        index.validate_structure()?;
        Ok(index)
    }

    /// Parses and structurally validates a fixed-size v1 payload without trusting its binding.
    pub fn parse(payload: &[u8]) -> Result<Self, AccelerationIndexError> {
        if payload.len() != PAYLOAD_SIZE {
            return Err(AccelerationIndexError::PayloadSize {
                expected: PAYLOAD_SIZE,
                actual: payload.len(),
            });
        }
        if payload[..4] != MAGIC {
            return Err(AccelerationIndexError::Magic);
        }
        let version = read_u16(payload, 4);
        if version != VERSION {
            return Err(AccelerationIndexError::Version(version));
        }
        let header_size = read_u16(payload, 6);
        if header_size != HEADER_SIZE {
            return Err(AccelerationIndexError::HeaderSize(header_size));
        }
        let profile = read_u16(payload, 8);
        if profile != PROFILE_GRAY8 {
            return Err(AccelerationIndexError::Profile(profile));
        }
        let flags = read_u16(payload, 10);
        if flags != FLAG_LSB_FIRST {
            return Err(AccelerationIndexError::Flags(flags));
        }

        let mut digest = [0u8; 32];
        digest.copy_from_slice(&payload[20..52]);
        let mut cursor = usize::from(HEADER_SIZE);
        let raw_prefix = read_prefix_table::<RAW_SYMBOLS>(payload, &mut cursor);
        let lz77_prefix = read_prefix_table::<LZ77_SYMBOLS>(payload, &mut cursor);
        debug_assert_eq!(cursor, PAYLOAD_SIZE);

        let predictor = payload[80];
        let channels = payload[81];
        let bits_per_sample = payload[82];
        let reserved = payload[83];
        if predictor != GRADIENT_PREDICTOR || channels != 1 || bits_per_sample != 8 || reserved != 0
        {
            return Err(AccelerationIndexError::ProfileFields {
                predictor,
                channels,
                bits_per_sample,
                reserved,
            });
        }

        let index = Self {
            codestream_len: read_u64(payload, 12),
            codestream_sha256: digest,
            width: read_u32(payload, 52),
            height: read_u32(payload, 56),
            token_bit_offset: read_u64(payload, 60),
            token_bit_len: read_u64(payload, 68),
            sample_count: read_u32(payload, 76),
            raw_prefix,
            lz77_prefix,
        };
        index.validate_structure()?;
        Ok(index)
    }

    /// Parses an index and verifies that it names exactly `codestream`.
    pub fn parse_bound(payload: &[u8], codestream: &[u8]) -> Result<Self, AccelerationIndexError> {
        let index = Self::parse(payload)?;
        index.verify_codestream(codestream)?;
        Ok(index)
    }

    /// Serializes the canonical fixed-size v1 payload.
    #[must_use]
    pub fn serialize(&self) -> [u8; PAYLOAD_SIZE] {
        let mut output = [0u8; PAYLOAD_SIZE];
        output[..4].copy_from_slice(&MAGIC);
        write_u16(&mut output, 4, VERSION);
        write_u16(&mut output, 6, HEADER_SIZE);
        write_u16(&mut output, 8, PROFILE_GRAY8);
        write_u16(&mut output, 10, FLAG_LSB_FIRST);
        write_u64(&mut output, 12, self.codestream_len);
        output[20..52].copy_from_slice(&self.codestream_sha256);
        write_u32(&mut output, 52, self.width);
        write_u32(&mut output, 56, self.height);
        write_u64(&mut output, 60, self.token_bit_offset);
        write_u64(&mut output, 68, self.token_bit_len);
        write_u32(&mut output, 76, self.sample_count);
        output[80] = GRADIENT_PREDICTOR;
        output[81] = 1;
        output[82] = 8;
        let mut cursor = usize::from(HEADER_SIZE);
        write_prefix_table(&mut output, &mut cursor, &self.raw_prefix);
        write_prefix_table(&mut output, &mut cursor, &self.lz77_prefix);
        debug_assert_eq!(cursor, PAYLOAD_SIZE);
        output
    }

    pub fn verify_codestream(&self, codestream: &[u8]) -> Result<(), AccelerationIndexError> {
        let actual_len =
            u64::try_from(codestream.len()).map_err(|_| AccelerationIndexError::SizeOverflow)?;
        if actual_len != self.codestream_len {
            return Err(AccelerationIndexError::CodestreamLength {
                expected: self.codestream_len,
                actual: actual_len,
            });
        }
        let actual: [u8; 32] = Sha256::digest(codestream).into();
        if actual != self.codestream_sha256 {
            return Err(AccelerationIndexError::CodestreamDigest);
        }
        Ok(())
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    #[must_use]
    pub const fn token_bit_offset(&self) -> u64 {
        self.token_bit_offset
    }

    #[must_use]
    pub const fn token_bit_len(&self) -> u64 {
        self.token_bit_len
    }

    #[must_use]
    pub const fn raw_prefix(&self) -> &[PrefixCodeEntry; RAW_SYMBOLS] {
        &self.raw_prefix
    }

    #[must_use]
    pub const fn lz77_prefix(&self) -> &[PrefixCodeEntry; LZ77_SYMBOLS] {
        &self.lz77_prefix
    }

    /// Proves that the standard group metadata encodes the prefix tables carried by this index.
    ///
    /// `group_start` is an absolute LSB-first bit offset in `codestream`. On success, the exact
    /// fixed Modular metadata and four prefix trees match and end precisely at
    /// [`token_bit_offset`](Self::token_bit_offset). This prevents an untrusted private box from
    /// changing the meaning of the standard `jxlc` bytes.
    pub fn validate_group_prefix(
        &self,
        codestream: &[u8],
        group_start: u64,
    ) -> Result<(), AccelerationIndexError> {
        self.verify_codestream(codestream)?;
        let expected = write_expected_group_prefix(self)?;
        let expected_len =
            u64::try_from(expected.bit_len()).map_err(|_| AccelerationIndexError::SizeOverflow)?;
        let expected_token_offset = group_start
            .checked_add(expected_len)
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        if self.token_bit_offset != expected_token_offset {
            return Err(AccelerationIndexError::TokenOffset {
                expected: expected_token_offset,
                actual: self.token_bit_offset,
            });
        }
        let codestream_bits = u64::try_from(codestream.len())
            .ok()
            .and_then(|bytes| bytes.checked_mul(8))
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        if expected_token_offset > codestream_bits {
            return Err(AccelerationIndexError::GroupPrefixRange);
        }
        for bit in 0..expected_len {
            let actual_offset = group_start
                .checked_add(bit)
                .ok_or(AccelerationIndexError::SizeOverflow)?;
            if read_bit(codestream, actual_offset) != read_bit(expected.as_bytes(), bit) {
                return Err(AccelerationIndexError::GroupPrefixMismatch { bit });
            }
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), AccelerationIndexError> {
        if self.width == 0 || self.height == 0 {
            return Err(AccelerationIndexError::EmptyExtent);
        }
        let expected_samples = self
            .width
            .checked_mul(self.height)
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        if self.sample_count != expected_samples {
            return Err(AccelerationIndexError::SampleCount {
                expected: expected_samples,
                actual: self.sample_count,
            });
        }
        if self.token_bit_len == 0 {
            return Err(AccelerationIndexError::EmptyTokenStream);
        }
        let codestream_bits = self
            .codestream_len
            .checked_mul(8)
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        let token_end = self
            .token_bit_offset
            .checked_add(self.token_bit_len)
            .ok_or(AccelerationIndexError::SizeOverflow)?;
        if token_end > codestream_bits {
            return Err(AccelerationIndexError::TokenRange {
                offset: self.token_bit_offset,
                length: self.token_bit_len,
                codestream_bits,
            });
        }
        validate_prefix_tables(&self.raw_prefix, &self.lz77_prefix)
    }
}

fn read_bit(bytes: &[u8], bit: u64) -> bool {
    let byte = usize::try_from(bit / 8)
        .ok()
        .and_then(|index| bytes.get(index))
        .copied()
        .unwrap_or(0);
    byte & (1 << (bit % 8)) != 0
}

fn write_expected_group_prefix(
    index: &Gray8AccelerationIndex,
) -> Result<BitWriter, AccelerationIndexError> {
    let mut output = BitWriter::new();
    // Fixed fast-lossless Modular metadata, adapted from zune-jpegxl 0.5.2. See the repository's
    // THIRD_PARTY.md and LICENSES/zune-jpegxl-MIT.txt.
    write(&mut output, 1, 1)?;
    write(&mut output, 1, 1)?;
    write(&mut output, 0, 1)?;
    write(&mut output, 1, 1)?;
    write(&mut output, 0, 2)?;
    write(&mut output, 1, 1)?;
    write(&mut output, 0, 4)?;
    write(&mut output, 0b100011, 6)?;
    write(&mut output, 1, 2)?;
    write(&mut output, 3, 2)?;
    for symbol in 0..4 {
        write(&mut output, symbol, 2)?;
    }
    write(&mut output, 0, 1)?;

    const TREE_INDICES: [usize; 26] = [
        1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0,
    ];
    const SYMBOL_BITS: [u64; 6] = [0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111];
    const SYMBOL_NBITS: [u8; 6] = [2, 2, 3, 3, 4, 4];
    for index in TREE_INDICES {
        write(&mut output, SYMBOL_BITS[index], SYMBOL_NBITS[index])?;
    }

    write(&mut output, 1, 1)?;
    write(&mut output, 0, 2)?;
    write(&mut output, 0b1010, 4)?;
    write(&mut output, 4, 4)?;
    write(&mut output, 0, 3)?;
    write(&mut output, 0, 3)?;
    write(&mut output, 1, 1)?;
    write(&mut output, 3, 2)?;
    for context in [4, 3, 2, 1, 0] {
        write(&mut output, context, 3)?;
    }
    write(&mut output, 1, 1)?;
    write(&mut output, 0, 4)?;
    for _ in 0..4 {
        write(&mut output, 0, 4)?;
    }
    write(&mut output, 1, 5)?;
    for _ in 0..4 {
        write(&mut output, 1, 1)?;
        write(&mut output, 8, 4)?;
        write(&mut output, 0, 8)?;
    }
    write(&mut output, 1, 2)?;
    write(&mut output, 0, 2)?;
    write(&mut output, 1, 1)?;
    write_prefix_tree(&mut output, &index.raw_prefix, &index.lz77_prefix)?;
    for _ in 0..3 {
        write_prefix_tree(&mut output, &UNUSED_RAW_PREFIX, &UNUSED_LZ77_PREFIX)?;
    }
    write(&mut output, 1, 1)?;
    write(&mut output, 1, 1)?;
    write(&mut output, 0, 2)?;
    Ok(output)
}

fn write_prefix_tree(
    writer: &mut BitWriter,
    raw: &[PrefixCodeEntry; RAW_SYMBOLS],
    lz77: &[PrefixCodeEntry; LZ77_SYMBOLS],
) -> Result<(), AccelerationIndexError> {
    let mut code_length_counts = [0u64; 18];
    code_length_counts[17] = 3 + 2 * (LZ77_SYMBOLS - 1) as u64;
    for entry in raw.iter().chain(lz77) {
        code_length_counts[usize::from(entry.bit_len)] += 1;
    }
    let mut code_length_nbits = [0u8; 18];
    compute_code_lengths(
        &code_length_counts,
        18,
        &[0; 18],
        &[5; 18],
        &mut code_length_nbits,
    );
    write(writer, 0, 2)?;

    const ORDER: [u8; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    const LENGTH_NBITS: [u8; 6] = [2, 4, 3, 2, 2, 4];
    const LENGTH_BITS: [u64; 6] = [0, 7, 3, 2, 1, 15];
    let mut num_code_lengths = ORDER.len();
    while code_length_nbits[usize::from(ORDER[num_code_lengths - 1])] == 0 {
        num_code_lengths -= 1;
    }
    for &ordered in &ORDER[..num_code_lengths] {
        let symbol = usize::from(code_length_nbits[usize::from(ordered)]);
        write(writer, LENGTH_BITS[symbol], LENGTH_NBITS[symbol])?;
    }

    let code_length_bits = canonical_bits(&code_length_nbits);
    for entry in raw {
        let length = usize::from(entry.bit_len);
        write(
            writer,
            u64::from(code_length_bits[length]),
            code_length_nbits[length],
        )?;
    }
    let num_lz77 = lz77
        .iter()
        .rposition(|entry| entry.bit_len != 0)
        .map_or(0, |index| index + 1);
    for repeated_bits in [0b010, 0b000, 0b010] {
        write(
            writer,
            u64::from(code_length_bits[17]),
            code_length_nbits[17],
        )?;
        write(writer, repeated_bits, 3)?;
    }
    for entry in &lz77[..num_lz77] {
        let length = usize::from(entry.bit_len);
        write(
            writer,
            u64::from(code_length_bits[length]),
            code_length_nbits[length],
        )?;
    }
    Ok(())
}

fn write(writer: &mut BitWriter, value: u64, bit_len: u8) -> Result<(), AccelerationIndexError> {
    writer
        .write_bits(value, bit_len)
        .map_err(|_| AccelerationIndexError::PrefixSerialization)
}

fn canonical_bits<const N: usize>(bit_lengths: &[u8; N]) -> [u16; N] {
    let mut counts = [0u16; 16];
    for &length in bit_lengths {
        counts[usize::from(length)] += 1;
    }
    let mut next_code = [0u16; 16];
    let mut code = 0u16;
    for length in 1..=15 {
        code = (code + counts[length - 1]) << 1;
        next_code[length] = code;
    }
    std::array::from_fn(|index| {
        let length = bit_lengths[index];
        if length == 0 {
            return 0;
        }
        let slot = usize::from(length);
        let bits = reverse_low_bits(next_code[slot], length);
        next_code[slot] = next_code[slot].wrapping_add(1);
        bits
    })
}

fn compute_code_lengths(
    frequencies: &[u64],
    count: usize,
    min_limits: &[u8],
    max_limits: &[u8],
    output: &mut [u8],
) {
    let mut compact_frequencies = [0u64; LZ77_SYMBOLS];
    let mut compact_min = [0u8; LZ77_SYMBOLS];
    let mut compact_max = [0u8; LZ77_SYMBOLS];
    let mut compact_count = 0;
    for index in 0..count {
        if frequencies[index] != 0 {
            compact_frequencies[compact_count] = frequencies[index];
            compact_min[compact_count] = min_limits[index];
            compact_max[compact_count] = max_limits[index];
            compact_count += 1;
        }
    }
    let mut compact_output = [0u8; LZ77_SYMBOLS];
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

fn validate_prefix_tables(
    raw: &[PrefixCodeEntry; RAW_SYMBOLS],
    lz77: &[PrefixCodeEntry; LZ77_SYMBOLS],
) -> Result<(), AccelerationIndexError> {
    if raw[..10].iter().any(|entry| entry.bit_len == 0) {
        return Err(AccelerationIndexError::MissingRawSymbol);
    }
    let mut counts = [0u32; 16];
    for (alphabet, entries) in [("raw", raw.as_slice()), ("lz77", lz77.as_slice())] {
        for (index, entry) in entries.iter().copied().enumerate() {
            if entry.bit_len > 15
                || (entry.bit_len == 0 && entry.bits != 0)
                || (entry.bit_len != 0 && u32::from(entry.bits) >= 1u32 << entry.bit_len)
            {
                return Err(AccelerationIndexError::PrefixEntry {
                    alphabet,
                    index,
                    bit_len: entry.bit_len,
                    bits: entry.bits,
                });
            }
            if entry.bit_len != 0 {
                counts[usize::from(entry.bit_len)] += 1;
            }
        }
    }
    let used = counts
        .iter()
        .enumerate()
        .skip(1)
        .try_fold(0u32, |sum, (length, count)| {
            sum.checked_add(count << (15 - length))
        })
        .ok_or(AccelerationIndexError::OversubscribedPrefix)?;
    if used > 1 << 15 {
        return Err(AccelerationIndexError::OversubscribedPrefix);
    }

    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for length in 1..=15 {
        code = (code + counts[length - 1]) << 1;
        next_code[length] = code;
    }
    for (alphabet, entries) in [("raw", raw.as_slice()), ("lz77", lz77.as_slice())] {
        for (index, entry) in entries.iter().copied().enumerate() {
            if entry.bit_len == 0 {
                continue;
            }
            let length = usize::from(entry.bit_len);
            let expected = reverse_low_bits(next_code[length] as u16, entry.bit_len);
            if entry.bits != expected {
                return Err(AccelerationIndexError::NonCanonicalPrefix {
                    alphabet,
                    index,
                    expected,
                    actual: entry.bits,
                });
            }
            next_code[length] += 1;
        }
    }
    Ok(())
}

fn reverse_low_bits(value: u16, bit_len: u8) -> u16 {
    value.reverse_bits() >> (16 - bit_len)
}

fn read_prefix_table<const N: usize>(payload: &[u8], cursor: &mut usize) -> [PrefixCodeEntry; N] {
    std::array::from_fn(|_| {
        let entry = PrefixCodeEntry {
            bit_len: payload[*cursor],
            bits: read_u16(payload, *cursor + 1),
        };
        *cursor += 3;
        entry
    })
}

fn write_prefix_table(output: &mut [u8], cursor: &mut usize, entries: &[PrefixCodeEntry]) {
    for entry in entries {
        output[*cursor] = entry.bit_len;
        write_u16(output, *cursor + 1, entry.bits);
        *cursor += 3;
    }
}

fn read_u16(input: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(input[offset..offset + 2].try_into().expect("fixed payload"))
}

fn read_u32(input: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(input[offset..offset + 4].try_into().expect("fixed payload"))
}

fn read_u64(input: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(input[offset..offset + 8].try_into().expect("fixed payload"))
}

fn write_u16(output: &mut [u8], offset: usize, value: u16) {
    output[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AccelerationIndexError {
    #[error("jwgp payload must contain exactly {expected} bytes, received {actual}")]
    PayloadSize { expected: usize, actual: usize },
    #[error("jwgp payload has an invalid magic value")]
    Magic,
    #[error("unsupported jwgp schema version {0}")]
    Version(u16),
    #[error("jwgp v1 header size must be 84, received {0}")]
    HeaderSize(u16),
    #[error("unsupported jwgp profile {0}")]
    Profile(u16),
    #[error("unsupported jwgp flags 0x{0:04x}")]
    Flags(u16),
    #[error(
        "jwgp Gray8 profile fields are invalid: predictor={predictor}, channels={channels}, bits={bits_per_sample}, reserved={reserved}"
    )]
    ProfileFields {
        predictor: u8,
        channels: u8,
        bits_per_sample: u8,
        reserved: u8,
    },
    #[error("jwgp profile has an empty image extent")]
    EmptyExtent,
    #[error("jwgp sample count mismatch: expected {expected}, received {actual}")]
    SampleCount { expected: u32, actual: u32 },
    #[error("jwgp token stream is empty")]
    EmptyTokenStream,
    #[error(
        "jwgp token offset mismatch: standard group metadata ends at {expected}, index names {actual}"
    )]
    TokenOffset { expected: u64, actual: u64 },
    #[error("jwgp standard group prefix exceeds the bound codestream")]
    GroupPrefixRange,
    #[error("jwgp standard group metadata differs from its indexed prefix tables at bit {bit}")]
    GroupPrefixMismatch { bit: u64 },
    #[error("jwgp token range {offset}+{length} exceeds the {codestream_bits}-bit codestream")]
    TokenRange {
        offset: u64,
        length: u64,
        codestream_bits: u64,
    },
    #[error("jwgp codestream length mismatch: expected {expected}, received {actual}")]
    CodestreamLength { expected: u64, actual: u64 },
    #[error("jwgp SHA-256 does not match the codestream")]
    CodestreamDigest,
    #[error("jwgp prefix entry {alphabet}[{index}] is invalid ({bit_len} bits, value {bits})")]
    PrefixEntry {
        alphabet: &'static str,
        index: usize,
        bit_len: u8,
        bits: u16,
    },
    #[error("jwgp raw prefix table is missing one of the required symbols 0..=9")]
    MissingRawSymbol,
    #[error("jwgp prefix code is oversubscribed")]
    OversubscribedPrefix,
    #[error("jwgp fixed prefix metadata could not be serialized")]
    PrefixSerialization,
    #[error(
        "jwgp prefix entry {alphabet}[{index}] is not canonical: expected {expected}, received {actual}"
    )]
    NonCanonicalPrefix {
        alphabet: &'static str,
        index: usize,
        expected: u16,
        actual: u16,
    },
    #[error("jwgp size arithmetic overflow")]
    SizeOverflow,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_tables() -> (
        [PrefixCodeEntry; RAW_SYMBOLS],
        [PrefixCodeEntry; LZ77_SYMBOLS],
    ) {
        // Complete 6-bit canonical alphabet, with the first twelve raw symbols and all LZ77
        // symbols present. Symbols unused by the fixed Gray8 profile remain zero.
        let mut raw = [PrefixCodeEntry::default(); RAW_SYMBOLS];
        let mut lz77 = [PrefixCodeEntry::default(); LZ77_SYMBOLS];
        for (code, entry) in raw.iter_mut().take(12).chain(&mut lz77).enumerate() {
            *entry = PrefixCodeEntry {
                bit_len: 6,
                bits: reverse_low_bits(code as u16, 6),
            };
        }
        (raw, lz77)
    }

    #[test]
    fn index_roundtrip_binds_the_exact_codestream() {
        let codestream = [0xff, 0x0a, 1, 2, 3, 4, 5, 6];
        let (raw, lz77) = valid_tables();
        let index = Gray8AccelerationIndex::new(&codestream, 2, 3, 17, 31, raw, lz77).unwrap();
        let payload = index.serialize();
        assert_eq!(payload.len(), PAYLOAD_SIZE);
        let parsed = Gray8AccelerationIndex::parse_bound(&payload, &codestream).unwrap();
        assert_eq!(parsed, index);
        assert_eq!(parsed.sample_count(), 6);

        let mut changed = codestream;
        changed[7] ^= 1;
        assert_eq!(
            parsed.verify_codestream(&changed).unwrap_err(),
            AccelerationIndexError::CodestreamDigest
        );
    }

    #[test]
    fn parser_rejects_noncanonical_prefix_and_out_of_bounds_tokens() {
        let codestream = [0xff, 0x0a, 1, 2, 3, 4, 5, 6];
        let (raw, lz77) = valid_tables();
        let index = Gray8AccelerationIndex::new(&codestream, 2, 3, 17, 31, raw, lz77).unwrap();
        let mut payload = index.serialize();
        payload[85] ^= 1;
        assert!(matches!(
            Gray8AccelerationIndex::parse(&payload),
            Err(AccelerationIndexError::NonCanonicalPrefix { .. })
        ));

        let mut payload = index.serialize();
        write_u64(&mut payload, 68, 1_000);
        assert!(matches!(
            Gray8AccelerationIndex::parse(&payload),
            Err(AccelerationIndexError::TokenRange { .. })
        ));
    }

    #[test]
    fn checked_in_group_prefix_matches_the_standard_codestream_bits() {
        let container = include_bytes!("../../../fixtures/gpu_gray8_lossless.jxl");
        let parsed = crate::parse(container, crate::ParseLimits::default()).unwrap();
        let payload = parsed
            .boxes_of_type(ACCELERATION_INDEX_BOX_TYPE)
            .next()
            .expect("fixture carries jwgp")
            .payload;
        let index = Gray8AccelerationIndex::parse_bound(payload, parsed.codestream()).unwrap();
        let prefix = write_expected_group_prefix(&index).unwrap();
        let group_start = index
            .token_bit_offset()
            .checked_sub(prefix.bit_len() as u64)
            .expect("prefix precedes tokens");
        index
            .validate_group_prefix(parsed.codestream(), group_start)
            .expect("standard group prefix exactly represents indexed tables");

        let mut changed = parsed.codestream().to_vec();
        let changed_bit = group_start + 9;
        changed[(changed_bit / 8) as usize] ^= 1 << (changed_bit % 8);
        let rebound = Gray8AccelerationIndex::new(
            &changed,
            index.width(),
            index.height(),
            index.token_bit_offset(),
            index.token_bit_len(),
            *index.raw_prefix(),
            *index.lz77_prefix(),
        )
        .unwrap();
        assert!(matches!(
            rebound.validate_group_prefix(&changed, group_start),
            Err(AccelerationIndexError::GroupPrefixMismatch { .. })
        ));
    }
}
