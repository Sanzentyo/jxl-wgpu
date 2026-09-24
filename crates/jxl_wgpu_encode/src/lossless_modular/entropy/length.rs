//! Global LZ77 length configuration. Canonical event metadata stays unchanged.
use super::*;

const LENGTH_SYMBOLS: usize = ALPHABET - hybrid::RAW_ALPHABET;
// Complete groups have at most 1024² samples. Canonical length tokens 0..31
// cover value = copied - 7 through 20 bits; token 32 is outside the ANS contract.
const LENGTH_BITS: u8 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LengthCoding(hybrid::HybridConfig);

pub(super) struct LengthHistograms {
    pub(super) counts: [[u64; LENGTH_SYMBOLS]; 4],
    pub(super) extra_bits: [u128; 4],
}

impl LengthCoding {
    pub(super) const fn canonical() -> Self {
        Self(hybrid::HybridConfig::canonical_length())
    }

    pub(super) fn candidates() -> Vec<Self> {
        hybrid::HybridConfig::for_alphabet(LENGTH_BITS, LENGTH_SYMBOLS)
            .into_iter()
            .map(Self)
            .collect()
    }

    pub(super) fn packed(self) -> u32 {
        self.0.packed()
    }

    pub(super) fn write(self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        // JPEG XL length configurations always declare log_alphabet_size = 8,
        // independently of the eventual entropy distribution's alphabet size.
        self.0.write(writer)
    }

    pub(super) fn histograms(
        self,
        canonical: &[[u64; LZ77_SYMBOLS]; 4],
    ) -> Result<LengthHistograms, EncodeError> {
        let split =
            self.0
                .split_only()
                .filter(|&split| split <= 4)
                .ok_or(BackendError::Invariant(
                    "length configuration cannot be derived from canonical histograms",
                ))?;
        let mut result = LengthHistograms {
            counts: [[0; LENGTH_SYMBOLS]; 4],
            extra_bits: [0; 4],
        };
        for (channel, histogram) in canonical.iter().enumerate() {
            if histogram[LENGTH_SYMBOLS..].iter().any(|&count| count != 0) {
                return Err(
                    BackendError::InvalidArtifact("ANS LZ77 alphabet exceeds 256 symbols").into(),
                );
            }
            for (token, &count) in histogram[..LENGTH_SYMBOLS].iter().enumerate() {
                // A direct canonical bin identifies one value. Remaining bins identify
                // one exponent, sufficient for all full-range 32-symbol configurations.
                let value = if token < 16 {
                    token as u32
                } else {
                    1 << (token - 12)
                };
                let (selected, bits) = if value < 1 << split {
                    (value, 0)
                } else {
                    let exponent = value.ilog2();
                    ((1 << split) + exponent - u32::from(split), exponent)
                };
                let target = &mut result.counts[channel][selected as usize];
                *target = target
                    .checked_add(count)
                    .ok_or(BackendError::InvalidArtifact("length histogram overflow"))?;
                result.extra_bits[channel] += u128::from(count) * u128::from(bits);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
