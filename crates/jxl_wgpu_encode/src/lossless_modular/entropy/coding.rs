//! One checked wire/symbol domain shared by selection and GPU lowering.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CodingPlan {
    pub(super) alphabet: AnsAlphabet,
    pub(super) min_symbol: usize,
    pub(super) length: length::LengthCoding,
}

impl CodingPlan {
    pub(super) const fn canonical() -> Self {
        Self {
            alphabet: AnsAlphabet::MAX,
            min_symbol: 224,
            length: length::LengthCoding::canonical(),
        }
    }

    pub(super) fn new(
        min_symbol: usize,
        length: length::LengthCoding,
    ) -> Result<Self, EncodeError> {
        let symbols = min_symbol
            .checked_add(length.symbols())
            .ok_or(BackendError::Invariant("ANS symbol domain overflow"))?;
        if min_symbol < 8 {
            return Err(BackendError::Invariant("LZ77 threshold is below its wire domain").into());
        }
        Ok(Self {
            alphabet: AnsAlphabet::for_symbols(symbols)?,
            min_symbol,
            length,
        })
    }

    pub(super) fn candidates() -> Vec<Self> {
        // Between raw-configuration boundaries, increasing the threshold only inserts
        // empty histogram bins. The wire-special 224 is the sole shorter-header exception.
        let mut thresholds: Vec<_> = hybrid::HybridConfig::candidates()
            .iter()
            .map(|config| config.symbols(32))
            .chain([224])
            .collect();
        thresholds.sort_unstable();
        thresholds.dedup();
        thresholds
            .into_iter()
            .flat_map(|threshold| {
                length::LengthCoding::candidates()
                    .into_iter()
                    .filter_map(move |length| Self::new(threshold, length).ok())
            })
            .collect()
    }

    pub(super) fn supports(self, config: hybrid::HybridConfig) -> bool {
        config.symbols(32) <= self.min_symbol
    }

    pub(super) fn write_lz77(self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        writer.write_bits(1, 1)?;
        if self.min_symbol == 224 {
            writer.write_bits(0, 2)?;
        } else {
            writer.write_bits(3, 2)?;
            writer.write_bits((self.min_symbol - 8) as u64, 15)?;
        }
        writer.write_bits(0b1010, 4)?; // min_length = 7
        self.length.write(writer)
    }
}

#[cfg(test)]
mod tests;
