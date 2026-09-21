/// GPU residual matching policy. Both modes emit standard JPEG XL LZ77 syntax.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum LosslessModularLz77 {
    /// Preserve the original zero-run coding and codestream bytes.
    #[default]
    ZeroRuns = 0,
    /// Greedy matching of arbitrary residual sequences, including overlapping matches.
    /// A three-symbol hash chain checks at most 32 candidates per token position;
    /// ties retain the nearest match. History resets at each group/channel.
    Greedy = 1,
}

impl LosslessModularLz77 {
    pub(super) fn hash_entries(self, pixels: u32) -> u32 {
        match self {
            Self::ZeroRuns => 0,
            Self::Greedy => pixels.next_power_of_two().min(1 << 16),
        }
    }

    // Groups contain at most 1024² samples, so these sums fit well inside u64.
    pub(super) fn scratch_words(self, pixels: u32) -> u64 {
        match self {
            Self::ZeroRuns => 0,
            Self::Greedy => 2 * u64::from(pixels) + u64::from(self.hash_entries(pixels)),
        }
    }
}
