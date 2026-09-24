//! Bounded hybrid-uint policy, GPU histogram admission, and validated frame metadata.
use super::*;

pub(super) const RAW_ALPHABET: usize = 224;
pub(in super::super) const PROFILES: usize = 37;
pub(in super::super) const PROFILE_BYTES: u64 =
    4 * (2 + PROFILES * clustering::CONTEXTS * RAW_ALPHABET) as u64;
type RawCounts = [[u64; RAW_ALPHABET]; clustering::CONTEXTS];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct HybridConfig {
    split: u8,
    msb: u8,
    lsb: u8,
}

impl HybridConfig {
    pub(super) fn candidates() -> &'static [Self] {
        static CONFIGS: std::sync::OnceLock<Vec<HybridConfig>> = std::sync::OnceLock::new();
        CONFIGS.get_or_init(|| {
            let mut configs = Vec::new();
            for split in 0..8 {
                for msb in 0..=split {
                    for lsb in 0..=split - msb {
                        let config = Self { split, msb, lsb };
                        // Every u32 must fit below the reserved LZ77 alphabet.
                        if config.max_token() < RAW_ALPHABET {
                            configs.push(config);
                        }
                    }
                }
            }
            assert_eq!(configs.len(), PROFILES);
            configs
        })
    }

    fn max_token(self) -> usize {
        (1 << self.split) + ((32 - usize::from(self.split)) << (self.msb + self.lsb)) - 1
    }

    pub(super) fn packed(self) -> u32 {
        u32::from(self.split) | (u32::from(self.msb) << 8) | (u32::from(self.lsb) << 16)
    }

    pub(super) fn write(self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        writer.write_bits(u64::from(self.split), 4)?;
        writer.write_bits(u64::from(self.msb), (8 - self.split.leading_zeros()) as u8)?;
        writer.write_bits(
            u64::from(self.lsb),
            (8 - (self.split - self.msb).leading_zeros()) as u8,
        )?;
        Ok(())
    }

    fn canonical_token(self, token: usize) -> usize {
        if token < 1 << self.split {
            (usize::BITS - token.leading_zeros()) as usize
        } else {
            ((token - (1 << self.split)) >> (self.msb + self.lsb)) + usize::from(self.split) + 1
        }
    }

    fn extra_bits(self, canonical: &[u64; RAW_SYMBOLS]) -> u128 {
        canonical
            .iter()
            .enumerate()
            .map(|(token, &count)| {
                let bits = if token <= usize::from(self.split) {
                    0
                } else {
                    token - 1 - usize::from(self.msb + self.lsb)
                };
                u128::from(count) * bits as u128
            })
            .sum()
    }
}

#[derive(Clone)]
pub(super) struct HybridCode {
    pub(super) config: HybridConfig,
    pub(super) code: AnsCode,
}

pub(super) struct HybridCounts {
    pub(super) config: HybridConfig,
    pub(super) counts: [[u64; ALPHABET]; clustering::CONTEXTS],
    pub(super) extra_bits: [u128; clustering::CONTEXTS],
}

pub(in super::super) struct FrameHistograms {
    pub(in super::super) raw: [[u64; RAW_SYMBOLS]; 4],
    pub(in super::super) lz77: [[u64; LZ77_SYMBOLS]; 4],
    pub(in super::super) distance: [u64; RAW_SYMBOLS],
    profiles: Vec<RawCounts>,
}

impl Default for FrameHistograms {
    fn default() -> Self {
        Self {
            raw: [[0; RAW_SYMBOLS]; 4],
            lz77: [[0; LZ77_SYMBOLS]; 4],
            distance: [0; RAW_SYMBOLS],
            profiles: Vec::new(),
        }
    }
}

impl FrameHistograms {
    fn canonical_raw(
        &self,
        mode: LosslessModularLz77,
    ) -> Result<[[u64; RAW_SYMBOLS]; clustering::CONTEXTS], EncodeError> {
        let mut counts = [[0; RAW_SYMBOLS]; clustering::CONTEXTS];
        if mode == LosslessModularLz77::ZeroRuns {
            if self.distance.iter().any(|&count| count != 0) {
                return Err(BackendError::InvalidArtifact(
                    "zero-run ANS contains explicit distances",
                )
                .into());
            }
            counts[0][1] = self.lz77.iter().flatten().try_fold(0u64, |sum, &count| {
                sum.checked_add(count).ok_or(BackendError::InvalidArtifact(
                    "ANS distance histogram overflow",
                ))
            })?;
        } else {
            counts[0] = self.distance;
        }
        counts[1..].copy_from_slice(&self.raw);
        Ok(counts)
    }

    /// Accept only completed, bounded profiles agreeing with independently validated
    /// canonical event histograms. No samples or residuals are recoded on the host.
    pub(in super::super) fn read_profiles(
        &mut self,
        mode: LosslessModularLz77,
        offset: u64,
        dispatches: usize,
        bytes: &[u8],
    ) -> Result<(), EncodeError> {
        let start = usize::try_from(offset)
            .map_err(|_| BackendError::InvalidArtifact("hybrid histogram offset overflow"))?;
        let end =
            start
                .checked_add(PROFILE_BYTES as usize)
                .ok_or(BackendError::InvalidArtifact(
                    "hybrid histogram range overflow",
                ))?;
        let data = bytes
            .get(start..end)
            .ok_or(BackendError::InvalidArtifact("truncated hybrid histograms"))?;
        let word =
            |index: usize| u32::from_le_bytes(data[index * 4..index * 4 + 4].try_into().unwrap());
        if u64::from(word(0)) != (dispatches * PROFILES) as u64 || word(1) != 0 {
            return Err(BackendError::InvalidArtifact("incomplete GPU hybrid histograms").into());
        }
        let canonical = self.canonical_raw(mode)?;
        let mut profiles = Vec::with_capacity(PROFILES);
        for (index, &config) in HybridConfig::candidates().iter().enumerate() {
            let mut profile = [[0; RAW_ALPHABET]; clustering::CONTEXTS];
            for (context, bins) in profile.iter_mut().enumerate() {
                let mut coarsened = [0u64; RAW_SYMBOLS];
                for (token, count) in bins.iter_mut().enumerate() {
                    *count = u64::from(word(
                        2 + (index * clustering::CONTEXTS + context) * RAW_ALPHABET + token,
                    ));
                    if token <= config.max_token() {
                        coarsened[config.canonical_token(token)] += *count;
                    } else if *count != 0 {
                        return Err(BackendError::InvalidArtifact(
                            "hybrid token exceeds u32 alphabet",
                        )
                        .into());
                    }
                }
                if coarsened != canonical[context] {
                    return Err(BackendError::InvalidArtifact(
                        "hybrid histogram disagrees with validated events",
                    )
                    .into());
                }
            }
            profiles.push(profile);
        }
        self.profiles = profiles;
        Ok(())
    }

    pub(in super::super) fn accumulate(&mut self, batch: Self) -> Result<(), EncodeError> {
        fn add<'a>(
            target: impl Iterator<Item = &'a mut u64>,
            source: impl Iterator<Item = u64>,
        ) -> Result<(), EncodeError> {
            for (total, count) in target.zip(source) {
                *total = total
                    .checked_add(count)
                    .ok_or(BackendError::InvalidArtifact("frame histogram overflow"))?;
            }
            Ok(())
        }
        add(
            self.raw.iter_mut().flatten(),
            batch.raw.into_iter().flatten(),
        )?;
        add(
            self.lz77.iter_mut().flatten(),
            batch.lz77.into_iter().flatten(),
        )?;
        add(self.distance.iter_mut(), batch.distance.into_iter())?;
        if self.profiles.is_empty() {
            self.profiles = batch.profiles;
        } else {
            if self.profiles.len() != batch.profiles.len() {
                return Err(BackendError::Invariant(
                    "hybrid profile inventory changed between batches",
                )
                .into());
            }
            add(
                self.profiles.iter_mut().flatten().flatten(),
                batch.profiles.into_iter().flatten().flatten(),
            )?;
        }
        Ok(())
    }

    pub(super) fn candidates(
        &self,
        mode: LosslessModularLz77,
    ) -> Result<Vec<HybridCounts>, EncodeError> {
        if self.profiles.len() != PROFILES {
            return Err(BackendError::InvalidArtifact(
                "ANS requires validated GPU hybrid profiles",
            )
            .into());
        }
        if self.lz77.iter().any(|counts| counts[32] != 0) {
            return Err(
                BackendError::InvalidArtifact("ANS LZ77 alphabet exceeds 256 symbols").into(),
            );
        }
        let canonical = self.canonical_raw(mode)?;
        Ok(HybridConfig::candidates()
            .iter()
            .zip(&self.profiles)
            .map(|(&config, profile)| {
                let mut counts = [[0; ALPHABET]; clustering::CONTEXTS];
                for (target, source) in counts.iter_mut().zip(profile) {
                    target[..RAW_ALPHABET].copy_from_slice(source);
                }
                for (target, source) in counts[1..].iter_mut().zip(&self.lz77) {
                    target[RAW_ALPHABET..].copy_from_slice(&source[..32]);
                }
                HybridCounts {
                    config,
                    counts,
                    extra_bits: canonical.map(|counts| config.extra_bits(&counts)),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
