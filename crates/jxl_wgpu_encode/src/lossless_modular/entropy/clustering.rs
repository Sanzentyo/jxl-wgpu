//! Exact partition search over the five existing Modular entropy contexts.
//! Only bounded GPU histogram metadata is inspected; symbols remain on GPU.
use super::*;

pub(super) const CONTEXTS: usize = 5;
pub(super) type ContextMap = [u8; CONTEXTS];

#[derive(Clone)]
struct Candidate {
    histogram: AnsHistogram,
    config: hybrid::HybridConfig,
    cost: u128,
}

struct Partition {
    subsets: Vec<u8>,
    map: ContextMap,
    cost: u128,
}

pub(super) struct ClusteredCode {
    tables: Vec<Candidate>,
    pub(super) map: ContextMap,
    pub(super) estimated_bits_q20: u128,
}

impl ClusteredCode {
    /// Materialize reverse aliases only after the global coding choice is final.
    pub(super) fn compile(self) -> Result<Vec<hybrid::HybridCode>, EncodeError> {
        self.tables
            .into_iter()
            .map(|candidate| {
                Ok(hybrid::HybridCode {
                    config: candidate.config,
                    code: candidate.histogram.compile()?,
                })
            })
            .collect()
    }
}

pub(super) fn cluster(
    profiles: &[hybrid::HybridCounts],
    header_copies: u64,
    alphabet: AnsAlphabet,
) -> Result<ClusteredCode, EncodeError> {
    if header_copies == 0 {
        return Err(BackendError::Invariant("ANS codebook has no header").into());
    }
    // Cache the 31 possible unions, then enumerate all 52 set partitions. Labels
    // follow the first source context in each cluster; ties use fewer clusters,
    // then the lexicographically smaller map. No floating-point choices occur.
    let candidates = (1..1u8 << CONTEXTS)
        .map(|subset| {
            let mut best: Option<Candidate> = None;
            for profile in profiles {
                let mut merged = [0u64; ALPHABET];
                let mut extra_bits = 0;
                for (context, counts) in profile.counts.iter().enumerate() {
                    if subset & (1 << context) == 0 {
                        continue;
                    }
                    extra_bits += profile.extra_bits[context];
                    for (total, &count) in merged.iter_mut().zip(counts) {
                        *total = total
                            .checked_add(count)
                            .ok_or(BackendError::InvalidArtifact(
                                "clustered ANS histogram overflow",
                            ))?;
                    }
                }
                let code = AnsHistogram::from_counts(&merged, alphabet)?;
                let mut header = BitWriter::new();
                code.write_histogram(&mut header)?;
                profile.config.write(&mut header, alphabet)?;
                let cost = code.estimated_data_bits(&merged)?
                    + ((extra_bits + header.bit_len() as u128 * u128::from(header_copies)) << 20);
                if best.as_ref().is_none_or(|previous| {
                    (cost, profile.config) < (previous.cost, previous.config)
                }) {
                    best = Some(Candidate {
                        histogram: code,
                        config: profile.config,
                        cost,
                    });
                }
            }
            best.ok_or_else(|| BackendError::Invariant("ANS hybrid search is empty").into())
        })
        .collect::<Result<Vec<_>, EncodeError>>()?;
    let mut best = None;
    visit(
        (1 << CONTEXTS) - 1,
        &mut Vec::with_capacity(CONTEXTS),
        &candidates,
        header_copies,
        &mut best,
    );
    let best = best.ok_or(BackendError::Invariant("ANS partition search is empty"))?;
    let tables = best
        .subsets
        .iter()
        .map(|&subset| candidates[subset as usize - 1].clone())
        .collect();
    Ok(ClusteredCode {
        tables,
        map: best.map,
        estimated_bits_q20: best.cost,
    })
}

fn visit(
    remaining: u8,
    subsets: &mut Vec<u8>,
    candidates: &[Candidate],
    header_copies: u64,
    best: &mut Option<Partition>,
) {
    if remaining == 0 {
        let mut map = [0; CONTEXTS];
        for (cluster, &subset) in subsets.iter().enumerate() {
            for (context, target) in map.iter_mut().enumerate() {
                if subset & (1 << context) != 0 {
                    *target = cluster as u8;
                }
            }
        }
        let width = context_map_width(&map);
        let cost = subsets
            .iter()
            .map(|&subset| candidates[subset as usize - 1].cost)
            .sum::<u128>()
            + (((3 + CONTEXTS as u128 * u128::from(width)) * u128::from(header_copies)) << 20);
        if best.as_ref().is_none_or(|previous| {
            (cost, subsets.len(), map) < (previous.cost, previous.subsets.len(), previous.map)
        }) {
            *best = Some(Partition {
                subsets: subsets.clone(),
                map,
                cost,
            });
        }
        return;
    }
    let first = 1 << remaining.trailing_zeros();
    let mut subset = remaining;
    while subset != 0 {
        if subset & first != 0 {
            subsets.push(subset);
            visit(remaining ^ subset, subsets, candidates, header_copies, best);
            subsets.pop();
        }
        subset = (subset - 1) & remaining;
    }
}

fn context_map_width(map: &ContextMap) -> u8 {
    u8::BITS as u8 - map.iter().max().unwrap().leading_zeros() as u8
}

pub(super) fn write_context_map(
    writer: &mut BitWriter,
    map: &ContextMap,
) -> Result<(), EncodeError> {
    let width = context_map_width(map);
    writer.write_bits(1, 1)?; // simple map, including the implicit all-zero map
    writer.write_bits(u64::from(width), 2)?;
    // The four MA leaves are visited in reverse channel order, then LZ77 distance.
    for &cluster in map.iter().rev() {
        writer.write_bits(u64::from(cluster), width)?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests;
