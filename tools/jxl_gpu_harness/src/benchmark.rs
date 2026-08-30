use std::hint::black_box;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capture::{CaptureFile, encode_f32};
use crate::compare::AccuracyThreshold;
use crate::error::{Error, Result};
use crate::replay::{ReplayBackend, verify_capture};
use crate::report::{CaseReport, CaseStatus, TimingStatistics};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkOptions {
    pub warmup: u32,
    pub iterations: u32,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            warmup: 3,
            iterations: 20,
        }
    }
}

impl BenchmarkOptions {
    pub fn validate(self) -> Result<Self> {
        if self.iterations == 0 {
            return Err(Error::InvalidConfig(
                "benchmark iterations must be nonzero".into(),
            ));
        }
        if self.iterations > 1_000_000 || self.warmup > 1_000_000 {
            return Err(Error::InvalidConfig(
                "benchmark iteration count is unreasonably large".into(),
            ));
        }
        Ok(self)
    }
}

pub fn benchmark_capture(
    capture: &CaptureFile,
    backend: &mut dyn ReplayBackend,
    threshold: &AccuracyThreshold,
    options: BenchmarkOptions,
) -> Result<CaseReport> {
    let options = options.validate()?;
    let verification = verify_capture(capture, backend, threshold)?;
    if verification.status != CaseStatus::Passed {
        return Ok(verification);
    }

    for _ in 0..options.warmup {
        let output = backend.execute(capture)?;
        black_box(blake3::hash(&encode_f32(&output)));
    }

    let mut samples =
        Vec::with_capacity(usize::try_from(options.iterations).map_err(|_| Error::LengthOverflow)?);
    let mut last_hash = None;
    for _ in 0..options.iterations {
        let start = Instant::now();
        let output = backend.execute(capture)?;
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let hash = blake3::hash(&encode_f32(&output)).to_hex().to_string();
        black_box(&hash);
        last_hash = Some(hash);
        samples.push(elapsed);
    }
    let timing = summarize_timings(&samples)?;
    Ok(CaseReport {
        output_hash: last_hash,
        timing: Some(timing),
        ..verification
    })
}

pub fn summarize_timings(samples: &[u64]) -> Result<TimingStatistics> {
    if samples.is_empty() {
        return Err(Error::InvalidConfig(
            "cannot summarize an empty timing sample".into(),
        ));
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let minimum = sorted[0];
    let median = sorted[(sorted.len() - 1) / 2];
    let p95 = sorted[(sorted.len() - 1) * 95 / 100];
    let mean = sorted.iter().map(|&value| value as f64).sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / sorted.len() as f64;
    Ok(TimingStatistics {
        samples: u32::try_from(sorted.len()).unwrap_or(u32::MAX),
        minimum_ns: minimum,
        median_ns: median,
        p95_ns: p95,
        mean_ns: mean,
        standard_deviation_ns: variance.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_percentiles_are_deterministic() {
        let timing = summarize_timings(&[50, 10, 30, 20, 40]).unwrap();
        assert_eq!(timing.minimum_ns, 10);
        assert_eq!(timing.median_ns, 30);
        assert_eq!(timing.p95_ns, 40);
        assert_eq!(timing.mean_ns, 30.0);
    }

    #[test]
    fn rejects_empty_timing_sample() {
        assert!(summarize_timings(&[]).is_err());
    }
}
