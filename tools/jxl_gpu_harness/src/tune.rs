use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::benchmark::{BenchmarkOptions, benchmark_capture};
use crate::capture::CaptureFile;
use crate::compare::AccuracyThreshold;
use crate::error::{Error, Result};
use crate::replay::{BackendKind, create_backend};
use crate::report::{CaseStatus, TimingStatistics};

pub const TUNING_PROFILE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningProfile {
    pub schema_version: u16,
    pub entries: BTreeMap<String, TunedCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TunedCase {
    pub operation: String,
    pub selected_backend: Option<BackendKind>,
    pub candidates: Vec<TuningCandidateResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningCandidateResult {
    pub backend: BackendKind,
    pub passed: bool,
    pub timing: Option<TimingStatistics>,
    pub message: Option<String>,
}

impl TuningProfile {
    pub fn tune(
        captures: &[CaptureFile],
        candidates: &[BackendKind],
        threshold_for: impl Fn(&CaptureFile) -> AccuracyThreshold,
        benchmark: BenchmarkOptions,
    ) -> Result<Self> {
        if candidates.is_empty() {
            return Err(Error::InvalidConfig(
                "tuning requires at least one backend candidate".into(),
            ));
        }
        let mut entries = BTreeMap::new();
        for capture in captures {
            let threshold = threshold_for(capture);
            let mut results = Vec::new();
            for &candidate in candidates {
                let result = match create_backend(candidate) {
                    Ok(mut backend) => {
                        match benchmark_capture(capture, backend.as_mut(), &threshold, benchmark) {
                            Ok(report) => TuningCandidateResult {
                                backend: candidate,
                                passed: report.status == CaseStatus::Passed,
                                timing: report.timing,
                                message: report.message,
                            },
                            Err(error) => TuningCandidateResult {
                                backend: candidate,
                                passed: false,
                                timing: None,
                                message: Some(error.to_string()),
                            },
                        }
                    }
                    Err(error) => TuningCandidateResult {
                        backend: candidate,
                        passed: false,
                        timing: None,
                        message: Some(error.to_string()),
                    },
                };
                results.push(result);
            }
            let selected_backend = results
                .iter()
                .filter(|result| result.passed)
                .filter_map(|result| {
                    result
                        .timing
                        .as_ref()
                        .map(|timing| (result.backend, timing.median_ns))
                })
                .min_by_key(|(_, median)| *median)
                .map(|(backend, _)| backend);
            entries.insert(
                capture.metadata.case_id.clone(),
                TunedCase {
                    operation: capture.metadata.operation.kind.as_str().into(),
                    selected_backend,
                    candidates: results,
                },
            );
        }
        Ok(Self {
            schema_version: TUNING_PROFILE_SCHEMA_VERSION,
            entries,
        })
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_vec_pretty(self)?)
            .map_err(|source| Error::io(path, source))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::capture::{OperationKind, PrecisionMode};
    use crate::config::SyntheticCaseConfig;
    use crate::synthetic::generate_case;

    use super::*;

    #[test]
    fn reference_candidate_is_selected() {
        let capture = generate_case(&SyntheticCaseConfig {
            name: "copy".into(),
            operation: OperationKind::Copy,
            width: 4,
            height: 4,
            channels: 1,
            seed: 9,
            precision: PrecisionMode::Exact,
            parameters: BTreeMap::new(),
        })
        .unwrap();
        let profile = TuningProfile::tune(
            &[capture],
            &[BackendKind::Reference],
            |_| AccuracyThreshold {
                require_exact: true,
                ..AccuracyThreshold::default()
            },
            BenchmarkOptions {
                warmup: 0,
                iterations: 1,
            },
        )
        .unwrap();
        assert_eq!(
            profile.entries["copy"].selected_backend,
            Some(BackendKind::Reference)
        );
    }
}
