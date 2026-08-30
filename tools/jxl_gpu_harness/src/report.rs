use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::codec::{
    CodecOperation, CpuReadbackMode, DeclaredExtent, GpuPixelFormat, OutputTarget, SizeClass,
    WorkloadExecutionModel, WorkloadSpec,
};
use crate::compare::{AccuracyMetrics, ThresholdEvaluation};
use crate::error::{Error, Result};

pub const RUN_REPORT_SCHEMA_VERSION: u16 = 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Passed,
    Failed,
    Unsupported,
    Incomplete,
    Unavailable,
    Error,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseReport {
    pub case_id: String,
    pub operation: String,
    pub backend: String,
    pub status: CaseStatus,
    pub output_hash: Option<String>,
    pub metrics: Option<AccuracyMetrics>,
    pub threshold: Option<ThresholdEvaluation>,
    pub timing: Option<TimingStatistics>,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TimingStatistics {
    pub samples: u32,
    pub minimum_ns: u64,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub mean_ns: f64,
    pub standard_deviation_ns: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterReport {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub pci_bus_id: String,
    pub subgroup_min_size: u32,
    pub subgroup_max_size: u32,
    pub features: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
}

/// Typed reason why a production GPU codec case did not execute successfully.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodecIssueKind {
    Unsupported,
    Incomplete,
    Unavailable,
    InvalidInput,
    Verification,
    Backend,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecIssue {
    pub kind: CodecIssueKind,
    pub component: String,
    pub code: String,
    pub detail: String,
}

impl CodecIssue {
    #[must_use]
    pub fn new(
        kind: CodecIssueKind,
        component: impl Into<String>,
        code: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            component: component.into(),
            code: code.into(),
            detail: detail.into(),
        }
    }

    #[must_use]
    pub const fn status(&self) -> CaseStatus {
        match self.kind {
            CodecIssueKind::Unsupported => CaseStatus::Unsupported,
            CodecIssueKind::Incomplete => CaseStatus::Incomplete,
            CodecIssueKind::Unavailable => CaseStatus::Unavailable,
            CodecIssueKind::Verification => CaseStatus::Failed,
            CodecIssueKind::InvalidInput | CodecIssueKind::Backend => CaseStatus::Error,
        }
    }
}

/// Result for one GPU-required decode, encode, or round-trip case.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodecCaseReport {
    pub case_id: String,
    pub path: String,
    pub operation: CodecOperation,
    pub workload: WorkloadSpec,
    pub output_target: OutputTarget,
    pub status: CaseStatus,
    pub adapter: Option<String>,
    pub pixel_format: GpuPixelFormat,
    pub size_class: SizeClass,
    pub declared_extent: Option<DeclaredExtent>,
    pub input_bytes: u64,
    /// Operation result bytes: encoded container bytes for encode, logical image bytes otherwise.
    pub output_bytes: u64,
    /// Sum of addressable bytes in decoded GPU image layouts.
    pub gpu_output_logical_bytes: u64,
    pub frame_count: u32,
    /// Codec queue submissions during measured operations; warmup is excluded.
    pub codec_submissions: u64,
    /// Blocking codec completion paths during measured operations.
    pub codec_completion_waits: u64,
    pub display_submissions: u64,
    pub display_completion_waits: u64,
    pub readback_submissions: u64,
    pub readback_completion_waits: u64,
    pub readback_logical_bytes: u64,
    pub readback_staging_bytes: u64,
    /// Number of decoded GPU frames transported by measured readback submissions.
    pub readback_source_frames: u64,
    /// Largest number of decoded GPU frames represented by one measured readback submission.
    pub readback_max_frames_per_submission: u64,
    pub readback_mode: Option<CpuReadbackMode>,
    /// `false` until several images or frames are encoded into one GPU submission.
    pub coalesced_gpu_batching: bool,
    pub output_hash: Option<String>,
    pub timing: Option<CodecTiming>,
    pub issue: Option<CodecIssue>,
    /// Opt-in external CPU comparator. This is never a production codec path.
    pub cpu_baseline: Option<CpuBaselineReport>,
}

impl CodecCaseReport {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        case_id: String,
        path: &Path,
        operation: CodecOperation,
        workload: WorkloadSpec,
        output_target: OutputTarget,
        pixel_format: GpuPixelFormat,
        size_class: SizeClass,
        declared_extent: Option<DeclaredExtent>,
        input_bytes: u64,
    ) -> Self {
        Self {
            case_id,
            path: path.display().to_string(),
            operation,
            workload,
            output_target,
            status: CaseStatus::Error,
            adapter: None,
            pixel_format,
            size_class,
            declared_extent,
            input_bytes,
            output_bytes: 0,
            gpu_output_logical_bytes: 0,
            frame_count: 0,
            codec_submissions: 0,
            codec_completion_waits: 0,
            display_submissions: 0,
            display_completion_waits: 0,
            readback_submissions: 0,
            readback_completion_waits: 0,
            readback_logical_bytes: 0,
            readback_staging_bytes: 0,
            readback_source_frames: 0,
            readback_max_frames_per_submission: 0,
            readback_mode: None,
            coalesced_gpu_batching: false,
            output_hash: None,
            timing: None,
            issue: None,
            cpu_baseline: None,
        }
    }
}

/// Outcome of an opt-in, development-only external CPU codec comparator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuBaselineStatus {
    Completed,
    VerificationFailed,
    Skipped,
    Unsupported,
    TimedOut,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuBaselineIssueKind {
    UnsupportedContract,
    ExecutableUnavailable,
    Timeout,
    ProcessExit,
    InvalidOutput,
    Io,
    NondeterministicOutput,
    VerificationMismatch,
    WorkerPanic,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuBaselineIssue {
    pub kind: CpuBaselineIssueKind,
    pub code: String,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuBaselineProcessModel {
    ExternalProcessPerOperation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuBaselineProvenance {
    pub implementation: String,
    pub executable: String,
    pub version: Option<String>,
    pub version_query_error: Option<String>,
    pub process_model: CpuBaselineProcessModel,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuBaselineConfiguration {
    pub warmup: u32,
    pub iterations: u32,
    pub timeout_ms_per_process: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuBaselineColdTiming {
    pub elapsed_ns: u64,
    pub includes_process_spawn: bool,
    pub includes_output_file_io_and_hash: bool,
    pub input_bytes: u64,
    pub decoded_bytes: u64,
    pub width: u32,
    pub height: u32,
    pub output_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpuBaselineWarmRepetition {
    pub samples: u32,
    pub minimum_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub mean_ns: f64,
    pub wall_ns: u64,
    pub operations_per_second: f64,
    pub parallelism: u32,
    pub workload_execution_model: WorkloadExecutionModel,
    pub process_model: CpuBaselineProcessModel,
    pub total_input_bytes: u64,
    pub total_decoded_bytes: u64,
    pub decoded_bytes_per_operation: u64,
    pub width: u32,
    pub height: u32,
    pub output_hash: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpuBaselineReport {
    pub status: CpuBaselineStatus,
    pub provenance: CpuBaselineProvenance,
    pub configuration: CpuBaselineConfiguration,
    pub cold: Option<CpuBaselineColdTiming>,
    pub warm_repetition: Option<CpuBaselineWarmRepetition>,
    /// Exact U8 pixel equality with the GPU decode + CPU-readback result.
    pub verified: Option<bool>,
    pub issue: Option<CpuBaselineIssue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodecTiming {
    pub operation_latency: TimingStatistics,
    pub workload: WorkloadTiming,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadTiming {
    pub operations: u64,
    pub parallelism: u32,
    pub execution_model: WorkloadExecutionModel,
    pub wall_ns: u64,
    pub operations_per_second: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunReport {
    pub schema_version: u16,
    pub command: String,
    pub generated_unix_ms: u128,
    pub package_version: String,
    pub target_os: String,
    pub target_arch: String,
    pub adapters: Vec<AdapterReport>,
    pub cases: Vec<CaseReport>,
    /// Present only for the GPU-required `codec` command; omitted by synthetic commands.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub codec_cases: Vec<CodecCaseReport>,
}

impl RunReport {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            schema_version: RUN_REPORT_SCHEMA_VERSION,
            command: command.into(),
            generated_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis()),
            package_version: env!("CARGO_PKG_VERSION").into(),
            target_os: std::env::consts::OS.into(),
            target_arch: std::env::consts::ARCH.into(),
            adapters: Vec::new(),
            cases: Vec::new(),
            codec_cases: Vec::new(),
        }
    }

    pub fn passed(&self) -> bool {
        let statuses = self
            .cases
            .iter()
            .map(|case| &case.status)
            .chain(self.codec_cases.iter().map(|case| &case.status));
        let statuses = statuses.collect::<Vec<_>>();
        statuses.iter().any(|status| **status == CaseStatus::Passed)
            && statuses.iter().all(|status| **status == CaseStatus::Passed)
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes).map_err(|source| Error::io(path, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_case_fails_run() {
        let mut report = RunReport::new("verify");
        report.cases.push(CaseReport {
            case_id: "case".into(),
            operation: "copy".into(),
            backend: "reference".into(),
            status: CaseStatus::Failed,
            output_hash: None,
            metrics: None,
            threshold: None,
            timing: None,
            message: None,
        });
        assert!(!report.passed());
        assert!(
            serde_json::to_string(&report)
                .unwrap()
                .contains("schema_version")
        );
    }

    #[test]
    fn empty_or_all_unsupported_run_does_not_pass() {
        let mut report = RunReport::new("verify");
        assert!(!report.passed());
        report.cases.push(CaseReport {
            case_id: "unsupported".into(),
            operation: "future_operation".into(),
            backend: "wgpu".into(),
            status: CaseStatus::Unsupported,
            output_hash: None,
            metrics: None,
            threshold: None,
            timing: None,
            message: Some("unsupported".into()),
        });
        assert!(!report.passed());
    }

    #[test]
    fn unsupported_and_incomplete_codec_cases_never_pass() {
        for (kind, status) in [
            (CodecIssueKind::Unsupported, CaseStatus::Unsupported),
            (CodecIssueKind::Incomplete, CaseStatus::Incomplete),
        ] {
            let issue = CodecIssue::new(kind, "codec", "profile", "not implemented");
            assert_eq!(issue.status(), status);
            let mut report = RunReport::new("codec");
            let mut case = CodecCaseReport::new(
                "case".into(),
                Path::new("case.jxl"),
                CodecOperation::Decode,
                WorkloadSpec::default(),
                OutputTarget::GpuResident,
                GpuPixelFormat::Rgba8,
                SizeClass::Small,
                None,
                10,
            );
            case.status = issue.status();
            case.issue = Some(issue);
            report.codec_cases.push(case);
            assert!(!report.passed());
        }
    }

    #[test]
    fn checked_in_json_schema_matches_report_version() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/run.schema.json")).unwrap();
        assert_eq!(
            schema["properties"]["schema_version"]["const"],
            RUN_REPORT_SCHEMA_VERSION
        );
        assert!(schema["$defs"]["cpu_baseline"].is_object());
        assert!(
            schema["$defs"]["codec_case"]["required"]
                .as_array()
                .unwrap()
                .iter()
                .any(|field| field == "cpu_baseline")
        );
    }

    #[test]
    fn codec_report_serializes_path_and_scheduling_telemetry() {
        let case = CodecCaseReport::new(
            "case".into(),
            Path::new("case.jxl"),
            CodecOperation::Decode,
            WorkloadSpec::default(),
            OutputTarget::CpuReadback,
            GpuPixelFormat::U8,
            SizeClass::Small,
            None,
            10,
        );
        let value = serde_json::to_value(case).unwrap();
        assert!(value["readback_mode"].is_null());
        assert_eq!(value["coalesced_gpu_batching"], false);
        assert_eq!(value["codec_completion_waits"], 0);
        assert_eq!(value["readback_staging_bytes"], 0);
        assert_eq!(value["readback_source_frames"], 0);
        assert_eq!(value["readback_max_frames_per_submission"], 0);
        assert!(value["cpu_baseline"].is_null());
    }

    #[test]
    fn schema_v4_distinguishes_aggregate_readback_from_codec_batching() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/run.schema.json")).unwrap();
        assert_eq!(schema["properties"]["schema_version"]["const"], 4);
        assert_eq!(
            schema["$defs"]["codec_case"]["properties"]["coalesced_gpu_batching"]["const"],
            false
        );
        let modes =
            schema["$defs"]["codec_case"]["properties"]["readback_mode"]["oneOf"][1]["enum"]
                .as_array()
                .unwrap();
        assert!(modes.iter().any(|mode| mode == "staged_copy"));
        assert!(modes.iter().any(|mode| mode == "direct_map"));
        assert!(modes.iter().any(|mode| mode == "aggregate_staged_copy"));
        for field in [
            "readback_source_frames",
            "readback_max_frames_per_submission",
        ] {
            assert!(
                schema["$defs"]["codec_case"]["required"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|required| required == field)
            );
        }
    }
}
