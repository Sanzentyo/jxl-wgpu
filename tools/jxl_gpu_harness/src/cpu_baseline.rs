//! Opt-in `djxl` CPU decode comparator for development measurements.
//!
//! This module is intentionally owned by the harness. Production codec crates never depend on,
//! spawn, or fall back to a CPU codec. Every measured operation below is a fresh external process;
//! the report makes that process model explicit.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::benchmark::summarize_timings;
use crate::codec::{
    CodecOperation, CodecRunOptions, GpuPixelFormat, OutputTarget, WorkloadKind, WorkloadSpec,
};
use crate::error::{Error, Result};
use crate::report::{
    CaseStatus, CodecCaseReport, CpuBaselineColdTiming, CpuBaselineConfiguration, CpuBaselineIssue,
    CpuBaselineIssueKind, CpuBaselineProcessModel, CpuBaselineProvenance, CpuBaselineReport,
    CpuBaselineStatus, CpuBaselineWarmRepetition,
};

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 3_600_000;
const MAX_DIAGNOSTIC_BYTES: usize = 16 * 1024;
const MAX_PGM_BYTES: u64 = 512 * 1024 * 1024;
const PROCESS_POLL_INTERVAL: Duration = Duration::from_micros(200);

static NEXT_SCRATCH_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct DjxlBaselineOptions {
    pub executable: PathBuf,
    pub workload: WorkloadSpec,
    pub timeout_per_process: Duration,
}

impl DjxlBaselineOptions {
    pub fn new(
        executable: PathBuf,
        mut workload: WorkloadSpec,
        warmup: Option<u32>,
        iterations: Option<u32>,
        timeout_ms: Option<u64>,
    ) -> Result<Self> {
        if let Some(warmup) = warmup {
            workload.warmup = warmup;
        }
        if let Some(iterations) = iterations {
            workload.iterations = iterations;
        }
        let workload = workload.validate()?;
        let timeout_ms = timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
        if timeout_ms == 0 || timeout_ms > MAX_TIMEOUT_MS {
            return Err(Error::InvalidConfig(format!(
                "cpu-baseline-timeout-ms must be in 1..={MAX_TIMEOUT_MS}"
            )));
        }
        Ok(Self {
            executable,
            workload,
            timeout_per_process: Duration::from_millis(timeout_ms),
        })
    }

    fn empty_report(&self) -> CpuBaselineReport {
        CpuBaselineReport {
            status: CpuBaselineStatus::Error,
            provenance: CpuBaselineProvenance {
                implementation: "libjxl_djxl".into(),
                executable: self.executable.display().to_string(),
                version: None,
                version_query_error: None,
                process_model: CpuBaselineProcessModel::ExternalProcessPerOperation,
            },
            configuration: CpuBaselineConfiguration {
                warmup: self.workload.warmup,
                iterations: self.workload.iterations,
                timeout_ms_per_process: u64::try_from(self.timeout_per_process.as_millis())
                    .unwrap_or(u64::MAX),
            },
            cold: None,
            warm_repetition: None,
            verified: None,
            issue: None,
        }
    }
}

/// Runs an external `djxl` comparator for a single GPU codec case.
///
/// Only decode + U8 + explicit CPU readback is byte-comparable today. Other requests return a
/// typed `unsupported` baseline record without starting the executable.
#[must_use]
pub fn run_djxl_baseline(
    input: &Path,
    gpu_options: &CodecRunOptions,
    gpu_report: &CodecCaseReport,
    options: &DjxlBaselineOptions,
) -> CpuBaselineReport {
    let mut report = options.empty_report();
    if let Some(issue) = unsupported_contract(gpu_options) {
        report.status = CpuBaselineStatus::Unsupported;
        report.issue = Some(issue);
        return report;
    }

    let scratch = match ScratchDirectory::create() {
        Ok(scratch) => scratch,
        Err(error) => {
            apply_failure(&mut report, BaselineFailure::Io(error.to_string()));
            return report;
        }
    };
    let runner = DjxlRunner {
        executable: &options.executable,
        input,
        scratch: &scratch,
        timeout: options.timeout_per_process,
        next_output: AtomicU64::new(0),
    };

    let cold = match runner.decode() {
        Ok(cold) => cold,
        Err(failure) => {
            apply_failure(&mut report, failure);
            return report;
        }
    };
    report.cold = Some(cold.as_report());

    let warm = match execute_warm_repetition(&runner, options.workload) {
        Ok(warm) => warm,
        Err(failure) => {
            query_version(&runner, &mut report);
            apply_failure(&mut report, failure);
            return report;
        }
    };
    if cold.output_hash != warm.output_hash
        || cold.decoded_bytes != warm.decoded_bytes_per_operation
        || cold.width != warm.width
        || cold.height != warm.height
    {
        query_version(&runner, &mut report);
        apply_failure(
            &mut report,
            BaselineFailure::NondeterministicOutput(
                "cold and warm-repetition djxl outputs differ".into(),
            ),
        );
        return report;
    }
    report.warm_repetition = Some(warm);
    query_version(&runner, &mut report);

    report.verified = verify_gpu_readback(gpu_report, report.warm_repetition.as_ref());
    if report.verified == Some(false) {
        report.status = CpuBaselineStatus::VerificationFailed;
        report.issue = Some(CpuBaselineIssue {
            kind: CpuBaselineIssueKind::VerificationMismatch,
            code: "gpu_cpu_pixel_mismatch".into(),
            detail: "djxl U8 pixels differ from the GPU decode + CPU-readback bytes".into(),
        });
    } else {
        report.status = CpuBaselineStatus::Completed;
    }
    report
}

fn unsupported_contract(options: &CodecRunOptions) -> Option<CpuBaselineIssue> {
    let detail = if options.operation != CodecOperation::Decode {
        Some("the djxl comparator only measures decode")
    } else if options.output_target != OutputTarget::CpuReadback {
        Some("byte verification requires --output-target cpu-readback")
    } else if options.format != GpuPixelFormat::U8 {
        Some("byte verification currently requires the normalized Gray8 U8 output format")
    } else if options.workload.kind == WorkloadKind::Animation {
        Some("the external PGM comparator does not model an animation session")
    } else {
        None
    };
    detail.map(|detail| CpuBaselineIssue {
        kind: CpuBaselineIssueKind::UnsupportedContract,
        code: "comparison_contract".into(),
        detail: detail.into(),
    })
}

fn verify_gpu_readback(
    gpu_report: &CodecCaseReport,
    baseline: Option<&CpuBaselineWarmRepetition>,
) -> Option<bool> {
    if gpu_report.status != CaseStatus::Passed {
        return None;
    }
    let baseline = baseline?;
    let gpu_hash = gpu_report.output_hash.as_deref()?;
    let operations = gpu_report.timing.as_ref()?.workload.operations;
    if operations == 0 || !gpu_report.readback_logical_bytes.is_multiple_of(operations) {
        return Some(false);
    }
    let gpu_bytes_per_operation = gpu_report.readback_logical_bytes / operations;
    Some(
        gpu_hash == baseline.output_hash
            && gpu_bytes_per_operation == baseline.decoded_bytes_per_operation,
    )
}

fn query_version(runner: &DjxlRunner<'_>, report: &mut CpuBaselineReport) {
    match runner.version() {
        Ok(version) => report.provenance.version = Some(version),
        Err(failure) => report.provenance.version_query_error = Some(failure.detail()),
    }
}

fn apply_failure(report: &mut CpuBaselineReport, failure: BaselineFailure) {
    let (status, kind, code) = match &failure {
        BaselineFailure::Unavailable(_) => (
            CpuBaselineStatus::Skipped,
            CpuBaselineIssueKind::ExecutableUnavailable,
            "executable_unavailable",
        ),
        BaselineFailure::Timeout(_) => (
            CpuBaselineStatus::TimedOut,
            CpuBaselineIssueKind::Timeout,
            "process_timeout",
        ),
        BaselineFailure::ProcessExit(_) => (
            CpuBaselineStatus::Error,
            CpuBaselineIssueKind::ProcessExit,
            "process_exit",
        ),
        BaselineFailure::InvalidOutput(_) => (
            CpuBaselineStatus::Error,
            CpuBaselineIssueKind::InvalidOutput,
            "invalid_pgm",
        ),
        BaselineFailure::Io(_) => (CpuBaselineStatus::Error, CpuBaselineIssueKind::Io, "io"),
        BaselineFailure::NondeterministicOutput(_) => (
            CpuBaselineStatus::Error,
            CpuBaselineIssueKind::NondeterministicOutput,
            "nondeterministic_output",
        ),
        BaselineFailure::WorkerPanic(_) => (
            CpuBaselineStatus::Error,
            CpuBaselineIssueKind::WorkerPanic,
            "worker_panic",
        ),
    };
    report.status = status;
    report.issue = Some(CpuBaselineIssue {
        kind,
        code: code.into(),
        detail: failure.detail(),
    });
}

struct DjxlRunner<'a> {
    executable: &'a Path,
    input: &'a Path,
    scratch: &'a ScratchDirectory,
    timeout: Duration,
    next_output: AtomicU64,
}

impl DjxlRunner<'_> {
    fn decode(&self) -> std::result::Result<DecodeInvocation, BaselineFailure> {
        let id = self.next_output.fetch_add(1, Ordering::Relaxed);
        let output_path = self.scratch.path.join(format!("decode-{id}.pgm"));
        let output_guard = OutputFileGuard(&output_path);
        let started = Instant::now();
        let mut command = Command::new(self.executable);
        command.arg(self.input).arg(&output_path).arg("--quiet");
        let process = run_process(&mut command, self.timeout)?;
        require_success(process)?;
        let image = read_pgm(&output_path)?;
        let output_hash = blake3::hash(&image.pixels).to_hex().to_string();
        let decoded_bytes = u64::try_from(image.pixels.len()).unwrap_or(u64::MAX);
        let elapsed_ns = nanos(started.elapsed().as_nanos());
        let input_bytes = std::fs::metadata(self.input)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        drop(output_guard);
        Ok(DecodeInvocation {
            elapsed_ns,
            input_bytes,
            decoded_bytes,
            width: image.width,
            height: image.height,
            output_hash,
        })
    }

    fn version(&self) -> std::result::Result<String, BaselineFailure> {
        let mut command = Command::new(self.executable);
        command.arg("--version");
        let process = run_process(&mut command, self.timeout)?;
        let process = require_success(process)?;
        let combined = if process.stdout.bytes.is_empty() {
            process.stderr.text()
        } else {
            process.stdout.text()
        };
        combined
            .lines()
            .find(|line| !line.trim().is_empty())
            .map(str::trim)
            .map(str::to_owned)
            .ok_or_else(|| BaselineFailure::InvalidOutput("djxl --version was empty".into()))
    }
}

#[derive(Clone, Debug)]
struct DecodeInvocation {
    elapsed_ns: u64,
    input_bytes: u64,
    decoded_bytes: u64,
    width: u32,
    height: u32,
    output_hash: String,
}

impl DecodeInvocation {
    fn as_report(&self) -> CpuBaselineColdTiming {
        CpuBaselineColdTiming {
            elapsed_ns: self.elapsed_ns,
            includes_process_spawn: true,
            includes_output_file_io_and_hash: true,
            input_bytes: self.input_bytes,
            decoded_bytes: self.decoded_bytes,
            width: self.width,
            height: self.height,
            output_hash: self.output_hash.clone(),
        }
    }
}

fn execute_warm_repetition(
    runner: &DjxlRunner<'_>,
    workload: WorkloadSpec,
) -> std::result::Result<CpuBaselineWarmRepetition, BaselineFailure> {
    let parallelism = workload.parallelism();
    for _ in 0..workload.warmup {
        execute_group(runner, parallelism)?;
    }

    let wall_started = Instant::now();
    let measured = if workload.kind == WorkloadKind::Concurrent {
        execute_worker_stream(runner, parallelism, workload.measured_groups())?
    } else {
        (0..workload.measured_groups()).try_fold(Vec::new(), |mut all, _| {
            all.extend(execute_group(runner, parallelism)?);
            Ok::<_, BaselineFailure>(all)
        })?
    };
    let wall_ns = nanos(wall_started.elapsed().as_nanos());
    summarize_invocations(measured, workload, wall_ns)
}

fn execute_group(
    runner: &DjxlRunner<'_>,
    parallelism: u32,
) -> std::result::Result<Vec<DecodeInvocation>, BaselineFailure> {
    if parallelism == 1 {
        return runner.decode().map(|value| vec![value]);
    }
    std::thread::scope(|scope| {
        let barrier = Arc::new(std::sync::Barrier::new(
            usize::try_from(parallelism).expect("validated parallelism fits usize"),
        ));
        let handles = (0..parallelism)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    runner.decode()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    BaselineFailure::WorkerPanic("a djxl burst worker panicked".into())
                })?
            })
            .collect()
    })
}

fn execute_worker_stream(
    runner: &DjxlRunner<'_>,
    workers: u32,
    operations_per_worker: u32,
) -> std::result::Result<Vec<DecodeInvocation>, BaselineFailure> {
    std::thread::scope(|scope| {
        let barrier = Arc::new(std::sync::Barrier::new(
            usize::try_from(workers).expect("validated worker count fits usize"),
        ));
        let handles = (0..workers)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                scope.spawn(move || {
                    barrier.wait();
                    (0..operations_per_worker)
                        .map(|_| runner.decode())
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    BaselineFailure::WorkerPanic(
                        "a persistent djxl workload worker panicked".into(),
                    )
                })?
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(|values| values.into_iter().flatten().collect())
    })
}

fn summarize_invocations(
    measured: Vec<DecodeInvocation>,
    workload: WorkloadSpec,
    wall_ns: u64,
) -> std::result::Result<CpuBaselineWarmRepetition, BaselineFailure> {
    let first = measured.first().ok_or_else(|| {
        BaselineFailure::InvalidOutput("djxl workload produced no samples".into())
    })?;
    if measured.iter().any(|sample| {
        sample.output_hash != first.output_hash
            || sample.decoded_bytes != first.decoded_bytes
            || sample.width != first.width
            || sample.height != first.height
    }) {
        return Err(BaselineFailure::NondeterministicOutput(
            "identical djxl inputs produced different PGM pixels".into(),
        ));
    }
    let samples = measured
        .iter()
        .map(|sample| sample.elapsed_ns)
        .collect::<Vec<_>>();
    let timing = summarize_timings(&samples)
        .map_err(|error| BaselineFailure::InvalidOutput(error.to_string()))?;
    let operations = u64::try_from(measured.len()).unwrap_or(u64::MAX);
    let operations_per_second = if wall_ns == 0 {
        0.0
    } else {
        operations as f64 * 1_000_000_000.0 / wall_ns as f64
    };
    let total_input_bytes = measured
        .iter()
        .map(|sample| sample.input_bytes)
        .fold(0_u64, u64::saturating_add);
    let total_decoded_bytes = measured
        .iter()
        .map(|sample| sample.decoded_bytes)
        .fold(0_u64, u64::saturating_add);
    Ok(CpuBaselineWarmRepetition {
        samples: timing.samples,
        minimum_ns: timing.minimum_ns,
        p50_ns: timing.median_ns,
        p95_ns: timing.p95_ns,
        mean_ns: timing.mean_ns,
        wall_ns,
        operations_per_second,
        parallelism: workload.parallelism(),
        workload_execution_model: workload.execution_model(),
        process_model: CpuBaselineProcessModel::ExternalProcessPerOperation,
        total_input_bytes,
        total_decoded_bytes,
        decoded_bytes_per_operation: first.decoded_bytes,
        width: first.width,
        height: first.height,
        output_hash: first.output_hash.clone(),
    })
}

struct ScratchDirectory {
    path: PathBuf,
}

impl ScratchDirectory {
    fn create() -> io::Result<Self> {
        let base = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        (0..32)
            .find_map(|_| {
                let id = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!(
                    "jxl-gpu-harness-djxl-{}-{timestamp}-{id}",
                    std::process::id()
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => Some(Ok(Self { path })),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .unwrap_or_else(|| {
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "could not allocate a unique djxl scratch directory",
                ))
            })
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct OutputFileGuard<'a>(&'a Path);

impl Drop for OutputFileGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

struct PgmImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

fn read_pgm(path: &Path) -> std::result::Result<PgmImage, BaselineFailure> {
    let file = File::open(path).map_err(|error| BaselineFailure::Io(error.to_string()))?;
    let mut bytes = Vec::new();
    file.take(MAX_PGM_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| BaselineFailure::Io(error.to_string()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_PGM_BYTES {
        return Err(BaselineFailure::InvalidOutput(format!(
            "djxl PGM exceeds the {MAX_PGM_BYTES}-byte harness limit"
        )));
    }
    parse_pgm(bytes)
}

fn parse_pgm(bytes: Vec<u8>) -> std::result::Result<PgmImage, BaselineFailure> {
    let mut cursor = 0;
    let magic = next_pgm_token(&bytes, &mut cursor)?;
    if magic != b"P5" {
        return Err(BaselineFailure::InvalidOutput(
            "djxl output is not a binary P5 PGM".into(),
        ));
    }
    let width = parse_pgm_u32(next_pgm_token(&bytes, &mut cursor)?, "width")?;
    let height = parse_pgm_u32(next_pgm_token(&bytes, &mut cursor)?, "height")?;
    let max_value = parse_pgm_u32(next_pgm_token(&bytes, &mut cursor)?, "max value")?;
    if width == 0 || height == 0 || max_value != 255 {
        return Err(BaselineFailure::InvalidOutput(format!(
            "unsupported PGM geometry/max value: {width}x{height}, max={max_value}"
        )));
    }
    let separator = bytes.get(cursor).copied().ok_or_else(|| {
        BaselineFailure::InvalidOutput("PGM header has no pixel separator".into())
    })?;
    if !separator.is_ascii_whitespace() {
        return Err(BaselineFailure::InvalidOutput(
            "PGM header is not separated from pixels".into(),
        ));
    }
    cursor += 1;
    if separator == b'\r' && bytes.get(cursor) == Some(&b'\n') {
        cursor += 1;
    }
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| BaselineFailure::InvalidOutput("PGM dimensions overflow".into()))?;
    let pixels = bytes
        .get(cursor..)
        .ok_or_else(|| BaselineFailure::InvalidOutput("PGM pixel offset is invalid".into()))?;
    if pixels.len() != expected {
        return Err(BaselineFailure::InvalidOutput(format!(
            "PGM contains {} pixel bytes; expected {expected}",
            pixels.len()
        )));
    }
    Ok(PgmImage {
        width,
        height,
        pixels: pixels.to_vec(),
    })
}

fn next_pgm_token<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
) -> std::result::Result<&'a [u8], BaselineFailure> {
    loop {
        while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
            *cursor += 1;
        }
        if bytes.get(*cursor) != Some(&b'#') {
            break;
        }
        while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
            *cursor += 1;
        }
    }
    let start = *cursor;
    while bytes
        .get(*cursor)
        .is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'#')
    {
        *cursor += 1;
    }
    if *cursor == start {
        Err(BaselineFailure::InvalidOutput(
            "PGM header is truncated".into(),
        ))
    } else {
        Ok(&bytes[start..*cursor])
    }
}

fn parse_pgm_u32(token: &[u8], field: &str) -> std::result::Result<u32, BaselineFailure> {
    std::str::from_utf8(token)
        .ok()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| BaselineFailure::InvalidOutput(format!("invalid PGM {field}")))
}

struct CapturedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

impl CapturedOutput {
    fn text(&self) -> String {
        let mut value = String::from_utf8_lossy(&self.bytes).into_owned();
        if self.truncated {
            value.push_str("\n[diagnostic truncated]");
        }
        value
    }
}

struct ProcessOutput {
    status: ExitStatus,
    stdout: CapturedOutput,
    stderr: CapturedOutput,
}

fn run_process(
    command: &mut Command,
    timeout: Duration,
) -> std::result::Result<ProcessOutput, BaselineFailure> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| match error.kind() {
            io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied => {
                BaselineFailure::Unavailable(error.to_string())
            }
            _ => BaselineFailure::Io(error.to_string()),
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| BaselineFailure::Io("failed to capture djxl stdout".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| BaselineFailure::Io("failed to capture djxl stderr".into()))?;

    std::thread::scope(|scope| {
        let stdout_reader = scope.spawn(move || read_bounded(stdout));
        let stderr_reader = scope.spawn(move || read_bounded(stderr));
        let started = Instant::now();
        let wait = loop {
            match child.try_wait() {
                Ok(Some(status)) => break ProcessWait::Exited(status),
                Ok(None) if started.elapsed() >= timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break ProcessWait::TimedOut;
                }
                Ok(None) => std::thread::sleep(PROCESS_POLL_INTERVAL),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    break ProcessWait::Io(error);
                }
            }
        };
        let stdout = stdout_reader
            .join()
            .map_err(|_| BaselineFailure::WorkerPanic("stdout reader panicked".into()))?
            .map_err(|error| BaselineFailure::Io(error.to_string()))?;
        let stderr = stderr_reader
            .join()
            .map_err(|_| BaselineFailure::WorkerPanic("stderr reader panicked".into()))?
            .map_err(|error| BaselineFailure::Io(error.to_string()))?;
        match wait {
            ProcessWait::Exited(status) => Ok(ProcessOutput {
                status,
                stdout,
                stderr,
            }),
            ProcessWait::TimedOut => Err(BaselineFailure::Timeout(format!(
                "djxl exceeded {} ms; stderr: {}",
                timeout.as_millis(),
                stderr.text()
            ))),
            ProcessWait::Io(error) => Err(BaselineFailure::Io(error.to_string())),
        }
    })
}

enum ProcessWait {
    Exited(ExitStatus),
    TimedOut,
    Io(io::Error),
}

fn read_bounded(mut reader: impl Read) -> io::Result<CapturedOutput> {
    let mut bytes = Vec::with_capacity(MAX_DIAGNOSTIC_BYTES);
    let mut truncated = false;
    let mut chunk = [0_u8; 4096];
    loop {
        let count = reader.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        let remaining = MAX_DIAGNOSTIC_BYTES.saturating_sub(bytes.len());
        let keep = remaining.min(count);
        bytes.extend_from_slice(&chunk[..keep]);
        truncated |= keep != count;
    }
    Ok(CapturedOutput { bytes, truncated })
}

fn require_success(output: ProcessOutput) -> std::result::Result<ProcessOutput, BaselineFailure> {
    if output.status.success() {
        Ok(output)
    } else {
        Err(BaselineFailure::ProcessExit(format!(
            "djxl exited with {}; stderr: {}",
            output.status,
            output.stderr.text()
        )))
    }
}

#[derive(Debug)]
enum BaselineFailure {
    Unavailable(String),
    Timeout(String),
    ProcessExit(String),
    InvalidOutput(String),
    Io(String),
    NondeterministicOutput(String),
    WorkerPanic(String),
}

impl BaselineFailure {
    fn detail(&self) -> String {
        let detail = match self {
            Self::Unavailable(detail)
            | Self::Timeout(detail)
            | Self::ProcessExit(detail)
            | Self::InvalidOutput(detail)
            | Self::Io(detail)
            | Self::NondeterministicOutput(detail)
            | Self::WorkerPanic(detail) => detail,
        };
        bounded_string(detail)
    }
}

fn bounded_string(value: &str) -> String {
    if value.len() <= MAX_DIAGNOSTIC_BYTES {
        return value.to_owned();
    }
    let mut boundary = MAX_DIAGNOSTIC_BYTES;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}\n[diagnostic truncated]", &value[..boundary])
}

const fn nanos(value: u128) -> u64 {
    if value > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{DeclaredExtent, SizeClass, WorkloadExecutionModel};
    use crate::report::{CodecTiming, TimingStatistics, WorkloadTiming};

    #[test]
    fn parses_binary_pgm_without_treating_pixel_whitespace_as_header() {
        let image = parse_pgm(b"P5\n# generated\n3 1\n255\n\x00\x0a\xff".to_vec()).unwrap();
        assert_eq!((image.width, image.height), (3, 1));
        assert_eq!(image.pixels, [0, 10, 255]);
    }

    #[test]
    fn invalid_timeout_is_rejected() {
        let result = DjxlBaselineOptions::new(
            PathBuf::from("djxl"),
            WorkloadSpec::default(),
            None,
            None,
            Some(0),
        );
        assert!(matches!(result, Err(Error::InvalidConfig(_))));
    }

    #[test]
    fn incompatible_gpu_contract_is_typed_unsupported_without_spawn() {
        let options = CodecRunOptions {
            operation: CodecOperation::Decode,
            workload: WorkloadSpec::default(),
            output_target: OutputTarget::GpuResident,
            format: GpuPixelFormat::U8,
            size_class: SizeClass::Small,
            extent: None,
        };
        let baseline = DjxlBaselineOptions::new(
            PathBuf::from("definitely-not-djxl"),
            options.workload,
            None,
            None,
            None,
        )
        .unwrap();
        let gpu = empty_gpu_report(&options);
        let report = run_djxl_baseline(Path::new("input.jxl"), &options, &gpu, &baseline);
        assert_eq!(report.status, CpuBaselineStatus::Unsupported);
        assert_eq!(
            report.issue.unwrap().kind,
            CpuBaselineIssueKind::UnsupportedContract
        );
    }

    #[test]
    fn missing_executable_is_typed_skipped() {
        let options = comparable_options();
        let baseline = DjxlBaselineOptions::new(
            PathBuf::from("definitely-not-a-real-djxl-executable"),
            options.workload,
            None,
            None,
            Some(10),
        )
        .unwrap();
        let gpu = empty_gpu_report(&options);
        let report = run_djxl_baseline(Path::new("input.jxl"), &options, &gpu, &baseline);
        assert_eq!(report.status, CpuBaselineStatus::Skipped);
        assert_eq!(
            report.issue.unwrap().kind,
            CpuBaselineIssueKind::ExecutableUnavailable
        );
    }

    #[test]
    fn diagnostic_strings_are_bounded() {
        let value = "x".repeat(MAX_DIAGNOSTIC_BYTES * 2);
        let bounded = bounded_string(&value);
        assert!(bounded.len() < value.len());
        assert!(bounded.ends_with("[diagnostic truncated]"));
    }

    #[cfg(unix)]
    #[test]
    fn fake_djxl_runs_cold_warm_and_exact_verification() {
        let scripts = ScratchDirectory::create().unwrap();
        let executable = scripts.path.join("fake-djxl");
        write_test_executable(
            &executable,
            "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then echo 'djxl fake-1'; exit 0; fi\nprintf 'P5\\n2 1\\n255\\n' > \"$2\"\nprintf '\\001\\002' >> \"$2\"\n",
        );

        let mut options = comparable_options();
        options.workload.kind = WorkloadKind::WarmSequential;
        options.workload.warmup = 1;
        options.workload.iterations = 2;
        let baseline =
            DjxlBaselineOptions::new(executable, options.workload, None, None, Some(1_000))
                .unwrap();
        let mut gpu = empty_gpu_report(&options);
        gpu.status = CaseStatus::Passed;
        gpu.output_hash = Some(blake3::hash(&[1, 2]).to_hex().to_string());
        gpu.readback_logical_bytes = 4;
        gpu.timing = Some(CodecTiming {
            operation_latency: TimingStatistics::default(),
            workload: WorkloadTiming {
                operations: 2,
                parallelism: 1,
                execution_model: WorkloadExecutionModel::Sequential,
                wall_ns: 1,
                operations_per_second: 1.0,
            },
        });
        let input = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/basic.jxl");
        let report = run_djxl_baseline(&input, &options, &gpu, &baseline);
        assert_eq!(report.status, CpuBaselineStatus::Completed);
        assert_eq!(report.verified, Some(true));
        assert_eq!(report.provenance.version.as_deref(), Some("djxl fake-1"));
        let warm = report.warm_repetition.unwrap();
        assert_eq!(warm.samples, 2);
        assert_eq!(warm.total_decoded_bytes, 4);
        assert_eq!(
            warm.process_model,
            CpuBaselineProcessModel::ExternalProcessPerOperation
        );
    }

    #[cfg(unix)]
    #[test]
    fn external_process_timeout_is_typed_and_bounded() {
        let scripts = ScratchDirectory::create().unwrap();
        let executable = scripts.path.join("slow-djxl");
        write_test_executable(&executable, "#!/bin/sh\nexec sleep 1\n");
        let options = comparable_options();
        let baseline =
            DjxlBaselineOptions::new(executable, options.workload, None, None, Some(10)).unwrap();
        let gpu = empty_gpu_report(&options);
        let report = run_djxl_baseline(Path::new("input.jxl"), &options, &gpu, &baseline);
        assert_eq!(report.status, CpuBaselineStatus::TimedOut);
        let issue = report.issue.unwrap();
        assert_eq!(issue.kind, CpuBaselineIssueKind::Timeout);
        assert!(issue.detail.len() <= MAX_DIAGNOSTIC_BYTES + 32);
    }

    #[cfg(unix)]
    #[test]
    fn nonzero_exit_drains_and_bounds_stderr() {
        let scripts = ScratchDirectory::create().unwrap();
        let executable = scripts.path.join("failing-djxl");
        let noise = "x".repeat(MAX_DIAGNOSTIC_BYTES * 2);
        write_test_executable(
            &executable,
            &format!("#!/bin/sh\nprintf '%s' '{noise}' >&2\nexit 9\n"),
        );
        let options = comparable_options();
        let baseline =
            DjxlBaselineOptions::new(executable, options.workload, None, None, Some(1_000))
                .unwrap();
        let gpu = empty_gpu_report(&options);
        let report = run_djxl_baseline(Path::new("input.jxl"), &options, &gpu, &baseline);
        assert_eq!(report.status, CpuBaselineStatus::Error);
        let issue = report.issue.unwrap();
        assert_eq!(issue.kind, CpuBaselineIssueKind::ProcessExit);
        assert!(issue.detail.len() <= MAX_DIAGNOSTIC_BYTES + 32);
        assert!(issue.detail.ends_with("[diagnostic truncated]"));
    }

    #[cfg(unix)]
    fn write_test_executable(path: &Path, source: &str) {
        use std::os::unix::fs::PermissionsExt;

        std::fs::write(path, source).unwrap();
        let mut permissions = std::fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(path, permissions).unwrap();
    }

    fn comparable_options() -> CodecRunOptions {
        CodecRunOptions {
            operation: CodecOperation::Decode,
            workload: WorkloadSpec::default(),
            output_target: OutputTarget::CpuReadback,
            format: GpuPixelFormat::U8,
            size_class: SizeClass::Small,
            extent: Some(DeclaredExtent {
                width: 2,
                height: 1,
            }),
        }
    }

    fn empty_gpu_report(options: &CodecRunOptions) -> CodecCaseReport {
        CodecCaseReport::new(
            "test".into(),
            Path::new("input.jxl"),
            options.operation,
            options.workload,
            options.output_target,
            options.format,
            options.size_class,
            options.extent,
            1,
        )
    }
}
