use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use jxl_gpu_harness::adapter::enumerate_adapters;
use jxl_gpu_harness::benchmark::{BenchmarkOptions, benchmark_capture};
use jxl_gpu_harness::capture::{CaptureFile, OperationKind, PrecisionMode};
use jxl_gpu_harness::codec::{
    CodecCorpusCase, CodecCorpusConfig, CodecOperation, CodecRunOptions, DeclaredExtent,
    GpuPixelFormat, OutputTarget, SizeClass, WorkloadKind, WorkloadSpec, request_backend,
    run_codec_case,
};
use jxl_gpu_harness::config::{CorpusConfig, SyntheticCaseConfig, ThresholdConfig};
use jxl_gpu_harness::error::{Error, Result};
use jxl_gpu_harness::replay::{BackendKind, create_backend, verify_capture};
use jxl_gpu_harness::report::{CaseReport, CaseStatus, RunReport};
use jxl_gpu_harness::synthetic::generate_case;
use jxl_gpu_harness::tune::TuningProfile;

const DEFAULT_CORPUS: &str = "tools/jxl_gpu_harness/corpus.toml";
const DEFAULT_THRESHOLDS: &str = "tools/jxl_gpu_harness/thresholds.toml";

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Enumerate adapters and portable limits visible to wgpu.
    Adapters(OutputArgs),
    /// Generate and verify every synthetic corpus case.
    Verify(CorpusRunArgs),
    /// Benchmark every synthetic corpus case.
    Bench(BenchArgs),
    /// Run GPU-required decode, encode, or round-trip workloads without a CPU codec fallback.
    Codec(CodecArgs),
    /// Generate one deterministic capture file.
    Capture(CaptureArgs),
    /// Replay and compare one capture file.
    Replay(ReplayArgs),
    /// Benchmark candidate backends and write a tuning profile.
    Tune(TuneArgs),
}

#[derive(Clone, Debug, Args)]
struct OutputArgs {
    /// Write JSON to this path instead of standard output.
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Clone, Debug, Args)]
struct CorpusRunArgs {
    #[arg(long, default_value = DEFAULT_CORPUS)]
    corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_THRESHOLDS)]
    thresholds: PathBuf,
    #[arg(long, value_enum, default_value = "reference")]
    backend: BackendKind,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Clone, Debug, Args)]
struct BenchArgs {
    #[command(flatten)]
    run: CorpusRunArgs,
    #[arg(long, default_value_t = 3)]
    warmup: u32,
    #[arg(long, default_value_t = 20)]
    iterations: u32,
}

#[derive(Clone, Debug, Args)]
struct CodecArgs {
    /// Codec inputs. Decode inputs must be JPEG XL files. May be combined with --corpus.
    #[arg(value_name = "INPUT")]
    inputs: Vec<PathBuf>,
    /// Versioned TOML list of codec cases.
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "decode")]
    operation: CodecOperation,
    #[arg(long, value_enum, default_value = "single-latency")]
    workload: WorkloadKind,
    #[arg(long, value_enum, default_value = "gpu-resident")]
    output_target: OutputTarget,
    #[arg(long, value_enum, default_value = "rgba8")]
    format: GpuPixelFormat,
    #[arg(long, value_enum, default_value = "auto")]
    size_class: SizeClass,
    /// Optional declared WIDTHxHEIGHT source extent or decode corpus hint.
    #[arg(long, value_parser = parse_extent)]
    extent: Option<DeclaredExtent>,
    #[arg(long, default_value_t = 0)]
    warmup: u32,
    #[arg(long, default_value_t = 1)]
    iterations: u32,
    #[arg(long, default_value_t = 1)]
    batch_size: u32,
    #[arg(long, default_value_t = 1)]
    concurrency: u32,
    #[arg(long, default_value_t = 2)]
    max_in_flight: u32,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Clone, Debug, Args)]
struct CaptureArgs {
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, default_value = "synthetic")]
    name: String,
    #[arg(long, value_enum)]
    operation: OperationKind,
    #[arg(long, default_value_t = 17)]
    width: u32,
    #[arg(long, default_value_t = 19)]
    height: u32,
    #[arg(long, default_value_t = 3)]
    channels: u16,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Operation parameter in NAME=VALUE form. May be repeated.
    #[arg(long = "parameter", value_parser = parse_parameter)]
    parameters: Vec<(String, f64)>,
    /// Replace an existing capture file.
    #[arg(long)]
    force: bool,
}

#[derive(Clone, Debug, Args)]
struct ReplayArgs {
    #[arg(long, short)]
    input: PathBuf,
    #[arg(long, default_value = DEFAULT_THRESHOLDS)]
    thresholds: PathBuf,
    #[arg(long, value_enum, default_value = "reference")]
    backend: BackendKind,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Clone, Debug, Args)]
struct TuneArgs {
    #[arg(long, default_value = DEFAULT_CORPUS)]
    corpus: PathBuf,
    #[arg(long, default_value = DEFAULT_THRESHOLDS)]
    thresholds: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "reference,wgpu"
    )]
    candidates: Vec<BackendKind>,
    #[arg(long, default_value_t = 2)]
    warmup: u32,
    #[arg(long, default_value_t = 10)]
    iterations: u32,
    /// Replace an existing tuning profile.
    #[arg(long)]
    force: bool,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> Result<bool> {
    match cli.command {
        Command::Adapters(args) => {
            let mut report = RunReport::new("adapters");
            report.adapters = enumerate_adapters();
            write_json(&report, args.output.as_deref())?;
            Ok(true)
        }
        Command::Verify(args) => run_corpus(args, None),
        Command::Bench(args) => run_corpus(
            args.run,
            Some(BenchmarkOptions {
                warmup: args.warmup,
                iterations: args.iterations,
            }),
        ),
        Command::Codec(args) => run_codec(args),
        Command::Capture(args) => {
            refuse_overwrite(&args.output, args.force)?;
            let parameters = args.parameters.into_iter().collect::<BTreeMap<_, _>>();
            let capture = generate_case(&SyntheticCaseConfig {
                name: args.name,
                operation: args.operation,
                width: args.width,
                height: args.height,
                channels: args.channels,
                seed: args.seed,
                precision: PrecisionMode::F32,
                parameters,
            })?;
            capture.write_path(&args.output)?;
            println!("{}", args.output.display());
            Ok(true)
        }
        Command::Replay(args) => {
            let capture = CaptureFile::read_path(&args.input)?;
            let thresholds = ThresholdConfig::load(&args.thresholds)?;
            let mut report = RunReport::new("replay");
            if args.backend == BackendKind::Wgpu {
                report.adapters = enumerate_adapters();
            }
            report
                .cases
                .push(case_result(&capture, args.backend, &thresholds, None));
            let passed = report.passed();
            write_json(&report, args.output.output.as_deref())?;
            Ok(passed)
        }
        Command::Tune(args) => {
            refuse_overwrite(&args.output, args.force)?;
            let corpus = CorpusConfig::load(&args.corpus)?;
            let thresholds = ThresholdConfig::load(&args.thresholds)?;
            let captures = generate_corpus(&corpus)?;
            let profile = TuningProfile::tune(
                &captures,
                &args.candidates,
                |capture| {
                    thresholds
                        .for_operation(&capture.metadata.operation.kind)
                        .clone()
                },
                BenchmarkOptions {
                    warmup: args.warmup,
                    iterations: args.iterations,
                },
            )?;
            let passed = profile
                .entries
                .values()
                .all(|entry| entry.selected_backend.is_some());
            profile.write_json(&args.output)?;
            println!("{}", args.output.display());
            Ok(passed)
        }
    }
}

fn run_corpus(args: CorpusRunArgs, benchmark: Option<BenchmarkOptions>) -> Result<bool> {
    let corpus = CorpusConfig::load(&args.corpus)?;
    let thresholds = ThresholdConfig::load(&args.thresholds)?;
    let captures = generate_corpus(&corpus)?;
    let mut report = RunReport::new(if benchmark.is_some() {
        "bench"
    } else {
        "verify"
    });
    if args.backend == BackendKind::Wgpu {
        report.adapters = enumerate_adapters();
    }
    report.cases.extend(
        captures
            .iter()
            .map(|capture| case_result(capture, args.backend, &thresholds, benchmark)),
    );
    let passed = report.passed();
    write_json(&report, args.output.output.as_deref())?;
    Ok(passed)
}

fn run_codec(args: CodecArgs) -> Result<bool> {
    let inputs = collect_codec_inputs(&args)?;
    let options = CodecRunOptions {
        operation: args.operation,
        workload: WorkloadSpec {
            kind: args.workload,
            warmup: args.warmup,
            iterations: args.iterations,
            batch_size: args.batch_size,
            concurrency: args.concurrency,
            max_in_flight: args.max_in_flight,
        },
        output_target: args.output_target,
        format: args.format,
        size_class: args.size_class,
        extent: args.extent,
    }
    .validate()?;
    let mut report = RunReport::new("codec");
    report.adapters = enumerate_adapters();
    let backend = request_backend()?;
    report.codec_cases.extend(
        inputs
            .iter()
            .map(|case| run_codec_case(case, backend.as_ref(), &options)),
    );
    let passed = report.passed();
    write_json(&report, args.output.output.as_deref())?;
    Ok(passed)
}

fn collect_codec_inputs(args: &CodecArgs) -> Result<Vec<CodecCorpusCase>> {
    let mut cases = match &args.corpus {
        Some(path) => CodecCorpusConfig::load(path)?.cases,
        None => Vec::new(),
    };
    cases.extend(args.inputs.iter().map(|path| {
        CodecCorpusCase {
            name: path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("image")
                .to_string(),
            path: path.clone(),
            size_class: SizeClass::Auto,
            extent: None,
        }
    }));
    if cases.is_empty() {
        return Err(Error::InvalidConfig(
            "codec command requires at least one input path or --corpus".into(),
        ));
    }
    let mut names = std::collections::BTreeSet::new();
    for case in &cases {
        if !names.insert(&case.name) {
            return Err(Error::InvalidConfig(format!(
                "duplicate codec case name {}",
                case.name
            )));
        }
    }
    Ok(cases)
}

fn case_result(
    capture: &CaptureFile,
    backend_kind: BackendKind,
    thresholds: &ThresholdConfig,
    benchmark: Option<BenchmarkOptions>,
) -> CaseReport {
    let result = create_backend(backend_kind).and_then(|mut backend| {
        let threshold = thresholds.for_operation(&capture.metadata.operation.kind);
        match benchmark {
            Some(options) => benchmark_capture(capture, backend.as_mut(), threshold, options),
            None => verify_capture(capture, backend.as_mut(), threshold),
        }
    });
    result.unwrap_or_else(|error| CaseReport {
        case_id: capture.metadata.case_id.clone(),
        operation: capture.metadata.operation.kind.as_str().into(),
        backend: backend_kind.as_str().into(),
        status: match error {
            Error::BackendUnavailable(_) => CaseStatus::Unavailable,
            Error::UnsupportedOperation { .. } => CaseStatus::Unsupported,
            _ => CaseStatus::Error,
        },
        output_hash: None,
        metrics: None,
        threshold: None,
        timing: None,
        message: Some(error.to_string()),
    })
}

fn generate_corpus(config: &CorpusConfig) -> Result<Vec<CaptureFile>> {
    config.cases.iter().map(generate_case).collect()
}

fn write_json(value: &impl serde::Serialize, path: Option<&Path>) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(path) = path {
        std::fs::write(path, bytes).map_err(|source| Error::io(path, source))?;
    } else {
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(&bytes)
            .and_then(|()| stdout.write_all(b"\n"))
            .map_err(|source| Error::io("<stdout>", source))?;
    }
    Ok(())
}

fn refuse_overwrite(path: &Path, force: bool) -> Result<()> {
    if path.exists() && !force {
        return Err(Error::InvalidConfig(format!(
            "{} already exists; pass --force to replace it",
            path.display()
        )));
    }
    Ok(())
}

fn parse_parameter(value: &str) -> std::result::Result<(String, f64), String> {
    let (name, value) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=VALUE".to_string())?;
    if name.trim().is_empty() {
        return Err("parameter name must not be empty".into());
    }
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid parameter value: {error}"))?;
    if !parsed.is_finite() {
        return Err("parameter value must be finite".into());
    }
    Ok((name.into(), parsed))
}

fn parse_extent(value: &str) -> std::result::Result<DeclaredExtent, String> {
    let (width, height) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| "expected WIDTHxHEIGHT".to_string())?;
    let extent = DeclaredExtent {
        width: width
            .parse()
            .map_err(|error| format!("invalid extent width: {error}"))?,
        height: height
            .parse()
            .map_err(|error| format!("invalid extent height: {error}"))?,
    };
    extent.validate().map_err(|error| error.to_string())
}
