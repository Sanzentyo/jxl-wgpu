//! GPU-only codec workload orchestration.
//!
//! Production runs here never instantiate the published CPU `jxl` decoder and never substitute
//! CPU pixels when a GPU frontend or backend is unavailable. Typed support boundaries from the
//! codec crates remain `unsupported` or `incomplete` report outcomes and cannot pass a run.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::ValueEnum;
use jxl_gpu_formats::{
    ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, ImageLayout, Packed422Order,
    PixelFormat, RgbChannelOrder, vpi::VpiPitchLinearFormat as Vpi,
};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    DisplayPipeline, DisplayTextureDescriptor, ImageReadbackPipeline, WgpuBackend,
    WgpuBackendConfig,
};
use jxl_wgpu_decode::{
    Error as DecodeError, F64OutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping,
};
use jxl_wgpu_encode::{BufferImageSource, EncodeError, LosslessGray8Encoder, WgpuContext};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::benchmark::summarize_timings;
use crate::error::{Error, Result};
use crate::report::{
    CaseStatus, CodecCaseReport, CodecIssue, CodecIssueKind, CodecTiming, WorkloadTiming,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum CodecOperation {
    Decode,
    Encode,
    RoundTrip,
}

impl CodecOperation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Encode => "encode",
            Self::RoundTrip => "round_trip",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadKind {
    SingleLatency,
    WarmSequential,
    /// Barrier-synchronized host-thread fan-out. This does not coalesce GPU work into a batch.
    ConcurrentBurst,
    Concurrent,
    Animation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OutputTarget {
    GpuResident,
    DisplayTexture,
    CpuReadback,
}

/// How measured operations are launched by the harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadExecutionModel {
    Sequential,
    BarrierSynchronizedHostFanout,
    PersistentHostWorkers,
    AnimationSession,
}

/// Explicit host-transfer path used for decoded images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuReadbackMode {
    StagedCopy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum SizeClass {
    #[default]
    Auto,
    Small,
    Odd,
    Large,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum GpuPixelFormat {
    U8,
    S8,
    U16,
    U32,
    S32,
    S16,
    #[serde(rename = "2s16")]
    #[value(name = "2s16")]
    TwoS16,
    F32,
    F64,
    #[serde(rename = "2f32")]
    #[value(name = "2f32")]
    TwoF32,
    Y8,
    Y16,
    I420,
    I422,
    I444,
    Nv12,
    Nv21,
    Nv16,
    Nv61,
    Nv24,
    Nv42,
    P010,
    P012,
    P016,
    Yuyv,
    Uyvy,
    Rgb8,
    Bgr8,
    Rgba8,
    Bgra8,
    Rgb8Planar,
    Bgr8Planar,
    Rgba8Planar,
    Bgra8Planar,
}

impl GpuPixelFormat {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::S8 => "s8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::S32 => "s32",
            Self::S16 => "s16",
            Self::TwoS16 => "2s16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::TwoF32 => "2f32",
            Self::Y8 => "y8",
            Self::Y16 => "y16",
            Self::I420 => "i420",
            Self::I422 => "i422",
            Self::I444 => "i444",
            Self::Nv12 => "nv12",
            Self::Nv21 => "nv21",
            Self::Nv16 => "nv16",
            Self::Nv61 => "nv61",
            Self::Nv24 => "nv24",
            Self::Nv42 => "nv42",
            Self::P010 => "p010",
            Self::P012 => "p012",
            Self::P016 => "p016",
            Self::Yuyv => "yuyv",
            Self::Uyvy => "uyvy",
            Self::Rgb8 => "rgb8",
            Self::Bgr8 => "bgr8",
            Self::Rgba8 => "rgba8",
            Self::Bgra8 => "bgra8",
            Self::Rgb8Planar => "rgb8_planar",
            Self::Bgr8Planar => "bgr8_planar",
            Self::Rgba8Planar => "rgba8_planar",
            Self::Bgra8Planar => "bgra8_planar",
        }
    }

    #[must_use]
    pub fn pixel_format(self) -> PixelFormat {
        let yuv = ColorSpecification::Defined(ColorSpec::bt709(
            ColorRange::Limited,
            ChromaLocation2d::CENTER,
        ));
        let rgb =
            ColorSpecification::Defined(ColorSpec::bt709(ColorRange::Full, ChromaLocation2d::BOTH));
        match self {
            Self::U8 => Vpi::U8.pixel_format(),
            Self::S8 => Vpi::S8.pixel_format(),
            Self::U16 => Vpi::U16.pixel_format(),
            Self::U32 => Vpi::U32.pixel_format(),
            Self::S32 => Vpi::S32.pixel_format(),
            Self::S16 => Vpi::S16.pixel_format(),
            Self::TwoS16 => Vpi::TwoS16.pixel_format(),
            Self::F32 => Vpi::F32.pixel_format(),
            Self::F64 => Vpi::F64.pixel_format(),
            Self::TwoF32 => Vpi::TwoF32.pixel_format(),
            Self::Y8 => PixelFormat::luma(8, yuv),
            Self::Y16 => PixelFormat::luma(16, yuv),
            Self::I420 => PixelFormat::i420(8, 8, yuv).expect("I420 descriptor is valid"),
            Self::I422 => PixelFormat::i422(8, 8, yuv).expect("I422 descriptor is valid"),
            Self::I444 => PixelFormat::i444(8, 8, yuv).expect("I444 descriptor is valid"),
            Self::Nv12 => PixelFormat::nv12(yuv),
            Self::Nv21 => PixelFormat::nv21(yuv),
            Self::Nv16 => PixelFormat::nv16(yuv),
            Self::Nv61 => PixelFormat::nv61(yuv),
            Self::Nv24 => PixelFormat::nv24(yuv),
            Self::Nv42 => PixelFormat::nv42(yuv),
            Self::P010 => PixelFormat::p010(yuv),
            Self::P012 => PixelFormat::p012(yuv),
            Self::P016 => PixelFormat::p016(yuv),
            Self::Yuyv => PixelFormat::packed_yuv4228(Packed422Order::Yuyv, yuv),
            Self::Uyvy => PixelFormat::packed_yuv4228(Packed422Order::Uyvy, yuv),
            Self::Rgb8 => PixelFormat::rgb8(RgbChannelOrder::Rgb, false, rgb),
            Self::Bgr8 => PixelFormat::rgb8(RgbChannelOrder::Bgr, false, rgb),
            Self::Rgba8 => PixelFormat::rgb8(RgbChannelOrder::Rgba, false, rgb),
            Self::Bgra8 => PixelFormat::rgb8(RgbChannelOrder::Bgra, false, rgb),
            Self::Rgb8Planar => PixelFormat::rgb8(RgbChannelOrder::Rgb, true, rgb),
            Self::Bgr8Planar => PixelFormat::rgb8(RgbChannelOrder::Bgr, true, rgb),
            Self::Rgba8Planar => PixelFormat::rgb8(RgbChannelOrder::Rgba, true, rgb),
            Self::Bgra8Planar => PixelFormat::rgb8(RgbChannelOrder::Bgra, true, rgb),
        }
    }

    fn output_request(self) -> std::result::Result<GpuOutputRequest, DecodeError> {
        let format = self.pixel_format();
        if self == Self::F64 {
            GpuOutputRequest::numeric(
                format,
                NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::NativeOrExactF32Widening),
            )
        } else if matches!(
            self,
            Self::U8
                | Self::S8
                | Self::U16
                | Self::U32
                | Self::S32
                | Self::S16
                | Self::TwoS16
                | Self::F32
                | Self::TwoF32
        ) {
            GpuOutputRequest::numeric(format, NumericSampleMapping::NormalizedGray8)
        } else {
            GpuOutputRequest::color(format)
        }
    }
}

impl fmt::Display for GpuPixelFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub kind: WorkloadKind,
    pub warmup: u32,
    pub iterations: u32,
    pub burst_size: u32,
    pub concurrency: u32,
    pub max_frame_slots: u32,
}

impl Default for WorkloadSpec {
    fn default() -> Self {
        Self {
            kind: WorkloadKind::SingleLatency,
            warmup: 0,
            iterations: 1,
            burst_size: 1,
            concurrency: 1,
            max_frame_slots: 2,
        }
    }
}

impl WorkloadSpec {
    pub fn validate(self) -> Result<Self> {
        if self.iterations == 0
            || self.burst_size == 0
            || self.concurrency == 0
            || self.max_frame_slots == 0
        {
            return Err(Error::InvalidConfig(
                "iterations, burst-size, concurrency, and max-frame-slots must be nonzero".into(),
            ));
        }
        if self.iterations > 1_000_000
            || self.warmup > 1_000_000
            || self.burst_size > 1_024
            || self.concurrency > 1_024
            || self.max_frame_slots > 1_024
        {
            return Err(Error::InvalidConfig(
                "codec workload dimensions are unreasonably large".into(),
            ));
        }
        self.total_measured_operations()?;
        Ok(self)
    }

    #[must_use]
    pub const fn parallelism(self) -> u32 {
        match self.kind {
            WorkloadKind::ConcurrentBurst => self.burst_size,
            WorkloadKind::Concurrent => self.concurrency,
            _ => 1,
        }
    }

    #[must_use]
    pub const fn execution_model(self) -> WorkloadExecutionModel {
        match self.kind {
            WorkloadKind::ConcurrentBurst => WorkloadExecutionModel::BarrierSynchronizedHostFanout,
            WorkloadKind::Concurrent => WorkloadExecutionModel::PersistentHostWorkers,
            WorkloadKind::Animation => WorkloadExecutionModel::AnimationSession,
            WorkloadKind::SingleLatency | WorkloadKind::WarmSequential => {
                WorkloadExecutionModel::Sequential
            }
        }
    }

    #[must_use]
    pub const fn measured_groups(self) -> u32 {
        match self.kind {
            WorkloadKind::SingleLatency => 1,
            _ => self.iterations,
        }
    }

    pub fn total_measured_operations(self) -> Result<u64> {
        u64::from(self.measured_groups())
            .checked_mul(u64::from(self.parallelism()))
            .ok_or(Error::LengthOverflow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclaredExtent {
    pub width: u32,
    pub height: u32,
}

impl DeclaredExtent {
    pub fn validate(self) -> Result<Self> {
        if self.width == 0 || self.height == 0 {
            return Err(Error::InvalidConfig(
                "declared extent must be non-empty".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecCorpusConfig {
    pub version: u16,
    pub cases: Vec<CodecCorpusCase>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodecCorpusCase {
    pub name: String,
    pub path: PathBuf,
    #[serde(default)]
    pub size_class: SizeClass,
    #[serde(default)]
    pub extent: Option<DeclaredExtent>,
}

impl CodecCorpusConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config_path = path.as_ref();
        let source = std::fs::read_to_string(config_path)
            .map_err(|source| Error::io(config_path, source))?;
        let mut config: Self = toml::from_str(&source)?;
        if config.version != 1 {
            return Err(Error::InvalidConfig(format!(
                "codec corpus version {} is unsupported",
                config.version
            )));
        }
        if config.cases.is_empty() {
            return Err(Error::InvalidConfig(
                "codec corpus must contain at least one case".into(),
            ));
        }
        let base = config_path.parent().unwrap_or_else(|| Path::new("."));
        let mut names = std::collections::BTreeSet::new();
        for case in &mut config.cases {
            if case.name.trim().is_empty() || !names.insert(case.name.clone()) {
                return Err(Error::InvalidConfig(format!(
                    "codec corpus contains an empty or duplicate case name: {}",
                    case.name
                )));
            }
            if let Some(extent) = case.extent {
                extent.validate()?;
            }
            if case.path.is_relative() {
                case.path = base.join(&case.path);
            }
        }
        Ok(config)
    }
}

#[derive(Clone, Debug)]
pub struct CodecRunOptions {
    pub operation: CodecOperation,
    pub workload: WorkloadSpec,
    pub output_target: OutputTarget,
    pub format: GpuPixelFormat,
    pub size_class: SizeClass,
    pub extent: Option<DeclaredExtent>,
}

impl CodecRunOptions {
    pub fn validate(mut self) -> Result<Self> {
        self.workload = self.workload.validate()?;
        if let Some(extent) = self.extent {
            extent.validate()?;
        }
        Ok(self)
    }
}

pub fn request_backend() -> Result<Option<WgpuBackend>> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig::default())) {
        Ok(backend) => Ok(Some(backend)),
        Err(jxl_wgpu::Error::NoAdapter) => Ok(None),
        Err(error) => Err(Error::BackendUnavailable(error.to_string())),
    }
}

pub fn run_codec_case(
    case: &CodecCorpusCase,
    backend: Option<&WgpuBackend>,
    options: &CodecRunOptions,
) -> CodecCaseReport {
    let encoded_bytes = std::fs::metadata(&case.path).map_or(0, |metadata| metadata.len());
    let extent = options.extent.or(case.extent);
    let size_class = match (options.size_class, case.size_class) {
        (SizeClass::Auto, SizeClass::Auto) => classify_extent(extent),
        (SizeClass::Auto, selected) => selected,
        (selected, _) => selected,
    };
    let mut report = CodecCaseReport::new(
        case.name.clone(),
        &case.path,
        options.operation,
        options.workload,
        options.output_target,
        options.format,
        size_class,
        extent,
        encoded_bytes,
    );
    let mut effective_options = options.clone();
    effective_options.extent = extent;

    let execution = match effective_options.operation {
        CodecOperation::Decode => run_decode(&case.path, backend, &effective_options),
        CodecOperation::Encode => run_encode(&case.path, backend, &effective_options),
        CodecOperation::RoundTrip => run_round_trip(&case.path, backend, &effective_options),
    };
    match execution {
        Ok(execution) => {
            report.status = CaseStatus::Passed;
            report.adapter = backend.map(|value| value.adapter_info().name.clone());
            report.frame_count = execution.frame_count;
            report.output_bytes = execution.output_bytes;
            report.gpu_output_logical_bytes = execution.gpu_output_logical_bytes;
            report.codec_submissions = execution.codec_submissions;
            report.codec_completion_waits = execution.codec_completion_waits;
            report.display_submissions = execution.display_submissions;
            report.display_completion_waits = execution.display_completion_waits;
            report.readback_submissions = execution.readback_submissions;
            report.readback_completion_waits = execution.readback_completion_waits;
            report.readback_logical_bytes = execution.readback_logical_bytes;
            report.readback_staging_bytes = execution.readback_staging_bytes;
            report.readback_mode =
                (execution.readback_submissions != 0).then_some(CpuReadbackMode::StagedCopy);
            report.output_hash = execution.output_hash;
            report.timing = Some(execution.timing);
        }
        Err(issue) => {
            report.status = issue.status();
            report.issue = Some(issue);
            report.adapter = backend.map(|value| value.adapter_info().name.clone());
        }
    }
    report
}

fn run_decode(
    path: &Path,
    backend: Option<&WgpuBackend>,
    options: &CodecRunOptions,
) -> std::result::Result<WorkloadExecution, CodecIssue> {
    validate_decode_path(path)?;
    let backend = backend.ok_or_else(|| {
        CodecIssue::new(
            CodecIssueKind::Unavailable,
            "wgpu_adapter",
            "no_adapter",
            "no compatible wgpu adapter was found",
        )
    })?;
    let encoded = Arc::<[u8]>::from(std::fs::read(path).map_err(|error| {
        CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "input",
            "read_failed",
            error.to_string(),
        )
    })?);
    let decoder = GpuDecoder::wgpu(backend.clone());
    let display = (options.output_target == OutputTarget::DisplayTexture)
        .then(|| DisplayPipeline::new(backend));
    let readback = (options.output_target == OutputTarget::CpuReadback)
        .then(|| ImageReadbackPipeline::new(backend));
    let max_frame_slots =
        NonZeroUsize::new(usize::try_from(options.workload.max_frame_slots).unwrap_or(usize::MAX))
            .expect("validated max-frame-slots is nonzero");
    let request = options
        .format
        .output_request()
        .map_err(decode_issue)?
        .with_max_frame_slots(max_frame_slots);
    let require_animation = options.workload.kind == WorkloadKind::Animation;

    execute_workload(options.workload, || {
        decode_once(
            &decoder,
            Arc::clone(&encoded),
            request.clone(),
            display.as_ref(),
            readback.as_ref(),
            require_animation,
        )
    })
}

fn run_encode(
    path: &Path,
    backend: Option<&WgpuBackend>,
    options: &CodecRunOptions,
) -> std::result::Result<WorkloadExecution, CodecIssue> {
    if options.workload.kind == WorkloadKind::Animation {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_encode_profile",
            "animation",
            "the executable GPU encoder does not support animation",
        ));
    }
    let prepared = prepare_gray8_encode(path, backend, options)?;
    execute_workload(options.workload, || {
        let encoded = prepared
            .encoder
            .encode_container(prepared.source.clone())
            .map_err(encode_issue)?;
        Ok(DecodeObservation {
            frame_count: 1,
            output_bytes: u64::try_from(encoded.len()).unwrap_or(u64::MAX),
            gpu_output_logical_bytes: 0,
            codec_submissions: 1,
            codec_completion_waits: 1,
            display_submissions: 0,
            display_completion_waits: 0,
            readback_submissions: 0,
            readback_completion_waits: 0,
            readback_logical_bytes: 0,
            readback_staging_bytes: 0,
            output_hash: Some(blake3::hash(&encoded).to_hex().to_string()),
            hash_mismatch: false,
        })
    })
}

struct PreparedGray8Encode {
    encoder: LosslessGray8Encoder,
    source: BufferImageSource,
    source_hash: String,
}

fn prepare_gray8_encode(
    path: &Path,
    backend: Option<&WgpuBackend>,
    options: &CodecRunOptions,
) -> std::result::Result<PreparedGray8Encode, CodecIssue> {
    let backend = backend.ok_or_else(no_adapter_issue)?;
    if options.format != GpuPixelFormat::U8 {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_encode_input",
            "input_format",
            "the executable encode profile currently accepts only VPI U8 non-color input",
        ));
    }
    let extent = options.extent.ok_or_else(|| {
        CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_encode_input",
            "missing_extent",
            "the gray8 encoder requires --extent WIDTHxHEIGHT or a corpus extent",
        )
    })?;
    if !(2..=256).contains(&extent.width) || !(2..=256).contains(&extent.height) {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_encode_profile",
            "extent",
            "the current gray8 profile requires width and height in 2..=256",
        ));
    }
    let mut bytes = std::fs::read(path).map_err(|error| {
        CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_encode_input",
            "read_failed",
            error.to_string(),
        )
    })?;
    let expected_len = u64::from(extent.width) * u64::from(extent.height);
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != expected_len {
        return Err(CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_encode_input",
            "byte_length",
            format!(
                "gray8 input requires {expected_len} bytes for {}x{}, received {}",
                extent.width,
                extent.height,
                bytes.len()
            ),
        ));
    }
    let source_hash = blake3::hash(&bytes).to_hex().to_string();
    let layout = ImageLayout::packed(
        Extent2d::new(extent.width, extent.height),
        options.format.pixel_format(),
    )
    .map_err(|error| {
        CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_encode_input",
            "layout",
            error.to_string(),
        )
    })?;
    let padded_len = bytes.len().div_ceil(4) * 4;
    bytes.resize(padded_len, 0);
    let buffer = Arc::new(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-gpu-harness gray8 source"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    let source = BufferImageSource::new(buffer, layout).map_err(encode_issue)?;
    let context = WgpuContext::from_backend(backend);
    Ok(PreparedGray8Encode {
        encoder: LosslessGray8Encoder::new(context),
        source,
        source_hash,
    })
}

fn run_round_trip(
    path: &Path,
    backend: Option<&WgpuBackend>,
    options: &CodecRunOptions,
) -> std::result::Result<WorkloadExecution, CodecIssue> {
    if options.workload.kind == WorkloadKind::Animation {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "round_trip_verification",
            "animation",
            "the executable GPU round-trip profile does not support animation",
        ));
    }
    let backend = backend.ok_or_else(no_adapter_issue)?;
    if options.output_target != OutputTarget::CpuReadback {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "round_trip_verification",
            "cpu_readback_required",
            "exact round-trip verification currently requires --output-target cpu-readback",
        ));
    }
    let prepared = prepare_gray8_encode(path, Some(backend), options)?;
    let decoder = GpuDecoder::wgpu(backend.clone());
    let request = options.format.output_request().map_err(decode_issue)?;
    let readback = ImageReadbackPipeline::new(backend);
    execute_workload(options.workload, || {
        let encoded = prepared
            .encoder
            .encode_container(prepared.source.clone())
            .map_err(encode_issue)?;
        let mut observation = decode_once(
            &decoder,
            Arc::from(encoded),
            request.clone(),
            None,
            Some(&readback),
            false,
        )?;
        if observation.output_hash.as_deref() != Some(prepared.source_hash.as_str()) {
            return Err(CodecIssue::new(
                CodecIssueKind::Verification,
                "round_trip_verification",
                "pixel_mismatch",
                "GPU decoded Gray8 bytes do not exactly match the encoder input",
            ));
        }
        observation.codec_submissions = observation.codec_submissions.saturating_add(1);
        observation.codec_completion_waits = observation.codec_completion_waits.saturating_add(1);
        Ok(observation)
    })
}

fn decode_once(
    decoder: &GpuDecoder<jxl_wgpu_decode::WgpuSubmissionEngine>,
    encoded: Arc<[u8]>,
    request: GpuOutputRequest,
    display: Option<&DisplayPipeline>,
    readback: Option<&ImageReadbackPipeline>,
    require_animation: bool,
) -> std::result::Result<DecodeObservation, CodecIssue> {
    let mut session = decoder
        .open_shared(encoded, request)
        .map_err(decode_issue)?;
    if require_animation && !session.metadata().is_animation() {
        return Err(CodecIssue::new(
            CodecIssueKind::Unsupported,
            "decode_profile",
            "animation_required",
            "the animation workload requires an animated JPEG XL codestream",
        ));
    }

    let mut observation = DecodeObservation::default();
    let mut readback_hash = blake3::Hasher::new();
    let mut readback_outputs = 0_u32;
    while let Some(frame) = session.next_frame().map_err(decode_issue)? {
        observation.frame_count = observation.frame_count.saturating_add(1);
        observation.codec_submissions = observation.codec_submissions.saturating_add(1);
        observation.codec_completion_waits = observation.codec_completion_waits.saturating_add(1);
        if let Some(readback) = readback {
            let result = readback
                .submit(frame.output())
                .map_err(readback_issue)?
                .wait()
                .map_err(readback_issue)?;
            observation.readback_logical_bytes = observation
                .readback_logical_bytes
                .saturating_add(result.stats.logical_bytes);
            observation.readback_staging_bytes = observation
                .readback_staging_bytes
                .saturating_add(result.stats.staging_bytes);
            for output in result.frame.outputs {
                readback_hash.update(&output.bytes);
                readback_outputs = readback_outputs.saturating_add(1);
            }
            observation.readback_submissions = observation.readback_submissions.saturating_add(1);
            observation.readback_completion_waits =
                observation.readback_completion_waits.saturating_add(1);
        }
        for output in &frame.output().outputs {
            observation.output_bytes = observation
                .output_bytes
                .saturating_add(output.layout.logical_size);
            observation.gpu_output_logical_bytes = observation
                .gpu_output_logical_bytes
                .saturating_add(output.layout.logical_size);
            if let Some(display) = display {
                display
                    .submit_image(output, DisplayTextureDescriptor::default())
                    .map_err(display_issue)?;
                observation.display_submissions = observation.display_submissions.saturating_add(1);
            }
        }
    }
    if readback_outputs != 0 {
        observation.output_hash = Some(readback_hash.finalize().to_hex().to_string());
    }
    Ok(observation)
}

fn execute_workload<F>(
    workload: WorkloadSpec,
    operation: F,
) -> std::result::Result<WorkloadExecution, CodecIssue>
where
    F: Fn() -> std::result::Result<DecodeObservation, CodecIssue> + Sync,
{
    let parallelism = workload.parallelism();
    for _ in 0..workload.warmup {
        execute_group(parallelism, &operation)?;
    }

    let wall_start = Instant::now();
    let measured = if workload.kind == WorkloadKind::Concurrent {
        execute_worker_stream(parallelism, workload.measured_groups(), &operation)?
    } else {
        let mut measured = Vec::new();
        for _ in 0..workload.measured_groups() {
            measured.extend(execute_group(parallelism, &operation)?);
        }
        measured
    };
    let wall_ns = nanos(wall_start.elapsed().as_nanos());
    let mut samples = Vec::with_capacity(measured.len());
    let mut aggregate = DecodeObservation::default();
    for (observation, duration) in measured {
        aggregate.add_assign(observation);
        samples.push(duration);
    }
    let timing = summarize_timings(&samples).map_err(|error| {
        CodecIssue::new(
            CodecIssueKind::Backend,
            "harness",
            "timing",
            error.to_string(),
        )
    })?;
    let operations = u64::try_from(samples.len()).unwrap_or(u64::MAX);
    let operations_per_second = if wall_ns == 0 {
        0.0
    } else {
        operations as f64 * 1_000_000_000.0 / wall_ns as f64
    };
    if aggregate.hash_mismatch {
        return Err(CodecIssue::new(
            CodecIssueKind::Backend,
            "gpu_encode",
            "nondeterministic_output",
            "identical measured inputs produced different encoded byte hashes",
        ));
    }
    Ok(WorkloadExecution {
        frame_count: aggregate.frame_count,
        output_bytes: aggregate.output_bytes,
        gpu_output_logical_bytes: aggregate.gpu_output_logical_bytes,
        codec_submissions: aggregate.codec_submissions,
        codec_completion_waits: aggregate.codec_completion_waits,
        display_submissions: aggregate.display_submissions,
        display_completion_waits: aggregate.display_completion_waits,
        readback_submissions: aggregate.readback_submissions,
        readback_completion_waits: aggregate.readback_completion_waits,
        readback_logical_bytes: aggregate.readback_logical_bytes,
        readback_staging_bytes: aggregate.readback_staging_bytes,
        output_hash: aggregate.output_hash,
        timing: CodecTiming {
            operation_latency: timing,
            workload: WorkloadTiming {
                operations,
                parallelism,
                execution_model: workload.execution_model(),
                wall_ns,
                operations_per_second,
            },
        },
    })
}

fn execute_group<F>(
    parallelism: u32,
    operation: &F,
) -> std::result::Result<Vec<(DecodeObservation, u64)>, CodecIssue>
where
    F: Fn() -> std::result::Result<DecodeObservation, CodecIssue> + Sync,
{
    if parallelism == 1 {
        let started = Instant::now();
        return operation()
            .map(|observation| vec![(observation, nanos(started.elapsed().as_nanos()))]);
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
                    let started = Instant::now();
                    operation()
                        .map(|observation| (observation, nanos(started.elapsed().as_nanos())))
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    CodecIssue::new(
                        CodecIssueKind::Backend,
                        "harness",
                        "worker_panic",
                        "a codec workload worker panicked",
                    )
                })?
            })
            .collect()
    })
}

fn execute_worker_stream<F>(
    workers: u32,
    operations_per_worker: u32,
    operation: &F,
) -> std::result::Result<Vec<(DecodeObservation, u64)>, CodecIssue>
where
    F: Fn() -> std::result::Result<DecodeObservation, CodecIssue> + Sync,
{
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
                        .map(|_| {
                            let started = Instant::now();
                            operation().map(|observation| {
                                (observation, nanos(started.elapsed().as_nanos()))
                            })
                        })
                        .collect::<std::result::Result<Vec<_>, _>>()
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().map_err(|_| {
                    CodecIssue::new(
                        CodecIssueKind::Backend,
                        "harness",
                        "worker_panic",
                        "a persistent codec workload worker panicked",
                    )
                })?
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(|per_worker| per_worker.into_iter().flatten().collect())
    })
}

fn encode_issue(error: EncodeError) -> CodecIssue {
    match error {
        EncodeError::Unsupported(error) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_encode",
            "unsupported_feature",
            error.to_string(),
        ),
        EncodeError::InvalidSource(detail) => CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_encode_input",
            "invalid_source",
            detail,
        ),
        EncodeError::InvalidConfiguration(detail) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_encode_profile",
            "configuration",
            detail,
        ),
        other => CodecIssue::new(
            CodecIssueKind::Backend,
            "gpu_encode",
            "execution",
            other.to_string(),
        ),
    }
}

fn no_adapter_issue() -> CodecIssue {
    CodecIssue::new(
        CodecIssueKind::Unavailable,
        "wgpu_adapter",
        "no_adapter",
        "no compatible wgpu adapter was found",
    )
}

fn display_issue(error: jxl_wgpu::Error) -> CodecIssue {
    match error {
        jxl_wgpu::Error::Unsupported(detail) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "display_pipeline",
            "pixel_format",
            detail,
        ),
        other => CodecIssue::new(
            CodecIssueKind::Backend,
            "display_pipeline",
            "display_submission",
            other.to_string(),
        ),
    }
}

fn readback_issue(error: jxl_wgpu::Error) -> CodecIssue {
    match error {
        jxl_wgpu::Error::ImageReadbackTransientLimit { required, limit } => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "cpu_readback",
            "transient_limit",
            format!("readback requires {required} transient bytes, limit is {limit}"),
        ),
        jxl_wgpu::Error::ImageReadbackDeviceLimit { required, limit } => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "cpu_readback",
            "device_limit",
            format!("readback requires {required} staging bytes, device limit is {limit}"),
        ),
        jxl_wgpu::Error::Unsupported(detail) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "cpu_readback",
            "platform",
            detail,
        ),
        jxl_wgpu::Error::ImageReadbackNoFrames
        | jxl_wgpu::Error::ImageReadbackFrameEmpty { .. }
        | jxl_wgpu::Error::ImageReadbackSourceUsage { .. }
        | jxl_wgpu::Error::ImageReadbackSourceSize { .. }
        | jxl_wgpu::Error::ImageLayout(_) => CodecIssue::new(
            CodecIssueKind::Backend,
            "cpu_readback",
            "frame_contract",
            error.to_string(),
        ),
        other => CodecIssue::new(
            CodecIssueKind::Backend,
            "cpu_readback",
            "execution",
            other.to_string(),
        ),
    }
}

fn classify_extent(extent: Option<DeclaredExtent>) -> SizeClass {
    let Some(extent) = extent else {
        return SizeClass::Auto;
    };
    if extent.width % 2 != 0 || extent.height % 2 != 0 {
        SizeClass::Odd
    } else if u64::from(extent.width) * u64::from(extent.height) >= 1920 * 1080 {
        SizeClass::Large
    } else {
        SizeClass::Small
    }
}

fn validate_decode_path(path: &Path) -> std::result::Result<(), CodecIssue> {
    let is_jxl = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jxl"));
    if is_jxl {
        Ok(())
    } else {
        Err(CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "input",
            "file_extension",
            format!("{} is not a .jxl file", path.display()),
        ))
    }
}

fn decode_issue(error: DecodeError) -> CodecIssue {
    match error {
        DecodeError::UnsupportedProfile(error) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_decode_profile",
            format!("{:?}", error.feature),
            error.detail,
        ),
        DecodeError::FrontendIncomplete(error) => CodecIssue::new(
            CodecIssueKind::Incomplete,
            "gpu_decode_frontend",
            format!("{:?}", error.stage),
            error.detail,
        ),
        DecodeError::Bitstream(error) => CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "jxl_bitstream",
            "parse",
            error.to_string(),
        ),
        DecodeError::PixelFormat(error) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "pixel_format",
            "invalid_format",
            error.to_string(),
        ),
        DecodeError::ImageLayout(error) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "pixel_format",
            "image_layout",
            error.to_string(),
        ),
        DecodeError::UnsupportedOutputFormat(detail) => CodecIssue::new(
            CodecIssueKind::Unsupported,
            "gpu_decode_output",
            "pixel_format",
            detail,
        ),
        DecodeError::AccelerationIndex(error) => CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_acceleration_index",
            "invalid_index",
            error.to_string(),
        ),
        DecodeError::DuplicateAccelerationIndex => CodecIssue::new(
            CodecIssueKind::InvalidInput,
            "gpu_acceleration_index",
            "duplicate_index",
            "the input contains more than one GPU acceleration index",
        ),
        DecodeError::Backend(message) => CodecIssue::new(
            CodecIssueKind::Backend,
            "gpu_decode_backend",
            "execution",
            message,
        ),
        other => CodecIssue::new(
            CodecIssueKind::Backend,
            "gpu_decode_session",
            "contract",
            other.to_string(),
        ),
    }
}

const fn nanos(value: u128) -> u64 {
    if value > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}

#[derive(Clone, Debug, Default)]
struct DecodeObservation {
    frame_count: u32,
    output_bytes: u64,
    gpu_output_logical_bytes: u64,
    codec_submissions: u64,
    codec_completion_waits: u64,
    display_submissions: u64,
    display_completion_waits: u64,
    readback_submissions: u64,
    readback_completion_waits: u64,
    readback_logical_bytes: u64,
    readback_staging_bytes: u64,
    output_hash: Option<String>,
    hash_mismatch: bool,
}

impl DecodeObservation {
    fn add_assign(&mut self, other: Self) {
        self.frame_count = self.frame_count.saturating_add(other.frame_count);
        self.output_bytes = self.output_bytes.saturating_add(other.output_bytes);
        self.gpu_output_logical_bytes = self
            .gpu_output_logical_bytes
            .saturating_add(other.gpu_output_logical_bytes);
        self.codec_submissions = self
            .codec_submissions
            .saturating_add(other.codec_submissions);
        self.codec_completion_waits = self
            .codec_completion_waits
            .saturating_add(other.codec_completion_waits);
        self.display_submissions = self
            .display_submissions
            .saturating_add(other.display_submissions);
        self.display_completion_waits = self
            .display_completion_waits
            .saturating_add(other.display_completion_waits);
        self.readback_submissions = self
            .readback_submissions
            .saturating_add(other.readback_submissions);
        self.readback_completion_waits = self
            .readback_completion_waits
            .saturating_add(other.readback_completion_waits);
        self.readback_logical_bytes = self
            .readback_logical_bytes
            .saturating_add(other.readback_logical_bytes);
        self.readback_staging_bytes = self
            .readback_staging_bytes
            .saturating_add(other.readback_staging_bytes);
        self.hash_mismatch |= other.hash_mismatch;
        match (&self.output_hash, other.output_hash) {
            (Some(left), Some(right)) if left != &right => self.hash_mismatch = true,
            (None, Some(hash)) => self.output_hash = Some(hash),
            _ => {}
        }
    }
}

struct WorkloadExecution {
    frame_count: u32,
    output_bytes: u64,
    gpu_output_logical_bytes: u64,
    codec_submissions: u64,
    codec_completion_waits: u64,
    display_submissions: u64,
    display_completion_waits: u64,
    readback_submissions: u64,
    readback_completion_waits: u64,
    readback_logical_bytes: u64,
    readback_staging_bytes: u64,
    output_hash: Option<String>,
    timing: CodecTiming,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_format_inventory_is_valid() {
        let formats = [
            GpuPixelFormat::U8,
            GpuPixelFormat::S8,
            GpuPixelFormat::U16,
            GpuPixelFormat::U32,
            GpuPixelFormat::S32,
            GpuPixelFormat::S16,
            GpuPixelFormat::TwoS16,
            GpuPixelFormat::F32,
            GpuPixelFormat::F64,
            GpuPixelFormat::TwoF32,
            GpuPixelFormat::Y8,
            GpuPixelFormat::Y16,
            GpuPixelFormat::I420,
            GpuPixelFormat::I422,
            GpuPixelFormat::I444,
            GpuPixelFormat::Nv12,
            GpuPixelFormat::Nv21,
            GpuPixelFormat::Nv16,
            GpuPixelFormat::Nv61,
            GpuPixelFormat::Nv24,
            GpuPixelFormat::Nv42,
            GpuPixelFormat::P010,
            GpuPixelFormat::P012,
            GpuPixelFormat::P016,
            GpuPixelFormat::Yuyv,
            GpuPixelFormat::Uyvy,
            GpuPixelFormat::Rgb8,
            GpuPixelFormat::Bgr8,
            GpuPixelFormat::Rgba8,
            GpuPixelFormat::Bgra8,
            GpuPixelFormat::Rgb8Planar,
            GpuPixelFormat::Bgr8Planar,
            GpuPixelFormat::Rgba8Planar,
            GpuPixelFormat::Bgra8Planar,
        ];
        for format in formats {
            format.pixel_format().validate().unwrap();
            format.output_request().unwrap();
        }
    }

    #[test]
    fn workload_shapes_cover_single_warm_burst_and_concurrent() {
        let mut spec = WorkloadSpec::default();
        assert_eq!(spec.total_measured_operations().unwrap(), 1);
        spec.kind = WorkloadKind::WarmSequential;
        spec.iterations = 7;
        assert_eq!(spec.parallelism(), 1);
        assert_eq!(spec.total_measured_operations().unwrap(), 7);
        spec.kind = WorkloadKind::ConcurrentBurst;
        spec.burst_size = 3;
        assert_eq!(spec.total_measured_operations().unwrap(), 21);
        assert_eq!(
            spec.execution_model(),
            WorkloadExecutionModel::BarrierSynchronizedHostFanout
        );
        spec.kind = WorkloadKind::Concurrent;
        spec.concurrency = 4;
        assert_eq!(spec.total_measured_operations().unwrap(), 28);
    }

    #[test]
    fn simultaneous_executor_preserves_all_results() {
        let spec = WorkloadSpec {
            kind: WorkloadKind::Concurrent,
            warmup: 1,
            iterations: 3,
            concurrency: 4,
            ..WorkloadSpec::default()
        };
        let execution = execute_workload(spec, || {
            Ok(DecodeObservation {
                frame_count: 1,
                output_bytes: 16,
                gpu_output_logical_bytes: 16,
                codec_submissions: 1,
                codec_completion_waits: 1,
                display_submissions: 0,
                display_completion_waits: 0,
                readback_submissions: 0,
                readback_completion_waits: 0,
                readback_logical_bytes: 0,
                readback_staging_bytes: 0,
                output_hash: None,
                hash_mismatch: false,
            })
        })
        .unwrap();
        assert_eq!(execution.timing.workload.operations, 12);
        assert_eq!(execution.timing.workload.parallelism, 4);
        assert_eq!(execution.frame_count, 12);
        assert_eq!(execution.output_bytes, 192);
        assert_eq!(execution.gpu_output_logical_bytes, 192);
        assert_eq!(execution.codec_submissions, 12);
        assert_eq!(execution.codec_completion_waits, 12);
        assert_eq!(
            execution.timing.workload.execution_model,
            WorkloadExecutionModel::PersistentHostWorkers
        );
    }

    #[test]
    fn concurrent_burst_reports_host_fanout_without_gpu_batching() {
        let spec = WorkloadSpec {
            kind: WorkloadKind::ConcurrentBurst,
            iterations: 2,
            burst_size: 3,
            ..WorkloadSpec::default()
        };
        let execution = execute_workload(spec, || {
            Ok(DecodeObservation {
                frame_count: 1,
                output_bytes: 4,
                gpu_output_logical_bytes: 4,
                codec_submissions: 1,
                codec_completion_waits: 1,
                ..DecodeObservation::default()
            })
        })
        .unwrap();
        assert_eq!(execution.timing.workload.operations, 6);
        assert_eq!(execution.codec_submissions, 6);
        assert_eq!(execution.codec_completion_waits, 6);
        assert_eq!(
            execution.timing.workload.execution_model,
            WorkloadExecutionModel::BarrierSynchronizedHostFanout
        );
    }

    #[test]
    fn explicit_cpu_readback_is_a_separate_output_target() {
        assert_ne!(OutputTarget::CpuReadback, OutputTarget::GpuResident);
        assert_eq!(
            serde_json::to_string(&OutputTarget::CpuReadback).unwrap(),
            "\"cpu_readback\""
        );
    }

    #[test]
    fn actual_gray8_encode_workload_runs_when_an_adapter_exists() {
        let Some(backend) = request_backend().unwrap() else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/basic.jxl");
        let case = CodecCorpusCase {
            name: "gray8_odd".into(),
            path,
            size_class: SizeClass::Odd,
            extent: Some(DeclaredExtent {
                width: 5,
                height: 13,
            }),
        };
        let options = CodecRunOptions {
            operation: CodecOperation::Encode,
            workload: WorkloadSpec::default(),
            output_target: OutputTarget::GpuResident,
            format: GpuPixelFormat::U8,
            size_class: SizeClass::Auto,
            extent: None,
        };
        let report = run_codec_case(&case, Some(&backend), &options);
        assert_eq!(report.status, CaseStatus::Passed);
        assert_eq!(report.codec_submissions, 1);
        assert_eq!(report.codec_completion_waits, 1);
        assert_eq!(report.gpu_output_logical_bytes, 0);
        assert!(!report.coalesced_gpu_batching);
        assert!(report.output_bytes > 0);
        assert!(report.output_hash.is_some());
    }

    #[test]
    fn actual_gray8_encode_decode_round_trip_is_exact_when_an_adapter_exists() {
        let Some(backend) = request_backend().unwrap() else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/basic.jxl");
        let case = CodecCorpusCase {
            name: "gray8_round_trip".into(),
            path,
            size_class: SizeClass::Odd,
            extent: Some(DeclaredExtent {
                width: 5,
                height: 13,
            }),
        };
        let options = CodecRunOptions {
            operation: CodecOperation::RoundTrip,
            workload: WorkloadSpec::default(),
            output_target: OutputTarget::CpuReadback,
            format: GpuPixelFormat::U8,
            size_class: SizeClass::Auto,
            extent: None,
        };
        let report = run_codec_case(&case, Some(&backend), &options);
        assert_eq!(report.status, CaseStatus::Passed, "{:?}", report.issue);
        assert_eq!(report.codec_submissions, 2);
        assert_eq!(report.codec_completion_waits, 2);
        assert_eq!(report.readback_submissions, 1);
        assert_eq!(report.readback_completion_waits, 1);
        assert_eq!(report.gpu_output_logical_bytes, 65);
        assert_eq!(report.readback_logical_bytes, 65);
        assert_eq!(report.readback_staging_bytes, 68);
        assert_eq!(report.readback_mode, Some(CpuReadbackMode::StagedCopy));
        assert!(!report.coalesced_gpu_batching);
        assert_eq!(report.output_bytes, 65);
        assert!(report.output_hash.is_some());
    }

    #[test]
    fn actual_gray8_decode_submits_native_nv12_display_without_host_readback() {
        let Some(backend) = request_backend().unwrap() else {
            return;
        };
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/basic.jxl");
        let encode_options = CodecRunOptions {
            operation: CodecOperation::Encode,
            workload: WorkloadSpec::default(),
            output_target: OutputTarget::GpuResident,
            format: GpuPixelFormat::U8,
            size_class: SizeClass::Odd,
            extent: Some(DeclaredExtent {
                width: 5,
                height: 13,
            }),
        };
        let prepared = prepare_gray8_encode(&path, Some(&backend), &encode_options).unwrap();
        let encoded = prepared.encoder.encode_container(prepared.source).unwrap();
        let decoder = GpuDecoder::wgpu(backend.clone());
        let display = DisplayPipeline::new(&backend);
        let observation = decode_once(
            &decoder,
            Arc::from(encoded),
            GpuPixelFormat::Nv12.output_request().unwrap(),
            Some(&display),
            None,
            false,
        )
        .unwrap();
        assert_eq!(observation.frame_count, 1);
        assert_eq!(observation.codec_submissions, 1);
        assert_eq!(observation.display_submissions, 1);
        assert!(observation.output_hash.is_none());
    }

    #[test]
    fn automatic_size_class_keeps_odd_and_large_cases_visible() {
        assert_eq!(
            classify_extent(Some(DeclaredExtent {
                width: 17,
                height: 18,
            })),
            SizeClass::Odd
        );
        assert_eq!(
            classify_extent(Some(DeclaredExtent {
                width: 1920,
                height: 1080,
            })),
            SizeClass::Large
        );
    }
}
