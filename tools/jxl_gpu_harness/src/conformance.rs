//! Deterministic, row-bounded image corpus support for GPU codec conformance.
//!
//! The checked-in manifest describes both the executable stock GPU profile and explicit future
//! coverage. Pixel generation is lazy: even a UHD case allocates at most one padded row while
//! hashing or writing an input.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use clap::ValueEnum;
use jxl_gpu_formats::{Channel, ImageLayout, PitchLinearPlaneLayout, PixelFormat, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::ImageReadbackPipeline;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping};
use jxl_wgpu_encode::{BufferImageSource, LosslessModularEncoder, WgpuContext};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

use crate::codec::{
    CodecOperation, DeclaredExtent, GpuPixelFormat, OutputTarget, SizeClass, WorkloadSpec,
};
use crate::error::{Error, Result};
use crate::report::{CaseStatus, CodecCaseReport, CodecIssue, CodecIssueKind};

/// Schema version for the conformance manifest and JSON report.
pub const CONFORMANCE_SCHEMA_VERSION: u16 = 1;
/// Default maximum allocation made for one generated row.
pub const DEFAULT_MAX_ROW_BYTES: u64 = 64 * 1024 * 1024;
const MAX_STOCK_GPU_SOURCE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceAction {
    Inventory,
    GpuRoundTrip,
    ExternalFixtures,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceCorpus {
    pub version: u16,
    pub cases: Vec<ConformanceCase>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub category: ResolutionClass,
    pub extent: ImageExtent,
    pub source: SourceDescriptor,
    pub row_layout: RowLayoutSpec,
    pub pattern: PatternSpec,
    pub expectation: ConformanceExpectation,
    pub support_note: String,
    #[serde(default)]
    pub external_reference: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionClass {
    Tiny,
    Odd,
    Square,
    Portrait,
    Landscape,
    Panorama,
    Tall,
    GroupBoundary255,
    GroupBoundary256,
    GroupBoundary257,
    Hd,
    Fhd,
    Uhd4k,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageExtent {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDescriptor {
    pub model: PixelModel,
    pub depth: SampleDepth,
    pub alpha: AlphaPattern,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelModel {
    Gray,
    Rgb,
    Rgba,
}

impl PixelModel {
    #[must_use]
    pub const fn channels(self) -> u32 {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleDepth {
    U8,
    U10,
    U12,
    U16,
}

impl SampleDepth {
    #[must_use]
    pub const fn bits(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U10 => 10,
            Self::U12 => 12,
            Self::U16 => 16,
        }
    }

    #[must_use]
    pub const fn bytes_per_sample(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U10 | Self::U12 | Self::U16 => 2,
        }
    }

    #[must_use]
    pub const fn max_sample(self) -> u16 {
        match self {
            Self::U8 => u8::MAX as u16,
            Self::U10 => (1 << 10) - 1,
            Self::U12 => (1 << 12) - 1,
            Self::U16 => u16::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaPattern {
    None,
    Opaque,
    Checkerboard,
    HorizontalRamp,
    VerticalRamp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowLayoutSpec {
    pub alignment: u32,
    pub extra_padding: u32,
    pub padding_byte: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternSpec {
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceExpectation {
    StockGpuRoundTrip,
    FutureGpuProfile,
}

impl ConformanceCorpus {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path).map_err(|source| Error::io(path, source))?;
        let corpus: Self = toml::from_str(&source)?;
        corpus.validate()?;
        Ok(corpus)
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != CONFORMANCE_SCHEMA_VERSION {
            return Err(Error::InvalidConfig(format!(
                "conformance corpus version {} is unsupported; expected {}",
                self.version, CONFORMANCE_SCHEMA_VERSION
            )));
        }
        if self.cases.is_empty() {
            return Err(Error::InvalidConfig(
                "conformance corpus must contain at least one case".into(),
            ));
        }
        let mut names = BTreeSet::new();
        for case in &self.cases {
            case.validate()?;
            if !names.insert(case.name.as_str()) {
                return Err(Error::InvalidConfig(format!(
                    "duplicate conformance case name {}",
                    case.name
                )));
            }
        }
        Ok(())
    }

    pub fn select<'a>(&'a self, requested: &[String]) -> Result<Vec<&'a ConformanceCase>> {
        if requested.is_empty() {
            return Ok(self.cases.iter().collect());
        }
        let wanted = requested
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if wanted.len() != requested.len() {
            return Err(Error::InvalidConfig(
                "a conformance case was selected more than once".into(),
            ));
        }
        for name in &wanted {
            if !self.cases.iter().any(|case| case.name == *name) {
                return Err(Error::InvalidConfig(format!(
                    "unknown conformance case {name}"
                )));
            }
        }
        Ok(self
            .cases
            .iter()
            .filter(|case| wanted.contains(case.name.as_str()))
            .collect())
    }
}

impl ConformanceCase {
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || !self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(Error::InvalidConfig(format!(
                "conformance case name {:?} must contain only ASCII letters, digits, '_' or '-'",
                self.name
            )));
        }
        if self.extent.width == 0 || self.extent.height == 0 {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} has an empty extent",
                self.name
            )));
        }
        let category_matches_extent = match self.category {
            ResolutionClass::Tiny => self.extent.width <= 8 && self.extent.height <= 8,
            ResolutionClass::Odd => self.extent.width % 2 == 1 && self.extent.height % 2 == 1,
            ResolutionClass::Square => self.extent.width == self.extent.height,
            ResolutionClass::Portrait => self.extent.height > self.extent.width,
            ResolutionClass::Landscape => self.extent.width > self.extent.height,
            ResolutionClass::Panorama => {
                u64::from(self.extent.width) >= u64::from(self.extent.height) * 4
            }
            ResolutionClass::Tall => {
                u64::from(self.extent.height) >= u64::from(self.extent.width) * 4
            }
            ResolutionClass::GroupBoundary255 => {
                self.extent.width == 255 && self.extent.height == 255
            }
            ResolutionClass::GroupBoundary256 => {
                self.extent.width == 256 && self.extent.height == 256
            }
            ResolutionClass::GroupBoundary257 => {
                self.extent.width == 257 && self.extent.height == 257
            }
            ResolutionClass::Hd => self.extent.width == 1280 && self.extent.height == 720,
            ResolutionClass::Fhd => self.extent.width == 1920 && self.extent.height == 1080,
            ResolutionClass::Uhd4k => self.extent.width == 3840 && self.extent.height == 2160,
        };
        if !category_matches_extent {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} extent {}x{} does not match category {:?}",
                self.name, self.extent.width, self.extent.height, self.category
            )));
        }
        if self.row_layout.alignment == 0 || !self.row_layout.alignment.is_power_of_two() {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} row alignment must be a nonzero power of two",
                self.name
            )));
        }
        match (self.source.model, self.source.alpha) {
            (PixelModel::Rgba, AlphaPattern::None) => {
                return Err(Error::InvalidConfig(format!(
                    "conformance case {} must declare an alpha pattern for RGBA",
                    self.name
                )));
            }
            (PixelModel::Gray | PixelModel::Rgb, AlphaPattern::None) => {}
            (PixelModel::Gray | PixelModel::Rgb, _) => {
                return Err(Error::InvalidConfig(format!(
                    "conformance case {} declares alpha for a non-RGBA model",
                    self.name
                )));
            }
            (PixelModel::Rgba, _) => {}
        }
        if self.expectation == ConformanceExpectation::StockGpuRoundTrip
            && !self.is_stock_gpu_round_trip()
        {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} is marked stock, but the stock profile is Gray U8 with each dimension in 1..2^30",
                self.name
            )));
        }
        if self.external_reference && self.source.model == PixelModel::Rgba {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} requests external PNM reference generation for RGBA; keep RGBA inventory-only until a portable alpha fixture transport is selected",
                self.name
            )));
        }
        LazyImage::new(self, u64::MAX).map(|_| ())
    }

    #[must_use]
    pub fn is_stock_gpu_round_trip(&self) -> bool {
        self.source.model == PixelModel::Gray
            && self.source.depth == SampleDepth::U8
            && self.source.alpha == AlphaPattern::None
            && (1..(1_u32 << 30)).contains(&self.extent.width)
            && (1..(1_u32 << 30)).contains(&self.extent.height)
    }

    #[must_use]
    pub const fn size_class(&self) -> SizeClass {
        match self.category {
            ResolutionClass::Odd => SizeClass::Odd,
            ResolutionClass::Tiny => SizeClass::Small,
            ResolutionClass::Square
            | ResolutionClass::Portrait
            | ResolutionClass::Landscape
            | ResolutionClass::Panorama
            | ResolutionClass::Tall
            | ResolutionClass::GroupBoundary255
            | ResolutionClass::GroupBoundary256
            | ResolutionClass::GroupBoundary257
            | ResolutionClass::Hd
            | ResolutionClass::Fhd
            | ResolutionClass::Uhd4k => SizeClass::Large,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedLayout {
    pub active_row_bytes: u64,
    pub row_stride: u64,
    pub active_bytes: u64,
    pub storage_bytes: u64,
}

/// A validated image descriptor whose rows are generated on demand.
#[derive(Clone, Copy, Debug)]
pub struct LazyImage<'a> {
    case: &'a ConformanceCase,
    layout: GeneratedLayout,
}

impl<'a> LazyImage<'a> {
    pub fn new(case: &'a ConformanceCase, max_row_bytes: u64) -> Result<Self> {
        let samples_per_row = u64::from(case.extent.width)
            .checked_mul(u64::from(case.source.model.channels()))
            .ok_or(Error::LengthOverflow)?;
        let active_row_bytes = samples_per_row
            .checked_mul(u64::from(case.source.depth.bytes_per_sample()))
            .ok_or(Error::LengthOverflow)?;
        let alignment = u64::from(case.row_layout.alignment);
        let aligned = active_row_bytes
            .checked_add(alignment - 1)
            .ok_or(Error::LengthOverflow)?
            & !(alignment - 1);
        let row_stride = aligned
            .checked_add(u64::from(case.row_layout.extra_padding))
            .ok_or(Error::LengthOverflow)?;
        if row_stride > max_row_bytes {
            return Err(Error::InvalidConfig(format!(
                "conformance case {} needs a {row_stride}-byte row, above the configured {max_row_bytes}-byte row limit",
                case.name
            )));
        }
        usize::try_from(row_stride).map_err(|_| Error::LengthOverflow)?;
        let active_bytes = active_row_bytes
            .checked_mul(u64::from(case.extent.height))
            .ok_or(Error::LengthOverflow)?;
        let storage_bytes = row_stride
            .checked_mul(u64::from(case.extent.height))
            .ok_or(Error::LengthOverflow)?;
        Ok(Self {
            case,
            layout: GeneratedLayout {
                active_row_bytes,
                row_stride,
                active_bytes,
                storage_bytes,
            },
        })
    }

    #[must_use]
    pub const fn layout(self) -> GeneratedLayout {
        self.layout
    }

    #[must_use]
    pub const fn rows(self) -> LazyRows<'a> {
        LazyRows {
            image: self,
            next_row: 0,
        }
    }

    pub fn hashes(self) -> Result<GenerationHashes> {
        let mut input = blake3::Hasher::new();
        let mut pixels = blake3::Hasher::new();
        for row in self.rows() {
            let row = row?;
            input.update(row.storage());
            pixels.update(row.active());
        }
        Ok(GenerationHashes {
            input_hash: input.finalize().to_hex().to_string(),
            pixel_hash: pixels.finalize().to_hex().to_string(),
        })
    }

    pub fn write_padded_raw(self, mut writer: impl Write) -> Result<GenerationSummary> {
        self.write_rows(&mut writer, true)
    }

    pub fn write_active_raw(self, mut writer: impl Write) -> Result<GenerationSummary> {
        self.write_rows(&mut writer, false)
    }

    fn write_rows(self, writer: &mut impl Write, padded: bool) -> Result<GenerationSummary> {
        let mut input = blake3::Hasher::new();
        let mut pixels = blake3::Hasher::new();
        let mut written = 0_u64;
        for row in self.rows() {
            let row = row?;
            input.update(row.storage());
            pixels.update(row.active());
            let bytes = if padded { row.storage() } else { row.active() };
            writer
                .write_all(bytes)
                .map_err(|source| Error::io("<generated image>", source))?;
            written = written
                .checked_add(u64::try_from(bytes.len()).map_err(|_| Error::LengthOverflow)?)
                .ok_or(Error::LengthOverflow)?;
        }
        Ok(GenerationSummary {
            layout: self.layout,
            written_bytes: written,
            hashes: GenerationHashes {
                input_hash: input.finalize().to_hex().to_string(),
                pixel_hash: pixels.finalize().to_hex().to_string(),
            },
        })
    }

    /// Write PGM, PPM, or PAM without materializing the image.
    pub fn write_pnm(self, mut writer: impl Write) -> Result<GenerationSummary> {
        let max_sample = self.case.source.depth.max_sample();
        let header = match self.case.source.model {
            PixelModel::Gray => format!(
                "P5\n{} {}\n{}\n",
                self.case.extent.width, self.case.extent.height, max_sample
            ),
            PixelModel::Rgb => format!(
                "P6\n{} {}\n{}\n",
                self.case.extent.width, self.case.extent.height, max_sample
            ),
            PixelModel::Rgba => format!(
                "P7\nWIDTH {}\nHEIGHT {}\nDEPTH 4\nMAXVAL {}\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
                self.case.extent.width, self.case.extent.height, max_sample
            ),
        };
        writer
            .write_all(header.as_bytes())
            .map_err(|source| Error::io("<generated PNM>", source))?;

        let mut input = blake3::Hasher::new();
        let mut pixels = blake3::Hasher::new();
        let mut written_bytes = u64::try_from(header.len()).map_err(|_| Error::LengthOverflow)?;
        for row in self.rows() {
            let mut row = row?;
            input.update(row.storage());
            pixels.update(row.active());
            if self.case.source.depth.bytes_per_sample() == 1 {
                writer
                    .write_all(row.active())
                    .map_err(|source| Error::io("<generated PNM>", source))?;
            } else {
                for sample in row.bytes[..row.active_len].chunks_exact_mut(2) {
                    sample.swap(0, 1);
                }
                writer
                    .write_all(row.active())
                    .map_err(|source| Error::io("<generated PNM>", source))?;
            }
            written_bytes = written_bytes
                .checked_add(self.layout.active_row_bytes)
                .ok_or(Error::LengthOverflow)?;
        }
        Ok(GenerationSummary {
            layout: self.layout,
            written_bytes,
            hashes: GenerationHashes {
                input_hash: input.finalize().to_hex().to_string(),
                pixel_hash: pixels.finalize().to_hex().to_string(),
            },
        })
    }

    pub fn inventory(self) -> Result<ConformanceInventory> {
        Ok(ConformanceInventory {
            case: self.case.clone(),
            layout: self.layout,
            hashes: self.hashes()?,
        })
    }
}

#[derive(Debug)]
pub struct LazyRows<'a> {
    image: LazyImage<'a>,
    next_row: u32,
}

impl Iterator for LazyRows<'_> {
    type Item = Result<GeneratedRow>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_row >= self.image.case.extent.height {
            return None;
        }
        let row = generate_row(self.image, self.next_row);
        self.next_row += 1;
        Some(row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.image.case.extent.height - self.next_row;
        let remaining = usize::try_from(remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LazyRows<'_> {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedRow {
    index: u32,
    active_len: usize,
    bytes: Vec<u8>,
}

impl GeneratedRow {
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[must_use]
    pub fn active(&self) -> &[u8] {
        &self.bytes[..self.active_len]
    }

    #[must_use]
    pub fn storage(&self) -> &[u8] {
        &self.bytes
    }
}

fn generate_row(image: LazyImage<'_>, y: u32) -> Result<GeneratedRow> {
    let active_len =
        usize::try_from(image.layout.active_row_bytes).map_err(|_| Error::LengthOverflow)?;
    let row_stride = usize::try_from(image.layout.row_stride).map_err(|_| Error::LengthOverflow)?;
    let mut bytes = vec![image.case.row_layout.padding_byte; row_stride];
    let channels = image.case.source.model.channels();
    let sample_bytes = image.case.source.depth.bytes_per_sample();
    for x in 0..image.case.extent.width {
        for channel in 0..channels {
            let sample = if image.case.source.model == PixelModel::Rgba && channel == 3 {
                alpha_sample(image.case, x, y)
            } else {
                color_sample(image.case, x, y, channel)
            };
            let sample_index = u64::from(x)
                .checked_mul(u64::from(channels))
                .and_then(|value| value.checked_add(u64::from(channel)))
                .ok_or(Error::LengthOverflow)?;
            let offset = sample_index
                .checked_mul(u64::from(sample_bytes))
                .ok_or(Error::LengthOverflow)?;
            let offset = usize::try_from(offset).map_err(|_| Error::LengthOverflow)?;
            if sample_bytes == 1 {
                bytes[offset] = u8::try_from(sample).map_err(|_| Error::LengthOverflow)?;
            } else {
                bytes[offset..offset + 2].copy_from_slice(&sample.to_le_bytes());
            }
        }
    }
    Ok(GeneratedRow {
        index: y,
        active_len,
        bytes,
    })
}

fn color_sample(case: &ConformanceCase, x: u32, y: u32, channel: u32) -> u16 {
    let coordinate = u64::from(x).wrapping_mul(0x9e37_79b1_85eb_ca87)
        ^ u64::from(y).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ u64::from(channel).wrapping_mul(0x1656_67b1_9e37_79f9)
        ^ case.pattern.seed;
    let mixed = mix64(coordinate);
    let modulus = u64::from(case.source.depth.max_sample()) + 1;
    u16::try_from(mixed % modulus).expect("sample modulus is at most u16::MAX + 1")
}

fn alpha_sample(case: &ConformanceCase, x: u32, y: u32) -> u16 {
    let max = case.source.depth.max_sample();
    match case.source.alpha {
        AlphaPattern::None => 0,
        AlphaPattern::Opaque => max,
        AlphaPattern::Checkerboard => {
            if (x / 8 + y / 8).is_multiple_of(2) {
                0
            } else {
                max
            }
        }
        AlphaPattern::HorizontalRamp => scale_coordinate(x, case.extent.width, max),
        AlphaPattern::VerticalRamp => scale_coordinate(y, case.extent.height, max),
    }
}

fn scale_coordinate(coordinate: u32, length: u32, max: u16) -> u16 {
    let denominator = u128::from(length.saturating_sub(1).max(1));
    let scaled = u128::from(coordinate) * u128::from(max) / denominator;
    u16::try_from(scaled).unwrap_or(max)
}

const fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationHashes {
    /// BLAKE3 of the complete padded row storage, excluding any file-format header.
    pub input_hash: String,
    /// BLAKE3 of active interleaved samples in little-endian canonical form.
    pub pixel_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationSummary {
    pub layout: GeneratedLayout,
    pub written_bytes: u64,
    pub hashes: GenerationHashes,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceInventory {
    pub case: ConformanceCase,
    pub layout: GeneratedLayout,
    pub hashes: GenerationHashes,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformanceCaseReport {
    pub inventory: ConformanceInventory,
    pub gpu_round_trip: Option<CodecCaseReport>,
    pub external_fixture: Option<ExternalFixtureReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformanceReport {
    pub schema_version: u16,
    pub action: ConformanceAction,
    pub manifest: String,
    pub dry_run: bool,
    pub cases: Vec<ConformanceCaseReport>,
}

impl ConformanceReport {
    #[must_use]
    pub fn passed(&self) -> bool {
        match self.action {
            ConformanceAction::Inventory => true,
            ConformanceAction::GpuRoundTrip => {
                self.cases.iter().any(|case| {
                    case.inventory.case.expectation == ConformanceExpectation::StockGpuRoundTrip
                }) && self.cases.iter().all(|case| {
                    case.inventory.case.expectation != ConformanceExpectation::StockGpuRoundTrip
                        || matches!(
                            case.gpu_round_trip.as_ref().map(|report| &report.status),
                            Some(CaseStatus::Passed)
                        )
                })
            }
            ConformanceAction::ExternalFixtures => {
                self.dry_run
                    || (self
                        .cases
                        .iter()
                        .any(|case| case.inventory.case.external_reference)
                        && self.cases.iter().all(|case| {
                            !case.inventory.case.external_reference
                                || matches!(
                                    case.external_fixture.as_ref().map(|report| report.status),
                                    Some(ExternalFixtureStatus::Passed)
                                )
                        }))
            }
        }
    }
}

/// Execute the stock GPU profile using the manifest's real padded row layout.
pub fn run_stock_gpu_round_trip(
    case: &ConformanceCase,
    backend: Option<&jxl_wgpu::WgpuBackend>,
    max_row_bytes: u64,
) -> Result<CodecCaseReport> {
    if case.expectation != ConformanceExpectation::StockGpuRoundTrip
        || !case.is_stock_gpu_round_trip()
    {
        return Err(Error::InvalidConfig(format!(
            "case {} is outside the stock GPU round-trip profile",
            case.name
        )));
    }
    let image = LazyImage::new(case, max_row_bytes)?;
    let source_path = PathBuf::from(format!("<lazy-conformance:{}>", case.name));
    let mut report = CodecCaseReport::new(
        case.name.clone(),
        &source_path,
        CodecOperation::RoundTrip,
        WorkloadSpec::default(),
        OutputTarget::CpuReadback,
        GpuPixelFormat::U8,
        case.size_class(),
        Some(DeclaredExtent {
            width: case.extent.width,
            height: case.extent.height,
        }),
        image.layout().storage_bytes,
    );
    let Some(backend) = backend else {
        report.status = CaseStatus::Unavailable;
        report.issue = Some(CodecIssue::new(
            CodecIssueKind::Unavailable,
            "wgpu_adapter",
            "no_adapter",
            "no compatible wgpu adapter was found",
        ));
        return Ok(report);
    };
    report.adapter = Some(backend.adapter_info().name.clone());
    match execute_stock_gpu_round_trip(case, image, backend) {
        Ok(execution) => {
            report.frame_count = 1;
            report.output_bytes = image.layout().active_bytes;
            report.gpu_output_logical_bytes = execution.output_logical_bytes;
            report.codec_submissions = 2;
            report.codec_completion_waits = 2;
            report.readback_submissions = 1;
            report.readback_completion_waits = 1;
            report.readback_logical_bytes = execution.readback_logical_bytes;
            report.readback_staging_bytes = execution.readback_staging_bytes;
            report.readback_mode = Some(crate::codec::CpuReadbackMode::StagedCopy);
            report.output_hash = Some(execution.output_hash.clone());
            if execution.output_hash == execution.expected_hash {
                report.status = CaseStatus::Passed;
            } else {
                report.status = CaseStatus::Failed;
                report.issue = Some(CodecIssue::new(
                    CodecIssueKind::Verification,
                    "conformance_corpus",
                    "pixel_hash_mismatch",
                    format!(
                        "decoded hash {} differs from generated hash {}",
                        execution.output_hash, execution.expected_hash
                    ),
                ));
            }
        }
        Err(detail) => {
            report.status = CaseStatus::Error;
            report.issue = Some(CodecIssue::new(
                CodecIssueKind::Backend,
                "conformance_corpus",
                "gpu_round_trip",
                detail,
            ));
        }
    }
    Ok(report)
}

struct StockGpuExecution {
    expected_hash: String,
    output_hash: String,
    output_logical_bytes: u64,
    readback_logical_bytes: u64,
    readback_staging_bytes: u64,
}

fn execute_stock_gpu_round_trip(
    case: &ConformanceCase,
    image: LazyImage<'_>,
    backend: &jxl_wgpu::WgpuBackend,
) -> std::result::Result<StockGpuExecution, String> {
    if image.layout().storage_bytes > MAX_STOCK_GPU_SOURCE_BYTES {
        return Err(format!(
            "stock GPU source needs {} bytes, above the {}-byte materialization limit",
            image.layout().storage_bytes,
            MAX_STOCK_GPU_SOURCE_BYTES
        ));
    }
    let capacity = usize::try_from(image.layout().storage_bytes)
        .map_err(|_| "padded source size does not fit host usize".to_string())?;
    let mut padded = Vec::with_capacity(capacity);
    let generated = image
        .write_padded_raw(&mut padded)
        .map_err(|error| error.to_string())?;
    let buffer_len = padded.len().div_ceil(4) * 4;
    padded.resize(buffer_len, 0);

    let extent = Extent2d::new(case.extent.width, case.extent.height);
    let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
    let layout = ImageLayout::from_planes(
        extent,
        format.clone(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: generated.layout.row_stride,
            sample_extent: extent,
            row_bytes: generated.layout.active_row_bytes,
        }],
    )
    .map_err(|error| format!("invalid padded Gray8 GPU layout: {error}"))?;
    let source_buffer = Arc::new(backend.device().create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("jxl-gpu-harness conformance padded source"),
            contents: &padded,
            usage: wgpu::BufferUsages::STORAGE,
        },
    ));
    let source = BufferImageSource::new(source_buffer, layout)
        .map_err(|error| format!("invalid GPU encoder source: {error}"))?;
    let encoder = LosslessModularEncoder::new(WgpuContext::from_backend(backend));
    let encoded = encoder
        .encode_container(source)
        .map_err(|error| format!("GPU encode failed: {error}"))?;

    let request = GpuOutputRequest::numeric(format, NumericSampleMapping::NormalizedGray8)
        .map_err(|error| format!("GPU output request failed: {error}"))?;
    let decoder = GpuDecoder::wgpu(backend.clone());
    let mut session = decoder
        .open_shared(Arc::<[u8]>::from(encoded), request)
        .map_err(|error| format!("GPU decode session failed: {error}"))?;
    let frame = session
        .next_frame()
        .map_err(|error| format!("GPU decode failed: {error}"))?
        .ok_or_else(|| "GPU decoder returned no frame".to_string())?;
    let readback = ImageReadbackPipeline::new(backend);
    let submission = readback
        .submit(frame.output())
        .map_err(|error| format!("GPU readback submission failed: {error}"))?;
    let readback_stats = submission.stats();
    let result = submission
        .wait()
        .map_err(|error| format!("GPU readback failed: {error}"))?;
    let output_logical_bytes = result
        .frame
        .outputs
        .iter()
        .try_fold(0_u64, |total, output| {
            total.checked_add(output.layout.logical_size)
        })
        .ok_or_else(|| "decoded logical byte count overflow".to_string())?;
    let output_hash = hash_gray8_outputs(&result.frame.outputs, extent)?;
    drop(frame);
    if session
        .next_frame()
        .map_err(|error| format!("GPU decode tail validation failed: {error}"))?
        .is_some()
    {
        return Err("stock still-image decode returned more than one frame".into());
    }
    Ok(StockGpuExecution {
        expected_hash: generated.hashes.pixel_hash,
        output_hash,
        output_logical_bytes,
        readback_logical_bytes: readback_stats.logical_bytes,
        readback_staging_bytes: readback_stats.staging_bytes,
    })
}

fn hash_gray8_outputs(
    outputs: &[jxl_wgpu::CpuImageOutput],
    expected_extent: Extent2d,
) -> std::result::Result<String, String> {
    if outputs.len() != 1 {
        return Err(format!(
            "stock Gray8 decode returned {} outputs instead of one",
            outputs.len()
        ));
    }
    let output = &outputs[0];
    let expected_format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
    if output.layout.extent != expected_extent {
        return Err(format!(
            "decoded extent {:?} differs from expected {expected_extent:?}",
            output.layout.extent
        ));
    }
    let plane = output
        .layout
        .plane(0)
        .ok_or_else(|| "decoded Gray8 output has no plane".to_string())?;
    if output.layout.planes.len() != 1
        || output.layout.format != expected_format
        || plane.sample_extent != expected_extent
        || plane.row_bytes != u64::from(expected_extent.width)
    {
        return Err("decoded Gray8 plane layout is not one byte per active pixel".into());
    }
    let mut hasher = blake3::Hasher::new();
    for y in 0..expected_extent.height {
        let start = plane
            .offset
            .checked_add(
                plane
                    .row_stride
                    .checked_mul(u64::from(y))
                    .ok_or_else(|| "decoded row offset overflow".to_string())?,
            )
            .ok_or_else(|| "decoded row offset overflow".to_string())?;
        let end = start
            .checked_add(plane.row_bytes)
            .ok_or_else(|| "decoded row end overflow".to_string())?;
        let start = usize::try_from(start)
            .map_err(|_| "decoded row start does not fit host usize".to_string())?;
        let end = usize::try_from(end)
            .map_err(|_| "decoded row end does not fit host usize".to_string())?;
        let row = output
            .bytes
            .get(start..end)
            .ok_or_else(|| "decoded output is shorter than its declared layout".to_string())?;
        hasher.update(row);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalFixtureOptions {
    pub apply: bool,
    pub force: bool,
    pub output_dir: PathBuf,
    pub cjxl: PathBuf,
    pub djxl: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalFixtureStatus {
    Planned,
    Passed,
    NotApplicable,
    ToolUnavailable,
    ProcessFailed,
    VerificationFailed,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalFixtureReport {
    pub status: ExternalFixtureStatus,
    pub cjxl: String,
    pub djxl: String,
    pub source_path: String,
    pub jxl_path: String,
    pub decoded_path: String,
    pub jxl_bytes: Option<u64>,
    pub jxl_hash: Option<String>,
    pub decoded_pixel_hash: Option<String>,
    pub exact: Option<bool>,
    pub message: Option<String>,
}

/// Plan or create a standard JXL fixture through development-only `cjxl` and verify with `djxl`.
pub fn external_fixture(
    case: &ConformanceCase,
    max_row_bytes: u64,
    options: &ExternalFixtureOptions,
) -> Result<ExternalFixtureReport> {
    let extension = match case.source.model {
        PixelModel::Gray => "pgm",
        PixelModel::Rgb => "ppm",
        PixelModel::Rgba => "pam",
    };
    let source_path = options
        .output_dir
        .join(format!("{}.source.{extension}", case.name));
    let jxl_path = options.output_dir.join(format!("{}.jxl", case.name));
    let decoded_path = options
        .output_dir
        .join(format!("{}.decoded.{extension}", case.name));
    let mut report = ExternalFixtureReport {
        status: ExternalFixtureStatus::Planned,
        cjxl: options.cjxl.display().to_string(),
        djxl: options.djxl.display().to_string(),
        source_path: source_path.display().to_string(),
        jxl_path: jxl_path.display().to_string(),
        decoded_path: decoded_path.display().to_string(),
        jxl_bytes: None,
        jxl_hash: None,
        decoded_pixel_hash: None,
        exact: None,
        message: None,
    };
    if !case.external_reference {
        report.status = ExternalFixtureStatus::NotApplicable;
        report.message =
            Some("manifest keeps this case inventory-only for external reference tooling".into());
        return Ok(report);
    }
    if !options.apply {
        report.message = Some("dry run; pass --apply to create fixture files".into());
        return Ok(report);
    }
    std::fs::create_dir_all(&options.output_dir)
        .map_err(|source| Error::io(&options.output_dir, source))?;
    if !options.force {
        for path in [&source_path, &jxl_path, &decoded_path] {
            if path.exists() {
                return Err(Error::InvalidConfig(format!(
                    "{} already exists; pass --force to replace generated fixtures",
                    path.display()
                )));
            }
        }
    } else {
        for path in [&source_path, &jxl_path, &decoded_path] {
            if path.exists() {
                std::fs::remove_file(path).map_err(|source| Error::io(path, source))?;
            }
        }
    }
    let image = LazyImage::new(case, max_row_bytes)?;
    let mut source_writer = BufWriter::new(
        File::create(&source_path).map_err(|source| Error::io(&source_path, source))?,
    );
    let generated = image.write_pnm(&mut source_writer)?;
    source_writer
        .flush()
        .map_err(|source| Error::io(&source_path, source))?;

    let encode = Command::new(&options.cjxl)
        .arg(&source_path)
        .arg(&jxl_path)
        .arg("--distance=0")
        .arg("--effort=7")
        .output();
    let encode = match encode {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.status = ExternalFixtureStatus::ToolUnavailable;
            report.message = Some(format!("could not execute cjxl: {error}"));
            return Ok(report);
        }
        Err(error) => {
            report.status = ExternalFixtureStatus::Error;
            report.message = Some(format!("could not execute cjxl: {error}"));
            return Ok(report);
        }
    };
    if !encode.status.success() {
        report.status = ExternalFixtureStatus::ProcessFailed;
        report.message = Some(process_message("cjxl", &encode));
        return Ok(report);
    }

    let decode = Command::new(&options.djxl)
        .arg(&jxl_path)
        .arg(&decoded_path)
        .output();
    let decode = match decode {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.status = ExternalFixtureStatus::ToolUnavailable;
            report.message = Some(format!("could not execute djxl: {error}"));
            return Ok(report);
        }
        Err(error) => {
            report.status = ExternalFixtureStatus::Error;
            report.message = Some(format!("could not execute djxl: {error}"));
            return Ok(report);
        }
    };
    if !decode.status.success() {
        report.status = ExternalFixtureStatus::ProcessFailed;
        report.message = Some(process_message("djxl", &decode));
        return Ok(report);
    }

    let decoded = read_pnm_pixel_hash(&decoded_path, case, max_row_bytes)?;
    let exact = decoded.width == case.extent.width
        && decoded.height == case.extent.height
        && decoded.channels == case.source.model.channels()
        && decoded.max_sample == case.source.depth.max_sample()
        && decoded.pixel_hash.as_deref() == Some(generated.hashes.pixel_hash.as_str());
    let metadata = std::fs::metadata(&jxl_path).map_err(|source| Error::io(&jxl_path, source))?;
    report.jxl_bytes = Some(metadata.len());
    report.jxl_hash = Some(hash_file(&jxl_path)?);
    report.decoded_pixel_hash = decoded.pixel_hash;
    report.exact = Some(exact);
    report.status = if exact {
        ExternalFixtureStatus::Passed
    } else {
        ExternalFixtureStatus::VerificationFailed
    };
    if !exact {
        report.message = Some(
            "djxl output metadata or canonical pixel hash differs from the generated source".into(),
        );
    }
    Ok(report)
}

fn process_message(tool: &str, output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr)
        .chars()
        .take(4096)
        .collect::<String>();
    format!("{tool} exited with {}: {stderr}", output.status)
}

#[derive(Debug)]
struct DecodedPnm {
    width: u32,
    height: u32,
    channels: u32,
    max_sample: u16,
    pixel_hash: Option<String>,
}

fn read_pnm_pixel_hash(
    path: &Path,
    expected: &ConformanceCase,
    max_row_bytes: u64,
) -> Result<DecodedPnm> {
    let file = File::open(path).map_err(|source| Error::io(path, source))?;
    let mut reader = BufReader::new(file);
    let magic = read_pnm_token(&mut reader).map_err(|source| Error::io(path, source))?;
    let channels = match magic.as_str() {
        "P5" => 1_u32,
        "P6" => 3_u32,
        _ => {
            return Err(Error::InvalidConfig(format!(
                "{} is not a binary PGM/PPM decoded fixture",
                path.display()
            )));
        }
    };
    let width = parse_pnm_u32(path, "width", &read_pnm_token(&mut reader))?;
    let height = parse_pnm_u32(path, "height", &read_pnm_token(&mut reader))?;
    let max_sample_u32 = parse_pnm_u32(path, "max sample", &read_pnm_token(&mut reader))?;
    let max_sample = u16::try_from(max_sample_u32).map_err(|_| {
        Error::InvalidConfig(format!(
            "{} has a PNM max sample above 65535",
            path.display()
        ))
    })?;
    if max_sample == 0 {
        return Err(Error::InvalidConfig(format!(
            "{} has a zero PNM max sample",
            path.display()
        )));
    }
    if width != expected.extent.width
        || height != expected.extent.height
        || channels != expected.source.model.channels()
        || max_sample != expected.source.depth.max_sample()
    {
        return Ok(DecodedPnm {
            width,
            height,
            channels,
            max_sample,
            pixel_hash: None,
        });
    }
    consume_pnm_separator(&mut reader).map_err(|source| Error::io(path, source))?;
    let sample_bytes = if max_sample <= u16::from(u8::MAX) {
        1_u64
    } else {
        2_u64
    };
    let row_bytes = u64::from(width)
        .checked_mul(u64::from(channels))
        .and_then(|value| value.checked_mul(sample_bytes))
        .ok_or(Error::LengthOverflow)?;
    if row_bytes > max_row_bytes {
        return Err(Error::InvalidConfig(format!(
            "decoded PNM row is {row_bytes} bytes, above the safety limit"
        )));
    }
    let row_bytes = usize::try_from(row_bytes).map_err(|_| Error::LengthOverflow)?;
    let mut row = vec![0_u8; row_bytes];
    let mut hasher = blake3::Hasher::new();
    for _ in 0..height {
        reader
            .read_exact(&mut row)
            .map_err(|source| Error::io(path, source))?;
        if sample_bytes == 1 {
            hasher.update(&row);
        } else {
            for sample in row.chunks_exact(2) {
                hasher.update(&[sample[1], sample[0]]);
            }
        }
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|source| Error::io(path, source))?
        != 0
    {
        return Err(Error::InvalidConfig(format!(
            "{} has trailing PNM payload bytes",
            path.display()
        )));
    }
    Ok(DecodedPnm {
        width,
        height,
        channels,
        max_sample,
        pixel_hash: Some(hasher.finalize().to_hex().to_string()),
    })
}

fn read_pnm_token(reader: &mut impl BufRead) -> std::io::Result<String> {
    let mut token = Vec::new();
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        let byte = buffer[0];
        if token.is_empty() && byte == b'#' {
            let consumed = buffer
                .iter()
                .position(|candidate| *candidate == b'\n')
                .map_or(buffer.len(), |position| position + 1);
            reader.consume(consumed);
        } else if byte.is_ascii_whitespace() {
            if token.is_empty() {
                reader.consume(1);
            } else {
                break;
            }
        } else {
            token.push(byte);
            reader.consume(1);
        }
    }
    if token.is_empty() {
        Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "missing PNM header token",
        ))
    } else {
        String::from_utf8(token)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
    }
}

fn consume_pnm_separator(reader: &mut impl BufRead) -> std::io::Result<()> {
    let buffer = reader.fill_buf()?;
    let Some(separator) = buffer.first().copied() else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "missing PNM payload separator",
        ));
    };
    if !separator.is_ascii_whitespace() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid PNM payload separator",
        ));
    }
    reader.consume(1);
    if separator == b'\r' {
        let buffer = reader.fill_buf()?;
        if buffer.first() == Some(&b'\n') {
            reader.consume(1);
        }
    }
    Ok(())
}

fn parse_pnm_u32(path: &Path, field: &str, token: &std::io::Result<String>) -> Result<u32> {
    let token = token
        .as_ref()
        .map_err(|error| Error::io(path, std::io::Error::new(error.kind(), error.to_string())))?;
    token.parse::<u32>().map_err(|error| {
        Error::InvalidConfig(format!(
            "{} has an invalid PNM {field}: {error}",
            path.display()
        ))
    })
}

fn hash_file(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|source| Error::io(path, source))?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0_u8; 64 * 1024];
    let mut hasher = blake3::Hasher::new();
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|source| Error::io(path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_case(model: PixelModel, depth: SampleDepth, alpha: AlphaPattern) -> ConformanceCase {
        ConformanceCase {
            name: "test-case".into(),
            category: ResolutionClass::Odd,
            extent: ImageExtent {
                width: 5,
                height: 3,
            },
            source: SourceDescriptor {
                model,
                depth,
                alpha,
            },
            row_layout: RowLayoutSpec {
                alignment: 8,
                extra_padding: 3,
                padding_byte: 0xa5,
            },
            pattern: PatternSpec { seed: 7 },
            expectation: ConformanceExpectation::FutureGpuProfile,
            support_note: "test".into(),
            external_reference: model != PixelModel::Rgba,
        }
    }

    #[test]
    fn padded_rows_are_deterministic_and_exactly_sized() {
        let case = test_case(PixelModel::Rgb, SampleDepth::U8, AlphaPattern::None);
        let image = LazyImage::new(&case, 1024).unwrap();
        assert_eq!(image.layout().active_row_bytes, 15);
        assert_eq!(image.layout().row_stride, 19);
        assert_eq!(image.layout().storage_bytes, 57);
        let rows = image.rows().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|row| row.storage().len() == 19));
        assert!(rows.iter().all(|row| row.storage()[15..] == [0xa5; 4]));
        let hashes = image.hashes().unwrap();
        assert_eq!(hashes, image.hashes().unwrap());
        let mut changed_padding = case.clone();
        changed_padding.row_layout.padding_byte = 0x5a;
        let changed_hashes = LazyImage::new(&changed_padding, 1024)
            .unwrap()
            .hashes()
            .unwrap();
        assert_eq!(hashes.pixel_hash, changed_hashes.pixel_hash);
        assert_ne!(hashes.input_hash, changed_hashes.input_hash);
    }

    #[test]
    fn rgba_alpha_patterns_have_exact_endpoints() {
        let mut case = test_case(
            PixelModel::Rgba,
            SampleDepth::U10,
            AlphaPattern::HorizontalRamp,
        );
        case.row_layout = RowLayoutSpec {
            alignment: 1,
            extra_padding: 0,
            padding_byte: 0,
        };
        let image = LazyImage::new(&case, 1024).unwrap();
        let row = image.rows().next().unwrap().unwrap();
        let alpha = |x: usize| {
            let offset = (x * 4 + 3) * 2;
            u16::from_le_bytes([row.active()[offset], row.active()[offset + 1]])
        };
        assert_eq!(alpha(0), 0);
        assert_eq!(alpha(4), 1023);
    }

    #[test]
    fn uhd_generation_is_lazy_and_row_bounded() {
        let mut case = test_case(PixelModel::Rgba, SampleDepth::U16, AlphaPattern::Opaque);
        case.extent = ImageExtent {
            width: 3840,
            height: 2160,
        };
        case.row_layout = RowLayoutSpec {
            alignment: 256,
            extra_padding: 64,
            padding_byte: 0xcd,
        };
        let image = LazyImage::new(&case, DEFAULT_MAX_ROW_BYTES).unwrap();
        assert_eq!(image.layout().active_row_bytes, 30_720);
        assert!(image.layout().storage_bytes > 66_000_000);
        let row = image.rows().next().unwrap().unwrap();
        assert_eq!(row.storage().len(), 30_784);
    }

    #[test]
    fn pnm_uses_big_endian_samples_without_changing_canonical_hash() {
        let case = test_case(PixelModel::Gray, SampleDepth::U16, AlphaPattern::None);
        let image = LazyImage::new(&case, 1024).unwrap();
        let mut encoded = Vec::new();
        let summary = image.write_pnm(&mut encoded).unwrap();
        let header = b"P5\n5 3\n65535\n";
        assert!(encoded.starts_with(header));
        let first = image.rows().next().unwrap().unwrap();
        assert_eq!(encoded[header.len()], first.active()[1]);
        assert_eq!(encoded[header.len() + 1], first.active()[0]);
        assert_eq!(summary.written_bytes, encoded.len() as u64);
        assert_eq!(summary.hashes, image.hashes().unwrap());
    }

    #[test]
    fn invalid_alpha_and_row_limits_are_rejected() {
        let mut case = test_case(PixelModel::Rgb, SampleDepth::U8, AlphaPattern::Opaque);
        assert!(case.validate().is_err());
        case.source.alpha = AlphaPattern::None;
        assert!(LazyImage::new(&case, 1).is_err());
        case.category = ResolutionClass::Hd;
        assert!(case.validate().is_err());
    }
}
