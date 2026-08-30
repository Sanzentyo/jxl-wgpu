use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::CAPTURE_SCHEMA_VERSION;
use crate::error::{Error, Result};

pub const CAPTURE_MAGIC: [u8; 8] = *b"JXLGPUC\0";
pub const DEFAULT_MAX_HEADER_BYTES: u64 = 1024 * 1024;
pub const DEFAULT_MAX_PAYLOAD_BYTES: u64 = 1024 * 1024 * 1024;

const FIXED_HEADER_BYTES: usize = 8 + 2 + 2 + 4 + 8 + 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureLimits {
    pub max_header_bytes: u64,
    pub max_payload_bytes: u64,
}

impl Default for CaptureLimits {
    fn default() -> Self {
        Self {
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
#[value(rename_all = "snake_case")]
pub enum OperationKind {
    Copy,
    Affine,
    Gaborish,
    Epf,
    Upsample,
    ChromaUpsample,
    YcbcrToRgb,
    PremultiplyAlpha,
}

impl OperationKind {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Copy => "copy",
            Self::Affine => "affine",
            Self::Gaborish => "gaborish",
            Self::Epf => "epf",
            Self::Upsample => "upsample",
            Self::ChromaUpsample => "chroma_upsample",
            Self::YcbcrToRgb => "ycbcr_to_rgb",
            Self::PremultiplyAlpha => "premultiply_alpha",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationSpec {
    pub kind: OperationKind,
    #[serde(default)]
    pub parameters: BTreeMap<String, f64>,
}

impl OperationSpec {
    pub fn parameter(&self, name: &str) -> Result<f64> {
        self.parameters.get(name).copied().ok_or_else(|| {
            Error::InvalidMetadata(format!(
                "operation {} is missing parameter {name}",
                self.kind.as_str()
            ))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    Exact,
    F32,
    F16Storage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SectionKind {
    Input,
    Expected,
    Parameter,
    Auxiliary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    F32,
    I32,
    U16,
    U8,
    Bytes,
}

impl DataType {
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::U16 => 2,
            Self::U8 | Self::Bytes => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    Planar,
    Interleaved,
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    pub width: u32,
    pub height: u32,
    pub channels: u16,
    /// Number of scalar elements between adjacent rows in a channel.
    pub row_stride: u64,
    /// Number of scalar elements between adjacent planar channels.
    pub channel_stride: u64,
    pub origin_x: i32,
    pub origin_y: i32,
    pub layout: TensorLayout,
}

impl TensorShape {
    pub fn planar(width: u32, height: u32, channels: u16) -> Result<Self> {
        let row_stride = u64::from(width);
        let channel_stride = row_stride
            .checked_mul(u64::from(height))
            .ok_or(Error::LengthOverflow)?;
        Ok(Self {
            width,
            height,
            channels,
            row_stride,
            channel_stride,
            origin_x: 0,
            origin_y: 0,
            layout: TensorLayout::Planar,
        })
    }

    pub fn minimum_elements(&self) -> Result<u64> {
        if self.width == 0 || self.height == 0 || self.channels == 0 {
            return Err(Error::InvalidTensor(
                "tensor dimensions and channel count must be nonzero".into(),
            ));
        }
        match self.layout {
            TensorLayout::Planar => {
                if self.row_stride < u64::from(self.width) {
                    return Err(Error::InvalidTensor(format!(
                        "row stride {} is shorter than width {}",
                        self.row_stride, self.width
                    )));
                }
                let minimum_channel_stride = self
                    .row_stride
                    .checked_mul(u64::from(self.height))
                    .ok_or(Error::LengthOverflow)?;
                if self.channel_stride < minimum_channel_stride {
                    return Err(Error::InvalidTensor(format!(
                        "channel stride {} is shorter than {}",
                        self.channel_stride, minimum_channel_stride
                    )));
                }
                self.channel_stride
                    .checked_mul(u64::from(self.channels))
                    .ok_or(Error::LengthOverflow)
            }
            TensorLayout::Interleaved => {
                let row_width = u64::from(self.width)
                    .checked_mul(u64::from(self.channels))
                    .ok_or(Error::LengthOverflow)?;
                if self.row_stride < row_width {
                    return Err(Error::InvalidTensor(format!(
                        "interleaved row stride {} is shorter than {}",
                        self.row_stride, row_width
                    )));
                }
                self.row_stride
                    .checked_mul(u64::from(self.height))
                    .ok_or(Error::LengthOverflow)
            }
            TensorLayout::Opaque => Err(Error::InvalidTensor(
                "opaque tensor has no scalar shape".into(),
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionDescriptor {
    pub id: u32,
    pub name: String,
    pub kind: SectionKind,
    pub data_type: DataType,
    pub element_count: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub blake3: String,
    pub tensor: Option<TensorShape>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureMetadata {
    pub schema_version: u16,
    pub case_id: String,
    pub operation: OperationSpec,
    pub seed: u64,
    pub precision: PrecisionMode,
    pub sections: Vec<SectionDescriptor>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureSection {
    pub id: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CaptureFile {
    pub metadata: CaptureMetadata,
    pub sections: Vec<CaptureSection>,
}

impl CaptureFile {
    pub fn new(
        case_id: impl Into<String>,
        operation: OperationSpec,
        seed: u64,
        precision: PrecisionMode,
    ) -> Self {
        Self {
            metadata: CaptureMetadata {
                schema_version: CAPTURE_SCHEMA_VERSION,
                case_id: case_id.into(),
                operation,
                seed,
                precision,
                sections: Vec::new(),
                tags: BTreeMap::new(),
            },
            sections: Vec::new(),
        }
    }

    pub fn add_section(
        &mut self,
        id: u32,
        name: impl Into<String>,
        kind: SectionKind,
        data_type: DataType,
        tensor: Option<TensorShape>,
        data: Vec<u8>,
    ) -> Result<()> {
        if self.sections.iter().any(|section| section.id == id) {
            return Err(Error::DuplicateSection(id));
        }
        let element_size = data_type.bytes_per_element();
        let data_len = u64::try_from(data.len()).map_err(|_| Error::LengthOverflow)?;
        if data_len % element_size != 0 {
            return Err(Error::InvalidSection(format!(
                "section {id} length {data_len} is not divisible by element size {element_size}"
            )));
        }
        let element_count = data_len / element_size;
        if let Some(shape) = &tensor {
            let expected = shape.minimum_elements()?;
            if expected != element_count {
                return Err(Error::InvalidSection(format!(
                    "section {id} contains {element_count} elements but its tensor requires {expected}"
                )));
            }
        }
        self.metadata.sections.push(SectionDescriptor {
            id,
            name: name.into(),
            kind,
            data_type,
            element_count,
            byte_offset: 0,
            byte_length: data_len,
            blake3: String::new(),
            tensor,
        });
        self.sections.push(CaptureSection { id, data });
        Ok(())
    }

    pub fn section(&self, id: u32) -> Result<&[u8]> {
        self.sections
            .iter()
            .find(|section| section.id == id)
            .map(|section| section.data.as_slice())
            .ok_or(Error::MissingSection(id))
    }

    pub fn descriptor(&self, id: u32) -> Result<&SectionDescriptor> {
        self.metadata
            .sections
            .iter()
            .find(|section| section.id == id)
            .ok_or(Error::MissingSection(id))
    }

    pub fn section_by_kind(&self, kind: SectionKind) -> Result<(&SectionDescriptor, &[u8])> {
        let descriptor = self
            .metadata
            .sections
            .iter()
            .find(|descriptor| descriptor.kind == kind)
            .ok_or_else(|| Error::InvalidMetadata(format!("missing {kind:?} section")))?;
        Ok((descriptor, self.section(descriptor.id)?))
    }

    /// Returns one uniquely named section of `kind`, or `None` when it is absent.
    pub fn section_by_name(
        &self,
        kind: SectionKind,
        name: &str,
    ) -> Result<Option<(&SectionDescriptor, &[u8])>> {
        let mut matches = self
            .metadata
            .sections
            .iter()
            .filter(|descriptor| descriptor.kind == kind && descriptor.name == name);
        let Some(descriptor) = matches.next() else {
            return Ok(None);
        };
        if matches.next().is_some() {
            return Err(Error::InvalidMetadata(format!(
                "more than one {kind:?} section is named {name}"
            )));
        }
        Ok(Some((descriptor, self.section(descriptor.id)?)))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut normalized = self.clone();
        normalized.normalize_descriptors()?;
        normalized.validate(CaptureLimits::default())?;

        let metadata_bytes = serde_json::to_vec(&normalized.metadata)?;
        let header_len = u32::try_from(metadata_bytes.len()).map_err(|_| Error::LengthOverflow)?;
        let payload_len =
            normalized
                .sections
                .iter()
                .try_fold(0_u64, |total, section| -> Result<u64> {
                    let length =
                        u64::try_from(section.data.len()).map_err(|_| Error::LengthOverflow)?;
                    total.checked_add(length).ok_or(Error::LengthOverflow)
                })?;
        let capacity = FIXED_HEADER_BYTES
            .checked_add(metadata_bytes.len())
            .and_then(|value| value.checked_add(usize::try_from(payload_len).ok()?))
            .ok_or(Error::LengthOverflow)?;
        let mut output = Vec::with_capacity(capacity);
        output.extend_from_slice(&CAPTURE_MAGIC);
        output.extend_from_slice(&CAPTURE_SCHEMA_VERSION.to_le_bytes());
        output.extend_from_slice(&0_u16.to_le_bytes());
        output.extend_from_slice(&header_len.to_le_bytes());
        output.extend_from_slice(&payload_len.to_le_bytes());
        output.extend_from_slice(blake3::hash(&metadata_bytes).as_bytes());
        output.extend_from_slice(&metadata_bytes);
        normalized
            .sections
            .iter()
            .for_each(|section| output.extend_from_slice(&section.data));
        Ok(output)
    }

    pub fn write_to(&self, mut writer: impl Write) -> Result<()> {
        writer
            .write_all(&self.to_bytes()?)
            .map_err(|source| Error::io("<writer>", source))
    }

    pub fn write_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|source| Error::io(path, source))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::read_with_limits(Cursor::new(bytes), CaptureLimits::default())
    }

    pub fn read_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|source| Error::io(path, source))?;
        Self::read_with_limits(file, CaptureLimits::default())
    }

    pub fn read_with_limits(mut reader: impl Read, limits: CaptureLimits) -> Result<Self> {
        let mut fixed = [0_u8; FIXED_HEADER_BYTES];
        reader
            .read_exact(&mut fixed)
            .map_err(|_| Error::Truncated("fixed header"))?;
        if fixed[..8] != CAPTURE_MAGIC {
            return Err(Error::InvalidMagic);
        }
        let version = u16::from_le_bytes([fixed[8], fixed[9]]);
        if version != CAPTURE_SCHEMA_VERSION {
            return Err(Error::UnsupportedSchema {
                found: version,
                expected: CAPTURE_SCHEMA_VERSION,
            });
        }
        let flags = u16::from_le_bytes([fixed[10], fixed[11]]);
        if flags != 0 {
            return Err(Error::UnsupportedFlags(flags));
        }
        let header_len = u64::from(u32::from_le_bytes(fixed[12..16].try_into().unwrap()));
        let payload_len = u64::from_le_bytes(fixed[16..24].try_into().unwrap());
        if header_len > limits.max_header_bytes {
            return Err(Error::HeaderTooLarge {
                actual: header_len,
                limit: limits.max_header_bytes,
            });
        }
        if payload_len > limits.max_payload_bytes {
            return Err(Error::PayloadTooLarge {
                actual: payload_len,
                limit: limits.max_payload_bytes,
            });
        }
        let mut metadata_bytes =
            vec![0_u8; usize::try_from(header_len).map_err(|_| Error::LengthOverflow)?];
        reader
            .read_exact(&mut metadata_bytes)
            .map_err(|_| Error::Truncated("metadata"))?;
        if blake3::hash(&metadata_bytes).as_bytes() != &fixed[24..56] {
            return Err(Error::HeaderHashMismatch);
        }
        let metadata: CaptureMetadata = serde_json::from_slice(&metadata_bytes)?;
        if metadata.schema_version != version {
            return Err(Error::InvalidMetadata(format!(
                "metadata schema {} disagrees with prefix schema {version}",
                metadata.schema_version
            )));
        }
        let payload_usize = usize::try_from(payload_len).map_err(|_| Error::LengthOverflow)?;
        let mut payload = vec![0_u8; payload_usize];
        reader
            .read_exact(&mut payload)
            .map_err(|_| Error::Truncated("payload"))?;
        let mut trailing = Vec::new();
        reader
            .take(limits.max_payload_bytes.saturating_add(1))
            .read_to_end(&mut trailing)
            .map_err(|source| Error::io("<reader>", source))?;
        if !trailing.is_empty() {
            return Err(Error::TrailingBytes {
                actual: u64::try_from(trailing.len()).unwrap_or(u64::MAX),
            });
        }

        let mut sections = Vec::with_capacity(metadata.sections.len());
        for descriptor in &metadata.sections {
            let start =
                usize::try_from(descriptor.byte_offset).map_err(|_| Error::LengthOverflow)?;
            let end = descriptor
                .byte_offset
                .checked_add(descriptor.byte_length)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(Error::LengthOverflow)?;
            let data = payload.get(start..end).ok_or_else(|| {
                Error::InvalidSection(format!(
                    "section {} range {}..{} exceeds payload length {}",
                    descriptor.id,
                    start,
                    end,
                    payload.len()
                ))
            })?;
            if blake3::hash(data).to_hex().as_str() != descriptor.blake3 {
                return Err(Error::SectionHashMismatch {
                    section_id: descriptor.id,
                });
            }
            sections.push(CaptureSection {
                id: descriptor.id,
                data: data.to_vec(),
            });
        }
        let capture = Self { metadata, sections };
        capture.validate(limits)?;
        Ok(capture)
    }

    pub fn validate(&self, limits: CaptureLimits) -> Result<()> {
        if self.metadata.schema_version != CAPTURE_SCHEMA_VERSION {
            return Err(Error::UnsupportedSchema {
                found: self.metadata.schema_version,
                expected: CAPTURE_SCHEMA_VERSION,
            });
        }
        if self.metadata.case_id.trim().is_empty() {
            return Err(Error::InvalidMetadata("case_id must not be empty".into()));
        }
        let mut descriptor_ids = BTreeSet::new();
        let mut data_ids = BTreeSet::new();
        let mut expected_offset = 0_u64;
        for descriptor in &self.metadata.sections {
            if !descriptor_ids.insert(descriptor.id) {
                return Err(Error::DuplicateSection(descriptor.id));
            }
            if descriptor.name.trim().is_empty() {
                return Err(Error::InvalidSection(format!(
                    "section {} has an empty name",
                    descriptor.id
                )));
            }
            if descriptor.byte_offset != expected_offset {
                return Err(Error::InvalidSection(format!(
                    "section {} starts at {}, expected {}",
                    descriptor.id, descriptor.byte_offset, expected_offset
                )));
            }
            let element_bytes = descriptor
                .element_count
                .checked_mul(descriptor.data_type.bytes_per_element())
                .ok_or(Error::LengthOverflow)?;
            if element_bytes != descriptor.byte_length {
                return Err(Error::InvalidSection(format!(
                    "section {} byte length {} disagrees with element count ({element_bytes})",
                    descriptor.id, descriptor.byte_length
                )));
            }
            if let Some(tensor) = &descriptor.tensor
                && tensor.minimum_elements()? != descriptor.element_count
            {
                return Err(Error::InvalidSection(format!(
                    "section {} tensor shape disagrees with element count",
                    descriptor.id
                )));
            }
            expected_offset = expected_offset
                .checked_add(descriptor.byte_length)
                .ok_or(Error::LengthOverflow)?;
        }
        if expected_offset > limits.max_payload_bytes {
            return Err(Error::PayloadTooLarge {
                actual: expected_offset,
                limit: limits.max_payload_bytes,
            });
        }
        for section in &self.sections {
            if !data_ids.insert(section.id) {
                return Err(Error::DuplicateSection(section.id));
            }
            let descriptor = self.descriptor(section.id)?;
            if u64::try_from(section.data.len()).map_err(|_| Error::LengthOverflow)?
                != descriptor.byte_length
            {
                return Err(Error::InvalidSection(format!(
                    "section {} data length disagrees with metadata",
                    section.id
                )));
            }
            if !descriptor.blake3.is_empty()
                && blake3::hash(&section.data).to_hex().as_str() != descriptor.blake3
            {
                return Err(Error::SectionHashMismatch {
                    section_id: section.id,
                });
            }
        }
        if descriptor_ids != data_ids {
            let missing = descriptor_ids
                .symmetric_difference(&data_ids)
                .next()
                .copied()
                .unwrap_or_default();
            return Err(Error::MissingSection(missing));
        }
        Ok(())
    }

    fn normalize_descriptors(&mut self) -> Result<()> {
        let by_id: BTreeMap<_, _> = self
            .sections
            .iter()
            .map(|section| (section.id, section.data.as_slice()))
            .collect();
        let mut offset = 0_u64;
        for descriptor in &mut self.metadata.sections {
            let data = by_id
                .get(&descriptor.id)
                .copied()
                .ok_or(Error::MissingSection(descriptor.id))?;
            descriptor.byte_offset = offset;
            descriptor.byte_length =
                u64::try_from(data.len()).map_err(|_| Error::LengthOverflow)?;
            descriptor.element_count =
                descriptor.byte_length / descriptor.data_type.bytes_per_element();
            descriptor.blake3 = blake3::hash(data).to_hex().to_string();
            offset = offset
                .checked_add(descriptor.byte_length)
                .ok_or(Error::LengthOverflow)?;
        }
        Ok(())
    }
}

pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

pub fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(Error::InvalidTensor(format!(
            "f32 payload length {} is not divisible by four",
            bytes.len()
        )));
    }
    Ok(chunks
        .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
        .collect())
}

pub fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(Error::InvalidTensor(format!(
            "i32 payload length {} is not divisible by four",
            bytes.len()
        )));
    }
    Ok(chunks
        .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_capture() -> CaptureFile {
        let mut capture = CaptureFile::new(
            "roundtrip",
            OperationSpec {
                kind: OperationKind::Copy,
                parameters: BTreeMap::new(),
            },
            7,
            PrecisionMode::Exact,
        );
        capture
            .add_section(
                1,
                "input",
                SectionKind::Input,
                DataType::F32,
                Some(TensorShape::planar(2, 2, 1).unwrap()),
                encode_f32(&[0.0, 1.0, 2.0, 3.0]),
            )
            .unwrap();
        capture
            .add_section(
                2,
                "expected",
                SectionKind::Expected,
                DataType::F32,
                Some(TensorShape::planar(2, 2, 1).unwrap()),
                encode_f32(&[0.0, 1.0, 2.0, 3.0]),
            )
            .unwrap();
        capture
    }

    #[test]
    fn canonical_roundtrip() {
        let encoded = sample_capture().to_bytes().unwrap();
        let decoded = CaptureFile::from_bytes(&encoded).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), encoded);
    }

    #[test]
    fn named_parameter_lookup_rejects_ambiguity() {
        let mut capture = sample_capture();
        for id in [3, 4] {
            capture
                .add_section(
                    id,
                    "sigma",
                    SectionKind::Parameter,
                    DataType::F32,
                    Some(TensorShape::planar(1, 1, 1).unwrap()),
                    encode_f32(&[-0.5]),
                )
                .unwrap();
        }
        assert!(matches!(
            capture.section_by_name(SectionKind::Parameter, "sigma"),
            Err(Error::InvalidMetadata(message)) if message.contains("more than one")
        ));
    }

    #[test]
    fn rejects_truncation_at_every_boundary() {
        let encoded = sample_capture().to_bytes().unwrap();
        for end in 0..encoded.len() {
            assert!(
                CaptureFile::from_bytes(&encoded[..end]).is_err(),
                "end={end}"
            );
        }
    }

    #[test]
    fn rejects_corrupt_metadata() {
        let mut encoded = sample_capture().to_bytes().unwrap();
        encoded[FIXED_HEADER_BYTES] ^= 0x40;
        assert!(matches!(
            CaptureFile::from_bytes(&encoded),
            Err(Error::HeaderHashMismatch)
        ));
    }

    #[test]
    fn rejects_corrupt_payload() {
        let mut encoded = sample_capture().to_bytes().unwrap();
        let last = encoded.len() - 1;
        encoded[last] ^= 0x40;
        assert!(matches!(
            CaptureFile::from_bytes(&encoded),
            Err(Error::SectionHashMismatch { .. })
        ));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut encoded = sample_capture().to_bytes().unwrap();
        encoded.push(0);
        assert!(matches!(
            CaptureFile::from_bytes(&encoded),
            Err(Error::TrailingBytes { actual: 1 })
        ));
    }

    #[test]
    fn rejects_configured_payload_limit() {
        let encoded = sample_capture().to_bytes().unwrap();
        let error = CaptureFile::read_with_limits(
            Cursor::new(encoded),
            CaptureLimits {
                max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
                max_payload_bytes: 1,
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::PayloadTooLarge { .. }));
    }

    #[test]
    fn f32_codec_preserves_bits() {
        let values = [0.0, -0.0, 1.0, f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let decoded = decode_f32(&encode_f32(&values)).unwrap();
        assert_eq!(
            decoded.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            values.map(f32::to_bits).to_vec(),
            "the capture codec must not canonicalize floating-point values"
        );
    }

    #[test]
    fn i32_codec_preserves_values() {
        let values = [i32::MIN, -1, 0, 1, i32::MAX];
        assert_eq!(decode_i32(&encode_i32(&values)).unwrap(), values);
    }
}
