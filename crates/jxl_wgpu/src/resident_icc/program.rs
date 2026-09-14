//! One four-word header followed by four-word stage records (opcode, inputs, outputs,
//! payload word offset). Payloads use checked relative word addresses. Curve tables are
//! immutable and shared; no shader address is derived from unchecked profile bytes.
use std::collections::{BTreeMap, HashMap};

use jxl_gpu_protocol::icc::{IccClutInterpolation, IccCurveSegmentKind, IccProgram, IccStage};

use super::*;

#[repr(u32)]
enum Opcode {
    ForwardCurves = 0,
    InverseCurves = 1,
    Affine = 2,
    Clut = 3,
    SegmentedCurves = 4,
    LabToXyz = 5,
    XyzToLab = 6,
    ClampedAffine = 7,
    MultilinearClut = 8,
    BlackPointConnection = 9,
    RgbTransfer = 10,
    ToneMapping = 11,
}

pub(super) fn program_size(transform: &IccTransform, limit: u64) -> Result<u64, ResidentIccError> {
    Ok(encode(
        transform,
        Sink {
            bytes: None,
            length: 0,
            limit,
        },
    )?
    .length)
}

pub(super) fn lower_program(
    transform: &IccTransform,
    memory: ResidentIccMemoryPlan,
) -> Result<Vec<u8>, ResidentIccError> {
    let sink = encode(
        transform,
        Sink {
            bytes: Some(Vec::with_capacity(memory.program_bytes as usize)),
            length: 0,
            limit: memory.program_bytes,
        },
    )?;
    debug_assert_eq!(sink.length, memory.program_bytes);
    Ok(sink.bytes.expect("upload encoding"))
}

/// The same checked layout walk counts or writes metadata. Counting does not allocate or
/// traverse large sample payloads. Shared curves and CLUTs occupy one physical record.
struct Sink {
    bytes: Option<Vec<u8>>,
    length: u64,
    limit: u64,
}

impl Sink {
    fn allocate(&mut self, words: usize) -> Result<u32, ResidentIccError> {
        let start = self.length;
        let end = start
            .checked_add(words as u64 * 4)
            .ok_or(ResidentIccError::Addressing)?;
        if end > self.limit {
            return Err(ResidentIccError::Limit {
                resource: "program bytes",
                required: end,
                available: self.limit,
            });
        }
        if let Some(bytes) = &mut self.bytes {
            bytes.resize(end as usize, 0);
        }
        self.length = end;
        Ok((start / 4) as u32)
    }
    fn word(&mut self, offset: u32, value: u32) {
        if let Some(bytes) = &mut self.bytes {
            bytes[offset as usize * 4..offset as usize * 4 + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
    }
    fn floats(&mut self, values: &[f32]) -> Result<u32, ResidentIccError> {
        let offset = self.allocate(values.len())?;
        if self.bytes.is_some() {
            for (i, v) in values.iter().enumerate() {
                self.word(offset + i as u32, v.to_bits());
            }
        }
        Ok(offset)
    }
    fn finite(&mut self, offset: u32, value: f64) -> Result<(), ResidentIccError> {
        let value = value as f32;
        if !value.is_finite() {
            return Err(ResidentIccError::Precision);
        }
        self.word(offset, value.to_bits());
        Ok(())
    }
}

fn encode(transform: &IccTransform, sink: Sink) -> Result<Sink, ResidentIccError> {
    let mut encoder = Encoder {
        sink,
        curves: HashMap::new(),
        segmented: BTreeMap::new(),
        cluts: BTreeMap::new(),
    };
    let start = encoder.program(transform.program())?;
    debug_assert_eq!(start, 0);
    Ok(encoder.sink)
}

struct Encoder<'a> {
    sink: Sink,
    curves: HashMap<&'a IccCurve, u32>,
    segmented: BTreeMap<usize, u32>,
    cluts: BTreeMap<usize, u32>,
}

impl<'a> Encoder<'a> {
    fn program(&mut self, program: &'a IccProgram) -> Result<u32, ResidentIccError> {
        let stages = program.stages();
        let start = self.sink.allocate(4 + stages.len() * 4)?;
        self.sink.word(start, stages.len() as u32);
        for (index, stage) in stages.iter().enumerate() {
            let (opcode, payload) = match stage {
                IccStage::Curves {
                    curves: selected,
                    inverse,
                } => {
                    let payload = self.sink.allocate(selected.len())?;
                    for (c, curve) in selected.iter().enumerate() {
                        let offset = if let Some(offset) = self.curves.get(curve) {
                            *offset
                        } else {
                            let offset = legacy_curve(&mut self.sink, curve)?;
                            self.curves.insert(curve, offset);
                            offset
                        };
                        self.sink.word(payload + c as u32, offset);
                    }
                    (
                        if *inverse {
                            Opcode::InverseCurves
                        } else {
                            Opcode::ForwardCurves
                        },
                        payload,
                    )
                }
                IccStage::Matrix(matrix) => {
                    let p = matrix.input_channels();
                    let payload = self.sink.allocate(matrix.offset().len() * (p + 1))?;
                    for (r, offset) in matrix.offset().iter().enumerate() {
                        for (c, value) in matrix.matrix()[r * p..(r + 1) * p]
                            .iter()
                            .chain([offset])
                            .enumerate()
                        {
                            self.sink
                                .finite(payload + (r * (p + 1) + c) as u32, *value)?;
                        }
                    }
                    (
                        if matrix.clamp_output() {
                            Opcode::ClampedAffine
                        } else {
                            Opcode::Affine
                        },
                        payload,
                    )
                }
                IccStage::Clut(clut) => {
                    let key = clut.values().as_ptr() as usize;
                    let payload = if let Some(payload) = self.cluts.get(&key) {
                        *payload
                    } else {
                        let payload = self.sink.allocate(clut.grid().len() * 2)?;
                        let mut stride = clut.output_channels() as u32;
                        for (c, &grid) in clut.grid().iter().enumerate().rev() {
                            self.sink.word(payload + c as u32 * 2, u32::from(grid));
                            self.sink.word(payload + c as u32 * 2 + 1, stride);
                            stride = stride
                                .checked_mul(u32::from(grid))
                                .ok_or(ResidentIccError::Addressing)?;
                        }
                        self.sink.floats(clut.values())?;
                        self.cluts.insert(key, payload);
                        payload
                    };
                    let opcode = match clut.interpolation() {
                        IccClutInterpolation::Tetrahedral => Opcode::Clut,
                        IccClutInterpolation::Multilinear => Opcode::MultilinearClut,
                    };
                    (opcode, payload)
                }
                IccStage::SegmentedCurves(curves) => {
                    let payload = self.sink.allocate(curves.len())?;
                    for (c, curve) in curves.iter().enumerate() {
                        let key = curve.segments().as_ptr() as usize;
                        let offset = if let Some(offset) = self.segmented.get(&key) {
                            *offset
                        } else {
                            let offset = self.sink.allocate(1 + curve.segments().len() * 10)?;
                            self.sink.word(offset, curve.segments().len() as u32);
                            for (i, segment) in curve.segments().iter().enumerate() {
                                let record = offset + 1 + i as u32 * 10;
                                self.sink.word(record, segment.upper.to_bits());
                                self.sink.word(record + 1, segment.lower.to_bits());
                                let (kind, values) = match &segment.kind {
                                    IccCurveSegmentKind::Power { gamma, a, b, c } => {
                                        (0, [*gamma, *a, *b, *c, 0.0])
                                    }
                                    IccCurveSegmentKind::Logarithmic { gamma, a, b, c, d } => {
                                        (1, [*gamma, *a, *b, *c, *d])
                                    }
                                    IccCurveSegmentKind::Exponential { a, b, c, d, e } => {
                                        (2, [*a, *b, *c, *d, *e])
                                    }
                                    IccCurveSegmentKind::Samples(samples) => {
                                        self.sink.word(record + 3, samples.len() as u32);
                                        let samples = self.sink.floats(samples)?;
                                        self.sink.word(record + 9, samples);
                                        (3, [0.0; 5])
                                    }
                                };
                                self.sink.word(record + 2, kind);
                                for (i, value) in values.into_iter().enumerate() {
                                    self.sink.word(record + 4 + i as u32, value.to_bits());
                                }
                            }
                            self.segmented.insert(key, offset);
                            offset
                        };
                        self.sink.word(payload + c as u32, offset);
                    }
                    (Opcode::SegmentedCurves, payload)
                }
                IccStage::BlackPointConnection(connection) => {
                    let payload = self.sink.allocate(8 + connection.input().len())?;
                    let source = self.program(connection.source())?;
                    self.sink.word(payload, source);
                    self.sink.word(payload + 1, connection.input().len() as u32);
                    for (c, value) in connection.target().into_iter().enumerate() {
                        self.sink.finite(payload + 4 + c as u32, value)?;
                    }
                    for (c, value) in connection.input().iter().enumerate() {
                        self.sink.finite(payload + 8 + c as u32, *value)?;
                    }
                    self.sink.word(start + 1, payload);
                    (Opcode::BlackPointConnection, payload)
                }
                IccStage::RgbTransfer {
                    encoding,
                    intensity,
                    to_linear,
                } => {
                    let (code, gamma) = crate::image_output::transfer_parameters(encoding.transfer);
                    let luminance = intensity.map_or(Ok([0.0; 4]), |intensity| {
                        crate::display_luminance(encoding.space, intensity.nits(), !to_linear)
                            .map_err(|_| ResidentIccError::Precision)
                    })?;
                    let payload = self.sink.allocate(8)?;
                    self.sink.word(payload, code);
                    self.sink.word(payload + 1, gamma.to_bits());
                    self.sink.word(payload + 2, u32::from(*to_linear));
                    self.sink
                        .word(payload + 3, intensity.map_or(0.0, |v| v.nits()).to_bits());
                    for (c, value) in luminance.into_iter().enumerate() {
                        self.sink.word(payload + 4 + c as u32, value.to_bits());
                    }
                    (Opcode::RgbTransfer, payload)
                }
                IccStage::ToneMapping(mapping) => {
                    let params = crate::ToneMappingParams::new(*mapping)
                        .map_err(|_| ResidentIccError::Precision)?;
                    let words = bytemuck::cast_slice::<_, u32>(std::slice::from_ref(&params));
                    let payload = self.sink.allocate(words.len())?;
                    for (index, &word) in words.iter().enumerate() {
                        self.sink.word(payload + index as u32, word);
                    }
                    (Opcode::ToneMapping, payload)
                }
                IccStage::LabToXyz => (Opcode::LabToXyz, 0),
                IccStage::XyzToLab => (Opcode::XyzToLab, 0),
            };
            let record = start + 4 + index as u32 * 4;
            self.sink.word(record, opcode as u32);
            self.sink.word(record + 1, stage.input_channels() as u32);
            self.sink.word(record + 2, stage.output_channels() as u32);
            self.sink.word(record + 3, payload);
        }
        Ok(start)
    }
}

fn legacy_curve(sink: &mut Sink, curve: &IccCurve) -> Result<u32, ResidentIccError> {
    let offset = sink.allocate(12)?;
    let mut record = CurveParams::zeroed();
    record.selectors[3] =
        u32::from(curve.inverse_direction() == Ok(IccInverseDirection::Decreasing));
    match curve.kind() {
        IccCurveKind::Identity => {}
        IccCurveKind::Gamma(value) => {
            record.selectors[0] = 1;
            record.parameters[0][0] = f32::from(*value) / 256.0;
        }
        IccCurveKind::Sampled(samples) => {
            record.selectors[0] = 2;
            record.selectors[2] = samples.len() as u32;
        }
        IccCurveKind::Parametric {
            function,
            parameters,
        } => {
            let [g, a, b, _, d, _, _] = parameters.map(|v| f64::from(v) / 65536.0);
            let largest_base = if *function == 0 {
                1.0
            } else {
                (a + b).max(0.0)
            };
            if (*function <= 2 || d <= 1.0) && largest_base.powf(g) > f64::from(f32::MAX) / 2.0 {
                return Err(ResidentIccError::Precision);
            }
            record.selectors[0] = 3;
            record.selectors[1] = u32::from(*function);
            for (i, value) in parameters.iter().enumerate() {
                record.parameters[i / 4][i % 4] = *value as f32 / 65536.0;
            }
        }
    }
    for (i, word) in bytemuck::cast_slice::<_, u32>(bytemuck::bytes_of(&record))
        .iter()
        .enumerate()
    {
        sink.word(offset + i as u32, *word);
    }
    if let IccCurveKind::Sampled(samples) = curve.kind() {
        let start = sink.allocate(samples.len())?;
        if sink.bytes.is_some() {
            for (i, sample) in samples.iter().enumerate() {
                sink.word(start + i as u32, u32::from(*sample));
            }
        }
    }
    Ok(offset)
}
