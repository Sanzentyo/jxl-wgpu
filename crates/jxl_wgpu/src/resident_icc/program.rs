//! One four-word header followed by four-word stage records (opcode, inputs, outputs,
//! payload word offset). Payloads use checked relative word addresses. Curve tables are
//! immutable and shared; no shader address is derived from unchecked profile bytes.
use std::collections::{BTreeMap, HashMap};

use jxl_gpu_protocol::icc::{IccClutInterpolation, IccCurveSegmentKind, IccStage};

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

fn encode(transform: &IccTransform, mut sink: Sink) -> Result<Sink, ResidentIccError> {
    let stages = transform.program().stages();
    sink.allocate(4 + stages.len() * 4)?;
    sink.word(0, stages.len() as u32);
    let mut curves: HashMap<&IccCurve, u32> = HashMap::new();
    let mut segmented = BTreeMap::new();
    let mut cluts = BTreeMap::new();
    for (index, stage) in stages.iter().enumerate() {
        let (opcode, payload) = match stage {
            IccStage::Curves {
                curves: selected,
                inverse,
            } => {
                let payload = sink.allocate(selected.len())?;
                for (c, curve) in selected.iter().enumerate() {
                    let offset = if let Some(offset) = curves.get(curve) {
                        *offset
                    } else {
                        let offset = legacy_curve(&mut sink, curve)?;
                        curves.insert(curve, offset);
                        offset
                    };
                    sink.word(payload + c as u32, offset);
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
                let payload = sink.allocate(matrix.offset().len() * (p + 1))?;
                for (r, offset) in matrix.offset().iter().enumerate() {
                    for (c, value) in matrix.matrix()[r * p..(r + 1) * p]
                        .iter()
                        .chain([offset])
                        .enumerate()
                    {
                        sink.finite(payload + (r * (p + 1) + c) as u32, *value)?;
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
                let payload = if let Some(payload) = cluts.get(&key) {
                    *payload
                } else {
                    let payload = sink.allocate(clut.grid().len() * 2)?;
                    let mut stride = clut.output_channels() as u32;
                    for (c, &grid) in clut.grid().iter().enumerate().rev() {
                        sink.word(payload + c as u32 * 2, u32::from(grid));
                        sink.word(payload + c as u32 * 2 + 1, stride);
                        stride = stride
                            .checked_mul(u32::from(grid))
                            .ok_or(ResidentIccError::Addressing)?;
                    }
                    sink.floats(clut.values())?;
                    cluts.insert(key, payload);
                    payload
                };
                let opcode = match clut.interpolation() {
                    IccClutInterpolation::Tetrahedral => Opcode::Clut,
                    IccClutInterpolation::Multilinear => Opcode::MultilinearClut,
                };
                (opcode, payload)
            }
            IccStage::SegmentedCurves(curves) => {
                let payload = sink.allocate(curves.len())?;
                for (c, curve) in curves.iter().enumerate() {
                    let key = curve.segments().as_ptr() as usize;
                    let offset = if let Some(offset) = segmented.get(&key) {
                        *offset
                    } else {
                        let offset = sink.allocate(1 + curve.segments().len() * 10)?;
                        sink.word(offset, curve.segments().len() as u32);
                        for (i, segment) in curve.segments().iter().enumerate() {
                            let record = offset + 1 + i as u32 * 10;
                            sink.word(record, segment.upper.to_bits());
                            sink.word(record + 1, segment.lower.to_bits());
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
                                    sink.word(record + 3, samples.len() as u32);
                                    let samples = sink.floats(samples)?;
                                    sink.word(record + 9, samples);
                                    (3, [0.0; 5])
                                }
                            };
                            sink.word(record + 2, kind);
                            for (i, value) in values.into_iter().enumerate() {
                                sink.word(record + 4 + i as u32, value.to_bits());
                            }
                        }
                        segmented.insert(key, offset);
                        offset
                    };
                    sink.word(payload + c as u32, offset);
                }
                (Opcode::SegmentedCurves, payload)
            }
            IccStage::LabToXyz => (Opcode::LabToXyz, 0),
            IccStage::XyzToLab => (Opcode::XyzToLab, 0),
        };
        let record = 4 + index as u32 * 4;
        sink.word(record, opcode as u32);
        sink.word(record + 1, stage.input_channels() as u32);
        sink.word(record + 2, stage.output_channels() as u32);
        sink.word(record + 3, payload);
    }
    Ok(sink)
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
