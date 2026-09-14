use std::collections::BTreeMap;

use super::{Reader, finish, invalid, limit, positions};
use crate::icc::{
    IccCurveSegment, IccCurveSegmentKind, IccError, IccLimits, IccSegmentedCurve, IccSignature,
    IccStage,
};

pub(super) fn set(
    data: Reader<'_>,
    p: usize,
    q: usize,
    limits: IccLimits,
) -> Result<IccStage, IccError> {
    if p != q {
        return invalid("curve set channels", 8);
    }
    let mut decoded = BTreeMap::new();
    let mut curves = Vec::with_capacity(p);
    for position @ (offset, size) in positions(data, 12, p as u64)? {
        let curve = if let Some(curve) = decoded.get(&position) {
            curve
        } else {
            let curve = parse(
                Reader(data.slice(offset, offset + size, "segmented curve")?),
                limits,
            )?;
            decoded.entry(position).or_insert(curve)
        };
        curves.push(curve.clone());
    }
    Ok(IccStage::SegmentedCurves(curves.into()))
}

fn parse(data: Reader<'_>, limits: IccLimits) -> Result<IccSegmentedCurve, IccError> {
    if data.signature(0)? != IccSignature(*b"curf") {
        return invalid("segmented curve signature", 0);
    }
    data.zeros(4, 8, "reserved segmented curve")?;
    data.zeros(10, 12, "reserved segment count")?;
    let count = data.u16(8)?;
    if count == 0 {
        return invalid("empty segmented curve", 8);
    }
    limit(
        "curve segments",
        u64::from(count),
        u64::from(limits.max_curve_segments),
    )?;
    let mut breakpoints = Vec::with_capacity(usize::from(count) + 1);
    breakpoints.push(f32::NEG_INFINITY);
    for i in 0..u64::from(count - 1) {
        let value = data.f32(12 + i * 4)?;
        if value < *breakpoints.last().expect("initial breakpoint") {
            return invalid("decreasing curve breakpoints", 12 + i * 4);
        }
        breakpoints.push(value);
    }
    breakpoints.push(f32::INFINITY);
    let mut cursor = 12 + u64::from(count - 1) * 4;
    let mut segments: Vec<IccCurveSegment> = Vec::with_capacity(usize::from(count));
    for index in 0..usize::from(count) {
        let lower = breakpoints[index];
        let upper = breakpoints[index + 1];
        let signature = data.signature(cursor)?;
        data.zeros(cursor + 4, cursor + 8, "reserved curve segment")?;
        let kind = match &signature.0 {
            b"parf" => {
                let function = data.u16(cursor + 8)?;
                data.zeros(cursor + 10, cursor + 12, "reserved formula function")?;
                let n = match function {
                    0 => 4,
                    1 | 2 => 5,
                    _ => {
                        return Err(IccError::CurveFunction {
                            tag: signature,
                            function,
                        });
                    }
                };
                let mut values = [0.0; 5];
                for (i, value) in values.iter_mut().enumerate().take(n) {
                    *value = data.f32(cursor + 12 + i as u64 * 4)?;
                }
                cursor += 12 + n as u64 * 4;
                match (function, values) {
                    (0, [gamma, a, b, c, _]) => IccCurveSegmentKind::Power { gamma, a, b, c },
                    (1, [gamma, a, b, c, d]) => {
                        IccCurveSegmentKind::Logarithmic { gamma, a, b, c, d }
                    }
                    (_, [a, b, c, d, e]) => IccCurveSegmentKind::Exponential { a, b, c, d, e },
                }
            }
            b"samf" => {
                if index == 0 || index + 1 == usize::from(count) || lower == upper {
                    return invalid("unbounded or zero-width sampled segment", cursor);
                }
                let count = data.u32(cursor + 8)?;
                if count == 0 {
                    return invalid("empty sampled segment", cursor + 8);
                }
                limit(
                    "curve samples",
                    u64::from(count) + 1,
                    u64::from(limits.max_curve_samples),
                )?;
                let end = cursor + 12 + u64::from(count) * 4;
                data.slice(cursor + 12, end, "float curve samples")?;
                let mut samples = Vec::with_capacity(count as usize + 1);
                // ICC stores samples S1..Sn. S0 is the preceding segment's value at the
                // finite breakpoint. This is metadata reconstruction, never pixel evaluation.
                let initial = endpoint(segments.last().expect("preceding segment"), lower);
                if !initial.is_finite() {
                    return invalid("sampled segment initial value", cursor);
                }
                samples.push(initial);
                for i in 0..u64::from(count) {
                    samples.push(data.f32(cursor + 12 + i * 4)?);
                }
                cursor = end;
                IccCurveSegmentKind::Samples(samples.into())
            }
            _ => return invalid("curve segment signature", cursor),
        };
        validate_formula(&kind, lower, upper)?;
        segments.push(IccCurveSegment { lower, upper, kind });
    }
    finish(data, cursor)?;
    Ok(IccSegmentedCurve {
        segments: segments.into(),
    })
}

// Extended bounds are metadata, including which finite endpoints belong to the segment.
// They reject non-real formulas before upload without evaluating any image samples.
#[derive(Clone, Copy)]
struct Bounds {
    ends: [(f64, bool); 2],
}

impl Bounds {
    fn affine(self, a: f64, b: f64) -> Self {
        if a == 0.0 {
            return Self {
                ends: [(b, true); 2],
            };
        }
        let mut ends = self.ends.map(|(x, included)| (a * x + b, included));
        if a < 0.0 {
            ends.swap(0, 1);
        }
        Self { ends }
    }
    fn contains_zero(self) -> bool {
        let [(lo, li), (hi, ui)] = self.ends;
        (lo < 0.0 || (lo == 0.0 && li)) && (hi > 0.0 || (hi == 0.0 && ui))
    }
    fn power(self, gamma: f64) -> Result<Self, IccError> {
        if gamma == 0.0 {
            return Ok(Self {
                ends: [(1.0, true); 2],
            });
        }
        if (self.ends[0].0 < 0.0 && gamma.fract() != 0.0) || (gamma < 0.0 && self.contains_zero()) {
            return invalid("non-real segmented power", 0);
        }
        let mut ends = std::array::from_fn(|i| {
            let (mut x, included) = self.ends[i];
            if x == 0.0 && !included && self.ends[1 - i].0 < 0.0 {
                x = -0.0;
            }
            (x.powf(gamma), included)
        });
        if ends[0].0 > ends[1].0 {
            ends.swap(0, 1);
        }
        if gamma > 0.0 && self.contains_zero() && ends[0].0 >= 0.0 {
            ends[0] = (0.0, true);
        }
        Ok(Self { ends })
    }
}

fn validate_formula(kind: &IccCurveSegmentKind, lower: f32, upper: f32) -> Result<(), IccError> {
    if lower == upper {
        return Ok(());
    } // An empty segment is never selected.
    let domain = Bounds {
        ends: [
            (f64::from(lower), false),
            (f64::from(upper), upper.is_finite()),
        ],
    };
    match kind {
        IccCurveSegmentKind::Power { gamma, a, b, .. } => {
            domain
                .affine(f64::from(*a), f64::from(*b))
                .power(f64::from(*gamma))?;
        }
        IccCurveSegmentKind::Logarithmic { gamma, b, c, .. } => {
            let argument = domain
                .power(f64::from(*gamma))?
                .affine(f64::from(*b), f64::from(*c));
            let (minimum, included) = argument.ends[0];
            if minimum < 0.0 || (minimum == 0.0 && included) {
                return invalid("non-positive segmented logarithm", 0);
            }
        }
        IccCurveSegmentKind::Exponential { b, c, d, .. } => {
            if *b < 0.0 && (*c != 0.0 || d.fract() != 0.0) {
                return invalid("non-real segmented exponential", 0);
            }
            if *b == 0.0 {
                let (minimum, included) = domain.affine(f64::from(*c), f64::from(*d)).ends[0];
                if minimum < 0.0 || (minimum == 0.0 && included) {
                    return invalid("undefined segmented exponential", 0);
                }
            }
        }
        IccCurveSegmentKind::Samples(_) => {}
    }
    Ok(())
}

fn endpoint(segment: &IccCurveSegment, x: f32) -> f32 {
    let x = f64::from(x);
    (match &segment.kind {
        IccCurveSegmentKind::Power { gamma, a, b, c } => {
            (f64::from(*a) * x + f64::from(*b)).powf(f64::from(*gamma)) + f64::from(*c)
        }
        IccCurveSegmentKind::Logarithmic { gamma, a, b, c, d } => {
            f64::from(*a) * (f64::from(*b) * x.powf(f64::from(*gamma)) + f64::from(*c)).log10()
                + f64::from(*d)
        }
        IccCurveSegmentKind::Exponential { a, b, c, d, e } => {
            f64::from(*a) * f64::from(*b).powf(f64::from(*c) * x + f64::from(*d)) + f64::from(*e)
        }
        IccCurveSegmentKind::Samples(samples) => {
            f64::from(*samples.last().expect("nonempty preceding segment"))
        }
    }) as f32
}
