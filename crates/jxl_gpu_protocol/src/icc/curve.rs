use std::sync::Arc;

use super::profile::{invalid, limit};
use super::{IccError, IccProfile, IccSignature};

/// Exact ICC curve data. Parameters are signed s15Fixed16, gamma is unsigned u8Fixed8,
/// and sampled values are unsigned 16-bit fractions with denominator 65535.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IccCurveKind {
    Identity,
    Gamma(u16),
    Sampled(Arc<[u16]>),
    Parametric { function: u16, parameters: [i32; 7] },
}

/// A validated curve, whose device domain and range are both [0, 1].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IccCurve {
    kind: IccCurveKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IccInverseDirection {
    Increasing,
    Decreasing,
}

impl IccCurve {
    #[must_use]
    pub const fn kind(&self) -> &IccCurveKind {
        &self.kind
    }

    pub(super) fn parse(profile: &IccProfile, tag: IccSignature) -> Result<Self, IccError> {
        let data = profile.required(tag)?;
        let kind = data.signature(0)?;
        let (kind, end) = match &kind.0 {
            b"curv" => {
                let count = data.u32(8)?;
                limit(
                    "curve samples",
                    u64::from(count),
                    u64::from(profile.max_curve_samples()),
                )?;
                let end = 12 + u64::from(count) * 2;
                let samples = data.slice(12, end, "curve samples")?;
                let curve = match count {
                    0 => IccCurveKind::Identity,
                    1 => {
                        let gamma = data.u16(12)?;
                        if gamma == 0 {
                            return Err(IccError::CurveParameters { tag });
                        }
                        IccCurveKind::Gamma(gamma)
                    }
                    _ => IccCurveKind::Sampled(
                        samples
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|v| u16::from_be_bytes(*v))
                            .collect(),
                    ),
                };
                (curve, end)
            }
            b"para" => {
                let function = data.u16(8)?;
                data.zeros(10, 12, "reserved curve function")?;
                let count = match function {
                    0 => 1,
                    1 => 3,
                    2 => 4,
                    3 => 5,
                    4 => 7,
                    _ => return Err(IccError::CurveFunction { tag, function }),
                };
                let mut parameters = [0; 7];
                for (index, parameter) in parameters.iter_mut().enumerate().take(count) {
                    *parameter = data.i32(12 + index as u64 * 4)?;
                }
                validate_parameters(function, parameters, tag)?;
                (
                    IccCurveKind::Parametric {
                        function,
                        parameters,
                    },
                    12 + count as u64 * 4,
                )
            }
            _ => return Err(IccError::TagType { tag, kind }),
        };
        if data.0.len() as u64 > end.next_multiple_of(4) {
            return invalid("curve element size", end);
        }
        data.zeros(end, data.0.len() as u64, "curve padding")?;
        Ok(Self { kind })
    }

    /// Validates inversion in the declared unit domain. Constant/nonmonotone curves are
    /// legitimate forward metadata but have no defined ICC inverse (Annex F.1).
    pub fn inverse_direction(&self) -> Result<IccInverseDirection, IccError> {
        let invalid = |reason| Err(IccError::CurveInverse { reason });
        match &self.kind {
            IccCurveKind::Identity | IccCurveKind::Gamma(_) => Ok(IccInverseDirection::Increasing),
            IccCurveKind::Sampled(samples) => {
                let first = samples[0];
                let last = samples[samples.len() - 1];
                if first == last {
                    return invalid("constant or nonmonotone sampled curve");
                }
                let increasing = first < last;
                if samples.windows(2).any(|pair| {
                    if increasing {
                        pair[0] > pair[1]
                    } else {
                        pair[0] < pair[1]
                    }
                }) {
                    return invalid("nonmonotone sampled curve");
                }
                Ok(if increasing {
                    IccInverseDirection::Increasing
                } else {
                    IccInverseDirection::Decreasing
                })
            }
            IccCurveKind::Parametric {
                function,
                parameters,
            } => {
                let p = parameters.map(|v| f64::from(v) / 65536.0);
                let (first, last) = (
                    parameter_value(*function, p, 0.0),
                    parameter_value(*function, p, 1.0),
                );
                if first > last && *function >= 3 && p[4] > 1.0 && p[3] < 0.0 {
                    return Ok(IccInverseDirection::Decreasing);
                }
                if first > last {
                    return invalid(
                        "decreasing parametric curves with an active power branch are not implemented",
                    );
                }
                if first == last {
                    return invalid("constant or nonmonotone parametric curve");
                }
                if *function >= 3 && p[4] > 0.0 {
                    if p[3] < 0.0 {
                        return invalid("decreasing lower branch");
                    }
                    if p[4] <= 1.0 {
                        let lower =
                            (p[3] * p[4] + if *function == 4 { p[6] } else { 0.0 }).clamp(0.0, 1.0);
                        let upper = parameter_value(*function, p, p[4]);
                        if lower > upper {
                            return invalid("decreasing parametric branch boundary");
                        }
                    }
                }
                Ok(IccInverseDirection::Increasing)
            }
        }
    }
}

// Evaluate only a fixed number of curve metadata endpoints to validate its mathematical
// domain and inversion. Pixel evaluation belongs exclusively to the GPU backend.
fn parameter_value(function: u16, p: [f64; 7], x: f64) -> f64 {
    let [g, a, b, c, d, e, f] = p;
    let value = match function {
        0 => x.powf(g),
        1 | 2 => {
            let power = if x < -b / a {
                0.0
            } else {
                (a * x + b).max(0.0).powf(g)
            };
            power + if function == 2 { c } else { 0.0 }
        }
        3 | 4 => {
            if x < d {
                c * x + if function == 4 { f } else { 0.0 }
            } else {
                (a * x + b).max(0.0).powf(g) + if function == 4 { e } else { 0.0 }
            }
        }
        _ => unreachable!("validated ICC parametric function"),
    };
    value.clamp(0.0, 1.0)
}

fn validate_parameters(
    function: u16,
    parameters: [i32; 7],
    tag: IccSignature,
) -> Result<(), IccError> {
    let [g, a, b, _, d, _, _] = parameters.map(|v| f64::from(v) / 65536.0);
    if g <= 0.0 || (function != 0 && a <= 0.0) {
        return Err(IccError::CurveParameters { tag });
    }
    if function >= 3 && d <= 1.0 && a * d.max(0.0) + b < 0.0 {
        return Err(IccError::CurveParameters { tag });
    }
    Ok(())
}
