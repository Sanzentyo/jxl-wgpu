use std::collections::BTreeMap;

use super::data::{Precision, clut_count};
use super::{Reader, invalid, processing_count};
use crate::icc::{
    IccAffine, IccClutInterpolation, IccCurve, IccError, IccLimits, IccSignature, IccStage,
};

struct Offsets {
    b: u64,
    matrix: u64,
    m: u64,
    clut: u64,
    a: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Curve,
    Matrix,
    Clut,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Region {
    start: u64,
    end: u64,
    kind: Kind,
}

pub(super) fn parse(
    data: Reader<'_>,
    tag: IccSignature,
    inputs: usize,
    outputs: usize,
    limits: IccLimits,
    reverse: bool,
    interpolation: IccClutInterpolation,
) -> Result<Vec<IccStage>, IccError> {
    data.zeros(10, 12, "reserved LUT A/B")?;
    let offsets = Offsets {
        b: u64::from(data.u32(12)?),
        matrix: u64::from(data.u32(16)?),
        m: u64::from(data.u32(20)?),
        clut: u64::from(data.u32(24)?),
        a: u64::from(data.u32(28)?),
    };
    for offset in [
        offsets.b,
        offsets.matrix,
        offsets.m,
        offsets.clut,
        offsets.a,
    ] {
        if offset == 0 {
            continue;
        }
        if offset < 32 || !offset.is_multiple_of(4) {
            return invalid("LUT element offset", offset);
        }
        data.slice(offset, offset + 4, "LUT element offset")?;
    }
    if offsets.b == 0
        || (offsets.matrix == 0) != (offsets.m == 0)
        || (offsets.clut == 0) != (offsets.a == 0)
    {
        return invalid("LUT processing combination", 12);
    }
    if offsets.clut == 0 && inputs != outputs {
        return invalid("LUT requires channel-changing CLUT", 8);
    }
    let (b_channels, a_channels) = if reverse {
        (inputs, outputs)
    } else {
        (outputs, inputs)
    };
    if offsets.matrix != 0 && b_channels != 3 {
        return invalid("LUT matrix channels", 8);
    }
    processing_count(
        1 + 2 * usize::from(offsets.matrix != 0) + 2 * usize::from(offsets.clut != 0),
        limits,
    )?;

    // Curves are individually shareable, including a suffix of another curve set.
    // Preflight all lengths and disjoint physical spans before allocating payloads.
    let mut regions = Vec::new();
    let b_positions = curve_positions(data, tag, offsets.b, b_channels, limits, &mut regions)?;
    let m_positions = curve_positions(data, tag, offsets.m, b_channels, limits, &mut regions)?;
    let a_positions = curve_positions(data, tag, offsets.a, a_channels, limits, &mut regions)?;
    if offsets.matrix != 0 {
        data.slice(offsets.matrix, offsets.matrix + 48, "LUT matrix")?;
        regions.push(Region {
            start: offsets.matrix,
            end: offsets.matrix + 48,
            kind: Kind::Matrix,
        });
    }
    let mut clut_layout = None;
    if offsets.clut != 0 {
        let start = offsets.clut;
        let header = data.slice(start, start + 20, "LUT CLUT header")?;
        let precision = match header[16] {
            1 => Precision::U8,
            2 => Precision::U16,
            _ => return invalid("LUT CLUT precision", start + 16),
        };
        data.zeros(start + inputs as u64, start + 16, "unused CLUT dimensions")?;
        data.zeros(start + 17, start + 20, "reserved LUT CLUT")?;
        let grid = &header[..inputs];
        let count = clut_count(grid, outputs, limits)?;
        let end = start + 20 + count * precision.bytes();
        data.slice(start, end, "LUT CLUT values")?;
        regions.push(Region {
            start,
            end,
            kind: Kind::Clut,
        });
        clut_layout = Some((precision, grid));
    }
    regions.sort_unstable();
    regions.dedup();
    let mut end = 32;
    for region in &regions {
        if region.start < end {
            return invalid("overlapping LUT elements", region.start);
        }
        data.zeros(
            region.end,
            region.end.next_multiple_of(4).min(data.0.len() as u64),
            "LUT element padding",
        )?;
        end = region.end;
    }
    let mut curves = BTreeMap::new();
    for region in regions {
        if region.kind == Kind::Curve {
            curves.insert(
                region.start,
                IccCurve::parse_data(
                    Reader(data.slice(region.start, region.end, "LUT curve")?),
                    tag,
                    limits.max_curve_samples,
                )?,
            );
        }
    }
    let set = |positions: Vec<u64>| {
        (!positions.is_empty()).then(|| IccStage::Curves {
            curves: positions
                .iter()
                .map(|position| curves[position].clone())
                .collect(),
            inverse: false,
        })
    };
    let b = set(b_positions);
    let m = set(m_positions);
    let a = set(a_positions);
    let matrix = if offsets.matrix == 0 {
        None
    } else {
        let coefficients = (0..12)
            .map(|i| {
                data.i32(offsets.matrix + i * 4)
                    .map(|v| f64::from(v) / 65536.0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Some(IccStage::Matrix(IccAffine::new(
            3,
            coefficients[..9].to_vec(),
            coefficients[9..].to_vec(),
            true,
        )?))
    };
    let clut = clut_layout
        .map(|(precision, grid)| {
            precision.clut(
                data,
                offsets.clut + 20,
                grid,
                outputs,
                limits,
                interpolation,
            )
        })
        .transpose()?;
    Ok(if reverse {
        [b, matrix, m, clut, a]
    } else {
        [a, clut, m, matrix, b]
    }
    .into_iter()
    .flatten()
    .collect())
}

fn curve_positions(
    data: Reader<'_>,
    tag: IccSignature,
    start: u64,
    count: usize,
    limits: IccLimits,
    regions: &mut Vec<Region>,
) -> Result<Vec<u64>, IccError> {
    if start == 0 {
        return Ok(Vec::new());
    }
    let mut cursor = start;
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        let tail = Reader(data.slice(cursor, data.0.len() as u64, "LUT curve set")?);
        let end = cursor + IccCurve::encoded_len(tail, tag, limits.max_curve_samples)?;
        positions.push(cursor);
        regions.push(Region {
            start: cursor,
            end,
            kind: Kind::Curve,
        });
        cursor = end.next_multiple_of(4);
    }
    Ok(positions)
}
