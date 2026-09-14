//! ICC.1:2022 §10.16. Position tables are storage locations, not execution order.
use std::collections::BTreeMap;

use super::profile::{Reader, invalid, limit};
use super::{
    IccAffine, IccClut, IccClutInterpolation, IccError, IccLimits, IccProfile, IccProgram,
    IccSignature, IccStage,
};

mod curve;

/// None means an unknown element makes this whole method inapplicable (8.10/10.16.1).
/// Invalid known metadata is an error, never a request to substitute another method.
pub(super) fn parse(
    profile: &IccProfile,
    tag: IccSignature,
    inputs: usize,
    outputs: usize,
) -> Result<Option<IccProgram>, IccError> {
    let data = profile.required(tag)?;
    if data.signature(0)? != IccSignature(*b"mpet") {
        return Err(IccError::TagType {
            tag,
            kind: data.signature(0)?,
        });
    }
    let limits = profile.limits();
    let (p, q) = channel_pair(data, limits)?;
    if (p, q) != (inputs, outputs) {
        return invalid("MPE profile channels", 8);
    }
    let count = data.u32(12)?;
    limit(
        "processing elements",
        u64::from(count),
        u64::from(limits.max_processing_elements),
    )?;
    if count == 0 {
        return invalid("empty MPE", 12);
    }
    let positions = positions(data, 16, u64::from(count))?;
    let mut current = inputs;
    let mut unknown = false;
    for &(offset, size) in &positions {
        let element = Reader(data.slice(offset, offset + size, "processing element")?);
        let (p, q) = channel_pair(element, limits)?;
        if p != current {
            return invalid("MPE channel continuity", offset + 8);
        }
        current = q;
        unknown |= !matches!(
            &element.signature(0)?.0,
            b"matf" | b"cvst" | b"clut" | b"bACS" | b"eACS"
        );
    }
    if current != outputs {
        return invalid("MPE output channels", 10);
    }
    if unknown {
        return Ok(None);
    }
    let mut decoded = BTreeMap::new();
    let mut stages = Vec::with_capacity(count as usize);
    for position @ (offset, size) in positions {
        let stage = if let Some(stage) = decoded.get(&position) {
            stage
        } else {
            let data = Reader(data.slice(offset, offset + size, "processing element")?);
            let (p, q) = channel_pair(data, limits)?;
            let stage = match &data.signature(0)?.0 {
                b"matf" => Some(matrix(data, p, q)?),
                b"cvst" => Some(curve::set(data, p, q, limits)?),
                b"clut" => Some(clut(data, p, q, limits)?),
                b"bACS" | b"eACS" => {
                    if p != q {
                        return invalid("pass-through channels", 8);
                    }
                    data.slice(12, 16, "ACS signature")?;
                    finish(data, 16)?;
                    None
                }
                _ => unreachable!("preflight recognizes all elements"),
            };
            decoded.entry(position).or_insert(stage)
        };
        if let Some(stage) = stage {
            stages.push(stage.clone());
        }
    }
    IccProgram::new(inputs, outputs, stages).map(Some)
}

fn channel_pair(data: Reader<'_>, limits: IccLimits) -> Result<(usize, usize), IccError> {
    data.zeros(4, 8, "reserved processing element")?;
    let p = data.u16(8)?;
    let q = data.u16(10)?;
    for count in [p, q] {
        if count == 0 {
            return invalid("zero processing channels", 8);
        }
        limit(
            "processing channels",
            u64::from(count),
            u64::from(limits.max_processing_channels),
        )?;
    }
    Ok((usize::from(p), usize::from(q)))
}

/// Check all positions before decoding or allocating payloads, preserving shared ranges.
fn positions(data: Reader<'_>, start: u64, count: u64) -> Result<Vec<(u64, u64)>, IccError> {
    let table_end = start + count * 8;
    data.slice(start, table_end, "positions table")?;
    let mut positions = Vec::with_capacity(count as usize);
    for index in 0..count {
        let entry = start + index * 8;
        let offset = u64::from(data.u32(entry)?);
        let size = u64::from(data.u32(entry + 4)?);
        if !offset.is_multiple_of(4) || offset < table_end || size < 12 {
            return invalid("processing element range", entry);
        }
        data.slice(offset, offset + size, "processing element range")?;
        positions.push((offset, size));
    }
    let mut stored = positions.clone();
    stored.sort_unstable();
    stored.dedup();
    let mut end = table_end;
    for (offset, size) in stored {
        if offset != end.next_multiple_of(4) {
            return invalid("overlapping or noncontiguous processing elements", offset);
        }
        data.zeros(end, offset, "processing element padding")?;
        end = offset + size;
    }
    finish(data, end)?;
    Ok(positions)
}

fn finish(data: Reader<'_>, end: u64) -> Result<(), IccError> {
    if data.0.len() as u64 > end.next_multiple_of(4) {
        return invalid("processing element size", end);
    }
    data.zeros(end, data.0.len() as u64, "processing element padding")
}

fn matrix(data: Reader<'_>, p: usize, q: usize) -> Result<IccStage, IccError> {
    let count = q * (p + 1);
    let end = 12 + count as u64 * 4;
    data.slice(12, end, "float matrix")?;
    finish(data, end)?;
    let values = (0..count)
        .map(|i| data.f32(12 + i as u64 * 4).map(f64::from))
        .collect::<Result<Vec<_>, _>>()?;
    let (matrix, offset) = values.split_at(p * q);
    Ok(IccStage::Matrix(IccAffine::new(
        p,
        matrix.to_vec(),
        offset.to_vec(),
        false,
    )?))
}

fn clut(data: Reader<'_>, p: usize, q: usize, limits: IccLimits) -> Result<IccStage, IccError> {
    if p > 16 {
        return invalid("CLUT input dimensions", 8);
    }
    let grid = data.slice(12, 12 + p as u64, "CLUT grid")?;
    data.zeros(12 + p as u64, 28, "unused CLUT dimensions")?;
    let mut count = q as u64;
    for &points in grid {
        if points < 2 {
            return invalid("CLUT grid points", 12);
        }
        // Limit after each multiplication, before allocation, so even 255^16 is bounded.
        count *= u64::from(points);
        limit("CLUT values", count, u64::from(limits.max_clut_values))?;
    }
    let end = 28 + count * 4;
    data.slice(28, end, "float CLUT values")?;
    finish(data, end)?;
    let values = (0..count)
        .map(|i| data.f32(28 + i * 4))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(IccStage::Clut(IccClut {
        grid: grid.into(),
        output_channels: q,
        values: values.into(),
        interpolation: IccClutInterpolation::Tetrahedral,
    }))
}
