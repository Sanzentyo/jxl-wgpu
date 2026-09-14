use super::data::{Precision, clut_count, finish};
use super::{Reader, invalid, limit, processing_count};
use crate::icc::{IccAffine, IccClutInterpolation, IccError, IccLimits, IccSignature, IccStage};

pub(super) fn parse(
    data: Reader<'_>,
    inputs: usize,
    outputs: usize,
    limits: IccLimits,
    xyz_input: bool,
    interpolation: IccClutInterpolation,
) -> Result<Vec<IccStage>, IccError> {
    data.zeros(11, 12, "reserved LUT table")?;
    let precision = if data.signature(0)? == IccSignature(*b"mft1") {
        Precision::U8
    } else {
        Precision::U16
    };
    let (input_entries, output_entries, start) = match precision {
        Precision::U8 => (256, 256, 48),
        Precision::U16 => (u64::from(data.u16(48)?), u64::from(data.u16(50)?), 52),
    };
    for entries in [input_entries, output_entries] {
        if !(2..=4096).contains(&entries) {
            return invalid("LUT table entries", 48);
        }
        limit(
            "curve samples",
            entries,
            u64::from(limits.max_curve_samples),
        )?;
    }
    processing_count(4, limits)?;
    let points = data.slice(10, 11, "LUT grid")?[0];
    let grid = vec![points; inputs];
    let clut_values = clut_count(&grid, outputs, limits)?;
    let clut_start = start + inputs as u64 * input_entries * precision.bytes();
    let output_start = clut_start + clut_values * precision.bytes();
    let end = output_start + outputs as u64 * output_entries * precision.bytes();
    data.slice(0, end, "LUT tables")?;
    finish(data, end)?;
    let matrix = (0..9)
        .map(|i| data.i32(12 + i * 4).map(|v| f64::from(v) / 65536.0))
        .collect::<Result<Vec<_>, _>>()?;
    if !xyz_input
        && matrix
            .iter()
            .enumerate()
            .any(|(i, &v)| v != f64::from(i / 3 == i % 3))
    {
        return invalid("LUT matrix requires PCS XYZ input", 12);
    }
    let mut stages = Vec::with_capacity(4);
    if xyz_input {
        stages.push(IccStage::Matrix(IccAffine::new(
            3,
            matrix,
            vec![0.0; 3],
            true,
        )?));
    }
    stages.push(precision.curves(data, start, inputs, input_entries)?);
    stages.push(precision.clut(data, clut_start, &grid, outputs, limits, interpolation)?);
    stages.push(precision.curves(data, output_start, outputs, output_entries)?);
    Ok(stages)
}
