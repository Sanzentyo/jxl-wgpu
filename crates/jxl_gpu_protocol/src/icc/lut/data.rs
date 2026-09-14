use super::{Reader, invalid, limit};
use crate::icc::{IccClut, IccClutInterpolation, IccCurve, IccError, IccLimits, IccStage};

#[derive(Clone, Copy)]
pub(super) enum Precision {
    U8,
    U16,
}

impl Precision {
    pub(super) const fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    fn values(self, data: Reader<'_>, start: u64, count: u64) -> Result<Vec<u16>, IccError> {
        let bytes = data.slice(start, start + count * self.bytes(), "LUT values")?;
        Ok(match self {
            Self::U8 => bytes.iter().map(|&v| u16::from(v) * 257).collect(),
            Self::U16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&v| u16::from_be_bytes(v))
                .collect(),
        })
    }

    pub(super) fn curves(
        self,
        data: Reader<'_>,
        start: u64,
        channels: usize,
        entries: u64,
    ) -> Result<IccStage, IccError> {
        let curves = (0..channels)
            .map(|c| {
                IccCurve::from_samples(self.values(
                    data,
                    start + c as u64 * entries * self.bytes(),
                    entries,
                )?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(IccStage::Curves {
            curves: curves.into(),
            inverse: false,
        })
    }

    pub(super) fn clut(
        self,
        data: Reader<'_>,
        start: u64,
        grid: &[u8],
        outputs: usize,
        limits: IccLimits,
        interpolation: IccClutInterpolation,
    ) -> Result<IccStage, IccError> {
        let count = clut_count(grid, outputs, limits)?;
        let values = self
            .values(data, start, count)?
            .into_iter()
            .map(|v| f32::from(v) / 65535.0)
            .collect();
        Ok(IccStage::Clut(IccClut {
            grid: grid.into(),
            output_channels: outputs,
            values,
            interpolation,
        }))
    }
}

pub(super) fn clut_count(grid: &[u8], outputs: usize, limits: IccLimits) -> Result<u64, IccError> {
    if grid.is_empty() || grid.len() > 16 {
        return invalid("CLUT input dimensions", 0);
    }
    let mut count = outputs as u64;
    for &points in grid {
        if points < 2 {
            return invalid("CLUT grid points", 0);
        }
        count *= u64::from(points);
        limit("CLUT values", count, u64::from(limits.max_clut_values))?;
    }
    Ok(count)
}

pub(super) fn finish(data: Reader<'_>, end: u64) -> Result<(), IccError> {
    if data.0.len() as u64 > end.next_multiple_of(4) {
        return invalid("LUT element size", end);
    }
    data.zeros(end, data.0.len() as u64, "LUT padding")
}
