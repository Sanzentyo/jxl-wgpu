use super::types::LosslessModularGroupSize;
use crate::{EncodeError, FrameGroupLayout};

/// Row-major JPEG XL pass-group grid used by one Modular frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularGroupGrid {
    pub width: u32,
    pub height: u32,
    pub group_size: LosslessModularGroupSize,
    pub columns: u32,
    pub rows: u32,
    pub groups: u32,
    pub lf_columns: u32,
    pub lf_rows: u32,
    pub lf_groups: u32,
}

impl LosslessModularGroupGrid {
    pub(super) fn for_extent(
        width: u32,
        height: u32,
        group_size: LosslessModularGroupSize,
    ) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width >= (1 << 30) || height >= (1 << 30) {
            return Err(EncodeError::InvalidConfiguration(
                "Modular dimensions must be in 1..2^30",
            ));
        }
        let dimension = group_size.dimension();
        let columns = width.div_ceil(dimension);
        let rows = height.div_ceil(dimension);
        let groups = columns
            .checked_mul(rows)
            .ok_or(EncodeError::InvalidSource("Modular group count overflow"))?;
        let lf_columns = width.div_ceil(dimension * 8);
        let lf_rows = height.div_ceil(dimension * 8);
        let lf_groups = lf_columns
            .checked_mul(lf_rows)
            .ok_or(EncodeError::InvalidSource(
                "Modular LF group count overflow",
            ))?;
        // FrameGroupLayout performs the normative TOC-entry bound as well. Do it here so an
        // impossible grid is rejected before any driver allocation or queue interaction.
        FrameGroupLayout::new(lf_groups, groups, 1)?;
        Ok(Self {
            width,
            height,
            group_size,
            columns,
            rows,
            groups,
            lf_columns,
            lf_rows,
            lf_groups,
        })
    }

    /// Resolves a canonical row-major pass-group index to its exact pixel rectangle.
    #[must_use]
    pub fn group(self, index: u32) -> Option<LosslessModularGroup> {
        if index >= self.groups {
            return None;
        }
        let column = index % self.columns;
        let row = index / self.columns;
        let dimension = self.group_size.dimension();
        let x = column.checked_mul(dimension)?;
        let y = row.checked_mul(dimension)?;
        Some(LosslessModularGroup {
            index,
            column,
            row,
            x,
            y,
            width: (self.width - x).min(dimension),
            height: (self.height - y).min(dimension),
        })
    }

    /// Iterates the standard JPEG XL TOC PassGroup order.
    pub fn ordered_groups(self) -> impl ExactSizeIterator<Item = LosslessModularGroup> {
        (0..self.groups).map(move |index| {
            self.group(index)
                .expect("an index from the checked group range is valid")
        })
    }
}

/// One GPU workgroup and its standard row-major JPEG XL PassGroup destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularGroup {
    pub index: u32,
    pub column: u32,
    pub row: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}
