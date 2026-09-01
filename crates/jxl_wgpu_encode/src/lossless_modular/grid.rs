use super::types::{LOSSLESS_MODULAR_GROUP_DIMENSION, LOSSLESS_MODULAR_LF_GROUP_DIMENSION};
use crate::{EncodeError, FrameGroupLayout};

/// Row-major JPEG XL pass-group grid used by one Modular frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularGroupGrid {
    pub width: u32,
    pub height: u32,
    pub columns: u32,
    pub rows: u32,
    pub groups: u32,
    pub lf_columns: u32,
    pub lf_rows: u32,
    pub lf_groups: u32,
}

impl LosslessModularGroupGrid {
    pub(super) fn for_extent(width: u32, height: u32) -> Result<Self, EncodeError> {
        if width == 0 || height == 0 || width >= (1 << 30) || height >= (1 << 30) {
            return Err(EncodeError::InvalidConfiguration(
                "Modular dimensions must be in 1..2^30",
            ));
        }
        let columns = width.div_ceil(LOSSLESS_MODULAR_GROUP_DIMENSION);
        let rows = height.div_ceil(LOSSLESS_MODULAR_GROUP_DIMENSION);
        let groups = columns
            .checked_mul(rows)
            .ok_or(EncodeError::InvalidSource("Modular group count overflow"))?;
        let lf_columns = width.div_ceil(LOSSLESS_MODULAR_LF_GROUP_DIMENSION);
        let lf_rows = height.div_ceil(LOSSLESS_MODULAR_LF_GROUP_DIMENSION);
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
        let x = column.checked_mul(LOSSLESS_MODULAR_GROUP_DIMENSION)?;
        let y = row.checked_mul(LOSSLESS_MODULAR_GROUP_DIMENSION)?;
        Some(LosslessModularGroup {
            index,
            column,
            row,
            x,
            y,
            width: (self.width - x).min(LOSSLESS_MODULAR_GROUP_DIMENSION),
            height: (self.height - y).min(LOSSLESS_MODULAR_GROUP_DIMENSION),
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
