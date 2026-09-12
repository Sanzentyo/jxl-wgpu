//! JPEG component grids shared by Modular and VarDCT reconstruction.

/// Effective component subsampling relative to the largest color grid.
/// JPEG XL permits zero or one bit of shift on each axis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct JpegComponentShift {
    pub horizontal: u32,
    pub vertical: u32,
}

impl JpegComponentShift {
    #[must_use]
    pub const fn is_subsampled(self) -> bool {
        self.horizontal != 0 || self.vertical != 0
    }

    pub(crate) fn shifted_extent(self, width: u32, height: u32) -> Option<[u32; 2]> {
        let horizontal = 1u32.checked_shl(self.horizontal)?;
        let vertical = 1u32.checked_shl(self.vertical)?;
        Some([width.div_ceil(horizontal), height.div_ceil(vertical)])
    }
}

const HORIZONTAL: [u32; 4] = [0, 1, 1, 0];
const VERTICAL: [u32; 4] = [0, 1, 0, 1];

/// Selectors have already passed `FrameInventory::validate_jpeg_sampling`.
pub(crate) fn block_alignment(selectors: [u32; 3]) -> [u32; 2] {
    selectors.into_iter().fold([0, 0], |maximum, value| {
        let index = value as usize;
        [
            maximum[0].max(HORIZONTAL[index]),
            maximum[1].max(VERTICAL[index]),
        ]
    })
}

pub(crate) fn component_shifts(selectors: [u32; 3]) -> [JpegComponentShift; 3] {
    let maximum = block_alignment(selectors);
    selectors.map(|value| {
        let index = value as usize;
        JpegComponentShift {
            horizontal: maximum[0] - HORIZONTAL[index],
            vertical: maximum[1] - VERTICAL[index],
        }
    })
}
