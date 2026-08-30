//! GPU ABI for JPEG XL adaptive LF smoothing.

use bytemuck::{Pod, Zeroable};

pub(crate) const ADAPTIVE_LF_SHADER: &str = include_str!("vardct_lf.wgsl");
pub(crate) const ADAPTIVE_LF_TILE: u32 = 16;

/// One 32-byte, 16-byte-aligned uniform shared with `vardct_lf.wgsl`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct AdaptiveLfParams {
    pub extent_and_offsets: [u32; 4],
    pub lf_scale: [f32; 4],
}

impl AdaptiveLfParams {
    pub(crate) fn new(
        width: u32,
        height: u32,
        input_offset: u32,
        output_offset: u32,
        lf_scale: [f32; 3],
    ) -> Self {
        Self {
            extent_and_offsets: [width, height, input_offset, output_offset],
            lf_scale: [lf_scale[0], lf_scale[1], lf_scale[2], 0.0],
        }
    }

    pub(crate) fn dispatch(self) -> [u32; 2] {
        [
            self.extent_and_offsets[0].div_ceil(ADAPTIVE_LF_TILE),
            self.extent_and_offsets[1].div_ceil(ADAPTIVE_LF_TILE),
        ]
    }
}

const _: () = {
    assert!(std::mem::size_of::<AdaptiveLfParams>() == 32);
    assert!(std::mem::align_of::<AdaptiveLfParams>() == 16);
};
