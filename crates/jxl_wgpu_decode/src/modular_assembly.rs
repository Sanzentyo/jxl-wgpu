//! Bounds-checked copies from locally reconstructed Modular planes to a frame arena.

use crate::modular_transform::GpuModularChannelLayout;
use crate::{Error, Result};

pub(crate) fn encode_plane_copies(
    encoder: &mut wgpu::CommandEncoder,
    group_arena: &wgpu::Buffer,
    frame_arena: &wgpu::Buffer,
    lane_offset: u64,
    planes: impl IntoIterator<Item = (GpuModularChannelLayout, GpuModularChannelLayout)>,
) -> Result<()> {
    for (source, destination) in planes {
        if source.width != destination.width
            || source.height != destination.height
            || source.bit_depth != destination.bit_depth
        {
            return Err(Error::EngineContract(
                "Modular subimage source and frame-arena destination geometries disagree",
            ));
        }
        let row_bytes = u64::from(source.width)
            .checked_mul(4)
            .ok_or_else(|| Error::backend("Modular subimage copy row size overflow"))?;
        for row in 0..source.height {
            let source_offset = u64::from(source.word_offset)
                .checked_add(u64::from(row) * u64::from(source.row_stride_words))
                .and_then(|words| words.checked_mul(4))
                .and_then(|bytes| lane_offset.checked_add(bytes))
                .ok_or_else(|| Error::backend("Modular subimage copy source offset overflow"))?;
            let destination_offset = u64::from(destination.word_offset)
                .checked_add(u64::from(row) * u64::from(destination.row_stride_words))
                .and_then(|words| words.checked_mul(4))
                .ok_or_else(|| {
                    Error::backend("Modular subimage copy destination offset overflow")
                })?;
            if source_offset
                .checked_add(row_bytes)
                .is_none_or(|end| end > group_arena.size())
                || destination_offset
                    .checked_add(row_bytes)
                    .is_none_or(|end| end > frame_arena.size())
            {
                return Err(Error::EngineContract(
                    "Modular subimage plane copy exceeds a resident GPU arena",
                ));
            }
            encoder.copy_buffer_to_buffer(
                group_arena,
                source_offset,
                frame_arena,
                destination_offset,
                row_bytes,
            );
        }
    }
    Ok(())
}
