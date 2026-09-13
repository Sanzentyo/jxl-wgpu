//! Bit-preserving GPU copies between independently pitched scalar F32 planes.
//!
//! Copy commands interpret neither color metadata nor floating-point values. All planes are
//! validated before recording; padding and neighboring channels remain untouched.

use jxl_gpu_formats::{ImageLayout, PlaneSampling, SampleKind};
use jxl_wgpu::{ResidentF32Plane, ResidentStorageBinding};

use super::FrameSurfaceError;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

pub(crate) fn planes(
    encoder: &mut wgpu::CommandEncoder,
    inputs: &[ResidentF32Plane<'_>],
    output: ResidentStorageBinding<'_>,
    layout: &ImageLayout,
) -> Result<(), FrameSurfaceError> {
    let invalid = |role, plane, reason| FrameSurfaceError::Copy {
        role,
        plane,
        reason,
    };
    if inputs.len() != layout.planes.len() || inputs.is_empty() {
        return Err(invalid(
            "output",
            0,
            "source and destination plane counts differ",
        ));
    }
    let layout =
        ImageLayout::from_planes(layout.extent, layout.format.clone(), layout.planes.clone())?;
    if layout.format.sample_kind != SampleKind::Float
        || layout.format.byte_order != jxl_gpu_formats::ByteOrder::Native
        || layout.format.planes.iter().any(|plane| {
            plane.sampling != PlaneSampling::FULL
                || plane.pixels_per_element != 1
                || plane.words.len() != 1
                || plane.words[0].fields.len() != 1
                || plane.words[0].fields[0].bits != 32
        })
    {
        return Err(invalid("output", 0, "copy requires scalar F32 planes"));
    }
    if !output.buffer.usage().contains(wgpu::BufferUsages::COPY_DST)
        || !output.offset.is_multiple_of(4)
        || output
            .offset
            .checked_add(output.size.get())
            .is_none_or(|end| end > output.buffer.size())
        || layout.logical_size > output.size.get()
    {
        return Err(invalid(
            "output",
            0,
            "invalid copy destination range or usage",
        ));
    }
    let row_bytes = u64::from(layout.extent.width) * 4;
    for (index, (input, destination)) in inputs.iter().zip(&layout.planes).enumerate() {
        let stride = if input.stride == 0 {
            input.width
        } else {
            input.stride
        };
        let required = u64::from(layout.extent.height - 1)
            .checked_mul(u64::from(stride) * 4)
            .and_then(|bytes| bytes.checked_add(row_bytes));
        if input.width < layout.extent.width
            || input.height < layout.extent.height
            || stride < input.width
            || required.is_none_or(|bytes| bytes > input.storage.size.get())
        {
            return Err(invalid(
                "input",
                index,
                "source does not cover the copied extent",
            ));
        }
        if !input
            .storage
            .buffer
            .usage()
            .contains(wgpu::BufferUsages::COPY_SRC)
            || !input.storage.offset.is_multiple_of(4)
            || input
                .storage
                .offset
                .checked_add(input.storage.size.get())
                .is_none_or(|end| end > input.storage.buffer.size())
        {
            return Err(invalid(
                "input",
                index,
                "invalid copy source range or usage",
            ));
        }
        if input.storage.buffer == output.buffer {
            return Err(invalid(
                "input",
                index,
                "source and destination must use distinct buffers",
            ));
        }
        if destination.row_bytes != row_bytes
            || !destination.offset.is_multiple_of(4)
            || !destination.row_stride.is_multiple_of(4)
        {
            return Err(invalid(
                "output",
                index,
                "destination is not a word-aligned scalar plane",
            ));
        }
    }
    for (input, destination) in inputs.iter().zip(&layout.planes) {
        let stride = u64::from(if input.stride == 0 {
            input.width
        } else {
            input.stride
        }) * 4;
        let output_offset = output.offset + destination.offset;
        if stride == row_bytes && destination.row_stride == row_bytes {
            encoder.copy_buffer_to_buffer(
                input.storage.buffer,
                input.storage.offset,
                output.buffer,
                output_offset,
                row_bytes * u64::from(layout.extent.height),
            );
        } else {
            for row in 0..u64::from(layout.extent.height) {
                encoder.copy_buffer_to_buffer(
                    input.storage.buffer,
                    input.storage.offset + row * stride,
                    output.buffer,
                    output_offset + row * destination.row_stride,
                    row_bytes,
                );
            }
        }
    }
    Ok(())
}
