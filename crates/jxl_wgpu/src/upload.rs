// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use jxl_gpu_protocol::{HostPlane, PlaneData, PlaneDesc, SampleType};

use crate::arena::ArenaAllocation;
use crate::{Error, Result};

#[derive(Debug)]
pub(crate) struct UploadedPlane {
    pub desc: PlaneDesc,
    pub buffer: Arc<wgpu::Buffer>,
    pub offset: u64,
    pub padded_size: u64,
}

/// A caller-provided GPU buffer bound directly to an `ImportedResident` plane.
#[derive(Clone, Debug)]
pub struct ResidentPlaneBinding {
    pub plane: jxl_gpu_protocol::PlaneId,
    pub buffer: Arc<wgpu::Buffer>,
    pub offset: u64,
    pub size: u64,
}

impl ResidentPlaneBinding {
    #[must_use]
    pub fn new(
        plane: jxl_gpu_protocol::PlaneId,
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        size: u64,
    ) -> Self {
        Self {
            plane,
            buffer,
            offset,
            size,
        }
    }
}

impl UploadedPlane {
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: wgpu::BufferSize::new(self.padded_size),
        })
    }
}

pub(crate) fn aligned_buffer_size(size: u64) -> Result<u64> {
    size.max(4)
        .checked_add(3)
        .map(|value| value & !3)
        .ok_or(Error::BufferSizeOverflow)
}

pub(crate) fn plane_logical_size(desc: &PlaneDesc) -> Result<u64> {
    let samples = desc.minimum_len().ok_or(Error::BufferSizeOverflow)?;
    let bytes = samples
        .checked_mul(desc.sample_type.bytes_per_sample())
        .ok_or(Error::BufferSizeOverflow)?;
    u64::try_from(bytes).map_err(|_| Error::BufferSizeOverflow)
}

pub(crate) fn upload_plane_to_slot<'a>(
    queue: &wgpu::Queue,
    desc: &PlaneDesc,
    fragments: impl IntoIterator<Item = &'a HostPlane>,
    slot: Arc<wgpu::Buffer>,
    slot_size: u64,
    allocation: &ArenaAllocation,
) -> Result<UploadedPlane> {
    let logical_size = plane_logical_size(desc)?;
    let padded_size = aligned_buffer_size(logical_size)?;
    validate_allocation(desc, allocation, logical_size, padded_size, slot_size)?;
    let mut bytes =
        vec![0_u8; usize::try_from(padded_size).map_err(|_| Error::BufferSizeOverflow)?];

    let mut saw_fragment = false;
    for fragment in fragments {
        saw_fragment = true;
        write_fragment(desc, fragment, &mut bytes)?;
    }
    if !saw_fragment {
        return Err(Error::MissingPlane(desc.id));
    }

    queue.write_buffer(&slot, 0, &bytes);

    Ok(UploadedPlane {
        desc: desc.clone(),
        buffer: slot,
        offset: 0,
        padded_size,
    })
}

pub(crate) fn plane_in_slot(
    desc: &PlaneDesc,
    slot: Arc<wgpu::Buffer>,
    slot_size: u64,
    allocation: &ArenaAllocation,
) -> Result<UploadedPlane> {
    let logical_size = plane_logical_size(desc)?;
    let padded_size = aligned_buffer_size(logical_size)?;
    validate_allocation(desc, allocation, logical_size, padded_size, slot_size)?;
    Ok(UploadedPlane {
        desc: desc.clone(),
        buffer: slot,
        offset: 0,
        padded_size,
    })
}

fn validate_allocation(
    desc: &PlaneDesc,
    allocation: &ArenaAllocation,
    logical_size: u64,
    padded_size: u64,
    slot_size: u64,
) -> Result<()> {
    if allocation.plane != desc.id || allocation.size != logical_size || padded_size > slot_size {
        return Err(Error::Execution(format!(
            "resident arena slot for plane {:?} is inconsistent with its {}-byte payload",
            desc.id, logical_size
        )));
    }
    Ok(())
}

fn write_fragment(desc: &PlaneDesc, fragment: &HostPlane, destination: &mut [u8]) -> Result<()> {
    fragment
        .validate()
        .map_err(|error| Error::InvalidPayload(error.to_string()))?;
    if fragment.id != desc.id {
        return Err(Error::InvalidPayload(format!(
            "fragment for {:?} was supplied while assembling {:?}",
            fragment.id, desc.id
        )));
    }
    if fragment.data.sample_type() != desc.sample_type {
        return Err(Error::InvalidPayload(format!(
            "plane {:?} has {:?} samples, expected {:?}",
            desc.id,
            fragment.data.sample_type(),
            desc.sample_type
        )));
    }
    let (origin_x, origin_y) = fragment.origin;
    let origin_x = u32::try_from(origin_x).map_err(|_| {
        Error::InvalidPayload(format!("plane {:?} has a negative x origin", desc.id))
    })?;
    let origin_y = u32::try_from(origin_y).map_err(|_| {
        Error::InvalidPayload(format!("plane {:?} has a negative y origin", desc.id))
    })?;
    let end_x = origin_x
        .checked_add(fragment.extent.width)
        .ok_or(Error::BufferSizeOverflow)?;
    let end_y = origin_y
        .checked_add(fragment.extent.height)
        .ok_or(Error::BufferSizeOverflow)?;
    if end_x > desc.extent.width || end_y > desc.extent.height {
        return Err(Error::InvalidPayload(format!(
            "plane {:?} fragment ({origin_x}, {origin_y}) {}x{} exceeds {}x{}",
            desc.id,
            fragment.extent.width,
            fragment.extent.height,
            desc.extent.width,
            desc.extent.height
        )));
    }

    let bytes_per_sample = desc.sample_type.bytes_per_sample();
    let source_stride = effective_stride(fragment.stride, fragment.extent.width)?;
    let destination_stride = effective_stride(desc.stride, desc.extent.width)?;
    let source = plane_bytes(&fragment.data);
    let row_bytes = usize::try_from(fragment.extent.width)
        .ok()
        .and_then(|width| width.checked_mul(bytes_per_sample))
        .ok_or(Error::BufferSizeOverflow)?;

    for row in 0..fragment.extent.height {
        let source_sample = usize::try_from(row)
            .ok()
            .and_then(|row| row.checked_mul(source_stride))
            .ok_or(Error::BufferSizeOverflow)?;
        let destination_sample = usize::try_from(origin_y + row)
            .ok()
            .and_then(|row| row.checked_mul(destination_stride))
            .and_then(|offset| offset.checked_add(origin_x as usize))
            .ok_or(Error::BufferSizeOverflow)?;
        let source_offset = source_sample
            .checked_mul(bytes_per_sample)
            .ok_or(Error::BufferSizeOverflow)?;
        let destination_offset = destination_sample
            .checked_mul(bytes_per_sample)
            .ok_or(Error::BufferSizeOverflow)?;
        let source_end = source_offset
            .checked_add(row_bytes)
            .ok_or(Error::BufferSizeOverflow)?;
        let destination_end = destination_offset
            .checked_add(row_bytes)
            .ok_or(Error::BufferSizeOverflow)?;
        let source_row = source
            .get(source_offset..source_end)
            .ok_or_else(|| Error::InvalidPayload("source row is out of bounds".into()))?;
        let destination_row = destination
            .get_mut(destination_offset..destination_end)
            .ok_or_else(|| Error::InvalidPayload("destination row is out of bounds".into()))?;
        destination_row.copy_from_slice(source_row);
    }
    Ok(())
}

fn effective_stride(stride: u32, width: u32) -> Result<usize> {
    usize::try_from(if stride == 0 { width } else { stride }).map_err(|_| Error::BufferSizeOverflow)
}

fn plane_bytes(data: &PlaneData) -> &[u8] {
    match data {
        PlaneData::I32(values) => bytemuck::cast_slice(values),
        PlaneData::F32(values) => bytemuck::cast_slice(values),
        PlaneData::F16(values) | PlaneData::U16(values) => bytemuck::cast_slice(values),
        PlaneData::U8(values) => values,
    }
}

pub(crate) const fn is_word_sample(sample_type: SampleType) -> bool {
    matches!(sample_type, SampleType::I32 | SampleType::F32)
}
