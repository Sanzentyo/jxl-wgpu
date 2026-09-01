// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::{BTreeMap, BTreeSet};

use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{
    ChromaAxis, EpfParams, GroupId, GroupPayload, MemoryMode, PlaneId, PlaneRole, RenderNode,
    RenderOp, RenderPlan, ResourceData, ResourceId, ResourceUpdate, SampleType,
};

use crate::context::WgpuBackend;
use crate::planner::{ExecutionPlan, FusedKernel, PlannedDispatch};
use crate::upload::{aligned_buffer_size, plane_logical_size};
use crate::vardct;
use crate::{Error, Result};

use super::{
    BlendUniform, Chroma2dUniform, ChromaUpsampleUniform, CopyParams, EpfUniform, ExtendUniform,
    GaborishRgbUniform, GaborishUniform, ImageOutputUniform, ModularParams, OutputEncoding,
    OutputMode, OutputTarget, PremultiplyUniform, SaveUniform, TransferUniform, UpsampleUniform,
    XybUniform, YcbcrUniform, stride,
};

pub(super) fn validate(plan: &RenderPlan) -> Result<()> {
    vardct::validate_plan(plan)?;
    for node in &plan.nodes {
        match &node.op {
            RenderOp::Copy
            | RenderOp::ModularToF32 { .. }
            | RenderOp::ChromaUpsample { .. }
            | RenderOp::Gaborish(_)
            | RenderOp::Epf(_)
            | RenderOp::VarDct
            | RenderOp::XybToRgb(_)
            | RenderOp::YcbcrToRgb
            | RenderOp::TransferFunction(_)
            | RenderOp::Blend(_)
            | RenderOp::PremultiplyAlpha { .. }
            | RenderOp::Extend { .. } => {}
            RenderOp::Upsample(params) if matches!(params.factor, 2 | 4 | 8) => {}
            RenderOp::Upsample(params) => {
                return Err(Error::Unsupported(format!(
                    "{}x upsampling is not supported; expected 2, 4, or 8",
                    params.factor
                )));
            }
            RenderOp::Convert {
                output_type: SampleType::I32 | SampleType::F32,
            } => {}
            RenderOp::Convert { output_type } => {
                return Err(Error::Unsupported(format!(
                    "conversion to {output_type:?} is not representable by the baseline kernels"
                )));
            }
            RenderOp::Save(save)
                if matches!(save.sample_type, SampleType::I32 | SampleType::F32) => {}
            RenderOp::Save(save) => {
                return Err(Error::Unsupported(format!(
                    "saving {:?} output is not representable by the baseline packing kernel",
                    save.sample_type
                )));
            }
            unsupported => {
                return Err(Error::Unsupported(format!(
                    "render operation {:?} ({}) has no portable GPU kernel",
                    unsupported.kind(),
                    node.name
                )));
            }
        }
    }
    Ok(())
}
pub(super) fn validate_resources(
    plan: &RenderPlan,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<()> {
    for node in &plan.nodes {
        for id in &node.resources {
            if !resources.contains_key(id) {
                return Err(Error::InvalidPayload(format!(
                    "node {} is missing late-bound resource {id:?}",
                    node.name
                )));
            }
        }
        match &node.op {
            RenderOp::Epf(params) => validate_epf_resource(plan, node, params, resources)?,
            RenderOp::VarDct => {}
            _ if node.resources.is_empty() => {}
            _ => {
                return Err(Error::Unsupported(format!(
                    "node {} uses late-bound resources whose operation-specific layout is not defined",
                    node.name
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_transient_budget(
    backend: &WgpuBackend,
    plan: &RenderPlan,
    execution: &ExecutionPlan,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
    output_target: OutputTarget<'_>,
) -> Result<u64> {
    let required = transient_bytes(plan, execution, groups, resources, output_target)?;
    let budget = backend.config.memory.max_transient_bytes;
    if required > budget {
        return Err(Error::ResourceLimit(format!(
            "submission needs {required} bytes of explicit transient GPU buffers, exceeding the configured limit of {budget} bytes"
        )));
    }
    Ok(required)
}

pub(in crate::scheduler) fn transient_bytes(
    plan: &RenderPlan,
    execution: &ExecutionPlan,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
    output_target: OutputTarget<'_>,
) -> Result<u64> {
    let planes = plan
        .planes
        .iter()
        .map(|plane| (plane.id, plane))
        .collect::<BTreeMap<_, _>>();
    let mut bytes = 0u64;
    for node in &plan.nodes {
        match &node.op {
            RenderOp::Copy => add_uniform::<CopyParams>(&mut bytes)?,
            RenderOp::ModularToF32 { .. } => add_uniform::<ModularParams>(&mut bytes)?,
            RenderOp::ChromaUpsample { .. } => add_uniform::<ChromaUpsampleUniform>(&mut bytes)?,
            RenderOp::Gaborish(_) => add_uniform::<GaborishUniform>(&mut bytes)?,
            RenderOp::Epf(params) => {
                add_uniform::<EpfUniform>(&mut bytes)?;
                if params.sigma_plane.is_none() {
                    add_slice::<f32>(&mut bytes, 1)?;
                }
            }
            RenderOp::VarDct => {
                add_bytes(
                    &mut bytes,
                    vardct::transient_bytes(plan, node, groups, resources)?,
                )?;
            }
            RenderOp::Upsample(params) => {
                add_uniform::<UpsampleUniform>(&mut bytes)?;
                add_slice::<f32>(&mut bytes, params.weights.len())?;
            }
            RenderOp::XybToRgb(_) => add_uniform::<XybUniform>(&mut bytes)?,
            RenderOp::YcbcrToRgb => {
                add_uniforms::<YcbcrUniform>(&mut bytes, node.outputs.len())?;
            }
            RenderOp::TransferFunction(_) => add_uniform::<TransferUniform>(&mut bytes)?,
            RenderOp::Blend(_) => add_uniform::<BlendUniform>(&mut bytes)?,
            RenderOp::PremultiplyAlpha { alpha_plane } => {
                let color_count = node.inputs.iter().filter(|id| *id != alpha_plane).count();
                add_uniforms::<PremultiplyUniform>(&mut bytes, color_count)?;
                if node.outputs.len() == node.inputs.len() {
                    add_uniform::<CopyParams>(&mut bytes)?;
                }
            }
            RenderOp::Convert { output_type } => {
                let [input] = node.inputs.as_slice() else {
                    return Err(Error::InvalidPayload(format!(
                        "Convert node '{}' does not have one input",
                        node.name
                    )));
                };
                let input = planes.get(input).ok_or(Error::MissingPlane(*input))?;
                if input.sample_type == *output_type {
                    add_uniform::<CopyParams>(&mut bytes)?;
                } else {
                    add_uniform::<ModularParams>(&mut bytes)?;
                }
            }
            RenderOp::Extend { .. } => add_uniform::<ExtendUniform>(&mut bytes)?,
            RenderOp::Save(save) => {
                let output = plan
                    .outputs
                    .iter()
                    .find(|output| output.id == save.output)
                    .ok_or_else(|| {
                        Error::InvalidPayload(format!("unknown output {:?}", save.output))
                    })?;
                let (logical_size, uniform_bytes) = match output_target.encoding {
                    OutputEncoding::Original => {
                        let logical_size = output
                            .extent
                            .area()
                            .and_then(|area| area.checked_mul(usize::from(output.channels)))
                            .and_then(|samples| {
                                samples.checked_mul(output.sample_type.bytes_per_sample())
                            })
                            .and_then(|size| u64::try_from(size).ok())
                            .ok_or(Error::BufferSizeOverflow)?;
                        let uniform_bytes = save
                            .channels
                            .len()
                            .checked_mul(std::mem::size_of::<SaveUniform>())
                            .and_then(|bytes| u64::try_from(bytes).ok())
                            .ok_or(Error::BufferSizeOverflow)?;
                        (logical_size, uniform_bytes)
                    }
                    OutputEncoding::Image(request) => (
                        ImageLayout::packed(output.extent, request.format.clone())?.logical_size,
                        u64::try_from(std::mem::size_of::<ImageOutputUniform>())
                            .map_err(|_| Error::BufferSizeOverflow)?,
                    ),
                };
                let packed_size = aligned_buffer_size(logical_size)?;
                add_bytes(&mut bytes, packed_size)?;
                if output_target.mode == OutputMode::CpuReadback && !output_target.direct_readback {
                    add_bytes(&mut bytes, packed_size)?;
                }
                add_bytes(&mut bytes, uniform_bytes)?;
            }
            unsupported => {
                return Err(Error::Unsupported(format!(
                    "cannot estimate transient storage for operation {:?}",
                    unsupported.kind()
                )));
            }
        }
    }
    for dispatch in &execution.dispatches {
        match dispatch.kernel {
            FusedKernel::Chroma2d => replace_uniforms::<ChromaUpsampleUniform, Chroma2dUniform>(
                &mut bytes,
                dispatch.node_indices.len(),
            )?,
            FusedKernel::GaborishRgb => replace_uniforms::<GaborishUniform, GaborishRgbUniform>(
                &mut bytes,
                dispatch.node_indices.len(),
            )?,
            FusedKernel::Single(_) => {}
        }
    }
    Ok(bytes)
}

fn replace_uniforms<Old, New>(total: &mut u64, old_count: usize) -> Result<()> {
    let old_bytes = old_count
        .checked_mul(std::mem::size_of::<Old>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    *total = total
        .checked_sub(old_bytes)
        .ok_or_else(|| Error::Execution("fused transient estimate underflowed".into()))?;
    add_uniform::<New>(total)
}

fn add_uniform<T>(total: &mut u64) -> Result<()> {
    add_bytes(
        total,
        u64::try_from(std::mem::size_of::<T>()).map_err(|_| Error::BufferSizeOverflow)?,
    )
}

fn add_uniforms<T>(total: &mut u64, count: usize) -> Result<()> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    add_bytes(total, bytes)
}

fn add_slice<T>(total: &mut u64, count: usize) -> Result<()> {
    let bytes = count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    add_bytes(total, bytes)
}

fn add_bytes(total: &mut u64, bytes: u64) -> Result<()> {
    *total = total.checked_add(bytes).ok_or(Error::BufferSizeOverflow)?;
    Ok(())
}

pub(super) fn validate_execution(
    device: &wgpu::Device,
    plan: &RenderPlan,
    execution: &ExecutionPlan,
) -> Result<()> {
    if execution.memory_mode != MemoryMode::Resident {
        return Err(Error::Unsupported(
            "the scheduler only executes resident allocation plans".into(),
        ));
    }
    validate_dispatch_metadata(plan, execution)?;
    if execution.arena.size_bytes == 0
        || execution.resident_bytes != execution.arena.size_bytes
        || execution.scratch_bytes != execution.arena.peak_scratch_bytes
    {
        return Err(Error::Execution(
            "resident execution metadata is inconsistent with its arena".into(),
        ));
    }

    let alignment = resident_alignment(device);
    let slot_sizes = resident_slot_sizes(execution, alignment)?;
    if let Some((&offset, &size)) = slot_sizes
        .iter()
        .find(|(_, size)| **size > device.limits().max_buffer_size)
    {
        return Err(Error::ResourceLimit(format!(
            "resident arena slot {offset} needs {size} bytes, exceeding the device buffer limit of {} bytes",
            device.limits().max_buffer_size
        )));
    }
    let planes: BTreeMap<_, _> = plan.planes.iter().map(|plane| (plane.id, plane)).collect();
    let mut required = plan
        .nodes
        .iter()
        .flat_map(|node| node.inputs.iter().chain(&node.outputs).copied())
        .collect::<BTreeSet<_>>();
    for node in &plan.nodes {
        if let RenderOp::Epf(params) = &node.op
            && let Some(sigma) = params.sigma_plane
        {
            required.insert(sigma);
        }
    }
    for plane in required {
        if execution.arena.allocation(plane).is_none() {
            return Err(Error::Execution(format!(
                "resident arena has no allocation for required plane {plane:?}"
            )));
        }
    }

    let mut allocated_planes = BTreeSet::new();
    for allocation in &execution.arena.allocations {
        if !allocated_planes.insert(allocation.plane) {
            return Err(Error::Execution(format!(
                "resident arena allocates plane {:?} more than once",
                allocation.plane
            )));
        }
        let desc = planes.get(&allocation.plane).ok_or_else(|| {
            Error::Execution(format!(
                "resident arena refers to unknown plane {:?}",
                allocation.plane
            ))
        })?;
        let logical_size = plane_logical_size(desc)?;
        let padded_size = aligned_buffer_size(logical_size)?;
        let end = allocation
            .offset
            .checked_add(padded_size)
            .ok_or(Error::BufferSizeOverflow)?;
        if allocation.size != logical_size
            || allocation.offset % alignment != 0
            || end > execution.arena.size_bytes
            || execution.tile_extents.get(&allocation.plane) != Some(&desc.extent)
        {
            return Err(Error::Execution(format!(
                "resident arena allocation for plane {:?} violates size, alignment, or extent invariants",
                allocation.plane
            )));
        }
        if padded_size > device.limits().max_storage_buffer_binding_size {
            return Err(Error::ResourceLimit(format!(
                "plane {:?} exceeds the device storage-binding limit",
                allocation.plane
            )));
        }
    }
    Ok(())
}

fn validate_dispatch_metadata(plan: &RenderPlan, execution: &ExecutionPlan) -> Result<()> {
    let mut expected_node = 0usize;
    for dispatch in &execution.dispatches {
        if dispatch.node_indices.is_empty() {
            return Err(Error::Execution(format!(
                "planned dispatch '{}' contains no nodes",
                dispatch.label
            )));
        }
        for &node_index in &dispatch.node_indices {
            if node_index != expected_node || node_index >= plan.nodes.len() {
                return Err(Error::Execution(format!(
                    "planned dispatch '{}' names node {node_index}, expected {expected_node}",
                    dispatch.label
                )));
            }
            expected_node = expected_node
                .checked_add(1)
                .ok_or(Error::BufferSizeOverflow)?;
        }
        let nodes = dispatch_nodes(plan, dispatch)?;
        if nodes
            .iter()
            .any(|node| node.precision != dispatch.precision)
        {
            return Err(Error::Execution(format!(
                "planned dispatch '{}' has stale precision metadata",
                dispatch.label
            )));
        }
        let resources = nodes
            .iter()
            .flat_map(|node| node.resources.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if resources != dispatch.resources {
            return Err(Error::Execution(format!(
                "planned dispatch '{}' has stale resource metadata",
                dispatch.label
            )));
        }
        if dispatch.workgroup_size != dispatch.variant.workgroup_size()
            || (!dispatch.kernel.is_workgroup_tunable()
                && dispatch.variant != dispatch.kernel.default_variant())
        {
            return Err(Error::Execution(format!(
                "planned dispatch '{}' has incompatible {:?} workgroup metadata",
                dispatch.label, dispatch.variant
            )));
        }
        match dispatch.kernel {
            FusedKernel::Single(kind) if nodes.len() == 1 && nodes[0].op.kind() == kind => {}
            FusedKernel::Chroma2d
                if nodes.len() == 2
                    && matches!(
                        &nodes[0].op,
                        RenderOp::ChromaUpsample {
                            axis: ChromaAxis::Horizontal
                        }
                    )
                    && matches!(
                        &nodes[1].op,
                        RenderOp::ChromaUpsample {
                            axis: ChromaAxis::Vertical
                        }
                    ) => {}
            FusedKernel::GaborishRgb
                if nodes.len() == 3
                    && nodes.iter().enumerate().all(|(channel, node)| {
                        matches!(
                            &node.op,
                            RenderOp::Gaborish(params)
                                if usize::from(params.channel) == channel
                        )
                    }) => {}
            _ => {
                return Err(Error::Execution(format!(
                    "planned dispatch '{}' does not match its fused kernel",
                    dispatch.label
                )));
            }
        }
    }
    if expected_node != plan.nodes.len() {
        return Err(Error::Execution(format!(
            "execution plan covers {expected_node} of {} render nodes",
            plan.nodes.len()
        )));
    }
    Ok(())
}

pub(in crate::scheduler) fn dispatch_nodes<'a>(
    plan: &'a RenderPlan,
    dispatch: &PlannedDispatch,
) -> Result<Vec<&'a RenderNode>> {
    dispatch
        .node_indices
        .iter()
        .map(|&index| {
            plan.nodes.get(index).ok_or_else(|| {
                Error::Execution(format!(
                    "planned dispatch '{}' names missing node {index}",
                    dispatch.label
                ))
            })
        })
        .collect()
}

pub(in crate::scheduler) fn premultiply_dispatch_count(
    node: &RenderNode,
    alpha: PlaneId,
) -> Result<u32> {
    let colors = node.inputs.iter().filter(|id| **id != alpha).count();
    let copies_alpha = usize::from(node.outputs.len() == node.inputs.len());
    u32::try_from(
        colors
            .checked_add(copies_alpha)
            .ok_or(Error::BufferSizeOverflow)?,
    )
    .map_err(|_| Error::BufferSizeOverflow)
}

pub(in crate::scheduler) fn resident_alignment(device: &wgpu::Device) -> u64 {
    u64::from(device.limits().min_storage_buffer_offset_alignment)
        .max(4)
        .next_power_of_two()
}

pub(in crate::scheduler) fn zero_required_slot_offsets(
    plan: &RenderPlan,
    execution: &ExecutionPlan,
) -> Result<BTreeSet<u64>> {
    let mut offsets = BTreeSet::new();
    for output in plan
        .nodes
        .iter()
        .filter(|node| matches!(node.op, RenderOp::VarDct))
        .flat_map(|node| &node.outputs)
    {
        let allocation = execution.arena.allocation(*output).ok_or_else(|| {
            Error::Execution(format!(
                "VarDCT output plane {output:?} has no resident allocation"
            ))
        })?;
        offsets.insert(allocation.offset);
    }
    Ok(offsets)
}

pub(in crate::scheduler) fn resident_slot_sizes(
    execution: &ExecutionPlan,
    alignment: u64,
) -> Result<BTreeMap<u64, u64>> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(Error::Execution(format!(
            "resident arena alignment {alignment} is invalid"
        )));
    }

    let mut slots = BTreeMap::<u64, u64>::new();
    for allocation in &execution.arena.allocations {
        let physical_size = align_up(allocation.size, alignment)?;
        if physical_size == 0 {
            return Err(Error::Execution(format!(
                "resident arena plane {:?} has an empty physical slot",
                allocation.plane
            )));
        }
        slots
            .entry(allocation.offset)
            .and_modify(|capacity| *capacity = (*capacity).max(physical_size))
            .or_insert(physical_size);
    }

    for (index, left) in execution.arena.allocations.iter().enumerate() {
        let left_end = left
            .offset
            .checked_add(align_up(left.size, alignment)?)
            .ok_or(Error::BufferSizeOverflow)?;
        for right in &execution.arena.allocations[index + 1..] {
            let lifetimes_overlap =
                left.first_use <= right.last_use && right.first_use <= left.last_use;
            let right_end = right
                .offset
                .checked_add(align_up(right.size, alignment)?)
                .ok_or(Error::BufferSizeOverflow)?;
            let storage_overlaps = left.offset < right_end && right.offset < left_end;
            if lifetimes_overlap && storage_overlaps {
                return Err(Error::Execution(format!(
                    "resident arena aliases simultaneously live planes {:?} and {:?}",
                    left.plane, right.plane
                )));
            }
        }
    }

    let mut physical_bytes = 0u64;
    for (&offset, &capacity) in &slots {
        if offset != physical_bytes {
            return Err(Error::Execution(format!(
                "resident arena slot {offset} is not contiguous with the preceding {physical_bytes} bytes"
            )));
        }
        physical_bytes = physical_bytes
            .checked_add(capacity)
            .ok_or(Error::BufferSizeOverflow)?;
    }
    if physical_bytes != execution.arena.size_bytes {
        return Err(Error::Execution(format!(
            "resident arena reports {} bytes but its physical slots require {physical_bytes} bytes",
            execution.arena.size_bytes
        )));
    }
    Ok(slots)
}

pub(in crate::scheduler) fn align_up(value: u64, alignment: u64) -> Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(Error::BufferSizeOverflow)
}

fn validate_epf_resource(
    plan: &RenderPlan,
    node: &RenderNode,
    params: &EpfParams,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<()> {
    let resource_id = params.sigma_resource.ok_or_else(|| {
        Error::InvalidPayload(format!("EPF node {} has no sigma resource", node.name))
    })?;
    if node.resources.as_slice() != [resource_id] {
        return Err(Error::InvalidPayload(format!(
            "EPF node {} must declare only sigma resource {resource_id:?}",
            node.name
        )));
    }
    let update = resources.get(&resource_id).ok_or_else(|| {
        Error::InvalidPayload(format!(
            "EPF node {} is missing sigma resource {resource_id:?}",
            node.name
        ))
    })?;
    match (params.sigma_plane, &update.data) {
        (None, ResourceData::F32(values)) if values.len() == 1 && values[0].is_finite() => Ok(()),
        (None, ResourceData::F32(values)) => Err(Error::InvalidPayload(format!(
            "EPF scalar sigma resource {resource_id:?} must contain exactly one finite F32 value, got {}",
            values.len()
        ))),
        (None, _) => Err(Error::InvalidPayload(format!(
            "EPF scalar sigma resource {resource_id:?} must use ResourceData::F32"
        ))),
        (Some(sigma_id), ResourceData::Plane(host)) => {
            let desc = plan
                .planes
                .iter()
                .find(|desc| desc.id == sigma_id)
                .ok_or_else(|| {
                    Error::InvalidPayload(format!(
                        "EPF sigma resource {resource_id:?} names unknown plane {sigma_id:?}"
                    ))
                })?;
            host.validate()
                .map_err(|error| Error::InvalidPayload(error.to_string()))?;
            if desc.role != PlaneRole::Parameter
                || desc.sample_type != SampleType::F32
                || host.id != sigma_id
                || host.extent != desc.extent
                || host.origin != (0, 0)
                || host.stride != stride(desc)
                || host.data.sample_type() != SampleType::F32
            {
                return Err(Error::InvalidPayload(format!(
                    "EPF resource {resource_id:?} must provide the complete declared F32 parameter plane {sigma_id:?}"
                )));
            }
            let jxl_gpu_protocol::PlaneData::F32(values) = &host.data else {
                return Err(Error::InvalidPayload(format!(
                    "EPF sigma plane {sigma_id:?} does not contain F32 samples"
                )));
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidPayload(format!(
                    "EPF sigma plane {sigma_id:?} contains non-finite values"
                )));
            }
            Ok(())
        }
        (Some(sigma_id), _) => Err(Error::InvalidPayload(format!(
            "EPF variable sigma resource {resource_id:?} must provide plane {sigma_id:?}"
        ))),
    }
}
