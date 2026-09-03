// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use jxl_gpu_protocol::{
    Extent2d, GroupId, GroupPayload, OutputDesc, OutputId, OutputLayout, PlaneDesc, PlaneId,
    PlaneRole, RenderNode, RenderOp, RenderPlan, ResourceData, ResourceId, ResourceUpdate,
    SampleType,
};

use crate::buffer_pool::PooledBuffer;
use crate::context::WgpuBackend;
use crate::planner::{ExecutionPlan, FusedKernel};
use crate::readback::{ReadbackRequest, stage_output};
use crate::upload::{
    ResidentPlaneBinding, UploadedPlane, plane_in_slot, upload_plane_to_slot,
};
use crate::vardct;
use crate::video::{
    ImageOutputRequest, ImageReadbackRequest, PackedImageOutput, stage_image_output,
};
use crate::{Error, Result};

mod nodes;
mod pipeline;
#[cfg(test)]
mod tests;
mod validation;

#[cfg(test)]
use nodes::io::prepare_image_output;
use pipeline::PipelineFactory;
#[cfg(test)]
use validation::transient_bytes;
use validation::{
    dispatch_nodes, premultiply_dispatch_count, resident_alignment, resident_slot_sizes,
    validate_execution, validate_resources, validate_transient_budget, zero_required_slot_offsets,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputMode {
    CpuReadback,
    GpuOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputEncoding<'a> {
    Original,
    Image(&'a ImageOutputRequest),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OutputTarget<'a> {
    mode: OutputMode,
    encoding: OutputEncoding<'a>,
    direct_readback: bool,
}

fn output_buffer_usage(target: OutputTarget<'_>) -> wgpu::BufferUsages {
    let transfer = if target.mode == OutputMode::CpuReadback && target.direct_readback {
        wgpu::BufferUsages::MAP_READ
    } else {
        wgpu::BufferUsages::COPY_SRC
    };
    wgpu::BufferUsages::STORAGE | transfer
}

#[derive(Debug)]
pub(crate) struct PackedOutput {
    pub id: OutputId,
    pub extent: Extent2d,
    pub sample_type: SampleType,
    pub channels: u8,
    pub layout: OutputLayout,
    pub logical_size: u64,
    pub buffer: Arc<wgpu::Buffer>,
}

#[derive(Debug)]
pub(crate) struct EncodedSubmission {
    pub command_buffer: wgpu::CommandBuffer,
    pub readbacks: Vec<ReadbackRequest>,
    pub packed_outputs: Vec<PackedOutput>,
    pub image_readbacks: Vec<ImageReadbackRequest>,
    pub packed_image_outputs: Vec<PackedImageOutput>,
    pub planned_dispatches: u32,
    pub compute_dispatches: u32,
    pub fused_dispatches: u32,
    pub direct_readback: bool,
    /// Physical resident-plane bytes addressed by this submission.
    pub resident_bytes: u64,
    /// Explicit uniforms, uploads, packed outputs, and staging bytes allocated
    /// for this submission. Pipeline/driver-private allocations are excluded.
    pub transient_bytes: u64,
    /// Internal buffers that are safe to reuse immediately after this command buffer is submitted
    /// to the backend's queue.
    pub recycle_after_submit: Vec<PooledBuffer>,
    /// Directly mapped CPU outputs that become reusable only after `wait` has unmapped them.
    pub recycle_after_wait: Vec<PooledBuffer>,
}

pub(crate) struct Scheduler;

impl Scheduler {
    pub(crate) fn validate(plan: &RenderPlan) -> Result<()> {
        validation::validate(plan)
    }
    pub(crate) fn encode(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::CpuReadback,
            OutputEncoding::Original,
        )
    }

    /// Computes the exact explicit transient allocation for [`Self::encode`] without creating
    /// GPU resources. Callers use this preflight value for backend-wide admission before any
    /// submission-owned buffer is allocated.
    pub(crate) fn estimate_transient(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::CpuReadback,
            OutputEncoding::Original,
        )
        .map(|(_, transient_bytes)| transient_bytes)
    }

    pub(crate) fn encode_gpu(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::GpuOnly,
            OutputEncoding::Original,
        )
    }

    /// Computes the exact explicit transient allocation for [`Self::encode_gpu`] without
    /// creating GPU resources.
    pub(crate) fn estimate_gpu_transient(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::GpuOnly,
            OutputEncoding::Original,
        )
        .map(|(_, transient_bytes)| transient_bytes)
    }

    pub(crate) fn encode_image(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        request: &ImageOutputRequest,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::CpuReadback,
            OutputEncoding::Image(request),
        )
    }

    /// Computes the exact explicit transient allocation for [`Self::encode_image`] without
    /// creating GPU resources.
    pub(crate) fn estimate_image_transient(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        request: &ImageOutputRequest,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::CpuReadback,
            OutputEncoding::Image(request),
        )
        .map(|(_, transient_bytes)| transient_bytes)
    }

    pub(crate) fn encode_gpu_image(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        request: &ImageOutputRequest,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::GpuOnly,
            OutputEncoding::Image(request),
        )
    }

    /// Computes the exact explicit transient allocation for [`Self::encode_gpu_image`] without
    /// creating GPU resources.
    pub(crate) fn estimate_gpu_image_transient(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        request: &ImageOutputRequest,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            OutputMode::GpuOnly,
            OutputEncoding::Image(request),
        )
        .map(|(_, transient_bytes)| transient_bytes)
    }

    fn preflight_with_mode<'a>(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        _imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        output_mode: OutputMode,
        output_encoding: OutputEncoding<'a>,
    ) -> Result<(OutputTarget<'a>, u64)> {
        Self::validate(plan)?;
        validate_resources(plan, resources)?;
        let output_target = OutputTarget {
            mode: output_mode,
            encoding: output_encoding,
            direct_readback: output_mode == OutputMode::CpuReadback
                && backend.direct_readback_enabled(),
        };
        validate_execution(&backend.device, plan, execution)?;
        let transient_bytes =
            validate_transient_budget(backend, plan, execution, groups, resources, output_target)?;
        Ok((output_target, transient_bytes))
    }

    fn encode_with_mode(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
        imported_planes: &BTreeMap<PlaneId, ResidentPlaneBinding>,
        output_mode: OutputMode,
        output_encoding: OutputEncoding<'_>,
    ) -> Result<EncodedSubmission> {
        let device = &backend.device;
        let (output_target, transient_bytes) = Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
            imported_planes,
            output_mode,
            output_encoding,
        )?;
        let alignment = resident_alignment(device);
        let slot_sizes = resident_slot_sizes(execution, alignment)?;
        let zero_required_slots = zero_required_slot_offsets(plan, execution)?;
        let mut slot_leases = slot_sizes
            .iter()
            .map(|(&offset, &size)| {
                let label = format!("jxl-wgpu resident plane slot {offset}");
                let usage = wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST;
                let lease = if zero_required_slots.contains(&offset) {
                    backend.buffers.acquire_zeroed(&label, size, usage)
                } else {
                    backend.buffers.acquire(&label, size, usage)
                };
                (offset, lease)
            })
            .collect::<BTreeMap<_, _>>();
        let slots = slot_leases
            .iter()
            .map(|(&offset, lease)| (offset, Arc::clone(lease.buffer())))
            .collect::<BTreeMap<_, _>>();
        let mut planes = BTreeMap::new();

        for desc in &plan.planes {
            if desc.role == PlaneRole::ImportedResident {
                let binding = imported_planes.get(&desc.id).ok_or_else(|| {
                    Error::Execution(format!(
                        "scheduler is missing imported resident plane {:?}",
                        desc.id
                    ))
                })?;
                let plane = UploadedPlane {
                    desc: desc.clone(),
                    buffer: Arc::clone(&binding.buffer),
                    offset: binding.offset,
                    padded_size: binding.size,
                };
                planes.insert(desc.id, plane);
                continue;
            }
            let Some(allocation) = execution.arena.allocation(desc.id) else {
                continue;
            };
            let slot_size = slot_sizes.get(&allocation.offset).copied().ok_or_else(|| {
                Error::Execution(format!(
                    "resident arena has no physical slot for plane {:?}",
                    desc.id
                ))
            })?;
            let slot = slots.get(&allocation.offset).cloned().ok_or_else(|| {
                Error::Execution(format!(
                    "resident arena did not allocate the physical slot for plane {:?}",
                    desc.id
                ))
            })?;
            let plane = if matches!(desc.role, PlaneRole::Source | PlaneRole::Parameter) {
                let mut fragments = groups
                    .values()
                    .flat_map(|group| group.planes.iter())
                    .filter(|plane| plane.id == desc.id)
                    .collect::<Vec<_>>();
                fragments.extend(resources.values().filter_map(|update| match &update.data {
                    ResourceData::Plane(plane) if plane.id == desc.id => Some(plane),
                    _ => None,
                }));
                upload_plane_to_slot(&backend.queue, desc, fragments, slot, slot_size, allocation)?
            } else {
                plane_in_slot(desc, slot, slot_size, allocation)?
            };
            planes.insert(desc.id, plane);
        }

        let mut upsample_weights_buffers = BTreeMap::new();
        for node in &plan.nodes {
            if let RenderOp::Upsample(params) = &node.op {
                if !upsample_weights_buffers.contains_key(&params.weights) {
                    let update = resources.get(&params.weights).ok_or_else(|| {
                        Error::InvalidPayload(format!(
                            "Upsample node '{}' is missing weights resource {:?}",
                            node.name, params.weights
                        ))
                    })?;
                    let weights_slice = match &update.data {
                        ResourceData::F32(values) => values.as_slice(),
                        _ => {
                            return Err(Error::InvalidPayload(format!(
                                "Upsample node '{}' weights resource must be ResourceData::F32",
                                node.name
                            )));
                        }
                    };
                    let factor = usize::from(params.factor.as_u8());
                    let expected_weights = factor * factor * 25;
                    if weights_slice.len() != expected_weights {
                        return Err(Error::InvalidPayload(format!(
                            "{}x Upsample has {} weights, expected {expected_weights}",
                            params.factor.as_u8(),
                            weights_slice.len()
                        )));
                    }
                    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("jxl-wgpu upsample weights"),
                        contents: bytemuck::cast_slice(weights_slice),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
                    upsample_weights_buffers.insert(params.weights, buffer);
                }
            }
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("jxl-wgpu frame"),
        });
        let mut readbacks = Vec::new();
        let mut packed_outputs = Vec::new();
        let mut image_readbacks = Vec::new();
        let mut packed_image_outputs = Vec::new();
        let mut recycle_after_submit = Vec::new();
        let mut recycle_after_wait = Vec::new();
        let mut saved_outputs = BTreeSet::new();
        let mut compute_dispatches = 0u32;
        let mut fused_dispatches = 0u32;

        for dispatch in &execution.dispatches {
            let factory = PipelineFactory {
                device,
                cache: &backend.pipelines,
                buffers: &backend.buffers,
                kernel_policy: &backend.config.kernel_policy,
                variant: dispatch.variant,
            };
            match &dispatch.kernel {
                FusedKernel::Single(expected_kind) => {
                    let [node_index] = dispatch.node_indices.as_slice() else {
                        return Err(Error::Execution(format!(
                            "single dispatch '{}' does not name exactly one node",
                            dispatch.label
                        )));
                    };
                    let node = plan.nodes.get(*node_index).ok_or_else(|| {
                        Error::Execution(format!(
                            "dispatch '{}' names missing node {node_index}",
                            dispatch.label
                        ))
                    })?;
                    if node.op.kind() != *expected_kind {
                        return Err(Error::Execution(format!(
                            "dispatch '{}' declares {expected_kind:?} for a {:?} node",
                            dispatch.label,
                            node.op.kind()
                        )));
                    }
                    let emitted = match &node.op {
                        RenderOp::Copy => {
                            nodes::io::encode_copy(&factory, &mut encoder, node, &planes)?;
                            1
                        }
                        RenderOp::ModularToF32 { multiplier, bias } => {
                            nodes::filters::encode_modular_to_f32(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *multiplier,
                                *bias,
                            )?;
                            1
                        }
                        RenderOp::ChromaUpsample { axis } => {
                            nodes::filters::encode_chroma_upsample(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *axis,
                            )?;
                            1
                        }
                        RenderOp::Gaborish(params) => {
                            nodes::filters::encode_gaborish(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                            )?;
                            1
                        }
                        RenderOp::Epf(params) => {
                            nodes::filters::encode_epf(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                resources,
                                params,
                            )?;
                            1
                        }
                        RenderOp::VarDct => vardct::encode(
                            backend,
                            &mut encoder,
                            plan,
                            node,
                            &planes,
                            groups,
                            resources,
                        )?,
                        RenderOp::Upsample(params) => {
                            let weights_buffer = upsample_weights_buffers
                                .get(&params.weights)
                                .ok_or_else(|| {
                                    Error::InvalidPayload(format!(
                                        "Upsample node '{}' missing prepared weights buffer {:?}",
                                        node.name, params.weights
                                    ))
                                })?;
                            nodes::filters::encode_upsample(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                                weights_buffer,
                            )?;
                            1
                        }
                        RenderOp::XybToRgb(params) => {
                            nodes::color::encode_xyb_to_rgb(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                            )?;
                            1
                        }
                        RenderOp::YcbcrToRgb => {
                            nodes::color::encode_ycbcr(&factory, &mut encoder, node, &planes)?;
                            u32::try_from(node.outputs.len())
                                .map_err(|_| Error::BufferSizeOverflow)?
                        }
                        RenderOp::TransferFunction(params) => {
                            nodes::color::encode_transfer_function(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                            )?;
                            1
                        }
                        RenderOp::Blend(params) => {
                            nodes::blend::encode_blend(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                            )?;
                            1
                        }
                        RenderOp::PremultiplyAlpha { alpha_plane } => {
                            nodes::blend::encode_premultiply(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *alpha_plane,
                            )?;
                            premultiply_dispatch_count(node, *alpha_plane)?
                        }
                        RenderOp::Convert { output_type } => {
                            nodes::blend::encode_convert(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *output_type,
                            )?;
                            1
                        }
                        RenderOp::Extend {
                            image_extent,
                            origin,
                        } => {
                            nodes::io::encode_extend(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *image_extent,
                                *origin,
                            )?;
                            1
                        }
                        RenderOp::AdaptiveLf(_) => {
                            return Err(Error::Unsupported(
                                "Adaptive LF node lowering is not yet implemented in scheduler".into(),
                            ));
                        }
                        RenderOp::Save(save) => {
                            if !saved_outputs.insert(save.output) {
                                return Err(Error::InvalidPayload(format!(
                                    "output {:?} is saved more than once",
                                    save.output
                                )));
                            }
                            match output_target.encoding {
                                OutputEncoding::Original => {
                                    let (packed, pooled) = nodes::io::encode_save(
                                        &factory,
                                        &mut encoder,
                                        node,
                                        &planes,
                                        plan,
                                        save,
                                        output_target,
                                    )?;
                                    match output_target.mode {
                                        OutputMode::CpuReadback => {
                                            readbacks.push(stage_output(
                                                device,
                                                &mut encoder,
                                                output_desc(plan, save.output)?,
                                                &packed.buffer,
                                                packed.logical_size,
                                                output_target.direct_readback,
                                            )?);
                                            let pooled = pooled.ok_or_else(|| {
                                                Error::Execution(
                                                    "CPU output was allocated outside the internal pool"
                                                        .into(),
                                                )
                                            })?;
                                            if output_target.direct_readback {
                                                recycle_after_wait.push(pooled);
                                            } else {
                                                recycle_after_submit.push(pooled);
                                            }
                                        }
                                        OutputMode::GpuOnly => {
                                            debug_assert!(pooled.is_none());
                                            packed_outputs.push(packed);
                                        }
                                    }
                                    u32::try_from(save.channels.len())
                                        .map_err(|_| Error::BufferSizeOverflow)?
                                }
                                OutputEncoding::Image(_) => {
                                    let (packed, pooled) = nodes::io::encode_image_save(
                                        &factory,
                                        &mut encoder,
                                        node,
                                        &planes,
                                        plan,
                                        save,
                                        output_target,
                                    )?;
                                    match output_target.mode {
                                        OutputMode::CpuReadback => {
                                            image_readbacks.push(stage_image_output(
                                                device,
                                                &mut encoder,
                                                &packed,
                                                output_target.direct_readback,
                                            )?);
                                            let pooled = pooled.ok_or_else(|| {
                                                Error::Execution(
                                                    "CPU image readback output was allocated outside the internal pool"
                                                        .into(),
                                                )
                                            })?;
                                            if output_target.direct_readback {
                                                recycle_after_wait.push(pooled);
                                            } else {
                                                recycle_after_submit.push(pooled);
                                            }
                                        }
                                        OutputMode::GpuOnly => {
                                            debug_assert!(pooled.is_none());
                                            packed_image_outputs.push(packed);
                                        }
                                    }
                                    1
                                }
                            }
                        }
                        unsupported => {
                            return Err(Error::Unsupported(format!(
                                "render operation {:?} ({}) has no portable GPU kernel",
                                unsupported.kind(),
                                node.name
                            )));
                        }
                    };
                    compute_dispatches = compute_dispatches
                        .checked_add(emitted)
                        .ok_or(Error::BufferSizeOverflow)?;
                }
                FusedKernel::Chroma2d => {
                    let nodes = dispatch_nodes(plan, dispatch)?;
                    nodes::filters::encode_chroma_2d(&factory, &mut encoder, &nodes, &planes)?;
                    compute_dispatches = compute_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                    fused_dispatches = fused_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                }
                FusedKernel::GaborishRgb => {
                    let nodes = dispatch_nodes(plan, dispatch)?;
                    nodes::filters::encode_gaborish_rgb(&factory, &mut encoder, &nodes, &planes)?;
                    compute_dispatches = compute_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                    fused_dispatches = fused_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                }
            }
        }

        for output in &plan.outputs {
            if !saved_outputs.contains(&output.id) {
                return Err(Error::InvalidPayload(format!(
                    "output {:?} has no Save node",
                    output.id
                )));
            }
        }

        // VarDCT writes only the declared DCT8 task rectangles, so pixels outside those rectangles
        // require a known-zero initial state. Only those cache candidates are cleared after every
        // Save has consumed them. Full-frame kernels and zero-filled source uploads overwrite their
        // complete bound regions and can reuse dirty allocations without this bandwidth cost. The
        // next queue write and submission are ordered after the tail clear.
        for (&offset, lease) in &mut slot_leases {
            if zero_required_slots.contains(&offset) && lease.cacheable() {
                encoder.clear_buffer(lease.buffer(), 0, None);
                lease.mark_zeroed_on_recycle();
            }
        }
        // Resident slots may be returned after submission, not completion. Every future use is
        // submitted to this same queue, whose total order prevents overlap with the old commands.
        recycle_after_submit.extend(slot_leases.into_values().filter(PooledBuffer::cacheable));

        Ok(EncodedSubmission {
            command_buffer: encoder.finish(),
            readbacks,
            packed_outputs,
            image_readbacks,
            packed_image_outputs,
            planned_dispatches: u32::try_from(execution.dispatches.len())
                .map_err(|_| Error::BufferSizeOverflow)?,
            compute_dispatches,
            fused_dispatches,
            direct_readback: output_target.direct_readback,
            resident_bytes: execution.resident_bytes,
            transient_bytes,
            recycle_after_submit,
            recycle_after_wait,
        })
    }
}
fn validate_storage_buffer_size(limits: &wgpu::Limits, size: u64, label: &str) -> Result<()> {
    let max_binding = limits.max_storage_buffer_binding_size;
    let maximum = limits.max_buffer_size.min(max_binding);
    if size > maximum {
        return Err(Error::ResourceLimit(format!(
            "{label} needs {size} bytes, exceeding the storage buffer binding limit {maximum}"
        )));
    }
    Ok(())
}
fn output_desc(plan: &RenderPlan, id: OutputId) -> Result<&OutputDesc> {
    plan.outputs
        .iter()
        .find(|output| output.id == id)
        .ok_or_else(|| Error::InvalidPayload(format!("unknown output {id:?}")))
}
fn unary_planes<'a>(
    node: &RenderNode,
    planes: &'a BTreeMap<PlaneId, UploadedPlane>,
) -> Result<(&'a UploadedPlane, &'a UploadedPlane)> {
    let [input] = node.inputs.as_slice() else {
        return Err(Error::Unsupported(format!(
            "{} requires exactly one input",
            node.name
        )));
    };
    let [output] = node.outputs.as_slice() else {
        return Err(Error::Unsupported(format!(
            "{} requires exactly one output",
            node.name
        )));
    };
    Ok((plane(planes, *input)?, plane(planes, *output)?))
}

fn plane(planes: &BTreeMap<PlaneId, UploadedPlane>, id: PlaneId) -> Result<&UploadedPlane> {
    planes.get(&id).ok_or(Error::MissingPlane(id))
}

fn require_f32_equal_extent(
    operation: &str,
    input: &UploadedPlane,
    output: &UploadedPlane,
) -> Result<()> {
    if input.desc.sample_type == SampleType::F32
        && output.desc.sample_type == SampleType::F32
        && input.desc.extent == output.desc.extent
    {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "{operation} requires equal-sized F32 input and output"
        )))
    }
}

fn stride(desc: &PlaneDesc) -> u32 {
    if desc.stride == 0 {
        desc.extent.width
    } else {
        desc.stride
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyParams {
    width: u32,
    height: u32,
    input_stride: u32,
    output_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModularParams {
    width: u32,
    height: u32,
    input_stride: u32,
    output_stride: u32,
    multiplier: f32,
    bias: f32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ChromaUpsampleUniform {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride: u32,
    output_stride: u32,
    axis: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Chroma2dUniform {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride: u32,
    output_stride: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GaborishUniform {
    width: u32,
    height: u32,
    input_stride: u32,
    output_stride: u32,
    weight0: f32,
    weight1: f32,
    weight2: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GaborishRgbUniform {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    weights_x: [f32; 4],
    weights_y: [f32; 4],
    weights_b: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AdaptiveLfUniform {
    extent_and_offsets: [u32; 4],
    lf_scale: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EpfUniform {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    sigma_width: u32,
    sigma_height: u32,
    sigma_stride: u32,
    sigma_is_plane: u32,
    sigma_scale: f32,
    border_sad_mul: f32,
    channel_scale_x: f32,
    channel_scale_y: f32,
    channel_scale_b: f32,
    min_sigma: f32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpsampleUniform {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    input_stride: u32,
    output_stride: u32,
    factor: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct YcbcrUniform {
    width: u32,
    height: u32,
    cb_stride: u32,
    y_stride: u32,
    cr_stride: u32,
    output_stride: u32,
    component: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct XybUniform {
    width: u32,
    height: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    input_stride_b: u32,
    output_stride_r: u32,
    output_stride_g: u32,
    output_stride_b: u32,
    matrix_r: [f32; 4],
    matrix_g: [f32; 4],
    matrix_b: [f32; 4],
    bias_cbrt: [f32; 4],
    scaled_bias: [f32; 4],
    intensity_scale: f32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TransferUniform {
    width: u32,
    height: u32,
    input_stride_r: u32,
    input_stride_g: u32,
    input_stride_b: u32,
    output_stride_r: u32,
    output_stride_g: u32,
    output_stride_b: u32,
    transfer: u32,
    gamma: f32,
    intensity_target: f32,
    min_nits: f32,
    luminance_rgb: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlendUniform {
    width: u32,
    height: u32,
    base_stride: u32,
    source_stride: u32,
    output_stride: u32,
    base_alpha_stride: u32,
    source_alpha_stride: u32,
    mode: u32,
    component: u32,
    clamp: u32,
    alpha_associated: u32,
    has_alpha: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PremultiplyUniform {
    width: u32,
    height: u32,
    color_stride: u32,
    alpha_stride: u32,
    output_stride: u32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ExtendUniform {
    width: u32,
    height: u32,
    frame_width: u32,
    frame_height: u32,
    frame_stride: u32,
    reference_stride: u32,
    output_stride: u32,
    origin_x: i32,
    origin_y: i32,
    has_reference: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SaveUniform {
    width: u32,
    height: u32,
    source_stride: u32,
    channels: u32,
    channel: u32,
    layout: u32,
    orientation: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageOutputUniform {
    width: u32,
    height: u32,
    source_width: u32,
    source_height: u32,
    r_stride: u32,
    g_stride: u32,
    b_stride: u32,
    kind: u32,
    channels: u32,
    order: u32,
    matrix: u32,
    range: u32,
    siting_x: u32,
    siting_y: u32,
    subsample_x: u32,
    subsample_y: u32,
    bits: u32,
    storage_bits: u32,
    plane0_offset: u32,
    plane0_stride: u32,
    plane1_offset: u32,
    plane1_stride: u32,
    plane2_offset: u32,
    plane2_stride: u32,
    plane3_offset: u32,
    plane3_stride: u32,
    logical_size: u32,
    dispatch_width: u32,
    orientation: u32,
    source_transfer: u32,
    target_transfer: u32,
    _padding: u32,
    primaries_r: [f32; 4],
    primaries_g: [f32; 4],
    primaries_b: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<CopyParams>() == 16);
    assert!(std::mem::align_of::<CopyParams>() == 4);
    assert!(std::mem::size_of::<ModularParams>() == 32);
    assert!(std::mem::align_of::<ModularParams>() == 4);
    assert!(std::mem::size_of::<ChromaUpsampleUniform>() == 32);
    assert!(std::mem::align_of::<ChromaUpsampleUniform>() == 4);
    assert!(std::mem::size_of::<Chroma2dUniform>() == 32);
    assert!(std::mem::align_of::<Chroma2dUniform>() == 4);
    assert!(std::mem::size_of::<GaborishUniform>() == 32);
    assert!(std::mem::align_of::<GaborishUniform>() == 4);
    assert!(std::mem::size_of::<GaborishRgbUniform>() == 80);
    assert!(std::mem::align_of::<GaborishRgbUniform>() == 4);
    assert!(std::mem::size_of::<EpfUniform>() == 80);
    assert!(std::mem::align_of::<EpfUniform>() == 4);
    assert!(std::mem::size_of::<UpsampleUniform>() == 32);
    assert!(std::mem::align_of::<UpsampleUniform>() == 4);
    assert!(std::mem::size_of::<YcbcrUniform>() == 32);
    assert!(std::mem::align_of::<YcbcrUniform>() == 4);
    assert!(std::mem::size_of::<XybUniform>() == 128);
    assert!(std::mem::align_of::<XybUniform>() == 4);
    assert!(std::mem::size_of::<TransferUniform>() == 64);
    assert!(std::mem::align_of::<TransferUniform>() == 4);
    assert!(std::mem::size_of::<BlendUniform>() == 48);
    assert!(std::mem::align_of::<BlendUniform>() == 4);
    assert!(std::mem::size_of::<PremultiplyUniform>() == 32);
    assert!(std::mem::align_of::<PremultiplyUniform>() == 4);
    assert!(std::mem::size_of::<ExtendUniform>() == 48);
    assert!(std::mem::align_of::<ExtendUniform>() == 4);
    assert!(std::mem::size_of::<SaveUniform>() == 32);
    assert!(std::mem::align_of::<SaveUniform>() == 4);
    assert!(std::mem::size_of::<ImageOutputUniform>() == 176);
    assert!(std::mem::align_of::<ImageOutputUniform>() == 4);
    assert!(std::mem::offset_of!(ImageOutputUniform, primaries_r) == 128);
    assert!(std::mem::offset_of!(ImageOutputUniform, primaries_g) == 144);
    assert!(std::mem::offset_of!(ImageOutputUniform, primaries_b) == 160);
};
