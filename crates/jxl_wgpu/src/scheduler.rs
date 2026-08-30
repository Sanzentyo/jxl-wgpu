// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_formats::{
    ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpace, ColorSpecification,
    ImageLayout, NumericFormatClass, Packed422Order, PixelFormat, PixelFormatClass,
    RgbChannelOrder, RgbStorage, TransferFunction as ImageTransferFunction, WgslNumericCapability,
    YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{
    BlendComponent, BlendMode, BlendParams, ChromaAxis, EpfParams, EpfPass, Extent2d, GroupId,
    GroupPayload, MemoryMode, OutputColorEncoding, OutputDesc, OutputId, OutputLayout,
    OutputOrientation, PlaneDesc, PlaneId, PlaneRole, RenderNode, RenderOp, RenderOpKind,
    RenderPlan, ResourceData, ResourceId, ResourceUpdate, RgbColorEncoding, RgbPrimaries,
    SampleType, TransferFunction as SourceTransferFunction,
};
use wgpu::util::DeviceExt;

use crate::autotune::KernelVariant;
use crate::buffer_pool::{BufferPool, PooledBuffer};
use crate::context::WgpuBackend;
use crate::pipeline_cache::{PipelineCache, PipelineKey};
use crate::planner::{ExecutionPlan, FusedKernel, PlannedDispatch};
use crate::readback::{ReadbackRequest, stage_output};
use crate::upload::{
    UploadedPlane, aligned_buffer_size, is_word_sample, plane_in_slot, plane_logical_size,
    upload_plane_to_slot,
};
use crate::vardct;
use crate::video::{
    ImageOutputRequest, ImageReadbackRequest, PackedImageOutput, stage_image_output,
};
use crate::{Error, Result};

const WORKGROUP_SIZE: u32 = 16;

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

struct PipelineFactory<'a> {
    device: &'a wgpu::Device,
    cache: &'a PipelineCache,
    buffers: &'a Arc<BufferPool>,
}

impl Scheduler {
    pub(crate) fn validate(plan: &RenderPlan) -> Result<()> {
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

    pub(crate) fn encode(
        backend: &WgpuBackend,
        plan: &RenderPlan,
        execution: &ExecutionPlan,
        groups: &BTreeMap<GroupId, GroupPayload>,
        resources: &BTreeMap<ResourceId, ResourceUpdate>,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
        request: &ImageOutputRequest,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
        request: &ImageOutputRequest,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
        request: &ImageOutputRequest,
    ) -> Result<EncodedSubmission> {
        Self::encode_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
        request: &ImageOutputRequest,
    ) -> Result<u64> {
        Self::preflight_with_mode(
            backend,
            plan,
            execution,
            groups,
            resources,
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
            output_mode,
            output_encoding,
        )?;
        let factory = PipelineFactory {
            device,
            cache: &backend.pipelines,
            buffers: &backend.buffers,
        };
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
                            encode_copy(&factory, &mut encoder, node, &planes)?;
                            1
                        }
                        RenderOp::ModularToF32 { multiplier, bias } => {
                            encode_modular_to_f32(
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
                            encode_chroma_upsample(&factory, &mut encoder, node, &planes, *axis)?;
                            1
                        }
                        RenderOp::Gaborish(params) => {
                            encode_gaborish(&factory, &mut encoder, node, &planes, params)?;
                            1
                        }
                        RenderOp::Epf(params) => {
                            encode_epf(&factory, &mut encoder, node, &planes, resources, params)?;
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
                            encode_upsample(&factory, &mut encoder, node, &planes, params)?;
                            1
                        }
                        RenderOp::XybToRgb(params) => {
                            encode_xyb_to_rgb(&factory, &mut encoder, node, &planes, params)?;
                            1
                        }
                        RenderOp::YcbcrToRgb => {
                            encode_ycbcr(&factory, &mut encoder, node, &planes)?;
                            u32::try_from(node.outputs.len())
                                .map_err(|_| Error::BufferSizeOverflow)?
                        }
                        RenderOp::TransferFunction(params) => {
                            encode_transfer_function(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                params,
                            )?;
                            1
                        }
                        RenderOp::Blend(params) => {
                            encode_blend(&factory, &mut encoder, node, &planes, params)?;
                            1
                        }
                        RenderOp::PremultiplyAlpha { alpha_plane } => {
                            encode_premultiply(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *alpha_plane,
                            )?;
                            premultiply_dispatch_count(node, *alpha_plane)?
                        }
                        RenderOp::Convert { output_type } => {
                            encode_convert(&factory, &mut encoder, node, &planes, *output_type)?;
                            1
                        }
                        RenderOp::Extend {
                            image_extent,
                            origin,
                        } => {
                            encode_extend(
                                &factory,
                                &mut encoder,
                                node,
                                &planes,
                                *image_extent,
                                *origin,
                            )?;
                            1
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
                                    let (packed, pooled) = encode_save(
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
                                    let (packed, pooled) = encode_image_save(
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
                    encode_chroma_2d(&factory, &mut encoder, &nodes, &planes)?;
                    compute_dispatches = compute_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                    fused_dispatches = fused_dispatches
                        .checked_add(1)
                        .ok_or(Error::BufferSizeOverflow)?;
                }
                FusedKernel::GaborishRgb => {
                    let nodes = dispatch_nodes(plan, dispatch)?;
                    encode_gaborish_rgb(&factory, &mut encoder, &nodes, &planes)?;
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

fn validate_resources(
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

fn validate_transient_budget(
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

fn transient_bytes(
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

fn validate_execution(
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
        let expected_variant = match dispatch.kernel {
            FusedKernel::Single(RenderOpKind::VarDct) => KernelVariant::Tile8x8,
            FusedKernel::Single(_) | FusedKernel::Chroma2d | FusedKernel::GaborishRgb => {
                KernelVariant::Tile16x16
            }
        };
        if dispatch.variant != expected_variant
            || dispatch.workgroup_size != expected_variant.workgroup_size()
        {
            return Err(Error::Execution(format!(
                "planned dispatch '{}' does not match the fixed {:?} shader workgroup",
                dispatch.label, expected_variant
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

fn dispatch_nodes<'a>(
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

fn premultiply_dispatch_count(node: &RenderNode, alpha: PlaneId) -> Result<u32> {
    let colors = node.inputs.iter().filter(|id| **id != alpha).count();
    let copies_alpha = usize::from(node.outputs.len() == node.inputs.len());
    u32::try_from(
        colors
            .checked_add(copies_alpha)
            .ok_or(Error::BufferSizeOverflow)?,
    )
    .map_err(|_| Error::BufferSizeOverflow)
}

fn resident_alignment(device: &wgpu::Device) -> u64 {
    u64::from(device.limits().min_storage_buffer_offset_alignment)
        .max(4)
        .next_power_of_two()
}

fn zero_required_slot_offsets(
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

fn resident_slot_sizes(execution: &ExecutionPlan, alignment: u64) -> Result<BTreeMap<u64, u64>> {
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

fn align_up(value: u64, alignment: u64) -> Result<u64> {
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

fn encode_copy(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    if input.desc.sample_type != output.desc.sample_type || input.desc.extent != output.desc.extent
    {
        return Err(Error::Unsupported(format!(
            "Copy requires identical types and extents, got {:?} {:?} -> {:?} {:?}",
            input.desc.sample_type, input.desc.extent, output.desc.sample_type, output.desc.extent
        )));
    }
    if !is_word_sample(input.desc.sample_type) {
        return Err(Error::Unsupported(
            "resident-arena Copy currently requires I32 or F32 planes".into(),
        ));
    }

    let params = CopyParams {
        width: input.desc.extent.width,
        height: input.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
    };
    let uniform = create_uniform(device, "jxl-wgpu copy params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu copy",
        wgpu::include_wgsl!("../shaders/copy.wgsl"),
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        input.desc.extent.width,
        input.desc.extent.height,
    );
    Ok(())
}

fn encode_modular_to_f32(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    multiplier: f32,
    bias: f32,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    if input.desc.sample_type != SampleType::I32
        || output.desc.sample_type != SampleType::F32
        || input.desc.extent != output.desc.extent
    {
        return Err(Error::Unsupported(
            "ModularToF32 requires equal-sized I32 input and F32 output".into(),
        ));
    }
    let params = ModularParams {
        width: input.desc.extent.width,
        height: input.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
        multiplier,
        bias,
        _padding: [0; 2],
    };
    let uniform = create_uniform(device, "jxl-wgpu modular params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu modular-to-f32",
        wgpu::include_wgsl!("../shaders/modular_to_f32.wgsl"),
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        input.desc.extent.width,
        input.desc.extent.height,
    );
    Ok(())
}

fn encode_chroma_upsample(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    axis: ChromaAxis,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    if input.desc.sample_type != SampleType::F32 || output.desc.sample_type != SampleType::F32 {
        return Err(Error::InvalidPayload(
            "ChromaUpsample requires F32 input and output".into(),
        ));
    }
    let extent_matches = match axis {
        ChromaAxis::Horizontal => {
            output.desc.extent.height == input.desc.extent.height
                && output.desc.extent.width.div_ceil(2) == input.desc.extent.width
        }
        ChromaAxis::Vertical => {
            output.desc.extent.width == input.desc.extent.width
                && output.desc.extent.height.div_ceil(2) == input.desc.extent.height
        }
    };
    if !extent_matches {
        return Err(Error::InvalidPayload(format!(
            "{axis:?} ChromaUpsample extent mismatch: {:?} -> {:?}; expected a possibly odd-cropped 2x extent",
            input.desc.extent, output.desc.extent
        )));
    }
    let params = ChromaUpsampleUniform {
        input_width: input.desc.extent.width,
        input_height: input.desc.extent.height,
        output_width: output.desc.extent.width,
        output_height: output.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
        axis: match axis {
            ChromaAxis::Horizontal => 0,
            ChromaAxis::Vertical => 1,
        },
        _padding: 0,
    };
    let uniform = create_uniform(device, "jxl-wgpu chroma upsample params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu chroma upsample",
        wgpu::include_wgsl!("../shaders/chroma_upsample.wgsl"),
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        output.desc.extent.width,
        output.desc.extent.height,
    );
    Ok(())
}

fn encode_chroma_2d(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    nodes: &[&RenderNode],
    planes: &BTreeMap<PlaneId, UploadedPlane>,
) -> Result<()> {
    let [horizontal, vertical] = nodes else {
        return Err(Error::Execution(
            "Chroma2d fusion requires exactly horizontal and vertical nodes".into(),
        ));
    };
    if !matches!(
        &horizontal.op,
        RenderOp::ChromaUpsample {
            axis: ChromaAxis::Horizontal
        }
    ) || !matches!(
        &vertical.op,
        RenderOp::ChromaUpsample {
            axis: ChromaAxis::Vertical
        }
    ) {
        return Err(Error::Execution(
            "Chroma2d fusion has stale operation metadata".into(),
        ));
    }
    let (input, intermediate) = unary_planes(horizontal, planes)?;
    let (vertical_input, output) = unary_planes(vertical, planes)?;
    if intermediate.desc.id != vertical_input.desc.id {
        return Err(Error::Execution(
            "Chroma2d fusion nodes do not share one intermediate plane".into(),
        ));
    }
    if input.desc.sample_type != SampleType::F32
        || intermediate.desc.sample_type != SampleType::F32
        || output.desc.sample_type != SampleType::F32
        || intermediate.desc.extent.height != input.desc.extent.height
        || intermediate.desc.extent.width.div_ceil(2) != input.desc.extent.width
        || output.desc.extent.width != intermediate.desc.extent.width
        || output.desc.extent.height.div_ceil(2) != intermediate.desc.extent.height
    {
        return Err(Error::InvalidPayload(
            "Chroma2d fusion requires valid possibly odd-cropped F32 H->V extents".into(),
        ));
    }
    let params = Chroma2dUniform {
        input_width: input.desc.extent.width,
        input_height: input.desc.extent.height,
        output_width: output.desc.extent.width,
        output_height: output.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
        _padding: [0; 2],
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu chroma 2d params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu chroma-2d",
        wgpu::include_wgsl!("../shaders/chroma_2d.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        output.desc.extent.width,
        output.desc.extent.height,
    );
    Ok(())
}

fn encode_gaborish(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    gaborish: &jxl_gpu_protocol::GaborishParams,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    require_f32_equal_extent("Gaborish", input, output)?;
    if [gaborish.weight0, gaborish.weight1, gaborish.weight2]
        .into_iter()
        .any(|weight| !weight.is_finite())
    {
        return Err(Error::InvalidPayload(
            "Gaborish weights must be finite".into(),
        ));
    }
    let params = GaborishUniform {
        width: input.desc.extent.width,
        height: input.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
        weight0: gaborish.weight0,
        weight1: gaborish.weight1,
        weight2: gaborish.weight2,
        _padding: 0,
    };
    let uniform = create_uniform(device, "jxl-wgpu gaborish params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu gaborish",
        wgpu::include_wgsl!("../shaders/gaborish.wgsl"),
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        input.desc.extent.width,
        input.desc.extent.height,
    );
    Ok(())
}

fn encode_gaborish_rgb(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    nodes: &[&RenderNode],
    planes: &BTreeMap<PlaneId, UploadedPlane>,
) -> Result<()> {
    let [node_x, node_y, node_b] = nodes else {
        return Err(Error::Execution(
            "GaborishRgb fusion requires exactly three nodes".into(),
        ));
    };
    let nodes = [*node_x, *node_y, *node_b];
    let mut inputs = Vec::with_capacity(3);
    let mut outputs = Vec::with_capacity(3);
    let mut weights = Vec::with_capacity(3);
    let mut input_ids = BTreeSet::new();
    let mut output_ids = BTreeSet::new();
    for (channel, node) in nodes.into_iter().enumerate() {
        let RenderOp::Gaborish(params) = &node.op else {
            return Err(Error::Execution(
                "GaborishRgb fusion has stale operation metadata".into(),
            ));
        };
        if usize::from(params.channel) != channel {
            return Err(Error::Execution(format!(
                "GaborishRgb node {channel} declares channel {}",
                params.channel
            )));
        }
        let (input, output) = unary_planes(node, planes)?;
        require_f32_equal_extent("GaborishRgb", input, output)?;
        if !input_ids.insert(input.desc.id) || !output_ids.insert(output.desc.id) {
            return Err(Error::Execution(
                "GaborishRgb fusion aliases logical channels".into(),
            ));
        }
        inputs.push(input);
        outputs.push(output);
        weights.push(params);
    }
    if !input_ids.is_disjoint(&output_ids)
        || inputs
            .iter()
            .chain(outputs.iter())
            .any(|plane| plane.desc.extent != inputs[0].desc.extent)
    {
        return Err(Error::Execution(
            "GaborishRgb fusion requires six distinct equal-sized planes".into(),
        ));
    }
    let extent = inputs[0].desc.extent;
    let params = GaborishRgbUniform {
        width: extent.width,
        height: extent.height,
        input_stride_x: stride(&inputs[0].desc),
        input_stride_y: stride(&inputs[1].desc),
        input_stride_b: stride(&inputs[2].desc),
        output_stride_x: stride(&outputs[0].desc),
        output_stride_y: stride(&outputs[1].desc),
        output_stride_b: stride(&outputs[2].desc),
        weights_x: [
            weights[0].weight0,
            weights[0].weight1,
            weights[0].weight2,
            0.0,
        ],
        weights_y: [
            weights[1].weight0,
            weights[1].weight1,
            weights[1].weight2,
            0.0,
        ],
        weights_b: [
            weights[2].weight0,
            weights[2].weight1,
            weights[2].weight2,
            0.0,
        ],
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu Gaborish RGB params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu gaborish-rgb",
        wgpu::include_wgsl!("../shaders/gaborish_rgb.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            inputs[0].binding(),
            inputs[1].binding(),
            inputs[2].binding(),
            outputs[0].binding(),
            outputs[1].binding(),
            outputs[2].binding(),
            uniform.as_entire_binding(),
        ],
        extent.width,
        extent.height,
    );
    Ok(())
}

fn encode_epf(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
    epf: &EpfParams,
) -> Result<()> {
    let device = factory.device;
    let [input_x_id, input_y_id, input_b_id] = node.inputs.as_slice() else {
        return Err(Error::InvalidPayload(format!(
            "EPF node {} requires exactly three input planes",
            node.name
        )));
    };
    let [output_x_id, output_y_id, output_b_id] = node.outputs.as_slice() else {
        return Err(Error::InvalidPayload(format!(
            "EPF node {} requires exactly three output planes",
            node.name
        )));
    };
    let inputs = [
        plane(planes, *input_x_id)?,
        plane(planes, *input_y_id)?,
        plane(planes, *input_b_id)?,
    ];
    let outputs = [
        plane(planes, *output_x_id)?,
        plane(planes, *output_y_id)?,
        plane(planes, *output_b_id)?,
    ];
    let extent = inputs[0].desc.extent;
    if inputs
        .iter()
        .chain(outputs.iter())
        .any(|plane| plane.desc.sample_type != SampleType::F32 || plane.desc.extent != extent)
    {
        return Err(Error::InvalidPayload(
            "EPF requires three equal-sized F32 inputs and outputs".into(),
        ));
    }

    let resource_id = epf.sigma_resource.ok_or_else(|| {
        Error::InvalidPayload(format!("EPF node {} has no sigma resource", node.name))
    })?;
    let update = resources.get(&resource_id).ok_or_else(|| {
        Error::InvalidPayload(format!(
            "EPF node {} is missing sigma resource {resource_id:?}",
            node.name
        ))
    })?;
    let scalar_buffer;
    let (sigma_binding, sigma_width, sigma_height, sigma_stride, sigma_is_plane) =
        match (epf.sigma_plane, &update.data) {
            (None, ResourceData::F32(values)) => {
                scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jxl-wgpu EPF scalar sigma"),
                    contents: bytemuck::cast_slice(values),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                (scalar_buffer.as_entire_binding(), 1, 1, 1, 0)
            }
            (Some(sigma_id), ResourceData::Plane(_)) => {
                let sigma = plane(planes, sigma_id)?;
                (
                    sigma.binding(),
                    sigma.desc.extent.width,
                    sigma.desc.extent.height,
                    stride(&sigma.desc),
                    1,
                )
            }
            _ => {
                return Err(Error::InvalidPayload(format!(
                    "EPF sigma resource {resource_id:?} changed representation after validation"
                )));
            }
        };
    let minimum_sigma_extent = (extent.width.div_ceil(8), extent.height.div_ceil(8));
    if sigma_is_plane != 0
        && (sigma_width < minimum_sigma_extent.0 || sigma_height < minimum_sigma_extent.1)
    {
        return Err(Error::InvalidPayload(format!(
            "EPF sigma plane is {sigma_width}x{sigma_height}, but image {:?} requires at least {}x{} blocks",
            extent, minimum_sigma_extent.0, minimum_sigma_extent.1
        )));
    }

    const MIN_SIGMA: f32 = -3.905_243;
    let uniform = create_uniform(
        device,
        "jxl-wgpu EPF params",
        &EpfUniform {
            width: extent.width,
            height: extent.height,
            input_stride_x: stride(&inputs[0].desc),
            input_stride_y: stride(&inputs[1].desc),
            input_stride_b: stride(&inputs[2].desc),
            output_stride_x: stride(&outputs[0].desc),
            output_stride_y: stride(&outputs[1].desc),
            output_stride_b: stride(&outputs[2].desc),
            sigma_width,
            sigma_height,
            sigma_stride,
            sigma_is_plane,
            sigma_scale: epf.sigma_scale,
            border_sad_mul: epf.border_sad_mul,
            channel_scale_x: epf.channel_scale[0],
            channel_scale_y: epf.channel_scale[1],
            channel_scale_b: epf.channel_scale[2],
            min_sigma: MIN_SIGMA,
            _padding: [0; 2],
        },
    );
    let (label, entry_point, pass_key) = match epf.pass {
        EpfPass::Pass0 => ("jxl-wgpu EPF pass 0", "epf0", 0),
        EpfPass::Pass1 => ("jxl-wgpu EPF pass 1", "epf1", 1),
        EpfPass::Pass2 => ("jxl-wgpu EPF pass 2", "epf2", 2),
    };
    let pipeline = create_pipeline_entry(
        factory,
        label,
        wgpu::include_wgsl!("../shaders/epf.wgsl"),
        entry_point,
        0x4550_4600 | pass_key,
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            inputs[0].binding(),
            inputs[1].binding(),
            inputs[2].binding(),
            sigma_binding,
            outputs[0].binding(),
            outputs[1].binding(),
            outputs[2].binding(),
            uniform.as_entire_binding(),
        ],
        extent.width,
        extent.height,
    );
    Ok(())
}

fn encode_upsample(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    upsample: &jxl_gpu_protocol::UpsampleParams,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    if input.desc.sample_type != SampleType::F32 || output.desc.sample_type != SampleType::F32 {
        return Err(Error::Unsupported(
            "Upsample requires F32 input and output".into(),
        ));
    }
    if !matches!(upsample.factor, 2 | 4 | 8) {
        return Err(Error::Unsupported(format!(
            "{}x upsampling is unsupported",
            upsample.factor
        )));
    }
    let factor = u32::from(upsample.factor);
    if output.desc.extent.width.div_ceil(factor) != input.desc.extent.width
        || output.desc.extent.height.div_ceil(factor) != input.desc.extent.height
    {
        return Err(Error::InvalidPayload(format!(
            "{}x Upsample extent mismatch in '{}': {:?} -> {:?}; expected a possibly odd-cropped extent",
            upsample.factor, node.name, input.desc.extent, output.desc.extent
        )));
    }
    let expected_weights = usize::from(upsample.factor)
        .checked_mul(usize::from(upsample.factor))
        .and_then(|phases| phases.checked_mul(25))
        .ok_or(Error::BufferSizeOverflow)?;
    if upsample.weights.len() != expected_weights {
        return Err(Error::InvalidPayload(format!(
            "{}x Upsample has {} weights, expected {expected_weights} phase-major weights",
            upsample.factor,
            upsample.weights.len()
        )));
    }
    let weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu upsample weights"),
        contents: bytemuck::cast_slice(&upsample.weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = UpsampleUniform {
        input_width: input.desc.extent.width,
        input_height: input.desc.extent.height,
        output_width: output.desc.extent.width,
        output_height: output.desc.extent.height,
        input_stride: stride(&input.desc),
        output_stride: stride(&output.desc),
        factor,
        _padding: 0,
    };
    let uniform = create_uniform(device, "jxl-wgpu upsample params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu upsample",
        wgpu::include_wgsl!("../shaders/upsample.wgsl"),
    );
    record_dispatch(
        device,
        encoder,
        &pipeline,
        &[
            input.binding(),
            weights.as_entire_binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        output.desc.extent.width,
        output.desc.extent.height,
    );
    Ok(())
}

fn encode_ycbcr(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
) -> Result<()> {
    let device = factory.device;
    if node.inputs.len() != 3 || node.outputs.len() != 3 {
        return Err(Error::Unsupported(
            "YCbCrToRgb requires Cb/Y/Cr inputs and R/G/B outputs".into(),
        ));
    }
    let cb = plane(planes, node.inputs[0])?;
    let y = plane(planes, node.inputs[1])?;
    let cr = plane(planes, node.inputs[2])?;
    let outputs = node
        .outputs
        .iter()
        .map(|id| plane(planes, *id))
        .collect::<Result<Vec<_>>>()?;
    let extent = cb.desc.extent;
    if [y, cr]
        .into_iter()
        .chain(outputs.iter().copied())
        .any(|item| item.desc.sample_type != SampleType::F32 || item.desc.extent != extent)
        || cb.desc.sample_type != SampleType::F32
    {
        return Err(Error::Unsupported(
            "YCbCrToRgb requires equal-sized F32 planes".into(),
        ));
    }
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu ycbcr-to-rgb",
        wgpu::include_wgsl!("../shaders/ycbcr_to_rgb.wgsl"),
    );
    for (component, output) in outputs.into_iter().enumerate() {
        let params = YcbcrUniform {
            width: extent.width,
            height: extent.height,
            cb_stride: stride(&cb.desc),
            y_stride: stride(&y.desc),
            cr_stride: stride(&cr.desc),
            output_stride: stride(&output.desc),
            component: component as u32,
            _padding: 0,
        };
        let uniform = create_uniform(device, "jxl-wgpu ycbcr params", &params);
        record_dispatch(
            device,
            encoder,
            &pipeline,
            &[
                cb.binding(),
                y.binding(),
                cr.binding(),
                output.binding(),
                uniform.as_entire_binding(),
            ],
            extent.width,
            extent.height,
        );
    }
    Ok(())
}

fn encode_xyb_to_rgb(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    xyb: &jxl_gpu_protocol::XybParams,
) -> Result<()> {
    let [x_id, y_id, b_id] = node.inputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "XYB conversion requires exactly three inputs".into(),
        ));
    };
    let [r_id, g_id, output_b_id] = node.outputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "XYB conversion requires exactly three outputs".into(),
        ));
    };
    let [x, y, b, r, g, output_b] = [
        plane(planes, *x_id)?,
        plane(planes, *y_id)?,
        plane(planes, *b_id)?,
        plane(planes, *r_id)?,
        plane(planes, *g_id)?,
        plane(planes, *output_b_id)?,
    ];
    let extent = x.desc.extent;
    if [y, b, r, g, output_b]
        .iter()
        .any(|plane| plane.desc.sample_type != SampleType::F32 || plane.desc.extent != extent)
        || x.desc.sample_type != SampleType::F32
    {
        return Err(Error::InvalidPayload(
            "XYB conversion requires six equal-extent F32 planes".into(),
        ));
    }

    let intensity_scale = 255.0 / xyb.intensity_target;
    let bias_cbrt = xyb.opsin_bias.map(f32::cbrt);
    let scaled_bias = xyb.opsin_bias.map(|value| value * intensity_scale);
    let params = XybUniform {
        width: extent.width,
        height: extent.height,
        input_stride_x: stride(&x.desc),
        input_stride_y: stride(&y.desc),
        input_stride_b: stride(&b.desc),
        output_stride_r: stride(&r.desc),
        output_stride_g: stride(&g.desc),
        output_stride_b: stride(&output_b.desc),
        matrix_r: [
            xyb.inverse_opsin_matrix[0][0],
            xyb.inverse_opsin_matrix[0][1],
            xyb.inverse_opsin_matrix[0][2],
            0.0,
        ],
        matrix_g: [
            xyb.inverse_opsin_matrix[1][0],
            xyb.inverse_opsin_matrix[1][1],
            xyb.inverse_opsin_matrix[1][2],
            0.0,
        ],
        matrix_b: [
            xyb.inverse_opsin_matrix[2][0],
            xyb.inverse_opsin_matrix[2][1],
            xyb.inverse_opsin_matrix[2][2],
            0.0,
        ],
        bias_cbrt: [bias_cbrt[0], bias_cbrt[1], bias_cbrt[2], 0.0],
        scaled_bias: [scaled_bias[0], scaled_bias[1], scaled_bias[2], 0.0],
        intensity_scale,
        _padding: [0; 3],
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu XYB params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu XYB-to-RGB",
        wgpu::include_wgsl!("../shaders/xyb_to_rgb.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            x.binding(),
            y.binding(),
            b.binding(),
            r.binding(),
            g.binding(),
            output_b.binding(),
            uniform.as_entire_binding(),
        ],
        extent.width,
        extent.height,
    );
    Ok(())
}

fn encode_transfer_function(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    transfer: &jxl_gpu_protocol::TransferParams,
) -> Result<()> {
    let [input_r_id, input_g_id, input_b_id] = node.inputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "transfer function requires exactly three inputs".into(),
        ));
    };
    let [output_r_id, output_g_id, output_b_id] = node.outputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "transfer function requires exactly three outputs".into(),
        ));
    };
    let [input_r, input_g, input_b, output_r, output_g, output_b] = [
        plane(planes, *input_r_id)?,
        plane(planes, *input_g_id)?,
        plane(planes, *input_b_id)?,
        plane(planes, *output_r_id)?,
        plane(planes, *output_g_id)?,
        plane(planes, *output_b_id)?,
    ];
    let extent = input_r.desc.extent;
    if [input_r, input_g, input_b, output_r, output_g, output_b]
        .iter()
        .any(|plane| plane.desc.sample_type != SampleType::F32 || plane.desc.extent != extent)
    {
        return Err(Error::InvalidPayload(
            "transfer function requires six equal-extent F32 planes".into(),
        ));
    }
    let transfer_code = match transfer.function {
        SourceTransferFunction::Linear => 0,
        SourceTransferFunction::Srgb => 1,
        SourceTransferFunction::Bt709 => 2,
        SourceTransferFunction::Pq => 3,
        SourceTransferFunction::Hlg => 4,
        SourceTransferFunction::Gamma => 5,
    };
    let params = TransferUniform {
        width: extent.width,
        height: extent.height,
        input_stride_r: stride(&input_r.desc),
        input_stride_g: stride(&input_g.desc),
        input_stride_b: stride(&input_b.desc),
        output_stride_r: stride(&output_r.desc),
        output_stride_g: stride(&output_g.desc),
        output_stride_b: stride(&output_b.desc),
        transfer: transfer_code,
        gamma: transfer.gamma,
        intensity_target: transfer.intensity_target,
        min_nits: transfer.min_nits,
        luminance_rgb: [
            transfer.luminance_rgb[0],
            transfer.luminance_rgb[1],
            transfer.luminance_rgb[2],
            0.0,
        ],
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu transfer params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu transfer function",
        wgpu::include_wgsl!("../shaders/transfer_function.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            input_r.binding(),
            input_g.binding(),
            input_b.binding(),
            output_r.binding(),
            output_g.binding(),
            output_b.binding(),
            uniform.as_entire_binding(),
        ],
        extent.width,
        extent.height,
    );
    Ok(())
}

fn encode_blend(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    blend: &BlendParams,
) -> Result<()> {
    let [output_id] = node.outputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "Blend requires exactly one output".into(),
        ));
    };
    let (base_id, source_id, base_alpha_id, source_alpha_id, component, has_alpha) =
        match (blend.component, node.inputs.as_slice()) {
            (BlendComponent::Color { .. }, [base, source]) => {
                (*base, *source, *base, *source, 0, false)
            }
            (BlendComponent::Color { .. }, [base, source, base_alpha, source_alpha]) => {
                (*base, *source, *base_alpha, *source_alpha, 0, true)
            }
            (BlendComponent::Alpha, [base, source]) => (*base, *source, *base, *source, 1, true),
            _ => {
                return Err(Error::InvalidPayload(
                    "Blend input arity does not match its component".into(),
                ));
            }
        };
    let base = plane(planes, base_id)?;
    let source = plane(planes, source_id)?;
    let base_alpha = plane(planes, base_alpha_id)?;
    let source_alpha = plane(planes, source_alpha_id)?;
    let output = plane(planes, *output_id)?;
    require_f32_equal_extent("Blend", base, source)?;
    require_f32_equal_extent("Blend", base, base_alpha)?;
    require_f32_equal_extent("Blend", base, source_alpha)?;
    require_f32_equal_extent("Blend", base, output)?;

    let mode = match blend.mode {
        BlendMode::Keep => 0,
        BlendMode::Replace => 1,
        BlendMode::Add => 2,
        BlendMode::Multiply => 3,
        BlendMode::BlendAbove => 4,
        BlendMode::BlendBelow => 5,
        BlendMode::AlphaWeightedAddAbove => 6,
        BlendMode::AlphaWeightedAddBelow => 7,
    };
    let alpha_associated = match blend.component {
        BlendComponent::Color { alpha_associated } => alpha_associated,
        BlendComponent::Alpha => false,
    };
    let extent = base.desc.extent;
    let uniform = create_uniform(
        factory.device,
        "jxl-wgpu blend params",
        &BlendUniform {
            width: extent.width,
            height: extent.height,
            base_stride: stride(&base.desc),
            source_stride: stride(&source.desc),
            output_stride: stride(&output.desc),
            base_alpha_stride: stride(&base_alpha.desc),
            source_alpha_stride: stride(&source_alpha.desc),
            mode,
            component,
            clamp: u32::from(blend.clamp),
            alpha_associated: u32::from(alpha_associated),
            has_alpha: u32::from(has_alpha),
        },
    );
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu JPEG XL blend",
        wgpu::include_wgsl!("../shaders/blend.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            base.binding(),
            source.binding(),
            base_alpha.binding(),
            source_alpha.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        extent.width,
        extent.height,
    );
    Ok(())
}

fn encode_premultiply(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    alpha_id: PlaneId,
) -> Result<()> {
    let device = factory.device;
    let alpha = plane(planes, alpha_id)?;
    if alpha.desc.sample_type != SampleType::F32 {
        return Err(Error::Unsupported(
            "PremultiplyAlpha requires an F32 alpha plane".into(),
        ));
    }
    let colors = node
        .inputs
        .iter()
        .copied()
        .filter(|id| *id != alpha_id)
        .collect::<Vec<_>>();
    let pairs = if node.outputs.len() == colors.len() {
        colors
            .iter()
            .copied()
            .zip(node.outputs.iter().copied())
            .collect::<Vec<_>>()
    } else if node.outputs.len() == node.inputs.len() {
        node.inputs
            .iter()
            .copied()
            .zip(node.outputs.iter().copied())
            .filter(|(input, _)| *input != alpha_id)
            .collect::<Vec<_>>()
    } else {
        return Err(Error::Unsupported(
            "PremultiplyAlpha output arity does not match its color inputs".into(),
        ));
    };
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu premultiply-alpha",
        wgpu::include_wgsl!("../shaders/premultiply_alpha.wgsl"),
    );
    for (color_id, output_id) in pairs {
        let color = plane(planes, color_id)?;
        let output = plane(planes, output_id)?;
        require_f32_equal_extent("PremultiplyAlpha", color, output)?;
        if color.desc.extent != alpha.desc.extent {
            return Err(Error::Unsupported(
                "PremultiplyAlpha color and alpha extents differ".into(),
            ));
        }
        let params = PremultiplyUniform {
            width: color.desc.extent.width,
            height: color.desc.extent.height,
            color_stride: stride(&color.desc),
            alpha_stride: stride(&alpha.desc),
            output_stride: stride(&output.desc),
            _padding: [0; 3],
        };
        let uniform = create_uniform(device, "jxl-wgpu premultiply params", &params);
        record_dispatch(
            device,
            encoder,
            &pipeline,
            &[
                color.binding(),
                alpha.binding(),
                output.binding(),
                uniform.as_entire_binding(),
            ],
            color.desc.extent.width,
            color.desc.extent.height,
        );
    }
    if node.outputs.len() == node.inputs.len() {
        let alpha_output = node
            .inputs
            .iter()
            .position(|id| *id == alpha_id)
            .and_then(|index| node.outputs.get(index))
            .copied()
            .ok_or_else(|| Error::InvalidPayload("alpha output is missing".into()))?;
        encode_copy_ids(factory, encoder, alpha_id, alpha_output, planes)?;
    }
    Ok(())
}

fn encode_convert(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    output_type: SampleType,
) -> Result<()> {
    let (input, output) = unary_planes(node, planes)?;
    if output.desc.sample_type != output_type {
        return Err(Error::InvalidPayload(format!(
            "Convert declares {output_type:?}, but output plane is {:?}",
            output.desc.sample_type
        )));
    }
    if input.desc.sample_type == output_type {
        return encode_copy(factory, encoder, node, planes);
    }
    if input.desc.sample_type == SampleType::I32 && output_type == SampleType::F32 {
        return encode_modular_to_f32(factory, encoder, node, planes, 1.0, 0.0);
    }
    Err(Error::Unsupported(format!(
        "Convert {:?} -> {output_type:?} is not representable without a rounding contract",
        input.desc.sample_type
    )))
}

fn encode_extend(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    image_extent: Extent2d,
    origin: (i32, i32),
) -> Result<()> {
    let (frame_id, reference_id, has_reference) = match node.inputs.as_slice() {
        [frame] => (*frame, *frame, false),
        [frame, reference] => (*frame, *reference, true),
        _ => {
            return Err(Error::InvalidPayload(
                "Extend requires a frame and optional reference input".into(),
            ));
        }
    };
    let [output_id] = node.outputs.as_slice() else {
        return Err(Error::InvalidPayload(
            "Extend requires exactly one output".into(),
        ));
    };
    let frame = plane(planes, frame_id)?;
    let reference = plane(planes, reference_id)?;
    let output = plane(planes, *output_id)?;
    if !matches!(frame.desc.sample_type, SampleType::I32 | SampleType::F32)
        || reference.desc.sample_type != frame.desc.sample_type
        || output.desc.sample_type != frame.desc.sample_type
        || output.desc.extent != image_extent
        || (has_reference && reference.desc.extent != image_extent)
    {
        return Err(Error::InvalidPayload(
            "Extend requires matching I32/F32 planes and a full-canvas output/reference".into(),
        ));
    }

    let uniform = create_uniform(
        factory.device,
        "jxl-wgpu extend params",
        &ExtendUniform {
            width: image_extent.width,
            height: image_extent.height,
            frame_width: frame.desc.extent.width,
            frame_height: frame.desc.extent.height,
            frame_stride: stride(&frame.desc),
            reference_stride: stride(&reference.desc),
            output_stride: stride(&output.desc),
            origin_x: origin.0,
            origin_y: origin.1,
            has_reference: u32::from(has_reference),
            _padding: [0; 2],
        },
    );
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu extend to image canvas",
        wgpu::include_wgsl!("../shaders/extend.wgsl"),
    );
    record_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            frame.binding(),
            reference.binding(),
            output.binding(),
            uniform.as_entire_binding(),
        ],
        image_extent.width,
        image_extent.height,
    );
    Ok(())
}

fn encode_save(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    plan: &RenderPlan,
    save: &jxl_gpu_protocol::SaveParams,
    output_target: OutputTarget<'_>,
) -> Result<(PackedOutput, Option<PooledBuffer>)> {
    let device = factory.device;
    let output = output_desc(plan, save.output)?;
    if output.sample_type != save.sample_type
        || usize::from(output.channels) != save.channels.len()
        || save.channels.is_empty()
        || node.inputs != save.channels
        || !matches!(save.sample_type, SampleType::I32 | SampleType::F32)
    {
        return Err(Error::Unsupported(format!(
            "Save contract for output {:?} is not representable",
            output.id
        )));
    }
    let channels = save
        .channels
        .iter()
        .map(|id| plane(planes, *id))
        .collect::<Result<Vec<_>>>()?;
    let source_extent = channels[0].desc.extent;
    if channels.iter().any(|channel| {
        channel.desc.sample_type != output.sample_type || channel.desc.extent != source_extent
    }) || save.orientation.map_extent(source_extent) != output.extent
    {
        return Err(Error::Unsupported(
            "Save channels must share a source extent whose oriented size matches the output"
                .into(),
        ));
    }
    let logical_size = output
        .extent
        .area()
        .and_then(|area| area.checked_mul(usize::from(output.channels)))
        .and_then(|samples| samples.checked_mul(output.sample_type.bytes_per_sample()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)?;
    let padded_size = aligned_buffer_size(logical_size)?;
    validate_storage_buffer_size(&device.limits(), padded_size, "packed output")?;
    let (packed, pooled) = allocate_output_buffer(
        factory,
        "jxl-wgpu packed output",
        padded_size,
        output_target,
    );
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu save",
        wgpu::include_wgsl!("../shaders/save.wgsl"),
    );
    for (channel_index, channel) in channels.into_iter().enumerate() {
        let params = SaveUniform {
            width: source_extent.width,
            height: source_extent.height,
            source_stride: stride(&channel.desc),
            channels: u32::from(output.channels),
            channel: channel_index as u32,
            layout: match output.layout {
                OutputLayout::Planar => 0,
                OutputLayout::Interleaved => 1,
            },
            orientation: orientation_code(save.orientation),
            _padding: 0,
        };
        let uniform = create_uniform(device, "jxl-wgpu save params", &params);
        record_dispatch(
            device,
            encoder,
            &pipeline,
            &[
                channel.binding(),
                packed.as_entire_binding(),
                uniform.as_entire_binding(),
            ],
            output.extent.width,
            output.extent.height,
        );
    }
    Ok((
        PackedOutput {
            id: output.id,
            extent: output.extent,
            sample_type: output.sample_type,
            channels: output.channels,
            layout: output.layout,
            logical_size,
            buffer: packed,
        },
        pooled,
    ))
}

fn encode_image_save(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    plan: &RenderPlan,
    save: &jxl_gpu_protocol::SaveParams,
    output_target: OutputTarget<'_>,
) -> Result<(PackedImageOutput, Option<PooledBuffer>)> {
    let OutputEncoding::Image(request) = output_target.encoding else {
        return Err(Error::Execution(
            "generic image encoder received the ordinary output target".into(),
        ));
    };
    let output = output_desc(plan, save.output)?;
    if output.sample_type != SampleType::F32
        || save.sample_type != SampleType::F32
        || save.channels.len() < 3
        || node.inputs != save.channels
    {
        return Err(Error::Unsupported(format!(
            "generic image output {:?} requires at least three F32 Save channels in R'G'B' order",
            output.id
        )));
    }
    let source_encoding = match output.color_encoding {
        OutputColorEncoding::Rgb(encoding) => encoding,
        OutputColorEncoding::NonColor => {
            return Err(Error::Unsupported(format!(
                "generic image output {:?} requires an explicit RGB source color encoding",
                output.id
            )));
        }
    };
    if source_encoding != request.source_encoding {
        return Err(Error::InvalidPayload(format!(
            "generic image output {:?} declares source color {:?}, but the request declares {:?}",
            output.id, source_encoding, request.source_encoding
        )));
    }
    let (source_transfer, target_transfer) =
        image_transfer_params(source_encoding, &request.format)?;
    let channels = save
        .channels
        .iter()
        .take(3)
        .map(|id| plane(planes, *id))
        .collect::<Result<Vec<_>>>()?;
    let source_extent = channels[0].desc.extent;
    if channels.iter().any(|channel| {
        channel.desc.sample_type != SampleType::F32 || channel.desc.extent != source_extent
    }) || save.orientation.map_extent(source_extent) != output.extent
    {
        return Err(Error::Unsupported(
            "generic image Save channels must be equal-sized F32 planes matching the oriented output"
                .into(),
        ));
    }

    let layout = ImageLayout::packed(output.extent, request.format.clone())?;
    let prepared = prepare_image_output(&layout)?;
    let padded_size = aligned_buffer_size(layout.logical_size)?;
    validate_storage_buffer_size(
        &factory.device.limits(),
        padded_size,
        "generic image output",
    )?;
    let (buffer, pooled) = allocate_output_buffer(
        factory,
        "jxl-wgpu generic image output",
        padded_size,
        output_target,
    );
    let word_count = layout.logical_size.div_ceil(4);
    let (dispatch_x, dispatch_y, dispatch_width) =
        linear_dispatch_shape(factory.device, word_count)?;
    let params = ImageOutputUniform {
        width: output.extent.width,
        height: output.extent.height,
        source_width: source_extent.width,
        source_height: source_extent.height,
        r_stride: stride(&channels[0].desc),
        g_stride: stride(&channels[1].desc),
        b_stride: stride(&channels[2].desc),
        kind: prepared.kind,
        channels: prepared.channels,
        order: prepared.order,
        matrix: prepared.matrix,
        range: prepared.range,
        siting_x: prepared.siting_x,
        siting_y: prepared.siting_y,
        subsample_x: prepared.subsample_x,
        subsample_y: prepared.subsample_y,
        bits: prepared.bits,
        storage_bits: prepared.storage_bits,
        plane0_offset: prepared.plane_offsets[0],
        plane0_stride: prepared.plane_strides[0],
        plane1_offset: prepared.plane_offsets[1],
        plane1_stride: prepared.plane_strides[1],
        plane2_offset: prepared.plane_offsets[2],
        plane2_stride: prepared.plane_strides[2],
        plane3_offset: prepared.plane_offsets[3],
        plane3_stride: prepared.plane_strides[3],
        logical_size: to_shader_u32(layout.logical_size)?,
        dispatch_width,
        orientation: orientation_code(save.orientation),
        source_transfer,
        target_transfer,
        _padding: 0,
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu generic image params", &params);
    let pipeline = create_pipeline(
        factory,
        "jxl-wgpu RGB to generic image",
        wgpu::include_wgsl!("../shaders/rgb_to_image.wgsl"),
    );
    record_linear_dispatch(
        factory.device,
        encoder,
        &pipeline,
        &[
            channels[0].binding(),
            channels[1].binding(),
            channels[2].binding(),
            buffer.as_entire_binding(),
            uniform.as_entire_binding(),
        ],
        dispatch_x,
        dispatch_y,
    );
    Ok((
        PackedImageOutput {
            id: output.id,
            layout,
            buffer,
        },
        pooled,
    ))
}

#[derive(Clone, Copy, Debug)]
struct PreparedImageOutput {
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
    plane_offsets: [u32; 4],
    plane_strides: [u32; 4],
}

fn prepare_image_output(layout: &ImageLayout) -> Result<PreparedImageOutput> {
    if layout.planes.len() != layout.format.planes.len() || layout.planes.len() > 4 {
        return Err(Error::Unsupported(format!(
            "generic GPU output supports 1..=4 planes, layout has {}",
            layout.planes.len()
        )));
    }
    let mut plane_offsets = [0; 4];
    let mut plane_strides = [0; 4];
    for (index, plane) in layout.planes.iter().enumerate() {
        plane_offsets[index] = to_shader_u32(plane.offset)?;
        plane_strides[index] = to_shader_u32(plane.row_stride)?;
    }

    let class = classify_image_output_format(&layout.format)?;
    match class {
        ColorFormatClass::Rgb8 { storage, order } => {
            let kind = match storage {
                RgbStorage::Interleaved => 0,
                RgbStorage::Planar => 1,
            };
            let (channels, order) = match order {
                RgbChannelOrder::Rgb => (3, 0),
                RgbChannelOrder::Bgr => (3, 1),
                RgbChannelOrder::Rgba => (4, 2),
                RgbChannelOrder::Bgra => (4, 3),
            };
            Ok(PreparedImageOutput {
                kind,
                channels,
                order,
                matrix: 1,
                range: 0,
                siting_x: 1,
                siting_y: 1,
                subsample_x: 1,
                subsample_y: 1,
                bits: 8,
                storage_bits: 8,
                plane_offsets,
                plane_strides,
            })
        }
        color => {
            let (matrix, range, siting_x, siting_y) = image_color_params(layout)?;
            let (subsample_x, subsample_y) = layout
                .format
                .chroma_subsampling
                .chroma_divisors()
                .unwrap_or((1, 1));
            let (kind, channels, order, bits, storage_bits) = match color {
                ColorFormatClass::Luma { bits, storage_bits } => {
                    (if bits == 8 { 2 } else { 3 }, 1, 0, bits, storage_bits)
                }
                ColorFormatClass::YuvPlanar {
                    bits, storage_bits, ..
                } => (4, 3, 0, bits, storage_bits),
                ColorFormatClass::YuvSemiplanar {
                    bits,
                    storage_bits,
                    chroma_order,
                    ..
                } => (
                    5,
                    3,
                    u32::from(chroma_order == ChromaOrder::CrCb),
                    bits,
                    storage_bits,
                ),
                ColorFormatClass::Yuv422Packed { order } => {
                    (6, 3, u32::from(order == Packed422Order::Uyvy), 8, 8)
                }
                ColorFormatClass::Rgb8 { .. } => {
                    unreachable!("RGB color classes were handled before YCbCr lowering")
                }
            };
            Ok(PreparedImageOutput {
                kind,
                channels,
                order,
                matrix,
                range,
                siting_x: u32::from(siting_x),
                siting_y: u32::from(siting_y),
                subsample_x: u32::from(subsample_x),
                subsample_y: u32::from(subsample_y),
                bits: u32::from(bits),
                storage_bits: u32::from(storage_bits),
                plane_offsets,
                plane_strides,
            })
        }
    }
}

fn classify_image_output_format(format: &PixelFormat) -> Result<ColorFormatClass> {
    match classify_pixel_format(format) {
        Ok(PixelFormatClass::Color(color)) => Ok(color),
        Ok(PixelFormatClass::Numeric(numeric)) => Err(numeric_image_output_error(numeric)),
        Err(error) => Err(Error::Unsupported(format!(
            "generic GPU output format is unsupported: {error}"
        ))),
    }
}

fn numeric_image_output_error(numeric: NumericFormatClass) -> Error {
    if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
        Error::Unsupported(
            "generic GPU output does not assign color semantics to numeric F64; portable WGSL also has no native F64 arithmetic"
                .into(),
        )
    } else {
        Error::Unsupported(format!(
            "generic GPU output does not assign color semantics to numeric format {numeric:?}"
        ))
    }
}

fn image_transfer_params(source: RgbColorEncoding, target: &PixelFormat) -> Result<(u32, u32)> {
    if source.primaries != RgbPrimaries::Bt709 {
        return Err(Error::Unsupported(format!(
            "generic GPU output supports only BT.709 source primaries, not {:?}",
            source.primaries
        )));
    }
    let source_transfer = match source.transfer {
        SourceTransferFunction::Linear => 0,
        SourceTransferFunction::Srgb => 1,
        SourceTransferFunction::Bt709 => 2,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output source transfer {unsupported:?} is unsupported"
            )));
        }
    };
    let target_color = match target.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Default | ColorSpecification::Undefined => {
            return Err(Error::Unsupported(
                "generic GPU output requires an explicit target color specification".into(),
            ));
        }
    };
    if target_color.space != ColorSpace::Bt709 {
        return Err(Error::Unsupported(format!(
            "generic GPU output supports only BT.709 target primaries, not {:?}",
            target_color.space
        )));
    }
    let target_transfer = match target_color.transfer {
        ImageTransferFunction::Linear => 0,
        ImageTransferFunction::Srgb | ImageTransferFunction::Sycc => 1,
        ImageTransferFunction::Bt709 => 2,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target transfer {unsupported:?} is unsupported"
            )));
        }
    };
    Ok((source_transfer, target_transfer))
}

fn image_color_params(layout: &ImageLayout) -> Result<(u32, u32, u8, u8)> {
    let color = match layout.format.color_spec {
        ColorSpecification::Defined(color) => color,
        ColorSpecification::Default | ColorSpecification::Undefined => {
            return Err(Error::Unsupported(
                "YCbCr GPU output requires an explicit matrix, range, and chroma location".into(),
            ));
        }
    };
    let matrix = match color.encoding {
        YcbcrEncoding::Bt601 => 0,
        YcbcrEncoding::Bt709 => 1,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "YCbCr GPU output matrix {unsupported:?} is unsupported"
            )));
        }
    };
    let range = match color.range {
        ColorRange::Full => 0,
        ColorRange::Limited => 1,
    };
    let (subsample_x, subsample_y) = layout
        .format
        .chroma_subsampling
        .chroma_divisors()
        .unwrap_or((1, 1));
    Ok((
        matrix,
        range,
        image_siting(color.chroma_location.horizontal, subsample_x)?,
        image_siting(color.chroma_location.vertical, subsample_y)?,
    ))
}

fn image_siting(location: ChromaLocation, divisor: u8) -> Result<u8> {
    if divisor == 1 {
        return Ok(1);
    }
    match location {
        ChromaLocation::Center => Ok(0),
        ChromaLocation::Even => Ok(1),
        unsupported => Err(Error::Unsupported(format!(
            "chroma location {unsupported:?} is unsupported for {divisor}:1 output"
        ))),
    }
}

fn allocate_output_buffer(
    factory: &PipelineFactory<'_>,
    label: &str,
    size: u64,
    output_target: OutputTarget<'_>,
) -> (Arc<wgpu::Buffer>, Option<PooledBuffer>) {
    let usage = output_buffer_usage(output_target);
    if output_target.mode == OutputMode::CpuReadback {
        let pooled = factory.buffers.acquire(label, size, usage);
        let buffer = Arc::clone(pooled.buffer());
        (buffer, Some(pooled))
    } else {
        // Public GPU outputs can outlive the frame session. They must never enter a cache whose
        // reuse lifetime is controlled by the backend rather than by the public Arc owner.
        let buffer = Arc::new(factory.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }));
        (buffer, None)
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

fn to_shader_u32(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::ResourceLimit(
            "generic image output exceeds the shader's 32-bit address space".into(),
        )
    })
}

const fn orientation_code(orientation: OutputOrientation) -> u32 {
    match orientation {
        OutputOrientation::Identity => 0,
        OutputOrientation::FlipHorizontal => 1,
        OutputOrientation::Rotate180 => 2,
        OutputOrientation::FlipVertical => 3,
        OutputOrientation::Transpose => 4,
        OutputOrientation::Rotate90Cw => 5,
        OutputOrientation::AntiTranspose => 6,
        OutputOrientation::Rotate90Ccw => 7,
    }
}

fn output_desc(plan: &RenderPlan, id: OutputId) -> Result<&OutputDesc> {
    plan.outputs
        .iter()
        .find(|output| output.id == id)
        .ok_or_else(|| Error::InvalidPayload(format!("unknown output {id:?}")))
}

fn encode_copy_ids(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    input: PlaneId,
    output: PlaneId,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
) -> Result<()> {
    let node = RenderNode {
        name: "internal copy".into(),
        op: RenderOp::Copy,
        inputs: vec![input],
        outputs: vec![output],
        resources: Vec::new(),
        scale: jxl_gpu_protocol::Scale2d::IDENTITY,
        border: jxl_gpu_protocol::Border2d::default(),
        precision: jxl_gpu_protocol::PrecisionContract::Exact,
    };
    encode_copy(factory, encoder, &node, planes)
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

fn create_uniform<T: Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn create_pipeline(
    factory: &PipelineFactory<'_>,
    label: &str,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    create_pipeline_entry(factory, label, descriptor, "main", 0)
}

fn create_pipeline_entry(
    factory: &PipelineFactory<'_>,
    label: &str,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
    entry_point: &'static str,
    layout_hash: u64,
) -> std::sync::Arc<wgpu::ComputePipeline> {
    let key = PipelineKey::new(label, entry_point, KernelVariant::Tile16x16, layout_hash);
    if let Some(pipeline) = factory.cache.get(&key) {
        return pipeline;
    }
    match factory.cache.get_or_insert_with(key, || {
        let module = factory.device.create_shader_module(descriptor);
        Ok::<_, std::convert::Infallible>(factory.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ))
    }) {
        Ok(pipeline) => pipeline,
        Err(never) => match never {},
    }
}

fn record_dispatch(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &[wgpu::BindingResource<'_>],
    width: u32,
    height: u32,
) {
    let layout = pipeline.get_bind_group_layout(0);
    let entries = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu dispatch bindings"),
        layout: &layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("jxl-wgpu dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(
        width.div_ceil(WORKGROUP_SIZE),
        height.div_ceil(WORKGROUP_SIZE),
        1,
    );
}

fn record_linear_dispatch(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    resources: &[wgpu::BindingResource<'_>],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let layout = pipeline.get_bind_group_layout(0);
    let entries = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect::<Vec<_>>();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu linear dispatch bindings"),
        layout: &layout,
        entries: &entries,
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("jxl-wgpu linear dispatch"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn linear_dispatch_shape(device: &wgpu::Device, word_count: u64) -> Result<(u32, u32, u32)> {
    let limit = device.limits().max_compute_workgroups_per_dimension;
    let required_x = word_count.div_ceil(256);
    let workgroups_x =
        u32::try_from(required_x.min(u64::from(limit))).map_err(|_| Error::BufferSizeOverflow)?;
    let dispatch_width = workgroups_x
        .checked_mul(256)
        .ok_or(Error::BufferSizeOverflow)?;
    let workgroups_y = u32::try_from(word_count.div_ceil(u64::from(dispatch_width)))
        .map_err(|_| Error::BufferSizeOverflow)?;
    if workgroups_y > limit {
        return Err(Error::ResourceLimit(format!(
            "generic image output needs a {workgroups_x}x{workgroups_y} dispatch, exceeding the device limit {limit}"
        )));
    }
    Ok((workgroups_x, workgroups_y, dispatch_width))
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
    assert!(std::mem::size_of::<ImageOutputUniform>() == 128);
    assert!(std::mem::align_of::<ImageOutputUniform>() == 4);
};

#[cfg(test)]
mod tests {
    use std::mem::{align_of, size_of};

    use jxl_gpu_protocol::{
        Border2d, Extent2d, OutputDesc, OutputId, PlaneRole, PrecisionContract, RenderNode,
        SaveParams, Scale2d,
    };

    use crate::arena::{ArenaAllocation, ArenaPlan};

    use super::*;

    fn abi_words<T: Pod>(value: &T) -> &[u32] {
        bytemuck::cast_slice(std::slice::from_ref(value))
    }

    fn assert_sequential_words<T: Pod>(value: &T) {
        let expected =
            (1..=u32::try_from(size_of::<T>() / size_of::<u32>()).unwrap()).collect::<Vec<_>>();
        assert_eq!(abi_words(value), expected);
    }

    fn assert_wgsl_fields(shader: &str, name: &str, expected: &[&str]) {
        let marker = format!("struct {name} {{");
        let (_, after_marker) = shader
            .split_once(&marker)
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is missing"));
        let (body, _) = after_marker
            .split_once("};")
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is not terminated"));
        let actual = body
            .lines()
            .filter_map(|line| line.split_once(':').map(|(field, _)| field.trim()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "WGSL field-order drift for {name}");
    }

    #[test]
    fn canonical_classifier_drives_vpi_image_output_support() {
        let mut color_count = 0;
        let mut numeric_count = 0;
        for predefined in jxl_gpu_formats::vpi::VpiPitchLinearFormat::ALL {
            let format = predefined.pixel_format();
            let layout = ImageLayout::packed(Extent2d::new(3, 3), format.clone()).unwrap();
            match classify_pixel_format(&format).unwrap() {
                PixelFormatClass::Color(_) => {
                    color_count += 1;
                    prepare_image_output(&layout)
                        .unwrap_or_else(|error| panic!("{}: {error}", predefined.name()));
                }
                PixelFormatClass::Numeric(numeric) => {
                    numeric_count += 1;
                    let Error::Unsupported(message) = prepare_image_output(&layout).unwrap_err()
                    else {
                        panic!("{} did not return typed Unsupported", predefined.name());
                    };
                    assert!(message.contains("does not assign color semantics"));
                    if numeric.wgsl == WgslNumericCapability::UnavailableFloat64 {
                        assert!(message.contains("no native F64 arithmetic"));
                    }
                }
            }
        }
        assert_eq!(color_count, 20);
        assert_eq!(numeric_count, 10);
    }

    #[test]
    fn uniform_abi_sizes_are_explicit_and_naturally_aligned() {
        let sizes = [
            ("CopyParams", size_of::<CopyParams>(), 16),
            ("ModularParams", size_of::<ModularParams>(), 32),
            (
                "ChromaUpsampleUniform",
                size_of::<ChromaUpsampleUniform>(),
                32,
            ),
            ("Chroma2dUniform", size_of::<Chroma2dUniform>(), 32),
            ("GaborishUniform", size_of::<GaborishUniform>(), 32),
            ("GaborishRgbUniform", size_of::<GaborishRgbUniform>(), 80),
            ("EpfUniform", size_of::<EpfUniform>(), 80),
            ("UpsampleUniform", size_of::<UpsampleUniform>(), 32),
            ("YcbcrUniform", size_of::<YcbcrUniform>(), 32),
            ("XybUniform", size_of::<XybUniform>(), 128),
            ("TransferUniform", size_of::<TransferUniform>(), 64),
            ("PremultiplyUniform", size_of::<PremultiplyUniform>(), 32),
            ("SaveUniform", size_of::<SaveUniform>(), 32),
            ("ImageOutputUniform", size_of::<ImageOutputUniform>(), 128),
        ];
        for (name, actual, expected) in sizes {
            assert_eq!(actual, expected, "Rust/WGSL ABI size drift for {name}");
            assert_eq!(actual % 16, 0, "uniform {name} is not 16-byte sized");
        }
        for (name, alignment) in [
            ("CopyParams", align_of::<CopyParams>()),
            ("ModularParams", align_of::<ModularParams>()),
            ("ChromaUpsampleUniform", align_of::<ChromaUpsampleUniform>()),
            ("Chroma2dUniform", align_of::<Chroma2dUniform>()),
            ("GaborishUniform", align_of::<GaborishUniform>()),
            ("GaborishRgbUniform", align_of::<GaborishRgbUniform>()),
            ("EpfUniform", align_of::<EpfUniform>()),
            ("UpsampleUniform", align_of::<UpsampleUniform>()),
            ("YcbcrUniform", align_of::<YcbcrUniform>()),
            ("XybUniform", align_of::<XybUniform>()),
            ("TransferUniform", align_of::<TransferUniform>()),
            ("BlendUniform", align_of::<BlendUniform>()),
            ("PremultiplyUniform", align_of::<PremultiplyUniform>()),
            ("ExtendUniform", align_of::<ExtendUniform>()),
            ("SaveUniform", align_of::<SaveUniform>()),
            ("ImageOutputUniform", align_of::<ImageOutputUniform>()),
        ] {
            assert_eq!(
                alignment, 4,
                "Rust/WGSL natural ABI alignment drift for {name}"
            );
        }
    }

    #[test]
    fn uniform_rust_word_order_matches_wgsl_field_order() {
        assert_sequential_words(&CopyParams {
            width: 1,
            height: 2,
            input_stride: 3,
            output_stride: 4,
        });
        assert_wgsl_fields(
            include_str!("../shaders/copy.wgsl"),
            "Params",
            &["width", "height", "input_stride", "output_stride"],
        );

        assert_sequential_words(&ModularParams {
            width: 1,
            height: 2,
            input_stride: 3,
            output_stride: 4,
            multiplier: f32::from_bits(5),
            bias: f32::from_bits(6),
            _padding: [7, 8],
        });
        assert_wgsl_fields(
            include_str!("../shaders/modular_to_f32.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride",
                "output_stride",
                "multiplier",
                "bias",
                "_pad0",
                "_pad1",
            ],
        );

        assert_sequential_words(&ChromaUpsampleUniform {
            input_width: 1,
            input_height: 2,
            output_width: 3,
            output_height: 4,
            input_stride: 5,
            output_stride: 6,
            axis: 7,
            _padding: 8,
        });
        assert_wgsl_fields(
            include_str!("../shaders/chroma_upsample.wgsl"),
            "Params",
            &[
                "input_width",
                "input_height",
                "output_width",
                "output_height",
                "input_stride",
                "output_stride",
                "axis",
                "_pad0",
            ],
        );

        assert_sequential_words(&Chroma2dUniform {
            input_width: 1,
            input_height: 2,
            output_width: 3,
            output_height: 4,
            input_stride: 5,
            output_stride: 6,
            _padding: [7, 8],
        });
        assert_wgsl_fields(
            include_str!("../shaders/chroma_2d.wgsl"),
            "Params",
            &[
                "input_width",
                "input_height",
                "output_width",
                "output_height",
                "input_stride",
                "output_stride",
                "_pad0",
                "_pad1",
            ],
        );

        assert_sequential_words(&GaborishUniform {
            width: 1,
            height: 2,
            input_stride: 3,
            output_stride: 4,
            weight0: f32::from_bits(5),
            weight1: f32::from_bits(6),
            weight2: f32::from_bits(7),
            _padding: 8,
        });
        assert_wgsl_fields(
            include_str!("../shaders/gaborish.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride",
                "output_stride",
                "weight0",
                "weight1",
                "weight2",
                "_pad0",
            ],
        );

        assert_sequential_words(&GaborishRgbUniform {
            width: 1,
            height: 2,
            input_stride_x: 3,
            input_stride_y: 4,
            input_stride_b: 5,
            output_stride_x: 6,
            output_stride_y: 7,
            output_stride_b: 8,
            weights_x: [
                f32::from_bits(9),
                f32::from_bits(10),
                f32::from_bits(11),
                f32::from_bits(12),
            ],
            weights_y: [
                f32::from_bits(13),
                f32::from_bits(14),
                f32::from_bits(15),
                f32::from_bits(16),
            ],
            weights_b: [
                f32::from_bits(17),
                f32::from_bits(18),
                f32::from_bits(19),
                f32::from_bits(20),
            ],
        });
        assert_wgsl_fields(
            include_str!("../shaders/gaborish_rgb.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride_x",
                "input_stride_y",
                "input_stride_b",
                "output_stride_x",
                "output_stride_y",
                "output_stride_b",
                "weight0_x",
                "weight1_x",
                "weight2_x",
                "_pad_x",
                "weight0_y",
                "weight1_y",
                "weight2_y",
                "_pad_y",
                "weight0_b",
                "weight1_b",
                "weight2_b",
                "_pad_b",
            ],
        );

        assert_sequential_words(&EpfUniform {
            width: 1,
            height: 2,
            input_stride_x: 3,
            input_stride_y: 4,
            input_stride_b: 5,
            output_stride_x: 6,
            output_stride_y: 7,
            output_stride_b: 8,
            sigma_width: 9,
            sigma_height: 10,
            sigma_stride: 11,
            sigma_is_plane: 12,
            sigma_scale: f32::from_bits(13),
            border_sad_mul: f32::from_bits(14),
            channel_scale_x: f32::from_bits(15),
            channel_scale_y: f32::from_bits(16),
            channel_scale_b: f32::from_bits(17),
            min_sigma: f32::from_bits(18),
            _padding: [19, 20],
        });
        assert_wgsl_fields(
            include_str!("../shaders/epf.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride_x",
                "input_stride_y",
                "input_stride_b",
                "output_stride_x",
                "output_stride_y",
                "output_stride_b",
                "sigma_width",
                "sigma_height",
                "sigma_stride",
                "sigma_is_plane",
                "sigma_scale",
                "border_sad_mul",
                "channel_scale_x",
                "channel_scale_y",
                "channel_scale_b",
                "min_sigma",
                "_pad0",
                "_pad1",
            ],
        );

        assert_sequential_words(&UpsampleUniform {
            input_width: 1,
            input_height: 2,
            output_width: 3,
            output_height: 4,
            input_stride: 5,
            output_stride: 6,
            factor: 7,
            _padding: 8,
        });
        assert_wgsl_fields(
            include_str!("../shaders/upsample.wgsl"),
            "Params",
            &[
                "input_width",
                "input_height",
                "output_width",
                "output_height",
                "input_stride",
                "output_stride",
                "factor",
                "_pad0",
            ],
        );

        assert_sequential_words(&YcbcrUniform {
            width: 1,
            height: 2,
            cb_stride: 3,
            y_stride: 4,
            cr_stride: 5,
            output_stride: 6,
            component: 7,
            _padding: 8,
        });
        assert_wgsl_fields(
            include_str!("../shaders/ycbcr_to_rgb.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "cb_stride",
                "y_stride",
                "cr_stride",
                "output_stride",
                "component",
                "_pad0",
            ],
        );

        assert_sequential_words(&XybUniform {
            width: 1,
            height: 2,
            input_stride_x: 3,
            input_stride_y: 4,
            input_stride_b: 5,
            output_stride_r: 6,
            output_stride_g: 7,
            output_stride_b: 8,
            matrix_r: [
                f32::from_bits(9),
                f32::from_bits(10),
                f32::from_bits(11),
                f32::from_bits(12),
            ],
            matrix_g: [
                f32::from_bits(13),
                f32::from_bits(14),
                f32::from_bits(15),
                f32::from_bits(16),
            ],
            matrix_b: [
                f32::from_bits(17),
                f32::from_bits(18),
                f32::from_bits(19),
                f32::from_bits(20),
            ],
            bias_cbrt: [
                f32::from_bits(21),
                f32::from_bits(22),
                f32::from_bits(23),
                f32::from_bits(24),
            ],
            scaled_bias: [
                f32::from_bits(25),
                f32::from_bits(26),
                f32::from_bits(27),
                f32::from_bits(28),
            ],
            intensity_scale: f32::from_bits(29),
            _padding: [30, 31, 32],
        });
        assert_wgsl_fields(
            include_str!("../shaders/xyb_to_rgb.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride_x",
                "input_stride_y",
                "input_stride_b",
                "output_stride_r",
                "output_stride_g",
                "output_stride_b",
                "matrix_r",
                "matrix_g",
                "matrix_b",
                "bias_cbrt",
                "scaled_bias",
                "intensity_scale",
                "_pad0",
                "_pad1",
                "_pad2",
            ],
        );

        assert_sequential_words(&TransferUniform {
            width: 1,
            height: 2,
            input_stride_r: 3,
            input_stride_g: 4,
            input_stride_b: 5,
            output_stride_r: 6,
            output_stride_g: 7,
            output_stride_b: 8,
            transfer: 9,
            gamma: f32::from_bits(10),
            intensity_target: f32::from_bits(11),
            min_nits: f32::from_bits(12),
            luminance_rgb: [
                f32::from_bits(13),
                f32::from_bits(14),
                f32::from_bits(15),
                f32::from_bits(16),
            ],
        });
        assert_wgsl_fields(
            include_str!("../shaders/transfer_function.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "input_stride_r",
                "input_stride_g",
                "input_stride_b",
                "output_stride_r",
                "output_stride_g",
                "output_stride_b",
                "transfer",
                "gamma",
                "intensity_target",
                "min_nits",
                "luminance_rgb",
            ],
        );

        assert_sequential_words(&BlendUniform {
            width: 1,
            height: 2,
            base_stride: 3,
            source_stride: 4,
            output_stride: 5,
            base_alpha_stride: 6,
            source_alpha_stride: 7,
            mode: 8,
            component: 9,
            clamp: 10,
            alpha_associated: 11,
            has_alpha: 12,
        });
        assert_wgsl_fields(
            include_str!("../shaders/blend.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "base_stride",
                "source_stride",
                "output_stride",
                "base_alpha_stride",
                "source_alpha_stride",
                "mode",
                "component",
                "clamp",
                "alpha_associated",
                "has_alpha",
            ],
        );

        assert_sequential_words(&PremultiplyUniform {
            width: 1,
            height: 2,
            color_stride: 3,
            alpha_stride: 4,
            output_stride: 5,
            _padding: [6, 7, 8],
        });
        assert_wgsl_fields(
            include_str!("../shaders/premultiply_alpha.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "color_stride",
                "alpha_stride",
                "output_stride",
                "_pad0",
                "_pad1",
                "_pad2",
            ],
        );

        assert_sequential_words(&ExtendUniform {
            width: 1,
            height: 2,
            frame_width: 3,
            frame_height: 4,
            frame_stride: 5,
            reference_stride: 6,
            output_stride: 7,
            origin_x: 8,
            origin_y: 9,
            has_reference: 10,
            _padding: [11, 12],
        });
        assert_wgsl_fields(
            include_str!("../shaders/extend.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "frame_width",
                "frame_height",
                "frame_stride",
                "reference_stride",
                "output_stride",
                "origin_x",
                "origin_y",
                "has_reference",
                "_pad0",
                "_pad1",
            ],
        );

        assert_sequential_words(&SaveUniform {
            width: 1,
            height: 2,
            source_stride: 3,
            channels: 4,
            channel: 5,
            layout: 6,
            orientation: 7,
            _padding: 8,
        });
        assert_wgsl_fields(
            include_str!("../shaders/save.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "source_stride",
                "channels",
                "channel",
                "output_layout",
                "orientation",
                "_pad0",
            ],
        );

        assert_sequential_words(&ImageOutputUniform {
            width: 1,
            height: 2,
            source_width: 3,
            source_height: 4,
            r_stride: 5,
            g_stride: 6,
            b_stride: 7,
            kind: 8,
            channels: 9,
            order: 10,
            matrix: 11,
            range: 12,
            siting_x: 13,
            siting_y: 14,
            subsample_x: 15,
            subsample_y: 16,
            bits: 17,
            storage_bits: 18,
            plane0_offset: 19,
            plane0_stride: 20,
            plane1_offset: 21,
            plane1_stride: 22,
            plane2_offset: 23,
            plane2_stride: 24,
            plane3_offset: 25,
            plane3_stride: 26,
            logical_size: 27,
            dispatch_width: 28,
            orientation: 29,
            source_transfer: 30,
            target_transfer: 31,
            _padding: 32,
        });
        assert_wgsl_fields(
            include_str!("../shaders/rgb_to_image.wgsl"),
            "Params",
            &[
                "width",
                "height",
                "source_width",
                "source_height",
                "r_stride",
                "g_stride",
                "b_stride",
                "kind",
                "channels",
                "order",
                "matrix",
                "range",
                "siting_x",
                "siting_y",
                "subsample_x",
                "subsample_y",
                "bits",
                "storage_bits",
                "plane0_offset",
                "plane0_stride",
                "plane1_offset",
                "plane1_stride",
                "plane2_offset",
                "plane2_stride",
                "plane3_offset",
                "plane3_stride",
                "logical_size",
                "dispatch_width",
                "orientation",
                "source_transfer",
                "target_transfer",
                "_padding",
            ],
        );
    }

    fn execution(allocations: Vec<ArenaAllocation>, size_bytes: u64) -> ExecutionPlan {
        ExecutionPlan {
            memory_mode: MemoryMode::Resident,
            dispatches: Vec::new(),
            arena: ArenaPlan {
                size_bytes,
                peak_live_bytes: size_bytes,
                peak_scratch_bytes: 0,
                allocations,
            },
            tile_extents: BTreeMap::new(),
            resident_bytes: size_bytes,
            scratch_bytes: 0,
            groups_per_batch: 1,
        }
    }

    fn allocation(
        plane: u32,
        offset: u64,
        size: u64,
        first_use: usize,
        last_use: usize,
    ) -> ArenaAllocation {
        ArenaAllocation {
            plane: PlaneId(plane),
            offset,
            size,
            first_use,
            last_use,
        }
    }

    #[test]
    fn resident_slots_reuse_disjoint_lifetimes_without_double_counting() {
        let execution = execution(
            vec![
                allocation(0, 0, 17, 0, 0),
                allocation(1, 0, 9, 1, 1),
                allocation(2, 32, 33, 0, 2),
            ],
            80,
        );

        assert_eq!(
            resident_slot_sizes(&execution, 16).unwrap(),
            BTreeMap::from([(0, 32), (32, 48)])
        );
    }

    #[test]
    fn resident_slots_reject_false_aggregate_budget() {
        let execution = execution(
            vec![allocation(0, 0, 17, 0, 0), allocation(1, 32, 33, 1, 1)],
            64,
        );

        assert!(matches!(
            resident_slot_sizes(&execution, 16),
            Err(Error::Execution(message)) if message.contains("physical slots require 80 bytes")
        ));
    }

    #[test]
    fn resident_slots_reject_simultaneously_live_aliases() {
        let execution = execution(
            vec![allocation(0, 0, 16, 0, 1), allocation(1, 0, 16, 1, 2)],
            16,
        );

        assert!(matches!(
            resident_slot_sizes(&execution, 16),
            Err(Error::Execution(message)) if message.contains("simultaneously live")
        ));
    }

    #[test]
    fn transient_estimate_includes_uniform_packing_and_readback_buffers() {
        let extent = Extent2d::new(3, 3);
        let output = OutputId(0);
        let plan = RenderPlan {
            planes: vec![
                PlaneDesc {
                    id: PlaneId(0),
                    extent,
                    stride: 3,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Source,
                },
                PlaneDesc {
                    id: PlaneId(1),
                    extent,
                    stride: 3,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                },
            ],
            nodes: vec![
                RenderNode {
                    name: "copy".into(),
                    op: RenderOp::Copy,
                    inputs: vec![PlaneId(0)],
                    outputs: vec![PlaneId(1)],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                },
                RenderNode {
                    name: "save".into(),
                    op: RenderOp::Save(SaveParams {
                        output,
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(1)],
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    inputs: vec![PlaneId(1)],
                    outputs: Vec::new(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                },
            ],
            outputs: vec![OutputDesc {
                id: output,
                extent,
                sample_type: SampleType::F32,
                channels: 1,
                layout: OutputLayout::Planar,
                color_encoding: OutputColorEncoding::NonColor,
            }],
        };
        let expected = std::mem::size_of::<CopyParams>()
            + std::mem::size_of::<SaveUniform>()
            + 2 * 3 * 3 * std::mem::size_of::<f32>();
        assert_eq!(
            transient_bytes(
                &plan,
                &execution(Vec::new(), 0),
                &BTreeMap::new(),
                &BTreeMap::new(),
                OutputTarget {
                    mode: OutputMode::CpuReadback,
                    encoding: OutputEncoding::Original,
                    direct_readback: false,
                },
            )
            .unwrap(),
            expected as u64
        );
        assert_eq!(
            transient_bytes(
                &plan,
                &execution(Vec::new(), 0),
                &BTreeMap::new(),
                &BTreeMap::new(),
                OutputTarget {
                    mode: OutputMode::GpuOnly,
                    encoding: OutputEncoding::Original,
                    direct_readback: false,
                },
            )
            .unwrap(),
            (expected - 3 * 3 * std::mem::size_of::<f32>()) as u64
        );
        assert_eq!(
            transient_bytes(
                &plan,
                &execution(Vec::new(), 0),
                &BTreeMap::new(),
                &BTreeMap::new(),
                OutputTarget {
                    mode: OutputMode::CpuReadback,
                    encoding: OutputEncoding::Original,
                    direct_readback: true,
                },
            )
            .unwrap(),
            (expected - 3 * 3 * std::mem::size_of::<f32>()) as u64
        );
    }

    #[test]
    fn packed_storage_size_is_checked_against_both_device_limits() {
        let limits = wgpu::Limits {
            max_buffer_size: 1024,
            max_storage_buffer_binding_size: 512,
            ..wgpu::Limits::default()
        };
        validate_storage_buffer_size(&limits, 512, "test output").unwrap();
        assert!(matches!(
            validate_storage_buffer_size(&limits, 516, "test output"),
            Err(Error::ResourceLimit(message))
                if message.contains("516 bytes") && message.contains("limit 512")
        ));

        let limits = wgpu::Limits {
            max_buffer_size: 256,
            max_storage_buffer_binding_size: 512,
            ..wgpu::Limits::default()
        };
        assert!(matches!(
            validate_storage_buffer_size(&limits, 260, "test output"),
            Err(Error::ResourceLimit(message))
                if message.contains("260 bytes") && message.contains("limit 256")
        ));
    }
}
