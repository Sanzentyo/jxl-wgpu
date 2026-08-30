// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Bounded VarDCT packet validation and DCT8 execution.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::{
    GroupId, GroupPayload, PackedCoefficients, PlaneId, RenderNode, RenderOp, RenderPlan,
    ResourceData, ResourceId, ResourceUpdate, TransformKind, VarDctPacket, VarDctResource,
};
use wgpu::util::DeviceExt;

use crate::autotune::KernelVariant;
use crate::context::WgpuBackend;
use crate::pipeline_cache::PipelineKey;
use crate::upload::UploadedPlane;
use crate::{Error, Result};

const DCT8_COEFFICIENTS_PER_CHANNEL: usize = 64;
const DCT8_CHANNELS: usize = 3;
const DCT8_COEFFICIENTS_PER_TASK: usize = DCT8_COEFFICIENTS_PER_CHANNEL * DCT8_CHANNELS;
const DCT8_WORKGROUP_STORAGE_BYTES: u32 = 2 * 192 * 4;

type FlattenedResource = (Vec<GpuResourceVector>, u32, u32, u32, u32);

#[derive(Clone, Copy, Debug)]
pub(super) struct Rect {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

impl Rect {
    fn from_task(task: &jxl_gpu_protocol::TransformTask) -> Result<Self> {
        let Some((x, y)) = task.destinations[0] else {
            return Err(Error::InvalidPayload(
                "optimized DCT8 path requires an X-channel destination".into(),
            ));
        };
        if task
            .destinations
            .iter()
            .any(|&destination| destination != Some((x, y)))
        {
            return Err(Error::InvalidPayload(
                "optimized DCT8 path requires equal enabled channel destinations".into(),
            ));
        }
        Ok(Self {
            x,
            y,
            width: 8,
            height: 8,
        })
    }

    fn right(self) -> Option<u32> {
        self.x.checked_add(self.width)
    }

    fn bottom(self) -> Option<u32> {
        self.y.checked_add(self.height)
    }

    pub(super) fn is_within(self, width: u32, height: u32) -> bool {
        self.right().is_some_and(|right| right <= width)
            && self.bottom().is_some_and(|bottom| bottom <= height)
    }
}

/// Finds one overlapping rectangle pair in O(n log n).
///
/// Rectangles that are active at the same x coordinate cannot overlap in y before the first
/// collision is found. That invariant makes the immediate y predecessor and successor sufficient
/// for each insertion, while a min-heap expires rectangles whose right edge only touches the next
/// left edge.
pub(super) fn find_rect_overlap(rects: &[Rect]) -> Result<Option<(usize, usize)>> {
    let mut starts = rects
        .iter()
        .enumerate()
        .map(|(index, rect)| {
            let right = rect.right().ok_or(Error::BufferSizeOverflow)?;
            let bottom = rect.bottom().ok_or(Error::BufferSizeOverflow)?;
            Ok((rect.x, rect.y, right, bottom, index))
        })
        .collect::<Result<Vec<_>>>()?;
    starts.sort_unstable_by_key(|&(x, y, right, bottom, index)| (x, y, right, bottom, index));

    let mut ends: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut active_y: BTreeSet<(u32, usize)> = BTreeSet::new();
    for (x, y, right, bottom, index) in starts {
        while let Some(&Reverse((active_right, active_index))) = ends.peek() {
            if active_right > x {
                break;
            }
            ends.pop();
            let active = rects[active_index];
            active_y.remove(&(active.y, active_index));
        }

        if let Some(&(_, previous_index)) = active_y.range(..=(y, usize::MAX)).next_back()
            && rects[previous_index]
                .bottom()
                .ok_or(Error::BufferSizeOverflow)?
                > y
        {
            return Ok(Some((previous_index, index)));
        }
        if let Some(&(next_y, next_index)) = active_y.range((y, 0)..).next()
            && next_y < bottom
        {
            return Ok(Some((next_index, index)));
        }

        active_y.insert((y, index));
        ends.push(Reverse((right, index)));
    }
    Ok(None)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuTask {
    coefficient_offset: u32,
    destination_x: u32,
    destination_y: u32,
    quant_index: u32,
    matrix_index: u32,
    correlation_index: u32,
    lf_index: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuResourceVector([f32; 4]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Dct8Uniform {
    task_count: u32,
    output_width: u32,
    output_height: u32,
    output_stride_x: u32,
    output_stride_y: u32,
    output_stride_b: u32,
    quant_offset: u32,
    matrix_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    _padding: [u32; 2],
    quant_biases: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<GpuTask>() == 28);
    assert!(std::mem::align_of::<GpuTask>() == 4);
    assert!(std::mem::size_of::<GpuResourceVector>() == 16);
    assert!(std::mem::align_of::<GpuResourceVector>() == 16);
    assert!(std::mem::size_of::<Dct8Uniform>() == 64);
    assert!(std::mem::align_of::<Dct8Uniform>() == 16);
};

struct PreparedVarDct {
    coefficients: Vec<i32>,
    tasks: Vec<GpuTask>,
    resource_vectors: Vec<GpuResourceVector>,
    quant_offset: u32,
    matrix_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    quant_biases: [f32; 4],
}

pub(crate) fn has_node(plan: &RenderPlan) -> bool {
    plan.nodes
        .iter()
        .any(|node| matches!(node.op, RenderOp::VarDct))
}

pub(crate) fn validate_plan(plan: &RenderPlan) -> Result<()> {
    let nodes = plan
        .nodes
        .iter()
        .filter(|node| matches!(node.op, RenderOp::VarDct))
        .collect::<Vec<_>>();
    if nodes.len() > 1 {
        return Err(Error::Unsupported(
            "the portable backend accepts at most one VarDCT node per render plan".into(),
        ));
    }
    if let Some(node) = nodes.first() {
        validate_node(plan, node)?;
    }
    Ok(())
}

fn validate_node(plan: &RenderPlan, node: &RenderNode) -> Result<()> {
    let RenderOp::VarDct = node.op else {
        return Ok(());
    };
    if !node.inputs.is_empty() || node.outputs.len() != DCT8_CHANNELS || node.resources.len() != 1 {
        return Err(Error::InvalidPayload(format!(
            "VarDCT node '{}' requires no plane inputs, three F32 outputs, and one typed resource",
            node.name
        )));
    }
    let outputs = node
        .outputs
        .iter()
        .map(|id| {
            plan.planes
                .iter()
                .find(|plane| plane.id == *id)
                .ok_or(Error::MissingPlane(*id))
        })
        .collect::<Result<Vec<_>>>()?;
    if outputs
        .iter()
        .any(|plane| plane.sample_type != jxl_gpu_protocol::SampleType::F32)
    {
        return Err(Error::InvalidPayload(
            "VarDCT requires three F32 destination planes".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_packet(packet: &VarDctPacket) -> Result<()> {
    crate::vardct_general::validate_packet(packet)
}

fn coefficient_len(coefficients: &PackedCoefficients) -> Result<usize> {
    match coefficients {
        PackedCoefficients::DenseI32(values) => Ok(values.len()),
        PackedCoefficients::PackedI16 { words, .. } => {
            words.len().checked_mul(2).ok_or(Error::BufferSizeOverflow)
        }
    }
}

fn validate_packed_overflow(
    coefficients: &PackedCoefficients,
    coefficient_len: usize,
) -> Result<()> {
    let PackedCoefficients::PackedI16 { overflow, .. } = coefficients else {
        return Ok(());
    };
    let mut indices = BTreeSet::new();
    for replacement in overflow {
        let index = usize::try_from(replacement.index).map_err(|_| Error::BufferSizeOverflow)?;
        if index >= coefficient_len || !indices.insert(replacement.index) {
            return Err(Error::InvalidPayload(format!(
                "packed coefficient overflow index {} is out of range or duplicated",
                replacement.index
            )));
        }
        if i16::try_from(replacement.value).is_ok() {
            return Err(Error::InvalidPayload(format!(
                "packed coefficient overflow value {} fits in i16",
                replacement.value
            )));
        }
    }
    Ok(())
}

fn decode_coefficients(coefficients: &PackedCoefficients) -> Result<Vec<i32>> {
    validate_packed_overflow(coefficients, coefficient_len(coefficients)?)?;
    match coefficients {
        PackedCoefficients::DenseI32(values) => Ok(values.clone()),
        PackedCoefficients::PackedI16 { words, overflow } => {
            let mut values = Vec::with_capacity(
                words
                    .len()
                    .checked_mul(2)
                    .ok_or(Error::BufferSizeOverflow)?,
            );
            for word in words {
                values.push(i32::from(*word as u16 as i16));
                values.push(i32::from((*word >> 16) as u16 as i16));
            }
            for replacement in overflow {
                values[replacement.index as usize] = replacement.value;
            }
            Ok(values)
        }
    }
}

fn validate_resource(resource: &VarDctResource) -> Result<()> {
    if resource
        .quant_biases
        .iter()
        .chain(resource.quant_scales.iter().flatten())
        .chain(
            resource
                .dequant_matrices
                .iter()
                .flat_map(|matrix| matrix.scales.iter().flatten()),
        )
        .chain(resource.correlations.iter().flatten())
        .chain(resource.lf_coefficients.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidPayload(
            "VarDCT resource contains a non-finite parameter".into(),
        ));
    }
    for (index, matrix) in resource.dequant_matrices.iter().enumerate() {
        if matrix.transform != TransformKind::Dct8
            || matrix.scales.len() != DCT8_COEFFICIENTS_PER_CHANNEL
        {
            return Err(Error::InvalidPayload(format!(
                "DCT8 dequant matrix {index} has transform {:?} and {} entries, expected Dct8/64",
                matrix.transform,
                matrix.scales.len(),
            )));
        }
    }
    Ok(())
}

fn flatten_resource(resource: &VarDctResource) -> Result<FlattenedResource> {
    validate_resource(resource)?;
    let mut vectors = Vec::new();
    let quant_offset = 0;
    vectors.extend(
        resource
            .quant_scales
            .iter()
            .map(|scale| GpuResourceVector([scale[0], scale[1], scale[2], 0.0])),
    );
    let matrix_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(resource.dequant_matrices.iter().flat_map(|matrix| {
        matrix
            .scales
            .iter()
            .map(|scale| GpuResourceVector([scale[0], scale[1], scale[2], 0.0]))
    }));
    let correlation_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(
        resource
            .correlations
            .iter()
            .map(|correlation| GpuResourceVector([correlation[0], correlation[1], 0.0, 0.0])),
    );
    let lf_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(
        resource
            .lf_coefficients
            .iter()
            .map(|lf| GpuResourceVector([lf[0], lf[1], lf[2], 0.0])),
    );
    Ok((
        vectors,
        quant_offset,
        matrix_offset,
        correlation_offset,
        lf_offset,
    ))
}

fn prepare(
    plan: &RenderPlan,
    node: &RenderNode,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<PreparedVarDct> {
    validate_node(plan, node)?;
    let resource_id = node.resources[0];
    let resource = resources.get(&resource_id).ok_or_else(|| {
        Error::InvalidPayload(format!(
            "VarDCT resource {resource_id:?} has not been supplied"
        ))
    })?;
    let ResourceData::VarDct(resource) = &resource.data else {
        return Err(Error::InvalidPayload(format!(
            "VarDCT resource {resource_id:?} does not contain typed VarDCT parameters"
        )));
    };
    validate_resource(resource)?;

    let output = plan
        .planes
        .iter()
        .find(|plane| plane.id == node.outputs[0])
        .ok_or(Error::MissingPlane(node.outputs[0]))?;
    let output_width = output.extent.width;
    let output_height = output.extent.height;
    let mut coefficients = Vec::new();
    let mut tasks = Vec::new();
    let mut task_rects = Vec::new();

    for (group_id, group) in groups {
        let Some(packet) = &group.vardct else {
            continue;
        };
        validate_packet(packet)?;
        let decoded = decode_coefficients(&packet.coefficients)?;
        for bucket in &packet.buckets {
            for task in &bucket.tasks {
                let rect = Rect::from_task(task)?;
                if !rect.is_within(output_width, output_height) {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT task in group {group_id:?} writes {rect:?} outside {output_width}x{output_height}"
                    )));
                }
                task_rects.push(rect);
                if usize::from(task.quant_index) >= resource.quant_scales.len()
                    || usize::from(task.dequant_matrix_index) >= resource.dequant_matrices.len()
                    || usize::from(task.correlation_index) >= resource.correlations.len()
                    || usize::try_from(task.lf_offset)
                        .ok()
                        .is_none_or(|index| index >= resource.lf_coefficients.len())
                {
                    return Err(Error::InvalidPayload(
                        "DCT8 task references a missing dequantization or correlation entry".into(),
                    ));
                }
                let source_start = task.coefficient_offset as usize;
                let source_end = source_start
                    .checked_add(DCT8_COEFFICIENTS_PER_TASK)
                    .ok_or(Error::BufferSizeOverflow)?;
                let coefficient_offset =
                    u32::try_from(coefficients.len()).map_err(|_| Error::BufferSizeOverflow)?;
                coefficients.extend_from_slice(&decoded[source_start..source_end]);
                tasks.push(GpuTask {
                    coefficient_offset,
                    destination_x: rect.x,
                    destination_y: rect.y,
                    quant_index: u32::from(task.quant_index),
                    matrix_index: u32::from(task.dequant_matrix_index),
                    correlation_index: u32::from(task.correlation_index),
                    lf_index: task.lf_offset,
                });
            }
        }
    }

    if let Some((first, second)) = find_rect_overlap(&task_rects)? {
        let rect = task_rects[first.max(second)];
        return Err(Error::InvalidPayload(format!(
            "VarDCT task destination {rect:?} overlaps another task"
        )));
    }

    let (resource_vectors, quant_offset, matrix_offset, correlation_offset, lf_offset) =
        flatten_resource(resource)?;
    Ok(PreparedVarDct {
        coefficients,
        tasks,
        resource_vectors,
        quant_offset,
        matrix_offset,
        correlation_offset,
        lf_offset,
        quant_biases: resource.quant_biases,
    })
}

pub(crate) fn transient_bytes(
    plan: &RenderPlan,
    node: &RenderNode,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<u64> {
    if crate::vardct_general::is_required(groups, resources, node.resources[0]) {
        return crate::vardct_general::transient_bytes(plan, node, groups, resources);
    }
    prepared_transient_bytes(&prepare(plan, node, groups, resources)?)
}

fn prepared_transient_bytes(prepared: &PreparedVarDct) -> Result<u64> {
    let mut bytes = 0u64;
    if !prepared.tasks.is_empty() {
        for allocation in [
            buffer_bytes(&prepared.coefficients)?,
            buffer_bytes(&prepared.tasks)?,
            buffer_bytes(&prepared.resource_vectors)?,
            u64::try_from(std::mem::size_of::<Dct8Uniform>())
                .map_err(|_| Error::BufferSizeOverflow)?,
        ] {
            bytes = bytes
                .checked_add(allocation)
                .ok_or(Error::BufferSizeOverflow)?;
        }
    }
    Ok(bytes)
}

pub(crate) fn encode(
    backend: &WgpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    plan: &RenderPlan,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<u32> {
    if crate::vardct_general::is_required(groups, resources, node.resources[0]) {
        return crate::vardct_general::encode(
            backend, encoder, plan, node, planes, groups, resources,
        );
    }
    let prepared = prepare(plan, node, groups, resources)?;
    let outputs = node
        .outputs
        .iter()
        .map(|id| planes.get(id).ok_or(Error::MissingPlane(*id)))
        .collect::<Result<Vec<_>>>()?;
    if prepared.tasks.is_empty() {
        return Ok(0);
    }
    encode_dct8(backend, encoder, &outputs, &prepared)?;
    Ok(1)
}

fn encode_dct8(
    backend: &WgpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    outputs: &[&UploadedPlane],
    prepared: &PreparedVarDct,
) -> Result<()> {
    let device = &backend.device;
    let task_count = u32::try_from(prepared.tasks.len()).map_err(|_| Error::BufferSizeOverflow)?;
    if task_count > device.limits().max_compute_workgroups_per_dimension {
        return Err(Error::ResourceLimit(format!(
            "VarDCT needs {task_count} workgroups, exceeding the device limit"
        )));
    }
    if device.limits().max_compute_workgroup_storage_size < DCT8_WORKGROUP_STORAGE_BYTES {
        return Err(Error::ResourceLimit(format!(
            "VarDCT DCT8 needs {DCT8_WORKGROUP_STORAGE_BYTES} bytes of workgroup storage, device permits {}",
            device.limits().max_compute_workgroup_storage_size
        )));
    }
    ensure_upload_fits(device, &prepared.coefficients)?;
    ensure_upload_fits(device, &prepared.tasks)?;
    ensure_upload_fits(device, &prepared.resource_vectors)?;
    let coefficients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT coefficients"),
        contents: bytemuck::cast_slice(&prepared.coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tasks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT tasks"),
        contents: bytemuck::cast_slice(&prepared.tasks),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT resources"),
        contents: bytemuck::cast_slice(&prepared.resource_vectors),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let extent = outputs[0].desc.extent;
    let uniform = Dct8Uniform {
        task_count,
        output_width: extent.width,
        output_height: extent.height,
        output_stride_x: stride(&outputs[0].desc),
        output_stride_y: stride(&outputs[1].desc),
        output_stride_b: stride(&outputs[2].desc),
        quant_offset: prepared.quant_offset,
        matrix_offset: prepared.matrix_offset,
        correlation_offset: prepared.correlation_offset,
        lf_offset: prepared.lf_offset,
        _padding: [0; 2],
        quant_biases: prepared.quant_biases,
    };
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu VarDCT DCT8 params"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipeline = pipeline(
        backend,
        "jxl-wgpu vardct-dct8",
        KernelVariant::Tile8x8,
        wgpu::include_wgsl!("../shaders/vardct_dct8.wgsl"),
    );
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("jxl-wgpu VarDCT DCT8 bindings"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: coefficients.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: tasks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resources.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: outputs[0].binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: outputs[1].binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: outputs[2].binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("jxl-wgpu VarDCT DCT8"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(task_count, 1, 1);
    Ok(())
}

fn ensure_upload_fits<T>(device: &wgpu::Device, values: &[T]) -> Result<()> {
    let bytes = buffer_bytes(values)?;
    let limits = device.limits();
    let maximum = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    if bytes > maximum {
        return Err(Error::ResourceLimit(format!(
            "VarDCT storage upload needs {bytes} bytes, exceeding the device storage-buffer limit {maximum}"
        )));
    }
    Ok(())
}

fn buffer_bytes<T>(values: &[T]) -> Result<u64> {
    values
        .len()
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(Error::BufferSizeOverflow)
}

fn stride(desc: &jxl_gpu_protocol::PlaneDesc) -> u32 {
    if desc.stride == 0 {
        desc.extent.width
    } else {
        desc.stride
    }
}

fn pipeline(
    backend: &WgpuBackend,
    label: &str,
    variant: KernelVariant,
    descriptor: wgpu::ShaderModuleDescriptor<'static>,
) -> Arc<wgpu::ComputePipeline> {
    let key = PipelineKey::new(label, "main", variant, 0);
    if let Some(pipeline) = backend.pipelines.get(&key) {
        return pipeline;
    }
    match backend.pipelines.get_or_insert_with(key, || {
        let module = backend.device.create_shader_module(descriptor);
        Ok::<_, std::convert::Infallible>(backend.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ))
    }) {
        Ok(pipeline) => pipeline,
        Err(never) => match never {},
    }
}

#[cfg(test)]
fn basis(position: usize, frequency: usize) -> f32 {
    if frequency == 0 {
        1.0
    } else {
        std::f32::consts::SQRT_2
            * (((2 * position + 1) * frequency) as f32 * std::f32::consts::PI / 16.0).cos()
    }
}

#[cfg(test)]
fn adjust_quantized(value: i32, small_bias: f32, large_bias: f32) -> f32 {
    let value_f = value as f32;
    if (-1..=1).contains(&value) {
        value_f * small_bias
    } else {
        value_f - large_bias / value_f
    }
}

#[cfg(test)]
fn dequantized_reference_block(
    coefficients: &[i32],
    quant_index: usize,
    matrix_index: usize,
    correlation_index: usize,
    lf_index: usize,
    resource: &VarDctResource,
) -> [[f32; 64]; 3] {
    let quant_scale = resource.quant_scales[quant_index];
    let matrix = &resource.dequant_matrices[matrix_index].scales;
    let correlation = resource.correlations[correlation_index];
    let mut dequantized = [[0.0; 64]; 3];
    for index in 0..64 {
        let mut values = [0.0; 3];
        for channel in 0..3 {
            values[channel] = adjust_quantized(
                coefficients[channel * 64 + index],
                resource.quant_biases[channel],
                resource.quant_biases[3],
            ) * quant_scale[channel]
                * matrix[index][channel];
        }
        dequantized[0][index] = values[0].mul_add(1.0, correlation[0] * values[1]);
        dequantized[1][index] = values[1];
        dequantized[2][index] = values[2].mul_add(1.0, correlation[1] * values[1]);
    }
    let lf = resource.lf_coefficients[lf_index];
    for channel in 0..3 {
        dequantized[channel][0] = lf[channel];
    }
    dequantized
}

#[cfg(test)]
fn scalar_reference_block(
    coefficients: &[i32],
    quant_index: usize,
    matrix_index: usize,
    correlation_index: usize,
    lf_index: usize,
    resource: &VarDctResource,
) -> [[f32; 64]; 3] {
    let dequantized = dequantized_reference_block(
        coefficients,
        quant_index,
        matrix_index,
        correlation_index,
        lf_index,
        resource,
    );
    let mut horizontal = [[0.0; 64]; 3];
    for channel in 0..3 {
        for frequency_y in 0..8 {
            for x in 0..8 {
                horizontal[channel][frequency_y * 8 + x] = (0..8).fold(0.0, |sum, frequency_x| {
                    dequantized[channel][frequency_y * 8 + frequency_x]
                        .mul_add(basis(x, frequency_x), sum)
                });
            }
        }
    }
    let mut output = [[0.0; 64]; 3];
    for channel in 0..3 {
        for y in 0..8 {
            for x in 0..8 {
                output[channel][x * 8 + y] = (0..8).fold(0.0, |sum, frequency_y| {
                    horizontal[channel][frequency_y * 8 + x].mul_add(basis(y, frequency_y), sum)
                });
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    fn abi_words<T: Pod>(value: &T) -> &[u32] {
        bytemuck::cast_slice(std::slice::from_ref(value))
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
    fn vardct_gpu_abi_sizes_are_explicit_and_aligned() {
        assert_eq!(size_of::<GpuTask>(), 28);
        assert_eq!(size_of::<GpuResourceVector>(), 16);
        assert_eq!(std::mem::align_of::<GpuResourceVector>(), 16);
        assert_eq!(size_of::<Dct8Uniform>(), 64);
        assert_eq!(size_of::<Dct8Uniform>() % 16, 0);
        assert_eq!(std::mem::align_of::<Dct8Uniform>(), 16);
        assert_eq!(DCT8_WORKGROUP_STORAGE_BYTES, 1536);
    }

    #[test]
    fn vardct_rust_word_order_matches_wgsl_field_order() {
        let task = GpuTask {
            coefficient_offset: 1,
            destination_x: 2,
            destination_y: 3,
            quant_index: 4,
            matrix_index: 5,
            correlation_index: 6,
            lf_index: 7,
        };
        assert_eq!(abi_words(&task), &[1, 2, 3, 4, 5, 6, 7]);

        let resource = GpuResourceVector([
            f32::from_bits(1),
            f32::from_bits(2),
            f32::from_bits(3),
            f32::from_bits(4),
        ]);
        assert_eq!(abi_words(&resource), &[1, 2, 3, 4]);

        let params = Dct8Uniform {
            task_count: 1,
            output_width: 2,
            output_height: 3,
            output_stride_x: 4,
            output_stride_y: 5,
            output_stride_b: 6,
            quant_offset: 7,
            matrix_offset: 8,
            correlation_offset: 9,
            lf_offset: 10,
            _padding: [11, 12],
            quant_biases: [
                f32::from_bits(13),
                f32::from_bits(14),
                f32::from_bits(15),
                f32::from_bits(16),
            ],
        };
        assert_eq!(abi_words(&params), &(1..=16).collect::<Vec<_>>());

        let shader = include_str!("../shaders/vardct_dct8.wgsl");
        assert!(
            shader.contains("var<storage, read> resources: array<vec4<f32>>;"),
            "WGSL resource element ABI drifted from a 16-byte vec4"
        );
        assert_wgsl_fields(
            shader,
            "Task",
            &[
                "coefficient_offset",
                "destination_x",
                "destination_y",
                "quant_index",
                "matrix_index",
                "correlation_index",
                "lf_index",
            ],
        );
        assert_wgsl_fields(
            shader,
            "Params",
            &[
                "task_count",
                "output_width",
                "output_height",
                "output_stride_x",
                "output_stride_y",
                "output_stride_b",
                "quant_offset",
                "matrix_offset",
                "correlation_offset",
                "lf_offset",
                "_pad0",
                "_pad1",
                "quant_biases",
            ],
        );
    }
    use jxl_gpu_protocol::{
        Border2d, CoefficientOverflow, Extent2d, FrameSessionDesc, GroupPayload, MemoryMode,
        OutputDesc, OutputId, OutputLayout, PackedCoefficients, PlaneData, PlaneDesc, PlaneRole,
        PrecisionContract, PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderPlan,
        ResourceUpdate, SaveParams, Scale2d, TransformBucket, TransformTask, VarDctDequantMatrix,
        VarDctPacket,
    };

    use crate::{WgpuBackend, WgpuBackendConfig};

    fn resource() -> VarDctResource {
        VarDctResource {
            quant_biases: [1.0, 1.0, 1.0, 0.0],
            quant_scales: vec![[1.0, 0.5, 2.0]],
            dequant_matrices: vec![VarDctDequantMatrix {
                transform: TransformKind::Dct8,
                scales: vec![[1.0, 1.0, 1.0]; 64],
            }],
            correlations: vec![[0.25, -0.5]],
            lf_coefficients: vec![[3.0, 4.0, 4.0], [-5.0, 2.0, 1.0]],
        }
    }

    fn dct8_task(coefficient_offset: u32, destination_x: u32, destination_y: u32) -> TransformTask {
        TransformTask {
            coefficient_offset,
            destinations: [Some((destination_x, destination_y)); 3],
            quant_index: 0,
            dequant_matrix_index: 0,
            correlation_index: 0,
            lf_offset: coefficient_offset / 192,
        }
    }

    fn plan(extent: Extent2d) -> RenderPlan {
        let channels = [PlaneId(0), PlaneId(1), PlaneId(2)];
        RenderPlan {
            planes: channels
                .iter()
                .map(|id| PlaneDesc {
                    id: *id,
                    extent,
                    stride: extent.width,
                    sample_type: jxl_gpu_protocol::SampleType::F32,
                    role: PlaneRole::Intermediate,
                })
                .collect(),
            nodes: vec![
                RenderNode {
                    name: "vardct".into(),
                    op: RenderOp::VarDct,
                    inputs: Vec::new(),
                    outputs: channels.to_vec(),
                    resources: vec![ResourceId(0)],
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::default(),
                },
                RenderNode {
                    name: "save".into(),
                    op: RenderOp::Save(SaveParams {
                        output: OutputId(0),
                        sample_type: jxl_gpu_protocol::SampleType::F32,
                        channels: channels.to_vec(),
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    inputs: channels.to_vec(),
                    outputs: Vec::new(),
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                },
            ],
            outputs: vec![OutputDesc {
                id: OutputId(0),
                extent,
                sample_type: jxl_gpu_protocol::SampleType::F32,
                channels: 3,
                layout: OutputLayout::Planar,
                color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
            }],
        }
    }

    fn frame(extent: Extent2d) -> FrameSessionDesc {
        FrameSessionDesc {
            frame_extent: extent,
            group_extent: extent,
            group_count: 1,
            precision: PrecisionPolicy::F32Only,
            memory_mode: MemoryMode::Resident,
            max_resident_bytes: 16 * 1024 * 1024,
            max_scratch_bytes: 16 * 1024 * 1024,
        }
    }

    fn test_backend() -> Option<WgpuBackend> {
        match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        })) {
            Ok(backend) => Some(backend),
            Err(Error::NoAdapter) => {
                eprintln!("skipping GPU test: no wgpu adapter is available");
                None
            }
            Err(error) => panic!("failed to initialize GPU test device: {error}"),
        }
    }

    fn pack_i16(values: &[i32]) -> PackedCoefficients {
        assert_eq!(values.len() % 2, 0);
        let mut words = Vec::with_capacity(values.len() / 2);
        let mut overflow = Vec::new();
        for (word_index, pair) in values.chunks_exact(2).enumerate() {
            let mut lanes = [0_u16; 2];
            for (lane_index, value) in pair.iter().copied().enumerate() {
                if let Ok(value) = i16::try_from(value) {
                    lanes[lane_index] = value as u16;
                } else {
                    overflow.push(CoefficientOverflow {
                        index: u32::try_from(word_index * 2 + lane_index).unwrap(),
                        value,
                    });
                }
            }
            words.push(u32::from(lanes[0]) | (u32::from(lanes[1]) << 16));
        }
        PackedCoefficients::PackedI16 { words, overflow }
    }

    fn codec_reference_block(
        coefficients: &[i32],
        quant_index: usize,
        matrix_index: usize,
        correlation_index: usize,
        lf_index: usize,
        resource: &VarDctResource,
    ) -> [[f32; 64]; 3] {
        let mut output = dequantized_reference_block(
            coefficients,
            quant_index,
            matrix_index,
            correlation_index,
            lf_index,
            resource,
        );
        for channel in &mut output {
            // This is the same public dispatch used by jxl::frame::group after dequantization.
            // DCT8 takes its DC from the LF plane, so feed the already dequantized natural-order
            // DC value through that argument exactly as transform_to_pixels expects.
            let mut lf = [channel[0]];
            jxl_transforms::transform::transform_to_pixels(
                jxl_transforms::transform_map::HfTransformType::DCT,
                &mut lf,
                channel,
            );
        }
        output
    }

    fn write_reference_block(
        destination: &mut [Vec<f32>; 3],
        output_width: usize,
        task: &TransformTask,
        coefficients: &[i32],
        resource: &VarDctResource,
    ) {
        let block = codec_reference_block(
            coefficients,
            usize::from(task.quant_index),
            usize::from(task.dequant_matrix_index),
            usize::from(task.correlation_index),
            task.lf_offset as usize,
            resource,
        );
        for channel in 0..3 {
            for y in 0..8 {
                for x in 0..8 {
                    let (destination_x, destination_y) = task.destinations[channel].unwrap();
                    let destination_index =
                        (destination_y as usize + y) * output_width + destination_x as usize + x;
                    destination[channel][destination_index] = block[channel][y * 8 + x];
                }
            }
        }
    }

    #[test]
    fn packed_i16_decodes_signed_lanes_and_overflow() {
        let packed = PackedCoefficients::PackedI16 {
            words: vec![0x8000_7fff, 0x0001_ffff],
            overflow: vec![CoefficientOverflow {
                index: 1,
                value: 40_000,
            }],
        };
        assert_eq!(
            decode_coefficients(&packed).unwrap(),
            [32_767, 40_000, -1, 1]
        );
    }

    #[test]
    fn dense_i32_decodes_without_repacking() {
        let packed = PackedCoefficients::DenseI32(vec![i32::MIN, -7, 0, 9, i32::MAX]);
        assert_eq!(
            decode_coefficients(&packed).unwrap(),
            [i32::MIN, -7, 0, 9, i32::MAX]
        );
    }

    fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn rectangle_sweep_distinguishes_touching_from_adversarial_overlap() {
        let touching = [
            rect(0, 0, 8, 8),
            rect(8, 0, 8, 8),
            rect(0, 8, 8, 8),
            rect(8, 8, 8, 8),
        ];
        assert_eq!(find_rect_overlap(&touching).unwrap(), None);

        for rectangles in [
            vec![rect(40, 40, 4, 4), rect(0, 10, 100, 2), rect(50, 0, 2, 100)],
            vec![rect(10, 10, 20, 20), rect(12, 12, 1, 1)],
            vec![rect(0, 0, 2, 100), rect(1, 99, 100, 2)],
            vec![rect(5, 5, 10, 10), rect(5, 14, 10, 10)],
        ] {
            assert!(find_rect_overlap(&rectangles).unwrap().is_some());
        }

        assert!(matches!(
            find_rect_overlap(&[rect(u32::MAX, 0, 1, 1)]),
            Err(Error::BufferSizeOverflow)
        ));
        assert!(matches!(
            find_rect_overlap(&[rect(0, u32::MAX, 1, 1)]),
            Err(Error::BufferSizeOverflow)
        ));
    }

    #[test]
    fn rectangle_sweep_scales_to_large_tiled_frames() {
        const TILES_PER_AXIS: u32 = 256;
        let mut rectangles = Vec::with_capacity((TILES_PER_AXIS * TILES_PER_AXIS) as usize);
        // Reverse both axes so the input order cannot accidentally substitute for the sweep sort.
        for y in (0..TILES_PER_AXIS).rev() {
            for x in (0..TILES_PER_AXIS).rev() {
                rectangles.push(rect(x * 8, y * 8, 8, 8));
            }
        }
        assert_eq!(rectangles.len(), 65_536);
        assert_eq!(find_rect_overlap(&rectangles).unwrap(), None);

        rectangles.push(rect(1023, 1023, 2, 2));
        assert!(find_rect_overlap(&rectangles).unwrap().is_some());
    }

    #[test]
    fn rectangle_sweep_matches_quadratic_oracle_on_scrambled_inputs() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut random_u32 = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 32) as u32
        };
        for count in 0..128usize {
            let rectangles = (0..count)
                .map(|_| {
                    rect(
                        random_u32() % 512,
                        random_u32() % 512,
                        random_u32() % 24 + 1,
                        random_u32() % 24 + 1,
                    )
                })
                .collect::<Vec<_>>();
            let quadratic = rectangles.iter().enumerate().any(|(index, first)| {
                rectangles[index + 1..].iter().any(|second| {
                    first.x < second.right().unwrap()
                        && second.x < first.right().unwrap()
                        && first.y < second.bottom().unwrap()
                        && second.y < first.bottom().unwrap()
                })
            });
            let swept = find_rect_overlap(&rectangles).unwrap();
            assert_eq!(swept.is_some(), quadratic, "rectangle count {count}");
            if let Some((first, second)) = swept {
                let first = rectangles[first];
                let second = rectangles[second];
                assert!(
                    first.x < second.right().unwrap()
                        && second.x < first.right().unwrap()
                        && first.y < second.bottom().unwrap()
                        && second.y < first.bottom().unwrap()
                );
            }
        }
    }

    #[test]
    fn transient_estimate_matches_every_explicit_vardct_buffer() {
        let prepared = PreparedVarDct {
            coefficients: vec![0; 192],
            tasks: vec![GpuTask::zeroed()],
            resource_vectors: vec![GpuResourceVector::zeroed(); 3],
            quant_offset: 0,
            matrix_offset: 0,
            correlation_offset: 0,
            lf_offset: 0,
            quant_biases: [0.0; 4],
        };
        let expected = 192 * std::mem::size_of::<i32>()
            + std::mem::size_of::<GpuTask>()
            + 3 * std::mem::size_of::<GpuResourceVector>()
            + std::mem::size_of::<Dct8Uniform>();
        assert_eq!(
            prepared_transient_bytes(&prepared).unwrap(),
            expected as u64
        );
    }

    #[test]
    fn scalar_reference_preserves_dc_and_applies_correlation() {
        let mut coefficients = vec![0; 192];
        coefficients[0] = 2;
        coefficients[64] = 8;
        coefficients[128] = 3;
        let output = scalar_reference_block(&coefficients, 0, 0, 0, 0, &resource());
        assert!(output[0].iter().all(|value| (*value - 3.0).abs() < 1.0e-6));
        assert!(output[1].iter().all(|value| (*value - 4.0).abs() < 1.0e-6));
        assert!(output[2].iter().all(|value| (*value - 4.0).abs() < 1.0e-6));
    }

    #[test]
    fn scalar_reference_matches_codec_transform_order_and_normalization() {
        let mut coefficients = vec![0; 192];
        for channel in 0..3 {
            for frequency_y in 0..8 {
                for frequency_x in 0..8 {
                    let index = channel * 64 + frequency_y * 8 + frequency_x;
                    coefficients[index] = ((index * 37 + 11) % 19) as i32 - 9;
                }
            }
        }
        coefficients[0] = 17;
        coefficients[64] = -13;
        coefficients[128] = 5;
        let scalar = scalar_reference_block(&coefficients, 0, 0, 0, 0, &resource());
        let codec = codec_reference_block(&coefficients, 0, 0, 0, 0, &resource());
        for channel in 0..3 {
            for index in 0..64 {
                let tolerance = 1.0e-4_f32.max(codec[channel][index].abs() * 5.0e-6);
                assert!(
                    (scalar[channel][index] - codec[channel][index]).abs() <= tolerance,
                    "channel {channel} sample {index}: scalar {}, codec {}, tolerance {tolerance}",
                    scalar[channel][index],
                    codec[channel][index]
                );
            }
        }

        let mut dc_only = vec![0; 192];
        // The entropy packet's coefficient-zero slot is deliberately ignored: codec DCT8 takes
        // DC from the separately decoded LF plane.
        dc_only[0] = 99;
        let codec_dc = codec_reference_block(&dc_only, 0, 0, 0, 0, &resource());
        assert!(
            codec_dc[0]
                .iter()
                .all(|value| (*value - 3.0).abs() < 1.0e-6)
        );
    }

    #[test]
    fn non_dct8_buckets_and_split_destinations_are_structurally_valid() {
        let non_dct8 = |tasks| VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(Vec::new()),
            buckets: vec![TransformBucket {
                transform: TransformKind::Dct4x4,
                tasks,
            }],
        };

        assert!(validate_packet(&non_dct8(Vec::new())).is_ok());
        assert!(
            validate_packet(&VarDctPacket {
                revision: 0,
                last_pass: 0,
                coefficients: PackedCoefficients::DenseI32(vec![0; 192]),
                buckets: vec![TransformBucket {
                    transform: TransformKind::Dct4x4,
                    tasks: vec![TransformTask {
                        coefficient_offset: 0,
                        destinations: [Some((0, 0)); 3],
                        quant_index: 0,
                        dequant_matrix_index: 0,
                        correlation_index: 0,
                        lf_offset: 0,
                    }],
                }],
            })
            .is_ok()
        );

        let mut split = dct8_task(0, 0, 0);
        split.destinations[2] = Some((4, 0));
        let split_packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(vec![0; 192]),
            buckets: vec![TransformBucket {
                transform: TransformKind::Dct8,
                tasks: vec![split],
            }],
        };
        assert!(validate_packet(&split_packet).is_ok());
    }

    #[test]
    fn enqueue_accepts_non_dct8_bucket_and_split_destinations() {
        let Some(backend) = test_backend() else {
            return;
        };
        let extent = Extent2d::new(8, 8);
        let mut session = backend
            .create_session(&frame(extent), Arc::new(plan(extent)))
            .expect("create VarDCT session");
        let payload = |packet| GroupPayload {
            group: GroupId(0),
            revision: 0,
            complete: true,
            planes: Vec::new(),
            vardct: Some(packet),
        };
        let mut split = dct8_task(0, 0, 0);
        split.destinations[1] = None;
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(vec![0; 192]),
            buckets: vec![
                TransformBucket {
                    transform: TransformKind::Dct4x4,
                    tasks: Vec::new(),
                },
                TransformBucket {
                    transform: TransformKind::Dct8,
                    tasks: vec![split],
                },
            ],
        };
        session
            .enqueue(payload(packet))
            .expect("full transform descriptors are accepted before GPU submission");
    }

    #[test]
    fn prepare_preserves_typed_task_overlap_errors() {
        let extent = Extent2d::new(16, 8);
        let plan = plan(extent);
        let node = &plan.nodes[0];
        let resources = BTreeMap::from([(
            ResourceId(0),
            ResourceUpdate {
                id: ResourceId(0),
                revision: 0,
                data: ResourceData::VarDct(resource()),
            },
        )]);
        let payload = |packet| {
            BTreeMap::from([(
                GroupId(0),
                GroupPayload {
                    group: GroupId(0),
                    revision: 0,
                    complete: true,
                    planes: Vec::new(),
                    vardct: Some(packet),
                },
            )])
        };

        let overlapping_tasks = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(vec![0; 384]),
            buckets: vec![TransformBucket {
                transform: TransformKind::Dct8,
                tasks: vec![dct8_task(0, 0, 0), dct8_task(192, 4, 0)],
            }],
        };
        assert!(matches!(
            prepare(&plan, node, &payload(overlapping_tasks), &resources),
            Err(Error::InvalidPayload(message))
                if message == "VarDCT task destination Rect { x: 4, y: 0, width: 8, height: 8 } overlaps another task"
        ));
    }

    #[test]
    fn gpu_dct8_matches_codec_reference_at_odd_tail() {
        let Some(backend) = test_backend() else {
            return;
        };
        eprintln!("running VarDCT test on {:?}", backend.adapter_info());

        let extent = Extent2d::new(19, 11);
        let resource = resource();
        let first = dct8_task(0, 1, 1);
        let second = dct8_task(192, 11, 3);
        let mut coefficients = vec![0; 384];
        coefficients[0] = 40_000;
        coefficients[1] = 3;
        coefficients[64] = 8;
        coefficients[72] = -2;
        coefficients[128] = 3;
        coefficients[192] = -5;
        coefficients[192 + 7] = 2;
        coefficients[192 + 64] = 4;
        coefficients[192 + 64 + 24] = -3;
        coefficients[192 + 128] = 1;
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: pack_i16(&coefficients),
            buckets: vec![TransformBucket {
                transform: TransformKind::Dct8,
                tasks: vec![first, second],
            }],
        };

        let mut expected = [vec![0.0; 19 * 11], vec![0.0; 19 * 11], vec![0.0; 19 * 11]];
        write_reference_block(&mut expected, 19, &first, &coefficients[..192], &resource);
        write_reference_block(
            &mut expected,
            19,
            &second,
            &coefficients[192..384],
            &resource,
        );

        let plan = Arc::new(plan(extent));
        let mut session = backend
            .create_session(&frame(extent), plan)
            .expect("create VarDCT session");
        session
            .update_resource(ResourceUpdate {
                id: ResourceId(0),
                revision: 0,
                data: ResourceData::VarDct(resource),
            })
            .expect("supply VarDCT dequantization resource");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: Vec::new(),
                vardct: Some(packet),
            })
            .expect("enqueue VarDCT packet");
        let token = session.submit(RenderIntent::Final).expect("submit VarDCT");
        let frame = session.wait(token).expect("read VarDCT output");
        let PlaneData::F32(actual) = &frame.outputs[0].data else {
            panic!("VarDCT output was not F32");
        };
        assert_eq!(actual.len(), 19 * 11 * 3);
        for channel in 0..3 {
            for (index, (&expected, &actual)) in expected[channel]
                .iter()
                .zip(&actual[channel * 19 * 11..(channel + 1) * 19 * 11])
                .enumerate()
            {
                let tolerance = 5.0e-4_f32.max(expected.abs() * 3.0e-6);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "channel {channel} sample {index}: GPU {actual}, CPU {expected}, tolerance {tolerance}"
                );
            }
        }
        assert!((actual[10 * 19 + 18] - expected[0][10 * 19 + 18]).abs() < 1.0e-4);
        assert_eq!(actual[9 * 19], 0.0);
        assert_eq!(actual[19 * 11 + 9 * 19], 0.0);
    }
}
