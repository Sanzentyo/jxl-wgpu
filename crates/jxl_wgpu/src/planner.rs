// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Capability, fusion, dispatch, and memory planning for a frame render graph.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use jxl_gpu_protocol::{
    BlendComponent, BlendParams, ChromaAxis, EpfParams, Extent2d, FrameSessionDesc, MemoryMode,
    PlaneId, PlaneRole, PrecisionContract, PrecisionPolicy, RenderNode, RenderOp, RenderOpKind,
    RenderPlan, ResourceId, SampleType,
};

use crate::arena::{ArenaPlan, ArenaPlanner};
use crate::autotune::{KernelPolicy, KernelVariant};
use crate::context::WgpuMemoryPolicy;
use crate::{Error, Result};

/// A shader template selected for one or more graph nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FusedKernel {
    Single(RenderOpKind),
    Chroma2d,
    GaborishRgb,
}

impl FusedKernel {
    pub const fn key(&self) -> &'static str {
        match self {
            Self::Single(RenderOpKind::Copy) => "copy",
            Self::Single(RenderOpKind::ModularToF32) => "modular_to_f32",
            Self::Single(RenderOpKind::ChromaUpsample) => "chroma_upsample",
            Self::Single(RenderOpKind::Gaborish) => "gaborish",
            Self::Single(RenderOpKind::Epf) => "epf",
            Self::Single(RenderOpKind::Upsample) => "upsample",
            Self::Single(RenderOpKind::VarDct) => "vardct",
            Self::Single(RenderOpKind::AddNoise) => "add_noise",
            Self::Single(RenderOpKind::XybToRgb) => "xyb_to_rgb",
            Self::Single(RenderOpKind::YcbcrToRgb) => "ycbcr_to_rgb",
            Self::Single(RenderOpKind::TransferFunction) => "transfer_function",
            Self::Single(RenderOpKind::Blend) => "blend",
            Self::Single(RenderOpKind::PremultiplyAlpha) => "premultiply_alpha",
            Self::Single(RenderOpKind::Convert) => "convert",
            Self::Single(RenderOpKind::Extend) => "extend",
            Self::Single(RenderOpKind::Save) => "save",
            Self::Chroma2d => "chroma_2d",
            Self::GaborishRgb => "gaborish_rgb",
        }
    }

    pub const fn default_variant(&self) -> KernelVariant {
        match self {
            Self::Single(RenderOpKind::VarDct) => KernelVariant::Tile8x8,
            Self::Single(_) | Self::Chroma2d | Self::GaborishRgb => KernelVariant::Tile16x16,
        }
    }

    pub const fn is_workgroup_tunable(&self) -> bool {
        !matches!(self, Self::Single(RenderOpKind::VarDct))
    }

    pub fn variant_for(&self, policy: &KernelPolicy) -> Result<KernelVariant> {
        let default = self.default_variant();
        if self.is_workgroup_tunable() {
            policy.variant_for(self.key(), default)
        } else {
            Ok(default)
        }
    }
}

/// One compute dispatch after safe fusion.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedDispatch {
    pub label: Arc<str>,
    pub kernel: FusedKernel,
    /// Original `RenderPlan::nodes` indices, in execution order.
    pub node_indices: Vec<usize>,
    pub resources: Vec<ResourceId>,
    pub precision: PrecisionContract,
    pub variant: KernelVariant,
    pub workgroup_size: (u32, u32),
    pub workgroups: (u32, u32, u32),
}

/// Fully resolved execution metadata for a frame session.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan {
    /// Currently always `Resident`. Explicit streaming requests are rejected until tiled
    /// execution is connected to the scheduler.
    pub memory_mode: MemoryMode,
    pub dispatches: Vec<PlannedDispatch>,
    pub arena: ArenaPlan,
    /// Physical extent used for each logical plane. In resident mode these are full-frame extents.
    pub tile_extents: BTreeMap<PlaneId, Extent2d>,
    /// Aggregate capacity of the physical resident plane slots.
    pub resident_bytes: u64,
    /// Peak simultaneously live capacity of intermediate plane allocations.
    pub scratch_bytes: u64,
    pub groups_per_batch: u32,
}

#[derive(Clone, Debug)]
pub struct Planner {
    limits: wgpu::Limits,
    memory: WgpuMemoryPolicy,
    kernel_policy: KernelPolicy,
    supported_ops: BTreeSet<RenderOpKind>,
    supports_f16: bool,
}

impl Planner {
    pub fn new(limits: wgpu::Limits, memory: WgpuMemoryPolicy) -> Self {
        Self {
            limits,
            memory,
            kernel_policy: KernelPolicy::Default,
            supported_ops: portable_supported_ops(),
            supports_f16: false,
        }
    }

    pub fn with_kernel_policy(mut self, kernel_policy: KernelPolicy) -> Self {
        self.kernel_policy = kernel_policy;
        self
    }

    pub fn with_supported_ops(
        mut self,
        supported_ops: impl IntoIterator<Item = RenderOpKind>,
    ) -> Self {
        self.supported_ops = supported_ops.into_iter().collect();
        self
    }

    pub const fn with_f16_support(mut self, supports_f16: bool) -> Self {
        self.supports_f16 = supports_f16;
        self
    }

    pub fn supported_ops(&self) -> &BTreeSet<RenderOpKind> {
        &self.supported_ops
    }

    pub fn plan(&self, frame: &FrameSessionDesc, plan: &RenderPlan) -> Result<ExecutionPlan> {
        plan.validate()?;
        self.validate_frame(frame)?;
        self.validate_nodes(frame, plan)?;

        if frame.memory_mode == MemoryMode::Streaming {
            return Err(Error::Unsupported(
                "streaming execution is not implemented by the scheduler; use Resident or Auto"
                    .into(),
            ));
        }

        let tile_extents = plan
            .planes
            .iter()
            .map(|plane| (plane.id, plane.extent))
            .collect();
        let dispatches = self.plan_dispatches(plan, &tile_extents)?;
        let mut node_steps = vec![usize::MAX; plan.nodes.len()];
        for (step, dispatch) in dispatches.iter().enumerate() {
            for &node_index in &dispatch.node_indices {
                node_steps[node_index] = step;
            }
        }

        let binding_alignment = u64::from(self.limits.min_storage_buffer_offset_alignment)
            .max(4)
            .next_power_of_two();
        // The virtual offsets identify independent lifetime slots. The scheduler materializes one
        // buffer per slot, so only an individual slot (not their aggregate budget) is constrained
        // by `max_buffer_size`.
        let arena = ArenaPlanner::new(0)
            .with_alignment(binding_alignment)?
            .plan_with_node_steps(plan, &node_steps)?;
        if let Some(allocation) = arena.allocations.iter().find(|allocation| {
            allocation
                .size
                .checked_add(binding_alignment - 1)
                .map(|size| size & !(binding_alignment - 1))
                .is_none_or(|size| size > self.limits.max_buffer_size)
        }) {
            return Err(Error::ResourceLimit(format!(
                "plane {:?} needs a physical slot larger than the device buffer limit of {} bytes",
                allocation.plane, self.limits.max_buffer_size
            )));
        }
        let max_binding_size = self.limits.max_storage_buffer_binding_size;
        if let Some(allocation) = arena.allocations.iter().find(|allocation| {
            allocation
                .size
                .max(4)
                .checked_add(3)
                .map(|size| size & !3)
                .is_none_or(|size| size > max_binding_size)
        }) {
            return Err(Error::ResourceLimit(format!(
                "plane {:?} needs a {}-byte storage binding, exceeding the device limit of {max_binding_size} bytes",
                allocation.plane, allocation.size
            )));
        }

        let resident_budget =
            effective_budget(self.memory.max_resident_bytes, frame.max_resident_bytes);
        let scratch_budget =
            effective_budget(self.memory.max_scratch_bytes, frame.max_scratch_bytes);
        if arena.size_bytes > resident_budget || arena.peak_scratch_bytes > scratch_budget {
            return Err(Error::ResourceLimit(format!(
                "resident execution needs {} bytes ({} bytes of live intermediates), exceeding resident={resident_budget} or scratch={scratch_budget}; streaming execution is unavailable",
                arena.size_bytes, arena.peak_scratch_bytes
            )));
        }
        let resident_bytes = arena.size_bytes;
        let scratch_bytes = arena.peak_scratch_bytes;
        let groups_per_batch = frame.group_count;

        Ok(ExecutionPlan {
            memory_mode: MemoryMode::Resident,
            dispatches,
            arena,
            tile_extents,
            resident_bytes,
            scratch_bytes,
            groups_per_batch,
        })
    }

    fn validate_frame(&self, frame: &FrameSessionDesc) -> Result<()> {
        if frame.frame_extent.is_empty() {
            return Err(Error::InvalidPayload("frame extent is empty".into()));
        }
        if frame.group_extent.is_empty() {
            return Err(Error::InvalidPayload("group extent is empty".into()));
        }
        if frame.group_count == 0 {
            return Err(Error::InvalidPayload("frame has no groups".into()));
        }
        Ok(())
    }

    fn validate_nodes(&self, frame: &FrameSessionDesc, plan: &RenderPlan) -> Result<()> {
        let planes: BTreeMap<_, _> = plan.planes.iter().map(|plane| (plane.id, plane)).collect();
        for (index, node) in plan.nodes.iter().enumerate() {
            let kind = node.op.kind();
            if !self.supported_ops.contains(&kind) {
                return Err(Error::Unsupported(format!(
                    "node {index} '{}' uses unsupported operation {kind:?}",
                    node.name
                )));
            }
            validate_precision(index, node, &planes)?;
            validate_operation(index, node, plan)?;

            let mut resources = BTreeSet::new();
            if node
                .resources
                .iter()
                .any(|resource| !resources.insert(*resource))
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} '{}' lists a resource more than once",
                    node.name
                )));
            }
        }

        let has_f16_intermediate = plan.planes.iter().any(|plane| {
            plane.sample_type == SampleType::F16
                && matches!(
                    plane.role,
                    PlaneRole::Source | PlaneRole::Intermediate | PlaneRole::Parameter
                )
        });
        if has_f16_intermediate {
            match frame.precision {
                PrecisionPolicy::F32Only => {
                    return Err(Error::Unsupported(
                        "F16 intermediate storage violates F32Only precision policy".into(),
                    ));
                }
                PrecisionPolicy::AllowF16Storage | PrecisionPolicy::MatchDecoder
                    if !self.supports_f16 =>
                {
                    return Err(Error::Unsupported(
                        "the plan requires F16 storage but SHADER_F16 is unavailable".into(),
                    ));
                }
                PrecisionPolicy::AllowF16Storage | PrecisionPolicy::MatchDecoder => {}
            }
        }
        Ok(())
    }

    fn plan_dispatches(
        &self,
        plan: &RenderPlan,
        extents: &BTreeMap<PlaneId, Extent2d>,
    ) -> Result<Vec<PlannedDispatch>> {
        let consumers = consumer_counts(plan);
        let roles: BTreeMap<_, _> = plan
            .planes
            .iter()
            .map(|plane| (plane.id, plane.role))
            .collect();
        let mut dispatches = Vec::new();
        let mut index = 0;
        while index < plan.nodes.len() {
            let (length, kernel) = known_fusion(plan, index, &consumers, &roles)
                .unwrap_or((1, FusedKernel::Single(plan.nodes[index].op.kind())));
            let indices: Vec<_> = (index..index + length).collect();
            let nodes = &plan.nodes[index..index + length];
            let variant = kernel.variant_for(&self.kernel_policy)?;
            let workgroup_size = variant.workgroup_size();
            variant.validate_for(kernel.key(), &self.limits, 0)?;
            let extent = dispatch_extent(nodes, extents)?;
            let workgroups = (
                extent.width.div_ceil(workgroup_size.0),
                extent.height.div_ceil(workgroup_size.1),
                1,
            );
            if workgroups.0 > self.limits.max_compute_workgroups_per_dimension
                || workgroups.1 > self.limits.max_compute_workgroups_per_dimension
            {
                return Err(Error::ResourceLimit(format!(
                    "dispatch '{}' needs {}x{} workgroups, device limit is {} per dimension",
                    joined_label(nodes),
                    workgroups.0,
                    workgroups.1,
                    self.limits.max_compute_workgroups_per_dimension
                )));
            }
            let resources = nodes
                .iter()
                .flat_map(|node| node.resources.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            dispatches.push(PlannedDispatch {
                label: Arc::from(joined_label(nodes)),
                kernel,
                node_indices: indices,
                resources,
                precision: nodes[0].precision,
                variant,
                workgroup_size,
                workgroups,
            });
            index += length;
        }
        Ok(dispatches)
    }
}

fn portable_supported_ops() -> BTreeSet<RenderOpKind> {
    [
        RenderOpKind::Copy,
        RenderOpKind::ModularToF32,
        RenderOpKind::ChromaUpsample,
        RenderOpKind::Gaborish,
        RenderOpKind::Epf,
        RenderOpKind::Upsample,
        RenderOpKind::VarDct,
        RenderOpKind::XybToRgb,
        RenderOpKind::YcbcrToRgb,
        RenderOpKind::TransferFunction,
        RenderOpKind::Blend,
        RenderOpKind::PremultiplyAlpha,
        RenderOpKind::Convert,
        RenderOpKind::Extend,
        RenderOpKind::Save,
    ]
    .into_iter()
    .collect()
}

fn validate_precision(
    index: usize,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, &jxl_gpu_protocol::PlaneDesc>,
) -> Result<()> {
    match node.precision {
        PrecisionContract::Exact if !exact_capable(node, planes) => {
            Err(Error::Unsupported(format!(
                "node {index} '{}' requests exact precision for {:?}",
                node.name,
                node.op.kind()
            )))
        }
        PrecisionContract::Float {
            absolute,
            relative,
            rmse,
        } if [absolute, relative, rmse]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.0) =>
        {
            Err(Error::InvalidPayload(format!(
                "node {index} '{}' has an invalid floating precision contract",
                node.name
            )))
        }
        PrecisionContract::Perceptual { min_psnr, .. }
            if !min_psnr.is_finite() || min_psnr < 0.0 =>
        {
            Err(Error::InvalidPayload(format!(
                "node {index} '{}' has an invalid perceptual precision contract",
                node.name
            )))
        }
        _ => Ok(()),
    }
}

fn exact_capable(
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, &jxl_gpu_protocol::PlaneDesc>,
) -> bool {
    match node.op.kind() {
        RenderOpKind::Copy | RenderOpKind::Extend | RenderOpKind::Save => true,
        RenderOpKind::Convert => node
            .inputs
            .iter()
            .chain(&node.outputs)
            .filter_map(|plane| planes.get(plane))
            .all(|plane| {
                matches!(
                    plane.sample_type,
                    SampleType::I32 | SampleType::U16 | SampleType::U8
                )
            }),
        _ => false,
    }
}

fn validate_operation(index: usize, node: &RenderNode, plan: &RenderPlan) -> Result<()> {
    if node.outputs.is_empty() && !matches!(node.op, RenderOp::Save(_)) {
        return Err(Error::InvalidPayload(format!(
            "node {index} '{}' produces no planes",
            node.name
        )));
    }
    match &node.op {
        RenderOp::Copy => {
            let [input] = node.inputs.as_slice() else {
                return Err(Error::InvalidPayload(format!(
                    "node {index} Copy requires exactly one input"
                )));
            };
            let [output] = node.outputs.as_slice() else {
                return Err(Error::InvalidPayload(format!(
                    "node {index} Copy requires exactly one output"
                )));
            };
            let input = plan
                .planes
                .iter()
                .find(|plane| plane.id == *input)
                .ok_or_else(|| Error::InvalidPayload(format!("node {index} input is unknown")))?;
            let output = plan
                .planes
                .iter()
                .find(|plane| plane.id == *output)
                .ok_or_else(|| Error::InvalidPayload(format!("node {index} output is unknown")))?;
            if input.sample_type != output.sample_type || input.extent != output.extent {
                return Err(Error::InvalidPayload(format!(
                    "node {index} Copy input and output types or extents differ"
                )));
            }
            if !matches!(input.sample_type, SampleType::I32 | SampleType::F32) {
                return Err(Error::Unsupported(format!(
                    "node {index} resident-arena Copy requires I32 or F32 planes"
                )));
            }
            Ok(())
        }
        RenderOp::ModularToF32 { multiplier, bias }
            if !multiplier.is_finite() || !bias.is_finite() =>
        {
            Err(Error::InvalidPayload(format!(
                "node {index} has non-finite modular conversion parameters"
            )))
        }
        RenderOp::Gaborish(params) => {
            if !params.weight0.is_finite()
                || !params.weight1.is_finite()
                || !params.weight2.is_finite()
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} has non-finite Gaborish weights"
                )));
            }
            let ([input_id], [output_id]) = (node.inputs.as_slice(), node.outputs.as_slice())
            else {
                return Err(Error::InvalidPayload(format!(
                    "node {index} Gaborish requires exactly one input and one output"
                )));
            };
            let input = plan
                .planes
                .iter()
                .find(|plane| plane.id == *input_id)
                .ok_or(Error::MissingPlane(*input_id))?;
            let output = plan
                .planes
                .iter()
                .find(|plane| plane.id == *output_id)
                .ok_or(Error::MissingPlane(*output_id))?;
            if input.sample_type != SampleType::F32
                || output.sample_type != SampleType::F32
                || input.extent != output.extent
                || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
                || node.border != jxl_gpu_protocol::Border2d::symmetric(1, 1)
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} Gaborish requires equal F32 extents, identity scale, and a one-pixel border"
                )));
            }
            Ok(())
        }
        RenderOp::ChromaUpsample { axis } => validate_chroma_upsample(index, node, plan, *axis),
        RenderOp::Epf(params) => validate_epf(index, node, params, plan),
        RenderOp::Upsample(params) => {
            let factor = usize::from(params.factor);
            let expected = factor
                .checked_mul(factor)
                .and_then(|phases| phases.checked_mul(25))
                .ok_or(Error::BufferSizeOverflow)?;
            if !matches!(params.factor, 2 | 4 | 8) || params.weights.len() != expected {
                return Err(Error::InvalidPayload(format!(
                    "node {index} has factor {} with {} weights; expected 2/4/8 and {expected} weights",
                    params.factor,
                    params.weights.len()
                )));
            }
            if node.scale.x != params.factor || node.scale.y != params.factor {
                return Err(Error::InvalidPayload(format!(
                    "node {index} scale does not match its {}x upsampling factor",
                    params.factor
                )));
            }
            if node.border != jxl_gpu_protocol::Border2d::symmetric(2, 2) {
                return Err(Error::InvalidPayload(format!(
                    "node {index} {}x upsampling requires a two-sample symmetric border",
                    params.factor
                )));
            }
            let ([input_id], [output_id]) = (node.inputs.as_slice(), node.outputs.as_slice())
            else {
                return Err(Error::InvalidPayload(format!(
                    "node {index} upsampling requires exactly one input and one output"
                )));
            };
            let input = plan
                .planes
                .iter()
                .find(|plane| plane.id == *input_id)
                .ok_or(Error::MissingPlane(*input_id))?;
            let output = plan
                .planes
                .iter()
                .find(|plane| plane.id == *output_id)
                .ok_or(Error::MissingPlane(*output_id))?;
            let factor = u32::from(params.factor);
            if input.sample_type != SampleType::F32
                || output.sample_type != SampleType::F32
                || output.extent.width.div_ceil(factor) != input.extent.width
                || output.extent.height.div_ceil(factor) != input.extent.height
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} {}x upsampling requires F32 planes and a possibly odd-cropped extent, got {:?} -> {:?}",
                    params.factor, input.extent, output.extent
                )));
            }
            Ok(())
        }
        RenderOp::VarDct => {
            if !node.inputs.is_empty() || node.outputs.len() != 3 || node.resources.len() != 1 {
                return Err(Error::InvalidPayload(format!(
                    "node {index} has an invalid VarDCT plane or resource contract"
                )));
            }
            Ok(())
        }
        RenderOp::XybToRgb(params) => validate_xyb_to_rgb(index, node, params, plan),
        RenderOp::TransferFunction(params) => validate_transfer_function(index, node, params, plan),
        RenderOp::Blend(params) => validate_blend(index, node, params, plan),
        RenderOp::PremultiplyAlpha { alpha_plane } => {
            let alpha_count = node
                .inputs
                .iter()
                .filter(|&&input| input == *alpha_plane)
                .count();
            let colors = node
                .inputs
                .iter()
                .copied()
                .filter(|input| input != alpha_plane)
                .collect::<Vec<_>>();
            if alpha_count != 1
                || colors.is_empty()
                || !matches!(node.outputs.len(), count if count == colors.len() || count == node.inputs.len())
                || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
                || node.border != jxl_gpu_protocol::Border2d::default()
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} PremultiplyAlpha has an invalid dependency, arity, scale, or border contract"
                )));
            }
            let descriptor = |id: PlaneId| {
                plan.planes
                    .iter()
                    .find(|plane| plane.id == id)
                    .ok_or(Error::MissingPlane(id))
            };
            let alpha = descriptor(*alpha_plane)?;
            if alpha.sample_type != SampleType::F32 {
                return Err(Error::InvalidPayload(format!(
                    "node {index} PremultiplyAlpha requires an F32 alpha plane"
                )));
            }
            let pairs = if node.outputs.len() == colors.len() {
                colors
                    .iter()
                    .copied()
                    .zip(node.outputs.iter().copied())
                    .collect::<Vec<_>>()
            } else {
                node.inputs
                    .iter()
                    .copied()
                    .zip(node.outputs.iter().copied())
                    .collect::<Vec<_>>()
            };
            for (input_id, output_id) in pairs {
                let input = descriptor(input_id)?;
                let output = descriptor(output_id)?;
                if input.sample_type != SampleType::F32
                    || output.sample_type != SampleType::F32
                    || input.extent != alpha.extent
                    || output.extent != input.extent
                {
                    return Err(Error::InvalidPayload(format!(
                        "node {index} PremultiplyAlpha requires equal-extent F32 color, alpha, and output planes"
                    )));
                }
            }
            Ok(())
        }
        RenderOp::Save(save) => {
            let output = plan.outputs.iter().find(|output| output.id == save.output);
            if save.channels.is_empty()
                || save
                    .channels
                    .iter()
                    .any(|channel| !node.inputs.contains(channel))
                || output.is_some_and(|output| {
                    output.sample_type != save.sample_type
                        || output.layout != save.layout
                        || usize::from(output.channels) != save.channels.len()
                })
            {
                return Err(Error::InvalidPayload(format!(
                    "node {index} has a save contract inconsistent with its inputs or output"
                )));
            }
            Ok(())
        }
        RenderOp::Extend {
            image_extent,
            origin,
        } => validate_extend(index, node, *image_extent, *origin, plan),
        _ => Ok(()),
    }
}

fn validate_extend(
    index: usize,
    node: &RenderNode,
    image_extent: Extent2d,
    origin: (i32, i32),
    plan: &RenderPlan,
) -> Result<()> {
    if image_extent.is_empty()
        || !matches!(node.inputs.len(), 1 | 2)
        || node.outputs.len() != 1
        || !node.resources.is_empty()
        || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
        || node.border != jxl_gpu_protocol::Border2d::default()
        || node.inputs.contains(&node.outputs[0])
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} Extend has an invalid extent, arity, dependency, scale, or border contract"
        )));
    }
    let descriptor = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or(Error::MissingPlane(id))
    };
    let frame = descriptor(node.inputs[0])?;
    let output = descriptor(node.outputs[0])?;
    if !matches!(frame.sample_type, SampleType::I32 | SampleType::F32)
        || output.sample_type != frame.sample_type
        || output.extent != image_extent
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} Extend requires matching I32 or F32 frame/output planes and the declared image extent"
        )));
    }
    if let Some(reference_id) = node.inputs.get(1) {
        let reference = descriptor(*reference_id)?;
        if reference.sample_type != frame.sample_type || reference.extent != image_extent {
            return Err(Error::InvalidPayload(format!(
                "node {index} Extend reference must match the output type and full image extent"
            )));
        }
    }

    let coordinate_is_safe = |frame_size: u32, image_size: u32, offset: i32| {
        let frame_size = i64::from(frame_size);
        let image_size = i64::from(image_size);
        let offset = i64::from(offset);
        frame_size + image_size <= i64::from(i32::MAX)
            && offset >= -frame_size
            && offset <= image_size
    };
    if !coordinate_is_safe(frame.extent.width, image_extent.width, origin.0)
        || !coordinate_is_safe(frame.extent.height, image_extent.height, origin.1)
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} Extend origin or extent is outside the safe JPEG XL canvas coordinate range"
        )));
    }
    Ok(())
}

fn validate_blend(
    index: usize,
    node: &RenderNode,
    params: &BlendParams,
    plan: &RenderPlan,
) -> Result<()> {
    let valid_arity = match params.component {
        BlendComponent::Alpha => node.inputs.len() == 2,
        BlendComponent::Color { alpha_associated } => {
            matches!(node.inputs.len(), 2 | 4) && (!alpha_associated || node.inputs.len() == 4)
        }
    };
    if !valid_arity
        || node.outputs.len() != 1
        || !node.resources.is_empty()
        || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
        || node.border != jxl_gpu_protocol::Border2d::default()
        || node.inputs.contains(&node.outputs[0])
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} Blend has an invalid component, arity, dependency, scale, or border contract"
        )));
    }

    let descriptor = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or(Error::MissingPlane(id))
    };
    let first = descriptor(node.inputs[0])?;
    for id in node.inputs.iter().chain(&node.outputs) {
        let plane = descriptor(*id)?;
        if plane.sample_type != SampleType::F32 || plane.extent != first.extent {
            return Err(Error::InvalidPayload(format!(
                "node {index} Blend requires equal-extent F32 base, source, alpha, and output planes"
            )));
        }
    }
    Ok(())
}

fn validate_xyb_to_rgb(
    index: usize,
    node: &RenderNode,
    params: &jxl_gpu_protocol::XybParams,
    plan: &RenderPlan,
) -> Result<()> {
    if !params.intensity_target.is_finite()
        || params.intensity_target <= 0.0
        || params.opsin_bias.iter().any(|value| !value.is_finite())
        || params
            .inverse_opsin_matrix
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} XYB conversion has invalid opsin or intensity parameters"
        )));
    }
    if node.inputs.len() != 3
        || node.outputs.len() != 3
        || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
        || node.border != jxl_gpu_protocol::Border2d::default()
        || !node.resources.is_empty()
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} XYB conversion requires three inputs, three outputs, identity scale, no border, and no resources"
        )));
    }

    let descriptor = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or(Error::MissingPlane(id))
    };
    let first = descriptor(node.inputs[0])?;
    for id in node.inputs.iter().chain(&node.outputs) {
        let plane = descriptor(*id)?;
        if plane.sample_type != SampleType::F32 || plane.extent != first.extent {
            return Err(Error::InvalidPayload(format!(
                "node {index} XYB conversion requires six equal-extent F32 planes"
            )));
        }
    }
    Ok(())
}

fn validate_transfer_function(
    index: usize,
    node: &RenderNode,
    params: &jxl_gpu_protocol::TransferParams,
    plan: &RenderPlan,
) -> Result<()> {
    let basic_parameters_valid = params.intensity_target.is_finite()
        && params.intensity_target > 0.0
        && params.min_nits.is_finite()
        && params.min_nits >= 0.0
        && params.min_nits <= params.intensity_target
        && params.gamma.is_finite()
        && params.luminance_rgb.iter().all(|value| value.is_finite());
    let function_parameters_valid = match params.function {
        jxl_gpu_protocol::TransferFunction::Gamma => {
            (0.0..=1.0).contains(&params.gamma) && params.gamma > 0.0
        }
        jxl_gpu_protocol::TransferFunction::Hlg => {
            params.luminance_rgb.iter().all(|&value| value >= 0.0)
                && params.luminance_rgb.iter().sum::<f32>() > 0.0
        }
        _ => true,
    };
    if !basic_parameters_valid || !function_parameters_valid {
        return Err(Error::InvalidPayload(format!(
            "node {index} has invalid transfer-function parameters"
        )));
    }
    if node.inputs.len() != 3
        || node.outputs.len() != 3
        || node.scale != jxl_gpu_protocol::Scale2d::IDENTITY
        || node.border != jxl_gpu_protocol::Border2d::default()
        || !node.resources.is_empty()
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} transfer function requires three inputs, three outputs, identity scale, no border, and no resources"
        )));
    }
    let descriptor = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or(Error::MissingPlane(id))
    };
    let first = descriptor(node.inputs[0])?;
    for id in node.inputs.iter().chain(&node.outputs) {
        let plane = descriptor(*id)?;
        if plane.sample_type != SampleType::F32 || plane.extent != first.extent {
            return Err(Error::InvalidPayload(format!(
                "node {index} transfer function requires six equal-extent F32 planes"
            )));
        }
    }
    Ok(())
}

fn validate_chroma_upsample(
    index: usize,
    node: &RenderNode,
    plan: &RenderPlan,
    axis: ChromaAxis,
) -> Result<()> {
    let ([input_id], [output_id]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
        return Err(Error::InvalidPayload(format!(
            "node {index} chroma upsample requires exactly one input and one output"
        )));
    };
    let plane = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or_else(|| {
                Error::InvalidPayload(format!(
                    "node {index} chroma upsample names unknown plane {id:?}"
                ))
            })
    };
    let input = plane(*input_id)?;
    let output = plane(*output_id)?;
    if input.sample_type != SampleType::F32 || output.sample_type != SampleType::F32 {
        return Err(Error::InvalidPayload(format!(
            "node {index} chroma upsample requires F32 input and output"
        )));
    }
    let (scale, border, extent_matches) = match axis {
        ChromaAxis::Horizontal => (
            jxl_gpu_protocol::Scale2d::new(2, 1),
            jxl_gpu_protocol::Border2d::symmetric(1, 0),
            output.extent.height == input.extent.height
                && output.extent.width.div_ceil(2) == input.extent.width,
        ),
        ChromaAxis::Vertical => (
            jxl_gpu_protocol::Scale2d::new(1, 2),
            jxl_gpu_protocol::Border2d::symmetric(0, 1),
            output.extent.width == input.extent.width
                && output.extent.height.div_ceil(2) == input.extent.height,
        ),
    };
    if node.scale != scale || node.border != border || !extent_matches {
        return Err(Error::InvalidPayload(format!(
            "node {index} {axis:?} chroma upsample has scale {:?}, border {:?}, and extent {:?} -> {:?}; expected {scale:?}, {border:?}, and a possibly odd-cropped 2x extent",
            node.scale, node.border, input.extent, output.extent
        )));
    }
    Ok(())
}

fn validate_epf(
    index: usize,
    node: &RenderNode,
    params: &EpfParams,
    plan: &RenderPlan,
) -> Result<()> {
    if !params.sigma_scale.is_finite()
        || !params.border_sad_mul.is_finite()
        || params.channel_scale.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidPayload(format!(
            "node {index} has non-finite EPF parameters"
        )));
    }
    if node.inputs.len() != 3 || node.outputs.len() != 3 {
        return Err(Error::InvalidPayload(format!(
            "node {index} EPF requires exactly three inputs and three outputs"
        )));
    }
    if node.scale != jxl_gpu_protocol::Scale2d::IDENTITY {
        return Err(Error::InvalidPayload(format!(
            "node {index} EPF must use identity scale"
        )));
    }
    let expected_border = match params.pass {
        jxl_gpu_protocol::EpfPass::Pass0 => 3,
        jxl_gpu_protocol::EpfPass::Pass1 => 2,
        jxl_gpu_protocol::EpfPass::Pass2 => 1,
    };
    if node.border != jxl_gpu_protocol::Border2d::symmetric(expected_border, expected_border) {
        return Err(Error::InvalidPayload(format!(
            "node {index} {:?} requires a {expected_border}-pixel symmetric border",
            params.pass
        )));
    }
    let sigma_resource = params
        .sigma_resource
        .ok_or_else(|| Error::InvalidPayload(format!("node {index} EPF has no sigma resource")))?;
    if node.resources.as_slice() != [sigma_resource] {
        return Err(Error::InvalidPayload(format!(
            "node {index} EPF must declare only sigma resource {sigma_resource:?}"
        )));
    }

    let plane = |id: PlaneId| {
        plan.planes
            .iter()
            .find(|plane| plane.id == id)
            .ok_or_else(|| {
                Error::InvalidPayload(format!("node {index} EPF names unknown plane {id:?}"))
            })
    };
    let image_extent = plane(node.inputs[0])?.extent;
    for id in node.inputs.iter().chain(&node.outputs) {
        let desc = plane(*id)?;
        if desc.sample_type != SampleType::F32 || desc.extent != image_extent {
            return Err(Error::InvalidPayload(format!(
                "node {index} EPF plane {id:?} must be F32 with extent {image_extent:?}"
            )));
        }
    }

    if let Some(sigma_id) = params.sigma_plane {
        let sigma = plane(sigma_id)?;
        let required_width = image_extent.width.div_ceil(8);
        let required_height = image_extent.height.div_ceil(8);
        if sigma.role != PlaneRole::Parameter
            || sigma.sample_type != SampleType::F32
            || sigma.extent.width < required_width
            || sigma.extent.height < required_height
        {
            return Err(Error::InvalidPayload(format!(
                "node {index} EPF sigma plane {sigma_id:?} must be an F32 parameter plane covering at least {required_width}x{required_height} blocks"
            )));
        }
    }
    Ok(())
}

fn effective_budget(configured: u64, requested: u64) -> u64 {
    match (configured, requested) {
        (0, 0) => u64::MAX,
        (0, requested) => requested,
        (configured, 0) => configured,
        (configured, requested) => configured.min(requested),
    }
}

#[cfg(test)]
fn parameter_bytes(plan: &RenderPlan) -> Result<u64> {
    plan.planes
        .iter()
        .filter(|plane| plane.role == PlaneRole::Parameter)
        .try_fold(0u64, |total, plane| {
            let samples = plane.minimum_len().ok_or(Error::BufferSizeOverflow)?;
            let bytes = samples
                .checked_mul(plane.sample_type.bytes_per_sample())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or(Error::BufferSizeOverflow)?;
            total.checked_add(bytes).ok_or(Error::BufferSizeOverflow)
        })
}

#[cfg(test)]
fn streaming_extents(
    frame: &FrameSessionDesc,
    plan: &RenderPlan,
) -> Result<BTreeMap<PlaneId, Extent2d>> {
    let plane_map: BTreeMap<_, _> = plan.planes.iter().map(|plane| (plane.id, plane)).collect();
    let mut extents = BTreeMap::new();
    for plane in plan
        .planes
        .iter()
        .filter(|plane| matches!(plane.role, PlaneRole::Source | PlaneRole::Parameter))
    {
        let extent = if plane.role == PlaneRole::Parameter {
            plane.extent
        } else {
            proportional_extent(plane.extent, frame.group_extent, frame.frame_extent)?
        };
        extents.insert(plane.id, extent);
    }

    for node in &plan.nodes {
        let input_extent = if matches!(node.op, RenderOp::VarDct) {
            let output = node.outputs.first().ok_or_else(|| {
                Error::InvalidPayload(format!("node '{}' has no spatial output", node.name))
            })?;
            proportional_extent(
                plane_map[output].extent,
                frame.group_extent,
                frame.frame_extent,
            )?
        } else {
            node.inputs
                .iter()
                .filter(|plane| {
                    plane_map
                        .get(plane)
                        .is_some_and(|plane| plane.role != PlaneRole::Parameter)
                })
                .filter_map(|plane| extents.get(plane).copied())
                .reduce(max_extent)
                .or_else(|| {
                    node.inputs
                        .iter()
                        .filter_map(|plane| extents.get(plane).copied())
                        .reduce(max_extent)
                })
                .ok_or_else(|| {
                    Error::InvalidPayload(format!("node '{}' has no spatial input", node.name))
                })?
        };
        let width = u64::from(input_extent.width)
            .checked_add(u64::from(node.border.left) + u64::from(node.border.right))
            .and_then(|width| width.checked_mul(u64::from(node.scale.x)))
            .ok_or(Error::BufferSizeOverflow)?;
        let height = u64::from(input_extent.height)
            .checked_add(u64::from(node.border.top) + u64::from(node.border.bottom))
            .and_then(|height| height.checked_mul(u64::from(node.scale.y)))
            .ok_or(Error::BufferSizeOverflow)?;
        // Reject invalid scale/halo arithmetic before clipping. A saturating estimate could hide
        // a plan that no legal dispatch can address.
        let propagated = Extent2d::new(
            u32::try_from(width).map_err(|_| Error::BufferSizeOverflow)?,
            u32::try_from(height).map_err(|_| Error::BufferSizeOverflow)?,
        );
        for output in &node.outputs {
            let full = plane_map[output].extent;
            let proportional = proportional_extent(full, frame.group_extent, frame.frame_extent)?;
            let candidate = Extent2d::new(
                propagated.width.max(proportional.width).min(full.width),
                propagated.height.max(proportional.height).min(full.height),
            );
            extents
                .entry(*output)
                .and_modify(|current| *current = max_extent(*current, candidate))
                .or_insert(candidate);
        }
    }

    for plane in &plan.planes {
        if let std::collections::btree_map::Entry::Vacant(entry) = extents.entry(plane.id) {
            entry.insert(proportional_extent(
                plane.extent,
                frame.group_extent,
                frame.frame_extent,
            )?);
        }
    }
    Ok(extents)
}

#[cfg(test)]
fn proportional_extent(full: Extent2d, tile: Extent2d, frame: Extent2d) -> Result<Extent2d> {
    Ok(Extent2d::new(
        proportional_dimension(full.width, tile.width, frame.width)?,
        proportional_dimension(full.height, tile.height, frame.height)?,
    ))
}

#[cfg(test)]
fn proportional_dimension(full: u32, tile: u32, frame: u32) -> Result<u32> {
    let numerator = u64::from(full)
        .checked_mul(u64::from(tile))
        .ok_or(Error::BufferSizeOverflow)?;
    let value = numerator
        .div_ceil(u64::from(frame))
        .max(1)
        .min(u64::from(full));
    u32::try_from(value).map_err(|_| Error::BufferSizeOverflow)
}

#[cfg(test)]
fn streaming_sizes(
    plan: &RenderPlan,
    extents: &BTreeMap<PlaneId, Extent2d>,
) -> Result<BTreeMap<PlaneId, u64>> {
    plan.planes
        .iter()
        .map(|plane| {
            let extent = extents[&plane.id];
            let stride = if plane.role == PlaneRole::Parameter && plane.stride != 0 {
                plane.stride
            } else {
                extent.width
            };
            let bytes = u64::from(stride)
                .checked_mul(u64::from(extent.height))
                .and_then(|samples| {
                    samples.checked_mul(u64::try_from(plane.sample_type.bytes_per_sample()).ok()?)
                })
                .ok_or(Error::BufferSizeOverflow)?;
            Ok((plane.id, bytes))
        })
        .collect()
}

fn max_extent(left: Extent2d, right: Extent2d) -> Extent2d {
    Extent2d::new(left.width.max(right.width), left.height.max(right.height))
}

fn consumer_counts(plan: &RenderPlan) -> BTreeMap<PlaneId, usize> {
    let mut counts = BTreeMap::new();
    for input in plan.nodes.iter().flat_map(|node| &node.inputs) {
        *counts.entry(*input).or_default() += 1;
    }
    counts
}

fn known_fusion(
    plan: &RenderPlan,
    start: usize,
    consumers: &BTreeMap<PlaneId, usize>,
    roles: &BTreeMap<PlaneId, PlaneRole>,
) -> Option<(usize, FusedKernel)> {
    let remaining = &plan.nodes[start..];
    if remaining.is_empty() {
        return None;
    }
    if remaining.len() >= 2
        && matches!(
            &remaining[0].op,
            RenderOp::ChromaUpsample {
                axis: ChromaAxis::Horizontal
            }
        )
        && matches!(
            &remaining[1].op,
            RenderOp::ChromaUpsample {
                axis: ChromaAxis::Vertical
            }
        )
        && fusible_chain(&remaining[..2], consumers, roles)
    {
        return Some((2, FusedKernel::Chroma2d));
    }
    if gaborish_rgb_fusion(plan, remaining) {
        return Some((3, FusedKernel::GaborishRgb));
    }
    None
}

fn gaborish_rgb_fusion(plan: &RenderPlan, remaining: &[RenderNode]) -> bool {
    if remaining.len() < 3 || remaining[..3].iter().any(|node| !node.resources.is_empty()) {
        return false;
    }
    let precision = remaining[0].precision;
    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut extent = None;
    for (channel, node) in remaining[..3].iter().enumerate() {
        let RenderOp::Gaborish(params) = &node.op else {
            return false;
        };
        if usize::from(params.channel) != channel || node.precision != precision {
            return false;
        }
        let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
            return false;
        };
        if !inputs.insert(*input) || !outputs.insert(*output) {
            return false;
        }
        let Some(input_desc) = plan.planes.iter().find(|plane| plane.id == *input) else {
            return false;
        };
        let Some(output_desc) = plan.planes.iter().find(|plane| plane.id == *output) else {
            return false;
        };
        if input_desc.extent != output_desc.extent
            || extent.is_some_and(|extent| extent != input_desc.extent)
        {
            return false;
        }
        extent = Some(input_desc.extent);
    }
    inputs.is_disjoint(&outputs)
}

fn fusible_chain(
    nodes: &[RenderNode],
    consumers: &BTreeMap<PlaneId, usize>,
    roles: &BTreeMap<PlaneId, PlaneRole>,
) -> bool {
    nodes
        .iter()
        .all(|node| node.precision == nodes[0].precision)
        && nodes.windows(2).all(|pair| {
            let outputs = &pair[0].outputs;
            !outputs.is_empty()
                && outputs.iter().all(|output| {
                    roles.get(output) == Some(&PlaneRole::Intermediate)
                        && consumers.get(output) == Some(&1)
                        && pair[1].inputs.contains(output)
                })
        })
}

fn dispatch_extent(
    nodes: &[RenderNode],
    extents: &BTreeMap<PlaneId, Extent2d>,
) -> Result<Extent2d> {
    let last = nodes.last().expect("a dispatch always contains a node");
    last.outputs
        .iter()
        .chain(&last.inputs)
        .filter_map(|plane| extents.get(plane).copied())
        .reduce(max_extent)
        .ok_or_else(|| {
            Error::InvalidPayload(format!("node '{}' has no dispatch extent", last.name))
        })
}

fn joined_label(nodes: &[RenderNode]) -> String {
    nodes
        .iter()
        .map(|node| node.name.as_ref())
        .collect::<Vec<_>>()
        .join("+")
}

#[cfg(test)]
mod tests {
    use jxl_gpu_protocol::{
        Border2d, ChromaAxis, OutputDesc, OutputId, OutputLayout, PlaneDesc, PrecisionContract,
        RenderOp, SaveParams, Scale2d, UpsampleParams,
    };

    use super::*;

    fn tuned_policy(kernel: &str, variant: KernelVariant) -> KernelPolicy {
        let mut profile = crate::AutotuneProfile::new(crate::AdapterFingerprint {
            name: "planner test".into(),
            vendor: 0,
            device: 0,
            device_type: "Cpu".into(),
            backend: "Empty".into(),
            driver: String::new(),
            driver_info: String::new(),
        });
        profile.record(crate::TunedKernel::from_samples(kernel, variant, &[1, 2, 3]).unwrap());
        KernelPolicy::Profile(profile)
    }

    fn memory() -> WgpuMemoryPolicy {
        WgpuMemoryPolicy {
            max_resident_bytes: 64 * 1024 * 1024,
            max_scratch_bytes: 16 * 1024 * 1024,
            max_transient_bytes: 16 * 1024 * 1024,
            max_in_flight_transient_bytes: 16 * 1024 * 1024,
            max_cached_buffer_bytes: 16 * 1024 * 1024,
            prefer_streaming: false,
        }
    }

    fn frame(memory_mode: MemoryMode) -> FrameSessionDesc {
        FrameSessionDesc {
            frame_extent: Extent2d::new(1024, 1024),
            group_extent: Extent2d::new(64, 64),
            group_count: 256,
            precision: PrecisionPolicy::F32Only,
            memory_mode,
            max_resident_bytes: 0,
            max_scratch_bytes: 0,
        }
    }

    fn plane(id: u32, role: PlaneRole, sample_type: SampleType) -> PlaneDesc {
        PlaneDesc {
            id: PlaneId(id),
            extent: Extent2d::new(1024, 1024),
            stride: 1024,
            sample_type,
            role,
        }
    }

    fn node(
        name: &'static str,
        op: RenderOp,
        inputs: &[u32],
        outputs: &[u32],
        precision: PrecisionContract,
    ) -> RenderNode {
        RenderNode {
            name: Arc::from(name),
            op,
            inputs: inputs.iter().copied().map(PlaneId).collect(),
            outputs: outputs.iter().copied().map(PlaneId).collect(),
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::default(),
            precision,
        }
    }

    fn copy_plan() -> RenderPlan {
        RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source, SampleType::F32),
                plane(1, PlaneRole::Output, SampleType::F32),
            ],
            nodes: vec![node(
                "copy",
                RenderOp::Copy,
                &[0],
                &[1],
                PrecisionContract::Exact,
            )],
            outputs: Vec::new(),
        }
    }

    #[test]
    fn profile_selects_non_default_tier_a_variant() {
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .with_kernel_policy(tuned_policy("copy", KernelVariant::Tile8x8))
            .plan(&frame(MemoryMode::Resident), &copy_plan())
            .unwrap();
        assert_eq!(execution.dispatches[0].variant, KernelVariant::Tile8x8);
        assert_eq!(execution.dispatches[0].workgroup_size, (8, 8));
    }

    #[test]
    fn chooses_resident_when_it_fits() {
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Auto), &copy_plan())
            .unwrap();
        assert_eq!(execution.memory_mode, MemoryMode::Resident);
        assert_eq!(execution.groups_per_batch, 256);
        assert_eq!(execution.dispatches[0].node_indices, [0]);
    }

    #[test]
    fn aggregate_resident_budget_may_exceed_the_per_buffer_device_limit() {
        let limits = wgpu::Limits {
            max_buffer_size: 5 * 1024 * 1024,
            ..wgpu::Limits::default()
        };
        let execution = Planner::new(limits, memory())
            .plan(&frame(MemoryMode::Resident), &copy_plan())
            .unwrap();
        assert!(execution.arena.size_bytes > 5 * 1024 * 1024);
        assert!(
            execution
                .arena
                .allocations
                .iter()
                .all(|allocation| allocation.size <= 5 * 1024 * 1024)
        );
    }

    #[test]
    fn rejects_an_individual_resident_slot_over_the_device_buffer_limit() {
        let limits = wgpu::Limits {
            max_buffer_size: 3 * 1024 * 1024,
            ..wgpu::Limits::default()
        };
        assert!(matches!(
            Planner::new(limits, memory()).plan(&frame(MemoryMode::Resident), &copy_plan()),
            Err(Error::ResourceLimit(message)) if message.contains("physical slot")
        ));
    }

    #[test]
    fn auto_rejects_budget_that_only_unimplemented_streaming_would_fit() {
        let mut policy = memory();
        policy.max_resident_bytes = 1024 * 1024;
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), policy)
                .plan(&frame(MemoryMode::Auto), &copy_plan()),
            Err(Error::ResourceLimit(message)) if message.contains("streaming execution is unavailable")
        ));
    }

    #[test]
    fn explicit_streaming_is_typed_unsupported_even_when_it_would_fit() {
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), memory())
                .plan(&frame(MemoryMode::Streaming), &copy_plan()),
            Err(Error::Unsupported(message)) if message.contains("streaming execution is not implemented")
        ));
    }

    #[test]
    fn explicit_resident_mode_reports_budget_failure() {
        let mut policy = memory();
        policy.max_resident_bytes = 1024;
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), policy)
                .plan(&frame(MemoryMode::Resident), &copy_plan()),
            Err(Error::ResourceLimit(_))
        ));
    }

    #[test]
    fn rejects_unsupported_operation() {
        let mut plan = copy_plan();
        plan.nodes[0].op = RenderOp::AddNoise { seed0: 1, seed1: 2 };
        plan.nodes[0].precision = PrecisionContract::default();
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), memory()).plan(&frame(MemoryMode::Auto), &plan),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn premultiply_requires_explicit_equal_f32_alpha_dependency() {
        let valid = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source, SampleType::F32),
                plane(1, PlaneRole::Source, SampleType::F32),
                plane(2, PlaneRole::Output, SampleType::F32),
            ],
            nodes: vec![node(
                "premultiply",
                RenderOp::PremultiplyAlpha {
                    alpha_plane: PlaneId(1),
                },
                &[0, 1],
                &[2],
                PrecisionContract::default(),
            )],
            outputs: Vec::new(),
        };
        Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Resident), &valid)
            .expect("explicit F32 alpha dependency is plannable");

        let mut hidden = valid.clone();
        hidden.nodes[0].inputs.pop();
        assert!(
            Planner::new(wgpu::Limits::default(), memory())
                .plan(&frame(MemoryMode::Resident), &hidden)
                .is_err()
        );

        let mut wrong_type = valid;
        wrong_type.planes[1].sample_type = SampleType::I32;
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), memory())
                .plan(&frame(MemoryMode::Resident), &wrong_type),
            Err(Error::InvalidPayload(message)) if message.contains("F32 alpha")
        ));
    }

    #[test]
    fn rejects_exact_contract_for_float_filter() {
        let mut plan = copy_plan();
        plan.nodes[0].op = RenderOp::Gaborish(jxl_gpu_protocol::GaborishParams {
            channel: 0,
            weight0: 1.0,
            weight1: 0.1,
            weight2: 0.2,
        });
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), memory()).plan(&frame(MemoryMode::Auto), &plan),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn accepts_odd_cropped_upsample_and_chroma_extents() {
        let odd_source = |id, role| PlaneDesc {
            id: PlaneId(id),
            extent: Extent2d::new(129, 129),
            stride: 129,
            sample_type: SampleType::F32,
            role,
        };
        let odd_output = |id, role| PlaneDesc {
            id: PlaneId(id),
            extent: Extent2d::new(257, 257),
            stride: 257,
            sample_type: SampleType::F32,
            role,
        };
        let mut weights = vec![0.0; 4 * 25];
        weights[12] = 1.0;
        let mut upsample = node(
            "odd upsample",
            RenderOp::Upsample(UpsampleParams {
                factor: 2,
                weights: weights.into(),
            }),
            &[0],
            &[1],
            PrecisionContract::default(),
        );
        upsample.scale = Scale2d::new(2, 2);
        upsample.border = Border2d::symmetric(2, 2);
        let upsample_plan = RenderPlan {
            planes: vec![
                odd_source(0, PlaneRole::Source),
                odd_output(1, PlaneRole::Intermediate),
            ],
            nodes: vec![upsample],
            outputs: Vec::new(),
        };
        Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Resident), &upsample_plan)
            .expect("odd-cropped 2x upsample must be plannable");

        let mut chroma = node(
            "odd horizontal chroma",
            RenderOp::ChromaUpsample {
                axis: ChromaAxis::Horizontal,
            },
            &[2],
            &[3],
            PrecisionContract::default(),
        );
        chroma.scale = Scale2d::new(2, 1);
        chroma.border = Border2d::symmetric(1, 0);
        let chroma_plan = RenderPlan {
            planes: vec![
                PlaneDesc {
                    id: PlaneId(2),
                    extent: Extent2d::new(129, 3),
                    stride: 129,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Source,
                },
                PlaneDesc {
                    id: PlaneId(3),
                    extent: Extent2d::new(257, 3),
                    stride: 257,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                },
            ],
            nodes: vec![chroma],
            outputs: Vec::new(),
        };
        Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Resident), &chroma_plan)
            .expect("odd-cropped horizontal chroma upsample must be plannable");
    }

    #[test]
    fn plans_implemented_horizontal_vertical_chroma_fusion() {
        let desc = |id, extent, role| PlaneDesc {
            id: PlaneId(id),
            extent,
            stride: extent.width,
            sample_type: SampleType::F32,
            role,
        };
        let mut horizontal = node(
            "horizontal chroma",
            RenderOp::ChromaUpsample {
                axis: ChromaAxis::Horizontal,
            },
            &[0],
            &[1],
            PrecisionContract::default(),
        );
        horizontal.scale = Scale2d::new(2, 1);
        horizontal.border = Border2d::symmetric(1, 0);
        let mut vertical = node(
            "vertical chroma",
            RenderOp::ChromaUpsample {
                axis: ChromaAxis::Vertical,
            },
            &[1],
            &[2],
            PrecisionContract::default(),
        );
        vertical.scale = Scale2d::new(1, 2);
        vertical.border = Border2d::symmetric(0, 1);
        let plan = RenderPlan {
            planes: vec![
                desc(0, Extent2d::new(5, 4), PlaneRole::Source),
                desc(1, Extent2d::new(9, 4), PlaneRole::Intermediate),
                desc(2, Extent2d::new(9, 7), PlaneRole::Intermediate),
            ],
            nodes: vec![horizontal, vertical],
            outputs: Vec::new(),
        };
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Resident), &plan)
            .unwrap();
        assert_eq!(execution.dispatches.len(), 1);
        assert_eq!(execution.dispatches[0].node_indices, [0, 1]);
        assert_eq!(execution.dispatches[0].kernel, FusedKernel::Chroma2d);
        assert_eq!(
            execution.dispatches[0].workgroup_size,
            KernelVariant::Tile16x16.workgroup_size()
        );
    }

    #[test]
    fn plans_implemented_three_channel_gaborish_fusion() {
        let mut nodes = Vec::new();
        for channel in 0..3 {
            let mut gaborish = node(
                "gaborish",
                RenderOp::Gaborish(jxl_gpu_protocol::GaborishParams {
                    channel,
                    weight0: 0.5,
                    weight1: 0.1,
                    weight2: 0.025,
                }),
                &[u32::from(channel)],
                &[u32::from(channel) + 3],
                PrecisionContract::default(),
            );
            gaborish.border = Border2d::symmetric(1, 1);
            nodes.push(gaborish);
        }
        let plan = RenderPlan {
            planes: (0..6)
                .map(|id| {
                    plane(
                        id,
                        if id < 3 {
                            PlaneRole::Source
                        } else {
                            PlaneRole::Intermediate
                        },
                        SampleType::F32,
                    )
                })
                .collect(),
            nodes,
            outputs: Vec::new(),
        };
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Resident), &plan)
            .unwrap();
        assert_eq!(execution.dispatches.len(), 1);
        assert_eq!(execution.dispatches[0].node_indices, [0, 1, 2]);
        assert_eq!(execution.dispatches[0].kernel, FusedKernel::GaborishRgb);
    }

    #[test]
    fn does_not_advertise_unimplemented_convert_save_fusion() {
        let plan = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source, SampleType::F32),
                plane(1, PlaneRole::Intermediate, SampleType::F32),
            ],
            nodes: vec![
                node(
                    "convert",
                    RenderOp::Convert {
                        output_type: SampleType::F32,
                    },
                    &[0],
                    &[1],
                    PrecisionContract::default(),
                ),
                node(
                    "save",
                    RenderOp::Save(SaveParams {
                        output: OutputId(0),
                        sample_type: SampleType::F32,
                        channels: vec![PlaneId(1)],
                        layout: OutputLayout::Planar,
                        orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                    }),
                    &[1],
                    &[],
                    PrecisionContract::default(),
                ),
            ],
            outputs: vec![OutputDesc {
                id: OutputId(0),
                extent: Extent2d::new(1024, 1024),
                sample_type: SampleType::F32,
                channels: 1,
                layout: OutputLayout::Planar,
                color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
            }],
        };
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .plan(&frame(MemoryMode::Auto), &plan)
            .unwrap();
        assert_eq!(execution.dispatches.len(), 2);
        assert_eq!(execution.dispatches[0].node_indices, [0]);
        assert_eq!(execution.dispatches[1].node_indices, [1]);
        assert_eq!(
            execution.dispatches[0].kernel,
            FusedKernel::Single(RenderOpKind::Convert)
        );
        assert_eq!(
            execution.dispatches[1].kernel,
            FusedKernel::Single(RenderOpKind::Save)
        );
    }

    #[test]
    fn does_not_advertise_unimplemented_restoration_fusion() {
        let mut epf = node(
            "epf1",
            RenderOp::Epf(EpfParams {
                pass: jxl_gpu_protocol::EpfPass::Pass1,
                sigma_scale: 1.0,
                border_sad_mul: 1.0,
                channel_scale: [1.0; 3],
                sigma_resource: Some(jxl_gpu_protocol::ResourceId(0)),
                sigma_plane: None,
            }),
            &[3, 1, 2],
            &[4, 5, 6],
            PrecisionContract::default(),
        );
        epf.resources = vec![jxl_gpu_protocol::ResourceId(0)];
        epf.border = Border2d::symmetric(2, 2);
        let mut gaborish = node(
            "gaborish",
            RenderOp::Gaborish(jxl_gpu_protocol::GaborishParams {
                channel: 0,
                weight0: 1.0,
                weight1: 0.1,
                weight2: 0.2,
            }),
            &[0],
            &[3],
            PrecisionContract::default(),
        );
        gaborish.border = Border2d::symmetric(1, 1);
        let plan = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source, SampleType::F32),
                plane(1, PlaneRole::Source, SampleType::F32),
                plane(2, PlaneRole::Source, SampleType::F32),
                plane(3, PlaneRole::Intermediate, SampleType::F32),
                plane(4, PlaneRole::Intermediate, SampleType::F32),
                plane(5, PlaneRole::Intermediate, SampleType::F32),
                plane(6, PlaneRole::Output, SampleType::F32),
            ],
            nodes: vec![gaborish, epf],
            outputs: Vec::new(),
        };
        let execution = Planner::new(wgpu::Limits::default(), memory())
            .with_supported_ops([RenderOpKind::Gaborish, RenderOpKind::Epf])
            .plan(&frame(MemoryMode::Auto), &plan)
            .unwrap();
        assert_eq!(execution.dispatches.len(), 2);
        assert_eq!(execution.dispatches[0].node_indices, [0]);
        assert_eq!(execution.dispatches[1].node_indices, [1]);
        assert_eq!(
            execution.dispatches[0].kernel,
            FusedKernel::Single(RenderOpKind::Gaborish)
        );
    }

    #[test]
    fn rejects_invalid_upsample_weights() {
        let mut plan = copy_plan();
        plan.nodes[0] = node(
            "upsample",
            RenderOp::Upsample(UpsampleParams {
                factor: 2,
                weights: vec![0.0; 99].into(),
            }),
            &[0],
            &[1],
            PrecisionContract::default(),
        );
        plan.nodes[0].scale = Scale2d::new(2, 2);
        assert!(matches!(
            Planner::new(wgpu::Limits::default(), memory()).plan(&frame(MemoryMode::Auto), &plan),
            Err(Error::InvalidPayload(_))
        ));
    }

    #[test]
    fn rejects_scale_and_halo_overflow_in_streaming_estimate() {
        let mut plan = copy_plan();
        for plane in &mut plan.planes {
            plane.extent = Extent2d::new(u32::MAX, u32::MAX);
            plane.stride = u32::MAX;
        }
        plan.nodes[0].scale = jxl_gpu_protocol::Scale2d::new(u8::MAX, u8::MAX);
        plan.nodes[0].border = Border2d::symmetric(u16::MAX, u16::MAX);
        let frame = FrameSessionDesc {
            frame_extent: Extent2d::new(u32::MAX, u32::MAX),
            group_extent: Extent2d::new(u32::MAX, u32::MAX),
            group_count: 1,
            precision: PrecisionPolicy::F32Only,
            memory_mode: MemoryMode::Streaming,
            max_resident_bytes: 0,
            max_scratch_bytes: 0,
        };
        assert!(matches!(
            streaming_extents(&frame, &plan),
            Err(Error::BufferSizeOverflow)
        ));
    }

    #[test]
    fn streaming_estimate_remains_checked_for_future_tiled_execution() {
        let plan = copy_plan();
        let frame = frame(MemoryMode::Auto);
        let extents = streaming_extents(&frame, &plan).unwrap();
        let sizes = streaming_sizes(&plan, &extents).unwrap();
        assert!(sizes.values().all(|size| *size > 0));
        assert_eq!(parameter_bytes(&plan).unwrap(), 0);
    }
}
