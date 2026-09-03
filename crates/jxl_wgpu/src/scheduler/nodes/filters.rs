// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::{BTreeMap, BTreeSet};

use jxl_gpu_protocol::{
    ChromaAxis, EpfParams, EpfPass, PlaneId, RenderNode, RenderOp, ResourceData, ResourceId,
    ResourceUpdate, SampleType,
};
use wgpu::util::DeviceExt;

use crate::upload::UploadedPlane;
use crate::{Error, Result};

use super::super::pipeline::{
    PipelineFactory, create_pipeline, create_pipeline_entry, create_uniform, record_dispatch,
};
use super::super::{
    Chroma2dUniform, ChromaUpsampleUniform, EpfUniform, GaborishRgbUniform, GaborishUniform,
    ModularParams, UpsampleUniform, plane, require_f32_equal_extent, stride, unary_planes,
};
pub(in crate::scheduler) fn encode_modular_to_f32(
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
        wgpu::include_wgsl!("../../../shaders/modular_to_f32.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_chroma_upsample(
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
        wgpu::include_wgsl!("../../../shaders/chroma_upsample.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_chroma_2d(
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
        wgpu::include_wgsl!("../../../shaders/chroma_2d.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_gaborish(
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
        wgpu::include_wgsl!("../../../shaders/gaborish.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_gaborish_rgb(
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
        wgpu::include_wgsl!("../../../shaders/gaborish_rgb.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_epf(
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
        wgpu::include_wgsl!("../../../shaders/epf.wgsl"),
        entry_point,
        0x4550_4600 | pass_key,
        factory.variant,
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_upsample(
    factory: &PipelineFactory<'_>,
    encoder: &mut wgpu::CommandEncoder,
    node: &RenderNode,
    planes: &BTreeMap<PlaneId, UploadedPlane>,
    upsample: &jxl_gpu_protocol::UpsampleParams,
    weights: &wgpu::Buffer,
) -> Result<()> {
    let device = factory.device;
    let (input, output) = unary_planes(node, planes)?;
    if input.desc.sample_type != SampleType::F32 || output.desc.sample_type != SampleType::F32 {
        return Err(Error::Unsupported(
            "Upsample requires F32 input and output".into(),
        ));
    }
    let factor = upsample.factor.as_u32();
    if output.desc.extent.width.div_ceil(factor) != input.desc.extent.width
        || output.desc.extent.height.div_ceil(factor) != input.desc.extent.height
    {
        return Err(Error::InvalidPayload(format!(
            "{}x Upsample extent mismatch in '{}': {:?} -> {:?}; expected a possibly odd-cropped extent",
            upsample.factor.as_u8(), node.name, input.desc.extent, output.desc.extent
        )));
    }
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
        wgpu::include_wgsl!("../../../shaders/upsample.wgsl"),
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
        factory.variant,
    );
    Ok(())
}
