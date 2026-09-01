// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;

use jxl_gpu_protocol::{BlendComponent, BlendMode, BlendParams, PlaneId, RenderNode, SampleType};

use crate::upload::UploadedPlane;
use crate::{Error, Result};

use super::super::pipeline::{PipelineFactory, create_pipeline, create_uniform, record_dispatch};
use super::super::{
    BlendUniform, PremultiplyUniform, plane, require_f32_equal_extent, stride, unary_planes,
};
use super::filters::encode_modular_to_f32;
use super::io::{encode_copy, encode_copy_ids};
pub(in crate::scheduler) fn encode_blend(
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
        wgpu::include_wgsl!("../../../shaders/blend.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_premultiply(
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
        wgpu::include_wgsl!("../../../shaders/premultiply_alpha.wgsl"),
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
            factory.variant,
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

pub(in crate::scheduler) fn encode_convert(
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
