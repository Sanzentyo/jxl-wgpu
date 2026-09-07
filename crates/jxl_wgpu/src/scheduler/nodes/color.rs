// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;

use jxl_gpu_protocol::{
    PlaneId, RenderNode, SampleType, TransferFunction as SourceTransferFunction,
};

use crate::upload::UploadedPlane;
use crate::{Error, Result};

use super::super::pipeline::{PipelineFactory, create_pipeline, create_uniform, record_dispatch};
use super::super::{TransferUniform, XybUniform, YcbcrUniform, plane, stride};
pub(in crate::scheduler) fn encode_ycbcr(
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
        wgpu::include_wgsl!("../../../shaders/ycbcr_to_rgb.wgsl"),
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
            factory.variant,
        );
    }
    Ok(())
}

pub(in crate::scheduler) fn encode_xyb_to_rgb(
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
        wgpu::include_wgsl!("../../../shaders/xyb_to_rgb.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_transfer_function(
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
        wgpu::include_wgsl!("../../../shaders/transfer_function.wgsl"),
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
        factory.variant,
    );
    Ok(())
}
