// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;

use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, TransferFunction as ImageTransferFunction,
};
use jxl_gpu_protocol::{
    PlaneId, RenderNode, RgbColorEncoding, RgbPrimaries, SampleType,
    TransferFunction as SourceTransferFunction,
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
#[derive(Clone, Copy, Debug)]
pub(in crate::scheduler) struct ImageColorTransform {
    pub(in crate::scheduler) source_transfer: u32,
    pub(in crate::scheduler) target_transfer: u32,
    pub(in crate::scheduler) primaries: [[f32; 4]; 3],
}

pub(in crate::scheduler) fn image_color_transform(
    source: RgbColorEncoding,
    target: &PixelFormat,
) -> Result<ImageColorTransform> {
    let source_transfer = match source.transfer {
        SourceTransferFunction::Linear => 0,
        SourceTransferFunction::Srgb => 1,
        SourceTransferFunction::Bt709 => 2,
        SourceTransferFunction::Pq => 3,
        SourceTransferFunction::Hlg => 4,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output source transfer {unsupported:?} has no complete numeric contract"
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
    let target_transfer = match target_color.transfer {
        ImageTransferFunction::Linear => 0,
        ImageTransferFunction::Srgb | ImageTransferFunction::Sycc => 1,
        ImageTransferFunction::Bt709 => 2,
        ImageTransferFunction::Pq => 3,
        ImageTransferFunction::Hlg => 4,
        ImageTransferFunction::Bt2020 => 5,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target transfer {unsupported:?} is unsupported"
            )));
        }
    };
    let primaries = primaries_transform(source.primaries, target_color.space)?;
    Ok(ImageColorTransform {
        source_transfer,
        target_transfer,
        primaries,
    })
}

type Matrix3 = [[f32; 3]; 3];

const IDENTITY_3: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const BT709_TO_XYZ: Matrix3 = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175],
    [0.019_333_9, 0.119_192, 0.950_304_1],
];
const BT2020_TO_XYZ: Matrix3 = [
    [0.636_958, 0.144_616_9, 0.168_881],
    [0.262_700_2, 0.677_998_1, 0.059_301_7],
    [0.0, 0.028_072_7, 1.060_985_1],
];
const DISPLAY_P3_TO_XYZ: Matrix3 = [
    [0.486_570_95, 0.265_667_7, 0.198_217_29],
    [0.228_974_57, 0.691_738_55, 0.079_286_91],
    [0.0, 0.045_113_38, 1.043_944_4],
];
const XYZ_TO_BT709: Matrix3 = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];
const XYZ_TO_BT2020: Matrix3 = [
    [1.716_651_2, -0.355_670_8, -0.253_366_3],
    [-0.666_684_4, 1.616_481_2, 0.015_768_5],
    [0.017_639_9, -0.042_770_6, 0.942_103_1],
];
const XYZ_TO_DISPLAY_P3: Matrix3 = [
    [2.493_497, -0.931_383_6, -0.402_710_8],
    [-0.829_489, 1.762_664, 0.023_624_7],
    [0.035_845_8, -0.076_172_4, 0.956_884_5],
];

fn primaries_transform(source: RgbPrimaries, target: ColorSpace) -> Result<[[f32; 4]; 3]> {
    let source_index = match source {
        RgbPrimaries::Bt709 => 0,
        RgbPrimaries::Bt2020 => 1,
        RgbPrimaries::DisplayP3 => 2,
        RgbPrimaries::Undefined => {
            return Err(Error::Unsupported(
                "generic GPU output requires defined source RGB primaries".into(),
            ));
        }
    };
    let target_index = match target {
        ColorSpace::Bt709 => 0,
        ColorSpace::Bt2020 => 1,
        ColorSpace::DisplayP3 => 2,
        unsupported => {
            return Err(Error::Unsupported(format!(
                "generic GPU output target primaries {unsupported:?} are unsupported"
            )));
        }
    };
    let matrix = if source_index == target_index {
        IDENTITY_3
    } else {
        let source_to_xyz = [BT709_TO_XYZ, BT2020_TO_XYZ, DISPLAY_P3_TO_XYZ][source_index];
        let xyz_to_target = [XYZ_TO_BT709, XYZ_TO_BT2020, XYZ_TO_DISPLAY_P3][target_index];
        multiply_matrix3(xyz_to_target, source_to_xyz)
    };
    Ok(matrix.map(|row| [row[0], row[1], row[2], 0.0]))
}

fn multiply_matrix3(lhs: Matrix3, rhs: Matrix3) -> Matrix3 {
    let mut product = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            product[row][column] = lhs[row][0] * rhs[0][column]
                + lhs[row][1] * rhs[1][column]
                + lhs[row][2] * rhs[2][column];
        }
    }
    product
}
