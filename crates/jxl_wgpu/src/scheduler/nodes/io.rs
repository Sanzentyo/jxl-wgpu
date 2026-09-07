// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::autotune::KernelVariant;
use crate::buffer_pool::PooledBuffer;
use crate::image_output::{ImageOutputParams, ImageOutputSource, RGB_TO_IMAGE_SHADER};
use crate::upload::{UploadedPlane, aligned_buffer_size, is_word_sample};
use crate::video::PackedImageOutput;
use crate::{Error, Result};
use jxl_gpu_formats::ImageLayout;
use jxl_gpu_protocol::{
    Extent2d, OutputColorEncoding, OutputLayout, OutputOrientation, PlaneId, RenderNode, RenderOp,
    RenderPlan, SampleType,
};

use super::super::pipeline::{
    PipelineFactory, create_pipeline, create_pipeline_with_variant, create_uniform,
    linear_dispatch_shape, record_dispatch, record_linear_dispatch,
};
use super::super::{
    CopyParams, ExtendUniform, OutputEncoding, OutputMode, OutputTarget, PackedOutput, SaveUniform,
    output_buffer_usage, output_desc, plane, stride, unary_planes, validate_storage_buffer_size,
};
pub(in crate::scheduler) fn encode_copy(
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
        wgpu::include_wgsl!("../../../shaders/copy.wgsl"),
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
pub(in crate::scheduler) fn encode_extend(
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
        wgpu::include_wgsl!("../../../shaders/extend.wgsl"),
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
        factory.variant,
    );
    Ok(())
}

pub(in crate::scheduler) fn encode_save(
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
        wgpu::include_wgsl!("../../../shaders/save.wgsl"),
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
            factory.variant,
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

pub(in crate::scheduler) fn encode_image_save(
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
    let padded_size = aligned_buffer_size(layout.logical_size)?;
    validate_storage_buffer_size(
        &factory.device.limits(),
        padded_size,
        "generic image output",
    )?;
    let variant = factory
        .kernel_policy
        .variant_for("rgb_to_image", KernelVariant::Lanes256)?;
    variant.validate_for("rgb_to_image", &factory.device.limits(), 0)?;
    let word_count = layout.logical_size.div_ceil(4);
    let (dispatch_x, dispatch_y, dispatch_width) =
        linear_dispatch_shape(factory.device, word_count, variant)?;
    let params = ImageOutputParams::new(
        &layout,
        ImageOutputSource {
            extent: source_extent,
            orientation: save.orientation,
            strides: [
                stride(&channels[0].desc),
                stride(&channels[1].desc),
                stride(&channels[2].desc),
            ],
            encoding: source_encoding,
        },
        dispatch_width,
    )?;
    let (buffer, pooled) = allocate_output_buffer(
        factory,
        "jxl-wgpu generic image output",
        padded_size,
        output_target,
    );
    let uniform = create_uniform(factory.device, "jxl-wgpu generic image params", &params);
    let pipeline = create_pipeline_with_variant(
        factory,
        "jxl-wgpu RGB to generic image",
        wgpu::ShaderModuleDescriptor {
            label: Some("shared image output"),
            source: wgpu::ShaderSource::Wgsl(RGB_TO_IMAGE_SHADER.into()),
        },
        "main",
        0,
        variant,
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
const fn orientation_code(orientation: OutputOrientation) -> u32 {
    orientation.to_exif_value() - 1
}
pub(in crate::scheduler) fn encode_copy_ids(
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
