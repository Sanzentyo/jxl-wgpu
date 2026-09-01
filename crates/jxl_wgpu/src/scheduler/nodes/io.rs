// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::autotune::KernelVariant;
use crate::buffer_pool::PooledBuffer;
use crate::upload::{UploadedPlane, aligned_buffer_size, is_word_sample};
use crate::video::PackedImageOutput;
use crate::{Error, Result};
use jxl_gpu_formats::{
    ChromaLocation, ChromaOrder, ColorFormatClass, ColorRange, ColorSpecification, ImageLayout,
    NumericFormatClass, Packed422Order, PixelFormat, PixelFormatClass, RgbChannelOrder, RgbStorage,
    WgslNumericCapability, YcbcrEncoding, classify_pixel_format,
};
use jxl_gpu_protocol::{
    Extent2d, OutputColorEncoding, OutputLayout, OutputOrientation, PlaneId, RenderNode, RenderOp,
    RenderPlan, SampleType,
};

use super::super::pipeline::{
    PipelineFactory, create_pipeline, create_pipeline_with_variant, create_uniform,
    linear_dispatch_shape, record_dispatch, record_linear_dispatch,
};
use super::super::{
    CopyParams, ExtendUniform, ImageOutputUniform, OutputEncoding, OutputMode, OutputTarget,
    PackedOutput, SaveUniform, output_buffer_usage, output_desc, plane, stride, unary_planes,
    validate_storage_buffer_size,
};
use super::color::image_color_transform;
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
    let color_transform = image_color_transform(source_encoding, &request.format)?;
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
    let variant = factory
        .kernel_policy
        .variant_for("rgb_to_image", KernelVariant::Lanes256)?;
    variant.validate_for("rgb_to_image", &factory.device.limits(), 0)?;
    let word_count = layout.logical_size.div_ceil(4);
    let (dispatch_x, dispatch_y, dispatch_width) =
        linear_dispatch_shape(factory.device, word_count, variant)?;
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
        source_transfer: color_transform.source_transfer,
        target_transfer: color_transform.target_transfer,
        _padding: 0,
        primaries_r: color_transform.primaries[0],
        primaries_g: color_transform.primaries[1],
        primaries_b: color_transform.primaries[2],
    };
    let uniform = create_uniform(factory.device, "jxl-wgpu generic image params", &params);
    let pipeline = create_pipeline_with_variant(
        factory,
        "jxl-wgpu RGB to generic image",
        wgpu::include_wgsl!("../../../shaders/rgb_to_image.wgsl"),
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

#[derive(Clone, Copy, Debug)]
pub(in crate::scheduler) struct PreparedImageOutput {
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

pub(in crate::scheduler) fn prepare_image_output(
    layout: &ImageLayout,
) -> Result<PreparedImageOutput> {
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
        YcbcrEncoding::Bt2020 => 2,
        YcbcrEncoding::Bt2020ConstantLuminance => 3,
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

fn to_shader_u32(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::ResourceLimit(
            "generic image output exceeds the shader's 32-bit address space".into(),
        )
    })
}
