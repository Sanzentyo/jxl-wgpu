//! General rectangular VarDCT execution without transform-sized workgroup allocations.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use jxl_gpu_protocol::{
    GroupId, GroupPayload, PackedCoefficients, PlaneId, RenderNode, RenderPlan, ResourceData,
    ResourceId, ResourceUpdate, TransformKind, VarDctResource,
};
use wgpu::util::DeviceExt;

use crate::autotune::KernelVariant;
use crate::context::WgpuBackend;
use crate::pipeline_cache::PipelineKey;
use crate::upload::UploadedPlane;
use crate::vardct::{Rect, find_rect_overlap};
use crate::{Error, Result};

const CHANNELS: usize = 3;
const WORKGROUP_SIZE: u32 = 64;
const PORTABLE_WORKGROUPS_PER_DIMENSION: usize = 65_535;

// Normative inverse AFV basis from jxl_transforms 0.6.0, which in turn follows
// the JPEG XL specification. Stored as vec4-aligned GPU resource data below.
#[allow(clippy::excessive_precision)]
const AFV_BASIS: [f32; 256] = [
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
    0.876902929799142,
    0.2206518106944235,
    -0.10140050393753763,
    -0.1014005039375375,
    0.2206518106944236,
    -0.10140050393753777,
    -0.10140050393753772,
    -0.10140050393753763,
    -0.10140050393753758,
    -0.10140050393753769,
    -0.1014005039375375,
    -0.10140050393753768,
    -0.10140050393753768,
    -0.10140050393753759,
    -0.10140050393753763,
    -0.10140050393753741,
    0.0,
    0.0,
    0.40670075830260755,
    0.44444816619734445,
    0.0,
    0.0,
    0.19574399372042936,
    0.2929100136981264,
    -0.40670075830260716,
    -0.19574399372042872,
    0.0,
    0.11379074460448091,
    -0.44444816619734384,
    -0.29291001369812636,
    -0.1137907446044814,
    0.0,
    0.0,
    0.0,
    -0.21255748058288748,
    0.3085497062849767,
    0.0,
    0.4706702258572536,
    -0.1621205195722993,
    0.0,
    -0.21255748058287047,
    -0.16212051957228327,
    -0.47067022585725277,
    -0.1464291867126764,
    0.3085497062849487,
    0.0,
    -0.14642918671266536,
    0.4251149611657548,
    0.0,
    -std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.4105377591765233,
    0.6235485373547691,
    -0.06435071657946274,
    -0.06435071657946266,
    0.6235485373547694,
    -0.06435071657946284,
    -0.0643507165794628,
    -0.06435071657946274,
    -0.06435071657946272,
    -0.06435071657946279,
    -0.06435071657946266,
    -0.06435071657946277,
    -0.06435071657946277,
    -0.06435071657946273,
    -0.06435071657946274,
    -0.0643507165794626,
    0.0,
    0.0,
    -0.4517556589999482,
    0.15854503551840063,
    0.0,
    -0.04038515160822202,
    0.0074182263792423875,
    0.39351034269210167,
    -0.45175565899994635,
    0.007418226379244351,
    0.1107416575309343,
    0.08298163094882051,
    0.15854503551839705,
    0.3935103426921022,
    0.0829816309488214,
    -0.45175565899994796,
    0.0,
    0.0,
    -0.304684750724869,
    0.5112616136591823,
    0.0,
    0.0,
    -0.290480129728998,
    -0.06578701549142804,
    0.304684750724884,
    0.2904801297290076,
    0.0,
    -0.23889773523344604,
    -0.5112616136592012,
    0.06578701549142545,
    0.23889773523345467,
    0.0,
    0.0,
    0.0,
    0.3017929516615495,
    0.25792362796341184,
    0.0,
    0.16272340142866204,
    0.09520022653475037,
    0.0,
    0.3017929516615503,
    0.09520022653475055,
    -0.16272340142866173,
    -0.35312385449816297,
    0.25792362796341295,
    0.0,
    -0.3531238544981624,
    -0.6035859033230976,
    0.0,
    0.0,
    0.40824829046386274,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.4082482904638628,
    -0.4082482904638635,
    0.0,
    0.0,
    -0.40824829046386296,
    0.0,
    0.4082482904638634,
    0.408248290463863,
    0.0,
    0.0,
    0.0,
    0.1747866975480809,
    0.0812611176717539,
    0.0,
    0.0,
    -0.3675398009862027,
    -0.307882213957909,
    -0.17478669754808135,
    0.3675398009862011,
    0.0,
    0.4826689115059883,
    -0.08126111767175039,
    0.30788221395790305,
    -0.48266891150598584,
    0.0,
    0.0,
    0.0,
    -0.21105601049335784,
    0.18567180916109802,
    0.0,
    0.0,
    0.49215859013738733,
    -0.38525013709251915,
    0.21105601049335806,
    -0.49215859013738905,
    0.0,
    0.17419412659916217,
    -0.18567180916109904,
    0.3852501370925211,
    -0.1741941265991621,
    0.0,
    0.0,
    0.0,
    -0.14266084808807264,
    -0.3416446842253372,
    0.0,
    0.7367497537172237,
    0.24627107722075148,
    -0.08574019035519306,
    -0.14266084808807344,
    0.24627107722075137,
    0.14883399227113567,
    -0.04768680350229251,
    -0.3416446842253373,
    -0.08574019035519267,
    -0.047686803502292804,
    -0.14266084808807242,
    0.0,
    0.0,
    -0.13813540350758585,
    0.3302282550303788,
    0.0,
    0.08755115000587084,
    -0.07946706605909573,
    -0.4613374887461511,
    -0.13813540350758294,
    -0.07946706605910261,
    0.49724647109535086,
    0.12538059448563663,
    0.3302282550303805,
    -0.4613374887461554,
    0.12538059448564315,
    -0.13813540350758452,
    0.0,
    0.0,
    -0.17437602599651067,
    0.0702790691196284,
    0.0,
    -0.2921026642334881,
    0.3623817333531167,
    0.0,
    -0.1743760259965108,
    0.36238173335311646,
    0.29210266423348785,
    -0.4326608024727445,
    0.07027906911962818,
    0.0,
    -0.4326608024727457,
    0.34875205199302267,
    0.0,
    0.0,
    0.11354987314994337,
    -0.07417504595810355,
    0.0,
    0.19402893032594343,
    -0.435190496523228,
    0.21918684838857466,
    0.11354987314994257,
    -0.4351904965232251,
    0.5550443808910661,
    -0.25468277124066463,
    -0.07417504595810233,
    0.2191868483885728,
    -0.25468277124066413,
    0.1135498731499429,
];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuTask {
    coefficient_offset: u32,
    scratch_or_basis_offset: u32,
    matrix_offset: u32,
    quant_index: u32,
    coefficient_origin_x: u32,
    lf_offset: u32,
    channel_mask: u32,
    coefficient_origin_y: u32,
    destination_x_x: u32,
    destination_y_x: u32,
    destination_x_y: u32,
    destination_y_y: u32,
    destination_x_b: u32,
    destination_y_b: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuResourceVector([f32; 4]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GeneralUniform {
    task_base: u32,
    task_count: u32,
    transform_width: u32,
    transform_height: u32,
    transform_area: u32,
    lf_width: u32,
    lf_height: u32,
    quant_offset: u32,
    correlation_offset: u32,
    lf_offset: u32,
    output_width_x: u32,
    output_height_x: u32,
    output_stride_x: u32,
    output_width_y: u32,
    output_height_y: u32,
    output_stride_y: u32,
    output_width_b: u32,
    output_height_b: u32,
    output_stride_b: u32,
    transform_kind: u32,
    correlation_width: u32,
    correlation_height: u32,
    _padding: [u32; 2],
    quant_biases: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<GpuTask>() == 64);
    assert!(std::mem::align_of::<GpuTask>() == 4);
    assert!(std::mem::size_of::<GpuResourceVector>() == 16);
    assert!(std::mem::align_of::<GpuResourceVector>() == 16);
    assert!(std::mem::size_of::<GeneralUniform>() == 112);
    assert!(std::mem::align_of::<GeneralUniform>() == 16);
};

#[derive(Debug)]
struct PreparedBucket {
    transform: TransformKind,
    tasks: Vec<GpuTask>,
}

#[derive(Debug)]
struct PreparedGeneral {
    coefficients: Vec<i32>,
    resources: Vec<GpuResourceVector>,
    quant_offset: u32,
    correlation_offset: u32,
    correlation_width: u32,
    correlation_height: u32,
    lf_offset: u32,
    quant_biases: [f32; 4],
    scratch_scalars: u32,
    buckets: Vec<PreparedBucket>,
}

#[derive(Debug)]
struct ResourceLayout {
    vectors: Vec<GpuResourceVector>,
    quant_offset: u32,
    correlation_offset: u32,
    correlation_width: u32,
    correlation_height: u32,
    lf_offset: u32,
    afv_basis_offset: u32,
    matrix_offsets: Vec<u32>,
}

pub(crate) fn is_required(
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
    resource_id: ResourceId,
) -> bool {
    let packet_requires_general = groups
        .values()
        .filter_map(|group| group.vardct.as_ref())
        .any(|packet| {
            packet.buckets.iter().any(|bucket| {
                bucket.transform != TransformKind::Dct8
                    || bucket.tasks.iter().any(|task| {
                        let first = task.destinations[0];
                        first.is_none()
                            || task
                                .destinations
                                .iter()
                                .any(|destination| *destination != first)
                    })
            })
        });
    let resource_requires_general = resources
        .get(&resource_id)
        .and_then(|update| match &update.data {
            ResourceData::VarDct(resource) => Some(
                resource
                    .dequant_matrices
                    .iter()
                    .any(|matrix| matrix.transform != TransformKind::Dct8),
            ),
            _ => None,
        })
        .unwrap_or(false);
    packet_requires_general || resource_requires_general
}

pub(crate) fn validate_packet(packet: &jxl_gpu_protocol::VarDctPacket) -> Result<()> {
    let coefficient_len = coefficient_len(&packet.coefficients)?;
    validate_packed_overflow(&packet.coefficients, coefficient_len)?;
    for bucket in &packet.buckets {
        let extent = bucket.transform.pixel_extent();
        let area = extent.area().ok_or(Error::BufferSizeOverflow)?;
        let coefficient_count = area
            .checked_mul(CHANNELS)
            .ok_or(Error::BufferSizeOverflow)?;
        for task in &bucket.tasks {
            if task.destinations.iter().all(Option::is_none) {
                return Err(Error::InvalidPayload(format!(
                    "VarDCT {:?} task has no enabled channel destination",
                    bucket.transform
                )));
            }
            let start =
                usize::try_from(task.coefficient_offset).map_err(|_| Error::BufferSizeOverflow)?;
            let end = start
                .checked_add(coefficient_count)
                .ok_or(Error::BufferSizeOverflow)?;
            if end > coefficient_len {
                return Err(Error::InvalidPayload(format!(
                    "VarDCT {:?} coefficient range {start}..{end} exceeds packet length {coefficient_len}",
                    bucket.transform
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn transient_bytes(
    plan: &RenderPlan,
    node: &RenderNode,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<u64> {
    let prepared = prepare(plan, node, groups, resources)?;
    prepared_transient_bytes(&prepared)
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
    let prepared = prepare(plan, node, groups, resources)?;
    if prepared
        .buckets
        .iter()
        .all(|bucket| bucket.tasks.is_empty())
    {
        return Ok(0);
    }
    let outputs = node
        .outputs
        .iter()
        .map(|id| planes.get(id).ok_or(Error::MissingPlane(*id)))
        .collect::<Result<Vec<_>>>()?;
    encode_prepared(backend, encoder, &outputs, &prepared)
}

fn prepare(
    plan: &RenderPlan,
    node: &RenderNode,
    groups: &BTreeMap<GroupId, GroupPayload>,
    resources: &BTreeMap<ResourceId, ResourceUpdate>,
) -> Result<PreparedGeneral> {
    if !node.inputs.is_empty() || node.outputs.len() != CHANNELS || node.resources.len() != 1 {
        return Err(Error::InvalidPayload(
            "VarDCT requires no plane inputs, three outputs, and one typed resource".into(),
        ));
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

    let resource_update = resources.get(&node.resources[0]).ok_or_else(|| {
        Error::InvalidPayload(format!(
            "VarDCT resource {:?} has not been supplied",
            node.resources[0]
        ))
    })?;
    let ResourceData::VarDct(resource) = &resource_update.data else {
        return Err(Error::InvalidPayload(format!(
            "VarDCT resource {:?} does not contain typed VarDCT parameters",
            node.resources[0]
        )));
    };
    let layout = flatten_resource(resource)?;

    let mut coefficients = Vec::new();
    let mut scratch_scalars = 0u32;
    let mut buckets = BTreeMap::<TransformKind, Vec<GpuTask>>::new();
    let mut rects: [Vec<Rect>; CHANNELS] = std::array::from_fn(|_| Vec::new());

    for (group_id, group) in groups {
        let Some(packet) = &group.vardct else {
            continue;
        };
        validate_packet(packet)?;
        let decoded = decode_coefficients(&packet.coefficients)?;
        for bucket in &packet.buckets {
            let extent = bucket.transform.pixel_extent();
            let area = extent.area().ok_or(Error::BufferSizeOverflow)?;
            let area_u32 = u32::try_from(area).map_err(|_| Error::BufferSizeOverflow)?;
            let coefficient_count = area
                .checked_mul(CHANNELS)
                .ok_or(Error::BufferSizeOverflow)?;
            let lf_area = bucket
                .transform
                .lf_extent()
                .area()
                .ok_or(Error::BufferSizeOverflow)?;

            for task in &bucket.tasks {
                let quant_index = usize::from(task.quant_index);
                let matrix_index = usize::from(task.dequant_matrix_index);
                let Some(&matrix_offset) = layout.matrix_offsets.get(matrix_index) else {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} task references missing matrix {matrix_index}",
                        bucket.transform
                    )));
                };
                let matrix = &resource.dequant_matrices[matrix_index];
                if matrix.transform != bucket.transform || matrix.scales.len() != area {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} task selected {:?}/{}-entry matrix {matrix_index}, expected {:?}/{area}",
                        bucket.transform,
                        matrix.transform,
                        matrix.scales.len(),
                        bucket.transform
                    )));
                }
                if quant_index >= resource.quant_scales.len() {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} task references missing quantization data",
                        bucket.transform
                    )));
                }
                let correlation_width_pixels = resource
                    .hf_correlation
                    .extent
                    .width
                    .checked_mul(64)
                    .ok_or(Error::BufferSizeOverflow)?;
                let correlation_height_pixels = resource
                    .hf_correlation
                    .extent
                    .height
                    .checked_mul(64)
                    .ok_or(Error::BufferSizeOverflow)?;
                let coefficient_rect = Rect {
                    x: task.coefficient_origin.0,
                    y: task.coefficient_origin.1,
                    width: extent.width,
                    height: extent.height,
                };
                if !coefficient_rect.is_within(correlation_width_pixels, correlation_height_pixels)
                {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} coefficient rectangle {coefficient_rect:?} exceeds the {}x{} HF correlation grid",
                        bucket.transform, correlation_width_pixels, correlation_height_pixels
                    )));
                }
                let lf_start =
                    usize::try_from(task.lf_offset).map_err(|_| Error::BufferSizeOverflow)?;
                let lf_end = lf_start
                    .checked_add(lf_area)
                    .ok_or(Error::BufferSizeOverflow)?;
                if lf_end > resource.lf_coefficients.len() {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} LF range {lf_start}..{lf_end} exceeds resource length {}",
                        bucket.transform,
                        resource.lf_coefficients.len()
                    )));
                }

                let source_start = usize::try_from(task.coefficient_offset)
                    .map_err(|_| Error::BufferSizeOverflow)?;
                let source_end = source_start
                    .checked_add(coefficient_count)
                    .ok_or(Error::BufferSizeOverflow)?;
                let coefficient_offset =
                    u32::try_from(coefficients.len()).map_err(|_| Error::BufferSizeOverflow)?;
                coefficients.extend_from_slice(&decoded[source_start..source_end]);

                let mut channel_mask = 0u32;
                let mut destinations = [(0u32, 0u32); CHANNELS];
                for (channel, destination) in task.destinations.iter().copied().enumerate() {
                    let Some((x, y)) = destination else {
                        continue;
                    };
                    let rect = Rect {
                        x,
                        y,
                        width: extent.width,
                        height: extent.height,
                    };
                    if !rect.is_within(
                        outputs[channel].extent.width,
                        outputs[channel].extent.height,
                    ) {
                        return Err(Error::InvalidPayload(format!(
                            "VarDCT {:?} task in group {group_id:?} writes channel {channel} {rect:?} outside {:?}",
                            bucket.transform, outputs[channel].extent
                        )));
                    }
                    channel_mask |= 1 << channel;
                    destinations[channel] = (x, y);
                    rects[channel].push(rect);
                }
                if channel_mask == 0 {
                    return Err(Error::InvalidPayload(format!(
                        "VarDCT {:?} task has no enabled destination",
                        bucket.transform
                    )));
                }
                let scratch_offset = scratch_scalars;
                scratch_scalars = scratch_scalars
                    .checked_add(area_u32.checked_mul(3).ok_or(Error::BufferSizeOverflow)?)
                    .ok_or(Error::BufferSizeOverflow)?;
                let task_scratch_offset = if matches!(
                    bucket.transform,
                    TransformKind::Afv0
                        | TransformKind::Afv1
                        | TransformKind::Afv2
                        | TransformKind::Afv3
                ) {
                    layout.afv_basis_offset
                } else {
                    scratch_offset
                };
                buckets.entry(bucket.transform).or_default().push(GpuTask {
                    coefficient_offset,
                    scratch_or_basis_offset: task_scratch_offset,
                    matrix_offset,
                    quant_index: u32::from(task.quant_index),
                    coefficient_origin_x: task.coefficient_origin.0,
                    lf_offset: task.lf_offset,
                    channel_mask,
                    coefficient_origin_y: task.coefficient_origin.1,
                    destination_x_x: destinations[0].0,
                    destination_y_x: destinations[0].1,
                    destination_x_y: destinations[1].0,
                    destination_y_y: destinations[1].1,
                    destination_x_b: destinations[2].0,
                    destination_y_b: destinations[2].1,
                    _pad1: 0,
                    _pad2: 0,
                });
            }
        }
    }

    for channel_rects in &rects {
        if let Some((_, second)) = find_rect_overlap(channel_rects)? {
            return Err(Error::InvalidPayload(format!(
                "VarDCT task destination {:?} overlaps another task in the same channel",
                channel_rects[second]
            )));
        }
    }
    Ok(PreparedGeneral {
        coefficients,
        resources: layout.vectors,
        quant_offset: layout.quant_offset,
        correlation_offset: layout.correlation_offset,
        correlation_width: layout.correlation_width,
        correlation_height: layout.correlation_height,
        lf_offset: layout.lf_offset,
        quant_biases: resource.quant_biases,
        scratch_scalars,
        buckets: buckets
            .into_iter()
            .map(|(transform, tasks)| PreparedBucket { transform, tasks })
            .collect(),
    })
}

fn flatten_resource(resource: &VarDctResource) -> Result<ResourceLayout> {
    if resource
        .quant_biases
        .iter()
        .chain(resource.quant_scales.iter().flatten())
        .chain(resource.hf_correlation.values.iter().flatten())
        .chain(resource.lf_coefficients.iter().flatten())
        .chain(
            resource
                .dequant_matrices
                .iter()
                .flat_map(|matrix| matrix.scales.iter().flatten()),
        )
        .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidPayload(
            "VarDCT resource contains a non-finite parameter".into(),
        ));
    }
    let correlation_area = resource
        .hf_correlation
        .extent
        .area()
        .ok_or(Error::BufferSizeOverflow)?;
    if resource.hf_correlation.extent.width == 0
        || resource.hf_correlation.extent.height == 0
        || resource.hf_correlation.values.len() != correlation_area
    {
        return Err(Error::InvalidPayload(format!(
            "VarDCT HF correlation grid {:?} contains {} values, expected {correlation_area}",
            resource.hf_correlation.extent,
            resource.hf_correlation.values.len()
        )));
    }
    let mut vectors = Vec::new();
    let quant_offset = 0;
    vectors.extend(
        resource
            .quant_scales
            .iter()
            .map(|value| GpuResourceVector([value[0], value[1], value[2], 0.0])),
    );
    let correlation_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(
        resource
            .hf_correlation
            .values
            .iter()
            .map(|value| GpuResourceVector([value[0], value[1], 0.0, 0.0])),
    );
    let lf_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(
        resource
            .lf_coefficients
            .iter()
            .map(|value| GpuResourceVector([value[0], value[1], value[2], 0.0])),
    );
    let mut matrix_offsets = Vec::with_capacity(resource.dequant_matrices.len());
    for matrix in &resource.dequant_matrices {
        let expected = matrix
            .transform
            .pixel_extent()
            .area()
            .ok_or(Error::BufferSizeOverflow)?;
        if matrix.scales.len() != expected {
            return Err(Error::InvalidPayload(format!(
                "VarDCT {:?} matrix has {} entries, expected {expected}",
                matrix.transform,
                matrix.scales.len()
            )));
        }
        matrix_offsets.push(u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?);
        vectors.extend(
            matrix
                .scales
                .iter()
                .map(|value| GpuResourceVector([value[0], value[1], value[2], 0.0])),
        );
    }
    let afv_basis_offset = u32::try_from(vectors.len()).map_err(|_| Error::BufferSizeOverflow)?;
    vectors.extend(
        AFV_BASIS
            .chunks_exact(4)
            .map(|values| GpuResourceVector([values[0], values[1], values[2], values[3]])),
    );
    Ok(ResourceLayout {
        vectors,
        quant_offset,
        correlation_offset,
        correlation_width: resource.hf_correlation.extent.width,
        correlation_height: resource.hf_correlation.extent.height,
        lf_offset,
        afv_basis_offset,
        matrix_offsets,
    })
}

fn prepared_transient_bytes(prepared: &PreparedGeneral) -> Result<u64> {
    if prepared
        .buckets
        .iter()
        .all(|bucket| bucket.tasks.is_empty())
    {
        return Ok(0);
    }
    let scratch_bytes = u64::from(prepared.scratch_scalars)
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(Error::BufferSizeOverflow)?;
    let mut bytes = buffer_bytes(&prepared.coefficients)?
        .checked_add(buffer_bytes(&prepared.resources)?)
        .and_then(|value| value.checked_add(scratch_bytes.checked_mul(2)?))
        .ok_or(Error::BufferSizeOverflow)?;
    for bucket in &prepared.buckets {
        bytes = bytes
            .checked_add(buffer_bytes(&bucket.tasks)?)
            .ok_or(Error::BufferSizeOverflow)?;
        let chunks = bucket
            .tasks
            .len()
            .div_ceil(PORTABLE_WORKGROUPS_PER_DIMENSION);
        bytes = bytes
            .checked_add(
                u64::try_from(chunks)
                    .ok()
                    .and_then(|chunks| {
                        chunks.checked_mul(std::mem::size_of::<GeneralUniform>() as u64)
                    })
                    .ok_or(Error::BufferSizeOverflow)?,
            )
            .ok_or(Error::BufferSizeOverflow)?;
    }
    Ok(bytes)
}

fn encode_prepared(
    backend: &WgpuBackend,
    encoder: &mut wgpu::CommandEncoder,
    outputs: &[&UploadedPlane],
    prepared: &PreparedGeneral,
) -> Result<u32> {
    let device = &backend.device;
    ensure_upload_fits(device, &prepared.coefficients)?;
    ensure_upload_fits(device, &prepared.resources)?;
    let scratch_bytes = u64::from(prepared.scratch_scalars)
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or(Error::BufferSizeOverflow)?;
    ensure_storage_bytes_fit(device, scratch_bytes)?;

    let coefficients = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu general VarDCT coefficients"),
        contents: bytemuck::cast_slice(&prepared.coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let resources = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("jxl-wgpu general VarDCT resources"),
        contents: bytemuck::cast_slice(&prepared.resources),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scratch_desc = wgpu::BufferDescriptor {
        label: Some("jxl-wgpu general VarDCT scratch"),
        size: scratch_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    };
    let dequantized = device.create_buffer(&scratch_desc);
    let horizontal = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("jxl-wgpu general VarDCT horizontal scratch"),
        ..scratch_desc
    });
    let dequantize_pipeline = pipeline(backend, "dequantize");
    let horizontal_pipeline = pipeline(backend, "horizontal_idct");
    let vertical_pipeline = pipeline(backend, "vertical_idct");

    let mut dispatches = 0u32;
    for bucket in &prepared.buckets {
        if bucket.tasks.is_empty() {
            continue;
        }
        ensure_upload_fits(device, &bucket.tasks)?;
        let tasks = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("jxl-wgpu general VarDCT tasks"),
            contents: bytemuck::cast_slice(&bucket.tasks),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let extent = bucket.transform.pixel_extent();
        let lf_extent = bucket.transform.lf_extent();
        let area = u32::try_from(extent.area().ok_or(Error::BufferSizeOverflow)?)
            .map_err(|_| Error::BufferSizeOverflow)?;
        let dequantize_x = area.div_ceil(WORKGROUP_SIZE);
        let transform_x = area
            .checked_mul(3)
            .ok_or(Error::BufferSizeOverflow)?
            .div_ceil(WORKGROUP_SIZE);

        for (chunk_index, chunk) in bucket
            .tasks
            .chunks(PORTABLE_WORKGROUPS_PER_DIMENSION)
            .enumerate()
        {
            let task_base = chunk_index
                .checked_mul(PORTABLE_WORKGROUPS_PER_DIMENSION)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or(Error::BufferSizeOverflow)?;
            let uniform = GeneralUniform {
                task_base,
                task_count: u32::try_from(chunk.len()).map_err(|_| Error::BufferSizeOverflow)?,
                transform_width: extent.width,
                transform_height: extent.height,
                transform_area: area,
                lf_width: lf_extent.width,
                lf_height: lf_extent.height,
                quant_offset: prepared.quant_offset,
                correlation_offset: prepared.correlation_offset,
                lf_offset: prepared.lf_offset,
                output_width_x: outputs[0].desc.extent.width,
                output_height_x: outputs[0].desc.extent.height,
                output_stride_x: stride(&outputs[0].desc),
                output_width_y: outputs[1].desc.extent.width,
                output_height_y: outputs[1].desc.extent.height,
                output_stride_y: stride(&outputs[1].desc),
                output_width_b: outputs[2].desc.extent.width,
                output_height_b: outputs[2].desc.extent.height,
                output_stride_b: stride(&outputs[2].desc),
                transform_kind: if bucket.transform.is_special() {
                    special_transform_code(bucket.transform)?
                } else {
                    0
                },
                correlation_width: prepared.correlation_width,
                correlation_height: prepared.correlation_height,
                _padding: [0; 2],
                quant_biases: prepared.quant_biases,
            };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu general VarDCT params"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            if bucket.transform.is_special() {
                let special_pipeline = special_pipeline(backend);
                let special_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("jxl-wgpu special VarDCT bindings"),
                    layout: &special_pipeline.get_bind_group_layout(0),
                    entries: &[
                        entry(0, &coefficients),
                        entry(1, &tasks),
                        entry(2, &resources),
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: outputs[0].binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: outputs[1].binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: outputs[2].binding(),
                        },
                        entry(8, &uniform),
                    ],
                });
                dispatch(
                    encoder,
                    "jxl-wgpu special VarDCT",
                    &special_pipeline,
                    &special_bind_group,
                    1,
                    u32::try_from(chunk.len()).map_err(|_| Error::BufferSizeOverflow)?,
                );
                dispatches = dispatches.checked_add(1).ok_or(Error::BufferSizeOverflow)?;
                continue;
            }

            let dequantize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu general VarDCT dequantize bindings"),
                layout: &dequantize_pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(0, &coefficients),
                    entry(1, &tasks),
                    entry(2, &resources),
                    entry(3, &dequantized),
                    entry(8, &uniform),
                ],
            });
            let horizontal_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu general VarDCT horizontal bindings"),
                layout: &horizontal_pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(1, &tasks),
                    entry(3, &dequantized),
                    entry(4, &horizontal),
                    entry(8, &uniform),
                ],
            });
            let vertical_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("jxl-wgpu general VarDCT vertical bindings"),
                layout: &vertical_pipeline.get_bind_group_layout(0),
                entries: &[
                    entry(1, &tasks),
                    entry(4, &horizontal),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: outputs[0].binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: outputs[1].binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: outputs[2].binding(),
                    },
                    entry(8, &uniform),
                ],
            });

            dispatch(
                encoder,
                "jxl-wgpu general VarDCT dequantize",
                &dequantize_pipeline,
                &dequantize_bind_group,
                dequantize_x,
                u32::try_from(chunk.len()).map_err(|_| Error::BufferSizeOverflow)?,
            );
            dispatch(
                encoder,
                "jxl-wgpu general VarDCT horizontal",
                &horizontal_pipeline,
                &horizontal_bind_group,
                transform_x,
                u32::try_from(chunk.len()).map_err(|_| Error::BufferSizeOverflow)?,
            );
            dispatch(
                encoder,
                "jxl-wgpu general VarDCT vertical",
                &vertical_pipeline,
                &vertical_bind_group,
                transform_x,
                u32::try_from(chunk.len()).map_err(|_| Error::BufferSizeOverflow)?,
            );
            dispatches = dispatches.checked_add(3).ok_or(Error::BufferSizeOverflow)?;
        }
    }
    Ok(dispatches)
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    label: &'static str,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    x: u32,
    y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

fn pipeline(backend: &WgpuBackend, entry_point: &'static str) -> Arc<wgpu::ComputePipeline> {
    let label = format!("jxl-wgpu vardct-general-{entry_point}");
    let key = PipelineKey::new(label.clone(), entry_point, KernelVariant::Tile8x8, 0);
    if let Some(pipeline) = backend.pipelines.get(&key) {
        return pipeline;
    }
    match backend.pipelines.get_or_insert_with(key, || {
        let module = backend
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/vardct_general.wgsl"));
        Ok::<_, std::convert::Infallible>(backend.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&label),
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

fn special_pipeline(backend: &WgpuBackend) -> Arc<wgpu::ComputePipeline> {
    let label = "jxl-wgpu vardct-special";
    let key = PipelineKey::new(label, "main", KernelVariant::Tile8x8, 0);
    if let Some(pipeline) = backend.pipelines.get(&key) {
        return pipeline;
    }
    match backend.pipelines.get_or_insert_with(key, || {
        let module = backend
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/vardct_special.wgsl"));
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

fn special_transform_code(transform: TransformKind) -> Result<u32> {
    match transform {
        TransformKind::Hornuss => Ok(0),
        TransformKind::Dct2x2 => Ok(1),
        TransformKind::Dct4x4 => Ok(2),
        TransformKind::Dct4x8 => Ok(3),
        TransformKind::Dct8x4 => Ok(4),
        TransformKind::Afv0 => Ok(5),
        TransformKind::Afv1 => Ok(6),
        TransformKind::Afv2 => Ok(7),
        TransformKind::Afv3 => Ok(8),
        _ => Err(Error::InvalidPayload(format!(
            "regular transform {transform:?} was routed to the special VarDCT kernel"
        ))),
    }
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

fn ensure_upload_fits<T>(device: &wgpu::Device, values: &[T]) -> Result<()> {
    ensure_storage_bytes_fit(device, buffer_bytes(values)?)
}

fn ensure_storage_bytes_fit(device: &wgpu::Device, bytes: u64) -> Result<()> {
    let limits = device.limits();
    let maximum = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    if bytes > maximum {
        return Err(Error::ResourceLimit(format!(
            "general VarDCT buffer needs {bytes} bytes, device permits {maximum}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use jxl_gpu_protocol::{
        Border2d, Extent2d, FrameSessionDesc, GroupPayload, MemoryMode, OutputDesc, OutputId,
        OutputLayout, PackedCoefficients, PlaneData, PlaneDesc, PlaneRole, PrecisionContract,
        PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderPlan, ResourceUpdate,
        SaveParams, Scale2d, TransformBucket, TransformTask, VarDctCorrelationGrid,
        VarDctDequantMatrix, VarDctPacket,
    };

    use crate::{WgpuBackend, WgpuBackendConfig};

    fn abi_words<T: Pod>(value: &T) -> &[u32] {
        bytemuck::cast_slice(std::slice::from_ref(value))
    }

    fn assert_wgsl_fields(shader: &str, name: &str, expected: &[&str]) {
        let module = naga::front::wgsl::parse_str(shader).expect("WGSL parses");
        let ty = module
            .types
            .iter()
            .map(|(_, ty)| ty)
            .find(|ty| ty.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("WGSL struct '{name}' is missing"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("WGSL type '{name}' is not a struct");
        };
        let actual = members
            .iter()
            .map(|member| member.name.as_deref().expect("WGSL struct member is named"))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "WGSL field-order drift for {name}");
    }

    #[test]
    fn general_vardct_rust_and_wgsl_abis_are_pinned() {
        let task = GpuTask {
            coefficient_offset: 1,
            scratch_or_basis_offset: 2,
            matrix_offset: 3,
            quant_index: 4,
            coefficient_origin_x: 5,
            lf_offset: 6,
            channel_mask: 7,
            coefficient_origin_y: 8,
            destination_x_x: 9,
            destination_y_x: 10,
            destination_x_y: 11,
            destination_y_y: 12,
            destination_x_b: 13,
            destination_y_b: 14,
            _pad1: 15,
            _pad2: 16,
        };
        assert_eq!(abi_words(&task), &(1..=16).collect::<Vec<_>>());

        let params = GeneralUniform {
            task_base: 1,
            task_count: 2,
            transform_width: 3,
            transform_height: 4,
            transform_area: 5,
            lf_width: 6,
            lf_height: 7,
            quant_offset: 8,
            correlation_offset: 9,
            lf_offset: 10,
            output_width_x: 11,
            output_height_x: 12,
            output_stride_x: 13,
            output_width_y: 14,
            output_height_y: 15,
            output_stride_y: 16,
            output_width_b: 17,
            output_height_b: 18,
            output_stride_b: 19,
            transform_kind: 20,
            correlation_width: 21,
            correlation_height: 22,
            _padding: [23, 24],
            quant_biases: [
                f32::from_bits(25),
                f32::from_bits(26),
                f32::from_bits(27),
                f32::from_bits(28),
            ],
        };
        assert_eq!(abi_words(&params), &(1..=28).collect::<Vec<_>>());

        let task_fields = [
            "coefficient_offset",
            "scratch_or_basis_offset",
            "matrix_offset",
            "quant_index",
            "coefficient_origin_x",
            "lf_offset",
            "channel_mask",
            "coefficient_origin_y",
            "destination_x_x",
            "destination_y_x",
            "destination_x_y",
            "destination_y_y",
            "destination_x_b",
            "destination_y_b",
            "_pad1",
            "_pad2",
        ];
        let param_fields = [
            "task_base",
            "task_count",
            "transform_width",
            "transform_height",
            "transform_area",
            "lf_width",
            "lf_height",
            "quant_offset",
            "correlation_offset",
            "lf_offset",
            "output_width_x",
            "output_height_x",
            "output_stride_x",
            "output_width_y",
            "output_height_y",
            "output_stride_y",
            "output_width_b",
            "output_height_b",
            "output_stride_b",
            "transform_kind",
            "correlation_width",
            "correlation_height",
            "_padding",
            "quant_biases",
        ];
        for shader in [
            include_str!("../shaders/vardct_general.wgsl"),
            include_str!("../shaders/vardct_special.wgsl"),
        ] {
            assert_wgsl_fields(shader, "Task", &task_fields);
            assert_wgsl_fields(shader, "Params", &param_fields);
        }
    }

    #[test]
    fn general_transient_estimate_includes_grid_resources_and_expanded_abi() {
        let prepared = PreparedGeneral {
            coefficients: vec![0; 192],
            resources: vec![GpuResourceVector::zeroed(); 4],
            quant_offset: 0,
            correlation_offset: 1,
            correlation_width: 2,
            correlation_height: 1,
            lf_offset: 3,
            quant_biases: [0.0; 4],
            scratch_scalars: 32,
            buckets: vec![PreparedBucket {
                transform: TransformKind::Dct8x16,
                tasks: vec![GpuTask::zeroed(); 2],
            }],
        };
        let expected = 192 * std::mem::size_of::<i32>()
            + 4 * std::mem::size_of::<GpuResourceVector>()
            + 2 * 32 * std::mem::size_of::<f32>()
            + 2 * std::mem::size_of::<GpuTask>()
            + std::mem::size_of::<GeneralUniform>();
        assert_eq!(
            prepared_transient_bytes(&prepared).unwrap(),
            expected as u64
        );
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

    fn plan(extent: Extent2d) -> RenderPlan {
        let channels = [PlaneId(0), PlaneId(1), PlaneId(2)];
        RenderPlan {
            planes: channels
                .iter()
                .map(|&id| PlaneDesc {
                    id,
                    extent,
                    stride: extent.width,
                    sample_type: jxl_gpu_protocol::SampleType::F32,
                    role: PlaneRole::Intermediate,
                })
                .collect(),
            nodes: vec![
                RenderNode {
                    name: "general-vardct".into(),
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
            max_resident_bytes: 32 * 1024 * 1024,
            max_scratch_bytes: 32 * 1024 * 1024,
        }
    }

    fn codec_transform(transform: TransformKind) -> jxl_transforms::transform_map::HfTransformType {
        use jxl_transforms::transform_map::HfTransformType as Hf;
        match transform {
            TransformKind::Dct8 => Hf::DCT,
            TransformKind::Hornuss => Hf::IDENTITY,
            TransformKind::Dct2x2 => Hf::DCT2X2,
            TransformKind::Dct4x4 => Hf::DCT4X4,
            TransformKind::Dct16x16 => Hf::DCT16X16,
            TransformKind::Dct32x32 => Hf::DCT32X32,
            TransformKind::Dct16x8 => Hf::DCT16X8,
            TransformKind::Dct8x16 => Hf::DCT8X16,
            TransformKind::Dct32x8 => Hf::DCT32X8,
            TransformKind::Dct8x32 => Hf::DCT8X32,
            TransformKind::Dct32x16 => Hf::DCT32X16,
            TransformKind::Dct16x32 => Hf::DCT16X32,
            TransformKind::Dct4x8 => Hf::DCT4X8,
            TransformKind::Dct8x4 => Hf::DCT8X4,
            TransformKind::Afv0 => Hf::AFV0,
            TransformKind::Afv1 => Hf::AFV1,
            TransformKind::Afv2 => Hf::AFV2,
            TransformKind::Afv3 => Hf::AFV3,
            TransformKind::Dct64x64 => Hf::DCT64X64,
            TransformKind::Dct64x32 => Hf::DCT64X32,
            TransformKind::Dct32x64 => Hf::DCT32X64,
            TransformKind::Dct128x128 => Hf::DCT128X128,
            TransformKind::Dct128x64 => Hf::DCT128X64,
            TransformKind::Dct64x128 => Hf::DCT64X128,
            TransformKind::Dct256x256 => Hf::DCT256X256,
            TransformKind::Dct256x128 => Hf::DCT256X128,
            TransformKind::Dct128x256 => Hf::DCT128X256,
        }
    }

    fn reference_block(
        transform: TransformKind,
        task: &TransformTask,
        coefficients: &[i32],
        resource: &VarDctResource,
    ) -> [Vec<f32>; 3] {
        let area = transform.pixel_extent().area().unwrap();
        let lf_area = transform.lf_extent().area().unwrap();
        let matrix = &resource.dequant_matrices[usize::from(task.dequant_matrix_index)].scales;
        let quant = resource.quant_scales[usize::from(task.quant_index)];
        let mut blocks: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; area]);
        for index in 0..area {
            let extent = transform.pixel_extent();
            let index_u32 = u32::try_from(index).unwrap();
            let (frequency_x, frequency_y) = if extent.height < extent.width {
                (index_u32 % extent.width, index_u32 / extent.width)
            } else {
                (index_u32 / extent.height, index_u32 % extent.height)
            };
            let cell_x = (task.coefficient_origin.0 + frequency_x) / 64;
            let cell_y = (task.coefficient_origin.1 + frequency_y) / 64;
            let correlation = resource.hf_correlation.values
                [(cell_y * resource.hf_correlation.extent.width + cell_x) as usize];
            let mut values = [0.0; 3];
            for channel in 0..3 {
                let value = coefficients[channel * area + index] as f32;
                values[channel] = value * quant[channel] * matrix[index][channel];
            }
            blocks[0][index] = correlation[0].mul_add(values[1], values[0]);
            blocks[1][index] = values[1];
            blocks[2][index] = correlation[1].mul_add(values[1], values[2]);
        }
        for channel in 0..3 {
            let mut lf = resource.lf_coefficients
                [task.lf_offset as usize..task.lf_offset as usize + lf_area]
                .iter()
                .map(|value| value[channel])
                .collect::<Vec<_>>();
            jxl_transforms::transform::transform_to_pixels(
                codec_transform(transform),
                &mut lf,
                &mut blocks[channel],
            );
        }
        blocks
    }

    fn packed_transform_index(
        transform: TransformKind,
        frequency_x: u32,
        frequency_y: u32,
    ) -> usize {
        let extent = transform.pixel_extent();
        let index = if extent.height < extent.width {
            frequency_y * extent.width + frequency_x
        } else {
            frequency_x * extent.height + frequency_y
        };
        usize::try_from(index).unwrap()
    }

    fn run_regular_transform_case(backend: &WgpuBackend, transform: TransformKind) -> Result<()> {
        assert!(!transform.is_special());
        let block_extent = transform.pixel_extent();
        let extent = Extent2d::new(block_extent.width + 3, block_extent.height + 3);
        let area = block_extent.area().unwrap();
        let lf_extent = transform.lf_extent();
        let lf_area = lf_extent.area().unwrap();
        let task = TransformTask {
            coefficient_offset: 0,
            destinations: [Some((1, 1)); 3],
            quant_index: 0,
            dequant_matrix_index: 0,
            coefficient_origin: (32, 32),
            lf_offset: 0,
        };
        let frequencies = [
            (block_extent.width - 1, block_extent.height - 1),
            (block_extent.width / 2, block_extent.height / 3),
            (block_extent.width - 2, lf_extent.height),
            (lf_extent.width, block_extent.height - 2),
        ];
        let mut coefficients = vec![0i32; area * 3];
        for channel in 0..3usize {
            for (slot, &(frequency_x, frequency_y)) in frequencies.iter().enumerate() {
                let index = packed_transform_index(transform, frequency_x, frequency_y);
                coefficients[channel * area + index] = i32::try_from((channel + 1) * (slot + 2))
                    .unwrap()
                    * if slot.is_multiple_of(2) { 1 } else { -1 };
            }
        }
        let correlation_extent = Extent2d::new(
            (task.coefficient_origin.0 + block_extent.width).div_ceil(64),
            (task.coefficient_origin.1 + block_extent.height).div_ceil(64),
        );
        let resource = VarDctResource {
            quant_biases: [1.0, 1.0, 1.0, 0.0],
            quant_scales: vec![[0.75, 1.0, 1.25]],
            dequant_matrices: vec![VarDctDequantMatrix {
                transform,
                scales: (0..area)
                    .map(|index| {
                        let scale = 0.75 + (index % 7) as f32 * 0.03125;
                        [scale, scale + 0.125, scale + 0.25]
                    })
                    .collect(),
            }],
            hf_correlation: VarDctCorrelationGrid {
                extent: correlation_extent,
                values: (0..correlation_extent.area().unwrap())
                    .map(|index| {
                        let scale = index as f32 + 1.0;
                        [0.03125 * scale, -0.046875 * scale]
                    })
                    .collect(),
            },
            lf_coefficients: (0..lf_area)
                .map(|index| {
                    let value = (index % 11) as f32 * 0.125 - 0.5;
                    [value, -0.75 * value, 0.5 * value]
                })
                .collect(),
        };
        let expected_block = reference_block(transform, &task, &coefficients, &resource);
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(coefficients),
            buckets: vec![TransformBucket {
                transform,
                tasks: vec![task],
            }],
        };

        let mut session = backend.create_session(&frame(extent), Arc::new(plan(extent)))?;
        session.update_resource(ResourceUpdate {
            id: ResourceId(0),
            revision: 0,
            data: ResourceData::VarDct(resource),
        })?;
        session.enqueue(GroupPayload {
            group: GroupId(0),
            revision: 0,
            complete: true,
            planes: Vec::new(),
            vardct: Some(packet),
        })?;
        let token = session.submit(RenderIntent::Final)?;
        let output = session.wait(token)?;
        let PlaneData::F32(actual) = &output.outputs[0].data else {
            panic!("{transform:?} output was not F32");
        };
        let plane_area = extent.area().unwrap();
        let maximum_dimension = block_extent.width.max(block_extent.height);
        let absolute_tolerance: f32 = match maximum_dimension {
            0..=32 => 2.0e-3,
            33..=64 => 4.0e-3,
            65..=128 => 8.0e-3,
            _ => 2.0e-2,
        };
        for channel in 0..3 {
            for y in 0..block_extent.height as usize {
                for x in 0..block_extent.width as usize {
                    let output_index = (y + 1) * extent.width as usize + x + 1;
                    let expected = expected_block[channel][y * block_extent.width as usize + x];
                    let actual = actual[channel * plane_area + output_index];
                    let tolerance = absolute_tolerance.max(expected.abs() * 2.5e-4);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{transform:?} channel {channel} ({x}, {y}): GPU {actual}, codec {expected}, tolerance {tolerance}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn every_regular_transform_matches_codec_oracle_on_gpu() {
        let Some(backend) = test_backend() else {
            return;
        };
        for transform in TransformKind::ALL {
            if transform.is_special()
                || matches!(
                    transform,
                    TransformKind::Dct128x128 | TransformKind::Dct256x256
                )
            {
                continue;
            }
            if let Err(error) = run_regular_transform_case(&backend, transform) {
                let device_limited_256 = transform
                    .pixel_extent()
                    .width
                    .max(transform.pixel_extent().height)
                    == 256
                    && matches!(
                        error,
                        Error::ResourceLimit(_) | Error::MemoryBackpressure(_)
                    );
                if device_limited_256 {
                    eprintln!("skipping {transform:?} GPU oracle case on this device: {error}");
                    continue;
                }
                panic!("{transform:?} GPU oracle case failed: {error}");
            }
        }
    }

    #[test]
    fn dct128_and_dct256_cross_cfl_cells_on_gpu() {
        let Some(backend) = test_backend() else {
            return;
        };
        for transform in [TransformKind::Dct128x128, TransformKind::Dct256x256] {
            if let Err(error) = run_regular_transform_case(&backend, transform) {
                let device_limited_256 = transform == TransformKind::Dct256x256
                    && matches!(
                        error,
                        Error::ResourceLimit(_) | Error::MemoryBackpressure(_)
                    );
                if device_limited_256 {
                    eprintln!("skipping {transform:?} GPU oracle case on this device: {error}");
                    continue;
                }
                panic!("{transform:?} cross-cell GPU oracle case failed: {error}");
            }
        }
    }

    #[test]
    fn mixed_rectangular_tasks_execute_at_odd_tail_on_gpu() {
        let Some(backend) = test_backend() else {
            return;
        };
        let extent = Extent2d::new(29, 25);
        let transforms = [TransformKind::Dct8x16, TransformKind::Dct16x8];
        let tasks = [
            TransformTask {
                coefficient_offset: 0,
                destinations: [Some((1, 1)), Some((2, 2)), None],
                quant_index: 0,
                dequant_matrix_index: 0,
                coefficient_origin: (0, 0),
                lf_offset: 0,
            },
            TransformTask {
                coefficient_offset: 384,
                destinations: [Some((20, 7)), Some((20, 8)), Some((19, 7))],
                quant_index: 0,
                dequant_matrix_index: 1,
                coefficient_origin: (0, 0),
                lf_offset: 2,
            },
        ];
        let mut coefficients = vec![0i32; 768];
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = ((index * 29 + 7) % 17) as i32 - 8;
        }
        let resource = VarDctResource {
            quant_biases: [1.0, 1.0, 1.0, 0.0],
            quant_scales: vec![[0.75, 1.25, 0.5]],
            dequant_matrices: transforms
                .iter()
                .map(|&transform| VarDctDequantMatrix {
                    transform,
                    scales: vec![[1.0, 0.5, 1.5]; 128],
                })
                .collect(),
            hf_correlation: VarDctCorrelationGrid {
                extent: Extent2d::new(1, 1),
                values: vec![[0.125, -0.25]],
            },
            lf_coefficients: vec![
                [2.0, -1.0, 0.5],
                [-0.25, 1.5, 3.0],
                [4.0, 0.75, -2.0],
                [1.25, -3.0, 0.125],
            ],
        };
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(coefficients.clone()),
            buckets: transforms
                .iter()
                .zip(tasks)
                .map(|(&transform, task)| TransformBucket {
                    transform,
                    tasks: vec![task],
                })
                .collect(),
        };

        let mut expected = [
            vec![0.0; extent.area().unwrap()],
            vec![0.0; extent.area().unwrap()],
            vec![0.0; extent.area().unwrap()],
        ];
        for ((&transform, task), coefficient_block) in transforms
            .iter()
            .zip(&tasks)
            .zip(coefficients.chunks_exact(384))
        {
            let block = reference_block(transform, task, coefficient_block, &resource);
            let block_extent = transform.pixel_extent();
            for channel in 0..3 {
                let Some((origin_x, origin_y)) = task.destinations[channel] else {
                    continue;
                };
                for y in 0..block_extent.height as usize {
                    for x in 0..block_extent.width as usize {
                        expected[channel][(origin_y as usize + y) * extent.width as usize
                            + origin_x as usize
                            + x] = block[channel][y * block_extent.width as usize + x];
                    }
                }
            }
        }

        let mut session = backend
            .create_session(&frame(extent), Arc::new(plan(extent)))
            .expect("create general VarDCT session");
        session
            .update_resource(ResourceUpdate {
                id: ResourceId(0),
                revision: 0,
                data: ResourceData::VarDct(resource),
            })
            .expect("supply general VarDCT resource");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: Vec::new(),
                vardct: Some(packet),
            })
            .expect("enqueue mixed transform packet");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit mixed transforms");
        let frame = session.wait(token).expect("read mixed transform output");
        let PlaneData::F32(actual) = &frame.outputs[0].data else {
            panic!("general VarDCT output was not F32");
        };
        let plane_area = extent.area().unwrap();
        for channel in 0..3 {
            for (index, (&expected, &actual)) in expected[channel]
                .iter()
                .zip(&actual[channel * plane_area..(channel + 1) * plane_area])
                .enumerate()
            {
                let tolerance = 1.0e-3_f32.max(expected.abs() * 2.0e-5);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "channel {channel} sample {index}: GPU {actual}, codec {expected}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn all_non_afv_special_transforms_match_codec_oracle_on_gpu() {
        let Some(backend) = test_backend() else {
            return;
        };
        let extent = Extent2d::new(29, 19);
        let transforms = [
            TransformKind::Hornuss,
            TransformKind::Dct2x2,
            TransformKind::Dct4x4,
            TransformKind::Dct4x8,
            TransformKind::Dct8x4,
        ];
        let origins = [(1, 1), (10, 1), (19, 1), (1, 10), (10, 10)];
        let tasks = std::array::from_fn::<_, 5, _>(|index| TransformTask {
            coefficient_offset: u32::try_from(index * 192).unwrap(),
            destinations: [Some(origins[index]); 3],
            quant_index: 0,
            dequant_matrix_index: u16::try_from(index).unwrap(),
            coefficient_origin: (0, 0),
            lf_offset: u32::try_from(index).unwrap(),
        });
        let mut coefficients = vec![0i32; transforms.len() * 192];
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = ((index * 31 + 5) % 13) as i32 - 6;
        }
        let resource = VarDctResource {
            quant_biases: [1.0, 1.0, 1.0, 0.0],
            quant_scales: vec![[0.5, 1.25, 0.75]],
            dequant_matrices: transforms
                .iter()
                .map(|&transform| VarDctDequantMatrix {
                    transform,
                    scales: vec![[1.0, 0.75, 1.5]; 64],
                })
                .collect(),
            hf_correlation: VarDctCorrelationGrid {
                extent: Extent2d::new(1, 1),
                values: vec![[0.2, -0.125]],
            },
            lf_coefficients: (0..transforms.len())
                .map(|index| {
                    let value = index as f32 + 1.0;
                    [value, -0.5 * value, 0.25 * value]
                })
                .collect(),
        };
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(coefficients.clone()),
            buckets: transforms
                .iter()
                .zip(tasks)
                .map(|(&transform, task)| TransformBucket {
                    transform,
                    tasks: vec![task],
                })
                .collect(),
        };
        let plane_area = extent.area().unwrap();
        let mut expected = [
            vec![0.0; plane_area],
            vec![0.0; plane_area],
            vec![0.0; plane_area],
        ];
        for (((&transform, task), &(origin_x, origin_y)), coefficient_block) in transforms
            .iter()
            .zip(&tasks)
            .zip(&origins)
            .zip(coefficients.chunks_exact(192))
        {
            let block = reference_block(transform, task, coefficient_block, &resource);
            for channel in 0..3 {
                for y in 0..8usize {
                    for x in 0..8usize {
                        expected[channel][(origin_y as usize + y) * extent.width as usize
                            + origin_x as usize
                            + x] = block[channel][y * 8 + x];
                    }
                }
            }
        }

        let mut session = backend
            .create_session(&frame(extent), Arc::new(plan(extent)))
            .expect("create special VarDCT session");
        session
            .update_resource(ResourceUpdate {
                id: ResourceId(0),
                revision: 0,
                data: ResourceData::VarDct(resource),
            })
            .expect("supply special VarDCT resource");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: Vec::new(),
                vardct: Some(packet),
            })
            .expect("enqueue special transform packet");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit special transforms");
        let frame = session.wait(token).expect("read special transform output");
        let PlaneData::F32(actual) = &frame.outputs[0].data else {
            panic!("special VarDCT output was not F32");
        };
        for channel in 0..3 {
            for (index, (&expected, &actual)) in expected[channel]
                .iter()
                .zip(&actual[channel * plane_area..(channel + 1) * plane_area])
                .enumerate()
            {
                let tolerance = 1.0e-3_f32.max(expected.abs() * 2.0e-5);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "channel {channel} sample {index}: GPU {actual}, codec {expected}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn all_afv_variants_match_codec_oracle_on_gpu() {
        let Some(backend) = test_backend() else {
            return;
        };
        let extent = Extent2d::new(19, 19);
        let transforms = [
            TransformKind::Afv0,
            TransformKind::Afv1,
            TransformKind::Afv2,
            TransformKind::Afv3,
        ];
        let origins = [(1, 1), (10, 1), (1, 10), (10, 10)];
        let tasks = std::array::from_fn::<_, 4, _>(|index| TransformTask {
            coefficient_offset: u32::try_from(index * 192).unwrap(),
            destinations: [Some(origins[index]); 3],
            quant_index: 0,
            dequant_matrix_index: u16::try_from(index).unwrap(),
            coefficient_origin: (0, 0),
            lf_offset: u32::try_from(index).unwrap(),
        });
        let mut coefficients = vec![0i32; transforms.len() * 192];
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = ((index * 37 + 11) % 19) as i32 - 9;
        }
        let resource = VarDctResource {
            quant_biases: [1.0, 1.0, 1.0, 0.0],
            quant_scales: vec![[0.625, 1.125, 0.875]],
            dequant_matrices: transforms
                .iter()
                .map(|&transform| VarDctDequantMatrix {
                    transform,
                    scales: vec![[1.0, 0.75, 1.25]; 64],
                })
                .collect(),
            hf_correlation: VarDctCorrelationGrid {
                extent: Extent2d::new(1, 1),
                values: vec![[0.1875, -0.0625]],
            },
            lf_coefficients: (0..transforms.len())
                .map(|index| {
                    let value = index as f32 + 0.75;
                    [value, -0.375 * value, 0.5 * value]
                })
                .collect(),
        };
        let packet = VarDctPacket {
            revision: 0,
            last_pass: 0,
            coefficients: PackedCoefficients::DenseI32(coefficients.clone()),
            buckets: transforms
                .iter()
                .zip(tasks)
                .map(|(&transform, task)| TransformBucket {
                    transform,
                    tasks: vec![task],
                })
                .collect(),
        };
        let plane_area = extent.area().unwrap();
        let mut expected = [
            vec![0.0; plane_area],
            vec![0.0; plane_area],
            vec![0.0; plane_area],
        ];
        for (((&transform, task), &(origin_x, origin_y)), coefficient_block) in transforms
            .iter()
            .zip(&tasks)
            .zip(&origins)
            .zip(coefficients.chunks_exact(192))
        {
            let block = reference_block(transform, task, coefficient_block, &resource);
            for channel in 0..3 {
                for y in 0..8usize {
                    for x in 0..8usize {
                        expected[channel][(origin_y as usize + y) * extent.width as usize
                            + origin_x as usize
                            + x] = block[channel][y * 8 + x];
                    }
                }
            }
        }

        let mut session = backend
            .create_session(&frame(extent), Arc::new(plan(extent)))
            .expect("create AFV VarDCT session");
        session
            .update_resource(ResourceUpdate {
                id: ResourceId(0),
                revision: 0,
                data: ResourceData::VarDct(resource),
            })
            .expect("supply AFV VarDCT resource");
        session
            .enqueue(GroupPayload {
                group: GroupId(0),
                revision: 0,
                complete: true,
                planes: Vec::new(),
                vardct: Some(packet),
            })
            .expect("enqueue AFV transform packet");
        let token = session
            .submit(RenderIntent::Final)
            .expect("submit AFV transforms");
        let frame = session.wait(token).expect("read AFV transform output");
        let PlaneData::F32(actual) = &frame.outputs[0].data else {
            panic!("AFV VarDCT output was not F32");
        };
        for channel in 0..3 {
            for (index, (&expected, &actual)) in expected[channel]
                .iter()
                .zip(&actual[channel * plane_area..(channel + 1) * plane_area])
                .enumerate()
            {
                let tolerance = 1.0e-3_f32.max(expected.abs() * 2.0e-5);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "channel {channel} sample {index}: GPU {actual}, codec {expected}, tolerance {tolerance}"
                );
            }
        }
    }
}
