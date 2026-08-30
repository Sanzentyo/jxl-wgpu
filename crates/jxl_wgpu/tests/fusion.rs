use std::collections::BTreeSet;
use std::sync::Arc;

use jxl_gpu_protocol::{
    Border2d, ChromaAxis, Extent2d, FrameSessionDesc, GroupId, GroupPayload, HostPlane, MemoryMode,
    OutputDesc, OutputId, OutputLayout, PlaneData, PlaneDesc, PlaneId, PlaneRole,
    PrecisionContract, PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderOpKind,
    RenderPlan, SampleType, SaveParams, Scale2d,
};
use jxl_wgpu::{
    Error, FusedKernel, KernelVariant, Planner, Result, WgpuBackend, WgpuBackendConfig,
    WgpuMemoryPolicy,
};

fn backend() -> Result<Option<WgpuBackend>> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..WgpuBackendConfig::default()
    })) {
        Ok(backend) => Ok(Some(backend)),
        Err(Error::NoAdapter) => Ok(None),
        Err(error) => Err(error),
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

fn plane(id: u32, extent: Extent2d, role: PlaneRole) -> PlaneDesc {
    PlaneDesc {
        id: PlaneId(id),
        extent,
        stride: extent.width,
        sample_type: SampleType::F32,
        role,
    }
}

fn node(name: &'static str, op: RenderOp, inputs: &[u32], outputs: &[u32]) -> RenderNode {
    RenderNode {
        name: name.into(),
        op,
        inputs: inputs.iter().copied().map(PlaneId).collect(),
        outputs: outputs.iter().copied().map(PlaneId).collect(),
        resources: Vec::new(),
        scale: Scale2d::IDENTITY,
        border: Border2d::default(),
        precision: PrecisionContract::default(),
    }
}

fn mirror_coordinate(mut value: i32, size: u32) -> u32 {
    if size <= 1 {
        return 0;
    }
    loop {
        if value < 0 {
            value = -value - 1;
        } else if value >= size as i32 {
            value = size as i32 * 2 - value - 1;
        } else {
            return value as u32;
        }
    }
}

fn chroma_axis(
    input: &[f32],
    input_extent: Extent2d,
    output_extent: Extent2d,
    axis: ChromaAxis,
) -> Vec<f32> {
    let mut output = vec![0.0; output_extent.width as usize * output_extent.height as usize];
    for y in 0..output_extent.height {
        for x in 0..output_extent.width {
            let mut source_x = x as i32;
            let mut source_y = y as i32;
            let mut neighbor_x = source_x;
            let mut neighbor_y = source_y;
            match axis {
                ChromaAxis::Horizontal => {
                    source_x = (x / 2) as i32;
                    neighbor_x = source_x + if x & 1 == 0 { -1 } else { 1 };
                }
                ChromaAxis::Vertical => {
                    source_y = (y / 2) as i32;
                    neighbor_y = source_y + if y & 1 == 0 { -1 } else { 1 };
                }
            }
            let sample = |sample_x: i32, sample_y: i32| {
                let sample_x = mirror_coordinate(sample_x, input_extent.width) as usize;
                let sample_y = mirror_coordinate(sample_y, input_extent.height) as usize;
                input[sample_y * input_extent.width as usize + sample_x]
            };
            let current = sample(source_x, source_y);
            let neighbor = sample(neighbor_x, neighbor_y);
            output[y as usize * output_extent.width as usize + x as usize] =
                neighbor.mul_add(0.25, current * 0.75);
        }
    }
    output
}

#[test]
fn fused_chroma_2d_matches_separable_scalar_on_odd_mirror_edges() -> Result<()> {
    let Some(backend) = backend()? else {
        eprintln!("skipping fusion test: no wgpu adapter is available");
        return Ok(());
    };
    let input_extent = Extent2d::new(5, 4);
    let horizontal_extent = Extent2d::new(9, 4);
    let output_extent = Extent2d::new(9, 7);
    let output_id = OutputId(0);
    let mut horizontal = node(
        "horizontal chroma",
        RenderOp::ChromaUpsample {
            axis: ChromaAxis::Horizontal,
        },
        &[0],
        &[1],
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
    );
    vertical.scale = Scale2d::new(1, 2);
    vertical.border = Border2d::symmetric(0, 1);
    let plan = Arc::new(RenderPlan {
        planes: vec![
            plane(0, input_extent, PlaneRole::Source),
            plane(1, horizontal_extent, PlaneRole::Intermediate),
            plane(2, output_extent, PlaneRole::Intermediate),
        ],
        nodes: vec![
            horizontal,
            vertical,
            node(
                "save chroma",
                RenderOp::Save(SaveParams {
                    output: output_id,
                    sample_type: SampleType::F32,
                    channels: vec![PlaneId(2)],
                    layout: OutputLayout::Planar,
                    orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                }),
                &[2],
                &[],
            ),
        ],
        outputs: vec![OutputDesc {
            id: output_id,
            extent: output_extent,
            sample_type: SampleType::F32,
            channels: 1,
            layout: OutputLayout::Planar,
            color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
        }],
    });
    let frame = frame(output_extent);
    let execution =
        Planner::new(backend.device().limits(), WgpuMemoryPolicy::default()).plan(&frame, &plan)?;
    assert_eq!(execution.dispatches.len(), 2);
    assert_eq!(execution.dispatches[0].kernel, FusedKernel::Chroma2d);
    assert_eq!(execution.dispatches[0].node_indices, [0, 1]);
    assert_eq!(execution.dispatches[0].variant, KernelVariant::Tile16x16);
    assert_eq!(execution.dispatches[0].workgroup_size, (16, 16));
    assert_eq!(execution.dispatches[0].workgroups, (1, 1, 1));
    let fused_slots = [PlaneId(0), PlaneId(1), PlaneId(2)]
        .map(|plane| execution.arena.allocation(plane).unwrap().offset)
        .into_iter()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        fused_slots.len(),
        3,
        "all planes visible to one fused dispatch need independent wgpu buffers"
    );
    assert_eq!(
        execution.dispatches[1].kernel,
        FusedKernel::Single(RenderOpKind::Save)
    );

    let source = (0..input_extent.width * input_extent.height)
        .map(|index| ((index * 29 + 7) % 37) as f32 / 11.0 - 1.25)
        .collect::<Vec<_>>();
    let horizontal_reference = chroma_axis(
        &source,
        input_extent,
        horizontal_extent,
        ChromaAxis::Horizontal,
    );
    let reference = chroma_axis(
        &horizontal_reference,
        horizontal_extent,
        output_extent,
        ChromaAxis::Vertical,
    );

    let mut session = backend.create_session(&frame, plan)?;
    session.enqueue(GroupPayload {
        group: GroupId(0),
        revision: 0,
        complete: true,
        planes: vec![HostPlane {
            id: PlaneId(0),
            extent: input_extent,
            stride: input_extent.width,
            origin: (0, 0),
            data: PlaneData::F32(source),
        }],
        vardct: None,
    })?;
    let token = session.submit(RenderIntent::Final)?;
    let stats = session.last_submission_stats().unwrap();
    assert_eq!(
        (
            stats.planned_dispatches,
            stats.compute_dispatches,
            stats.fused_dispatches,
            stats.direct_readback,
        ),
        (
            2,
            2,
            1,
            backend
                .device()
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        )
    );
    assert!(stats.resident_bytes > 0);
    assert!(stats.transient_bytes > 0);
    let rendered = session.wait(token)?;
    let PlaneData::F32(actual) = &rendered.outputs[0].data else {
        panic!("fused chroma output is not F32");
    };
    for (index, (&actual, &expected)) in actual.iter().zip(&reference).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "chroma sample {index}: GPU {actual}, scalar {expected}"
        );
    }
    Ok(())
}

fn gaborish_reference(input: &[f32], extent: Extent2d, weights: [f32; 3]) -> Vec<f32> {
    let sample = |x: i32, y: i32| {
        let x = mirror_coordinate(x, extent.width) as usize;
        let y = mirror_coordinate(y, extent.height) as usize;
        input[y * extent.width as usize + x]
    };
    let mut output = vec![0.0; input.len()];
    for y in 0..extent.height {
        for x in 0..extent.width {
            let x = x as i32;
            let y = y as i32;
            let center = sample(x, y) * weights[0];
            let axial = sample(x, y - 1) + sample(x - 1, y) + sample(x, y + 1) + sample(x + 1, y);
            let diagonal = sample(x - 1, y - 1)
                + sample(x + 1, y - 1)
                + sample(x - 1, y + 1)
                + sample(x + 1, y + 1);
            output[y as usize * extent.width as usize + x as usize] =
                diagonal.mul_add(weights[2], axial.mul_add(weights[1], center));
        }
    }
    output
}

#[test]
fn fused_gaborish_rgb_matches_scalar_with_channel_specific_weights() -> Result<()> {
    let Some(backend) = backend()? else {
        eprintln!("skipping fusion test: no wgpu adapter is available");
        return Ok(());
    };
    let extent = Extent2d::new(19, 11);
    let weights = [
        [0.55, 0.08, 0.0325],
        [0.61, 0.065, 0.0325],
        [0.49, 0.095, 0.0325],
    ];
    let output_id = OutputId(0);
    let mut nodes = Vec::new();
    for channel in 0..3_u16 {
        let mut gaborish = node(
            "gaborish",
            RenderOp::Gaborish(jxl_gpu_protocol::GaborishParams {
                channel,
                weight0: weights[channel as usize][0],
                weight1: weights[channel as usize][1],
                weight2: weights[channel as usize][2],
            }),
            &[u32::from(channel)],
            &[u32::from(channel) + 3],
        );
        gaborish.border = Border2d::symmetric(1, 1);
        nodes.push(gaborish);
    }
    nodes.push(node(
        "save RGB",
        RenderOp::Save(SaveParams {
            output: output_id,
            sample_type: SampleType::F32,
            channels: vec![PlaneId(3), PlaneId(4), PlaneId(5)],
            layout: OutputLayout::Planar,
            orientation: jxl_gpu_protocol::OutputOrientation::Identity,
        }),
        &[3, 4, 5],
        &[],
    ));
    let plan = Arc::new(RenderPlan {
        planes: (0..6)
            .map(|id| {
                plane(
                    id,
                    extent,
                    if id < 3 {
                        PlaneRole::Source
                    } else {
                        PlaneRole::Intermediate
                    },
                )
            })
            .collect(),
        nodes,
        outputs: vec![OutputDesc {
            id: output_id,
            extent,
            sample_type: SampleType::F32,
            channels: 3,
            layout: OutputLayout::Planar,
            color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
        }],
    });
    let frame = frame(extent);
    let execution =
        Planner::new(backend.device().limits(), WgpuMemoryPolicy::default()).plan(&frame, &plan)?;
    assert_eq!(execution.dispatches.len(), 2);
    assert_eq!(execution.dispatches[0].kernel, FusedKernel::GaborishRgb);
    assert_eq!(execution.dispatches[0].node_indices, [0, 1, 2]);
    assert_eq!(execution.dispatches[0].variant, KernelVariant::Tile16x16);
    assert_eq!(execution.dispatches[0].workgroup_size, (16, 16));
    assert_eq!(execution.dispatches[0].workgroups, (2, 1, 1));
    let fused_slots = (0..6)
        .map(|plane| execution.arena.allocation(PlaneId(plane)).unwrap().offset)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        fused_slots.len(),
        6,
        "all six RGB bindings need independent wgpu buffers"
    );

    let sources = (0..3_u32)
        .map(|channel| {
            (0..extent.width * extent.height)
                .map(|index| ((index * (17 + channel * 4) + channel * 13) % 101) as f32 / 29.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reference = sources
        .iter()
        .zip(weights)
        .flat_map(|(source, weights)| gaborish_reference(source, extent, weights))
        .collect::<Vec<_>>();
    let mut session = backend.create_session(&frame, plan)?;
    session.enqueue(GroupPayload {
        group: GroupId(0),
        revision: 0,
        complete: true,
        planes: sources
            .iter()
            .enumerate()
            .map(|(channel, values)| HostPlane {
                id: PlaneId(channel as u32),
                extent,
                stride: extent.width,
                origin: (0, 0),
                data: PlaneData::F32(values.clone()),
            })
            .collect(),
        vardct: None,
    })?;
    let token = session.submit(RenderIntent::Final)?;
    let stats = session.last_submission_stats().unwrap();
    assert_eq!(
        (
            stats.planned_dispatches,
            stats.compute_dispatches,
            stats.fused_dispatches,
            stats.direct_readback,
        ),
        (
            2,
            4,
            1,
            backend
                .device()
                .features()
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        )
    );
    assert!(stats.resident_bytes > 0);
    assert!(stats.transient_bytes > 0);
    let rendered = session.wait(token)?;
    let PlaneData::F32(actual) = &rendered.outputs[0].data else {
        panic!("fused Gaborish output is not F32");
    };
    for (index, (&actual, &expected)) in actual.iter().zip(&reference).enumerate() {
        let tolerance = 1.0e-6_f32.max(expected.abs() * 2.0e-6);
        assert!(
            (actual - expected).abs() <= tolerance,
            "Gaborish sample {index}: GPU {actual}, scalar {expected}, tolerance {tolerance}"
        );
    }
    Ok(())
}
