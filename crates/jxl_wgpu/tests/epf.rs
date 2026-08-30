use std::sync::Arc;

use jxl_gpu_protocol::{
    Border2d, EpfParams, EpfPass, Extent2d, FrameSessionDesc, GroupId, GroupPayload, HostPlane,
    MemoryMode, OutputDesc, OutputId, OutputLayout, PlaneData, PlaneDesc, PlaneId, PlaneRole,
    PrecisionContract, PrecisionPolicy, RenderIntent, RenderNode, RenderOp, RenderPlan,
    ResourceData, ResourceId, ResourceUpdate, SampleType, SaveParams, Scale2d,
};
use jxl_wgpu::{Error, WgpuBackend, WgpuBackendConfig};

const WIDTH: u32 = 19;
const HEIGHT: u32 = 11;
const MIN_SIGMA: f32 = -3.905_243;
const CHANNEL_SCALE: [f32; 3] = [40.0, 5.0, 3.5];
const BORDER_SAD_MUL: f32 = 2.3 / 3.0;

#[derive(Clone, Copy)]
enum SigmaMode {
    Constant(f32),
    Variable,
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

fn pass_parameters(pass: EpfPass) -> (f32, u16) {
    match pass {
        EpfPass::Pass0 => (0.9, 3),
        EpfPass::Pass1 => (1.0, 2),
        EpfPass::Pass2 => (6.5, 1),
    }
}

fn source_channels() -> [Vec<f32>; 3] {
    std::array::from_fn(|channel| {
        (0..HEIGHT)
            .flat_map(|y| {
                (0..WIDTH).map(move |x| {
                    let mixed = (x * 37 + y * 17 + channel as u32 * 53 + x * y * 3) % 101;
                    (mixed as f32 - 50.0) / 31.0 + x as f32 * 0.013 - y as f32 * 0.021
                })
            })
            .collect()
    })
}

fn sigma_extent() -> Extent2d {
    Extent2d::new(WIDTH.div_ceil(8) + 2, HEIGHT.div_ceil(8))
}

fn variable_sigma() -> Vec<f32> {
    let extent = sigma_extent();
    (0..extent.height)
        .flat_map(|y| {
            (0..extent.width).map(move |x| match (x + y * extent.width) % 5 {
                0 => -0.35,
                1 => -4.25,
                2 => -1.15,
                3 => -0.72,
                _ => MIN_SIGMA,
            })
        })
        .collect()
}

fn plane_desc(id: u32, role: PlaneRole, stride: u32) -> PlaneDesc {
    PlaneDesc {
        id: PlaneId(id),
        extent: Extent2d::new(WIDTH, HEIGHT),
        stride,
        sample_type: SampleType::F32,
        role,
    }
}

fn epf_plan(pass: EpfPass, mode: SigmaMode) -> Arc<RenderPlan> {
    let (_, border) = pass_parameters(pass);
    let sigma_plane = matches!(mode, SigmaMode::Variable).then_some(PlaneId(6));
    let mut planes = vec![
        plane_desc(0, PlaneRole::Source, WIDTH + 1),
        plane_desc(1, PlaneRole::Source, WIDTH + 2),
        plane_desc(2, PlaneRole::Source, WIDTH + 3),
        plane_desc(3, PlaneRole::Intermediate, WIDTH + 4),
        plane_desc(4, PlaneRole::Intermediate, WIDTH + 5),
        plane_desc(5, PlaneRole::Intermediate, WIDTH + 6),
    ];
    if sigma_plane.is_some() {
        let extent = sigma_extent();
        planes.push(PlaneDesc {
            id: PlaneId(6),
            extent,
            stride: extent.width,
            sample_type: SampleType::F32,
            role: PlaneRole::Parameter,
        });
    }
    let resource = ResourceId(0);
    Arc::new(RenderPlan {
        planes,
        nodes: vec![
            RenderNode {
                name: "EPF reference comparison".into(),
                op: RenderOp::Epf(EpfParams {
                    pass,
                    sigma_scale: pass_parameters(pass).0,
                    border_sad_mul: BORDER_SAD_MUL,
                    channel_scale: CHANNEL_SCALE,
                    sigma_resource: Some(resource),
                    sigma_plane,
                }),
                inputs: vec![PlaneId(0), PlaneId(1), PlaneId(2)],
                outputs: vec![PlaneId(3), PlaneId(4), PlaneId(5)],
                resources: vec![resource],
                scale: Scale2d::IDENTITY,
                border: Border2d::symmetric(border, border),
                precision: PrecisionContract::Float {
                    absolute: 2.0e-5,
                    relative: 2.0e-5,
                    rmse: 2.0e-6,
                },
            },
            RenderNode {
                name: "save EPF output".into(),
                op: RenderOp::Save(SaveParams {
                    output: OutputId(0),
                    sample_type: SampleType::F32,
                    channels: vec![PlaneId(3), PlaneId(4), PlaneId(5)],
                    layout: OutputLayout::Planar,
                    orientation: jxl_gpu_protocol::OutputOrientation::Identity,
                }),
                inputs: vec![PlaneId(3), PlaneId(4), PlaneId(5)],
                outputs: Vec::new(),
                resources: Vec::new(),
                scale: Scale2d::IDENTITY,
                border: Border2d::default(),
                precision: PrecisionContract::Exact,
            },
        ],
        outputs: vec![OutputDesc {
            id: OutputId(0),
            extent: Extent2d::new(WIDTH, HEIGHT),
            sample_type: SampleType::F32,
            channels: 3,
            layout: OutputLayout::Planar,
            color_encoding: jxl_gpu_protocol::OutputColorEncoding::NonColor,
        }],
    })
}

fn frame_desc() -> FrameSessionDesc {
    FrameSessionDesc {
        frame_extent: Extent2d::new(WIDTH, HEIGHT),
        group_extent: Extent2d::new(WIDTH, HEIGHT),
        group_count: 1,
        precision: PrecisionPolicy::F32Only,
        memory_mode: MemoryMode::Resident,
        max_resident_bytes: 16 * 1024 * 1024,
        max_scratch_bytes: 16 * 1024 * 1024,
    }
}

fn padded(values: &[f32], stride: u32) -> Vec<f32> {
    let mut padded = vec![9_876.0; stride as usize * HEIGHT as usize];
    for y in 0..HEIGHT as usize {
        let source = &values[y * WIDTH as usize..(y + 1) * WIDTH as usize];
        let start = y * stride as usize;
        padded[start..start + WIDTH as usize].copy_from_slice(source);
    }
    padded
}

fn group_payload(channels: &[Vec<f32>; 3]) -> GroupPayload {
    GroupPayload {
        group: GroupId(0),
        revision: 0,
        complete: true,
        planes: channels
            .iter()
            .enumerate()
            .map(|(channel, values)| {
                let stride = WIDTH + channel as u32 + 1;
                HostPlane {
                    id: PlaneId(channel as u32),
                    extent: Extent2d::new(WIDTH, HEIGHT),
                    stride,
                    origin: (0, 0),
                    data: PlaneData::F32(padded(values, stride)),
                }
            })
            .collect(),
        vardct: None,
    }
}

fn sigma_update(mode: SigmaMode) -> ResourceUpdate {
    let data = match mode {
        SigmaMode::Constant(value) => ResourceData::F32(vec![value]),
        SigmaMode::Variable => {
            let extent = sigma_extent();
            ResourceData::Plane(HostPlane {
                id: PlaneId(6),
                extent,
                stride: extent.width,
                origin: (0, 0),
                data: PlaneData::F32(variable_sigma()),
            })
        }
    };
    ResourceUpdate {
        id: ResourceId(0),
        revision: 0,
        data,
    }
}

fn run_gpu(
    backend: &WgpuBackend,
    pass: EpfPass,
    mode: SigmaMode,
    channels: &[Vec<f32>; 3],
) -> Result<[Vec<f32>; 3], Error> {
    let mut session = backend.create_session(&frame_desc(), epf_plan(pass, mode))?;
    session.update_resource(sigma_update(mode))?;
    session.enqueue(group_payload(channels))?;
    let token = session.submit(RenderIntent::Final)?;
    let frame = session.wait(token)?;
    let output = frame
        .outputs
        .first()
        .ok_or_else(|| Error::Execution("EPF test produced no output".into()))?;
    let PlaneData::F32(values) = &output.data else {
        return Err(Error::Execution("EPF test output is not F32".into()));
    };
    let area = WIDTH as usize * HEIGHT as usize;
    if values.len() != area * 3 {
        return Err(Error::Execution(format!(
            "EPF test produced {} samples, expected {}",
            values.len(),
            area * 3
        )));
    }
    Ok(std::array::from_fn(|channel| {
        values[channel * area..(channel + 1) * area].to_vec()
    }))
}

fn sample(channels: &[Vec<f32>; 3], channel: usize, x: i32, y: i32) -> f32 {
    let x = mirror_coordinate(x, WIDTH) as usize;
    let y = mirror_coordinate(y, HEIGHT) as usize;
    channels[channel][y * WIDTH as usize + x]
}

fn mirror_coordinate(mut value: i32, size: u32) -> u32 {
    if size <= 1 {
        return 0;
    }
    let size = size as i32;
    loop {
        if value < 0 {
            value = -value - 1;
        } else if value >= size {
            value = size * 2 - value - 1;
        } else {
            return value as u32;
        }
    }
}

fn sigma_at(mode: SigmaMode, x: u32, y: u32) -> f32 {
    match mode {
        SigmaMode::Constant(value) => value,
        SigmaMode::Variable => {
            let extent = sigma_extent();
            variable_sigma()[(y / 8 * extent.width + x / 8) as usize]
        }
    }
}

fn plus_sad(channels: &[Vec<f32>; 3], x: i32, y: i32, offset: (i32, i32)) -> f32 {
    const PLUS: [(i32, i32); 5] = [(0, -1), (-1, 0), (0, 0), (1, 0), (0, 1)];
    channels.iter().enumerate().fold(0.0, |acc, (channel, _)| {
        let sad = PLUS.iter().fold(0.0, |sad, (plus_x, plus_y)| {
            let center = sample(channels, channel, x + plus_x, y + plus_y);
            let candidate = sample(
                channels,
                channel,
                x + offset.0 + plus_x,
                y + offset.1 + plus_y,
            );
            sad + (candidate - center).abs()
        });
        sad.mul_add(CHANNEL_SCALE[channel], acc)
    })
}

fn reference_epf(channels: &[Vec<f32>; 3], pass: EpfPass, mode: SigmaMode) -> [Vec<f32>; 3] {
    let (sigma_scale, _) = pass_parameters(pass);
    let offsets: &[(i32, i32)] = match pass {
        EpfPass::Pass0 => &[
            (0, -2),
            (-1, -1),
            (0, -1),
            (1, -1),
            (-2, 0),
            (-1, 0),
            (1, 0),
            (2, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (0, 2),
        ],
        EpfPass::Pass1 => &[(0, -1), (-1, 0), (1, 0), (0, 1)],
        EpfPass::Pass2 => &[(0, -1), (-1, 0), (1, 0), (0, 1)],
    };
    let mut output: [Vec<f32>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(WIDTH as usize * HEIGHT as usize));
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let sigma = sigma_at(mode, x, y);
            if MIN_SIGMA > sigma {
                for (channel, destination) in output.iter_mut().enumerate() {
                    destination.push(sample(channels, channel, x as i32, y as i32));
                }
                continue;
            }
            let sm = sigma_scale * 1.65;
            let border = matches!(x % 8, 0 | 7) || matches!(y % 8, 0 | 7);
            let sad_mul = if border { sm * BORDER_SAD_MUL } else { sm };
            let inverse_sigma = sigma * sad_mul;
            let weights = if pass == EpfPass::Pass2 {
                offsets
                    .iter()
                    .map(|offset| {
                        let sad = (sample(channels, 0, x as i32 + offset.0, y as i32 + offset.1)
                            - sample(channels, 0, x as i32, y as i32))
                        .abs()
                        .mul_add(
                            CHANNEL_SCALE[0],
                            (sample(channels, 1, x as i32 + offset.0, y as i32 + offset.1)
                                - sample(channels, 1, x as i32, y as i32))
                            .abs()
                            .mul_add(
                                CHANNEL_SCALE[1],
                                (sample(channels, 2, x as i32 + offset.0, y as i32 + offset.1)
                                    - sample(channels, 2, x as i32, y as i32))
                                .abs()
                                    * CHANNEL_SCALE[2],
                            ),
                        );
                        sad.mul_add(inverse_sigma, 1.0).max(0.0)
                    })
                    .collect::<Vec<_>>()
            } else {
                offsets
                    .iter()
                    .map(|offset| {
                        plus_sad(channels, x as i32, y as i32, *offset)
                            .mul_add(inverse_sigma, 1.0)
                            .max(0.0)
                    })
                    .collect::<Vec<_>>()
            };
            let weight_sum = weights.iter().fold(1.0, |sum, weight| sum + weight);
            for (channel, destination) in output.iter_mut().enumerate() {
                let center = sample(channels, channel, x as i32, y as i32);
                let accumulate = |value: f32, (offset, weight): (&(i32, i32), &f32)| {
                    sample(channels, channel, x as i32 + offset.0, y as i32 + offset.1)
                        .mul_add(*weight, value)
                };
                let value = if pass == EpfPass::Pass2 {
                    offsets.iter().zip(&weights).fold(center, accumulate)
                } else {
                    offsets.iter().zip(&weights).rev().fold(center, accumulate)
                };
                destination.push(value / weight_sum);
            }
        }
    }
    output
}

#[test]
fn all_epf_passes_match_scalar_reference_for_constant_and_variable_sigma() {
    let Some(backend) = test_backend() else {
        return;
    };
    #[cfg(target_os = "macos")]
    assert_eq!(backend.adapter_info().backend, wgpu::Backend::Metal);
    let channels = source_channels();
    for pass in [EpfPass::Pass0, EpfPass::Pass1, EpfPass::Pass2] {
        for mode in [SigmaMode::Constant(-0.58), SigmaMode::Variable] {
            let expected = reference_epf(&channels, pass, mode);
            let actual = run_gpu(&backend, pass, mode, &channels)
                .unwrap_or_else(|error| panic!("{pass:?} GPU execution failed: {error}"));
            let mut maximum_error = 0.0f32;
            let mut squared_error = 0.0f64;
            for (expected_channel, actual_channel) in expected.iter().zip(&actual) {
                for (&expected, &actual) in expected_channel.iter().zip(actual_channel) {
                    let error = (actual - expected).abs();
                    maximum_error = maximum_error.max(error);
                    squared_error += f64::from(error) * f64::from(error);
                }
            }
            let samples = (WIDTH * HEIGHT * 3) as f64;
            let rmse = (squared_error / samples).sqrt();
            assert!(
                maximum_error <= 2.0e-5 && rmse <= 2.0e-6,
                "{pass:?} {} sigma differs from scalar reference: max={maximum_error:e}, rmse={rmse:e}",
                if matches!(mode, SigmaMode::Variable) {
                    "variable"
                } else {
                    "constant"
                }
            );
        }
    }
}

#[test]
fn malformed_epf_resources_return_typed_errors() {
    let Some(backend) = test_backend() else {
        return;
    };
    let channels = source_channels();

    let mut missing = backend
        .create_session(
            &frame_desc(),
            epf_plan(EpfPass::Pass2, SigmaMode::Constant(-0.5)),
        )
        .expect("create missing-resource test session");
    missing
        .enqueue(group_payload(&channels))
        .expect("enqueue missing-resource source planes");
    assert!(matches!(
        missing.submit(RenderIntent::Final),
        Err(Error::InvalidPayload(message)) if message.contains("missing")
    ));

    let mut malformed = backend
        .create_session(
            &frame_desc(),
            epf_plan(EpfPass::Pass1, SigmaMode::Constant(-0.5)),
        )
        .expect("create malformed-resource test session");
    malformed
        .update_resource(ResourceUpdate {
            id: ResourceId(0),
            revision: 0,
            data: ResourceData::F32(vec![-0.5, -0.6]),
        })
        .expect("resource revisions accept opaque data before operation validation");
    malformed
        .enqueue(group_payload(&channels))
        .expect("enqueue malformed-resource source planes");
    assert!(matches!(
        malformed.submit(RenderIntent::Final),
        Err(Error::InvalidPayload(message)) if message.contains("exactly one")
    ));

    let mut undersized_plan = Arc::unwrap_or_clone(epf_plan(EpfPass::Pass0, SigmaMode::Variable));
    let sigma = undersized_plan
        .planes
        .iter_mut()
        .find(|plane| plane.id == PlaneId(6))
        .expect("variable plan declares sigma plane");
    sigma.extent = Extent2d::new(1, 1);
    sigma.stride = 1;
    assert!(matches!(
        backend.create_session(&frame_desc(), Arc::new(undersized_plan)),
        Err(Error::InvalidPayload(message)) if message.contains("covering at least")
    ));
}
