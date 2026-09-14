use super::*;

struct Case {
    profile: IccProfile,
    format: PixelFormat,
    extent: Extent2d,
    orientation: OutputOrientation,
    index: usize,
}

#[test]
fn device_components_alpha_orientation_and_padding_match_independent_words() {
    let backend = backend().expect("device output conformance requires a GPU");
    let device = backend.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ICC device packing conformance"),
        source: wgpu::ShaderSource::Wgsl(DEVICE_OUTPUT_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ICC device packing conformance"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut cases = 0;
    let mut bytes = 0;
    for (name, _) in PROFILES {
        let profile = profile(name);
        for extent in [
            Extent2d::new(17, 9),
            Extent2d::new(1, 5),
            Extent2d::new(7, 1),
        ] {
            for sample in [ColorSample::U8, ColorSample::F32] {
                for storage in [ColorStorage::Interleaved, ColorStorage::Planar] {
                    for alpha in [false, true] {
                        for code in 1..=8 {
                            let mut format =
                                PixelFormat::icc_device(profile.clone(), sample, storage, alpha)
                                    .unwrap();
                            if cases % 2 == 1 {
                                if storage == ColorStorage::Planar {
                                    format.planes.reverse();
                                } else {
                                    format.planes[0].words.reverse();
                                }
                            }
                            bytes += verify(
                                &backend,
                                &pipeline,
                                Case {
                                    profile: profile.clone(),
                                    format,
                                    extent,
                                    orientation: OutputOrientation::from_exif_value(code).unwrap(),
                                    index: cases,
                                },
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1152);
    eprintln!("ICC device packing: {cases} guarded dispatches, {bytes} checked bytes");
}

fn coordinate(x: u32, y: u32, extent: Extent2d, orientation: OutputOrientation) -> (u32, u32) {
    let w = extent.width;
    let h = extent.height;
    match orientation.to_exif_value() {
        1 => (x, y),
        2 => (w - 1 - x, y),
        3 => (w - 1 - x, h - 1 - y),
        4 => (x, h - 1 - y),
        5 => (y, x),
        6 => (h - 1 - y, x),
        7 => (h - 1 - y, w - 1 - x),
        8 => (y, w - 1 - x),
        _ => unreachable!(),
    }
}

fn verify(backend: &WgpuBackend, pipeline: &wgpu::ComputePipeline, case: Case) -> usize {
    let count = usize::from(
        case.profile
            .header()
            .device_space
            .device_channels()
            .unwrap(),
    );
    let encoding = if (case.index / 4).is_multiple_of(2) {
        ResidentIccSampleEncoding::Direct
    } else {
        ResidentIccSampleEncoding::Complement
    };
    let conversion = match case.index % 4 {
        2 => AlphaConversion::Unpremultiply,
        3 => AlphaConversion::Premultiply,
        _ => AlphaConversion::Preserve,
    };
    let mut values = vec![-12345.0_f32; 7];
    let mut append = |c: usize, alpha: bool| {
        let stride = case.extent.width + 1 + c as u32 % 3;
        let plane = ResidentIccPlane {
            offset: values.len() as u32,
            stride,
        };
        values.resize(
            values.len() + (stride * case.extent.height) as usize + 3,
            -12345.0,
        );
        for y in 0..case.extent.height {
            for x in 0..case.extent.width {
                let p = y * case.extent.width + x;
                let v = if alpha {
                    [0.0, 0.25, 0.5, 1.0][p as usize % 4]
                } else {
                    (((p as usize * 13 + c * 7) % 33) as f32 - 8.0) / 16.0
                };
                values[(plane.offset + y * stride + x) as usize] =
                    if !alpha && encoding == ResidentIccSampleEncoding::Complement {
                        1.0 - v
                    } else {
                        v
                    };
            }
        }
        plane
    };
    let alpha = (!case.index.is_multiple_of(4)).then(|| append(count, true));
    let mut planes: Vec<_> = (0..count).rev().map(|c| append(c, false)).collect();
    planes.reverse();
    let mut layout =
        ImageLayout::packed(case.orientation.map_extent(case.extent), case.format).unwrap();
    let mut offset = 3;
    for plane in &mut layout.planes {
        plane.offset = offset;
        plane.row_stride += 1 + case.index as u64 % 3;
        offset = plane.end_offset().unwrap() + 5;
    }
    layout = ImageLayout::from_planes(layout.extent, layout.format, layout.planes).unwrap();
    let source = DeviceOutputSource {
        profile: &case.profile,
        extent: case.extent,
        orientation: case.orientation,
        planes: &planes,
        alpha,
        input_words: values.len() as u64,
        sample_encoding: encoding,
        alpha_conversion: conversion,
    };
    let params = DeviceOutputParams::new(&layout, source, 64).unwrap();
    if case.index == 0 {
        let mut bad = source;
        bad.input_words = u64::from(planes.iter().map(|p| p.offset).max().unwrap());
        assert!(matches!(
            DeviceOutputParams::new(&layout, bad, 64),
            Err(jxl_wgpu::Error::InvalidPayload(_))
        ));
        assert!(matches!(
            DeviceOutputParams::new(&layout, source, 0),
            Err(jxl_wgpu::Error::InvalidPayload(_))
        ));
        let mut bad = layout.clone();
        bad.logical_size -= 1;
        assert!(matches!(
            DeviceOutputParams::new(&bad, source, 64),
            Err(jxl_wgpu::Error::InvalidPayload(_))
        ));
    }
    let size = layout.logical_size.div_ceil(4) as usize * 4;
    let mut expected = vec![0u8; size];
    let read =
        |plane: ResidentIccPlane, x, y| values[(plane.offset + y * plane.stride + x) as usize];
    for (target, format) in layout.planes.iter().zip(&layout.format.planes) {
        for (position, word) in format.words.iter().enumerate() {
            let PackingFieldKind::Channel(channel) = word.fields[0].kind else {
                unreachable!()
            };
            let bytes = usize::from(word.fields[0].bits / 8);
            for y in 0..case.extent.height {
                for x in 0..case.extent.width {
                    let a = alpha.map_or(1.0, |plane| read(plane, x, y));
                    let value = match channel {
                        Channel::Alpha => a,
                        Channel::Device(c) => {
                            let stored = read(planes[c as usize], x, y);
                            let v = if encoding == ResidentIccSampleEncoding::Complement {
                                1.0 - stored
                            } else {
                                stored
                            };
                            match conversion {
                                AlphaConversion::Preserve => v,
                                AlphaConversion::Unpremultiply => v / a.max(1.0 / 67108864.0),
                                AlphaConversion::Premultiply => v * a.max(1.0 / 67108864.0),
                            }
                        }
                        _ => unreachable!(),
                    };
                    let (dx, dy) = coordinate(x, y, case.extent, case.orientation);
                    let address = target.offset as usize
                        + dy as usize * target.row_stride as usize
                        + (dx as usize * format.words.len() + position) * bytes;
                    if bytes == 4 {
                        expected[address..address + 4].copy_from_slice(&value.to_le_bytes());
                    } else {
                        expected[address] = (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
                    }
                }
            }
        }
    }
    check_gpu(backend, pipeline, &values, &params, &expected, case.index);
    expected.len()
}

fn check_gpu(
    backend: &WgpuBackend,
    pipeline: &wgpu::ComputePipeline,
    values: &[f32],
    params: &DeviceOutputParams,
    expected: &[u8],
    case: usize,
) {
    let device = backend.device();
    let guard = device.limits().min_storage_buffer_offset_alignment.max(4) as usize;
    let input_bytes = bytemuck::cast_slice(values);
    let mut source = vec![0xa5; guard];
    source.extend_from_slice(input_bytes);
    source.resize(source.len() + guard, 0xa5);
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC device source and guards"),
        contents: &source,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC device output and guards"),
        contents: &vec![0xa5; guard * 2 + expected.len()],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC device output parameters"),
        contents: bytemuck::bytes_of(params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ICC device packing conformance"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &input,
                    offset: guard as u64,
                    size: NonZeroU64::new(input_bytes.len() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &output,
                    offset: guard as u64,
                    size: NonZeroU64::new(expected.len() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: uniform.as_entire_binding(),
            },
        ],
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICC device packing readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, (expected.len() as u32 / 4).div_ceil(64), 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    let submission = backend.queue().submit([encoder.finish()]);
    let (send, receive) = mpsc::sync_channel(1);
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = send.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receive.recv().unwrap().unwrap();
    let bytes = readback.slice(..).get_mapped_range().unwrap();
    assert!(
        bytes[..guard].iter().all(|&b| b == 0xa5),
        "case {case} prefix guard"
    );
    assert!(
        bytes[guard + expected.len()..].iter().all(|&b| b == 0xa5),
        "case {case} suffix guard"
    );
    if let Some((offset, (&actual, &expected))) = bytes[guard..guard + expected.len()]
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (a, e))| a != e)
    {
        panic!("case {case}, byte {offset}: GPU {actual:#04x}, expected {expected:#04x}");
    }
    drop(bytes);
    readback.unmap();
}
