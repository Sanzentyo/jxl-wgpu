use super::*;

struct Case {
    transform: TransformKind,
    test: u32,
    values: Vec<f32>,
}

fn cases() -> Vec<Case> {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/forward_vardct.bin"
    ))
    .expect("committed native oracle");
    assert_eq!(&bytes[..8], b"JXLFWD01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    let count = words.next().unwrap();
    assert_eq!(count, 667);
    let cases = (0..count)
        .map(|_| {
            let transform = TransformKind::ALL[words.next().unwrap() as usize];
            let test = words.next().unwrap();
            let extent = transform.pixel_extent();
            assert_eq!(words.next(), Some(extent.width));
            assert_eq!(words.next(), Some(extent.height));
            let size = 3 * (extent.area().unwrap() + transform.lf_extent().area().unwrap());
            let values = words
                .by_ref()
                .take(size)
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), size);
            Case {
                transform,
                test,
                values,
            }
        })
        .collect();
    assert!(words.next().is_none());
    assert!((bytes.len() - 8).is_multiple_of(4));
    cases
}

fn sample(case: &Case, channel: usize, x: usize, y: usize) -> f32 {
    let extent = case.transform.pixel_extent();
    if case.test != 0 {
        let position = case.test as usize - 1;
        let (target, amplitude) = match channel {
            0 => (position, 1.0),
            1 => ((position + 17) % 64, -0.5),
            _ => (63 - position, 0.25),
        };
        return if y * extent.width as usize + x == target {
            amplitude
        } else {
            0.0
        };
    }
    match channel {
        1 => 0.375,
        2 => {
            if x == extent.width as usize - 1 && y == extent.height as usize / 2 {
                -0.75
            } else {
                0.0
            }
        }
        _ => (((x * 37 + y * 101 + x * y * 3) % 509) as i32 - 254) as f32 / 256.0,
    }
}

const POISON: u32 = 0x7fc0_0bad;
const PREFIX: u64 = 256;
const TAIL: u64 = 16;

fn source_buffers(device: &wgpu::Device, case: &Case) -> [wgpu::Buffer; 3] {
    let extent = case.transform.pixel_extent();
    std::array::from_fn(|channel| {
        let stride = extent.width as usize + channel + 3;
        let origin = 3 + channel * 2;
        let mut values =
            vec![POISON; PREFIX as usize / 4 + origin + stride * extent.height as usize + 4];
        for y in 0..extent.height as usize {
            for x in 0..extent.width as usize {
                values[PREFIX as usize / 4 + origin + y * stride + x] =
                    sample(case, channel, x, y).to_bits();
            }
        }
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("forward oracle strided source with poisoned padding"),
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE,
        })
    })
}

fn output_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("forward oracle guarded output"),
        contents: bytemuck::cast_slice(&vec![POISON; (PREFIX + size + TAIL) as usize / 4]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn binding(buffer: &wgpu::Buffer) -> ResidentStorageBinding<'_> {
    ResidentStorageBinding {
        buffer,
        offset: PREFIX,
        size: std::num::NonZeroU64::new(buffer.size() - PREFIX).unwrap(),
    }
}

fn inputs<'a>(
    case: &Case,
    sources: &'a [wgpu::Buffer; 3],
    coefficients: &'a wgpu::Buffer,
    low_frequency: &'a wgpu::Buffer,
) -> ForwardVarDctInputs<'a> {
    let extent = case.transform.pixel_extent();
    ForwardVarDctInputs {
        transform: case.transform,
        sources: std::array::from_fn(|channel| ResidentF32Plane {
            storage: binding(&sources[channel]),
            width: extent.width,
            height: extent.height,
            stride: extent.width + channel as u32 + 3,
        }),
        origins: [3, 5, 7],
        coefficients: binding(coefficients),
        low_frequency: binding(low_frequency),
    }
}

fn run(backend: &crate::WgpuBackend, pipeline: &ForwardVarDctPipeline, case: &Case) -> Vec<f32> {
    let device = backend.device();
    let memory = ForwardVarDctMemoryPlan::new(case.transform);
    let sources = source_buffers(device, case);
    let coefficients = output_buffer(device, memory.coefficient_bytes);
    let lf = output_buffer(device, memory.lf_bytes);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forward oracle explicit readback"),
        size: coefficients.size() + lf.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = device.create_command_encoder(&Default::default());
    let scratch = pipeline
        .encode(
            device,
            &mut commands,
            inputs(case, &sources, &coefficients, &lf),
        )
        .unwrap();
    assert_eq!(scratch.memory, memory);
    assert_eq!(scratch.parameters.size(), memory.parameter_bytes);
    assert_eq!(scratch.basis.size(), memory.basis_bytes);
    assert_eq!(
        scratch.horizontal.as_ref().map_or(0, wgpu::Buffer::size),
        memory.horizontal_bytes
    );
    commands.copy_buffer_to_buffer(&coefficients, 0, &staging, 0, coefficients.size());
    commands.copy_buffer_to_buffer(&lf, 0, &staging, coefficients.size(), lf.size());
    let submission = backend.queue().submit([commands.finish()]);
    let (send, recv) = std::sync::mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    recv.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    let mut values = Vec::new();
    for (start, size) in [
        (0, memory.coefficient_bytes),
        (coefficients.size(), memory.lf_bytes),
    ] {
        let prefix = (start / 4) as usize;
        let first = ((start + PREFIX) / 4) as usize;
        let end = ((start + PREFIX + size) / 4) as usize;
        assert!(words[prefix..first].iter().all(|word| *word == POISON));
        assert!(
            words[end..end + TAIL as usize / 4]
                .iter()
                .all(|word| *word == POISON)
        );
        values.extend(words[first..end].iter().map(|word| f32::from_bits(*word)));
    }
    values
}

fn backend() -> crate::WgpuBackend {
    let backend = pollster::block_on(crate::WgpuBackend::request_default(
        crate::WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    ))
    .expect("forward VarDCT proof requires an actual GPU adapter");
    eprintln!("forward VarDCT adapter: {:?}", backend.adapter_info());
    backend
}

#[test]
fn gpu_all_transforms_match_native_coefficients_and_lf_for_every_linear_variant() {
    let backend = backend();
    let cases = cases();
    let mut baseline = Vec::new();
    // Regression tolerances fixed before executing the new kernels. These do
    // not constitute the JPEG XL end-to-end precision or quality contract.
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let pipeline = ForwardVarDctPipeline::new(backend.device(), variant).unwrap();
        let mut peak_coefficient = 0.0_f32;
        let mut peak_lf = 0.0_f32;
        for (index, case) in cases.iter().enumerate() {
            let actual = run(&backend, &pipeline, case);
            let area = case.transform.pixel_extent().area().unwrap();
            for (position, (&value, &reference)) in actual.iter().zip(&case.values).enumerate() {
                let error = (value - reference).abs();
                let limit = if position < 3 * area { 2e-6 } else { 2e-5 };
                assert!(
                    value.is_finite() && error <= limit * (1.0 + reference.abs()),
                    "{:?}/{}, {variant:?}, scalar {position}: {value} != {reference}, error {error}",
                    case.transform,
                    case.test
                );
                let peak = if position < 3 * area {
                    &mut peak_coefficient
                } else {
                    &mut peak_lf
                };
                *peak = peak.max(error);
            }
            let bits = actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            if variant == KernelVariant::Scalar {
                baseline.push(bits);
            } else {
                assert_eq!(
                    bits, baseline[index],
                    "{:?}/{}: {variant:?}",
                    case.transform, case.test
                );
            }
        }
        eprintln!(
            "{variant:?}: 667 cases, peak coefficient error={peak_coefficient:e}, LF={peak_lf:e}"
        );
    }
}

#[test]
fn rejects_invalid_geometry_ranges_aliases_and_kernel_shapes_before_dispatch() {
    let backend = backend();
    let device = backend.device();
    let pipeline = ForwardVarDctPipeline::new(device, KernelVariant::Lanes64).unwrap();
    let case = Case {
        transform: TransformKind::Dct16x8,
        test: 0,
        values: Vec::new(),
    };
    let memory = ForwardVarDctMemoryPlan::new(case.transform);
    let sources = source_buffers(device, &case);
    let coefficients = output_buffer(device, memory.coefficient_bytes);
    let lf = output_buffer(device, memory.lf_bytes);
    let valid = inputs(&case, &sources, &coefficients, &lf);
    let reject = |input| {
        let mut commands = device.create_command_encoder(&Default::default());
        pipeline.encode(device, &mut commands, input).unwrap_err()
    };
    let mut invalid = valid;
    invalid.sources[1].width += 1;
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::SourceGeometry { channel: 1 }
    ));
    invalid = valid;
    invalid.sources[2].stride = 1;
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::SourceGeometry { channel: 2 }
    ));
    invalid = valid;
    invalid.origins[0] = u32::MAX;
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::SourceAddress { channel: 0 }
    ));
    invalid = valid;
    invalid.origins[2] += 1024;
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::BindingSize {
            role: "source B",
            ..
        }
    ));
    invalid = valid;
    invalid.coefficients.size = std::num::NonZeroU64::new(memory.coefficient_bytes - 4).unwrap();
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::BindingSize {
            role: "coefficients",
            ..
        }
    ));
    invalid = valid;
    invalid.low_frequency.size = std::num::NonZeroU64::new(memory.lf_bytes - 4).unwrap();
    assert!(matches!(
        reject(invalid),
        ForwardVarDctError::BindingSize { role: "LF", .. }
    ));
    invalid = valid;
    invalid.low_frequency = valid.coefficients;
    assert!(matches!(reject(invalid), ForwardVarDctError::Alias { .. }));
    invalid = valid;
    invalid.sources[0].storage = valid.coefficients;
    assert!(matches!(reject(invalid), ForwardVarDctError::Alias { .. }));
    invalid = valid;
    invalid.coefficients.offset += 4;
    assert!(matches!(reject(invalid), ForwardVarDctError::Storage(_)));
    let wrong_usage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: coefficients.size(),
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    invalid = valid;
    invalid.coefficients = binding(&wrong_usage);
    assert!(matches!(reject(invalid), ForwardVarDctError::Storage(_)));
    assert!(ForwardVarDctPipeline::new(device, KernelVariant::Tile8x8).is_err());
}

#[test]
fn insufficient_pipeline_limits_return_typed_errors() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("actual adapter required for forward pipeline admission");
    for (name, required, available) in [
        ("max_bindings_per_bind_group", 9, 8),
        ("max_storage_buffers_per_shader_stage", 6, 5),
        ("max_uniform_buffer_binding_size", 64, 32),
    ] {
        let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
        match name {
            "max_bindings_per_bind_group" => limits.max_bindings_per_bind_group = available as u32,
            "max_storage_buffers_per_shader_stage" => {
                limits.max_storage_buffers_per_shader_stage = available as u32
            }
            _ => limits.max_uniform_buffer_binding_size = available,
        }
        let (device, _) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .unwrap();
        let error = ForwardVarDctPipeline::new(&device, KernelVariant::Lanes64)
            .err()
            .expect("pipeline limit must be rejected before compilation");
        assert!(matches!(error, ForwardVarDctError::DeviceLimit {
            name: actual_name, required: actual_required, available: actual_available,
        } if actual_name == name && actual_required == required && actual_available == available));
    }
}

#[test]
fn shader_abi_and_memory_plan_match_allocations() {
    let module = naga::front::wgsl::parse_str(SHADER).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let uniform = module
        .types
        .iter()
        .find_map(|(_, ty)| (ty.name.as_deref() == Some("Params")).then_some(&ty.inner));
    let Some(naga::TypeInner::Struct { members, span }) = uniform else {
        panic!("missing shader parameters");
    };
    assert_eq!(*span, size_of::<Params>() as u32);
    assert_eq!(
        members[8].offset,
        std::mem::offset_of!(Params, strides) as u32
    );
    assert_eq!(
        members[9].offset,
        std::mem::offset_of!(Params, basis_offsets) as u32
    );
    let task = module
        .types
        .iter()
        .find_map(|(_, ty)| (ty.name.as_deref() == Some("Task")).then_some(&ty.inner));
    let Some(naga::TypeInner::Struct { members, span }) = task else {
        panic!("missing shader task descriptor");
    };
    assert_eq!(*span, size_of::<ForwardVarDctTask>() as u32);
    assert_eq!(
        members[1].offset,
        std::mem::offset_of!(ForwardVarDctTask, coefficient_offset) as u32
    );
    assert_eq!(
        members[2].offset,
        std::mem::offset_of!(ForwardVarDctTask, lf_offset) as u32
    );
    for transform in TransformKind::ALL {
        let memory = ForwardVarDctMemoryPlan::new(transform);
        assert_eq!(
            memory.basis_bytes,
            (basis::matrix(transform).weights.len() * 4) as u64
        );
        assert_eq!(
            memory.transient_bytes,
            memory.parameter_bytes
                + memory.basis_bytes
                + memory.horizontal_bytes
                + memory.task_bytes
        );
        assert_eq!(
            memory.coefficient_bytes,
            transform.pixel_extent().area().unwrap() as u64 * 12
        );
        assert_eq!(
            memory.lf_bytes,
            transform.lf_extent().area().unwrap() as u64 * 12
        );
    }
}

#[test]
fn batches_match_native_transforms_with_disjoint_reordered_ranges_and_guarded_crops() {
    let backend = backend();
    let device = backend.device();
    let cases = cases()
        .into_iter()
        .filter(|case| case.test == 0)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 27);
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let pipeline = ForwardVarDctPipeline::new(device, variant).unwrap();
        for case in &cases {
            let extent = case.transform.pixel_extent();
            let memory = ForwardVarDctMemoryPlan::for_batch(case.transform, 2).unwrap();
            let coefficient_len = memory.coefficient_bytes as usize / 8;
            let lf_len = memory.lf_bytes as usize / 8;
            let width = extent.width * 2 + 4;
            let height = extent.height * 2 + 3;
            let mut tasks = [
                ForwardVarDctTask {
                    origins: [3, 5, 7],
                    coefficient_offset: coefficient_len as u32 + 7,
                    lf_offset: lf_len as u32 + 11,
                },
                ForwardVarDctTask {
                    origins: [0; 3],
                    coefficient_offset: 3,
                    lf_offset: 5,
                },
            ];
            for channel in 0..3 {
                let stride = width + channel as u32 + 3;
                tasks[1].origins[channel] =
                    tasks[0].origins[channel] + extent.width + 4 + stride * (extent.height + 3);
            }
            let sources: [wgpu::Buffer; 3] = std::array::from_fn(|channel| {
                let stride = width + channel as u32 + 3;
                let mut values = vec![
                    POISON;
                    PREFIX as usize / 4
                        + tasks[0].origins[channel] as usize
                        + (stride * height) as usize
                ];
                for (task_index, task) in tasks.iter().enumerate() {
                    for y in 0..extent.height {
                        for x in 0..extent.width {
                            let value = sample(case, channel, x as usize, y as usize)
                                * if task_index == 0 { 1.0 } else { -0.5 };
                            let address = PREFIX as usize / 4
                                + (task.origins[channel] + y * stride + x) as usize;
                            values[address] = value.to_bits();
                        }
                    }
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batched native oracle crop source"),
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            });
            let coefficients = output_buffer(device, (2 * coefficient_len + 7) as u64 * 4);
            let lf = output_buffer(device, (2 * lf_len + 11) as u64 * 4);
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batched native oracle readback"),
                size: coefficients.size() + lf.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let inputs = ForwardVarDctBatchInputs {
                transform: case.transform,
                sources: std::array::from_fn(|channel| ResidentF32Plane {
                    storage: binding(&sources[channel]),
                    width,
                    height,
                    stride: width + channel as u32 + 3,
                }),
                tasks: &tasks,
                coefficients: binding(&coefficients),
                low_frequency: binding(&lf),
            };
            let mut commands = device.create_command_encoder(&Default::default());
            let scratch = pipeline
                .encode_batch(device, &mut commands, inputs)
                .unwrap();
            assert_eq!(scratch.memory, memory);
            assert_eq!(scratch.tasks.size(), 40);
            assert_eq!(
                scratch.parameters.size()
                    + scratch.basis.size()
                    + scratch.tasks.size()
                    + scratch.horizontal.as_ref().map_or(0, wgpu::Buffer::size),
                memory.transient_bytes
            );
            commands.copy_buffer_to_buffer(&coefficients, 0, &staging, 0, coefficients.size());
            commands.copy_buffer_to_buffer(&lf, 0, &staging, coefficients.size(), lf.size());
            let submission = backend.queue().submit([commands.finish()]);
            let (send, recv) = std::sync::mpsc::sync_channel(1);
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    send.send(result).unwrap()
                });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: None,
                })
                .unwrap();
            recv.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            let mut expected = vec![None; words.len()];
            for (index, task) in tasks.iter().enumerate() {
                let scale = if index == 0 { 1.0 } else { -0.5 };
                for (base, values, tolerance) in [
                    (
                        PREFIX as usize / 4 + task.coefficient_offset as usize,
                        &case.values[..coefficient_len],
                        2e-6,
                    ),
                    (
                        (coefficients.size() + PREFIX) as usize / 4 + task.lf_offset as usize,
                        &case.values[coefficient_len..],
                        2e-5,
                    ),
                ] {
                    for (offset, value) in values.iter().enumerate() {
                        expected[base + offset] = Some((value * scale, tolerance));
                    }
                }
            }
            for (word, expected) in words.iter().zip(expected) {
                if let Some((expected, tolerance)) = expected {
                    let actual = f32::from_bits(*word);
                    assert!(
                        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                        "{:?}/{variant:?}: actual {actual}, expected {expected}",
                        case.transform
                    );
                } else {
                    assert_eq!(*word, POISON, "batch escaped its output range");
                }
            }
            drop(mapped);
            staging.unmap();
            let mut commands = device.create_command_encoder(&Default::default());
            for invalid in [
                vec![],
                vec![tasks[0], tasks[0]],
                vec![ForwardVarDctTask {
                    coefficient_offset: u32::MAX,
                    ..tasks[0]
                }],
                vec![ForwardVarDctTask {
                    lf_offset: u32::MAX,
                    ..tasks[0]
                }],
            ] {
                assert!(matches!(
                    pipeline.encode_batch(
                        device,
                        &mut commands,
                        ForwardVarDctBatchInputs {
                            tasks: &invalid,
                            ..inputs
                        }
                    ),
                    Err(ForwardVarDctError::TaskLayout)
                ));
            }
        }
    }
    assert!(ForwardVarDctMemoryPlan::for_batch(TransformKind::Dct256x256, u32::MAX).is_err());
}
