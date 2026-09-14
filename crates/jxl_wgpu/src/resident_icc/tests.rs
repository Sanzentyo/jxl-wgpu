use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn singular_gpu_black_connection_rejects_status_and_leaves_pixels_untouched() {
    let backend =
        pollster::block_on(crate::WgpuBackend::request_default(Default::default())).unwrap();
    let device = backend.device();
    // A finite source-black probe with Z equal to PCS D50 makes the connection singular.
    // Exercise the real preparation and image entry points, including their pass ordering.
    let mut words = vec![
        1, 8, 0, 0, // main header
        9, 3, 3, 8, // connection stage
        20, 3, 0, 0, // source program and endpoint channels
        0, 0, 0, 0, // target black
        0, 0, 0, 0, // device endpoint and padding
        1, 0, 0, 0, // probe header
        2, 3, 3, 28, // constant affine
    ];
    for black in [0.1_f32, 0.1, 0.8249] {
        words.extend([0, 0, 0, black.to_bits()]);
    }
    let program = ResidentIccProgram {
        buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("singular ICC connection metadata"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        }),
        input_channels: 3,
        output_channels: 3,
        memory: ResidentIccMemoryPlan {
            program_bytes: words.len() as u64 * 4,
            dispatch_bytes: 304,
            validation_bytes: 4,
        },
    };
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC singular test input"),
        contents: bytemuck::cast_slice(&[0.5_f32; 6]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let guard = [-12345.25_f32; 6];
    let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ICC singular test output"),
        contents: bytemuck::cast_slice(&guard),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ICC singular test readback"),
        size: 24,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let binding = |buffer| ResidentStorageBinding {
        buffer,
        offset: 0,
        size: std::num::NonZeroU64::new(24).unwrap(),
    };
    let planes = [0, 2, 4].map(|offset| ResidentIccPlane { offset, stride: 2 });
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Tile16x16,
    ] {
        let pipeline = ResidentIccPipeline::with_variant(device, variant).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let dispatch = pipeline
            .encode(
                device,
                &mut encoder,
                &program,
                ResidentIccInputs {
                    input: binding(&input),
                    output: binding(&output),
                    extent: Extent2d::new(2, 1),
                    input_planes: &planes,
                    output_planes: &planes,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 24);
        let submission = backend.queue().submit([encoder.finish()]);
        let validation = dispatch.validation_buffer().unwrap();
        let completions = [&readback, validation].map(|buffer| {
            let (send, receive) = std::sync::mpsc::sync_channel(1);
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                send.send(result).unwrap();
            });
            receive
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        for completion in completions {
            completion.recv().unwrap().unwrap();
        }
        let status = validation.slice(..).get_mapped_range().unwrap();
        assert_eq!(&*status, 1_u32.to_le_bytes());
        assert_eq!(
            ResidentIccDispatch::validate_status(&status),
            Err(ResidentIccError::Precision)
        );
        drop(status);
        validation.unmap();
        let pixels = readback.slice(..).get_mapped_range().unwrap();
        assert_eq!(&*pixels, bytemuck::cast_slice::<f32, u8>(&guard));
        drop(pixels);
        readback.unmap();
    }
}

#[test]
fn profile_binding_capabilities_fail_before_pipeline_creation() {
    assert!(
        validate_capabilities(&wgpu::Limits {
            max_uniform_buffers_per_shader_stage: 0,
            max_uniform_buffer_binding_size: 0,
            ..Default::default()
        })
        .is_ok()
    );
    for (limits, resource) in [
        (
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 3,
                ..Default::default()
            },
            "storage bindings",
        ),
        (
            wgpu::Limits {
                max_bind_groups: 0,
                ..Default::default()
            },
            "bind groups",
        ),
        (
            wgpu::Limits {
                max_bindings_per_bind_group: 3,
                ..Default::default()
            },
            "binding slots",
        ),
        (
            wgpu::Limits {
                max_storage_buffer_binding_size: 303,
                ..Default::default()
            },
            "dispatch storage bytes",
        ),
    ] {
        assert!(
            matches!(validate_capabilities(&limits), Err(ResidentIccError::Limit { resource: actual, .. }) if actual == resource)
        );
    }
}

#[test]
fn icc_shader_and_dispatch_abi_are_webgpu_portable() {
    let module = naga::front::wgsl::parse_str(include_str!("../../shaders/icc.wgsl")).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let (_, params) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
        .unwrap();
    let naga::TypeInner::Struct { members, span } = &params.inner else {
        panic!("dispatch parameters are not a struct");
    };
    assert_eq!(*span, std::mem::size_of::<DispatchParams>() as u32);
    assert_eq!(
        members
            .iter()
            .map(|member| member.offset)
            .collect::<Vec<_>>(),
        vec![0, 16, 80, 144, 208, 272, 288, 300]
    );
}
