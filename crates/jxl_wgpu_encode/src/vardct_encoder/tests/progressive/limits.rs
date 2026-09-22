use super::*;

#[test]
fn progressive_arenas_and_toc_counts_are_bounded() {
    let code = fixed_prefix_code().unwrap();
    let hf = HfEntropyPlan::single_cluster_prefix().unwrap();
    for (width, height) in [(1, 1), (257, 17), (2057, 17), (16_384, 16_384)] {
        let frame = VarDctFrameLayout::tiled_dct8(width, height).unwrap();
        let layout = ArtifactLayout::for_tiled_grid(frame, &code, &hf).unwrap();
        for invalid in [0, 12, usize::MAX] {
            assert!(matches!(
                layout.with_passes(invalid),
                Err(EncodeError::InvalidConfiguration(_))
            ));
        }
        let mut grid = TiledVarDctGrid::new(width, height).unwrap();
        for passes in [2, 3, 5, 11] {
            grid.passes = passes;
            assert_eq!(
                grid.toc_entries().unwrap(),
                2 + frame.lf_group_count().unwrap()
                    + u32::from(passes) * frame.ac_group_count().unwrap()
            );
        }
        if width == 16_384 {
            // Checked arithmetic rejects this arena without attempting a multi-GB allocation.
            assert!(matches!(
                layout.with_passes(11),
                Err(EncodeError::InvalidConfiguration(
                    "VarDCT pass arena overflow"
                ))
            ));
        } else {
            assert!(layout.with_passes(11).unwrap().with_passes(2).is_err());
        }
    }
    let mut zero_hf = ArtifactLayout::new(VarDctStrategy::Dct8, &code).unwrap();
    zero_hf.ac_descriptor_len = 0;
    assert!(zero_hf.with_passes(2).is_err());
}

#[test]
fn progressive_storage_and_request_mismatches_reject_before_admission() {
    const LIMIT: u64 = 64 * 1024;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_storage_buffer_binding_size = LIMIT;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
    let source = padded_rgb_source_sized(&context, 65, 17, &reference::pattern(65, 17));
    let baseline = TiledVarDctEncoder::new(context.clone()).unwrap();
    assert!(
        baseline
            .memory_plan(&source)
            .unwrap()
            .artifact_storage_bytes
            < LIMIT
    );
    baseline.encode(source.clone()).unwrap();
    let config = VarDctConfig {
        progressive: maximum(),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    assert_eq!(encoder.capabilities().max_progressive_passes, 11);
    assert!(matches!(
        encoder.submit(source),
        Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
            name: "max_storage_buffer_binding_size",
            required,
            available: LIMIT
        })) if required > LIMIT
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

    let source = padded_rgb_source_sized(&context, 8, 8, &reference::pattern(8, 8));
    let backend =
        VarDctBackend::new_with_config(&context, VarDctStrategy::Dct8, config.clone()).unwrap();
    let request = FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: ProgressivePlan::single(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: 8,
        canvas_height: 8,
        options: FrameOptions::default(),
    };
    assert!(matches!(
        backend.submit(&context, GpuFrameSource::Buffer(source), &request),
        Err(EncodeError::InvalidConfiguration(
            "the requested VarDCT passes do not match the backend configuration"
        ))
    ));
    assert_eq!(context.memory_budget().snapshot().reserved_bytes, 0);
}
