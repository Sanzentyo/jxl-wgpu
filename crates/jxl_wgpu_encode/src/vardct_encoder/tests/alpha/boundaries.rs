use super::super::super::modular_plane::{Pipeline, Plan};
use super::*;
use crate::{BackendError, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource};

#[test]
fn alpha_input_cannot_assemble_without_validated_side_plane_fragments() {
    let config = VarDctConfig {
        alpha: Some(AlphaAssociation::Unassociated),
        ..Default::default()
    };
    let color = VarDctColorPlan::new(&config).unwrap();
    let fixture = super::super::DcFixture {
        words: vec![],
        bits: 0,
    };
    assert!(matches!(
        build_frame_packet(
            fixture.artifact(),
            &fixed_prefix_code().unwrap(),
            &HfEntropyPlan::single_cluster_prefix().unwrap(),
            VarDctFrameLayout::single(VarDctStrategy::Dct8),
            &config,
            &still_control(8, 8),
            &color
        ),
        Err(EncodeError::Backend(BackendError::InvalidArtifact(_)))
    ));
}

fn drain(context: &WgpuContext) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < until {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn alpha_input_artifacts_reject_missing_rows_forged_lengths_and_trailing_bits() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(259, 263);
    let config = VarDctConfig {
        alpha: Some(AlphaAssociation::Unassociated),
        ..Default::default()
    };
    let words = input(extent, config.sample_format);
    let source = upload(&context, extent, &config, &words, Storage::Planar, true);
    let layout = crate::source::SourceLayout::new(
        &source.layout,
        source.buffer.size(),
        u64::from(
            context
                .device()
                .limits()
                .min_storage_buffer_offset_alignment,
        ),
    )
    .unwrap();
    let region = layout.region(0, 0, extent.width, extent.height).unwrap();
    let mut component = [region.components[3]];
    layout
        .full_windows
        .rebase(&mut component, [region.offsets[3], 0, 0, 0])
        .unwrap();
    let code = fixed_prefix_code().unwrap();
    let plan = Plan::new(
        VarDctFrameLayout::tiled_dct8(extent.width, extent.height).unwrap(),
        component[0],
        255,
        true,
        &crate::ProgressivePlan::single(),
        &code,
    )
    .unwrap();
    let bytes = plan.memory.artifact_bytes;
    for (buffer, binding, axis) in [
        (bytes - 1, u64::MAX, u32::MAX),
        (u64::MAX, bytes - 1, u32::MAX),
        (u64::MAX, u64::MAX, extent.height - 1),
    ] {
        assert!(matches!(
            plan.validate_limits(buffer, binding, axis),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::DeviceLimit { .. }
            ))
        ));
    }
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha artifact test"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = context.device().create_command_encoder(&Default::default());
    let scratch = Pipeline::new(context.device()).encode(
        context.device(),
        &mut commands,
        plan,
        layout.full_windows.entries(
            &source.buffer,
            super::super::super::dispatch::SOURCE_BINDINGS,
        ),
        &staging,
        0,
    );
    context.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    drop(scratch);
    plan.validate(&bytes, &code).unwrap();
    for mutation in 0..8 {
        let mut corrupt = bytes.clone();
        match mutation {
            0 => corrupt[0] ^= 1,
            1 => corrupt[4] ^= 1,
            2 => corrupt[8] ^= 1,
            3 => corrupt[12..16].copy_from_slice(&u32::MAX.to_le_bytes()),
            4 => corrupt[12..16].fill(0),
            5 => *corrupt.last_mut().unwrap() = 0x80,
            6 => {
                corrupt.truncate(corrupt.len() - 4);
            }
            _ => corrupt.extend([0; 4]),
        }
        assert!(
            matches!(
                plan.validate(&corrupt, &code),
                Err(BackendError::InvalidArtifact(_))
            ),
            "mutation {mutation}"
        );
    }
}

#[test]
fn alpha_input_budget_admission_cancellation_and_color_failure_release_every_buffer() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(25, 17);
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        let config = VarDctConfig {
            alpha: Some(AlphaAssociation::Associated),
            sample_format: ColorSampleFormat::float(channels, 32, 8).unwrap(),
            dequant_matrices: raw_matrices::selected([VarDctStrategy::Dct8]),
            progressive: progressive::combined(),
            ..Default::default()
        };
        let words = input(extent, config.sample_format);
        let source = upload(&context, extent, &config, &words, Storage::Split, true);
        for topology in [layouts::Topology::Map, layouts::Topology::Tiled] {
            let backend = topology.backend(&context, extent, &config);
            let memory = backend.memory_plan(&source).unwrap();
            let alpha = memory.alpha.unwrap();
            assert_eq!(
                alpha.total_bytes,
                alpha.parameter_bytes + alpha.artifact_bytes + alpha.readback_bytes
            );
            assert_eq!(
                memory.readback_bytes,
                memory.artifact_storage_bytes
                    + memory.raw_matrix_artifact_bytes
                    + alpha.readback_bytes
            );
            for limit in [memory.owned_bytes_per_job - 1, memory.owned_bytes_per_job] {
                let bounded = WgpuContext::with_memory_budget(
                    Arc::new(context.device().clone()),
                    Arc::new(context.queue().clone()),
                    NonZeroU64::new(limit).unwrap(),
                )
                .unwrap();
                let backend = topology.backend(&bounded, extent, &config);
                let request = layouts::request(extent, &config);
                let result =
                    backend.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request);
                if limit < memory.owned_bytes_per_job {
                    assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    continue;
                }
                drop(result.unwrap());
                drain(&bounded);
                backend
                    .submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request)
                    .unwrap()
                    .wait()
                    .unwrap();
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                let mut invalid = words.clone();
                invalid[0] = f32::NAN.to_bits();
                let source = upload(&bounded, extent, &config, &invalid, Storage::Split, true);
                assert!(matches!(
                    backend
                        .submit(&bounded, GpuFrameSource::Buffer(source), &request)
                        .unwrap()
                        .wait(),
                    Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
                ));
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn alpha_input_gpu_readback_retains_exact_integer_and_float_samples_after_session_drop() {
    use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
    use jxl_test_support::gpu::planes::open_fragmented;
    use jxl_wgpu_decode::{NumericSampleMapping, WgpuDecodeEngine};
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoders = [
        GpuDecoder::wgpu(gpu.clone()).unwrap(),
        GpuDecoder::new(
            WgpuDecodeEngine::new(gpu.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ];
    let readback = ImageReadbackPipeline::new(&gpu);
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for samples in [
            ColorSampleFormat::integer(channels, 31).unwrap(),
            ColorSampleFormat::float(channels, 32, 8).unwrap(),
        ] {
            for extent in [Extent2d::new(17, 13), Extent2d::new(259, 3)] {
                let config = VarDctConfig {
                    sample_format: samples,
                    alpha: Some(AlphaAssociation::Associated),
                    progressive: progressive::combined()
                        .with_downsampling(vec![crate::ProgressiveDownsampling {
                            factor: 1,
                            last_pass: 0,
                        }])
                        .unwrap(),
                    ..Default::default()
                };
                let words = input(extent, samples);
                let encoder =
                    TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
                let bytes = encoder
                    .encode(upload(
                        &context,
                        extent,
                        &config,
                        &words,
                        Storage::Planar,
                        true,
                    ))
                    .unwrap();
                let request = if samples.float_precision().is_some() {
                    GpuOutputRequest::numeric(
                        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                        NumericSampleMapping::NativeFloat,
                    )
                } else {
                    GpuOutputRequest::numeric(
                        crate::LosslessModularFormat::Gray.pixel_format(31).unwrap(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                }
                .unwrap()
                .with_extra_channel(0)
                .unwrap();
                let expected: Vec<u8> = expected_alpha(&words, samples)
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect();
                for (decoder, fragmented) in decoders.iter().zip([false, true]) {
                    let mut session = if fragmented {
                        open_fragmented(decoder, &bytes, request.clone())
                    } else {
                        decoder.open(&bytes, request.clone()).unwrap()
                    };
                    let frame = session.next_frame().unwrap().unwrap();
                    assert!(session.next_frame().unwrap().is_none());
                    drop(session);
                    let result = readback.submit(frame.output()).unwrap().wait().unwrap();
                    assert_eq!(result.frame.outputs[0].bytes, expected);
                    drop(frame);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
