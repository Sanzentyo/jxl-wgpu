use super::super::super::modular_plane::{ImagePlan, Limits, Pipeline};
use super::*;
use crate::{BackendError, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource, VarDctBackend};

fn pair(context: &WgpuContext, extent: Extent2d) -> (VarDctConfig, BufferImageSource) {
    let config = VarDctConfig {
        extra_channels: vec![declaration(ExtraChannelKind::Depth, 13, 0); 2],
        ..Default::default()
    };
    let words: Vec<_> = (0..extent.area().unwrap())
        .map(|i| (i as u32 * 379) & 8191)
        .collect();
    let scalar = scalar_source(
        context,
        extent,
        config.extra_channels[0].precision(),
        &words,
    );
    let source = color_source(context, extent)
        .with_extra_channels(vec![scalar.clone(), scalar])
        .unwrap();
    (config, source)
}

fn drain(context: &WgpuContext) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn extra_input_budget_union_cancellation_and_source_mismatch_preserve_admission() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(17, 13);
    let (config, source) = pair(&context, extent);
    let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
    let memory = backend.memory_plan(&source).unwrap();
    let scalar = &source.extra_channels()[0];
    let alignment = u64::from(
        context
            .device()
            .limits()
            .min_storage_buffer_offset_alignment,
    );
    let windows = |source: &BufferImageSource| {
        crate::source::SourceLayout::new(&source.layout, source.buffer.size(), alignment)
            .unwrap()
            .full_windows
    };
    assert_eq!(
        memory.source_binding_bytes,
        windows(&source).addressed_bytes().unwrap() + windows(scalar).addressed_bytes().unwrap()
    );
    let extras = memory.extra_channels.unwrap();
    assert_eq!(
        extras.total_bytes,
        extras.parameter_bytes + extras.artifact_bytes + extras.readback_bytes
    );
    assert_eq!(
        memory.readback_bytes,
        memory.artifact_storage_bytes + extras.readback_bytes
    );
    assert!(memory.alpha.is_none());
    let request = layouts::request(extent, &config);
    for limit in [memory.owned_bytes_per_job - 1, memory.owned_bytes_per_job] {
        let bounded = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let backend = VarDctBackend::new_tiled_dct8_with_config(&bounded, config.clone()).unwrap();
        let result = backend.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request);
        if limit < memory.owned_bytes_per_job {
            assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
        } else {
            drop(result.unwrap());
            drain(&bounded);
            backend
                .submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
        }
    }
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let still_memory = encoder.memory_plan(&source).unwrap();
    assert!(still_memory.extra_channel_metadata_bytes > 0);
    assert_eq!(
        still_memory.owned_bytes_per_job,
        memory.owned_bytes_per_job + still_memory.extra_channel_metadata_bytes
    );
    for limit in [
        still_memory.owned_bytes_per_job - 1,
        still_memory.owned_bytes_per_job,
    ] {
        let bounded = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(bounded.clone(), config.clone()).unwrap();
        let result = encoder.encode(source.clone());
        if limit < still_memory.owned_bytes_per_job {
            assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
        } else {
            result.unwrap();
        }
        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
    }
    for attachments in [vec![], vec![scalar.clone()], vec![scalar.clone(); 3]] {
        assert!(matches!(
            backend.memory_plan(&source.clone().with_extra_channels(attachments).unwrap()),
            Err(EncodeError::InvalidSource(_))
        ));
    }
    let unreadable = BufferImageSource::new(
        Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: scalar.buffer.size(),
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })),
        scalar.layout.clone(),
    )
    .unwrap();
    assert!(matches!(
        backend.memory_plan(
            &source
                .clone()
                .with_extra_channels(vec![unreadable, scalar.clone()])
                .unwrap()
        ),
        Err(EncodeError::InvalidSource(_))
    ));
    let (_, owned_source) = pair(&context, extent);
    let source_lifetime = Arc::downgrade(&owned_source.extra_channels()[0].buffer);
    drop(
        backend
            .submit(&context, GpuFrameSource::Buffer(owned_source), &request)
            .unwrap(),
    );
    drain(&context);
    assert!(source_lifetime.upgrade().is_none());
    for (wrong_extent, wrong_precision) in [
        (Extent2d::new(16, 13), SamplePrecision::integer(13).unwrap()),
        (extent, SamplePrecision::integer(12).unwrap()),
        (extent, SamplePrecision::float(13, 5).unwrap()),
    ] {
        let wrong = scalar_source(
            &context,
            wrong_extent,
            wrong_precision,
            &vec![0; wrong_extent.area().unwrap()],
        );
        assert!(
            backend
                .memory_plan(
                    &source
                        .clone()
                        .with_extra_channels(vec![wrong, scalar.clone()])
                        .unwrap()
                )
                .is_err()
        );
    }
    assert!(
        source
            .clone()
            .with_extra_channels(vec![source.clone()])
            .is_err()
    );
    assert!(
        source
            .clone()
            .with_extra_channels(vec![scalar.clone(); 257])
            .is_err()
    );
    let mut unsupported = crate::MixedModeConfig::default();
    unsupported.vardct.extra_channels = config.extra_channels.clone();
    assert!(crate::MixedModeEncoder::new(context.clone(), unsupported).is_err());
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn extra_input_artifact_identity_prevents_plane_swaps_and_malformed_publication() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(17, 13);
    let (config, source) = pair(&context, extent);
    let limits = context.device().limits();
    let main = crate::source::SourceLayout::new(
        &source.layout,
        source.buffer.size(),
        u64::from(limits.min_storage_buffer_offset_alignment),
    )
    .unwrap();
    let code = fixed_prefix_code().unwrap();
    let plan = ImagePlan::new(
        VarDctFrameLayout::tiled_dct8(extent.width, extent.height).unwrap(),
        &VarDctColorPlan::new(&config).unwrap().samples,
        &source,
        &main,
        &config.progressive,
        &code,
        Limits {
            buffer: limits.max_buffer_size,
            binding: limits.max_storage_buffer_binding_size,
            workgroups: limits.max_compute_workgroups_per_dimension,
            alignment: u64::from(limits.min_storage_buffer_offset_alignment),
        },
    )
    .unwrap();
    let readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: plan.memory.readback_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = context.device().create_command_encoder(&Default::default());
    let scratch = plan.encode(
        &Pipeline::new(context.device()),
        context.device(),
        &mut commands,
        &source,
        &readback,
        0,
    );
    context.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
    readback.unmap();
    drop(scratch);
    assert_eq!(plan.validate(&bytes, &code).unwrap().len(), 2);
    for mutation in 0..6 {
        let mut bad = bytes.clone();
        let middle = bad.len() / 2;
        match mutation {
            0 => {
                let (a, b) = bad.split_at_mut(middle);
                a.swap_with_slice(b);
            }
            1 => bad[middle] ^= 1,
            2 => bad[middle + 12..middle + 16].copy_from_slice(&u32::MAX.to_le_bytes()),
            3 => {
                bad.truncate(middle);
            }
            4 => bad.push(0),
            _ => *bad.last_mut().unwrap() = 0x80,
        }
        assert!(
            matches!(
                plan.validate(&bad, &code),
                Err(BackendError::InvalidArtifact(_))
            ),
            "mutation {mutation}"
        );
    }
}

#[test]
fn extra_input_packed_alpha_icc_raw_matrices_and_nonfinite_color_share_completion() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(8, 8);
    for transform in [
        crate::VarDctColorTransform::Xyb,
        crate::VarDctColorTransform::Original,
    ] {
        let profile = icc::profile(false);
        let mut config = icc::config(&profile, transform);
        config.alpha = Some(crate::AlphaAssociation::Associated);
        config.extra_channels = vec![
            declaration(ExtraChannelKind::Depth, 31, 1),
            ExtraChannel::new(
                ExtraChannelKind::Thermal,
                SamplePrecision::float(32, 8).unwrap(),
                0,
                vec![],
            )
            .unwrap(),
        ];
        config.dequant_matrices = raw_matrices::selected([VarDctStrategy::Dct8]);
        let words = alpha::input(extent, config.sample_format);
        let scalar_words: Vec<Vec<u32>> = config
            .extra_channels
            .iter()
            .map(|d| {
                (0..d.source_extent(extent).area().unwrap())
                    .map(|i| {
                        if d.precision()
                            .color(crate::ColorChannels::Gray)
                            .float_precision()
                            .is_some()
                        {
                            [0x80000000, 0x7fc00001, 0xff800000, 0x3f800000][i % 4]
                        } else {
                            (i as u32 * 379_331) & 0x7fff_ffff
                        }
                    })
                    .collect()
            })
            .collect();
        let attachments: Vec<_> = config
            .extra_channels
            .iter()
            .zip(&scalar_words)
            .map(|(d, w)| scalar_source(&context, d.source_extent(extent), d.precision(), w))
            .collect();
        let source = alpha::upload(&context, extent, &config, &words, Storage::Split, true)
            .with_extra_channels(attachments.clone())
            .unwrap();
        let definitions =
            crate::sample_format::ImageSamplePlan::new(config.sample_format, config.alpha)
                .with_extra_channels(
                    &config.extra_channels,
                    config.max_extra_channel_metadata_bytes,
                )
                .unwrap()
                .extra_channels;
        let expected: Vec<Vec<u32>> =
            std::iter::once(words.as_chunks::<4>().0.iter().map(|p| p[3]).collect())
                .chain(scalar_words)
                .collect();
        for topology in [
            layouts::Topology::Single,
            layouts::Topology::Map,
            layouts::Topology::Tiled,
        ] {
            let backend = topology.backend(&context, extent, &config);
            let request = layouts::request(extent, &config);
            let memory = backend.memory_plan(&source).unwrap();
            assert_eq!(
                memory.readback_bytes,
                memory.artifact_storage_bytes
                    + memory.raw_matrix_artifact_bytes
                    + memory.extra_channels.unwrap().readback_bytes
            );
            assert!(memory.extra_channels.unwrap().total_bytes > memory.alpha.unwrap().total_bytes);
            let artifacts = backend
                .submit(&context, GpuFrameSource::Buffer(source.clone()), &request)
                .unwrap()
                .wait()
                .unwrap();
            let color = VarDctColorPlan::new(&config).unwrap();
            let header = color
                .image_header(
                    &crate::ImageSequenceDescriptor::new(8, 8, crate::AnimationHeader::Still)
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(header.extra_storage_bytes, 0); // ICC already owns the whole header.
            let (header, permit) = header.finish(context.memory_budget()).unwrap();
            let mut bytes = header.bytes().to_vec();
            bytes.extend(assemble_frame(artifacts.packets).unwrap().into_bytes());
            drop(permit);
            check_words(&bytes, 0, &definitions, extent, &expected);
            let (_, native) =
                extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), 3).unwrap();
            assert_numeric(&native[0], &expected[0], definitions[0].precision());
            assert_numeric(&native[2], &expected[2], definitions[2].precision());
            let mut invalid = words.clone();
            invalid[0] = f32::NAN.to_bits();
            let invalid = alpha::upload(&context, extent, &config, &invalid, Storage::Split, true)
                .with_extra_channels(attachments.clone())
                .unwrap();
            assert!(matches!(
                backend
                    .submit(&context, GpuFrameSource::Buffer(invalid), &request)
                    .unwrap()
                    .wait(),
                Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
            ));
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}
