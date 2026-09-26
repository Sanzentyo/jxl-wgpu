use super::*;
use std::num::NonZeroU64;

fn request(extent: Extent2d, config: &VarDctConfig, modular: bool) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: if modular {
            EncodeProfile::ModularLossless {
                sample_bit_depth: config.sample_format.bit_depth(),
            }
        } else {
            EncodeProfile::VarDct {
                quantization: config.quantization,
            }
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: extent.width,
        canvas_height: extent.height,
        options: FrameOptions::default(),
    }
}

fn drain(context: &WgpuContext, owner: Option<&std::sync::Weak<wgpu::Buffer>>) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (context.memory_stats().reserved_bytes != 0
        || owner.is_some_and(|p| p.upgrade().is_some()))
        && std::time::Instant::now() < until
    {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
    assert!(owner.is_none_or(|p| p.upgrade().is_none()));
}

fn budget_and_cancel<B: GpuEncodeBackend>(
    context: &WgpuContext,
    source: BufferImageSource,
    request: &FrameEncodeRequest,
    bytes: u64,
    backend: impl Fn(&WgpuContext) -> B,
) {
    for limit in [bytes - 1, bytes] {
        let bounded = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let backend = backend(&bounded);
        let result = backend.submit(&bounded, GpuFrameSource::Buffer(source.clone()), request);
        if limit < bytes {
            assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
        } else {
            drop(result.unwrap());
            drain(&bounded, None);
            backend
                .submit(&bounded, GpuFrameSource::Buffer(source.clone()), request)
                .unwrap()
                .wait()
                .unwrap();
        }
        drain(&bounded, None);
    }
    let owner = Arc::downgrade(&source.buffer);
    drop(
        backend(context)
            .submit(context, GpuFrameSource::Buffer(source), request)
            .unwrap(),
    );
    drain(context, Some(&owner));
}

#[test]
fn cmyk_primary_views_keep_exact_admission_and_release_one_shared_owner() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    let extent = Extent2d::new(17, 9);
    let mut config = config(&profile, 8, 0, true);
    let words: Vec<_> = (0..extent.area().unwrap() * 5)
        .map(|v| v as u32 * 7 % 256)
        .collect();
    let make_source = |config: &VarDctConfig| {
        input(
            &context,
            extent,
            config,
            Storage::Planar,
            CmykSampleEncoding::InkAmounts,
            &words,
        )
    };
    let source = make_source(&config);
    let memory = LosslessModularBackend::new(&context)
        .memory_plan(&source)
        .unwrap();
    let binding_bytes = memory.source_binding_bytes;
    budget_and_cancel(
        &context,
        source,
        &request(extent, &config, true),
        memory.owned_bytes_per_job,
        LosslessModularBackend::new,
    );
    for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
        config.color_transform = transform;
        let source = make_source(&config);
        let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
        let memory = backend.memory_plan(&source).unwrap();
        assert_eq!(memory.source_binding_bytes, binding_bytes);
        if let Some(icc) = memory.icc {
            assert_eq!(icc.input_bytes, 24 * 16 * 4 * 4);
            assert_eq!(icc.linear_bytes, 24 * 16 * 3 * 4);
            assert_eq!(
                icc.total_bytes,
                icc.input_bytes + icc.linear_bytes + icc.program_bytes + icc.parameter_bytes
            );
        } else {
            assert_eq!(transform, VarDctColorTransform::Original);
        }
        budget_and_cancel(
            &context,
            source,
            &request(extent, &config, false),
            memory.owned_bytes_per_job,
            |ctx| VarDctBackend::new_tiled_dct8_with_config(ctx, config.clone()).unwrap(),
        );
    }
}

#[test]
fn cmyk_ambiguous_black_conventions_sampling_and_nonfinite_color_reject_without_authority() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    // At 1x1 a different factor has identical rounded dimensions; semantics must still reject.
    let extent = Extent2d::new(1, 1);
    let mut config = config(&profile, 32, 8, true);
    let words = vec![0.5f32.to_bits(); 5];
    let implicit = raw_source(
        &context,
        extent,
        config.pixel_format(),
        Storage::Planar,
        &words,
    );
    let modular = LosslessModularBackend::new(&context);
    assert!(matches!(
        modular.memory_plan(&implicit),
        Err(EncodeError::InvalidSource(_))
    ));
    let source = implicit
        .with_cmyk_encoding(CmykSampleEncoding::Complemented)
        .unwrap();
    for modular_codec in [true, false] {
        let mut request = request(extent, &config, modular_codec);
        request.options.extra_channel_upsampling =
            vec![UpsamplingFactor::One, UpsamplingFactor::Two];
        if modular_codec {
            assert!(matches!(
                modular.submit(&context, GpuFrameSource::Buffer(source.clone()), &request),
                Err(EncodeError::InvalidConfiguration(_))
            ));
        } else {
            let backend =
                VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
            assert!(matches!(
                backend.submit(&context, GpuFrameSource::Buffer(source.clone()), &request),
                Err(EncodeError::InvalidConfiguration(_))
            ));
        }
    }
    config.extra_channels = vec![
        ExtraChannel::new(
            ExtraChannelKind::Black,
            SamplePrecision::integer(8).unwrap(),
            0,
            Vec::new(),
        )
        .unwrap(),
    ];
    assert!(matches!(
        TiledVarDctEncoder::new_with_config(context.clone(), config.clone()),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        LosslessModularBackend::with_config(
            &context,
            LosslessModularConfig {
                extra_channels: config.extra_channels.clone(),
                ..Default::default()
            }
        )
        .memory_plan(&source),
        Err(EncodeError::InvalidConfiguration(_))
    ));
    config.extra_channels.clear();
    config.color_transform = VarDctColorTransform::Xyb;
    let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
    for component in 0..4 {
        let mut invalid = words.clone();
        invalid[component] = f32::NAN.to_bits();
        let input = input(
            &context,
            extent,
            &config,
            Storage::Planar,
            CmykSampleEncoding::Complemented,
            &invalid,
        );
        assert!(matches!(
            backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(input),
                    &request(extent, &config, false)
                )
                .unwrap()
                .wait(),
            Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
        ));
    }
    drain(&context, None);
}

#[test]
fn associated_ink_input_rejects_in_every_frontend_before_sequence_admission() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    let extent = Extent2d::new(9, 5);
    let mut config = config(&profile, 8, 0, true);
    config.alpha = Some(AlphaAssociation::Associated);
    let words = vec![128; extent.area().unwrap() * 5];
    let source = input(
        &context,
        extent,
        &config,
        Storage::Planar,
        CmykSampleEncoding::InkAmounts,
        &words,
    );
    let modular = LosslessModularEncoder::new(context.clone())
        .with_alpha_association(AlphaAssociation::Associated)
        .with_image_options(config.image_options)
        .unwrap();
    assert!(matches!(
        modular.memory_plan(&source),
        Err(EncodeError::InvalidSource(_))
    ));
    assert!(matches!(
        modular.submit(source.clone()),
        Err(EncodeError::InvalidSource(_))
    ));
    let descriptor = LosslessModularSequenceDescriptor::from_pixel_format(
        extent.width,
        extent.height,
        &config.pixel_format(),
        AnimationHeader::Still,
    )
    .unwrap();
    let mut sequence = modular.begin_sequence(descriptor).unwrap();
    assert!(matches!(
        sequence.submit_last_frame(source.clone(), FrameOptions::default()),
        Err(EncodeError::InvalidSource(_))
    ));
    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
    drop(sequence);
    let vardct = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    assert!(matches!(
        vardct.memory_plan(&source),
        Err(EncodeError::InvalidSource(_))
    ));
    let mixed = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: config,
            ..Default::default()
        },
    )
    .unwrap();
    let mut sequence = mixed
        .begin_sequence(
            ImageSequenceDescriptor::new(extent.width, extent.height, AnimationHeader::Still)
                .unwrap(),
        )
        .unwrap();
    let before = context.memory_stats().reserved_bytes;
    for codec in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        assert!(matches!(
            sequence.submit_last_frame(source.clone(), codec, FrameOptions::default()),
            Err(EncodeError::InvalidSource(_))
        ));
        assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
        assert_eq!(context.memory_stats().reserved_bytes, before);
    }
    drop(sequence);
    drain(&context, None);
}
