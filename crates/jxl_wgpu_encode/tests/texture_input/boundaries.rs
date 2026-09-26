use super::*;
use std::num::NonZeroU64;

fn request(extent: Extent2d, modular: bool) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: if modular {
            EncodeProfile::ModularLossless {
                sample_bit_depth: ColorSampleFormat::GRAY8.bit_depth(),
            }
        } else {
            EncodeProfile::VarDct {
                quantization: VarDctQuantization::default(),
            }
        },
        progressive: ProgressivePlan::single(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: extent.width,
        canvas_height: extent.height,
        options: FrameOptions::default(),
    }
}

fn drain(context: &WgpuContext, texture: Option<&std::sync::Weak<wgpu::Texture>>) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (context.memory_stats().reserved_bytes != 0
        || texture.is_some_and(|owner| owner.upgrade().is_some()))
        && std::time::Instant::now() < deadline
    {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
    assert!(texture.is_none_or(|owner| owner.upgrade().is_none()));
}

fn admission<B: GpuEncodeBackend>(
    context: &WgpuContext,
    input: TextureImageSource,
    bytes: u64,
    request: &FrameEncodeRequest,
    make: impl Fn(&WgpuContext) -> B,
) {
    for limit in [bytes - 1, bytes] {
        let bounded = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let backend = make(&bounded);
        let job = backend.submit(&bounded, input.clone().into(), request);
        if limit < bytes {
            assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
        } else {
            drop(job.unwrap());
            drain(&bounded, None);
            backend
                .submit(&bounded, input.clone().into(), request)
                .unwrap()
                .wait()
                .unwrap();
        }
        drain(&bounded, None);
    }
    let owner = Arc::downgrade(&input.texture);
    drop(
        make(context)
            .submit(context, input.into(), request)
            .unwrap(),
    );
    drain(context, Some(&owner));
}

#[test]
fn copy_storage_is_reserved_once_and_released_after_completion_or_cancellation() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(129, 17);
    let raw: Vec<_> = (0..extent.area().unwrap())
        .map(|i| (i * 37) as u8)
        .collect();
    let source = || {
        texture(
            &context,
            extent,
            ColorSampleFormat::GRAY8.pixel_format(),
            wgpu::TextureFormat::R8Unorm,
            &raw,
        )
    };
    let copy_bytes = (256 * (extent.height as u64 - 1) + extent.width as u64).div_ceil(4) * 4;
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        let config = LosslessModularConfig {
            entropy,
            ..Default::default()
        };
        let backend = LosslessModularBackend::with_config(&context, config.clone());
        let input = source();
        let plan = backend.memory_plan(&input).unwrap();
        assert_eq!(plan.source_copy_bytes, copy_bytes);
        assert_eq!(
            plan.owned_bytes_per_job,
            plan.parameter_storage_bytes
                + plan.artifact_storage_bytes
                + plan.readback_bytes
                + copy_bytes
        );
        assert_eq!(
            plan.addressed_bytes_per_job,
            plan.owned_bytes_per_job + raw.len() as u64
        );
        assert_eq!(
            plan.gpu_submission_count,
            if entropy == LosslessModularEntropyCoding::Ans {
                2
            } else {
                1
            }
        );
        admission(
            &context,
            input,
            plan.owned_bytes_per_job,
            &request(extent, true),
            |ctx| LosslessModularBackend::with_config(ctx, config.clone()),
        );
    }
    let config = VarDctConfig {
        sample_format: ColorSampleFormat::GRAY8,
        color_transform: VarDctColorTransform::Original,
        ..Default::default()
    };
    let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
    let input = source();
    let plan = backend.memory_plan(&input).unwrap();
    let base = backend
        .memory_plan(buffer(&context, extent, config.pixel_format(), &raw))
        .unwrap();
    assert_eq!(
        plan.owned_bytes_per_job,
        base.owned_bytes_per_job + copy_bytes
    );
    assert_eq!(
        plan.addressed_bytes_per_job,
        plan.owned_bytes_per_job + raw.len() as u64
    );
    admission(
        &context,
        input,
        plan.owned_bytes_per_job,
        &request(extent, false),
        |ctx| VarDctBackend::new_tiled_dct8_with_config(ctx, config.clone()).unwrap(),
    );
}

#[test]
fn texture_storage_and_mutated_descriptors_reject_before_admission() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let format = ColorSampleFormat::GRAY8.pixel_format();
    let valid = texture(
        &context,
        extent,
        format.clone(),
        wgpu::TextureFormat::R8Uint,
        &[71; 45],
    );
    for mutate in 0..5 {
        let mut input = valid.clone();
        match mutate {
            0 => input.texture_format = wgpu::TextureFormat::R8Unorm,
            1 => input.mip_level = u32::MAX,
            2 => input.array_layer = 3,
            3 => input.pixel_format = ColorSampleFormat::RGB8.pixel_format(),
            _ => input.pixel_format.planes[0].pixels_per_element = 2,
        }
        let modular = LosslessModularBackend::new(&context);
        let vardct = VarDctBackend::new_tiled_dct8_with_config(
            &context,
            VarDctConfig {
                sample_format: ColorSampleFormat::GRAY8,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(matches!(
            modular.memory_plan(&input),
            Err(EncodeError::InvalidSource(_))
        ));
        assert!(
            modular
                .submit(&context, input.clone().into(), &request(extent, true))
                .is_err()
        );
        assert!(vardct.memory_plan(&input).is_err());
        assert!(
            vardct
                .submit(&context, input.into(), &request(extent, false))
                .is_err()
        );
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
    for (dimension, storage, usage, samples) in [
        (
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::R8Uint,
            wgpu::TextureUsages::COPY_DST,
            1,
        ),
        (
            wgpu::TextureDimension::D3,
            wgpu::TextureFormat::R8Uint,
            wgpu::TextureUsages::COPY_SRC,
            1,
        ),
        (
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::COPY_SRC,
            1,
        ),
        (
            wgpu::TextureDimension::D2,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            4,
        ),
    ] {
        let texture = Arc::new(context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("valid GPU texture outside the encoder source contract"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension,
            format: storage,
            usage,
            view_formats: &[],
        }));
        assert!(matches!(
            TextureImageSource::new(texture, storage, format.clone(), 0, 0),
            Err(EncodeError::InvalidSource(_))
        ));
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn nonfinite_texture_color_never_publishes_vardct_or_preview_authority() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(8, 8);
    let config = VarDctConfig {
        sample_format: ColorSampleFormat::float(ColorChannels::Gray, 32, 8).unwrap(),
        ..Default::default()
    };
    let mut raw = bytes(&vec![0.5f32.to_bits(); extent.area().unwrap()], 4);
    raw[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    let input = texture(
        &context,
        extent,
        config.pixel_format(),
        wgpu::TextureFormat::R32Float,
        &raw,
    );
    let encoder =
        VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config).unwrap();
    assert!(matches!(
        encoder.submit(input.clone()).unwrap().wait(),
        Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
    ));
    let mut sequence = encoder
        .begin_sequence(
            ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still)
                .unwrap()
                .with_preview(PreviewSize::new(8, 8).unwrap()),
        )
        .unwrap();
    let baseline = context.memory_stats().reserved_bytes;
    assert!(
        sequence
            .preview_memory_plan(&input, FrameOptions::default())
            .is_ok()
    );
    assert!(matches!(
        sequence
            .submit_preview(input, FrameOptions::default())
            .unwrap()
            .wait(),
        Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
    ));
    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
    assert_eq!(context.memory_stats().reserved_bytes, baseline);
    drop(sequence);
    drain(&context, None);
}

#[test]
fn independent_associated_alpha_uses_the_same_cmyk_convention_check_for_both_storages() {
    use jxl_gpu_formats::ColorSpecification;
    use jxl_gpu_protocol::icc::IccProfile;
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let profile = IccProfile::parse(
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/lut/lut16_xyz_4.icc"),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap();
    let alpha = ExtraChannel::new(
        ExtraChannelKind::Alpha(AlphaAssociation::Associated),
        SamplePrecision::integer(8).unwrap(),
        0,
        Vec::new(),
    )
    .unwrap();
    let config = VarDctConfig {
        source_color: ColorSpecification::Icc(profile.clone()),
        image_options: ImageOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        },
        extra_channels: vec![alpha.clone()],
        color_transform: VarDctColorTransform::Original,
        ..Default::default()
    };
    let extra = buffer(
        &context,
        extent,
        alpha.precision().pixel_format(),
        &[128; 45],
    );
    let input = texture(
        &context,
        extent,
        config.pixel_format(),
        wgpu::TextureFormat::Rgba8Uint,
        &[31; 180],
    )
    .with_extra_channels(vec![extra.clone()])
    .unwrap();
    let canonical = buffer(&context, extent, config.pixel_format(), &[31; 180])
        .with_extra_channels(vec![extra])
        .unwrap();
    let modular = LosslessModularEncoder::with_config(
        context.clone(),
        LosslessModularConfig {
            extra_channels: vec![alpha],
            ..Default::default()
        },
    )
    .with_image_options(config.image_options)
    .unwrap();
    let vardct = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let mixed = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: config,
            ..Default::default()
        },
    )
    .unwrap();
    let descriptor = ImageSequenceDescriptor::new(9, 5, AnimationHeader::Still).unwrap();
    for source in [
        GpuFrameSource::from(input.clone()),
        GpuFrameSource::from(canonical.clone()),
    ] {
        assert!(matches!(
            modular.memory_plan(&source),
            Err(EncodeError::InvalidSource(_))
        ));
        assert!(matches!(
            modular.submit(source.clone()),
            Err(EncodeError::InvalidSource(_))
        ));
        assert!(matches!(
            vardct.memory_plan(&source),
            Err(EncodeError::InvalidSource(_))
        ));
        let mut sequence = mixed.begin_sequence(descriptor.clone()).unwrap();
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
    }
    let input = input
        .with_cmyk_encoding(CmykSampleEncoding::Complemented)
        .unwrap();
    let canonical = canonical
        .with_cmyk_encoding(CmykSampleEncoding::Complemented)
        .unwrap();
    assert_eq!(
        modular.encode(input.clone()).unwrap(),
        modular.encode(canonical.clone()).unwrap()
    );
    assert_eq!(
        vardct.encode(input).unwrap(),
        vardct.encode(canonical).unwrap()
    );
    drain(&context, None);
}
