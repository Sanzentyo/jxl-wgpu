use super::*;
use std::num::NonZeroU64;

fn request(extent: Extent2d, modular: bool) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: if modular {
            EncodeProfile::ModularLossless {
                sample_bit_depth: ColorSampleFormat::float(ColorChannels::Rgb, 32, 8)
                    .unwrap()
                    .bit_depth(),
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

fn drain(context: &WgpuContext, owner: Option<&SourceOwners>) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (context.memory_stats().reserved_bytes != 0 || owner.is_some_and(|p| !p.released()))
        && std::time::Instant::now() < deadline
    {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
    assert!(owner.is_none_or(SourceOwners::released));
}

fn admission<B: GpuEncodeBackend>(
    context: &WgpuContext,
    fixture: &Case,
    bytes: u64,
    modular: bool,
    textures: bool,
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
        let input = fixture.stored_input(&bounded, YuvRgbTransfer::Preserve, textures);
        let owner = SourceOwners::of(&input);
        let job = backend.submit(
            &bounded,
            input.into(),
            &request(fixture.layout.extent, modular),
        );
        if limit < bytes {
            assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
        } else {
            drop(job.unwrap());
        }
        drain(&bounded, Some(&owner));
        if limit == bytes {
            backend
                .submit(
                    &bounded,
                    fixture
                        .stored_input(&bounded, YuvRgbTransfer::Preserve, textures)
                        .into(),
                    &request(fixture.layout.extent, modular),
                )
                .unwrap()
                .wait()
                .unwrap();
            drain(&bounded, None);
        }
    }
}

#[test]
fn converted_input_budget_is_exact_and_cancellation_retires_sources_for_both_codecs() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let fixture = case(
        Extent2d::new(129, 17),
        Packing::Semi(ChromaOrder::CbCr),
        ChromaSubsampling::Cs420,
        8,
        false,
        ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER),
    );
    for textures in [false, true] {
        let input = fixture.stored_input(&context, YuvRgbTransfer::Preserve, textures);
        let conversion = 12 * fixture.layout.extent.area().unwrap() as u64 + 112;
        let strict_backend = LosslessModularBackend::new(&context);
        let mut strict_request = request(fixture.layout.extent, true);
        strict_request.minimum_determinism = Determinism::CrossDevice;
        assert!(matches!(
            strict_backend.memory_plan_for_request(&input, &strict_request),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::InputDeterminism { .. }
            ))
        ));
        assert!(matches!(
            strict_backend.submit(&context, input.clone().into(), &strict_request),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::InputDeterminism { .. }
            ))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, 0);
        let binding = if textures {
            0
        } else {
            fixture.bytes.len() as u64
        };
        let mut copy = 0u64;
        for plane in &fixture.layout.planes {
            copy = copy.div_ceil(4) * 4;
            copy +=
                plane.row_bytes.div_ceil(256) * 256 * (u64::from(plane.sample_extent.height) - 1)
                    + plane.row_bytes;
        }
        let copy = if textures { copy.div_ceil(4) * 4 } else { 0 };
        let texture_bytes = if textures {
            fixture
                .layout
                .planes
                .iter()
                .map(|plane| plane.row_bytes * u64::from(plane.sample_extent.height))
                .sum()
        } else {
            0
        };
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let config = LosslessModularConfig {
                entropy,
                ..Default::default()
            };
            let backend = LosslessModularBackend::with_config(&context, config.clone());
            let plan = backend.memory_plan(&input).unwrap();
            assert_eq!(plan.source_conversion_bytes, conversion);
            assert_eq!(plan.source_copy_bytes, copy);
            assert_eq!(plan.source_texture_bytes, texture_bytes);
            assert_eq!(plan.source_binding_bytes, binding);
            assert_eq!(plan.peak_source_binding_bytes, binding);
            assert_eq!(
                plan.owned_bytes_per_job,
                plan.parameter_storage_bytes
                    + plan.artifact_storage_bytes
                    + plan.readback_bytes
                    + conversion
                    + copy
            );
            assert_eq!(
                plan.addressed_bytes_per_job,
                plan.owned_bytes_per_job + binding + texture_bytes
            );
            admission(
                &context,
                &fixture,
                plan.owned_bytes_per_job,
                true,
                textures,
                |ctx| LosslessModularBackend::with_config(ctx, config.clone()),
            );
        }
        let config = config(&input);
        let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
        let plan = backend.memory_plan(&input).unwrap();
        assert_eq!(plan.source_conversion_bytes, conversion);
        assert_eq!(plan.source_binding_bytes, binding);
        assert_eq!(
            plan.addressed_bytes_per_job,
            plan.owned_bytes_per_job + binding + texture_bytes
        );
        admission(
            &context,
            &fixture,
            plan.owned_bytes_per_job,
            false,
            textures,
            |ctx| VarDctBackend::new_tiled_dct8_with_config(ctx, config.clone()).unwrap(),
        );
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn yuv_and_independent_scalar_aliases_are_counted_once() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let fixture = case(
        Extent2d::new(17, 9),
        Packing::Semi(ChromaOrder::CrCb),
        ChromaSubsampling::Cs420,
        8,
        false,
        ColorSpec::bt709(ColorRange::Full, ChromaLocation2d::EVEN),
    );
    let raw = upload(&context, fixture.layout.clone(), &fixture.bytes);
    let scalar = BufferImageSource::new(
        raw.buffer.clone(),
        ImageLayout::from_planes(
            raw.layout.extent,
            SamplePrecision::integer(8).unwrap().pixel_format(),
            vec![raw.layout.planes[0].clone()],
        )
        .unwrap(),
    )
    .unwrap();
    let input = YuvImageSource::new(raw, YuvRgbTransfer::Preserve)
        .unwrap()
        .with_extra_channels(vec![scalar])
        .unwrap();
    let depth = ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::integer(8).unwrap(),
        0,
        Vec::new(),
    )
    .unwrap();
    let modular = LosslessModularEncoder::with_config(
        context.clone(),
        LosslessModularConfig {
            extra_channels: vec![depth.clone()],
            ..Default::default()
        },
    );
    let mut cfg = config(&input);
    cfg.extra_channels = vec![depth];
    let vardct = TiledVarDctEncoder::new_with_config(context.clone(), cfg).unwrap();
    assert_eq!(
        modular.memory_plan(&input).unwrap().source_binding_bytes,
        fixture.bytes.len() as u64
    );
    assert_eq!(
        vardct.memory_plan(&input).unwrap().source_binding_bytes,
        fixture.bytes.len() as u64
    );
    let encoded = modular.encode(input.clone()).unwrap();
    let words = modular_words::channel_frames(&encoded).remove(0);
    fixture.assert_rgb(&words, false);
    assert_eq!(
        words[3].words,
        fixture
            .codes
            .y
            .iter()
            .map(|&v| u32::from(v))
            .collect::<Vec<_>>()
    );
    let encoded = vardct.encode(input).unwrap();
    assert_eq!(
        modular_integer::vardct_extra_words(&encoded, 0)[0],
        words[3]
    );
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn malformed_or_ambiguous_yuv_inputs_and_noncommuting_alpha_reject_without_admission() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let fixture = case(
        Extent2d::new(9, 5),
        Packing::Planar,
        ChromaSubsampling::Cs420,
        10,
        true,
        ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER),
    );
    let original = upload(&context, fixture.layout.clone(), &fixture.bytes);
    for mutation in 0..14 {
        let mut raw = original.clone();
        let mut output = YuvRgbTransfer::Preserve;
        match mutation {
            0 => raw.layout.logical_size += 4,
            1 => raw.layout.planes[0].offset = u64::MAX,
            2 => raw.layout.planes[1].plane_index = 0,
            3 => raw.layout.planes[2].offset = raw.layout.planes[1].offset,
            4 => raw.layout.planes[0].row_stride = 0,
            5 => raw.layout.format.color_spec = ColorSpecification::Undefined,
            6 => raw.layout.format.sample_kind = SampleKind::Signed,
            7 => raw.layout.extent.width = 0,
            8 => {
                raw.buffer = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: fixture.bytes.len() as u64,
                    usage: wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            }
            13 => raw.layout.format.swizzle = Swizzle::ZYX1,
            9 => {
                raw.buffer = Arc::new(context.device().create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }))
            }
            other => {
                let ColorSpecification::Defined(ref mut color) = raw.layout.format.color_spec
                else {
                    unreachable!()
                };
                if other == 10 {
                    color.chroma_location = ChromaLocation2d::BOTH;
                }
                if other == 11 {
                    color.transfer = TransferFunction::Pq;
                    output = YuvRgbTransfer::Linear;
                }
                if other == 12 {
                    color.encoding = YcbcrEncoding::Bt2020ConstantLuminance;
                }
            }
        }
        assert!(
            YuvImageSource::new(raw, output).is_err(),
            "mutation {mutation}"
        );
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
    assert!(
        LosslessModularEncoder::new(context.clone())
            .memory_plan(original)
            .is_err()
    );
    let alpha = ExtraChannel::new(
        ExtraChannelKind::Alpha(AlphaAssociation::Associated),
        SamplePrecision::integer(8).unwrap(),
        0,
        Vec::new(),
    )
    .unwrap();
    let extra_layout =
        ImageLayout::packed(fixture.layout.extent, alpha.precision().pixel_format()).unwrap();
    let extra = upload(
        &context,
        extra_layout.clone(),
        &vec![128; extra_layout.logical_size.div_ceil(4) as usize * 4],
    );
    let linear = fixture
        .input(&context, YuvRgbTransfer::Linear)
        .with_extra_channels(vec![extra.clone()])
        .unwrap();
    let mut cfg = config(&linear);
    cfg.extra_channels = vec![alpha.clone()];
    let modular = LosslessModularEncoder::with_config(
        context.clone(),
        LosslessModularConfig {
            extra_channels: vec![alpha],
            ..Default::default()
        },
    );
    let vardct = TiledVarDctEncoder::new_with_config(context.clone(), cfg.clone()).unwrap();
    let mixed = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: cfg,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(
        modular.memory_plan(&linear),
        Err(EncodeError::InvalidSource(_))
    ));
    assert!(matches!(
        vardct.memory_plan(&linear),
        Err(EncodeError::InvalidSource(_))
    ));
    let mut sequence = mixed
        .begin_sequence(
            ImageSequenceDescriptor::new(9, 5, AnimationHeader::Still)
                .unwrap()
                .with_preview(PreviewSize::new(9, 5).unwrap()),
        )
        .unwrap();
    let baseline = context.memory_stats().reserved_bytes;
    for codec in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        assert!(matches!(
            sequence.submit_preview(linear.clone(), codec, FrameOptions::default()),
            Err(EncodeError::InvalidSource(_))
        ));
        assert!(matches!(
            sequence.submit_last_frame(linear.clone(), codec, FrameOptions::default()),
            Err(EncodeError::InvalidSource(_))
        ));
        assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
        assert_eq!(context.memory_stats().reserved_bytes, baseline);
    }
    drop(sequence);
    // Preserving the encoded RGB domain does not apply a nonlinear curve to associated color.
    let preserved = fixture
        .input(&context, YuvRgbTransfer::Preserve)
        .with_extra_channels(vec![extra])
        .unwrap();
    let mut identity = preserved.source().as_buffer().unwrap().clone();
    let ColorSpecification::Defined(ref mut color) = identity.layout.format.color_spec else {
        unreachable!()
    };
    color.transfer = TransferFunction::Linear;
    modular
        .encode(YuvImageSource::new(identity, YuvRgbTransfer::Linear).unwrap())
        .unwrap();
    modular.encode(preserved).unwrap();
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
