use super::*;
use crate::{
    FrameKind, ImageSequenceDescriptor, LosslessModularColorTransform, LosslessModularConfig,
    LosslessModularEncoder, LosslessModularEntropyCoding, LosslessModularGroupSize,
    LosslessModularPredictor, LosslessModularRctType, LosslessModularSqueeze, MixedModeConfig,
    MixedModeEncoder, MixedModeFrameEncoding, MixedModeMemoryPlan, VarDctTransformSelection,
};
use jxl_test_support::oracles::modular_words::original_frames;

const MODULAR: MixedModeFrameEncoding = MixedModeFrameEncoding::Modular;
const VARDCT: MixedModeFrameEncoding = MixedModeFrameEncoding::VarDct;

fn config(entropy: LosslessModularEntropyCoding) -> MixedModeConfig {
    MixedModeConfig {
        modular: LosslessModularConfig {
            entropy,
            group_size: LosslessModularGroupSize::Pixels128,
            color_transform: LosslessModularColorTransform::LocalRct(
                LosslessModularRctType::new(17).unwrap(),
            ),
            predictor: LosslessModularPredictor::Weighted,
            local_transforms: LosslessModularSqueeze::HorizontalThenVertical.into(),
            ..Default::default()
        },
        vardct: VarDctConfig {
            color_transform: VarDctColorTransform::Original,
            progressive: progressive::combined(),
            ..configuration()
        },
        ..Default::default()
    }
}

// Use the existing independent single-codec APIs to obtain physical-frame references.
// Whole-sequence oracles below also decode the actual mixed stream, including its controls.
pub(super) struct Stills {
    modular: LosslessModularEncoder,
    vardct: Option<VarDctEncoder>,
    tiled: Option<TiledVarDctEncoder>,
}

impl Stills {
    pub(super) fn new(context: &WgpuContext, config: &MixedModeConfig) -> Self {
        let (vardct, tiled) = match &config.vardct_transform {
            VarDctTransformSelection::Single(strategy) => (
                Some(
                    VarDctEncoder::new_with_config(
                        context.clone(),
                        *strategy,
                        config.vardct.clone(),
                    )
                    .unwrap(),
                ),
                None,
            ),
            VarDctTransformSelection::Map(map) => (
                Some(
                    VarDctEncoder::new_with_strategy_map(
                        context.clone(),
                        map.clone(),
                        config.vardct.clone(),
                    )
                    .unwrap(),
                ),
                None,
            ),
            VarDctTransformSelection::TiledDct8 => (
                None,
                Some(
                    TiledVarDctEncoder::new_with_config(context.clone(), config.vardct.clone())
                        .unwrap(),
                ),
            ),
        };
        Self {
            modular: LosslessModularEncoder::with_config(context.clone(), config.modular.clone()),
            vardct,
            tiled,
        }
    }

    pub(super) fn encode(
        &self,
        source: BufferImageSource,
        mode: MixedModeFrameEncoding,
    ) -> Vec<u8> {
        match mode {
            MixedModeFrameEncoding::Modular => self.modular.encode(source),
            MixedModeFrameEncoding::VarDct => match &self.vardct {
                Some(encoder) => encoder.encode(source),
                None => self.tiled.as_ref().unwrap().encode(source),
            },
        }
        .unwrap()
    }
}

fn layers(width: usize, height: usize, animated: bool) -> Vec<Layer> {
    [
        (FrameKind::ReferenceOnly, 0, 0, BlendMode::Replace, 0, 3, 0),
        (FrameKind::Regular, -2, 1, BlendMode::Add, 3, 0, 0),
        (FrameKind::Regular, 0, 0, BlendMode::Add, 0, 1, 7),
        (FrameKind::ReferenceOnly, 0, 0, BlendMode::Replace, 0, 2, 0),
        (FrameKind::Regular, 1, -2, BlendMode::Multiply, 2, 0, 11),
        (FrameKind::Regular, -1, 2, BlendMode::Replace, 1, 0, 0),
        (FrameKind::Regular, 0, 0, BlendMode::Multiply, 0, 0, 13),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(index, (kind, x, y, mode, source, save, duration))| Layer {
            width,
            height,
            options: FrameOptions {
                kind,
                crop: Some(FrameCrop::new(x, y, width as u32, height as u32).unwrap()),
                ..options(
                    if animated { duration } else { 0 },
                    (animated && kind == FrameKind::Regular).then_some(1000 + index as u32),
                    mode,
                    source,
                    save,
                )
            },
        },
    )
    .collect()
}

fn physical_samples(still: &[u8], input: &[[u8; 3]], mode: MixedModeFrameEncoding) -> Vec<f32> {
    let native = native_updates(still, false).expect("required native mixed-sequence oracle");
    let native = floats(&native.last().unwrap().pixels);
    let rust = rust_frame_planes(still).remove(0).0;
    compare("Rust/native physical source", &rust, &native);
    let rgb8: Vec<_> = native
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            p[..3]
                .iter()
                .map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect();
    assert!(psnr(input, &rgb8) > 30.0);
    if mode == MODULAR {
        assert_eq!(rgb8, input.iter().flatten().copied().collect::<Vec<_>>());
        let words = original_frames(still);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].planes.len(), 3);
        for (c, plane) in words[0].planes.iter().enumerate() {
            assert_eq!(
                plane,
                &input.iter().map(|p| i32::from(p[c])).collect::<Vec<_>>()
            );
        }
    }
    rust
}

#[test]
fn mixed_mode_sequences_preserve_cross_codec_references_and_indexed_presentations() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        for (selection, width, height) in [
            (VarDctTransformSelection::Single(VarDctStrategy::Dct8), 8, 8),
            (
                VarDctTransformSelection::Map(mixed::packed_map(25, 17, false)),
                25,
                17,
            ),
            (VarDctTransformSelection::TiledDct8, 259, 19),
        ] {
            let config = MixedModeConfig {
                vardct_transform: selection,
                ..config(entropy)
            };
            let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
            let stills = Stills::new(&context, &config);
            for animated in [false, true] {
                let desc = descriptor(
                    width,
                    height,
                    if animated {
                        timebase(60_000, 1001, 2, true)
                    } else {
                        AnimationHeader::Still
                    },
                );
                let layers = layers(width, height, animated);
                let mut session = encoder.begin_sequence(desc.clone()).unwrap();
                let mut submissions = Vec::new();
                let mut samples = Vec::new();
                let mut modes = Vec::new();
                let mut passes = Vec::new();
                for (index, layer) in layers.iter().enumerate() {
                    let mode = if (index + usize::from(animated)) % 2 == 0 {
                        MODULAR
                    } else {
                        VARDCT
                    };
                    let input = pixels(width, height, index);
                    let source = padded_rgb_source_sized(&context, width, height, &input);
                    samples.push(physical_samples(
                        &stills.encode(source.clone(), mode),
                        &input,
                        mode,
                    ));
                    let last = index + 1 == layers.len();
                    let plan = session
                        .memory_plan(&source, mode, layer.options.clone(), last)
                        .unwrap();
                    match plan {
                        MixedModeMemoryPlan::Modular(plan) => {
                            assert_eq!(mode, MODULAR);
                            assert_eq!(
                                plan.streaming,
                                entropy == LosslessModularEntropyCoding::Ans
                            );
                        }
                        MixedModeMemoryPlan::VarDct(_) => assert_eq!(mode, VARDCT),
                    }
                    let job = if last {
                        session.submit_last_frame(source, mode, layer.options.clone())
                    } else {
                        session.submit_frame(source, mode, layer.options.clone())
                    }
                    .unwrap();
                    submissions.push(job);
                    modes.push(mode);
                    passes.push(
                        if mode == MODULAR || layer.options.kind == FrameKind::ReferenceOnly {
                            1
                        } else {
                            5
                        },
                    );
                    assert_eq!(
                        session.next_frame_index(),
                        FrameIndex::new(index as u32 + 1)
                    );
                }
                for (index, job) in submissions.into_iter().rev().enumerate() {
                    let artifacts = if index % 2 == 0 {
                        pollster::block_on(job)
                    } else {
                        job.wait()
                    }
                    .unwrap();
                    session.insert(artifacts).unwrap();
                }
                let encoded = session
                    .finish_indexed_container(Default::default(), Default::default())
                    .unwrap();
                let inventory = jxl_gpu_bitstream::parse(&encoded, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                for (frame, mode) in inventory.frames.iter().zip(modes) {
                    assert_eq!(
                        frame.encoding,
                        if mode == MODULAR {
                            jxl_gpu_bitstream::FrameEncoding::Modular
                        } else {
                            jxl_gpu_bitstream::FrameEncoding::VarDct
                        }
                    );
                }
                check_sequence_with_passes(
                    &backend,
                    &encoded,
                    &desc,
                    &layers,
                    &samples,
                    &passes,
                    (VarDctColorTransform::Original, CompositionOracle::JxlOxide),
                );
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn mixed_mode_forced_stills_match_both_existing_encoders() {
    let context = test_context().expect("actual GPU required for mixed forced alternatives");
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        let config = config(entropy);
        let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
        let stills = Stills::new(&context, &config);
        for mode in [MODULAR, VARDCT] {
            let source = padded_rgb_source_sized(&context, 13, 7, &pixels(13, 7, 0));
            let mut session = encoder
                .begin_sequence(descriptor(13, 7, AnimationHeader::Still))
                .unwrap();
            let job = session
                .submit_last_frame(source.clone(), mode, Default::default())
                .unwrap();
            session.insert(job.wait().unwrap()).unwrap();
            assert!(matches!(
                session.submit_last_frame(source.clone(), mode, Default::default()),
                Err(EncodeError::SessionClosed)
            ));
            let encoded = session.finish_raw().unwrap();
            let baseline = stills.encode(source, mode);
            if mode == VARDCT {
                assert_eq!(encoded, baseline);
            } else {
                let header = |bytes: &[u8]| {
                    jxl_gpu_bitstream::parse(bytes, Default::default())
                        .unwrap()
                        .codestream_inventory(Default::default())
                        .unwrap()
                        .image_header
                };
                let actual = header(&encoded);
                let mut expected = header(&baseline);
                assert!(!actual.modular_16bit_buffers);
                assert!(expected.modular_16bit_buffers);
                // The mixed image reserves i32 LF buffers for its configured VarDCT codec.
                // All other metadata and the complete physical Modular frame remain identical.
                expected.modular_16bit_buffers = false;
                assert_eq!(actual, expected);
                let offset = actual.bit_range.length.div_ceil(8) as usize;
                assert_eq!(&encoded[offset..], &baseline[offset..]);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

fn drain(context: &WgpuContext) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn mixed_mode_admission_retry_cancellation_and_mode_specific_budgets() {
    let original = test_context().expect("actual GPU required for mixed ownership");
    let source = padded_rgb_source_sized(&original, 17, 9, &pixels(17, 9, 0));
    let desc = descriptor(17, 9, AnimationHeader::Still);
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        let config = config(entropy);
        let encoder = MixedModeEncoder::new(original.clone(), config.clone()).unwrap();
        let session = encoder.begin_sequence(desc.clone()).unwrap();
        for mode in [MODULAR, VARDCT] {
            let bytes = session
                .memory_plan(&source, mode, Default::default(), true)
                .unwrap()
                .owned_bytes_per_job();
            for limit in [bytes - 1, bytes] {
                let context = WgpuContext::with_memory_budget(
                    Arc::new(original.device().clone()),
                    Arc::new(original.queue().clone()),
                    NonZeroU64::new(limit).unwrap(),
                )
                .unwrap();
                let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
                let mut sequence = encoder.begin_sequence(desc.clone()).unwrap();
                let submitted =
                    sequence.submit_last_frame(source.clone(), mode, Default::default());
                if limit < bytes {
                    assert!(matches!(submitted, Err(EncodeError::MemoryBackpressure(_))));
                    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
                    assert_eq!(context.memory_stats().reserved_bytes, 0);
                    continue;
                }
                sequence.insert(submitted.unwrap().wait().unwrap()).unwrap();
                assert_eq!(
                    rust_frame_planes(&sequence.finish_container().unwrap()).len(),
                    1
                );
                drain(&context);
                // Drop the session before its independently owned job. This also exercises
                // the native streamed ANS cancellation path and subsequent encoder reuse.
                let mut abandoned = encoder.begin_sequence(desc.clone()).unwrap();
                let job = abandoned
                    .submit_last_frame(source.clone(), mode, Default::default())
                    .unwrap();
                drop(abandoned);
                drop(job);
                drain(&context);
                let mut recovered = encoder.begin_sequence(desc.clone()).unwrap();
                let job = recovered
                    .submit_last_frame(source.clone(), mode, Default::default())
                    .unwrap();
                recovered.insert(pollster::block_on(job).unwrap()).unwrap();
                assert_eq!(rust_frame_planes(&recovered.finish_raw().unwrap()).len(), 1);
                encoder.clear_buffer_pool();
                drain(&context);
            }
        }
        // A resident job pins its complete reservation until result collection. The other
        // codec must share this budget; failure must not consume the final frame/index.
        let modular_bytes = session
            .memory_plan(&source, MODULAR, Default::default(), false)
            .unwrap()
            .owned_bytes_per_job();
        let vardct_bytes = session
            .memory_plan(&source, VARDCT, Default::default(), false)
            .unwrap()
            .owned_bytes_per_job();
        let limit = modular_bytes.max(vardct_bytes);
        for (first_mode, last_mode) in if entropy == LosslessModularEntropyCoding::Prefix {
            vec![(MODULAR, VARDCT), (VARDCT, MODULAR)]
        } else {
            vec![(VARDCT, MODULAR)]
        } {
            let context = WgpuContext::with_memory_budget(
                Arc::new(original.device().clone()),
                Arc::new(original.queue().clone()),
                NonZeroU64::new(limit).unwrap(),
            )
            .unwrap();
            let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
            let mut sequence = encoder.begin_sequence(desc.clone()).unwrap();
            let first = sequence
                .submit_frame(source.clone(), first_mode, Default::default())
                .unwrap();
            assert!(matches!(
                sequence.submit_last_frame(source.clone(), last_mode, Default::default()),
                Err(EncodeError::MemoryBackpressure(_))
            ));
            assert_eq!(sequence.next_frame_index(), FrameIndex::new(1));
            sequence.insert(first.wait().unwrap()).unwrap();
            let last = sequence
                .submit_last_frame(source.clone(), last_mode, Default::default())
                .unwrap();
            sequence.insert(last.wait().unwrap()).unwrap();
            assert_eq!(rust_frame_planes(&sequence.finish_raw().unwrap()).len(), 1);
            drain(&context);
        }
    }
}

#[test]
fn mixed_mode_rejects_incompatible_contracts_before_advancing() {
    let context = test_context().expect("actual GPU required for mixed rejection");
    assert!(
        MixedModeEncoder::new(
            context.clone(),
            MixedModeConfig {
                vardct: VarDctConfig::default(),
                ..Default::default()
            }
        )
        .is_err()
    );
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct_transform: VarDctTransformSelection::Single(VarDctStrategy::Dct8),
            ..config(LosslessModularEntropyCoding::Prefix)
        },
    )
    .unwrap();
    let desc = ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still).unwrap();
    let source = padded_rgb_source_sized(&context, 8, 8, &pixels(8, 8, 0));
    for mode in [MODULAR, VARDCT] {
        let mut sequence = encoder.begin_sequence(desc.clone()).unwrap();
        let invalid = [
            FrameOptions {
                save_before_color_transform: true,
                ..Default::default()
            },
            FrameOptions {
                kind: FrameKind::ReferenceOnly,
                ..Default::default()
            },
            FrameOptions {
                timing: FrameTiming {
                    duration_ticks: 1,
                    timecode: None,
                },
                ..Default::default()
            },
            FrameOptions {
                color_blend: FrameBlend {
                    alpha_channel: 0,
                    mode: BlendMode::Blend,
                    ..Default::default()
                },
                ..Default::default()
            },
            FrameOptions {
                crop: Some(FrameCrop::new(0, 0, 7, 8).unwrap()),
                ..Default::default()
            },
        ];
        for options in invalid {
            assert!(
                sequence
                    .memory_plan(&source, mode, options.clone(), true)
                    .is_err()
            );
            assert!(
                sequence
                    .submit_last_frame(source.clone(), mode, options)
                    .is_err()
            );
            assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
        let mut unknown = source.clone();
        unknown.layout.format.color_spec = jxl_gpu_formats::ColorSpecification::Undefined;
        assert!(matches!(
            sequence.memory_plan(&unknown, mode, Default::default(), true),
            Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
        ));
        assert!(
            sequence
                .submit_last_frame(unknown, mode, Default::default())
                .is_err()
        );
        let job = sequence
            .submit_last_frame(source.clone(), mode, Default::default())
            .unwrap();
        sequence.insert(job.wait().unwrap()).unwrap();
        assert_eq!(rust_frame_planes(&sequence.finish_raw().unwrap()).len(), 1);
    }
    // Geometry belongs to the chosen codec, not the union of its capability flags.
    let larger = padded_rgb_source_sized(&context, 13, 9, &pixels(13, 9, 0));
    let crop = FrameOptions {
        crop: Some(FrameCrop::new(-2, -1, 13, 9).unwrap()),
        ..Default::default()
    };
    let mut sequence = encoder.begin_sequence(desc.clone()).unwrap();
    assert!(
        sequence
            .memory_plan(&larger, VARDCT, crop.clone(), true)
            .is_err()
    );
    assert!(
        sequence
            .submit_last_frame(larger.clone(), VARDCT, crop.clone())
            .is_err()
    );
    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
    sequence
        .memory_plan(&larger, MODULAR, crop.clone(), false)
        .unwrap();
    let job = sequence.submit_frame(larger, MODULAR, crop).unwrap();
    sequence.insert(job.wait().unwrap()).unwrap();
    let job = sequence
        .submit_last_frame(source, VARDCT, options(0, None, BlendMode::Add, 0, 0))
        .unwrap();
    sequence.insert(job.wait().unwrap()).unwrap();
    let encoded = sequence.finish_raw().unwrap();
    let native = floats(
        &native_updates(&encoded, false)
            .unwrap()
            .last()
            .unwrap()
            .pixels,
    );
    compare(
        "cropped Modular with fixed VarDCT",
        &rust_frame_planes(&encoded).remove(0).0,
        &native,
    );
    assert!(matches!(
        encoder.begin_sequence(desc.clone()).unwrap().finish_raw(),
        Err(EncodeError::MissingFinalFrame)
    ));
    let mut incomplete = encoder.begin_sequence(desc).unwrap();
    let source = padded_rgb_source_sized(&context, 8, 8, &pixels(8, 8, 0));
    let job = incomplete
        .submit_last_frame(source, MODULAR, Default::default())
        .unwrap();
    assert!(incomplete.finish_raw().is_err());
    drop(job);
    drain(&context);
}
