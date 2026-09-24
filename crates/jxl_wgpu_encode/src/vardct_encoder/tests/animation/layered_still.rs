use super::*;
use crate::{FrameKind, VarDctSequenceDescriptor};

fn layers(width: usize, height: usize) -> Vec<Layer> {
    [
        (FrameKind::ReferenceOnly, 0, 0, BlendMode::Replace, 0, 3),
        (FrameKind::Regular, -2, 1, BlendMode::Add, 3, 0),
        (
            FrameKind::Regular,
            width as i32 + 3,
            0,
            BlendMode::Replace,
            0,
            1,
        ),
        (FrameKind::Regular, 1, -2, BlendMode::Multiply, 1, 0),
    ]
    .into_iter()
    .map(|(kind, x, y, mode, source, save)| Layer {
        width,
        height,
        options: FrameOptions {
            kind,
            crop: Some(FrameCrop::new(x, y, width as u32, height as u32).unwrap()),
            ..options(0, None, mode, source, save)
        },
    })
    .collect()
}

#[test]
fn layered_still_vardct_composes_reference_and_cropped_layers_across_all_backends() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let config = VarDctConfig {
        progressive: progressive::combined(),
        ..configuration()
    };
    for (encoder, width, height) in [
        (
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
                .unwrap(),
            8,
            8,
        ),
        (
            VarDctEncoder::new_with_strategy_map(
                context.clone(),
                mixed::packed_map(25, 17, false),
                config.clone(),
            )
            .unwrap(),
            25,
            17,
        ),
    ] {
        let desc = VarDctSequenceDescriptor::new(
            width as u32 - 1,
            height as u32 - 1,
            AnimationHeader::Still,
        )
        .unwrap();
        assert!(matches!(
            encoder.begin_animation(desc.clone()),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        let layers = layers(width, height);
        let (encoded, samples) = encode_layers(
            &context,
            encoder.begin_sequence(desc.clone()).unwrap(),
            &layers,
            |source| encoder.encode(source).unwrap(),
            true,
        );
        check_sequence_with_oracle(
            &backend,
            &encoded,
            &desc,
            &layers,
            &samples,
            5,
            CompositionOracle::IndependentStills,
        );
    }
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let desc = VarDctSequenceDescriptor::new(257, 17, AnimationHeader::Still).unwrap();
    let mut layers = layers(259, 19);
    layers[1].width = 9;
    layers[1].height = 7;
    layers[1].options.crop = Some(FrameCrop::new(-2, 1, 9, 7).unwrap());
    let (encoded, samples) = encode_layers(
        &context,
        encoder.begin_sequence(desc.clone()).unwrap(),
        &layers,
        |source| encoder.encode(source).unwrap(),
        true,
    );
    check_sequence_with_oracle(
        &backend,
        &encoded,
        &desc,
        &layers,
        &samples,
        5,
        CompositionOracle::IndependentStills,
    );

    // A cropped final frame without a prefix is also a still, with an initially empty canvas.
    let desc = VarDctSequenceDescriptor::new(17, 9, AnimationHeader::Still).unwrap();
    let layers = [Layer {
        width: 9,
        height: 7,
        options: FrameOptions {
            crop: Some(FrameCrop::new(-2, 1, 9, 7).unwrap()),
            ..Default::default()
        },
    }];
    let (encoded, samples) = encode_layers(
        &context,
        encoder.begin_sequence(desc.clone()).unwrap(),
        &layers,
        |source| encoder.encode(source).unwrap(),
        false,
    );
    check_sequence_with_oracle(
        &backend,
        &encoded,
        &desc,
        &layers,
        &samples,
        5,
        CompositionOracle::IndependentStills,
    );

    let source = padded_rgb_source_sized(&context, 17, 9, &pixels(17, 9, 0));
    let baseline = encoder.encode(source.clone()).unwrap();
    let mut sequence = encoder.begin_sequence(desc).unwrap();
    let job = sequence
        .submit_last_frame(source, Default::default())
        .unwrap();
    sequence.insert(job.wait().unwrap()).unwrap();
    assert_eq!(sequence.finish_raw().unwrap(), baseline);
}

#[test]
fn layered_still_vardct_invalid_timing_final_retry_and_cancellation_keep_ownership() {
    let context = test_context().expect("actual GPU required");
    let source = padded_rgb_source_sized(&context, 9, 7, &pixels(9, 7, 0));
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
    for limit in [bytes - 1, bytes] {
        let context = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
        let desc = VarDctSequenceDescriptor::new(9, 7, AnimationHeader::Still).unwrap();
        assert!(matches!(
            encoder.begin_animation(desc.clone()),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        let mut sequence = encoder.begin_sequence(desc.clone()).unwrap();
        for timing in [
            FrameTiming {
                duration_ticks: 1,
                timecode: None,
            },
            FrameTiming {
                duration_ticks: 0,
                timecode: Some(0),
            },
        ] {
            assert!(matches!(
                sequence.submit_last_frame(
                    source.clone(),
                    FrameOptions {
                        timing,
                        ..Default::default()
                    }
                ),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
        let pending = sequence.submit_frame(source.clone(), Default::default());
        if limit < bytes {
            assert!(matches!(pending, Err(EncodeError::MemoryBackpressure(_))));
            assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
            continue;
        }
        let pending = pending.unwrap();
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, bytes);
        assert!(matches!(
            sequence.submit_last_frame(source.clone(), Default::default()),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        assert_eq!(sequence.next_frame_index(), FrameIndex::new(1));
        sequence.insert(pending.wait().unwrap()).unwrap();
        let last = sequence
            .submit_last_frame(source.clone(), Default::default())
            .unwrap();
        sequence.insert(last.wait().unwrap()).unwrap();
        let raw = sequence.finish_raw().unwrap();
        assert_eq!(
            native_updates(&raw, false)
                .unwrap()
                .iter()
                .filter(|u| u.complete)
                .count(),
            1
        );
        let mut abandoned = encoder.begin_sequence(desc.clone()).unwrap();
        let job = abandoned
            .submit_frame(source.clone(), Default::default())
            .unwrap();
        drop(abandoned);
        drop(job);
        context
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while encoder.in_flight_memory_stats().reserved_bytes != 0
            && std::time::Instant::now() < deadline
        {
            context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let mut recovered = encoder.begin_sequence(desc).unwrap();
        let job = recovered
            .submit_frame(source.clone(), Default::default())
            .unwrap();
        recovered.insert(job.wait().unwrap()).unwrap();
        assert!(matches!(
            recovered.finish_raw(),
            Err(EncodeError::MissingFinalFrame)
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
