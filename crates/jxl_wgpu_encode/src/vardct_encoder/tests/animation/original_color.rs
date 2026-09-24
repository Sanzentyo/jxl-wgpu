use super::*;
use crate::{FrameKind, VarDctSequenceDescriptor};

fn layers(width: usize, height: usize, animated: bool) -> Vec<Layer> {
    [
        (FrameKind::ReferenceOnly, 0, 0, BlendMode::Replace, 0, 3, 0),
        (
            FrameKind::Regular,
            -2,
            1,
            BlendMode::Add,
            3,
            1,
            if animated { 7 } else { 0 },
        ),
        (
            FrameKind::Regular,
            1,
            -2,
            BlendMode::Multiply,
            1,
            0,
            if animated { 11 } else { 0 },
        ),
    ]
    .into_iter()
    .map(|(kind, x, y, mode, source, save, duration)| Layer {
        width,
        height,
        options: FrameOptions {
            kind,
            crop: Some(FrameCrop::new(x, y, width as u32, height as u32).unwrap()),
            ..options(duration, None, mode, source, save)
        },
    })
    .collect()
}

#[test]
fn original_rgb_sequences_bind_color_across_all_backends_and_frame_kinds() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let config = VarDctConfig {
        color_transform: VarDctColorTransform::Original,
        progressive: progressive::combined(),
        ..configuration()
    };
    for animation in [AnimationHeader::Still, timebase(60_000, 1001, 2, false)] {
        for (encoder, width, height) in [
            (
                VarDctEncoder::new_with_config(
                    context.clone(),
                    VarDctStrategy::Dct8,
                    config.clone(),
                )
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
            let desc =
                VarDctSequenceDescriptor::new(width as u32, height as u32, animation).unwrap();
            let layers = layers(width, height, animation.is_animation());
            let (encoded, samples) = encode_layers(
                &context,
                encoder.begin_sequence(desc.clone()).unwrap(),
                &layers,
                |source| encoder.encode(source).unwrap(),
                true,
            );
            check_sequence_in_domain(
                &backend,
                &encoded,
                &desc,
                &layers,
                &samples,
                5,
                (VarDctColorTransform::Original, CompositionOracle::JxlOxide),
            );
        }
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        let desc = VarDctSequenceDescriptor::new(259, 19, animation).unwrap();
        let layers = layers(259, 19, animation.is_animation());
        let (encoded, samples) = encode_layers(
            &context,
            encoder.begin_sequence(desc.clone()).unwrap(),
            &layers,
            |source| encoder.encode(source).unwrap(),
            true,
        );
        check_sequence_in_domain(
            &backend,
            &encoded,
            &desc,
            &layers,
            &samples,
            5,
            (VarDctColorTransform::Original, CompositionOracle::JxlOxide),
        );
    }
    // A descriptor has no implicit XYB color contract: the same checked geometry can be
    // bound to either encoder, and a one-frame sequence must equal its still convenience API.
    let desc = VarDctSequenceDescriptor::new(13, 7, AnimationHeader::Still).unwrap();
    let input = padded_rgb_source_sized(&context, 13, 7, &pixels(13, 7, 0));
    let mut streams = Vec::new();
    for color_transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
        let encoder = TiledVarDctEncoder::new_with_config(
            context.clone(),
            VarDctConfig {
                color_transform,
                ..config.clone()
            },
        )
        .unwrap();
        let mut session = encoder.begin_sequence(desc.clone()).unwrap();
        let job = session
            .submit_last_frame(input.clone(), FrameOptions::default())
            .unwrap();
        session.insert(job.wait().unwrap()).unwrap();
        let stream = session.finish_raw().unwrap();
        assert_eq!(stream, encoder.encode(input.clone()).unwrap());
        streams.push(stream);
    }
    assert_ne!(streams[0], streams[1]);
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
