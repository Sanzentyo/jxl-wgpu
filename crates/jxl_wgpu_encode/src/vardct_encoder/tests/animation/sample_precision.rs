use super::*;
use crate::{
    FrameKind, MixedModeConfig, MixedModeEncoder, MixedModeFrameEncoding, RgbSequenceDescriptor,
    VarDctTransformSelection,
};
use jxl_test_support::oracles::modular_words::original_frames;

fn layers(width: usize, height: usize, animated: bool) -> Vec<Layer> {
    [
        (FrameKind::ReferenceOnly, 0, 0, BlendMode::Replace, 0, 3, 0),
        (FrameKind::Regular, -2, 1, BlendMode::Add, 3, 1, 7),
        (FrameKind::Regular, 1, -2, BlendMode::Multiply, 1, 0, 11),
    ]
    .into_iter()
    .map(|(kind, x, y, mode, source, save, duration)| Layer {
        width,
        height,
        options: FrameOptions {
            kind,
            crop: Some(FrameCrop::new(x, y, width as u32, height as u32).unwrap()),
            ..options(
                if animated { duration } else { 0 },
                None,
                mode,
                source,
                save,
            )
        },
    })
    .collect()
}

#[test]
fn integer_precision_mixed_sequences_share_depth_across_codecs_and_reference_frames() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let pixels = color::PixelOracles::new(&backend);
    for bits in [1, 9, 16, 17, 24, 31] {
        for (transform, w, h) in [
            (VarDctTransformSelection::Single(VarDctStrategy::Dct8), 8, 8),
            (
                VarDctTransformSelection::Map(mixed::packed_map(25, 17, false)),
                25,
                17,
            ),
            (VarDctTransformSelection::TiledDct8, 259, 3),
        ] {
            let config = MixedModeConfig {
                vardct: VarDctConfig {
                    progressive: progressive::combined(),
                    ..precision::configuration(bits, VarDctColorTransform::Original)
                },
                vardct_transform: transform,
                ..Default::default()
            };
            let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
            let stills = mixed_mode::Stills::new(&context, &config);
            assert_eq!(encoder.sample_format(), config.vardct.sample_format);
            for animation in [AnimationHeader::Still, timebase(60_000, 1001, 2, false)] {
                for &reference_only in if bits == 31 {
                    &[true, false][..]
                } else {
                    &[true][..]
                } {
                    eprintln!("{bits}-bit/{w}x{h}/{animation:?}");
                    let desc = RgbSequenceDescriptor::new(w as u32, h as u32, animation).unwrap();
                    let mut session = encoder.begin_sequence(desc.clone()).unwrap();
                    let mut layers = layers(w, h, animation.is_animation());
                    if !reference_only {
                        for layer in &mut layers {
                            layer.options.kind = FrameKind::Regular;
                            layer.options.crop = None;
                            layer.options.color_blend = FrameBlend::default();
                        }
                    }
                    let mut submissions = Vec::new();
                    let mut inputs = Vec::new();
                    let mut samples = Vec::new();
                    for i in 0..layers.len() {
                        let mode = if i == 1 {
                            MixedModeFrameEncoding::VarDct
                        } else {
                            MixedModeFrameEncoding::Modular
                        };
                        let input = precision::pixels(w, h, bits, i as u32 * 91);
                        let source = precision::source(&context, w, h, bits, &input, true);
                        let baseline = stills.encode(source.clone(), mode);
                        precision::check_header(&baseline, bits, VarDctColorTransform::Original);
                        samples.push(precision::check_pixels(&pixels, &baseline, &input, bits));
                        if mode == MixedModeFrameEncoding::Modular {
                            let words = original_frames(&baseline);
                            assert_eq!(words.len(), 1);
                            for (c, plane) in words[0].planes.iter().enumerate() {
                                assert_eq!(
                                    *plane,
                                    input.iter().map(|p| p[c] as i32).collect::<Vec<_>>()
                                );
                            }
                        }
                        inputs.push((source, mode));
                    }
                    for (i, ((source, mode), layer)) in inputs.into_iter().zip(&layers).enumerate()
                    {
                        // The checked common precision applies before either codec acquires memory.
                        let mut wrong = source.clone();
                        wrong.layout.format = crate::RgbSampleFormat::RGB8.pixel_format();
                        assert!(
                            session
                                .memory_plan(&wrong, mode, layer.options.clone(), i == 2)
                                .is_err()
                        );
                        assert_eq!(session.next_frame_index(), FrameIndex::new(i as u32));
                        let job = if i == 2 {
                            session.submit_last_frame(source, mode, layer.options.clone())
                        } else {
                            session.submit_frame(source, mode, layer.options.clone())
                        }
                        .unwrap();
                        submissions.push(job);
                    }
                    for job in submissions.into_iter().rev() {
                        session.insert(job.wait().unwrap()).unwrap();
                    }
                    let encoded = session
                        .finish_indexed_container(Default::default(), Default::default())
                        .unwrap();
                    precision::check_header(&encoded, bits, VarDctColorTransform::Original);
                    check_sequence_with_passes(
                        &backend,
                        &encoded,
                        &desc,
                        &layers,
                        &samples,
                        &[1, 5, 1],
                        (
                            VarDctColorTransform::Original,
                            if bits == 31 && reference_only {
                                // Both external whole-stream Rust decoders have 31-bit mixed blending
                                // limitations: oxide overflows its i32 divisor; jxl disagrees with native
                                // composition. Retain native/GPU whole-stream checks and independent
                                // composition of Rust-decoded physical stills for this combination.
                                CompositionOracle::IndependentStills
                            } else if bits == 31 {
                                // jxl-oxide 0.13 overflows (1i32 << 31) - 1 while normalizing Modular.
                                // Rust jxl independently verifies the additional full-canvas Replace streams.
                                CompositionOracle::RustJxl
                            } else {
                                CompositionOracle::JxlOxide
                            },
                        ),
                    );
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn integer_precision_vardct_sequences_bind_precision_with_both_color_domains() {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let pixels = color::PixelOracles::new(&backend);
    let (w, h) = (13, 7);
    for bits in [9, 16, 31] {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = VarDctConfig {
                progressive: progressive::combined(),
                ..precision::configuration(bits, color)
            };
            let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
            for animation in [AnimationHeader::Still, timebase(60, 1, 0, false)] {
                let desc = RgbSequenceDescriptor::new(w as u32, h as u32, animation).unwrap();
                let layers = layers(w, h, animation.is_animation());
                let mut session = encoder.begin_sequence(desc.clone()).unwrap();
                let mut samples = Vec::new();
                let mut jobs = Vec::new();
                let mut inputs = Vec::new();
                for i in 0..layers.len() {
                    let input = precision::pixels(w, h, bits, i as u32 * 31);
                    let source = precision::source(&context, w, h, bits, &input, true);
                    let baseline = encoder.encode(source.clone()).unwrap();
                    samples.push(precision::check_pixels(&pixels, &baseline, &input, bits));
                    inputs.push(source);
                }
                for (i, (source, layer)) in inputs.into_iter().zip(&layers).enumerate() {
                    jobs.push(
                        if i == 2 {
                            session.submit_last_frame(source, layer.options.clone())
                        } else {
                            session.submit_frame(source, layer.options.clone())
                        }
                        .unwrap(),
                    );
                }
                for job in jobs.into_iter().rev() {
                    session.insert(job.wait().unwrap()).unwrap();
                }
                let encoded = session
                    .finish_indexed_container(Default::default(), Default::default())
                    .unwrap();
                precision::check_header(&encoded, bits, color);
                check_sequence_with_passes(
                    &backend,
                    &encoded,
                    &desc,
                    &layers,
                    &samples,
                    &[1, 5, 5],
                    (color, CompositionOracle::JxlOxide),
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
