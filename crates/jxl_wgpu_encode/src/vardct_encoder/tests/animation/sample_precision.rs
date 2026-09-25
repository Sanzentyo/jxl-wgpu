use super::*;
use crate::RgbSampleFormat;
use crate::{
    FrameKind, MixedModeConfig, MixedModeEncoder, MixedModeFrameEncoding, RgbSequenceDescriptor,
    VarDctTransformSelection,
};
use jxl_gpu_formats::SampleKind;
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

fn check_mixed_sequences(formats: impl IntoIterator<Item = RgbSampleFormat>) {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let pixels = color::PixelOracles::new(&backend);
    for sample in formats {
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
                    sample_format: sample,
                    ..precision::configuration(8, VarDctColorTransform::Original)
                },
                vardct_transform: transform,
                ..Default::default()
            };
            let encoder = MixedModeEncoder::new(context.clone(), config.clone()).unwrap();
            let stills = mixed_mode::Stills::new(&context, &config);
            assert_eq!(encoder.sample_format(), config.vardct.sample_format);
            for animation in [AnimationHeader::Still, timebase(60_000, 1001, 2, false)] {
                for &reference_only in if sample == RgbSampleFormat::integer(31).unwrap()
                    || sample.float_precision().is_some()
                {
                    &[true, false][..]
                } else {
                    &[true][..]
                } {
                    eprintln!("{sample:?}/{w}x{h}/{animation:?}");
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
                        let input = source_samples(w, h, sample, i as u32 * 91);
                        let source = input_source(&context, w, h, sample, &input);
                        let baseline = stills.encode(source.clone(), mode);
                        assert_header(&baseline, sample, VarDctColorTransform::Original);
                        samples.push(check_pixels(&pixels, &baseline, &input, sample));
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
                    assert_header(&encoded, sample, VarDctColorTransform::Original);
                    check_sequence_with_passes(
                        &backend,
                        &encoded,
                        &desc,
                        &layers,
                        &samples,
                        &[1, 5, 1],
                        (
                            VarDctColorTransform::Original,
                            if sample.float_precision().is_some() && reference_only {
                                // Whole-stream oxide and Rust jxl disagree with native floating
                                // mixed blending. Preserve these cases with native/GPU output and
                                // independent Rust-decoded physical-still composition. Extra
                                // full-canvas Replace streams retain whole-stream Rust jxl checks.
                                CompositionOracle::IndependentStills
                            } else if sample.float_precision().is_some() {
                                CompositionOracle::RustJxl
                            } else if sample == RgbSampleFormat::integer(31).unwrap()
                                && reference_only
                            {
                                // Both external whole-stream Rust decoders have 31-bit mixed blending
                                // limitations: oxide overflows its i32 divisor; jxl disagrees with native
                                // composition. Retain native/GPU whole-stream checks and independent
                                // composition of Rust-decoded physical stills for this combination.
                                CompositionOracle::IndependentStills
                            } else if sample == RgbSampleFormat::integer(31).unwrap() {
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

fn check_vardct_sequences(formats: impl IntoIterator<Item = RgbSampleFormat>) {
    let backend = backend();
    let context = WgpuContext::from_backend(&backend);
    let pixels = color::PixelOracles::new(&backend);
    let (w, h) = (13, 7);
    for sample in formats {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = VarDctConfig {
                progressive: progressive::combined(),
                sample_format: sample,
                ..precision::configuration(8, color)
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
                    let input = source_samples(w, h, sample, i as u32 * 31);
                    let source = input_source(&context, w, h, sample, &input);
                    let baseline = encoder.encode(source.clone()).unwrap();
                    samples.push(check_pixels(&pixels, &baseline, &input, sample));
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
                assert_header(&encoded, sample, color);
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

fn float_formats() -> Vec<RgbSampleFormat> {
    [(5, 2), (16, 5), (24, 7), (32, 8)]
        .map(|(bits, exponent)| RgbSampleFormat::float(bits, exponent).unwrap())
        .to_vec()
}

fn source_samples(w: usize, h: usize, format: RgbSampleFormat, seed: u32) -> Vec<[u32; 3]> {
    match format.float_precision() {
        None => precision::pixels(w, h, format.bits_per_sample(), seed),
        Some(p) => {
            let mut pixels = floating::pixels(w, h, p);
            let offset = seed as usize % pixels.len();
            pixels.rotate_left(offset);
            pixels
        }
    }
}

fn input_source(
    context: &WgpuContext,
    w: usize,
    h: usize,
    format: RgbSampleFormat,
    words: &[[u32; 3]],
) -> BufferImageSource {
    precision::source_with_kind(
        context,
        w,
        h,
        format.bits_per_sample(),
        format
            .float_precision()
            .map_or(SampleKind::Unsigned, SampleKind::CustomFloat),
        words,
        true,
    )
}

fn assert_header(encoded: &[u8], format: RgbSampleFormat, color: VarDctColorTransform) {
    match format.float_precision() {
        None => precision::check_header(encoded, format.bits_per_sample(), color),
        Some(p) => floating::check_header(encoded, p),
    }
}

fn check_pixels(
    oracles: &color::PixelOracles,
    encoded: &[u8],
    words: &[[u32; 3]],
    format: RgbSampleFormat,
) -> Vec<f32> {
    match format.float_precision() {
        None => precision::check_pixels(oracles, encoded, words, format.bits_per_sample()),
        Some(p) => floating::check_pixels(oracles, encoded, words, p),
    }
}

#[test]
fn integer_precision_mixed_sequences_share_depth_across_codecs_and_reference_frames() {
    check_mixed_sequences(
        [1, 9, 16, 17, 24, 31].map(|bits| RgbSampleFormat::integer(bits).unwrap()),
    );
}

#[test]
fn integer_precision_vardct_sequences_bind_precision_with_both_color_domains() {
    check_vardct_sequences([9, 16, 31].map(|bits| RgbSampleFormat::integer(bits).unwrap()));
}

#[test]
fn floating_precision_mixed_sequences_share_precision_across_codecs_and_reference_frames() {
    check_mixed_sequences(float_formats());
}

#[test]
fn floating_precision_vardct_sequences_bind_precision_with_both_color_domains() {
    check_vardct_sequences(float_formats());
}
