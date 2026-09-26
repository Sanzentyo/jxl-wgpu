use super::*;
use crate::{
    AnimationHeader, FrameOptions, GpuEncodeBackend, GpuEncoder, GpuFrameSource,
    ImageSequenceDescriptor, ProgressiveDownsampling, UpsamplingFactor, VarDctBackend,
};
use jxl_test_support::oracles::resampling::{Arithmetic, Plane, Sample};

mod sequence;

const FACTORS: [UpsamplingFactor; 4] = [
    UpsamplingFactor::One,
    UpsamplingFactor::Two,
    UpsamplingFactor::Four,
    UpsamplingFactor::Eight,
];

fn input(
    context: &WgpuContext,
    extent: Extent2d,
    definitions: &[ExtraChannel],
    factors: &[UpsamplingFactor],
    seed: usize,
) -> (BufferImageSource, Vec<modular_integer::ExtraWords>) {
    let expected: Vec<_> = definitions
        .iter()
        .zip(factors)
        .enumerate()
        .map(|(index, (d, f))| {
            let divisor = (1 << d.dimension_shift()) * f.factor();
            let width = extent.width.div_ceil(divisor);
            let height = extent.height.div_ceil(divisor);
            modular_integer::ExtraWords {
                width,
                height,
                words: (0..width as usize * height as usize)
                    .map(|i| {
                        let word = ((i * 379 + index * 127 + seed * 257) % 8192) as u32;
                        if d.precision()
                            .color(crate::ColorChannels::Gray)
                            .exponent_bits()
                            == 0
                        {
                            word & d.precision().mask()
                        } else {
                            (word as f32 / 16384.0 + 0.25).to_bits()
                        }
                    })
                    .collect(),
            }
        })
        .collect();
    let attachments = expected
        .iter()
        .zip(definitions)
        .map(|(words, d)| {
            scalar_source(
                context,
                Extent2d::new(words.width, words.height),
                d.precision(),
                &words.words,
            )
        })
        .collect();
    (
        color_source(context, extent)
            .with_extra_channels(attachments)
            .unwrap(),
        expected,
    )
}

fn check_header(
    bytes: &[u8],
    index: usize,
    definitions: &[ExtraChannel],
    factors: &[UpsamplingFactor],
) {
    let image = jxl_oxide::JxlImage::read_with_defaults(bytes).unwrap();
    let frame = image.frame(index).unwrap().header();
    // The independent parser exposes wire factors separately from image-header dim_shift.
    assert_eq!(
        frame.ec_upsampling,
        factors.iter().map(|f| f.factor()).collect::<Vec<_>>()
    );
    assert_eq!(frame.upsampling, 1);
    assert_eq!(
        image.image_header().metadata.ec_info.len(),
        definitions.len()
    );
    for (parsed, d) in image
        .image_header()
        .metadata
        .ec_info
        .iter()
        .zip(definitions)
    {
        assert_eq!(parsed.dim_shift, u32::from(d.dimension_shift()));
    }
}

fn render_extra(
    decoder: &GpuDecoder<jxl_wgpu_decode::WgpuDecodeEngine>,
    readback: &ImageReadbackPipeline,
    bytes: &[u8],
    index: usize,
    expected: &[Sample],
    mapping: jxl_wgpu_decode::NumericSampleMapping,
    fragmented: bool,
) -> Vec<u8> {
    let request = GpuOutputRequest::numeric(
        SamplePrecision::float(32, 8).unwrap().pixel_format(),
        mapping,
    )
    .unwrap()
    .with_extra_channel(index as u32)
    .unwrap();
    let mut session = if fragmented {
        jxl_test_support::gpu::planes::open_fragmented(decoder, bytes, request)
    } else {
        decoder.open(bytes, request).unwrap()
    };
    let frame = session.next_frame().unwrap().unwrap();
    let output = readback.submit(frame.output()).unwrap().wait().unwrap();
    let raw = output.frame.outputs[0].bytes.clone();
    let actual = extra_channels::floats(&raw);
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        expected.check(actual, &format!("GPU extra {index}"), i);
    }
    assert!(session.next_frame().unwrap().is_none());
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    raw
}

#[test]
fn extra_sampling_all_intrinsic_and_frame_factors_route_exact_words_and_render_independent_pixels()
{
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let fragmented = GpuDecoder::new(
        jxl_wgpu_decode::WgpuDecodeEngine::new(gpu.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let readback = ImageReadbackPipeline::new(&gpu);
    // A leading small global channel, then a full-size channel that terminates the prefix,
    // followed by every intrinsic/frame pair. Later small planes must retain LF/pass routing.
    let mut definitions = vec![declaration(ExtraChannelKind::Depth, 13, 3)];
    let mut factors = vec![UpsamplingFactor::Eight];
    for shift in 0..=3 {
        for factor in FACTORS {
            definitions.push(declaration(ExtraChannelKind::Depth, 13, shift));
            factors.push(factor);
        }
    }
    definitions[16] = ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::float(32, 8).unwrap(),
        3,
        Vec::new(),
    )
    .unwrap();
    let (native_definitions, native_factors): (Vec<_>, Vec<_>) = definitions
        .iter()
        .zip(&factors)
        .filter(|(d, f)| f.factor() << d.dimension_shift() <= 8)
        .map(|(d, f)| (d.clone(), *f))
        .unzip();
    for (definitions, factors, native_compatible) in [
        (definitions, factors, false),
        (native_definitions, native_factors, true),
    ] {
        for extent in [
            Extent2d::new(1, 1),
            Extent2d::new(255, 3),
            Extent2d::new(256, 3),
            Extent2d::new(257, 3),
            Extent2d::new(259, 263),
            Extent2d::new(2048, 9),
            Extent2d::new(2049, 17),
            Extent2d::new(4097, 19),
        ] {
            let (source, words) = input(&context, extent, &definitions, &factors, 0);
            for progression in [
                crate::ProgressivePlan::single(),
                progressive::combined()
                    .with_downsampling(vec![
                        ProgressiveDownsampling {
                            factor: 4,
                            last_pass: 0,
                        },
                        ProgressiveDownsampling {
                            factor: 2,
                            last_pass: 2,
                        },
                        ProgressiveDownsampling {
                            factor: 1,
                            last_pass: 3,
                        },
                    ])
                    .unwrap(),
            ] {
                let encoder = TiledVarDctEncoder::new_with_config(
                    context.clone(),
                    VarDctConfig {
                        extra_channels: definitions.clone(),
                        progressive: progression,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut session = encoder
                    .begin_sequence(
                        ImageSequenceDescriptor::new(
                            extent.width,
                            extent.height,
                            AnimationHeader::Still,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let job = session
                    .submit_last_frame(
                        source.clone(),
                        FrameOptions {
                            extra_channel_upsampling: factors.clone(),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                session.insert(job.wait().unwrap()).unwrap();
                let bytes = session.finish_raw().unwrap();
                check_header(&bytes, 0, &definitions, &factors);
                assert_eq!(modular_integer::vardct_extra_words(&bytes, 0), words);
                let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let expected = |index: usize, arithmetic| {
                    let d = &definitions[index];
                    let words = &words[index];
                    Plane {
                        width: words.width as usize,
                        height: words.height as usize,
                        samples: words
                            .words
                            .iter()
                            .map(|&word| {
                                Sample::decoded(word, d.precision().bit_depth(), arithmetic)
                            })
                            .collect(),
                    }
                    .reconstruct(
                        factors[index].factor() << d.dimension_shift(),
                        extent.width as usize,
                        extent.height as usize,
                        &inventory.image_header.upsampling_weights,
                        arithmetic,
                    )
                };
                // libjxl's renderer supports effective factors <= 8. Extended factors retain
                // exact independent Modular words, the existing F64 intervals and GPU rendering.
                let native = native_compatible.then(|| {
                    extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), definitions.len())
                        .unwrap()
                        .1
                });
                if let Some(native) = &native {
                    for (index, actual) in native.iter().enumerate() {
                        for (pixel, (&actual, reference)) in actual
                            .iter()
                            .zip(expected(index, Arithmetic::Native).samples)
                            .enumerate()
                        {
                            reference.check(actual, "native sampling", pixel);
                        }
                    }
                }
                // jxl-render's known single-sample padding defect does not affect its raw-word
                // decoder above. Thin rendered grids use the independent repeated-mirror oracle.
                if words
                    .iter()
                    .all(|plane| plane.width >= 2 && plane.height >= 2)
                {
                    let image = jxl_oxide::JxlImage::read_with_defaults(bytes.as_slice()).unwrap();
                    let rendered = image.render_frame(0).unwrap();
                    let pixels = rendered.image_all_channels();
                    assert_eq!(pixels.channels(), 3 + definitions.len());
                    assert_eq!(
                        pixels.buf().len(),
                        extent.area().unwrap() * pixels.channels()
                    );
                    for index in 0..definitions.len() {
                        let reference = expected(index, Arithmetic::Rust);
                        for (pixel, (actual, reference)) in pixels
                            .buf()
                            .chunks_exact(pixels.channels())
                            .zip(reference.samples)
                            .enumerate()
                        {
                            reference.check(actual[3 + index], "jxl-oxide sampling", pixel);
                        }
                    }
                }
                // Full interpolation is covered at both odd 2D pass boundaries and an LF boundary,
                // including every effective factor 1..64 plus floating samples.
                if [
                    Extent2d::new(1, 1),
                    Extent2d::new(259, 263),
                    Extent2d::new(4097, 19),
                ]
                .contains(&extent)
                {
                    let selected: &[usize] = if native_compatible {
                        &[0, 1, 2, 3]
                    } else {
                        &[1, 2, 3, 4, 8, 12, 0, 16]
                    };
                    for &index in selected {
                        let reference = expected(index, Arithmetic::Wgsl);
                        let mapping = if definitions[index]
                            .precision()
                            .color(crate::ColorChannels::Gray)
                            .exponent_bits()
                            == 0
                        {
                            jxl_wgpu_decode::NumericSampleMapping::NormalizedUnsigned
                        } else {
                            jxl_wgpu_decode::NumericSampleMapping::NativeFloat
                        };
                        let whole = render_extra(
                            &decoder,
                            &readback,
                            &bytes,
                            index,
                            &reference.samples,
                            mapping,
                            false,
                        );
                        if let Some(native) = &native {
                            for (actual, &reference) in
                                extra_channels::floats(&whole).iter().zip(&native[index])
                            {
                                assert!((actual - reference).abs() <= 2e-6);
                            }
                        }
                        if index == 0 {
                            assert_eq!(
                                whole,
                                render_extra(
                                    &fragmented,
                                    &readback,
                                    &bytes,
                                    index,
                                    &reference.samples,
                                    mapping,
                                    true
                                )
                            );
                        }
                    }
                }
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn extra_sampling_request_admission_uses_reduced_sources_and_releases_cancelled_ownership() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(259, 263);
    let definitions = vec![declaration(ExtraChannelKind::Depth, 13, 1)];
    let factors = [UpsamplingFactor::Four];
    let config = VarDctConfig {
        extra_channels: definitions.clone(),
        ..Default::default()
    };
    let (source, _) = input(&context, extent, &definitions, &factors, 0);
    let (full_source, _) = input(&context, extent, &definitions, &[UpsamplingFactor::One], 0);
    let backend = VarDctBackend::new_tiled_dct8_with_config(&context, config.clone()).unwrap();
    let mut request = layouts::request(extent, &config);
    request.options.extra_channel_upsampling = factors.to_vec();
    let memory = backend.memory_plan_for_request(&source, &request).unwrap();
    assert!(
        memory.owned_bytes_per_job
            < backend
                .memory_plan(&full_source)
                .unwrap()
                .owned_bytes_per_job
    );
    assert!(backend.memory_plan(&source).is_err()); // still/default request is factor one
    assert!(
        backend
            .memory_plan_for_request(&full_source, &request)
            .is_err()
    );
    for limit in [memory.owned_bytes_per_job - 1, memory.owned_bytes_per_job] {
        let bounded = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(limit).unwrap(),
        )
        .unwrap();
        let backend = VarDctBackend::new_tiled_dct8_with_config(&bounded, config.clone()).unwrap();
        let encoder = GpuEncoder::new(bounded.clone(), backend);
        let result = encoder.submit_frame(GpuFrameSource::Buffer(source.clone()), request.clone());
        if limit < memory.owned_bytes_per_job {
            assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
        } else {
            drop(result.unwrap());
            boundaries::drain(&bounded);
            encoder
                .submit_frame(GpuFrameSource::Buffer(source.clone()), request.clone())
                .unwrap()
                .wait()
                .unwrap();
        }
        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
    }
    let weak = Arc::downgrade(&source.extra_channels()[0].buffer);
    drop(
        backend
            .submit(&context, GpuFrameSource::Buffer(source), &request)
            .unwrap(),
    );
    boundaries::drain(&context);
    assert!(weak.upgrade().is_none());
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let mut session = encoder
        .begin_sequence(
            ImageSequenceDescriptor::new(extent.width, extent.height, AnimationHeader::Still)
                .unwrap(),
        )
        .unwrap();
    let header_bytes = context.memory_stats().reserved_bytes;
    let (source, _) = input(&context, extent, &definitions, &factors, 0);
    for options in [
        FrameOptions::default(),
        FrameOptions {
            extra_channel_upsampling: vec![UpsamplingFactor::Four; 2],
            ..Default::default()
        },
    ] {
        assert!(session.submit_last_frame(source.clone(), options).is_err());
        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
    }
    let result = session
        .submit_last_frame(source, request.options)
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(result.frame_index.get(), 0);
    session.insert(result).unwrap();
    session.finish_raw().unwrap();
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn extra_sampling_maximum_channel_table_retains_every_factor_and_source_identity() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(1, 1);
    let definitions: Vec<_> = (0..256)
        .map(|index| declaration(ExtraChannelKind::Depth, 13, (index / 4 % 4) as u8))
        .collect();
    let factors: Vec<_> = (0..256).map(|index| FACTORS[index % 4]).collect();
    let (source, words) = input(&context, extent, &definitions, &factors, 0);
    let encoder = TiledVarDctEncoder::new_with_config(
        context.clone(),
        VarDctConfig {
            extra_channels: definitions.clone(),
            ..Default::default()
        },
    )
    .unwrap();
    let mut session = encoder
        .begin_sequence(ImageSequenceDescriptor::new(1, 1, AnimationHeader::Still).unwrap())
        .unwrap();
    let retained = context.memory_stats().reserved_bytes;
    for count in [255, 257] {
        assert!(matches!(
            session.submit_last_frame(
                source.clone(),
                FrameOptions {
                    extra_channel_upsampling: vec![UpsamplingFactor::One; count],
                    ..Default::default()
                }
            ),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, retained);
    }
    let result = session
        .submit_last_frame(
            source,
            FrameOptions {
                extra_channel_upsampling: factors.clone(),
                ..Default::default()
            },
        )
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(result.frame_index.get(), 0);
    session.insert(result).unwrap();
    let bytes = session.finish_raw().unwrap();
    check_header(&bytes, 0, &definitions, &factors);
    assert_eq!(modular_integer::vardct_extra_words(&bytes, 0), words);
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
