use super::*;
use crate::{ColorChannels, ColorSampleFormat, ProgressiveDownsampling, VarDctColorTransform};

#[test]
fn extra_input_every_scalar_precision_preserves_special_words_independently_of_color() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(9, 7);
    let precisions: Vec<_> = (1..=31)
        .map(|bits| SamplePrecision::integer(bits).unwrap())
        .chain(
            floating::all_precisions()
                .into_iter()
                .map(|p| SamplePrecision::float(p.bits(), p.exponent_bits()).unwrap()),
        )
        .collect();
    assert_eq!(precisions.len(), 185);
    let words: Vec<Vec<u32>> = precisions
        .iter()
        .map(|&precision| {
            let samples: Vec<_> = match precision.color(ColorChannels::Gray).float_precision() {
                Some(p) => {
                    let fraction = p.bits() - p.exponent_bits() - 1;
                    let sign = 1 << (p.bits() - 1);
                    let infinity = ((1 << p.exponent_bits()) - 1) << fraction;
                    let one = ((1 << (p.exponent_bits() - 1)) - 1) << fraction;
                    vec![
                        0,
                        sign,
                        1,
                        sign | 1,
                        one,
                        one - 1,
                        one | sign,
                        infinity,
                        infinity | sign,
                        infinity | 1,
                        infinity | (1 << (fraction - 1)),
                        precision.mask(),
                    ]
                }
                None => vec![
                    0,
                    1,
                    precision.mask(),
                    precision.mask() - 1,
                    precision.mask() / 3,
                ],
            };
            samples
                .into_iter()
                .cycle()
                .take(extent.area().unwrap())
                .collect()
        })
        .collect();
    let definitions: Vec<_> = precisions
        .iter()
        .map(|&p| ExtraChannel::new(ExtraChannelKind::Depth, p, 0, Vec::new()).unwrap())
        .collect();
    let attachments: Vec<_> = precisions
        .iter()
        .zip(&words)
        .map(|(&p, w)| scalar_source(&context, extent, p, w))
        .collect();
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = VarDctConfig {
                sample_format: ColorSampleFormat::integer(channels, 8).unwrap(),
                color_transform: transform,
                extra_channels: definitions.clone(),
                ..Default::default()
            };
            let source = if channels == ColorChannels::Rgb {
                color_source(&context, extent)
            } else {
                let format = config.pixel_format();
                let values: Vec<_> = (0..extent.area().unwrap())
                    .map(|i| (i * 127 % 256) as u32)
                    .collect();
                let (layout, bytes) = Packing {
                    storage: Storage::Split,
                    reversed: false,
                    shifted: false,
                }
                .pack(format, extent, &values, 17);
                BufferImageSource::new(
                    Arc::new(context.device().create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("gray color beside independent extras"),
                            contents: &bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        },
                    )),
                    layout,
                )
                .unwrap()
            }
            .with_extra_channels(attachments.clone())
            .unwrap();
            let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
            let bytes = encoder.encode(source).unwrap();
            check_words(&bytes, 0, &definitions, extent, &words);
            let (_, native) =
                extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), definitions.len())
                    .unwrap();
            for ((actual, words), &p) in native.iter().zip(&words).zip(&precisions) {
                assert_numeric(actual, words, p);
            }
            assert_eq!(context.memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn extra_input_global_prefix_lf_groups_and_progressive_passes_preserve_shifted_words() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoders = [
        GpuDecoder::wgpu(gpu.clone()).unwrap(),
        GpuDecoder::new(
            jxl_wgpu_decode::WgpuDecodeEngine::new(gpu.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ];
    let readback = ImageReadbackPipeline::new(&gpu);
    // Order is significant: the leading small plane can be global, but small planes
    // following the first large plane are coded in LF/pass groups.
    let definitions: Vec<_> = [1, 0, 3, 2, 1]
        .into_iter()
        .map(|shift| declaration(ExtraChannelKind::Depth, 13, shift))
        .collect();
    for extent in [
        Extent2d::new(1, 1),
        Extent2d::new(255, 3),
        Extent2d::new(256, 3),
        Extent2d::new(257, 3),
        Extent2d::new(259, 263),
        Extent2d::new(2048, 9),
        Extent2d::new(4097, 19),
    ] {
        let words: Vec<Vec<u32>> = definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| {
                (0..definition.source_extent(extent).area().unwrap())
                    .map(|i| ((i * 379 + index * 127) % 8192) as u32)
                    .collect()
            })
            .collect();
        let attachments = definitions
            .iter()
            .zip(&words)
            .map(|(definition, words)| {
                scalar_source(
                    &context,
                    definition.source_extent(extent),
                    definition.precision(),
                    words,
                )
            })
            .collect();
        let source = color_source(&context, extent)
            .with_extra_channels(attachments)
            .unwrap();
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
            let config = VarDctConfig {
                extra_channels: definitions.clone(),
                progressive: progression,
                ..Default::default()
            };
            let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
            let bytes = encoder.encode(source.clone()).unwrap();
            check_words(&bytes, 0, &definitions, extent, &words);
            let (_, native) =
                extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), definitions.len())
                    .unwrap();
            for (index, expected) in native.iter().enumerate() {
                let request = GpuOutputRequest::numeric(
                    jxl_gpu_formats::PixelFormat::non_color(
                        jxl_gpu_formats::SampleKind::Float,
                        32,
                        &[jxl_gpu_formats::Channel::X],
                    ),
                    jxl_wgpu_decode::NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(index as u32)
                .unwrap();
                let mut whole = None;
                for (decoder, fragmented) in decoders.iter().zip([false, true]) {
                    let mut session = if fragmented {
                        jxl_test_support::gpu::planes::open_fragmented(
                            decoder,
                            &bytes,
                            request.clone(),
                        )
                    } else {
                        decoder.open(&bytes, request.clone()).unwrap()
                    };
                    let frame = session.next_frame().unwrap().unwrap();
                    assert!(session.next_frame().unwrap().is_none());
                    drop(session);
                    let output = readback.submit(frame.output()).unwrap().wait().unwrap();
                    let raw = &output.frame.outputs[0].bytes;
                    if let Some(whole) = &whole {
                        assert_eq!(raw, whole);
                    } else {
                        whole = Some(raw.clone());
                    }
                    let actual = extra_channels::floats(raw);
                    assert_eq!(actual.len(), expected.len());
                    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                        assert!(
                            (actual - expected).abs() <= 2e-6,
                            "{extent:?}/extra {index}/pixel {i}: {actual} vs {expected}"
                        );
                    }
                    drop(frame);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}
