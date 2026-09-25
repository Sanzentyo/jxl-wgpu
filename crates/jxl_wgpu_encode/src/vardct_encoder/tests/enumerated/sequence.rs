use super::*;
use crate::{
    FrameIndex, FrameOptions, FrameTiming, ImageSequenceDescriptor, MixedModeConfig,
    MixedModeEncoder, MixedModeFrameEncoding, VarDctTransformSelection,
};
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::gpu::planes::open_fragmented;
use jxl_test_support::oracles::extra_channels::{floats, libjxl_output};
use jxl_wgpu_decode::WgpuDecodeEngine;

#[test]
fn enumerated_sequences_share_color_across_modular_vardct_and_rejected_frames() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoders = [
        GpuDecoder::wgpu(gpu.clone()).unwrap(),
        GpuDecoder::new(
            WgpuDecodeEngine::new(gpu.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ];
    let readback = ImageReadbackPipeline::new(&gpu);
    let profiles = IccProfileOracle::compile();
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for index in 0..7 {
            let (extent, topology) = match index % 3 {
                0 => (
                    Extent2d::new(8, 8),
                    VarDctTransformSelection::Single(VarDctStrategy::Dct8),
                ),
                1 => (
                    Extent2d::new(25, 17),
                    VarDctTransformSelection::Map(mixed::packed_map(25, 17, false)),
                ),
                _ => (Extent2d::new(259, 3), VarDctTransformSelection::TiledDct8),
            };
            for (mixed, transform) in [
                (true, VarDctColorTransform::Original),
                (false, VarDctColorTransform::Original),
                (false, VarDctColorTransform::Xyb),
            ] {
                let mut config = oracle::config(channels, index * 8, transform);
                config.progressive = progressive::combined();
                if index % 2 == 1 {
                    config.sample_format = ColorSampleFormat::float(channels, 32, 8).unwrap();
                }
                let animation = AnimationHeader::Animation {
                    ticks_per_second_numerator: 60.try_into().unwrap(),
                    ticks_per_second_denominator: 1.try_into().unwrap(),
                    num_loops: 2,
                    have_timecodes: false,
                };
                let desc =
                    ImageSequenceDescriptor::new(extent.width, extent.height, animation).unwrap();
                let sources: Vec<_> = (0..3)
                    .map(|i| {
                        let mut words = input(extent, config.sample_format).0;
                        words.rotate_left(i * channels.count() as usize);
                        let source = upload(&context, extent, &config, &words, i % 2 == 1);
                        let mut wrong = source.clone();
                        wrong.layout.format.color_spec = ColorSpecification::Undefined;
                        let mut options = FrameOptions {
                            timing: FrameTiming {
                                duration_ticks: 7 + i as u32,
                                timecode: None,
                            },
                            ..Default::default()
                        };
                        // RGB reference composition exercises cross-codec image-color identity.
                        if channels == ColorChannels::Rgb && i > 0 {
                            options.color_blend = crate::FrameBlend {
                                mode: crate::BlendMode::Multiply,
                                source_reference: crate::ReferenceSlot::new(1).unwrap(),
                                clamp: true,
                            };
                        }
                        options.save_as_reference =
                            crate::ReferenceSlot::new(u8::from(i != 2)).unwrap();
                        (source, wrong, options)
                    })
                    .collect();
                let bytes = if mixed {
                    let encoder = MixedModeEncoder::new(
                        context.clone(),
                        MixedModeConfig {
                            vardct: config.clone(),
                            vardct_transform: topology.clone(),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let mut session = encoder.begin_sequence(desc.clone()).unwrap();
                    let mut jobs = Vec::new();
                    for (i, (source, wrong, options)) in sources.into_iter().enumerate() {
                        let mode = if i == 1 {
                            MixedModeFrameEncoding::VarDct
                        } else {
                            MixedModeFrameEncoding::Modular
                        };
                        let reserved = context.memory_stats().reserved_bytes;
                        assert!(session.submit_frame(wrong, mode, options.clone()).is_err());
                        assert_eq!(session.next_frame_index(), FrameIndex::new(i as u32));
                        assert_eq!(context.memory_stats().reserved_bytes, reserved);
                        jobs.push(
                            if i == 2 {
                                session.submit_last_frame(source, mode, options)
                            } else {
                                session.submit_frame(source, mode, options)
                            }
                            .unwrap(),
                        );
                    }
                    for job in jobs.into_iter().rev() {
                        session.insert(job.wait().unwrap()).unwrap();
                    }
                    session
                        .finish_indexed_container(Default::default(), Default::default())
                        .unwrap()
                } else {
                    let mut session = match &topology {
                        VarDctTransformSelection::Single(strategy) => {
                            VarDctEncoder::new_with_config(
                                context.clone(),
                                *strategy,
                                config.clone(),
                            )
                            .unwrap()
                            .begin_sequence(desc.clone())
                            .unwrap()
                        }
                        VarDctTransformSelection::Map(map) => VarDctEncoder::new_with_strategy_map(
                            context.clone(),
                            map.clone(),
                            config.clone(),
                        )
                        .unwrap()
                        .begin_sequence(desc.clone())
                        .unwrap(),
                        VarDctTransformSelection::TiledDct8 => {
                            TiledVarDctEncoder::new_with_config(context.clone(), config.clone())
                                .unwrap()
                                .begin_sequence(desc.clone())
                                .unwrap()
                        }
                    };
                    let mut jobs = Vec::new();
                    for (i, (source, wrong, options)) in sources.into_iter().enumerate() {
                        let reserved = context.memory_stats().reserved_bytes;
                        assert!(session.submit_frame(wrong, options.clone()).is_err());
                        assert_eq!(session.next_frame_index(), FrameIndex::new(i as u32));
                        assert_eq!(context.memory_stats().reserved_bytes, reserved);
                        jobs.push(
                            if i == 2 {
                                session.submit_last_frame(source, options)
                            } else {
                                session.submit_frame(source, options)
                            }
                            .unwrap(),
                        );
                    }
                    for job in jobs.into_iter().rev() {
                        session.insert(job.wait().unwrap()).unwrap();
                    }
                    session
                        .finish_indexed_container(Default::default(), Default::default())
                        .unwrap()
                };
                eprintln!("enumerated sequence {channels:?}/{index}/{mixed}/{transform:?}");
                check_header(&bytes, &config, &profiles);
                let native = libjxl_output(&bytes, &["--original"]).unwrap();
                let rust = oracle::rust_frames(&bytes, &config);
                assert_eq!(rust.len(), 3);
                let frame_samples = extent.width as usize * extent.height as usize * 4;
                assert_eq!(native.len(), 3 * frame_samples);
                let expected: Vec<_> = native.chunks_exact(frame_samples).collect();
                for (rgb, native) in rust.iter().zip(&expected) {
                    super::super::animation::compare(
                        "native/Rust enumerated sequence",
                        rgb,
                        native,
                    );
                }
                let format = PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    ColorSpecification::Defined(oracle::wire_spec(&config)),
                );
                let mut whole = Vec::new();
                for (i, decoder) in decoders.iter().enumerate() {
                    let request = GpuOutputRequest::color(format.clone()).unwrap();
                    let mut session = if i == 0 {
                        decoder.open(&bytes, request).unwrap()
                    } else {
                        open_fragmented(decoder, &bytes, request)
                    };
                    let mut actual = Vec::new();
                    while let Some(frame) = session.next_frame().unwrap() {
                        let read = readback.submit(frame.output()).unwrap().wait().unwrap();
                        actual.push(floats(&read.frame.outputs[0].bytes));
                    }
                    assert_eq!(actual.len(), expected.len());
                    for (a, b) in actual.iter().zip(&expected) {
                        super::super::animation::compare("GPU/native enumerated sequence", a, b);
                    }
                    if i == 0 {
                        whole = actual;
                    } else {
                        assert_eq!(actual, whole);
                    }
                    drop(session);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
