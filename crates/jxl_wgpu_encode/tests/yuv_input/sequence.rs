use super::*;
use jxl_test_support::{
    gpu::planes::{open_fragmented, read_bytes},
    oracles::extra_channels,
};
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping};

fn compare_gpu(
    gpu: &WgpuBackend,
    encoded: &[u8],
    color: ColorSpecification,
    displayed: Extent2d,
    frames: usize,
    extras: usize,
) {
    let native =
        extra_channels::libjxl_output(encoded, &["--original", "--preserve-alpha"]).unwrap();
    let pixels = displayed.area().unwrap();
    let stride = pixels * (4 + extras);
    assert_eq!(native.len(), frames * stride);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    for selected in std::iter::once(None).chain((0..extras).map(Some)) {
        let request = if let Some(index) = selected {
            GpuOutputRequest::numeric(
                SamplePrecision::float(32, 8).unwrap().pixel_format(),
                NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap()
        } else {
            GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                color.clone(),
            ))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        };
        let mut session = open_fragmented(&decoder, encoded, request);
        for frame in 0..frames {
            let decoded = session.next_frame().unwrap().unwrap();
            let actual = extra_channels::floats(&read_bytes(gpu, &decoded.output().outputs[0]));
            let offset = frame * stride + selected.map_or(0, |i| pixels * (4 + i));
            let expected = &native[offset..offset + actual.len()];
            for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                let bound = if selected.is_some() {
                    2e-6
                } else {
                    2e-4 * (1.0 + b.abs())
                };
                assert!(
                    a.is_finite() && (a - b).abs() <= bound,
                    "frame {frame} selected {selected:?} sample {i}: {a} vs {b}"
                );
            }
        }
        assert!(session.next_frame().unwrap().is_none());
    }
}

#[test]
fn yuv_vardct_original_and_xyb_consume_the_same_independently_checked_rgb() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let modular = LosslessModularEncoder::new(context.clone());
    for extent in [Extent2d::new(8, 8), Extent2d::new(17, 9)] {
        for linear in [false, true] {
            let fixture = case(
                extent,
                Packing::Semi(ChromaOrder::CrCb),
                ChromaSubsampling::Cs420,
                12,
                true,
                ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER),
            );
            let input = fixture.input(
                &context,
                if linear {
                    YuvRgbTransfer::Linear
                } else {
                    YuvRgbTransfer::Preserve
                },
            );
            let texture_input = fixture.texture_input(&context, input.rgb_transfer());
            let color = input.pixel_format().color_spec.clone();
            let checked = modular.encode(input.clone()).unwrap();
            let words = modular_words::channel_frames(&checked).remove(0);
            fixture.assert_rgb(&words, linear);
            let canonical = decoded_input(&context, input.pixel_format().clone(), &words);
            for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
                let cfg = VarDctConfig {
                    color_transform: transform,
                    ..config(&input)
                };
                let encoder =
                    TiledVarDctEncoder::new_with_config(context.clone(), cfg.clone()).unwrap();
                let encoded = encoder.encode(input.clone()).unwrap();
                assert_eq!(encoded, encoder.encode(canonical.clone()).unwrap());
                assert_eq!(encoded, encoder.encode(texture_input.clone()).unwrap());
                compare_gpu(&gpu, &encoded, color.clone(), extent, 1, 0);
                if extent.width == 8 {
                    let fixed =
                        VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, cfg)
                            .unwrap();
                    let encoded = fixed.encode(input.clone()).unwrap();
                    assert_eq!(encoded, fixed.encode(canonical.clone()).unwrap());
                    assert_eq!(encoded, fixed.encode(texture_input.clone()).unwrap());
                    compare_gpu(&gpu, &encoded, color.clone(), extent, 1, 0);
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn yuv_mixed_sequences_share_sampling_crop_references_and_independent_channels() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let modular = LosslessModularEncoder::new(context.clone());
    let canvas = Extent2d::new(33, 19);
    let alpha = ExtraChannel::new(
        ExtraChannelKind::Alpha(AlphaAssociation::Unassociated),
        SamplePrecision::integer(8).unwrap(),
        0,
        Vec::new(),
    )
    .unwrap();
    let depth = ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::integer(13).unwrap(),
        1,
        b"depth".to_vec(),
    )
    .unwrap();
    let mut inputs = Vec::new();
    for index in 0..3 {
        let displayed = if index == 2 {
            Extent2d::new(29, 17)
        } else {
            canvas
        };
        let factor = if index == 1 {
            UpsamplingFactor::Two
        } else {
            UpsamplingFactor::One
        };
        let fixture = case(
            factor.source_extent(displayed),
            Packing::Semi(ChromaOrder::CbCr),
            ChromaSubsampling::Cs420,
            10,
            false,
            ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER),
        );
        let input = fixture.input(&context, YuvRgbTransfer::Preserve);
        let encoded = modular.encode(input.clone()).unwrap();
        let mut words = modular_words::channel_frames(&encoded).remove(0);
        fixture.assert_rgb(&words, false);
        let canonical = decoded_input(&context, input.pixel_format().clone(), &words);
        let mut extras = Vec::new();
        for (def, upsampling) in [(&alpha, factor), (&depth, UpsamplingFactor::Four)] {
            let extent = def.source_extent_with_upsampling(displayed, upsampling);
            let scalar: Vec<u32> = (0..extent.area().unwrap())
                .map(|i| {
                    if def.precision() == SamplePrecision::integer(8).unwrap() {
                        (i * 37 + 99) as u32 % 256
                    } else {
                        (i * 991 + 777) as u32 % 8192
                    }
                })
                .collect();
            let stride = if def.precision() == SamplePrecision::integer(8).unwrap() {
                1
            } else {
                2
            };
            let mut raw: Vec<_> = scalar
                .iter()
                .flat_map(|v| v.to_le_bytes().into_iter().take(stride))
                .collect();
            raw.resize(raw.len().div_ceil(4) * 4, 0xa7);
            extras.push(upload(
                &context,
                ImageLayout::packed(extent, def.precision().pixel_format()).unwrap(),
                &raw,
            ));
            words.push(modular_integer::ExtraWords {
                width: extent.width,
                height: extent.height,
                words: scalar,
            });
        }
        inputs.push((
            input.with_extra_channels(extras.clone()).unwrap(),
            canonical.with_extra_channels(extras.clone()).unwrap(),
            fixture
                .texture_input(&context, YuvRgbTransfer::Preserve)
                .with_extra_channels(extras)
                .unwrap(),
            words,
        ));
    }
    let mut cfg = config(&inputs[0].0);
    cfg.extra_channels = vec![alpha, depth];
    let color = cfg.source_color.clone();
    let encoder = MixedModeEncoder::new(
        context.clone(),
        MixedModeConfig {
            vardct: cfg,
            ..Default::default()
        },
    )
    .unwrap();
    for reverse in [false, true] {
        let mut outputs = Vec::new();
        for storage in 0..3 {
            let mut sequence = encoder
                .begin_sequence(
                    ImageSequenceDescriptor::new(
                        canvas.width,
                        canvas.height,
                        AnimationHeader::Animation {
                            ticks_per_second_numerator: 100.try_into().unwrap(),
                            ticks_per_second_denominator: 1.try_into().unwrap(),
                            num_loops: 0,
                            have_timecodes: false,
                        },
                    )
                    .unwrap(),
                )
                .unwrap();
            for (index, (input, canonical, textures, _)) in inputs.iter().enumerate() {
                let source: GpuFrameSource = match storage {
                    0 => canonical.into(),
                    1 => input.into(),
                    _ => textures.into(),
                };
                let factor = if index == 1 {
                    UpsamplingFactor::Two
                } else {
                    UpsamplingFactor::One
                };
                let codec = if (index != 1) != reverse {
                    MixedModeFrameEncoding::Modular
                } else {
                    MixedModeFrameEncoding::VarDct
                };
                let blend = FrameBlend {
                    mode: BlendMode::Blend,
                    source_reference: ReferenceSlot::new(if index == 1 { 1 } else { 3 }).unwrap(),
                    alpha_channel: 0,
                    clamp: true,
                };
                let options = if index == 0 {
                    FrameOptions {
                        kind: FrameKind::ReferenceOnly,
                        save_as_reference: ReferenceSlot::new(1).unwrap(),
                        extra_channel_upsampling: vec![factor, UpsamplingFactor::Four],
                        ..Default::default()
                    }
                } else {
                    FrameOptions {
                        upsampling: factor,
                        extra_channel_upsampling: vec![factor, UpsamplingFactor::Four],
                        crop: (index == 2).then(|| FrameCrop::new(3, -1, 29, 17).unwrap()),
                        color_blend: blend,
                        extra_channel_blends: vec![blend; 2],
                        save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 })
                            .unwrap(),
                        timing: FrameTiming {
                            duration_ticks: 1,
                            timecode: None,
                        },
                        ..Default::default()
                    }
                };
                sequence
                    .memory_plan(&source, codec, options.clone(), index == 2)
                    .unwrap();
                let job = if index == 2 {
                    sequence.submit_last_frame(source, codec, options)
                } else {
                    sequence.submit_frame(source, codec, options)
                }
                .unwrap();
                sequence.insert(job.wait().unwrap()).unwrap();
            }
            outputs.push(sequence.finish_raw().unwrap());
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0], outputs[2]);
        for (index, (_, _, _, words)) in inputs.iter().enumerate() {
            if (index != 1) != reverse {
                assert_eq!(
                    &modular_integer::modular_channel_words(&outputs[1], index),
                    words
                );
            } else {
                assert_eq!(
                    modular_integer::vardct_extra_words(&outputs[1], index),
                    words[3..]
                );
            }
        }
        compare_gpu(&gpu, &outputs[1], color.clone(), canvas, 2, 2);
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn yuv_previews_release_conversion_storage_before_packets_are_retained() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let fixture = case(
        extent,
        Packing::Semi(ChromaOrder::CrCb),
        ChromaSubsampling::Cs420,
        16,
        true,
        ColorSpec::bt2020_ncl(ColorRange::Full, ChromaLocation2d::CENTER),
    );
    for textures in [false, true] {
        let source = || fixture.stored_input(&context, YuvRgbTransfer::Linear, textures);
        let cfg = config(&source());
        let descriptor = ImageSequenceDescriptor::new(9, 5, AnimationHeader::Still)
            .unwrap()
            .with_preview(PreviewSize::new(9, 5).unwrap());
        let mixed = MixedModeEncoder::new(
            context.clone(),
            MixedModeConfig {
                vardct: cfg.clone(),
                ..Default::default()
            },
        )
        .unwrap();
        let mut outputs = Vec::new();
        for codec in [
            MixedModeFrameEncoding::Modular,
            MixedModeFrameEncoding::VarDct,
        ] {
            let mut sequence = mixed.begin_sequence(descriptor.clone()).unwrap();
            let input = source();
            let owner = SourceOwners::of(&input);
            sequence
                .preview_memory_plan(&input, codec, FrameOptions::default())
                .unwrap();
            let baseline = context.memory_stats().reserved_bytes;
            let preview = sequence
                .submit_preview(input, codec, FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            assert!(owner.released());
            assert_eq!(
                context.memory_stats().reserved_bytes,
                baseline + preview.reserved_bytes()
            );
            sequence.insert_preview(preview).unwrap();
            let main = sequence
                .submit_last_frame(source(), codec, FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            sequence.insert(main).unwrap();
            outputs.push((
                codec == MixedModeFrameEncoding::Modular,
                sequence.finish_raw().unwrap(),
            ));
        }
        let modular = LosslessModularEncoder::new(context.clone());
        let mut sequence = modular
            .begin_sequence(
                LosslessModularSequenceDescriptor::from_pixel_format(
                    9,
                    5,
                    source().pixel_format(),
                    AnimationHeader::Still,
                )
                .unwrap()
                .with_preview(PreviewSize::new(9, 5).unwrap()),
            )
            .unwrap();
        sequence
            .preview_memory_plan(source(), FrameOptions::default())
            .unwrap();
        let preview = sequence
            .submit_preview(source(), FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert_preview(preview).unwrap();
        let main = sequence
            .submit_last_frame(source(), FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert(main).unwrap();
        outputs.push((true, sequence.finish_raw().unwrap()));
        let vardct = TiledVarDctEncoder::new_with_config(context.clone(), cfg).unwrap();
        let mut sequence = vardct.begin_sequence(descriptor).unwrap();
        sequence
            .preview_memory_plan(source(), FrameOptions::default())
            .unwrap();
        let preview = sequence
            .submit_preview(source(), FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert_preview(preview).unwrap();
        let main = sequence
            .submit_last_frame(source(), FrameOptions::default())
            .unwrap()
            .wait()
            .unwrap();
        sequence.insert(main).unwrap();
        outputs.push((false, sequence.finish_raw().unwrap()));
        for (is_modular, encoded) in outputs {
            let native =
                extra_channels::libjxl_output(&encoded, &["--preview", "--original"]).unwrap();
            assert_eq!(native.len(), extent.area().unwrap() * 4);
            if is_modular {
                let words = modular_words::original_preview(&encoded);
                assert_eq!((words.bits, words.exponent_bits), (32, 8));
                let planes: Vec<_> = words
                    .planes
                    .into_iter()
                    .map(|words| modular_integer::ExtraWords {
                        width: 9,
                        height: 5,
                        words: words.into_iter().map(|w| w as u32).collect(),
                    })
                    .collect();
                fixture.assert_rgb(&planes, true);
            }
            compare_gpu(
                &gpu,
                &encoded,
                source().pixel_format().color_spec.clone(),
                extent,
                1,
                0,
            );
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
