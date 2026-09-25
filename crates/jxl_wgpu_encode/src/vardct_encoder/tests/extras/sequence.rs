use super::*;
use crate::{
    AlphaAssociation, AnimationHeader, BlendMode, FrameBlend, FrameCrop, FrameKind, FrameOptions,
    FrameTiming, ImageSequenceDescriptor, ReferenceSlot, VarDctColorTransform,
};

fn definitions(association: AlphaAssociation) -> Vec<ExtraChannel> {
    (0..11)
        .map(|index| {
            ExtraChannel::new(
                if [1, 10].contains(&index) {
                    ExtraChannelKind::Alpha(association)
                } else {
                    ExtraChannelKind::Depth
                },
                SamplePrecision::float(32, 8).unwrap(),
                if index == 3 { 1 } else { 0 },
                Vec::new(),
            )
            .unwrap()
        })
        .collect()
}

fn input(
    context: &WgpuContext,
    extent: Extent2d,
    definitions: &[ExtraChannel],
    seed: usize,
) -> BufferImageSource {
    color_source(context, extent)
        .with_extra_channels(
            definitions
                .iter()
                .enumerate()
                .map(|(index, d)| {
                    let extent = d.source_extent(extent);
                    let words: Vec<_> = (0..extent.area().unwrap())
                        .map(|i| (0.25 + ((i + index + seed) % 4) as f32 / 8.0).to_bits())
                        .collect();
                    scalar_source(context, extent, d.precision(), &words)
                })
                .collect(),
        )
        .unwrap()
}

fn blend(mode: BlendMode, reference: u8, selector: u32) -> FrameBlend {
    FrameBlend {
        mode,
        source_reference: ReferenceSlot::new(reference).unwrap(),
        clamp: false,
        alpha_channel: if matches!(mode, BlendMode::Blend | BlendMode::MultiplyAdd) {
            selector
        } else {
            0
        },
    }
}

fn compare(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "sample {i}: {actual} vs {expected}"
        );
    }
}

#[test]
fn extra_input_sequences_select_alpha_and_references_independently_for_each_plane() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&gpu);
    for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
        let definitions = definitions(association);
        let config = VarDctConfig {
            color_transform: VarDctColorTransform::Original,
            extra_channels: definitions.clone(),
            progressive: progressive::combined(),
            ..Default::default()
        };
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
        for selector in [1, 10] {
            for mode in [
                BlendMode::Replace,
                BlendMode::Add,
                BlendMode::Blend,
                BlendMode::MultiplyAdd,
                BlendMode::Multiply,
            ] {
                let descriptor = ImageSequenceDescriptor::new(
                    9,
                    9,
                    AnimationHeader::Animation {
                        ticks_per_second_numerator: 1000.try_into().unwrap(),
                        ticks_per_second_denominator: 1.try_into().unwrap(),
                        num_loops: 2,
                        have_timecodes: true,
                    },
                )
                .unwrap();
                let mut session = encoder.begin_sequence(descriptor).unwrap();
                for index in 0..3 {
                    let extent = if index == 2 {
                        Extent2d::new(8, 8)
                    } else {
                        Extent2d::new(9, 9)
                    };
                    let options = if index == 0 {
                        FrameOptions {
                            kind: FrameKind::ReferenceOnly,
                            save_as_reference: ReferenceSlot::new(1).unwrap(),
                            ..Default::default()
                        }
                    } else {
                        FrameOptions {
                            timing: FrameTiming {
                                duration_ticks: index as u32 + 2,
                                timecode: Some(100 + index as u32),
                            },
                            crop: (index == 2).then(|| FrameCrop::new(-1, 1, 8, 8).unwrap()),
                            color_blend: blend(
                                if index == 1 { BlendMode::Add } else { mode },
                                1,
                                selector,
                            ),
                            extra_channel_blends: (0..definitions.len())
                                .map(|extra| {
                                    blend(
                                        if index == 1 { BlendMode::Add } else { mode },
                                        if index == 1 || extra % 2 == 0 { 1 } else { 3 },
                                        if extra % 2 == 0 { 1 } else { selector },
                                    )
                                })
                                .collect(),
                            save_as_reference: ReferenceSlot::new(if index == 1 { 3 } else { 0 })
                                .unwrap(),
                            ..Default::default()
                        }
                    };
                    let source = input(&context, extent, &definitions, index);
                    let job = if index == 2 {
                        session.submit_last_frame(source, options)
                    } else {
                        session.submit_frame(source, options)
                    }
                    .unwrap();
                    session.insert(job.wait().unwrap()).unwrap();
                }
                let bytes = session.finish_raw().unwrap();
                let native = extra_channels::libjxl_output(&bytes, &[]).unwrap();
                let pixels = 81;
                let stride = pixels * (4 + definitions.len());
                assert_eq!(native.len(), 2 * stride);
                for selected in [None, Some(0), Some(1), Some(3), Some(10)] {
                    let request = if let Some(index) = selected {
                        GpuOutputRequest::numeric(
                            SamplePrecision::float(32, 8).unwrap().pixel_format(),
                            jxl_wgpu_decode::NumericSampleMapping::NativeFloat,
                        )
                        .unwrap()
                        .with_extra_channel(index)
                        .unwrap()
                    } else {
                        GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
                            jxl_gpu_formats::RgbChannelOrder::Rgba,
                            false,
                            vardct_rgb8_format().color_spec,
                        ))
                        .unwrap()
                    };
                    let mut session = decoder.open(&bytes, request).unwrap();
                    for frame_index in 0..2 {
                        let frame = session.next_frame().unwrap().unwrap();
                        assert_eq!(frame.metadata.duration.ticks, frame_index as u32 + 3);
                        assert_eq!(frame.metadata.timecode, Some(frame_index as u32 + 101));
                        let output = readback.submit(frame.output()).unwrap().wait().unwrap();
                        let (offset, count) = selected.map_or((0, pixels * 4), |index| {
                            ((4 + index as usize) * pixels, pixels)
                        });
                        let start = frame_index * stride + offset;
                        compare(
                            &extra_channels::floats(&output.frame.outputs[0].bytes),
                            &native[start..start + count],
                            if selected.is_none() { 2e-4 } else { 2e-6 },
                        );
                    }
                    assert!(session.next_frame().unwrap().is_none());
                    drop(session);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                }
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn extra_input_sequence_rejects_invalid_selectors_and_inputs_without_consuming_finality() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(9, 9);
    let definitions = definitions(AlphaAssociation::Unassociated);
    let config = VarDctConfig {
        extra_channels: definitions.clone(),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let mut session = encoder
        .begin_sequence(ImageSequenceDescriptor::new(9, 9, AnimationHeader::Still).unwrap())
        .unwrap();
    let source = input(&context, extent, &definitions, 0);
    let header_bytes = context.memory_stats().reserved_bytes;
    assert!(header_bytes > 0);
    for options in [
        FrameOptions {
            color_blend: blend(BlendMode::Blend, 0, 11),
            ..Default::default()
        },
        FrameOptions {
            extra_channel_blends: vec![FrameBlend::default(); 10],
            ..Default::default()
        },
        FrameOptions {
            extra_channel_blends: vec![blend(BlendMode::MultiplyAdd, 0, 11); 11],
            ..Default::default()
        },
        FrameOptions {
            color_blend: FrameBlend {
                alpha_channel: 1,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        assert!(matches!(
            session.submit_last_frame(source.clone(), options),
            Err(EncodeError::InvalidConfiguration(_))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, header_bytes);
    }
    assert!(
        session
            .submit_last_frame(color_source(&context, extent), Default::default())
            .is_err()
    );
    let job = session
        .submit_last_frame(source, Default::default())
        .unwrap();
    let result = job.wait().unwrap();
    assert_eq!(result.frame_index.get(), 0);
    session.insert(result).unwrap();
    session.finish_raw().unwrap();
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
