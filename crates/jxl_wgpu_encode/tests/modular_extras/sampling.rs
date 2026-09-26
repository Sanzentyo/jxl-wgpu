use super::*;
use jxl_test_support::oracles::resampling::{Arithmetic, Plane, Sample};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping};

#[test]
fn squeeze_wire_ranges_split_at_the_full_input_count_without_reordering() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(3, 3);
    let definitions = vec![definition(8, 0); 256];
    let (source, expected) = inputs(
        &context,
        extent,
        &definitions,
        extent,
        &[UpsamplingFactor::One; 256],
    );
    for (in_place, selection) in [
        (false, None),
        (true, None),
        (false, Some((3, 256))),
        (true, Some((255, 4))),
    ] {
        let mut squeeze = LosslessModularSqueeze::HorizontalThenVertical.with_in_place(in_place);
        if let Some((begin, count)) = selection {
            squeeze = squeeze.with_channels(begin, count).unwrap();
        }
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    extra_channels: definitions.clone(),
                    entropy,
                    local_transforms: squeeze.clone().into(),
                    ..Default::default()
                },
            );
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(&modular_words::channel_frames(&bytes)[0], &expected);
            assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
        }
    }
}

#[test]
fn sampled_color_and_scalar_grids_share_the_frame_header_plan() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let decoder = GpuDecoder::wgpu(gpu.clone()).unwrap();
    let definitions = vec![definition(8, 0), definition(13, 1), definition(31, 3)];
    let displayed = Extent2d::new(2051, 19);
    for factor in [
        UpsamplingFactor::One,
        UpsamplingFactor::Two,
        UpsamplingFactor::Four,
        UpsamplingFactor::Eight,
    ] {
        for (factors, native_compatible) in [
            (
                [factor, UpsamplingFactor::Four, UpsamplingFactor::One],
                true,
            ),
            (
                [factor, UpsamplingFactor::Eight, UpsamplingFactor::Eight],
                false,
            ),
        ] {
            let (source, expected) = inputs(
                &context,
                factor.source_extent(displayed),
                &definitions,
                displayed,
                &factors,
            );
            for entropy in [
                LosslessModularEntropyCoding::Prefix,
                LosslessModularEntropyCoding::Ans,
            ] {
                let encoder = LosslessModularEncoder::with_config(
                    context.clone(),
                    LosslessModularConfig {
                        extra_channels: definitions.clone(),
                        entropy,
                        group_size: LosslessModularGroupSize::Pixels128,
                        tree_mode: LosslessModularTreeMode::LocalPerGroup,
                        ..Default::default()
                    },
                );
                let mut sequence = encoder
                    .begin_sequence(
                        LosslessModularSequenceDescriptor::new(
                            displayed.width,
                            displayed.height,
                            LosslessModularFormat::Rgb,
                            8,
                            AnimationHeader::Still,
                        )
                        .unwrap(),
                    )
                    .unwrap();
                let frame = sequence
                    .submit_last_frame(
                        source.clone(),
                        FrameOptions {
                            upsampling: factor,
                            extra_channel_upsampling: factors.to_vec(),
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .wait()
                    .unwrap();
                sequence.insert(frame).unwrap();
                let bytes = sequence.finish_raw().unwrap();
                // Unmodified libjxl 0.12 rejects effective extra factors above eight.
                // Extended grids keep the independent raw-word and F64/GPU checks below.
                if native_compatible {
                    assert_eq!(&modular_words::channel_frames(&bytes)[0], &expected);
                }
                assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
                if native_compatible {
                    let header = modular_words::sampling_headers(&bytes);
                    assert_eq!(
                        header[0].coded,
                        [
                            factor.source_extent(displayed).width,
                            factor.source_extent(displayed).height
                        ]
                    );
                }
                let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let plane = &expected[5];
                let reference = Plane {
                    width: plane.width as usize,
                    height: plane.height as usize,
                    samples: plane
                        .words
                        .iter()
                        .map(|&word| {
                            Sample::decoded(
                                word,
                                definitions[2].precision().bit_depth(),
                                Arithmetic::Wgsl,
                            )
                        })
                        .collect(),
                }
                .reconstruct(
                    factors[2].factor() << definitions[2].dimension_shift(),
                    displayed.width as usize,
                    displayed.height as usize,
                    &inventory.image_header.upsampling_weights,
                    Arithmetic::Wgsl,
                );
                let request = GpuOutputRequest::numeric(
                    SamplePrecision::float(32, 8).unwrap().pixel_format(),
                    NumericSampleMapping::NormalizedUnsigned,
                )
                .unwrap()
                .with_extra_channel(2)
                .unwrap();
                let mut session = decoder.open(&bytes, request).unwrap();
                let frame = session.next_frame().unwrap().unwrap();
                let raw =
                    jxl_test_support::gpu::planes::read_bytes(&gpu, &frame.output().outputs[0]);
                let actual = extra_channels::floats(&raw);
                assert_eq!(actual.len(), reference.samples.len());
                for (index, (&actual, expected)) in actual.iter().zip(reference.samples).enumerate()
                {
                    expected.check(actual, "independent Modular resampling", index);
                }
                assert!(session.next_frame().unwrap().is_none());
            }
        }
    }
}

#[test]
fn palette_and_ordered_transforms_preserve_independent_channel_order() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let definitions = vec![
        definition(7, 0),
        definition(9, 0),
        definition(11, 0),
        definition(13, 3),
    ];
    let extent = Extent2d::new(261, 5);
    let (source, expected) = inputs(
        &context,
        extent,
        &definitions,
        extent,
        &[UpsamplingFactor::One; 4],
    );
    let program = LosslessModularLocalTransforms::sequence(vec![
        LosslessModularTransform::Rct {
            begin_channel: 2,
            rct_type: LosslessModularRctType::YCOCG,
        },
        LosslessModularTransform::Squeeze(
            LosslessModularSqueezeStep::new(true, 2, 3, false).unwrap(),
        ),
        LosslessModularTransform::Rct {
            begin_channel: 2,
            rct_type: LosslessModularRctType::new(35).unwrap(),
        },
    ])
    .unwrap();
    for local_transforms in [
        LosslessModularSqueeze::HorizontalThenVertical.into(),
        LosslessModularSqueeze::HorizontalThenVertical
            .with_channels(2, 3)
            .unwrap()
            .into(),
        program,
    ] {
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    extra_channels: definitions.clone(),
                    entropy,
                    palette: Some(
                        LosslessModularPalette::new(256)
                            .unwrap()
                            .with_components(1, 2)
                            .unwrap(),
                    ),
                    local_transforms: local_transforms.clone(),
                    group_size: LosslessModularGroupSize::Pixels128,
                    ..Default::default()
                },
            );
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(&modular_words::channel_frames(&bytes)[0], &expected);
            assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
        }
    }
    let invalid = LosslessModularEncoder::with_config(
        context.clone(),
        LosslessModularConfig {
            extra_channels: definitions,
            group_size: LosslessModularGroupSize::Pixels128,
            local_transforms: LosslessModularSqueeze::Horizontal
                .with_channels(6, 1)
                .unwrap()
                .into(),
            ..Default::default()
        },
    );
    assert!(matches!(
        invalid.memory_plan(&source),
        Err(EncodeError::InvalidModularSqueezeChannels { channels: 6, .. })
    ));
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
