use super::*;
use jxl_gpu_formats::{ColorSample, ColorSpecification, ColorStorage};
use jxl_gpu_protocol::icc::IccProfile;

#[test]
fn preview_icc_identity_and_samples_survive_modular_and_all_vardct_backends() {
    let rig = Rig::new();
    let profile = IccProfile::parse(
        std::fs::read(jxl_test_support::fixtures::embedded_icc::directory().join("rgb.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    let native_profile = jxl_test_support::oracles::icc_profile::IccProfileOracle::compile();
    let preview = PreviewSize::new(8, 8).unwrap();
    let coded = preview.extent();
    let samples = ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap();
    let config = VarDctConfig {
        color_transform: VarDctColorTransform::Original,
        sample_format: samples,
        source_color: ColorSpecification::Icc(profile.clone()),
        image_options: ImageOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut preview_source = source(&rig, coded, samples, false, 1);
    preview_source.layout.format.color_spec = config.source_color.clone();
    let mut main_source = source(&rig, coded, samples, false, 7);
    main_source.layout.format.color_spec = config.source_color.clone();
    for topology in 0..4 {
        let descriptor = ImageSequenceDescriptor::new(16, 16, AnimationHeader::Still)
            .unwrap()
            .with_preview(preview);
        let options = FrameOptions {
            upsampling: UpsamplingFactor::Two,
            ..Default::default()
        };
        let bytes = if topology == 0 {
            let encoder = LosslessModularEncoder::new(rig.context.clone())
                .with_image_options(config.image_options)
                .unwrap();
            let mut session = encoder
                .begin_sequence(
                    LosslessModularSequenceDescriptor::from_pixel_format(
                        16,
                        16,
                        &main_source.layout.format,
                        AnimationHeader::Still,
                    )
                    .unwrap()
                    .with_preview(preview),
                )
                .unwrap();
            let output = session
                .submit_preview(preview_source.clone(), FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            session.insert_preview(output).unwrap();
            let frame = session
                .submit_last_frame(main_source.clone(), options)
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            session.finish_container().unwrap()
        } else {
            let fixed = match topology {
                1 => Some(
                    VarDctEncoder::new_with_config(
                        rig.context.clone(),
                        VarDctStrategy::Dct8,
                        config.clone(),
                    )
                    .unwrap(),
                ),
                2 => Some(
                    VarDctEncoder::new_with_strategy_map(
                        rig.context.clone(),
                        VarDctStrategyMap::new(
                            8,
                            8,
                            vec![VarDctTransform::new(0, 0, VarDctStrategy::Dct8)],
                        )
                        .unwrap(),
                        config.clone(),
                    )
                    .unwrap(),
                ),
                _ => None,
            };
            let tiled =
                TiledVarDctEncoder::new_with_config(rig.context.clone(), config.clone()).unwrap();
            let mut session = if let Some(fixed) = fixed {
                fixed.begin_sequence(descriptor)
            } else {
                tiled.begin_sequence(descriptor)
            }
            .unwrap();
            let output = session
                .submit_preview(preview_source.clone(), FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            session.insert_preview(output).unwrap();
            let frame = session
                .submit_last_frame(main_source.clone(), options)
                .unwrap()
                .wait()
                .unwrap();
            session.insert(frame).unwrap();
            session.finish_container().unwrap()
        };
        assert_eq!(
            native_profile.read(&bytes).profile,
            profile.bytes().as_ref()
        );
        for selection in [ImageSelection::Preview, ImageSelection::Main] {
            let mut flags = vec!["--original-icc", "--no-cms"];
            if selection == ImageSelection::Preview {
                flags.push("--preview");
            }
            let native = oracle::libjxl_output(&bytes, &flags).unwrap();
            let request = GpuOutputRequest::color(
                PixelFormat::icc_device(
                    profile.clone(),
                    ColorSample::F32,
                    ColorStorage::Interleaved,
                    false,
                )
                .unwrap(),
            )
            .unwrap()
            .with_image_selection(selection);
            let mut session = open_fragmented(&rig.decoder, &bytes, request);
            let frame = session.next_frame().unwrap().unwrap();
            let actual = read_bytes(&rig.gpu, &frame.output().outputs[0]);
            let actual = oracle::floats(&actual);
            assert_eq!(actual.len() / 3, native.len() / 4);
            for (a, b) in actual
                .as_chunks::<3>()
                .0
                .iter()
                .zip(native.as_chunks::<4>().0)
            {
                for (&a, &b) in a.iter().zip(b) {
                    assert!(
                        a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                        "ICC preview/main {topology}: {a} vs {b}"
                    );
                }
            }
            assert!(session.next_frame().unwrap().is_none());
        }
        assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn preview_has_its_own_independent_scalar_sources_and_nonleading_alpha() {
    let rig = Rig::new();
    let preview = PreviewSize::new(9, 5).unwrap();
    let main = Extent2d::new(17, 9);
    let config = VarDctConfig {
        color_transform: VarDctColorTransform::Original,
        extra_channels: vec![
            ExtraChannel::new(
                ExtraChannelKind::Depth,
                SamplePrecision::integer(13).unwrap(),
                0,
                b"preview depth".to_vec(),
            )
            .unwrap(),
            ExtraChannel::new(
                ExtraChannelKind::Alpha(AlphaAssociation::Unassociated),
                SamplePrecision::integer(7).unwrap(),
                0,
                Vec::new(),
            )
            .unwrap(),
        ],
        ..Default::default()
    };
    let mut sources = Vec::new();
    let mut expected = Vec::new();
    for (extent, seed) in [(preview.extent(), 1), (main, 7)] {
        let mut inputs = Vec::new();
        let mut planes = Vec::new();
        for extra in &config.extra_channels {
            let words = (0..extent.width * extent.height)
                .map(|i| {
                    (i * 73 + seed * 11)
                        & ((1u32
                            << extra
                                .precision()
                                .color(ColorChannels::Gray)
                                .bits_per_sample())
                            - 1)
                })
                .collect::<Vec<_>>();
            let (layout, bytes) = Packing {
                storage: Storage::Packed,
                reversed: true,
                shifted: true,
            }
            .pack(extra.precision().pixel_format(), extent, &words, 4099);
            inputs.push(
                BufferImageSource::new(
                    Arc::new(rig.context.device().create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("preview independent scalar"),
                            contents: &bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        },
                    )),
                    layout,
                )
                .unwrap(),
            );
            planes.push(words);
        }
        sources.push(
            source(&rig, extent, ColorSampleFormat::RGB8, false, seed)
                .with_extra_channels(inputs)
                .unwrap(),
        );
        expected.push(planes);
    }
    let encoder = TiledVarDctEncoder::new_with_config(rig.context.clone(), config).unwrap();
    let mut session = encoder
        .begin_sequence(
            ImageSequenceDescriptor::new(main.width, main.height, AnimationHeader::Still)
                .unwrap()
                .with_preview(preview),
        )
        .unwrap();
    let output = session
        .submit_preview(sources.remove(0), FrameOptions::default())
        .unwrap()
        .wait()
        .unwrap();
    session.insert_preview(output).unwrap();
    let frame = session
        .submit_last_frame(sources.remove(0), FrameOptions::default())
        .unwrap()
        .wait()
        .unwrap();
    session.insert(frame).unwrap();
    let bytes = session.finish_raw().unwrap();
    rig.check(
        &bytes,
        main,
        preview,
        OutputOrientation::from_exif_value(1).unwrap(),
        2,
    );
    for (selection, planes) in [ImageSelection::Preview, ImageSelection::Main]
        .into_iter()
        .zip(expected)
    {
        for (channel, expected) in planes.into_iter().enumerate() {
            let request = GpuOutputRequest::numeric(
                SamplePrecision::integer([13, 7][channel])
                    .unwrap()
                    .pixel_format(),
                jxl_wgpu_decode::NumericSampleMapping::NativeUnsigned,
            )
            .unwrap()
            .with_extra_channel(channel as u32)
            .unwrap()
            .with_image_selection(selection);
            let mut session = rig.decoder.open(&bytes, request).unwrap();
            let frame = session.next_frame().unwrap().unwrap();
            let actual = read_bytes(&rig.gpu, &frame.output().outputs[0]);
            let words = if channel == 0 {
                actual
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|word| u32::from(u16::from_le_bytes(*word)))
                    .collect::<Vec<_>>()
            } else {
                actual.iter().map(|&word| u32::from(word)).collect()
            };
            assert_eq!(words, expected);
        }
    }
}
