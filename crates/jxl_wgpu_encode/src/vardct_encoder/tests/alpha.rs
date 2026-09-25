//! Exact alpha words are independent of lossy color coefficients and physical source packing.
use super::*;
use crate::{AlphaAssociation, ColorChannels, ColorSampleFormat};
use jxl_gpu_formats::ByteOrder;
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::oracles::extra_channels;

mod boundaries;
mod sequence;

fn formats(channels: ColorChannels) -> impl Iterator<Item = ColorSampleFormat> {
    (1..=31)
        .map(move |bits| ColorSampleFormat::integer(channels, bits).unwrap())
        .chain(
            floating::all_precisions().into_iter().map(move |p| {
                ColorSampleFormat::float(channels, p.bits(), p.exponent_bits()).unwrap()
            }),
        )
}

pub(super) fn input(extent: Extent2d, samples: ColorSampleFormat) -> Vec<u32> {
    let rgb = match samples.float_precision() {
        Some(p) => floating::pixels(extent.width as usize, extent.height as usize, p),
        None => precision::pixels(
            extent.width as usize,
            extent.height as usize,
            samples.bits_per_sample(),
            137,
        ),
    };
    rgb.into_iter()
        .enumerate()
        .flat_map(|(i, rgb)| {
            let alpha = match samples.float_precision() {
                Some(p) => {
                    let fraction = u32::from(p.bits() - p.exponent_bits() - 1);
                    let one = ((1u32 << (p.exponent_bits() - 1)) - 1) << fraction;
                    [
                        0,
                        one,
                        one - 1,
                        1,
                        one - (1 << (fraction - 1)),
                        one - (1 << fraction),
                    ][i % 6]
                }
                None => {
                    let mask = samples.sample_mask();
                    [0, mask, 1, mask - 1, mask / 3, (i as u32 * 347) & mask][i % 6]
                }
            };
            rgb.into_iter()
                .take(samples.channels().count() as usize)
                .chain([alpha])
        })
        .collect()
}

pub(super) fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    config: &VarDctConfig,
    words: &[u32],
    storage: Storage,
    reversed: bool,
) -> BufferImageSource {
    let mut format = config.pixel_format();
    format.byte_order = if reversed {
        ByteOrder::Big
    } else {
        ByteOrder::Native
    };
    let (layout, bytes) = Packing {
        storage,
        reversed,
        shifted: reversed,
    }
    .pack(format, extent, words, 4099);
    BufferImageSource::new(
        Arc::new(
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("alpha source with poisoned component/row padding"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

fn expected_alpha(words: &[u32], samples: ColorSampleFormat) -> Vec<u32> {
    words
        .chunks_exact(samples.channels().count() as usize + 1)
        .map(|pixel| *pixel.last().unwrap())
        .collect()
}

fn check_alpha(bytes: &[u8], words: &[u32], samples: ColorSampleFormat) {
    let expected = check_alpha_words_and_native(bytes, words, samples);
    let rust = extra_channels::rust_planes(bytes);
    assert_eq!(rust.1.len(), 1);
    check_numeric_alpha(&rust.1[0], &expected);
}

fn check_alpha_words_and_native(
    bytes: &[u8],
    words: &[u32],
    samples: ColorSampleFormat,
) -> Vec<f32> {
    let expected = expected_alpha(words, samples);
    let extras = jxl_test_support::oracles::modular_integer::extra_planes(bytes, 0);
    assert_eq!(extras.len(), 1);
    let raw: Vec<u32> = extras[0].iter().map(|&v| v as u32).collect();
    assert_eq!(raw, expected, "exact independently decoded alpha words");
    let native = extra_channels::libjxl_planes(bytes, expected.len(), 1)
        .expect("required native alpha oracle");
    let expected: Vec<_> = expected
        .into_iter()
        .map(|word| match samples.float_precision() {
            Some(p) => f32::from_bits(jxl_test_support::oracles::sample_bits::custom_binary32(
                word,
                p.bits(),
                p.exponent_bits(),
            )),
            None => (f64::from(word) / f64::from(samples.sample_mask())) as f32,
        })
        .collect();
    check_numeric_alpha(&native.1[0], &expected);
    expected
}

fn check_numeric_alpha(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &value)) in actual.iter().zip(expected).enumerate() {
        if value.is_nan() {
            assert!(actual.is_nan());
            continue;
        }
        if value == 0.0 || value.is_infinite() {
            assert_eq!(actual.to_bits(), value.to_bits());
            continue;
        }
        assert!(
            (actual - value).abs() <= 2.0 * f32::EPSILON * value.abs().max(f32::MIN_POSITIVE),
            "alpha {i}: {actual} vs {value}"
        );
    }
}

fn check_header(bytes: &[u8], config: &VarDctConfig) {
    let header = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    assert_eq!(header.bit_depth, config.sample_format.bit_depth());
    assert_eq!(header.extra_channels.len(), 1);
    assert_eq!(header.extra_channels[0].bit_depth, header.bit_depth);
    assert_eq!(header.extra_channels[0].dimension_shift, 0);
    assert_eq!(
        header.extra_channels[0].channel_type,
        jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha {
            associated: config.alpha == Some(AlphaAssociation::Associated),
        }
    );
    assert_eq!(
        header.grayscale,
        config.sample_format.channels() == ColorChannels::Gray
    );
}

#[test]
fn alpha_input_every_precision_and_association_preserves_exact_words_and_source_layouts() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(8, 8);
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for samples in formats(channels) {
            let words = input(extent, samples);
            for color_transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
                for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
                    let config = VarDctConfig {
                        sample_format: samples,
                        alpha: Some(association),
                        color_transform,
                        ..Default::default()
                    };
                    let encoder = VarDctEncoder::new_with_config(
                        context.clone(),
                        VarDctStrategy::Dct8,
                        config.clone(),
                    )
                    .unwrap();
                    let bytes = encoder
                        .encode(upload(
                            &context,
                            extent,
                            &config,
                            &words,
                            Storage::Packed,
                            false,
                        ))
                        .unwrap();
                    check_header(&bytes, &config);
                    check_alpha(&bytes, &words, samples);
                    // All legal sample widths, endian handling and swizzles use the same plane plan.
                    for storage in [Storage::Packed, Storage::Planar, Storage::Split] {
                        assert_eq!(
                            encoder
                                .encode(upload(&context, extent, &config, &words, storage, true))
                                .unwrap(),
                            bytes,
                            "{samples:?}/{color_transform:?}/{association:?}/{storage:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn alpha_input_progression_orders_and_packed_fields_share_one_side_plane_plan() {
    let context = test_context().unwrap();
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for samples in [
            ColorSampleFormat::integer(channels, 7).unwrap(),
            ColorSampleFormat::float(channels, 24, 7).unwrap(),
        ] {
            for (topology, extent) in [
                (layouts::Topology::Map, Extent2d::new(25, 17)),
                (layouts::Topology::Tiled, Extent2d::new(259, 263)),
            ] {
                let words = input(extent, samples);
                for early in [false, true] {
                    let progression = if early {
                        progressive::combined()
                            .with_downsampling(vec![crate::ProgressiveDownsampling {
                                factor: 1,
                                last_pass: 0,
                            }])
                            .unwrap()
                    } else {
                        progressive::combined()
                    };
                    let config = VarDctConfig {
                        sample_format: samples,
                        alpha: Some(AlphaAssociation::Associated),
                        progressive: progression,
                        group_order: crate::VarDctGroupOrder::saliency_first(),
                        coefficient_orders: orders::selected(VarDctStrategy::ALL),
                        dequant_matrices: raw_matrices::selected([VarDctStrategy::Dct8]),
                        ..Default::default()
                    };
                    let backend = topology.backend(&context, extent, &config);
                    let bytes = layouts::encode(
                        &context,
                        &backend,
                        &config,
                        upload(&context, extent, &config, &words, Storage::Packed, false),
                    );
                    check_header(&bytes, &config);
                    if early && (extent.width > 256 || extent.height > 256) {
                        // jxl 0.6.0's downsampling_bracket saturates 0 - 1 to 0. It therefore
                        // requests the same full-resolution Modular plane again in later passes.
                        // libjxl uses signed -1 (an empty bracket), as does independent jxl-oxide.
                        assert!(matches!(
                            extra_channels::try_rust_frame_planes_with_profile(&bytes),
                            Err(jxl::error::Error::SectionTooShort)
                        ));
                        // Retained jxl-oxide integer planes independently preserve every bit;
                        // native libjxl checks normalization, including custom-float zeros.
                        check_alpha_words_and_native(&bytes, &words, samples);
                    } else {
                        check_alpha(&bytes, &words, samples);
                    }
                    let mut storages = vec![Storage::ThreeBytes];
                    if samples.bits_per_sample() == 7 {
                        storages.extend([Storage::SharedWord, Storage::MixedWords]);
                    }
                    for storage in storages {
                        assert_eq!(
                            layouts::encode(
                                &context,
                                &backend,
                                &config,
                                upload(&context, extent, &config, &words, storage, true)
                            ),
                            bytes
                        );
                    }
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn alpha_input_all_strategies_keep_color_coefficients_independent_of_alpha() {
    use crate::{GpuEncodeBackend, GpuFrameSource, VarDctBackend};
    let context = test_context().unwrap();
    for strategy in VarDctStrategy::ALL {
        let extent = strategy.pixel_extent();
        let samples = ColorSampleFormat::float(ColorChannels::Rgb, 16, 5).unwrap();
        let words = input(extent, samples);
        let colors: Vec<_> = words
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| p[..3].iter().copied())
            .collect();
        for color_transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = VarDctConfig {
                sample_format: samples,
                alpha: Some(AlphaAssociation::Associated),
                color_transform,
                lf_metadata: custom_lf_metadata(),
                coefficient_orders: orders::selected([strategy]),
                ..Default::default()
            };
            let mut baseline = None;
            for alpha in [None, config.alpha] {
                let config = VarDctConfig {
                    alpha,
                    ..config.clone()
                };
                let backend =
                    VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
                let source = upload(
                    &context,
                    extent,
                    &config,
                    if alpha.is_some() { &words } else { &colors },
                    Storage::Planar,
                    true,
                );
                let (ac, lengths, artifact) = backend
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(source),
                        &layouts::request(extent, &config),
                    )
                    .unwrap()
                    .wait_with_ac_fragments_for_test()
                    .unwrap();
                if let Some(reference) = &baseline {
                    assert_eq!(
                        &(ac, lengths),
                        reference,
                        "{strategy:?}/{color_transform:?}"
                    );
                    let mut bytes = super::image_header_with_color(
                        extent.width,
                        extent.height,
                        crate::AnimationHeader::Still,
                        &backend.color_plan,
                    )
                    .unwrap()
                    .bytes()
                    .to_vec();
                    bytes.extend_from_slice(assemble_frame(artifact.packets).unwrap().bytes());
                    check_alpha(&bytes, &words, samples);
                } else {
                    baseline = Some((ac, lengths));
                }
            }
        }
    }
}

#[test]
fn alpha_input_preserves_signed_zero_subnormals_infinities_and_nan_payloads() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(17, 13);
    for samples in formats(ColorChannels::Rgb).filter(|samples| samples.float_precision().is_some())
    {
        let p = samples.float_precision().unwrap();
        let fraction = p.bits() - p.exponent_bits() - 1;
        let sign = 1u32 << (p.bits() - 1);
        let infinity = ((1u32 << p.exponent_bits()) - 1) << fraction;
        let values = [
            0,
            sign,
            1,
            sign | 1,
            infinity,
            sign | infinity,
            infinity | 1,
            infinity | (1 << (fraction - 1)) | 1,
        ];
        let mut words = input(extent, samples);
        for (i, pixel) in words.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            pixel[3] = values[i % values.len()];
        }
        let config = VarDctConfig {
            sample_format: samples,
            alpha: Some(AlphaAssociation::Unassociated),
            ..Default::default()
        };
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        let bytes = encoder
            .encode(upload(
                &context,
                extent,
                &config,
                &words,
                Storage::Split,
                true,
            ))
            .unwrap();
        check_alpha(&bytes, &words, samples);
    }
}

#[test]
fn alpha_input_icc_device_channels_keep_profile_color_and_alpha_storage_independent() {
    use jxl_gpu_formats::{Channel, ColorModel, ColorSpecification, PackingFieldKind, Swizzle};
    use jxl_gpu_protocol::icc::IccProfile;
    let context = test_context().unwrap();
    for gray in [false, true] {
        let profile = IccProfile::parse(
            fs::read(
                jxl_test_support::fixtures::embedded_icc::directory().join(if gray {
                    "gray.icc"
                } else {
                    "rgb.icc"
                }),
            )
            .unwrap()
            .into(),
            Default::default(),
        )
        .unwrap();
        for color_transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let samples = ColorSampleFormat::float(
                if gray {
                    ColorChannels::Gray
                } else {
                    ColorChannels::Rgb
                },
                32,
                8,
            )
            .unwrap();
            let config = VarDctConfig {
                sample_format: samples,
                alpha: Some(AlphaAssociation::Unassociated),
                color_transform,
                source_color: ColorSpecification::Icc(profile.clone()),
                color_options: crate::ImageColorOptions {
                    rendering_intent: profile.header().rendering_intent,
                    ..Default::default()
                },
                ..Default::default()
            };
            for (topology, extent) in [
                (layouts::Topology::Single, Extent2d::new(8, 8)),
                (layouts::Topology::Map, Extent2d::new(25, 17)),
                (layouts::Topology::Tiled, Extent2d::new(259, 3)),
            ] {
                let words = input(extent, samples);
                let backend = topology.backend(&context, extent, &config);
                let source = upload(&context, extent, &config, &words, Storage::Planar, false);
                let memory = backend.memory_plan(&source).unwrap();
                assert!(memory.alpha.is_some());
                assert_eq!(
                    memory.icc.is_some(),
                    color_transform == VarDctColorTransform::Xyb
                );
                let bytes = layouts::encode(&context, &backend, &config, source.clone());
                check_header(&bytes, &config);
                check_alpha(&bytes, &words, samples);
                let mut format = source.layout.format.clone();
                format.model = ColorModel::IccDevice;
                format.swizzle = Swizzle::Device;
                for word in format.planes.iter_mut().flat_map(|plane| &mut plane.words) {
                    for field in &mut word.fields {
                        if let PackingFieldKind::Channel(channel) = &mut field.kind {
                            *channel = match channel {
                                Channel::X => Channel::Device(0),
                                Channel::Y => Channel::Device(1),
                                Channel::Z => Channel::Device(2),
                                Channel::W => Channel::Alpha,
                                _ => unreachable!(),
                            };
                        }
                    }
                }
                let layout =
                    ImageLayout::from_planes(extent, format, source.layout.planes).unwrap();
                let device_source = BufferImageSource::new(source.buffer, layout).unwrap();
                assert_eq!(
                    layouts::encode(&context, &backend, &config, device_source),
                    bytes
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn alpha_input_global_and_pass_groups_interoperate_with_both_color_domains() {
    let context = test_context().unwrap();
    for color_transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
        for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
            let config = VarDctConfig {
                alpha: Some(AlphaAssociation::Unassociated),
                color_transform,
                sample_format: ColorSampleFormat::integer(channels, 8).unwrap(),
                ..Default::default()
            };
            let encoder =
                TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
            for (w, h) in [
                (1, 1),
                (17, 13),
                (256, 256),
                (257, 17),
                (17, 257),
                (259, 261),
                (2051, 9),
            ] {
                let extent = Extent2d::new(w, h);
                let words = input(extent, config.sample_format);
                let bytes = encoder
                    .encode(upload(
                        &context,
                        extent,
                        &config,
                        &words,
                        Storage::Packed,
                        false,
                    ))
                    .unwrap();
                check_alpha(&bytes, &words, config.sample_format);
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
