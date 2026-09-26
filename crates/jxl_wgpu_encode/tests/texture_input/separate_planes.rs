use super::*;
use jxl_gpu_formats::{ByteOrder, PlaneFormat, PlaneSampling};

/// Repack canonical words into logical plane groups, independently of the encoder copy plan.
pub(super) fn split(
    context: &WgpuContext,
    extent: Extent2d,
    mut format: PixelFormat,
    raw: &[u8],
    groups: &[usize],
    big: bool,
) -> TexturePlanesSource {
    assert_eq!(format.planes.len(), 1);
    let words = &format.planes[0].words;
    assert_eq!(groups.iter().sum::<usize>(), words.len());
    let stride: usize = words.iter().map(|word| word.bits() as usize / 8).sum();
    assert_eq!(raw.len(), extent.area().unwrap() * stride);
    let mut planes = Vec::new();
    let mut stored = Vec::new();
    let mut first = 0;
    let mut offset = 0;
    for &count in groups {
        let group = &words[first..first + count];
        for pixel in raw.chunks_exact(stride) {
            let mut start = offset;
            for word in group {
                let len = word.bits() as usize / 8;
                let mut value = pixel[start..start + len].to_vec();
                if big {
                    value.reverse();
                }
                stored.extend(value);
                start += len;
            }
        }
        planes.push(PlaneFormat {
            sampling: PlaneSampling::FULL,
            pixels_per_element: 1,
            words: group.to_vec(),
        });
        offset += group
            .iter()
            .map(|word| word.bits() as usize / 8)
            .sum::<usize>();
        first += count;
    }
    format.planes = planes;
    format.byte_order = if big {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    };
    let mut layout = ImageLayout::packed(extent, format.clone()).unwrap();
    let mut end = 0;
    for plane in &mut layout.planes {
        plane.offset = end;
        end = plane.end_offset().unwrap();
    }
    let layout = ImageLayout::from_planes(extent, format.clone(), layout.planes).unwrap();
    assert_eq!(layout.logical_size, stored.len() as u64);
    let planes = jxl_test_support::gpu::textures::upload_planes(
        context.device(),
        context.queue(),
        &layout,
        &stored,
    )
    .into_iter()
    .map(|texture| {
        let format = texture.format();
        TexturePlaneSource::new(texture, format, 1, 1).unwrap()
    })
    .collect();
    TexturePlanesSource::new(extent, format, planes).unwrap()
}

#[test]
fn separate_and_split_texture_planes_preserve_integer_and_float_words() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let encoders = [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ]
    .map(|entropy| {
        LosslessModularEncoder::with_config(
            context.clone(),
            LosslessModularConfig {
                entropy,
                ..Default::default()
            },
        )
    });
    for (bits, exponent) in [
        (1, 0),
        (8, 0),
        (13, 0),
        (31, 0),
        (8, 3),
        (16, 5),
        (24, 7),
        (32, 8),
    ] {
        for (channels, alpha) in [
            (ColorChannels::Gray, true),
            (ColorChannels::Rgb, false),
            (ColorChannels::Rgb, true),
        ] {
            let sample = if exponent == 0 {
                ColorSampleFormat::integer(channels, bits)
            } else {
                ColorSampleFormat::float(channels, bits, exponent)
            }
            .unwrap();
            let config = VarDctConfig {
                sample_format: sample,
                alpha: alpha.then_some(AlphaAssociation::Unassociated),
                ..Default::default()
            };
            let count = channels.count() as usize + usize::from(alpha);
            // All exponent patterns and payload bits pass through without texture interpretation.
            let words: Vec<_> = (0..extent.area().unwrap() * count)
                .map(|i| (i as u32).wrapping_mul(741103597) & (u32::MAX >> (32 - bits)))
                .collect();
            let raw = bytes(&words, sample.word_bytes() as usize);
            let expected = planes(extent, &words, count);
            let canonical = buffer(&context, extent, config.pixel_format(), &raw);
            let planar = vec![1; count];
            let grouped = if count == 4 {
                vec![2, 2]
            } else if count == 3 {
                vec![1, 2]
            } else {
                planar.clone()
            };
            for (groups, big) in [(&planar, false), (&grouped, true)] {
                let input = split(&context, extent, config.pixel_format(), &raw, groups, big);
                for encoder in &encoders {
                    let plan = encoder.memory_plan(&input).unwrap();
                    assert_eq!(plan.source_texture_bytes, raw.len() as u64);
                    assert_eq!(plan.source_binding_bytes, 0);
                    assert_eq!(plan.source_conversion_bytes, 0);
                    let encoded = encoder.encode(input.clone()).unwrap();
                    assert_eq!(encoded, encoder.encode(canonical.clone()).unwrap());
                    assert_eq!(modular_words::channel_frames(&encoded)[0], expected);
                    assert_eq!(
                        modular_integer::modular_channel_words(&encoded, 0),
                        expected
                    );
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn cmyk_alpha_planes_align_wide_texels_and_preserve_black_and_independent_depth() {
    use jxl_gpu_formats::ColorSpecification;
    use jxl_gpu_protocol::icc::IccProfile;
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let profile = IccProfile::parse(
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/lut/lut16_xyz_4.icc"),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap();
    for float in [false, true] {
        let sample = if float {
            ColorSampleFormat::float(ColorChannels::Rgb, 32, 8)
        } else {
            ColorSampleFormat::integer(ColorChannels::Rgb, 8)
        }
        .unwrap();
        let depth = ExtraChannel::new(
            ExtraChannelKind::Depth,
            SamplePrecision::integer(13).unwrap(),
            1,
            b"depth".to_vec(),
        )
        .unwrap();
        let config = VarDctConfig {
            sample_format: sample,
            alpha: Some(AlphaAssociation::Unassociated),
            source_color: ColorSpecification::Icc(profile.clone()),
            extra_channels: vec![depth.clone()],
            image_options: ImageOptions {
                rendering_intent: profile.header().rendering_intent,
                ..Default::default()
            },
            ..Default::default()
        };
        let words: Vec<_> = (0..extent.area().unwrap() * 5)
            .map(|i| {
                if float {
                    ((i % 251) as f32 / 251.0).to_bits()
                } else {
                    (i * 17 % 251) as u32
                }
            })
            .collect();
        let raw = bytes(&words, sample.word_bytes() as usize);
        let scalar_extent = depth.source_extent(extent);
        let scalars: Vec<_> = (0..scalar_extent.area().unwrap())
            .map(|i| (i * 71) as u32)
            .collect();
        let extra = buffer(
            &context,
            scalar_extent,
            depth.precision().pixel_format(),
            &bytes(&scalars, 2),
        );
        // The first R32 plane ends 4 mod 16; the next RGBA32 copy must align its own offset.
        let groups: &[usize] = if float { &[1, 4] } else { &[1, 1, 1, 1, 1] };
        let mut input = split(&context, extent, config.pixel_format(), &raw, groups, true)
            .with_extra_channels(vec![extra.clone()])
            .unwrap();
        let mut canonical = buffer(&context, extent, config.pixel_format(), &raw)
            .with_extra_channels(vec![extra])
            .unwrap();
        if float {
            input = input
                .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                .unwrap();
            canonical = canonical
                .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                .unwrap();
        }
        let encoder = LosslessModularEncoder::with_config(
            context.clone(),
            LosslessModularConfig {
                extra_channels: vec![depth],
                ..Default::default()
            },
        )
        .with_image_options(config.image_options)
        .unwrap();
        let encoded = encoder.encode(input.clone()).unwrap();
        assert_eq!(encoded, encoder.encode(canonical.clone()).unwrap());
        let mut expected = planes(extent, &words, 5);
        if !float {
            for plane in expected.iter_mut().take(4) {
                for word in &mut plane.words {
                    *word = 255 - *word;
                }
            }
        }
        expected.swap(3, 4); // JPEG XL extras: alpha, primary Black, attached depth.
        expected.push(modular_integer::ExtraWords {
            width: scalar_extent.width,
            height: scalar_extent.height,
            words: scalars,
        });
        assert_eq!(modular_words::channel_frames(&encoded)[0], expected);
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let encoder = TiledVarDctEncoder::new_with_config(
                context.clone(),
                VarDctConfig {
                    color_transform: transform,
                    ..config.clone()
                },
            )
            .unwrap();
            let encoded = encoder.encode(input.clone()).unwrap();
            assert_eq!(encoded, encoder.encode(canonical.clone()).unwrap());
            assert_eq!(
                modular_integer::vardct_extra_words(&encoded, 0),
                expected[3..]
            );
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn aliased_subresources_count_once_but_every_owned_plane_copy_is_reserved() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let words = vec![47; extent.area().unwrap() * 3];
    let mut input = split(
        &context,
        extent,
        ColorSampleFormat::RGB8.pixel_format(),
        &bytes(&words, 1),
        &[1, 1, 1],
        false,
    );
    input.planes[1] = input.planes[0].clone();
    input.planes[2] = input.planes[0].clone();
    let encoder = LosslessModularEncoder::new(context.clone());
    let mut outputs = Vec::new();
    for (layer, selected_bytes) in [(1, 45), (2, 90)] {
        input.planes[2].array_layer = layer;
        let plan = encoder.memory_plan(&input).unwrap();
        assert_eq!(plan.source_texture_bytes, selected_bytes);
        assert_eq!(plan.source_copy_bytes, 3108);
        assert_eq!(plan.source_binding_bytes, 0);
        assert_eq!(
            plan.addressed_bytes_per_job,
            plan.owned_bytes_per_job + selected_bytes
        );
        outputs.push(encoder.encode(input.clone()).unwrap());
    }
    assert_eq!(
        modular_words::channel_frames(&outputs[0])[0],
        planes(extent, &words, 3)
    );
    let second: Vec<_> = (0..45).flat_map(|_| [47, 47, 0x55]).collect();
    assert_eq!(
        modular_words::channel_frames(&outputs[1])[0],
        planes(extent, &second, 3)
    );
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn altered_plane_geometry_and_formats_reject_before_admission() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let input = split(
        &context,
        extent,
        ColorSampleFormat::RGB8.pixel_format(),
        &[47; 135],
        &[1, 1, 1],
        false,
    );
    let encoder = LosslessModularEncoder::new(context.clone());
    let vardct =
        TiledVarDctEncoder::new_with_config(context.clone(), VarDctConfig::default()).unwrap();
    for mutation in 0..8 {
        let mut source = input.clone();
        match mutation {
            0 => source.extent.width += 1,
            1 => source.extent.height = 0,
            2 => {
                source.planes.pop();
            }
            3 => source.planes.push(source.planes[0].clone()),
            4 => source.planes[1].mip_level = 0,
            5 => source.planes[1].array_layer = 3,
            6 => source.planes[1].texture_format = wgpu::TextureFormat::R8Unorm,
            _ => source.pixel_format.planes[1].words[0].fields[0].bits = 16,
        }
        assert!(encoder.memory_plan(&source).is_err(), "mutation {mutation}");
        assert!(encoder.submit(source.clone()).is_err());
        assert!(vardct.memory_plan(&source).is_err());
        assert!(vardct.submit(source).is_err());
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn separate_plane_previews_release_all_sources_and_match_buffer_packets() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let words: Vec<_> = (0..135).map(|i| i * 17 % 256).collect();
    let raw = bytes(&words, 1);
    let encoder = MixedModeEncoder::new(context.clone(), MixedModeConfig::default()).unwrap();
    let descriptor = ImageSequenceDescriptor::new(9, 5, AnimationHeader::Still)
        .unwrap()
        .with_preview(PreviewSize::new(9, 5).unwrap());
    for codec in [
        MixedModeFrameEncoding::Modular,
        MixedModeFrameEncoding::VarDct,
    ] {
        let mut outputs = Vec::new();
        for textures in [false, true] {
            let mut sequence = encoder.begin_sequence(descriptor.clone()).unwrap();
            let input = split(
                &context,
                extent,
                ColorSampleFormat::RGB8.pixel_format(),
                &raw,
                &[1, 2],
                false,
            );
            let owners: Vec<_> = input
                .planes
                .iter()
                .map(|plane| Arc::downgrade(&plane.texture))
                .collect();
            let source: GpuFrameSource = if textures {
                input.into()
            } else {
                drop(input);
                buffer(
                    &context,
                    extent,
                    ColorSampleFormat::RGB8.pixel_format(),
                    &raw,
                )
                .into()
            };
            let baseline = context.memory_stats().reserved_bytes;
            let preview = sequence
                .submit_preview(source, codec, FrameOptions::default())
                .unwrap()
                .wait()
                .unwrap();
            assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
            assert_eq!(
                context.memory_stats().reserved_bytes,
                baseline + preview.reserved_bytes()
            );
            sequence.insert_preview(preview).unwrap();
            let main = sequence
                .submit_last_frame(
                    buffer(
                        &context,
                        extent,
                        ColorSampleFormat::RGB8.pixel_format(),
                        &raw,
                    ),
                    codec,
                    FrameOptions::default(),
                )
                .unwrap()
                .wait()
                .unwrap();
            sequence.insert(main).unwrap();
            outputs.push(sequence.finish_raw().unwrap());
        }
        assert_eq!(outputs[0], outputs[1]);
        let native = jxl_test_support::oracles::extra_channels::libjxl_output(
            &outputs[1],
            &["--preview", "--original"],
        )
        .unwrap();
        assert_eq!(native.len(), 45 * 4);
        if codec == MixedModeFrameEncoding::Modular {
            for (pixel, reference) in native
                .as_chunks::<4>()
                .0
                .iter()
                .zip(words.as_chunks::<3>().0)
            {
                for (value, &word) in pixel.iter().zip(reference) {
                    assert!((value - word as f32 / 255.0).abs() <= 1e-6);
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn nonfinite_later_texture_plane_cannot_publish_preview_authority() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let config = VarDctConfig {
        sample_format: ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap(),
        ..Default::default()
    };
    let mut words = vec![0.5f32.to_bits(); 8 * 8 * 3];
    words[4] = f32::NAN.to_bits();
    let input = split(
        &context,
        Extent2d::new(8, 8),
        config.pixel_format(),
        &bytes(&words, 4),
        &[1, 1, 1],
        true,
    );
    let owners: Vec<_> = input
        .planes
        .iter()
        .map(|plane| Arc::downgrade(&plane.texture))
        .collect();
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let mut sequence = encoder
        .begin_sequence(
            ImageSequenceDescriptor::new(8, 8, AnimationHeader::Still)
                .unwrap()
                .with_preview(PreviewSize::new(8, 8).unwrap()),
        )
        .unwrap();
    let baseline = context.memory_stats().reserved_bytes;
    assert!(matches!(
        sequence
            .submit_preview(input, FrameOptions::default())
            .unwrap()
            .wait(),
        Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
    ));
    assert_eq!(sequence.next_frame_index(), FrameIndex::new(0));
    assert_eq!(context.memory_stats().reserved_bytes, baseline);
    assert!(owners.iter().all(|owner| owner.upgrade().is_none()));
}
