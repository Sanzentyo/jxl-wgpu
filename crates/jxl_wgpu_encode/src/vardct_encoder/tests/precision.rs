//! Integer precision crosses source storage, normalization, metadata and frame codecs.

pub(super) mod linear;

use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuEncodeJob, GpuFrameSource, RgbSampleFormat, VarDctBackend,
    VarDctGroupOrder,
};
use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{
    ByteOrder, Channel, ChromaSubsampling, ColorModel, ColorSpecification, PackingField,
    PackingWord, PixelFormat, PlaneFormat, PlaneSampling, SampleKind, Swizzle,
};

pub(super) fn configuration(bits: u8, color_transform: VarDctColorTransform) -> VarDctConfig {
    VarDctConfig {
        sample_format: RgbSampleFormat::integer(bits).unwrap(),
        color_transform,
        quantization: VarDctQuantization::new(
            35_252,
            256,
            crate::VarDctHfMultiplier::new(48).unwrap(),
        )
        .unwrap(),
        ..Default::default()
    }
}

pub(super) fn pixels(width: usize, height: usize, bits: u8, seed: u32) -> Vec<[u32; 3]> {
    let maximum = (1u64 << bits) - 1;
    (0..width * height)
        .map(|i| {
            std::array::from_fn(|channel| {
                // Full-range endpoints, irregular low bits and cross-channel texture.
                let n = (i as u64 * 2_654_435_761 + channel as u64 * 1_234_567 + u64::from(seed))
                    % 4_294_967_296;
                match (i + channel) % 19 {
                    0 => 0,
                    1 => maximum as u32,
                    _ => ((n * maximum + 2_147_483_647) / 4_294_967_295) as u32,
                }
            })
        })
        .collect()
}

pub(super) fn normalized(pixels: &[[u32; 3]], bits: u8) -> Vec<[f64; 3]> {
    let maximum = ((1u64 << bits) - 1) as f64;
    pixels
        .iter()
        .map(|p| p.map(|v| f64::from(v) / maximum))
        .collect()
}

/// Independent packing, including poisoned high bits, unaligned words and padded rows.
pub(super) fn source(
    context: &WgpuContext,
    width: usize,
    height: usize,
    bits: u8,
    pixels: &[[u32; 3]],
    poison: bool,
) -> BufferImageSource {
    let source = source_with_kind(
        context,
        width,
        height,
        bits,
        SampleKind::Unsigned,
        pixels,
        poison,
    );
    assert_eq!(
        RgbSampleFormat::integer(bits).unwrap().pixel_format(),
        source.layout.format
    );
    source
}

pub(super) fn source_with_kind(
    context: &WgpuContext,
    width: usize,
    height: usize,
    bits: u8,
    kind: SampleKind,
    pixels: &[[u32; 3]],
    poison: bool,
) -> BufferImageSource {
    assert_eq!(pixels.len(), width * height);
    let bytes = match bits {
        1..=8 => 1,
        9..=16 => 2,
        _ => 4,
    };
    let offset = if poison { 261 } else { 5 };
    let row_bytes = width * 3 * bytes;
    let row_stride = row_bytes + 5;
    let size = (offset + row_stride * (height - 1) + row_bytes).next_multiple_of(4);
    let mut allocation = vec![if poison { 0xa5 } else { 0 }; size];
    for (i, pixel) in pixels.iter().enumerate() {
        let start = offset + i / width * row_stride + i % width * 3 * bytes;
        for (channel, &value) in pixel.iter().enumerate() {
            let word = value
                | if poison {
                    u32::MAX.checked_shl(u32::from(bits)).unwrap_or(0)
                } else {
                    0
                };
            allocation[start + channel * bytes..start + (channel + 1) * bytes]
                .copy_from_slice(&word.to_le_bytes()[..bytes]);
        }
    }
    let format = PixelFormat {
        model: ColorModel::Rgb,
        color_spec: ColorSpecification::Default,
        chroma_subsampling: ChromaSubsampling::None,
        sample_kind: kind,
        byte_order: ByteOrder::Native,
        swizzle: Swizzle::XYZ1,
        planes: vec![PlaneFormat {
            sampling: PlaneSampling::FULL,
            pixels_per_element: 1,
            words: [Channel::X, Channel::Y, Channel::Z]
                .map(|channel| {
                    let mut fields = vec![PackingField::channel(channel, bits)];
                    if bits as usize != 8 * bytes {
                        fields.insert(0, PackingField::padding((8 * bytes) as u8 - bits));
                    }
                    PackingWord { fields }
                })
                .into(),
        }],
    };
    let extent = Extent2d::new(width as u32, height as u32);
    let layout = ImageLayout::from_planes(
        extent,
        format,
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: offset as u64,
            row_stride: row_stride as u64,
            sample_extent: extent,
            row_bytes: row_bytes as u64,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("integer RGB precision fixture"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

fn request(width: usize, height: usize, config: &VarDctConfig) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: width as u32,
        canvas_height: height as u32,
        options: FrameOptions::default(),
    }
}

pub(super) fn check_header(encoded: &[u8], bits: u8, color: VarDctColorTransform) {
    let inventory = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        inventory.image_header.bit_depth,
        SampleBitDepth::Integer {
            bits_per_sample: u32::from(bits)
        }
    );
    assert_eq!(
        inventory.image_header.xyb_encoded,
        color == VarDctColorTransform::Xyb
    );
    assert!(inventory.frames.iter().all(|f| !f.do_ycbcr));
    if inventory
        .frames
        .iter()
        .any(|f| f.encoding == jxl_gpu_bitstream::FrameEncoding::VarDct)
    {
        assert!(!inventory.image_header.modular_16bit_buffers);
    }
}

pub(super) fn check_pixels(
    oracles: &color::PixelOracles,
    encoded: &[u8],
    input: &[[u32; 3]],
    bits: u8,
) -> Vec<f32> {
    let (actual, rust) = oracles.check_decoders_with_rust(encoded);
    check_quality(&actual, input, bits);
    rust
}

fn check_quality(actual: &[f32], input: &[[u32; 3]], bits: u8) {
    let expected = normalized(input, bits);
    check_normalized_quality(actual, &expected);
}

pub(super) fn check_normalized_quality(actual: &[f32], expected: &[[f64; 3]]) {
    assert_eq!(actual.len(), expected.len() * 4);
    let error = actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected)
        .flat_map(|(a, b)| (0..3).map(move |c| (f64::from(a[c]) - b[c]).powi(2)))
        .sum::<f64>()
        / (3 * expected.len()) as f64;
    assert!(
        -10.0 * error.log10() > 30.0,
        "source PSNR {}",
        -10.0 * error.log10()
    );
}

fn single(
    context: &WgpuContext,
    config: &VarDctConfig,
    strategy: VarDctStrategy,
    oracle: &native::Oracle,
    input: &[[u32; 3]],
) -> Vec<u8> {
    let extent = strategy.pixel_extent();
    let (w, h) = (extent.width as usize, extent.height as usize);
    let bits = config.sample_format.bits_per_sample();
    let components: Vec<_> = normalized(input, bits)
        .into_iter()
        .map(|p| match config.color_transform {
            VarDctColorTransform::Xyb => reference::xyb_normalized(p),
            VarDctColorTransform::Original => p,
        })
        .collect();
    let coefficients = native::forward_samples(&components, w, h, oracle);
    let backend = VarDctBackend::new_with_config(context, strategy, config.clone()).unwrap();
    let (words, length, artifacts) = backend
        .submit(
            context,
            GpuFrameSource::Buffer(source(context, w, h, bits, input, true)),
            &request(w, h, config),
        )
        .unwrap()
        .wait_with_ac_for_test()
        .unwrap();
    assert!(native::check_ac(&words, length, &coefficients, oracle, config.clone()) > 0);
    let frame = assemble_frame(artifacts.packets).unwrap();
    let mut encoded = super::super::bitstream::image_header(
        extent.width,
        extent.height,
        AnimationHeader::Still,
        backend.color_plan,
    )
    .unwrap()
    .bytes()
    .to_vec();
    encoded.extend_from_slice(frame.bytes());
    encoded
}

#[test]
fn integer_precision_all_depths_bind_storage_gpu_normalization_and_headers() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&backend);
    let oracles = color::PixelOracles::new(&backend);
    let native = native::native_oracles();
    for bits in 1..=31 {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = configuration(bits, color);
            let input = pixels(8, 8, bits, 0);
            let encoded = single(&context, &config, VarDctStrategy::Dct8, &native[0], &input);
            check_header(&encoded, bits, color);
            check_pixels(&oracles, &encoded, &input, bits);
            let encoder = VarDctEncoder::new_with_config(
                context.clone(),
                VarDctStrategy::Dct8,
                config.clone(),
            )
            .unwrap();
            assert_eq!(encoder.sample_format(), config.sample_format);
            assert_eq!(
                encoded,
                encoder
                    .encode(source(&context, 8, 8, bits, &input, false))
                    .unwrap(),
                "unused bits, binding offset or row padding affected {bits}-bit output"
            );
            for (w, h, mapped) in [(25, 17, true), (259, 3, false)] {
                let config = VarDctConfig {
                    progressive: progressive::combined(),
                    group_order: VarDctGroupOrder::saliency_first(),
                    ..config.clone()
                };
                let input = pixels(w, h, bits, 17);
                let source = source(&context, w, h, bits, &input, true);
                let encoded = if mapped {
                    VarDctEncoder::new_with_strategy_map(
                        context.clone(),
                        mixed::packed_map(w as u32, h as u32, false),
                        config,
                    )
                    .unwrap()
                    .encode(source)
                } else {
                    TiledVarDctEncoder::new_with_config(context.clone(), config)
                        .unwrap()
                        .encode(source)
                }
                .unwrap();
                check_header(&encoded, bits, color);
                check_pixels(&oracles, &encoded, &input, bits);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn integer_precision_wide_words_cover_every_transform_with_independent_coefficients() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&backend);
    let oracles = linear::LinearPixelOracles::new(&backend);
    let native = native::native_oracles();
    for (bits, color) in [
        (16, VarDctColorTransform::Xyb),
        (31, VarDctColorTransform::Original),
    ] {
        for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&native) {
            eprintln!("{bits}-bit/{color:?}/{strategy:?}");
            let extent = strategy.pixel_extent();
            let input = pixels(extent.width as usize, extent.height as usize, bits, 91);
            let config = VarDctConfig {
                coefficient_orders: orders::selected([strategy]),
                lf_metadata: custom_lf_metadata(),
                ..configuration(bits, color)
            };
            let encoded = single(&context, &config, strategy, oracle, &input);
            check_header(&encoded, bits, color);
            oracles.check(&encoded, &input, bits);
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn integer_precision_saliency_proxy_is_bounded_and_exact_in_every_workgroup_variant() {
    let (device, queue, info) = test_device().expect("actual GPU required");
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context = test_context_with_variants(
            &device,
            &queue,
            &info,
            &[(TILED_KERNEL_KEY, variant), (FORWARD_KERNEL_KEY, variant)],
        )
        .unwrap();
        for bits in [1, 9, 17, 31] {
            let config = VarDctConfig {
                group_order: VarDctGroupOrder::saliency_first(),
                ..configuration(bits, VarDctColorTransform::Original)
            };
            let (w, h) = (259, 3);
            let input = pixels(w, h, bits, 71);
            let proxy: Vec<_> = input
                .iter()
                .map(|p| {
                    p.map(|v| {
                        if bits < 8 {
                            (f64::from(v) * 255.0 / ((1u64 << bits) - 1) as f64).round() as u8
                        } else {
                            (u64::from(v) / (1u64 << (bits - 8))) as u8
                        }
                    })
                })
                .collect();
            for mapped in [false, true] {
                let encoder = if mapped {
                    VarDctBackend::new_with_strategy_map(
                        &context,
                        mixed::packed_map(w as u32, h as u32, false),
                        config.clone(),
                    )
                } else {
                    VarDctBackend::new_tiled_dct8_with_config(&context, config.clone())
                }
                .unwrap();
                assert_eq!(encoder.workgroup_variant(), variant);
                let (records, _) = encoder
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(source(&context, w, h, bits, &input, true)),
                        &request(w, h, &config),
                    )
                    .unwrap()
                    .wait_with_saliency_for_test()
                    .unwrap();
                assert_eq!(
                    records
                        .iter()
                        .map(|r| (u64::from(r.edges), u64::from(r.contrast)))
                        .collect::<Vec<_>>(),
                    saliency::oracle(w, h, &proxy)
                );
            }
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn integer_precision_rejects_mismatches_before_admission_and_releases_canceled_jobs() {
    for bits in [0, 32, 64, 255] {
        assert!(RgbSampleFormat::integer(bits).is_err());
    }
    let context = test_context().expect("actual GPU required");
    let (w, h) = (25, 17);
    for bits in [1, 8, 9, 16, 17, 31] {
        let input = pixels(w, h, bits, 33);
        let source = source(&context, w, h, bits, &input, true);
        let config = configuration(bits, VarDctColorTransform::Original);
        for mapped in [false, true] {
            let make = |context: &WgpuContext| {
                if mapped {
                    VarDctBackend::new_with_strategy_map(
                        context,
                        mixed::packed_map(w as u32, h as u32, false),
                        config.clone(),
                    )
                } else {
                    VarDctBackend::new_tiled_dct8_with_config(context, config.clone())
                }
                .unwrap()
            };
            let encoder = make(&context);
            let plan = encoder.memory_plan(&source).unwrap();
            let bytes = plan.owned_bytes_per_job;
            let request = request(w, h, &config);
            let mut invalid = Vec::new();
            let mut wrong = source.clone();
            wrong.layout.format = RgbSampleFormat::integer(if bits == 8 { 7 } else { 8 })
                .unwrap()
                .pixel_format();
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.byte_order = ByteOrder::Big;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.sample_kind = SampleKind::Signed;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].row_bytes -= 1;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].row_stride = u64::MAX;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].offset += 4;
            invalid.push(wrong);
            for wrong in invalid {
                assert!(encoder.memory_plan(&wrong).is_err());
                assert!(
                    encoder
                        .submit(&context, GpuFrameSource::Buffer(wrong), &request)
                        .is_err()
                );
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
            for limit in [bytes - 1, bytes] {
                let bounded = WgpuContext::with_memory_budget(
                    Arc::new(context.device().clone()),
                    Arc::new(context.queue().clone()),
                    NonZeroU64::new(limit).unwrap(),
                )
                .unwrap();
                let encoder = make(&bounded);
                let job =
                    encoder.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request);
                if limit < bytes {
                    assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
                } else {
                    let job = job.unwrap();
                    assert_eq!(bounded.memory_stats().reserved_bytes, bytes);
                    assert!(matches!(
                        encoder.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request),
                        Err(EncodeError::MemoryBackpressure(_))
                    ));
                    drop(job);
                    bounded
                        .device()
                        .poll(wgpu::PollType::wait_indefinitely())
                        .unwrap();
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                    while bounded.memory_stats().reserved_bytes != 0
                        && std::time::Instant::now() < deadline
                    {
                        bounded.device().poll(wgpu::PollType::Poll).unwrap();
                        std::thread::yield_now();
                    }
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    encoder
                        .submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request)
                        .unwrap()
                        .wait()
                        .unwrap();
                }
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            }
        }
    }
}
