//! Physical storage is independent of RGB precision, color and transform topology.
use super::*;
use crate::{
    AnimationHeader, BackendError, ColorSampleFormat, Determinism, EncodeProfile,
    FrameEncodeRequest, FrameIndex, FrameOptions, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource,
    VarDctBackend, VarDctGroupOrder,
};
use jxl_gpu_formats::{ByteOrder, ColorSpecification, FloatPrecision, Swizzle, SwizzleComponent};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};

#[derive(Clone, Copy, Debug)]
pub(super) enum Topology {
    Single,
    Map,
    Tiled,
}

impl Topology {
    pub(super) fn backend(
        self,
        context: &WgpuContext,
        extent: Extent2d,
        config: &VarDctConfig,
    ) -> VarDctBackend {
        match self {
            Self::Single => {
                VarDctBackend::new_with_config(context, VarDctStrategy::Dct8, config.clone())
            }
            Self::Map => VarDctBackend::new_with_strategy_map(
                context,
                mixed::packed_map(extent.width, extent.height, false),
                config.clone(),
            ),
            Self::Tiled => VarDctBackend::new_tiled_dct8_with_config(context, config.clone()),
        }
        .unwrap()
    }
}

fn configuration(format: ColorSampleFormat, color: VarDctColorTransform) -> VarDctConfig {
    VarDctConfig {
        sample_format: format,
        ..precision::configuration(8, color)
    }
}

fn pixels(extent: Extent2d, format: ColorSampleFormat) -> Vec<[u32; 3]> {
    let (w, h) = (extent.width as usize, extent.height as usize);
    match format.float_precision() {
        Some(p) => floating::pixels(w, h, p),
        None => precision::pixels(w, h, format.bits_per_sample(), 91),
    }
}

fn check_pixels(
    oracles: &color::PixelOracles,
    bytes: &[u8],
    input: &[[u32; 3]],
    format: ColorSampleFormat,
) {
    match format.float_precision() {
        Some(p) => {
            floating::check_pixels(oracles, bytes, input, p);
        }
        None => {
            precision::check_pixels(oracles, bytes, input, format.bits_per_sample());
        }
    }
}

fn canonical(
    context: &WgpuContext,
    extent: Extent2d,
    format: ColorSampleFormat,
    input: &[[u32; 3]],
) -> BufferImageSource {
    precision::source_with_kind(
        context,
        extent.width as usize,
        extent.height as usize,
        format.bits_per_sample(),
        format.pixel_format().sample_kind,
        input,
        false,
    )
}

pub(super) fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    format: ColorSampleFormat,
    input: &[[u32; 3]],
    packing: Packing,
    order: ByteOrder,
) -> BufferImageSource {
    let mut pixel_format = format.pixel_format();
    pixel_format.byte_order = order;
    let (layout, bytes) = packing.pack(pixel_format, extent, input.as_flattened(), 4099);
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VarDCT source layout with poisoned padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

pub(super) fn sequence_source(
    context: &WgpuContext,
    extent: Extent2d,
    format: ColorSampleFormat,
    input: &[[u32; 3]],
    index: usize,
) -> BufferImageSource {
    upload(
        context,
        extent,
        format,
        input,
        Packing {
            storage: [Storage::Packed, Storage::Planar, Storage::Split][index % 3],
            reversed: true,
            shifted: true,
        },
        [ByteOrder::Native, ByteOrder::Big, ByteOrder::Little][index % 3],
    )
}

pub(super) fn request(extent: Extent2d, config: &VarDctConfig) -> FrameEncodeRequest {
    FrameEncodeRequest {
        frame_index: FrameIndex::new(0),
        is_last: true,
        profile: EncodeProfile::VarDct {
            quantization: config.quantization,
        },
        progressive: config.progressive.clone(),
        minimum_determinism: Determinism::SameDevice,
        animation: AnimationHeader::Still,
        canvas_width: extent.width,
        canvas_height: extent.height,
        options: FrameOptions::default(),
    }
}

pub(super) fn encode(
    context: &WgpuContext,
    backend: &VarDctBackend,
    config: &VarDctConfig,
    source: BufferImageSource,
) -> Vec<u8> {
    let extent = source.layout.extent;
    let artifacts = backend
        .submit(
            context,
            GpuFrameSource::Buffer(source),
            &request(extent, config),
        )
        .unwrap()
        .wait()
        .unwrap();
    let mut bytes = super::super::bitstream::image_header(
        extent.width,
        extent.height,
        AnimationHeader::Still,
        backend.color_plan,
    )
    .unwrap()
    .bytes()
    .to_vec();
    bytes.extend_from_slice(assemble_frame(artifacts.packets).unwrap().bytes());
    bytes
}

#[test]
fn source_layouts_all_precisions_preserve_independently_checked_codestreams() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracles = color::PixelOracles::new(&gpu);
    let formats = (1..=31)
        .map(|b| ColorSampleFormat::integer(crate::ColorChannels::Rgb, b).unwrap())
        .chain(floating::all_precisions().into_iter().map(floating::format));
    let extent = Extent2d::new(8, 8);
    for format in formats {
        let input = pixels(extent, format);
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = configuration(format, color);
            let backend = Topology::Single.backend(&context, extent, &config);
            let baseline = encode(
                &context,
                &backend,
                &config,
                canonical(&context, extent, format, &input),
            );
            check_pixels(&oracles, &baseline, &input, format);
            for index in 0..3 {
                let input = sequence_source(&context, extent, format, &input, index);
                assert_eq!(
                    encode(&context, &backend, &config, input),
                    baseline,
                    "{format:?}/{color:?}/{index}"
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn source_layouts_shared_mixed_and_three_byte_words_preserve_mapped_and_tiled_frames() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracles = color::PixelOracles::new(&gpu);
    for format in [1, 7, 8, 10, 16, 24, 31]
        .map(|b| ColorSampleFormat::integer(crate::ColorChannels::Rgb, b).unwrap())
        .into_iter()
        .chain(
            [(5, 2), (16, 5), (24, 7), (32, 8)]
                .map(|(b, e)| ColorSampleFormat::float(crate::ColorChannels::Rgb, b, e).unwrap()),
        )
    {
        for (topology, extent) in [
            (Topology::Map, Extent2d::new(25, 17)),
            (Topology::Tiled, Extent2d::new(259, 3)),
        ] {
            let input = pixels(extent, format);
            for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
                let mut config = configuration(format, color);
                config.progressive = progressive::combined();
                config.group_order = VarDctGroupOrder::saliency_first();
                let backend = topology.backend(&context, extent, &config);
                let baseline = encode(
                    &context,
                    &backend,
                    &config,
                    canonical(&context, extent, format, &input),
                );
                check_pixels(&oracles, &baseline, &input, format);
                let mut storages = vec![Storage::Packed, Storage::Planar, Storage::Split];
                if format.bits_per_sample() <= 10 {
                    storages.push(Storage::SharedWord);
                }
                if format.bits_per_sample() <= 8 {
                    storages.push(Storage::MixedWords);
                }
                if format.bits_per_sample() <= 24 {
                    storages.push(Storage::ThreeBytes);
                }
                for storage in storages {
                    for order in [ByteOrder::Little, ByteOrder::Big] {
                        let source = upload(
                            &context,
                            extent,
                            format,
                            &input,
                            Packing {
                                storage,
                                reversed: true,
                                shifted: true,
                            },
                            order,
                        );
                        let plan = backend.memory_plan(&source).unwrap();
                        if source.layout.planes.len() > 1 {
                            assert!(plan.source_binding_bytes < source.buffer.size());
                        }
                        assert_eq!(
                            encode(&context, &backend, &config, source),
                            baseline,
                            "{format:?}/{topology:?}/{color:?}/{storage:?}/{order:?}"
                        );
                    }
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn source_layouts_all_rgb_permutations_select_logical_components() {
    let context = test_context().expect("actual GPU required");
    let extent = Extent2d::new(17, 9);
    for format in [
        ColorSampleFormat::integer(crate::ColorChannels::Rgb, 7).unwrap(),
        ColorSampleFormat::float(crate::ColorChannels::Rgb, 16, 5).unwrap(),
    ] {
        let input = pixels(extent, format);
        let config = configuration(format, VarDctColorTransform::Xyb);
        let backend = Topology::Tiled.backend(&context, extent, &config);
        let baseline = encode(
            &context,
            &backend,
            &config,
            canonical(&context, extent, format, &input),
        );
        for permutation in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let stored: Vec<_> = input
                .iter()
                .map(|pixel| {
                    let mut physical = [0; 3];
                    for logical in 0..3 {
                        physical[permutation[logical]] = pixel[logical];
                    }
                    physical
                })
                .collect();
            for storage in [Storage::Packed, Storage::Planar, Storage::Split] {
                let mut source = upload(
                    &context,
                    extent,
                    format,
                    &stored,
                    Packing {
                        storage,
                        reversed: false,
                        shifted: true,
                    },
                    ByteOrder::Big,
                );
                let components = [
                    SwizzleComponent::X,
                    SwizzleComponent::Y,
                    SwizzleComponent::Z,
                ];
                source.layout.format.swizzle = Swizzle::Xyzw([
                    components[permutation[0]],
                    components[permutation[1]],
                    components[permutation[2]],
                    SwizzleComponent::One,
                ]);
                assert_eq!(encode(&context, &backend, &config, source), baseline);
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn source_layouts_nonfinite_samples_in_each_plane_and_last_word_never_publish() {
    let context = test_context().expect("actual GPU required");
    let extent = Extent2d::new(25, 17);
    for p in [
        FloatPrecision::new(5, 2).unwrap(),
        FloatPrecision::BINARY16,
        FloatPrecision::new(24, 7).unwrap(),
        FloatPrecision::BINARY32,
    ] {
        let format = floating::format(p);
        let input = pixels(extent, format);
        let config = configuration(format, VarDctColorTransform::Original);
        let fraction = p.bits() - p.exponent_bits() - 1;
        let infinity = ((1 << p.exponent_bits()) - 1) << fraction;
        for topology in [Topology::Map, Topology::Tiled] {
            let backend = topology.backend(&context, extent, &config);
            for index in 0..3 {
                for c in 0..3 {
                    for value in [infinity, infinity | 1, infinity | (1 << (p.bits() - 1))] {
                        let mut corrupt = input.clone();
                        corrupt.last_mut().unwrap()[c] = value;
                        let source = sequence_source(&context, extent, format, &corrupt, index);
                        let result = backend
                            .submit(
                                &context,
                                GpuFrameSource::Buffer(source),
                                &request(extent, &config),
                            )
                            .unwrap()
                            .wait();
                        assert!(
                            matches!(
                                result,
                                Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
                            ),
                            "{p:?}/{topology:?}/{index}/{c}: {result:?}"
                        );
                        assert_eq!(context.memory_stats().reserved_bytes, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn source_layouts_admission_cancellation_and_completion_obey_exact_budget() {
    let context = test_context().expect("actual GPU required");
    let extent = Extent2d::new(25, 17);
    for format in [
        ColorSampleFormat::integer(crate::ColorChannels::Rgb, 31).unwrap(),
        ColorSampleFormat::float(crate::ColorChannels::Rgb, 24, 7).unwrap(),
    ] {
        let input = pixels(extent, format);
        let config = configuration(format, VarDctColorTransform::Original);
        for topology in [Topology::Map, Topology::Tiled] {
            let backend = topology.backend(&context, extent, &config);
            for index in [1, 2] {
                let source = sequence_source(&context, extent, format, &input, index);
                let owned = backend.memory_plan(&source).unwrap().owned_bytes_per_job;
                let mut invalid = Vec::new();
                let mut wrong = source.clone();
                wrong.layout.logical_size -= 1;
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.planes[1].offset = wrong.layout.planes[0].offset;
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.planes[1].row_stride = u64::MAX;
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.format.color_spec = ColorSpecification::Undefined;
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.format.swizzle = Swizzle::Xyzw([
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::Z,
                    SwizzleComponent::One,
                ]);
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.planes[0].sample_extent.width -= 1;
                invalid.push(wrong);
                let mut wrong = source.clone();
                wrong.layout.planes[0].offset = wrong.buffer.size();
                invalid.push(wrong);
                for wrong in invalid {
                    assert!(backend.memory_plan(&wrong).is_err());
                    assert!(
                        backend
                            .submit(
                                &context,
                                GpuFrameSource::Buffer(wrong),
                                &request(extent, &config)
                            )
                            .is_err()
                    );
                    assert_eq!(context.memory_stats().reserved_bytes, 0);
                }
                for limit in [owned - 1, owned] {
                    let bounded = WgpuContext::with_memory_budget(
                        Arc::new(context.device().clone()),
                        Arc::new(context.queue().clone()),
                        NonZeroU64::new(limit).unwrap(),
                    )
                    .unwrap();
                    let backend = topology.backend(&bounded, extent, &config);
                    let job = backend.submit(
                        &bounded,
                        GpuFrameSource::Buffer(source.clone()),
                        &request(extent, &config),
                    );
                    if limit < owned {
                        assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
                    } else {
                        let job = job.unwrap();
                        assert_eq!(bounded.memory_stats().reserved_bytes, owned);
                        assert!(matches!(
                            backend.submit(
                                &bounded,
                                GpuFrameSource::Buffer(source.clone()),
                                &request(extent, &config)
                            ),
                            Err(EncodeError::MemoryBackpressure(_))
                        ));
                        drop(job);
                        bounded
                            .device()
                            .poll(wgpu::PollType::wait_indefinitely())
                            .unwrap();
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(2);
                        while bounded.memory_stats().reserved_bytes != 0
                            && std::time::Instant::now() < deadline
                        {
                            bounded.device().poll(wgpu::PollType::Poll).unwrap();
                            std::thread::yield_now();
                        }
                        // Completion callbacks release canceled jobs, including every source binding.
                        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                        let result = backend
                            .submit(
                                &bounded,
                                GpuFrameSource::Buffer(source.clone()),
                                &request(extent, &config),
                            )
                            .unwrap()
                            .wait()
                            .unwrap();
                        // Validated packets have been copied to host ownership; GPU storage is free
                        // even while those completed packets remain available to the caller.
                        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                        drop(result);
                    }
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

#[test]
fn source_layouts_saliency_uses_logical_samples_in_every_workgroup_variant() {
    let (device, queue, info) = test_device().expect("actual GPU required");
    let extent = Extent2d::new(259, 3);
    let formats = [
        ColorSampleFormat::integer(crate::ColorChannels::Rgb, 7).unwrap(),
        ColorSampleFormat::integer(crate::ColorChannels::Rgb, 31).unwrap(),
        ColorSampleFormat::float(crate::ColorChannels::Rgb, 5, 2).unwrap(),
        ColorSampleFormat::float(crate::ColorChannels::Rgb, 32, 8).unwrap(),
    ];
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
        for format in formats {
            let input = pixels(extent, format);
            let proxy: Vec<_> = input
                .iter()
                .map(|p| {
                    p.map(|v| {
                        if let Some(p) = format.float_precision() {
                            (floating::value(v, p).clamp(0.0, 1.0) * 255.0).round() as u8
                        } else if format.bits_per_sample() < 8 {
                            let mask = (1 << format.bits_per_sample()) - 1;
                            ((v * 255 + mask / 2) / mask) as u8
                        } else {
                            (v >> (format.bits_per_sample() - 8)) as u8
                        }
                    })
                })
                .collect();
            let expected = saliency::oracle(extent.width as usize, extent.height as usize, &proxy);
            let mut config = configuration(format, VarDctColorTransform::Xyb);
            config.group_order = VarDctGroupOrder::saliency_first();
            for topology in [Topology::Map, Topology::Tiled] {
                let backend = topology.backend(&context, extent, &config);
                assert_eq!(backend.workgroup_variant(), variant);
                let source = sequence_source(&context, extent, format, &input, 1);
                let (records, _) = backend
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(source),
                        &request(extent, &config),
                    )
                    .unwrap()
                    .wait_with_saliency_for_test()
                    .unwrap();
                assert_eq!(
                    records
                        .iter()
                        .map(|r| (u64::from(r.edges), u64::from(r.contrast)))
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}
