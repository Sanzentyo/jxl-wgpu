use super::*;
use jxl_gpu_formats::{ColorSpecification, SampleKind};

#[test]
fn gray_input_nonfinite_samples_never_publish_a_frame() {
    let context = test_context().expect("actual GPU required");
    let extent = Extent2d::new(13, 9);
    for format in all_formats().filter(|f| f.float_precision().is_some()) {
        let p = format.float_precision().unwrap();
        let fraction = p.bits() - p.exponent_bits() - 1;
        let infinity = ((1u32 << p.exponent_bits()) - 1) << fraction;
        let sign = 1u32 << (p.bits() - 1);
        for (topology, color) in [
            (Topology::Map, VarDctColorTransform::Xyb),
            (Topology::Tiled, VarDctColorTransform::Original),
        ] {
            let config = configuration(format, color);
            let backend = topology.backend(&context, extent, &config);
            for (index, invalid) in [
                infinity,
                infinity | sign,
                infinity | 1,
                infinity | sign | (1 << (fraction - 1)),
            ]
            .into_iter()
            .enumerate()
            {
                let mut input = pixels(extent, format);
                let last = input.len() - 1;
                input[if index.is_multiple_of(2) { 0 } else { last }] = invalid;
                let source = upload(&context, extent, format, &input, true);
                let result = backend
                    .submit(
                        &context,
                        GpuFrameSource::Buffer(source),
                        &layouts::request(extent, &config),
                    )
                    .unwrap()
                    .wait();
                assert!(
                    matches!(
                        result,
                        Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
                    ),
                    "{format:?}/{topology:?}/{invalid:x}: {result:?}"
                );
                assert_eq!(context.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn gray_input_admission_cancellation_and_retained_packets_obey_exact_budget() {
    let context = test_context().expect("actual GPU required");
    let extent = Extent2d::new(25, 17);
    for format in [
        ColorSampleFormat::integer(ColorChannels::Gray, 31).unwrap(),
        ColorSampleFormat::float(ColorChannels::Gray, 24, 7).unwrap(),
    ] {
        let input = pixels(extent, format);
        let source = upload(&context, extent, format, &input, true);
        let config = configuration(format, VarDctColorTransform::Original);
        for topology in [Topology::Map, Topology::Tiled] {
            let backend = topology.backend(&context, extent, &config);
            let owned = backend.memory_plan(&source).unwrap().owned_bytes_per_job;
            let mut invalid = Vec::new();
            let mut wrong = source.clone();
            wrong.layout.format = ColorSampleFormat::integer(ColorChannels::Rgb, 31)
                .unwrap()
                .pixel_format();
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.color_spec = ColorSpecification::Undefined;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.sample_kind = SampleKind::Signed;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.swizzle = Swizzle::XYZ1;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].offset = wrong.buffer.size();
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].row_stride = u64::MAX;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.logical_size -= 1;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.planes[0].sample_extent.width -= 1;
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.swizzle = Swizzle::Xyzw([
                SwizzleComponent::W,
                SwizzleComponent::Zero,
                SwizzleComponent::Zero,
                SwizzleComponent::W,
            ]);
            invalid.push(wrong);
            for wrong in invalid {
                assert!(backend.memory_plan(&wrong).is_err());
                assert!(
                    backend
                        .submit(
                            &context,
                            GpuFrameSource::Buffer(wrong),
                            &layouts::request(extent, &config)
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
                let submit = || {
                    backend.submit(
                        &bounded,
                        GpuFrameSource::Buffer(source.clone()),
                        &layouts::request(extent, &config),
                    )
                };
                if limit < owned {
                    assert!(matches!(submit(), Err(EncodeError::MemoryBackpressure(_))));
                } else {
                    let job = submit().unwrap();
                    assert_eq!(bounded.memory_stats().reserved_bytes, owned);
                    assert!(matches!(submit(), Err(EncodeError::MemoryBackpressure(_))));
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
                    let retained = submit().unwrap().wait().unwrap();
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    assert!(!assemble_frame(retained.packets).unwrap().bytes().is_empty());
                }
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            }
        }
    }
}

#[test]
fn gray_input_saliency_and_packing_are_exact_in_every_workgroup_variant() {
    let (device, queue, info) = test_device().expect("actual GPU required");
    let extent = Extent2d::new(259, 3);
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
        for format in [
            ColorSampleFormat::integer(ColorChannels::Gray, 7).unwrap(),
            ColorSampleFormat::integer(ColorChannels::Gray, 31).unwrap(),
            ColorSampleFormat::float(ColorChannels::Gray, 5, 2).unwrap(),
            ColorSampleFormat::float(ColorChannels::Gray, 32, 8).unwrap(),
        ] {
            let input = pixels(extent, format);
            let proxy: Vec<_> = input
                .iter()
                .map(|&v| {
                    [if let Some(p) = format.float_precision() {
                        (floating::value(v, p).clamp(0.0, 1.0) * 255.0).round() as u8
                    } else if format.bits_per_sample() < 8 {
                        let mask = (1 << format.bits_per_sample()) - 1;
                        ((v * 255 + mask / 2) / mask) as u8
                    } else {
                        (v >> (format.bits_per_sample() - 8)) as u8
                    }; 3]
                })
                .collect();
            let expected = saliency::oracle(extent.width as usize, extent.height as usize, &proxy);
            let mut config = configuration(format, VarDctColorTransform::Xyb);
            config.group_order = crate::VarDctGroupOrder::saliency_first();
            for topology in [Topology::Map, Topology::Tiled] {
                let backend = topology.backend(&context, extent, &config);
                assert_eq!(backend.workgroup_variant(), variant);
                let baseline = layouts::encode(
                    &context,
                    &backend,
                    &config,
                    upload(&context, extent, format, &input, false),
                );
                for (index, channel) in [Channel::X, Channel::Y, Channel::Z, Channel::W]
                    .into_iter()
                    .enumerate()
                {
                    let storage = if format.bits_per_sample() <= 24 {
                        Storage::ThreeBytes
                    } else {
                        Storage::Packed
                    };
                    let source = upload_packing(
                        &context,
                        extent,
                        format,
                        &input,
                        Packing {
                            storage,
                            reversed: false,
                            shifted: true,
                        },
                        if index.is_multiple_of(2) {
                            ByteOrder::Little
                        } else {
                            ByteOrder::Big
                        },
                        channel,
                    );
                    assert_eq!(
                        layouts::encode(&context, &backend, &config, source.clone()),
                        baseline
                    );
                    let (records, _) = backend
                        .submit(
                            &context,
                            GpuFrameSource::Buffer(source),
                            &layouts::request(extent, &config),
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
        }
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn gray_input_degenerate_axes_cross_group_and_lf_group_boundaries() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracle = color::PixelOracles::new(&gpu);
    for extent in [
        Extent2d::new(1, 1),
        Extent2d::new(1, 257),
        Extent2d::new(255, 1),
        Extent2d::new(256, 1),
        Extent2d::new(2049, 1),
        Extent2d::new(1, 2049),
    ] {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let format = ColorSampleFormat::float(ColorChannels::Gray, 16, 5).unwrap();
            let input = pixels(extent, format);
            let config = VarDctConfig {
                progressive: progressive::combined(),
                ..configuration(format, color)
            };
            for topology in [Topology::Map, Topology::Tiled] {
                let backend = topology.backend(&context, extent, &config);
                let bytes = layouts::encode(
                    &context,
                    &backend,
                    &config,
                    upload(&context, extent, format, &input, true),
                );
                check_header(&bytes, format, color);
                precision::check_normalized_quality(
                    &oracle.check_decoders(&bytes),
                    &normalized(&input, format),
                );
            }
        }
    }
}
