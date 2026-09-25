use super::*;
use crate::VarDctGroupOrder;

#[test]
fn floating_precision_saliency_proxy_is_bounded_and_exact_in_every_workgroup_variant() {
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
        for p in [
            FloatPrecision::new(5, 2).unwrap(),
            FloatPrecision::BINARY16,
            FloatPrecision::new(24, 7).unwrap(),
            FloatPrecision::BINARY32,
        ] {
            let config = VarDctConfig {
                group_order: VarDctGroupOrder::saliency_first(),
                sample_format: format(p),
                ..precision::configuration(8, VarDctColorTransform::Original)
            };
            let (w, h) = (259, 3);
            let input = pixels(w, h, p);
            let proxy: Vec<_> = input
                .iter()
                .map(|pixel| pixel.map(|v| (value(v, p).clamp(0.0, 1.0) * 255.0).round() as u8))
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
                        GpuFrameSource::Buffer(source(&context, w, h, p, &input)),
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
fn floating_precision_rejects_mismatches_before_admission_and_releases_canceled_jobs() {
    for bits in 0..=u8::MAX {
        for exponent in 0..=u8::MAX {
            assert_eq!(
                ColorSampleFormat::float(crate::ColorChannels::Rgb, bits, exponent).is_ok(),
                FloatPrecision::new(bits, exponent).is_ok()
            );
        }
    }
    let context = test_context().expect("actual GPU required");
    let (w, h) = (25, 17);
    for p in [
        FloatPrecision::new(5, 2).unwrap(),
        FloatPrecision::BINARY16,
        FloatPrecision::new(24, 7).unwrap(),
        FloatPrecision::BINARY32,
    ] {
        let input = pixels(w, h, p);
        let source = source(&context, w, h, p, &input);
        let config = VarDctConfig {
            sample_format: format(p),
            ..precision::configuration(8, VarDctColorTransform::Original)
        };
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
            assert_eq!(plan.source_validation_bytes != 0, mapped);
            assert_eq!(plan.source_validation_bytes % 256, 0);
            let bytes = plan.owned_bytes_per_job;
            let request = request(w, h, &config);
            let mut invalid = Vec::new();
            let mut wrong = source.clone();
            wrong.layout.format = ColorSampleFormat::float(
                crate::ColorChannels::Rgb,
                16,
                if p == FloatPrecision::BINARY16 { 4 } else { 5 },
            )
            .unwrap()
            .pixel_format();
            invalid.push(wrong);
            let mut wrong = source.clone();
            wrong.layout.format.color_spec = jxl_gpu_formats::ColorSpecification::Undefined;
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

#[test]
fn floating_precision_conversion_preserves_finite_binary32_bits() {
    let context = test_context().expect("actual GPU required");
    let device = context.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("floating source field conversion boundaries"),
        source: wgpu::ShaderSource::Wgsl(
            shader_source(
                r"
            @compute @workgroup_size(64)
            fn check_float_fields(@builtin(local_invocation_index) i: u32) {
                if i < arrayLength(&source_words) {
                    artifact_words[i] = bitcast<u32>(normalize_source_sample(source_words[i]));
                }
            }
        ",
            )
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("floating source field conversion boundaries"),
        layout: None,
        module: &shader,
        entry_point: Some("check_float_fields"),
        compilation_options: Default::default(),
        cache: None,
    });
    for p in all_precisions() {
        let f = p.bits() - p.exponent_bits() - 1;
        let bias = (1 << (p.exponent_bits() - 1)) - 1;
        let infinity = ((1u32 << p.exponent_bits()) - 1) << f;
        let values: Vec<_> = [0, 1, (1 << f) - 1, 1 << f, bias << f, infinity - 1]
            .into_iter()
            .flat_map(|v| [v, v | (1 << (p.bits() - 1))])
            .collect();
        let expected: Vec<_> = values
            .iter()
            .map(|&v| (value(v, p) as f32).to_bits())
            .collect();
        let mut params: super::super::super::types::VarDctKernelParams =
            bytemuck::Zeroable::zeroed();
        params.source_sample_mask = format(p).sample_mask();
        params.source_exponent_bits = u32::from(p.exponent_bits());
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (values.len() * 4) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut commands = device.create_command_encoder(&Default::default());
        {
            let mut pass = commands.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        commands.copy_buffer_to_buffer(&output, 0, &staging, 0, size);
        let submission = context.queue().submit([commands.finish()]);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging.map_async(wgpu::MapMode::Read, .., move |r| tx.send(r).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.get_mapped_range(..).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&mapped), expected, "{p:?}");
        drop(mapped);
        staging.unmap();
    }
}

#[test]
fn floating_precision_extended_original_values_and_finite_overflow_are_distinct() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracles = color::PixelOracles::new(&gpu);
    let native = native::native_oracles();
    let p = FloatPrecision::BINARY32;
    let config = VarDctConfig {
        sample_format: format(p),
        ..precision::configuration(8, VarDctColorTransform::Original)
    };
    let words: Vec<_> = (0..64)
        .map(|i| std::array::from_fn(|c| [-0.5f32, -0.0, 0.25, 1.25, 1.5][(i + c) % 5].to_bits()))
        .collect();
    let encoded = single(&context, &config, VarDctStrategy::Dct8, &native[0], &words);
    check_pixels(&oracles, &encoded, &words, p);
    for mapped in [false, true] {
        let encoder = if mapped {
            VarDctBackend::new_with_strategy_map(
                &context,
                mixed::packed_map(8, 8, false),
                config.clone(),
            )
        } else {
            VarDctBackend::new_tiled_dct8_with_config(&context, config.clone())
        }
        .unwrap();
        let words = vec![[f32::MAX.to_bits(); 3]; 64];
        let result = encoder
            .submit(
                &context,
                GpuFrameSource::Buffer(source(&context, 8, 8, p, &words)),
                &request(8, 8, &config),
            )
            .unwrap()
            .wait();
        assert!(matches!(
            result,
            Err(EncodeError::Backend(
                BackendError::VarDctQuantizationOverflow { .. }
            ))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}
