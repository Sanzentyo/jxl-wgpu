use super::*;
use crate::{EncodeError, GpuEncodeJob, UnsupportedFeature};

#[test]
fn enumerated_normalization_matches_f64_at_transfer_and_signed_float_boundaries() {
    let context = test_context().unwrap();
    let device = context.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("declared source color normalization"),
        source: wgpu::ShaderSource::Wgsl(
            shader_source(
                r"
            @compute @workgroup_size(64)
            fn probe(@builtin(local_invocation_index) i: u32) {
                if i < params.width {
                    let value = normalize_rgb(i, 0u);
                    for (var c = 0u; c < 3u; c += 1u) {
                        artifact_words[3u*i+c] = bitcast<u32>(value[c]);
                    }
                }
            }
        ",
            )
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for index in 0..49 {
            for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
                let mut config = oracle::config(channels, index, transform);
                config.sample_format = ColorSampleFormat::float(channels, 32, 8).unwrap();
                let values = [
                    -0.25f32,
                    -0.0,
                    0.0,
                    f32::from_bits(1),
                    0.0031308,
                    0.018,
                    0.04045,
                    0.081,
                    0.5,
                    0.75,
                    1.0,
                ];
                let rgb: Vec<_> = (0..values.len())
                    .map(|i| {
                        std::array::from_fn(|c| {
                            f64::from(
                                values[(i + if channels == ColorChannels::Gray {
                                    0
                                } else {
                                    c * 2
                                }) % values.len()],
                            )
                        })
                    })
                    .collect();
                let words: Vec<_> = rgb
                    .iter()
                    .flat_map(|p| {
                        p[..channels.count() as usize]
                            .iter()
                            .map(|v| (*v as f32).to_bits())
                    })
                    .collect();
                let extent = Extent2d::new(rgb.len() as u32, 1);
                let source = upload(&context, extent, &config, &words, false);
                let layout = crate::source::SourceLayout::new(
                    &source.layout,
                    source.buffer.size(),
                    u64::from(device.limits().min_storage_buffer_offset_alignment),
                )
                .unwrap();
                let plan = VarDctColorPlan::new(&config).unwrap();
                let mut params: super::super::super::types::VarDctKernelParams =
                    bytemuck::Zeroable::zeroed();
                params.width = extent.width;
                let (sources, offsets) =
                    plan.bind_sources(&layout.region(0, 0, extent.width, 1).unwrap());
                params.sources = sources;
                for (source, offset) in params.sources.iter_mut().zip(offsets) {
                    source.byte_offset = u32::try_from(offset).unwrap();
                }
                params.source_sample_mask = u32::MAX;
                params.source_exponent_bits = 8;
                params.source_color = plan.gpu;
                params.color_normalization = plan.normalization();
                let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                let size = (rgb.len() * 12) as u64;
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
                let mut entries: Vec<_> = [0, 12, 13, 14]
                    .map(|binding| wgpu::BindGroupEntry {
                        binding,
                        resource: source.buffer.as_entire_binding(),
                    })
                    .into();
                entries.extend([
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: parameters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ]);
                let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &entries,
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
                staging.map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap()
                });
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(submission),
                        timeout: None,
                    })
                    .unwrap();
                rx.recv().unwrap().unwrap();
                let mapping = staging.get_mapped_range(..).unwrap();
                let actual: &[f32] = bytemuck::cast_slice(&mapping);
                let expected = oracle::components(&rgb, &config);
                // BT.709 has a discontinuity at 0.081, whose nearest F32 is just above
                // the exact decimal. A F32 predicate can select either limiting branch.
                // Evaluate that declared boundary independently; the arithmetic bound stays fixed.
                let boundary_rgb: Vec<_> = rgb
                    .iter()
                    .map(|p| {
                        p.map(|v| {
                            if oracle::wire_spec(&config).transfer == TransferFunction::Bt709
                                && v == f64::from(0.081_f32)
                            {
                                0.081
                            } else {
                                v
                            }
                        })
                    })
                    .collect();
                let boundary = oracle::components(&boundary_rgb, &config);
                for (i, (&a, &b)) in actual.iter().zip(expected.as_flattened()).enumerate() {
                    let other = boundary.as_flattened()[i];
                    let low = b.min(other);
                    let high = b.max(other);
                    let error = 2e-4 * (1.0 + b.abs().max(other.abs()));
                    assert!(
                        a.is_finite()
                            && b.is_finite()
                            && f64::from(a) >= low - error
                            && f64::from(a) <= high + error,
                        "{channels:?}/{index}/{transform:?}/{i}: {a} vs [{low}, {high}]"
                    );
                }
                drop(mapping);
                staging.unmap();
            }
        }
    }
}

#[test]
fn enumerated_admission_rejects_unbound_color_before_budget_and_releases_cancelled_work() {
    let context = test_context().unwrap();
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        let mut config = oracle::config(channels, 32, VarDctColorTransform::Xyb);
        config.sample_format = ColorSampleFormat::float(channels, 32, 8).unwrap();
        let extent = Extent2d::new(25, 17);
        let words = input(extent, config.sample_format).0;
        let source = upload(&context, extent, &config, &words, true);
        for topology in [layouts::Topology::Map, layouts::Topology::Tiled] {
            let encoder = topology.backend(&context, extent, &config);
            let bytes = encoder.memory_plan(&source).unwrap().owned_bytes_per_job;
            for limit in [bytes - 1, bytes] {
                let bounded = WgpuContext::with_memory_budget(
                    Arc::new(context.device().clone()),
                    Arc::new(context.queue().clone()),
                    NonZeroU64::new(limit).unwrap(),
                )
                .unwrap();
                let encoder = topology.backend(&bounded, extent, &config);
                let request = layouts::request(extent, &config);
                for color in [
                    ColorSpecification::Default,
                    ColorSpecification::Undefined,
                    oracle::spec(ColorSpace::Bt709, TransferFunction::Srgb),
                    oracle::spec(ColorSpace::DisplayP3, TransferFunction::Undefined),
                ] {
                    let mut wrong = source.clone();
                    wrong.layout.format.color_spec = color;
                    assert!(matches!(
                        encoder.submit(&bounded, GpuFrameSource::Buffer(wrong), &request),
                        Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
                    ));
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                }
                let job =
                    encoder.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request);
                if limit < bytes {
                    assert!(matches!(job, Err(EncodeError::MemoryBackpressure(_))));
                    continue;
                }
                drop(job.unwrap());
                bounded
                    .device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while bounded.memory_stats().reserved_bytes != 0
                    && std::time::Instant::now() < until
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
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                for word in [
                    f32::NAN.to_bits(),
                    f32::INFINITY.to_bits(),
                    f32::NEG_INFINITY.to_bits(),
                ] {
                    let mut bad = words.clone();
                    bad[17] = word;
                    let source = upload(&bounded, extent, &config, &bad, true);
                    assert!(matches!(
                        encoder
                            .submit(&bounded, GpuFrameSource::Buffer(source), &request)
                            .unwrap()
                            .wait(),
                        Err(EncodeError::Backend(
                            crate::BackendError::VarDctNonFiniteSource
                        ))
                    ));
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                }
            }
        }
    }
}

#[test]
fn enumerated_plan_checks_metadata_and_accepts_only_equivalent_wire_aliases() {
    for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
        let config = VarDctConfig {
            color_transform: transform,
            ..Default::default()
        };
        let plan = VarDctColorPlan::new(&config).unwrap();
        for transfer in [TransferFunction::Srgb, TransferFunction::Sycc] {
            let format = VarDctConfig {
                source_color: oracle::spec(ColorSpace::Bt709, transfer),
                ..config.clone()
            }
            .pixel_format();
            assert!(plan.matches_format(&format));
        }
        for transfer in [
            TransferFunction::Undefined,
            TransferFunction::Smpte240M,
            TransferFunction::Bt2020,
        ] {
            assert!(
                VarDctColorPlan::new(&VarDctConfig {
                    source_color: oracle::spec(ColorSpace::Bt709, transfer),
                    ..config.clone()
                })
                .is_err()
            );
        }
        for color in [
            ColorSpecification::Undefined,
            oracle::spec(ColorSpace::Undefined, TransferFunction::Srgb),
        ] {
            assert!(
                VarDctColorPlan::new(&VarDctConfig {
                    source_color: color,
                    ..config.clone()
                })
                .is_err()
            );
        }
        let ColorSpecification::Defined(spec) =
            oracle::spec(ColorSpace::Bt709, TransferFunction::Srgb)
        else {
            unreachable!()
        };
        for spec in [
            ColorSpec {
                range: ColorRange::Limited,
                ..spec
            },
            ColorSpec {
                encoding: YcbcrEncoding::Bt709,
                ..spec
            },
        ] {
            assert!(
                VarDctColorPlan::new(&VarDctConfig {
                    source_color: ColorSpecification::Defined(spec),
                    ..config.clone()
                })
                .is_err()
            );
        }
        for bits in [0, 0x8000, 0xbc00] {
            assert!(
                VarDctColorPlan::new(&VarDctConfig {
                    color_options: crate::ImageColorOptions {
                        intensity_target: FiniteF16::from_bits(bits).unwrap(),
                        ..Default::default()
                    },
                    ..config.clone()
                })
                .is_err()
            );
        }
        for bits in [1, 0x03ff, 0x7bff] {
            let value = FiniteF16::from_bits(bits).unwrap();
            let plan = VarDctColorPlan::new(&VarDctConfig {
                color_options: crate::ImageColorOptions {
                    intensity_target: value,
                    ..Default::default()
                },
                ..config.clone()
            })
            .unwrap();
            assert_eq!(plan.gpu.intensity, value.to_f32());
        }
    }
    // Wire-equivalent custom declarations must not be treated as different frame colors.
    let config = oracle::config(ColorChannels::Rgb, 48, VarDctColorTransform::Xyb);
    let plan = VarDctColorPlan::new(&config).unwrap();
    let equivalent = VarDctConfig {
        source_color: ColorSpecification::Defined(oracle::wire_spec(&config)),
        ..config
    };
    assert!(plan.matches_format(&equivalent.pixel_format()));
}
