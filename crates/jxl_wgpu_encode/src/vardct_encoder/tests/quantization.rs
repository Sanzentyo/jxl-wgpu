use super::*;
use crate::{
    BackendError, EncodeProfile, FramePacketSet, GroupPacket, GroupPacketKind, ProfileCapability,
    VarDctHfMultiplier, VarDctStrategyMap, VarDctTransform,
};
use jxl_gpu_bitstream::ContainerStreamScanner;
use jxl_wgpu_decode::WgpuDecodeEngine;

fn quantization(global: u32, lf: u32, hf: u32) -> VarDctQuantization {
    VarDctQuantization::new(global, lf, VarDctHfMultiplier::new(hf).unwrap()).unwrap()
}

fn packet(bytes: &[u8]) -> BoundedVarDctPacketPlan {
    let inventory = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    BoundedVarDctPacketPlan::parse(bytes, &inventory).unwrap()
}

pub(super) fn assert_decoders_agree(bytes: &[u8], width: usize, height: usize) -> Vec<u8> {
    assert_decoders_agree_with_reference(
        bytes,
        width,
        height,
        &decode_rgb8_sized(bytes, width, height),
    )
}

pub(super) fn assert_decoders_agree_with_reference(
    bytes: &[u8],
    width: usize,
    height: usize,
    reference: &[u8],
) -> Vec<u8> {
    let (device, queue, info) =
        test_device().expect("actual GPU required for quantizer validation");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    )
    .unwrap();
    let readback = ImageReadbackPipeline::new(&backend);
    let mut whole = None;
    for cap in [u64::MAX, 40] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
        let mut session = if cap == u64::MAX {
            decoder.open(bytes, request).unwrap()
        } else {
            let mut stream = decoder.stream(request).unwrap();
            let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
            for chunk in bytes.chunks(7) {
                for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
            }
            for event in transport.finish_input().unwrap() {
                stream.push_transport_event(&event).unwrap();
            }
            assert!(stream.stats().retained_spans > 2);
            stream.finish().unwrap()
        };
        let frame = session.next_frame().unwrap().unwrap();
        let output = readback.submit(frame.output()).unwrap().wait().unwrap();
        let gpu = output.frame.outputs[0].bytes.clone();
        if let Some(whole) = &whole {
            assert_eq!(&gpu, whole, "fragmented input through 40-byte GPU windows");
        } else {
            whole = Some(gpu);
        }
        drop(output);
        drop(frame);
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
    let reference = reference.to_vec();
    let gpu = whole.unwrap();
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let input = directory.join("quantization.jxl");
    let output = directory.join("quantization.ppm");
    fs::write(&input, bytes).unwrap();
    assert!(
        Command::new("djxl")
            .arg(&input)
            .arg(&output)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap()
            .success()
    );
    let native = read_ppm_rgb8(&output, width, height);
    eprintln!(
        "endpoint {:?}: GPU/reference {}, native/reference {}, GPU/native {}, files {}",
        (packet(bytes).global_scale, packet(bytes).quant_lf),
        max_abs_error(&gpu, &reference),
        max_abs_error(&native, &reference),
        max_abs_error(&gpu, &native),
        input.display()
    );
    assert!(max_abs_error(&native, &reference) <= 1);
    assert!(max_abs_error(&gpu, &reference) <= 1);

    fs::remove_dir_all(directory).unwrap();
    reference
}

#[test]
fn signed_hf_metadata_endpoints_interoperate_through_whole_and_fragmented_packets() {
    let code = fixed_prefix_code().unwrap();
    let hf = HfEntropyPlan::single_cluster_prefix().unwrap();
    let dc = [332, 153, -6];
    let fixture = cpu_test_artifact(dc, &code);
    let packets = build_frame_packet(
        fixture.artifact(),
        &code,
        &hf,
        VarDctFrameLayout::single(VarDctStrategy::Dct8),
        &VarDctConfig::default(),
        &still_control(8, 8),
        VarDctColorPlan::new(VarDctColorTransform::Xyb),
    )
    .unwrap();
    let header = packets.frame_header.clone();
    let layout = packets.layout;
    let image = image_header(8, 8, crate::AnimationHeader::Still).unwrap();
    let mut base = image.bytes().to_vec();
    base.extend_from_slice(assemble_frame(packets).unwrap().bytes());
    let plan = packet(&base);
    let expected = decode_rgb8(&base);
    for raw in [i32::MIN, -1, 0, 1, 254, 255, 256, 65_535, i32::MAX] {
        let mut group = BitWriter::new();
        // Preserve the independently parsed LF-global descriptor, then author a
        // one-block LF/HF packet with the full signed raw metadata domain.
        for bit in plan.lf_global.offset..u64::from(plan.entropy_bit_offset) {
            group
                .write_bits(u64::from((base[bit as usize / 8] >> (bit % 8)) & 1), 1)
                .unwrap();
        }
        let sample = |writer: &mut BitWriter, value: i32| {
            let value = i64::from(value);
            let packed = if value >= 0 {
                value * 2
            } else {
                -value * 2 - 1
            } as u32;
            let extra = 31u32.saturating_sub(packed.leading_zeros());
            code.write_raw(
                writer,
                u32::from(packed != 0) + extra,
                extra,
                packed.saturating_sub(1 << extra),
            )
            .unwrap();
        };
        group.write_bits(0, 2).unwrap(); // no extra LF precision
        group.write_bits(3, 4).unwrap(); // global Gradient tree, no local transforms
        for value in dc {
            sample(&mut group, value);
        }
        // One first block needs zero count bits. The second Modular image has
        // zero CfL maps, DCT8 strategy, the raw HF sample, and zero sharpness.
        group.write_bits(3, 4).unwrap();
        for value in [0, 0, 0, raw, 0] {
            sample(&mut group, value);
        }
        hf.write_global(
            &mut group,
            1,
            false,
            &Default::default(),
            Default::default(),
        )
        .unwrap();
        group.align_to_byte().unwrap();
        let packets = FramePacketSet::new(
            header.clone(),
            layout,
            [GroupPacket::new(
                GroupPacketKind::Single,
                group.into_bytes(),
            )],
        )
        .unwrap();
        let mut bytes = image.bytes().to_vec();
        bytes.extend_from_slice(assemble_frame(packets).unwrap().bytes());
        eprintln!("signed raw HF metadata: {raw}");
        assert_eq!(assert_decoders_agree(&bytes, 8, 8), expected);
    }
}

#[test]
fn quantizer_controls_cover_syntax_ranges_and_negotiate_exactly() {
    for global in [1, 2048, 2049, 4096, 4097, 8192, 8193, 73728] {
        for lf in [1, 16, 32, 256, 65536] {
            for hf in [1, 2, 255, 256] {
                let value = quantization(global, lf, hf);
                assert_eq!(
                    (
                        value.global_scale(),
                        value.quant_lf(),
                        value.hf_multiplier().get()
                    ),
                    (global, lf, hf)
                );
            }
        }
    }
    for value in [0, 73729, u32::MAX] {
        assert!(VarDctQuantization::new(value, 1, VarDctHfMultiplier::default()).is_err());
    }
    for value in [0, 65537, u32::MAX] {
        assert!(VarDctQuantization::new(1, value, VarDctHfMultiplier::default()).is_err());
    }
    for value in [0, 257, i32::MAX as u32, u32::MAX] {
        assert!(VarDctHfMultiplier::new(value).is_err());
    }
    let value = quantization(12345, 17, 255);
    let capability = ProfileCapability::VarDct {
        quantization: value,
    };
    assert!(capability.supports(EncodeProfile::VarDct {
        quantization: value
    }));
    for other in [
        quantization(12346, 17, 255),
        quantization(12345, 18, 255),
        quantization(12345, 17, 256),
    ] {
        assert!(!capability.supports(EncodeProfile::VarDct {
            quantization: other
        }));
    }
}

#[test]
fn maximal_global_lf_product_and_full_hf_metadata_preserve_solid_white() {
    let context = test_context().expect("actual GPU required for quantization endpoints");
    for value in [
        quantization(1, 1, 1),
        quantization(8193, 257, 128),
        quantization(73728, 65536, 256),
    ] {
        let config = VarDctConfig {
            quantization: value,
            ..Default::default()
        };
        let pixels = vec![[255; 3]; 13 * 21];
        let map = VarDctStrategyMap::new(
            13,
            21,
            vec![
                VarDctTransform::new(0, 0, VarDctStrategy::Dct16x16),
                VarDctTransform::new(0, 2, VarDctStrategy::Dct8x16)
                    .with_hf_multiplier(VarDctHfMultiplier::new(1).unwrap()),
            ],
        )
        .unwrap();
        let encoder =
            VarDctEncoder::new_with_strategy_map(context.clone(), map, config.clone()).unwrap();
        let tiled = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
        for bytes in [
            encoder
                .encode(padded_rgb_source_sized(&context, 13, 21, &pixels))
                .unwrap(),
            tiled
                .encode(padded_rgb_source_sized(&context, 13, 21, &pixels))
                .unwrap(),
        ] {
            let plan = packet(&bytes);
            assert_eq!(
                (plan.global_scale, plan.quant_lf),
                (value.global_scale(), value.quant_lf())
            );
            let decoded = assert_decoders_agree(&bytes, 13, 21);
            if value.quant_lf() > 1 {
                assert!(
                    max_abs_error(&decoded, &vec![255; 13 * 21 * 3]) <= 1,
                    "the LF scale product must not wrap in u32"
                );
            }
        }
    }
}

#[test]
fn gpu_lf_quantization_overflow_is_reported_and_releases_memory() {
    let context = test_context().expect("actual GPU required for overflow validation");
    let metadata =
        VarDctLfMetadata::new([f16(0x0016); 3], 84, [f16(0), f16(0x3c00)], [0, 0]).unwrap();
    for hf_multiplier in [1, 6, 256] {
        let config = VarDctConfig {
            quantization: quantization(73728, 65536, hf_multiplier),
            lf_metadata: metadata,
            ..Default::default()
        };
        let pixels = reference::pattern(8, 8);
        let single =
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
                .unwrap();
        let tiled = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
        for result in [
            single.encode(padded_rgb_source_sized(&context, 8, 8, &pixels)),
            tiled.encode(padded_rgb_source_sized(&context, 8, 8, &pixels)),
        ] {
            assert!(
                matches!(result, Err(EncodeError::Backend(BackendError::VarDctQuantizationOverflow {
                low_frequency, high_frequency
            })) if low_frequency && !high_frequency),
                "unexpected overflow result: {result:?}"
            );
        }
        assert_eq!(single.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(tiled.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn gpu_quantizer_preserves_i32_endpoints_and_flags_out_of_range_values() {
    let context = test_context().expect("actual GPU required for integer conversion boundaries");
    let device = context.device();
    let values = [
        -2147483648.0f32,
        2147483520.0,
        -2147483904.0,
        2147483648.0,
        0.0,
        17.25,
        -17.75,
    ];
    let source = shader_source(
        r"
        @group(0) @binding(9) var<storage, read> boundary_values: array<f32>;
        @compute @workgroup_size(64)
        fn check_boundaries(@builtin(local_invocation_index) index: u32) {
            let count = arrayLength(&boundary_values);
            if index < count {
                let error = select(LF_QUANTIZATION_OVERFLOW, HF_QUANTIZATION_OVERFLOW, index % 2u == 1u);
                artifact_words[index] = bitcast<u32>(quantize_checked(boundary_values[index], error));
            }
            workgroupBarrier();
            if index == 0u { artifact_words[count] = atomicLoad(&quantization_error); }
        }
    ",
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("VarDCT quantizer integer boundaries"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("VarDCT quantizer integer boundaries"),
        layout: None,
        module: &shader,
        entry_point: Some("check_boundaries"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quantizer f32 endpoint samples"),
        contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantizer boundary artifact"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quantizer boundary readback"),
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("quantizer boundary bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: input.as_entire_binding(),
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
    commands.copy_buffer_to_buffer(&output, 0, &staging, 0, 32);
    let submission = context.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap()
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let bytes = staging.get_mapped_range(..).unwrap();
    let words: &[i32] = bytemuck::cast_slice(&bytes);
    assert_eq!(&words[..7], &[i32::MIN, 2147483520, 0, 0, 0, 17, -18]);
    assert_eq!(words[7] as u32, 0xc000_0000);
    drop(bytes);
    staging.unmap();
}

#[test]
fn cropped_transform_sources_work_with_stricter_storage_offset_alignment() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let mut baseline = Vec::new();
    for alignment in [256, 1024] {
        let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
        limits.min_storage_buffer_offset_alignment = alignment;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .unwrap();
        let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
        let mut encoded = Vec::new();
        for (width, height, transforms) in [
            (8, 8, vec![VarDctTransform::new(0, 0, VarDctStrategy::Dct8)]),
            (
                13,
                21,
                vec![
                    VarDctTransform::new(0, 0, VarDctStrategy::Dct16x16),
                    VarDctTransform::new(0, 2, VarDctStrategy::Dct8x16),
                ],
            ),
        ] {
            let map = VarDctStrategyMap::new(width, height, transforms).unwrap();
            let encoder =
                VarDctEncoder::new_with_strategy_map(context.clone(), map, VarDctConfig::default())
                    .unwrap();
            let pixels = reference::pattern(width as usize, height as usize);
            encoded.push(
                encoder
                    .encode(padded_rgb_source_sized(
                        &context,
                        width as usize,
                        height as usize,
                        &pixels,
                    ))
                    .unwrap(),
            );
        }
        if alignment == 256 {
            baseline = encoded;
        } else {
            assert_eq!(encoded, baseline);
        }
    }
}
