use super::*;

#[test]
fn progressive_updates_match_native_and_bounded_input_under_all_variants() {
    use jxl_gpu_bitstream::ContainerStreamScanner;
    use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction};
    use jxl_test_support::oracles::progressive::{native_updates, scalar_linear_updates};
    use jxl_wgpu_decode::WgpuDecodeEngine;
    let (device, queue, info) = test_device().expect("actual GPU required for progressive images");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info.clone(),
        WgpuBackendConfig::default(),
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let readback = ImageReadbackPipeline::new(&backend);
    let mut streams = Vec::new();
    for (width, height, progressive, group_order, cap) in [
        (
            17,
            1,
            plan(&[(1, 3), (8, 0)]),
            VarDctGroupOrder::default(),
            40,
        ),
        (
            13,
            21,
            plan(&[(2, 0), (4, 0), (8, 0)]),
            VarDctGroupOrder::default(),
            40,
        ),
        (
            257,
            17,
            plan(&[(8, 3), (8, 1), (8, 0)]),
            VarDctGroupOrder::default(),
            256,
        ),
        (2057, 17, combined(), VarDctGroupOrder::default(), 256),
        (1, 1, maximum(), VarDctGroupOrder::default(), 40),
        (
            513,
            257,
            plan(&[(2, 0), (4, 0), (8, 0)])
                .with_downsampling(delivery::endpoints(&[(4, 0), (2, 1)]))
                .unwrap(),
            VarDctGroupOrder::center_first(),
            1024,
        ),
        (
            2057,
            1,
            combined()
                .with_downsampling(delivery::endpoints(&[(8, 0), (4, 1), (2, 2), (1, 3)]))
                .unwrap(),
            VarDctGroupOrder::explicit(vec![8, 0, 7, 1, 6, 2, 5, 3, 4]).unwrap(),
            256,
        ),
    ] {
        let config = VarDctConfig {
            progressive,
            group_order,
            coefficient_orders: orders::selected([VarDctStrategy::Dct8]),
            dequant_matrices: matrices::selected()
                .with_raw_matrix(
                    VarDctStrategy::Dct8,
                    f16(0x0400),
                    raw_matrices::samples(VarDctStrategy::Dct8),
                )
                .unwrap(),
            lf_metadata: custom_lf_metadata(),
            ..Default::default()
        };
        let pixels = reference::pattern(width, height);
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        let grid = encoder.grid(&source).unwrap();
        assert_eq!(usize::from(grid.passes), config.progressive.passes().len());
        let bytes = encoder.encode(source.clone()).unwrap();
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert_eq!(
            inventory.frames[0].sections.len(),
            grid.toc_entries().unwrap() as usize
        );
        assert_eq!(
            pollster::block_on(encoder.submit(source).unwrap()).unwrap(),
            bytes
        );
        let native = scalar_linear_updates(&bytes);
        let simd =
            native_updates(&bytes, true).expect("native SIMD progressive oracle is required");
        assert_eq!(native.len(), simd.len());
        assert_eq!(native.len(), config.progressive.passes().len() + 1);
        delivery::check_file_order_and_native_prefixes(
            &bytes,
            &config,
            &inventory.frames[0],
            &native,
        );
        // A third independent decoder checks the complete linear image, including raw matrices.
        let mut oxide = jxl_oxide::JxlImage::read_with_defaults(bytes.as_slice()).unwrap();
        oxide.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb_linear(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let oxide = oxide.render_frame(0).unwrap().image_all_channels();
        assert_eq!(oxide.buf().len(), width * height * 3);
        let mut format = PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            vardct_rgb8_format().color_spec,
        );
        let ColorSpecification::Defined(ref mut color) = format.color_spec else {
            unreachable!()
        };
        color.transfer = TransferFunction::Linear;
        let mut whole = None;
        for window in [u64::MAX, cap] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(window).unwrap()),
            );
            let request = GpuOutputRequest::color(format.clone())
                .unwrap()
                .with_progressive_output(true);
            let mut session = if window == u64::MAX {
                decoder.open(&bytes, request).unwrap()
            } else {
                let mut scanner = ContainerStreamScanner::new(decoder.container_stream_limits());
                let mut stream = decoder.stream(request).unwrap();
                for chunk in bytes.chunks(7) {
                    for event in scanner.push_chunk(Arc::from(chunk)).unwrap() {
                        stream.push_transport_event(&event).unwrap();
                    }
                }
                for event in scanner.finish_input().unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
                stream.finish().unwrap()
            };
            let mut actual = Vec::new();
            let mut held = Vec::new();
            while let Some(frame) = session.next_update().unwrap() {
                let stage = actual.len();
                let expected = &native[stage];
                let simd = &simd[stage];
                assert_eq!(expected.complete, simd.complete);
                assert_eq!(expected.step, simd.step);
                assert_eq!(expected.ratio, simd.ratio);
                assert_eq!(
                    expected.ratio,
                    delivery::expected_ratio(&config.progressive, stage)
                );
                assert_eq!(frame.is_complete(), expected.complete);
                assert_eq!(expected.step, stage);
                if let Some(progress) = frame.progression() {
                    assert_eq!(progress.intended_downsampling(), expected.ratio);
                }
                let pixels = readback
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes
                    .clone();
                assert_eq!(pixels.len(), expected.pixels.len());
                assert_eq!(pixels.len(), simd.pixels.len());
                let bound = if stage == 0 {
                    1e-5
                } else if frame.is_complete() {
                    1e-4
                } else {
                    2e-4
                };
                let mut peak = 0.0f32;
                for ((gpu, native), simd) in pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(expected.pixels.as_chunks::<4>().0)
                    .zip(simd.pixels.as_chunks::<4>().0)
                {
                    let (gpu, native) = (f32::from_le_bytes(*gpu), f32::from_le_bytes(*native));
                    assert!(gpu.is_finite() && native.is_finite());
                    peak = peak.max((gpu - native).abs());
                    let code = |linear: f32| {
                        let value = linear.clamp(0.0, 1.0);
                        let srgb = if value <= 0.0031308 {
                            value * 12.92
                        } else {
                            1.055 * value.powf(1.0 / 2.4) - 0.055
                        };
                        (srgb * 255.0).round() as u8
                    };
                    assert!(code(gpu).abs_diff(code(native)) <= 1);
                    let simd = f32::from_le_bytes(*simd);
                    assert!(simd.is_finite());
                    assert!(code(gpu).abs_diff(code(simd)) <= 1);
                }
                assert!(
                    peak < bound,
                    "{width}x{height}, window {window}, stage {stage}: {peak} >= {bound}"
                );
                if frame.is_complete() {
                    for (pixel, reference) in pixels
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(oxide.buf().as_chunks::<3>().0)
                    {
                        for (gpu, reference) in pixel.as_chunks::<4>().0.iter().zip(reference) {
                            assert!((f32::from_le_bytes(*gpu) - reference).abs() < 1e-4);
                        }
                    }
                }
                held.push(jxl_wgpu::GpuImageFrame {
                    token: frame.output().token,
                    outputs: frame
                        .output()
                        .outputs
                        .iter()
                        .map(|output| jxl_wgpu::GpuImageOutput {
                            id: output.id,
                            layout: output.layout.clone(),
                            buffer: output.buffer.clone(),
                        })
                        .collect(),
                    changed: frame.output().changed.clone(),
                });
                actual.push(pixels);
            }
            assert_eq!(actual.len(), native.len());
            drop(session);
            for (frame, original) in held.iter().zip(&actual) {
                assert_eq!(
                    &readback
                        .submit(frame)
                        .unwrap()
                        .wait()
                        .unwrap()
                        .frame
                        .outputs[0]
                        .bytes,
                    original
                );
            }
            let mut final_only = decoder
                .open(&bytes, GpuOutputRequest::color(format.clone()).unwrap())
                .unwrap();
            let frame = final_only.next_frame().unwrap().unwrap();
            assert_eq!(
                &readback
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap()
                    .frame
                    .outputs[0]
                    .bytes,
                actual.last().unwrap()
            );
            drop(frame);
            assert!(final_only.next_frame().unwrap().is_none());
            drop(final_only);
            drop(held);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
            if let Some(whole) = &whole {
                assert_eq!(&actual, whole);
            } else {
                whole = Some(actual);
            }
        }
        streams.push((width, height, pixels, config, bytes));
    }
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context =
            test_context_with_variants(&device, &queue, &info, &[(TILED_KERNEL_KEY, variant)])
                .unwrap();
        for (width, height, pixels, config, expected) in &streams {
            let encoder =
                TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
            assert_eq!(
                &encoder
                    .encode(padded_rgb_source_sized(&context, *width, *height, pixels))
                    .unwrap(),
                expected
            );
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn progressive_mixed_maps_preserve_all_strategies_and_lf_boundaries() {
    let (device, queue, info) = test_device().expect("actual GPU required for progressive maps");
    let context = WgpuContext::new(device.clone(), queue.clone()).unwrap();
    let mut streams = Vec::new();
    for map in [
        mixed::packed_map(512, 512, true),
        mixed::packed_map(2057, 17, false),
        mixed::packed_map(13, 21, false),
    ] {
        let extent = map.extent();
        let (width, height) = (extent.width as usize, extent.height as usize);
        let config = VarDctConfig {
            coefficient_orders: orders::selected(VarDctStrategy::ALL),
            dequant_matrices: raw_matrices::selected(VarDctStrategy::ALL),
            lf_metadata: custom_lf_metadata(),
            ..Default::default()
        };
        let pixels = reference::pattern(width, height);
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let baseline =
            VarDctEncoder::new_with_strategy_map(context.clone(), map.clone(), config.clone())
                .unwrap()
                .encode(source.clone())
                .unwrap();
        let config = VarDctConfig {
            progressive: combined()
                .with_downsampling(delivery::endpoints(&[(4, 1), (2, 2)]))
                .unwrap(),
            group_order: VarDctGroupOrder::explicit(
                (0..(width.div_ceil(256) * height.div_ceil(256)) as u32)
                    .rev()
                    .collect(),
            )
            .unwrap(),
            ..config
        };
        let encoder =
            VarDctEncoder::new_with_strategy_map(context.clone(), map.clone(), config.clone())
                .unwrap();
        let bytes = encoder.encode(source.clone()).unwrap();
        assert_eq!(
            pollster::block_on(encoder.submit(source).unwrap()).unwrap(),
            bytes
        );
        assert_eq!(
            decode_rgb8_sized(&bytes, width, height),
            decode_rgb8_sized(&baseline, width, height)
        );
        // The all-family raw matrices are large; whole-input native/GPU pixels are
        // checked here. Dedicated pass-image cases exercise bounded transport.
        let backend = WgpuBackend::from_device(
            device.as_ref().clone(),
            queue.as_ref().clone(),
            info.clone(),
            WgpuBackendConfig::default(),
        )
        .unwrap();
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let mut session = decoder
            .open(
                &bytes,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        let gpu = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap()
            .frame
            .outputs[0]
            .bytes
            .clone();
        let rust = decode_rgb8_sized(&bytes, width, height);
        assert!(max_abs_error(&gpu, &rust) <= 1);
        let directory = oracle_directory();
        fs::create_dir_all(&directory).unwrap();
        assert!(max_abs_error(&native_rgb8(&directory, &bytes, width, height), &rust) <= 1);
        fs::remove_dir_all(directory).unwrap();
        drop(frame);
        assert!(session.next_frame().unwrap().is_none());
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        streams.push((map, config, pixels, bytes));
    }
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context =
            test_context_with_variants(&device, &queue, &info, &[(FORWARD_KERNEL_KEY, variant)])
                .unwrap();
        for (map, config, pixels, expected) in &streams {
            let extent = map.extent();
            let encoder =
                VarDctEncoder::new_with_strategy_map(context.clone(), map.clone(), config.clone())
                    .unwrap();
            let source = padded_rgb_source_sized(
                &context,
                extent.width as usize,
                extent.height as usize,
                pixels,
            );
            assert_eq!(&encoder.encode(source).unwrap(), expected);
        }
    }
}
