use super::*;
use crate::VarDctCoefficientOrders;

const REPRESENTATIVES: [VarDctStrategy; 13] = [
    VarDctStrategy::Dct8,
    VarDctStrategy::Hornuss,
    VarDctStrategy::Dct16x16,
    VarDctStrategy::Dct32x32,
    VarDctStrategy::Dct16x8,
    VarDctStrategy::Dct32x8,
    VarDctStrategy::Dct32x16,
    VarDctStrategy::Dct64x64,
    VarDctStrategy::Dct64x32,
    VarDctStrategy::Dct128x128,
    VarDctStrategy::Dct128x64,
    VarDctStrategy::Dct256x256,
    VarDctStrategy::Dct256x128,
];

fn permutations(strategy: VarDctStrategy) -> [Vec<u32>; 3] {
    let extent = strategy.pixel_extent();
    let area = extent.width * extent.height;
    std::array::from_fn(|channel| {
        let mut order = (0..area).collect::<Vec<_>>();
        let tail = &mut order[(area / 64) as usize..];
        match channel {
            0 => tail.reverse(),
            1 => tail.rotate_left(3),
            _ => {
                // A deterministic shuffle, independent of the production Lehmer encoder.
                let mut state = 0x3141_5926u32;
                for index in (1..tail.len()).rev() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    tail.swap(index, state as usize % (index + 1));
                }
            }
        }
        order
    })
}

pub(super) fn selected(
    strategies: impl IntoIterator<Item = VarDctStrategy>,
) -> VarDctCoefficientOrders {
    strategies
        .into_iter()
        .fold(VarDctCoefficientOrders::default(), |orders, strategy| {
            orders.with_order(strategy, permutations(strategy)).unwrap()
        })
}

#[test]
fn all_custom_order_families_round_trip_through_independent_entropy_decoder() {
    for mask in [0u32, 1, 0x13, 0x5f, 0x1fff] {
        let orders = selected(
            REPRESENTATIVES
                .into_iter()
                .enumerate()
                .filter_map(|(id, strategy)| (mask & (1 << id) != 0).then_some(strategy)),
        );
        let mut output = BitWriter::new();
        orders.write(&mut output).unwrap();
        let bit_len = output.bit_len();
        output.align_to_byte().unwrap();
        let bytes = output.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&bytes);
        use jxl_bitstream::U;
        assert_eq!(bits.read_u32(0x5f, 0x13, 0, U(13)).unwrap(), mask);
        if mask != 0 {
            let mut decoder = jxl_coding::Decoder::parse(&mut bits, 8).unwrap();
            decoder.begin(&mut bits).unwrap();
            for (id, strategy) in REPRESENTATIVES.into_iter().enumerate() {
                if mask & (1 << id) == 0 {
                    continue;
                }
                for expected in permutations(strategy) {
                    let len = expected.len() as u32;
                    let actual =
                        jxl_coding::read_permutation(&mut bits, &mut decoder, len, len / 64)
                            .unwrap();
                    assert_eq!(
                        actual,
                        expected
                            .into_iter()
                            .map(|rank| rank as usize)
                            .collect::<Vec<_>>()
                    );
                }
            }
            decoder.finalize().unwrap();
        }
        assert_eq!(bits.num_read_bits(), bit_len);
    }
}

#[test]
fn custom_orders_reject_invalid_permutations_and_preserve_natural_lf_prefix() {
    for strategy in VarDctStrategy::ALL {
        let extent = strategy.pixel_extent();
        let len = (extent.width * extent.height) as usize;
        for channel in 0..3 {
            for invalid in 0..5 {
                let mut orders = permutations(strategy);
                match invalid {
                    0 => {
                        orders[channel].pop();
                    }
                    1 => orders[channel].push(0),
                    2 => orders[channel][len - 1] = orders[channel][len - 2],
                    3 => orders[channel][len - 1] = u32::MAX,
                    _ => orders[channel].swap(0, len / 64),
                }
                assert!(
                    matches!(VarDctCoefficientOrders::default().with_order(strategy, orders),
                    Err(EncodeError::VarDctCoefficientOrder { channel: failed, .. }) if failed == channel as u8)
                );
            }
        }
        let identity = std::array::from_fn(|_| (0..len as u32).collect());
        let restored = selected([strategy]).with_order(strategy, identity).unwrap();
        assert_eq!(restored, VarDctCoefficientOrders::default());
        let mut bits = BitWriter::new();
        restored.write(&mut bits).unwrap();
        assert_eq!(bits.bit_len(), 2);
    }
    let orders = selected([VarDctStrategy::Dct16x8, VarDctStrategy::Afv0]);
    assert_eq!(
        orders.permutations(VarDctStrategy::Dct16x8),
        orders.permutations(VarDctStrategy::Dct8x16)
    );
    assert_eq!(
        orders.permutations(VarDctStrategy::Afv0),
        orders.permutations(VarDctStrategy::Hornuss)
    );
    assert!(orders.permutations(VarDctStrategy::Dct8).is_none());
}

#[test]
fn tiled_custom_orders_preserve_pixels_across_edges_groups_windows_and_variants() {
    let (device, queue, info) =
        test_device().expect("actual GPU required for custom coefficient orders");
    let context = WgpuContext::new(device.clone(), queue.clone()).unwrap();
    let config = VarDctConfig {
        coefficient_orders: selected([VarDctStrategy::Dct8]),
        lf_metadata: custom_lf_metadata(),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let baseline =
        TiledVarDctEncoder::new_with_config(context.clone(), config_with_lf(config.lf_metadata))
            .unwrap();
    let mut cases = Vec::new();
    for (width, height) in [(1, 1), (13, 21), (257, 17), (2057, 17)] {
        let pixels = reference::pattern(width, height);
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let expected = baseline.encode(source.clone()).unwrap();
        let actual = encoder.encode(source.clone()).unwrap();
        assert_ne!(actual, expected);
        let decoded = quantization::assert_decoders_agree(&actual, width, height);
        assert_eq!(
            decoded,
            decode_rgb8_sized(&expected, width, height),
            "orders cannot change quantized pixels"
        );
        assert_eq!(
            pollster::block_on(encoder.submit(source).unwrap()).unwrap(),
            actual
        );
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        cases.push((width, height, pixels, actual));
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
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        for (width, height, pixels, expected) in &cases {
            let source = padded_rgb_source_sized(&context, *width, *height, pixels);
            assert_eq!(
                &encoder.encode(source).unwrap(),
                expected,
                "{variant:?}/{width}x{height}"
            );
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn custom_tiled_order_storage_survives_cancellation_and_obeys_exact_admission() {
    let context = test_context().expect("actual GPU required for custom order lifetime");
    for (case, dequant_matrices) in [
        Default::default(),
        matrices::selected(),
        raw_matrices::selected(VarDctStrategy::ALL),
        raw_matrices::selected(VarDctStrategy::ALL),
        raw_matrices::selected(VarDctStrategy::ALL),
    ]
    .into_iter()
    .enumerate()
    {
        let config = VarDctConfig {
            group_order: if case == 3 {
                crate::VarDctGroupOrder::centered_at(256, 0)
            } else if case == 4 {
                crate::VarDctGroupOrder::saliency_first()
            } else {
                Default::default()
            },
            progressive: if case >= 3 {
                progressive::maximum()
            } else {
                Default::default()
            },
            dequant_matrices,
            coefficient_orders: selected([VarDctStrategy::Dct8]),
            ..Default::default()
        };
        let source = padded_rgb_source_sized(&context, 257, 17, &reference::pattern(257, 17));
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        let memory = encoder.memory_plan(&source).unwrap();
        assert_eq!(
            memory.saliency_metadata_bytes,
            if case == 4 { 256 } else { 0 }
        );
        if case == 4 {
            let raster = TiledVarDctEncoder::new_with_config(
                context.clone(),
                VarDctConfig {
                    group_order: Default::default(),
                    ..config.clone()
                },
            )
            .unwrap()
            .memory_plan(&source)
            .unwrap();
            assert_eq!(
                memory.artifact_storage_bytes,
                raster.artifact_storage_bytes + 256
            );
            assert_eq!(memory.readback_bytes, raster.readback_bytes + 256);
            assert_eq!(memory.owned_bytes_per_job, raster.owned_bytes_per_job + 512);
        }
        assert_eq!(memory.quantization_metadata_bytes, 1536);
        assert_eq!(
            memory.owned_bytes_per_job,
            memory.parameter_storage_bytes
                + memory.artifact_storage_bytes
                + memory.readback_bytes
                + memory.raw_matrix_input_bytes
                + memory.raw_matrix_artifact_bytes
                + 1536
        );
        for deficit in [1, 0] {
            let limited = WgpuContext::with_memory_budget(
                Arc::new(context.device().clone()),
                Arc::new(context.queue().clone()),
                NonZeroU64::new(memory.owned_bytes_per_job - deficit).unwrap(),
            )
            .unwrap();
            let encoder =
                TiledVarDctEncoder::new_with_config(limited.clone(), config.clone()).unwrap();
            if deficit != 0 {
                assert!(matches!(
                    encoder.submit(source.clone()),
                    Err(EncodeError::MemoryBackpressure(_))
                ));
            } else {
                let job = encoder.submit(source.clone()).unwrap();
                assert_eq!(
                    encoder.in_flight_memory_stats().reserved_bytes,
                    memory.owned_bytes_per_job
                );
                drop(job);
                let fence = limited.queue().submit([]);
                limited
                    .device()
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(fence),
                        timeout: None,
                    })
                    .unwrap();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                while encoder.in_flight_memory_stats().reserved_bytes != 0
                    && std::time::Instant::now() < deadline
                {
                    limited.device().poll(wgpu::PollType::Poll).unwrap();
                    std::thread::yield_now();
                }
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                let bytes = encoder.encode(source.clone()).unwrap();
                assert_eq!(decode_rgb8_sized(&bytes, 257, 17).len(), 257 * 17 * 3);
            }
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
