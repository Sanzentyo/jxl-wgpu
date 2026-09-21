//! Mixed-transform AC oracles, group assembly, edge padding and admission.

use super::super::dispatch::VarDctBackend;
use super::super::strategy_map::{TransformPlan, VarDctStrategyMap, VarDctTransform};
use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuFrameSource, ProgressivePlan,
};

fn placement(x: u32, y: u32, strategy: VarDctStrategy) -> VarDctTransform {
    VarDctTransform::new(x, y, strategy)
}

fn packed_map(width: u32, height: u32, include_all: bool) -> VarDctStrategyMap {
    let (w, h) = (width.div_ceil(8), height.div_ceil(8));
    let mut occupied = vec![false; (w * h) as usize];
    let mut transforms = Vec::new();
    let mut place = |strategy: VarDctStrategy| {
        let extent = strategy.lf_extent();
        for y in 0..h {
            for x in 0..w {
                if extent.width > w - x
                    || extent.height > h - y
                    || extent.width > 32 - x % 32
                    || extent.height > 32 - y % 32
                {
                    continue;
                }
                if (y..y + extent.height)
                    .any(|row| (x..x + extent.width).any(|col| occupied[(row * w + col) as usize]))
                {
                    continue;
                }
                for row in y..y + extent.height {
                    for col in x..x + extent.width {
                        occupied[(row * w + col) as usize] = true;
                    }
                }
                transforms.push(placement(x, y, strategy));
                return true;
            }
        }
        false
    };
    let mut strategies = VarDctStrategy::ALL;
    strategies.sort_by_key(|strategy| std::cmp::Reverse(strategy.lf_extent().area().unwrap()));
    if include_all {
        for strategy in strategies {
            assert!(place(strategy), "{strategy:?}");
        }
    }
    for strategy in strategies {
        while place(strategy) {}
    }
    VarDctStrategyMap::new(width, height, transforms).unwrap()
}

#[test]
fn maps_reject_holes_overlap_bounds_and_group_crossing_and_canonicalize_order() {
    use VarDctStrategy::*;
    for (width, height, tasks) in [
        (0, 8, vec![placement(0, 0, Dct8)]),
        (8, 0, vec![placement(0, 0, Dct8)]),
        (16_385, 8, vec![placement(0, 0, Dct8)]),
        (8, 8, vec![]),
        (16, 8, vec![placement(0, 0, Dct8)]),
        (16, 8, vec![placement(0, 0, Dct8), placement(0, 0, Dct8)]),
        (
            16,
            16,
            vec![placement(0, 0, Dct16x16), placement(1, 1, Dct8)],
        ),
        (8, 8, vec![placement(u32::MAX, u32::MAX, Dct8)]),
        (8, 8, vec![placement(0, 0, Dct16x16)]),
        (264, 16, vec![placement(31, 0, Dct16x16)]),
        (16, 264, vec![placement(0, 31, Dct16x16)]),
    ] {
        assert!(VarDctStrategyMap::new(width, height, tasks).is_err());
    }
    let map = packed_map(512, 512, true);
    let mut reversed = map.transforms().to_vec();
    reversed.reverse();
    assert_eq!(VarDctStrategyMap::new(512, 512, reversed).unwrap(), map);
    let plan = TransformPlan::new(map, &Default::default()).unwrap();
    assert_eq!(plan.batches.len(), 27);
    assert_eq!(plan.memory.forward.parameter_bytes, 27 * 64);
    assert_eq!(
        plan.memory.task_metadata_bytes,
        plan.tasks.len() as u64 * 44
    );
    assert!(
        plan.ac_words
            < plan.tasks.len() as u32
                * plan
                    .tasks
                    .iter()
                    .map(|task| task.ac_word_capacity)
                    .max()
                    .unwrap()
    );
    assert_eq!(plan.memory.coefficient_bytes, 512 * 512 * 12);
}

#[test]
fn mixed_strategies_have_native_checked_ac_and_interoperate_across_lf_groups_and_edges() {
    let (device, queue, info) =
        test_device().expect("actual GPU required for mixed strategy evidence");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info.clone(),
        WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&backend);
    let oracles = native::native_oracles();
    let custom_matrices = matrices::selected();
    let matrix_oracle = matrices::oracle(&custom_matrices);
    let mut custom_oracles = native::native_oracles();
    for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&mut custom_oracles) {
        oracle.dequant = matrices::oracle_scales(&matrix_oracle, strategy, &custom_matrices);
    }
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let quantized_map = |width, height, all| {
        let map = packed_map(width, height, all);
        let transforms = map
            .transforms()
            .iter()
            .copied()
            .enumerate()
            .map(|(index, task)| match index % 4 {
                0 => task,
                which => task.with_hf_multiplier(
                    crate::VarDctHfMultiplier::new([0, 1, 255, 256][which]).unwrap(),
                ),
            })
            .collect();
        VarDctStrategyMap::new(width, height, transforms).unwrap()
    };
    let mut maps = vec![
        packed_map(512, 512, true),
        packed_map(2057, 17, false),
        VarDctStrategyMap::new(
            13,
            21,
            vec![
                placement(0, 0, VarDctStrategy::Dct16x16),
                placement(0, 2, VarDctStrategy::Dct8x16),
            ],
        )
        .unwrap(),
        VarDctStrategyMap::new(
            24,
            16,
            vec![
                placement(0, 0, VarDctStrategy::Dct8),
                placement(1, 0, VarDctStrategy::Dct16x16),
                placement(0, 1, VarDctStrategy::Dct8),
            ],
        )
        .unwrap(),
        quantized_map(512, 512, true),
        quantized_map(2057, 17, false),
    ];
    maps.extend_from_within(..3);
    maps.extend_from_within(..3);
    let mut streams = Vec::new();
    for (case, map) in maps.into_iter().enumerate() {
        let Extent2d { width, height } = map.extent();
        let (w, h) = (width as usize, height as usize);
        let pixels = reference::pattern(w, h);
        let metadata = if case == 1 {
            custom_lf_metadata()
        } else {
            VarDctLfMetadata::default()
        };
        let oracles = if case >= 9 { &custom_oracles } else { &oracles };
        let config = VarDctConfig {
            dequant_matrices: if case >= 9 {
                custom_matrices.clone()
            } else {
                Default::default()
            },
            lf_metadata: metadata,
            quantization: if case >= 4 {
                VarDctQuantization::new(13000, 257, crate::VarDctHfMultiplier::new(19).unwrap())
                    .unwrap()
            } else {
                VarDctQuantization::default()
            },
            coefficient_orders: if case >= 6 {
                orders::selected(map.transforms().iter().map(|task| task.strategy))
            } else {
                Default::default()
            },
        };
        let plan = TransformPlan::new(map.clone(), &config).unwrap();
        let encoder =
            VarDctBackend::new_with_strategy_map(&context, map.clone(), config.clone()).unwrap();

        let source = padded_rgb_source_sized(&context, w, h, &pixels);
        let request = FrameEncodeRequest {
            frame_index: FrameIndex::new(0),
            is_last: true,
            profile: EncodeProfile::VarDct {
                quantization: config.quantization,
            },
            progressive: ProgressivePlan::single(),
            minimum_determinism: Determinism::SameDevice,
            animation: AnimationHeader::Still,
            canvas_width: width,
            canvas_height: height,
            options: FrameOptions::default(),
        };
        let job = encoder
            .submit(&context, GpuFrameSource::Buffer(source), &request)
            .unwrap();
        let (words, lengths, artifacts) = job.wait_with_ac_fragments_for_test().unwrap();
        let mut invalid_lengths = lengths.clone();
        invalid_lengths[0] = 0;
        assert!(
            plan.validate_ac(
                &words,
                &invalid_lengths,
                &HfEntropyPlan::single_cluster_prefix().unwrap()
            )
            .is_err()
        );
        let mut nonzero = 0;
        for (index, task) in plan.tasks.iter().enumerate() {
            let (tw, th) = (task.width as usize, task.height as usize);
            let patch = (0..tw * th)
                .map(|pixel| {
                    let x = (task.block_x as usize * 8 + pixel % tw).min(w - 1);
                    let y = (task.block_y as usize * 8 + pixel / tw).min(h - 1);
                    pixels[y * w + x]
                })
                .collect::<Vec<_>>();
            let oracle = &oracles[task.strategy as usize];
            let coefficients = native::forward(&patch, tw, th, oracle);
            let start = task.ac_word_offset as usize;
            nonzero += native::check_ac(
                &words[start..start + task.ac_word_capacity as usize],
                lengths[index],
                &coefficients,
                oracle,
                VarDctConfig {
                    quantization: VarDctQuantization::new(
                        config.quantization.global_scale(),
                        config.quantization.quant_lf(),
                        crate::VarDctHfMultiplier::new(task.hf_multiplier).unwrap(),
                    )
                    .unwrap(),
                    ..config.clone()
                },
            );
        }
        assert!(nonzero > 0, "mixed textured images must retain AC");
        let frame = assemble_frame(artifacts.packets).unwrap();
        let mut stream = image_header(width, height).unwrap().bytes().to_vec();
        stream.extend_from_slice(frame.bytes());
        let rust = decode_rgb8_sized(&stream, w, h);
        let mut session = decoder
            .open(
                &stream,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        let output = readback.submit(frame.output()).unwrap().wait().unwrap();
        let gpu = output.frame.outputs[0].bytes.clone();
        drop(frame);
        assert!(session.next_frame().unwrap().is_none());
        let input = directory.join(format!("mixed-{case}.jxl"));
        let output = input.with_extension("ppm");
        fs::write(&input, &stream).unwrap();
        assert!(
            Command::new("djxl")
                .arg(&input)
                .arg(&output)
                .args(["--num_threads=0", "--quiet"])
                .status()
                .unwrap()
                .success()
        );
        let native_pixels = read_ppm_rgb8(&output, w, h);
        eprintln!(
            "quantizer case {case} {config:?}: GPU/Rust {}, native/Rust {}, GPU/native {}, files {}",
            max_abs_error(&gpu, &rust),
            max_abs_error(&native_pixels, &rust),
            max_abs_error(&gpu, &native_pixels),
            input.display()
        );
        fs::write(input.with_extension("gpu.rgb"), &gpu).unwrap();
        fs::write(input.with_extension("rust.rgb"), &rust).unwrap();
        assert!(
            max_abs_error(&native_pixels, &rust) <= 1,
            "native/Rust case {case}"
        );
        assert!(max_abs_error(&gpu, &rust) <= 1, "GPU/Rust case {case}");

        eprintln!(
            "mixed {width}x{height}: {} transforms, {nonzero} checked nonzero AC coefficients",
            plan.tasks.len()
        );
        streams.push((map, config, pixels, stream));
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
            let mut reordered = map.transforms().to_vec();
            reordered.reverse();
            let map = VarDctStrategyMap::new(extent.width, extent.height, reordered).unwrap();
            let encoder =
                VarDctEncoder::new_with_strategy_map(context.clone(), map, config.clone()).unwrap();
            let source = padded_rgb_source_sized(
                &context,
                extent.width as usize,
                extent.height as usize,
                pixels,
            );
            assert_eq!(
                &encoder.encode(source).unwrap(),
                expected,
                "{variant:?}/{extent:?}"
            );
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn mixed_jobs_admit_exact_memory_reject_wrong_extents_and_release_after_cancellation() {
    let context = test_context().expect("actual GPU required for mixed memory evidence");
    for (case, coefficient_orders) in [
        Default::default(),
        orders::selected(VarDctStrategy::ALL),
        orders::selected(VarDctStrategy::ALL),
    ]
    .into_iter()
    .enumerate()
    {
        let map = packed_map(512, 512, true);
        let metadata = VarDctLfMetadata::default();
        let custom = VarDctConfig {
            coefficient_orders,
            dequant_matrices: if case == 2 {
                matrices::selected()
            } else {
                Default::default()
            },
            ..config_with_lf(metadata)
        };
        let pixels = reference::pattern(512, 512);
        let encoder =
            VarDctEncoder::new_with_strategy_map(context.clone(), map.clone(), custom.clone())
                .unwrap();
        let source = padded_rgb_source_sized(&context, 512, 512, &pixels);
        let memory = encoder.memory_plan(&source).unwrap();
        assert_eq!(memory.kernel_layout, VarDctKernelLayout::StrategyMap);
        let wrong = padded_rgb_source_sized(&context, 8, 8, &[[0; 3]; 64]);
        assert!(matches!(
            encoder.submit(wrong),
            Err(EncodeError::InvalidSource(_))
        ));
        let insufficient = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(memory.owned_bytes_per_job - 1).unwrap(),
        )
        .unwrap();
        let encoder =
            VarDctEncoder::new_with_strategy_map(insufficient, map.clone(), custom.clone())
                .unwrap();
        assert!(matches!(
            encoder.submit(source.clone()),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let exact = WgpuContext::with_memory_budget(
            Arc::new(context.device().clone()),
            Arc::new(context.queue().clone()),
            NonZeroU64::new(memory.owned_bytes_per_job).unwrap(),
        )
        .unwrap();
        let encoder = VarDctEncoder::new_with_strategy_map(exact.clone(), map, custom).unwrap();
        let submission = encoder.submit(source.clone()).unwrap();
        assert_eq!(
            encoder.in_flight_memory_stats().reserved_bytes,
            memory.owned_bytes_per_job
        );
        drop(submission);
        let fence = exact.queue().submit([]);
        exact
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
            exact.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let bytes = encoder.encode(source).unwrap();
        assert_eq!(decode_rgb8_sized(&bytes, 512, 512).len(), 512 * 512 * 3);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
