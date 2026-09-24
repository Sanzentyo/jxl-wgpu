//! Every-strategy compressed-coefficient and decoded-image interoperability.

use super::native::{check_ac, forward, native_oracles};

use super::super::dispatch::VarDctBackend;
use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuFrameSource, ProgressivePlan,
};

#[test]
fn all_27_strategies_emit_native_checked_nonzero_ac_and_interoperate() {
    let (device, queue, info) =
        test_device().expect("actual GPU required for all-strategy AC evidence");
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
    let oracles = native_oracles();
    let custom_matrices = matrices::selected();
    let matrix_oracle = matrices::oracle(&custom_matrices);
    let mut custom_oracles = native_oracles();
    let mut raw_oracles = native_oracles();
    for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&mut raw_oracles) {
        oracle.dequant = raw_matrices::oracle_scales(strategy);
    }
    for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&mut custom_oracles) {
        oracle.dequant = matrices::oracle_scales(&matrix_oracle, strategy, &custom_matrices);
    }
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let djxl = "djxl";
    let mut streams = Vec::new();
    let mut gpu_mismatches = Vec::new();
    eprintln!("single-transform AC adapter: {info:?}; pinned native matrices and transforms");
    for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&oracles) {
        let Extent2d { width, height } = strategy.pixel_extent();
        let (w, h) = (width as usize, height as usize);
        let pixels = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                [
                    ((x * 37 + y * 19) % 256) as u8,
                    ((x * 13 + y * 53) % 256) as u8,
                    ((x * 71 + y * 11) % 256) as u8,
                ]
            })
            .collect::<Vec<_>>();
        let coefficients = forward(&pixels, w, h, oracle);
        let mut natural_pixels = Vec::new();
        for (custom, metadata) in [
            VarDctLfMetadata::default(),
            custom_lf_metadata(),
            VarDctLfMetadata::default(),
            custom_lf_metadata(),
            custom_lf_metadata(),
            custom_lf_metadata(),
        ]
        .into_iter()
        .enumerate()
        {
            let oracle = if custom == 5 {
                &raw_oracles[strategy.codestream_id() as usize]
            } else if custom == 4 {
                &custom_oracles[strategy.codestream_id() as usize]
            } else {
                oracle
            };
            let config = VarDctConfig {
                dequant_matrices: if custom == 5 {
                    raw_matrices::selected([strategy])
                } else if custom == 4 {
                    custom_matrices.clone()
                } else {
                    Default::default()
                },
                coefficient_orders: if custom >= 2 {
                    orders::selected([strategy])
                } else {
                    Default::default()
                },
                ..config_with_lf(metadata)
            };
            let encoder =
                VarDctBackend::new_with_config(&context, strategy, config.clone()).unwrap();
            let source = padded_rgb_source_sized(&context, w, h, &pixels);
            let request = FrameEncodeRequest {
                frame_index: FrameIndex::new(0),
                is_last: true,
                profile: EncodeProfile::VarDct {
                    quantization: VarDctQuantization::default(),
                },
                progressive: ProgressivePlan::single(),
                minimum_determinism: Determinism::SameDevice,
                animation: AnimationHeader::Still,
                canvas_width: width,
                canvas_height: height,
                options: FrameOptions::default(),
            };
            let job = encoder
                .submit(&context, GpuFrameSource::Buffer(source.clone()), &request)
                .unwrap();
            let (words, bits, artifacts) = job.wait_with_ac_for_test().unwrap();
            let nonzero = check_ac(&words, bits, &coefficients, oracle, config.clone());
            assert!(nonzero > 0, "textured input must emit nonzero AC");
            let frame = assemble_frame(artifacts.packets).unwrap();
            let mut stream = image_header(width, height, crate::AnimationHeader::Still)
                .unwrap()
                .bytes()
                .to_vec();
            stream.extend_from_slice(frame.bytes());
            let convenience =
                VarDctEncoder::new_with_config(context.clone(), strategy, config.clone()).unwrap();
            assert_eq!(
                pollster::block_on(convenience.submit(source).unwrap()).unwrap(),
                stream
            );
            let rust = decode_rgb8_sized(&stream, w, h);
            if custom < 2 {
                natural_pixels.push(rust.clone());
            } else if custom < 4 {
                assert_eq!(
                    rust,
                    natural_pixels[custom - 2],
                    "custom orders preserve quantized pixels"
                );
            }
            let mut session = decoder
                .open(
                    &stream,
                    GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
                )
                .unwrap();
            let frame = session.next_frame().unwrap().unwrap();
            let output = readback.submit(frame.output()).unwrap().wait().unwrap();
            let error = max_abs_error(&output.frame.outputs[0].bytes, &rust);
            if error > 1 {
                gpu_mismatches.push((strategy, custom, error));
            }
            drop(frame);
            assert!(session.next_frame().unwrap().is_none());
            let input = directory.join(format!("{strategy:?}-{custom}.jxl"));
            let output = input.with_extension("ppm");
            fs::write(&input, &stream).unwrap();
            assert!(
                Command::new(djxl)
                    .arg(&input)
                    .arg(&output)
                    .args(["--num_threads=0", "--quiet"])
                    .status()
                    .unwrap()
                    .success()
            );
            let native = read_ppm_rgb8(&output, w, h);
            let error = max_abs_error(&native, &rust);
            assert!(
                error <= 1,
                "{strategy:?}/{custom}: native/Rust max error {error}"
            );
            eprintln!("{strategy:?}/{custom}: {nonzero} nonzero coefficients, {bits} AC bits");
            streams.push((strategy, config, pixels.clone(), stream));
        }
    }
    assert!(
        gpu_mismatches.is_empty(),
        "GPU/Rust mismatches: {gpu_mismatches:?}"
    );
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
        for (strategy, config, pixels, expected) in &streams {
            let Extent2d { width, height } = strategy.pixel_extent();
            let encoder =
                VarDctEncoder::new_with_config(context.clone(), *strategy, config.clone()).unwrap();
            let actual = encoder
                .encode(padded_rgb_source_sized(
                    &context,
                    width as usize,
                    height as usize,
                    pixels,
                ))
                .unwrap();
            assert_eq!(&actual, expected, "{strategy:?}/{variant:?}");
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
        eprintln!(
            "{variant:?}: {} single-transform codestreams are byte-identical",
            streams.len()
        );
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn maximum_transform_fragment_and_invalid_counts_are_bounded() {
    use super::super::ac::validate_transform_fragments;
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let layout =
        ArtifactLayout::new(VarDctStrategy::Dct256x256, &fixed_prefix_code().unwrap()).unwrap();
    let maximum = 65_536 - 1_024;
    let tokens = (0..3).flat_map(|_| {
        std::iter::once(maximum).chain(std::iter::repeat_n(u32::MAX, maximum as usize))
    });
    let (mut words, bits) = ac::write_tokens(tokens, &entropy);
    assert_eq!(words.len(), layout.ac_words_per_block as usize);
    words.resize(layout.ac_words_per_block as usize, 0);
    validate_transform_fragments(
        &words,
        &[bits],
        layout.ac_words_per_block,
        maximum,
        &entropy,
    )
    .unwrap();
    for invalid in [0, bits - 1, bits + 1, layout.ac_words_per_block * 32 + 1] {
        assert!(
            validate_transform_fragments(
                &words,
                &[invalid],
                layout.ac_words_per_block,
                maximum,
                &entropy
            )
            .is_err()
        );
    }
    let (mut invalid, length) = ac::write_tokens([maximum + 1, 0, 0], &entropy);
    invalid.resize(layout.ac_words_per_block as usize, 0);
    assert!(
        validate_transform_fragments(
            &invalid,
            &[length],
            layout.ac_words_per_block,
            maximum,
            &entropy
        )
        .is_err()
    );
    eprintln!(
        "largest transform: {bits} maximum AC bits in {} words",
        layout.ac_words_per_block
    );
}

#[test]
fn single_transform_dispatch_grid_is_checked_before_recording() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("actual adapter required for single-transform dispatch admission");
    let info = adapter.get_info();
    let pixels = reference::pattern(8, 8);
    for axis_limit in [4, 8] {
        let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
        limits.max_compute_workgroups_per_dimension = axis_limit;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        }))
        .unwrap();
        let context = test_context_with_variants(
            &Arc::new(device),
            &Arc::new(queue),
            &info,
            &[(FORWARD_KERNEL_KEY, KernelVariant::Scalar)],
        )
        .unwrap();
        let source = padded_rgb_source_sized(&context, 8, 8, &pixels);
        let encoder = VarDctEncoder::new(context, VarDctStrategy::Dct8).unwrap();
        if axis_limit == 4 {
            assert!(matches!(
                encoder.submit(source),
                Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
                    name: "max_compute_workgroups_per_dimension",
                    required: 16,
                    available: 4,
                }))
            ));
        } else {
            let encoded = encoder.encode(source).unwrap();
            assert_eq!(decode_rgb8_sized(&encoded, 8, 8).len(), 8 * 8 * 3);
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn single_transform_metadata_storage_is_admitted_before_submission() {
    const LIMIT: u64 = 500 * 1024;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("actual GPU required for single-transform admission evidence");
    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_storage_buffer_binding_size = LIMIT;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
    let strategy = VarDctStrategy::Dct256x128;
    let pixels = reference::pattern(128, 256);
    let source = padded_rgb_source_sized(&context, 128, 256, &pixels);
    let layout = ArtifactLayout::new(strategy, &fixed_prefix_code().unwrap()).unwrap();
    assert!(source.buffer.size() < LIMIT && layout.artifact_bytes() < LIMIT);
    let encoder = VarDctEncoder::new(context, strategy).unwrap();
    assert!(matches!(
        encoder.submit(source),
        Err(EncodeError::Unsupported(UnsupportedFeature::DeviceLimit {
            name: "max_storage_buffer_binding_size",
            required: 786_432,
            available: LIMIT
        }))
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}
