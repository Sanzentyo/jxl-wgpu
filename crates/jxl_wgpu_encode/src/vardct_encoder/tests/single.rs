//! Every-strategy compressed-coefficient and decoded-image interoperability.

use jxl_gpu_bitstream::BitReader;

use super::super::dispatch::{VarDctBackend, profile_distance};
use super::*;
use crate::{
    AnimationHeader, Determinism, EncodeProfile, FrameEncodeRequest, FrameIndex, FrameOptions,
    GpuEncodeBackend, GpuFrameSource, ProgressivePlan,
};

struct Oracle {
    order: Vec<usize>,
    dequant: [Vec<f64>; 3],
    basis: Vec<f64>,
}

fn native_oracles() -> Vec<Oracle> {
    let bytes = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../jxl_gpu_protocol/test-data/vardct_metadata.bin"
    ))
    .unwrap();
    assert_eq!(&bytes[..8], b"JXLQNT01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    assert_eq!(words.next(), Some(27));
    let mut oracles = (0..27)
        .map(|id| {
            assert_eq!(words.next(), Some(id));
            let area = words.next().unwrap() as usize;
            let order = words
                .by_ref()
                .take(area)
                .map(|word| word as usize)
                .collect();
            let dequant = std::array::from_fn(|_| {
                words
                    .by_ref()
                    .take(area)
                    .map(|word| f64::from(f32::from_bits(word)))
                    .collect()
            });
            Oracle {
                order,
                dequant,
                basis: if area == 64 {
                    vec![0.0; 4096]
                } else {
                    Vec::new()
                },
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(words.next(), None);
    let bytes = fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../jxl_wgpu/test-data/forward_vardct.bin"
    ))
    .unwrap();
    assert_eq!(&bytes[..8], b"JXLFWD01");
    let mut words = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word));
    let records = words.next().unwrap();
    assert_eq!(records, 667);
    let mut impulses = [0; 27];
    for _ in 0..records {
        let id = words.next().unwrap() as usize;
        let test = words.next().unwrap();
        let width = words.next().unwrap() as usize;
        let height = words.next().unwrap() as usize;
        let area = width * height;
        for coefficient in 0..3 * (area + area / 64) {
            let value = f64::from(f32::from_bits(words.next().unwrap()));
            if test != 0 && coefficient < area {
                oracles[id].basis[coefficient * 64 + test as usize - 1] = value;
            }
        }
        impulses[id] += usize::from(test != 0);
    }
    assert_eq!(words.next(), None);
    for (oracle, impulses) in oracles.iter().zip(impulses) {
        assert_eq!(impulses, if oracle.basis.is_empty() { 0 } else { 64 });
    }
    oracles
}

fn forward(pixels: &[[u8; 3]], width: usize, height: usize, oracle: &Oracle) -> Vec<[f64; 3]> {
    let xyb = pixels
        .iter()
        .map(|&pixel| reference::xyb(pixel))
        .collect::<Vec<_>>();
    let area = width * height;
    if !oracle.basis.is_empty() {
        return (0..area)
            .map(|coefficient| {
                std::array::from_fn(|channel| {
                    xyb.iter()
                        .enumerate()
                        .map(|(pixel, value)| {
                            value[channel] * oracle.basis[coefficient * 64 + pixel]
                        })
                        .sum()
                })
            })
            .collect();
    }
    let cosines = |size: usize| {
        (0..size * size)
            .map(|index| {
                let frequency = index / size;
                let position = index % size;
                if frequency == 0 {
                    1.0 / size as f64
                } else {
                    std::f64::consts::SQRT_2
                        * (std::f64::consts::PI * frequency as f64 * (position as f64 + 0.5)
                            / size as f64)
                            .cos()
                        / size as f64
                }
            })
            .collect::<Vec<_>>()
    };
    let horizontal_basis = cosines(width);
    let vertical_basis = cosines(height);
    let mut horizontal = vec![[0.0; 3]; area];
    for y in 0..height {
        for fx in 0..width {
            horizontal[y * width + fx] = std::array::from_fn(|channel| {
                (0..width)
                    .map(|x| xyb[y * width + x][channel] * horizontal_basis[fx * width + x])
                    .sum()
            });
        }
    }
    let mut result = vec![[0.0; 3]; area];
    for fy in 0..height {
        for fx in 0..width {
            let wire = if height < width {
                fy * width + fx
            } else {
                fx * height + fy
            };
            result[wire] = std::array::from_fn(|channel| {
                (0..height)
                    .map(|y| horizontal[y * width + fx][channel] * vertical_basis[fy * height + y])
                    .sum()
            });
        }
    }
    result
}

fn check_ac(
    words: &[u32],
    bit_len: u32,
    coefficients: &[[f64; 3]],
    oracle: &Oracle,
    metadata: VarDctLfMetadata,
) -> usize {
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let mut reader = BitReader::new(&bytes);
    let slopes = metadata
        .base_correlation()
        .map(|value| f64::from(value.to_f32()));
    let skip = coefficients.len() / 64;
    let mut nonzero = 0;
    for channel in [1, 0, 2] {
        let mut remaining = ac::read_unsigned(&mut reader, &entropy);
        assert!(remaining as usize <= coefficients.len() - skip);
        for &index in &oracle.order[skip..] {
            let packed = if remaining == 0 {
                0
            } else {
                ac::read_unsigned(&mut reader, &entropy)
            };
            remaining -= u32::from(packed != 0);
            nonzero += usize::from(packed != 0);
            let actual = if packed.is_multiple_of(2) {
                (packed / 2) as i32
            } else {
                -((packed / 2) as i32) - 1
            };
            let coefficient = coefficients[index];
            let decorrelated = match channel {
                0 => coefficient[0] - slopes[0] * coefficient[1],
                1 => coefficient[1],
                _ => coefficient[2] - slopes[1] * coefficient[1],
            };
            let expected = (decorrelated * (8813.0 * 6.0 / 65536.0) * [1.25, 1.0, 1.0][channel]
                / oracle.dequant[channel][index])
                .round() as i32;
            // Fixed f64/native oracle regression bound: at most one quantizer
            // step across f32 rounding boundaries, not a distance-quality claim.
            assert!(
                (actual - expected).abs() <= 1,
                "channel {channel}, coefficient {index}: GPU={actual}, reference={expected}"
            );
        }
        assert_eq!(remaining, 0);
    }
    assert_eq!(reader.bit_offset(), u64::from(bit_len));
    assert!(nonzero > 0, "a textured input must not take a zero-AC path");
    nonzero
}

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
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let djxl = "djxl";
    let mut streams = Vec::new();
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
        for (custom, metadata) in [VarDctLfMetadata::default(), custom_lf_metadata()]
            .into_iter()
            .enumerate()
        {
            let encoder =
                VarDctBackend::new_with_lf_metadata(&context, strategy, metadata).unwrap();
            let source = padded_rgb_source_sized(&context, w, h, &pixels);
            let request = FrameEncodeRequest {
                frame_index: FrameIndex::new(0),
                is_last: true,
                profile: EncodeProfile::VarDct {
                    distance: profile_distance(),
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
            let nonzero = check_ac(&words, bits, &coefficients, oracle, metadata);
            let frame = assemble_frame(artifacts.packets).unwrap();
            let mut stream = image_header(width, height).unwrap().bytes().to_vec();
            stream.extend_from_slice(frame.bytes());
            let convenience =
                VarDctEncoder::new_with_lf_metadata(context.clone(), strategy, metadata).unwrap();
            assert_eq!(
                pollster::block_on(convenience.submit(source).unwrap()).unwrap(),
                stream
            );
            let rust = decode_rgb8_sized(&stream, w, h);
            let mut session = decoder
                .open(
                    &stream,
                    GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
                )
                .unwrap();
            let frame = session.next_frame().unwrap().unwrap();
            let output = readback.submit(frame.output()).unwrap().wait().unwrap();
            let error = max_abs_error(&output.frame.outputs[0].bytes, &rust);
            assert!(
                error <= 1,
                "{strategy:?}/{custom}: GPU/Rust max error {error}"
            );
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
            streams.push((strategy, metadata, pixels.clone(), stream));
        }
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
        for (strategy, metadata, pixels, expected) in &streams {
            let Extent2d { width, height } = strategy.pixel_extent();
            let encoder =
                VarDctEncoder::new_with_lf_metadata(context.clone(), *strategy, *metadata).unwrap();
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
        eprintln!("{variant:?}: 54 single-transform codestreams are byte-identical");
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
        std::iter::once(maximum).chain(std::iter::repeat_n(262_142, maximum as usize))
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
    const LIMIT: u64 = 400 * 1024;
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
            required: 524_288,
            available: LIMIT
        }))
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}
