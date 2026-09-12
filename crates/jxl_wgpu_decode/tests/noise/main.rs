#![cfg(not(target_arch = "wasm32"))]

use jxl_test_support::gpu::planes;
use jxl_test_support::oracles::extra_channels as oracle;

use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

use fixtures::{encoded, zero_noise};
use jxl_test_support::fixtures::noise as fixtures;

fn planned_bytes(
    backend: &WgpuBackend,
    session: &mut jxl_wgpu_decode::GpuDecodeSession<jxl_wgpu_decode::WgpuDecodeSubmissionSession>,
) -> u64 {
    // Probe the public admission contract, including Modular's composition producer. First
    // exhaust output admission, then leave exactly its allocation free to exhaust scratch.
    let budget = backend.transient_memory_budget();
    let mut required = 0;
    for _ in 0..2 {
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        let held = budget
            .try_reserve(budget.snapshot().limit_bytes - required)
            .unwrap();
        let progress = session
            .prefetch(std::num::NonZeroUsize::new(1).unwrap())
            .unwrap();
        assert_eq!(progress.submitted, 0);
        let Some(jxl_wgpu_decode::PrefetchBackpressure::Memory(
            jxl_wgpu::MemoryBudgetError::Exhausted {
                requested_bytes,
                reserved_bytes,
                limit_bytes,
            },
        )) = progress.backpressure
        else {
            panic!("expected memory admission pressure: {progress:?}");
        };
        assert_eq!(reserved_bytes, limit_bytes);
        assert_eq!(budget.snapshot().reserved_bytes, held.bytes());
        required += requested_bytes;
        drop(held);
    }
    required
}

fn wait_for_release(backend: &WgpuBackend) {
    let budget = backend.transient_memory_budget();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while budget.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(budget.snapshot().reserved_bytes, 0);
}

#[test]
fn noise_scratch_is_admitted_before_submission_and_released_on_retry_and_cancellation() {
    use std::num::NonZeroUsize;
    let backend = match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    })) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("noise adapter: {error}"),
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap();
    for name in [
        "vardct_257x17",
        "modular_257x17",
        "modular_rgb_group256",
        "modular_gray",
        "vardct_rgb_257x17",
        "vardct_rgb_gray",
        "jpeg_444",
        "jpeg_422",
        "jpeg_440",
        "jpeg_420",
        "jpeg_gray",
    ] {
        let bytes = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let zero = zero_noise(&bytes, &inventory, None);
        let mut empty = decoder.open(&zero, request.clone()).unwrap();
        let empty_bytes = planned_bytes(&backend, &mut empty);
        let component_extent = match name {
            "jpeg_422" => Some((272, 24)),
            "jpeg_440" => Some((264, 32)),
            "jpeg_420" => Some((272, 32)),
            _ => None,
        };
        if component_extent.is_some() {
            let memory = empty
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .unwrap();
            assert_eq!(memory.pre_restoration_upsample_bytes, 0);
            assert_eq!(memory.pre_restoration_upsample_uniform_bytes, 0);
            assert_eq!(memory.noise_bytes, 0);
            assert_eq!(memory.noise_uniform_bytes, 0);
        }
        drop(empty);
        let mut pending = decoder.open(&bytes, request.clone()).unwrap();
        let required = planned_bytes(&backend, &mut pending);
        let component_bytes = component_extent.map_or(0, |(width, height)| {
            let memory = pending
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats()
                .unwrap();
            // Both chroma components need padded, full-resolution destinations for noise.
            let bytes = width * height * 4 * 2;
            assert_eq!(memory.pre_restoration_upsample_bytes, bytes);
            assert_eq!(memory.pre_restoration_upsample_uniform_bytes, 64);
            bytes + 64
        });
        assert_eq!(
            required - empty_bytes,
            257 * 17 * 12
                + jxl_wgpu::ResidentNoisePlan::UNIFORM_BYTES
                + component_bytes
                + if !inventory.image_header.xyb_encoded
                    && inventory.frames[0].encoding == jxl_gpu_bitstream::FrameEncoding::Modular
                {
                    // Noise introduces the color renderer for unfiltered original color:
                    // three normalized planes, three aligned render destinations, and packing.
                    let alignment = u64::from(
                        backend
                            .device()
                            .limits()
                            .min_storage_buffer_offset_alignment,
                    )
                    .max(4);
                    257 * 17 * 12
                        + (257_u64 * 17 * 4).div_ceil(alignment) * alignment * 3
                        + 80
                        + 352
                } else {
                    0
                },
            "{name}"
        );
        let budget = backend.transient_memory_budget();
        assert_eq!(budget.snapshot().reserved_bytes, 0);
        let held = budget
            .try_reserve(budget.snapshot().limit_bytes - required + 1)
            .unwrap();
        let progress = pending.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert!(matches!(
            progress.backpressure,
            Some(jxl_wgpu_decode::PrefetchBackpressure::Memory(_))
        ));
        assert_eq!(budget.snapshot().reserved_bytes, held.bytes());
        drop(held);
        let frame = pollster::block_on(pending.next_frame_async())
            .unwrap()
            .unwrap();
        drop(frame);
        drop(pending);
        wait_for_release(&backend);
        let mut abandoned = planes::open_fragmented(&decoder, &bytes, request.clone());
        assert_eq!(
            abandoned
                .prefetch(NonZeroUsize::new(1).unwrap())
                .unwrap()
                .submitted,
            1
        );
        drop(abandoned);
        wait_for_release(&backend);
    }
}

#[test]
fn both_coding_modes_synthesize_signaled_noise_on_gpu_from_whole_and_bounded_input() {
    check_cases(
        &[
            "vardct_257x17",
            "modular_257x17",
            "vardct_up2",
            "vardct_up4",
            "vardct_up8",
            "modular_up2",
            "modular_up4",
            "modular_up8",
            "mixed_frames",
            "modular_xyb_group128",
            "modular_xyb_group256",
            "modular_xyb_group512",
            "modular_xyb_group1024",
            "modular_rgb_group128",
            "modular_rgb_group256",
            "modular_rgb_group512",
            "modular_rgb_group1024",
            "modular_gray",
            "modular_rgb_up2",
            "modular_rgb_up4",
            "modular_rgb_up8",
        ],
        Reference::Srgb,
    );
}

#[test]
fn noise_uses_base_correlations_independently_of_lf_slopes() {
    check_cases(
        &["vardct_lf_correlation", "vardct_base_correlation"],
        Reference::LinearCorrelation,
    );
}

#[test]
fn original_rgb_vardct_noise_follows_restoration_and_frame_upsampling() {
    check_cases(
        &[
            "vardct_rgb_257x17",
            "vardct_rgb_gray",
            "vardct_rgb_up2",
            "vardct_rgb_up4",
            "vardct_rgb_up8",
            "vardct_rgb_gray16",
            "vardct_rgb_float32_up4",
            "vardct_rgb_frames",
        ],
        Reference::Srgb,
    );
}

#[test]
fn ycbcr_noise_follows_component_upsampling() {
    check_cases(
        &["jpeg_444", "jpeg_422", "jpeg_440", "jpeg_420", "jpeg_gray"],
        Reference::Srgb,
    );
}

#[test]
fn original_rgb_noise_preserves_single_channel_implicit_palette_values() {
    check_cases(&["modular_rgb_palette"], Reference::ImplicitPalette);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Reference {
    Srgb,
    // Native linear output avoids extended-range sRGB approximation error; checked snapshots
    // make this regression mandatory even without libjxl installed. Rust jxl 0.6 incorrectly
    // uses LF-adjusted correlations for noise, so it is not an oracle for the LF-slope case.
    LinearCorrelation,
    // ISO/IEC 18181-1 H.6.4 applies implicit entries to single-channel palettes too.
    // libjxl 0.12 clamps these indices in its single-channel, zero-delta, Zero-predictor path.
    ImplicitPalette,
}

fn check_cases(names: &[&str], reference: Reference) {
    let backend = match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    })) {
        Ok(backend) => backend,
        Err(jxl_wgpu::Error::NoAdapter) => return,
        Err(error) => panic!("noise adapter: {error}"),
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(std::num::NonZeroU64::new(256).unwrap()),
    );
    for &name in names {
        let bytes = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(inventory.frames.iter().all(|frame| frame.flags & 1 != 0));
        let image = &inventory.image_header;
        let frame = &inventory.frames[0];
        if name.starts_with("vardct_rgb_") || name.starts_with("jpeg_") {
            assert!(!image.xyb_encoded);
            assert_eq!(frame.encoding, jxl_gpu_bitstream::FrameEncoding::VarDct);
            assert_eq!(frame.do_ycbcr, name.starts_with("jpeg_"));
            assert_eq!(image.grayscale, name.contains("_gray"));
            let (depth, orientation) = match name {
                "vardct_rgb_gray16" => (
                    jxl_gpu_bitstream::SampleBitDepth::Integer {
                        bits_per_sample: 16,
                    },
                    6,
                ),
                "vardct_rgb_float32_up4" => (
                    jxl_gpu_bitstream::SampleBitDepth::Float {
                        bits_per_sample: 32,
                        exponent_bits_per_sample: 8,
                    },
                    1,
                ),
                _ => (
                    jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample: 8 },
                    if name == "vardct_rgb_frames" { 8 } else { 1 },
                ),
            };
            assert_eq!(image.bit_depth, depth);
            assert_eq!(image.orientation, orientation);
            let sampling = match name {
                "jpeg_422" => [0, 2, 0],
                "jpeg_440" => [0, 3, 0],
                "jpeg_420" => [0, 1, 0],
                _ => [0; 3],
            };
            assert_eq!(frame.jpeg_upsampling, sampling);
            let factor = name
                .rsplit_once("_up")
                .map_or(1, |(_, factor)| factor.parse::<u32>().unwrap());
            assert_eq!(frame.upsampling, factor);
            assert_eq!(
                frame.restoration_filter,
                if factor == 1 {
                    jxl_gpu_bitstream::RestorationFilterInventory::Custom {
                        gaborish: jxl_gpu_bitstream::GaborishInventory::Disabled,
                        epf: jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled,
                    }
                } else {
                    jxl_gpu_bitstream::RestorationFilterInventory::Default
                }
            );
        }
        assert_eq!(inventory.frames[0].noise_seed, [1, 0]);
        if let Some((_, dimension)) = name.rsplit_once("_group") {
            let dimension: u32 = dimension.parse().unwrap();
            assert_eq!(128 << inventory.frames[0].group_size_shift, dimension);
            assert_eq!(inventory.image_header.width, dimension + 1);
        }
        if name.starts_with("modular_rgb_") || name == "modular_gray" {
            assert!(!inventory.image_header.xyb_encoded);
        }
        if reference == Reference::LinearCorrelation {
            let mut range = inventory.frames[0].sections[0].bits;
            range.offset += 80;
            range.length -= 80;
            let prefix =
                jxl_wgpu_decode::vardct::frontend::LfGlobalPrefix::parse(&bytes, range).unwrap();
            assert_eq!(prefix.lf_correlation.colour_factor, 84);
            let (base, factors) = if name == "vardct_lf_correlation" {
                ([0.0, 1.0], [12, -19])
            } else {
                ([0.125, 0.875], [0, 0])
            };
            assert_eq!(prefix.lf_correlation.base, base);
            assert_eq!(prefix.lf_correlation.lf_factors, factors);
        }
        let animation = image.animation.is_some();
        if animation {
            assert_eq!(
                inventory
                    .frames
                    .iter()
                    .map(|frame| frame.noise_seed)
                    .collect::<Vec<_>>(),
                [[1, 0], [1, 1], [1, 2], [2, 0], [3, 0]]
            );
        }
        let mut noisy_words = None;
        for zero_model in [false, true] {
            let bytes = if zero_model {
                zero_noise(&bytes, &inventory, None)
            } else {
                bytes.clone()
            };
            let pixels =
                inventory.image_header.width as usize * inventory.image_header.height as usize;
            let rust = (reference != Reference::LinearCorrelation).then(|| {
                let frames = if animation {
                    oracle::rust_frame_planes(&bytes)
                } else {
                    vec![oracle::rust_planes(&bytes)]
                };
                frames
                    .into_iter()
                    .flat_map(|(rgb, extras)| {
                        assert!(extras.is_empty());
                        rgb
                    })
                    .collect::<Vec<f32>>()
            });
            let snapshot = (reference == Reference::LinearCorrelation).then(|| {
                let suffix = if zero_model { "zero.linear" } else { "linear" };
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("test-data/noise/{name}.{suffix}.f32.hex"));
                std::fs::read_to_string(path)
                    .unwrap()
                    .split_whitespace()
                    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
                    .collect::<Vec<_>>()
            });
            let native = if reference == Reference::ImplicitPalette {
                None
            } else if reference == Reference::LinearCorrelation {
                oracle::libjxl_output(&bytes, &["--linear"])
            } else if animation {
                oracle::libjxl_output(&bytes, &[])
            } else {
                oracle::libjxl_planes(&bytes, pixels, 0).map(|(rgb, _)| rgb)
            };
            if reference == Reference::ImplicitPalette && zero_model {
                let rust = rust.as_ref().unwrap();
                assert_eq!(&rust[..4], &[35.0 / 255.0, 4.0 / 255.0, 39.0 / 255.0, 1.0]);
                assert_eq!(rust[5], 0.0);
            }
            let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
            if reference == Reference::LinearCorrelation {
                let jxl_gpu_formats::ColorSpecification::Defined(ref mut defined) = color else {
                    panic!("defined RGB color");
                };
                defined.transfer = jxl_gpu_formats::TransferFunction::Linear;
            }
            let request =
                GpuOutputRequest::color(PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color))
                    .unwrap();
            let mut expected_words = None;
            for fragmented in [false, true] {
                let mut session = if fragmented {
                    planes::open_fragmented(&bounded, &bytes, request.clone())
                } else {
                    whole.open(&bytes, request.clone()).unwrap()
                };
                let mut words = Vec::new();
                while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
                    words.extend(planes::read(&backend, &frame.output().outputs[0]));
                }
                assert_eq!(words.len(), pixels * 4 * if animation { 3 } else { 1 });
                if let Some(expected) = &expected_words {
                    assert_eq!(&words, expected, "{name}");
                }
                for (reference, label) in rust
                    .as_ref()
                    .map(|data| (data, "Rust"))
                    .into_iter()
                    .chain(snapshot.as_ref().map(|data| (data, "snapshot")))
                    .chain(native.as_ref().map(|data| (data, "libjxl")))
                {
                    assert_eq!(reference.len(), words.len());
                    let mut maximum = 0.0_f32;
                    for (&word, &expected) in words.iter().zip(reference) {
                        let actual = f32::from_bits(word);
                        assert!(
                            actual.is_finite() && expected.is_finite(),
                            "{name} {label}: non-finite color sample"
                        );
                        maximum = maximum.max((actual - expected).abs());
                    }
                    eprintln!(
                        "{name} zero={zero_model} fragmented={fragmented} {label} maxAE={maximum}"
                    );
                    // The two CPU transform implementations have different rounding; the native
                    // bound is a quarter of one RGB8 code. The RNG itself is checked bit-exactly.
                    let limit = if label == "Rust" {
                        0.0001
                    } else {
                        1.0 / 1024.0
                    };
                    assert!(maximum < limit, "{name} {label}: maxAE {maximum}");
                }
                if !fragmented {
                    if zero_model {
                        assert_ne!(
                            Some(&words),
                            noisy_words.as_ref(),
                            "noise must change {name}"
                        );
                    } else {
                        noisy_words = Some(words.clone());
                    }
                }
                expected_words = Some(words);
                drop(session);
                backend
                    .device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                assert_eq!(whole.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(bounded.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
