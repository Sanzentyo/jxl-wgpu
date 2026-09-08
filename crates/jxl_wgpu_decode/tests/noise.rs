#![cfg(not(target_arch = "wasm32"))]

#[path = "common/extra_channel_oracle.rs"]
mod oracle;
#[path = "support/planes.rs"]
mod planes;

use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

fn encoded(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("test-data/noise/{name}.jxl.hex"));
    let text: String = std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .collect();
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

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
    for name in ["vardct_257x17", "modular_257x17"] {
        let bytes = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let mut zero = bytes.clone();
        let start = inventory.frames[0].sections[0].bytes.offset as usize;
        zero[start..start + 10].fill(0);
        let mut empty = decoder.open(&zero, request.clone()).unwrap();
        let empty_bytes = planned_bytes(&backend, &mut empty);
        drop(empty);
        let mut pending = decoder.open(&bytes, request.clone()).unwrap();
        let required = planned_bytes(&backend, &mut pending);
        assert_eq!(
            required - empty_bytes,
            257 * 17 * 12 + jxl_wgpu::ResidentNoisePlan::UNIFORM_BYTES
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
    for name in [
        "vardct_257x17",
        "modular_257x17",
        "vardct_up2",
        "vardct_up4",
        "vardct_up8",
        "modular_up2",
        "modular_up4",
        "modular_up8",
        "mixed_frames",
    ] {
        let bytes = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(inventory.frames.iter().all(|frame| frame.flags & 1 != 0));
        assert_eq!(inventory.frames[0].noise_seed, [1, 0]);
        if name == "mixed_frames" {
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
            let mut bytes = bytes.clone();
            if zero_model {
                for frame in &inventory.frames {
                    assert_eq!(frame.flags & (2 | 16), 0);
                    let section = frame
                        .sections
                        .iter()
                        .find(|section| {
                            matches!(
                                section.kind,
                                jxl_gpu_bitstream::FrameSectionKind::Single
                                    | jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGlobal
                            )
                        })
                        .unwrap();
                    assert_eq!(section.bits.offset % 8, 0);
                    let start = section.bytes.offset as usize;
                    bytes[start..start + 10].fill(0);
                }
            }
            let pixels =
                inventory.image_header.width as usize * inventory.image_header.height as usize;
            let rust_frames = if name == "mixed_frames" {
                oracle::rust_frame_planes(&bytes)
            } else {
                vec![oracle::rust_planes(&bytes)]
            };
            let rust: Vec<f32> = rust_frames
                .into_iter()
                .flat_map(|(rgb, extras)| {
                    assert!(extras.is_empty());
                    rgb
                })
                .collect();
            let native = if name == "mixed_frames" {
                oracle::libjxl_output(&bytes, &[])
            } else {
                oracle::libjxl_planes(&bytes, pixels, 0).map(|(rgb, _)| rgb)
            };
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            ))
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
                assert_eq!(
                    words.len(),
                    pixels * 4 * if name == "mixed_frames" { 3 } else { 1 }
                );
                if let Some(expected) = &expected_words {
                    assert_eq!(&words, expected, "{name}");
                }
                for (reference, label) in std::iter::once((&rust, "Rust"))
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
