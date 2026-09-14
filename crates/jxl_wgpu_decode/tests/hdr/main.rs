#![cfg(not(target_arch = "wasm32"))]

use jxl_test_support::fixtures::hdr as corpus;
use jxl_test_support::oracles::hdr as oracle;
mod icc;
mod independent;
mod output;
mod reference;
mod transfer;

use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use std::num::{NonZeroU64, NonZeroUsize};

fn backend() -> WgpuBackend {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("HDR adapter: {:?}", backend.adapter_info());
    backend
}

fn compare(actual: &[u32], reference: &[f32], tolerance: f32, name: &str) {
    assert_eq!(actual.len(), reference.len(), "{name}");
    let mut maximum = 0.0_f32;
    for (index, (&word, &expected)) in actual.iter().zip(reference).enumerate() {
        let actual = f32::from_bits(word);
        let error = (actual - expected).abs() / (1.0 + expected.abs());
        let limit = if index % 4 == 3 { 2e-6 } else { tolerance };
        assert!(
            actual.is_finite() && expected.is_finite() && error <= limit,
            "{name}/{index}: GPU {actual}, reference {expected}, error {error}, limit {limit}"
        );
        maximum = maximum.max(error);
    }
    eprintln!("{name}: maximum normalized error {maximum}");
}

#[test]
fn original_hdr_stills_and_composed_progression_match_native_whole_and_bounded() {
    let backend = backend();
    for case in corpus::cases() {
        eprintln!("HDR original {}", case.name);
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&inventory);
        let reference = case.reference(false);
        let linear_reference = (case.xyb && !case.sequence).then(|| case.reference(true));
        assert_eq!(reference.len(), case.frame_words() * case.frame_count());
        let mut baseline = None;
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let request = GpuOutputRequest::color(case.format(case.transfer, case.space))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_max_frame_slots(
                    NonZeroUsize::new(
                        inventory
                            .frames
                            .iter()
                            .map(|frame| frame.num_passes as usize + 1)
                            .sum(),
                    )
                    .unwrap(),
                )
                .with_progressive_output(true);
            let mut session = if limit.is_some() {
                planes::open_fragmented(&decoder, &data, request)
            } else {
                decoder.open(&data, request).unwrap()
            };
            let mut held = Vec::new();
            let mut snapshots = Vec::new();
            let mut frames = 0;
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                let words = planes::read(&backend, &update.output().outputs[0]);
                if update.progression().is_none() {
                    let mut maximum = 0.0_f64;
                    for (pixel, words) in words.as_chunks::<4>().0.iter().enumerate() {
                        let index = frames * case.width * case.height + pixel;
                        let bounds = reference::original_bounds(
                            &case,
                            &reference,
                            linear_reference.as_deref(),
                            index,
                        );
                        for c in 0..4 {
                            let actual = f64::from(f32::from_bits(words[c]));
                            let expected = f64::from(reference[index * 4 + c]);
                            let [low, high] = bounds[c];
                            assert!(
                                actual.is_finite() && actual >= low && actual <= high,
                                "{} original {frames}/{pixel}/{c}: {actual}, native {expected}, interval [{low}, {high}]",
                                case.name
                            );
                            maximum =
                                maximum.max((actual - expected).abs() / (1.0 + expected.abs()));
                        }
                    }
                    eprintln!("{}: max normalized original error {maximum}", case.name);
                    frames += 1;
                }
                snapshots.push((update.progression(), words));
                held.push(update);
            }
            assert_eq!(frames, case.frame_count());
            for (frame, (_, expected)) in held.iter().zip(&snapshots) {
                assert_eq!(
                    &planes::read(&backend, &frame.output().outputs[0]),
                    expected
                );
            }
            if let Some(baseline) = &baseline {
                assert_eq!(baseline, &snapshots, "{}: bounded progression", case.name);
            } else {
                baseline = Some(snapshots);
            }
            drop((held, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let request = GpuOutputRequest::color(case.format(case.transfer, case.space))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
        let mut session = decoder.open(&data, request).unwrap();
        for (_, expected) in baseline
            .unwrap()
            .iter()
            .filter(|(progression, _)| progression.is_none())
        {
            let frame = session.next_frame().unwrap().unwrap();
            assert_eq!(
                &planes::read(&backend, &frame.output().outputs[0]),
                expected,
                "{} final-only output",
                case.name
            );
        }
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
