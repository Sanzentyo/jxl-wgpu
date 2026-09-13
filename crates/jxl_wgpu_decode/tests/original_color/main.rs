#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::FrameEncoding;
use jxl_test_support::{fixtures::original_color as corpus, gpu::planes};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, OrientationPolicy, WgpuDecodeEngine,
};
use std::num::{NonZeroU64, NonZeroUsize};

mod numeric;
mod oracle;
mod output;

fn tolerance(case: &corpus::Case) -> f32 {
    if case.mode.encoding() == FrameEncoding::Modular && !case.mode.xyb() {
        1e-5
    } else {
        1.0 / 1024.0
    }
}

fn compare(actual: &[u32], expected: &[f32], tolerance: f32, name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let mut maximum = 0f32;
    for (index, (&word, &reference)) in actual.iter().zip(expected).enumerate() {
        let actual = f32::from_bits(word);
        let error = (actual - reference).abs() / (1.0 + reference.abs());
        let bound = if index % 4 == 3 { 2e-6 } else { tolerance };
        assert!(
            actual.is_finite() && reference.is_finite() && error <= bound,
            "{name}/{index}: GPU {actual}, native {reference}, error {error}, bound {bound}"
        );
        maximum = maximum.max(error);
    }
    eprintln!("{name}: max normalized original-color error {maximum}");
}

#[test]
fn original_sdr_profiles_preserve_color_and_progressive_reference_frames() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for case in corpus::cases() {
        eprintln!("original profile {}", case.name);
        let data = case.bytes();
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        case.validate(&inventory);
        if case.mode.ycbcr() {
            assert_eq!(case.encode_ycbcr(), data);
        }
        let reference = case.reference();
        let tolerance = tolerance(&case);
        let mut baseline = None;
        for limit in [None, NonZeroU64::new(256)] {
            let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
            if let Some(limit) = limit {
                engine = engine.with_stream_window_limit(limit);
            }
            let decoder = GpuDecoder::new(engine);
            let request = GpuOutputRequest::color(case.format())
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                .with_orientation_policy(OrientationPolicy::Keep)
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
            let mut snapshots = Vec::new();
            let mut images = Vec::new();
            let mut final_count = 0;
            while let Some(update) = if limit.is_some() {
                pollster::block_on(session.next_update_async()).unwrap()
            } else {
                session.next_update().unwrap()
            } {
                let words = planes::read(&backend, &update.output().outputs[0]);
                if update.progression().is_none() {
                    compare(
                        &words,
                        &reference[final_count * 37 * 19 * 4..(final_count + 1) * 37 * 19 * 4],
                        tolerance,
                        &case.name,
                    );
                    final_count += 1;
                }
                snapshots.push((update.progression(), words));
                images.push(update);
            }
            assert_eq!(final_count, if case.sequence { 4 } else { 1 });
            for (image, (_, words)) in images.iter().zip(&snapshots) {
                assert_eq!(
                    &planes::read(&backend, &image.output().outputs[0]),
                    words,
                    "retained {}",
                    case.name
                );
            }
            if let Some(baseline) = &baseline {
                assert_eq!(baseline, &snapshots, "fragmented {}", case.name);
            } else {
                baseline = Some(snapshots);
            }
            drop((images, session));
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let request = GpuOutputRequest::color(case.format())
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_orientation_policy(OrientationPolicy::Keep);
        let mut session = decoder.open(&data, request).unwrap();
        for (_, words) in baseline
            .unwrap()
            .iter()
            .filter(|(progression, _)| progression.is_none())
        {
            let update = session.next_frame().unwrap().unwrap();
            assert_eq!(
                &planes::read(&backend, &update.output().outputs[0]),
                words,
                "final-only {}",
                case.name
            );
        }
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
