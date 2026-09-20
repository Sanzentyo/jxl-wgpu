use super::*;
use std::num::{NonZeroU32, NonZeroUsize};

use jxl_wgpu_encode::{
    AnimationHeader, FrameOptions, FrameTiming, LosslessModularAnimationDescriptor,
};

#[test]
fn wide_animation_retains_exact_replace_frames_and_timing() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
        .expect("required GPU adapter");
    let context = WgpuContext::from_backend(&backend);
    let encoder = LosslessModularEncoder::new(context.clone());
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let extent = Extent2d::new(257, 3);
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        for bits in [17, 24, 31] {
            let descriptor = LosslessModularAnimationDescriptor::new(
                extent.width,
                extent.height,
                format,
                bits,
                AnimationHeader::Animation {
                    ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
                    ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
                    num_loops: 2,
                    have_timecodes: true,
                },
            )
            .unwrap();
            let mut assembly = encoder.begin_animation(descriptor).unwrap();
            let mut pending = Vec::new();
            let mut expected = Vec::new();
            for index in 0..3 {
                let (source, samples) = source(&context, extent, format, bits, index);
                expected.push(samples);
                let options = FrameOptions {
                    timing: FrameTiming {
                        duration_ticks: index + 2,
                        timecode: Some(100 + index),
                    },
                    ..Default::default()
                };
                pending.push(if index == 2 {
                    assembly.submit_last_frame(source, options).unwrap()
                } else {
                    assembly.submit_frame(source, options).unwrap()
                });
            }
            // Completion and insertion order need not match presentation order.
            for (index, job) in pending.into_iter().rev().enumerate() {
                let frame = if index == 1 {
                    job.wait().unwrap()
                } else {
                    pollster::block_on(job).unwrap()
                };
                assembly.insert(frame).unwrap();
            }
            let encoded = assembly.finish_container().unwrap();
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
            let references: Vec<_> = expected.iter().map(Vec::as_slice).collect();
            check_frame_oracles(&encoded, &references, format, bits);
            for (decoder, fragmented) in [(&whole, false), (&bounded, true)] {
                let pixel_format = format.pixel_format(bits).unwrap();
                let request = if format == LosslessModularFormat::Gray {
                    GpuOutputRequest::numeric(pixel_format, NumericSampleMapping::NativeUnsigned)
                } else {
                    GpuOutputRequest::color(pixel_format)
                }
                .unwrap()
                .with_max_frame_slots(NonZeroUsize::new(4).unwrap());
                let mut session = if fragmented {
                    open_fragmented(decoder, &encoded, request)
                } else {
                    decoder.open(&encoded, request).unwrap()
                };
                assert_eq!(session.metadata().loop_count, Some(2));
                let mut retained = Vec::new();
                let mut ticks = 0;
                for index in 0..3 {
                    let frame = pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .unwrap();
                    assert_eq!(frame.metadata.index, index);
                    assert_eq!(frame.metadata.duration.ticks, index as u32 + 2);
                    assert_eq!(
                        frame.metadata.duration.as_seconds(),
                        (index + 2) as f64 / 100.0
                    );
                    assert_eq!(frame.metadata.timecode, Some(100 + index as u32));
                    assert_eq!(frame.metadata.presentation_ticks, ticks);
                    assert_eq!(frame.metadata.is_last, index == 2);
                    ticks += index as u64 + 2;
                    retained.push(frame);
                }
                assert!(
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .is_none()
                );
                drop(session);
                for (frame, expected) in retained.iter().zip(&expected) {
                    let actual = read_bytes(&backend, &frame.output().outputs[0]);
                    let words: Vec<_> = actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| u32::from_le_bytes(*v))
                        .collect();
                    assert_eq!(
                        &words, expected,
                        "GPU animation {format:?}/{bits}, bounded={fragmented}"
                    );
                }
                drop(retained);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
