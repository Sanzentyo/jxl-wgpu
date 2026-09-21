use super::*;
use jxl_wgpu_encode::{
    AnimationHeader, FrameOptions, FrameTiming, LosslessModularAnimationDescriptor,
};
use std::num::{NonZeroU32, NonZeroUsize};

#[test]
fn animation_frames_accept_distinct_layouts_without_changing_samples_or_timing() {
    let rig = Rig::new();
    let extent = Extent2d::new(257, 3);
    let encoder = LosslessModularEncoder::new(rig.context.clone());
    for (kind, bits) in [
        (SampleKind::Unsigned, 8),
        (SampleKind::Unsigned, 31),
        (SampleKind::Float, 16),
        (SampleKind::Float, 32),
    ] {
        let base = Case {
            format: LosslessModularFormat::Rgba,
            bits,
            kind,
            storage: Storage::Planar,
            reversed: true,
            byte_order: ByteOrder::Big,
            shifted: true,
        };
        let timing = AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 2,
            have_timecodes: true,
        };
        let descriptor = if kind == SampleKind::Float {
            LosslessModularAnimationDescriptor::new_float(
                extent.width,
                extent.height,
                base.format,
                bits,
                timing,
            )
        } else {
            LosslessModularAnimationDescriptor::new(
                extent.width,
                extent.height,
                base.format,
                bits,
                timing,
            )
        }
        .unwrap();
        let mut assembly = encoder.begin_animation(descriptor).unwrap();
        let mut pending = Vec::new();
        let mut expected = Vec::new();
        for (index, storage) in [Storage::Planar, Storage::Split, Storage::Packed]
            .into_iter()
            .enumerate()
        {
            let case = Case {
                storage,
                byte_order: if index == 1 {
                    ByteOrder::Little
                } else {
                    ByteOrder::Big
                },
                ..base
            };
            let mut samples = case.samples(extent);
            samples.rotate_left(index * 7);
            let input = upload(&rig.context, &case, extent, &samples, 4099);
            expected.push(samples);
            let options = FrameOptions {
                timing: FrameTiming {
                    duration_ticks: index as u32 + 2,
                    timecode: Some(100 + index as u32),
                },
                ..Default::default()
            };
            pending.push(
                if index == 2 {
                    assembly.submit_last_frame(input, options)
                } else {
                    assembly.submit_frame(input, options)
                }
                .unwrap(),
            );
        }
        for (index, job) in pending.into_iter().rev().enumerate() {
            assembly
                .insert(if index == 1 {
                    job.wait().unwrap()
                } else {
                    pollster::block_on(job).unwrap()
                })
                .unwrap();
        }
        let encoded = assembly.finish_container().unwrap();
        let references: Vec<_> = expected.iter().map(Vec::as_slice).collect();
        let native = check_frame_oracles(&encoded, &references, &base);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
            let request = if kind == SampleKind::Float {
                GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            } else {
                GpuOutputRequest::color(base.format.pixel_format(bits).unwrap()).unwrap()
            }
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
            for (index, frame) in retained.iter().enumerate() {
                let bytes = read_bytes(&rig.backend, &frame.output().outputs[0]);
                if kind == SampleKind::Float {
                    let pixels = expected[index].len() / 4;
                    let start = index * pixels * 5;
                    assert_eq!(
                        bytes,
                        native[start..start + pixels * 4]
                            .iter()
                            .flat_map(|value| value.to_bits().to_le_bytes())
                            .collect::<Vec<_>>()
                    );
                } else {
                    let width = bits.next_power_of_two().max(8) as usize / 8;
                    assert_eq!(
                        bytes,
                        expected[index]
                            .iter()
                            .flat_map(|value| { value.to_le_bytes()[..width].to_vec() })
                            .collect::<Vec<_>>()
                    );
                }
            }
            drop(retained);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
