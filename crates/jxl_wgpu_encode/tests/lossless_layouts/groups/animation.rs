use super::*;
use jxl_wgpu_encode::{
    AnimationHeader, FrameOptions, FrameTiming, LosslessModularAnimationDescriptor,
};
use std::num::NonZeroU32;

#[test]
fn cropped_frames_use_their_own_group_grid_across_reference_composition() {
    let rig = Rig::new();
    for size in LosslessModularGroupSize::ALL {
        for tree in TREES {
            let encoder = encoder(&rig, size, tree);
            check_cropped_frames(&rig, &encoder, size);
        }
    }
}

pub(crate) fn check_cropped_frames(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    size: LosslessModularGroupSize,
) {
    use jxl_wgpu_encode::{BlendMode, FrameBlend, FrameCrop, ReferenceSlot};
    let edge = size.dimension();
    let canvas = Extent2d::new(8 * edge + 17, 3);
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 32,
        kind: SampleKind::Float,
        storage: Storage::Split,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let descriptor = LosslessModularAnimationDescriptor::new_float(
        canvas.width,
        canvas.height,
        case.format,
        32,
        AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 2,
            have_timecodes: true,
        },
    )
    .unwrap();
    let mut animation = encoder.begin_animation(descriptor).unwrap();
    let mut jobs = Vec::new();
    let mut extents = Vec::new();
    for (index, (crop, mode, extra_mode, reference, save)) in [
        (None, BlendMode::Replace, BlendMode::Replace, 0, 1),
        (
            Some(FrameCrop::new(-3, 1, edge - 1, 2).unwrap()),
            BlendMode::Add,
            BlendMode::Replace,
            1,
            2,
        ),
        (
            Some(FrameCrop::new(edge as i32 - 1, -1, edge + 1, 3).unwrap()),
            BlendMode::Multiply,
            BlendMode::Add,
            2,
            0,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let extent = crop.map_or(canvas, |crop| Extent2d::new(crop.width(), crop.height()));
        extents.push(extent);
        let samples: Vec<_> = (0..extent.area().unwrap() * 4)
            .map(|component| {
                if component % 4 == 3 {
                    0.5f32.to_bits()
                } else {
                    (0.125 + ((component + index * 7) % 9) as f32 / 32.0).to_bits()
                }
            })
            .collect();
        let source = upload(&rig.context, &case, extent, &samples, 4099);
        let options = FrameOptions {
            timing: FrameTiming {
                duration_ticks: index as u32 + 2,
                timecode: Some(index as u32 + 20),
            },
            crop,
            color_blend: FrameBlend {
                mode,
                source_reference: ReferenceSlot::new(reference).unwrap(),
                clamp: false,
            },
            extra_channel_blends: vec![FrameBlend {
                mode: extra_mode,
                source_reference: ReferenceSlot::new(u8::from(index != 0)).unwrap(),
                clamp: false,
            }],
            save_as_reference: ReferenceSlot::new(save).unwrap(),
            ..Default::default()
        };
        jobs.push(
            if index == 2 {
                animation.submit_last_frame(source, options)
            } else {
                animation.submit_frame(source, options)
            }
            .unwrap(),
        );
    }
    for job in jobs.into_iter().rev() {
        animation.insert(pollster::block_on(job).unwrap()).unwrap();
    }
    let encoded = animation.finish_container().unwrap();
    check_header(&encoded, size, &extents);
    let native = native(&encoded);
    let rust = extra_channels::rust_frame_planes(&encoded);
    assert_eq!(rust.len(), 3);
    assert_eq!(native.len(), canvas.area().unwrap() * 5 * 3);
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
    .with_max_frame_slots(std::num::NonZeroUsize::new(4).unwrap());
    let mut baseline = None;
    for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
        let mut session = if fragmented {
            open_fragmented(decoder, &encoded, request.clone())
        } else {
            decoder.open(&encoded, request.clone()).unwrap()
        };
        let mut frames = Vec::new();
        while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
            frames.push(frame);
        }
        assert_eq!(frames.len(), 3);
        drop(session);
        let mut actual_frames = Vec::new();
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.metadata.index, index);
            assert_eq!(frame.metadata.duration.ticks, index as u32 + 2);
            assert_eq!(frame.metadata.timecode, Some(index as u32 + 20));
            let words = read(&rig.backend, &frame.output().outputs[0]);
            let native =
                &native[index * canvas.area().unwrap() * 5..][..canvas.area().unwrap() * 4];
            assert_eq!(words.len(), native.len());
            for (component, word) in words.iter().enumerate() {
                let value = f32::from_bits(*word);
                let bound = if component % 4 == 3 { 4e-7 } else { 3e-6 };
                for expected in [native[component], rust[index].0[component]] {
                    assert!(
                        value.is_finite()
                            && expected.is_finite()
                            && (value - expected).abs() / expected.abs().max(1.0) < bound,
                        "{size:?}, frame {index}, component {component}: {value} vs {expected}"
                    );
                }
            }
            actual_frames.push(words);
        }
        if let Some(baseline) = &baseline {
            assert_eq!(&actual_frames, baseline);
        } else {
            baseline = Some(actual_frames);
        }
        drop(frames);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn every_group_size_keeps_animation_words_timing_and_retained_frames() {
    let rig = Rig::new();
    for size in LosslessModularGroupSize::ALL {
        let encoder = encoder(&rig, size, TREES[1]);
        for (format, kind, bits) in [
            (LosslessModularFormat::Rgba, SampleKind::Unsigned, 31),
            (LosslessModularFormat::GrayAlpha, SampleKind::Float, 32),
        ] {
            check_animation_words(&rig, &encoder, size, format, kind, bits);
        }
    }
}

pub(crate) fn check_animation_words(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    size: LosslessModularGroupSize,
    format: LosslessModularFormat,
    kind: SampleKind,
    bits: u8,
) {
    check_animation_words_with_oracle(rig, encoder, size, format, kind, bits, check_frame_oracles);
}

pub(crate) fn check_animation_words_with_oracle(
    rig: &Rig,
    encoder: &LosslessModularEncoder,
    size: LosslessModularGroupSize,
    format: LosslessModularFormat,
    kind: SampleKind,
    bits: u8,
    oracle: FrameOracle,
) {
    let case = Case {
        format,
        bits,
        kind,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let extent = Extent2d::new(size.dimension() + 1, 3);
    let template = upload(&rig.context, &case, extent, &case.samples(extent), 4099);
    let descriptor = LosslessModularAnimationDescriptor::from_pixel_format(
        extent.width,
        extent.height,
        &template.layout.format,
        AnimationHeader::Animation {
            ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
            ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
            num_loops: 2,
            have_timecodes: true,
        },
    )
    .unwrap();
    let mut animation = encoder.begin_animation(descriptor).unwrap();
    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for index in 0..3 {
        let mut samples = case.samples(extent);
        samples.rotate_left(index * format.channel_count() as usize);
        let source = upload(
            &rig.context,
            &Case {
                storage: [Storage::Packed, Storage::Planar, Storage::Split][index],
                ..case
            },
            extent,
            &samples,
            8195,
        );
        let options = FrameOptions {
            timing: FrameTiming {
                duration_ticks: index as u32 + 2,
                timecode: Some(index as u32 + 20),
            },
            ..Default::default()
        };
        jobs.push(
            if index == 2 {
                animation.submit_last_frame(source, options)
            } else {
                animation.submit_frame(source, options)
            }
            .unwrap(),
        );
        expected.push(samples);
    }
    for (index, job) in jobs.into_iter().rev().enumerate() {
        animation
            .insert(if index == 1 {
                job.wait().unwrap()
            } else {
                pollster::block_on(job).unwrap()
            })
            .unwrap();
    }
    let encoded = animation.finish_container().unwrap();
    check_header(&encoded, size, &[extent; 3]);
    oracle(
        &encoded,
        &expected.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        &case,
    );
    color::check_numeric(rig, &encoded, &expected, &case);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}
