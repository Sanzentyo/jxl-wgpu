use super::*;
use jxl_wgpu_encode::{
    AnimationHeader, BlendMode, FrameBlend, FrameCrop, FrameOptions, FrameTiming,
    LosslessModularAnimationDescriptor, ReferenceSlot,
};
use std::num::NonZeroU32;

fn samples(case: &Case, extent: Extent2d, frame: usize) -> Vec<u32> {
    let channels = case.format.channel_count() as usize;
    (0..extent.area().unwrap() * channels)
        .map(|index| {
            let value = if index % channels == channels - 1 {
                [0.5f32, 0.25, 0.75][frame % 3]
            } else {
                0.125
                    + ((index / channels + frame * 3) % 7) as f32 / 32.0
                    + (index % channels) as f32 / 16.0
            };
            if case.kind == SampleKind::Unsigned {
                (f64::from(value) * f64::from(u32::MAX >> (32 - case.bits))).round() as u32
            } else if case.bits == 32 {
                value.to_bits()
            } else {
                let bits = value.to_bits();
                assert_eq!(bits & 0x1fff, 0);
                ((bits >> 16) & 0x8000)
                    | ((((bits >> 23) & 255) - 112) << 10)
                    | ((bits >> 13) & 1023)
            }
        })
        .collect()
}

#[test]
fn gray_and_rgb_alpha_compose_all_blends_with_independent_reference_fields() {
    let rig = Rig::new();
    let canvas = Extent2d::new(17, 3);
    let timing = AnimationHeader::Animation {
        ticks_per_second_numerator: NonZeroU32::new(100).unwrap(),
        ticks_per_second_denominator: NonZeroU32::new(1).unwrap(),
        num_loops: 2,
        have_timecodes: true,
    };
    for format in [
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgba,
    ] {
        for association in [AlphaAssociation::Unassociated, AlphaAssociation::Associated] {
            let encoder = LosslessModularEncoder::new(rig.context.clone())
                .with_alpha_association(association);
            for (kind, bits) in [
                (SampleKind::Unsigned, 8),
                (SampleKind::Unsigned, 31),
                (SampleKind::Float, 16),
                (SampleKind::Float, 32),
            ] {
                let case = Case {
                    format,
                    kind,
                    bits,
                    storage: Storage::Planar,
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let descriptor = if kind == SampleKind::Float {
                    LosslessModularAnimationDescriptor::new_float(
                        canvas.width,
                        canvas.height,
                        format,
                        bits,
                        timing,
                    )
                } else {
                    LosslessModularAnimationDescriptor::new(
                        canvas.width,
                        canvas.height,
                        format,
                        bits,
                        timing,
                    )
                }
                .unwrap();
                let mut assembly = encoder.begin_animation(descriptor).unwrap();
                // Partial frames preserve the explicitly selected color/alpha backgrounds;
                // the final full Replace still serializes the alpha Multiply reference.
                let contracts = [
                    (BlendMode::Replace, 0, BlendMode::Replace, 0, None, 1),
                    (
                        BlendMode::Add,
                        1,
                        BlendMode::Replace,
                        1,
                        Some(FrameCrop::new(-2, 1, 13, 1).unwrap()),
                        2,
                    ),
                    (BlendMode::Blend, 2, BlendMode::Blend, 1, None, 3),
                    (BlendMode::Multiply, 1, BlendMode::Add, 2, None, 2),
                    (
                        BlendMode::MultiplyAdd,
                        3,
                        BlendMode::Replace,
                        2,
                        Some(FrameCrop::new(5, 0, 7, 2).unwrap()),
                        1,
                    ),
                    (BlendMode::Replace, 0, BlendMode::Multiply, 1, None, 0),
                ];
                let mut pending = Vec::new();
                for (index, (mode, source, alpha_mode, alpha_source, crop, save)) in
                    contracts.into_iter().enumerate()
                {
                    let extent =
                        crop.map_or(canvas, |crop| Extent2d::new(crop.width(), crop.height()));
                    let input = upload(
                        &rig.context,
                        &Case {
                            storage: if index % 2 == 0 {
                                Storage::Planar
                            } else {
                                Storage::Packed
                            },
                            byte_order: if index % 2 == 0 {
                                ByteOrder::Big
                            } else {
                                ByteOrder::Little
                            },
                            ..case
                        },
                        extent,
                        &samples(&case, extent, index),
                        4099,
                    );
                    let options = FrameOptions {
                        timing: FrameTiming {
                            duration_ticks: index as u32 + 2,
                            timecode: Some(index as u32 + 20),
                        },
                        crop,
                        color_blend: FrameBlend {
                            alpha_channel: 0,
                            mode,
                            source_reference: ReferenceSlot::new(source).unwrap(),
                            clamp: false,
                        },
                        extra_channel_blends: vec![FrameBlend {
                            alpha_channel: 0,
                            mode: alpha_mode,
                            source_reference: ReferenceSlot::new(alpha_source).unwrap(),
                            clamp: false,
                        }],
                        save_as_reference: ReferenceSlot::new(save).unwrap(),
                        ..Default::default()
                    };
                    pending.push(
                        if index == 5 {
                            assembly.submit_last_frame(input, options)
                        } else {
                            assembly.submit_frame(input, options)
                        }
                        .unwrap(),
                    );
                }
                for (index, job) in pending.into_iter().rev().enumerate() {
                    assembly
                        .insert(if index % 2 == 0 {
                            job.wait().unwrap()
                        } else {
                            pollster::block_on(job).unwrap()
                        })
                        .unwrap();
                }
                let data = assembly.finish_container().unwrap();
                assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                let native_preserved = native(&data);
                let native_straight =
                    extra_channels::libjxl_output(&data, &["--original", "--keep-orientation"])
                        .expect("required native composed output");
                let rust = extra_channels::rust_frame_planes(&data);
                let pixels = canvas.area().unwrap();
                assert_eq!(native_preserved.len(), 6 * pixels * 5);
                assert_eq!(native_straight.len(), 6 * pixels * 5);
                assert_eq!(rust.len(), 6);
                for policy in [
                    AlphaOutputPolicy::Preserve,
                    AlphaOutputPolicy::Unassociated,
                    AlphaOutputPolicy::Associated,
                ] {
                    let gpu = output::color_frames(&rig, &data, policy, 6);
                    for (index, frame) in gpu.iter().enumerate() {
                        let preserved = &native_preserved[index * pixels * 5..][..pixels * 4];
                        let mut expected = preserved.to_vec();
                        for pixel in expected.as_chunks_mut::<4>().0 {
                            let alpha = pixel[3].max(2.0f32.powi(-26));
                            let factor = match (association, policy) {
                                (AlphaAssociation::Associated, AlphaOutputPolicy::Unassociated) => {
                                    1.0 / alpha
                                }
                                (AlphaAssociation::Unassociated, AlphaOutputPolicy::Associated) => {
                                    alpha
                                }
                                _ => 1.0,
                            };
                            for color in &mut pixel[..3] {
                                *color *= factor;
                            }
                        }
                        let unpremultiplied = association == AlphaAssociation::Associated
                            && policy == AlphaOutputPolicy::Unassociated;
                        output::compare(frame, &expected, unpremultiplied);
                        // Rust jxl 0.6 preserves source-associated color even when its
                        // premultiply_output option is false; libjxl explicitly unpremultiplies.
                        if policy == AlphaOutputPolicy::Preserve {
                            output::compare(frame, &rust[index].0, false);
                        }
                        if policy == AlphaOutputPolicy::Unassociated {
                            output::compare(
                                frame,
                                &native_straight[index * pixels * 5..][..pixels * 4],
                                unpremultiplied,
                            );
                        }
                    }
                }
            }
        }
    }
}
