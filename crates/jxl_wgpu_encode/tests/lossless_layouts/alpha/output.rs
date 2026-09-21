use super::*;

pub(super) fn color_frames(
    rig: &Rig,
    data: &[u8],
    policy: AlphaOutputPolicy,
    frame_count: usize,
) -> Vec<Vec<f32>> {
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(policy)
    .with_max_frame_slots(std::num::NonZeroUsize::new(frame_count + 1).unwrap());
    let mut whole = None;
    for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
        let mut session = if fragmented {
            open_fragmented(decoder, data, request.clone())
        } else {
            decoder.open(data, request.clone()).unwrap()
        };
        let mut frames = Vec::new();
        while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
            assert_eq!(frame.metadata.index, frames.len());
            assert_eq!(frame.metadata.is_last, frames.len() + 1 == frame_count);
            if frame_count > 1 {
                let index = frames.len();
                assert_eq!(frame.metadata.duration.ticks, index as u32 + 2);
                assert_eq!(frame.metadata.timecode, Some(index as u32 + 20));
                assert_eq!(
                    frame.metadata.presentation_ticks,
                    (0..index as u64).map(|index| index + 2).sum::<u64>()
                );
            }
            frames.push(frame);
        }
        assert_eq!(frames.len(), frame_count);
        drop(session);
        let bits: Vec<_> = frames
            .iter()
            .map(|frame| read(&rig.backend, &frame.output().outputs[0]))
            .collect();
        drop(frames);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        if let Some(whole) = &whole {
            assert_eq!(&bits, whole);
        } else {
            whole = Some(bits);
        }
    }
    whole
        .unwrap()
        .into_iter()
        .map(|frame| frame.into_iter().map(f32::from_bits).collect())
        .collect()
}

// Comparison in the stored-color scale keeps the existing alpha floor and precision contract.
// Exact numeric checks separately cover every source word, including invisible colors.
#[track_caller]
pub(super) fn compare(actual: &[f32], expected: &[f32], unpremultiplied: bool) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected_value.is_finite());
        let scale = if unpremultiplied && index % 4 != 3 {
            expected[index / 4 * 4 + 3].max(2.0f32.powi(-26))
        } else {
            1.0
        };
        let error =
            (actual - expected_value).abs() * scale / (expected_value * scale).abs().max(1.0);
        let bound = if index % 4 == 3 { 4e-7 } else { 3e-6 };
        assert!(
            error < bound,
            "component {index}: {actual} vs {expected_value}, scaled error {error}"
        );
    }
}

fn finite_samples(case: &Case, extent: Extent2d) -> Vec<u32> {
    let channels = case.format.channel_count();
    let mask = u32::MAX >> (32 - case.bits);
    (0..extent.width * extent.height * channels)
        .map(|index| {
            let alpha = index % channels == channels - 1;
            if case.kind == SampleKind::Unsigned {
                let values = [0, 1, 2, mask / 4, mask / 2, mask - 1, mask];
                values[(index as usize + usize::from(!alpha) * 3) % values.len()]
            } else if case.bits == 16 {
                // These all have explicit source/reference pairs in the shared test source.
                [0, 1, 0x400, 0x3c00][(index as usize + usize::from(!alpha)) % 4]
            } else if alpha {
                [
                    0.0f32,
                    2.0f32.powi(-30),
                    2.0f32.powi(-26),
                    2.0f32.powi(-25),
                    0.25,
                    0.5,
                    1.0,
                ][(index / channels) as usize % 7]
                    .to_bits()
            } else {
                [0.0f32, -0.25, 0.125, 0.5, 0.875, 1.0][index as usize % 6].to_bits()
            }
        })
        .collect()
}

#[test]
fn preserved_premultiplied_and_straight_presentations_match_native_and_source_arithmetic() {
    let rig = Rig::new();
    let extent = Extent2d::new(257, 3);
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
                    bits,
                    kind,
                    storage: Storage::Planar,
                    reversed: true,
                    byte_order: ByteOrder::Big,
                    shifted: true,
                };
                let expected = finite_samples(&case, extent);
                let input = upload(&rig.context, &case, extent, &expected, 4099);
                let data = encoder.encode(input).unwrap();
                let original = check_oracles(&data, &expected, &case);
                let pixels = extent.area().unwrap();
                let original = &original[..pixels * 4];
                let native_straight =
                    extra_channels::libjxl_output(&data, &["--original", "--keep-orientation"])
                        .expect("required native unassociated output");
                for policy in [
                    AlphaOutputPolicy::Preserve,
                    AlphaOutputPolicy::Unassociated,
                    AlphaOutputPolicy::Associated,
                ] {
                    let mut reference = original.to_vec();
                    for pixel in reference.as_chunks_mut::<4>().0 {
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
                    let output = color_frames(&rig, &data, policy, 1);
                    let unpremultiplied = association == AlphaAssociation::Associated
                        && policy == AlphaOutputPolicy::Unassociated;
                    compare(&output[0], &reference, unpremultiplied);
                    if policy == AlphaOutputPolicy::Unassociated {
                        compare(&output[0], &native_straight[..pixels * 4], unpremultiplied);
                    }
                }
                color::check_numeric(&rig, &data, &[expected], &case);
            }
        }
    }
}
