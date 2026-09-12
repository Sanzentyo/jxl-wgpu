use super::*;
use jxl_wgpu_decode::SpotColorPolicy;

fn unpack(layout: &ImageLayout, bytes: &[u8], planar: bool) -> Vec<f32> {
    let width = layout.extent.width as usize;
    (0..layout.extent.area().unwrap())
        .flat_map(|pixel| {
            (0..4).map(move |c| {
                let stored = if planar && c < 3 { 2 - c } else { c };
                let plane = &layout.planes[if planar { stored } else { 0 }];
                let offset = plane.offset as usize
                    + pixel / width * plane.row_stride as usize
                    + pixel % width * if planar { 4 } else { 16 }
                    + if planar { 0 } else { stored * 4 };
                f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
            })
        })
        .collect()
}

fn compare_output(
    actual: &[f32],
    expected: &[u8],
    associated: bool,
    policy: AlphaOutputPolicy,
    color_limit: f32,
) {
    assert_eq!(actual.len() * 4, expected.len());
    for (actual, expected) in actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<16>().0.iter())
    {
        let expected: Vec<_> = expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let alpha = expected[3].max(1.0 / 67108864.0);
        let factor = match (associated, policy) {
            (true, AlphaOutputPolicy::Unassociated) => 1.0 / alpha,
            (false, AlphaOutputPolicy::Associated) => alpha,
            _ => 1.0,
        };
        for c in 0..4 {
            let reference = expected[c] * if c == 3 { 1.0 } else { factor };
            assert!(actual[c].is_finite() && reference.is_finite());
            let scale = if c != 3 && associated && policy == AlphaOutputPolicy::Unassociated {
                alpha
            } else {
                1.0
            };
            let error = (actual[c] - reference).abs() * scale / (reference * scale).abs().max(1.0);
            let limit = if c == 3 { 4e-7 } else { color_limit };
            assert!(
                error < limit,
                "channel {c}: {} vs {reference}, error {error} >= {limit}",
                actual[c]
            );
        }
    }
}

#[test]
fn extra_updates_apply_orientation_alpha_spots_and_target_format_at_every_stage() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for name in [
        "vardct_extras_rgba_progressive",
        "vardct_extras_associated_squeeze",
        "floating/vardct_extras_float_squeeze",
        "vardct_extras_associated_thin",
    ] {
        let data = fixture(name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let frame = &inventory.frames[0];
        let associated = inventory
            .image_header
            .extra_channels
            .iter()
            .find_map(|e| match e.channel_type {
                jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated } => {
                    Some(associated)
                }
                _ => None,
            })
            .unwrap();
        for keep in [false, true] {
            let linear = keep;
            let spots = !keep;
            let Some(mut expected) = prefix_images_options(&data, frame, keep, linear, spots)
            else {
                return;
            };
            let Some(native) = native_updates_with_spots(&data, linear, keep, false, spots) else {
                return;
            };
            expected.push(Some(native.last().unwrap().pixels.clone()));
            let mut color = vardct_rgb8_format().color_spec;
            if linear {
                let jxl_gpu_formats::ColorSpecification::Defined(ref mut spec) = color else {
                    unreachable!()
                };
                spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
            }
            let format = PixelFormat::rgb_f32(
                if keep {
                    jxl_gpu_formats::RgbChannelOrder::Bgra
                } else {
                    jxl_gpu_formats::RgbChannelOrder::Rgba
                },
                keep,
                color,
            );
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
            );
            for policy in [
                AlphaOutputPolicy::Preserve,
                AlphaOutputPolicy::Associated,
                AlphaOutputPolicy::Unassociated,
            ] {
                eprintln!("{name}: keep{keep}, {policy:?}");
                let request = GpuOutputRequest::color(format.clone())
                    .unwrap()
                    .with_progressive_output(true)
                    .with_alpha_output_policy(policy)
                    .with_orientation_policy(if keep {
                        OrientationPolicy::Keep
                    } else {
                        OrientationPolicy::Apply
                    })
                    .with_spot_color_policy(if spots {
                        SpotColorPolicy::Render
                    } else {
                        SpotColorPolicy::Preserve
                    });
                let mut session = open_incremental(&decoder, &data, request.clone());
                let mut held = Vec::new();
                let mut pixels = Vec::new();
                while let Some(update) = session.next_update().unwrap() {
                    let stage = pixels.len();
                    let actual = read(&backend, update.output());
                    if let Some(expected) = &expected[stage] {
                        let values = unpack(&update.output().outputs[0].layout, &actual, keep);
                        compare_output(
                            &values,
                            expected,
                            associated,
                            policy,
                            if stage == 0 {
                                2e-5
                            } else if linear {
                                1e-4
                            } else {
                                6e-4
                            },
                        );
                    }
                    held.push(update);
                    pixels.push(actual);
                }
                assert_eq!(pixels.len(), expected.len());
                let mut final_only = decoder
                    .open(&data, request.with_progressive_output(false))
                    .unwrap();
                let final_frame = final_only.next_frame().unwrap().unwrap();
                assert_eq!(
                    read(&backend, final_frame.output()),
                    *pixels.last().unwrap()
                );
                for (update, bytes) in held.iter().zip(&pixels) {
                    assert_eq!(update.metadata, final_frame.metadata);
                    assert_eq!(read(&backend, update.output()), *bytes);
                }
                drop(final_frame);
                drop(final_only);
                drop(held);
                drop(session);
                drain_gpu(&backend, 0);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}
