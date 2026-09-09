use super::*;
#[path = "sequence_lf.rs"]
mod lf;
#[path = "sequence_lifecycle.rs"]
mod lifecycle;
use jxl_gpu_bitstream::{FrameEncoding, FrameType};
use jxl_wgpu_decode::{FrameExecutionPlan, FrameProgression, OrientationPolicy};

fn cases() -> [(&'static str, &'static str); 7] {
    [
        (
            "rgb",
            include_str!("../../../test-data/sequence_vardct_rgb.jxl.hex"),
        ),
        (
            "gray",
            include_str!("../../../test-data/sequence_vardct_gray.jxl.hex"),
        ),
        (
            "lf",
            include_str!("../../../test-data/sequence_vardct_dc.jxl.hex"),
        ),
        (
            "mixed",
            include_str!("../../../test-data/sequence_mixed_jpeg_modular.jxl.hex"),
        ),
        (
            "composed",
            include_str!("../../../test-data/composition_vardct.jxl.hex"),
        ),
        (
            "composed_gray",
            include_str!("../../../test-data/composition_vardct_gray.jxl.hex"),
        ),
        (
            "composed_lf",
            include_str!("../../../test-data/composition_vardct_dc.jxl.hex"),
        ),
    ]
}

pub(super) fn request(keep: bool) -> GpuOutputRequest {
    let mut format = PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        vardct_rgb8_format().color_spec,
    );
    let jxl_gpu_formats::ColorSpecification::Defined(ref mut color) = format.color_spec else {
        unreachable!()
    };
    color.transfer = jxl_gpu_formats::TransferFunction::Linear;
    GpuOutputRequest::color(format)
        .unwrap()
        .with_progressive_output(true)
        .with_max_frame_slots(NonZeroUsize::new(3).unwrap())
        .with_orientation_policy(if keep {
            OrientationPolicy::Keep
        } else {
            OrientationPolicy::Apply
        })
}

pub(super) fn relative_error(actual: &[u8], expected: &[u8]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .map(|(a, b)| {
            let a = f32::from_le_bytes(a.try_into().unwrap());
            let b = f32::from_le_bytes(b.try_into().unwrap());
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs() / b.abs().max(1.0)
        })
        .fold(0.0, f32::max)
}

#[test]
fn animated_and_composed_coefficient_updates_preserve_presentations_and_match_libjxl() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, hex) in cases() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert!(inventory.image_header.animation.is_some());
        assert!(inventory.image_header.extra_channels.is_empty());
        for keep in [false, true] {
            let request = request(keep);
            let plan = FrameExecutionPlan::negotiate_with_orientation(
                &inventory,
                request.orientation_policy(),
            )
            .unwrap();
            let native = if name.starts_with("composed") {
                sequence_oracle::composed(&data, &inventory, name, keep)
            } else {
                native_updates_oriented(&data, true, keep)
            };
            let mut whole = None;
            for cap in [u64::MAX, 40] {
                let decoder = GpuDecoder::new(
                    WgpuDecodeEngine::new(backend.clone())
                        .unwrap()
                        .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
                );
                let mut session = if cap == u64::MAX {
                    decoder.open(&data, request.clone()).unwrap()
                } else {
                    open_incremental(&decoder, &data, request.clone())
                };
                session.prefetch(session.resolved_frame_slots()).unwrap();
                let mut presentation = 0;
                let mut coefficients = 0;
                let mut native_index = 0;
                let mut held = Vec::new();
                let mut pixels = Vec::new();
                let mut finals = Vec::new();
                while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                    let planned = &plan.presentations[presentation];
                    let physical = &inventory.frames[planned.physical_frames.end - 1];
                    assert_eq!(update.metadata, planned.metadata, "{name}");
                    assert_eq!(
                        update.output().outputs[0].layout.extent,
                        plan.metadata.extent
                    );
                    let actual = read(&backend, update.output());
                    assert!(
                        actual
                            .chunks_exact(4)
                            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                    );
                    eprintln!(
                        "{name} keep{keep} cap{cap} presentation{presentation}: {:?}",
                        update.progression()
                    );
                    let lf = matches!(
                        update.progression(),
                        Some(FrameProgression::LowFrequency { .. })
                    );
                    if let Some(FrameProgression::Coefficients {
                        physical_frame_index,
                        completed_passes,
                        total_passes,
                        ..
                    }) = update.progression()
                    {
                        assert_eq!(physical_frame_index, physical.frame_index);
                        assert_eq!(u32::from(total_passes), physical.num_passes);
                        assert_eq!(usize::from(completed_passes), coefficients);
                        coefficients += 1;
                    }
                    if let Some(native) = &native
                        && !lf
                    {
                        let expected = &native[native_index];
                        assert_eq!(
                            expected.frame, presentation,
                            "{name}: native step {}",
                            expected.step
                        );
                        assert_eq!(
                            expected.complete,
                            update.is_complete(),
                            "{name}: native step {}",
                            expected.step
                        );
                        assert_eq!(expected.duration, update.metadata.duration.ticks);
                        assert_eq!(expected.timecode, update.metadata.timecode.unwrap_or(0));
                        assert_eq!(expected.is_last, update.metadata.is_last);
                        if let Some(progression) = update.progression() {
                            assert_eq!(progression.intended_downsampling(), expected.ratio);
                        }
                        let error = relative_error(&actual, &expected.pixels);
                        eprintln!(
                            "{name} presentation{presentation} step{} native linear error {error}",
                            expected.step
                        );
                        let limit = if name.starts_with("composed") {
                            // Extended sRGB reference composition amplifies inverse-transform
                            // differences also present in the unchanged final-only path.
                            1e-3
                        } else if expected.step == 0 && !expected.complete {
                            1e-5
                        } else {
                            2e-4
                        };
                        assert!(
                            error < limit,
                            "{name}: presentation{presentation} step{} error{error}",
                            expected.step
                        );
                        native_index += 1;
                    }
                    if update.is_complete() {
                        assert_eq!(
                            coefficients,
                            if physical.encoding == FrameEncoding::VarDct
                                && physical.frame_type == FrameType::Regular
                            {
                                physical.num_passes as usize
                            } else {
                                0
                            },
                            "{name}: presentation{presentation}"
                        );
                        coefficients = 0;
                        presentation += 1;
                        finals.push(actual.clone());
                    }
                    // Retain byte ownership while releasing the presentation's frame-slot lease.
                    held.push(owned(update.output()));
                    pixels.push(actual);
                }
                assert_eq!(presentation, plan.presentations.len());
                if let Some(native) = &native {
                    assert_eq!(native_index, native.len(), "{name}");
                }
                assert_eq!(session.frames_submitted(), plan.presentations.len());
                for (image, expected) in held.iter().zip(&pixels) {
                    assert_eq!(&read(&backend, image), expected, "{name}: immutable image");
                }
                let mut final_only = decoder
                    .open(&data, request.clone().with_progressive_output(false))
                    .unwrap();
                for expected in &finals {
                    let frame = final_only.next_frame().unwrap().unwrap();
                    assert_eq!(
                        &read(&backend, frame.output()),
                        expected,
                        "{name}: final-only pixels"
                    );
                }
                assert!(final_only.next_frame().unwrap().is_none());
                if let Some(whole) = &whole {
                    assert_eq!(&pixels, whole, "{name}: input windows");
                } else {
                    whole = Some(pixels);
                }
                drop(final_only);
                drop(session);
                drop(held);
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
