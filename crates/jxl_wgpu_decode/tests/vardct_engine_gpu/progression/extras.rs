//! libjxl disables progression events with extra channels. Its public prefix flush still
//! provides an independent oracle at each physical pass boundary, with original alpha association.
use super::*;
use jxl_wgpu_decode::{AlphaOutputPolicy, FrameProgression, OrientationPolicy};
mod composition;
mod lifecycle;
mod output;

fn fixture(name: &str) -> Vec<u8> {
    encoded(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("test-data")
                .join(format!("{name}.jxl.hex")),
        )
        .unwrap(),
    )
}

fn request(keep: bool, policy: AlphaOutputPolicy) -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_progressive_output(true)
    .with_max_frame_slots(NonZeroUsize::new(1).unwrap())
    .with_alpha_output_policy(policy)
    .with_orientation_policy(if keep {
        OrientationPolicy::Keep
    } else {
        OrientationPolicy::Apply
    })
}

fn prefix_images(
    data: &[u8],
    frame: &jxl_gpu_bitstream::FrameInventory,
    keep: bool,
) -> Option<Vec<Option<Vec<u8>>>> {
    prefix_images_options(data, frame, keep, false, true)
}

fn prefix_images_options(
    data: &[u8],
    frame: &jxl_gpu_bitstream::FrameInventory,
    keep: bool,
    linear: bool,
    spots: bool,
) -> Option<Vec<Option<Vec<u8>>>> {
    // Inventory offsets address the logical codestream, including when the fixture is boxed.
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let data = parsed.codestream();
    (0..frame.num_passes)
        .map(|completed| {
            let end = frame
                .sections
                .iter()
                .filter_map(|section| match section.kind {
                    jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. }
                        if pass_index >= completed =>
                    {
                        Some(section.bytes.offset as usize)
                    }
                    _ => None,
                })
                .min();
            // A single TOC entry has no byte-aligned section boundary to flush independently.
            let Some(end) = end else { return Some(None) };
            let updates = native_updates_with_spots(&data[..end], linear, keep, true, spots)?;
            let last = updates.last().expect("a flushed prefix image");
            assert!(!last.complete);
            Some(Some(last.pixels.clone()))
        })
        .collect()
}

fn compare(name: &str, actual: &[u8], expected: &[u8], completed: usize) {
    assert_eq!(actual.len(), expected.len());
    let mut max = [0_f32; 4];
    for (a, b) in actual
        .as_chunks::<16>()
        .0
        .iter()
        .zip(expected.as_chunks::<16>().0.iter())
    {
        for c in 0..4 {
            let a = f32::from_le_bytes(a[c * 4..c * 4 + 4].try_into().unwrap());
            let b = f32::from_le_bytes(b[c * 4..c * 4 + 4].try_into().unwrap());
            assert!(
                a.is_finite() && b.is_finite(),
                "{name}: nonfinite channel {c}"
            );
            max[c] = max[c].max((a - b).abs() / b.abs().max(1.0));
        }
    }
    let color_limit = if completed == 0 { 2e-5 } else { 6e-4 };
    assert!(
        max[..3].iter().all(|&v| v < color_limit),
        "{name} pass{completed}: {max:?}"
    );
    assert!(
        max[3] < 4e-7,
        "{name} pass{completed}: alpha error {}",
        max[3]
    );
    eprintln!("{name} pass{completed}: {max:?}");
}

#[test]
fn extra_pass_images_match_native_prefixes_without_mutating_later_reconstruction() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for name in [
        "vardct_extras_rgba_progressive",
        "vardct_extras_distributed_progressive",
        "vardct_extras_associated_squeeze",
        "vardct_extras_resampled_squeeze",
        "vardct_extras_transformed",
        "vardct_extras_associated_gray",
        "floating/vardct_extras_float_squeeze",
        "integer/vardct_extras_integer_rgb_extended31",
    ] {
        let data = fixture(name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(!inventory.image_header.extra_channels.is_empty());
        assert_eq!(inventory.frames.len(), 1);
        let frame = &inventory.frames[0];
        eprintln!(
            "{name}: {} passes, {} extras",
            frame.num_passes,
            inventory.image_header.extra_channels.len()
        );
        let Some(expected) = prefix_images(&data, frame, true) else {
            return;
        };
        let Some(native) = native_updates_oriented(&data, false, true) else {
            return;
        };
        let native_final = &native.last().unwrap().pixels;
        let request = request(true, AlphaOutputPolicy::Preserve);
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
            let mut held = Vec::new();
            let mut pixels = Vec::new();
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                let completed = pixels.len();
                let actual = read(&backend, update.output());
                if let Some(progression) = update.progression() {
                    assert!(matches!(progression, FrameProgression::Coefficients {
                        physical_frame_index: 0, completed_passes, total_passes, ..
                    } if usize::from(completed_passes) == completed && u32::from(total_passes) == frame.num_passes));
                    if let Some(reference) = &expected[completed] {
                        compare(name, &actual, reference, completed);
                    }
                } else {
                    assert_eq!(completed, frame.num_passes as usize);
                    compare(name, &actual, native_final, completed);
                }
                held.push(update);
                pixels.push(actual);
            }
            assert_eq!(pixels.len(), frame.num_passes as usize + 1);
            assert_eq!(session.frames_submitted(), 1);
            assert_eq!(session.active_frame_slots(), 1);
            let mut baseline = decoder
                .open(&data, request.clone().with_progressive_output(false))
                .unwrap();
            let final_frame = baseline.next_frame().unwrap().unwrap();
            assert_eq!(
                read(&backend, final_frame.output()),
                *pixels.last().unwrap()
            );
            for (update, actual) in held.iter().zip(&pixels) {
                assert_eq!(update.metadata, final_frame.metadata);
                assert_eq!(read(&backend, update.output()), *actual);
            }
            if let Some(expected) = &whole {
                assert_eq!(
                    &pixels, expected,
                    "{name}: fragmented input/windowed execution"
                );
            } else {
                whole = Some(pixels);
            }
            drop(final_frame);
            drop(baseline);
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
