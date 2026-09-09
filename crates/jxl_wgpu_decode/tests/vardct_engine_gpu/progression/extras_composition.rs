//! Standalone native prefix flushes, then independent associated-alpha composition in F64 sRGB.
//! The oracle checks every final presentation against libjxl's original coalesced sequence.
use super::super::sequence_oracle::layers;
use super::*;
use jxl_gpu_bitstream::{FrameBlendMode, FrameInventory};

struct Expected {
    physical: u32,
    presentation: usize,
    completed: Option<u8>,
    pixels: Vec<u8>,
}

fn blend(
    pixels: &[u8],
    frame: &FrameInventory,
    references: &[Option<Vec<f64>>; 4],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let foreground: Vec<_> = pixels
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    let mut canvas = vec![0.0; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let sx = x as i64 - i64::from(frame.x0);
            let sy = y as i64 - i64::from(frame.y0);
            let inside =
                sx >= 0 && sy >= 0 && sx < i64::from(frame.width) && sy < i64::from(frame.height);
            for c in 0..4 {
                let mut operation = if c < 3 {
                    frame.color_blend
                } else {
                    frame.extra_channel_blends[0]
                };
                // Color Blend also writes its alpha using that alpha's independently selected background.
                if c == 3 && frame.color_blend.mode == FrameBlendMode::Blend {
                    operation.mode = FrameBlendMode::Blend;
                    operation.clamp = frame.color_blend.clamp;
                }
                let offset = (y * width + x) * 4 + c;
                let base = references[operation.source as usize]
                    .as_ref()
                    .map_or(0.0, |p| p[offset]);
                canvas[offset] = base;
                if !inside {
                    continue;
                }
                let position = (sy as usize * frame.width as usize + sx as usize) * 4;
                let top = foreground[position + c];
                let alpha = foreground[position + 3];
                let alpha = if operation.clamp {
                    alpha.clamp(0.0, 1.0)
                } else {
                    alpha
                };
                canvas[offset] = match operation.mode {
                    FrameBlendMode::Replace => top,
                    FrameBlendMode::Add => base + top,
                    FrameBlendMode::Multiply => {
                        base * if operation.clamp {
                            top.clamp(0.0, 1.0)
                        } else {
                            top
                        }
                    }
                    FrameBlendMode::MultiplyAdd if c == 3 => base,
                    FrameBlendMode::MultiplyAdd => base + alpha * top,
                    FrameBlendMode::Blend if c == 3 => 1.0 - (1.0 - alpha) * (1.0 - base),
                    FrameBlendMode::Blend => top + (1.0 - alpha) * base,
                };
            }
        }
    }
    canvas
}

fn oracle(data: &[u8], keep: bool) -> Option<Vec<Expected>> {
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let image = &inventory.image_header;
    assert_eq!(image.extra_channels.len(), 1);
    assert!(matches!(
        image.extra_channels[0].channel_type,
        jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated: true }
    ));
    let native = native_updates_oriented(data, false, true)?;
    let finals: Vec<_> = native.iter().filter(|image| image.complete).collect();
    let width = image.width as usize;
    let height = image.height as usize;
    let mut references = std::array::from_fn(|_| None);
    let mut result = Vec::new();
    let mut presentation = 0;
    for (index, frame) in inventory.frames.iter().enumerate() {
        let headers =
            std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
                format!("test-data/progressive_composition/associated_vardct_layer{index}.headers"),
            ))
            .unwrap();
        let standalone = layers::reframe(data, std::slice::from_ref(frame), &headers);
        let standalone_inventory = jxl_gpu_bitstream::parse(&standalone, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let visible = frame.duration_ticks != 0 || frame.is_last;
        let mut stages = if visible {
            prefix_images(&standalone, &standalone_inventory.frames[0], true)?
                .into_iter()
                .map(|image| image.expect("sectioned progressive layer"))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let native = native_updates_oriented(&standalone, false, true)?;
        stages.push(native.last().unwrap().pixels.clone());
        let final_index = stages.len() - 1;
        for (completed, pixels) in stages.into_iter().enumerate() {
            let composed = blend(&pixels, frame, &references, width, height);
            if visible {
                let final_image = completed == final_index;
                let pixels = super::super::sequence_oracle::present(
                    &composed,
                    width,
                    height,
                    if keep { 1 } else { image.orientation },
                );
                if final_image {
                    let reference: Vec<_> = finals[presentation]
                        .pixels
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .collect();
                    let reference = super::super::sequence_oracle::present(
                        &reference,
                        width,
                        height,
                        if keep { 1 } else { image.orientation },
                    );
                    let error = sequence::relative_error(&pixels, &reference);
                    assert!(
                        error < 3e-6,
                        "independent alpha composition presentation{presentation}: {error}"
                    );
                }
                result.push(Expected {
                    physical: frame.frame_index,
                    presentation,
                    completed: (!final_image).then_some(completed as u8),
                    pixels,
                });
            }
            if completed == final_index
                && !frame.is_last
                && (frame.duration_ticks == 0 || frame.save_as_reference != 0)
            {
                references[frame.save_as_reference as usize] = Some(composed);
            }
        }
        if visible {
            presentation += 1;
        }
    }
    assert_eq!(presentation, finals.len());
    Some(result)
}

#[test]
fn alpha_composed_updates_use_committed_references_and_match_independent_native_layers() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let data = fixture("composition_associated_vardct");
    for keep in [false, true] {
        let Some(expected) = oracle(&data, keep) else {
            return;
        };
        let mut request =
            sequence::request(keep).with_alpha_output_policy(AlphaOutputPolicy::Preserve);
        request = request.with_max_frame_slots(NonZeroUsize::new(1).unwrap());
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
            let mut finals = Vec::new();
            while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
                let reference = &expected[pixels.len()];
                assert_eq!(update.metadata.index, reference.presentation);
                let actual = read(&backend, update.output());
                assert_eq!(
                    update.progression().and_then(|p| p.completed_passes()),
                    reference.completed
                );
                if let Some(FrameProgression::Coefficients {
                    physical_frame_index,
                    ..
                }) = update.progression()
                {
                    assert_eq!(physical_frame_index, reference.physical);
                }
                let error = sequence::relative_error(&actual, &reference.pixels);
                assert!(
                    error < 1e-3,
                    "keep{keep} cap{cap} physical{} {:?}: {error}",
                    reference.physical,
                    reference.completed
                );
                if update.is_complete() {
                    finals.push(actual.clone());
                }
                held.push(owned(update.output()));
                pixels.push(actual);
            }
            assert_eq!(pixels.len(), expected.len());
            let mut baseline = decoder
                .open(&data, request.clone().with_progressive_output(false))
                .unwrap();
            for pixels in finals {
                let frame = baseline.next_frame().unwrap().unwrap();
                assert_eq!(read(&backend, frame.output()), pixels);
            }
            assert!(baseline.next_frame().unwrap().is_none());
            for (image, pixels) in held.iter().zip(&pixels) {
                assert_eq!(read(&backend, image), *pixels);
            }
            if let Some(expected) = &whole {
                assert_eq!(&pixels, expected);
            } else {
                whole = Some(pixels);
            }
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
