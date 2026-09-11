//! Independent libjxl layer flushes followed by scalar f64 composition in codestream sRGB.
//! libjxl 0.12 FlushImage deliberately rejects cropped/blended frames. Only their image/frame
//! headers are replaced here; shared test support verifies every entropy interpretation field.
use super::*;
use jxl_gpu_bitstream::{CodestreamInventory, FrameBlendMode, FrameType};

pub(super) use common::progressive_layers as layers;

pub(super) fn present(pixels: &[f64], width: usize, height: usize, orientation: u32) -> Vec<u8> {
    let mut output = vec![0; pixels.len() * 4];
    let stride = if orientation >= 5 { height } else { width };
    for y in 0..height {
        for x in 0..width {
            let (dx, dy) = match orientation {
                1 => (x, y),
                2 => (width - 1 - x, y),
                3 => (width - 1 - x, height - 1 - y),
                4 => (x, height - 1 - y),
                5 => (y, x),
                6 => (height - 1 - y, x),
                7 => (height - 1 - y, width - 1 - x),
                8 => (y, width - 1 - x),
                _ => unreachable!(),
            };
            for c in 0..4 {
                let value = pixels[(y * width + x) * 4 + c];
                let value = if c == 3 {
                    value
                } else {
                    value.signum()
                        * if value.abs() <= 0.04045 {
                            value.abs() / 12.92
                        } else {
                            ((value.abs() + 0.055) / 1.055).powf(2.4)
                        }
                } as f32;
                let offset = ((dy * stride + dx) * 4 + c) * 4;
                output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    output
}

#[derive(Debug)]
pub(super) struct LfImage {
    pub frame: usize,
    pub physical_frame_index: u32,
    pub pixels: Vec<u8>,
}

fn lf1_pixels(data: &[u8]) -> Vec<u8> {
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let last_lf = inventory
        .frames
        .iter()
        .rposition(|frame| frame.lf_level == 1)
        .unwrap();
    let end = inventory.frames[last_lf + 1].header_bits.offset as usize / 8;
    let mut input = &data[..end];
    let decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let ProcessingResult::Complete {
        result: mut decoder,
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("complete image header")
    };
    decoder.set_pixel_format(JxlPixelFormat::rgb_f32(0));
    let (width, height) = decoder.basic_info().size;
    let ProcessingResult::NeedsMoreInput {
        fallback: mut decoder,
        ..
    } = decoder.process(&mut input, None).unwrap()
    else {
        panic!("LF prefix has no main header")
    };
    let mut pixels = vec![0; width * height * 12];
    assert!(
        decoder
            .flush_pixels(
                &mut [JxlOutputBuffer::new(&mut pixels, height, width * 12)],
                None
            )
            .unwrap()
    );
    pixels
        .chunks_exact(12)
        .flat_map(|rgb| rgb.iter().copied().chain(1_f32.to_le_bytes()))
        .collect()
}

pub(super) fn composed(
    data: &[u8],
    inventory: &CodestreamInventory,
    name: &str,
    keep: bool,
) -> Option<Vec<NativeUpdate>> {
    composed_with_lf(data, inventory, name, keep).map(|(updates, _)| updates)
}

pub(super) fn composed_with_lf(
    data: &[u8],
    inventory: &CodestreamInventory,
    name: &str,
    keep: bool,
) -> Option<(Vec<NativeUpdate>, Vec<LfImage>)> {
    let native = native_updates_oriented(data, false, true)?;
    let native_finals: Vec<_> = native.iter().filter(|step| step.complete).collect();
    let family = match name {
        "composed" => "vardct",
        "composed_gray" => "vardct_gray",
        "composed_lf" => "vardct_dc",
        _ => unreachable!(),
    };
    let image = &inventory.image_header;
    assert!(image.extra_channels.is_empty());
    let width = image.width as usize;
    let height = image.height as usize;
    let mut references: [Option<Vec<f64>>; 4] = std::array::from_fn(|_| None);
    let mut updates = Vec::new();
    let mut first = 0;
    let mut lf_images = Vec::new();
    let mut presentation = 0;
    for (index, frame) in inventory.frames.iter().enumerate() {
        if frame.frame_type == FrameType::LowFrequency {
            continue;
        }
        assert_eq!(frame.frame_type, FrameType::Regular);
        assert!(!frame.save_before_color_transform);
        let layer = std::str::from_utf8(&frame.name_bytes)
            .unwrap()
            .strip_prefix("composed-")
            .unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "test-data/progressive_composition/{family}_layer{layer}.headers"
        ));
        let mut dependencies = vec![frame.clone()];
        let mut source = frame.lf_source_frame;
        while let Some(id) = source {
            let dependency = inventory
                .frames
                .iter()
                .find(|frame| frame.frame_index == id)
                .unwrap();
            dependencies.push(dependency.clone());
            source = dependency.lf_source_frame;
        }
        dependencies.reverse();
        let standalone =
            layers::reframe(data, &dependencies, &std::fs::read_to_string(path).unwrap());
        if (frame.duration_ticks != 0 || frame.is_last)
            && let Some(source) = frame.lf_source_frame.filter(|id| *id as usize >= first)
        {
            let pixels = blend(&lf1_pixels(&standalone), frame, &references, width, height);
            lf_images.push(LfImage {
                frame: presentation,
                physical_frame_index: source,
                pixels: present(
                    &pixels,
                    width,
                    height,
                    if keep { 1 } else { image.orientation },
                ),
            });
        }
        let native_layer = native_updates_oriented(&standalone, false, true)?;
        let count = frame.num_passes as usize + 1; // DC, nonfinal AC passes, final image
        assert!(native_layer.len() >= count);
        let mut completed = None;
        for (step, update) in native_layer.iter().rev().take(count).rev().enumerate() {
            assert_eq!(update.complete, step + 1 == count);
            assert_eq!(
                update.pixels.len(),
                frame.width as usize * frame.height as usize * 16
            );
            let composed = blend(&update.pixels, frame, &references, width, height);
            if frame.duration_ticks != 0 || frame.is_last {
                if update.complete {
                    let oracle = native_finals[presentation];
                    assert_eq!(
                        (oracle.duration, oracle.timecode, oracle.is_last),
                        (
                            frame.duration_ticks,
                            frame.timecode.unwrap_or(0),
                            frame.is_last
                        )
                    );
                    let oracle: Vec<_> = oracle
                        .pixels
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                        .collect();
                    let error = sequence::relative_error(
                        &present(&composed, width, height, 1),
                        &present(&oracle, width, height, 1),
                    );
                    assert!(
                        error < 1e-5,
                        "{name} presentation{presentation}: independent composition oracle error {error}"
                    );
                }
                updates.push(NativeUpdate {
                    frame: presentation,
                    duration: frame.duration_ticks,
                    timecode: frame.timecode.unwrap_or(0),
                    is_last: frame.is_last,
                    step,
                    ratio: update.ratio,
                    complete: update.complete,
                    pixels: present(
                        &composed,
                        width,
                        height,
                        if keep { 1 } else { image.orientation },
                    ),
                });
            }
            if update.complete {
                completed = Some(composed);
            }
        }
        if !frame.is_last && (frame.duration_ticks == 0 || frame.save_as_reference != 0) {
            references[frame.save_as_reference as usize] = completed;
        }
        if frame.duration_ticks != 0 || frame.is_last {
            presentation += 1;
            first = index + 1;
        }
    }
    assert_eq!(presentation, native_finals.len());
    Some((updates, lf_images))
}

fn blend(
    pixels: &[u8],
    frame: &jxl_gpu_bitstream::FrameInventory,
    references: &[Option<Vec<f64>>; 4],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let mut composed = references[frame.color_blend.source as usize]
        .clone()
        .unwrap_or_else(|| [0.0, 0.0, 0.0, 1.0].repeat(width * height));
    for y in 0..frame.height as usize {
        for x in 0..frame.width as usize {
            let dx = x as i64 + i64::from(frame.x0);
            let dy = y as i64 + i64::from(frame.y0);
            if dx < 0 || dy < 0 || dx >= width as i64 || dy >= height as i64 {
                continue;
            }
            for c in 0..3 {
                let offset = ((y * frame.width as usize + x) * 4 + c) * 4;
                let foreground = f64::from(f32::from_le_bytes(
                    pixels[offset..offset + 4].try_into().unwrap(),
                ));
                let background = &mut composed[(dy as usize * width + dx as usize) * 4 + c];
                *background = match frame.color_blend.mode {
                    FrameBlendMode::Replace | FrameBlendMode::Blend => foreground,
                    FrameBlendMode::Add | FrameBlendMode::MultiplyAdd => *background + foreground,
                    FrameBlendMode::Multiply => {
                        *background
                            * if frame.color_blend.clamp {
                                foreground.clamp(0.0, 1.0)
                            } else {
                                foreground
                            }
                    }
                };
            }
        }
    }
    composed
}
