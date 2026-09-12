//! Native standalone prefix images followed by independent F64 crop/reference composition.
use jxl_gpu_bitstream::{ExtraChannelTypeInventory, FrameBlendMode, FrameInventory};

use super::progressive::{native_updates_options, native_updates_oriented};
use crate::fixtures::progressive_layers;

pub struct ExpectedUpdate {
    pub physical: u32,
    pub presentation: usize,
    pub completed: Option<u8>,
    pub pixels: Vec<f64>,
}

pub fn prefix_end(frame: &FrameInventory, completed: u8) -> usize {
    frame
        .sections
        .iter()
        .filter_map(|section| match section.kind {
            jxl_gpu_bitstream::FrameSectionKind::PassGroup { pass_index, .. }
                if pass_index >= u32::from(completed) =>
            {
                Some(section.bytes.offset as usize)
            }
            _ => None,
        })
        .min()
        .expect("sectioned progressive frame")
}

pub fn floats(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| {
            let value = f32::from_le_bytes(*word);
            assert!(value.is_finite());
            f64::from(value)
        })
        .collect()
}

pub fn relative_error(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| {
            assert!(a.is_finite() && b.is_finite());
            (a - b).abs() / b.abs().max(1.0)
        })
        .fold(0.0, f64::max)
}

/// The corpus uses either no alpha or one associated alpha with independent blend/source metadata.
fn blend(
    foreground: &[f64],
    frame: &FrameInventory,
    references: &[Option<Vec<f64>>; 4],
    width: usize,
    height: usize,
    alpha: bool,
) -> Vec<f64> {
    assert_eq!(
        foreground.len(),
        frame.width as usize * frame.height as usize * 4
    );
    let mut canvas = vec![0.0; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let sx = x as i64 - i64::from(frame.x0);
            let sy = y as i64 - i64::from(frame.y0);
            let inside =
                sx >= 0 && sy >= 0 && sx < i64::from(frame.width) && sy < i64::from(frame.height);
            for c in 0..4 {
                let offset = (y * width + x) * 4 + c;
                if c == 3 && !alpha {
                    canvas[offset] = 1.0;
                    continue;
                }
                let mut operation = if c < 3 {
                    frame.color_blend
                } else {
                    frame.extra_channel_blends[0]
                };
                // Color source-over also blends its selected alpha against that alpha's own source.
                if c == 3 && frame.color_blend.mode == FrameBlendMode::Blend {
                    operation.mode = FrameBlendMode::Blend;
                    operation.clamp = frame.color_blend.clamp;
                }
                let base = references[operation.source as usize]
                    .as_ref()
                    .map_or(0.0, |p| p[offset]);
                canvas[offset] = base;
                if !inside {
                    continue;
                }
                let source = (sy as usize * frame.width as usize + sx as usize) * 4;
                let top = foreground[source + c];
                let a = foreground[source + 3];
                let a = if operation.clamp {
                    a.clamp(0.0, 1.0)
                } else {
                    a
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
                    FrameBlendMode::MultiplyAdd => base + a * top,
                    FrameBlendMode::Blend if c == 3 => 1.0 - (1.0 - a) * (1.0 - base),
                    FrameBlendMode::Blend => top + (1.0 - a) * base,
                };
            }
        }
    }
    canvas
}

pub fn updates(data: &[u8], directory: &str, family: &str) -> Option<Vec<ExpectedUpdate>> {
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let inventory = parsed.codestream_inventory(Default::default()).unwrap();
    let image = &inventory.image_header;
    let alpha = !image.extra_channels.is_empty();
    assert!(image.extra_channels.len() <= 1);
    if alpha {
        assert_eq!(
            image.extra_channels[0].channel_type,
            ExtraChannelTypeInventory::Alpha { associated: true }
        );
    }
    let native = native_updates_oriented(data, false, true)?;
    let finals: Vec<_> = native.iter().filter(|step| step.complete).collect();
    let (width, height) = (image.width as usize, image.height as usize);
    let mut references = std::array::from_fn(|_| None);
    let mut result = Vec::new();
    let mut presentation = 0;
    for (index, frame) in inventory.frames.iter().enumerate() {
        assert_eq!(frame.frame_type, jxl_gpu_bitstream::FrameType::Regular);
        assert!(!frame.save_before_color_transform);
        let path = crate::decoder_directory().join(format!(
            "test-data/{directory}/{family}_layer{index}.headers"
        ));
        let headers = std::fs::read_to_string(path).unwrap();
        let standalone = progressive_layers::reframe(data, std::slice::from_ref(frame), &headers);
        let standalone_inventory = jxl_gpu_bitstream::parse(&standalone, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let visible = frame.duration_ticks != 0 || frame.is_last;
        let mut stages = Vec::new();
        if visible {
            for completed in 0..frame.num_passes {
                let end = prefix_end(&standalone_inventory.frames[0], completed as u8);
                let prefix = native_updates_options(&standalone[..end], false, true, true)?;
                stages.push(floats(
                    &prefix.last().expect("flushable layer prefix").pixels,
                ));
            }
        }
        let native_layer = native_updates_oriented(&standalone, false, true)?;
        stages.push(floats(&native_layer.last().unwrap().pixels));
        let last = stages.len() - 1;
        for (completed, pixels) in stages.into_iter().enumerate() {
            let composed = blend(&pixels, frame, &references, width, height, alpha);
            if visible {
                if completed == last {
                    let native = finals[presentation];
                    assert_eq!(
                        (native.duration, native.timecode, native.is_last),
                        (
                            frame.duration_ticks,
                            frame.timecode.unwrap_or(0),
                            frame.is_last
                        )
                    );
                    let error = relative_error(&composed, &floats(&native.pixels));
                    assert!(
                        error < 3e-6,
                        "{family} presentation{presentation}: independent composition error {error}"
                    );
                }
                result.push(ExpectedUpdate {
                    physical: frame.frame_index,
                    presentation,
                    completed: (completed != last).then_some(completed as u8),
                    pixels: composed.clone(),
                });
            }
            if completed == last
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

pub fn orient(pixels: &[f64], width: usize, height: usize, orientation: u32) -> Vec<f64> {
    let mut output = vec![0.0; pixels.len()];
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
            output[(dy * stride + dx) * 4..(dy * stride + dx + 1) * 4]
                .copy_from_slice(&pixels[(y * width + x) * 4..(y * width + x + 1) * 4]);
        }
    }
    output
}
