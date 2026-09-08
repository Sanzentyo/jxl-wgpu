//! Independent full-frame and every-plane oracle checks shared by source precision corpora.
use super::planes::{open_fragmented, read};
use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};
use std::num::NonZeroU64;
fn text(directory: &str, name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-data")
            .join(directory)
            .join(name),
    )
    .unwrap()
}

fn encoded(directory: &str, name: &str) -> Vec<u8> {
    let hex = text(directory, &format!("{name}.jxl.hex"))
        .split_whitespace()
        .collect::<String>();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn reference(directory: &str, name: &str, suffix: &str) -> Vec<u32> {
    text(directory, &format!("{name}.{suffix}"))
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).unwrap())
        .collect()
}

fn numeric(mapping: NumericSampleMapping) -> GpuOutputRequest {
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        mapping,
    )
    .unwrap()
}

fn color() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_spot_color_policy(SpotColorPolicy::Preserve)
}

fn compare(name: &str, actual: &[u32], expected: &[u32], color: bool, unpremultiplied: bool) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let tolerance = if color {
        if name.contains("vardct") {
            0.003
        } else if name.contains("_lossy_") {
            1e-4
        } else {
            2e-5
        }
    } else {
        2e-6
    };
    for (index, (&actual_word, &expected_word)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (f32::from_bits(actual_word), f32::from_bits(expected_word));
        // Compare reconstructed color before output unpremultiplication, matching the existing
        // VarDCT oracle contract when a tiny alpha amplifies IDCT rounding differences.
        let scale = if unpremultiplied && index % 4 != 3 {
            f32::from_bits(expected[index / 4 * 4 + 3]).max(1.0 / 67108864.0)
        } else {
            1.0
        };
        let limit = if color && index % 4 == 3 {
            2e-6
        } else {
            tolerance
        };
        assert!(
            a.is_finite() && (a - b).abs() * scale <= limit * (1.0 + (b * scale).abs()),
            "{name}/{index}: {a} vs {b}, difference {}",
            (a - b).abs()
        );
    }
}

pub fn check_rendering(directory: &str, names: Vec<String>) {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in names {
        let data = encoded(directory, &name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let channels = image.extra_channels.len();
        let base = reference(directory, &name, "f32.hex");
        let frame_words = pixels * (4 + channels);
        let frames = base.len() / frame_words;
        assert!(frames > 0 && base.len().is_multiple_of(frame_words));
        if name.contains("progressive_dc") {
            assert!(inventory.frames.len() > 1);
        }
        for selected in 0..channels + 3 {
            let color_output = selected >= channels;
            let request = if !color_output {
                numeric(
                    if matches!(
                        image.extra_channels[selected].bit_depth,
                        SampleBitDepth::Float { .. }
                    ) {
                        NumericSampleMapping::NativeFloat
                    } else {
                        NumericSampleMapping::NormalizedUnsigned
                    },
                )
                .with_extra_channel(selected as u32)
                .unwrap()
            } else if selected == channels + 1 {
                color().with_spot_color_policy(SpotColorPolicy::Render)
            } else if selected == channels + 2 {
                color().with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            } else {
                color()
            };
            let expected: Vec<_> = if selected == channels + 1 {
                reference(directory, &name, "spots.f32.hex")
            } else if selected == channels + 2 {
                reference(directory, &name, "associated.f32.hex")
            } else {
                base.chunks_exact(frame_words)
                    .flat_map(|frame| {
                        let range = if color_output {
                            0..pixels * 4
                        } else {
                            pixels * (4 + selected)..pixels * (5 + selected)
                        };
                        frame[range].iter().copied()
                    })
                    .collect()
            };
            let mut first = Vec::new();
            for (decoder, fragmented) in [(&whole, false), (&bounded, true)] {
                let mut session = if fragmented {
                    open_fragmented(decoder, &data, request.clone())
                } else {
                    decoder
                        .open(&data, request.clone())
                        .unwrap_or_else(|e| panic!("{name}/{selected}: open: {e}"))
                };
                assert_eq!(session.metadata().extra_channels, image.extra_channels);
                let mut actual = Vec::new();
                let mut count = 0;
                while let Some(frame) = pollster::block_on(session.next_frame_async())
                    .unwrap_or_else(|e| panic!("{name}/{selected}: decode: {e}"))
                {
                    actual.extend(read(&backend, &frame.output().outputs[0]));
                    count += 1;
                }
                assert_eq!(count, frames, "{name}/{selected}");
                let associated = image
                    .extra_channels
                    .iter()
                    .find_map(|extra| match extra.channel_type {
                        jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated } => {
                            Some(associated)
                        }
                        _ => None,
                    })
                    .unwrap_or(false);
                compare(
                    &format!("{name}/{selected}"),
                    &actual,
                    &expected,
                    color_output,
                    color_output && associated && selected != channels + 2,
                );
                if fragmented {
                    assert_eq!(
                        actual, first,
                        "{name}/{selected}: bounded input changes words"
                    );
                } else {
                    first = actual;
                }
                drop(session);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
}
