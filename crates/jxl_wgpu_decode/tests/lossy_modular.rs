#![cfg(not(target_arch = "wasm32"))]
#[path = "common/extra_channel_oracle.rs"]
mod oracle;
#[path = "support/planes.rs"]
mod planes;
#[path = "support/rendering.rs"]
mod rendering;

fn cases() -> Vec<String> {
    let directory =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-data/lossy_modular");
    let mut names: Vec<_> = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter_map(|name| name.to_str()?.strip_suffix(".jxl.hex").map(str::to_owned))
        .collect();
    names.sort();
    names
}

fn encoded(name: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("test-data/lossy_modular/{name}.jxl.hex")),
    )
    .unwrap()
    .split_whitespace()
    .collect::<String>();
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn reference(name: &str, suffix: &str) -> Vec<f32> {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("test-data/lossy_modular/{name}.{suffix}.hex")),
    )
    .unwrap()
    .split_whitespace()
    .map(|word| f32::from_bits(u32::from_str_radix(word, 16).unwrap()))
    .collect()
}

#[test]
fn lossy_modular_reconstructs_color_and_every_extra_plane() {
    let names = cases();
    assert_eq!(names.len(), 19);
    rendering::check_rendering("lossy_modular", names);
}

#[test]
fn lossy_modular_references_match_independent_rust_and_live_libjxl_decoders() {
    for name in cases() {
        let data = encoded(&name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let expected = reference(&name, "f32");
        let rust = if image.animation.is_none() {
            vec![oracle::rust_planes(&data)]
        } else {
            oracle::rust_frame_planes(&data)
        };
        let rust: Vec<_> = rust
            .into_iter()
            .flat_map(|(color, extras)| color.into_iter().chain(extras.into_iter().flatten()))
            .collect();
        assert_eq!(rust.len(), expected.len(), "{name}");
        let frame_words = pixels * (4 + image.extra_channels.len());
        // Rust jxl 0.6 preserves stored associated RGB. libjxl's ordinary output unpremultiplies
        // with a 2^-26 floor, so use its explicit preserved-association output for this comparison.
        let mut rust_reference = expected.clone();
        if image
            .extra_channels
            .iter()
            .find_map(|extra| match extra.channel_type {
                jxl_gpu_bitstream::ExtraChannelTypeInventory::Alpha { associated } => {
                    Some(associated)
                }
                _ => None,
            })
            .unwrap_or(false)
        {
            let associated = reference(&name, "associated.f32");
            for (frame, color) in rust_reference
                .chunks_exact_mut(frame_words)
                .zip(associated.chunks_exact(pixels * 4))
            {
                frame[..pixels * 4].copy_from_slice(color);
            }
        }
        // The existing corpus pins Rust jxl's clamped-Multiply operand reversal. Only the initial
        // Replace presentation is an independent authority for these multi-extra reference chains;
        // all later presentations remain checked against live libjxl and the GPU differential.
        let compared_words = if image.animation.is_some() {
            frame_words
        } else {
            expected.len()
        };
        for (index, (&actual, &expected)) in rust
            .iter()
            .zip(&rust_reference)
            .take(compared_words)
            .enumerate()
        {
            let color = index % frame_words < pixels * 4 && index % 4 != 3;
            let tolerance = if color { 1e-4 } else { 2e-6 };
            assert!(
                actual.is_finite()
                    && (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
                "{name}/{index}: Rust {actual} vs libjxl {expected}"
            );
        }
        let live = if image.animation.is_none() {
            oracle::libjxl_planes(&data, pixels, image.extra_channels.len()).map(
                |(color, extras)| {
                    color
                        .into_iter()
                        .chain(extras.into_iter().flatten())
                        .collect::<Vec<_>>()
                },
            )
        } else {
            oracle::libjxl_output(&data, &[])
        };
        if let Some(live) = live {
            assert_eq!(
                live, expected,
                "{name}: checked-in and live libjxl output differ"
            );
        }
    }
}

#[test]
fn lossy_modular_quantizes_only_at_final_requested_output() {
    use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
    use jxl_wgpu_decode::{
        GpuDecoder, GpuOutputRequest, ModularChannels, SpotColorPolicy, native_modular_pixel_format,
    };
    let backend =
        pollster::block_on(jxl_wgpu::WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for name in [
        "extras_lossy_epf3",
        "extras_lossy_associated",
        "extras_lossy_gray",
        "extras_lossy_resampled4",
        "extras_lossy_original",
        "extras_lossy_float",
        "composition_extras_lossy_rgb",
        "composition_extras_lossy_float",
    ] {
        let data = encoded(name);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let expected = reference(name, "f32");
        for (format, bits, channels) in [
            (jxl_wgpu_decode::vardct_rgb8_format(), 8, 3),
            (
                PixelFormat::rgb8(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ),
                8,
                4,
            ),
            (
                native_modular_pixel_format(ModularChannels::Rgba, 16).unwrap(),
                16,
                4,
            ),
        ] {
            let request = GpuOutputRequest::color(format)
                .unwrap()
                .with_spot_color_policy(SpotColorPolicy::Preserve);
            let mut session = decoder.open(&data, request).unwrap();
            let mut expected_frames =
                expected.chunks_exact(pixels * (4 + image.extra_channels.len()));
            while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
                let output = planes::read_bytes(&backend, &frame.output().outputs[0]);
                let expected = &expected_frames.next().unwrap()[..pixels * 4];
                let words: Vec<u32> = if bits == 8 {
                    output.into_iter().map(u32::from).collect()
                } else {
                    output
                        .chunks_exact(2)
                        .map(|word| u32::from(u16::from_le_bytes(word.try_into().unwrap())))
                        .collect()
                };
                assert_eq!(words.len(), pixels * channels);
                let maximum = (1u32 << bits) - 1;
                for (actual, expected) in words.into_iter().zip(
                    expected
                        .chunks_exact(4)
                        .flat_map(|pixel| &pixel[..channels]),
                ) {
                    let reference =
                        (f64::from(expected.clamp(0.0, 1.0)) * f64::from(maximum)).round() as u32;
                    assert!(
                        actual.abs_diff(reference) <= if bits == 8 { 1 } else { 7 },
                        "{name}: {actual} vs {reference}"
                    );
                }
            }
            assert!(expected_frames.next().is_none());
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
