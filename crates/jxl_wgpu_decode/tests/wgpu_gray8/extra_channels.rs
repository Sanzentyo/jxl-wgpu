use super::*;
use jxl_gpu_bitstream::{InventoryLimits, SampleBitDepth, parse};
use jxl_gpu_formats::RgbChannelOrder;
use jxl_gpu_protocol::OutputOrientation;
use jxl_wgpu_decode::{OrientationPolicy, WgpuDecodeEngine};

fn cases() -> [(&'static str, &'static str); 6] {
    [
        (
            "data_only",
            include_str!("../../test-data/extras_data_only.jxl.hex"),
        ),
        (
            "rgb12",
            include_str!("../../test-data/extras_rgb12.jxl.hex"),
        ),
        (
            "gray8",
            include_str!("../../test-data/extras_gray8.jxl.hex"),
        ),
        (
            "gray_alpha",
            include_str!("../../test-data/extras_gray_alpha.jxl.hex"),
        ),
        ("rgba", include_str!("../../test-data/extras_rgba.jxl.hex")),
        (
            "transformed",
            include_str!("../../test-data/extras_transformed.jxl.hex"),
        ),
    ]
}

fn encoded(hex: &str) -> Vec<u8> {
    let digits = hex.split_whitespace().collect::<String>();
    digits
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn bits(depth: SampleBitDepth) -> u8 {
    let SampleBitDepth::Integer { bits_per_sample } = depth else {
        panic!("integer fixture")
    };
    bits_per_sample as u8
}

fn expected_code(x: u32, y: u32, channel: u32, bits: u8) -> u32 {
    let mask = (1 << bits) - 1;
    match x % 11 {
        0 => 0,
        1 => mask,
        _ => (193 * x + 317 * y + 97 * channel + ((x ^ y) * (23 + channel))) & mask,
    }
}

fn rust_planes(encoded: &[u8], apply: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (mut color, mut extras) = extra_channel_oracle::rust_planes(encoded);
    if !apply {
        // jxl 0.6.0 exposes adjust_orientation but does not use it in its render pipeline.
        let inventory = parse(encoded, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let image = &inventory.image_header;
        let extent = OutputOrientation::from_exif_value(image.orientation)
            .unwrap()
            .map_extent(Extent2d::new(image.width, image.height));
        let orientation = inventory.image_header.orientation;
        color = frame_sequence::keep_codestream_order(&color, extent, 4, orientation);
        for plane in &mut extras {
            *plane = frame_sequence::keep_codestream_order(plane, extent, 1, orientation);
        }
    }
    (color, extras)
}

#[path = "../common/extra_channel_oracle.rs"]
mod extra_channel_oracle;
use extra_channel_oracle::{floats, libjxl_planes};

#[test]
fn arbitrary_modular_extra_channels_preserve_native_codes_and_independent_precision() {
    let Some(backend) = backend() else { return };
    for (name, hex) in cases() {
        let data = encoded(hex);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let image = &inventory.image_header;
        let original = Extent2d::new(image.width, image.height);
        let orientation = OutputOrientation::from_exif_value(image.orientation).unwrap();
        let colors = if image.grayscale { 1 } else { 3 };
        let (_, reference) = rust_planes(&data, false);
        let (_, oriented) = rust_planes(&data, true);
        let libjxl = libjxl_planes(
            &data,
            (image.width * image.height) as usize,
            image.extra_channels.len(),
        );
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let bounded = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
        );
        for (index, extra) in image.extra_channels.iter().enumerate() {
            let depth = bits(extra.bit_depth);
            for floating in [false, true] {
                let request = if floating {
                    GpuOutputRequest::numeric(
                        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                        NumericSampleMapping::NormalizedUnsigned,
                    )
                    .unwrap()
                } else {
                    GpuOutputRequest::numeric(
                        LosslessModularFormat::Gray.pixel_format(depth).unwrap(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                    .unwrap()
                }
                .with_extra_channel(index as u32)
                .unwrap()
                .with_orientation_policy(if floating {
                    OrientationPolicy::Apply
                } else {
                    OrientationPolicy::Keep
                });
                let selected = if floating && index + 1 == image.extra_channels.len() {
                    &bounded
                } else {
                    &decoder
                };
                let mut session = if floating && index + 1 == image.extra_channels.len() {
                    frame_sequence::incremental(selected, &data, request)
                } else {
                    selected
                        .open(&data, request)
                        .unwrap_or_else(|e| panic!("{name} plane {index}: {e}"))
                };
                assert_eq!(&session.metadata().extra_channels, &image.extra_channels);
                let jxl_wgpu_decode::DecodeProfile::ModularLossless {
                    channels,
                    prediction,
                    ..
                } = session.profile()
                else {
                    panic!("Modular source")
                };
                assert_eq!(channels.color_count(), colors);
                assert_eq!(channels.extra_count() as usize, image.extra_channels.len());
                if name == "data_only" {
                    assert!(matches!(
                        prediction,
                        jxl_wgpu_decode::ModularPredictionProfile::MetaAdaptive {
                            node_count: 1,
                            decision_node_count: 0,
                            max_depth: 0,
                            ..
                        }
                    ));
                }
                if name == "transformed" {
                    let stats = session
                        .submission_session()
                        .modular()
                        .unwrap()
                        .memory_stats();
                    assert!(stats.inverse_transform_count > 1);
                }
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let output = &frame.output().outputs[0];
                let bytes = read_output(&backend, output);
                assert_eq!(
                    output.layout.extent,
                    if floating {
                        orientation.map_extent(original)
                    } else {
                        original
                    }
                );
                for y in 0..image.height {
                    for x in 0..image.width {
                        let expected = expected_code(x, y, colors + index as u32, depth);
                        let pos = (y * image.width + x) as usize;
                        assert!(
                            (reference[index][pos] * ((1u32 << depth) - 1) as f32
                                - expected as f32)
                                .abs()
                                < 0.02,
                            "{name} Rust plane {index} source mismatch at {x},{y}: {} vs code {expected}, depth {depth}",
                            reference[index][pos]
                        );
                        if floating {
                            let actual =
                                f32::from_le_bytes(bytes[pos * 4..pos * 4 + 4].try_into().unwrap());
                            assert!(
                                (actual - oriented[index][pos]).abs() < 2e-7,
                                "{name} F32 plane {index} at {x},{y}: {actual}"
                            );
                            if let Some((_, libjxl)) = &libjxl {
                                assert!(
                                    (actual - libjxl[index][pos]).abs() < 2e-7,
                                    "{name} libjxl plane {index} at {pos}"
                                );
                            }
                        } else {
                            let actual = if depth <= 8 {
                                u32::from(bytes[pos])
                            } else {
                                u32::from(u16::from_le_bytes(
                                    bytes[pos * 2..pos * 2 + 2].try_into().unwrap(),
                                ))
                            };
                            assert_eq!(actual, expected, "{name} native plane {index} at {x},{y}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gray_alpha_and_multiple_extras_select_color_and_first_alpha_without_colorizing_data() {
    let Some(backend) = backend() else { return };
    for (name, hex) in cases() {
        let data = encoded(hex);
        let (expected, _) = rust_planes(&data, true);
        let inventory = parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let libjxl = libjxl_planes(
            &data,
            expected.len() / 4,
            inventory.image_header.extra_channels.len(),
        );
        let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
        let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap()
        .with_spot_color_policy(jxl_wgpu_decode::SpotColorPolicy::Preserve);
        let mut session = decoder
            .open(&data, request)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let frame = session.next_frame().unwrap().unwrap();
        let actual = floats(&read_output(&backend, &frame.output().outputs[0]));
        assert_eq!(actual.len(), expected.len(), "{name} F32 sample count");
        assert!(actual.iter().all(|v| v.is_finite()), "{name} finite F32");
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 2e-7, "{name} RGBA error {error}");
        if let Some((libjxl, _)) = libjxl {
            assert_eq!(actual.len(), libjxl.len(), "{name} libjxl sample count");
            let error = actual
                .iter()
                .zip(&libjxl)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(error < 2e-7, "{name} libjxl RGBA error {error}");
        }
        let depth = bits(inventory.image_header.bit_depth);
        let request =
            GpuOutputRequest::color(LosslessModularFormat::Rgba.pixel_format(depth).unwrap())
                .unwrap()
                .with_spot_color_policy(jxl_wgpu_decode::SpotColorPolicy::Preserve);
        let mut session = decoder.open(&data, request).unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        let bytes = read_output(&backend, &frame.output().outputs[0]);
        let native: Vec<u32> = if depth <= 8 {
            bytes.iter().copied().map(u32::from).collect()
        } else {
            bytes
                .chunks_exact(2)
                .map(|v| u32::from(u16::from_le_bytes(v.try_into().unwrap())))
                .collect()
        };
        assert_eq!(native.len(), expected.len());
        for (index, (&actual, &reference)) in native.iter().zip(&expected).enumerate() {
            let expected =
                (reference.clamp(0.0, 1.0) * ((1u32 << depth) - 1) as f32).round() as u32;
            assert_eq!(actual, expected, "{name} native RGBA sample {index}");
        }
    }
}

#[test]
fn extra_channel_selection_validates_indices_and_preserves_retry_and_cancellation_ownership() {
    use jxl_wgpu_decode::{Error, SpotColorPolicy, UnsupportedCodestreamFeature};
    let Some(backend) = backend() else { return };
    let data = encoded(include_str!("../../test-data/extras_rgb12.jxl.hex"));
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let scalar = || {
        GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
    };
    for index in [9, u32::MAX] {
        assert!(
            matches!(decoder.open(&data, scalar().with_extra_channel(index).unwrap()),
            Err(Error::ExtraChannelIndex { index: actual, count: 9 }) if actual == index)
        );
    }
    let composed = encoded(include_str!(
        "../../test-data/composition_gray_alpha.jxl.hex"
    ));
    assert!(
        matches!(decoder.open(&composed, scalar().with_extra_channel(0).unwrap()),
        Err(Error::UnsupportedProfile(ref e)) if e.feature == UnsupportedCodestreamFeature::ExtraChannels)
    );
    let color = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap();
    assert!(matches!(
        color.clone().with_extra_channel(0),
        Err(Error::UnsupportedOutputFormat(_))
    ));
    assert!(
        matches!(decoder.open(&data, color.clone()), Err(Error::UnsupportedProfile(ref e))
        if e.feature == UnsupportedCodestreamFeature::ExtraChannels)
    );
    assert!(
        decoder
            .open(
                &data,
                color.with_spot_color_policy(SpotColorPolicy::Preserve)
            )
            .is_ok()
    );
    assert!(matches!(
        jxl_wgpu_decode::ModularChannelCounts::new(false, u32::MAX),
        Err(Error::ModularChannelCountOverflow {
            color_channels: 3,
            extra_channels: u32::MAX
        })
    ));
    let memory = backend.transient_memory_budget();
    assert_eq!(memory.snapshot().reserved_bytes, 0);
    for cancel in [false, true] {
        let mut session =
            frame_sequence::incremental(&decoder, &data, scalar().with_extra_channel(4).unwrap());
        let guard = memory
            .try_reserve(memory.snapshot().available_bytes)
            .unwrap();
        let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(progress.submitted, 0);
        assert!(matches!(
            progress.backpressure,
            Some(PrefetchBackpressure::Memory(_))
        ));
        drop(guard);
        assert_eq!(
            session
                .prefetch(NonZeroUsize::new(1).unwrap())
                .unwrap()
                .queued,
            1
        );
        let retained = if cancel {
            None
        } else {
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            Some(frame.output().outputs[0].buffer.clone())
        };
        drop(session);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        let bytes = retained.as_ref().map_or(0, GpuBufferLease::size);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while memory.snapshot().reserved_bytes != bytes && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, bytes);
        drop(retained);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
