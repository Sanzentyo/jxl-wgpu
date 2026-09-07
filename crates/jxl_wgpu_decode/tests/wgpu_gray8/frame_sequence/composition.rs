use super::*;
use jxl_gpu_formats::{ColorSpecification, RgbChannelOrder, TransferFunction};
use jxl_wgpu_decode::{OrientationPolicy, PrefetchBackpressure};

fn srgb_to_linear(value: f32) -> f32 {
    value.signum()
        * if value.abs() <= 0.04045 {
            value.abs() / 12.92
        } else {
            ((value.abs() + 0.055) / 1.055).powf(2.4)
        }
}

#[test]
fn floating_composition_packs_only_after_blending_and_orientation() {
    let Some(backend) = backend() else {
        return;
    };
    for case in composition_cases() {
        let bytes = encoded(&case);
        let inventory = parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let original = rust_float_frames(&bytes, case.format);
        let image = &inventory.image_header;
        let displayed = jxl_gpu_protocol::OutputOrientation::from_exif_value(image.orientation)
            .unwrap()
            .map_extent(Extent2d::new(image.width, image.height));
        for (orientation, planar, linear) in [
            (OrientationPolicy::Apply, false, false),
            (OrientationPolicy::Keep, true, true),
        ] {
            let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
            if let ColorSpecification::Defined(spec) = &mut color
                && linear
            {
                spec.transfer = TransferFunction::Linear;
            }
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                if planar {
                    RgbChannelOrder::Bgra
                } else {
                    RgbChannelOrder::Rgba
                },
                planar,
                color,
            ))
            .unwrap()
            .with_orientation_policy(orientation);
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
            );
            let mut session = incremental(&decoder, &bytes, request);
            for (index, expected) in original.iter().enumerate() {
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                let output = &frame.output().outputs[0];
                let channels = case.format.channel_count() as usize;
                let expected = if orientation == OrientationPolicy::Keep {
                    keep_codestream_order(expected, displayed, channels, image.orientation)
                } else {
                    expected.clone()
                };
                let data = read_output(&backend, output);
                let width = output.layout.extent.width as usize;
                let mut maximum = 0f32;
                for (pixel, oracle) in expected.chunks_exact(channels).enumerate() {
                    for channel in 0..4 {
                        let stored = if planar && channel < 3 {
                            2 - channel
                        } else {
                            channel
                        };
                        let plane = &output.layout.planes[if planar { stored } else { 0 }];
                        let offset = plane.offset as usize
                            + (pixel / width) * plane.row_stride as usize
                            + (pixel % width) * if planar { 4 } else { 16 }
                            + if planar { 0 } else { stored * 4 };
                        let actual =
                            f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                        let mut reference = if channel == 3 && channels != 4 {
                            1.0
                        } else {
                            oracle[if channels == 1 { 0 } else { channel }]
                        };
                        if linear && channel < 3 {
                            reference = srgb_to_linear(reference);
                        }
                        assert!(actual.is_finite());
                        let error = if !linear && channel < 3 {
                            (srgb_to_linear(actual) - srgb_to_linear(reference)).abs()
                        } else {
                            (actual - reference).abs()
                        };
                        let reference_linear = if !linear && channel < 3 {
                            srgb_to_linear(reference)
                        } else {
                            reference
                        };
                        maximum = maximum.max(error / reference_linear.abs().max(1.0));
                    }
                }
                let limit = if case.vardct { 1e-4 } else { 3e-6 };
                assert!(
                    maximum < limit,
                    "{} presentation {index} {orientation:?} planar {planar}: scaled linear/alpha error {maximum}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn multiply_clamps_foreground_and_retains_extended_reference_values() {
    let Some(backend) = backend() else {
        return;
    };
    let case = Case {
        name: "gray_clamp",
        hex: include_str!("../../../test-data/composition_gray_clamp.jxl.hex"),
        format: LosslessModularFormat::Gray,
        bits: 8,
        vardct: false,
    };
    let bytes = encoded(&case);
    let djxl = djxl_frames(&case, &bytes);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = decoder
        .open(
            &bytes,
            GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                jxl_wgpu_decode::vardct_rgb8_format().color_spec,
            ))
            .unwrap(),
        )
        .unwrap();
    let mut outputs = Vec::new();
    while let Some(frame) = session.next_frame().unwrap() {
        let data = read_output(&backend, &frame.output().outputs[0]);
        let gray = data
            .chunks_exact(16)
            .map(|p| f32::from_le_bytes(p[..4].try_into().unwrap()))
            .collect::<Vec<_>>();
        if let Some(djxl) = &djxl {
            let quantized = gray
                .iter()
                .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u16)
                .collect::<Vec<_>>();
            let error = quantized
                .iter()
                .zip(&djxl[outputs.len()])
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                error <= 1,
                "presentation {}: djxl quantization error {error}",
                outputs.len()
            );
        }
        outputs.push(gray);
    }
    // Original coordinate (253,6) maps to (10,253) under orientation 6. The deterministic
    // sources are 124/255 and 226/255 at the two-layer Add, then 191/255 for Multiply.
    // Clamp applies to 191/255, so the result remains above one until integer presentation.
    let index = 253 * 17 + 10;
    assert!((outputs[1][index] - 350.0 / 255.0).abs() < 2e-7);
    assert!((outputs[2][index] - 350.0 * 191.0 / (255.0 * 255.0)).abs() < 3e-7);
    // jxl 0.6.0 instead clamps the reference (350/255) in this case and yields 191/255.
    // The checked-in fixture plus the analytic value and djxl oracle guard the normative result.
}

#[test]
fn composition_admission_dependencies_and_cancellation_release_owned_bytes() {
    let Some(backend) = backend() else {
        return;
    };
    for case in composition_cases()
        .into_iter()
        .filter(|case| matches!(case.name, "rgba16" | "vardct_dc" | "mixed"))
    {
        let bytes = encoded(&case);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(4096).unwrap()),
        );
        let mut session = incremental(&decoder, &bytes, request(&case));
        let memory = backend.transient_memory_budget();
        let guard = memory
            .try_reserve(memory.snapshot().available_bytes)
            .unwrap();
        for _ in 0..2 {
            let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
            assert_eq!(progress.submitted, 0);
            assert!(matches!(
                progress.backpressure,
                Some(PrefetchBackpressure::Memory(_))
            ));
        }
        drop(guard);
        let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
        assert_eq!(progress.queued, 1);
        assert_eq!(
            progress.backpressure,
            Some(PrefetchBackpressure::FrameDependency { index: 0 })
        );
        let first = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let retained = first.output().outputs[0].buffer.clone();
        // The previous caller-visible output can stay alive while the reference-dependent next
        // presentation is submitted. Dependency admission is separate from frame-slot ownership.
        let progress = session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
        assert_eq!(progress.queued, 1);
        assert_eq!(
            progress.backpressure,
            Some(PrefetchBackpressure::FrameDependency { index: 1 })
        );
        drop(first);
        drop(session);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while memory.snapshot().reserved_bytes != retained.size()
            && std::time::Instant::now() < deadline
        {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, retained.size());
        drop(retained);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
