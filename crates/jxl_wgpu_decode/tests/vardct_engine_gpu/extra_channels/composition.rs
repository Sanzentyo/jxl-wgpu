use super::*;
use jxl_gpu_bitstream::SampleBitDepth;
use jxl_wgpu_decode::AlphaOutputPolicy;
use jxl_wgpu_encode::LosslessModularFormat;

fn cases() -> [(&'static str, &'static str); 7] {
    [
        (
            "rgb",
            include_str!("../../../test-data/composition_extras_rgb.jxl.hex"),
        ),
        (
            "gray",
            include_str!("../../../test-data/composition_extras_gray.jxl.hex"),
        ),
        (
            "vardct",
            include_str!("../../../test-data/composition_extras_vardct.jxl.hex"),
        ),
        (
            "distributed",
            include_str!("../../../test-data/composition_extras_distributed.jxl.hex"),
        ),
        (
            "resampled",
            include_str!("../../../test-data/composition_extras_resampled.jxl.hex"),
        ),
        (
            "vardct_resampled",
            include_str!("../../../test-data/composition_extras_vardct_resampled.jxl.hex"),
        ),
        (
            "data",
            include_str!("../../../test-data/composition_extras_data.jxl.hex"),
        ),
    ]
}

fn reference(
    data: &[u8],
    pixels: usize,
    count: usize,
    keep: bool,
) -> Option<Vec<oracle::FloatPlanes>> {
    let mut options = vec!["--preserve-alpha"];
    if keep {
        options.push("--keep-orientation");
    }
    oracle::libjxl_output(data, &options).map(|values| {
        assert_eq!(values.len(), 6 * pixels * (4 + count));
        values
            .chunks_exact(pixels * (4 + count))
            .map(|frame| {
                (
                    frame[..pixels * 4].to_vec(),
                    frame[pixels * 4..]
                        .chunks_exact(pixels)
                        .map(<[f32]>::to_vec)
                        .collect(),
                )
            })
            .collect()
    })
}

fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs() / expected.abs().max(1.0);
        assert!(
            error < tolerance,
            "{name}/{index}: {actual} vs {expected}, error {error}"
        );
    }
}

#[test]
fn all_extra_channels_compose_with_independent_references_and_alpha_selectors() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in cases() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let count = image.extra_channels.len();
        let pixels = (image.width * image.height) as usize;
        let native = reference(&data, pixels, count, false);
        // Rust jxl 0.6 has the documented dimension-shift and clamped-Multiply defects.
        // Its first Replace presentation independently checks all unshifted input planes;
        // libjxl checks the complete reference chain, including extended clamped-Mul values.
        let rust = (!name.contains("resampled")).then(|| {
            let mut frames = oracle::rust_frame_planes(&data);
            frames.truncate(1);
            frames
        });
        let Some(expected) = native.as_ref().or(rust.as_ref()) else {
            continue;
        };
        let color = associated::floating_request(AlphaOutputPolicy::Preserve, false, false);
        let whole = associated::decode(&backend, &data, color.clone(), false);
        assert_eq!(
            whole,
            associated::decode(&backend, &data, color, true),
            "{name}: color input windows"
        );
        assert_eq!(whole.len(), 6);
        let tolerance = if matches!(name, "vardct" | "distributed" | "vardct_resampled") {
            3e-3
        } else {
            8e-6
        };
        for (frame, (_, bytes)) in whole.iter().enumerate() {
            if let Some(expected) = expected.get(frame) {
                compare(
                    &format!("{name}/frame{frame}/color"),
                    &oracle::floats(bytes),
                    &expected.0,
                    tolerance,
                );
            }
            if let Some(rust) = rust.as_ref().and_then(|frames| frames.get(frame)) {
                compare(
                    &format!("{name}/frame{frame}/Rust-color"),
                    &oracle::floats(bytes),
                    &rust.0,
                    tolerance,
                );
            }
        }
        for index in 0..count {
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap();
            let whole = associated::decode(&backend, &data, request.clone(), false);
            assert_eq!(
                whole,
                associated::decode(
                    &backend,
                    &data,
                    request.with_alpha_output_policy(AlphaOutputPolicy::Associated),
                    true
                ),
                "{name}/extra{index}: bounded numeric planes"
            );
            for (frame, (_, bytes)) in whole.iter().enumerate() {
                if let Some(expected) = expected.get(frame) {
                    compare(
                        &format!("{name}/frame{frame}/extra{index}"),
                        &oracle::floats(bytes),
                        &expected.1[index],
                        4e-6,
                    );
                }
                if let Some(rust) = rust.as_ref().and_then(|frames| frames.get(frame)) {
                    compare(
                        &format!("{name}/frame{frame}/Rust-extra{index}"),
                        &oracle::floats(bytes),
                        &rust.1[index],
                        4e-6,
                    );
                }
            }
        }
    }
}

#[test]
fn composed_extra_native_quantization_and_orientation_follow_the_selected_declaration() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in cases()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "gray" | "vardct_resampled"))
    {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let Some(expected) = reference(
            &data,
            (image.width * image.height) as usize,
            image.extra_channels.len(),
            true,
        ) else {
            continue;
        };
        for (index, declaration) in image.extra_channels.iter().enumerate() {
            let SampleBitDepth::Integer { bits_per_sample } = declaration.bit_depth else {
                unreachable!()
            };
            let request = GpuOutputRequest::numeric(
                LosslessModularFormat::Gray
                    .pixel_format(bits_per_sample as u8)
                    .unwrap(),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Keep);
            let whole = associated::decode(&backend, &data, request.clone(), false);
            assert_eq!(
                whole,
                associated::decode(&backend, &data, request, true),
                "{name}/extra{index}: native input windows"
            );
            let sample_bytes = bits_per_sample.div_ceil(8) as usize;
            let maximum = ((1u32 << bits_per_sample) - 1) as f32;
            for (frame, (layout, bytes)) in whole.iter().enumerate() {
                assert_eq!(layout.extent, Extent2d::new(image.width, image.height));
                for (pixel, bytes) in bytes.chunks_exact(sample_bytes).enumerate() {
                    let actual = if sample_bytes == 1 {
                        u32::from(bytes[0])
                    } else {
                        u32::from(u16::from_le_bytes(bytes.try_into().unwrap()))
                    };
                    let expected = (expected[frame].1[index][pixel].clamp(0.0, 1.0) * maximum + 0.5)
                        .floor() as u32;
                    assert!(
                        actual.abs_diff(expected) <= 1,
                        "{name}/frame{frame}/extra{index}/{pixel}: {actual} vs {expected}"
                    );
                    assert!(actual <= maximum as u32);
                }
            }
        }
    }
}

#[test]
fn composed_color_output_uses_the_first_alpha_after_independent_channel_blending() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in cases()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "gray" | "vardct"))
    {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let Some(expected) = reference(
            &data,
            (image.width * image.height) as usize,
            image.extra_channels.len(),
            true,
        ) else {
            continue;
        };
        let source_associated = name == "gray";
        for policy in [
            AlphaOutputPolicy::Preserve,
            AlphaOutputPolicy::Associated,
            AlphaOutputPolicy::Unassociated,
        ] {
            let request = associated::floating_request(policy, true, true);
            let frames = associated::decode(&backend, &data, request, true);
            assert_eq!(frames.len(), expected.len());
            for ((layout, bytes), (reference, _)) in frames.iter().zip(&expected) {
                let mut values = associated::unpack(layout, bytes, true);
                let mut expected = reference.clone();
                associated::linearize(&mut expected);
                associated::associate(&mut expected, policy, source_associated);
                if source_associated && policy == AlphaOutputPolicy::Unassociated {
                    // Evaluate reconstruction error before division by near-zero alpha.
                    for (actual, expected) in values
                        .as_chunks_mut::<4>()
                        .0
                        .iter_mut()
                        .zip(expected.as_chunks_mut::<4>().0.iter_mut())
                    {
                        let scale = expected[3].max(1.0 / 67108864.0);
                        for channel in 0..3 {
                            actual[channel] *= scale;
                            expected[channel] *= scale;
                        }
                    }
                }
                compare(
                    name,
                    &values,
                    &expected,
                    if source_associated { 2e-5 } else { 3e-3 },
                );
            }
        }
    }
}

#[test]
fn all_channel_references_and_abandoned_work_remain_accounted_until_completion() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in cases()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "rgb" | "distributed" | "resampled"))
    {
        let data = encoded(hex);
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(1024).unwrap()),
        );
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
        .with_extra_channel(8)
        .unwrap()
        .with_max_frame_slots(NonZeroUsize::new(3).unwrap());
        let mut session = resampled::fragmented(&decoder, &data, request);
        let memory = backend.transient_memory_budget();
        let guard = memory
            .try_reserve(memory.snapshot().available_bytes)
            .unwrap();
        for _ in 0..2 {
            let progress = session.prefetch(NonZeroUsize::new(3).unwrap()).unwrap();
            assert_eq!(progress.submitted, 0);
            assert!(
                matches!(progress.backpressure, Some(PrefetchBackpressure::Memory(_))),
                "{name}: {progress:?}"
            );
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
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while memory.snapshot().reserved_bytes != retained.size()
            && std::time::Instant::now() < deadline
        {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            memory.snapshot().reserved_bytes,
            retained.size(),
            "{name}: hidden reference or pending surface leaked"
        );
        drop(retained);
        assert_eq!(memory.snapshot().reserved_bytes, 0);
    }
}
