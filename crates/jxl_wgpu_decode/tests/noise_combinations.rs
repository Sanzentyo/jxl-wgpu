#![cfg(not(target_arch = "wasm32"))]

#[path = "support/noise.rs"]
mod fixtures;
#[path = "common/extra_channel_oracle.rs"]
mod oracle;
#[path = "support/planes.rs"]
mod planes;

use std::num::{NonZeroU64, NonZeroUsize};

use fixtures::{encoded, zero_noise};
use jxl_gpu_bitstream::{CodestreamInventory, FrameEncoding, FrameType};
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine};

fn inventory(data: &[u8]) -> CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

fn request(extra: Option<u32>) -> GpuOutputRequest {
    match extra {
        Some(index) => GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NormalizedUnsigned,
        )
        .unwrap()
        .with_extra_channel(index)
        .unwrap(),
        None => GpuOutputRequest::color(PixelFormat::rgb_f32(
            RgbChannelOrder::Rgba,
            false,
            jxl_wgpu_decode::vardct_rgb8_format().color_spec,
        ))
        .unwrap(),
    }
}

struct Renderer {
    backend: WgpuBackend,
    whole: GpuDecoder<WgpuDecodeEngine>,
    bounded: GpuDecoder<WgpuDecodeEngine>,
}

impl Renderer {
    fn new() -> Option<Self> {
        let backend = match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        })) {
            Ok(backend) => backend,
            Err(jxl_wgpu::Error::NoAdapter) => return None,
            Err(error) => panic!("noise combinations adapter: {error}"),
        };
        let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
        let bounded = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        );
        Some(Self {
            backend,
            whole,
            bounded,
        })
    }

    fn released(&self) {
        // Cancellation drops the frontend immediately; map-completion callbacks still own
        // submitted planes and source spans until the runtime-neutral poller retires them.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while (self.backend.submission_poller().in_flight() != 0
            || self
                .backend
                .transient_memory_budget()
                .snapshot()
                .reserved_bytes
                != 0)
            && std::time::Instant::now() < deadline
        {
            self.backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::yield_now();
        }
        assert_eq!(self.backend.submission_poller().in_flight(), 0);
        for decoder in [&self.whole, &self.bounded] {
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                0
            );
        }
    }

    fn decode(&self, data: &[u8], request: GpuOutputRequest, fragmented: bool) -> Vec<u32> {
        let mut session = if fragmented {
            planes::open_fragmented(&self.bounded, data, request)
        } else {
            self.whole.open(data, request).unwrap()
        };
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let words = planes::read(&self.backend, &frame.output().outputs[0]);
        drop(frame);
        assert!(
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .is_none()
        );
        drop(session);
        self.released();
        words
    }
}

fn max_error(actual: &[u32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&word, &expected)| {
            let actual = f32::from_bits(word);
            assert!(actual.is_finite() && expected.is_finite());
            (actual - expected).abs()
        })
        .fold(0.0_f32, f32::max)
}

fn check(actual: &[u32], expected: &[f32], limit: f32, label: &str) {
    let error = max_error(actual, expected);
    eprintln!("{label}: maxAE={error}");
    assert!(error < limit, "{label}: maxAE={error}, limit={limit}");
}

fn oxide_rgba(data: &[u8], pixels: usize) -> Vec<f32> {
    // A third, pinned, test-only oracle: both libjxl and Rust jxl have vertical-subsampling
    // restoration defects. The scalar Gaborish check below independently audits this selection.
    let mut image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
        jxl_oxide::RenderingIntent::Relative,
    ));
    let render = image.render_frame(0).unwrap();
    let frame = render.image_all_channels();
    assert_eq!(frame.buf().len(), pixels * 3);
    frame
        .buf()
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], p[2], 1.0])
        .collect()
}

fn gaborish_rgb(rgb: &[f32], width: usize, height: usize) -> Vec<f32> {
    // Default Gaborish uses the same normalized kernel for every component. It commutes
    // with the affine YCbCr-to-RGB transform. No clipping occurs for these SDR inputs.
    // Evaluate in f64 using an independent nine-tap loop, with one-pixel symmetric borders.
    let axial = 1.1 * 0.104699568;
    let diagonal = 1.1 * 0.055680538;
    let divisor = 1.0 + 4.0 * (axial + diagonal);
    let mut output = vec![1.0; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let mut sum = 0.0;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        let sx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                        let sy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                        let weight = match (dx == 0, dy == 0) {
                            (true, true) => 1.0,
                            (false, false) => diagonal,
                            _ => axial,
                        };
                        sum += f64::from(rgb[(sy * width + sx) * 4 + channel]) * weight;
                    }
                }
                output[(y * width + x) * 4 + channel] = (sum / divisor) as f32;
            }
        }
    }
    output
}

#[test]
fn jpeg_restoration_and_noise_execute_after_component_expansion() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for (sampling, shifts, vertical) in [
        ("444", [0, 0, 0], false),
        ("422", [0, 2, 0], false),
        ("440", [0, 3, 0], true),
        ("420", [0, 1, 0], true),
        ("gray", [0, 0, 0], false),
    ] {
        let base = encoded(&format!("jpeg_{sampling}"));
        let base = zero_noise(&base, &inventory(&base), None);
        let base_rgb = oracle::rust_planes(&base).0;
        let gab_rgb = gaborish_rgb(&base_rgb, 257, 17);
        for (filter, gab, iterations) in [
            ("gab", true, 0),
            ("epf1", false, 1),
            ("gab_epf2", true, 2),
            ("gab_epf3", true, 3),
        ] {
            let name = format!("jpeg_{sampling}_{filter}");
            let data = encoded(&name);
            let inventory = inventory(&data);
            assert_eq!(
                (inventory.image_header.width, inventory.image_header.height),
                (257, 17)
            );
            assert!(!inventory.image_header.xyb_encoded);
            assert_eq!(inventory.image_header.grayscale, sampling == "gray");
            assert_eq!(inventory.frames.len(), 1);
            let frame = &inventory.frames[0];
            assert_eq!(frame.encoding, FrameEncoding::VarDct);
            assert!(frame.do_ycbcr);
            assert_eq!(frame.jpeg_upsampling, shifts);
            assert_eq!(frame.noise_seed, [1, 0]);
            let jxl_gpu_bitstream::RestorationFilterInventory::Custom { gaborish, epf } =
                frame.restoration_filter
            else {
                panic!("custom filters")
            };
            assert_eq!(
                gaborish,
                if gab {
                    jxl_gpu_bitstream::GaborishInventory::Default
                } else {
                    jxl_gpu_bitstream::GaborishInventory::Disabled
                }
            );
            if iterations == 0 {
                assert_eq!(
                    epf,
                    jxl_gpu_bitstream::EdgePreservingFilterInventory::Disabled
                );
            } else {
                let jxl_gpu_bitstream::EdgePreservingFilterInventory::Enabled {
                    iterations: actual,
                    sharp_lut: Some(lut),
                    sigma: Some(sigma),
                    ..
                } = epf
                else {
                    panic!("active EPF")
                };
                assert_eq!(actual, iterations);
                assert!(lut.iter().all(|v| v.to_f32() == 1.0));
                assert_eq!(sigma.quant_mul.unwrap().to_f32(), 8.0);
            }
            let mut noisy = None;
            for zero in [false, true] {
                let bytes = if zero {
                    zero_noise(&data, &inventory, None)
                } else {
                    data.clone()
                };
                // Gray-to-RGB encoding conversion in jxl-oxide replicates/converts one gray
                // channel; it does not expose the same three noisy YCbCr components returned
                // by libjxl/Rust jxl's RGBA pixel-format request. Keep those two gray oracles.
                let oxide = (sampling != "gray").then(|| oxide_rgba(&bytes, 257 * 17));
                let rust = (!vertical).then(|| oracle::rust_planes(&bytes).0);
                // Native libjxl's fast render pipeline disagrees with scalar Gaborish on
                // vertically subsampled input for Gaborish alone and Gaborish+EPF3.
                let native = if !vertical || iterations == 1 || iterations == 2 {
                    oracle::libjxl_output(&bytes, &[])
                } else {
                    None
                };
                let actual = renderer.decode(&bytes, request(None), false);
                assert_eq!(
                    renderer.decode(&bytes, request(None), true),
                    actual,
                    "{name}: fragmented"
                );
                if let Some(reference) = oxide {
                    check(
                        &actual,
                        &reference,
                        1e-5,
                        &format!("{name} zero={zero} oxide"),
                    );
                }
                if let Some(reference) = rust {
                    check(&actual, &reference, 1e-5, &format!("{name} Rust"));
                }
                if let Some(reference) = native {
                    check(&actual, &reference, 1.0 / 1024.0, &format!("{name} native"));
                }
                if zero {
                    if iterations == 0 {
                        check(&actual, &gab_rgb, 1e-5, &format!("{name} scalar Gaborish"));
                    } else {
                        let unfiltered = if gab { &gab_rgb } else { &base_rgb };
                        assert!(
                            max_error(&actual, unfiltered) > 0.01,
                            "{name}: EPF must actually filter"
                        );
                    }
                    assert_ne!(
                        noisy.as_ref(),
                        Some(&actual),
                        "{name}: noise must change output"
                    );
                } else {
                    noisy = Some(actual);
                }
            }
        }
    }
}

#[test]
fn lf_noise_survives_each_dependency_and_preserves_alpha_and_depth() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for name in [
        "lf_modular_gab0",
        "lf_modular_gab1",
        "lf_vardct_gab0",
        "lf_vardct_gab1",
        "lf_nested_modular_gab1",
        "lf_nested_vardct_gab1",
        "lf_progressive_ac",
    ] {
        let data = encoded(name);
        let inventory = inventory(&data);
        let image = &inventory.image_header;
        let pixels = image.width as usize * image.height as usize;
        let levels = if name.contains("nested") || name == "lf_progressive_ac" {
            2
        } else {
            1
        };
        assert_eq!(inventory.frames.len(), levels + 1);
        assert_eq!(
            image.extra_channels.len(),
            if name == "lf_progressive_ac" { 0 } else { 2 }
        );
        assert_eq!(
            inventory.frames[0].encoding,
            if name.contains("modular") || name == "lf_progressive_ac" {
                FrameEncoding::Modular
            } else {
                FrameEncoding::VarDct
            }
        );
        for (index, frame) in inventory.frames.iter().enumerate() {
            let level = (levels - index) as u32;
            assert_eq!(frame.lf_level, level);
            assert_eq!(
                frame.color_sample_extent(),
                Some((
                    image.width.div_ceil(1 << (3 * level)),
                    image.height.div_ceil(1 << (3 * level)),
                ))
            );
            assert_eq!(
                frame.lf_source_frame,
                index.checked_sub(1).map(|v| v as u32)
            );
            assert_eq!(
                frame.noise_seed,
                if index < levels {
                    [0, index as u32 + 1]
                } else {
                    [1, 0]
                }
            );
            assert_eq!(
                frame.frame_type,
                if index < levels {
                    FrameType::LowFrequency
                } else {
                    FrameType::Regular
                }
            );
            assert_eq!(frame.flags & 1, u64::from(index < levels));
        }
        let mut colors = Vec::<Vec<u32>>::new();
        let mut extras = Vec::new();
        // Toggle each model independently, including both nested LF producers. The regular
        // frame has no noise of its own, so every effect must arrive through a resident LF slot.
        for mask in 0..1_usize << levels {
            let mut bytes = data.clone();
            for index in 0..levels {
                if mask & (1 << index) == 0 {
                    bytes = zero_noise(&bytes, &inventory, Some(index));
                }
            }
            let (rust_color, rust_extras) = oracle::rust_planes(&bytes);
            // This high-contrast chain amplifies native IDCT rounding in dark sRGB samples.
            // Keep the tight Rust sRGB comparison and independently compare linear native
            // output, as for the existing custom-correlation corpus. Do not loosen either bound.
            let native = if name == "lf_progressive_ac" {
                let jxl_gpu_formats::ColorSpecification::Defined(mut color) =
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec
                else {
                    panic!("defined color")
                };
                color.transfer = jxl_gpu_formats::TransferFunction::Linear;
                let linear_request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_gpu_formats::ColorSpecification::Defined(color),
                ))
                .unwrap();
                let linear = renderer.decode(&bytes, linear_request.clone(), false);
                assert_eq!(renderer.decode(&bytes, linear_request, true), linear);
                if let Some(reference) = oracle::libjxl_output(&bytes, &["--linear"]) {
                    check(
                        &linear,
                        &reference,
                        1.0 / 1024.0,
                        &format!("{name} mask={mask} native linear"),
                    );
                }
                None
            } else {
                oracle::libjxl_planes(&bytes, pixels, image.extra_channels.len())
            };
            for extra in
                std::iter::once(None).chain((0..image.extra_channels.len() as u32).map(Some))
            {
                let actual = renderer.decode(&bytes, request(extra), false);
                assert_eq!(
                    renderer.decode(&bytes, request(extra), true),
                    actual,
                    "{name}: mask={mask}, extra={extra:?}"
                );
                let rust = extra.map_or(&rust_color, |index| &rust_extras[index as usize]);
                let limit = if extra.is_none() { 1e-4 } else { 1e-6 };
                check(
                    &actual,
                    rust,
                    limit,
                    &format!("{name} mask={mask} extra={extra:?} Rust"),
                );
                if let Some((color, extras)) = &native {
                    check(
                        &actual,
                        extra.map_or(color, |index| &extras[index as usize]),
                        if extra.is_none() { 1.0 / 1024.0 } else { 1e-6 },
                        &format!("{name} native"),
                    );
                }
                if let Some(index) = extra {
                    if mask == 0 {
                        extras.push(actual);
                    } else {
                        assert_eq!(
                            actual, extras[index as usize],
                            "{name}: noise changed extra"
                        );
                    }
                } else {
                    if mask != 0 {
                        assert!(
                            actual
                                .chunks_exact(4)
                                .zip(colors[0].chunks_exact(4))
                                .all(|(a, b)| a[3] == b[3]),
                            "{name}: noise changed alpha"
                        );
                    }
                    colors.push(actual);
                }
            }
        }
        for mask in 1..1_usize << levels {
            for index in 0..levels {
                if mask & (1 << index) != 0 {
                    let other = colors[mask ^ (1 << index)]
                        .iter()
                        .map(|&v| f32::from_bits(v))
                        .collect::<Vec<_>>();
                    assert!(
                        max_error(&colors[mask], &other) > 1e-5,
                        "{name}: LF model {index} must affect mask {mask}"
                    );
                }
            }
        }
    }
}

#[test]
fn cancelling_noisy_lf_dependencies_releases_resident_planes_and_fragmented_input() {
    use std::task::{Context, Poll, Waker};
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for name in [
        "lf_nested_modular_gab1",
        "lf_nested_vardct_gab1",
        "lf_progressive_ac",
    ] {
        let data = encoded(name);
        let mut session = planes::open_fragmented(&renderer.bounded, &data, request(None));
        drop(
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap(),
        );
        let count = session.submission_session().submissions_per_frame();
        drop(session);
        renderer.released();
        assert!(count >= 8);
        for stop in [1, count / 2, count - 1] {
            let mut session = planes::open_fragmented(&renderer.bounded, &data, request(None));
            session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            let mut context = Context::from_waker(Waker::noop());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while session.submission_session().submissions_per_frame() < stop {
                assert!(
                    std::time::Instant::now() < deadline,
                    "{name}: cancellation timeout"
                );
                assert!(matches!(
                    session.poll_next_frame(&mut context),
                    Poll::Pending
                ));
                std::thread::yield_now();
            }
            assert!(
                renderer
                    .bounded
                    .engine()
                    .in_flight_memory_stats()
                    .reserved_bytes
                    > 0
            );
            drop(session);
            renderer.released();
        }
    }
}
