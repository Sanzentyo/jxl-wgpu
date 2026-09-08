#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "support/preview.rs"]
mod fixtures;
#[allow(dead_code)]
#[path = "common/extra_channel_oracle.rs"]
mod oracle;
#[path = "support/planes.rs"]
mod planes;

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use fixtures::inventory;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{
    AnimationMetadata, FrameExecutionPlan, FrameMetadata, GpuDecoder, GpuOutputRequest,
    ImageSelection, ImageSelectionError, NumericSampleMapping, OrientationPolicy,
    SelectedImageInventory, WgpuDecodeEngine,
};

fn source(name: &str) -> Vec<u8> {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("test-data/{name}.jxl.hex")),
    )
    .unwrap()
    .split_whitespace()
    .collect::<String>();
    text.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}

fn fixture(name: &str) -> Vec<u8> {
    source(&format!("preview/{name}"))
}

fn cases() -> Vec<(String, &'static str)> {
    let mut cases = Vec::new();
    for mode in ["modular", "vardct"] {
        cases.extend([
            (mode.to_owned(), "noise/vardct_257x17"),
            (format!("nonfinal_{mode}"), "noise/vardct_257x17"),
            (
                format!("main_modular_preview_{mode}"),
                "noise/modular_257x17",
            ),
            (format!("lf_{mode}"), "noise/lf_progressive_ac"),
            (format!("animation_{mode}"), "noise/mixed_frames"),
        ]);
        for div8 in 0..=1 {
            for ratio in 0..8 {
                if div8 != 0 || ratio != 0 {
                    cases.push((
                        format!("ratio_{mode}_{div8}_{ratio}"),
                        "noise/vardct_257x17",
                    ));
                }
            }
        }
    }
    cases.extend(
        [
            ("alpha_modular", "extras_rgba"),
            ("alpha_vardct", "vardct_extras_rgba"),
            ("rgb_modular", "noise/modular_rgb_group256"),
            ("rgb_vardct", "noise/vardct_rgb_257x17"),
            ("jpeg_420", "noise/jpeg_420"),
            ("float_vardct", "noise/vardct_rgb_float32_up4"),
            ("resampled_modular", "noise/modular_up4"),
            ("resampled_vardct", "noise/vardct_up4"),
        ]
        .map(|(name, original)| (name.to_owned(), original)),
    );
    assert_eq!(cases.len(), 48);
    cases
}

fn request(selection: ImageSelection) -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_image_selection(selection)
}

type Decoded = (AnimationMetadata, Vec<(FrameMetadata, Vec<u32>)>);

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
            Err(error) => panic!("preview adapter: {error}"),
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

    fn decode(&self, data: &[u8], request: GpuOutputRequest, fragmented: bool) -> Decoded {
        let mut session = if fragmented {
            planes::open_fragmented(&self.bounded, data, request)
        } else {
            self.whole.open(data, request).unwrap()
        };
        let metadata = session.metadata().clone();
        let mut frames = Vec::new();
        while let Some(frame) = pollster::block_on(session.next_frame_async()).unwrap() {
            frames.push((
                frame.metadata.clone(),
                planes::read(&self.backend, &frame.output().outputs[0]),
            ));
        }
        drop(session);
        self.released();
        (metadata, frames)
    }

    fn check_fragmented(&self, data: &[u8], request: GpuOutputRequest) -> Decoded {
        let whole = self.decode(data, request.clone(), false);
        assert_eq!(
            self.decode(data, request, true),
            whole,
            "bounded transport changes output"
        );
        whole
    }
}

fn check(actual: &[u32], expected: &[f32], limit: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&word, &value)| {
            let actual = f32::from_bits(word);
            assert!(actual.is_finite() && value.is_finite(), "{label}");
            (actual - value).abs()
        })
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: maxAE={error}");
    assert!(error < limit, "{label}: maxAE={error}, limit={limit}");
}

#[test]
fn every_preview_geometry_and_color_path_matches_native_with_exact_bounded_output() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for (name, _) in cases() {
        let data = fixture(&name);
        let info = inventory(&data);
        let (width, height) = info.image_header.preview_size.unwrap();
        let mut noisy = None;
        for zero in [false, true] {
            if zero && info.frames[0].flags & 1 == 0 {
                continue;
            }
            let bytes = if zero {
                fixtures::zero_noise(&data)
            } else {
                data.clone()
            };
            let (metadata, frames) =
                renderer.check_fragmented(&bytes, request(ImageSelection::Preview));
            assert_eq!(metadata.extent, Extent2d::new(width, height), "{name}");
            assert!(!metadata.is_animation(), "{name}");
            assert_eq!(metadata.extra_channels, info.image_header.extra_channels);
            assert_eq!(frames.len(), 1, "{name}");
            let (frame, actual) = &frames[0];
            assert_eq!(frame.index, 0);
            assert_eq!(frame.duration.ticks, 0);
            assert_eq!(frame.presentation_ticks, 0);
            assert_eq!(frame.timecode, None);
            assert!(frame.is_last && frame.is_keyframe);
            if let Some(reference) = oracle::libjxl_output(&bytes, &["--preview"]) {
                check(
                    actual,
                    &reference,
                    1.0 / 1024.0,
                    &format!("{name} zero={zero} native preview"),
                );
            }
            if zero {
                assert_ne!(
                    noisy.as_ref(),
                    Some(actual),
                    "{name}: preview noise must be observable"
                );
            } else {
                noisy = Some(actual.clone());
            }
        }
    }
}

#[test]
fn main_frames_keep_noise_lf_references_composition_and_timing_after_preview() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for (name, original) in cases()
        .into_iter()
        .filter(|(name, _)| !name.starts_with("ratio_"))
    {
        let data = fixture(&name);
        let mut original = source(original);
        if name.starts_with("lf_") {
            // Independent seed-equivalent control without a preview: repeat the independent
            // root LF frame before the original chain. Its first result is overwritten, but
            // it advances the nonvisible counter once, just as a preview does. This also
            // avoids the Rust oracle's preview counter/skip behavior.
            let parsed = jxl_gpu_bitstream::parse(&original, Default::default()).unwrap();
            let bytes = parsed.codestream();
            let info = inventory(bytes);
            let start = info.frames[0].header_bits.offset as usize / 8;
            let end = info.frames[1].header_bits.offset as usize / 8;
            original = [&bytes[..end], &bytes[start..]].concat();
        }
        let actual = renderer.check_fragmented(&data, request(ImageSelection::Main));
        // Independent codestream with the same main entropy and noise state, without a preview.
        let expected = renderer.decode(&original, request(ImageSelection::Main), false);
        assert_eq!(
            actual, expected,
            "{name}: preview changed main reconstruction/timing"
        );
        let rust = oracle::rust_frame_planes(&original);
        assert_eq!(actual.1.len(), rust.len());
        for ((_, pixels), (color, _)) in actual.1.iter().zip(rust) {
            check(pixels, &color, 1e-4, &format!("{name} main Rust"));
        }
        // The existing progressive-AC fixture magnifies native sRGB IDCT rounding in dark
        // samples. Its independent native gate uses linear F32, as in the LF noise corpus.
        let linear = name.starts_with("lf_");
        let mut output = request(ImageSelection::Main);
        if linear {
            let jxl_gpu_formats::ColorSpecification::Defined(mut color) =
                output.format().color_spec
            else {
                panic!("defined color");
            };
            color.transfer = jxl_gpu_formats::TransferFunction::Linear;
            output = GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                jxl_gpu_formats::ColorSpecification::Defined(color),
            ))
            .unwrap();
        }
        if let Some(native) = oracle::libjxl_output(&data, if linear { &["--linear"] } else { &[] })
        {
            let native_actual = if linear {
                renderer.check_fragmented(&data, output)
            } else {
                actual
            };
            let pixels =
                native_actual.0.extent.width as usize * native_actual.0.extent.height as usize;
            let stride = pixels * (4 + native_actual.0.extra_channels.len());
            assert_eq!(native.len(), native_actual.1.len() * stride);
            for ((_, actual), expected) in native_actual.1.iter().zip(native.chunks_exact(stride)) {
                check(
                    actual,
                    &expected[..pixels * 4],
                    1.0 / 1024.0,
                    &format!("{name} main native"),
                );
            }
        }
    }
}

#[test]
fn orientation_uses_the_selected_image_extent_and_keep_retains_encoded_coordinates() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for name in ["modular", "vardct"] {
        let data = fixture(name);
        for selection in [ImageSelection::Preview, ImageSelection::Main] {
            let baseline = renderer.decode(&data, request(selection), false);
            for orientation in 1..=8 {
                let bytes = fixtures::orient(&data, orientation);
                let keep = request(selection).with_orientation_policy(OrientationPolicy::Keep);
                assert_eq!(
                    renderer.check_fragmented(&bytes, keep),
                    baseline,
                    "{name}: Keep {orientation}"
                );
                let actual = renderer.check_fragmented(&bytes, request(selection));
                let size = baseline.0.extent;
                assert_eq!(
                    actual.0.extent,
                    if orientation < 5 {
                        size
                    } else {
                        Extent2d::new(size.height, size.width)
                    }
                );
                let options = if selection == ImageSelection::Preview {
                    &["--preview"][..]
                } else {
                    &[]
                };
                if let Some(native) = oracle::libjxl_output(&bytes, options) {
                    check(
                        &actual.1[0].1,
                        &native,
                        1.0 / 1024.0,
                        &format!("{name} {selection:?} orientation={orientation}"),
                    );
                }
            }
        }
    }
}

#[test]
fn preview_alpha_selection_preserves_the_declared_extra_plane() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for (name, original) in [
        ("alpha_modular", "extras_rgba"),
        ("alpha_vardct", "vardct_extras_rgba"),
    ] {
        let data = fixture(name);
        let original = source(original);
        let (_, extras) = oracle::rust_planes(&original);
        for (index, expected) in extras.iter().enumerate() {
            let output = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap()
            .with_extra_channel(index as u32)
            .unwrap()
            .with_image_selection(ImageSelection::Preview);
            let actual = renderer.check_fragmented(&data, output);
            check(
                &actual.1[0].1,
                expected,
                1e-6,
                &format!("{name} preview extra={index}"),
            );
        }
    }
}

#[test]
fn selected_image_lowering_preserves_physical_ids_ranges_and_dependency_versions() {
    for (name, _) in cases() {
        let source = Arc::new(inventory(&fixture(&name)));
        let original = (*source).clone();
        assert_eq!(source.frames[0].noise_seed, [0, 1]);
        if name.starts_with("lf_") {
            assert_eq!(
                source
                    .frames
                    .iter()
                    .map(|frame| frame.noise_seed)
                    .collect::<Vec<_>>(),
                [[0, 1], [0, 2], [0, 3], [1, 0]]
            );
        }
        assert!(matches!(
            FrameExecutionPlan::negotiate(&source),
            Err(jxl_wgpu_decode::FramePlanError::ImageNotSelected)
        ));
        for selection in [ImageSelection::Preview, ImageSelection::Main] {
            let selected = SelectedImageInventory::new(Arc::clone(&source), selection).unwrap();
            assert_eq!(selected.selection(), selection);
            assert_eq!(selected.source_inventory(), &original);
            let lowered = selected.reconstruction_inventory();
            assert!(lowered.image_header.preview_size.is_none());
            let retained = if selection == ImageSelection::Preview {
                &source.frames[..1]
            } else {
                &source.frames[1..]
            };
            assert_eq!(lowered.frames.len(), retained.len());
            for (position, (frame, original)) in lowered.frames.iter().zip(retained).enumerate() {
                assert_eq!(frame.frame_index, original.frame_index);
                assert_eq!(lowered.frame_position(frame.frame_index), Some(position));
                assert_eq!(frame.header_bits, original.header_bits);
                assert_eq!(frame.toc_bits, original.toc_bits);
                assert_eq!(frame.sections, original.sections);
                assert_eq!(frame.noise_seed, original.noise_seed);
                assert_eq!(frame.lf_source_frame, original.lf_source_frame);
            }
            assert_eq!(lowered.frame_position(u32::MAX), None);
            if selection == ImageSelection::Main {
                assert_eq!(lowered.frame_position(0), None);
            }
            let plan = FrameExecutionPlan::negotiate(lowered).unwrap();
            for node in &plan.nodes {
                if let Some(source) = node.lf_source_frame {
                    let producer = &plan.nodes[lowered.frame_position(source).unwrap()];
                    assert!(producer.lf_last_use.unwrap() >= node.frame_index);
                }
                for reference in node.references.iter().flatten() {
                    assert!(
                        lowered.frame_position(reference.frame_index).unwrap()
                            < lowered.frame_position(node.frame_index).unwrap()
                    );
                }
            }
        }
        assert_eq!(*source, original, "lowering mutated the source");
    }
}

#[test]
fn preview_admission_retry_cancellation_and_missing_image_release_both_budgets() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    let ordinary = source("noise/vardct_257x17");
    assert!(matches!(
        renderer
            .whole
            .open(&ordinary, request(ImageSelection::Preview)),
        Err(jxl_wgpu_decode::Error::ImageSelection(
            ImageSelectionError::MissingPreview
        ))
    ));
    renderer.released();
    for name in ["modular", "vardct", "lf_vardct", "animation_modular"] {
        let data = fixture(name);
        for selection in [ImageSelection::Preview, ImageSelection::Main] {
            let mut session = planes::open_fragmented(&renderer.bounded, &data, request(selection));
            let budget = renderer.backend.transient_memory_budget();
            let held = budget.try_reserve(budget.snapshot().limit_bytes).unwrap();
            let progress = session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
            assert_eq!(progress.submitted, 0);
            assert!(matches!(
                progress.backpressure,
                Some(jxl_wgpu_decode::PrefetchBackpressure::Memory(_))
            ));
            assert_eq!(budget.snapshot().reserved_bytes, held.bytes());
            drop(held);
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            drop(session);
            assert!(
                budget.snapshot().reserved_bytes > 0,
                "caller lease must retain output"
            );
            drop(frame);
            renderer.released();
            let mut session = planes::open_fragmented(&renderer.bounded, &data, request(selection));
            assert_eq!(
                session
                    .prefetch(NonZeroUsize::new(1).unwrap())
                    .unwrap()
                    .submitted,
                1
            );
            drop(session);
            renderer.released();
        }
    }
}

#[test]
fn whole_and_incremental_inventory_agree_across_the_single_preview_boundary() {
    use jxl_gpu_bitstream::{CodestreamInventoryEvent, CodestreamStreamScanner};
    for (name, _) in cases() {
        let data = fixture(&name);
        let expected = inventory(&data);
        for chunk_size in [1, 43] {
            let mut scanner = CodestreamStreamScanner::new(Default::default());
            let mut image = None;
            let mut frames = Vec::new();
            let mut completed = Vec::new();
            let mut ended = false;
            for (index, chunk) in data.chunks(chunk_size).enumerate() {
                let bytes = jxl_gpu_bitstream::StreamSlice::from_shared(Arc::from(chunk));
                let events = scanner
                    .push_chunk((index * chunk_size) as u64, bytes)
                    .unwrap();
                for event in events {
                    match event {
                        CodestreamInventoryEvent::ImageHeader(value) => {
                            image = Some((*value).clone())
                        }
                        CodestreamInventoryEvent::FrameStart(value) => {
                            frames.push((*value).clone())
                        }
                        CodestreamInventoryEvent::FrameEnd { frame_index } => {
                            completed.push(frame_index)
                        }
                        CodestreamInventoryEvent::SectionChunk {
                            frame_index,
                            section,
                            section_offset,
                            bytes,
                        } => {
                            let start = (section.bytes.offset + section_offset) as usize;
                            assert_eq!(bytes.bytes(), &data[start..start + bytes.len()]);
                            assert_eq!(
                                expected.frames[frame_index as usize].sections
                                    [section.toc_index as usize],
                                section
                            );
                        }
                        CodestreamInventoryEvent::End { .. } => panic!("premature end"),
                    }
                }
            }
            for event in scanner.finish_input(data.len() as u64).unwrap() {
                if let CodestreamInventoryEvent::End {
                    codestream_bytes,
                    frame_count,
                } = event
                {
                    assert_eq!(codestream_bytes, data.len() as u64);
                    assert_eq!(frame_count as usize, expected.frames.len());
                    ended = true;
                } else {
                    panic!("unexpected final event");
                }
            }
            assert!(ended, "{name}");
            assert_eq!(image.as_ref(), Some(&expected.image_header), "{name}");
            assert_eq!(frames, expected.frames, "{name}");
            assert_eq!(completed, (0..frames.len() as u32).collect::<Vec<_>>());
            assert!(frames[0].is_preview && frames[1..].iter().all(|frame| !frame.is_preview));
        }
    }
}

#[test]
fn malformed_preview_inventories_cannot_cross_the_image_selection_boundary() {
    use jxl_gpu_bitstream::{FrameBlendMode, FrameType};
    let original = inventory(&fixture("nonfinal_vardct"));
    let invalid: &[fn(&mut jxl_gpu_bitstream::CodestreamInventory)] = &[
        |data| data.frames[0].frame_index = 1,
        |data| data.frames[1].frame_index = 2,
        |data| data.frames[0].is_preview = false,
        |data| data.frames[1].is_preview = true,
        |data| data.image_header.preview_size = None,
        |data| data.image_header.preview_size = Some((0, 27)),
        |data| data.image_header.preview_size = Some((15, 4097)),
        |data| data.frames[0].frame_type = FrameType::LowFrequency,
        |data| data.frames[0].lf_level = 1,
        |data| data.frames[0].lf_source_frame = Some(0),
        |data| data.frames[0].flags |= 0x20,
        |data| data.frames[0].save_as_reference = 1,
        |data| data.frames[0].have_crop = true,
        |data| data.frames[0].x0 = 1,
        |data| data.frames[0].width -= 1,
        |data| data.frames[0].color_blend.mode = FrameBlendMode::Add,
    ];
    for mutate in invalid {
        let mut data = original.clone();
        mutate(&mut data);
        for selection in [ImageSelection::Preview, ImageSelection::Main] {
            assert!(matches!(
                SelectedImageInventory::new(Arc::new(data.clone()), selection),
                Err(ImageSelectionError::InvalidInventory(_))
            ));
        }
    }
    let mut no_main = original;
    no_main.frames.truncate(1);
    assert!(matches!(
        SelectedImageInventory::new(Arc::new(no_main), ImageSelection::Preview),
        Err(ImageSelectionError::MissingMainImage)
    ));
}

#[test]
fn malformed_or_incomplete_preview_streams_release_input_before_any_submission() {
    use jxl_gpu_bitstream::{ContainerStreamEvent, ContainerStreamScanner, StreamSlice};
    let Some(renderer) = Renderer::new() else {
        return;
    };
    let data = fixture("nonfinal_vardct");
    let info = inventory(&data);
    let frame_start = info.frames[0].header_bits.offset as usize;
    for frame_type in 1..4 {
        let mut bytes = data.clone();
        // all_default=0 followed by the two-bit frame type; previews must be Regular.
        for bit in 0..2 {
            let at = frame_start + 1 + bit;
            bytes[at / 8] =
                (bytes[at / 8] & !(1 << (at % 8))) | (((frame_type >> bit) & 1) << (at % 8));
        }
        assert!(
            renderer
                .whole
                .open(&bytes, request(ImageSelection::Preview))
                .is_err()
        );
        let mut stream = renderer
            .bounded
            .stream(request(ImageSelection::Preview))
            .unwrap();
        let event = ContainerStreamEvent::CodestreamChunk {
            logical_offset: 0,
            bytes: StreamSlice::from_shared(Arc::from(bytes)),
        };
        assert!(stream.push_transport_event(&event).is_err());
        assert_eq!(stream.stats().retained_codestream_bytes, 0);
        assert_eq!(stream.stats().retained_spans, 0);
        drop(stream);
        renderer.released();
    }
    for end in [
        2,
        info.frames[1].header_bits.offset as usize / 8 - 1,
        info.frames[1].header_bits.offset as usize / 8,
        data.len() - 1,
    ] {
        assert!(
            renderer
                .whole
                .open(&data[..end], request(ImageSelection::Preview))
                .is_err()
        );
        let mut stream = renderer
            .bounded
            .stream(request(ImageSelection::Preview))
            .unwrap();
        let mut transport = ContainerStreamScanner::new(renderer.bounded.container_stream_limits());
        for chunk in data[..end].chunks(43) {
            for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                stream.push_transport_event(&event).unwrap();
            }
        }
        let mut failed = false;
        for event in transport.finish_input().unwrap() {
            if stream.push_transport_event(&event).is_err() {
                failed = true;
                break;
            }
        }
        assert!(
            failed,
            "a complete preview cannot stand in for a missing main image"
        );
        assert_eq!(stream.stats().retained_codestream_bytes, 0);
        drop(stream);
        renderer.released();
    }
}

#[test]
fn entropy_errors_are_confined_to_the_selected_image_and_release_its_resources() {
    let Some(renderer) = Renderer::new() else {
        return;
    };
    for name in ["modular", "vardct"] {
        let data = fixture(name);
        let info = inventory(&data);
        for (physical, damaged_selection, other) in [
            (0, ImageSelection::Preview, ImageSelection::Main),
            (1, ImageSelection::Main, ImageSelection::Preview),
        ] {
            let mut damaged = data.clone();
            for section in &info.frames[physical].sections {
                damaged[section.bytes.offset as usize..section.bytes.end().unwrap() as usize]
                    .fill(0xff);
            }
            assert_eq!(inventory(&damaged), info, "only entropy bytes changed");
            assert_eq!(
                renderer.check_fragmented(&damaged, request(other)),
                renderer.decode(&data, request(other), false)
            );
            // A selected corrupt image may fail during bounded metadata negotiation or GPU
            // validation, but must never become a successful presentation.
            if let Ok(mut session) = renderer.whole.open(&damaged, request(damaged_selection)) {
                assert!(pollster::block_on(session.next_frame_async()).is_err());
                drop(session);
            }
            renderer.released();
            let mut stream = renderer.bounded.stream(request(damaged_selection)).unwrap();
            let mut transport = jxl_gpu_bitstream::ContainerStreamScanner::new(
                renderer.bounded.container_stream_limits(),
            );
            for chunk in damaged.chunks(43) {
                for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                    stream.push_transport_event(&event).unwrap();
                }
            }
            for event in transport.finish_input().unwrap() {
                stream.push_transport_event(&event).unwrap();
            }
            if let Ok(mut session) = stream.finish() {
                assert!(pollster::block_on(session.next_frame_async()).is_err());
                drop(session);
            }
            renderer.released();
        }
    }
}
