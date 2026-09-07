#![cfg(not(target_arch = "wasm32"))]

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};

use jxl::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use jxl_gpu_bitstream::{
    ContainerStreamScanner, EdgePreservingFilterInventory, GaborishInventory, InventoryLimits,
    ParseLimits, RestorationFilterInventory,
};
use jxl_gpu_formats::{Channel, ImageLayout, PitchLinearPlaneLayout, PixelFormat, SampleKind};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use jxl_wgpu::{
    DisplayColorEncoding, DisplayPipeline, DisplayTexture, DisplayTextureDescriptor,
    ImageReadbackPipeline, MemoryBudget, MemoryBudgetError, ResidentVarDctMemoryPlan, WgpuBackend,
    WgpuBackendConfig,
};
use jxl_wgpu_decode::vardct::engine::vardct_rgb8_format;
use jxl_wgpu_decode::vardct::packet::{
    BoundedVarDctPacketError, BoundedVarDctPacketPlan, GpuVarDctPacketError,
};
use jxl_wgpu_decode::{
    DecodeProfile, Error as DecodeError, GpuDecodeSession, GpuDecoder, GpuOutputRequest,
    NumericSampleMapping, PrefetchBackpressure, VarDctDecodeError, VarDctSubmissionEngine,
    WgpuDecodeEngine, WgpuDecodeSubmissionSession,
};
use jxl_wgpu_encode::{
    BufferImageSource, TiledVarDctEncoder, VarDctColorEncoding, VarDctEncoder, VarDctStrategy,
    WgpuContext,
};
use wgpu::util::DeviceExt;

mod common;

static DJXL_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn open_incremental(
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    encoded: &[u8],
    request: GpuOutputRequest,
) -> GpuDecodeSession<WgpuDecodeSubmissionSession> {
    let mut stream = decoder.stream(request).unwrap();
    let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
    let first = encoded.len().min(1);
    let second = encoded.len().min(17);
    for range in [0..first, first..second, second..encoded.len()] {
        for event in transport.push_chunk(Arc::from(&encoded[range])).unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
    }
    for event in transport.finish_input().unwrap() {
        stream.push_transport_event(&event).unwrap();
    }
    assert!(stream.is_ready());
    assert!(stream.stats().retained_spans >= 2);
    stream.finish().unwrap()
}

fn device() -> Option<(wgpu::AdapterInfo, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu bounded VarDCT decoder oracle"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((info, device, queue))
}

fn solid_source(
    context: &WgpuContext,
    strategy: VarDctStrategy,
    rgb: [u8; 3],
) -> BufferImageSource {
    let (width, height) = strategy.block_extent();
    let extent = Extent2d::new(u32::from(width), u32::from(height));
    let pixels = extent.area().unwrap();
    let bytes = rgb.repeat(pixels);
    let layout = ImageLayout::from_planes(
        extent,
        VarDctColorEncoding::SrgbD65.pixel_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: u64::from(extent.width) * 3,
            sample_extent: extent,
            row_bytes: u64::from(extent.width) * 3,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solid VarDCT decoder oracle source"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

fn tiled_source(context: &WgpuContext, extent: Extent2d) -> BufferImageSource {
    let mut bytes = Vec::with_capacity(extent.area().unwrap() * 3);
    for y in 0..extent.height {
        for x in 0..extent.width {
            bytes.extend_from_slice(&[
                ((x * 17 + y * 3) & 255) as u8,
                ((y * 29 + x * 5) & 255) as u8,
                (((x + y) * 11) & 255) as u8,
            ]);
        }
    }
    let layout = ImageLayout::from_planes(
        extent,
        VarDctColorEncoding::SrgbD65.pixel_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: u64::from(extent.width) * 3,
            sample_extent: extent,
            row_bytes: u64::from(extent.width) * 3,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("odd tiled VarDCT decoder oracle source"),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

fn rust_jxl_rgb8(codestream: &[u8], extent: Extent2d) -> Vec<u8> {
    let mut input = codestream;
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    assert_eq!(
        decoder.basic_info().size,
        (extent.width as usize, extent.height as usize)
    );
    decoder.set_pixel_format(JxlPixelFormat::rgb8(0));
    let mut frame = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let mut pixels = vec![0u8; extent.area().unwrap() * 3];
    let mut buffers = [JxlOutputBuffer::new(
        &mut pixels,
        extent.height as usize,
        extent.width as usize * 3,
    )];
    loop {
        match frame.process(&mut input, &mut buffers, None).unwrap() {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => frame = fallback,
        }
    }
    pixels
}

fn maximum_error(left: &[u8], right: &[u8]) -> u8 {
    assert_eq!(
        left.len(),
        right.len(),
        "oracle image byte counts must match"
    );
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left.abs_diff(right))
        .max()
        .unwrap_or(0)
}

fn assert_presentation_matches_oracles(
    backend: &WgpuBackend,
    name: &str,
    encoded: &[u8],
    extent: Extent2d,
) -> Vec<u8> {
    let expected = rust_jxl_rgb8(encoded, extent);
    let djxl = djxl_ppm(encoded, extent);
    let mut whole_output = None;
    for cap in [u64::MAX, 256] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
        let mut session = if cap == 256 {
            open_incremental(&decoder, encoded, request)
        } else {
            decoder.open(encoded, request).unwrap()
        };
        let frame = if cap == 256 {
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap()
        } else {
            session.next_frame().unwrap().unwrap()
        };
        assert!(session.next_frame().unwrap().is_none());
        assert_eq!(frame.output().outputs[0].layout.extent, extent);
        let readback = ImageReadbackPipeline::new(backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(readback.frame.outputs[0].layout.extent, extent);
        let actual = &readback.frame.outputs[0].bytes;
        let error = maximum_error(actual, &expected);
        assert!(error <= 1, "{name}, cap {cap}: Rust jxl error {error}");
        if let Some(djxl) = &djxl {
            let error = maximum_error(actual, djxl);
            assert!(error <= 1, "{name}, cap {cap}: djxl error {error}");
        }
        if let Some(whole) = &whole_output {
            assert_eq!(actual, whole, "{name}: bounded windows changed pixels");
        } else {
            whole_output = Some(actual.clone());
        }
        eprintln!("{name}, cap {cap}: max error {error}");
        drop(readback);
        drop(frame);
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
    whole_output.unwrap()
}

#[test]
fn all_eight_orientations_normalize_multigroup_progressive_output_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    eprintln!("orientation adapter: {info:?}");
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for value in 1..=8 {
        let encoded = common::vardct_orientation(value);
        let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert_eq!(inventory.image_header.orientation, value);
        let packet = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
        assert_eq!((packet.profile.width, packet.profile.height), (257, 17));
        assert_eq!(packet.profile.group_count, 2);
        assert_eq!(packet.profile.coefficient_shifts, [0, 0, 0]);
        let orientation = OutputOrientation::from_exif_value(value).unwrap();
        let extent = orientation.map_extent(Extent2d::new(257, 17));
        assert_presentation_matches_oracles(
            &backend,
            &format!("orientation {value}"),
            &encoded,
            extent,
        );
    }
}

#[test]
fn grayscale_presentation_combines_orientation_resampling_and_progressive_dc_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    eprintln!("grayscale adapter: {info:?}");
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    for (name, coded_extent, orientation, upsampling, frame_count) in [
        ("single", Extent2d::new(17, 9), 3, 1, 1),
        ("progressive", Extent2d::new(257, 129), 1, 1, 1),
        ("upsample", Extent2d::new(515, 259), 8, 4, 1),
        ("multilf", Extent2d::new(2056, 17), 5, 1, 1),
        ("dc_ac", Extent2d::new(1024, 128), 6, 1, 3),
        ("jpeg", Extent2d::new(173, 101), 8, 1, 1),
    ] {
        let encoded = common::vardct_gray(name);
        let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        assert!(inventory.image_header.grayscale);
        assert_eq!(inventory.image_header.xyb_encoded, name != "jpeg");
        assert_eq!(inventory.image_header.orientation, orientation);
        assert_eq!(inventory.frames.len(), frame_count);
        assert_eq!(inventory.frames.last().unwrap().upsampling, upsampling);
        if frame_count == 1 {
            let packet = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
            if name == "multilf" {
                assert_eq!(packet.profile.low_frequency_group_count, 2);
            }
            if name == "progressive" {
                assert_eq!(packet.profile.coefficient_shifts, [1, 0]);
            }
        }
        let extent = OutputOrientation::from_exif_value(orientation)
            .unwrap()
            .map_extent(coded_extent);
        let pixels = assert_presentation_matches_oracles(&backend, name, &encoded, extent);
        assert!(
            pixels
                .chunks_exact(3)
                .all(|pixel| pixel[0] == pixel[1] && pixel[1] == pixel[2]),
            "{name}: grayscale must have no chromatic residuals"
        );
    }
}

#[test]
fn jpeg_subsampling_is_expanded_in_codestream_coordinates_before_orientation() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let encoded = common::vardct_oriented_jpeg();
    let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert!(!inventory.image_header.xyb_encoded);
    assert_eq!(inventory.image_header.orientation, 6);
    let packet = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
    let shifts = packet.profile.channel_shifts;
    assert_eq!((shifts[0].horizontal, shifts[0].vertical), (1, 1));
    assert_eq!(shifts[1], Default::default());
    assert_eq!(shifts[0], shifts[2]);
    assert_presentation_matches_oracles(
        &backend,
        "oriented JPEG 420",
        &encoded,
        Extent2d::new(101, 173),
    );
}

#[test]
fn frame_upsampling_uses_header_weights_and_presentation_extent_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    eprintln!("frame upsampling adapter: {info:?}");
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let cases = [
        ("2", Extent2d::new(515, 259), 2),
        ("4", Extent2d::new(1027, 133), 4),
        ("8", Extent2d::new(2053, 67), 8),
        ("8_custom", Extent2d::new(584, 456), 8),
        ("2_multilf", Extent2d::new(4111, 17), 2),
        ("4_thin", Extent2d::new(17, 1), 4),
        ("8_single", Extent2d::new(7, 5), 8),
    ];
    let mut default_up8_weights = None;
    for (name, extent, factor) in cases {
        let encoded = common::vardct_upsampling(name);
        let inventory = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let packet = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
        assert_eq!(packet.profile.upsampling, factor);
        assert_eq!(
            (packet.profile.width, packet.profile.height),
            (
                extent.width.div_ceil(factor),
                extent.height.div_ceil(factor)
            )
        );
        assert_eq!(
            (packet.profile.output_width, packet.profile.output_height),
            (extent.width, extent.height)
        );
        if name == "2_multilf" {
            assert_eq!(packet.profile.low_frequency_group_count, 2);
        }
        if name == "4" {
            assert_eq!(packet.profile.coefficient_shifts.len(), 3);
        }
        if name == "8" {
            default_up8_weights = Some(inventory.image_header.upsampling_weights.up8);
        }
        if name == "8_custom" {
            assert!(
                Some(inventory.image_header.upsampling_weights.up8) != default_up8_weights,
                "the custom fixture must carry nondefault weights"
            );
        }
        let expected = rust_jxl_rgb8(&encoded, extent);
        let djxl = djxl_ppm(&encoded, extent);
        let mut whole_output = None;
        for cap in [u64::MAX, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
            let mut session = if cap == 256 {
                open_incremental(&decoder, &encoded, request)
            } else {
                decoder.open(&encoded, request).unwrap()
            };
            let memory = session
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats();
            assert_eq!(
                memory.frame_upsample_bytes,
                u64::from(extent.width) * u64::from(extent.height) * 12
            );
            assert_eq!(
                memory.frame_upsample_weight_bytes,
                u64::from(factor * factor) * 25 * 4
            );
            assert_eq!(memory.frame_upsample_uniform_bytes, 96);
            if name == "2" && cap == 256 {
                let budget =
                    MemoryBudget::new(NonZeroU64::new(memory.frame_upsample_bytes).unwrap());
                let limited = GpuDecoder::new(
                    VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
                        .unwrap(),
                );
                assert!(matches!(
                    limited.open(
                        &encoded,
                        GpuOutputRequest::color(vardct_rgb8_format()).unwrap()
                    ),
                    Err(DecodeError::VarDct(
                        VarDctDecodeError::MemoryBudgetTooSmall { .. }
                    ))
                ));
                assert_eq!(budget.snapshot().reserved_bytes, 0);
            }
            let frame = if cap == 256 {
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_frame().unwrap().unwrap()
            };
            assert!(session.next_frame().unwrap().is_none());
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(readback.frame.outputs[0].layout.extent, extent);
            let actual = &readback.frame.outputs[0].bytes;
            let error = maximum_error(actual, &expected);
            assert!(
                error <= 1,
                "upsample {name}, cap {cap}: Rust jxl error {error}"
            );
            if let Some(djxl) = &djxl {
                let error = maximum_error(actual, djxl);
                assert!(error <= 1, "upsample {name}, cap {cap}: djxl error {error}");
            }
            if let Some(whole) = &whole_output {
                assert_eq!(actual, whole);
            } else {
                whole_output = Some(actual.clone());
            }
            eprintln!("upsample {name}, cap {cap}: max error {error}");
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn progressive_ac_passes_accumulate_with_independent_tables_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    eprintln!("progressive AC adapter: {info:?}");
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let cases = [
        (
            "spectral",
            common::vardct_progressive_spectral(),
            Extent2d::new(257, 129),
            false,
        ),
        (
            "quantized",
            common::vardct_progressive_quantized(),
            Extent2d::new(515, 259),
            true,
        ),
        (
            "multilf",
            common::vardct_progressive_multilf(),
            Extent2d::new(2056, 17),
            false,
        ),
    ];
    let mut distinct_pass_orders = false;
    for (name, encoded, extent, shifted) in cases {
        let inventory = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
            .unwrap()
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        let packet = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
        let entropy = packet.hf_coefficients.as_ref().unwrap();
        eprintln!(
            "{name}: shifts {:?}, groups {}, LF groups {}",
            packet.profile.coefficient_shifts,
            packet.profile.group_count,
            packet.profile.low_frequency_group_count
        );
        assert_eq!(entropy.passes.len(), if shifted { 2 } else { 3 });
        assert_eq!(
            entropy.passes.len(),
            inventory.frames[0].num_passes as usize
        );
        assert_eq!(
            packet
                .profile
                .coefficient_shifts
                .iter()
                .any(|&shift| shift != 0),
            shifted
        );
        assert!(
            entropy
                .passes
                .windows(2)
                .any(|pair| pair[0].metadata != pair[1].metadata)
        );
        distinct_pass_orders |= entropy
            .passes
            .windows(2)
            .any(|pair| pair[0].order_words != pair[1].order_words);
        if name == "quantized" {
            assert!(
                inventory.frames[0]
                    .sections
                    .iter()
                    .any(|section| section.bitstream_index != section.toc_index)
            );
        }
        let expected = rust_jxl_rgb8(encoded, extent);
        let djxl = djxl_ppm(encoded, extent);
        let mut whole_output = None;
        for cap in [u64::MAX, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
            let mut session = if cap == 256 {
                let mut input = decoder.stream(request).unwrap();
                let mut transport = ContainerStreamScanner::new(decoder.container_stream_limits());
                for chunk in encoded.chunks(37) {
                    for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                        input.push_transport_event(&event).unwrap();
                    }
                }
                for event in transport.finish_input().unwrap() {
                    input.push_transport_event(&event).unwrap();
                }
                input.finish().unwrap()
            } else {
                decoder.open(encoded, request).unwrap()
            };
            let memory = session
                .submission_session()
                .vardct()
                .unwrap()
                .memory_stats();
            assert_eq!(
                memory.hf_status_bytes,
                packet.profile.group_count * entropy.passes.len() as u64 * 32
            );
            assert_eq!(
                memory.hf_execution_state_bytes,
                packet.profile.group_count * entropy.passes.len() as u64 * 464
            );
            if cap == 256 {
                assert!(memory.hf_stream_window_bytes <= 256);
                assert!(
                    memory.hf_stream_batch_count
                        > packet.profile.group_count as usize * entropy.passes.len()
                );
            }
            let frame = if cap == 256 {
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_frame().unwrap().unwrap()
            };
            assert!(session.next_frame().unwrap().is_none());
            let readback = ImageReadbackPipeline::new(&backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let actual = &readback.frame.outputs[0].bytes;
            let error = maximum_error(actual, &expected);
            assert!(error <= 1, "{name}, cap {cap}: Rust jxl error {error}");
            if let Some(djxl) = &djxl {
                let error = maximum_error(actual, djxl);
                assert!(error <= 1, "{name}, cap {cap}: djxl error {error}");
            }
            if let Some(whole_output) = &whole_output {
                assert_eq!(actual, whole_output);
            } else {
                whole_output = Some(actual.clone());
            }
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
    assert!(
        distinct_pass_orders,
        "the corpus must exercise different coefficient orders between passes"
    );
}

#[test]
fn progressive_ac_combines_with_recursive_gpu_resident_dc() {
    let encoded = common::vardct_progressive_dc_ac();
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let inventory = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert_eq!(inventory.frames.len(), 3);
    assert_eq!(
        inventory.frames.last().unwrap().progressive_passes.shifts,
        [1]
    );
    let extent = Extent2d::new(1024, 128);
    let expected = rust_jxl_rgb8(encoded, extent);
    let djxl = djxl_ppm(encoded, extent);
    for cap in [u64::MAX, 256] {
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
        );
        let mut session = decoder
            .open(
                encoded,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            session.submission_session(),
            WgpuDecodeSubmissionSession::ProgressiveDc(_)
        ));
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        assert!(session.next_frame().unwrap().is_none());
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0].bytes;
        let error = maximum_error(actual, &expected);
        assert!(
            error <= 1,
            "progressive DC+AC cap {cap}: Rust jxl error {error}"
        );
        if let Some(djxl) = &djxl {
            let error = maximum_error(actual, djxl);
            assert!(
                error <= 1,
                "progressive DC+AC cap {cap}: djxl error {error}"
            );
        }
        drop(readback);
        drop(frame);
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn progressive_ac_late_corruption_and_cancellation_release_all_pass_storage() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let encoded = common::vardct_progressive_quantized();
    let inventory = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    let last_pass = packet
        .hf_coefficients
        .as_ref()
        .unwrap()
        .passes
        .last()
        .unwrap();
    let range = last_pass
        .pass_groups
        .iter()
        .max_by_key(|range| range.length)
        .unwrap();
    let mut damaged = encoded.to_vec();
    let end = (range.end().unwrap() / 8) as usize;
    assert!(range.length > 256);
    damaged[end - 32..end].fill(0xff);
    let request = || GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
    let mut damaged_session = decoder.open(&damaged, request()).unwrap();
    assert!(matches!(
        damaged_session.next_frame(),
        Err(DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
    ));
    drop(damaged_session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut abandoned = decoder.open(encoded, request()).unwrap();
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
    drop(abandoned);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while decoder.engine().in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn jpeg_transcode_sampling_layouts_match_reference_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let extent = Extent2d::new(264, 64);
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let cases = [
        (
            "4:4:4",
            common::jpeg_transcode_444(),
            [67_584, 67_584, 67_584],
        ),
        (
            "4:2:2",
            common::jpeg_transcode_422(),
            [34_816, 69_632, 34_816],
        ),
        (
            "4:4:0",
            common::jpeg_transcode_440(),
            [33_792, 67_584, 33_792],
        ),
        (
            "4:2:0",
            common::jpeg_transcode_raw_matrix(),
            [17_408, 69_632, 17_408],
        ),
    ];
    for (name, encoded, resident_plane_bytes) in cases {
        let mut session = decoder
            .open(
                encoded,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let memory = session
            .submission_session()
            .vardct()
            .expect("JPEG reconstruction selects the VarDCT submission session")
            .memory_stats();
        assert_eq!(memory.resident_plane_bytes, resident_plane_bytes, "{name}");
        assert_eq!(
            memory.resident_image_bytes,
            resident_plane_bytes.into_iter().sum::<u64>(),
            "{name}",
        );
        let frame = session.next_frame().unwrap().unwrap();
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0].bytes;
        let rust = rust_jxl_rgb8(encoded, extent);
        assert_eq!(actual.len(), rust.len(), "{name}");
        let rust_error = maximum_error(actual, &rust);
        assert!(
            rust_error <= 1,
            "{name} JPEG-transcode GPU output diverges from Rust jxl by {rust_error}",
        );
        if let Some(djxl) = djxl_ppm(encoded, extent) {
            let djxl_error = maximum_error(actual, &djxl);
            assert!(
                djxl_error <= 1,
                "{name} JPEG-transcode GPU output diverges from djxl by {djxl_error}",
            );
        }
    }
}

fn djxl_ppm(codestream: &[u8], extent: Extent2d) -> Option<Vec<u8>> {
    fn next_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> &'a [u8] {
        loop {
            while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
                *cursor += 1;
            }
            if bytes.get(*cursor) != Some(&b'#') {
                break;
            }
            while bytes.get(*cursor).is_some_and(|&byte| byte != b'\n') {
                *cursor += 1;
            }
        }
        let start = *cursor;
        while bytes
            .get(*cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            *cursor += 1;
        }
        &bytes[start..*cursor]
    }

    if std::process::Command::new("djxl")
        .arg("--version")
        .output()
        .is_err()
    {
        return None;
    }
    let nonce = format!(
        "{}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        DJXL_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed),
    );
    let input = std::env::temp_dir().join(format!("jxl-wgpu-vardct-{nonce}.jxl"));
    let output = std::env::temp_dir().join(format!("jxl-wgpu-vardct-{nonce}.ppm"));
    std::fs::write(&input, codestream).unwrap();
    let command = std::process::Command::new("djxl")
        .args([&input, &output])
        .output()
        .unwrap();
    assert!(
        command.status.success(),
        "djxl rejected bounded VarDCT packet: {}",
        String::from_utf8_lossy(&command.stderr)
    );
    let ppm = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_file(input);
    let _ = std::fs::remove_file(output);
    let mut cursor = 0;
    let channels = match next_token(&ppm, &mut cursor) {
        b"P5" => 1,
        b"P6" => 3,
        magic => panic!("djxl emitted unsupported PNM magic {magic:?}"),
    };
    assert_eq!(
        std::str::from_utf8(next_token(&ppm, &mut cursor))
            .unwrap()
            .parse::<u32>()
            .unwrap(),
        extent.width
    );
    assert_eq!(
        std::str::from_utf8(next_token(&ppm, &mut cursor))
            .unwrap()
            .parse::<u32>()
            .unwrap(),
        extent.height
    );
    let maximum = std::str::from_utf8(next_token(&ppm, &mut cursor))
        .unwrap()
        .parse::<u32>()
        .unwrap();
    if ppm.get(cursor..cursor + 2) == Some(b"\r\n") {
        cursor += 2;
    } else {
        assert!(ppm.get(cursor).is_some_and(u8::is_ascii_whitespace));
        cursor += 1;
    }
    let pixels = &ppm[cursor..];
    let pixels: Vec<u8> = match maximum {
        255 => pixels.to_vec(),
        65_535 => pixels
            .chunks_exact(2)
            .map(|pair| {
                let value = u16::from_be_bytes([pair[0], pair[1]]);
                ((u32::from(value) + 128) / 257) as u8
            })
            .collect(),
        _ => panic!("djxl PPM uses unsupported maximum {maximum}"),
    };
    assert_eq!(pixels.len(), extent.area().unwrap() * channels);
    Some(if channels == 1 {
        pixels.into_iter().flat_map(|value| [value; 3]).collect()
    } else {
        pixels
    })
}

fn read_display_texture(backend: &WgpuBackend, texture: &DisplayTexture) -> Vec<u8> {
    let bytes_per_row = texture.extent.width.checked_mul(4).unwrap().div_ceil(256) * 256;
    let size = u64::from(bytes_per_row) * u64::from(texture.extent.height);
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("bounded VarDCT display oracle staging"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bounded VarDCT display oracle copy"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: texture.texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: texture.extent.width,
            height: texture.extent.height,
            depth_or_array_layers: 1,
        },
    );
    let submission = backend.queue().submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let row_bytes = usize::try_from(texture.extent.width * 4).unwrap();
    let mut packed = Vec::with_capacity(
        usize::try_from(texture.extent.width * texture.extent.height * 4).unwrap(),
    );
    for y in 0..texture.extent.height {
        let offset = usize::try_from(y * bytes_per_row).unwrap();
        packed.extend_from_slice(&mapped[offset..offset + row_bytes]);
    }
    drop(mapped);
    staging.unmap();
    packed
}

fn linear_srgb_code(code: u8) -> u8 {
    let encoded = f32::from(code) / 255.0;
    let linear = if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    };
    (linear * 255.0).round() as u8
}

const SUPPORTED_STRATEGIES: [VarDctStrategy; 9] = [
    VarDctStrategy::Dct8,
    VarDctStrategy::Dct16x16,
    VarDctStrategy::Dct32x32,
    VarDctStrategy::Dct16x8,
    VarDctStrategy::Dct8x16,
    VarDctStrategy::Dct32x8,
    VarDctStrategy::Dct8x32,
    VarDctStrategy::Dct32x16,
    VarDctStrategy::Dct16x32,
];

#[test]
fn one_decoder_routes_modular_and_all_bounded_vardct_packets_on_gpu() {
    let Some((info, device, queue)) = device() else {
        eprintln!("skipping bounded VarDCT engine oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();

    let modular_request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let modular_input = common::gpu_gray8_lossless();
    let mut modular_session = open_incremental(&decoder, modular_input, modular_request);
    let modular_codestream_bytes = jxl_gpu_bitstream::parse(modular_input, ParseLimits::default())
        .unwrap()
        .codestream()
        .len() as u64;
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        modular_codestream_bytes
    );
    assert!(matches!(
        modular_session.profile(),
        DecodeProfile::ModularLossless { .. }
    ));
    assert!(modular_session.submission_session().modular().is_some());
    let modular_frame = modular_session.next_frame().unwrap().unwrap();
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    let modular_readback = ImageReadbackPipeline::new(&backend)
        .submit(modular_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let expected_modular = (0..13u32)
        .flat_map(|y| {
            (0..17u32).map(move |x| {
                if y < 3 {
                    0
                } else {
                    ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
                }
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(modular_readback.frame.outputs[0].bytes, expected_modular);
    drop(modular_readback);
    drop(modular_frame);
    drop(modular_session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let rgb = [19, 103, 229];
    let mut dct8_packet = None;

    for (index, strategy) in SUPPORTED_STRATEGIES.into_iter().enumerate() {
        let (width, height) = strategy.block_extent();
        let extent = Extent2d::new(u32::from(width), u32::from(height));
        let encoded = VarDctEncoder::new(context.clone(), strategy)
            .unwrap()
            .encode(solid_source(&context, strategy, rgb))
            .unwrap();
        if strategy == VarDctStrategy::Dct8 {
            dct8_packet = Some(encoded.clone());
        }
        let oracle = djxl_ppm(&encoded, extent);
        let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
        let mut session = if index == 0 {
            let session = open_incremental(&decoder, &encoded, request);
            let codestream_bytes = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
                .unwrap()
                .codestream()
                .len() as u64;
            assert_eq!(
                decoder.incremental_input_budget().snapshot().reserved_bytes,
                codestream_bytes
            );
            session
        } else {
            decoder.open(&encoded, request).unwrap()
        };
        let memory = session
            .submission_session()
            .vardct()
            .expect("VarDCT input selects the VarDCT submission session")
            .memory_stats();
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        let frame = if index == 0 {
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap()
        } else {
            session.next_frame().unwrap().unwrap()
        };
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
        assert_eq!(
            frame.output().outputs[0].layout.format,
            vardct_rgb8_format()
        );
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            memory.output_lease_bytes
        );

        let display_rgba = if index == 0 {
            let displayed = DisplayPipeline::new(&backend)
                .submit_image(
                    &frame.output().outputs[0],
                    DisplayTextureDescriptor::default(),
                )
                .expect("explicit sRGB VarDCT output is directly displayable");
            assert_eq!(
                displayed.texture.color_encoding,
                DisplayColorEncoding::LinearBt709
            );
            Some(read_display_texture(&backend, &displayed.texture))
        } else {
            None
        };

        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0].bytes;
        assert_eq!(actual.len(), extent.area().unwrap() * 3);
        if let Some(oracle) = oracle {
            assert_eq!(actual.len(), oracle.len());
            let maximum_error = actual
                .iter()
                .zip(&oracle)
                .map(|(&gpu, &cpu)| gpu.abs_diff(cpu))
                .max()
                .unwrap();
            assert!(
                maximum_error <= 1,
                "{strategy:?} resident output differs from djxl by {maximum_error} codes"
            );
        }
        if let Some(rgba) = display_rgba {
            assert_eq!(rgba.len() / 4, actual.len() / 3);
            for (display, encoded) in rgba.chunks_exact(4).zip(actual.chunks_exact(3)) {
                let expected = [
                    linear_srgb_code(encoded[0]),
                    linear_srgb_code(encoded[1]),
                    linear_srgb_code(encoded[2]),
                ];
                assert!(display[0].abs_diff(expected[0]) <= 1);
                assert!(display[1].abs_diff(expected[1]) <= 1);
                assert!(display[2].abs_diff(expected[2]) <= 1);
                assert_eq!(display[3], 255);
            }
        }
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            memory.output_lease_bytes,
            "readback and decode must share and release the backend byte budget"
        );
        let retained = frame.output().outputs[0].buffer.clone();
        drop(readback);
        drop(frame);
        drop(session);
        assert_eq!(
            decoder.engine().in_flight_memory_stats().reserved_bytes,
            memory.output_lease_bytes,
            "the last GPU output-buffer clone owns the decode reservation"
        );
        drop(retained);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }

    let mut corrupted = dct8_packet.expect("Dct8 is in the accepted strategy matrix");
    let parsed = jxl_gpu_bitstream::parse(&corrupted, ParseLimits::default()).unwrap();
    assert_eq!(parsed.codestream().len(), corrupted.len());
    let inventory = parsed
        .codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: u64::try_from(corrupted.len()).unwrap(),
            ..InventoryLimits::default()
        })
        .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
    let entropy_bit = usize::try_from(packet.entropy_bit_offset).unwrap();
    let modular_header_bit = entropy_bit + 2;
    corrupted[modular_header_bit / 8] ^= 1 << (modular_header_bit % 8);

    let error = match decoder.open(
        &corrupted,
        GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
    ) {
        Err(error) => error,
        Ok(_) => panic!("corrupt local MA metadata must be rejected before submission"),
    };
    assert!(matches!(
        error,
        DecodeError::VarDct(VarDctDecodeError::Packet(
            BoundedVarDctPacketError::ModularTree(_)
        ))
    ));
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn combined_single_packet_resumes_across_bounded_gpu_windows() {
    let Some((info, device, queue)) = device() else {
        eprintln!("skipping combined VarDCT packet oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let extent = Extent2d::new(32, 32);
    let encoded = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct32x32)
        .unwrap()
        .encode(tiled_source(&context, extent))
        .unwrap();
    let parsed = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let plan = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
    assert!(plan.hf_global.is_none());
    assert!(plan.requires_lf_staging());

    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let codestream_bytes = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
        .unwrap()
        .codestream()
        .len() as u64;
    let mut session = open_incremental(
        &decoder,
        &encoded,
        GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
    );
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        codestream_bytes
    );
    let vardct = session.submission_session().vardct().unwrap();
    let memory = vardct.memory_stats();
    assert!(memory.deferred_hf_modular_metadata);
    assert!(memory.packet_stream_window_bytes > 0);
    assert!(memory.packet_stream_window_bytes <= 40);
    assert!(memory.packet_stream_batch_count > 2);
    assert_eq!(
        vardct.submissions_per_frame(),
        memory.packet_stream_batch_count + 2
    );

    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(&encoded, extent);
    assert!(maximum_error(actual, &rust) <= 1);
    if let Some(djxl) = djxl_ppm(&encoded, extent) {
        assert!(maximum_error(actual, &djxl) <= 1);
    }
    drop(readback);
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut abandoned = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
    drop(abandoned);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while decoder.engine().in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let group = plan.groups.first().unwrap();
    let stream_start = u64::from(plan.entropy_bit_offset);
    let stream_end = group.lf_group.end().unwrap();
    let damage_bit = stream_start + (stream_end - stream_start) * 17 / 20;
    let damage_start = usize::try_from(damage_bit / 8).unwrap();
    let damage_end = (damage_start + 8).min(usize::try_from(stream_end.div_ceil(8)).unwrap());
    let mut damaged = encoded.clone();
    for byte in &mut damaged[damage_start..damage_end] {
        *byte ^= 0xa5;
    }
    let mut damaged_session = decoder
        .open(
            &damaged,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let error = damaged_session.next_frame().unwrap_err();
    assert!(
        matches!(
            error,
            DecodeError::VarDct(VarDctDecodeError::PacketGpu(
                GpuVarDctPacketError::Entropy { .. }
            ))
        ),
        "unexpected combined-packet corruption error: {error:?}"
    );
    drop(damaged_session);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn tiled_dct8_spans_empty_pass_groups_and_odd_padded_edges_on_gpu() {
    let Some((info, device, queue)) = device() else {
        eprintln!("skipping tiled VarDCT engine oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();

    for extent in [Extent2d::new(257, 17), Extent2d::new(513, 259)] {
        let encoded = encoder.encode(tiled_source(&context, extent)).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits {
                max_frames: 1,
                max_total_section_bytes: encoded.len() as u64,
                ..InventoryLimits::default()
            })
            .unwrap();
        let plan = BoundedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
        let blocks = extent.width.div_ceil(8) * extent.height.div_ceil(8);
        assert_eq!(plan.groups.len(), 1);
        let group = &plan.groups[0];
        assert_eq!(group.task_capacity, blocks);
        assert!(plan.hf_global.is_some());
        assert!(plan.profile.group_count >= 2);
        let hf_coefficients = plan
            .hf_coefficients
            .as_ref()
            .expect("multi-entry VarDCT parses the descriptor-only HF coefficient plan");
        assert_eq!(hf_coefficients.num_hf_presets, 1);
        assert_eq!(hf_coefficients.passes[0].context_map.len(), 495 * 15);
        assert_eq!(hf_coefficients.block_context_map.len(), 39);
        assert_eq!(
            hf_coefficients.passes[0].pass_groups.len() as u64,
            plan.profile.group_count
        );
        assert!(hf_coefficients.passes[0].metadata.len() >= 28);
        assert_eq!(hf_coefficients.passes[0].lz77_window_words, 0);
        let control = group.packet_control(&plan).unwrap();
        let correlations = extent.width.div_ceil(64) * extent.height.div_ceil(64);
        assert_eq!(control.offsets[0], 0);
        assert_eq!(control.offsets[1], correlations);
        assert_eq!(control.offsets[2], 2 * correlations);
        assert_eq!(control.offsets[3], 2 * correlations + blocks);
        assert_eq!(control.capacities[0], blocks * 8 * 8 * 3);
        assert_eq!(control.capacities[1], 2 * correlations + 3 * blocks);
        assert_eq!(control.capacities[3], blocks);

        let mut session = decoder
            .open(
                &encoded,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let memory = session
            .submission_session()
            .vardct()
            .expect("VarDCT input selects the VarDCT submission session")
            .memory_stats();
        assert_eq!(
            memory.resident_image_bytes,
            u64::from(extent.width.div_ceil(8) * 8) * u64::from(extent.height.div_ceil(8) * 8) * 12,
        );
        let resident_plan = ResidentVarDctMemoryPlan::new(blocks * 8 * 8 * 3).unwrap();
        assert_eq!(memory.resident_transient_bytes, resident_plan.total_bytes);
        let frame = if extent.width == 257 {
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap()
        } else {
            session.next_frame().unwrap().unwrap()
        };
        let retained = (extent.width == 257).then(|| frame.output().outputs[0].buffer.clone());
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0].bytes;
        assert_eq!(actual.len(), extent.area().unwrap() * 3);

        let rust = rust_jxl_rgb8(&encoded, extent);
        assert!(
            maximum_error(actual, &rust) <= 1,
            "{}x{} tiled GPU output diverges from Rust jxl",
            extent.width,
            extent.height,
        );
        if let Some(djxl) = djxl_ppm(&encoded, extent) {
            assert!(
                maximum_error(actual, &djxl) <= 1,
                "{}x{} tiled GPU output diverges from djxl",
                extent.width,
                extent.height,
            );
        }
        drop(readback);
        drop(frame);
        drop(session);
        if let Some(retained) = retained {
            assert_eq!(
                decoder.engine().in_flight_memory_stats().reserved_bytes,
                memory.output_lease_bytes,
                "the tiled output reservation follows the final GPU buffer clone",
            );
            drop(retained);
        }
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn libjxl_nonzero_ac_custom_order_matches_reference_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let encoded = common::green_queen_vardct_nonzero_ac();
    let extent = Extent2d::new(438, 589);
    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let plan = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    assert!(plan.needs_self_correcting);
    let hf = plan.hf_coefficients.as_ref().unwrap();
    assert_eq!(hf.passes[0].pass_groups.len(), 6);
    assert_eq!(hf.passes[0].order_coordinate_offset_words, 13 * 3 * 4);
    let descriptors =
        bytemuck::cast_slice::<u32, jxl_wgpu_decode::vardct::artifact::GpuHfOrderDescriptor>(
            &hf.passes[0].order_words[..hf.passes[0].order_coordinate_offset_words as usize],
        );
    assert_eq!(descriptors.len(), 13 * 3);
    assert_eq!([descriptors[0].width, descriptors[0].height], [8, 8]);
    assert_ne!(descriptors[0].offset, descriptors[1].offset);

    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(encoded, extent);
    assert_eq!(actual.len(), rust.len());
    assert!(
        maximum_error(actual, &rust) <= 1,
        "nonzero-AC GPU output diverges from Rust jxl",
    );
    if let Some(djxl) = djxl_ppm(encoded, extent) {
        assert!(
            maximum_error(actual, &djxl) <= 1,
            "nonzero-AC GPU output diverges from djxl",
        );
    }
}

#[test]
fn global_packet_and_nonzero_ac_resume_across_bounded_gpu_stream_windows() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let engine = WgpuDecodeEngine::new(backend.clone())
        .unwrap()
        .with_stream_window_limit(NonZeroU64::new(256).unwrap());
    let decoder = GpuDecoder::new(engine);
    let encoded = common::green_queen_vardct_nonzero_ac();
    let extent = Extent2d::new(438, 589);
    let expected = rust_jxl_rgb8(encoded, extent);

    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let vardct = session.submission_session().vardct().unwrap();
    let memory = vardct.memory_stats();
    assert!(memory.packet_stream_window_bytes > 0);
    assert!(memory.packet_stream_window_bytes <= 256);
    assert!(memory.packet_stream_batch_count > 1);
    assert!(memory.hf_stream_window_bytes > 0);
    assert!(memory.hf_stream_window_bytes <= 256);
    assert!(memory.hf_stream_batch_count > 6);
    assert_eq!(
        vardct.submissions_per_frame(),
        memory.packet_stream_batch_count - 1 + memory.hf_stream_batch_count + 2
    );
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(maximum_error(&readback.frame.outputs[0].bytes, &expected) <= 1);
    drop(readback);
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut async_session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let async_frame = pollster::block_on(async_session.next_frame_async())
        .unwrap()
        .unwrap();
    let async_readback = ImageReadbackPipeline::new(&backend)
        .submit(async_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(maximum_error(&async_readback.frame.outputs[0].bytes, &expected) <= 1);
    drop(async_readback);
    drop(async_frame);
    drop(async_session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    let damaged_range = packet.hf_coefficients.as_ref().unwrap().passes[0]
        .pass_groups
        .iter()
        .max_by_key(|range| range.length)
        .copied()
        .unwrap();
    let mut damaged = encoded.to_vec();
    let byte_start = usize::try_from(damaged_range.offset.div_ceil(8)).unwrap();
    let byte_end = usize::try_from(damaged_range.end().unwrap() / 8).unwrap();
    let damage_start = byte_start + (byte_end - byte_start) * 3 / 4;
    for byte in &mut damaged[damage_start..(damage_start + 32).min(byte_end)] {
        *byte ^= 0x5a;
    }
    let mut damaged_session = decoder
        .open(
            &damaged,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        damaged_session.next_frame(),
        Err(DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_)))
    ));
    drop(damaged_session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut abandoned = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
    drop(abandoned);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while decoder.engine().in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn vardct_stream_windows_adapt_to_the_shared_frame_budget() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let encoded = common::green_queen_vardct_nonzero_ac();
    let reference = rust_jxl_rgb8(encoded, Extent2d::new(438, 589));
    let request = GpuOutputRequest::color(vardct_rgb8_format()).unwrap();
    let memory_at_limit = |limit| {
        let decoder = GpuDecoder::new(
            VarDctSubmissionEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(limit).unwrap()),
        );
        let session = decoder.open(encoded, request.clone()).unwrap();
        session.submission_session().memory_stats()
    };
    let minimum = memory_at_limit(40);
    let configured = memory_at_limit(256);
    assert!(minimum.total_frame_bytes < configured.total_frame_bytes);
    let budget_limit =
        minimum.total_frame_bytes + (configured.total_frame_bytes - minimum.total_frame_bytes) / 2;
    assert!(budget_limit < configured.total_frame_bytes);

    let budget = MemoryBudget::new(NonZeroU64::new(budget_limit).unwrap());
    let engine = VarDctSubmissionEngine::with_memory_budget(backend.clone(), budget.clone())
        .unwrap()
        .with_stream_window_limit(NonZeroU64::new(256).unwrap());
    assert_eq!(engine.stream_window_limit(), NonZeroU64::new(256));
    let decoder = GpuDecoder::new(engine);
    let mut session = decoder.open(encoded, request.clone()).unwrap();
    let memory = session.submission_session().memory_stats();
    assert!(memory.resolved_stream_window_limit_bytes >= 40);
    assert!(memory.resolved_stream_window_limit_bytes < 256);
    assert_eq!(memory.resolved_stream_window_limit_bytes % 4, 0);
    assert!(memory.total_frame_bytes <= budget_limit);
    assert!(memory.packet_stream_window_bytes <= memory.resolved_stream_window_limit_bytes,);
    assert!(memory.hf_stream_window_bytes <= memory.resolved_stream_window_limit_bytes);

    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(maximum_error(&readback.frame.outputs[0].bytes, &reference,) <= 1);
    drop(readback);
    drop(frame);
    drop(session);
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    let mut abandoned = decoder.open(encoded, request.clone()).unwrap();
    let mut backpressured = decoder.open(encoded, request.clone()).unwrap();
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(budget.snapshot().reserved_bytes > 0);
    let progress = backpressured
        .prefetch(NonZeroUsize::new(1).unwrap())
        .unwrap();
    assert_eq!(progress.submitted, 0);
    assert_eq!(progress.queued, 0);
    assert!(
        matches!(
            progress.backpressure,
            Some(PrefetchBackpressure::Memory(
                MemoryBudgetError::Exhausted { .. }
            ))
        ),
        "unexpected concurrent VarDCT backpressure: {progress:?}",
    );
    drop(abandoned);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while budget.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    let retried_frame = backpressured.next_frame().unwrap().unwrap();
    let retried_readback = ImageReadbackPipeline::new(&backend)
        .submit(retried_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(maximum_error(&retried_readback.frame.outputs[0].bytes, &reference) <= 1);
    drop(retried_readback);
    drop(retried_frame);
    drop(backpressured);
    assert_eq!(budget.snapshot().reserved_bytes, 0);

    let insufficient_limit = minimum.total_frame_bytes - 1;
    let insufficient_budget = MemoryBudget::new(NonZeroU64::new(insufficient_limit).unwrap());
    let insufficient = GpuDecoder::new(
        VarDctSubmissionEngine::with_memory_budget(backend, insufficient_budget)
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let error = match insufficient.open(encoded, request) {
        Ok(_) => panic!("minimum-window VarDCT layout unexpectedly fit"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        DecodeError::VarDct(VarDctDecodeError::MemoryBudgetTooSmall {
            required_bytes,
            limit_bytes,
        }) if required_bytes == minimum.total_frame_bytes && limit_bytes == insufficient_limit
    ));
}

#[test]
fn libjxl_center_first_permuted_toc_matches_reference_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let encoded = common::green_queen_vardct_permuted();
    let extent = Extent2d::new(438, 589);
    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert!(inventory.frames[0].toc_permuted);
    let physical_group_order = inventory.frames[0]
        .sections
        .iter()
        .filter_map(|section| match section.kind {
            jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index,
            } => Some(group_index),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(physical_group_order.len(), 6);
    assert_ne!(physical_group_order, (0..6).collect::<Vec<_>>());

    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(encoded, extent);
    assert_eq!(actual.len(), rust.len());
    let rust_error = maximum_error(actual, &rust);
    assert!(
        rust_error <= 1,
        "center-first GPU output diverges from Rust jxl by {rust_error}",
    );
    if let Some(djxl) = djxl_ppm(encoded, extent) {
        let djxl_error = maximum_error(actual, &djxl);
        assert!(
            djxl_error <= 1,
            "center-first GPU output diverges from djxl by {djxl_error}",
        );
    }
}

#[test]
fn libjxl_mixed_strategies_and_capacity_strided_metadata_match_reference_on_gpu() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let encoded = common::green_queen_vardct_mixed();
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    let plan = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    let extent = Extent2d::new(plan.profile.width, plan.profile.height);
    assert_eq!(extent, Extent2d::new(257, 257));
    assert_eq!(plan.groups.len(), 1);
    assert_eq!(plan.groups[0].extra_precision, 1);
    assert_eq!(plan.groups[0].task_capacity, 33 * 33);
    let hf = plan.hf_coefficients.as_ref().unwrap();
    assert_eq!(hf.num_block_clusters, 3);
    let descriptors =
        bytemuck::cast_slice::<u32, jxl_wgpu_decode::vardct::artifact::GpuHfOrderDescriptor>(
            &hf.passes[0].order_words[..hf.passes[0].order_coordinate_offset_words as usize],
        );
    let custom_orders = (0..13)
        .filter(|&order| descriptors[order * 3].offset != descriptors[order * 3 + 1].offset)
        .collect::<Vec<_>>();
    assert_eq!(custom_orders, [0, 1]);
    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(encoded, extent);
    assert_eq!(actual.len(), rust.len());
    let rust_error = maximum_error(actual, &rust);
    assert!(
        rust_error <= 1,
        "mixed-strategy GPU output diverges from Rust jxl by {rust_error}",
    );
    if let Some(djxl) = djxl_ppm(encoded, extent) {
        let djxl_error = maximum_error(actual, &djxl);
        assert!(
            djxl_error <= 1,
            "mixed-strategy GPU output diverges from djxl by {djxl_error}",
        );
    }
}

fn assert_multiple_lf_groups(
    encoded: &[u8],
    adaptive_lf_smoothing: bool,
    stream_window_limit: Option<NonZeroU64>,
) {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert!(matches!(
        inventory.frames[0].restoration_filter,
        RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Default,
            epf: EdgePreservingFilterInventory::Enabled { iterations: 1, .. },
        }
    ));
    let plan = BoundedVarDctPacketPlan::parse(encoded, &inventory).unwrap();
    let extent = Extent2d::new(plan.profile.width, plan.profile.height);
    assert_eq!(extent, Extent2d::new(2056, 256));
    assert_eq!(plan.profile.adaptive_lf_smoothing, adaptive_lf_smoothing);
    assert_eq!(plan.profile.low_frequency_group_count, 2);
    assert_eq!(plan.profile.group_count, 9);
    assert_eq!(plan.groups.len(), 2);
    assert_eq!(plan.groups[0].rect.x, 0);
    assert_eq!(plan.groups[0].rect.width, 2048);
    assert_eq!(plan.groups[1].rect.x, 2048);
    assert_eq!(plan.groups[1].rect.width, 8);
    assert_eq!(plan.groups[0].lf_stream_index, 1);
    assert_eq!(plan.groups[1].lf_stream_index, 2);
    assert_eq!(plan.groups[0].hf_stream_index, 5);
    assert_eq!(plan.groups[1].hf_stream_index, 6);

    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    let engine = match stream_window_limit {
        Some(limit) => engine.with_stream_window_limit(limit),
        None => engine,
    };
    let decoder = GpuDecoder::new(engine);
    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let vardct = session
        .submission_session()
        .vardct()
        .expect("multiple LF groups select the VarDCT submission session");
    let memory = vardct.memory_stats();
    if let Some(stream_window_limit) = stream_window_limit {
        assert!(!memory.deferred_hf_modular_metadata);
        assert!(memory.packet_stream_window_bytes > 0);
        assert!(memory.packet_stream_window_bytes <= stream_window_limit.get());
        assert!(memory.packet_stream_batch_count > 2);
        let downstream_submissions = if memory.hf_stream_window_bytes == 0 {
            1
        } else {
            memory.hf_stream_batch_count + 2
        };
        assert_eq!(
            vardct.submissions_per_frame(),
            memory.packet_stream_batch_count - 1 + downstream_submissions,
        );
    } else {
        assert_eq!(vardct.submissions_per_frame(), 1);
    }
    assert_eq!(memory.packet_status_bytes, 2 * 64);
    assert_eq!(memory.adaptive_lf_uniform_bytes != 0, adaptive_lf_smoothing,);
    assert_eq!(
        memory.validation_staging_bytes,
        memory.packet_status_bytes
            + 2 * std::mem::size_of::<jxl_wgpu_decode::vardct::artifact::GpuVarDctArtifactStatus>()
                as u64
            + memory.hf_status_bytes,
    );

    let frame = if stream_window_limit.is_some() {
        pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap()
    } else {
        session.next_frame().unwrap().unwrap()
    };
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(encoded, extent);
    assert_eq!(actual.len(), rust.len());
    let rust_error = maximum_error(actual, &rust);
    assert!(
        rust_error <= 1,
        "multiple-LF-group GPU output diverges from Rust jxl by {rust_error}",
    );
    if let Some(djxl) = djxl_ppm(encoded, extent) {
        let djxl_error = maximum_error(actual, &djxl);
        assert!(
            djxl_error <= 1,
            "multiple-LF-group GPU output diverges from djxl by {djxl_error}",
        );
    }
}

#[test]
fn standard_multiple_lf_groups_share_one_resident_image_and_status_map() {
    assert_multiple_lf_groups(common::testsrc_vardct_multi_lf(), true, None);
}

#[test]
fn shared_global_tree_packets_resume_across_bounded_gpu_windows() {
    assert_multiple_lf_groups(
        common::testsrc_vardct_multi_lf(),
        true,
        NonZeroU64::new(256),
    );
}

#[test]
fn ordinary_cjxl_local_trees_resume_lf_and_hf_across_bounded_packet_windows() {
    let Some(encoded) = common::cjxl_local_tree_codestream() else {
        return;
    };
    let Some((info, device, queue)) = device() else {
        eprintln!("skipping local-tree VarDCT frame oracle: no adapter");
        return;
    };
    let extent = Extent2d::new(2056, 256);
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    let codestream_bytes = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default())
        .unwrap()
        .codestream()
        .len() as u64;
    let mut session = open_incremental(
        &decoder,
        &encoded,
        GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
    );
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        codestream_bytes
    );
    let vardct = session
        .submission_session()
        .vardct()
        .expect("ordinary cjxl selects the VarDCT submission session");
    let memory = vardct.memory_stats();
    assert!(memory.deferred_hf_modular_metadata);
    assert!(memory.packet_stream_window_bytes > 0);
    assert!(memory.packet_stream_window_bytes <= 256);
    assert!(memory.packet_stream_batch_count > 2);
    assert_eq!(memory.packet_execution_state_bytes, 2 * 128);
    let downstream_submissions = if memory.hf_stream_window_bytes == 0 {
        1
    } else {
        memory.hf_stream_batch_count + 2
    };
    let submissions_before_hf_plan = memory.packet_stream_batch_count + downstream_submissions;
    assert_eq!(vardct.submissions_per_frame(), submissions_before_hf_plan);

    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        codestream_bytes,
        "staged local-HF parsing retains compressed spans after the LF submission"
    );
    assert!(matches!(
        session
            .front_pending_frame()
            .unwrap()
            .unvalidated_gpu_frame(),
        Err(DecodeError::VarDct(
            VarDctDecodeError::UnvalidatedOutputNotSubmitted
        ))
    ));

    let frame = session.next_frame().unwrap().unwrap();
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0,
        "the final local-HF submission releases compressed spans"
    );
    let vardct = session
        .submission_session()
        .vardct()
        .expect("ordinary cjxl retains the VarDCT submission session");
    let hf_packet_batches = vardct.hf_packet_stream_batch_count();
    assert!(hf_packet_batches > 2);
    assert_eq!(
        vardct.submissions_per_frame(),
        submissions_before_hf_plan + hf_packet_batches - 1,
    );
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(&encoded, extent);
    let rust_error = maximum_error(actual, &rust);
    assert!(
        rust_error <= 1,
        "local-tree GPU output diverges from Rust jxl by {rust_error}",
    );
    if let Some(djxl) = djxl_ppm(&encoded, extent) {
        let djxl_error = maximum_error(actual, &djxl);
        assert!(
            djxl_error <= 1,
            "local-tree GPU output diverges from djxl by {djxl_error}",
        );
    }
    drop(readback);
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut async_session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let async_frame = pollster::block_on(async_session.next_frame_async())
        .unwrap()
        .unwrap();
    let async_readback = ImageReadbackPipeline::new(&backend)
        .submit(async_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(maximum_error(&async_readback.frame.outputs[0].bytes, &rust) <= 1);
    drop(async_readback);
    drop(async_frame);
    drop(async_session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let mut abandoned = open_incremental(
        &decoder,
        &encoded,
        GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
    );
    abandoned.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        codestream_bytes
    );
    assert!(decoder.engine().in_flight_memory_stats().reserved_bytes > 0);
    drop(abandoned);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0,
        "cancelling a staged local-HF session releases its source spans immediately"
    );
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while decoder.engine().in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let parsed = jxl_gpu_bitstream::parse(&encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits {
            max_frames: 1,
            max_total_section_bytes: encoded.len() as u64,
            ..InventoryLimits::default()
        })
        .unwrap();
    let packet = BoundedVarDctPacketPlan::parse(&encoded, &inventory).unwrap();
    let group = packet.groups.first().unwrap();
    let group_end = group.lf_group.end().unwrap();
    let damage_bit = group.lf_group.offset + (group_end - group.lf_group.offset) * 9 / 10;
    let damage_start = usize::try_from(damage_bit / 8).unwrap();
    let mut damaged = encoded.clone();
    for byte in damaged
        .get_mut(damage_start..damage_start + 32)
        .expect("the late HF packet damage range is inside the first LF group")
    {
        *byte ^= 0xa5;
    }
    let mut damaged_session = decoder
        .open(
            &damaged,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let error = damaged_session.next_frame().unwrap_err();
    assert!(matches!(
        error,
        DecodeError::VarDct(VarDctDecodeError::PacketGpu(
            GpuVarDctPacketError::Entropy { .. }
        ))
    ));
    drop(damaged_session);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn multiple_lf_groups_can_skip_adaptive_smoothing_without_a_cpu_copy() {
    assert_multiple_lf_groups(
        common::testsrc_vardct_multi_lf_skip_smoothing(),
        false,
        None,
    );
}

#[test]
fn libjxl_gaborish_executes_between_resident_vardct_and_output_pack() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let encoded = common::green_queen_vardct_gaborish();
    let extent = Extent2d::new(438, 589);
    let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
    let inventory = parsed
        .codestream_inventory(InventoryLimits::default())
        .unwrap();
    assert!(matches!(
        inventory.frames[0].restoration_filter,
        RestorationFilterInventory::Custom {
            gaborish: GaborishInventory::Default,
            epf: EdgePreservingFilterInventory::Disabled,
        }
    ));

    let mut session = decoder
        .open(
            encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let memory = session
        .submission_session()
        .vardct()
        .expect("VarDCT input selects the VarDCT submission session")
        .memory_stats();
    assert_eq!(
        memory.restoration_scratch_bytes,
        memory.resident_image_bytes
    );
    assert_eq!(memory.gaborish_uniform_bytes, 80);
    assert_eq!(
        memory.transient_bytes + memory.output_lease_bytes,
        memory.total_frame_bytes
    );
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let rust = rust_jxl_rgb8(encoded, extent);
    assert_eq!(actual.len(), rust.len());
    assert!(
        maximum_error(actual, &rust) <= 1,
        "resident Gaborish output diverges from Rust jxl",
    );
    if let Some(djxl) = djxl_ppm(encoded, extent) {
        assert!(
            maximum_error(actual, &djxl) <= 1,
            "resident Gaborish output diverges from djxl",
        );
    }
}

#[test]
fn libjxl_epf2_and_epf3_execute_on_odd_resident_extent() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let extent = Extent2d::new(257, 17);
    let mut epf2_output = None;
    for (encoded, iterations) in [
        (common::green_queen_crop_vardct_epf2(), 2_u32),
        (common::green_queen_crop_vardct_epf3(), 3_u32),
    ] {
        let parsed = jxl_gpu_bitstream::parse(encoded, ParseLimits::default()).unwrap();
        let inventory = parsed
            .codestream_inventory(InventoryLimits::default())
            .unwrap();
        match (iterations, inventory.frames[0].restoration_filter) {
            (2, RestorationFilterInventory::Default) => {}
            (
                3,
                RestorationFilterInventory::Custom {
                    gaborish: GaborishInventory::Default,
                    epf: EdgePreservingFilterInventory::Enabled { iterations: 3, .. },
                },
            ) => {}
            (_, actual) => panic!("unexpected EPF restoration inventory: {actual:?}"),
        }

        let mut session = decoder
            .open(
                encoded,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let memory = session
            .submission_session()
            .vardct()
            .expect("EPF fixture selects the VarDCT submission session")
            .memory_stats();
        assert_eq!(
            memory.restoration_scratch_bytes,
            memory.resident_image_bytes
        );
        assert_eq!(memory.gaborish_uniform_bytes, 80);
        assert_eq!(memory.epf_sigma_bytes, 33 * 3 * 4);
        assert_eq!(memory.epf_sigma_uniform_bytes, 80);
        assert_eq!(memory.epf_filter_uniform_bytes, u64::from(iterations) * 80);
        assert_eq!(
            memory.transient_bytes + memory.output_lease_bytes,
            memory.total_frame_bytes
        );

        let frame = session.next_frame().unwrap().unwrap();
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0].bytes;
        let rust = rust_jxl_rgb8(encoded, extent);
        assert_eq!(actual.len(), rust.len());
        let rust_error = maximum_error(actual, &rust);
        assert!(
            rust_error <= 1,
            "resident EPF{iterations} output diverges from Rust jxl by {rust_error}",
        );
        if let Some(djxl) = djxl_ppm(encoded, extent) {
            let djxl_error = maximum_error(actual, &djxl);
            assert!(
                djxl_error <= 1,
                "resident EPF{iterations} output diverges from djxl by {djxl_error}",
            );
        }
        if let Some(epf2) = &epf2_output {
            assert_ne!(actual, epf2, "EPF3 must execute its additional EPF0 pass");
        } else {
            epf2_output = Some(actual.to_vec());
        }
    }
}

#[test]
fn libjxl_progressive_dc_chain_stays_gpu_resident_until_one_visible_output() {
    let Some(encoded) = common::cjxl_progressive_dc_codestream(1) else {
        return;
    };
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device,
        queue,
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        session.submission_session(),
        WgpuDecodeSubmissionSession::ProgressiveDc(_)
    ));
    assert_eq!(session.submission_session().submissions_per_frame(), 2);

    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    assert!(session.next_frame().unwrap().is_none());
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let expected = rust_jxl_rgb8(&encoded, Extent2d::new(1_024, 128));
    assert_eq!(actual.len(), expected.len());
    let error = maximum_error(actual, &expected);
    assert!(
        error <= 1,
        "GPU-resident progressive-DC output diverges from Rust jxl by {error}",
    );

    let mut blocking_session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let blocking_frame = blocking_session.next_frame().unwrap().unwrap();
    assert!(blocking_session.next_frame().unwrap().is_none());
    let blocking_readback = ImageReadbackPipeline::new(&backend)
        .submit(blocking_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        blocking_readback.frame.outputs[0].bytes.as_slice(),
        actual.as_slice()
    );
}

#[test]
fn multi_level_progressive_dc_executes_general_single_packet_ac_on_gpu() {
    let Some(encoded) = common::cjxl_progressive_dc_codestream(2) else {
        return;
    };
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        session.submission_session(),
        WgpuDecodeSubmissionSession::ProgressiveDc(_)
    ));
    assert_eq!(session.submission_session().submissions_per_frame(), 4);
    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    assert!(matches!(
        session
            .front_pending_frame()
            .unwrap()
            .unvalidated_gpu_frame(),
        Err(DecodeError::VarDct(
            VarDctDecodeError::UnvalidatedOutputNotSubmitted
        ))
    ));
    let frame = session.next_frame().unwrap().unwrap();
    assert!(session.next_frame().unwrap().is_none());
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    let expected = rust_jxl_rgb8(&encoded, Extent2d::new(1_024, 128));
    assert_eq!(actual.len(), expected.len());
    let error = maximum_error(actual, &expected);
    assert!(
        error <= 1,
        "multi-level GPU-resident progressive-DC output diverges from Rust jxl by {error}",
    );

    let mut async_session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let async_frame = pollster::block_on(async_session.next_frame_async())
        .unwrap()
        .unwrap();
    assert!(async_session.next_frame().unwrap().is_none());
    let async_readback = ImageReadbackPipeline::new(&backend)
        .submit(async_frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        async_readback.frame.outputs[0].bytes.as_slice(),
        actual.as_slice()
    );
    drop(async_readback);
    drop(async_frame);
    drop(async_session);
    drop(readback);
    drop(frame);
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let probe_decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut probe = probe_decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    probe.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    let initial_reservation = probe_decoder
        .engine()
        .in_flight_memory_stats()
        .reserved_bytes;
    assert!(initial_reservation > 0);
    let snapshot = backend.transient_memory_stats();
    let budget_blocker = backend
        .transient_memory_budget()
        .try_reserve(snapshot.limit_bytes - snapshot.reserved_bytes)
        .unwrap();
    assert!(matches!(
        probe.next_frame(),
        Err(DecodeError::MemoryBackpressure(
            MemoryBudgetError::Exhausted {
                requested_bytes,
                ..
            }
        )) if requested_bytes > 0
    ));
    drop(budget_blocker);
    drop(probe);
    let fence = backend.queue().submit(std::iter::empty());
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .unwrap();
    assert_eq!(
        probe_decoder
            .engine()
            .in_flight_memory_stats()
            .reserved_bytes,
        0
    );
}
