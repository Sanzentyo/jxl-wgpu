#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};

use jxl::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use jxl_gpu_bitstream::{
    EdgePreservingFilterInventory, GaborishInventory, InventoryLimits, ParseLimits,
    RestorationFilterInventory,
};
use jxl_gpu_formats::{Channel, ImageLayout, PitchLinearPlaneLayout, PixelFormat, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    DisplayColorEncoding, DisplayPipeline, DisplayTexture, DisplayTextureDescriptor,
    ImageReadbackPipeline, WgpuBackend, WgpuBackendConfig,
};
use jxl_wgpu_decode::vardct::engine::vardct_rgb8_format;
use jxl_wgpu_decode::vardct::packet::{BoundedVarDctPacketPlan, GpuVarDctPacketError};
use jxl_wgpu_decode::{
    DecodeProfile, Error as DecodeError, GpuDecoder, GpuOutputRequest, NumericSampleMapping,
    VarDctDecodeError,
};
use jxl_wgpu_encode::{
    BufferImageSource, TiledVarDctEncoder, VarDctColorEncoding, VarDctEncoder, VarDctStrategy,
    WgpuContext,
};
use wgpu::util::DeviceExt;

mod common;

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
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left.abs_diff(right))
        .max()
        .unwrap_or(0)
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
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
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
    assert_eq!(next_token(&ppm, &mut cursor), b"P6");
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
    Some(match maximum {
        255 => pixels.to_vec(),
        65_535 => pixels
            .chunks_exact(2)
            .map(|pair| {
                let value = u16::from_be_bytes([pair[0], pair[1]]);
                ((u32::from(value) + 128) / 257) as u8
            })
            .collect(),
        _ => panic!("djxl PPM uses unsupported maximum {maximum}"),
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
    let mut modular_session = decoder
        .open(common::gpu_gray8_lossless(), modular_request)
        .unwrap();
    assert!(matches!(
        modular_session.profile(),
        DecodeProfile::ModularLossless { .. }
    ));
    assert!(modular_session.submission_session().modular().is_some());
    let modular_frame = modular_session.next_frame().unwrap().unwrap();
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
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        let frame = if index == 0 {
            pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap()
        } else {
            session.next_frame().unwrap().unwrap()
        };
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

    let mut rejected = decoder
        .open(
            &corrupted,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .expect("the corrupted image entropy remains host-parseable");
    rejected.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    let unvalidated = rejected
        .front_pending_frame()
        .unwrap()
        .unvalidated_gpu_frame()
        .unwrap();
    let output_bytes = unvalidated.outputs[0].buffer.reserved_bytes();
    let error = match rejected.next_frame() {
        Err(error) => error,
        Ok(_) => panic!("corrupt VarDCT image entropy must not become authoritative"),
    };
    assert!(matches!(
        error,
        DecodeError::VarDct(VarDctDecodeError::PacketGpu(GpuVarDctPacketError::LfHeader))
    ));
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        output_bytes,
        "explicitly unvalidated output stays accounted but must be discarded"
    );
    drop(unvalidated);
    drop(rejected);
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
        assert_eq!(plan.uniform_transform, None);
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
        assert_eq!(hf_coefficients.context_map.len(), 495 * 15);
        assert_eq!(hf_coefficients.block_context_map.len(), 39);
        assert_eq!(
            hf_coefficients.pass_groups.len() as u64,
            plan.profile.group_count
        );
        assert!(hf_coefficients.metadata.len() >= 28);
        assert_eq!(hf_coefficients.lz77_window_words, 0);
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
            memory.xyb_plane_bytes,
            u64::from(extent.width.div_ceil(8) * 8) * u64::from(extent.height.div_ceil(8) * 8) * 12,
        );
        assert_eq!(
            memory.resident_transient_bytes,
            2 * u64::from(blocks * 8 * 8 * 3) * std::mem::size_of::<f32>() as u64 + 27 * 128,
        );
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
    assert_eq!(hf.pass_groups.len(), 6);
    assert_eq!(hf.order_coordinate_offset_words, 13 * 3 * 4);
    let descriptors = bytemuck::cast_slice::<
        u32,
        jxl_wgpu_decode::vardct::artifact::GpuHfOrderDescriptor,
    >(&hf.order_words[..hf.order_coordinate_offset_words as usize]);
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
    assert_eq!(plan.uniform_transform, None);
    assert_eq!(plan.groups.len(), 1);
    assert_eq!(plan.groups[0].extra_precision, 1);
    assert_eq!(plan.groups[0].task_capacity, 33 * 33);
    let hf = plan.hf_coefficients.as_ref().unwrap();
    assert_eq!(hf.num_block_clusters, 3);
    let descriptors = bytemuck::cast_slice::<
        u32,
        jxl_wgpu_decode::vardct::artifact::GpuHfOrderDescriptor,
    >(&hf.order_words[..hf.order_coordinate_offset_words as usize]);
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

fn assert_multiple_lf_groups(encoded: &[u8], adaptive_lf_smoothing: bool) {
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
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
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
    assert_eq!(vardct.submissions_per_frame(), 1);
    assert_eq!(memory.packet_status_bytes, 2 * 64);
    assert_eq!(memory.adaptive_lf_uniform_bytes != 0, adaptive_lf_smoothing,);
    assert_eq!(
        memory.validation_staging_bytes,
        memory.packet_status_bytes
            + 2 * std::mem::size_of::<jxl_wgpu_decode::vardct::artifact::GpuVarDctArtifactStatus>()
                as u64
            + memory.hf_status_bytes,
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
    assert_multiple_lf_groups(common::testsrc_vardct_multi_lf(), true);
}

#[test]
fn multiple_lf_groups_can_skip_adaptive_smoothing_without_a_cpu_copy() {
    assert_multiple_lf_groups(common::testsrc_vardct_multi_lf_skip_smoothing(), false);
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
    assert_eq!(memory.restoration_scratch_bytes, memory.xyb_plane_bytes);
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
        assert_eq!(memory.restoration_scratch_bytes, memory.xyb_plane_bytes);
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
