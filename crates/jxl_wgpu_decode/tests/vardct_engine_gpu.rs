#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroUsize;
use std::sync::{Arc, mpsc};

use jxl_gpu_bitstream::{InventoryLimits, ParseLimits};
use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    DisplayColorEncoding, DisplayPipeline, DisplayTexture, DisplayTextureDescriptor,
    ImageReadbackPipeline, WgpuBackend, WgpuBackendConfig,
};
use jxl_wgpu_decode::vardct::engine::vardct_rgb8_format;
use jxl_wgpu_decode::vardct::packet::{FixedVarDctPacketPlan, GpuVarDctPacketError};
use jxl_wgpu_decode::{Error as DecodeError, GpuDecoder, GpuOutputRequest, VarDctDecodeError};
use jxl_wgpu_encode::{
    BufferImageSource, VarDctColorEncoding, VarDctEncoder, VarDctStrategy, WgpuContext,
};
use wgpu::util::DeviceExt;

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
fn all_bounded_regular_packets_reach_display_and_reject_corrupt_entropy() {
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
    let decoder = GpuDecoder::vardct_wgpu(backend.clone());
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
        let memory = session.submission_session().memory_stats();
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
    let packet = FixedVarDctPacketPlan::parse(parsed.codestream(), &inventory).unwrap();
    let entropy_bit = usize::try_from(packet.entropy_bit_offset).unwrap();
    corrupted[entropy_bit / 8] ^= 1 << (entropy_bit % 8);

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
