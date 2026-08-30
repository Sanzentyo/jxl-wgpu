#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{ImageReadbackPipeline, WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::vardct::engine::vardct_rgb8_format;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest};
use jxl_wgpu_encode::{BufferImageSource, VarDctEncoder, VarDctStrategy, WgpuContext};
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

fn solid_source(context: &WgpuContext, rgb: [u8; 3]) -> BufferImageSource {
    let extent = Extent2d::new(8, 8);
    let bytes = rgb.repeat(64);
    let layout = ImageLayout::from_planes(
        extent,
        vardct_rgb8_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: 0,
            row_stride: 24,
            sample_extent: extent,
            row_bytes: 24,
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

fn djxl_ppm(codestream: &[u8]) -> Option<Vec<u8>> {
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
    assert_eq!(next_token(&ppm, &mut cursor), b"8");
    assert_eq!(next_token(&ppm, &mut cursor), b"8");
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

#[test]
fn standard_packet_reaches_resident_rgb8_and_matches_djxl() {
    let Some((info, device, queue)) = device() else {
        eprintln!("skipping bounded VarDCT engine oracle: no adapter");
        return;
    };
    let context = WgpuContext::new(Arc::new(device.clone()), Arc::new(queue.clone())).unwrap();
    let encoded = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8)
        .unwrap()
        .encode(solid_source(&context, [19, 103, 229]))
        .unwrap();
    let oracle = djxl_ppm(&encoded);
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
    let mut session = decoder
        .open(
            &encoded,
            GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
        )
        .unwrap();
    let memory = session.submission_session().memory_stats();
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    let frame = session.next_frame().unwrap().unwrap();
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        memory.output_lease_bytes
    );
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    let actual = &readback.frame.outputs[0].bytes;
    assert_eq!(actual.len(), 8 * 8 * 3);
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
            "resident VarDCT output differs from djxl by {maximum_error} codes"
        );
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
