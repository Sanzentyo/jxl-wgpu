#![cfg(not(target_arch = "wasm32"))]

use std::num::{NonZeroU64, NonZeroUsize};
use std::process::Command;
use std::sync::{Arc, mpsc};

use jxl::api::{
    Endianness, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use jxl_gpu_formats::{
    Channel, ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, ImageLayout,
    PitchLinearPlaneLayout, PixelFormat, SampleKind, TransferFunction, classify_pixel_format,
    convert_rgb_f32, vpi::VpiPitchLinearFormat as Vpi,
};
use jxl_gpu_protocol::{ChangedRegions, Extent2d, OutputId, SubmissionToken};
use jxl_wgpu::{
    DirectReadbackPolicy, DisplayPipeline, DisplayTextureDescriptor, GpuBufferLease, GpuImageFrame,
    GpuImageOutput, ImageReadbackPipeline, MemoryBudget, SUBMISSION_POLLER_CAPACITY,
    ShaderF64Policy, WgpuBackend, WgpuBackendConfig,
};
use jxl_wgpu_decode::{
    F64OutputPath, F64OutputPolicy, GpuDecoder, GpuOutputRequest, ModularChannels,
    ModularPredictor, ModularReconstructionSpecialization, NumericSampleMapping, OutputWritePath,
    PrefetchBackpressure, WgpuSubmissionEngine,
};
use jxl_wgpu_encode::{
    BufferImageSource, LosslessModularEncoder, LosslessModularFormat, WgpuContext,
};
use wgpu::util::DeviceExt;

mod common;

use common::gpu_gray8_lossless as indexed_gray8;
const VPI_COLOR_FORMATS: [Vpi; 20] = [
    Vpi::Y8,
    Vpi::Y8Er,
    Vpi::Y16,
    Vpi::Y16Er,
    Vpi::Nv12,
    Vpi::Nv12Er,
    Vpi::Nv24,
    Vpi::Nv24Er,
    Vpi::Uyvy,
    Vpi::UyvyEr,
    Vpi::Yuyv,
    Vpi::YuyvEr,
    Vpi::Rgb8,
    Vpi::Bgr8,
    Vpi::Rgba8,
    Vpi::Bgra8,
    Vpi::Rgb8Planar,
    Vpi::Bgr8Planar,
    Vpi::Rgba8Planar,
    Vpi::Bgra8Planar,
];
const VPI_NUMERIC_FORMATS: [Vpi; 10] = [
    Vpi::U8,
    Vpi::S8,
    Vpi::U16,
    Vpi::U32,
    Vpi::S32,
    Vpi::S16,
    Vpi::TwoS16,
    Vpi::F32,
    Vpi::F64,
    Vpi::TwoF32,
];

fn backend() -> Option<WgpuBackend> {
    let config = WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        ..WgpuBackendConfig::default()
    };
    backend_with_config(config)
}

fn direct_readback_backend() -> Option<WgpuBackend> {
    pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Auto,
        ..WgpuBackendConfig::default()
    }))
    .ok()
}

fn backend_with_config(config: WgpuBackendConfig) -> Option<WgpuBackend> {
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
        label: Some("jxl-wgpu indexed Gray8 decode test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    WgpuBackend::from_device(device, queue, info, config).ok()
}

fn native_f64_backend() -> Option<WgpuBackend> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    if !adapter.features().contains(wgpu::Features::SHADER_F64) {
        return None;
    }
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("jxl-wgpu indexed Gray8 native F64 decode test"),
        required_features: wgpu::Features::SHADER_F64,
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    let config = WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        shader_f64_policy: ShaderF64Policy::Require,
        ..WgpuBackendConfig::default()
    };
    WgpuBackend::from_device(device, queue, info, config).ok()
}

fn read_output(backend: &WgpuBackend, output: &jxl_wgpu::GpuImageOutput) -> Vec<u8> {
    let staging = backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("indexed Gray8 test output readback"),
        size: output.buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("indexed Gray8 test readback commands"),
        });
    commands.copy_buffer_to_buffer(
        output.buffer.as_wgpu_buffer(),
        0,
        &staging,
        0,
        output.buffer.size(),
    );
    let (sender, receiver) = mpsc::sync_channel(1);
    commands.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    let submission = backend.queue().submit([commands.finish()]);
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU output readback completes");
    receiver
        .recv()
        .expect("mapping callback runs")
        .expect("output mapping succeeds");
    let mapped = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped range is valid");
    let bytes = mapped[..output.layout.logical_size as usize].to_vec();
    drop(mapped);
    staging.unmap();
    bytes
}

fn expected_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(17 * 13);
    for y in 0..13u32 {
        for x in 0..17u32 {
            pixels.push(if y < 3 {
                0
            } else {
                ((x * 17 + y * 31 + (x * y) % 19) & 255) as u8
            });
        }
    }
    pixels
}

fn patterned_pixels(width: u32, height: u32) -> Vec<u8> {
    (0..u64::from(width) * u64::from(height))
        .map(|index| {
            let x = index % u64::from(width);
            let y = index / u64::from(width);
            ((x * 37 + y * 71 + (x * y) % 251) & 255) as u8
        })
        .collect()
}

fn encode_standard_gray8(
    backend: &WgpuBackend,
    width: u32,
    height: u32,
    pixels: &[u8],
    container: bool,
) -> Vec<u8> {
    let layout = ImageLayout::packed(
        Extent2d::new(width, height),
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
    )
    .unwrap();
    let buffer = Arc::new(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("standard multi-group Gray8 decode test source"),
                contents: pixels,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    let source = BufferImageSource::new(buffer, layout).unwrap();
    let encoder = LosslessModularEncoder::new(WgpuContext::from_backend(backend));
    if container {
        encoder.encode_container(source).unwrap()
    } else {
        encoder.encode(source).unwrap()
    }
}

fn patterned_modular_samples(
    format: LosslessModularFormat,
    bits_per_sample: u8,
    width: u32,
    height: u32,
) -> Vec<u16> {
    let channels = usize::try_from(format.channel_count()).unwrap();
    let mask = (1u32 << bits_per_sample) - 1;
    let mut samples = Vec::with_capacity(width as usize * height as usize * channels);
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let channel = channel as u32;
                let mut value = x
                    .wrapping_mul(257 + channel * 19)
                    .wrapping_add(y.wrapping_mul(509 + channel * 31))
                    .wrapping_add((x ^ y).wrapping_mul(17 + channel * 13))
                    .wrapping_add(channel * 997)
                    & mask;
                if (x + y * 3 + channel * 5).is_multiple_of(97) {
                    value = mask;
                } else if (x * 7 + y + channel * 11).is_multiple_of(89) {
                    value = 0;
                }
                samples.push(value as u16);
            }
        }
    }
    samples
}

fn packed_modular_bytes(samples: &[u16], bits_per_sample: u8) -> Vec<u8> {
    if bits_per_sample <= 8 {
        samples.iter().map(|&sample| sample as u8).collect()
    } else {
        samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect()
    }
}

fn encode_standard_modular_with_odd_stride(
    backend: &WgpuBackend,
    format: LosslessModularFormat,
    bits_per_sample: u8,
    width: u32,
    height: u32,
    samples: &[u16],
    container: bool,
) -> Vec<u8> {
    let pixel_format = format.pixel_format(bits_per_sample).unwrap();
    let channels = u64::from(format.channel_count());
    let bytes_per_sample = u64::from(if bits_per_sample <= 8 { 1u8 } else { 2u8 });
    let row_bytes = u64::from(width) * channels * bytes_per_sample;
    let row_stride = row_bytes + if row_bytes.is_multiple_of(2) { 1 } else { 2 };
    assert!(!row_stride.is_multiple_of(2));
    let offset = 3u64;
    let layout = ImageLayout::from_planes(
        Extent2d::new(width, height),
        pixel_format,
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset,
            row_stride,
            sample_extent: Extent2d::new(width, height),
            row_bytes,
        }],
    )
    .unwrap();
    let mut source_bytes = vec![0xa5; layout.logical_size.div_ceil(4) as usize * 4];
    let channels = usize::try_from(channels).unwrap();
    let bytes_per_sample = usize::try_from(bytes_per_sample).unwrap();
    for y in 0..height as usize {
        for x in 0..width as usize {
            for channel in 0..channels {
                let sample = samples[(y * width as usize + x) * channels + channel];
                let start = offset as usize
                    + y * row_stride as usize
                    + (x * channels + channel) * bytes_per_sample;
                if bytes_per_sample == 1 {
                    source_bytes[start] = sample as u8;
                } else {
                    source_bytes[start..start + 2].copy_from_slice(&sample.to_le_bytes());
                }
            }
        }
    }
    let buffer = Arc::new(
        backend
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("standard Modular odd-stride decode conformance source"),
                contents: &source_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    let source = BufferImageSource::new(buffer, layout).unwrap();
    let encoder = LosslessModularEncoder::new(WgpuContext::from_backend(backend));
    if container {
        encoder.encode_container(source).unwrap()
    } else {
        encoder.encode(source).unwrap()
    }
}

fn rust_jxl_decode_integer(
    encoded: &[u8],
    format: LosslessModularFormat,
    bits_per_sample: u8,
) -> Result<((usize, usize), Vec<u16>), String> {
    let mut input = encoded;
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match decoder
            .process(&mut input, None)
            .map_err(|error| error.to_string())?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("Rust jxl oracle needed more input before image info".into());
                }
                decoder = fallback;
            }
        }
    };
    let basic_info = decoder.basic_info();
    if basic_info.bit_depth
        != (JxlBitDepth::Int {
            bits_per_sample: u32::from(bits_per_sample),
        })
    {
        return Err(format!(
            "Rust jxl oracle depth {:?} does not match {bits_per_sample}",
            basic_info.bit_depth
        ));
    }
    let size = basic_info.size;
    let data_format = if bits_per_sample <= 8 {
        JxlDataFormat::U8 {
            bit_depth: bits_per_sample,
        }
    } else {
        JxlDataFormat::U16 {
            endianness: Endianness::LittleEndian,
            bit_depth: bits_per_sample,
        }
    };
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: match format {
            LosslessModularFormat::Gray => JxlColorType::Grayscale,
            LosslessModularFormat::Rgb => JxlColorType::Rgb,
            LosslessModularFormat::Rgba => JxlColorType::Rgba,
        },
        color_data_format: Some(data_format),
        extra_channel_format: vec![None; usize::from(format.has_alpha())],
    });
    let mut frame = loop {
        match decoder
            .process(&mut input, None)
            .map_err(|error| error.to_string())?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("Rust jxl oracle needed more input before frame info".into());
                }
                decoder = fallback;
            }
        }
    };
    let channels = usize::try_from(format.channel_count()).unwrap();
    let bytes_per_sample = data_format.bytes_per_sample();
    let row_bytes = size.0 * channels * bytes_per_sample;
    let mut bytes = vec![0u8; row_bytes * size.1];
    {
        let mut buffers = [JxlOutputBuffer::new(&mut bytes, size.1, row_bytes)];
        loop {
            match frame
                .process(&mut input, &mut buffers, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("Rust jxl oracle needed more input while rendering".into());
                    }
                    frame = fallback;
                }
            }
        }
    }
    let samples = if bytes_per_sample == 1 {
        bytes.into_iter().map(u16::from).collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|sample| u16::from_le_bytes([sample[0], sample[1]]))
            .collect()
    };
    Ok((size, samples))
}

fn rust_jxl_decode_gray8(encoded: &[u8]) -> Result<((usize, usize), Vec<u8>), String> {
    let mut input = encoded;
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match decoder
            .process(&mut input, None)
            .map_err(|error| error.to_string())?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("Rust jxl oracle needed more input before image info".into());
                }
                decoder = fallback;
            }
        }
    };
    let size = decoder.basic_info().size;
    decoder.set_pixel_format(JxlPixelFormat {
        color_type: JxlColorType::Grayscale,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: Vec::new(),
    });
    let mut frame = loop {
        match decoder
            .process(&mut input, None)
            .map_err(|error| error.to_string())?
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.is_empty() {
                    return Err("Rust jxl oracle needed more input before frame info".into());
                }
                decoder = fallback;
            }
        }
    };
    let mut pixels = vec![0u8; size.0 * size.1];
    {
        let mut buffers = [JxlOutputBuffer::new(&mut pixels, size.1, size.0)];
        loop {
            match frame
                .process(&mut input, &mut buffers, None)
                .map_err(|error| error.to_string())?
            {
                ProcessingResult::Complete { .. } => break,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        return Err("Rust jxl oracle needed more input while rendering".into());
                    }
                    frame = fallback;
                }
            }
        }
    }
    Ok((size, pixels))
}

fn parse_binary_pgm(bytes: &[u8]) -> Result<((usize, usize), &[u8]), String> {
    fn token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a [u8], String> {
        loop {
            while bytes
                .get(*cursor)
                .is_some_and(|byte| byte.is_ascii_whitespace())
            {
                *cursor += 1;
            }
            if bytes.get(*cursor) != Some(&b'#') {
                break;
            }
            while bytes.get(*cursor).is_some_and(|byte| *byte != b'\n') {
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
        bytes
            .get(start..*cursor)
            .filter(|token| !token.is_empty())
            .ok_or_else(|| "truncated PGM header".into())
    }

    let mut cursor = 0usize;
    if token(bytes, &mut cursor)? != b"P5" {
        return Err("djxl did not emit a binary grayscale PGM".into());
    }
    let width = std::str::from_utf8(token(bytes, &mut cursor)?)
        .map_err(|error| error.to_string())?
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let height = std::str::from_utf8(token(bytes, &mut cursor)?)
        .map_err(|error| error.to_string())?
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    if token(bytes, &mut cursor)? != b"255" {
        return Err("djxl PGM did not use 8-bit samples".into());
    }
    match bytes.get(cursor..) {
        Some([b'\r', b'\n', ..]) => cursor += 2,
        Some([byte, ..]) if byte.is_ascii_whitespace() => cursor += 1,
        _ => return Err("djxl PGM is missing its raster separator".into()),
    }
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| "djxl PGM extent overflow".to_string())?;
    let pixels = bytes
        .get(cursor..)
        .filter(|pixels| pixels.len() == expected)
        .ok_or_else(|| "djxl PGM raster length mismatch".to_string())?;
    Ok(((width, height), pixels))
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn target_nonlinear(sample: u8, transfer: TransferFunction) -> f32 {
    let encoded = f32::from(sample) / 255.0;
    match transfer {
        TransferFunction::Srgb | TransferFunction::Sycc => encoded,
        TransferFunction::Linear => srgb_to_linear(encoded),
        TransferFunction::Bt709 | TransferFunction::Bt2020 => {
            let linear = srgb_to_linear(encoded);
            if linear < 0.018 {
                4.5 * linear
            } else {
                1.099 * linear.powf(0.45) - 0.099
            }
        }
        unsupported => panic!("test inventory contains unsupported transfer {unsupported:?}"),
    }
}

fn expected_numeric_bytes(
    format: &PixelFormat,
    layout: &jxl_gpu_formats::ImageLayout,
    source: &[u8],
    native_f64: bool,
) -> Vec<u8> {
    let numeric = classify_pixel_format(format)
        .unwrap()
        .numeric()
        .expect("numeric test format is classified as numeric");
    let mut bytes = vec![0; layout.logical_size as usize];
    let plane = &layout.planes[0];
    let component_bytes = usize::from(numeric.bits_per_component / 8);
    let components = usize::from(numeric.components);
    for y in 0..layout.extent.height as usize {
        for x in 0..layout.extent.width as usize {
            let sample = source[y * layout.extent.width as usize + x];
            let encoded = match (numeric.sample_kind, numeric.bits_per_component) {
                (SampleKind::Unsigned, 8) => vec![sample],
                (SampleKind::Unsigned, 16) => (u16::from(sample) * 257).to_le_bytes().to_vec(),
                (SampleKind::Unsigned, 32) => {
                    (u32::from(sample) * 16_843_009).to_le_bytes().to_vec()
                }
                (SampleKind::Signed, bits @ (8 | 16 | 32)) => {
                    let maximum = match bits {
                        8 => i8::MAX as u64,
                        16 => i16::MAX as u64,
                        32 => i32::MAX as u64,
                        _ => unreachable!(),
                    };
                    let mapped = (u64::from(sample) * maximum + 127) / 255;
                    match bits {
                        8 => vec![mapped as u8],
                        16 => (mapped as i16).to_le_bytes().to_vec(),
                        32 => (mapped as i32).to_le_bytes().to_vec(),
                        _ => unreachable!(),
                    }
                }
                (SampleKind::Float, 32) => (f32::from(sample) / 255.0).to_le_bytes().to_vec(),
                (SampleKind::Float, 64) => {
                    let normalized = if native_f64 {
                        f64::from(sample) / 255.0
                    } else {
                        // Compatibility widens normalized f32 exactly; it does not redo /255 in f64.
                        f64::from(f32::from(sample) / 255.0)
                    };
                    normalized.to_le_bytes().to_vec()
                }
                unsupported => panic!("unsupported numeric test class {unsupported:?}"),
            };
            assert_eq!(encoded.len(), component_bytes);
            for component in 0..components {
                let start = plane.offset as usize
                    + y * plane.row_stride as usize
                    + (x * components + component) * component_bytes;
                bytes[start..start + component_bytes].copy_from_slice(&encoded);
            }
        }
    }
    bytes
}

#[test]
fn indexed_jxl_entropy_and_gradient_reconstruct_exact_gray8_on_gpu() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 decode test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder
        .open(indexed_gray8(), request)
        .expect("indexed profile opens without CPU fallback");
    let frame = session
        .next_frame()
        .expect("GPU decode succeeds")
        .expect("one frame is returned");
    let output = &frame.output().outputs[0];
    assert_eq!(
        (output.layout.extent.width, output.layout.extent.height),
        (17, 13)
    );
    assert_eq!(read_output(&backend, output), expected_pixels());
}

#[test]
fn indexed_gray8_direct_maps_the_tracked_output_on_supported_uma() {
    let Some(backend) = direct_readback_backend() else {
        eprintln!("skipping direct Gray8 readback test: no wgpu adapter");
        return;
    };
    if !backend.direct_readback_enabled() {
        eprintln!("skipping direct Gray8 readback test: direct mapping is unavailable");
        return;
    }

    let decoder = GpuDecoder::wgpu(backend.clone());
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let output = &frame.output().outputs[0];
    assert!(output.buffer.usage().contains(wgpu::BufferUsages::MAP_READ));

    let readback = ImageReadbackPipeline::new(&backend);
    let abandoned = readback.submit(frame.output()).unwrap();
    assert!(abandoned.stats().direct_mapped);
    assert_eq!(abandoned.stats().staging_bytes, 0);
    let abandoned_submission = abandoned.submission().clone();
    drop(abandoned);
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(abandoned_submission),
            timeout: None,
        })
        .unwrap();

    let completed = readback.submit(frame.output()).unwrap();
    assert!(completed.stats().direct_mapped);
    assert_eq!(completed.stats().logical_bytes, 17 * 13);
    assert_eq!(completed.stats().staging_bytes, 0);
    assert_eq!(completed.stats().padding_bytes, 0);
    let result = completed.wait().unwrap();
    assert_eq!(result.frame.outputs[0].bytes, expected_pixels());
}

#[test]
fn standard_raw_and_jxlc_multigroup_extreme_aspects_reconstruct_exactly_on_gpu() {
    let Some(backend) = backend() else {
        eprintln!("skipping standard multi-group Gray8 test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    for (width, height, container) in [
        (513, 257, false),
        (513, 257, true),
        (1, 513, false),
        (769, 1, true),
        (257, 257, false),
        (516, 3, false),
    ] {
        let expected = patterned_pixels(width, height);
        let encoded = encode_standard_gray8(&backend, width, height, &expected, container);
        let (oracle_extent, oracle) = rust_jxl_decode_gray8(&encoded).unwrap_or_else(|error| {
            panic!("{width}x{height} container={container} Rust jxl oracle failed: {error}")
        });
        assert_eq!(oracle_extent, (width as usize, height as usize));
        assert_eq!(oracle, expected, "Rust jxl oracle {width}x{height}");
        let request = GpuOutputRequest::numeric(
            PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
            NumericSampleMapping::NormalizedGray8,
        )
        .unwrap();
        let mut session = decoder.open(&encoded, request).unwrap_or_else(|error| {
            panic!("{width}x{height} container={container} did not open: {error}")
        });
        assert!(matches!(
            session.profile(),
            jxl_wgpu_decode::DecodeProfile::ModularLossless {
                prediction: jxl_wgpu_decode::ModularPredictionProfile::MetaAdaptive {
                    node_count,
                    decision_node_count,
                    leaf_context_count,
                    max_depth,
                    ..
                },
                grouping: jxl_wgpu_decode::ModularGrouping::MultipleGroups { .. },
                ..
            } if decision_node_count.checked_add(leaf_context_count) == Some(node_count)
                && leaf_context_count != 0
                && max_depth != 0
        ));
        if (width, height, container) == (513, 257, false) {
            let stats = session.submission_session().memory_stats();
            assert!(stats.parallel_group_lanes > 1);
            assert!(stats.parallel_group_lanes <= 64);
            assert_eq!(stats.max_lz77_window_words, 1);
            assert_eq!(stats.max_lz77_scratch_words, 0);
            assert_eq!(stats.max_logical_reconstruction_sample_words, 256 * 256);
            assert_eq!(stats.max_physical_reconstruction_sample_words, 256 * 2);
            assert_eq!(stats.reconstruction_lane_stride_bytes, 256 * 2 * 4);
            assert_eq!(
                stats.output_specialization,
                jxl_wgpu_decode::ModularOutputSpecialization::DirectNormalizedGray8
            );
            assert_eq!(
                stats.reconstruction_specialization,
                ModularReconstructionSpecialization::ChannelFixed {
                    predictor: ModularPredictor::Gradient,
                    offset: 0,
                    multiplier: 1,
                    channel_count: 1,
                    clusters: [1, 2, 3, 4],
                }
            );
            assert_eq!(stats.group_workgroup_size, 64);
            assert_eq!(stats.max_dispatch_workgroups, 1);
            assert_eq!(
                stats.reconstruction_scratch_bytes,
                stats.reconstruction_lane_stride_bytes * stats.parallel_group_lanes as u64
            );
            assert!(stats.stream_window_bytes < stats.transient_bytes);
            assert_eq!(stats.stream_batch_count, 1);
            assert_eq!(stats.submissions_per_frame, 1);
            assert_eq!(stats.output_write_path, OutputWritePath::AtomicBytes);
        }
        if (width, height, container) == (516, 3, false) {
            let stats = session.submission_session().memory_stats();
            assert_eq!(stats.max_lz77_window_words, 1);
            assert_eq!(stats.max_lz77_scratch_words, 0);
            assert_eq!(stats.output_write_path, OutputWritePath::WordAligned);
        }
        let frame = session
            .next_frame()
            .unwrap_or_else(|error| {
                panic!("{width}x{height} container={container} failed: {error}")
            })
            .expect("one frame is returned");
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.extent, Extent2d::new(width, height));
        assert_eq!(
            read_output(&backend, output),
            expected,
            "{width}x{height} container={container}"
        );
    }
}

#[test]
fn standard_modular_native_matrix_is_exact_on_gpu_and_rust_oracle() {
    let Some(backend) = backend() else {
        eprintln!("skipping standard Modular native matrix: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let (width, height) = (257, 3);
    let formats = [
        (LosslessModularFormat::Gray, ModularChannels::Gray),
        (LosslessModularFormat::Rgb, ModularChannels::Rgb),
        (LosslessModularFormat::Rgba, ModularChannels::Rgba),
    ];
    let depths = [8u8, 10, 12, 16];
    for (case_index, ((format, expected_channels), bits_per_sample)) in formats
        .iter()
        .flat_map(|format| depths.iter().map(move |&bits| (*format, bits)))
        .enumerate()
    {
        let samples = patterned_modular_samples(format, bits_per_sample, width, height);
        let container = case_index % 2 != 0;
        let encoded = encode_standard_modular_with_odd_stride(
            &backend,
            format,
            bits_per_sample,
            width,
            height,
            &samples,
            container,
        );
        let (oracle_extent, oracle) =
            rust_jxl_decode_integer(&encoded, format, bits_per_sample).unwrap_or_else(|error| {
                panic!(
                    "{format:?} {bits_per_sample}-bit container={container} Rust oracle failed: {error}"
                )
            });
        assert_eq!(oracle_extent, (width as usize, height as usize));
        assert_eq!(
            oracle, samples,
            "{format:?} {bits_per_sample}-bit Rust oracle"
        );

        let pixel_format = format.pixel_format(bits_per_sample).unwrap();
        let request = if format == LosslessModularFormat::Gray {
            GpuOutputRequest::numeric(pixel_format, NumericSampleMapping::NativeUnsigned).unwrap()
        } else {
            GpuOutputRequest::color(pixel_format).unwrap()
        };
        let mut session = decoder.open(&encoded, request).unwrap_or_else(|error| {
            panic!("{format:?} {bits_per_sample}-bit container={container} did not open: {error}")
        });
        assert!(matches!(
            session.profile(),
            jxl_wgpu_decode::DecodeProfile::ModularLossless {
                bits_per_sample: actual_bits,
                channels: actual_channels,
                grouping: jxl_wgpu_decode::ModularGrouping::MultipleGroups { .. },
                ..
            } if actual_bits == bits_per_sample && actual_channels == expected_channels
        ));
        let bytes_per_sample = if bits_per_sample <= 8 { 1u64 } else { 2 };
        let row_bytes = u64::from(width) * u64::from(format.channel_count()) * bytes_per_sample;
        let stats = session.submission_session().memory_stats();
        assert_eq!(stats.max_lz77_window_words, 1);
        assert_eq!(stats.max_lz77_scratch_words, 0);
        let logical_sample_words = 256 * height * format.channel_count();
        assert_eq!(
            stats.max_logical_reconstruction_sample_words,
            logical_sample_words
        );
        assert_eq!(
            stats.max_physical_reconstruction_sample_words, logical_sample_words,
            "native {format:?} must retain the generic full-group layout"
        );
        assert_eq!(
            stats.output_specialization,
            jxl_wgpu_decode::ModularOutputSpecialization::FinalizePass
        );
        assert_eq!(
            stats.reconstruction_specialization,
            ModularReconstructionSpecialization::ChannelFixed {
                predictor: ModularPredictor::Gradient,
                offset: 0,
                multiplier: 1,
                channel_count: u8::try_from(format.channel_count()).unwrap(),
                clusters: [1, 2, 3, 4],
            },
            "{format:?} {bits_per_sample}-bit channel specialization"
        );
        assert_eq!(
            stats.output_write_path,
            if row_bytes.is_multiple_of(4) {
                OutputWritePath::WordAligned
            } else {
                OutputWritePath::AtomicBytes
            },
            "{format:?} {bits_per_sample}-bit output path"
        );
        let frame = session
            .next_frame()
            .unwrap_or_else(|error| {
                panic!("{format:?} {bits_per_sample}-bit container={container} failed: {error}")
            })
            .expect("one native Modular frame is returned");
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.extent, Extent2d::new(width, height));
        assert_eq!(
            read_output(&backend, output),
            packed_modular_bytes(&samples, bits_per_sample),
            "{format:?} {bits_per_sample}-bit container={container} GPU output"
        );
    }
}

#[test]
fn standard_modular_fused_rgb_and_rgba_groups_are_exact_on_gpu() {
    let Some(backend) = backend() else {
        eprintln!("skipping fused standard Modular color test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let (width, height) = (17, 13);
    for (format, bits_per_sample, container) in [
        (LosslessModularFormat::Rgb, 10, false),
        (LosslessModularFormat::Rgba, 16, true),
    ] {
        let samples = patterned_modular_samples(format, bits_per_sample, width, height);
        let encoded = encode_standard_modular_with_odd_stride(
            &backend,
            format,
            bits_per_sample,
            width,
            height,
            &samples,
            container,
        );
        let request =
            GpuOutputRequest::color(format.pixel_format(bits_per_sample).unwrap()).unwrap();
        let mut session = decoder.open(&encoded, request).unwrap();
        assert!(matches!(
            session.profile(),
            jxl_wgpu_decode::DecodeProfile::ModularLossless {
                grouping: jxl_wgpu_decode::ModularGrouping::SingleGroup,
                ..
            }
        ));
        let frame = session.next_frame().unwrap().unwrap();
        assert_eq!(
            read_output(&backend, &frame.output().outputs[0]),
            packed_modular_bytes(&samples, bits_per_sample),
            "fused {format:?} {bits_per_sample}-bit"
        );
    }
}

#[test]
fn standard_multigroup_codestream_is_exact_in_djxl_when_available() {
    if Command::new("djxl").arg("--version").output().is_err() {
        eprintln!("skipping djxl oracle: djxl is not installed");
        return;
    }
    let Some(backend) = backend() else {
        eprintln!("skipping djxl oracle: no wgpu adapter");
        return;
    };
    let (width, height) = (513, 257);
    let expected = patterned_pixels(width, height);
    let encoded = encode_standard_gray8(&backend, width, height, &expected, true);
    let unique = format!(
        "jxl-wgpu-decode-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos()
    );
    let input = std::env::temp_dir().join(format!("{unique}.jxl"));
    let output = std::env::temp_dir().join(format!("{unique}.pgm"));
    std::fs::write(&input, encoded).expect("write temporary djxl input");
    let command = Command::new("djxl")
        .arg(&input)
        .arg(&output)
        .arg("--quiet")
        .output()
        .expect("run djxl oracle");
    let decoded = std::fs::read(&output).unwrap_or_else(|error| {
        panic!(
            "djxl failed: status={} stderr={} read={error}",
            command.status,
            String::from_utf8_lossy(&command.stderr)
        )
    });
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
    assert!(
        command.status.success(),
        "djxl failed: {}",
        String::from_utf8_lossy(&command.stderr)
    );
    let (extent, pixels) = parse_binary_pgm(&decoded).expect("parse djxl PGM output");
    assert_eq!(extent, (width as usize, height as usize));
    assert_eq!(pixels, expected);
}

#[test]
fn every_multigroup_gpu_status_is_validated_from_one_map() {
    let Some(backend) = backend() else {
        eprintln!("skipping aggregate group-status test: no wgpu adapter");
        return;
    };
    let (width, height) = (513, 257);
    let pixels = patterned_pixels(width, height);
    let mut encoded = encode_standard_gray8(&backend, width, height, &pixels, false);
    let parsed = jxl_gpu_bitstream::parse(&encoded, jxl_gpu_bitstream::ParseLimits::default())
        .expect("generated raw codestream parses");
    let inventory = parsed
        .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
        .expect("generated raw codestream inventories");
    let damaged = inventory.frames[0]
        .sections
        .iter()
        .find(|section| {
            section.kind
                == jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                    pass_index: 0,
                    group_index: 1,
                }
        })
        .expect("second pass group exists")
        .bytes;
    let start = usize::try_from(damaged.offset).unwrap();
    let end = usize::try_from(damaged.end().unwrap()).unwrap();
    encoded[start] &= 0x0f;
    encoded[start + 1..end].fill(0);

    let decoder = GpuDecoder::wgpu(backend);
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder
        .open(&encoded, request)
        .expect("damaged entropy remains valid bounded metadata");
    let error = session
        .next_frame()
        .expect_err("damaged second group must fail GPU status validation");
    assert!(
        error
            .to_string()
            .contains("group 1 rejected entropy stream"),
        "unexpected aggregate status error: {error}"
    );
}

#[test]
fn stock_still_prefetch_submits_exactly_one_frame_before_waiting() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 prefetch test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend);
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    let progress = session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
    assert_eq!(progress.submitted, 1);
    assert_eq!(progress.queued, 1);
    assert!(progress.end_reached);
    assert_eq!(progress.backpressure, None);
    assert_eq!(session.active_frame_slots(), 1);

    let frame = session.next_frame().unwrap().unwrap();
    assert_eq!(frame.metadata.index, 0);
    assert!(frame.metadata.is_last);
}

#[test]
fn unvalidated_handoff_queues_display_and_readback_before_codec_validation() {
    let Some(backend) = backend() else {
        eprintln!("skipping unvalidated GPU handoff test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let mut session = decoder
        .open(
            indexed_gray8(),
            GpuOutputRequest::color(Vpi::Y8Er.pixel_format()).unwrap(),
        )
        .expect("indexed profile opens");
    session
        .prefetch(NonZeroUsize::new(1).unwrap())
        .expect("decode is submitted without a validation wait");

    assert_eq!(session.pending_frames().len(), 1);
    let unvalidated = session
        .front_pending_frame()
        .expect("ordered pending queue has its front")
        .unvalidated_gpu_frame()
        .expect("stock pending output can be leased before validation");
    assert_eq!(unvalidated.outputs.len(), 1);
    assert!(unvalidated.outputs[0].buffer.reserved_bytes() > 0);

    let display = DisplayPipeline::new(&backend)
        .submit_unvalidated_image(&unvalidated.outputs[0], DisplayTextureDescriptor::default())
        .expect("same-queue display dispatch is accepted before validation");
    let readback = ImageReadbackPipeline::new(&backend)
        .submit_unvalidated(&unvalidated)
        .expect("same-queue readback is accepted before validation");
    assert_eq!(session.queued_frames(), 1);
    assert!(!session.is_finished());

    // Waiting for the later readback submission proves the earlier decode and display submissions
    // executed in queue order. It does not consume the decoder's validation result.
    let readback = readback.wait().expect("unvalidated transport completes");
    assert_eq!(readback.token, unvalidated.token);
    assert_eq!(readback.outputs.len(), 1);
    assert_eq!(display.texture.extent, Extent2d::new(17, 13));
    assert_eq!(session.queued_frames(), 1);
    assert!(!session.is_finished());

    let validated = session
        .next_frame()
        .expect("codec status validates after consumers were queued")
        .expect("one validated frame is returned");
    assert_eq!(validated.output().token, unvalidated.token);
    assert_eq!(
        readback.outputs[0].bytes,
        read_output(&backend, &validated.output().outputs[0])
    );
}

#[test]
fn unvalidated_lease_survives_abandoned_session_without_losing_budget_tracking() {
    let Some(backend) = backend() else {
        eprintln!("skipping unvalidated ownership test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
    let complete_job_bytes = decoder.engine().in_flight_memory_stats().reserved_bytes;
    let unvalidated = session
        .front_pending_frame()
        .unwrap()
        .unvalidated_gpu_frame()
        .unwrap();
    let output_bytes = unvalidated.outputs[0].buffer.reserved_bytes();
    assert!(complete_job_bytes > output_bytes);
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        complete_job_bytes,
        "cloning a tracked lease shares rather than duplicates its reservation"
    );

    drop(session);
    let commands = backend
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("unvalidated ownership completion fence"),
        });
    let fence = backend.queue().submit([commands.finish()]);
    backend
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("abandoned decode callback completes");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while decoder.engine().in_flight_memory_stats().reserved_bytes != output_bytes
        && std::time::Instant::now() < deadline
    {
        backend
            .device()
            .poll(wgpu::PollType::Poll)
            .expect("drive abandoned decode callback");
        std::thread::yield_now();
    }
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        output_bytes,
        "only the caller-held unvalidated output remains reserved"
    );
    drop(unvalidated);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn indexed_gray8_writes_native_limited_nv12_without_rgb_readback() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 NV12 test: no wgpu adapter");
        return;
    };
    let mut spec = ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER);
    // Keep the source codestream's nonlinear sRGB transfer so this test isolates native YUV
    // packing/range conversion from transfer-function conversion.
    spec.transfer = TransferFunction::Srgb;
    let format = PixelFormat::nv12(ColorSpecification::Defined(spec));
    let decoder = GpuDecoder::wgpu(backend.clone());
    let mut session = decoder
        .open(indexed_gray8(), GpuOutputRequest::color(format).unwrap())
        .expect("native NV12 request is supported");
    let frame = session
        .next_frame()
        .expect("GPU NV12 decode succeeds")
        .expect("one frame is returned");
    let output = &frame.output().outputs[0];
    let bytes = read_output(&backend, output);
    let y_plane = &output.layout.planes[0];
    for (index, sample) in expected_pixels().into_iter().enumerate() {
        let expected = (16.0 + 219.0 * f32::from(sample) / 255.0).round() as u8;
        assert_eq!(bytes[y_plane.offset as usize + index], expected);
    }
    let uv_plane = &output.layout.planes[1];
    let uv = &bytes[uv_plane.byte_range().unwrap().start as usize
        ..uv_plane.byte_range().unwrap().end as usize];
    assert!(uv.iter().all(|&sample| sample == 128));
}

#[test]
fn indexed_gray8_conforms_to_all_vpi_color_formats_on_gpu() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 VPI format test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    let source = expected_pixels();

    for predefined in VPI_COLOR_FORMATS {
        let format = predefined.pixel_format();
        let ColorSpecification::Defined(spec) = format.color_spec else {
            panic!("{} must carry explicit color semantics", predefined.name());
        };
        let target = source
            .iter()
            .copied()
            .map(|sample| target_nonlinear(sample, spec.transfer))
            .collect::<Vec<_>>();
        let expected = convert_rgb_f32(
            [&target, &target, &target],
            jxl_gpu_protocol::Extent2d::new(17, 13),
            &format,
        )
        .unwrap_or_else(|error| panic!("{} oracle conversion failed: {error}", predefined.name()));

        let mut session = decoder
            .open(indexed_gray8(), GpuOutputRequest::color(format).unwrap())
            .unwrap_or_else(|error| panic!("{} request was rejected: {error}", predefined.name()));
        let frame = session
            .next_frame()
            .unwrap_or_else(|error| panic!("{} GPU decode failed: {error}", predefined.name()))
            .unwrap_or_else(|| panic!("{} returned no frame", predefined.name()));
        let output = &frame.output().outputs[0];
        assert_eq!(
            output.layout,
            expected.layout,
            "{} layout",
            predefined.name()
        );
        assert_eq!(
            read_output(&backend, output),
            expected.bytes,
            "{} bytes",
            predefined.name()
        );
        drop(frame);
        drop(session);
    }
}

#[test]
fn indexed_gray8_conforms_to_all_vpi_numeric_formats_on_gpu() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 numeric VPI format test: no wgpu adapter");
        return;
    };
    assert_eq!(VPI_COLOR_FORMATS.len() + VPI_NUMERIC_FORMATS.len(), 30);
    let decoder = GpuDecoder::wgpu(backend.clone());
    let source = expected_pixels();

    for predefined in VPI_NUMERIC_FORMATS {
        let format = predefined.pixel_format();
        let mapping = if predefined == Vpi::F64 {
            NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::ExactF32Widening)
        } else {
            NumericSampleMapping::NormalizedGray8
        };
        let mut session = decoder
            .open(
                indexed_gray8(),
                GpuOutputRequest::numeric(format.clone(), mapping).unwrap(),
            )
            .unwrap_or_else(|error| panic!("{} request was rejected: {error}", predefined.name()));
        let frame = session
            .next_frame()
            .unwrap_or_else(|error| panic!("{} GPU decode failed: {error}", predefined.name()))
            .unwrap_or_else(|| panic!("{} returned no frame", predefined.name()));
        let output = &frame.output().outputs[0];
        assert_eq!(output.layout.format, format, "{} format", predefined.name());
        assert_eq!(
            read_output(&backend, output),
            expected_numeric_bytes(&format, &output.layout, &source, false),
            "{} bytes",
            predefined.name()
        );
        drop(frame);
        drop(session);
    }
}

#[test]
fn indexed_gray8_uses_native_f64_when_shader_f64_is_enabled() {
    let Some(backend) = native_f64_backend() else {
        eprintln!(
            "skipping native F64 decode test: the selected adapter does not expose SHADER_F64"
        );
        return;
    };
    assert!(backend.native_f64_enabled());
    let engine = WgpuSubmissionEngine::new(backend.clone());
    assert!(engine.capabilities().native_f64_arithmetic);
    let decoder = GpuDecoder::new(engine);
    let format = Vpi::F64.pixel_format();
    let request = GpuOutputRequest::numeric(
        format.clone(),
        NumericSampleMapping::NormalizedGray8F64(F64OutputPolicy::NativeRequired),
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    assert_eq!(
        session.submission_session().f64_output_path(),
        Some(F64OutputPath::NativeArithmetic)
    );
    let frame = session.next_frame().unwrap().unwrap();
    let output = &frame.output().outputs[0];
    assert_eq!(
        read_output(&backend, output),
        expected_numeric_bytes(&format, &output.layout, &expected_pixels(), true)
    );
}

#[test]
fn indexed_gpu_future_reports_and_releases_bounded_memory() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 async test: no wgpu adapter");
        return;
    };
    let engine = WgpuSubmissionEngine::with_memory_budget(
        backend.clone(),
        MemoryBudget::new(NonZeroU64::new(2 * 1024 * 1024).unwrap()),
    );
    let decoder = GpuDecoder::new(engine);
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    let stats = session.submission_session().memory_stats();
    assert!(stats.per_frame_bytes > 128 * 1024);
    assert_eq!(stats.max_frame_slots, 2);
    assert_eq!(
        stats.max_frame_window_bytes,
        stats.per_frame_bytes * stats.max_frame_slots as u64
    );
    assert_eq!(
        stats.per_frame_bytes,
        stats.output_lease_bytes + stats.transient_bytes
    );
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let frame = pollster::block_on(session.next_frame_async())
        .expect("runtime-neutral GPU future succeeds")
        .expect("one frame is returned");
    assert_eq!(
        read_output(&backend, &frame.output().outputs[0]),
        expected_pixels()
    );
    let output_lease = frame.output().outputs[0].buffer.clone();
    assert_eq!(output_lease.reserved_bytes(), stats.output_lease_bytes);
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        stats.output_lease_bytes
    );
    drop(frame);
    drop(session);
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        stats.output_lease_bytes,
        "the output lease outlives its decode session"
    );
    drop(output_lease);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn decode_output_and_generic_readback_share_backend_memory_admission() {
    const SHARED_LIMIT: u64 = 1024 * 1024;
    let mut config = WgpuBackendConfig {
        enable_timestamps: false,
        direct_readback_policy: DirectReadbackPolicy::Disabled,
        ..WgpuBackendConfig::default()
    };
    config.memory.max_in_flight_transient_bytes = SHARED_LIMIT;
    let Some(backend) = backend_with_config(config) else {
        eprintln!("skipping decode/readback shared-budget test: no wgpu adapter");
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone());
    assert_eq!(decoder.engine().memory_budget_bytes(), SHARED_LIMIT);
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();
    let decoded = session.next_frame().unwrap().unwrap();
    let decode_output_bytes = decoded.output().outputs[0].buffer.reserved_bytes();
    assert!(decode_output_bytes > 0);
    assert_eq!(
        backend.transient_memory_stats().reserved_bytes,
        decode_output_bytes
    );

    let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
    let layout = ImageLayout::packed(
        Extent2d::new(u32::try_from(SHARED_LIMIT).unwrap(), 1),
        format,
    )
    .unwrap();
    assert_eq!(layout.logical_size, SHARED_LIMIT);
    let source = Arc::new(backend.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("shared decode/readback budget test source"),
        size: SHARED_LIMIT,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));
    let large_frame = GpuImageFrame {
        token: SubmissionToken(77),
        outputs: vec![GpuImageOutput {
            id: OutputId(0),
            layout,
            buffer: GpuBufferLease::new(source),
        }],
        changed: ChangedRegions::default(),
    };
    let readback = ImageReadbackPipeline::new(&backend);
    assert!(matches!(
        readback.submit(&large_frame),
        Err(jxl_wgpu::Error::MemoryBackpressure(
            jxl_wgpu::MemoryBudgetError::Exhausted { .. }
        ))
    ));

    drop(decoded);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    let result = readback
        .submit(&large_frame)
        .expect("readback is admitted after the decode output lease is dropped")
        .wait()
        .expect("admitted readback completes");
    assert_eq!(result.frame.outputs[0].bytes.len() as u64, SHARED_LIMIT);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
}

#[test]
fn output_leases_backpressure_other_sessions_until_dropped() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 memory-pressure test: no wgpu adapter");
        return;
    };
    let format = PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]);
    let probe = GpuDecoder::new(WgpuSubmissionEngine::new(backend.clone()));
    let probe_session = probe
        .open(
            indexed_gray8(),
            GpuOutputRequest::numeric(format.clone(), NumericSampleMapping::NormalizedGray8)
                .unwrap(),
        )
        .unwrap();
    let per_frame = probe_session
        .submission_session()
        .memory_stats()
        .per_frame_bytes;
    drop(probe_session);
    drop(probe);

    let decoder = GpuDecoder::new(WgpuSubmissionEngine::with_memory_budget(
        backend,
        MemoryBudget::new(NonZeroU64::new(per_frame).unwrap()),
    ));
    let mut first = decoder
        .open(
            indexed_gray8(),
            GpuOutputRequest::numeric(format.clone(), NumericSampleMapping::NormalizedGray8)
                .unwrap(),
        )
        .unwrap();
    let mut second = decoder
        .open(
            indexed_gray8(),
            GpuOutputRequest::numeric(format, NumericSampleMapping::NormalizedGray8).unwrap(),
        )
        .unwrap();
    let frame = first.next_frame().unwrap().unwrap();
    let output_bytes = first.submission_session().memory_stats().output_lease_bytes;
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        output_bytes
    );

    let error = match second.next_frame() {
        Err(error) => error,
        Ok(_) => panic!("the second session must be byte-backpressured"),
    };
    assert!(matches!(
        error,
        jxl_wgpu_decode::Error::MemoryBackpressure(_)
    ));
    drop(frame);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    let second_frame = second
        .next_frame()
        .expect("memory backpressure is retryable")
        .expect("the retried session returns its frame");
    drop(second_frame);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn poller_saturation_is_retryable_without_consuming_the_decode_source() {
    let Some(backend) = backend() else {
        eprintln!("skipping indexed Gray8 poll-admission test: no wgpu adapter");
        return;
    };
    let mut held = (0..SUBMISSION_POLLER_CAPACITY)
        .map(|_| backend.submission_poller().try_reserve().unwrap())
        .collect::<Vec<_>>();
    let decoder = GpuDecoder::wgpu(backend.clone());
    let request = GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Unsigned, 8, &[Channel::X]),
        NumericSampleMapping::NormalizedGray8,
    )
    .unwrap();
    let mut session = decoder.open(indexed_gray8(), request).unwrap();

    let blocked = session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
    assert_eq!(blocked.submitted, 0);
    assert_eq!(blocked.queued, 0);
    assert_eq!(
        blocked.backpressure,
        Some(PrefetchBackpressure::SubmissionPoller(
            jxl_wgpu::SubmissionPollerError::Full {
                capacity: SUBMISSION_POLLER_CAPACITY,
            }
        ))
    );
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);

    drop(held.pop());
    let retried = session.prefetch(NonZeroUsize::new(2).unwrap()).unwrap();
    assert_eq!(retried.submitted, 1);
    assert_eq!(retried.queued, 1);
    assert!(retried.end_reached);
    let frame = session
        .next_frame()
        .expect("poll admission backpressure is retryable")
        .expect("the preserved source still decodes");
    assert_eq!(
        read_output(&backend, &frame.output().outputs[0]),
        expected_pixels()
    );
    drop(frame);
    drop(held);
}
