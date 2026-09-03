//! Existing VarDCT semantic, ABI, and GPU interoperability tests.

use std::fs;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use jxl::api::{
    JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult, states,
};
use jxl_gpu_bitstream::{BitWriter, FiniteF16};
use jxl_gpu_formats::{ImageLayout, PitchLinearPlaneLayout};
use jxl_gpu_protocol::Extent2d;
use jxl_wgpu::{
    AdapterFingerprint, AutotuneProfile, ImageReadbackPipeline, KernelPolicy, KernelVariant,
    TunedKernel, WgpuBackend, WgpuBackendConfig,
};
use jxl_wgpu_decode::{
    GpuDecoder, GpuOutputRequest, VarDctOutputFormat, vardct::packet::BoundedVarDctPacketPlan,
    vardct_output_format,
};
use wgpu::util::DeviceExt;

use crate::prefix::{PrefixCode, RAW_SYMBOLS};

use super::bitstream::{build_frame_packet, image_header};
use super::dispatch::{
    BOUNDED_KERNEL_KEY, LARGE_SHADER, SCALABLE_QUANTIZE_KERNEL_KEY, SHADER, TiledVarDctEncoder,
    VarDctEncoder, align_up, clamped_gradient_i32, fixed_artifact_data, gradient_residual_i32,
    signed_token,
};
use super::entropy::{HfEntropyPlan, fixed_prefix_code, prefix_entries};
use super::types::{
    GpuPrefixEntry, MAX_AC_FRAGMENT_WORDS, MAX_BLOCKS, MAX_COEFFICIENTS, MAX_DC_FRAGMENT_WORDS,
    MAX_DC_SAMPLES, SCALABLE_HEADER_WORDS, SCALABLE_SECTION_ALIGNMENT_WORDS,
    ScalableArtifactLayout, ScalableDcFragmentDescriptor, ScalableVarDctArtifactHeader,
    ScalableVarDctKernelParams, TiledVarDctGrid, VarDctColorEncoding, VarDctFrameLayout,
    VarDctKernelArtifact, VarDctKernelLayout, VarDctKernelParams, VarDctLfMetadata, VarDctStrategy,
};
use crate::{BufferImageSource, EncodeError, UnsupportedFeature, WgpuContext, assemble_frame};

#[cfg(test)]
fn cpu_test_artifact(q_yxb: [i32; 3], code: &PrefixCode) -> VarDctKernelArtifact {
    let mut fragment = BitWriter::new();
    let mut histogram = [0u32; RAW_SYMBOLS];
    let mut quantized_dc_yxb = [0i32; MAX_DC_SAMPLES];
    let mut raw_tokens = [0u32; MAX_DC_SAMPLES];
    let mut extra_bits = [0u32; MAX_DC_SAMPLES];
    for (channel, value) in q_yxb.into_iter().enumerate() {
        let index = channel * MAX_BLOCKS;
        quantized_dc_yxb[index] = value;
        let packed = if value >= 0 {
            (value as u32) << 1
        } else {
            ((-i64::from(value)) as u32) * 2 - 1
        };
        let nbits = if packed == 0 {
            0
        } else {
            31 - packed.leading_zeros()
        };
        let token = u32::from(packed != 0) + nbits;
        let extra = packed.saturating_sub(1u32 << nbits);
        code.write_raw(&mut fragment, token, nbits, extra).unwrap();
        histogram[token as usize] += 1;
        raw_tokens[index] = token;
        extra_bits[index] = extra;
    }
    let bit_len = fragment.bit_len() as u32;
    let bytes = fragment.into_bytes();
    let mut words = [0u32; MAX_DC_FRAGMENT_WORDS];
    for (index, byte) in bytes.into_iter().enumerate() {
        words[index / 4] |= u32::from(byte) << ((index % 4) * 8);
    }
    let mut strategy_map = [0u32; MAX_BLOCKS];
    strategy_map[0] = 1 << 8;
    let mut quantized_xyb = [0; 3 * MAX_COEFFICIENTS];
    quantized_xyb[MAX_COEFFICIENTS] = q_yxb[0];
    quantized_xyb[0] = q_yxb[1];
    quantized_xyb[2 * MAX_COEFFICIENTS] = q_yxb[2];
    VarDctKernelArtifact {
        strategy_map,
        quantized_dc_yxb,
        dc_raw_tokens: raw_tokens,
        dc_extra_bits: extra_bits,
        dc_fragment_words: words,
        dc_fragment_bit_len: bit_len,
        dc_sample_count: 3,
        block_count: 1,
        strategy: 0,
        raw_histogram: histogram,
        dc_padding: [0; 9],
        ac_fragment_words: [0; MAX_AC_FRAGMENT_WORDS],
        ac_fragment_bit_len: 0,
        ac_token_count: 0,
        ac_histogram: [0; RAW_SYMBOLS],
        ac_padding: [0; 43],
        forward_xyb_bits: [0; 3 * MAX_COEFFICIENTS],
        quantized_xyb,
    }
}

fn f16(bits: u16) -> FiniteF16 {
    FiniteF16::from_bits(bits).expect("test binary16 value is finite")
}

fn custom_lf_metadata() -> VarDctLfMetadata {
    VarDctLfMetadata::new(
        [f16(0x2c00), f16(0x3000), f16(0x3400)],
        256,
        [f16(0xb800), f16(0x3800)],
        [-16, 32],
    )
    .unwrap()
}

fn decode_rgb8_sized(codestream: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut input = codestream;
    let mut decoder = JxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    assert_eq!(decoder.basic_info().size, (width, height));
    decoder.set_pixel_format(JxlPixelFormat::rgb8(0));
    let mut frame = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let mut pixels = vec![0u8; width * height * 3];
    let mut buffers = [JxlOutputBuffer::new(&mut pixels, height, width * 3)];
    loop {
        match frame.process(&mut input, &mut buffers, None).unwrap() {
            ProcessingResult::Complete { .. } => break,
            ProcessingResult::NeedsMoreInput { fallback, .. } => frame = fallback,
        }
    }
    pixels
}

fn decode_rgb8(codestream: &[u8]) -> Vec<u8> {
    decode_rgb8_sized(codestream, 8, 8)
}

fn test_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::AdapterInfo)> {
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
        label: Some("jxl-wgpu VarDCT encoder test"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue), info))
}

fn test_context() -> Option<WgpuContext> {
    let (device, queue, _) = test_device()?;
    WgpuContext::new(device, queue).ok()
}

fn test_context_with_variants(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    info: &wgpu::AdapterInfo,
    variants: &[(&str, KernelVariant)],
) -> Option<WgpuContext> {
    let mut profile = AutotuneProfile::new(AdapterFingerprint::from_adapter_info(info));
    for &(kernel, variant) in variants {
        profile.record(TunedKernel::from_samples(kernel, variant, &[1])?);
    }
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info.clone(),
        WgpuBackendConfig {
            enable_timestamps: false,
            kernel_policy: KernelPolicy::Profile(profile),
            ..WgpuBackendConfig::default()
        },
    )
    .ok()?;
    Some(WgpuContext::from_backend(&backend))
}

fn padded_rgb_source(context: &WgpuContext, pixels: &[[u8; 3]; 64]) -> BufferImageSource {
    padded_rgb_source_sized(context, 8, 8, pixels)
}

fn padded_rgb_source_sized(
    context: &WgpuContext,
    width: usize,
    height: usize,
    pixels: &[[u8; 3]],
) -> BufferImageSource {
    const OFFSET: u64 = 5;
    let row_bytes = (width * 3) as u64;
    let row_stride = row_bytes + 5;
    let extent = Extent2d::new(width as u32, height as u32);
    let allocation_size =
        align_up(OFFSET + row_stride * (height as u64 - 1) + row_bytes, 4).unwrap();
    let mut allocation = vec![0xa5; allocation_size as usize];
    for y in 0..height {
        let start = usize::try_from(OFFSET + row_stride * y as u64).unwrap();
        for x in 0..width {
            allocation[start + x * 3..start + x * 3 + 3].copy_from_slice(&pixels[y * width + x]);
        }
    }
    let layout = ImageLayout::from_planes(
        extent,
        VarDctColorEncoding::SrgbD65.pixel_format(),
        vec![PitchLinearPlaneLayout {
            plane_index: 0,
            offset: OFFSET,
            row_stride,
            sample_extent: extent,
            row_bytes,
        }],
    )
    .unwrap();
    let buffer = Arc::new(
        context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("jxl-wgpu padded VarDCT RGB fixture"),
                contents: &allocation,
                usage: wgpu::BufferUsages::STORAGE,
            }),
    );
    BufferImageSource::new(buffer, layout).unwrap()
}

fn psnr(reference: &[[u8; 3]], actual: &[u8]) -> f64 {
    let squared_error = reference
        .iter()
        .flatten()
        .zip(actual)
        .map(|(&expected, &observed)| {
            let difference = f64::from(expected) - f64::from(observed);
            difference * difference
        })
        .sum::<f64>();
    if squared_error == 0.0 {
        return f64::INFINITY;
    }
    let mse = squared_error / actual.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn max_abs_error(left: &[u8], right: &[u8]) -> u8 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left.abs_diff(right))
        .max()
        .unwrap_or(0)
}

fn oracle_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the test clock is after the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "jxl-wgpu-vardct-oracle-{}-{nonce}",
        std::process::id()
    ))
}

fn ppm_bytes(pixels: &[[u8; 3]], width: usize, height: usize) -> Vec<u8> {
    let mut output = format!("P6\n{width} {height}\n255\n").into_bytes();
    output.extend(pixels.iter().flatten().copied());
    output
}

fn next_ppm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> &'a [u8] {
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

fn read_ppm_rgb8(path: &Path, width: usize, height: usize) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    let mut cursor = 0usize;
    assert_eq!(next_ppm_token(&bytes, &mut cursor), b"P6");
    assert_eq!(
        next_ppm_token(&bytes, &mut cursor),
        width.to_string().as_bytes(),
    );
    assert_eq!(
        next_ppm_token(&bytes, &mut cursor),
        height.to_string().as_bytes(),
    );
    assert_eq!(next_ppm_token(&bytes, &mut cursor), b"255");
    assert!(bytes[cursor].is_ascii_whitespace());
    if bytes[cursor] == b'\r' && bytes.get(cursor + 1) == Some(&b'\n') {
        cursor += 2;
    } else {
        cursor += 1;
    }
    assert_eq!(bytes.len() - cursor, width * height * 3);
    bytes[cursor..].to_vec()
}

#[test]
fn fixed_control_plane_decodes_as_standard_black_vardct() {
    let code = fixed_prefix_code().unwrap();
    let hf_entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    assert!(prefix_entries(&code).iter().all(|entry| entry.bit_len > 0));
    let artifact = cpu_test_artifact([0, 0, 0], &code);
    let frame = assemble_frame(
        build_frame_packet(
            fixed_artifact_data(&artifact),
            &code,
            &hf_entropy,
            VarDctFrameLayout::single(VarDctStrategy::Dct8),
            VarDctLfMetadata::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut codestream = image_header(8, 8).unwrap().bytes().to_vec();
    codestream.extend_from_slice(frame.bytes());
    let decoded = decode_rgb8(&codestream);
    assert_eq!(decoded, vec![0; 8 * 8 * 3]);
}

#[test]
fn fixed_control_plane_accepts_nonzero_quantized_xyb_dc() {
    let code = fixed_prefix_code().unwrap();
    let hf_entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    // libjxl's DCT8 oracle quantizes a solid red block close to these
    // Y/X/(B-Y) values with this profile's global DC scale.
    let artifact = cpu_test_artifact([332, 153, -6], &code);
    let frame = assemble_frame(
        build_frame_packet(
            fixed_artifact_data(&artifact),
            &code,
            &hf_entropy,
            VarDctFrameLayout::single(VarDctStrategy::Dct8),
            VarDctLfMetadata::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut codestream = image_header(8, 8).unwrap().bytes().to_vec();
    codestream.extend_from_slice(frame.bytes());
    let decoded = decode_rgb8(&codestream);
    for pixel in decoded.chunks_exact(3) {
        assert!(pixel[0] > 240, "red={}", pixel[0]);
        assert!(pixel[1] < 16, "green={}", pixel[1]);
        assert!(pixel[2] < 16, "blue={}", pixel[2]);
    }
}

#[test]
fn custom_lf_metadata_roundtrips_through_the_standard_control_plane() {
    let metadata = custom_lf_metadata();
    let code = fixed_prefix_code().unwrap();
    let hf_entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let artifact = cpu_test_artifact([332, 153, -6], &code);
    let frame = assemble_frame(
        build_frame_packet(
            fixed_artifact_data(&artifact),
            &code,
            &hf_entropy,
            VarDctFrameLayout::single(VarDctStrategy::Dct8),
            metadata,
        )
        .unwrap(),
    )
    .unwrap();
    let mut codestream = image_header(8, 8).unwrap().bytes().to_vec();
    codestream.extend_from_slice(frame.bytes());
    let inventory =
        jxl_gpu_bitstream::parse(&codestream, jxl_gpu_bitstream::ParseLimits::default())
            .unwrap()
            .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
            .unwrap();
    let plan = BoundedVarDctPacketPlan::parse(&codestream, &inventory).unwrap();
    assert_eq!(plan.lf_dequantization.multipliers, [0.0625, 0.125, 0.25]);
    assert_eq!(plan.lf_correlation.colour_factor, 256);
    assert_eq!(plan.lf_correlation.base, [-0.5, 0.5]);
    assert_eq!(plan.lf_correlation.lf_factors, [-16, 32]);
    assert_eq!(decode_rgb8(&codestream).len(), 8 * 8 * 3);
}

#[test]
fn custom_lf_metadata_rejects_non_interoperable_ranges() {
    assert!(matches!(
        VarDctLfMetadata::new(
            [f16(0), f16(0x3000), f16(0x3400)],
            84,
            [f16(0), f16(0x3c00)],
            [0, 0],
        ),
        Err(EncodeError::VarDctLfDequantization {
            channel: "X",
            value: 0.0,
        })
    ));
    assert!(matches!(
        VarDctLfMetadata::new(
            [f16(0x2800), f16(0x3400), f16(0x3800)],
            1,
            [f16(0), f16(0x3c00)],
            [0, 0],
        ),
        Err(EncodeError::VarDctColourFactor { value: 1 })
    ));
    assert!(matches!(
        VarDctLfMetadata::new(
            [f16(0x2800), f16(0x3400), f16(0x3800)],
            84,
            [f16(0xc480), f16(0x3c00)],
            [0, 0],
        ),
        Err(EncodeError::VarDctBaseCorrelation {
            channel: "X",
            value: -4.5,
        })
    ));
}

#[test]
fn abi_records_are_pod_and_word_aligned() {
    fn assert_pod<T: bytemuck::Pod>() {}
    assert_pod::<GpuPrefixEntry>();
    assert_pod::<VarDctKernelParams>();
    assert_pod::<VarDctKernelArtifact>();
    assert_pod::<ScalableVarDctKernelParams>();
    assert_pod::<ScalableVarDctArtifactHeader>();
    assert_pod::<ScalableDcFragmentDescriptor>();
    assert_eq!(std::mem::size_of::<VarDctKernelParams>(), 512);
    assert_eq!(std::mem::size_of::<VarDctKernelArtifact>(), 26_880);
    assert_eq!(std::mem::align_of::<VarDctKernelArtifact>(), 4);
    assert_eq!(std::mem::size_of::<ScalableVarDctKernelParams>(), 256);
    assert_eq!(std::mem::size_of::<ScalableVarDctArtifactHeader>(), 256);
    assert_eq!(std::mem::size_of::<ScalableDcFragmentDescriptor>(), 8);

    let mut params: ScalableVarDctKernelParams = bytemuck::Zeroable::zeroed();
    params.fragment_descriptor_offset = 0x55;
    params.fragment_descriptor_len = 0x56;
    params.lf_groups_x = 0x57;
    params.lf_groups_y = 0x58;
    let params = [params];
    let parameter_words = bytemuck::cast_slice::<ScalableVarDctKernelParams, u32>(&params);
    assert_eq!(&parameter_words[55..59], &[0x55, 0x56, 0x57, 0x58]);

    let mut header: ScalableVarDctArtifactHeader = bytemuck::Zeroable::zeroed();
    header.fragment_descriptor_offset = 0x41;
    header.fragment_descriptor_len = 0x42;
    header.lf_groups_x = 0x43;
    header.lf_groups_y = 0x44;
    header.lf_group_count = 0x45;
    let headers = [header];
    let header_words = bytemuck::cast_slice::<ScalableVarDctArtifactHeader, u32>(&headers);
    assert_eq!(&header_words[41..46], &[0x41, 0x42, 0x43, 0x44, 0x45]);
}

#[test]
fn naga_validates_vardct_shaders() {
    let module = naga::front::wgsl::parse_str(SHADER).expect("VarDCT WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("VarDCT WGSL validates");

    let module = naga::front::wgsl::parse_str(LARGE_SHADER).expect("scalable VarDCT WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("scalable VarDCT WGSL validates");
}

#[test]
fn strategy_ir_uses_exact_standard_codestream_order() {
    for (id, strategy) in VarDctStrategy::ALL.into_iter().enumerate() {
        assert_eq!(usize::from(strategy.codestream_id()), id);
    }
    assert_eq!(VarDctStrategy::Dct16x8.block_extent(), (8, 16));
    assert_eq!(VarDctStrategy::Dct8x16.block_extent(), (16, 8));
    assert_eq!(VarDctStrategy::Dct256x128.block_extent(), (128, 256));
    assert_eq!(VarDctStrategy::Dct256x128.block_grid(), (16, 32));
    assert_eq!(VarDctStrategy::Dct128x256.block_extent(), (256, 128));
    assert_eq!(VarDctStrategy::Dct128x256.block_grid(), (32, 16));
    assert_eq!(VarDctStrategy::EXECUTABLE, VarDctStrategy::ALL);
    assert!(
        VarDctStrategy::EXECUTABLE
            .into_iter()
            .all(VarDctStrategy::is_executable)
    );
    assert!(VarDctStrategy::Hornuss.is_executable());
    assert!(VarDctStrategy::Dct64x64.is_executable());
}

#[test]
fn artifact_gradient_validation_matches_wgsl_wrapping_without_panicking() {
    fn wgsl_gradient(top: i32, left: i32, top_left: i32) -> i32 {
        let wrapped = i32::from_ne_bytes(
            u32::from_ne_bytes(top.to_ne_bytes())
                .wrapping_add(u32::from_ne_bytes(left.to_ne_bytes()))
                .wrapping_sub(u32::from_ne_bytes(top_left.to_ne_bytes()))
                .to_ne_bytes(),
        );
        wrapped.clamp(top.min(left), top.max(left))
    }

    for (actual, top, left, top_left) in [
        (i32::MAX, i32::MAX, i32::MAX, i32::MIN),
        (i32::MIN, i32::MIN, i32::MIN, i32::MAX),
        (0, i32::MIN, i32::MAX, 0),
        (i32::MAX, i32::MIN, 1, i32::MAX),
        (i32::MIN, -1, i32::MAX, i32::MIN),
    ] {
        let expected = wgsl_gradient(top, left, top_left);
        assert_eq!(clamped_gradient_i32(top, left, top_left), expected);
        let residual = gradient_residual_i32(actual, top, left, top_left);
        assert_eq!(residual, actual.wrapping_sub(expected));
        assert!(std::panic::catch_unwind(|| signed_token(residual)).is_ok());
    }
}

#[test]
fn scalable_layout_is_checked_and_preserves_large_orientation() {
    let code = fixed_prefix_code().unwrap();
    let portrait = ScalableArtifactLayout::new(VarDctStrategy::Dct256x128, &code).unwrap();
    let landscape = ScalableArtifactLayout::new(VarDctStrategy::Dct128x256, &code).unwrap();
    let largest = ScalableArtifactLayout::new(VarDctStrategy::Dct256x256, &code).unwrap();
    assert_eq!(portrait.strategy_len, 16 * 32);
    assert_eq!(landscape.strategy_len, 32 * 16);
    assert_eq!(portrait.dc_len, 3 * 16 * 32);
    assert_eq!(portrait, landscape);
    assert_eq!(portrait.fragment_descriptor_offset, SCALABLE_HEADER_WORDS);
    assert_eq!(portrait.fragment_descriptor_len, 2);
    assert_eq!(portrait.strategy_offset, 2 * SCALABLE_HEADER_WORDS);
    assert_eq!(
        portrait.strategy_offset % SCALABLE_SECTION_ALIGNMENT_WORDS,
        0
    );
    assert_eq!(portrait.dc_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
    assert_eq!(portrait.token_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
    assert_eq!(portrait.extra_offset % SCALABLE_SECTION_ALIGNMENT_WORDS, 0);
    assert_eq!(
        portrait.fragment_offset % SCALABLE_SECTION_ALIGNMENT_WORDS,
        0
    );
    assert_eq!(
        portrait.artifact_words % SCALABLE_SECTION_ALIGNMENT_WORDS,
        0
    );
    assert!(portrait.fragment_max_bits > 0);
    assert_eq!(portrait.artifact_bytes(), 25_856);
    assert_eq!(largest.strategy_len, 1_024);
    assert_eq!(largest.dc_len, 3_072);
    assert_eq!(largest.fragment_max_bits, 76_800);
    assert_eq!(largest.artifact_bytes(), 51_200);
}

#[test]
fn gpu_profile_encodes_exact_black_from_padded_rgb() {
    let Some(context) = test_context() else {
        return;
    };
    let encoder = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
    let source = padded_rgb_source(&context, &[[0, 0, 0]; 64]);
    let plan = encoder.memory_plan(&source).unwrap();
    assert_eq!(plan.kernel_layout, VarDctKernelLayout::Bounded);
    assert_eq!(plan.source_binding_bytes, 232);
    assert_eq!(plan.parameter_storage_bytes, 512);
    assert_eq!(plan.artifact_storage_bytes, 26_880);
    assert_eq!(plan.readback_bytes, 26_880);
    assert_eq!(plan.owned_bytes_per_job, 54_272);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

    let codestream = encoder.encode(source).unwrap();
    assert_eq!(decode_rgb8(&codestream), vec![0; 8 * 8 * 3]);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn every_linear_workgroup_produces_identical_bounded_and_scalable_codestreams() {
    let Some((device, queue, info)) = test_device() else {
        return;
    };
    let default_context = WgpuContext::new(Arc::clone(&device), Arc::clone(&queue)).unwrap();

    let mut bounded_pixels = [[0u8; 3]; 64];
    for y in 0..8usize {
        for x in 0..8usize {
            bounded_pixels[y * 8 + x] = [
                (x * 29 + y * 5) as u8,
                (y * 31 + x * 3) as u8,
                ((x + y) * 17) as u8,
            ];
        }
    }
    let default_bounded_encoder =
        VarDctEncoder::new(default_context.clone(), VarDctStrategy::Dct8).unwrap();
    let default_bounded_source = padded_rgb_source(&default_context, &bounded_pixels);
    assert_eq!(
        default_bounded_encoder
            .memory_plan(&default_bounded_source)
            .unwrap()
            .kernel_layout,
        VarDctKernelLayout::Bounded,
    );
    let default_bounded = default_bounded_encoder
        .encode(default_bounded_source)
        .unwrap();

    let scalable_width = 32usize;
    let scalable_height = 64usize;
    let scalable_pixels = (0..scalable_height)
        .flat_map(|y| {
            (0..scalable_width).map(move |x| {
                [
                    (x * 255 / (scalable_width - 1)) as u8,
                    (y * 255 / (scalable_height - 1)) as u8,
                    ((x * 11 + y * 7) & 0xff) as u8,
                ]
            })
        })
        .collect::<Vec<_>>();
    let default_scalable_encoder =
        VarDctEncoder::new(default_context.clone(), VarDctStrategy::Dct64x32).unwrap();
    let default_scalable_source = padded_rgb_source_sized(
        &default_context,
        scalable_width,
        scalable_height,
        &scalable_pixels,
    );
    assert_eq!(
        default_scalable_encoder
            .memory_plan(&default_scalable_source)
            .unwrap()
            .kernel_layout,
        VarDctKernelLayout::Scalable,
    );
    let default_scalable = default_scalable_encoder
        .encode(default_scalable_source)
        .unwrap();

    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context = test_context_with_variants(
            &device,
            &queue,
            &info,
            &[
                (BOUNDED_KERNEL_KEY, variant),
                (SCALABLE_QUANTIZE_KERNEL_KEY, variant),
            ],
        )
        .unwrap();

        let bounded = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
        assert_eq!(bounded.workgroup_variant(), variant);
        assert_eq!(
            bounded
                .encode(padded_rgb_source(&context, &bounded_pixels))
                .unwrap(),
            default_bounded,
            "bounded variant={variant:?}",
        );

        let scalable = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct64x32).unwrap();
        assert_eq!(scalable.workgroup_variant(), variant);
        assert_eq!(
            scalable
                .encode(padded_rgb_source_sized(
                    &context,
                    scalable_width,
                    scalable_height,
                    &scalable_pixels,
                ))
                .unwrap(),
            default_scalable,
            "scalable variant={variant:?}",
        );
    }

    let incompatible = test_context_with_variants(
        &device,
        &queue,
        &info,
        &[(BOUNDED_KERNEL_KEY, KernelVariant::Tile8x8)],
    )
    .unwrap();
    assert!(matches!(
        VarDctEncoder::new(incompatible, VarDctStrategy::Dct8),
        Err(EncodeError::KernelPolicy(jxl_wgpu::Error::Unsupported(_)))
    ));
}

#[test]
fn tiled_dct8_emits_multiple_ac_groups_for_odd_black_extent() {
    let Some(context) = test_context() else {
        return;
    };
    let width = 257usize;
    let height = 17usize;
    let pixels = vec![[0u8; 3]; width * height];
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let source = padded_rgb_source_sized(&context, width, height, &pixels);
    let async_source = source.clone();
    let plan = encoder.memory_plan(&source).unwrap();
    let grid = encoder.grid(&source).unwrap();
    assert_eq!(plan.kernel_layout, VarDctKernelLayout::TiledDct8);
    assert_eq!(plan.parameter_storage_bytes, 256);
    assert_eq!((grid.block_columns, grid.block_rows), (33, 3));
    assert_eq!(grid.block_count().unwrap(), 99);
    assert_eq!((grid.ac_group_columns, grid.ac_group_rows), (2, 1));
    assert_eq!(grid.ac_group_count().unwrap(), 2);
    assert_eq!(grid.toc_entries().unwrap(), 5);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);

    let codestream = encoder.encode(source).unwrap();
    let inventory =
        jxl_gpu_bitstream::parse(&codestream, jxl_gpu_bitstream::ParseLimits::default())
            .unwrap()
            .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
            .unwrap();
    let frame = &inventory.frames[0];
    assert_eq!(frame.group_count, 2);
    assert_eq!(frame.low_frequency_group_count, 1);
    assert_eq!(frame.sections.len(), 5);
    assert_eq!(
        frame
            .sections
            .iter()
            .map(|section| section.kind)
            .collect::<Vec<_>>(),
        vec![
            jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGlobal,
            jxl_gpu_bitstream::FrameSectionKind::LowFrequencyGroup { group_index: 0 },
            jxl_gpu_bitstream::FrameSectionKind::HighFrequencyGlobal,
            jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index: 0,
            },
            jxl_gpu_bitstream::FrameSectionKind::PassGroup {
                pass_index: 0,
                group_index: 1,
            },
        ]
    );
    assert_eq!(
        decode_rgb8_sized(&codestream, width, height),
        vec![0; width * height * 3]
    );
    let async_codestream = pollster::block_on(encoder.submit(async_source).unwrap()).unwrap();
    assert_eq!(async_codestream, codestream);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn tiled_dct8_preserves_asymmetric_solid_and_lf_gradient() {
    let Some(context) = test_context() else {
        return;
    };
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    for (width, height, solid) in [
        (513usize, 259usize, [255u8, 0, 0]),
        (768usize, 513usize, [0u8, 255, 0]),
    ] {
        let solid_pixels = vec![solid; width * height];
        let solid_stream = encoder
            .encode(padded_rgb_source_sized(
                &context,
                width,
                height,
                &solid_pixels,
            ))
            .unwrap();
        let decoded_solid = decode_rgb8_sized(&solid_stream, width, height);
        let solid_quality = psnr(&solid_pixels, &decoded_solid);
        assert!(
            solid_quality > 30.0,
            "{width}x{height} solid PSNR={solid_quality}",
        );

        let gradient = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    [
                        (x * 255 / (width - 1)) as u8,
                        (y * 255 / (height - 1)) as u8,
                        ((x + y) * 255 / (width + height - 2)) as u8,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let gradient_stream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &gradient))
            .unwrap();
        let decoded_gradient = decode_rgb8_sized(&gradient_stream, width, height);
        let gradient_quality = psnr(&gradient, &decoded_gradient);
        assert!(
            gradient_quality > 9.0,
            "{width}x{height} LF gradient PSNR={gradient_quality}",
        );
    }
}

#[test]
fn tiled_dct8_grid_covers_multiple_lf_groups_and_checked_16k_axes() {
    let horizontal = TiledVarDctGrid::new(2_049, 1).unwrap();
    assert_eq!(
        (horizontal.lf_group_columns, horizontal.lf_group_rows),
        (2, 1)
    );
    assert_eq!(horizontal.lf_group_count().unwrap(), 2);
    assert_eq!(horizontal.toc_entries().unwrap(), 13);

    let square = TiledVarDctGrid::new(16_384, 16_384).unwrap();
    assert_eq!((square.lf_group_columns, square.lf_group_rows), (8, 8));
    assert_eq!(square.lf_group_count().unwrap(), 64);
    assert_eq!(square.ac_group_count().unwrap(), 4_096);
    assert_eq!(square.toc_entries().unwrap(), 4_162);

    assert!(matches!(
        TiledVarDctGrid::new(16_385, 1),
        Err(EncodeError::Unsupported(
            UnsupportedFeature::TiledVarDctDimensions {
                width: 16_385,
                height: 1,
                max_dimension: 16_384,
            }
        ))
    ));
}

#[test]
fn tiled_dct8_gpu_encodes_checked_16k_panorama_axes() {
    let Some(context) = test_context() else {
        return;
    };
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    for (width, height) in [(16_384usize, 1usize), (1usize, 16_384usize)] {
        let pixels = vec![[0u8; 3]; width * height];
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let grid = encoder.grid(&source).unwrap();
        assert_eq!(grid.lf_group_count().unwrap(), 8);
        assert_eq!(grid.ac_group_count().unwrap(), 64);
        assert_eq!(grid.toc_entries().unwrap(), 74);

        let codestream = encoder.encode(source).unwrap();
        assert_eq!(
            decode_rgb8_sized(&codestream, width, height),
            vec![0; width * height * 3],
            "{width}x{height}",
        );
    }
}

#[test]
fn tiled_dct8_reports_fused_single_group_ambiguity_as_a_typed_error() {
    assert!(matches!(
        TiledVarDctGrid::new(17, 9),
        Err(EncodeError::Unsupported(
            UnsupportedFeature::TiledVarDctSingleAcGroup {
                width: 17,
                height: 9,
                group_dimension: 256,
            }
        ))
    ));
}

#[test]
fn abandoned_tiled_job_holds_and_releases_its_exact_budget() {
    let Some(base_context) = test_context() else {
        return;
    };
    let width = 513usize;
    let height = 259usize;
    let pixels = vec![[0u8; 3]; width * height];
    let provisional = TiledVarDctEncoder::new(base_context.clone()).unwrap();
    let provisional_source = padded_rgb_source_sized(&base_context, width, height, &pixels);
    let plan = provisional.memory_plan(&provisional_source).unwrap();
    assert_eq!(plan.kernel_layout, VarDctKernelLayout::TiledDct8);
    assert_eq!(
        plan.owned_bytes_per_job,
        256 + 2 * plan.artifact_storage_bytes
    );

    let limited_context = WgpuContext::with_memory_budget(
        Arc::new(base_context.device().clone()),
        Arc::new(base_context.queue().clone()),
        NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
    )
    .unwrap();
    let encoder = TiledVarDctEncoder::new(limited_context.clone()).unwrap();
    let source = padded_rgb_source_sized(&limited_context, width, height, &pixels);
    let abandoned = encoder.submit(source.clone()).unwrap();
    assert_eq!(
        encoder.in_flight_memory_stats().reserved_bytes,
        plan.owned_bytes_per_job
    );
    assert!(matches!(
        encoder.submit(source),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    drop(abandoned);

    let fence_commands =
        limited_context
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("abandoned tiled VarDCT completion fence"),
            });
    let fence = limited_context.queue().submit([fence_commands.finish()]);
    limited_context
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("abandoned tiled VarDCT work completes");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while encoder.in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        limited_context
            .device()
            .poll(wgpu::PollType::Poll)
            .expect("drive abandoned tiled VarDCT map callback");
        std::thread::yield_now();
    }
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn abandoned_scalable_job_retains_and_releases_its_exact_budget() {
    let Some(base_context) = test_context() else {
        return;
    };
    let strategy = VarDctStrategy::Dct256x256;
    let pixels = vec![[0u8; 3]; 256 * 256];
    let provisional = VarDctEncoder::new(base_context.clone(), strategy).unwrap();
    let provisional_source = padded_rgb_source_sized(&base_context, 256, 256, &pixels);
    let plan = provisional.memory_plan(&provisional_source).unwrap();
    let limited_context = WgpuContext::with_memory_budget(
        Arc::new(base_context.device().clone()),
        Arc::new(base_context.queue().clone()),
        NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
    )
    .unwrap();
    let encoder = VarDctEncoder::new(limited_context.clone(), strategy).unwrap();
    let source = padded_rgb_source_sized(&limited_context, 256, 256, &pixels);

    let abandoned = encoder.submit(source.clone()).unwrap();
    assert_eq!(
        encoder.in_flight_memory_stats().reserved_bytes,
        plan.owned_bytes_per_job
    );
    assert!(matches!(
        encoder.submit(source),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    drop(abandoned);

    let fence_commands =
        limited_context
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("abandoned scalable VarDCT completion fence"),
            });
    let fence = limited_context.queue().submit([fence_commands.finish()]);
    limited_context
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(fence),
            timeout: None,
        })
        .expect("abandoned scalable VarDCT work completes");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while encoder.in_flight_memory_stats().reserved_bytes != 0
        && std::time::Instant::now() < deadline
    {
        limited_context
            .device()
            .poll(wgpu::PollType::Poll)
            .expect("drive abandoned scalable VarDCT map callback");
        std::thread::yield_now();
    }
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn every_executable_strategy_emits_a_standard_black_codestream() {
    let Some(context) = test_context() else {
        return;
    };
    for strategy in VarDctStrategy::EXECUTABLE {
        let (width, height) = strategy.block_extent();
        let width = usize::from(width);
        let height = usize::from(height);
        let pixels = vec![[0, 0, 0]; width * height];
        let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let plan = encoder.memory_plan(&source).unwrap();
        if strategy.uses_scalable_kernel() {
            let layout =
                ScalableArtifactLayout::new(strategy, &fixed_prefix_code().unwrap()).unwrap();
            assert_eq!(plan.kernel_layout, VarDctKernelLayout::Scalable);
            assert_eq!(plan.parameter_storage_bytes, 256);
            assert_eq!(plan.artifact_storage_bytes, layout.artifact_bytes());
            assert_eq!(plan.readback_bytes, layout.artifact_bytes());
            assert_eq!(plan.owned_bytes_per_job, 256 + 2 * layout.artifact_bytes());
        } else {
            assert_eq!(plan.kernel_layout, VarDctKernelLayout::Bounded);
        }
        let codestream = encoder.encode(source).unwrap();
        assert_eq!(
            decode_rgb8_sized(&codestream, width, height),
            vec![0; width * height * 3],
            "strategy={strategy:?}",
        );
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn every_executable_strategy_preserves_solid_color_and_lf_gradient() {
    let Some(context) = test_context() else {
        return;
    };
    for strategy in VarDctStrategy::EXECUTABLE {
        let (width, height) = strategy.block_extent();
        let width = usize::from(width);
        let height = usize::from(height);
        let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();

        let red = vec![[255, 0, 0]; width * height];
        let red_stream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &red))
            .unwrap();
        let decoded_red = decode_rgb8_sized(&red_stream, width, height);
        let red_quality = psnr(&red, &decoded_red);
        assert!(
            red_quality > 30.0,
            "strategy={strategy:?}, PSNR={red_quality}"
        );

        let mut gradient = vec![[0u8; 3]; width * height];
        for y in 0..height {
            for x in 0..width {
                gradient[y * width + x] = [
                    (x * 255 / (width - 1)) as u8,
                    (y * 255 / (height - 1)) as u8,
                    ((x + y) * 255 / (width + height - 2)) as u8,
                ];
            }
        }
        let gradient_stream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &gradient))
            .unwrap();
        let decoded_gradient = decode_rgb8_sized(&gradient_stream, width, height);
        let gradient_quality = psnr(&gradient, &decoded_gradient);
        assert!(
            gradient_quality > 9.0,
            "strategy={strategy:?}, PSNR={gradient_quality}",
        );
    }
}

#[test]
fn gpu_profile_is_same_device_deterministic_and_bounded_quality() {
    let Some(context) = test_context() else {
        return;
    };
    let encoder = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
    let mut fixture = [[0u8; 3]; 64];
    for y in 0..8usize {
        for x in 0..8usize {
            fixture[y * 8 + x] = [
                (x * 31 + y * 3) as u8,
                (y * 31 + x * 3) as u8,
                ((x + y) * 16) as u8,
            ];
        }
    }
    let first_source = padded_rgb_source(&context, &fixture);
    let second_source = first_source.clone();
    let first = encoder.encode(first_source).unwrap();
    let second = pollster::block_on(encoder.submit(second_source).unwrap()).unwrap();
    assert_eq!(first, second);
    let decoded = decode_rgb8(&first);
    let quality = psnr(&fixture, &decoded);
    assert!(quality > 9.0, "PSNR={quality}");

    let red = [[255, 0, 0]; 64];
    let red_stream = encoder.encode(padded_rgb_source(&context, &red)).unwrap();
    let decoded_red = decode_rgb8(&red_stream);
    let red_quality = psnr(&red, &decoded_red);
    assert!(red_quality > 30.0, "solid-red PSNR={red_quality}");
    for pixel in decoded_red.chunks_exact(3) {
        assert!(pixel[0] > 248);
        assert!(pixel[1] < 8);
        assert!(pixel[2] < 8);
    }
}

#[test]
fn custom_lf_metadata_gpu_encoder_and_decoders_agree() {
    const DJXL: &str = "/opt/homebrew/bin/djxl";
    let Some((device, queue, info)) = test_device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let metadata = custom_lf_metadata();
    let encoder =
        VarDctEncoder::new_with_lf_metadata(context.clone(), VarDctStrategy::Dct8, metadata)
            .unwrap();
    assert_eq!(encoder.lf_metadata(), metadata);
    let fixture = std::array::from_fn::<_, 64, _>(|index| {
        let x = index % 8;
        let y = index / 8;
        [
            (x * 31 + y * 3) as u8,
            (y * 31 + x * 3) as u8,
            ((x + y) * 16) as u8,
        ]
    });
    let source = padded_rgb_source(&context, &fixture);
    let codestream = encoder.encode(source.clone()).unwrap();
    let async_codestream = pollster::block_on(encoder.submit(source).unwrap()).unwrap();
    assert_eq!(codestream, async_codestream);

    let rust_pixels = decode_rgb8(&codestream);
    let quality = psnr(&fixture, &rust_pixels);
    assert!(quality > 9.0, "custom LF metadata PSNR={quality}");
    let gpu_decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut session = gpu_decoder
        .open(
            &codestream,
            GpuOutputRequest::color(vardct_output_format(VarDctOutputFormat::Rgb8)).unwrap(),
        )
        .unwrap();
    let frame = session.next_frame().unwrap().unwrap();
    let readback = ImageReadbackPipeline::new(&backend)
        .submit(frame.output())
        .unwrap()
        .wait()
        .unwrap();
    assert!(max_abs_error(&readback.frame.outputs[0].bytes, &rust_pixels) <= 1);
    drop(frame);
    assert!(session.next_frame().unwrap().is_none());

    if Path::new(DJXL).is_file() {
        let directory = oracle_directory();
        fs::create_dir_all(&directory).unwrap();
        let codestream_path = directory.join("custom-lf.jxl");
        let ppm_path = directory.join("custom-lf.ppm");
        fs::write(&codestream_path, &codestream).unwrap();
        let status = Command::new(DJXL)
            .arg(&codestream_path)
            .arg(&ppm_path)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap();
        assert!(status.success());
        let libjxl_pixels = read_ppm_rgb8(&ppm_path, 8, 8);
        assert!(max_abs_error(&libjxl_pixels, &rust_pixels) <= 1);
        fs::remove_dir_all(directory).unwrap();
    }

    let tiled = TiledVarDctEncoder::new_with_lf_metadata(context.clone(), metadata).unwrap();
    let tiled_fixture = vec![[255, 0, 0]; 257];
    let tiled_stream = tiled
        .encode(padded_rgb_source_sized(&context, 257, 1, &tiled_fixture))
        .unwrap();
    let tiled_pixels = decode_rgb8_sized(&tiled_stream, 257, 1);
    let tiled_quality = psnr(&tiled_fixture, &tiled_pixels);
    assert!(
        tiled_quality > 30.0,
        "custom LF tiled solid PSNR={tiled_quality}",
    );
}

#[test]
fn libjxl_cli_and_rust_oracles_agree_on_gpu_codestream() {
    const CJXL: &str = "/opt/homebrew/bin/cjxl";
    const DJXL: &str = "/opt/homebrew/bin/djxl";
    if !Path::new(CJXL).is_file() || !Path::new(DJXL).is_file() {
        return;
    }
    let Some(context) = test_context() else {
        return;
    };
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    for strategy in VarDctStrategy::EXECUTABLE {
        let (width, height) = strategy.block_extent();
        let width = usize::from(width);
        let height = usize::from(height);
        let mut fixture = vec![[0u8; 3]; width * height];
        for y in 0..height {
            for x in 0..width {
                fixture[y * width + x] = [
                    (x * 255 / (width - 1)) as u8,
                    (y * 255 / (height - 1)) as u8,
                    ((x + y) * 255 / (width + height - 2)) as u8,
                ];
            }
        }
        let encoder = VarDctEncoder::new(context.clone(), strategy).unwrap();
        let codestream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &fixture))
            .unwrap();
        let rust_pixels = decode_rgb8_sized(&codestream, width, height);

        let stem = strategy.codestream_id().to_string();
        let gpu_path = directory.join(format!("gpu-{stem}.jxl"));
        let gpu_ppm_path = directory.join(format!("gpu-{stem}.ppm"));
        let source_path = directory.join(format!("source-{stem}.ppm"));
        let reference_path = directory.join(format!("reference-{stem}.jxl"));
        let reference_ppm_path = directory.join(format!("reference-{stem}.ppm"));
        fs::write(&gpu_path, &codestream).unwrap();
        fs::write(&source_path, ppm_bytes(&fixture, width, height)).unwrap();

        let gpu_decode = Command::new(DJXL)
            .arg(&gpu_path)
            .arg(&gpu_ppm_path)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap();
        assert!(gpu_decode.success(), "strategy={strategy:?}");
        let libjxl_pixels = read_ppm_rgb8(&gpu_ppm_path, width, height);
        assert!(
            max_abs_error(&libjxl_pixels, &rust_pixels) <= 1,
            "strategy={strategy:?}",
        );

        let reference_encode = Command::new(CJXL)
            .arg(&source_path)
            .arg(&reference_path)
            .args([
                "-d",
                "25",
                "-e",
                "1",
                "-m",
                "0",
                "--progressive_dc=0",
                "--resampling=1",
                "--epf=0",
                "--gaborish=0",
                "--container=0",
                "--num_threads=0",
                "--quiet",
            ])
            .status()
            .unwrap();
        assert!(reference_encode.success(), "strategy={strategy:?}");
        let reference_codestream = fs::read(&reference_path).unwrap();
        let rust_reference = decode_rgb8_sized(&reference_codestream, width, height);
        let reference_decode = Command::new(DJXL)
            .arg(&reference_path)
            .arg(&reference_ppm_path)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap();
        assert!(reference_decode.success(), "strategy={strategy:?}");
        let libjxl_reference = read_ppm_rgb8(&reference_ppm_path, width, height);
        assert!(
            max_abs_error(&libjxl_reference, &rust_reference) <= 1,
            "strategy={strategy:?}",
        );
        assert!(psnr(&fixture, &rust_pixels) > 9.0);
        assert!(psnr(&fixture, &rust_reference) > 9.0);
    }

    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn tiled_dct8_rust_jxl_djxl_and_cjxl_oracles_cover_group_edges() {
    const CJXL: &str = "/opt/homebrew/bin/cjxl";
    const DJXL: &str = "/opt/homebrew/bin/djxl";
    if !Path::new(CJXL).is_file() || !Path::new(DJXL).is_file() {
        return;
    }
    let Some((device, queue, info)) = test_device() else {
        return;
    };
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info,
        WgpuBackendConfig {
            enable_timestamps: false,
            ..WgpuBackendConfig::default()
        },
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let encoder = TiledVarDctEncoder::new(context.clone()).unwrap();
    let gpu_decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&backend);
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();

    for (case, width, height) in [
        ("odd-group-edge", 257usize, 17usize),
        ("larger-asymmetric", 768usize, 513usize),
        ("multi-lf-horizontal", 2_056usize, 256usize),
        ("multi-lf-vertical", 256usize, 2_056usize),
    ] {
        let fixture = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    [
                        (x * 255 / (width - 1)) as u8,
                        (y * 255 / (height - 1)) as u8,
                        ((x + y) * 255 / (width + height - 2)) as u8,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let codestream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &fixture))
            .unwrap();
        let grid = TiledVarDctGrid::new(width as u32, height as u32).unwrap();
        let inventory =
            jxl_gpu_bitstream::parse(&codestream, jxl_gpu_bitstream::ParseLimits::default())
                .unwrap()
                .codestream_inventory(jxl_gpu_bitstream::InventoryLimits::default())
                .unwrap();
        assert_eq!(
            inventory.frames[0].low_frequency_group_count,
            u64::from(grid.lf_group_count().unwrap()),
            "case={case}",
        );
        let rust_pixels = decode_rgb8_sized(&codestream, width, height);
        assert!(psnr(&fixture, &rust_pixels) > 9.0, "case={case}");

        let mut gpu_session = gpu_decoder
            .open(
                &codestream,
                GpuOutputRequest::color(vardct_output_format(VarDctOutputFormat::Rgb8)).unwrap(),
            )
            .unwrap();
        let gpu_frame = gpu_session.next_frame().unwrap().unwrap();
        assert_eq!(gpu_frame.output().outputs.len(), 1, "case={case}");
        assert_eq!(
            gpu_frame.output().outputs[0].layout.extent,
            Extent2d::new(width as u32, height as u32),
            "case={case}",
        );
        let gpu_pixels = readback.submit(gpu_frame.output()).unwrap().wait().unwrap();
        assert!(
            max_abs_error(&gpu_pixels.frame.outputs[0].bytes, &rust_pixels) <= 1,
            "case={case}",
        );
        drop(gpu_frame);
        assert!(gpu_session.next_frame().unwrap().is_none(), "case={case}");

        let source_path = directory.join(format!("source-{case}.ppm"));
        let gpu_path = directory.join(format!("gpu-{case}.jxl"));
        let gpu_ppm_path = directory.join(format!("gpu-{case}.ppm"));
        let reference_path = directory.join(format!("reference-{case}.jxl"));
        let reference_ppm_path = directory.join(format!("reference-{case}.ppm"));
        fs::write(&source_path, ppm_bytes(&fixture, width, height)).unwrap();
        fs::write(&gpu_path, &codestream).unwrap();

        let gpu_decode = Command::new(DJXL)
            .arg(&gpu_path)
            .arg(&gpu_ppm_path)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap();
        assert!(gpu_decode.success(), "case={case}");
        let libjxl_pixels = read_ppm_rgb8(&gpu_ppm_path, width, height);
        assert!(
            max_abs_error(&libjxl_pixels, &rust_pixels) <= 1,
            "case={case}",
        );

        let reference_encode = Command::new(CJXL)
            .arg(&source_path)
            .arg(&reference_path)
            .args([
                "-d",
                "25",
                "-e",
                "1",
                "-m",
                "0",
                "--progressive_dc=0",
                "--resampling=1",
                "--epf=0",
                "--gaborish=0",
                "--container=0",
                "--num_threads=0",
                "--quiet",
            ])
            .status()
            .unwrap();
        assert!(reference_encode.success(), "case={case}");
        let reference_codestream = fs::read(&reference_path).unwrap();
        let rust_reference = decode_rgb8_sized(&reference_codestream, width, height);
        let reference_decode = Command::new(DJXL)
            .arg(&reference_path)
            .arg(&reference_ppm_path)
            .args(["--num_threads=0", "--quiet"])
            .status()
            .unwrap();
        assert!(reference_decode.success(), "case={case}");
        let libjxl_reference = read_ppm_rgb8(&reference_ppm_path, width, height);
        assert!(
            max_abs_error(&libjxl_reference, &rust_reference) <= 1,
            "case={case}",
        );
        assert!(psnr(&fixture, &rust_reference) > 9.0, "case={case}");
    }

    fs::remove_dir_all(directory).unwrap();
}
