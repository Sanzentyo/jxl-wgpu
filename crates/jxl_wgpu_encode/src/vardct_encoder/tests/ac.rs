use std::collections::BTreeMap;

use jxl_gpu_bitstream::{BitReader, BitWriter, FrameSectionKind};

use super::super::ac::validate_blocks;
use super::super::entropy::{HfEntropyPlan, fixed_prefix_code};
use super::super::types::{ScalableArtifactLayout, VarDctFrameLayout};
use super::{
    Arc, Command, EncodeError, Extent2d, GpuDecoder, GpuOutputRequest, ImageReadbackPipeline,
    KernelVariant, NonZeroU64, Path, SCALABLE_QUANTIZE_KERNEL_KEY, TiledVarDctEncoder,
    VarDctEncoder, VarDctLfMetadata, VarDctStrategy, WgpuBackend, WgpuBackendConfig, WgpuContext,
    custom_lf_metadata, decode_rgb8, decode_rgb8_sized, fs, max_abs_error, oracle_directory,
    padded_rgb_source, padded_rgb_source_sized, read_ppm_rgb8, reference, test_context,
    test_context_with_variants, test_device, vardct_rgb8_format,
};

pub(super) fn write_tokens(
    values: impl IntoIterator<Item = u32>,
    entropy: &HfEntropyPlan,
) -> (Vec<u32>, u32) {
    let mut writer = BitWriter::new();
    for value in values {
        let extra = if value == 0 {
            0
        } else {
            31 - value.leading_zeros()
        };
        let token = if value == 0 { 0 } else { extra + 1 };
        entropy
            .code
            .write_raw(&mut writer, token, extra, value.saturating_sub(1 << extra))
            .unwrap();
    }
    let bits = writer.bit_len() as u32;
    let mut bytes = writer.into_bytes();
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    (
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect(),
        bits,
    )
}

#[test]
fn ac_fragment_capacity_and_corruption_are_checked() {
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let layout = ScalableArtifactLayout::for_tiled_grid(
        VarDctFrameLayout::tiled_dct8(1, 1).unwrap(),
        &fixed_prefix_code().unwrap(),
        &entropy,
    )
    .unwrap();
    let stride = layout.ac_words_per_block as usize;
    assert_eq!(stride, 125);
    let check = |values: &[u32]| {
        let (mut words, bits) = write_tokens(values.iter().copied(), &entropy);
        words.resize(stride, 0);
        validate_blocks(&words, &[bits], stride as u32, &entropy)
    };
    assert!(check(&[0, 0, 0]).is_ok());
    assert!(check(&[64, 0, 0]).is_err());
    assert!(check(&[1, 262_143, 0, 0]).is_err());
    assert!(check(&[1, 1, 0, 0, 0]).is_err()); // trailing token
    assert!(check(&[1, 0, 0]).is_err()); // count without its nonzero coefficient
    assert!(check(&[63]).is_err());

    // Three maximal-magnitude dense channels exercise the last allocated word.
    let values = (0..3).flat_map(|_| std::iter::once(63).chain(std::iter::repeat_n(262_142, 63)));
    let (mut words, bits) = write_tokens(values, &entropy);
    assert_eq!(words.len(), stride);
    eprintln!("maximal DCT8 fragment: {bits} bits in {stride} words");
    words.resize(stride, 0);
    assert!(bits <= stride as u32 * 32);
    validate_blocks(&words, &[bits], stride as u32, &entropy).unwrap();
    for invalid in [0, bits - 1, bits + 1, stride as u32 * 32 + 1] {
        assert!(validate_blocks(&words, &[invalid], stride as u32, &entropy).is_err());
    }
    let mut padded = words.clone();
    padded[bits as usize / 32] |= 1 << (bits % 32);
    assert!(validate_blocks(&padded, &[bits], stride as u32, &entropy).is_err());
    assert!(validate_blocks(&words, &[], stride as u32, &entropy).is_err());
    assert!(validate_blocks(&words[..stride - 1], &[bits], stride as u32, &entropy).is_err());
    assert!(validate_blocks(&words, &[bits], 0, &entropy).is_err());
}

fn read_unsigned(reader: &mut BitReader<'_>, entropy: &HfEntropyPlan) -> u32 {
    let entries = entropy.gpu_entries();
    let mut bits = 0u32;
    for length in 1..=15 {
        bits |= (reader.read_bits(1).unwrap() as u32) << (length - 1);
        if let Some(symbol) = entries
            .iter()
            .position(|entry| entry.bit_len == length && entry.bits == bits)
        {
            return if symbol == 0 {
                0
            } else {
                (1 << (symbol - 1)) + reader.read_bits((symbol - 1) as u8).unwrap() as u32
            };
        }
    }
    panic!("invalid test coefficient prefix");
}

fn check_coefficients(
    stream: &[u8],
    pixels: &[[u8; 3]],
    width: usize,
    height: usize,
    metadata: VarDctLfMetadata,
) {
    let inventory = jxl_gpu_bitstream::parse(stream, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let entropy = HfEntropyPlan::single_cluster_prefix().unwrap();
    let order = reference::natural_order();
    let mut reference_blocks = BTreeMap::new();
    let columns = width.div_ceil(8);
    let rows = height.div_ceil(8);
    let group_columns = width.div_ceil(256);
    let mut seen = vec![false; columns * rows];
    let mut nonzero = 0;
    for section in &inventory.frames[0].sections {
        let FrameSectionKind::PassGroup {
            group_index,
            pass_index: 0,
        } = section.kind
        else {
            continue;
        };
        let x0 = group_index as usize % group_columns * 32;
        let y0 = group_index as usize / group_columns * 32;
        let start = section.bytes.offset as usize;
        let end = start + section.bytes.length as usize;
        let mut reader = BitReader::new(&stream[start..end]);
        for by in y0..(y0 + 32).min(rows) {
            for bx in x0..(x0 + 32).min(columns) {
                assert!(!std::mem::replace(&mut seen[by * columns + bx], true));
                let block = reference::block(pixels, width, height, bx, by);
                let expected = reference_blocks
                    .entry(block)
                    .or_insert_with(|| reference::quantized_ac(&block, metadata));
                for channel in [1, 0, 2] {
                    let mut remaining = read_unsigned(&mut reader, &entropy);
                    assert!(remaining <= 63);
                    for &offset in &order[1..] {
                        let actual = if remaining == 0 {
                            0
                        } else {
                            let packed = read_unsigned(&mut reader, &entropy);
                            remaining -= u32::from(packed != 0);
                            nonzero += usize::from(packed != 0);
                            if packed.is_multiple_of(2) {
                                (packed / 2) as i32
                            } else {
                                -((packed / 2) as i32) - 1
                            }
                        };
                        assert!(
                            (actual - expected[channel][offset]).abs() <= 1,
                            "{width}x{height} block={bx},{by} channel={channel} wire={offset}: GPU={actual}, f64={}",
                            expected[channel][offset]
                        );
                    }
                    assert_eq!(remaining, 0);
                }
            }
        }
        assert!(reader.remaining_bits() < 8);
        assert_eq!(reader.read_bits(reader.remaining_bits() as u8).unwrap(), 0);
    }
    assert!(seen.into_iter().all(|block| block));
    assert!(nonzero > 0);
}

#[test]
fn tiled_ac_matches_f64_across_group_boundaries_and_custom_correlation() {
    let context = test_context().expect("actual GPU required for tiled AC conformance evidence");
    for (width, height, custom) in [
        (257, 1, false),
        (1, 257, false),
        (255, 257, false),
        (257, 255, true),
        (263, 265, false),
        (2057, 17, true),
        (17, 2057, false),
        (2057, 2057, false),
    ] {
        let metadata = if custom {
            custom_lf_metadata()
        } else {
            VarDctLfMetadata::default()
        };
        let pixels = reference::pattern(width, height);
        let encoder = TiledVarDctEncoder::new_with_lf_metadata(context.clone(), metadata).unwrap();
        let stream = encoder
            .encode(padded_rgb_source_sized(&context, width, height, &pixels))
            .unwrap();
        check_coefficients(&stream, &pixels, width, height, metadata);
    }
}

#[test]
fn fused_tiled_ac_and_all_workgroup_sizes_interoperate() {
    let (device, queue, info) =
        test_device().expect("actual GPU required for tiled AC interoperability");
    let backend = WgpuBackend::from_device(
        device.as_ref().clone(),
        queue.as_ref().clone(),
        info.clone(),
        WgpuBackendConfig {
            enable_timestamps: false,
            ..Default::default()
        },
    )
    .unwrap();
    let context = WgpuContext::from_backend(&backend);
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let readback = ImageReadbackPipeline::new(&backend);
    let directory = oracle_directory();
    fs::create_dir_all(&directory).unwrap();
    let djxl = "/opt/homebrew/bin/djxl";
    let mut streams = Vec::new();
    eprintln!(
        "tiled AC adapter: {info:?}; djxl available: {}",
        Path::new(djxl).is_file()
    );
    for (width, height, custom) in [
        (1, 1, false),
        (7, 5, false),
        (8, 8, false),
        (17, 9, true),
        (256, 256, false),
        (263, 265, false),
        (263, 265, true),
        (1, 257, false),
        (257, 1, true),
        (2057, 17, true),
        (17, 2057, false),
    ] {
        let pixels = reference::pattern(width, height);
        let metadata = if custom {
            custom_lf_metadata()
        } else {
            VarDctLfMetadata::default()
        };
        let encoder = TiledVarDctEncoder::new_with_lf_metadata(context.clone(), metadata).unwrap();
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let stream = encoder.encode(source.clone()).unwrap();
        assert_eq!(
            pollster::block_on(encoder.submit(source).unwrap()).unwrap(),
            stream
        );
        if width == 8 && height == 8 {
            let bounded = VarDctEncoder::new(context.clone(), VarDctStrategy::Dct8).unwrap();
            let block = reference::block(&pixels, 8, 8, 0, 0);
            let reference = bounded.encode(padded_rgb_source(&context, &block)).unwrap();
            assert_eq!(stream, reference);
            let decoded = decode_rgb8(&stream);
            assert_ne!(
                &decoded[0..3],
                &decoded[3..6],
                "checkerboard AC must survive quantization"
            );
        }
        let rust_pixels = decode_rgb8_sized(&stream, width, height);
        let mut session = decoder
            .open(
                &stream,
                GpuOutputRequest::color(vardct_rgb8_format()).unwrap(),
            )
            .unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        assert_eq!(
            frame.output().outputs[0].layout.extent,
            Extent2d::new(width as u32, height as u32)
        );
        let output = readback.submit(frame.output()).unwrap().wait().unwrap();
        let error = max_abs_error(&output.frame.outputs[0].bytes, &rust_pixels);
        assert!(
            error <= 1,
            "{width}x{height}: GPU/Rust RGB8 max error {error}"
        );
        drop(frame);
        assert!(session.next_frame().unwrap().is_none());
        if Path::new(djxl).is_file() {
            let input = directory.join(format!("{width}x{height}.jxl"));
            let output = input.with_extension("ppm");
            fs::write(&input, &stream).unwrap();
            assert!(
                Command::new(djxl)
                    .arg(input)
                    .arg(&output)
                    .args(["--num_threads=0", "--quiet"])
                    .status()
                    .unwrap()
                    .success()
            );
            let native = read_ppm_rgb8(&output, width, height);
            let error = max_abs_error(&native, &rust_pixels);
            assert!(
                error <= 1,
                "{width}x{height}: libjxl/Rust RGB8 max error {error}"
            );
        }
        if width <= 263 && height <= 265 {
            streams.push((width, height, metadata, pixels, stream));
        }
    }
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
            &[(SCALABLE_QUANTIZE_KERNEL_KEY, variant)],
        )
        .unwrap();
        for (width, height, metadata, pixels, expected) in &streams {
            let encoder =
                TiledVarDctEncoder::new_with_lf_metadata(context.clone(), *metadata).unwrap();
            assert_eq!(
                &encoder
                    .encode(padded_rgb_source_sized(&context, *width, *height, pixels))
                    .unwrap(),
                expected,
                "variant={variant:?}, {width}x{height}"
            );
        }
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn nonzero_ac_admission_includes_every_fragment_and_readback_byte() {
    let base = test_context().expect("actual GPU required for tiled AC admission evidence");
    let pixels = reference::pattern(17, 9);
    let source = padded_rgb_source_sized(&base, 17, 9, &pixels);
    let provisional = TiledVarDctEncoder::new(base.clone()).unwrap();
    let plan = provisional.memory_plan(&source).unwrap();
    assert_eq!(
        plan.owned_bytes_per_job,
        512 + 2 * plan.artifact_storage_bytes
    );
    let limited = WgpuContext::with_memory_budget(
        Arc::new(base.device().clone()),
        Arc::new(base.queue().clone()),
        NonZeroU64::new(plan.owned_bytes_per_job - 1).unwrap(),
    )
    .unwrap();
    let encoder = TiledVarDctEncoder::new(limited).unwrap();
    assert!(matches!(
        encoder.submit(source.clone()),
        Err(EncodeError::MemoryBackpressure(_))
    ));
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    let exact = WgpuContext::with_memory_budget(
        Arc::new(base.device().clone()),
        Arc::new(base.queue().clone()),
        NonZeroU64::new(plan.owned_bytes_per_job).unwrap(),
    )
    .unwrap();
    let encoder = TiledVarDctEncoder::new(exact).unwrap();
    let stream = encoder.encode(source).unwrap();
    assert_eq!(decode_rgb8_sized(&stream, 17, 9).len(), 17 * 9 * 3);
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn tiled_ac_storage_is_checked_against_the_device_binding_limit() {
    const LIMIT: u64 = 1 << 20;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("actual GPU required for tiled AC device-limit evidence");
    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_storage_buffer_binding_size = LIMIT;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let context = WgpuContext::new(Arc::new(device), Arc::new(queue)).unwrap();
    let pixels = reference::pattern(513, 259);
    let source = padded_rgb_source_sized(&context, 513, 259, &pixels);
    assert!(source.buffer.size() < LIMIT);
    let encoder = TiledVarDctEncoder::new(context).unwrap();
    let layout = ScalableArtifactLayout::for_tiled_grid(
        VarDctFrameLayout::tiled_dct8(513, 259).unwrap(),
        &fixed_prefix_code().unwrap(),
        &HfEntropyPlan::single_cluster_prefix().unwrap(),
    )
    .unwrap();
    assert!(layout.artifact_bytes() > LIMIT);
    assert!(
        matches!(encoder.submit(source), Err(EncodeError::Unsupported(
        crate::UnsupportedFeature::DeviceLimit {
            name: "max_storage_buffer_binding_size", required, available: LIMIT,
        }
    )) if required == layout.artifact_bytes())
    );
    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
}
