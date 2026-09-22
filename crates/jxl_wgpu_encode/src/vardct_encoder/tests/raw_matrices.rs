//! Raw mode-7 metadata, independent entropy reconstruction and public interoperability.

use super::*;
use crate::{VarDctDequantMatrices, VarDctMatrixEncoding};

pub(super) fn samples(strategy: VarDctStrategy) -> [Vec<i32>; 3] {
    let extent = strategy.pixel_extent();
    let width = extent.width.min(extent.height) as usize;
    std::array::from_fn(|c| {
        (0..extent.area().unwrap())
            .map(|i| {
                let (x, y) = (i % width, i / width);
                (8 + (13 * x + 7 * y + x * y + 11 * c) % 53) as i32
            })
            .collect()
    })
}

pub(super) fn selected(
    strategies: impl IntoIterator<Item = VarDctStrategy>,
) -> VarDctDequantMatrices {
    strategies.into_iter().fold(
        Default::default(),
        |set: VarDctDequantMatrices, strategy| {
            set.with_raw_matrix(strategy, f16(0x0400), samples(strategy))
                .unwrap()
        },
    )
}

pub(super) fn oracle_scales(strategy: VarDctStrategy) -> [Vec<f64>; 3] {
    // Independent scalar specification: raw matrix scales are q / 16384 in canonical order.
    samples(strategy).map(|channel| {
        channel
            .into_iter()
            .map(|q| f64::from(q) / 16384.0)
            .collect()
    })
}

#[test]
fn raw_matrix_configuration_rejects_bad_shapes_values_and_preserves_family_identity() {
    for strategy in VarDctStrategy::ALL {
        for kind in 0..7 {
            let mut values = samples(strategy);
            let denominator = match kind {
                4 => f16(0),
                5 => f16(0xbc00),
                _ => f16(0x7bff),
            };
            match kind {
                0 => {
                    values[2].pop();
                }
                1 => values[0].push(1),
                2 => values[1][3] = 0,
                3 => values[2][0] = -1,
                6 => values[0][1] = i32::MAX,
                _ => {}
            }
            assert!(matches!(
                VarDctDequantMatrices::default().with_raw_matrix(strategy, denominator, values),
                Err(EncodeError::VarDctMatrix(
                    jxl_gpu_protocol::VarDctMatrixError::Value { .. }
                ))
            ));
        }
    }
    let set = selected([VarDctStrategy::Dct16x8, VarDctStrategy::Afv3]);
    let raw = set.raw_matrix(VarDctStrategy::Dct8x16).unwrap();
    assert_eq!(
        raw.extent(),
        Extent2d {
            width: 8,
            height: 16
        }
    );
    assert_eq!(raw.denominator(), f16(0x0400));
    assert_eq!(
        raw.channels(),
        samples(VarDctStrategy::Dct16x8)
            .each_ref()
            .map(Vec::as_slice)
    );
    assert_eq!(
        set.raw_matrix(VarDctStrategy::Afv0),
        set.raw_matrix(VarDctStrategy::Afv3)
    );
    assert!(set.encoding(VarDctStrategy::Dct16x8).is_none());
    let reset = set
        .with_matrix(VarDctStrategy::Dct8x16, VarDctMatrixEncoding::Default)
        .unwrap()
        .with_matrix(VarDctStrategy::Afv1, VarDctMatrixEncoding::Default)
        .unwrap();
    assert_eq!(reset, VarDctDequantMatrices::default());
    assert!(
        selected([VarDctStrategy::Dct8])
            .write(&mut BitWriter::new())
            .is_err(),
        "raw matrices cannot be serialized without GPU fragments"
    );
}

#[test]
fn raw_matrix_gpu_fragments_round_trip_all_families_and_reject_corrupted_artifacts() {
    use super::super::raw_matrices::{Pipeline, Plan};
    let context = test_context().expect("actual GPU required for raw matrix entropy");
    let matrices =
        VarDctStrategy::ALL
            .into_iter()
            .fold(VarDctDequantMatrices::default(), |set, strategy| {
                let area = strategy.pixel_extent().area().unwrap();
                let values = std::array::from_fn(|c| {
                    (0..area)
                        .map(|i| {
                            [1, i32::MAX, 2, 65_536, 127, 128, 256, 16_777_215][(i + 3 * c) % 8]
                        })
                        .collect()
                });
                set.with_raw_matrix(strategy, f16(1), values).unwrap()
            });
    let code = fixed_prefix_code().unwrap();
    let plan = Plan::new(&matrices, &code).unwrap().unwrap();
    plan.validate_limits(&context.device().limits()).unwrap();
    for (storage, buffer, groups) in [
        (plan.input_bytes() - 1, u64::MAX, u32::MAX),
        (u64::MAX, plan.artifact_bytes() - 1, u32::MAX),
        (u64::MAX, u64::MAX, 16),
    ] {
        let mut limits = context.device().limits();
        limits.max_storage_buffer_binding_size = storage;
        limits.max_buffer_size = buffer;
        limits.max_compute_workgroups_per_dimension = groups;
        assert!(matches!(
            plan.validate_limits(&limits),
            Err(EncodeError::Unsupported(
                UnsupportedFeature::DeviceLimit { .. }
            ))
        ));
    }
    let pipeline = Pipeline::new(context.device());
    let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raw matrix test readback"),
        size: plan.artifact_bytes(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut commands = context.device().create_command_encoder(&Default::default());
    let scratch = pipeline.encode(context.device(), &mut commands, &plan, &staging, 0);
    let submission = context.queue().submit([commands.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    let bytes = mapped.to_vec();
    drop(mapped);
    staging.unmap();
    drop(scratch);
    let fragments = plan.validate(&bytes).unwrap();
    for family in 0..17 {
        let mut output = BitWriter::new();
        super::super::entropy::write_prefix_config(&mut output, &code, 1).unwrap();
        fragments.append(&mut output, family).unwrap();
        let bit_len = output.bit_len();
        let encoded = output.into_bytes();
        let mut bits = jxl_bitstream::Bitstream::new(&encoded);
        let mut entropy = jxl_coding::Decoder::parse(&mut bits, 1).unwrap();
        entropy.begin(&mut bits).unwrap();
        let strategy = VarDctStrategy::ALL
            .into_iter()
            .find(|s| s.dequant_matrix_index() == family)
            .unwrap();
        let raw = matrices.raw_matrix(strategy).unwrap();
        let width = raw.extent().width as usize;
        for expected in raw.channels() {
            let mut decoded = vec![0i64; expected.len()];
            for (i, &sample) in expected.iter().enumerate() {
                let packed = entropy.read_varint(&mut bits, 0).unwrap();
                let residual = i64::from(packed >> 1) ^ -i64::from(packed & 1);
                let (x, y) = (i % width, i / width);
                let left = if x > 0 {
                    decoded[i - 1]
                } else if y > 0 {
                    decoded[i - width]
                } else {
                    0
                };
                let north = if y > 0 { decoded[i - width] } else { left };
                let northwest = if x > 0 && y > 0 {
                    decoded[i - width - 1]
                } else {
                    left
                };
                decoded[i] =
                    residual + (left + north - northwest).clamp(left.min(north), left.max(north));
                assert_eq!(decoded[i], i64::from(sample), "family {family}, index {i}");
            }
        }
        entropy.finalize().unwrap();
        assert_eq!(bits.num_read_bits(), bit_len);
    }
    for mutation in 0..7 {
        let mut corrupt = bytes.clone();
        match mutation {
            0 => corrupt[0] = 0,
            1 => corrupt[4] ^= 1,
            2 => corrupt[8] ^= 1,
            3 => corrupt[12..16].copy_from_slice(&u32::MAX.to_le_bytes()),
            4 => corrupt[17 * 16] ^= 1,
            5 => *corrupt.last_mut().unwrap() ^= 0x80,
            _ => {
                corrupt.truncate(corrupt.len() - 4);
            }
        }
        assert!(plan.validate(&corrupt).is_err(), "mutation {mutation}");
    }
}

#[test]
fn raw_tiled_matrices_interoperate_through_windows_and_all_workgroup_variants() {
    let (device, queue, info) = test_device().expect("actual GPU required for raw matrices");
    let context = WgpuContext::new(device.clone(), queue.clone()).unwrap();
    let config = VarDctConfig {
        dequant_matrices: matrices::selected()
            .with_raw_matrix(
                VarDctStrategy::Dct8,
                f16(0x0400),
                samples(VarDctStrategy::Dct8),
            )
            .unwrap(),
        coefficient_orders: orders::selected([VarDctStrategy::Dct8]),
        lf_metadata: custom_lf_metadata(),
        ..Default::default()
    };
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
    let mut cases = Vec::new();
    for (width, height) in [(1, 1), (13, 21), (257, 17), (2057, 17)] {
        let pixels = reference::pattern(width, height);
        let source = padded_rgb_source_sized(&context, width, height, &pixels);
        let bytes = encoder.encode(source.clone()).unwrap();
        assert_eq!(
            pollster::block_on(encoder.submit(source).unwrap()).unwrap(),
            bytes
        );
        quantization::assert_decoders_agree(&bytes, width, height);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        cases.push((width, height, pixels, bytes));
    }
    for variant in [
        KernelVariant::Scalar,
        KernelVariant::Lanes32,
        KernelVariant::Lanes64,
        KernelVariant::Lanes128,
        KernelVariant::Lanes256,
    ] {
        let context =
            test_context_with_variants(&device, &queue, &info, &[(TILED_KERNEL_KEY, variant)])
                .unwrap();
        let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
        for (width, height, pixels, expected) in &cases {
            assert_eq!(
                &encoder
                    .encode(padded_rgb_source_sized(&context, *width, *height, pixels))
                    .unwrap(),
                expected
            );
        }
    }
}

#[test]
fn raw_matrix_full_signed_sample_range_interoperates_without_narrowing() {
    let context = test_context().expect("actual GPU required for wide raw matrix samples");
    let pixels = reference::pattern(8, 8);
    let source = padded_rgb_source_sized(&context, 8, 8, &pixels);
    for values in [
        [1, 127, 128, 255, 256, 65_536, 16_777_215, i32::MAX],
        [131_072; 8],
    ] {
        let config = VarDctConfig {
            dequant_matrices: VarDctDequantMatrices::default()
                .with_raw_matrix(
                    VarDctStrategy::Dct8,
                    f16(1),
                    std::array::from_fn(|c| (0..64).map(|i| values[(i + c * 3) % 8]).collect()),
                )
                .unwrap(),
            ..Default::default()
        };
        let encoder =
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config).unwrap();
        let bytes = encoder.encode(source.clone()).unwrap();
        // Rust jxl 0.6.0 disagrees on these wide samples. Keep the one-code bound
        // against independent jxl-oxide and native libjxl; the corpus records the
        // discrepancy, whose cause is not established. Other cases still use jxl.
        let mut image = jxl_oxide::JxlImage::read_with_defaults(bytes.as_slice()).unwrap();
        image.request_color_encoding(jxl_oxide::EnumColourEncoding::srgb(
            jxl_oxide::RenderingIntent::Relative,
        ));
        let render = image.render_frame(0).unwrap();
        let reference: Vec<_> = render
            .image_all_channels()
            .buf()
            .iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        assert_eq!(reference.len(), 8 * 8 * 3);
        eprintln!(
            "wide raw matrix Rust/oxide difference: {}",
            max_abs_error(&decode_rgb8_sized(&bytes, 8, 8), &reference)
        );
        quantization::assert_decoders_agree_with_reference(&bytes, 8, 8, &reference);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn rectangular_raw_matrix_rasters_resume_among_parametric_families() {
    let context = test_context().expect("actual GPU required for rectangular raw matrices");
    for strategy in [
        VarDctStrategy::Dct16x8,
        VarDctStrategy::Dct8x16,
        VarDctStrategy::Dct32x8,
        VarDctStrategy::Dct8x32,
    ] {
        let extent = strategy.pixel_extent();
        let (width, height) = (extent.width as usize, extent.height as usize);
        let config = VarDctConfig {
            dequant_matrices: matrices::selected()
                .with_raw_matrix(strategy, f16(0x0400), samples(strategy))
                .unwrap(),
            coefficient_orders: orders::selected([strategy]),
            lf_metadata: custom_lf_metadata(),
            ..Default::default()
        };
        let encoder = VarDctEncoder::new_with_config(context.clone(), strategy, config).unwrap();
        let bytes = encoder
            .encode(padded_rgb_source_sized(
                &context,
                width,
                height,
                &reference::pattern(width, height),
            ))
            .unwrap();
        quantization::assert_decoders_agree(&bytes, width, height);
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
