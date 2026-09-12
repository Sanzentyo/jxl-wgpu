use super::*;
use jxl_gpu_formats::{
    ChromaLocation2d, ColorFormatClass, ColorRange, ColorSpace, ColorSpec, ColorSpecification,
    PixelFormatClass, RgbChannelOrder, RgbSample, TransferFunction, classify_pixel_format,
    convert_rgb_f32, vpi::VpiPitchLinearFormat,
};

fn source_rgb_for_target(samples: &[f32], format: &PixelFormat) -> [Vec<f32>; 3] {
    let ColorSpecification::Defined(color) = format.color_spec else {
        panic!("oracle requires explicit color");
    };
    assert_eq!(color.space, ColorSpace::Bt709);
    std::array::from_fn(|channel| {
        samples
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| {
                let value = pixel[channel];
                if matches!(
                    color.transfer,
                    TransferFunction::Srgb | TransferFunction::Sycc
                ) {
                    return value;
                }
                let magnitude = value.abs();
                let linear = if magnitude <= 0.04045 {
                    magnitude / 12.92
                } else {
                    ((magnitude + 0.055) / 1.055).powf(2.4)
                };
                let encoded = match color.transfer {
                    TransferFunction::Linear => linear,
                    TransferFunction::Bt709 => {
                        if linear < 0.018 {
                            4.5 * linear
                        } else {
                            1.099 * linear.powf(0.45) - 0.099
                        }
                    }
                    other => panic!("unhandled oracle transfer {other:?}"),
                };
                encoded.copysign(value)
            })
            .collect()
    })
}

fn assert_color_codes(
    name: &str,
    actual: &[u8],
    expected: &[u8],
    layout: &ImageLayout,
    float_tolerance: f32,
) {
    assert_eq!(actual.len(), expected.len());
    let class = classify_pixel_format(&layout.format).unwrap();
    if matches!(
        class,
        PixelFormatClass::Color(ColorFormatClass::Rgb {
            sample: RgbSample::F32,
            ..
        })
    ) {
        let ColorSpecification::Defined(color) = layout.format.color_spec else {
            unreachable!()
        };
        // Measure reconstruction error in linear light. Near black, the sRGB OETF amplifies
        // a small inverse-transform difference by up to 12.92; testing its encoded values at
        // the same threshold would impose a different reconstruction tolerance by brightness.
        // The shared output tests separately check the OETF itself in encoded coordinates.
        let to_linear = |value: f32| {
            let magnitude = value.abs();
            let linear = match color.transfer {
                TransferFunction::Linear => magnitude,
                TransferFunction::Srgb => {
                    if magnitude <= 0.04045 {
                        magnitude / 12.92
                    } else {
                        ((magnitude + 0.055) / 1.055).powf(2.4)
                    }
                }
                TransferFunction::Bt709 => {
                    if magnitude < 0.081 {
                        magnitude / 4.5
                    } else {
                        ((magnitude + 0.099) / 1.099).powf(1.0 / 0.45)
                    }
                }
                other => panic!("unsupported F32 error domain {other:?}"),
            };
            linear.copysign(value)
        };
        let mut maximum = 0.0f32;
        let mut maximum_encoded = 0.0f32;
        for (actual, expected) in actual
            .as_chunks::<4>()
            .0
            .iter()
            .zip(expected.as_chunks::<4>().0.iter())
        {
            let actual = f32::from_le_bytes(*actual);
            let expected = f32::from_le_bytes(*expected);
            assert!(actual.is_finite() && expected.is_finite());
            maximum_encoded = maximum_encoded.max((actual - expected).abs());
            maximum = maximum.max((to_linear(actual) - to_linear(expected)).abs());
        }
        eprintln!("{name}: F32 maximum linear error {maximum}, encoded error {maximum_encoded}");
        assert!(
            maximum < float_tolerance,
            "{name}: F32 maximum error {maximum}"
        );
        return;
    }
    let (bits, storage) = match class {
        PixelFormatClass::Color(
            ColorFormatClass::Luma { bits, storage_bits }
            | ColorFormatClass::YuvPlanar {
                bits, storage_bits, ..
            }
            | ColorFormatClass::YuvSemiplanar {
                bits, storage_bits, ..
            },
        ) => (bits, storage_bits),
        PixelFormatClass::Color(_) => (8, 8),
        _ => panic!("color oracle requires color storage"),
    };
    let step = usize::from(storage / 8);
    let shift = storage - bits;
    let read = |bytes: &[u8], offset: usize| {
        if step == 1 {
            u16::from(bytes[offset])
        } else {
            u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
        }
    };
    let mut maximum = 0;
    let mut covered = vec![false; actual.len()];
    for plane in &layout.planes {
        for row in 0..plane.sample_extent.height {
            let start = (plane.offset + u64::from(row) * plane.row_stride) as usize;
            for offset in (start..start + plane.row_bytes as usize).step_by(step) {
                let word = read(actual, offset);
                assert_eq!(word & ((1 << shift) - 1), 0, "{name}: unused sample bits");
                maximum = maximum.max((word >> shift).abs_diff(read(expected, offset) >> shift));
                covered[offset..offset + step].fill(true);
            }
        }
    }
    for (offset, &is_sample) in covered.iter().enumerate() {
        if !is_sample {
            assert_eq!(actual[offset], 0, "{name}: plane/row padding");
        }
    }
    let tolerance = if bits <= 12 { 1 } else { 4 };
    eprintln!("{name}: {bits}-bit maximum code error {maximum}");
    assert!(
        maximum <= tolerance,
        "{name}: {bits}-bit code error {maximum} exceeds {tolerance}"
    );
}

fn output_cases() -> Vec<(String, PixelFormat)> {
    let mut cases = VpiPitchLinearFormat::ALL
        .iter()
        .filter_map(|&vpi| {
            let format = vpi.pixel_format();
            matches!(
                classify_pixel_format(&format),
                Ok(PixelFormatClass::Color(_))
            )
            .then(|| (vpi.name().to_owned(), format))
        })
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 20);
    let color = ColorSpecification::Defined(ColorSpec {
        transfer: TransferFunction::Srgb,
        ..ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER)
    });
    cases.extend(
        [
            ("I444", PixelFormat::i444(8, 8, color).unwrap()),
            ("I422", PixelFormat::i422(8, 8, color).unwrap()),
            ("I420", PixelFormat::i420(8, 8, color).unwrap()),
            ("NV21", PixelFormat::nv21(color)),
            ("NV42", PixelFormat::nv42(color)),
            ("P010", PixelFormat::p010(color)),
            ("P012", PixelFormat::p012(color)),
            ("P016", PixelFormat::p016(color)),
            ("I420-12", PixelFormat::i420(12, 16, color).unwrap()),
        ]
        .map(|(name, format)| (name.to_owned(), format)),
    );
    let linear = ColorSpecification::Defined(ColorSpec {
        transfer: TransferFunction::Linear,
        ..ColorSpec::bt709(ColorRange::Full, ChromaLocation2d::CENTER)
    });
    cases.push((
        "linear-BGRA".to_owned(),
        PixelFormat::rgb8(RgbChannelOrder::Bgra, false, linear),
    ));
    cases.push((
        "linear-BGRA-F32".to_owned(),
        PixelFormat::rgb_f32(RgbChannelOrder::Bgra, false, linear),
    ));
    let srgb = ColorSpecification::Defined(ColorSpec {
        transfer: TransferFunction::Srgb,
        ..ColorSpec::bt709(ColorRange::Full, ChromaLocation2d::CENTER)
    });
    for order in [
        RgbChannelOrder::Rgb,
        RgbChannelOrder::Bgr,
        RgbChannelOrder::Rgba,
        RgbChannelOrder::Bgra,
    ] {
        for planar in [false, true] {
            cases.push((
                format!("{order:?}-F32-planar{planar}"),
                PixelFormat::rgb_f32(order, planar, srgb),
            ));
        }
    }
    cases
}

fn check_formats(
    backend: &WgpuBackend,
    name: &str,
    encoded: &[u8],
    extent: Extent2d,
    formats: &[(String, PixelFormat)],
) {
    let rust = rust_jxl_rgb_f32(encoded, extent);
    let djxl = djxl_rgb_f32(encoded, extent);
    let grayscale = jxl_gpu_bitstream::parse(encoded, ParseLimits::default())
        .unwrap()
        .codestream_inventory(InventoryLimits::default())
        .unwrap()
        .image_header
        .grayscale;
    for (label, format) in formats {
        let reference = |samples: &[f32]| {
            let rgb = source_rgb_for_target(samples, format);
            convert_rgb_f32([&rgb[0], &rgb[1], &rgb[2]], extent, format).unwrap()
        };
        let expected = reference(&rust);
        let float_linear = format.sample_kind == jxl_gpu_formats::SampleKind::Float
            && matches!(format.color_spec, ColorSpecification::Defined(spec) if spec.transfer == TransferFunction::Linear);
        let djxl_expected = if float_linear {
            djxl_rgb_f32_with_color(
                encoded,
                extent,
                Some(if grayscale {
                    "Gra_D65_Rel_Lin"
                } else {
                    "RGB_D65_SRG_Rel_Lin"
                }),
            )
            .map(|samples| {
                let planes: [Vec<f32>; 3] = std::array::from_fn(|channel| {
                    samples
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|pixel| pixel[channel])
                        .collect()
                });
                convert_rgb_f32([&planes[0], &planes[1], &planes[2]], extent, format).unwrap()
            })
        } else {
            djxl.as_ref().map(|samples| reference(samples))
        };
        if format.sample_kind == jxl_gpu_formats::SampleKind::Float
            && let Some(djxl) = &djxl_expected
        {
            let maximum = expected
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(djxl.bytes.as_chunks::<4>().0.iter())
                .map(|(a, b)| (f32::from_le_bytes(*a) - f32::from_le_bytes(*b)).abs())
                .fold(0.0f32, f32::max);
            eprintln!("{name}/{label}: Rust-djxl F32 reference disagreement {maximum}");
        }
        let mut whole = None;
        for cap in [u64::MAX, 256] {
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
            );
            let request = GpuOutputRequest::color(format.clone()).unwrap();
            let mut session = if cap == 256 {
                open_incremental(&decoder, encoded, request)
            } else {
                decoder.open(encoded, request).unwrap()
            };
            if let Some(vardct) = session.submission_session().vardct() {
                let memory = vardct.memory_stats().unwrap();
                assert_eq!(
                    memory.output_lease_bytes,
                    expected.layout.logical_size.div_ceil(4) * 4
                );
                assert_eq!(memory.output_uniform_bytes, 352);
            }
            let frame = if cap == 256 {
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_frame().unwrap().unwrap()
            };
            assert!(session.next_frame().unwrap().is_none());
            let readback = ImageReadbackPipeline::new(backend)
                .submit(frame.output())
                .unwrap()
                .wait()
                .unwrap();
            let output = &readback.frame.outputs[0];
            assert_eq!(output.layout, expected.layout);
            assert_color_codes(
                &format!("{name}/{label}/Rust/cap{cap}"),
                &output.bytes,
                &expected.bytes,
                &expected.layout,
                2e-5,
            );
            if let Some(expected) = &djxl_expected {
                assert_color_codes(
                    &format!("{name}/{label}/djxl/cap{cap}"),
                    &output.bytes,
                    &expected.bytes,
                    &expected.layout,
                    1e-4,
                );
            }
            if let Some(whole) = &whole {
                assert_eq!(&output.bytes, whole);
            } else {
                whole = Some(output.bytes.clone());
            }
            drop(readback);
            drop(frame);
            drop(session);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

#[test]
fn generic_color_outputs_preserve_oriented_high_depth_vardct_precision() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    eprintln!("generic VarDCT output adapter: {info:?}");
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    check_formats(
        &backend,
        "RGB16",
        &corpus::vardct_depth_combined("rgb_16_multilf"),
        Extent2d::new(17, 2056),
        &output_cases(),
    );
}

#[test]
fn generic_color_output_combines_jpeg_gray_resampling_and_recursive_dc() {
    let Some((info, device, queue)) = device() else {
        return;
    };
    let backend =
        WgpuBackend::from_device(device, queue, info, WgpuBackendConfig::default()).unwrap();
    let formats = output_cases()
        .into_iter()
        .filter(|(name, _)| {
            matches!(
                name.as_str(),
                "P016" | "I420" | "linear-BGRA" | "linear-BGRA-F32" | "Rgba-F32-planartrue"
            )
        })
        .collect::<Vec<_>>();
    for (name, encoded, extent) in [
        (
            "gray12-up4",
            corpus::vardct_depth_combined("gray_12_upsample"),
            Extent2d::new(259, 515),
        ),
        (
            "gray16-dc",
            corpus::vardct_depth_combined("gray_16_dc"),
            Extent2d::new(128, 1024),
        ),
        (
            "jpeg420",
            corpus::vardct_oriented_jpeg(),
            Extent2d::new(101, 173),
        ),
    ] {
        check_formats(&backend, name, &encoded, extent, &formats);
    }
}

#[test]
fn generic_color_output_converts_d65_primaries_against_djxl() {
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
    let encoded = corpus::vardct_depth_combined("rgb_16_multilf");
    let extent = Extent2d::new(17, 2056);
    for (space, transfer, profile) in [
        (
            ColorSpace::DisplayP3,
            TransferFunction::Srgb,
            "RGB_D65_DCI_Rel_SRG",
        ),
        (
            ColorSpace::Bt2020,
            TransferFunction::Bt709,
            "RGB_D65_202_Rel_709",
        ),
    ] {
        let Some(samples) = djxl_rgb_f32_with_color(&encoded, extent, Some(profile)) else {
            return;
        };
        let format = PixelFormat::rgb8(
            RgbChannelOrder::Bgra,
            true,
            ColorSpecification::Defined(ColorSpec {
                space,
                transfer,
                ..ColorSpec::bt709(ColorRange::Full, ChromaLocation2d::CENTER)
            }),
        );
        let rgb: [Vec<f32>; 3] = std::array::from_fn(|channel| {
            samples
                .as_chunks::<3>()
                .0
                .iter()
                .map(|pixel| pixel[channel])
                .collect()
        });
        let expected = convert_rgb_f32([&rgb[0], &rgb[1], &rgb[2]], extent, &format).unwrap();
        let mut session =
            open_incremental(&decoder, &encoded, GpuOutputRequest::color(format).unwrap());
        let frame = pollster::block_on(session.next_frame_async())
            .unwrap()
            .unwrap();
        let readback = ImageReadbackPipeline::new(&backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let actual = &readback.frame.outputs[0];
        assert_eq!(actual.layout, expected.layout);
        assert_color_codes(
            profile,
            &actual.bytes,
            &expected.bytes,
            &expected.layout,
            1e-4,
        );
        drop(readback);
        drop(frame);
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
