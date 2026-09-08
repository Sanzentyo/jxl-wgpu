use super::*;
use jxl_wgpu_decode::AlphaOutputPolicy;
use jxl_wgpu_encode::LosslessModularFormat;

fn stills() -> Vec<(&'static str, &'static str)> {
    macro_rules! pair {
        ($name:literal) => {
            [
                (
                    concat!("modular_", $name),
                    include_str!(concat!("../../../test-data/extras_", $name, ".jxl.hex")),
                ),
                (
                    concat!("vardct_", $name),
                    include_str!(concat!(
                        "../../../test-data/vardct_extras_",
                        $name,
                        ".jxl.hex"
                    )),
                ),
            ]
        };
    }
    [
        pair!("associated_rgb"),
        pair!("associated_same"),
        pair!("associated_gray"),
        pair!("associated_data"),
        pair!("associated_thin"),
        pair!("associated_resampled"),
        pair!("associated_squeeze"),
        pair!("rgba"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

pub(super) fn floating_request(
    policy: AlphaOutputPolicy,
    linear: bool,
    keep: bool,
) -> GpuOutputRequest {
    let mut color = vardct_rgb8_format().color_spec;
    if linear {
        let jxl_gpu_formats::ColorSpecification::Defined(ref mut spec) = color else {
            unreachable!()
        };
        spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
    }
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        if keep {
            RgbChannelOrder::Bgra
        } else {
            RgbChannelOrder::Rgba
        },
        keep,
        color,
    ))
    .unwrap()
    .with_alpha_output_policy(policy)
    .with_orientation_policy(if keep {
        OrientationPolicy::Keep
    } else {
        OrientationPolicy::Apply
    })
    .with_spot_color_policy(SpotColorPolicy::Preserve)
}

pub(super) fn decode(
    backend: &WgpuBackend,
    data: &[u8],
    request: GpuOutputRequest,
    bounded: bool,
) -> Vec<(ImageLayout, Vec<u8>)> {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if bounded {
        engine = engine.with_stream_window_limit(NonZeroU64::new(1024).unwrap());
    }
    let decoder = GpuDecoder::new(engine);
    let mut session = if bounded {
        resampled::fragmented(&decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let inventory = jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(
        session.metadata().extra_channels,
        inventory.image_header.extra_channels
    );
    let mut frames = Vec::new();
    loop {
        let frame = if bounded {
            pollster::block_on(session.next_frame_async())
        } else {
            session.next_frame()
        }
        .unwrap();
        let Some(frame) = frame else { break };
        let read = ImageReadbackPipeline::new(backend)
            .submit(frame.output())
            .unwrap()
            .wait()
            .unwrap();
        let output = &read.frame.outputs[0];
        frames.push((output.layout.clone(), output.bytes.clone()));
    }
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    frames
}

pub(super) fn unpack(layout: &ImageLayout, bytes: &[u8], planar: bool) -> Vec<f32> {
    let width = layout.extent.width as usize;
    (0..layout.extent.area().unwrap())
        .flat_map(|pixel| {
            (0..4).map(move |c| {
                let stored = if planar && c < 3 { 2 - c } else { c };
                let plane = &layout.planes[if planar { stored } else { 0 }];
                let offset = plane.offset as usize
                    + (pixel / width) * plane.row_stride as usize
                    + (pixel % width) * if planar { 4 } else { 16 }
                    + if planar { 0 } else { stored * 4 };
                f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
            })
        })
        .collect()
}

pub(super) fn associate(values: &mut [f32], policy: AlphaOutputPolicy, source_associated: bool) {
    for pixel in values.chunks_exact_mut(4) {
        let alpha = pixel[3].max(1.0 / 67108864.0);
        let factor = match (policy, source_associated) {
            (AlphaOutputPolicy::Unassociated, true) => 1.0 / alpha,
            (AlphaOutputPolicy::Associated, false) => alpha,
            _ => 1.0,
        };
        for color in &mut pixel[..3] {
            *color *= factor;
        }
    }
}

pub(super) fn linearize(values: &mut [f32]) {
    for pixel in values.chunks_exact_mut(4) {
        for value in &mut pixel[..3] {
            let magnitude = value.abs();
            *value = if magnitude <= 0.04045 {
                magnitude / 12.92
            } else {
                ((magnitude + 0.055) / 1.055).powf(2.4)
            }
            .copysign(*value);
        }
    }
}

fn compare(name: &str, values: &[f32], expected: &[f32], unpremultiplied: bool, tolerance: f32) {
    assert_eq!(values.len(), expected.len());
    let mut maximum = 0.0_f32;
    for (index, (a, b)) in values.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{name}/{index}: {a} vs {b}");
        // Undo output scaling only for the comparison so alpha near zero cannot hide a color
        // reconstruction error or amplify the documented CPU/GPU IDCT tolerance arbitrarily.
        let scale = if unpremultiplied && index % 4 != 3 {
            expected[index / 4 * 4 + 3].max(1.0 / 67108864.0)
        } else {
            1.0
        };
        let error = (a - b).abs() * scale / (b * scale).abs().max(1.0);
        let limit = if index % 4 == 3 { 4e-7 } else { tolerance };
        maximum = maximum.max(error);
        assert!(
            error < limit,
            "{name}/{index}: {a} vs {b}, scaled error {error} >= {limit}"
        );
    }
    eprintln!("{name}: scaled maximum {maximum}");
}

#[test]
fn associated_stills_preserve_or_convert_alpha_after_color_and_resampling() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in stills() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let associated = image
            .extra_channels
            .iter()
            .find_map(|e| match e.channel_type {
                ExtraChannelTypeInventory::Alpha { associated } => Some(associated),
                _ => None,
            })
            .unwrap();
        assert_eq!(associated, name.contains("associated"));
        let pixels = (image.width * image.height) as usize;
        let rust = (!name.contains("resampled")).then(|| oracle::rust_planes(&data).0);
        for (linear, keep) in [(false, false), (true, true)] {
            let options = if linear {
                vec!["--preserve-alpha", "--linear", "--keep-orientation"]
            } else {
                vec!["--preserve-alpha"]
            };
            let native_oracle =
                oracle::libjxl_output(&data, &options).map(|v| v[..pixels * 4].to_vec());
            let analytic = linear
                .then(|| oracle::libjxl_output(&data, &["--preserve-alpha", "--keep-orientation"]))
                .flatten()
                .map(|v| {
                    let mut v = v[..pixels * 4].to_vec();
                    linearize(&mut v);
                    v
                });
            for policy in [
                AlphaOutputPolicy::Preserve,
                AlphaOutputPolicy::Unassociated,
                AlphaOutputPolicy::Associated,
            ] {
                let request = floating_request(policy, linear, keep);
                let whole = decode(&backend, &data, request.clone(), false);
                let bounded = decode(&backend, &data, request, true);
                assert_eq!(whole, bounded, "{name}/{policy:?}: bounded input");
                assert_eq!(whole.len(), 1);
                let values = unpack(&whole[0].0, &whole[0].1, keep);
                let unpremultiplied = associated && policy == AlphaOutputPolicy::Unassociated;
                if let Some(reference) = &native_oracle {
                    let mut reference = reference.clone();
                    associate(&mut reference, policy, associated);
                    compare(
                        name,
                        &values,
                        &reference,
                        unpremultiplied,
                        if linear {
                            1e-4
                        } else if name.starts_with("modular") {
                            3e-6
                        } else {
                            0.002
                        },
                    );
                }
                if let Some(reference) = &analytic {
                    let mut reference = reference.clone();
                    associate(&mut reference, policy, associated);
                    compare(
                        name,
                        &values,
                        &reference,
                        unpremultiplied,
                        if name.starts_with("modular") {
                            3e-6
                        } else {
                            0.002
                        },
                    );
                }
                if !linear && let Some(reference) = &rust {
                    let mut reference = reference.clone();
                    associate(&mut reference, policy, associated);
                    compare(
                        name,
                        &values,
                        &reference,
                        unpremultiplied,
                        if name.starts_with("modular") {
                            3e-6
                        } else {
                            0.0003
                        },
                    );
                }
                if !linear {
                    verify_integer(&backend, name, &data, image.bit_depth, policy, &values);
                }
            }
        }
    }
}

fn verify_integer(
    backend: &WgpuBackend,
    name: &str,
    data: &[u8],
    depth: jxl_gpu_bitstream::SampleBitDepth,
    policy: AlphaOutputPolicy,
    reference: &[f32],
) {
    let jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } = depth else {
        unreachable!()
    };
    let bits = if name.starts_with("vardct") {
        8
    } else {
        bits_per_sample as u8
    };
    for alpha in [false, true] {
        let mut format = (if alpha {
            LosslessModularFormat::Rgba
        } else {
            LosslessModularFormat::Rgb
        })
        .pixel_format(bits)
        .unwrap();
        if name.starts_with("vardct") {
            format.color_spec = vardct_rgb8_format().color_spec;
        }
        let request = GpuOutputRequest::color(format.clone())
            .unwrap()
            .with_alpha_output_policy(policy)
            .with_spot_color_policy(SpotColorPolicy::Preserve);
        let frames = decode(backend, data, request.clone(), false);
        assert_eq!(frames, decode(backend, data, request, true));
        let channels = if alpha { 4 } else { 3 };
        let samples: Vec<u16> = if bits > 8 {
            frames[0]
                .1
                .chunks_exact(2)
                .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
                .collect()
        } else {
            frames[0].1.iter().map(|v| u16::from(*v)).collect()
        };
        let maximum = ((1u32 << bits) - 1) as f32;
        for (i, sample) in samples.iter().enumerate() {
            let expected = (reference[i / channels * 4 + i % channels].clamp(0.0, 1.0) * maximum)
                .round() as u16;
            assert!(
                sample.abs_diff(expected) <= 1,
                "{name}/{policy:?}/{alpha}/{i}: {sample} vs {expected}"
            );
        }
    }
}

#[test]
fn alpha_output_policy_never_changes_selected_extra_samples() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in stills()
        .into_iter()
        .filter(|(name, _)| name.contains("associated"))
    {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let pixels = (image.width * image.height) as usize;
        let Some((_, expected)) = oracle::libjxl_planes(&data, pixels, image.extra_channels.len())
        else {
            continue;
        };
        for (index, extra) in image.extra_channels.iter().enumerate() {
            let jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } = extra.bit_depth
            else {
                unreachable!()
            };
            for floating in [false, true] {
                let request = scalar::scalar_request(index as u32, bits_per_sample as u8, floating)
                    .with_orientation_policy(OrientationPolicy::Apply);
                let whole = decode(
                    &backend,
                    &data,
                    request
                        .clone()
                        .with_alpha_output_policy(AlphaOutputPolicy::Unassociated),
                    false,
                );
                let bounded = decode(
                    &backend,
                    &data,
                    request.with_alpha_output_policy(AlphaOutputPolicy::Associated),
                    true,
                );
                assert_eq!(
                    whole, bounded,
                    "{name}/{index}/{floating}: output policy affected extra"
                );
                if floating {
                    for (a, b) in oracle::floats(&whole[0].1)
                        .into_iter()
                        .zip(&expected[index])
                    {
                        assert!((a - b).abs() < 4e-7);
                    }
                } else {
                    let mask = ((1u32 << bits_per_sample) - 1) as f32;
                    for (a, b) in whole[0]
                        .1
                        .chunks_exact(if bits_per_sample > 8 { 2 } else { 1 })
                        .zip(&expected[index])
                    {
                        let a = if bits_per_sample > 8 {
                            u16::from_le_bytes(a.try_into().unwrap())
                        } else {
                            u16::from(a[0])
                        };
                        assert!(a.abs_diff((b * mask).round() as u16) <= 1);
                    }
                }
            }
        }
    }
}

#[test]
fn alpha_conversion_precedes_yuv_subsampling_and_quantization() {
    use jxl_gpu_formats::{
        ChromaLocation2d, ColorRange, ColorSpec, ColorSpecification, TransferFunction,
        YcbcrEncoding, convert_rgb_f32,
    };
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let srgb = ColorSpecification::Defined(ColorSpec {
        transfer: TransferFunction::Srgb,
        ..ColorSpec::bt709(ColorRange::Limited, ChromaLocation2d::CENTER)
    });
    let cl = ColorSpecification::Defined(ColorSpec {
        encoding: YcbcrEncoding::Bt2020ConstantLuminance,
        ..ColorSpec::bt2020_ncl(ColorRange::Limited, ChromaLocation2d::CENTER)
    });
    for hex in [
        include_str!("../../../test-data/vardct_extras_associated_resampled.jxl.hex"),
        include_str!("../../../test-data/vardct_extras_rgba.jxl.hex"),
    ] {
        let data = encoded(hex);
        for format in [
            PixelFormat::nv12(srgb),
            PixelFormat::p010(cl),
            PixelFormat::packed_yuv4228(jxl_gpu_formats::Packed422Order::Yuyv, srgb),
            PixelFormat::packed_yuv4228(jxl_gpu_formats::Packed422Order::Uyvy, srgb),
        ] {
            for policy in [
                AlphaOutputPolicy::Preserve,
                AlphaOutputPolicy::Unassociated,
                AlphaOutputPolicy::Associated,
            ] {
                let ColorSpecification::Defined(mut rgb_color) = format.color_spec else {
                    unreachable!()
                };
                rgb_color.range = ColorRange::Full;
                let floating = GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    ColorSpecification::Defined(rgb_color),
                ))
                .unwrap()
                .with_alpha_output_policy(policy);
                let rgba = decode(&backend, &data, floating, false);
                let values = oracle::floats(&rgba[0].1);
                let planes: [Vec<f32>; 3] =
                    std::array::from_fn(|c| values.chunks_exact(4).map(|p| p[c]).collect());
                let expected = if format.color_spec == cl {
                    p010_cl(&values, rgba[0].0.extent, &format)
                } else {
                    convert_rgb_f32(
                        [&planes[0], &planes[1], &planes[2]],
                        rgba[0].0.extent,
                        &format,
                    )
                    .unwrap()
                };
                let request = GpuOutputRequest::color(format.clone())
                    .unwrap()
                    .with_alpha_output_policy(policy);
                let actual = decode(&backend, &data, request.clone(), false);
                assert_eq!(actual, decode(&backend, &data, request, true));
                assert_eq!(actual[0].0, expected.layout);
                let high_depth = expected.layout.format.color_spec == cl;
                for (a, b) in actual[0]
                    .1
                    .chunks_exact(if high_depth { 2 } else { 1 })
                    .zip(expected.bytes.chunks_exact(if high_depth { 2 } else { 1 }))
                {
                    let (a, b) = if high_depth {
                        (
                            u16::from_le_bytes(a.try_into().unwrap()),
                            u16::from_le_bytes(b.try_into().unwrap()),
                        )
                    } else {
                        (u16::from(a[0]), u16::from(b[0]))
                    };
                    if high_depth {
                        assert_eq!(a & 63, 0);
                    }
                    assert!(
                        a.abs_diff(b) <= if high_depth { 64 } else { 1 },
                        "{policy:?}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

pub(super) fn p010_cl(
    values: &[f32],
    extent: Extent2d,
    format: &PixelFormat,
) -> jxl_gpu_formats::ConvertedImage {
    // Independent f64 BT.2020 constant-luminance equations, followed by centered 2x2 chroma
    // averaging and the limited 10-bit P010 storage contract.
    fn transfer(value: f64, encode: bool) -> f64 {
        let alpha = 1.09929682680944;
        let beta = 0.018053968510807;
        let m = value.abs();
        let mapped = if encode {
            if m < beta {
                4.5 * m
            } else {
                alpha * m.powf(0.45) - (alpha - 1.0)
            }
        } else if m < 4.5 * beta {
            m / 4.5
        } else {
            ((m + alpha - 1.0) / alpha).powf(1.0 / 0.45)
        };
        mapped.copysign(value)
    }
    let yuv: Vec<[f64; 3]> = values
        .chunks_exact(4)
        .map(|p| {
            let rgb = [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])];
            let linear = rgb.map(|v| transfer(v, false));
            let y = transfer(
                0.2627 * linear[0] + 0.6780 * linear[1] + 0.0593 * linear[2],
                true,
            );
            [
                y,
                (rgb[2] - y) / if rgb[2] > y { 1.5816 } else { 1.9404 } + 0.5,
                (rgb[0] - y) / if rgb[0] > y { 0.9936 } else { 1.7184 } + 0.5,
            ]
        })
        .collect();
    let layout = ImageLayout::packed(extent, format.clone()).unwrap();
    let mut bytes = vec![0; layout.logical_size as usize];
    for (pi, plane) in layout.planes.iter().enumerate() {
        for y in 0..plane.sample_extent.height {
            for x in 0..plane.sample_extent.width {
                for c in 0..if pi == 0 { 1 } else { 2 } {
                    let value = if pi == 0 {
                        yuv[(y * extent.width + x) as usize][0]
                    } else {
                        let mut sum = 0.0;
                        let mut count = 0;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let (sx, sy) = (2 * x + dx, 2 * y + dy);
                                if sx < extent.width && sy < extent.height {
                                    sum += yuv[(sy * extent.width + sx) as usize][c + 1];
                                    count += 1;
                                }
                            }
                        }
                        sum / f64::from(count)
                    };
                    let code = if pi == 0 {
                        64.0 + 876.0 * value
                    } else {
                        512.0 + 896.0 * (value - 0.5)
                    };
                    let stored = (code.round().clamp(0.0, 1023.0) as u16) << 6;
                    let offset = plane.offset as usize
                        + y as usize * plane.row_stride as usize
                        + x as usize * if pi == 0 { 2 } else { 4 }
                        + c * 2;
                    bytes[offset..offset + 2].copy_from_slice(&stored.to_le_bytes());
                }
            }
        }
    }
    jxl_gpu_formats::ConvertedImage { layout, bytes }
}

#[test]
fn associated_composition_keeps_reference_values_until_final_packing() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    for (name, hex) in [
        (
            "modular_rgb",
            include_str!("../../../test-data/composition_associated_rgb.jxl.hex"),
        ),
        (
            "modular_gray",
            include_str!("../../../test-data/composition_associated_gray.jxl.hex"),
        ),
        (
            "vardct",
            include_str!("../../../test-data/composition_associated_vardct.jxl.hex"),
        ),
    ] {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        assert!(inventory.frames.len() >= 9);
        let pixels = (inventory.image_header.width * inventory.image_header.height) as usize;
        let rust = oracle::rust_frame_planes(&data);
        for (linear, keep) in [(false, false), (true, true)] {
            let options = if linear && name == "vardct" {
                vec!["--preserve-alpha", "--keep-orientation"]
            } else if linear {
                vec!["--preserve-alpha", "--linear", "--keep-orientation"]
            } else {
                vec!["--preserve-alpha"]
            };
            let Some(reference) = oracle::libjxl_output(&data, &options) else {
                continue;
            };
            let analytic = linear.then(|| {
                oracle::libjxl_output(&data, &["--preserve-alpha", "--keep-orientation"]).unwrap()
            });
            for policy in [
                AlphaOutputPolicy::Preserve,
                AlphaOutputPolicy::Unassociated,
                AlphaOutputPolicy::Associated,
            ] {
                let request = floating_request(policy, linear, keep);
                let frames = decode(&backend, &data, request.clone(), false);
                assert_eq!(frames, decode(&backend, &data, request, true));
                assert_eq!(reference.len(), frames.len() * pixels * 5);
                if !linear {
                    let jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } =
                        inventory.image_header.bit_depth
                    else {
                        unreachable!()
                    };
                    let bits = if name == "vardct" {
                        8
                    } else {
                        bits_per_sample as u8
                    };
                    let mut format = LosslessModularFormat::Rgba.pixel_format(bits).unwrap();
                    if name == "vardct" {
                        format.color_spec = vardct_rgb8_format().color_spec;
                    }
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(policy);
                    let packed = decode(&backend, &data, request.clone(), false);
                    assert_eq!(packed, decode(&backend, &data, request, true));
                    for ((_, bytes), (layout, floats)) in packed.iter().zip(&frames) {
                        let values = unpack(layout, floats, false);
                        let mask = ((1u32 << bits) - 1) as f32;
                        for (sample, value) in
                            bytes.chunks_exact(if bits > 8 { 2 } else { 1 }).zip(values)
                        {
                            let code = if bits > 8 {
                                u16::from_le_bytes(sample.try_into().unwrap())
                            } else {
                                u16::from(sample[0])
                            };
                            assert!(
                                code.abs_diff((value.clamp(0.0, 1.0) * mask).round() as u16) <= 1
                            );
                        }
                    }
                }
                for (index, ((layout, bytes), reference)) in frames
                    .iter()
                    .zip(reference.chunks_exact(pixels * 5))
                    .enumerate()
                {
                    let mut expected = reference[..pixels * 4].to_vec();
                    if linear && name == "vardct" {
                        linearize(&mut expected);
                    }
                    associate(&mut expected, policy, true);
                    let actual = unpack(layout, bytes, keep);
                    let (checked_actual, checked_expected) = if let Some(original) = &analytic
                        && name != "vardct"
                    {
                        // libjxl's CMS uses an approximation outside the unit sRGB cube. Its
                        // in-gamut output is checked directly; the analytic comparison below
                        // independently checks every extended value without that approximation.
                        let original =
                            &original[index * pixels * 5..index * pixels * 5 + pixels * 4];
                        let mut a = Vec::new();
                        let mut b = Vec::new();
                        for ((source, actual), expected) in original
                            .chunks_exact(4)
                            .zip(actual.chunks_exact(4))
                            .zip(expected.chunks_exact(4))
                        {
                            if source[..3].iter().all(|v| (0.0..=1.0).contains(v)) {
                                a.extend_from_slice(actual);
                                b.extend_from_slice(expected);
                            }
                        }
                        assert!(!a.is_empty());
                        (a, b)
                    } else {
                        (actual.clone(), expected)
                    };
                    compare(
                        name,
                        &checked_actual,
                        &checked_expected,
                        policy == AlphaOutputPolicy::Unassociated,
                        if linear && name != "vardct" {
                            1e-4
                        } else if name.starts_with("modular") {
                            8e-6
                        } else {
                            0.003
                        },
                    );
                    let mut independent = if let Some(analytic) = &analytic {
                        let mut values =
                            analytic[index * pixels * 5..index * pixels * 5 + pixels * 4].to_vec();
                        linearize(&mut values);
                        values
                    } else {
                        rust[index].0.clone()
                    };
                    associate(&mut independent, policy, true);
                    compare(
                        name,
                        &unpack(layout, bytes, keep),
                        &independent,
                        policy == AlphaOutputPolicy::Unassociated,
                        if name.starts_with("modular") {
                            8e-6
                        } else {
                            0.003
                        },
                    );
                }
            }
        }
    }
}
