#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use jxl_gpu_bitstream::{
    gain_map::{GainMapBundle, GainMapMetadata, JHGM},
    metadata::{BrotliOptions, MetadataBox, MetadataCompression, MetadataSelection},
};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    GpuDecoder, GpuOutputRequest, OrientationPolicy, gain_map::GainMapDecodeError,
};

mod interop;
mod output;

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/gain_map/generated")
}
fn floats(name: &str) -> Vec<f64> {
    let bytes = std::fs::read(directory().join(name)).unwrap();
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "{name}: partial F32");
    words
        .iter()
        .map(|bytes| f64::from(f32::from_le_bytes(*bytes)))
        .collect()
}
fn format(wide: bool) -> PixelFormat {
    let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
    let ColorSpecification::Defined(ref mut fields) = color else {
        panic!()
    };
    fields.space = if wide {
        ColorSpace::Bt2020
    } else {
        ColorSpace::Bt709
    };
    fields.transfer = TransferFunction::Linear;
    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, color)
}
fn sample(map: &[f64], w: usize, h: usize, x: usize, y: usize, c: usize) -> f64 {
    let px = ((x as f64 + 0.5) * w as f64 / 17.0 - 0.5).clamp(0.0, (w - 1) as f64);
    let py = ((y as f64 + 0.5) * h as f64 / 9.0 - 0.5).clamp(0.0, (h - 1) as f64);
    let x0 = px.floor() as usize;
    let y0 = py.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = px - x0 as f64;
    let fy = py - y0 as f64;
    let at = |x, y| map[(y * w + x) * 4 + c];
    ((1.0 - fy) * ((1.0 - fx) * at(x0, y0) + fx * at(x1, y0))
        + fy * ((1.0 - fx) * at(x0, y1) + fx * at(x1, y1)))
    .clamp(0.0, 1.0)
}
fn oracle(
    base: &[f64],
    map: &[f64],
    w: usize,
    h: usize,
    metadata: &GainMapMetadata,
    wide: bool,
) -> Vec<f64> {
    assert_eq!(base.len(), 17 * 9 * 4);
    assert_eq!(map.len(), w * h * 4);
    let matrix = jxl_test_support::oracles::color::matrix(
        ColorSpace::Bt709,
        if wide {
            ColorSpace::Bt2020
        } else {
            ColorSpace::Bt709
        },
    );
    base.iter()
        .enumerate()
        .map(|(index, value)| {
            let c = index % 4;
            if c == 3 {
                return *value;
            }
            let (x, y) = (index / 4 % 17, index / 4 / 17);
            let channel = metadata.channels[c];
            let gain = sample(map, w, h, x, y, c).powf(1.0 / channel.gamma.value());
            let log_gain = channel.min.value() * (1.0 - gain) + channel.max.value() * gain;
            let pixel = index / 4 * 4;
            let linear: f64 = (0..3).map(|i| matrix[c][i] * base[pixel + i]).sum();
            (linear + channel.base_offset.value()) * log_gain.exp2()
                - channel.alternate_offset.value()
        })
        .collect()
}

#[test]
fn native_jhgm_images_reconstruct_on_gpu_with_color_resampling_alpha_and_orientation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    eprintln!("gain-map GPU: {:?}", backend.adapter_info());
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let cases = std::fs::read_to_string(directory().join("cases.txt")).unwrap();
    let mut count = 0;
    let mut values = 0;
    let mut max_error = 0.0_f64;
    let mut retained = Vec::new();
    for (case_index, case) in cases.lines().enumerate() {
        let fields = case.split_whitespace().collect::<Vec<_>>();
        let name = fields[0];
        let wide = fields[4] == "1";
        let width = fields[5].parse().unwrap();
        let height = fields[6].parse().unwrap();
        let orientation = fields[7].parse::<u32>().unwrap();
        let bytes = std::fs::read(directory().join(format!("{name}.jxl"))).unwrap();
        let parsed = jxl_gpu_bitstream::parse(&bytes, Default::default()).unwrap();
        let item = parsed
            .auxiliary_boxes()
            .iter()
            .find(|b| b.box_type == JHGM)
            .unwrap();
        let bundle = GainMapBundle::parse(item.payload, Default::default()).unwrap();
        let expected = floats(&format!("{name}.expected.f32"));
        assert_eq!(expected.len(), 17 * 9 * 4);
        let reference = oracle(
            &floats(&format!("{name}.base.f32")),
            &floats(&format!("{name}.gain.f32")),
            width,
            height,
            bundle.metadata(),
            wide,
        );
        // libultrahdr's primary conversion uses six-decimal coefficients. Check its gain
        // formula against its own working pixels; GPU color is independently checked above
        // against the F64 chromaticity-derived matrix, without widening either tolerance.
        let native_reference = oracle(
            &floats(&format!("{name}.working.f32")),
            &floats(&format!("{name}.gain.f32")),
            width,
            height,
            bundle.metadata(),
            false,
        );
        for (native, precise) in expected.iter().zip(&native_reference) {
            assert!(
                (native - precise).abs() < 3e-6 * (1.0 + precise.abs()),
                "{name}: native gain formula {native} vs {precise}"
            );
        }
        let mut metadata = parsed
            .metadata(&MetadataSelection::Types(vec![JHGM]), Default::default())
            .unwrap();
        if case_index % 2 == 0 {
            metadata
                .replace(
                    JHGM,
                    Some(
                        MetadataBox::new(
                            JHGM,
                            item.payload,
                            MetadataCompression::Brotli(BrotliOptions::default()),
                            Default::default(),
                        )
                        .unwrap(),
                    ),
                    Default::default(),
                )
                .unwrap();
        }
        let rewritten = metadata.write_container(parsed.codestream()).unwrap();
        for policy in [OrientationPolicy::Keep, OrientationPolicy::Apply] {
            let request = GpuOutputRequest::color(format(wide))
                .unwrap()
                .with_orientation_policy(policy);
            let frame = pollster::block_on(decoder.decode_alternate(
                if policy == OrientationPolicy::Keep {
                    &bytes
                } else {
                    &rewritten
                },
                request,
                Default::default(),
            ))
            .unwrap_or_else(|error| panic!("{name} {policy:?}: {error}"));
            let output = &frame.output().outputs[0];
            let words = planes::read(&backend, output);
            let orientation = if policy == OrientationPolicy::Keep {
                1
            } else {
                orientation
            };
            assert_eq!(
                output.layout.extent.width,
                if orientation >= 5 { 9 } else { 17 }
            );
            assert_eq!(
                output.layout.extent.height,
                if orientation >= 5 { 17 } else { 9 }
            );
            for y in 0..9 {
                for x in 0..17 {
                    let (ox, oy) = match orientation {
                        1 => (x, y),
                        2 => (16 - x, y),
                        3 => (16 - x, 8 - y),
                        4 => (x, 8 - y),
                        5 => (y, x),
                        6 => (8 - y, x),
                        7 => (8 - y, 16 - x),
                        8 => (y, 16 - x),
                        _ => unreachable!(),
                    };
                    for c in 0..4 {
                        let actual = f64::from(f32::from_bits(
                            words[(oy * output.layout.extent.width as usize + ox) * 4 + c],
                        ));
                        let expected = reference[(y * 17 + x) * 4 + c];
                        let error = (actual - expected).abs();
                        max_error = max_error.max(error);
                        assert!(
                            actual.is_finite()
                                && error
                                    <= if c == 3 {
                                        2e-7
                                    } else {
                                        2e-4 * (1.0 + expected.abs())
                                    },
                            "{name} {policy:?} {x},{y},{c}: GPU {actual}, F64 {expected}, delta {error}"
                        );
                        values += 1;
                    }
                }
            }
            retained.push((frame, words));
            count += 1;
        }
    }
    for (frame, expected) in &retained {
        assert_eq!(
            planes::read(&backend, &frame.output().outputs[0]),
            *expected
        );
    }
    drop(retained);
    assert_eq!(count, 128);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    eprintln!(
        "gain-map {count} images, {values} values, max F64 error {max_error:e}; retained outputs stable and all bytes released"
    );
}

#[test]
fn unsupported_gain_profiles_and_metadata_limits_fail_before_gpu_allocation() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let source = std::fs::read(directory().join("case_0.jxl")).unwrap();
    let parsed = jxl_gpu_bitstream::parse(&source, Default::default()).unwrap();
    let request = || GpuOutputRequest::color(format(false)).unwrap();
    let missing = pollster::block_on(decoder.decode_alternate(
        parsed.codestream(),
        request(),
        Default::default(),
    ));
    assert!(matches!(
        missing,
        Err(jxl_wgpu_decode::Error::GainMap(
            GainMapDecodeError::BoxCount(0)
        ))
    ));
    let item = parsed
        .auxiliary_boxes()
        .iter()
        .find(|b| b.box_type == JHGM)
        .unwrap();
    let bundle = GainMapBundle::parse(item.payload, Default::default()).unwrap();
    let duplicate = jxl_gpu_bitstream::write_container_with_boxes(
        parsed.codestream(),
        &[jxl_gpu_bitstream::ContainerBox {
            box_type: JHGM,
            payload: item.payload,
        }; 2],
    )
    .unwrap();
    assert!(matches!(
        pollster::block_on(decoder.decode_alternate(&duplicate, request(), Default::default())),
        Err(jxl_wgpu_decode::Error::GainMap(
            GainMapDecodeError::BoxCount(2)
        ))
    ));
    assert!(matches!(
        pollster::block_on(decoder.decode_alternate(
            &source,
            request().with_progressive_output(true),
            Default::default(),
        )),
        Err(jxl_wgpu_decode::Error::GainMap(
            GainMapDecodeError::Unsupported(_)
        ))
    ));
    // The baseline has alpha. Reusing it as an auxiliary stream is a valid bundle whose
    // rendering profile must be rejected before either image allocates GPU storage.
    let payload = GainMapBundle::new(
        bundle.metadata().clone(),
        &[],
        &[],
        parsed.codestream(),
        Default::default(),
    )
    .unwrap()
    .encode(Default::default())
    .unwrap();
    let with_auxiliary_alpha = jxl_gpu_bitstream::write_container_with_boxes(
        parsed.codestream(),
        &[jxl_gpu_bitstream::ContainerBox {
            box_type: JHGM,
            payload: &payload,
        }],
    )
    .unwrap();
    assert!(matches!(
        pollster::block_on(decoder.decode_alternate(
            &with_auxiliary_alpha,
            request(),
            Default::default()
        )),
        Err(jxl_wgpu_decode::Error::GainMap(
            GainMapDecodeError::Unsupported("auxiliary alpha/extra channels")
        ))
    ));
    for base_headroom in [1, 3] {
        let mut metadata = bundle.metadata().clone();
        metadata.base_hdr_headroom.numerator = base_headroom;
        let payload =
            GainMapBundle::new(metadata, &[], &[], bundle.codestream(), Default::default())
                .unwrap()
                .encode(Default::default())
                .unwrap();
        let bytes = jxl_gpu_bitstream::write_container_with_boxes(
            parsed.codestream(),
            &[jxl_gpu_bitstream::ContainerBox {
                box_type: JHGM,
                payload: &payload,
            }],
        )
        .unwrap();
        assert!(matches!(
            pollster::block_on(decoder.decode_alternate(&bytes, request(), Default::default())),
            Err(jxl_wgpu_decode::Error::GainMap(
                GainMapDecodeError::Unsupported(_)
            ))
        ));
    }
    let limits = jxl_wgpu_decode::gain_map::GainMapDecodeLimits {
        bundle: jxl_gpu_bitstream::gain_map::GainMapLimits {
            max_codestream_bytes: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(matches!(
        pollster::block_on(decoder.decode_alternate(&source, request(), limits)),
        Err(jxl_wgpu_decode::Error::GainMapMetadata(
            jxl_gpu_bitstream::gain_map::GainMapError::Limit { .. }
        ))
    ));
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}
