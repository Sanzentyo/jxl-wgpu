use super::*;
use jxl_gpu_protocol::DisplayIntensity;
use jxl_test_support::{fixtures::hdr as corpus, oracles::hdr};
use jxl_wgpu_decode::gain_map::{GainMapRendering, GainMapRendition};

use super::reference::{Map, Reference};

#[test]
fn gain_samples_keep_the_existing_original_color_precision() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let mut values = 0;
    for index in (0..16).step_by(2) {
        let map = Map::load(index);
        let mut output = format(false);
        let ColorSpecification::Defined(ref mut fields) = output.color_spec else {
            unreachable!()
        };
        fields.transfer = TransferFunction::Srgb;
        let mut session = decoder
            .open(&map.code, GpuOutputRequest::color(output).unwrap())
            .unwrap();
        let frame = session.next_frame().unwrap().unwrap();
        let actual = planes::read(&backend, &frame.output().outputs[0]);
        map.check_pixels(&actual);
        values += actual.len();
    }
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    eprintln!(
        "gain samples: 8 GPU outputs, {values} values at the existing codec precision bounds"
    );
}

#[test]
fn both_headroom_directions_interpolate_clamp_and_preserve_baseline_bits() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let native = native::Oracle::new();
    let mut count = 0;
    let mut values = 0;
    for index in (0..16).step_by(2) {
        let mut map = Map::load(index);
        let base = std::fs::read(directory().join(format!("case_{index}.jxl"))).unwrap();
        let pixels = floats(&format!("case_{index}.base.f32"));
        let request = || {
            GpuOutputRequest::color(format(false))
                .unwrap()
                .with_orientation_policy(OrientationPolicy::Keep)
        };
        let mut session = decoder.open(&base, request()).unwrap();
        let original_frame = session.next_frame().unwrap().unwrap();
        let original = planes::read(&backend, &original_frame.output().outputs[0]);
        drop((original_frame, session));
        for reverse in [false, true] {
            map.metadata.base_hdr_headroom.numerator = if reverse { 3 } else { 1 };
            map.metadata.alternate_hdr_headroom.numerator = if reverse { 1 } else { 3 };
            let encoded = map.container(&base);
            for headroom in [0.0, 1.0, 2.0, 3.0, 4.0] {
                let weight = if reverse {
                    -((3.0_f64 - headroom) / 2.0).clamp(0.0, 1.0)
                } else {
                    ((headroom - 1.0_f64) / 2.0).clamp(0.0, 1.0)
                };
                let reference = Reference {
                    map: &map,
                    extent: [17, 9],
                    weight,
                    scale: 1.0,
                };
                if let Some(native) = &native {
                    reference.check_native(native, &pixels, headroom);
                }
                let frame = pollster::block_on(decoder.decode_gain_map(
                    &encoded,
                    request(),
                    GainMapRendering {
                        rendition: GainMapRendition::DisplayHeadroom(headroom),
                        ..Default::default()
                    },
                    Default::default(),
                ))
                .unwrap();
                let actual = planes::read(&backend, &frame.output().outputs[0]);
                if weight == 0.0 {
                    assert_eq!(
                        actual, original,
                        "baseline endpoint {index}/{reverse}/{headroom}"
                    );
                }
                let expected = reference.image(&pixels);
                assert_eq!(actual.len(), expected.len());
                for (i, (&word, value)) in actual.iter().zip(&expected).enumerate() {
                    let actual = f64::from(f32::from_bits(word));
                    let bound = if i % 4 == 3 {
                        2e-7
                    } else {
                        2e-4 * (1.0 + value.abs())
                    };
                    assert!(
                        actual.is_finite() && (actual - value).abs() <= bound,
                        "{index}/{reverse}/{headroom}/{i}: {actual} vs {value}"
                    );
                }
                drop(frame);
                count += 1;
                values += actual.len();
            }
        }
    }
    assert_eq!(count, 80);
    assert_eq!(values, 48_960);
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    eprintln!(
        "gain-map headroom: {count} GPU outputs, {values} values, {} live native signed-weight comparisons; baseline endpoints exact",
        if native.is_some() { count } else { 0 }
    );
}

#[test]
fn hdr_baselines_apply_reference_white_and_gain_before_requested_transfers() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let native = native::Oracle::new();
    let mut count = 0;
    let mut values = 0;
    for (index, case) in corpus::cases()
        .into_iter()
        .filter(|c| !c.sequence)
        .enumerate()
    {
        let mut map = Map::load((index % 4) * 4);
        let bytes = case.bytes();
        case.validate(
            &jxl_gpu_bitstream::parse(&bytes, Default::default())
                .unwrap()
                .codestream_inventory(Default::default())
                .unwrap(),
        );
        let original = case.reference(false);
        let base: Vec<_> = if case.xyb {
            case.reference(true).into_iter().map(f64::from).collect()
        } else {
            original
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|rgba| {
                    let rgb = hdr::to_linear(
                        [rgba[0], rgba[1], rgba[2]].map(f64::from),
                        case.transfer,
                        case.space,
                        case.nits,
                    );
                    [rgb[0], rgb[1], rgb[2], f64::from(rgba[3])]
                })
                .collect()
        };
        for reverse in [false, true] {
            map.metadata.base_hdr_headroom.numerator = if reverse { 3 } else { 1 };
            map.metadata.alternate_hdr_headroom.numerator = if reverse { 0 } else { 4 };
            let encoded = map.container(&bytes);
            let selections = if reverse {
                [(0.0, -1.0), (1.5, -0.5), (2.25, -0.25), (4.0, 0.0)]
            } else {
                [(4.0, 1.0), (2.5, 0.5), (1.75, 0.25), (0.0, 0.0)]
            };
            for ((headroom, weight), (target, white)) in selections.into_iter().zip([
                (TransferFunction::Linear, 203.0_f32),
                (TransferFunction::Pq, 500.0),
                (TransferFunction::Hlg, 100.0),
                (TransferFunction::Hlg, 100.0),
            ]) {
                let reference = Reference {
                    map: &map,
                    extent: [case.width, case.height],
                    weight,
                    scale: case.nits / f64::from(white),
                };
                if let Some(native) = &native {
                    reference.check_native(native, &base, headroom);
                }
                let request = GpuOutputRequest::color(case.format(target, case.space))
                    .unwrap()
                    .with_orientation_policy(OrientationPolicy::Keep);
                let frame = pollster::block_on(decoder.decode_gain_map(
                    &encoded,
                    request.clone(),
                    GainMapRendering {
                        rendition: if weight.abs() == 1.0 {
                            GainMapRendition::Alternate
                        } else {
                            GainMapRendition::DisplayHeadroom(headroom)
                        },
                        reference_white: DisplayIntensity::new(white).unwrap(),
                    },
                    Default::default(),
                ))
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
                let words = planes::read(&backend, &frame.output().outputs[0]);
                assert_eq!(words.len(), case.width * case.height * 4);
                count += 1;
                values += words.len();
                if weight == 0.0 {
                    let mut session = decoder.open(&bytes, request).unwrap();
                    let ordinary = session.next_frame().unwrap().unwrap();
                    assert_eq!(words, planes::read(&backend, &ordinary.output().outputs[0]));
                    drop((ordinary, session, frame));
                    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    // Preserve ordinary same-encoding bypasses. An HLG roundtrip through its
                    // signed below-black extension is not an identity and is not executed here.
                    continue;
                }
                for (pixel, actual) in words.as_chunks::<4>().0.iter().enumerate() {
                    let at = pixel * 4;
                    let rgb = [base[at], base[at + 1], base[at + 2]];
                    let source = if case.xyb {
                        rgb.map(|v| {
                            [
                                v - f64::from(case.tolerance()) * (1.0 + v.abs()),
                                v + f64::from(case.tolerance()) * (1.0 + v.abs()),
                            ]
                        })
                    } else {
                        hdr::linear_interval(
                            [original[at], original[at + 1], original[at + 2]].map(f64::from),
                            case.transfer,
                            case.space,
                            case.nits,
                            f64::from(case.tolerance()),
                        )
                    };
                    let xy = [pixel % case.width, pixel / case.width];
                    let bounds = reference.interval(source, xy);
                    let center = reference.rgb(rgb, xy);
                    let gain_interval = std::array::from_fn(|c| {
                        [
                            bounds[c][0] - 2e-4 * (1.0 + center[c].abs()),
                            bounds[c][1] + 2e-4 * (1.0 + center[c].abs()),
                        ]
                    });
                    let interval =
                        hdr::from_linear_interval(gain_interval, target, case.space, case.nits);
                    let expected = hdr::from_linear(center, target, case.space, case.nits);
                    for c in 0..4 {
                        let value = f64::from(f32::from_bits(actual[c]));
                        let [low, high] = if c == 3 {
                            [base[at + 3] - 2e-6, base[at + 3] + 2e-6]
                        } else {
                            let packing = 5e-5 * (1.0 + expected[c].abs());
                            [interval[c][0] - packing, interval[c][1] + packing]
                        };
                        assert!(
                            value.is_finite() && value >= low && value <= high,
                            "{} reverse={reverse} {target:?} white={white} {pixel}/{c}: {value}, interval [{low},{high}]",
                            case.name
                        );
                    }
                }
                drop(frame);
                assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
                assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
            }
        }
    }
    assert_eq!(count, 384);
    assert_eq!(values, 774_656);
    eprintln!(
        "gain-map HDR baselines: 48 native HDR streams, {count} GPU outputs, {values} values, {} live native applications across both directions, PQ/HLG, four intensities, codecs, XYB and three reference whites",
        if native.is_some() { count } else { 0 }
    );
}

#[test]
fn unused_maps_and_underflowing_weights_have_distinct_offset_semantics() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let base = std::fs::read(directory().join("case_0.jxl")).unwrap();
    let pixels = floats("case_0.base.f32");
    let request = || {
        GpuOutputRequest::color(format(false))
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Keep)
    };
    let mut session = decoder.open(&base, request()).unwrap();
    let original_frame = session.next_frame().unwrap().unwrap();
    let original = planes::read(&backend, &original_frame.output().outputs[0]);
    drop((original_frame, session));
    for equal in [false, true] {
        let mut map = Map::load(0);
        map.metadata.base_hdr_headroom.numerator = if equal { 2 } else { 0 };
        map.metadata.alternate_hdr_headroom.numerator = 2;
        map.code = vec![0xff, 0x0a]; // Incomplete image, retained but unused by a baseline selection.
        let encoded = map.container(&base);
        let rendering = GainMapRendering {
            rendition: if equal {
                GainMapRendition::Alternate
            } else {
                GainMapRendition::DisplayHeadroom(0.0)
            },
            ..Default::default()
        };
        let frame = pollster::block_on(decoder.decode_gain_map(
            &encoded,
            request(),
            rendering,
            Default::default(),
        ))
        .unwrap();
        assert_eq!(planes::read(&backend, &frame.output().outputs[0]), original);
    }
    let map = Map::load(0);
    let encoded = map.container(&base);
    let frame = pollster::block_on(decoder.decode_gain_map(
        &encoded,
        request(),
        GainMapRendering {
            rendition: GainMapRendition::DisplayHeadroom(1e-300),
            ..Default::default()
        },
        Default::default(),
    ))
    .unwrap();
    let words = planes::read(&backend, &frame.output().outputs[0]);
    assert_eq!(words.len(), pixels.len());
    for (i, actual) in words.into_iter().enumerate() {
        let c = i % 4;
        let expected = if c == 3 {
            pixels[i]
        } else {
            pixels[i] + map.metadata.channels[c].base_offset.value()
                - map.metadata.channels[c].alternate_offset.value()
        };
        assert!((f64::from(f32::from_bits(actual)) - expected).abs() < 2e-6);
    }
    drop(frame);
    let invalid = pollster::block_on(decoder.decode_gain_map(
        &encoded,
        request(),
        GainMapRendering {
            reference_white: DisplayIntensity::new(f32::from_bits(1)).unwrap(),
            ..Default::default()
        },
        Default::default(),
    ));
    assert!(matches!(
        invalid,
        Err(jxl_wgpu_decode::Error::GainMap(
            GainMapDecodeError::Unsupported("reference-white scaling exceeds portable F32 range")
        ))
    ));
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    eprintln!(
        "gain-map endpoint edge cases: 3 GPU outputs, {} values; unused map and tiny weight semantics verified",
        3 * pixels.len()
    );
}
