use super::*;
use jxl_test_support::oracles::composed as composed_oracle;
use jxl_wgpu_decode::{AlphaOutputPolicy, FrameExecutionPlan, OrientationPolicy, WgpuDecodeEngine};

mod ownership;

fn cases() -> [(&'static str, &'static str, &'static str); 4] {
    [
        (
            "modular_pass_rgb",
            "modular_composition",
            include_str!("../../../test-data/modular_composition/modular_pass_rgb.jxl.hex"),
        ),
        (
            "modular_pass_gray_alpha",
            "modular_composition",
            include_str!("../../../test-data/modular_composition/modular_pass_gray_alpha.jxl.hex"),
        ),
        (
            "modular_pass_float",
            "modular_composition",
            include_str!("../../../test-data/modular_composition/modular_pass_float.jxl.hex"),
        ),
        (
            "associated_vardct",
            "progressive_composition",
            include_str!("../../../test-data/composition_associated_vardct.jxl.hex"),
        ),
    ]
}

fn float_mapping(depth: jxl_gpu_bitstream::SampleBitDepth) -> NumericSampleMapping {
    match depth {
        jxl_gpu_bitstream::SampleBitDepth::Integer { .. } => {
            NumericSampleMapping::NormalizedUnsigned
        }
        jxl_gpu_bitstream::SampleBitDepth::Float { .. } => NumericSampleMapping::NativeFloat,
    }
}

fn owned(frame: &GpuImageFrame) -> GpuImageFrame {
    GpuImageFrame {
        token: frame.token,
        changed: frame.changed.clone(),
        outputs: frame
            .outputs
            .iter()
            .map(|output| GpuImageOutput {
                id: output.id,
                layout: output.layout.clone(),
                buffer: output.buffer.clone(),
            })
            .collect(),
    }
}

#[test]
fn composed_modular_and_numeric_updates_match_independent_native_layers() {
    let Some(backend) = backend() else {
        return;
    };
    for (name, directory, hex) in cases() {
        let data = hex_bytes(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let Some(expected) = composed_oracle::updates(&data, directory, name) else {
            return;
        };
        let mut requests = vec![(
            None,
            None,
            rgba_request().with_alpha_output_policy(AlphaOutputPolicy::Preserve),
        )];
        if !image.extra_channels.is_empty() {
            requests.push((
                Some(3),
                None,
                GpuOutputRequest::numeric(
                    Vpi::F32.pixel_format(),
                    float_mapping(image.extra_channels[0].bit_depth),
                )
                .unwrap()
                .with_extra_channel(0)
                .unwrap()
                .with_progressive_output(true),
            ));
            if let jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } =
                image.extra_channels[0].bit_depth
            {
                requests.push((
                    Some(3),
                    Some(bits_per_sample as u8),
                    GpuOutputRequest::numeric(
                        jxl_wgpu_decode::native_modular_pixel_format(
                            ModularChannels::Gray,
                            bits_per_sample as u8,
                        )
                        .unwrap(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                    .unwrap()
                    .with_extra_channel(0)
                    .unwrap()
                    .with_progressive_output(true),
                ));
            }
        }
        if image.grayscale {
            requests.push((
                Some(0),
                None,
                GpuOutputRequest::numeric(Vpi::F32.pixel_format(), float_mapping(image.bit_depth))
                    .unwrap()
                    .with_progressive_output(true),
            ));
            if matches!(
                image.bit_depth,
                jxl_gpu_bitstream::SampleBitDepth::Integer { .. }
            ) {
                requests.push((
                    Some(0),
                    Some(16),
                    GpuOutputRequest::numeric(
                        Vpi::U16.pixel_format(),
                        NumericSampleMapping::NativeUnsigned,
                    )
                    .unwrap()
                    .with_progressive_output(true),
                ));
            }
        }
        for (scalar, integer_bits, request) in requests {
            for orientation in [OrientationPolicy::Apply, OrientationPolicy::Keep] {
                let request = request
                    .clone()
                    .with_orientation_policy(orientation)
                    .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
                let plan = FrameExecutionPlan::negotiate_with_orientation(&inventory, orientation)
                    .unwrap();
                let mut whole = None;
                for cap in [u64::MAX, 40] {
                    let decoder = GpuDecoder::new(
                        WgpuDecodeEngine::new(backend.clone())
                            .unwrap()
                            .with_stream_window_limit(NonZeroU64::new(cap).unwrap()),
                    );
                    let mut session = if cap == u64::MAX {
                        decoder.open(&data, request.clone()).unwrap()
                    } else {
                        frame_sequence::incremental(&decoder, &data, request.clone())
                    };
                    let mut held = Vec::new();
                    let mut pixels = Vec::new();
                    let mut finals = Vec::new();
                    while let Some(update) =
                        pollster::block_on(session.next_update_async()).unwrap()
                    {
                        let reference = &expected[pixels.len()];
                        assert_eq!(
                            update.metadata,
                            plan.presentations[reference.presentation].metadata
                        );
                        assert_eq!(
                            update
                                .progression()
                                .and_then(FrameProgression::completed_passes),
                            reference.completed
                        );
                        if let Some(progression) = update.progression() {
                            assert_eq!(
                                matches!(progression, FrameProgression::Modular { .. }),
                                name.starts_with("modular")
                            );
                            assert_eq!(progression.physical_frame_index(), reference.physical);
                            let physical = &inventory.frames[reference.physical as usize];
                            let completed = progression.completed_passes().unwrap();
                            let intended = physical
                                .progressive_passes
                                .last_pass
                                .iter()
                                .zip(&physical.progressive_passes.downsampling)
                                .filter(|(last, _)| u32::from(completed) > **last)
                                .fold(8, |current, (_, &target)| current.min(target));
                            assert_eq!(progression.intended_downsampling(), intended);
                        }
                        let output = &update.output().outputs[0];
                        assert_eq!(output.layout.extent, plan.metadata.extent);
                        let actual = read_output(&backend, output);
                        let oriented = composed_oracle::orient(
                            &reference.pixels,
                            image.width as usize,
                            image.height as usize,
                            if orientation == OrientationPolicy::Keep {
                                1
                            } else {
                                image.orientation
                            },
                        );
                        let reference: Vec<_> = if let Some(channel) = scalar {
                            oriented
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|pixel| pixel[channel])
                                .collect()
                        } else {
                            oriented
                        };
                        let error = if let Some(bits) = integer_bits {
                            let mask = (1u32 << bits) - 1;
                            let sample_bytes = if bits <= 8 { 1 } else { 2 };
                            assert_eq!(actual.len(), reference.len() * sample_bytes);
                            actual
                                .chunks_exact(sample_bytes)
                                .zip(&reference)
                                .map(|(bytes, reference)| {
                                    let code = if sample_bytes == 1 {
                                        u32::from(bytes[0])
                                    } else {
                                        u32::from(u16::from_le_bytes(bytes.try_into().unwrap()))
                                    };
                                    assert!(code <= mask, "high padding bits remain zero");
                                    let expected = (reference.clamp(0.0, 1.0) * f64::from(mask))
                                        .round()
                                        as u32;
                                    let error = code.abs_diff(expected);
                                    assert!(
                                        error <= 1,
                                        "{name} {scalar:?}: native integer error {error}"
                                    );
                                    f64::from(error) / f64::from(mask)
                                })
                                .fold(0.0, f64::max)
                        } else {
                            composed_oracle::relative_error(
                                &composed_oracle::floats(&actual),
                                &reference,
                            )
                        };
                        eprintln!(
                            "{name} scalar{scalar:?} integer{integer_bits:?} {orientation:?} cap{cap} {:?}: {error}",
                            update.progression()
                        );
                        let limit = if name == "associated_vardct" && scalar.is_none() {
                            1e-3
                        } else {
                            3e-6
                        };
                        assert!(integer_bits.is_some() || error < limit);
                        if update.is_complete() {
                            finals.push(actual.clone());
                        }
                        held.push(owned(update.output()));
                        pixels.push(actual);
                    }
                    assert_eq!(pixels.len(), expected.len());
                    assert_eq!(session.frames_submitted(), plan.presentations.len());
                    let mut baseline = decoder
                        .open(&data, request.clone().with_progressive_output(false))
                        .unwrap();
                    for expected in finals {
                        let frame = baseline.next_frame().unwrap().unwrap();
                        assert_eq!(read_output(&backend, &frame.output().outputs[0]), expected);
                    }
                    assert!(baseline.next_frame().unwrap().is_none());
                    for (frame, expected) in held.iter().zip(&pixels) {
                        assert_eq!(read_output(&backend, &frame.outputs[0]), *expected);
                    }
                    if let Some(whole) = &whole {
                        assert_eq!(&pixels, whole);
                    } else {
                        whole = Some(pixels);
                    }
                    drop((session, baseline, held));
                    lifecycle::drain(&backend);
                    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
                    assert_eq!(
                        decoder.incremental_input_budget().snapshot().reserved_bytes,
                        0
                    );
                }
            }
        }
    }
}
