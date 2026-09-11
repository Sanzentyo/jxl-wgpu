use super::*;

type Held = (jxl_wgpu::GpuImageOutput, Vec<u8>);

pub(super) fn hold(backend: &WgpuBackend, output: &jxl_wgpu::GpuImageOutput) -> Held {
    (
        jxl_wgpu::GpuImageOutput {
            id: output.id,
            layout: output.layout.clone(),
            buffer: output.buffer.clone(),
        },
        read_output(backend, output),
    )
}

pub(super) fn released(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    held: Vec<Held>,
) {
    retired(backend);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    assert_eq!(
        decoder.engine().in_flight_memory_stats().reserved_bytes,
        held.iter().map(|(o, _)| o.buffer.size()).sum::<u64>()
    );
    for (output, bytes) in &held {
        assert_eq!(read_output(backend, output), *bytes);
    }
    drop(held);
    retired(backend);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

pub(super) fn integer_error(bytes: &[u8], values: &[f64], bits: u32) {
    let stride = if bits <= 16 { 2 } else { 4 };
    assert_eq!(bytes.len(), values.len() * stride);
    let maximum = (1_u32 << bits) - 1;
    for (word, &value) in bytes.chunks_exact(stride).zip(values) {
        let actual = if stride == 2 {
            u32::from(u16::from_le_bytes(word.try_into().unwrap()))
        } else {
            u32::from_le_bytes(word.try_into().unwrap())
        };
        let expected = (value.clamp(0.0, 1.0) * f64::from(maximum)).round() as u32;
        assert!(actual <= maximum, "native high padding bits changed");
        assert!(
            actual.abs_diff(expected) <= 1,
            "native {bits}-bit sample: {actual} != {expected}"
        );
    }
}

#[test]
fn lf_native_wide_extras_and_cancellation_keep_exact_ownership() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases().into_iter().filter(|(name, _)| {
        name.starts_with("integer_associated") || name.starts_with("resampled_associated")
    }) {
        for composed in [false, true] {
            let Some(check) = Check::new(&name, levels, composed) else {
                return;
            };
            let decoder = GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            );
            for extra in 0..2 {
                let SampleBitDepth::Integer { bits_per_sample } =
                    check.image.extra_channels[extra].bit_depth
                else {
                    unreachable!()
                };
                let request = GpuOutputRequest::numeric(
                    jxl_wgpu_decode::native_modular_pixel_format(
                        jxl_wgpu_decode::ModularChannels::Gray,
                        bits_per_sample as u8,
                    )
                    .unwrap(),
                    NumericSampleMapping::NativeUnsigned,
                )
                .unwrap()
                .with_extra_channel(extra as u32)
                .unwrap()
                .with_max_frame_slots(NonZeroUsize::new(1).unwrap());
                let expected = |planes| {
                    check.values(
                        planes,
                        Some(extra as u32),
                        OrientationPolicy::Apply,
                        AlphaOutputPolicy::Preserve,
                    )
                };
                let mut baseline = decoder.open(&check.encoded, request.clone()).unwrap();
                let frame = baseline.next_frame().unwrap().unwrap();
                let final_bytes = read_output(&backend, &frame.output().outputs[0]);
                integer_error(&final_bytes, &expected(&check.native), bits_per_sample);
                drop(frame);
                drop(baseline);
                for boundary in 0..=usize::from(levels) {
                    let mut session = incremental(
                        &decoder,
                        &check.encoded,
                        request.clone().with_progressive_output(true),
                    );
                    session.prefetch(NonZeroUsize::new(1).unwrap()).unwrap();
                    let mut held = Vec::new();
                    for index in 0..boundary {
                        let update = pollster::block_on(session.next_update_async())
                            .unwrap()
                            .unwrap();
                        assert!(
                            matches!(update.progression(), Some(FrameProgression::LowFrequency { physical_frame_index, .. }) if physical_frame_index as usize == index)
                        );
                        let output = hold(&backend, &update.output().outputs[0]);
                        integer_error(
                            &output.1,
                            &expected(&check.previews[index]),
                            bits_per_sample,
                        );
                        held.push(output);
                    }
                    // Drain once after every LF image; all earlier boundaries cancel with
                    // output leases still alive, including queued work before the first image.
                    if boundary == usize::from(levels) {
                        let frame = pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap();
                        assert_eq!(
                            read_output(&backend, &frame.output().outputs[0]),
                            final_bytes
                        );
                    }
                    drop(session);
                    released(&backend, &decoder, held);
                }
            }
        }
    }
}

#[test]
fn lf_deep_and_floating_corruption_cannot_publish_unvalidated_images() {
    let Some(backend) = backend() else { return };
    for (name, levels) in cases()
        .into_iter()
        .filter(|(name, _)| name.starts_with("integer_associated") || name.starts_with("floating_"))
    {
        reject_corruption(&backend, &name, levels, &fixture(&name, ".composed"));
    }
}

pub(super) fn reject_corruption(backend: &WgpuBackend, name: &str, levels: u8, encoded: &[u8]) {
    let inventory = parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    for damaged in [
        0,
        usize::from(levels) - 1,
        usize::from(levels),
        usize::from(levels) + 1,
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>()
    {
        let mut corrupt = encoded[..inventory.frames[0].header_bits.offset as usize / 8].to_vec();
        for (index, frame) in inventory.frames.iter().enumerate() {
            let mut packets = payloads(encoded, frame);
            if index == damaged {
                let end = global_end(encoded, frame) / 8 - frame.sections[0].bytes.offset as usize;
                packets[0].truncate(end - 1);
            }
            corrupt.extend(reassemble(encoded, frame, packets));
        }
        parse(&corrupt, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let decoder = GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        );
        for extra in [None, Some(1)] {
            let mut session = incremental(
                &decoder,
                &corrupt,
                request(&inventory.image_header, extra).with_progressive_output(true),
            );
            let mut held = Vec::new();
            loop {
                match pollster::block_on(session.next_update_async()) {
                    Ok(Some(update)) => {
                        assert!(matches!(
                            update.progression(),
                            Some(FrameProgression::LowFrequency { .. })
                        ));
                        held.push(hold(backend, &update.output().outputs[0]));
                    }
                    Err(error) => {
                        assert!(
                            matches!(
                                error,
                                Error::ModularEntropyRejected { .. } | Error::VarDct(_)
                            ),
                            "{name} damaged={damaged}: {error:?}"
                        );
                        break;
                    }
                    Ok(None) => panic!("{name} damaged={damaged}: corrupt entropy accepted"),
                }
            }
            assert_eq!(
                held.len(),
                if damaged == usize::from(levels) + 1 {
                    usize::from(levels)
                } else {
                    0
                }
            );
            drop(session);
            released(backend, &decoder, held);
        }
    }
}
