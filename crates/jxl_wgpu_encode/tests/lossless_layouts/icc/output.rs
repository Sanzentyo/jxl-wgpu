use super::*;

#[test]
fn embedded_icc_output_keeps_independent_color_reference_and_retained_alpha() {
    let rig = Rig::new();
    for gray in [false, true] {
        let fixture = jxl_test_support::fixtures::embedded_icc::Case {
            gray,
            encoding: jxl_gpu_bitstream::FrameEncoding::Modular,
            xyb: false,
        };
        let profile = profile(gray);
        let encoder = encoder(&rig, &profile, TREES[usize::from(gray)]);
        let case = Case {
            format: if gray {
                LosslessModularFormat::GrayAlpha
            } else {
                LosslessModularFormat::Rgba
            },
            bits: 32,
            kind: SampleKind::Float,
            storage: Storage::Planar,
            reversed: true,
            byte_order: ByteOrder::Big,
            shifted: true,
        };
        let samples = fixture.input();
        let mut source = upload(&rig.context, &case, Extent2d::new(17, 9), &samples, 4099);
        attach(&mut source, &profile, gray);
        let encoded = encoder.encode_container(source).unwrap();
        check_frame_samples(&encoded, &[&samples], &case, &original(&encoded));
        let expected = fixture.linear_reference();
        let native = extra_channels::libjxl_output(
            &encoded,
            &["--linear", "--preserve-alpha", "--keep-orientation"],
        )
        .expect("required native ICC color oracle");
        // libjxl's native CMM and the scalar oracle have different curve/matrix precision.
        // Compare its output bit-for-bit to the unchanged native-encoded fixture, while the
        // GPU keeps the existing independent scalar bound below.
        let native_fixture = extra_channels::libjxl_output(
            &fixture.bytes(),
            &["--linear", "--preserve-alpha", "--keep-orientation"],
        )
        .expect("required native fixture color oracle");
        assert_eq!(
            native
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            native_fixture
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let mut color = jxl_wgpu_decode::vardct_rgb8_format().color_spec;
        let ColorSpecification::Defined(ref mut spec) = color else {
            panic!("enumerated linear target")
        };
        spec.transfer = jxl_gpu_formats::TransferFunction::Linear;
        let mut baseline = None;
        for (decoder, fragmented) in rig.decoders.iter().zip([false, true]) {
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                color.clone(),
            ))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
            let mut session = if fragmented {
                open_fragmented(decoder, &encoded, request)
            } else {
                decoder.open(&encoded, request).unwrap()
            };
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            assert!(
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .is_none()
            );
            drop(session);
            let actual = read(&rig.backend, &frame.output().outputs[0]);
            assert_eq!(actual.len(), expected.len());
            for (index, (&word, &expected)) in actual.iter().zip(&expected).enumerate() {
                let value = f32::from_bits(word);
                if index % 4 == 3 {
                    let channels = case.format.channel_count() as usize;
                    assert_eq!(word, samples[(index / 4 + 1) * channels - 1]);
                    assert_eq!(word, native[index].to_bits());
                } else {
                    // The existing embedded-ICC RGB/Gray matrix/TRC corpus uses this bound
                    // against its independent scalar/CMS reference; no tolerance is widened.
                    assert!(
                        value.is_finite() && (value - expected).abs() <= 2e-4,
                        "gray={gray}, component {index}: {value} vs {expected}"
                    );
                }
            }
            if let Some(baseline) = &baseline {
                assert_eq!(&actual, baseline);
            } else {
                baseline = Some(actual);
            }
            drop(frame);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
    }
}
