use super::*;

#[test]
fn color_declarations_keep_exact_admission_and_cancelled_source_release() {
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::Rgba,
        bits: 31,
        kind: SampleKind::Unsigned,
        storage: Storage::Planar,
        reversed: true,
        byte_order: ByteOrder::Big,
        shifted: true,
    };
    let color = spec(ColorSpace::Bt2020, TransferFunction::Pq);
    let options = ImageOptions {
        rendering_intent: IccRenderingIntent::Absolute,
        intensity_target: FiniteF16::from_bits(0x63d0).unwrap(),
        ..Default::default()
    };
    let encoder = LosslessModularEncoder::new(rig.context.clone())
        .with_image_options(options)
        .unwrap();
    for extent in [Extent2d::new(257, 9), Extent2d::new(16_384, 1)] {
        let expected = case.samples(extent);
        let plain = upload(&rig.context, &case, extent, &expected, 4099);
        let input = colored(plain.clone(), color);
        let plan = encoder.memory_plan(&input).unwrap();
        assert_eq!(plan, encoder.memory_plan(&plain).unwrap());
        assert_eq!(plan.streaming, extent.width == 16_384);
        let limited = |bytes| {
            LosslessModularEncoder::new(
                WgpuContext::with_memory_budget(
                    Arc::new(rig.context.device().clone()),
                    Arc::new(rig.context.queue().clone()),
                    NonZeroU64::new(bytes).unwrap(),
                )
                .unwrap(),
            )
            .with_image_options(options)
            .unwrap()
        };
        let short = limited(plan.owned_bytes_per_job - 1);
        let failure = match short.submit(input.clone()) {
            Ok(job) => pollster::block_on(job).unwrap_err(),
            Err(error) => error,
        };
        assert!(matches!(failure, EncodeError::MemoryBackpressure(_)));
        assert_eq!(short.buffer_pool_stats().allocation_misses, 0);
        assert_eq!(short.in_flight_memory_stats().reserved_bytes, 0);
        let exact = limited(plan.owned_bytes_per_job);
        let mut cancelled = input.clone();
        cancelled.buffer = Arc::new(input.buffer.as_ref().clone());
        let owner = Arc::downgrade(&cancelled.buffer);
        drop(exact.submit(cancelled).unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while owner.upgrade().is_some()
            || exact.in_flight_memory_stats().reserved_bytes != 0
            || exact.buffer_pool_stats().leased_buffer_sets != 0
        {
            assert!(
                std::time::Instant::now() < deadline,
                "cancelled color job retained source or budget"
            );
            rig.context.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let encoded = exact.encode(input.clone()).unwrap();
        assert_eq!(
            encoded,
            pollster::block_on(exact.submit(input).unwrap()).unwrap()
        );
        assert_eq!(exact.in_flight_memory_stats().reserved_bytes, 0);
        assert!(exact.buffer_pool_stats().reuse_hits > 0);
        check_oracles(&encoded, &expected, &case);
        check_numeric(&rig, &encoded, &[expected], &case);
    }
}

#[test]
fn unsupported_source_metadata_never_allocates_a_gpu_job() {
    let rig = Rig::new();
    let case = Case {
        format: LosslessModularFormat::Rgb,
        bits: 8,
        kind: SampleKind::Unsigned,
        storage: Storage::Packed,
        reversed: false,
        byte_order: ByteOrder::Native,
        shifted: false,
    };
    let extent = Extent2d::new(17, 3);
    let input = upload(&rig.context, &case, extent, &case.samples(extent), 0);
    let encoder = LosslessModularEncoder::new(rig.context.clone());
    let base = spec(ColorSpace::Bt709, TransferFunction::Srgb);
    for color in [
        ColorSpec {
            range: ColorRange::Limited,
            ..base
        },
        ColorSpec {
            encoding: YcbcrEncoding::Bt709,
            ..base
        },
        ColorSpec {
            transfer: TransferFunction::Bt2020,
            ..base
        },
        ColorSpec {
            space: ColorSpace::Sensor,
            ..base
        },
        ColorSpec {
            space: ColorSpace::CustomRgb(RgbChromaticities {
                red: RgbChromaticities::BT709.green,
                ..RgbChromaticities::BT709
            }),
            ..base
        },
    ] {
        assert!(matches!(
            encoder.submit(colored(input.clone(), color)),
            Err(EncodeError::Unsupported(_))
        ));
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        assert_eq!(encoder.buffer_pool_stats().allocation_misses, 0);
    }
    for bits in [0, 0x8000, 0xbc00] {
        let result =
            LosslessModularEncoder::new(rig.context.clone()).with_image_options(ImageOptions {
                intensity_target: FiniteF16::from_bits(bits).unwrap(),
                ..Default::default()
            });
        assert!(matches!(result, Err(EncodeError::InvalidConfiguration(_))));
    }
}

#[test]
fn default_srgb_keeps_canonical_bytes_and_custom_gray_omits_the_private_shortcut() {
    let rig = Rig::new();
    let encoder = LosslessModularEncoder::new(rig.context.clone());
    for format in [
        LosslessModularFormat::Gray,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ] {
        let case = Case {
            format,
            bits: 8,
            kind: SampleKind::Unsigned,
            storage: Storage::Packed,
            reversed: false,
            byte_order: ByteOrder::Native,
            shifted: false,
        };
        let extent = Extent2d::new(17, 3);
        let expected = case.samples(extent);
        let plain = upload(&rig.context, &case, extent, &expected, 0);
        let default = encoder.encode_container(plain.clone()).unwrap();
        let explicit = encoder
            .encode_container(colored(
                plain.clone(),
                spec(ColorSpace::Bt709, TransferFunction::Srgb),
            ))
            .unwrap();
        assert_eq!(default, explicit);
        let custom = encoder
            .encode_container(colored(
                plain,
                spec(ColorSpace::Bt709, TransferFunction::Linear),
            ))
            .unwrap();
        let file = jxl_gpu_bitstream::parse(&custom, Default::default()).unwrap();
        assert!(
            file.boxes_of_type(jxl_gpu_bitstream::ACCELERATION_INDEX_BOX_TYPE)
                .next()
                .is_none()
        );
        check_oracles(&custom, &expected, &case);
        check_numeric(&rig, &custom, &[expected], &case);
    }
}
