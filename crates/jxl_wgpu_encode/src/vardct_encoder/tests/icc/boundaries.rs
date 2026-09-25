use super::*;
use crate::{BackendError, GpuEncodeBackend, GpuEncodeJob, GpuFrameSource};
use jxl_gpu_protocol::icc::{IccDirection, IccRenderingIntent, IccSignature};

fn bounded(context: &WgpuContext, bytes: u64) -> WgpuContext {
    WgpuContext::with_memory_budget(
        Arc::new(context.device().clone()),
        Arc::new(context.queue().clone()),
        NonZeroU64::new(bytes).unwrap(),
    )
    .unwrap()
}

fn drain(context: &WgpuContext) {
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while context.memory_stats().reserved_bytes != 0 && std::time::Instant::now() < until {
        context.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn icc_admission_checks_profile_identity_intent_method_and_limits() {
    for gray in [false, true] {
        let profile = profile(gray);
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let mut config = config(&profile, transform);
            config.max_icc_profile_bytes = profile.bytes().len() as u64;
            let plan = VarDctColorPlan::new(&config).unwrap();
            let device_format = PixelFormat::icc_device(
                profile.clone(),
                ColorSample::F32,
                ColorStorage::Planar,
                false,
            )
            .unwrap();
            assert!(plan.matches_format(&device_format));
            config.max_icc_profile_bytes -= 1;
            assert!(matches!(
                VarDctColorPlan::new(&config),
                Err(EncodeError::IccLimit { .. })
            ));
            config.max_icc_profile_bytes += 1;
            config.color_options.rendering_intent =
                if profile.header().rendering_intent == IccRenderingIntent::Relative {
                    IccRenderingIntent::Absolute
                } else {
                    IccRenderingIntent::Relative
                };
            assert!(matches!(
                VarDctColorPlan::new(&config),
                Err(EncodeError::InvalidConfiguration(_))
            ));
            config.color_options.rendering_intent = profile.header().rendering_intent;
            config.sample_format = ColorSampleFormat::float(
                if gray {
                    ColorChannels::Rgb
                } else {
                    ColorChannels::Gray
                },
                32,
                8,
            )
            .unwrap();
            assert!(matches!(
                VarDctColorPlan::new(&config),
                Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
            ));
        }
        let bad = jxl_test_support::fixtures::icc::with_nonfinite_matrix_mpe(
            &profile,
            IccDirection::DeviceToPcs,
            IccRenderingIntent::Relative,
        );
        // Original coding only preserves device components. An unsupported selected CMM
        // method cannot silently become a fallback when XYB requires that method.
        assert!(VarDctColorPlan::new(&config(&bad, VarDctColorTransform::Original)).is_ok());
        assert!(matches!(
            VarDctColorPlan::new(&config(&bad, VarDctColorTransform::Xyb)),
            Err(EncodeError::Icc(_))
        ));
    }
}

#[test]
fn icc_jobs_enforce_exact_storage_budget_and_release_cancellation_and_nonfinite_inputs() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(25, 17);
    for gray in [false, true] {
        for transform in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            let config = config(&profile(gray), transform);
            let input = words(extent, config.sample_format.channels());
            let source = upload(&context, extent, &config, &input, true);
            for topology in [layouts::Topology::Map, layouts::Topology::Tiled] {
                let backend = topology.backend(&context, extent, &config);
                let plan = backend.memory_plan(&source).unwrap();
                assert!(plan.icc_profile_bytes > 0);
                assert_eq!(plan.icc_storage_bytes, 0); // image metadata is session-owned
                assert_eq!(plan.icc.is_some(), transform == VarDctColorTransform::Xyb);
                if let Some(icc) = plan.icc {
                    assert_eq!(icc.input_bytes, 32 * 24 * if gray { 1 } else { 3 } * 4);
                    assert_eq!(icc.linear_bytes, icc.input_bytes);
                    assert_eq!(
                        icc.total_bytes,
                        icc.input_bytes
                            + icc.linear_bytes
                            + icc.program_bytes
                            + icc.parameter_bytes
                    );
                    assert!(plan.source_validation_bytes >= 256);
                }
                for limit in [plan.owned_bytes_per_job - 1, plan.owned_bytes_per_job] {
                    let bounded = bounded(&context, limit);
                    let backend = topology.backend(&bounded, extent, &config);
                    let request = layouts::request(extent, &config);
                    let mut wrong = source.clone();
                    wrong.layout.format.color_spec = ColorSpecification::Default;
                    assert!(matches!(
                        backend.submit(&bounded, GpuFrameSource::Buffer(wrong), &request),
                        Err(EncodeError::Unsupported(UnsupportedFeature::InputFormat))
                    ));
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    let result =
                        backend.submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request);
                    if limit < plan.owned_bytes_per_job {
                        assert!(matches!(result, Err(EncodeError::MemoryBackpressure(_))));
                        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                        continue;
                    }
                    drop(result.unwrap());
                    drain(&bounded);
                    backend
                        .submit(&bounded, GpuFrameSource::Buffer(source.clone()), &request)
                        .unwrap()
                        .wait()
                        .unwrap();
                    assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut invalid = input.clone();
                        invalid[17][0] = value.to_bits();
                        let source = upload(&bounded, extent, &config, &invalid, true);
                        assert!(matches!(
                            backend
                                .submit(&bounded, GpuFrameSource::Buffer(source), &request)
                                .unwrap()
                                .wait(),
                            Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
                        ));
                        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                    }
                }
            }
        }
    }
}

#[test]
fn icc_still_and_sequence_headers_hold_one_reservation_until_assembly_or_drop() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(8, 8);
    for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
        let config = config(&profile(false), transform);
        let source = upload(
            &context,
            extent,
            &config,
            &words(extent, ColorChannels::Rgb),
            true,
        );
        let encoder =
            VarDctEncoder::new_with_config(context.clone(), VarDctStrategy::Dct8, config.clone())
                .unwrap();
        let plan = encoder.memory_plan(&source).unwrap();
        assert!(plan.icc_storage_bytes >= plan.icc_profile_bytes * 2);
        for limit in [plan.owned_bytes_per_job - 1, plan.owned_bytes_per_job] {
            let bounded = bounded(&context, limit);
            let encoder = VarDctEncoder::new_with_config(
                bounded.clone(),
                VarDctStrategy::Dct8,
                config.clone(),
            )
            .unwrap();
            if limit < plan.owned_bytes_per_job {
                assert!(matches!(
                    encoder.submit(source.clone()),
                    Err(EncodeError::MemoryBackpressure(_))
                ));
                assert_eq!(bounded.memory_stats().reserved_bytes, 0);
                continue;
            }
            drop(encoder.submit(source.clone()).unwrap());
            drain(&bounded);
            let job = encoder.submit_container(source.clone()).unwrap();
            // A future remains allocated after Ready; its header permit must not.
            let mut job = Box::pin(job);
            pollster::block_on(job.as_mut()).unwrap();
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            let mut invalid = words(extent, ColorChannels::Rgb);
            invalid[7][1] = f32::NAN.to_bits();
            let mut failed = Box::pin(
                encoder
                    .submit(upload(&bounded, extent, &config, &invalid, true))
                    .unwrap(),
            );
            assert!(matches!(
                pollster::block_on(failed.as_mut()),
                Err(EncodeError::Backend(BackendError::VarDctNonFiniteSource))
            ));
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            let descriptor =
                crate::ImageSequenceDescriptor::new(8, 8, crate::AnimationHeader::Still).unwrap();
            let mut session = encoder.begin_sequence(descriptor.clone()).unwrap();
            assert_eq!(
                bounded.memory_stats().reserved_bytes,
                plan.icc_storage_bytes
            );
            let job = session
                .submit_last_frame(source.clone(), Default::default())
                .unwrap();
            session.insert(job.wait().unwrap()).unwrap();
            assert_eq!(
                bounded.memory_stats().reserved_bytes,
                plan.icc_storage_bytes
            );
            session
                .finish_indexed_container(Default::default(), Default::default())
                .unwrap();
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
            drop(encoder.begin_sequence(descriptor).unwrap());
            assert_eq!(bounded.memory_stats().reserved_bytes, 0);
        }
        let bounded = bounded(&context, plan.icc_storage_bytes - 1);
        let encoder =
            VarDctEncoder::new_with_config(bounded.clone(), VarDctStrategy::Dct8, config).unwrap();
        assert!(matches!(
            encoder.begin_sequence(
                crate::ImageSequenceDescriptor::new(8, 8, crate::AnimationHeader::Still).unwrap()
            ),
            Err(EncodeError::MemoryBackpressure(_))
        ));
        assert_eq!(bounded.memory_stats().reserved_bytes, 0);
    }
}

#[test]
fn icc_finite_samples_with_overflowing_color_conversion_never_publish_packets() {
    let context = test_context().unwrap();
    let profile = jxl_test_support::fixtures::icc::with_matrix_mpe(
        &profile(false),
        IccDirection::DeviceToPcs,
        IccRenderingIntent::Relative,
    );
    let mut bytes = profile.bytes().to_vec();
    let offset = profile.tag(IccSignature(*b"D2B1")).unwrap().offset as usize + 24 + 12;
    bytes[offset..offset + 4].copy_from_slice(&1e20_f32.to_be_bytes());
    let profile = IccProfile::parse(bytes.into(), Default::default()).unwrap();
    let config = config(&profile, VarDctColorTransform::Xyb);
    let extent = Extent2d::new(25, 17);
    for topology in [layouts::Topology::Map, layouts::Topology::Tiled] {
        let backend = topology.backend(&context, extent, &config);
        let input = vec![[1e30_f32.to_bits(); 3]; extent.area().unwrap()];
        let source = upload(&context, extent, &config, &input, true);
        assert!(matches!(
            backend
                .submit(
                    &context,
                    GpuFrameSource::Buffer(source),
                    &layouts::request(extent, &config)
                )
                .unwrap()
                .wait(),
            Err(EncodeError::Backend(
                BackendError::VarDctColorConversionNonFinite
            ))
        ));
        assert_eq!(context.memory_stats().reserved_bytes, 0);
    }
}
