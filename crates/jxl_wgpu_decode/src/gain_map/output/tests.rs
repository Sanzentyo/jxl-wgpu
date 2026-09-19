use super::*;
use crate::{GpuDecoder, gain_map::GainMapRendering};
use jxl_gpu_protocol::{RgbColorSpace, TransferFunction, icc::IccProfile};
use jxl_test_support::gpu::planes;

#[test]
fn preserve_alpha_uses_the_primary_declaration_after_straight_gain_math() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(root.join("test-data/gain_map/generated/case_0.jxl")).unwrap();
    let mut image = jxl_gpu_bitstream::parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let profile = IccProfile::parse(
        std::fs::read(root.join("../jxl_wgpu/test-data/icc/gamma_v4.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgba,
        false,
        ColorSpecification::Icc(profile),
    ))
    .unwrap()
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve);
    let mut straight = Vec::new();
    for associated in [false, true] {
        image.extra_channels[0].channel_type = ExtraChannelTypeInventory::Alpha { associated };
        let plan = Plan::new(
            &backend,
            &request,
            &image,
            RgbColorEncoding::LINEAR_BT709,
            &GainMapMetadata::default(),
            1.0,
            DisplayIntensity::new(203.0).unwrap(),
        )
        .unwrap();
        // The incoming surface has already been unassociated for gain application. Only the
        // original primary declaration decides what Preserve means in the final device space.
        let frame = pollster::block_on(
            decoder.decode_alternate(
                &bytes,
                GpuOutputRequest::color(plan.gain.layout().format.clone())
                    .unwrap()
                    .with_orientation_policy(OrientationPolicy::Keep)
                    .with_alpha_output_policy(AlphaOutputPolicy::Unassociated),
                Default::default(),
            ),
        )
        .unwrap();
        let frame = pollster::block_on(plan.finish(&backend, frame)).unwrap();
        let words = planes::read(&backend, &frame.output().outputs[0]);
        if associated {
            let mut changed = 0;
            for (output, source) in words
                .as_chunks::<4>()
                .0
                .iter()
                .zip(straight.as_chunks::<4>().0)
            {
                assert_eq!(output[3], source[3]);
                let alpha = f64::from(f32::from_bits(source[3])).max(1.0 / 67108864.0);
                for c in 0..3 {
                    let value = f64::from(f32::from_bits(output[c]));
                    let expected = f64::from(f32::from_bits(source[c])) * alpha;
                    assert!((value - expected).abs() <= 2.0 * f64::from(f32::EPSILON));
                    changed += usize::from(output[c] != source[c]);
                }
            }
            assert!(changed > 0);
        } else {
            straight = words;
        }
    }
    assert_eq!(backend.transient_memory_stats().reserved_bytes, 0);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}

#[test]
fn icc_after_gain_rejects_and_cancels_without_retaining_the_intermediate() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let memory = backend.transient_memory_budget();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(root.join("test-data/gain_map/generated/case_0.jxl")).unwrap();
    let image = jxl_gpu_bitstream::parse(&bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    let profile = IccProfile::parse(
        std::fs::read(root.join("../jxl_wgpu/test-data/icc/lut/ab_channels15.icc"))
            .unwrap()
            .into(),
        Default::default(),
    )
    .unwrap();
    let request = GpuOutputRequest::color(
        PixelFormat::icc_device(
            profile,
            jxl_gpu_formats::ColorSample::F32,
            jxl_gpu_formats::ColorStorage::Planar,
            true,
        )
        .unwrap(),
    )
    .unwrap();
    let working = RgbColorEncoding {
        space: RgbColorSpace::Bt709,
        transfer: TransferFunction::Linear,
    };
    for outcome in 0..3 {
        let plan = Plan::new(
            &backend,
            &request,
            &image,
            working,
            &GainMapMetadata::default(),
            1.0,
            DisplayIntensity::new(203.0).unwrap(),
        )
        .unwrap();
        assert_eq!(memory.snapshot().reserved_bytes, 0);
        let intermediate_request = GpuOutputRequest::color(plan.gain.layout().format.clone())
            .unwrap()
            .with_orientation_policy(OrientationPolicy::Keep)
            .with_alpha_output_policy(AlphaOutputPolicy::Unassociated);
        let frame = pollster::block_on(decoder.decode_gain_map(
            &bytes,
            intermediate_request,
            GainMapRendering::default(),
            Default::default(),
        ))
        .unwrap();
        assert_eq!(&frame.output().outputs[0].layout, plan.gain.layout());
        assert_eq!(
            memory.snapshot().reserved_bytes,
            frame.output().outputs[0].buffer.size()
        );
        match outcome {
            0 => {
                let output = pollster::block_on(plan.finish(&backend, frame)).unwrap();
                drop(plan);
                assert_eq!(
                    memory.snapshot().reserved_bytes,
                    output.output().outputs[0].buffer.size()
                );
                assert!(
                    planes::read(&backend, &output.output().outputs[0])
                        .into_iter()
                        .all(|word| f32::from_bits(word).is_finite())
                );
                drop(output);
            }
            1 => {
                let blocker = memory
                    .try_reserve(memory.snapshot().available_bytes)
                    .unwrap();
                let result = pollster::block_on(plan.finish(&backend, frame));
                assert!(matches!(result, Err(crate::Error::MemoryBackpressure(_))));
                drop((blocker, plan));
            }
            2 => {
                let icc = plan.icc.as_ref().unwrap();
                let source = Surface {
                    buffer: frame.output().outputs[0].buffer.clone(),
                    layout: Arc::clone(&icc.source),
                    encoding: icc.encoding.clone(),
                };
                // Cancel the submitted ICC operation without polling it. Drop the gain frame,
                // its slot and the plan as well; the completion must own their GPU resources.
                let work = icc.presentation.pack(&backend, &source).unwrap();
                drop((work, source, frame, plan));
            }
            _ => unreachable!(),
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while memory.snapshot().reserved_bytes != 0 && std::time::Instant::now() < deadline {
            backend.device().poll(wgpu::PollType::Poll).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(memory.snapshot().reserved_bytes, 0);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
