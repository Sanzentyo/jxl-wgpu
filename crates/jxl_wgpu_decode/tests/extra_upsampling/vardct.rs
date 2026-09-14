use super::*;
use reference::{Plane, Sample};
use std::path::Path;

fn expected(
    width: usize,
    height: usize,
    scale: u32,
    weights: &jxl_gpu_bitstream::UpsamplingWeightsInventory,
    arithmetic: Arithmetic,
) -> Vec<Plane> {
    (0..4)
        .map(|channel| {
            let samples = (0..height / 8)
                .flat_map(|y| {
                    (0..width / 8).map(move |x| {
                        let code = ((x * 11 + y * 17 + channel * 53) % 97) as i32;
                        Sample {
                            value: f64::from(if channel == 1 { code % 17 } else { code - 43 })
                                / 16.0,
                            error: 0.0,
                        }
                    })
                })
                .collect();
            Plane {
                width: width / 8,
                height: height / 8,
                samples,
            }
            .reconstruct(
                8 * scale,
                width * scale as usize,
                height * scale as usize,
                weights,
                arithmetic,
            )
        })
        .collect()
}

#[test]
fn native_vardct_entropy_reconstructs_extended_extra_grids() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for (width, height) in [(24, 16), (272, 24)] {
        let base = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "test-data/extra_upsampling/vardct/vardct_{width}x{height}.jxl"
        )))
        .unwrap();
        for custom in [false, true] {
            let base = if custom {
                jxl_test_support::corpus::with_custom_upsampling_weights(&base)
            } else {
                base.clone()
            };
            for scale in [1, 2, 4, 8] {
                let data = jxl_test_support::fixtures::resampling::scale_presentation(&base, scale);
                let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
                    .unwrap()
                    .codestream_inventory(Default::default())
                    .unwrap();
                let context = format!("VarDCT {width}x{height} scale={scale} custom={custom}");
                eprintln!("{context}");
                let frame = &inventory.frames[0];
                assert_eq!(frame.encoding, jxl_gpu_bitstream::FrameEncoding::VarDct);
                assert_eq!(frame.upsampling, scale);
                assert_eq!(frame.extra_channel_upsampling, [8 * scale; 4]);
                assert_eq!(
                    (
                        inventory.image_header.width as usize,
                        inventory.image_header.height as usize
                    ),
                    (width * scale as usize, height * scale as usize)
                );
                let cpu = expected(
                    width,
                    height,
                    scale,
                    &inventory.image_header.upsampling_weights,
                    Arithmetic::Rust,
                );
                let mut image = jxl_oxide::JxlImage::read_with_defaults(data.as_slice()).unwrap();
                image.set_render_spot_color(false);
                let rendered = image.render_frame(0).unwrap();
                let pixels = rendered.image_all_channels();
                assert_eq!(pixels.channels(), 7);
                for (index, pixel) in pixels.buf().as_chunks::<7>().0.iter().enumerate() {
                    for channel in 0..4 {
                        cpu[channel].samples[index].check(
                            pixel[channel + 3],
                            &format!("{context} rust extra {channel}"),
                            index,
                        );
                    }
                }
                if scale == 1 {
                    let (_, extras) = extra_channels::libjxl_planes(&data, width * height, 4)
                        .expect("native VarDCT control oracle required");
                    for (channel, values) in extras.iter().enumerate() {
                        for (index, &value) in values.iter().enumerate() {
                            cpu[channel].samples[index].check(
                                value,
                                &format!("{context} native extra {channel}"),
                                index,
                            );
                        }
                    }
                }
                let expected = expected(
                    width,
                    height,
                    scale,
                    &inventory.image_header.upsampling_weights,
                    Arithmetic::Wgsl,
                );
                for channel in 0..4 {
                    let request = GpuOutputRequest::numeric(
                        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                        NumericSampleMapping::NativeFloat,
                    )
                    .unwrap()
                    .with_extra_channel(channel)
                    .unwrap();
                    let mut baseline = None;
                    for bounded in [false, true] {
                        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                        if bounded {
                            engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
                        }
                        let decoder = GpuDecoder::new(engine);
                        let mut session = if bounded {
                            planes::open_fragmented(&decoder, &data, request.clone())
                        } else {
                            decoder.open(&data, request.clone()).unwrap()
                        };
                        let frame = pollster::block_on(session.next_frame_async())
                            .unwrap()
                            .unwrap();
                        assert!(
                            pollster::block_on(session.next_frame_async())
                                .unwrap()
                                .is_none()
                        );
                        let pixels = planes::read(&backend, &frame.output().outputs[0]);
                        assert_eq!(pixels.len(), expected[channel as usize].samples.len());
                        for (index, &word) in pixels.iter().enumerate() {
                            expected[channel as usize].samples[index].check(
                                f32::from_bits(word),
                                &format!("{context} GPU extra {channel}"),
                                index,
                            );
                        }
                        if let Some(baseline) = &baseline {
                            assert_eq!(&pixels, baseline);
                        }
                        baseline = Some(pixels.clone());
                        drop(session);
                        assert_eq!(planes::read(&backend, &frame.output().outputs[0]), pixels);
                        drop(frame);
                        assert_eq!(
                            backend.transient_memory_budget().snapshot().reserved_bytes,
                            0
                        );
                    }
                }
            }
        }
    }
}
