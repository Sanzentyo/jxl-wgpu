use jxl_gpu_bitstream::SampleBitDepth;
use jxl_gpu_formats::{Channel, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_test_support::{gpu::planes, oracles::extra_channels};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, SpotColorPolicy,
    WgpuDecodeEngine,
};
use std::num::NonZeroU64;

mod admission;
mod corpus;
mod reference;
mod vardct;
use reference::Arithmetic;

#[test]
fn independent_intervals_agree_with_rust_and_native_control_reconstruction() {
    let mut rust_cases = 0;
    let mut native_cases = 0;
    for (case, custom) in oracle_cases() {
        eprintln!("oracle {} custom={custom}", case.name);
        let (data, _, expected) = case.load(Arithmetic::Rust, custom);
        // jxl-render 0.12.4 PaddedGrid::mirror_edges_padding reads padding as
        // input when a coded axis has fewer than two samples. Those thin cases
        // retain the independent repeated-mirror oracle and native 8x controls.
        if case.width.div_ceil(case.extra_factor as usize) >= 2
            && case.height.div_ceil(case.extra_factor as usize) >= 2
        {
            rust_cases += 1;
            let mut image = jxl_oxide::JxlImage::read_with_defaults(data.as_slice()).unwrap();
            image.set_render_spot_color(false);
            let rendered = image.render_frame(0).unwrap();
            let pixels = rendered.image_all_channels();
            assert_eq!(pixels.channels(), 7);
            assert_eq!(pixels.buf().len(), case.width * case.height * 7);
            for (index, pixel) in pixels.buf().as_chunks::<7>().0.iter().enumerate() {
                for (channel, &value) in pixel.iter().enumerate() {
                    expected[channel].samples[index].check(
                        value,
                        &format!("{} jxl-oxide channel {channel}", case.name),
                        index,
                    );
                }
            }
        }
        if case.extra_factor == 8 {
            native_cases += 1;
            let (_, _, expected) = case.load(Arithmetic::Native, custom);
            let (color, extras) = extra_channels::libjxl_planes(&data, case.width * case.height, 4)
                .expect("native control oracle required");
            for (index, pixel) in color.as_chunks::<4>().0.iter().enumerate() {
                for (channel, &value) in pixel.iter().enumerate() {
                    let plane = if channel == 3 { 4 } else { channel };
                    expected[plane].samples[index].check(
                        value,
                        &format!("{} libjxl color {channel}", case.name),
                        index,
                    );
                }
            }
            for (channel, values) in extras.iter().enumerate() {
                for (index, &value) in values.iter().enumerate() {
                    expected[channel + 3].samples[index].check(
                        value,
                        &format!("{} libjxl extra {channel}", case.name),
                        index,
                    );
                }
            }
        }
    }
    assert_eq!((rust_cases, native_cases), (35, 21));
}

#[test]
fn gpu_resampling_preserves_components_transport_and_output_lifetimes() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for (case, custom) in oracle_cases() {
        eprintln!("GPU {} custom={custom}", case.name);
        let (data, inventory, expected) = case.load(Arithmetic::Wgsl, custom);
        for selected in 0..5 {
            let request = if selected == 4 {
                GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_spot_color_policy(SpotColorPolicy::Preserve)
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            } else {
                GpuOutputRequest::numeric(
                    PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                    if matches!(
                        inventory.image_header.extra_channels[selected].bit_depth,
                        SampleBitDepth::Float { .. }
                    ) {
                        NumericSampleMapping::NativeFloat
                    } else {
                        NumericSampleMapping::NormalizedUnsigned
                    },
                )
                .unwrap()
                .with_extra_channel(selected as u32)
                .unwrap()
            };
            let mut baseline = None;
            for limit in [None, NonZeroU64::new(256)] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if let Some(limit) = limit {
                    engine = engine.with_stream_window_limit(limit);
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = if limit.is_some() {
                    planes::open_fragmented(&decoder, &data, request.clone())
                } else {
                    decoder.open(&data, request.clone()).unwrap()
                };
                assert_eq!(
                    session.metadata().extra_channels,
                    inventory.image_header.extra_channels
                );
                let frame = pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap();
                assert!(
                    pollster::block_on(session.next_frame_async())
                        .unwrap()
                        .is_none()
                );
                let pixels = planes::read(&backend, &frame.output().outputs[0]);
                drop(session);
                assert_eq!(planes::read(&backend, &frame.output().outputs[0]), pixels);
                let channels = if selected == 4 { 4 } else { 1 };
                assert_eq!(pixels.len(), case.width * case.height * channels);
                for (index, pixel) in pixels.chunks_exact(channels).enumerate() {
                    for (channel, &word) in pixel.iter().enumerate() {
                        let plane = if selected < 4 {
                            selected + 3
                        } else if channel == 3 {
                            4
                        } else {
                            channel
                        };
                        expected[plane].samples[index].check(
                            f32::from_bits(word),
                            &case.name,
                            index,
                        );
                    }
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&pixels, baseline);
                }
                baseline = Some(pixels);
                drop(frame);
                assert_eq!(
                    backend.transient_memory_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}

fn oracle_cases() -> impl Iterator<Item = (corpus::Case, bool)> {
    corpus::cases().into_iter().map(|case| (case, false)).chain(
        corpus::cases()
            .into_iter()
            .filter(|case| (case.width, case.height) == (129, 97))
            .map(|case| (case, true)),
    )
}
