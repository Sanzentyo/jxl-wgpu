use super::display::{check_metadata, encode, half};
use super::*;
use jxl_gpu_formats::{ColorSpace, ColorSpecification, RgbChannelOrder, TransferFunction};
use jxl_gpu_protocol::LuminanceRange;
use jxl_test_support::oracles::{hdr, icc_profile::IccProfileOracle, tone_mapping::Mapping};

#[test]
fn encoded_light_metadata_drives_explicit_gpu_tone_mapping_with_preserved_alpha() {
    let rig = Rig::new();
    let native = IccProfileOracle::compile();
    let extent = Extent2d::new(8, 8);
    let samples = ColorSampleFormat::float(ColorChannels::Rgb, 32, 8).unwrap();
    let ColorSpecification::Defined(mut color) = jxl_wgpu_decode::vardct_rgb8_format().color_spec
    else {
        unreachable!()
    };
    color.transfer = TransferFunction::Linear;
    let mut input_format = format(samples, true);
    input_format.color_spec = ColorSpecification::Defined(color);
    // Straddle both protected intervals and exercise shadows, shoulder, black and highlights.
    let words = (0..64)
        .flat_map(|p| {
            let light = [0.0f32, 0.005, 0.01, 0.02, 0.05, 0.1, 0.4, 1.0][p % 8];
            [light * 0.8, light, light * 1.1, (p % 5) as f32 / 4.0].map(f32::to_bits)
        })
        .collect::<Vec<_>>();
    let (layout, bytes) = Packing {
        storage: Storage::Planar,
        reversed: true,
        shifted: true,
    }
    .pack(input_format, extent, &words, 4099);
    let source = BufferImageSource::new(
        Arc::new(
            rig.context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("independent tone-mapping input"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap();
    for topology in [0, 1] {
        for (threshold, protected) in [
            (ToneMappingThreshold::default(), 0.0),
            (ToneMappingThreshold::DisplayFraction(half(0x3000)), 10.0),
            (ToneMappingThreshold::AbsoluteNits(half(0x4d00)), 20.0),
        ] {
            let options = ImageOptions {
                intensity_target: half(0x63d0), // 1000 nit
                min_nits: half(0x2c00),         // 1/16 nit
                linear_below: threshold,
                intrinsic_size: Some(IntrinsicSize::new(1 << 31, 1 << 30).unwrap()),
                ..Default::default()
            };
            let (bytes, _) = encode(
                &rig,
                source.clone(),
                VarDctConfig {
                    sample_format: samples,
                    alpha: Some(AlphaAssociation::Unassociated),
                    color_transform: VarDctColorTransform::Original,
                    source_color: ColorSpecification::Defined(color),
                    image_options: options,
                    ..Default::default()
                },
                topology,
            );
            check_metadata(&bytes, &native, extent, options);
            // These are independently decoded original linear samples, not GPU readback or
            // production conversion coefficients. Keep the established codec interval.
            let reference =
                extra_channels::libjxl_output(&bytes, &["--original", "--preserve-alpha"]).unwrap();
            assert_eq!(reference.len(), 64 * 5);
            let mapping = Mapping {
                source: [0.0625, 1000.0],
                target: [0.03125, 80.0],
                protected,
            };
            let weights = hdr::luminance(ColorSpace::Bt709);
            let expected = reference[..64 * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| {
                    let linear = [pixel[0], pixel[1], pixel[2]].map(f64::from);
                    let range = linear.map(|v| {
                        let radius = 2e-4 * (1.0 + v.abs());
                        [v - radius, v + radius]
                    });
                    (
                        mapping.apply(linear, weights, [1.0; 3]),
                        mapping.interval(range, weights, [1.0; 3]),
                        pixel[3],
                    )
                })
                .collect::<Vec<_>>();
            let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
                RgbChannelOrder::Rgba,
                false,
                ColorSpecification::Defined(color),
            ))
            .unwrap()
            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            .with_tone_mapping(LuminanceRange::new(0.03125, 80.0).unwrap());
            let mut baseline = None;
            for fragmented in [false, true] {
                let mut session = if fragmented {
                    open_fragmented(&rig.decoder, &bytes, request.clone())
                } else {
                    rig.decoder.open(&bytes, request.clone()).unwrap()
                };
                let frame = session.next_frame().unwrap().unwrap();
                assert_eq!(frame.output().outputs[0].layout.extent, extent);
                assert!(session.next_frame().unwrap().is_none());
                drop(session);
                let actual = read_bytes(&rig.gpu, &frame.output().outputs[0]);
                let values = extra_channels::floats(&actual);
                for (p, (pixel, (center, bounds, alpha))) in
                    values.as_chunks::<4>().0.iter().zip(&expected).enumerate()
                {
                    for c in 0..3 {
                        let round = 8e-5 * (1.0 + center[c].abs());
                        let value = f64::from(pixel[c]);
                        assert!(
                            value.is_finite()
                                && value >= bounds[c][0] - round
                                && value <= bounds[c][1] + round,
                            "codec {topology}, protected {protected}, {p}/{c}: {value}, expected {} in {:?}",
                            center[c],
                            bounds[c]
                        );
                    }
                    assert_eq!(pixel[3].to_bits(), alpha.to_bits());
                }
                if let Some(baseline) = &baseline {
                    assert_eq!(&actual, baseline);
                }
                baseline = Some(actual);
                drop(frame);
                assert_eq!(
                    rig.decoder.engine().in_flight_memory_stats().reserved_bytes,
                    0
                );
            }
            assert_eq!(rig.context.memory_stats().reserved_bytes, 0);
        }
    }
}
