use super::*;
use jxl_gpu_formats::{Channel, SampleKind};
use jxl_wgpu_decode::NumericSampleMapping;
use jxl_wgpu_encode::LosslessModularFormat;

fn scalar_request(index: u32, bits: u8, floating: bool) -> GpuOutputRequest {
    let (format, mapping) = if floating {
        (
            PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
            NumericSampleMapping::NormalizedUnsigned,
        )
    } else {
        (
            LosslessModularFormat::Gray.pixel_format(bits).unwrap(),
            NumericSampleMapping::NativeUnsigned,
        )
    };
    GpuOutputRequest::numeric(format, mapping)
        .unwrap()
        .with_extra_channel(index)
        .unwrap()
        .with_orientation_policy(if floating {
            OrientationPolicy::Apply
        } else {
            OrientationPolicy::Keep
        })
}

#[test]
fn every_global_vardct_extra_plane_has_independent_native_and_float_output() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(1024).unwrap()),
    );
    for (name, hex) in fixtures() {
        let data = encoded(hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let image = &inventory.image_header;
        let original = Extent2d::new(image.width, image.height);
        let oriented = OutputOrientation::from_exif_value(image.orientation)
            .unwrap()
            .map_extent(original);
        let (_, expected) = oracle::rust_planes(&data);
        let libjxl =
            oracle::libjxl_planes(&data, original.area().unwrap(), image.extra_channels.len());
        for (index, extra) in image.extra_channels.iter().enumerate() {
            let jxl_gpu_bitstream::SampleBitDepth::Integer { bits_per_sample } = extra.bit_depth
            else {
                unreachable!()
            };
            for floating in [false, true] {
                let request = scalar_request(index as u32, bits_per_sample as u8, floating);
                // Scalar data does not request spot-color rendering, even with the default policy.
                let mut session = if floating {
                    open_incremental(&bounded, &data, request)
                } else {
                    decoder
                        .open(&data, request)
                        .unwrap_or_else(|e| panic!("{name}/{index}: {e}"))
                };
                assert_eq!(session.metadata().extra_channels, image.extra_channels);
                let frame = if floating {
                    pollster::block_on(session.next_frame_async())
                } else {
                    session.next_frame()
                }
                .unwrap_or_else(|e| panic!("{name}/{index}: {e}"))
                .unwrap();
                let memory = session
                    .submission_session()
                    .vardct()
                    .unwrap()
                    .memory_stats()
                    .unwrap();
                assert_eq!(memory.resident_image_bytes, 0);
                assert_eq!(memory.resident_plane_bytes, [0; 3]);
                assert_eq!(memory.resident_transient_bytes, 0);
                assert_eq!(memory.restoration_scratch_bytes, 0);
                assert_eq!(memory.gaborish_uniform_bytes, 0);
                assert_eq!(memory.epf_sigma_bytes, 0);
                assert_eq!(memory.epf_filter_uniform_bytes, 0);
                assert_eq!(memory.frame_upsample_bytes, 0);
                assert_eq!(memory.pre_restoration_upsample_bytes, 0);
                assert_eq!(memory.output_uniform_bytes, 64);
                assert_eq!(memory.output_status_bytes, 4);
                let result = ImageReadbackPipeline::new(&backend)
                    .submit(frame.output())
                    .unwrap()
                    .wait()
                    .unwrap();
                let output = &result.frame.outputs[0];
                assert_eq!(
                    output.layout.extent,
                    if floating { oriented } else { original }
                );
                if floating {
                    let actual = oracle::floats(&output.bytes);
                    assert_eq!(actual.len(), expected[index].len());
                    for (position, (a, b)) in actual.iter().zip(&expected[index]).enumerate() {
                        assert!(
                            (a - b).abs() < 2e-7,
                            "{name}/{index}/{position}: {a} vs {b}"
                        );
                        if let Some((_, planes)) = &libjxl {
                            assert!(
                                (a - planes[index][position]).abs() < 2e-7,
                                "{name}/{index}: libjxl"
                            );
                        }
                    }
                } else {
                    let bytes_per_sample = bits_per_sample.div_ceil(8) as usize;
                    let mask = (1_u32 << bits_per_sample) - 1;
                    let channel = if image.grayscale { 1 } else { 3 } + index as u32;
                    for y in 0..image.height {
                        for x in 0..image.width {
                            let code = match x % 11 {
                                0 => 0,
                                1 => mask,
                                _ => {
                                    (193 * x + 317 * y + 97 * channel + (x ^ y) * (23 + channel))
                                        & mask
                                }
                            };
                            let offset = (y * image.width + x) as usize * bytes_per_sample;
                            let actual = if bytes_per_sample == 1 {
                                u32::from(output.bytes[offset])
                            } else {
                                u32::from(u16::from_le_bytes(
                                    output.bytes[offset..offset + 2].try_into().unwrap(),
                                ))
                            };
                            assert_eq!(actual, code, "{name}/{index}/{x},{y}");
                        }
                    }
                }
                drop(result);
                drop(frame);
                drop(session);
                assert_eq!(
                    decoder.engine().in_flight_memory_stats().reserved_bytes,
                    0,
                    "{name}/{index}: lifetime"
                );
            }
        }
    }
}

#[test]
fn scalar_selection_validates_depth_and_rejects_later_color_entropy_damage() {
    let Ok(backend) = pollster::block_on(WgpuBackend::request_default(Default::default())) else {
        return;
    };
    let data = encoded(include_str!(
        "../../../test-data/vardct_extras_rgba_progressive.jxl.hex"
    ));
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    assert!(matches!(
        decoder.open(&data, scalar_request(1, 5, false)),
        Err(DecodeError::ExtraChannelIndex { .. })
    ));
    assert!(matches!(
        decoder.open(&data, scalar_request(0, 8, false)),
        Err(DecodeError::VarDct(VarDctDecodeError::ScalarOutput(_)))
    ));
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    // Keep the complete global extra stream and corrupt only the final color AC packet.
    let section = inventory.frames[0]
        .sections
        .iter()
        .rev()
        .find(|s| {
            matches!(
                s.kind,
                jxl_gpu_bitstream::FrameSectionKind::PassGroup { .. }
            )
        })
        .unwrap();
    let start = section.bits.offset as usize / 8;
    let end = section.bits.end().unwrap() as usize / 8;
    let mut damaged = data.clone();
    damaged[start..end].fill(0);
    let mut session = decoder.open(&damaged, scalar_request(0, 5, true)).unwrap();
    let error = pollster::block_on(session.next_frame_async()).unwrap_err();
    assert!(
        matches!(
            error,
            DecodeError::VarDct(VarDctDecodeError::HfCoefficientGpu(_))
        ),
        "{error}"
    );
    assert!(
        session
            .submission_session()
            .vardct()
            .unwrap()
            .memory_stats()
            .is_some(),
        "the global extra stream must validate before the damaged color stream fails"
    );
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
}
