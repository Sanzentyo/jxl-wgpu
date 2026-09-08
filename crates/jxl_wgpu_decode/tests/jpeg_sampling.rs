#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "support/noise.rs"]
mod noise;
#[path = "common/extra_channel_oracle.rs"]
mod oracle;
#[path = "support/planes.rs"]
mod planes;
#[path = "support/jpeg_sampling.rs"]
mod sampling;

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::{CodestreamInventory, CodestreamStreamError, InventoryError};
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::vardct::frontend::{StandardVarDctProfile, VarDctFrontendError};
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

fn selectors() -> impl Iterator<Item = [u32; 3]> {
    (0..64).map(|index| [index / 16, index / 4 % 4, index % 4])
}

fn encoded([cb, y, cr]: [u32; 3], odd: bool) -> Vec<u8> {
    let prefix = if odd { "odd" } else { "sampling" };
    fixture(&format!("{prefix}_{cb}{y}{cr}"))
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("test-data/jpeg_sampling/{name}.jxl.hex"));
    let text = std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .collect::<String>();
    text.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}

fn inventory(data: &[u8]) -> Result<CodestreamInventory, InventoryError> {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
}

fn request() -> GpuOutputRequest {
    GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
}

fn backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    })) {
        Ok(backend) => Some(backend),
        Err(jxl_wgpu::Error::NoAdapter) => None,
        Err(error) => panic!("JPEG sampling adapter: {error}"),
    }
}

fn decode(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    fragmented: bool,
) -> Vec<u32> {
    let mut session = if fragmented {
        planes::open_fragmented(decoder, data, request())
    } else {
        decoder.open(data, request()).unwrap()
    };
    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    let actual = planes::read(backend, &frame.output().outputs[0]);
    drop(frame);
    assert!(
        pollster::block_on(session.next_frame_async())
            .unwrap()
            .is_none()
    );
    drop(session);
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    actual
}

fn compare(actual: &[u32], expected: &[f32], limit: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(&word, &reference)| {
            let value = f32::from_bits(word);
            assert!(value.is_finite() && reference.is_finite());
            (value - reference).abs()
        })
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: maxAE={error}");
    assert!(error < limit, "{label}: maxAE={error}, limit={limit}");
}

#[test]
fn every_component_sampling_layout_matches_both_decoders_with_bounded_input() {
    let Some(backend) = backend() else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for (odd, width, height) in [(false, 272, 32), (true, 257, 17)] {
        let mut equal_outputs = [None, None, None, None];
        for selectors in selectors() {
            let data = encoded(selectors, odd);
            let original = inventory(&data).unwrap();
            assert_eq!(
                (original.image_header.width, original.image_header.height),
                (width, height)
            );
            let frame = &original.frames[0];
            assert_eq!(frame.jpeg_upsampling, selectors);
            assert_eq!(frame.flags, 129);
            assert_eq!(frame.group_count, 2);
            let equal = selectors == [selectors[0]; 3];
            let mut outputs = Vec::new();
            for smoothing in [false, true] {
                if smoothing && !equal {
                    continue;
                }
                let data = if smoothing {
                    sampling::rewrite(&data, selectors, true)
                } else {
                    data.clone()
                };
                let parsed = inventory(&data).unwrap();
                let profile = StandardVarDctProfile::negotiate(&parsed).unwrap();
                assert_eq!(profile.adaptive_lf_smoothing, smoothing);
                assert_eq!(
                    profile.channel_shifts.iter().all(|v| !v.is_subsampled()),
                    equal
                );
                for zero in [false, true] {
                    let data = if zero {
                        noise::zero_noise(&data, &parsed, None)
                    } else {
                        data.clone()
                    };
                    let actual = decode(&backend, &whole, &data, false);
                    assert_eq!(actual, decode(&backend, &bounded, &data, true));
                    let label =
                        format!("{width}x{height} {selectors:?} smoothing={smoothing} zero={zero}");
                    compare(
                        &actual,
                        &oracle::rust_planes(&data).0,
                        1e-5,
                        &format!("{label} Rust"),
                    );
                    if let Some((color, extras)) =
                        oracle::libjxl_planes(&data, width as usize * height as usize, 0)
                    {
                        assert!(extras.is_empty());
                        compare(&actual, &color, 1.0 / 1024.0, &format!("{label} native"));
                    }
                    if equal && !odd {
                        let index = usize::from(smoothing) * 2 + usize::from(zero);
                        if let Some(reference) = &equal_outputs[index] {
                            assert_eq!(
                                &actual, reference,
                                "equal sampling factors must reconstruct identically"
                            );
                        } else {
                            equal_outputs[index] = Some(actual.clone());
                        }
                    }
                    outputs.push(actual);
                }
            }
            assert_ne!(
                outputs[0], outputs[1],
                "{selectors:?}: noise must affect pixels"
            );
            if equal && !odd {
                assert_ne!(
                    outputs[1], outputs[3],
                    "{selectors:?}: adaptive LF smoothing must affect pixels"
                );
            }
        }
    }
}

#[test]
fn equal_sampling_factors_preserve_lf_chroma_correlation() {
    let Some(backend) = backend() else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for (odd, prefix, pixels) in [(false, "sampling", 272 * 32), (true, "odd", 257 * 17)] {
        let mut equal_outputs = [None, None, None, None];
        for raw in 0..4 {
            let original = encoded([raw; 3], odd);
            let correlated = fixture(&format!("correlation_{prefix}_{raw}"));
            assert_eq!(correlated, sampling::correlated(&original));
            let uncorrelated = decode(&backend, &whole, &original, false);
            for smoothing in [false, true] {
                let data = sampling::rewrite(&correlated, [raw; 3], smoothing);
                let parsed = inventory(&data).unwrap();
                for zero in [false, true] {
                    let data = if zero {
                        noise::zero_noise(&data, &parsed, None)
                    } else {
                        data.clone()
                    };
                    let actual = decode(&backend, &whole, &data, false);
                    assert_eq!(actual, decode(&backend, &bounded, &data, true));
                    let label =
                        format!("correlation {prefix} raw={raw} smoothing={smoothing} zero={zero}");
                    // Rust jxl 0.6 uses LF-adjusted rather than base correlation for noise.
                    // Keep its independent zero-model oracle; native checks both models.
                    if zero {
                        compare(
                            &actual,
                            &oracle::rust_planes(&data).0,
                            1e-5,
                            &format!("{label} Rust"),
                        );
                    }
                    if let Some((color, _)) = oracle::libjxl_planes(&data, pixels, 0) {
                        compare(&actual, &color, 1.0 / 1024.0, &format!("{label} native"));
                    }
                    if !smoothing && !zero {
                        assert!(
                            actual != uncorrelated,
                            "{label}: LF correlation must affect pixels"
                        );
                    }
                    if !odd {
                        let index = usize::from(smoothing) * 2 + usize::from(zero);
                        if let Some(reference) = &equal_outputs[index] {
                            assert_eq!(
                                &actual, reference,
                                "{label}: equal factors changed LF correlation"
                            );
                        } else {
                            equal_outputs[index] = Some(actual);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_sampling_is_rejected_before_section_delivery_and_profile_planning() {
    for selectors in selectors().filter(|v| *v != [v[0]; 3]) {
        let data = encoded(selectors, false);
        let mut modified = inventory(&data).unwrap();
        modified.frames[0].flags &= !128;
        let expected = InventoryError::SubsampledAdaptiveLfSmoothing {
            jpeg_upsampling: selectors,
        };
        assert_eq!(
            StandardVarDctProfile::negotiate(&modified).unwrap_err(),
            VarDctFrontendError::InvalidFrameHeader(expected.clone())
        );
        let invalid = sampling::rewrite(&data, selectors, true);
        assert_eq!(inventory(&invalid).unwrap_err(), expected);
        for chunk_size in [1, 43, invalid.len()] {
            let mut transport = jxl_gpu_bitstream::ContainerStreamScanner::new(Default::default());
            let mut scanner = jxl_gpu_bitstream::CodestreamStreamScanner::new(Default::default());
            let result = (|| {
                for chunk in invalid.chunks(chunk_size) {
                    for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                        scanner.push_transport_event(&event)?;
                    }
                }
                for event in transport.finish_input().unwrap() {
                    scanner.push_transport_event(&event)?;
                }
                Ok::<_, CodestreamStreamError>(())
            })();
            assert_eq!(
                result.unwrap_err(),
                CodestreamStreamError::Inventory(expected.clone())
            );
            assert_eq!(scanner.stats().frames_started, 0);
            assert_eq!(scanner.stats().section_bytes_emitted, 0);
            assert_eq!(
                scanner.finish_input(invalid.len() as u64).unwrap_err(),
                CodestreamStreamError::Failed
            );
        }
    }
    let mut modified = inventory(&encoded([0; 3], false)).unwrap();
    for (selectors, do_ycbcr) in [
        ([4, 0, 0], true),
        ([0, u32::MAX, 0], true),
        ([0, 0, 4], true),
        ([1; 3], false),
    ] {
        modified.frames[0].jpeg_upsampling = selectors;
        modified.frames[0].do_ycbcr = do_ycbcr;
        assert_eq!(
            StandardVarDctProfile::negotiate(&modified).unwrap_err(),
            VarDctFrontendError::InvalidFrameHeader(InventoryError::InvalidJpegSampling {
                jpeg_upsampling: selectors,
                do_ycbcr,
            })
        );
    }
}

#[test]
fn invalid_adaptive_sampling_never_admits_gpu_work_or_retains_input() {
    let Some(backend) = backend() else {
        return;
    };
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for selectors in selectors().filter(|v| *v != [v[0]; 3]) {
        let data = sampling::rewrite(&encoded(selectors, false), selectors, true);
        assert!(matches!(decoder.open(&data, request()),
            Err(jxl_wgpu_decode::Error::CodestreamInventory(InventoryError::SubsampledAdaptiveLfSmoothing {
                jpeg_upsampling,
            })) if jpeg_upsampling == selectors));
        let mut stream = decoder.stream(request()).unwrap();
        let mut transport = jxl_gpu_bitstream::ContainerStreamScanner::new(Default::default());
        let result = (|| {
            for chunk in data.chunks(7) {
                for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                    stream.push_transport_event(&event)?;
                }
            }
            for event in transport.finish_input().unwrap() {
                stream.push_transport_event(&event)?;
            }
            Ok::<_, jxl_wgpu_decode::Error>(())
        })();
        assert!(
            matches!(result, Err(jxl_wgpu_decode::Error::CodestreamStream(
            CodestreamStreamError::Inventory(InventoryError::SubsampledAdaptiveLfSmoothing { jpeg_upsampling })
        )) if jpeg_upsampling == selectors)
        );
        let stats = stream.stats();
        assert_eq!(stats.codestream.frames_started, 0);
        assert_eq!(stats.codestream.section_bytes_emitted, 0);
        assert_eq!(stats.retained_codestream_bytes, 0);
        assert_eq!(stats.retained_spans, 0);
        assert_eq!(stats.input_budget.reserved_bytes, 0);
        assert!(matches!(
            stream.finish(),
            Err(jxl_wgpu_decode::Error::IncrementalInputPoisoned)
        ));
        assert_eq!(backend.submission_poller().in_flight(), 0);
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
