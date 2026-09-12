#![cfg(not(target_arch = "wasm32"))]

use std::io::Read;
use std::num::NonZeroU64;

use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::gpu::planes;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use sha2::{Digest, Sha256};

mod features;
mod progression;

fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap()
}

fn reference() -> Vec<f32> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-data/splines/animation_spline.npy.gz");
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(
        Sha256::digest(&bytes).as_slice(),
        jxl_test_support::offline::hex::unhex(
            "a571c5cbba58affeeb43c44c13f81e2b1962727eb9d4e017e4f25d95c7388f10"
        )
    );
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let offset = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
    let header = std::str::from_utf8(&bytes[10..offset]).unwrap();
    assert!(header.contains("'descr': '<f4'"));
    assert!(header.contains("'fortran_order': False"));
    assert!(header.contains("'shape': (60, 320, 320, 3)"));
    let (words, remainder) = bytes[offset..].as_chunks::<4>();
    assert!(remainder.is_empty());
    assert_eq!(words.len(), 60 * 320 * 320 * 3);
    words.iter().copied().map(f32::from_le_bytes).collect()
}

#[test]
fn official_spline_animation_meets_every_frame_bound_for_whole_and_bounded_input() {
    let backend = backend();
    let expected = reference();
    let data = include_bytes!("../../../../fixtures/animation_spline.jxl");
    let request = GpuOutputRequest::color(PixelFormat::rgb_f32(
        RgbChannelOrder::Rgb,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap();
    let mut whole = Vec::new();
    for limit in [None, NonZeroU64::new(256)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        let mut session = if limit.is_some() {
            planes::open_fragmented(&decoder, data, request.clone())
        } else {
            decoder.open(data, request.clone()).unwrap()
        };
        for (index, expected) in expected
            .as_chunks::<{ 320 * 320 * 3 }>()
            .0
            .iter()
            .enumerate()
        {
            let frame = if limit.is_some() {
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .unwrap()
            } else {
                session.next_frame().unwrap().unwrap()
            };
            let actual = planes::read(&backend, &frame.output().outputs[0]);
            assert_eq!(frame.metadata.index, index);
            assert!((frame.metadata.duration.as_seconds() - 0.02).abs() < 1e-12);
            assert_eq!(frame.metadata.is_last, index == 59);
            assert_eq!(actual.len(), expected.len());
            let mut squared = [0.0; 3];
            let mut peak = 0.0f64;
            for (i, (&word, &reference)) in actual.iter().zip(expected).enumerate() {
                let value = f32::from_bits(word);
                assert!(value.is_finite());
                let error = f64::from(value) - f64::from(reference);
                squared[i % 3] += error * error;
                peak = peak.max(error.abs());
            }
            let rmse = squared.map(|sum| (sum / (320.0 * 320.0)).sqrt());
            eprintln!("frame {index}, window {limit:?}: RMSE {rmse:?}, peak {peak}");
            if let Ok(directory) = std::env::var("JXL_SPLINE_OUTPUT_DIR") {
                let path =
                    std::path::Path::new(&directory).join(format!("frame-{index:03}.rgb.f32"));
                let bytes: Vec<_> = actual.iter().flat_map(|word| word.to_le_bytes()).collect();
                std::fs::write(path, bytes).unwrap();
            }
            assert!(
                rmse.into_iter().all(|error| error <= 0.0001),
                "frame {index}: RMSE {rmse:?}"
            );
            assert!(peak <= 0.004, "frame {index}: peak {peak}");
            if limit.is_none() {
                whole.push(actual);
            } else {
                assert_eq!(actual, whole[index], "bounded frame {index}");
            }
        }
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    }
}
