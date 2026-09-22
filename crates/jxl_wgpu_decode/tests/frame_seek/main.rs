#![cfg(not(target_arch = "wasm32"))]

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_bitstream::FrameIndex;
use jxl_gpu_formats::{PixelFormat, RgbChannelOrder};
use jxl_test_support::gpu::planes;
use jxl_test_support::oracles::progressive::native_updates;
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::{
    BoundFrameIndex, FrameExecutionPlan, FrameSeekError, FrameSeekLimits, GpuDecoder,
    GpuOutputRequest, ImageSelection, SelectedImageInventory, WgpuDecodeEngine,
};

mod native;
mod ownership;
mod planning;

fn source(name: &str) -> Vec<u8> {
    jxl_test_support::offline::hex::unhex(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("test-data/{name}.jxl.hex")),
        )
        .unwrap(),
    )
}

fn inventory(data: &[u8]) -> Arc<jxl_gpu_bitstream::CodestreamInventory> {
    Arc::new(
        jxl_gpu_bitstream::parse(data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap(),
    )
}

fn indexed(data: &[u8], index: &FrameIndex, fragments: bool) -> Vec<u8> {
    use jxl_gpu_bitstream::{ContainerBox, FRAME_INDEX_BOX_TYPE, FragmentedContainerWriter};
    let payload = index.encode(Default::default()).unwrap();
    let parsed = jxl_gpu_bitstream::parse(data, Default::default()).unwrap();
    let boxed = ContainerBox {
        box_type: FRAME_INDEX_BOX_TYPE,
        payload: &payload,
    };
    if fragments {
        let mut writer = FragmentedContainerWriter::new();
        writer.push_box(boxed).unwrap();
        let chunks: Vec<_> = parsed
            .codestream()
            .chunks(parsed.codestream().len().div_ceil(7))
            .collect();
        for (i, chunk) in chunks.iter().enumerate() {
            writer.push_fragment(chunk, i + 1 == chunks.len()).unwrap();
        }
        writer.finish().unwrap()
    } else {
        jxl_gpu_bitstream::write_container_with_boxes(parsed.codestream(), &[boxed]).unwrap()
    }
}

fn request() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb8(
        RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
}

fn backend() -> WgpuBackend {
    pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    }))
    .expect("actual GPU required for seek validation")
}

fn released(backend: &WgpuBackend) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while (backend.submission_poller().in_flight() != 0
        || backend.transient_memory_budget().snapshot().reserved_bytes != 0)
        && std::time::Instant::now() < deadline
    {
        backend.device().poll(wgpu::PollType::Poll).unwrap();
        std::thread::yield_now();
    }
    assert_eq!(backend.submission_poller().in_flight(), 0);
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}

#[test]
fn indexed_gpu_seeks_preserve_native_pixels_original_timing_and_reference_chains() {
    let backend = backend();
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
    );
    for name in [
        "sequence_modular_gray",
        "sequence_modular_rgb12",
        "sequence_modular_rgba16",
        "sequence_modular_many",
        "sequence_vardct_rgb",
        "sequence_vardct_gray",
        "sequence_vardct_dc",
        "sequence_mixed_jpeg_modular",
        "sequence_layered_still",
        "composition_gray",
        "composition_gray_alpha",
        "composition_rgba8",
        "composition_rgba16",
        "composition_rgba_mixed_depth",
        "composition_vardct",
        "composition_vardct_dc",
        "composition_mixed",
        "composition_still",
        "preview/animation_modular",
        "preview/animation_vardct",
        "noise/mixed_frames",
    ] {
        let data = source(name);
        let index = BoundFrameIndex::new(inventory(&data), None, Default::default()).unwrap();
        let container = indexed(&data, index.index(), true);
        let native: Vec<_> = native_updates(&container, false)
            .expect("native oracle required")
            .into_iter()
            .filter(|frame| frame.complete)
            .collect();
        let mut sequence = whole.open(&data, request()).unwrap();
        let mut expected = Vec::new();
        while let Some(frame) = sequence.next_frame().unwrap() {
            let pixels = planes::read_bytes(&backend, &frame.output().outputs[0]);
            let oracle = &native[expected.len()];
            let native_pixels: Vec<_> = oracle
                .pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| {
                    let value = f32::from_le_bytes(*word);
                    assert!(value.is_finite());
                    (value.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect();
            assert_eq!(pixels.len(), native_pixels.len(), "{name}");
            let error = pixels
                .iter()
                .zip(&native_pixels)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap();
            assert!(error <= 1, "{name}: native RGBA8 error {error}");
            assert_eq!(frame.metadata.duration.ticks, oracle.duration, "{name}");
            assert_eq!(
                frame.metadata.timecode.unwrap_or(0),
                oracle.timecode,
                "{name}"
            );
            expected.push((frame.metadata.clone(), pixels));
        }
        assert_eq!(expected.len(), native.len(), "{name}");
        drop(sequence);
        for (decoder, input, asynchronous) in [(&whole, &data, false), (&bounded, &container, true)]
        {
            for target in (0..expected.len()).rev() {
                let mut seek = decoder
                    .open_seek(
                        input,
                        request(),
                        target,
                        Default::default(),
                        Default::default(),
                    )
                    .unwrap_or_else(|error| panic!("{name}/{target}: {error}"));
                let plan = index.seek(target, Default::default()).unwrap();
                assert_eq!(seek.plan().physical_frames(), plan.physical_frames());
                let frame = if asynchronous {
                    pollster::block_on(seek.next_frame_async())
                } else {
                    seek.next_frame()
                }
                .unwrap()
                .unwrap();
                assert_eq!(frame.metadata, expected[target].0, "{name}/{target}");
                let actual = planes::read_bytes(&backend, &frame.output().outputs[0]);
                assert!(
                    actual == expected[target].1,
                    "{name}/{target}: seek differs from sequential output; first mismatch {:?}",
                    actual
                        .iter()
                        .zip(&expected[target].1)
                        .position(|(a, b)| a != b)
                );
                assert_eq!(seek.frames_submitted(), plan.preroll_presentations() + 1);
                assert!(seek.next_frame().unwrap().is_none());
                assert!(seek.submission_session().is_none());
                drop(frame);
                drop(seek);
                released(&backend);
            }
        }
    }
}
