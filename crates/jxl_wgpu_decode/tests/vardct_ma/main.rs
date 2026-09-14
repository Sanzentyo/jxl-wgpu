#![cfg(not(target_arch = "wasm32"))]

mod references;

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::Arc;

use jxl_test_support::gpu::planes;
use jxl_test_support::oracles::extra_channels as oracle;
use jxl_wgpu::{WgpuBackend, WgpuBackendConfig};
use jxl_wgpu_decode::vardct::packet::BoundedVarDctPacketPlan;
use jxl_wgpu_decode::{GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};

fn directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-data/vardct_ma")
}

fn backend() -> Option<WgpuBackend> {
    match pollster::block_on(WgpuBackend::request_default(WgpuBackendConfig {
        enable_timestamps: false,
        ..Default::default()
    })) {
        Ok(backend) => {
            eprintln!("VarDCT MA adapter: {:?}", backend.adapter_info());
            Some(backend)
        }
        Err(jxl_wgpu::Error::NoAdapter) => {
            eprintln!("skipping VarDCT MA GPU conformance: no adapter");
            None
        }
        Err(error) => panic!("VarDCT MA adapter: {error}"),
    }
}

fn request() -> GpuOutputRequest {
    GpuOutputRequest::color(jxl_gpu_formats::PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
}

fn decode(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<WgpuDecodeEngine>,
    data: &[u8],
    fragmented: bool,
) -> (Vec<u32>, usize) {
    let mut session = if fragmented {
        let mut stream = decoder.stream(request()).unwrap();
        let mut transport =
            jxl_gpu_bitstream::ContainerStreamScanner::new(decoder.container_stream_limits());
        for chunk in data.chunks(7) {
            for event in transport.push_chunk(Arc::from(chunk)).unwrap() {
                stream.push_transport_event(&event).unwrap();
            }
        }
        for event in transport.finish_input().unwrap() {
            stream.push_transport_event(&event).unwrap();
        }
        assert!(stream.stats().retained_spans > 2);
        stream.finish().unwrap()
    } else {
        decoder.open(data, request()).unwrap()
    };
    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    let actual = planes::read(backend, &frame.output().outputs[0]);
    let submissions = session.submission_session().submissions_per_frame();
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
    (actual, submissions)
}

fn compare(actual: &[u32], reference: &[f32], limit: f32, label: &str) {
    assert_eq!(actual.len(), reference.len());
    let error = actual
        .iter()
        .zip(reference)
        .map(|(&word, &expected)| {
            let value = f32::from_bits(word);
            assert!(value.is_finite() && expected.is_finite());
            (value - expected).abs()
        })
        .fold(0.0_f32, f32::max);
    eprintln!("{label}: maxAE={error}");
    assert!(error < limit, "{label}: maxAE={error}, limit={limit}");
}

#[test]
fn native_previous_channel_trees_decode_whole_and_bounded() {
    let Some(backend) = backend() else {
        return;
    };
    let whole = GpuDecoder::wgpu(backend.clone()).unwrap();
    let bounded = GpuDecoder::new(
        WgpuDecodeEngine::new(backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(40).unwrap()),
    );
    let manifest = std::fs::read_to_string(directory().join("manifest.txt")).unwrap();
    assert_eq!(manifest.lines().count(), 72);
    let mut outputs = std::collections::BTreeMap::new();
    for line in manifest.lines() {
        let columns = line.split_whitespace().collect::<Vec<_>>();
        let name = columns[0];
        eprintln!("native MA fixture {name}");
        let width = columns[1].parse::<usize>().unwrap();
        let height = columns[2].parse::<usize>().unwrap();
        let property = columns[3].parse::<u32>().unwrap();
        let weighted = columns[4] == "1";
        let hex = std::fs::read_to_string(directory().join(format!("{name}.jxl.hex"))).unwrap();
        let data = jxl_test_support::offline::hex::unhex(&hex);
        let inventory = jxl_gpu_bitstream::parse(&data, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        let plan = BoundedVarDctPacketPlan::parse(&data, &inventory).unwrap();
        // Inspect the packed MA ABI, independently of whether the selected leaf
        // happens to be used by this channel. The fixture must retain its split.
        assert_eq!(plan.modular_metadata[0], 3);
        let root = plan.modular_metadata[5] as usize;
        assert_eq!(plan.modular_metadata[root], 0);
        assert_eq!(plan.modular_metadata[root + 1], property);
        assert_eq!(plan.needs_self_correcting, weighted);
        let (actual, submissions) = decode(&backend, &whole, &data, false);
        let (fragmented, resumed_submissions) = decode(&backend, &bounded, &data, true);
        assert_eq!(actual, fragmented, "{name}: bounded continuation");
        if width > 8 {
            assert!(resumed_submissions > submissions, "{name}: must resume");
        }
        if let Some(previous) = outputs.get(&(width, height)) {
            assert_eq!(
                &actual, previous,
                "{name}: changing the MA tree must preserve pixels"
            );
        } else {
            outputs.insert((width, height), actual.clone());
        }
        compare(
            &actual,
            &oracle::rust_planes(&data).0,
            1e-4,
            &format!("{name} Rust"),
        );
        if let Some((color, extras)) = oracle::libjxl_planes(&data, width * height, 0) {
            assert!(extras.is_empty());
            // Native inverse-color/transfer approximations use the same F32
            // budget as the JPEG sampling corpus. Integer MA probes remain exact.
            compare(&actual, &color, 1.0 / 1024.0, &format!("{name} libjxl"));
        }
    }
}
