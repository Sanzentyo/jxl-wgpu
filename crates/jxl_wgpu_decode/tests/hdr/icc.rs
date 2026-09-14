use super::{backend, corpus, oracle, planes};
use jxl_gpu_formats::{
    ColorSpace, ColorSpecification, PixelFormat, RgbChannelOrder, TransferFunction,
};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_test_support::oracles::color;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, WgpuDecodeEngine};
use std::{
    num::{NonZeroU64, NonZeroUsize},
    path::Path,
};

mod embedded;
mod pcs;
mod profiles;

fn frames(
    backend: &WgpuBackend,
    data: &[u8],
    request: GpuOutputRequest,
    planar: bool,
    channels: usize,
    bounded: bool,
) -> Vec<Vec<u32>> {
    let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
    if bounded {
        engine = engine.with_stream_window_limit(NonZeroU64::new(256).unwrap());
    }
    let decoder = GpuDecoder::new(engine);
    let format = request.format().clone();
    let request = request
        .with_progressive_output(!planar)
        .with_max_frame_slots(NonZeroUsize::new(64).unwrap());
    let mut session = if bounded {
        planes::open_fragmented(&decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let mut held = Vec::new();
    while let Some(update) = pollster::block_on(session.next_update_async()).unwrap() {
        assert_eq!(update.output().outputs[0].layout.format, format);
        let words = planes::read(backend, &update.output().outputs[0]);
        held.push((update, words));
    }
    drop(session);
    let mut output = Vec::new();
    for (update, words) in held {
        assert_eq!(planes::read(backend, &update.output().outputs[0]), words);
        if update.progression().is_none() {
            let pixels = words.len() / channels;
            assert_eq!(words.len(), pixels * channels);
            output.push(if planar {
                (0..pixels)
                    .flat_map(|p| (0..channels).map(move |c| (p, c)))
                    .map(|(p, c)| words[c * pixels + p])
                    .collect()
            } else {
                words
            });
        }
    }
    assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
    assert_eq!(
        decoder.incremental_input_budget().snapshot().reserved_bytes,
        0
    );
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
    output
}
