#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_bitstream::FrameEncoding;
use jxl_test_support::{fixtures::embedded_icc as corpus, gpu::planes};
use jxl_wgpu::{GpuImageFrame, WgpuBackend};
use jxl_wgpu_decode::{
    GpuDecoder, GpuOutputRequest, GpuSubmissionEngine, GpuSubmissionSession, NumericSampleMapping,
    VarDctSubmissionEngine, WgpuDecodeEngine, WgpuSubmissionEngine,
};
use std::num::NonZeroU64;

mod color;
mod numeric;
mod profile;
mod ycbcr;

fn inventory(data: &[u8]) -> jxl_gpu_bitstream::CodestreamInventory {
    jxl_gpu_bitstream::parse(data, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
}

fn numeric_float() -> GpuOutputRequest {
    use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
    GpuOutputRequest::numeric(
        PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
        NumericSampleMapping::NativeFloat,
    )
    .unwrap()
}

fn decode<E>(
    backend: &WgpuBackend,
    decoder: &GpuDecoder<E>,
    data: &[u8],
    request: GpuOutputRequest,
    fragmented: bool,
) -> Vec<u8>
where
    E: GpuSubmissionEngine,
    E::Session: GpuSubmissionSession<Frame = GpuImageFrame>,
{
    let mut session = if fragmented {
        planes::open_fragmented(decoder, data, request)
    } else {
        decoder.open(data, request).unwrap()
    };
    let frame = pollster::block_on(session.next_frame_async())
        .unwrap()
        .unwrap();
    let output = &frame.output().outputs[0];
    assert!(matches!(
        output.layout.format.color_spec,
        jxl_gpu_formats::ColorSpecification::Undefined
    ));
    let bytes = planes::read_bytes(backend, output);
    assert!(
        pollster::block_on(session.next_frame_async())
            .unwrap()
            .is_none()
    );
    bytes
}

#[test]
fn native_icc_declarations_and_scalar_alpha_survive_both_codecs() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    for limit in [None, NonZeroU64::new(256)] {
        let mut common = WgpuDecodeEngine::new(backend.clone()).unwrap();
        let mut modular = WgpuSubmissionEngine::new(backend.clone());
        let mut vardct = VarDctSubmissionEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            common = common.with_stream_window_limit(limit);
            modular = modular.with_stream_window_limit(limit);
            vardct = vardct.with_stream_window_limit(limit);
        }
        let common = GpuDecoder::new(common);
        let modular = GpuDecoder::new(modular);
        let vardct = GpuDecoder::new(vardct);
        for case in corpus::cases() {
            let data = case.bytes();
            case.validate(&inventory(&data));
            let width = if case.gray { 2 } else { 4 };
            let expected: Vec<_> = case
                .input()
                .chunks_exact(width)
                .flat_map(|pixel| pixel[width - 1].to_le_bytes())
                .collect();
            let request = numeric_float().with_extra_channel(0).unwrap();
            let actual = decode(&backend, &common, &data, request.clone(), limit.is_some());
            assert_eq!(actual, expected, "{} common {limit:?}", case.name());
            let actual = match case.encoding {
                FrameEncoding::Modular => {
                    decode(&backend, &modular, &data, request, limit.is_some())
                }
                FrameEncoding::VarDct => decode(&backend, &vardct, &data, request, limit.is_some()),
            };
            assert_eq!(actual, expected, "{} producer {limit:?}", case.name());
            if case.encoding == FrameEncoding::Modular && !case.xyb {
                for channel in 0..width - 1 {
                    let expected: Vec<_> = case
                        .input()
                        .chunks_exact(width)
                        .flat_map(|pixel| pixel[channel].to_le_bytes())
                        .collect();
                    let request = numeric_float().with_color_channel(channel as u32).unwrap();
                    for actual in [
                        decode(&backend, &common, &data, request.clone(), limit.is_some()),
                        decode(&backend, &modular, &data, request, limit.is_some()),
                    ] {
                        assert_eq!(actual, expected, "{} color {channel}", case.name());
                    }
                }
            }
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[test]
fn icc_color_conversion_requires_execution_and_cannot_be_enabled_by_metadata() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoder = GpuDecoder::wgpu(backend.clone()).unwrap();
    for case in corpus::cases().filter(|case| case.xyb) {
        let data = case.bytes();
        let request = GpuOutputRequest::color(jxl_wgpu_decode::vardct_rgb8_format()).unwrap();
        assert!(matches!(decoder.open(&data, request),
            Err(jxl_wgpu_decode::Error::UnsupportedProfile(error))
                if error.feature == jxl_wgpu_decode::UnsupportedCodestreamFeature::ColorEncoding));
        assert_eq!(
            backend.transient_memory_budget().snapshot().reserved_bytes,
            0
        );
    }
}
