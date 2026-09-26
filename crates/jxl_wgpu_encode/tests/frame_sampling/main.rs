#![cfg(not(target_arch = "wasm32"))]

mod admission;
mod modular;
mod sequence;
mod vardct;

use std::num::{NonZeroU8, NonZeroU64};
use std::sync::Arc;

use jxl_gpu_formats::{ByteOrder, Channel, PackingFieldKind, PixelFormat, Swizzle};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::gpu::planes::{open_fragmented, read_bytes};
use jxl_test_support::oracles::{extra_channels, modular_integer, modular_words, resampling};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine,
};
use jxl_wgpu_encode::*;
use wgpu::util::DeviceExt;

const FACTORS: [UpsamplingFactor; 3] = [
    UpsamplingFactor::Two,
    UpsamplingFactor::Four,
    UpsamplingFactor::Eight,
];

struct Rig {
    gpu: WgpuBackend,
    context: WgpuContext,
    decoders: [GpuDecoder<WgpuDecodeEngine>; 2],
}

impl Rig {
    fn new() -> Self {
        let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
        let context = WgpuContext::from_backend(&gpu);
        let decoders = [
            GpuDecoder::wgpu(gpu.clone()).unwrap(),
            GpuDecoder::new(
                WgpuDecodeEngine::new(gpu.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            ),
        ];
        Self {
            gpu,
            context,
            decoders,
        }
    }

    fn render(&self, bytes: &[u8], request: GpuOutputRequest, fragmented: bool) -> Vec<u8> {
        let decoder = &self.decoders[usize::from(fragmented)];
        let mut session = if fragmented {
            open_fragmented(decoder, bytes, request)
        } else {
            decoder.open(bytes, request).unwrap()
        };
        let frame = session.next_frame().unwrap().unwrap();
        assert!(session.next_frame().unwrap().is_none());
        drop(session);
        // Read after dropping the producer to exercise retained output ownership.
        let bytes = read_bytes(&self.gpu, &frame.output().outputs[0]);
        drop(frame);
        assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        bytes
    }
}

fn packed_format(samples: ColorSampleFormat, alpha: bool) -> PixelFormat {
    let mut format = samples.pixel_format();
    if alpha {
        let mut word = format.planes[0].words[0].clone();
        word.fields.last_mut().unwrap().kind = PackingFieldKind::Channel(Channel::W);
        format.planes[0].words.push(word);
        format.swizzle = if samples.channels() == ColorChannels::Gray {
            Swizzle::X00W
        } else {
            Swizzle::XYZW
        };
    }
    format
}

fn words(extent: Extent2d, samples: ColorSampleFormat, alpha: bool, seed: u32) -> Vec<u32> {
    let count = samples.channels().count() + u32::from(alpha);
    (0..extent.width * extent.height * count)
        .map(|index| {
            let code = ((index * 97 + index / count * 11 + seed * 79) % 127) + 48;
            if samples.exponent_bits() == 0 {
                code * (((1u64 << samples.bits_per_sample()) - 1) as u32) / 255
            } else {
                (code as f32 / 256.0).to_bits()
            }
        })
        .collect()
}

fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    mut format: PixelFormat,
    words: &[u32],
    variant: usize,
) -> BufferImageSource {
    format.byte_order = [ByteOrder::Big, ByteOrder::Little, ByteOrder::Native][variant % 3];
    let (layout, bytes) = Packing {
        storage: [Storage::Split, Storage::Planar, Storage::Packed][variant % 3],
        reversed: true,
        shifted: true,
    }
    .pack(format, extent, words, 4099);
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frame sampling input with poisoned source padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

fn presented(coded: Extent2d, factor: UpsamplingFactor) -> Extent2d {
    if factor == UpsamplingFactor::One {
        return coded;
    }
    // Both axes exercise ceil division, with a one-sample axis remaining one pixel.
    Extent2d::new(
        if coded.width == 1 {
            1
        } else {
            coded.width * factor.factor() - 1
        },
        if coded.height == 1 {
            1
        } else {
            coded.height * factor.factor() - 1
        },
    )
}

fn check_header(
    bytes: &[u8],
    index: usize,
    extent: Extent2d,
    coded: Extent2d,
    factor: UpsamplingFactor,
) {
    let image = jxl_oxide::JxlImage::read_with_defaults(bytes).unwrap();
    let frame = image.frame(index).unwrap().header();
    assert_eq!((frame.width, frame.height), (extent.width, extent.height));
    assert_eq!(
        (frame.color_sample_width(), frame.color_sample_height()),
        (coded.width, coded.height)
    );
    assert_eq!(frame.upsampling, factor.factor());
}

fn color_request() -> GpuOutputRequest {
    GpuOutputRequest::color(PixelFormat::rgb_f32(
        jxl_gpu_formats::RgbChannelOrder::Rgba,
        false,
        jxl_wgpu_decode::vardct_rgb8_format().color_spec,
    ))
    .unwrap()
    .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
}

fn native(bytes: &[u8]) -> Vec<f32> {
    extra_channels::libjxl_output(
        bytes,
        &["--original", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("required libjxl 0.12.0 original output")
}

fn check_color(actual: &[u8], expected: &[f32], label: &str) {
    let actual = extra_channels::floats(actual);
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
            "{label}/{i}: {a} vs {b}"
        );
    }
}

fn progression() -> ProgressivePlan {
    ProgressivePlan::new(
        [(2, 2), (4, 0), (8, 0)]
            .into_iter()
            .map(|(square, shift)| ProgressivePass {
                coefficient_square: NonZeroU8::new(square).unwrap(),
                shift,
            })
            .collect(),
    )
    .unwrap()
    .with_downsampling(vec![
        ProgressiveDownsampling {
            factor: 4,
            last_pass: 0,
        },
        ProgressiveDownsampling {
            factor: 2,
            last_pass: 1,
        },
    ])
    .unwrap()
}
