#![cfg(not(target_arch = "wasm32"))]

mod alpha;
mod animation;
mod color;
mod groups;
mod icc;
mod lifetime;
mod predictors;
mod rct;
mod source;

use std::num::NonZeroU64;
use std::sync::Arc;

use jxl_gpu_formats::{ByteOrder, PixelFormat, RgbChannelOrder, SampleKind};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::gpu::planes::{open_fragmented, read, read_bytes};
use jxl_test_support::oracles::{extra_channels, modular_integer};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    AlphaOutputPolicy, GpuDecoder, GpuOutputRequest, NumericSampleMapping, WgpuDecodeEngine,
};
use jxl_wgpu_encode::{
    BufferImageSource, EncodeError, LosslessModularEncoder, LosslessModularFormat,
    LosslessModularTreeMode, WgpuContext,
};
use source::{Case, Storage, upload};

struct Rig {
    backend: WgpuBackend,
    context: WgpuContext,
    decoders: [GpuDecoder<WgpuDecodeEngine>; 2],
}

impl Rig {
    fn new() -> Self {
        let backend = pollster::block_on(WgpuBackend::request_default(Default::default()))
            .expect("required GPU adapter");
        let context = WgpuContext::from_backend(&backend);
        let decoders = [
            GpuDecoder::wgpu(backend.clone()).unwrap(),
            GpuDecoder::new(
                WgpuDecodeEngine::new(backend.clone())
                    .unwrap()
                    .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
            ),
        ];
        Self {
            backend,
            context,
            decoders,
        }
    }

    fn check(&self, encoder: &LosslessModularEncoder, case: &Case, extent: Extent2d) {
        let expected = case.samples(extent);
        let input = upload(&self.context, case, extent, &expected, 4099);
        let plan = encoder.memory_plan(&input).unwrap();
        assert_eq!(plan.format, case.format);
        assert_eq!(plan.bits_per_sample, case.bits);
        assert_eq!(
            plan.exponent_bits_per_sample != 0,
            case.kind == SampleKind::Float
        );
        let encoded = pollster::block_on(encoder.submit_container(input).unwrap()).unwrap();
        let canonical = upload(&self.context, &case.canonical(), extent, &expected, 0);
        assert_eq!(
            encoded,
            encoder.encode_container(canonical).unwrap(),
            "{case:?}"
        );
        assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        let native = check_oracles(&encoded, &expected, case);
        self.check_gpu(&encoded, &expected, case, &native);
    }

    fn check_gpu(&self, encoded: &[u8], expected: &[u32], case: &Case, native: &[f32]) {
        for (decoder, fragmented) in self.decoders.iter().zip([false, true]) {
            let request = if case.kind == SampleKind::Float {
                GpuOutputRequest::color(PixelFormat::rgb_f32(
                    RgbChannelOrder::Rgba,
                    false,
                    jxl_wgpu_decode::vardct_rgb8_format().color_spec,
                ))
                .unwrap()
                .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
            } else if case.format == LosslessModularFormat::Gray {
                GpuOutputRequest::numeric(
                    case.format.pixel_format(case.bits).unwrap(),
                    NumericSampleMapping::NativeUnsigned,
                )
                .unwrap()
            } else {
                GpuOutputRequest::color(case.format.pixel_format(case.bits).unwrap()).unwrap()
            };
            let mut session = if fragmented {
                open_fragmented(decoder, encoded, request)
            } else {
                decoder.open(encoded, request).unwrap()
            };
            let frame = pollster::block_on(session.next_frame_async())
                .unwrap()
                .unwrap();
            assert!(
                pollster::block_on(session.next_frame_async())
                    .unwrap()
                    .is_none()
            );
            drop(session);
            if case.kind == SampleKind::Float {
                let pixels = expected.len() / case.format.channel_count() as usize;
                assert_eq!(
                    read(&self.backend, &frame.output().outputs[0]),
                    native[..pixels * 4]
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "GPU float {case:?}, fragmented={fragmented}"
                );
            } else {
                let bytes = read_bytes(&self.backend, &frame.output().outputs[0]);
                let width = case.bits.next_power_of_two().max(8) as usize / 8;
                let actual: Vec<u32> = bytes
                    .chunks_exact(width)
                    .map(|sample| {
                        sample.iter().enumerate().fold(0, |word, (byte, value)| {
                            word | (u32::from(*value) << (byte * 8))
                        })
                    })
                    .collect();
                assert_eq!(
                    actual, expected,
                    "GPU integer {case:?}, fragmented={fragmented}"
                );
            }
            drop(frame);
            assert_eq!(decoder.engine().in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}

fn native(encoded: &[u8]) -> Vec<f32> {
    extra_channels::libjxl_output(
        encoded,
        &["--original", "--preserve-alpha", "--keep-orientation"],
    )
    .expect("required libjxl 0.12.0 original-component oracle")
}

fn check_oracles(encoded: &[u8], expected: &[u32], case: &Case) -> Vec<f32> {
    check_frame_oracles(encoded, &[expected], case)
}

fn check_frame_oracles(encoded: &[u8], frames: &[&[u32]], case: &Case) -> Vec<f32> {
    let native = native(encoded);
    check_frame_samples(encoded, frames, case, &native);
    native
}

fn check_frame_samples(encoded: &[u8], frames: &[&[u32]], case: &Case, native: &[f32]) {
    let pixels = frames[0].len() / case.format.channel_count() as usize;
    let frame_size = pixels * (4 + usize::from(case.format.has_alpha()));
    assert_eq!(native.len(), frame_size * frames.len());
    for (frame_index, expected) in frames.iter().enumerate() {
        let channels = case.format.channel_count() as usize;
        let pixels = expected.len() / channels;
        let planes = modular_integer::original_planes(encoded, frame_index);
        assert_eq!(planes.len(), channels);
        for (channel, plane) in planes.iter().enumerate() {
            assert_eq!(plane.len(), pixels);
            for (pixel, &word) in plane.iter().enumerate() {
                assert_eq!(
                    word as u32,
                    expected[pixel * channels + channel],
                    "independent exact words {case:?}/{pixel}/{channel}"
                );
            }
        }
        let native = &native[frame_index * frame_size..(frame_index + 1) * frame_size];
        for pixel in 0..pixels {
            for channel in 0..4 {
                let value = if channel == 3 && !case.format.has_alpha() {
                    1.0f32
                } else {
                    let index = if channel == 3 {
                        channels - 1
                    } else if case.format.color_channel_count() == 1 {
                        0
                    } else {
                        channel
                    };
                    case.normalized(expected[pixel * channels + index])
                };
                let actual = native[pixel * 4 + channel];
                if case.kind == SampleKind::Float {
                    assert_eq!(
                        actual.to_bits(),
                        value.to_bits(),
                        "native IEEE {case:?}/{pixel}/{channel}"
                    );
                } else {
                    assert!(
                        (actual - value).abs() <= 2e-7,
                        "native integer {case:?}/{pixel}/{channel}"
                    );
                }
            }
            if case.format.has_alpha() {
                let value = case.normalized(expected[pixel * channels + channels - 1]);
                let actual = native[pixels * 4 + pixel];
                if case.kind == SampleKind::Float {
                    assert_eq!(actual.to_bits(), value.to_bits());
                } else {
                    assert!((actual - value).abs() <= 2e-7);
                }
            }
        }
    }
}

const TREES: [LosslessModularTreeMode; 2] = [
    LosslessModularTreeMode::SharedGlobal,
    LosslessModularTreeMode::LocalPerGroup,
];

#[test]
fn every_integer_depth_keeps_exact_words_across_storage_layouts() {
    let rig = Rig::new();
    for tree in TREES {
        let encoder = LosslessModularEncoder::with_tree_mode(rig.context.clone(), tree);
        for bits in 1..=31 {
            let case = Case {
                format: [
                    LosslessModularFormat::Gray,
                    LosslessModularFormat::Rgb,
                    LosslessModularFormat::Rgba,
                ][bits as usize % 3],
                bits,
                kind: SampleKind::Unsigned,
                storage: [Storage::Planar, Storage::Split, Storage::Packed]
                    [(bits as usize / 3) % 3],
                reversed: true,
                byte_order: if bits % 2 == 0 {
                    ByteOrder::Big
                } else {
                    ByteOrder::Little
                },
                shifted: true,
            };
            let (width, height) = [(1, 257), (255, 2), (256, 1), (257, 3)][bits as usize % 4];
            rig.check(&encoder, &case, Extent2d::new(width, height));
        }
    }
}

#[test]
fn ieee_words_survive_swizzles_planar_storage_and_endianness() {
    let rig = Rig::new();
    for tree in TREES {
        let encoder = LosslessModularEncoder::with_tree_mode(rig.context.clone(), tree);
        for format in [
            LosslessModularFormat::Gray,
            LosslessModularFormat::Rgb,
            LosslessModularFormat::Rgba,
        ] {
            for bits in [16, 32] {
                for storage in [Storage::Planar, Storage::Split] {
                    let case = Case {
                        format,
                        bits,
                        kind: SampleKind::Float,
                        storage,
                        reversed: true,
                        byte_order: ByteOrder::Big,
                        shifted: true,
                    };
                    rig.check(&encoder, &case, Extent2d::new(257, 3));
                }
            }
        }
    }
}

#[test]
fn shared_words_and_mixed_storage_widths_preserve_channel_and_padding_boundaries() {
    let rig = Rig::new();
    for tree in TREES {
        let encoder = LosslessModularEncoder::with_tree_mode(rig.context.clone(), tree);
        for (format, bits, storage) in [
            (LosslessModularFormat::Rgb, 10, Storage::SharedWord),
            (LosslessModularFormat::Rgba, 7, Storage::SharedWord),
            (LosslessModularFormat::Rgb, 24, Storage::ThreeBytes),
            (LosslessModularFormat::Rgba, 8, Storage::MixedWords),
        ] {
            for byte_order in [ByteOrder::Little, ByteOrder::Big] {
                let case = Case {
                    format,
                    bits,
                    kind: SampleKind::Unsigned,
                    storage,
                    reversed: true,
                    byte_order,
                    shifted: true,
                };
                rig.check(&encoder, &case, Extent2d::new(257, 3));
            }
        }
    }
}
