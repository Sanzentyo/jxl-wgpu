#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_protocol::Extent2d;
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::oracles::{extra_channels, modular_integer, modular_words};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_encode::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;

mod boundaries;
mod sampling;
mod sequence;

#[test]
fn scalar_inputs_keep_each_color_topology_and_main_precision() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(7, 5);
    let definitions = vec![definition(13, 1)];
    for (format, bits, exponent) in [
        (LosslessModularFormat::Gray, 16, 5),
        (LosslessModularFormat::GrayAlpha, 31, 0),
        (LosslessModularFormat::Rgb, 8, 0),
        (LosslessModularFormat::Rgba, 10, 3),
    ] {
        let (base, mut expected) = inputs(
            &context,
            extent,
            &definitions,
            extent,
            &[UpsamplingFactor::One],
        );
        let scalar = base.extra_channels()[0].clone();
        let extra = expected.pop().unwrap();
        let channels = format.channel_count() as usize;
        let values: Vec<_> = (0..extent.width * extent.height * channels as u32)
            .map(|v| v.wrapping_mul(772_113) & (u32::MAX >> (32 - bits)))
            .collect();
        let pixel_format = if exponent == 0 {
            format.pixel_format(bits).unwrap()
        } else {
            format.custom_float_pixel_format(
                jxl_gpu_formats::FloatPrecision::new(bits, exponent).unwrap(),
            )
        };
        let source = source(&context, extent, pixel_format, &values)
            .with_extra_channels(vec![scalar])
            .unwrap();
        let mut expected: Vec<_> = (0..channels)
            .map(|c| modular_integer::ExtraWords {
                width: extent.width,
                height: extent.height,
                words: values.iter().skip(c).step_by(channels).copied().collect(),
            })
            .collect();
        expected.push(extra);
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let encoder = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    extra_channels: definitions.clone(),
                    entropy,
                    ..Default::default()
                },
            );
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(&modular_words::channel_frames(&bytes)[0], &expected);
            assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
        }
    }
}

fn source(
    context: &WgpuContext,
    extent: Extent2d,
    format: jxl_gpu_formats::PixelFormat,
    words: &[u32],
) -> BufferImageSource {
    let (layout, bytes) = Packing {
        storage: Storage::Packed,
        reversed: true,
        shifted: true,
    }
    .pack(format, extent, words, 4099);
    BufferImageSource::new(
        Arc::new(
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Modular independent source"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

fn definition(bits: u8, shift: u8) -> ExtraChannel {
    ExtraChannel::new(
        ExtraChannelKind::Depth,
        SamplePrecision::integer(bits).unwrap(),
        shift,
        format!("depth-{bits}-{shift}").into_bytes(),
    )
    .unwrap()
}

#[test]
fn modular_extras_share_transforms_and_entropy_with_packed_alpha() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(17, 13);
    let definitions = vec![definition(13, 0), definition(31, 0), definition(7, 0)];
    let main: Vec<_> = (0..extent.width * extent.height * 4)
        .map(|v| v % 8 * 17)
        .collect();
    let mut expected: Vec<Vec<u32>> = (0..4)
        .map(|channel| main.iter().skip(channel).step_by(4).copied().collect())
        .collect();
    let mut inputs = Vec::new();
    for (index, d) in definitions.iter().enumerate() {
        let words: Vec<_> = (0..extent.width * extent.height)
            .map(|v| v.wrapping_mul(372_143 + index as u32) & ((1u32 << [13, 31, 7][index]) - 1))
            .collect();
        inputs.push(source(
            &context,
            extent,
            d.precision().pixel_format(),
            &words,
        ));
        expected.push(words);
    }
    let source = source(
        &context,
        extent,
        LosslessModularFormat::Rgba.pixel_format(8).unwrap(),
        &main,
    )
    .with_extra_channels(inputs)
    .unwrap();
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        for squeeze in [
            LosslessModularSqueeze::None,
            LosslessModularSqueeze::HorizontalThenVertical,
        ] {
            let config = LosslessModularConfig {
                extra_channels: definitions.clone(),
                entropy,
                local_transforms: squeeze.into(),
                ..Default::default()
            };
            let encoder = LosslessModularEncoder::with_config(context.clone(), config);
            let bytes = encoder.encode(source.clone()).unwrap();
            let actual = modular_integer::original_planes(&bytes, 0);
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(
                    actual.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    *expected
                );
            }
            let physical = modular_integer::modular_channel_words(&bytes, 0);
            assert_eq!(modular_words::channel_frames(&bytes), vec![physical]);
            let native = extra_channels::libjxl_output(&bytes, &["--original", "--preserve-alpha"])
                .expect("required libjxl 0.12 output");
            assert!(!native.is_empty());
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

fn inputs(
    context: &WgpuContext,
    color_extent: Extent2d,
    definitions: &[ExtraChannel],
    displayed: Extent2d,
    factors: &[UpsamplingFactor],
) -> (BufferImageSource, Vec<modular_integer::ExtraWords>) {
    let color: Vec<_> = (0..color_extent.width * color_extent.height * 3)
        .map(|v| (v * 13) % 256)
        .collect();
    let mut expected: Vec<_> = (0..3)
        .map(|c| modular_integer::ExtraWords {
            width: color_extent.width,
            height: color_extent.height,
            words: color.iter().skip(c).step_by(3).copied().collect(),
        })
        .collect();
    let mut attachments = Vec::new();
    for (index, (d, factor)) in definitions.iter().zip(factors).enumerate() {
        let extent = d.source_extent_with_upsampling(displayed, *factor);
        let precision = d.precision().color(ColorChannels::Gray);
        let bits = precision.bits_per_sample();
        let mask = u32::MAX >> (32 - bits);
        let words: Vec<_> = (0..extent.width * extent.height)
            .map(|v| {
                if precision.exponent_bits() == 8 && bits == 32 {
                    [
                        0,
                        0x8000_0000,
                        0x3f80_0000,
                        0x7f80_0000,
                        0xff80_0000,
                        0x7fc0_0123,
                        1,
                    ][v as usize % 7]
                } else {
                    v.wrapping_mul(372_143 + index as u32) & mask
                }
            })
            .collect();
        let mut format = d.precision().pixel_format();
        format.byte_order = if index % 2 == 0 {
            jxl_gpu_formats::ByteOrder::Big
        } else {
            jxl_gpu_formats::ByteOrder::Little
        };
        attachments.push(source(context, extent, format, &words));
        expected.push(modular_integer::ExtraWords {
            width: extent.width,
            height: extent.height,
            words,
        });
    }
    (
        source(
            context,
            color_extent,
            ColorSampleFormat::RGB8.pixel_format(),
            &color,
        )
        .with_extra_channels(attachments)
        .unwrap(),
        expected,
    )
}

#[test]
fn physical_grids_follow_global_lf_and_pass_routes_with_exact_float_words() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let definitions = vec![
        definition(31, 0),
        ExtraChannel::new(
            ExtraChannelKind::Optional,
            SamplePrecision::float(32, 8).unwrap(),
            1,
            Vec::new(),
        )
        .unwrap(),
        definition(13, 3),
        definition(1, 2),
    ];
    for extent in [
        Extent2d::new(17, 9),
        Extent2d::new(259, 131),
        Extent2d::new(2051, 9),
        Extent2d::new(9, 2051),
    ] {
        let (source, expected) = inputs(
            &context,
            extent,
            &definitions,
            extent,
            &[UpsamplingFactor::One; 4],
        );
        for entropy in [
            LosslessModularEntropyCoding::Prefix,
            LosslessModularEntropyCoding::Ans,
        ] {
            let config = LosslessModularConfig {
                extra_channels: definitions.clone(),
                entropy,
                group_size: LosslessModularGroupSize::Pixels128,
                ..Default::default()
            };
            let encoder = LosslessModularEncoder::with_config(context.clone(), config);
            let bytes = encoder.encode(source.clone()).unwrap();
            assert_eq!(
                &modular_words::channel_frames(&bytes)[0],
                &expected,
                "{extent:?}/{entropy:?}"
            );
            assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn every_scalar_precision_fits_the_full_channel_declaration() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let mut definitions: Vec<_> = (1..=31)
        .map(|bits| SamplePrecision::integer(bits).unwrap())
        .chain((2..=8).flat_map(|exponent| {
            (2..=23).map(move |fraction| {
                SamplePrecision::float(1 + exponent + fraction, exponent).unwrap()
            })
        }))
        .enumerate()
        .map(|(index, precision)| {
            ExtraChannel::new(
                ExtraChannelKind::Optional,
                precision,
                0,
                format!("sample-{index}\0λ").into_bytes(),
            )
            .unwrap()
        })
        .collect();
    assert_eq!(definitions.len(), 185);
    definitions.resize(256, definition(8, 0));
    let extent = Extent2d::new(3, 2);
    let (source, expected) = inputs(
        &context,
        extent,
        &definitions,
        extent,
        &[UpsamplingFactor::One; 256],
    );
    for entropy in [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ] {
        let encoder = LosslessModularEncoder::with_config(
            context.clone(),
            LosslessModularConfig {
                extra_channels: definitions.clone(),
                entropy,
                ..Default::default()
            },
        );
        let bytes = encoder.encode(source.clone()).unwrap();
        assert_eq!(&modular_words::channel_frames(&bytes)[0], &expected);
        assert_eq!(modular_integer::modular_channel_words(&bytes, 0), expected);
    }
}
