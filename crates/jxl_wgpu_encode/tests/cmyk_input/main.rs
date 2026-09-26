#![cfg(not(target_arch = "wasm32"))]

use jxl_gpu_formats::{ByteOrder, ColorSpecification};
use jxl_gpu_protocol::{Extent2d, icc::IccProfile};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::oracles::{modular_integer, modular_words};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_encode::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;

mod boundaries;
mod precision;
mod sequence;

fn profile(name: &str) -> IccProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jxl_wgpu/test-data/icc/lut")
        .join(format!("{name}.icc"));
    IccProfile::parse(std::fs::read(path).unwrap().into(), Default::default()).unwrap()
}

fn config(profile: &IccProfile, bits: u8, exponent: u8, alpha: bool) -> VarDctConfig {
    VarDctConfig {
        sample_format: if exponent == 0 {
            ColorSampleFormat::integer(ColorChannels::Rgb, bits)
        } else {
            ColorSampleFormat::float(ColorChannels::Rgb, bits, exponent)
        }
        .unwrap(),
        alpha: alpha.then_some(AlphaAssociation::Unassociated),
        source_color: ColorSpecification::Icc(profile.clone()),
        image_options: ImageOptions {
            rendering_intent: profile.header().rendering_intent,
            ..Default::default()
        },
        color_transform: VarDctColorTransform::Original,
        ..Default::default()
    }
}

fn input(
    context: &WgpuContext,
    extent: Extent2d,
    config: &VarDctConfig,
    storage: Storage,
    encoding: CmykSampleEncoding,
    words: &[u32],
) -> BufferImageSource {
    raw_source(context, extent, config.pixel_format(), storage, words)
        .with_cmyk_encoding(encoding)
        .unwrap()
}

fn raw_source(
    context: &WgpuContext,
    extent: Extent2d,
    mut format: jxl_gpu_formats::PixelFormat,
    storage: Storage,
    words: &[u32],
) -> BufferImageSource {
    format.byte_order = ByteOrder::Big;
    let (layout, bytes) = Packing {
        storage,
        reversed: true,
        shifted: true,
    }
    .pack(format, extent, words, 1031);
    BufferImageSource::new(
        Arc::new(
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("CMYK source with poisoned padding"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

fn expected(
    extent: Extent2d,
    words: &[u32],
    bits: u8,
    alpha: bool,
    encoding: CmykSampleEncoding,
) -> Vec<modular_integer::ExtraWords> {
    let count = 4 + usize::from(alpha);
    let order: &[usize] = if alpha {
        &[0, 1, 2, 4, 3]
    } else {
        &[0, 1, 2, 3]
    };
    let mask = u32::MAX >> (32 - bits);
    order
        .iter()
        .map(|&component| modular_integer::ExtraWords {
            width: extent.width,
            height: extent.height,
            words: words
                .iter()
                .skip(component)
                .step_by(count)
                .map(|&word| {
                    if component < 4 && encoding == CmykSampleEncoding::InkAmounts {
                        mask - word
                    } else {
                        word
                    }
                })
                .collect(),
        })
        .collect()
}

#[test]
fn modular_cmyk_views_preserve_all_words_and_primary_alpha() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    let extent = Extent2d::new(17, 9);
    for alpha in [false, true] {
        let config = config(&profile, 13, 0, alpha);
        let words: Vec<_> = (0..extent.area().unwrap() * (4 + usize::from(alpha)))
            .map(|i| (i as u32 * 3791) & 8191)
            .collect();
        for storage in [Storage::Packed, Storage::Planar, Storage::Split] {
            for encoding in [
                CmykSampleEncoding::InkAmounts,
                CmykSampleEncoding::Complemented,
            ] {
                let source = input(&context, extent, &config, storage, encoding, &words);
                let reference = expected(extent, &words, 13, alpha, encoding);
                for entropy in [
                    LosslessModularEntropyCoding::Prefix,
                    LosslessModularEntropyCoding::Ans,
                ] {
                    let encoder = LosslessModularEncoder::with_config(
                        context.clone(),
                        LosslessModularConfig {
                            entropy,
                            ..Default::default()
                        },
                    )
                    .with_image_options(config.image_options)
                    .unwrap();
                    let encoded = encoder.encode(source.clone()).unwrap();
                    assert_eq!(modular_words::channel_frames(&encoded)[0], reference);
                    assert_eq!(
                        modular_integer::modular_channel_words(&encoded, 0),
                        reference
                    );
                }
            }
        }
    }
}

#[test]
fn vardct_cmyk_original_and_xyb_preserve_the_exact_black_plane() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let profile = profile("lut16_xyz_4");
    let extent = Extent2d::new(17, 9);
    for alpha in [false, true] {
        let mut config = config(&profile, 8, 0, alpha);
        let words: Vec<_> = (0..extent.area().unwrap() * (4 + usize::from(alpha)))
            .map(|i| (i as u32 * 7) % 251)
            .collect();
        let source = input(
            &context,
            extent,
            &config,
            Storage::Planar,
            CmykSampleEncoding::InkAmounts,
            &words,
        );
        let reference = expected(extent, &words, 8, alpha, CmykSampleEncoding::InkAmounts);
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            config.color_transform = transform;
            let encoder =
                TiledVarDctEncoder::new_with_config(context.clone(), config.clone()).unwrap();
            let encoded = encoder.encode(source.clone()).unwrap();
            assert_eq!(
                modular_integer::vardct_extra_words(&encoded, 0),
                reference[3..]
            );
        }
    }
}
