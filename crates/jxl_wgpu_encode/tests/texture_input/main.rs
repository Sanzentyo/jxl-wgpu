#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use jxl_gpu_formats::{ImageLayout, PixelFormat};
use jxl_gpu_protocol::Extent2d;
use jxl_test_support::oracles::{modular_integer, modular_words};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_encode::*;
use wgpu::util::DeviceExt;

mod boundaries;
mod separate_planes;
mod sequence;

fn buffer(
    context: &WgpuContext,
    extent: Extent2d,
    format: PixelFormat,
    bytes: &[u8],
) -> BufferImageSource {
    let layout = ImageLayout::packed(extent, format).unwrap();
    assert_eq!(bytes.len() as u64, layout.logical_size);
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().div_ceil(4) * 4, 0xa7);
    BufferImageSource::new(
        Arc::new(
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("texture test buffer reference"),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

fn texture(
    context: &WgpuContext,
    extent: Extent2d,
    format: PixelFormat,
    storage: wgpu::TextureFormat,
    bytes: &[u8],
) -> TextureImageSource {
    // Distinct neighboring subresources expose wrong mip/layer selection. The selected mip's
    // odd dimensions also exercise the copy's 256-byte rows and final partial storage word.
    let texture = Arc::new(context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("texture input with poisoned neighboring subresources"),
        size: wgpu::Extent3d {
            width: extent.width * 2,
            height: extent.height * 2,
            depth_or_array_layers: 3,
        },
        mip_level_count: 2,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: storage,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    }));
    let texel = storage.block_copy_size(None).unwrap();
    for mip in 0..2 {
        let width = extent.width * (2 >> mip);
        let height = extent.height * (2 >> mip);
        for layer in 0..3 {
            let poison = vec![0x53 + layer as u8; (width * height * texel) as usize];
            let selected = mip == 1 && layer == 1;
            context.queue().write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                if selected { bytes } else { &poison },
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * texel),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    TextureImageSource::new(texture, storage, format, 1, 1).unwrap()
}

fn bytes(words: &[u32], stride: usize) -> Vec<u8> {
    words
        .iter()
        .flat_map(|word| word.to_le_bytes().into_iter().take(stride))
        .collect()
}

fn planes(extent: Extent2d, words: &[u32], channels: usize) -> Vec<modular_integer::ExtraWords> {
    (0..channels)
        .map(|component| modular_integer::ExtraWords {
            width: extent.width,
            height: extent.height,
            words: words
                .iter()
                .skip(component)
                .step_by(channels)
                .copied()
                .collect(),
        })
        .collect()
}

#[test]
fn copied_mips_preserve_integer_and_float_words_without_texture_color_conversion() {
    use wgpu::TextureFormat as T;
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(9, 5);
    let encoders = [
        LosslessModularEntropyCoding::Prefix,
        LosslessModularEntropyCoding::Ans,
    ]
    .map(|entropy| {
        LosslessModularEncoder::with_config(
            context.clone(),
            LosslessModularConfig {
                entropy,
                ..Default::default()
            },
        )
    });
    // Native normalized formats are byte carriers here; sRGB must not pass through a sampler.
    for (storage, bits, exponent, channels, alpha) in [
        (T::R8Unorm, 8, 0, ColorChannels::Gray, false),
        (T::Rg8Uint, 8, 0, ColorChannels::Gray, true),
        (T::Rgba8UnormSrgb, 8, 0, ColorChannels::Rgb, true),
        (T::Bgra8UnormSrgb, 8, 0, ColorChannels::Rgb, true),
        (T::R16Uint, 13, 0, ColorChannels::Gray, false),
        (T::Rgba16Uint, 16, 0, ColorChannels::Rgb, true),
        (T::R32Uint, 31, 0, ColorChannels::Gray, false),
        (T::R16Float, 16, 5, ColorChannels::Gray, false),
        (T::Rg16Float, 16, 5, ColorChannels::Gray, true),
        (T::Rgba16Float, 16, 5, ColorChannels::Rgb, true),
        (T::R32Float, 32, 8, ColorChannels::Gray, false),
        (T::Rgba32Float, 32, 8, ColorChannels::Rgb, true),
    ] {
        let config = VarDctConfig {
            sample_format: if exponent == 0 {
                ColorSampleFormat::integer(channels, bits)
            } else {
                ColorSampleFormat::float(channels, bits, exponent)
            }
            .unwrap(),
            alpha: alpha.then_some(AlphaAssociation::Unassociated),
            ..Default::default()
        };
        let mut format = config.pixel_format();
        let count = format.planes[0].words.len();
        let width = format.planes[0].words[0].bits() as usize / 8;
        let special: &[u32] = if bits == 16 {
            &[0, 0x8000, 1, 0x3c00, 0x7c00, 0xfc00, 0x7e31, 0xfe57]
        } else {
            &[
                0, 0x80000000, 1, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc23456,
            ]
        };
        let words: Vec<_> = (0..extent.area().unwrap() * count)
            .map(|i| {
                if exponent == 0 {
                    (i as u32).wrapping_mul(741103597) & (u32::MAX >> (32 - bits))
                } else {
                    special[i % special.len()]
                }
            })
            .collect();
        let mut stored = words.clone();
        if storage == T::Bgra8UnormSrgb {
            format = PixelFormat::rgb8(
                jxl_gpu_formats::RgbChannelOrder::Bgra,
                false,
                format.color_spec,
            );
            for pixel in stored.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        let raw = bytes(&stored, width);
        let input = texture(&context, extent, format.clone(), storage, &raw);
        let canonical = buffer(&context, extent, format, &raw);
        let expected = planes(extent, &words, count);
        for encoder in &encoders {
            let plan = encoder.memory_plan(&input).unwrap();
            assert_eq!(
                plan.source_copy_bytes,
                ((extent.height as u64 - 1) * 256 + raw.len() as u64 / extent.height as u64)
                    .div_ceil(4)
                    * 4
            );
            assert_eq!(plan.source_texture_bytes, raw.len() as u64);
            assert_eq!(plan.source_binding_bytes, 0);
            let encoded = encoder.encode(input.clone()).unwrap();
            assert_eq!(
                encoded,
                encoder.encode(canonical.clone()).unwrap(),
                "{storage:?}"
            );
            assert_eq!(
                modular_words::channel_frames(&encoded)[0],
                expected,
                "native {storage:?}"
            );
            assert_eq!(
                modular_integer::modular_channel_words(&encoded, 0),
                expected,
                "Rust {storage:?}"
            );
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn texture_vardct_and_cmyk_share_exact_buffer_results_and_scalar_inputs() {
    use jxl_gpu_formats::ColorSpecification;
    use jxl_gpu_protocol::icc::IccProfile;
    use wgpu::TextureFormat as T;
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let extent = Extent2d::new(17, 9);
    let profile = IccProfile::parse(
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../jxl_wgpu/test-data/icc/lut/lut16_xyz_4.icc"),
        )
        .unwrap()
        .into(),
        Default::default(),
    )
    .unwrap();
    for cmyk in [false, true] {
        for float in [false, true] {
            let precision = if float {
                ColorSampleFormat::float(ColorChannels::Rgb, 32, 8)
            } else {
                ColorSampleFormat::integer(ColorChannels::Rgb, 8)
            }
            .unwrap();
            let definition = ExtraChannel::new(
                ExtraChannelKind::Depth,
                SamplePrecision::integer(13).unwrap(),
                1,
                b"depth".to_vec(),
            )
            .unwrap();
            let config = VarDctConfig {
                sample_format: precision,
                alpha: (!cmyk).then_some(AlphaAssociation::Unassociated),
                source_color: if cmyk {
                    ColorSpecification::Icc(profile.clone())
                } else {
                    ColorSpecification::Default
                },
                image_options: ImageOptions {
                    rendering_intent: if cmyk {
                        profile.header().rendering_intent
                    } else {
                        ImageOptions::default().rendering_intent
                    },
                    ..Default::default()
                },
                extra_channels: vec![definition.clone()],
                ..Default::default()
            };
            let words: Vec<_> = (0..extent.area().unwrap() * 4)
                .map(|i| {
                    if float {
                        ((i % 251) as f32 / 251.0).to_bits()
                    } else {
                        i as u32 * 13 % 251
                    }
                })
                .collect();
            let raw = bytes(&words, if float { 4 } else { 1 });
            let scalar_extent = definition.source_extent(extent);
            let scalar: Vec<_> = (0..scalar_extent.area().unwrap())
                .map(|i| i as u32 * 17)
                .collect();
            let extra = buffer(
                &context,
                scalar_extent,
                definition.precision().pixel_format(),
                &bytes(&scalar, 2),
            );
            let mut input = texture(
                &context,
                extent,
                config.pixel_format(),
                if float { T::Rgba32Float } else { T::Rgba8Uint },
                &raw,
            )
            .with_extra_channels(vec![extra.clone()])
            .unwrap();
            let mut canonical = buffer(&context, extent, config.pixel_format(), &raw)
                .with_extra_channels(vec![extra])
                .unwrap();
            if cmyk && float {
                input = input
                    .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                    .unwrap();
                canonical = canonical
                    .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                    .unwrap();
            }
            let mut planar = separate_planes::split(
                &context,
                extent,
                config.pixel_format(),
                &raw,
                &[2, 2],
                true,
            )
            .with_extra_channels(canonical.extra_channels().to_vec())
            .unwrap();
            if cmyk && float {
                planar = planar
                    .with_cmyk_encoding(CmykSampleEncoding::Complemented)
                    .unwrap();
            }
            let modular = LosslessModularEncoder::with_config(
                context.clone(),
                LosslessModularConfig {
                    extra_channels: config.extra_channels.clone(),
                    ..Default::default()
                },
            )
            .with_image_options(config.image_options)
            .unwrap();
            let encoded = modular.encode(input.clone()).unwrap();
            assert_eq!(encoded, modular.encode(canonical.clone()).unwrap());
            assert_eq!(encoded, modular.encode(planar.clone()).unwrap());
            let mut expected = planes(extent, &words, 4);
            if cmyk && !float {
                for plane in &mut expected {
                    for word in &mut plane.words {
                        *word = 255 - *word;
                    }
                }
            }
            expected.push(modular_integer::ExtraWords {
                width: scalar_extent.width,
                height: scalar_extent.height,
                words: scalar.clone(),
            });
            assert_eq!(modular_words::channel_frames(&encoded)[0], expected);
            for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
                let encoder = TiledVarDctEncoder::new_with_config(
                    context.clone(),
                    VarDctConfig {
                        color_transform: transform,
                        ..config.clone()
                    },
                )
                .unwrap();
                let encoded = encoder.encode(input.clone()).unwrap();
                assert_eq!(
                    encoded,
                    encoder.encode(canonical.clone()).unwrap(),
                    "{cmyk}/{float}/{transform:?}"
                );
                assert_eq!(encoded, encoder.encode(planar.clone()).unwrap());
                let expected_primary: Vec<_> = words
                    .iter()
                    .skip(3)
                    .step_by(4)
                    .map(|&v| if cmyk && !float { 255 - v } else { v })
                    .collect();
                assert_eq!(
                    modular_integer::vardct_extra_words(&encoded, 0),
                    [
                        modular_integer::ExtraWords {
                            width: extent.width,
                            height: extent.height,
                            words: expected_primary
                        },
                        modular_integer::ExtraWords {
                            width: scalar_extent.width,
                            height: scalar_extent.height,
                            words: scalar.clone()
                        },
                    ]
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
