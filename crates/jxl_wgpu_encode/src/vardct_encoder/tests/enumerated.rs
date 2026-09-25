//! One declared source color governs GPU components, stream metadata and sequence admission.
use super::*;
use crate::{
    AnimationHeader, ColorChannels, ColorSampleFormat, GpuEncodeBackend, GpuFrameSource,
    VarDctBackend,
};
use jxl_gpu_formats::{
    ColorRange, ColorSpace, ColorSpec, ColorSpecification, TransferFunction, YcbcrEncoding,
};
use jxl_gpu_protocol::icc::IccRenderingIntent;
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::oracles::icc_profile::IccProfileOracle;

mod boundaries;
mod oracle;
mod sequence;

fn input(extent: Extent2d, sample: ColorSampleFormat) -> (Vec<u32>, Vec<[f64; 3]>) {
    let mut words = Vec::new();
    let pixels = (0..extent.width * extent.height)
        .map(|i| {
            let mut rgb = [0.0; 3];
            for (c, component) in rgb
                .iter_mut()
                .enumerate()
                .take(sample.channels().count() as usize)
            {
                let value = (64 + (i * 13 + c as u32 * 37) % 128) as f64 / 255.0;
                let (word, value) = if sample.float_precision().is_some() {
                    assert_eq!(
                        sample,
                        ColorSampleFormat::float(sample.channels(), 32, 8).unwrap()
                    );
                    ((value as f32).to_bits(), f64::from(value as f32))
                } else {
                    let max = (1u64 << sample.bits_per_sample()) - 1;
                    let word = (value * max as f64).round() as u32;
                    (word, f64::from(word) / max as f64)
                };
                words.push(word);
                *component = value;
            }
            if sample.channels() == ColorChannels::Gray {
                rgb = [rgb[0]; 3];
            }
            rgb
        })
        .collect();
    (words, pixels)
}

fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    config: &VarDctConfig,
    words: &[u32],
    alternate: bool,
) -> BufferImageSource {
    let mut format = config.pixel_format();
    if alternate {
        format.byte_order = jxl_gpu_formats::ByteOrder::Big;
    }
    let (layout, bytes) = Packing {
        storage: if alternate && config.sample_format.channels() == ColorChannels::Rgb {
            Storage::Split
        } else {
            Storage::Packed
        },
        reversed: alternate,
        shifted: alternate,
    }
    .pack(format, extent, words, 4099);
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("enumerated color, independent poisoned packing"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

fn check_header(bytes: &[u8], config: &VarDctConfig, profiles: &IccProfileOracle) {
    let header = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap()
        .image_header;
    assert_eq!(
        header.grayscale,
        config.sample_format.channels() == ColorChannels::Gray
    );
    assert_eq!(header.bit_depth, config.sample_format.bit_depth());
    assert_eq!(
        header.xyb_encoded,
        config.color_transform == VarDctColorTransform::Xyb
    );
    assert_eq!(
        header.tone_mapping.intensity_target.to_f32(),
        config.color_options.intensity_target.to_f32()
    );
    assert_eq!(
        profiles.read(bytes).profile,
        profiles.create(&oracle::declaration(config)).profile
    );
}

fn single(
    context: &WgpuContext,
    config: &VarDctConfig,
    strategy: VarDctStrategy,
    native: &native::Oracle,
) -> Vec<u8> {
    let extent = strategy.pixel_extent();
    let (words, values) = input(extent, config.sample_format);
    let components = oracle::components(&values, config);
    let coefficients = native::forward_samples(
        &components,
        extent.width as usize,
        extent.height as usize,
        native,
    );
    let backend = VarDctBackend::new_with_config(context, strategy, config.clone()).unwrap();
    let source = upload(context, extent, config, &words, false);
    let (ac, bits, artifacts) = backend
        .submit(
            context,
            GpuFrameSource::Buffer(source),
            &layouts::request(extent, config),
        )
        .unwrap()
        .wait_with_ac_for_test()
        .unwrap();
    native::check_ac(&ac, bits, &coefficients, native, config.clone());
    let mut bytes = super::super::bitstream::image_header(
        extent.width,
        extent.height,
        AnimationHeader::Still,
        backend.color_plan,
    )
    .unwrap()
    .bytes()
    .to_vec();
    bytes.extend_from_slice(assemble_frame(artifacts.packets).unwrap().bytes());
    bytes
}

#[test]
fn enumerated_colors_bind_wire_declarations_to_independent_coefficients_and_pixels() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = color::PixelOracles::new(&gpu);
    let profiles = IccProfileOracle::compile();
    let native = native::native_oracles();
    for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
        for index in 0..49 {
            for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
                let config = oracle::config(channels, index, transform);
                eprintln!("enumerated {channels:?} {index} {transform:?}");
                let encoded = single(&context, &config, VarDctStrategy::Dct8, &native[0]);
                check_header(&encoded, &config, &profiles);
                let spec = oracle::wire_spec(&config);
                let actual = pixels.check_with_reference(
                    &encoded,
                    ColorSpecification::Defined(spec),
                    &oracle::pixels(&encoded, &config),
                );
                precision::check_normalized_quality(
                    &actual,
                    &input(Extent2d::new(8, 8), config.sample_format).1,
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn enumerated_colors_keep_all_strategies_layouts_progression_and_image_white() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels = color::PixelOracles::new(&gpu);
    let native = native::native_oracles();
    for (index, (strategy, native)) in VarDctStrategy::ALL.into_iter().zip(&native).enumerate() {
        for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
            let mut config = oracle::config(
                if index % 2 == 0 {
                    ColorChannels::Gray
                } else {
                    ColorChannels::Rgb
                },
                index * 9 % 49,
                transform,
            );
            config.sample_format =
                ColorSampleFormat::float(config.sample_format.channels(), 32, 8).unwrap();
            config.coefficient_orders = orders::selected([strategy]);
            config.lf_metadata = custom_lf_metadata();
            eprintln!("enumerated {strategy:?} {transform:?}");
            let encoded = single(&context, &config, strategy, native);
            pixels.check_with_reference(
                &encoded,
                ColorSpecification::Defined(oracle::wire_spec(&config)),
                &oracle::pixels(&encoded, &config),
            );
        }
    }
    for index in 0..7 {
        for channels in [ColorChannels::Gray, ColorChannels::Rgb] {
            for transform in [VarDctColorTransform::Original, VarDctColorTransform::Xyb] {
                let mut config = oracle::config(channels, index * 8, transform);
                config.progressive = progressive::combined();
                config.group_order = crate::VarDctGroupOrder::saliency_first();
                for (topology, extent) in [
                    (layouts::Topology::Map, Extent2d::new(25, 17)),
                    (layouts::Topology::Tiled, Extent2d::new(259, 3)),
                ] {
                    let backend = topology.backend(&context, extent, &config);
                    let words = input(extent, config.sample_format).0;
                    let bytes = layouts::encode(
                        &context,
                        &backend,
                        &config,
                        upload(&context, extent, &config, &words, false),
                    );
                    assert_eq!(
                        layouts::encode(
                            &context,
                            &backend,
                            &config,
                            upload(&context, extent, &config, &words, true)
                        ),
                        bytes
                    );
                    pixels.check_with_reference(
                        &bytes,
                        ColorSpecification::Defined(oracle::wire_spec(&config)),
                        &oracle::pixels(&bytes, &config),
                    );
                }
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
