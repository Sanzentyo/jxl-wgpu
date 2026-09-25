//! Gray is an image contract with one logical component, not an RGB packing special case.
use super::*;
use crate::{
    AnimationHeader, BackendError, ColorChannels, ColorSampleFormat, GpuEncodeBackend,
    GpuEncodeJob, GpuFrameSource, VarDctBackend,
};
use jxl_gpu_formats::{ByteOrder, Channel, PackingFieldKind, Swizzle, SwizzleComponent};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use layouts::Topology;

fn configuration(sample_format: ColorSampleFormat, color: VarDctColorTransform) -> VarDctConfig {
    VarDctConfig {
        sample_format,
        ..precision::configuration(8, color)
    }
}

pub(super) fn pixels(extent: Extent2d, format: ColorSampleFormat) -> Vec<u32> {
    let (w, h) = (extent.width as usize, extent.height as usize);
    let rgb = match format.float_precision() {
        Some(p) => floating::pixels(w, h, p),
        None => precision::pixels(w, h, format.bits_per_sample(), 91),
    };
    rgb.into_iter().map(|p| p[0]).collect()
}

fn normalized(input: &[u32], format: ColorSampleFormat) -> Vec<[f64; 3]> {
    input
        .iter()
        .map(|&v| {
            [match format.float_precision() {
                Some(p) => floating::value(v, p),
                None => f64::from(v) / ((1u64 << format.bits_per_sample()) - 1) as f64,
            }; 3]
        })
        .collect()
}

pub(super) fn upload(
    context: &WgpuContext,
    extent: Extent2d,
    format: ColorSampleFormat,
    input: &[u32],
    shifted: bool,
) -> BufferImageSource {
    upload_packing(
        context,
        extent,
        format,
        input,
        Packing {
            storage: Storage::Packed,
            reversed: false,
            shifted,
        },
        if shifted {
            ByteOrder::Big
        } else {
            ByteOrder::Native
        },
        if shifted { Channel::W } else { Channel::X },
    )
}

fn upload_packing(
    context: &WgpuContext,
    extent: Extent2d,
    format: ColorSampleFormat,
    input: &[u32],
    packing: Packing,
    byte_order: ByteOrder,
    channel: Channel,
) -> BufferImageSource {
    assert_eq!(format.channels(), ColorChannels::Gray);
    let mut pixel_format = format.pixel_format();
    pixel_format.byte_order = byte_order;
    let (mut layout, bytes) = packing.pack(pixel_format, extent, input, 4099);
    if channel != Channel::X {
        for field in &mut layout.format.planes[0].words[0].fields {
            if field.kind == PackingFieldKind::Channel(Channel::X) {
                field.kind = PackingFieldKind::Channel(channel);
            }
        }
        layout.format.swizzle = Swizzle::Xyzw([
            match channel {
                Channel::Y => SwizzleComponent::Y,
                Channel::Z => SwizzleComponent::Z,
                Channel::W => SwizzleComponent::W,
                _ => unreachable!(),
            },
            SwizzleComponent::Zero,
            SwizzleComponent::Zero,
            SwizzleComponent::One,
        ]);
    }
    let buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gray source with poisoned row/word padding"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    BufferImageSource::new(Arc::new(buffer), layout).unwrap()
}

fn all_formats() -> impl Iterator<Item = ColorSampleFormat> {
    (1..=31)
        .map(|bits| ColorSampleFormat::integer(ColorChannels::Gray, bits).unwrap())
        .chain(floating::all_precisions().into_iter().map(|p| {
            ColorSampleFormat::float(ColorChannels::Gray, p.bits(), p.exponent_bits()).unwrap()
        }))
}

fn single(
    context: &WgpuContext,
    config: &VarDctConfig,
    strategy: VarDctStrategy,
    oracle: &native::Oracle,
    input: &[u32],
) -> Vec<u8> {
    let extent = strategy.pixel_extent();
    let components: Vec<_> = normalized(input, config.sample_format)
        .into_iter()
        .map(|v| match config.color_transform {
            VarDctColorTransform::Xyb => reference::xyb_normalized(v),
            VarDctColorTransform::Original => v,
        })
        .collect();
    let coefficients = native::forward_samples(
        &components,
        extent.width as usize,
        extent.height as usize,
        oracle,
    );
    let backend = VarDctBackend::new_with_config(context, strategy, config.clone()).unwrap();
    let source = upload(context, extent, config.sample_format, input, false);
    let plan = backend.memory_plan(&source).unwrap();
    // Three working components alias one physical channel; caller bytes are counted once.
    assert_eq!(plan.source_binding_bytes, source.buffer.size());
    let (ac, bits, artifacts) = backend
        .submit(
            context,
            GpuFrameSource::Buffer(source),
            &layouts::request(extent, config),
        )
        .unwrap()
        .wait_with_ac_for_test()
        .unwrap();
    native::check_ac(&ac, bits, &coefficients, oracle, config.clone());
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

fn check_header(bytes: &[u8], format: ColorSampleFormat, color: VarDctColorTransform) {
    let inventory = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert!(inventory.image_header.grayscale);
    assert_eq!(inventory.image_header.bit_depth, format.bit_depth());
    assert_eq!(
        inventory.image_header.xyb_encoded,
        color == VarDctColorTransform::Xyb
    );
    assert!(!inventory.image_header.modular_16bit_buffers);
    assert!(inventory.frames.iter().all(|frame| !frame.do_ycbcr));
}

#[test]
fn gray_input_all_precisions_retain_metadata_coefficients_pixels_and_layouts() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let oracle = color::PixelOracles::new(&gpu);
    let native = native::native_oracles();
    for format in all_formats() {
        for color in [VarDctColorTransform::Xyb, VarDctColorTransform::Original] {
            for (topology, extent) in [
                (Topology::Single, Extent2d::new(8, 8)),
                (Topology::Map, Extent2d::new(25, 17)),
                (Topology::Tiled, Extent2d::new(259, 3)),
            ] {
                let input = pixels(extent, format);
                let mut config = configuration(format, color);
                if !matches!(topology, Topology::Single) {
                    config.progressive = progressive::combined();
                    config.group_order = crate::VarDctGroupOrder::saliency_first();
                }
                let backend = topology.backend(&context, extent, &config);
                let bytes = if matches!(topology, Topology::Single) {
                    single(&context, &config, VarDctStrategy::Dct8, &native[0], &input)
                } else {
                    layouts::encode(
                        &context,
                        &backend,
                        &config,
                        upload(&context, extent, format, &input, false),
                    )
                };
                check_header(&bytes, format, color);
                let actual = oracle.check_decoders(&bytes);
                precision::check_normalized_quality(&actual, &normalized(&input, format));
                assert_eq!(
                    layouts::encode(
                        &context,
                        &backend,
                        &config,
                        upload(&context, extent, format, &input, true)
                    ),
                    bytes,
                    "{format:?}/{topology:?}/{color:?}"
                );
            }
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

#[test]
fn gray_input_all_strategies_preserve_orders_and_lf_metadata() {
    let gpu = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let context = WgpuContext::from_backend(&gpu);
    let pixels_oracle = color::PixelOracles::new(&gpu);
    let native = native::native_oracles();
    for (format, color) in [
        (
            ColorSampleFormat::integer(ColorChannels::Gray, 31).unwrap(),
            VarDctColorTransform::Original,
        ),
        (
            ColorSampleFormat::float(ColorChannels::Gray, 16, 5).unwrap(),
            VarDctColorTransform::Xyb,
        ),
        (
            ColorSampleFormat::float(ColorChannels::Gray, 32, 8).unwrap(),
            VarDctColorTransform::Original,
        ),
    ] {
        for (strategy, oracle) in VarDctStrategy::ALL.into_iter().zip(&native) {
            eprintln!("{format:?}/{color:?}/{strategy:?}");
            let input = pixels(strategy.pixel_extent(), format);
            let config = VarDctConfig {
                coefficient_orders: orders::selected([strategy]),
                lf_metadata: custom_lf_metadata(),
                quantization: VarDctQuantization::new(
                    65536,
                    256,
                    crate::VarDctHfMultiplier::new(256).unwrap(),
                )
                .unwrap(),
                ..configuration(format, color)
            };
            let bytes = single(&context, &config, strategy, oracle, &input);
            check_header(&bytes, format, color);
            let actual = pixels_oracle.check_decoders(&bytes);
            precision::check_normalized_quality(&actual, &normalized(&input, format));
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}

mod boundaries;
