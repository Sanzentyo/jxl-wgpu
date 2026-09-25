use super::*;
use crate::{ExtraChannel, ExtraChannelKind, SamplePrecision};
use jxl_test_support::fixtures::source_layout::{Packing, Storage};
use jxl_test_support::oracles::{extra_channels, modular_integer};

mod boundaries;
mod matrix;
mod metadata;
mod sampling;
mod sequence;

fn color_source(context: &WgpuContext, extent: Extent2d) -> BufferImageSource {
    padded_rgb_source_sized(
        context,
        extent.width as usize,
        extent.height as usize,
        &reference::pattern(extent.width as usize, extent.height as usize),
    )
}

fn normalized(word: u32, precision: SamplePrecision) -> f32 {
    let format = precision.color(crate::ColorChannels::Gray);
    match format.float_precision() {
        Some(p) => f32::from_bits(jxl_test_support::oracles::sample_bits::custom_binary32(
            word,
            p.bits(),
            p.exponent_bits(),
        )),
        None => (f64::from(word) / f64::from(precision.mask())) as f32,
    }
}

fn assert_numeric(actual: &[f32], words: &[u32], precision: SamplePrecision) {
    assert_eq!(actual.len(), words.len());
    for (i, (&actual, &word)) in actual.iter().zip(words).enumerate() {
        let expected = normalized(word, precision);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else if expected == 0.0 || expected.is_infinite() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "sample {i}/{precision:?}"
            );
        } else {
            assert!(
                (actual - expected).abs()
                    <= 2.0 * f32::EPSILON * expected.abs().max(f32::MIN_POSITIVE),
                "sample {i}/{precision:?}: {actual} vs {expected}"
            );
        }
    }
}

fn check_words(
    bytes: &[u8],
    frame: usize,
    definitions: &[ExtraChannel],
    extent: Extent2d,
    words: &[Vec<u32>],
) {
    let decoded = modular_integer::vardct_extra_words(bytes, frame);
    assert_eq!(decoded.len(), definitions.len());
    for ((decoded, definition), words) in decoded.iter().zip(definitions).zip(words) {
        assert_eq!(
            Extent2d::new(decoded.width, decoded.height),
            definition.source_extent(extent)
        );
        assert_eq!(&decoded.words, words);
    }
}

fn scalar_source(
    context: &WgpuContext,
    extent: Extent2d,
    precision: SamplePrecision,
    words: &[u32],
) -> BufferImageSource {
    let mut format = precision.pixel_format();
    format.byte_order = jxl_gpu_formats::ByteOrder::Big;
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
                    label: Some("independent extra source"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        ),
        layout,
    )
    .unwrap()
}

fn declaration(kind: ExtraChannelKind, bits: u8, shift: u8) -> ExtraChannel {
    ExtraChannel::new(
        kind,
        SamplePrecision::integer(bits).unwrap(),
        shift,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn extra_input_independent_scalar_sources_form_one_global_modular_stream() {
    let context = test_context().unwrap();
    let extent = Extent2d::new(17, 13);
    let config = VarDctConfig {
        extra_channels: vec![
            declaration(ExtraChannelKind::Depth, 13, 0),
            declaration(
                ExtraChannelKind::Alpha(crate::AlphaAssociation::Unassociated),
                7,
                0,
            ),
            declaration(ExtraChannelKind::Cfa { channel: 274 }, 31, 0),
        ],
        ..Default::default()
    };
    let mut expected = Vec::new();
    let inputs = config
        .extra_channels
        .iter()
        .map(|extra| {
            let values: Vec<_> = (0..extent.area().unwrap())
                .map(|i| (i as u32 * 372_143) & extra.precision().mask())
                .collect();
            let source = scalar_source(&context, extent, extra.precision(), &values);
            expected.push(values);
            source
        })
        .collect();
    let source = padded_rgb_source_sized(
        &context,
        extent.width as usize,
        extent.height as usize,
        &reference::pattern(extent.width as usize, extent.height as usize),
    )
    .with_extra_channels(inputs)
    .unwrap();
    let encoder = TiledVarDctEncoder::new_with_config(context.clone(), config).unwrap();
    let bytes = encoder.encode(source).unwrap();
    let words = modular_integer::extra_planes(&bytes, 0);
    assert_eq!(words.len(), expected.len());
    for (decoded, expected) in words.iter().zip(&expected) {
        assert_eq!(
            decoded.iter().map(|&v| v as u32).collect::<Vec<_>>(),
            *expected
        );
    }
    let (_, decoded) =
        extra_channels::libjxl_planes(&bytes, extent.area().unwrap(), expected.len()).unwrap();
    for ((decoded, words), bits) in decoded.iter().zip(&expected).zip([13, 7, 31]) {
        let mask = (1u32 << bits) - 1;
        for (&decoded, &word) in decoded.iter().zip(words) {
            let expected = (f64::from(word) / f64::from(mask)) as f32;
            assert!(
                (decoded - expected).abs()
                    <= 2.0 * f32::EPSILON * expected.abs().max(f32::MIN_POSITIVE)
            );
        }
    }
    assert_eq!(context.memory_stats().reserved_bytes, 0);
}
