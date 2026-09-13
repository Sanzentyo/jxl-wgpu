use super::{corpus, decode, numeric_float, profile};
use jxl_test_support::offline::unhex;
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{
    GpuDecoder, GpuOutputRequest, ModularChannels, NumericSampleMapping, WgpuDecodeEngine,
    native_modular_pixel_format,
};
use std::num::NonZeroU64;

pub(super) fn sample_fixture(directory: &str, name: &str) -> (Vec<u8>, Vec<u32>) {
    let root = corpus::directory().parent().unwrap().join(directory);
    let bytes = unhex(&std::fs::read_to_string(root.join(format!("{name}.jxl.hex"))).unwrap());
    let suffix = if directory == "integer" { "u32" } else { "f32" };
    let words = std::fs::read_to_string(root.join(format!("{name}.{suffix}.hex")))
        .unwrap()
        .split_whitespace()
        .map(|word| u32::from_str_radix(word, 16).unwrap())
        .collect();
    (bytes, words)
}

#[test]
fn embedded_icc_does_not_change_wide_integer_codes_or_ieee754_words() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let decoders = [
        GpuDecoder::wgpu(backend.clone()).unwrap(),
        GpuDecoder::new(
            WgpuDecodeEngine::new(backend.clone())
                .unwrap()
                .with_stream_window_limit(NonZeroU64::new(256).unwrap()),
        ),
    ];
    let rgb = corpus::cases()
        .find(|case| !case.gray && !case.xyb)
        .unwrap()
        .bytes();
    let gray = corpus::cases()
        .find(|case| case.gray && !case.xyb)
        .unwrap()
        .bytes();
    for (name, color_bits, alpha_bits) in [
        ("17-3-31-33x5-p0-r0", 17, 31),
        ("31-3-5-33x5-p0-r0", 31, 5),
        ("31-3-24-33x5-p0-r0", 31, 24),
    ] {
        let (bytes, words) = sample_fixture("integer", name);
        let bytes = profile::replace(&bytes, &rgb);
        for channel in 0..4 {
            let bits = if channel == 3 { alpha_bits } else { color_bits };
            let request = GpuOutputRequest::numeric(
                native_modular_pixel_format(ModularChannels::Gray, bits).unwrap(),
                NumericSampleMapping::NativeUnsigned,
            )
            .unwrap();
            let request = if channel == 3 {
                request.with_extra_channel(0)
            } else {
                request.with_color_channel(channel)
            }
            .unwrap();
            let storage = usize::from(bits.next_power_of_two().max(8) / 8);
            let expected: Vec<_> = words
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| p[channel as usize].to_le_bytes().into_iter().take(storage))
                .collect();
            for (index, decoder) in decoders.iter().enumerate() {
                assert_eq!(
                    decode(&backend, decoder, &bytes, request.clone(), index != 0),
                    expected,
                    "{name}/{channel}/{index}"
                );
            }
        }
    }
    for name in ["5-2", "16-5", "24-7", "32-8"] {
        let (bytes, words) = sample_fixture("floating", name);
        let bytes = profile::replace(&bytes, &gray);
        let expected: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        for (index, decoder) in decoders.iter().enumerate() {
            assert_eq!(
                decode(&backend, decoder, &bytes, numeric_float(), index != 0),
                expected,
                "floating {name}/{index}"
            );
        }
    }
    assert_eq!(
        backend.transient_memory_budget().snapshot().reserved_bytes,
        0
    );
}
