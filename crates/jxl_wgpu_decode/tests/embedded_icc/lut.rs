use super::output;
use jxl_gpu_bitstream::FrameEncoding;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest};
use serde::Deserialize;
use std::{num::NonZeroU64, path::Path};

#[derive(Deserialize)]
struct Manifest {
    width: usize,
    height: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    source: String,
    target: String,
    channels: usize,
    target_channels: usize,
    modular: bool,
}

fn reference(directory: &Path, name: &str, intent: IccRenderingIntent) -> Vec<[f32; 6]> {
    let bytes =
        std::fs::read(directory.join(format!("{name}_{}.reference", intent as u32))).unwrap();
    let (records, tail) = bytes.as_chunks::<28>();
    assert!(tail.is_empty());
    records
        .iter()
        .map(|record| {
            let values = std::array::from_fn(|c| {
                f32::from_le_bytes(record[c * 4..c * 4 + 4].try_into().unwrap())
            });
            let [native, exact, lower, upper, native_lower, native_upper] = values;
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(lower <= exact && exact <= upper);
            assert!(native_lower <= native && native <= native_upper);
            assert!(matches!(
                u32::from_le_bytes(record[24..].try_into().unwrap()),
                0 | 32
            ));
            values
        })
        .collect()
}

#[test]
fn embedded_legacy_luts_convert_gray_and_rgb_through_both_codecs_and_all_intents() {
    verify_decoder_corpus("lut");
}

#[test]
fn embedded_v2_luts_prepare_black_connections_through_both_codecs_and_all_intents() {
    verify_decoder_corpus("black");
}

fn verify_decoder_corpus(corpus: &str) {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let profiles = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../jxl_wgpu/test-data/icc")
        .join(corpus);
    let directory = profiles.join("decoder");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.cases.len(), 24);
    let count = manifest.width * manifest.height;
    assert_eq!(count, 153);
    let mut presentations = 0;
    let mut components = 0;
    let mut distinct = [0; 4];
    for case in manifest.cases {
        let data = std::fs::read(directory.join(format!("{}.jxl", case.name))).unwrap();
        let inventory = super::inventory(&data);
        let embedded = inventory.image_header.embedded_icc.as_ref().unwrap();
        let source = std::fs::read(profiles.join(format!("{}.icc", case.source))).unwrap();
        assert_eq!(embedded.profile.as_ref(), source);
        assert_eq!(inventory.image_header.grayscale, case.channels == 1);
        assert!(!inventory.image_header.xyb_encoded);
        assert_eq!(inventory.frames.len(), 1);
        assert_eq!(
            inventory.frames[0].encoding,
            if case.modular {
                FrameEncoding::Modular
            } else {
                FrameEncoding::VarDct
            }
        );
        let target = IccProfile::parse(
            std::fs::read(profiles.join(format!("{}.icc", case.target)))
                .unwrap()
                .into(),
            Default::default(),
        )
        .unwrap();
        let native = std::fs::read(directory.join(format!("{}.f32le", case.name))).unwrap();
        let (native, tail) = native.as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(native.len(), count * (case.channels + 1));
        let relative = reference(&directory, &case.name, IccRenderingIntent::Relative);
        for intent in [
            IccRenderingIntent::Perceptual,
            IccRenderingIntent::Relative,
            IccRenderingIntent::Saturation,
            IccRenderingIntent::Absolute,
        ] {
            let expected = reference(&directory, &case.name, intent);
            assert_eq!(expected.len(), count * case.target_channels);
            distinct[intent as usize] += expected
                .iter()
                .zip(&relative)
                .filter(|(a, b)| a[3] < b[2] || b[3] < a[2])
                .count();
            for planar in [false, true] {
                let color = ColorSpecification::Icc(target.clone());
                let format = if case.target_channels == 1 {
                    PixelFormat::gray_f32(true, planar, color)
                } else {
                    PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                };
                for limit in [None, NonZeroU64::new(256)] {
                    let request = GpuOutputRequest::color(format.clone())
                        .unwrap()
                        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                        .with_icc_rendering_intent(intent);
                    let frames = output::frames(&backend, &data, request, limit);
                    assert_eq!(frames.len(), 1);
                    let pixels = &frames[0];
                    assert_eq!(pixels.len(), count * (case.target_channels + 1));
                    for pixel in 0..count {
                        let word = |c| {
                            pixels[if planar {
                                c * count + pixel
                            } else {
                                pixel * (case.target_channels + 1) + c
                            }]
                        };
                        assert_eq!(
                            word(case.target_channels),
                            u32::from_le_bytes(native[pixel * (case.channels + 1) + case.channels]),
                            "alpha {}",
                            case.name
                        );
                        for c in 0..case.target_channels {
                            let actual = f32::from_bits(word(c));
                            let reference = expected[pixel * case.target_channels + c];
                            assert!(
                                actual.is_finite()
                                    && reference[2] <= actual
                                    && actual <= reference[3],
                                "{} -> {} {intent:?} pixel {pixel} channel {c} planar {planar} window {limit:?}: GPU {actual}, independent {reference:?}",
                                case.name,
                                case.target
                            );
                            components += 1;
                        }
                    }
                    presentations += 1;
                }
            }
        }
    }
    assert_eq!(presentations, 384);
    assert_eq!(components, 117504);
    assert!(distinct[0] > 1000 && distinct[2] > 1000);
    assert_eq!(distinct[1], 0);
    assert_eq!(distinct[3], 0); // Equal media whites; absolute selects the relative LUT.
    eprintln!(
        "{corpus} LUT decoder: {presentations} presentations, {components} components, distinct from relative {distinct:?}"
    );
}
