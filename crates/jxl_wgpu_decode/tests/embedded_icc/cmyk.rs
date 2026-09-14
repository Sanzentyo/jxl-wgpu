use super::*;
use jxl_gpu_formats::{Channel, PixelFormat, SampleKind};
use sha2::{Digest, Sha256};
use std::io::Read;

const LAYERS: &[u8] = include_bytes!("../../test-data/cmyk/layers.jxl");

fn checked(bytes: &[u8], hash: &str) {
    assert_eq!(
        Sha256::digest(bytes).as_slice(),
        jxl_test_support::offline::hex::unhex(hash)
    );
}

fn layers_reference() -> Vec<[f32; 5]> {
    checked(
        LAYERS,
        "d732c8836bf1abeadf310d2e07387a32813ed4690d32650c1c25e541b80eed4a",
    );
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(&include_bytes!("../../test-data/cmyk/layers.npy.gz")[..])
        .read_to_end(&mut bytes)
        .unwrap();
    checked(
        &bytes,
        "a01913d4e4b1a89bd96e5de82a5dfb9925c7827ee6380ad60c0b1c4becb53880",
    );
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let offset = 10 + usize::from(u16::from_le_bytes(bytes[8..10].try_into().unwrap()));
    let header = std::str::from_utf8(&bytes[10..offset]).unwrap();
    assert!(header.contains("'descr': '<f4'"));
    assert!(header.contains("'fortran_order': False"));
    assert!(header.contains("'shape': (1, 512, 512, 5)"));
    let (pixels, remainder) = bytes[offset..].as_chunks::<20>();
    assert!(remainder.is_empty());
    assert_eq!(pixels.len(), 512 * 512);
    pixels
        .iter()
        .map(|pixel| {
            let words = pixel.as_chunks::<4>().0;
            std::array::from_fn(|channel| f32::from_le_bytes(words[channel]))
        })
        .collect()
}

#[test]
fn official_cmyk_layers_meet_all_five_channel_bounds_for_whole_and_bounded_input() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let expected = layers_reference();
    let inventory = inventory(LAYERS);
    let profile = &inventory
        .image_header
        .embedded_icc
        .as_ref()
        .unwrap()
        .profile;
    checked(
        profile,
        "4855b8fabb96bdc6495d45d089bb8c8efb1ae18389e0dc9e75a5f701a9c0b662",
    );
    assert_eq!(
        &**profile,
        include_bytes!("../../test-data/cmyk/layers.icc")
    );
    assert!(!inventory.image_header.xyb_encoded);
    assert_eq!(inventory.image_header.extra_channels.len(), 2);
    assert!(matches!(
        inventory.image_header.extra_channels[0].channel_type,
        jxl_gpu_bitstream::ExtraChannelTypeInventory::Black
    ));
    let mut whole = Vec::new();
    for limit in [None, NonZeroU64::new(256)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        for channel in 0..5 {
            let request = GpuOutputRequest::numeric(
                PixelFormat::non_color(SampleKind::Float, 32, &[Channel::X]),
                NumericSampleMapping::NormalizedUnsigned,
            )
            .unwrap();
            let request = if channel < 3 {
                request.with_color_channel(channel as u32)
            } else {
                request.with_extra_channel(channel as u32 - 3)
            }
            .unwrap();
            let bytes = decode(&backend, &decoder, LAYERS, request, limit.is_some());
            let (actual, remainder) = bytes.as_chunks::<4>();
            assert!(remainder.is_empty());
            assert_eq!(actual.len(), expected.len());
            let mut squared = 0.0;
            let mut peak = 0.0f64;
            for (value, expected) in actual.iter().zip(&expected) {
                let value = f32::from_le_bytes(*value);
                assert!(value.is_finite());
                let error = f64::from(value) - f64::from(expected[channel]);
                squared += error * error;
                peak = peak.max(error.abs());
            }
            let rmse = (squared / expected.len() as f64).sqrt();
            eprintln!("CMYK layers channel {channel}, window {limit:?}: RMSE {rmse}, peak {peak}");
            assert!(rmse <= 0.000976562, "channel {channel}: RMSE {rmse}");
            assert!(peak <= 0.000976562, "channel {channel}: peak {peak}");
            if limit.is_none() {
                whole.push(bytes);
            } else {
                assert_eq!(bytes, whole[channel], "bounded channel {channel}");
            }
            assert_eq!(
                backend.transient_memory_budget().snapshot().reserved_bytes,
                0
            );
        }
    }
}

#[derive(serde::Deserialize)]
struct Manifest {
    width: usize,
    height: usize,
    frames: usize,
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    source: String,
    target: String,
    channels: usize,
    mode: u32,
    black: usize,
}

#[test]
fn cmyk_luts_use_the_independently_composed_black_plane_before_rgb_gray_and_spot_output() {
    use jxl_gpu_formats::{ColorSpecification, RgbChannelOrder};
    use jxl_gpu_protocol::icc::{IccProfile, IccRenderingIntent};
    use jxl_wgpu_decode::{AlphaOutputPolicy, SpotColorPolicy};
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let profiles = directory.join("../jxl_wgpu/test-data/icc/lut");
    let directory = directory.join("test-data/cmyk/generated");
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest.cases.len(), 18);
    assert_eq!(manifest.frames, 3);
    let count = manifest.width * manifest.height;
    assert_eq!(count, 153);
    let mut components = 0;
    let mut presentations = 0;
    for case in manifest.cases {
        eprintln!("CMYK corpus {}", case.name);
        let data = std::fs::read(directory.join(format!("{}.jxl", case.name))).unwrap();
        let inventory = inventory(&data);
        assert_eq!(inventory.image_header.width as usize, manifest.width);
        assert_eq!(inventory.image_header.height as usize, manifest.height);
        assert_eq!(inventory.frames.len(), 3);
        assert!(!inventory.image_header.xyb_encoded);
        assert!(!inventory.image_header.grayscale);
        let embedded = &inventory
            .image_header
            .embedded_icc
            .as_ref()
            .unwrap()
            .profile;
        assert_eq!(
            embedded.as_ref(),
            std::fs::read(profiles.join(format!("{}.icc", case.source))).unwrap()
        );
        assert!(matches!(
            inventory.image_header.extra_channels[case.black].channel_type,
            jxl_gpu_bitstream::ExtraChannelTypeInventory::Black
        ));
        for frame in &inventory.frames {
            assert_eq!(
                frame.encoding,
                if case.mode == 0 {
                    FrameEncoding::Modular
                } else {
                    FrameEncoding::VarDct
                }
            );
            assert_eq!(frame.do_ycbcr, case.mode == 2);
        }
        assert_eq!(inventory.frames[2].color_blend.source, 2);
        assert_eq!(
            inventory.frames[2].extra_channel_blends[case.black].source,
            1
        );
        let original = std::fs::read(directory.join(format!("{}.f32", case.name))).unwrap();
        let (original, tail) = original.as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(original.len(), count * manifest.frames * 6);
        let target = IccProfile::parse(
            std::fs::read(profiles.join(format!("{}.icc", case.target)))
                .unwrap()
                .into(),
            Default::default(),
        )
        .unwrap();
        for spots in [false, true] {
            for intent in [
                IccRenderingIntent::Perceptual,
                IccRenderingIntent::Relative,
                IccRenderingIntent::Saturation,
                IccRenderingIntent::Absolute,
            ] {
                let expected = super::lut::reference(
                    &directory,
                    &format!("{}_{}", case.name, u32::from(spots)),
                    intent,
                );
                assert_eq!(expected.len(), manifest.frames * count * case.channels);
                for planar in [false, true] {
                    let color = ColorSpecification::Icc(target.clone());
                    let format = if case.channels == 1 {
                        PixelFormat::gray_f32(true, planar, color)
                    } else {
                        PixelFormat::rgb_f32(RgbChannelOrder::Rgba, planar, color)
                    };
                    let mut whole = Vec::new();
                    for limit in [None, NonZeroU64::new(256)] {
                        let request = GpuOutputRequest::color(format.clone())
                            .unwrap()
                            .with_max_frame_slots(std::num::NonZeroUsize::new(3).unwrap())
                            .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
                            .with_spot_color_policy(if spots {
                                SpotColorPolicy::Render
                            } else {
                                SpotColorPolicy::Preserve
                            })
                            .with_icc_rendering_intent(intent);
                        let frames = super::output::frames(&backend, &data, request, limit);
                        assert_eq!(frames.len(), manifest.frames);
                        for (frame, pixels) in frames.iter().enumerate() {
                            assert_eq!(pixels.len(), count * (case.channels + 1));
                            for p in 0..count {
                                let word = |c| {
                                    pixels[if planar {
                                        c * count + p
                                    } else {
                                        p * (case.channels + 1) + c
                                    }]
                                };
                                assert_eq!(
                                    word(case.channels),
                                    u32::from_le_bytes(original[(frame * count + p) * 6 + 4])
                                );
                                for c in 0..case.channels {
                                    let actual = f32::from_bits(word(c));
                                    let reference =
                                        expected[(frame * count + p) * case.channels + c];
                                    assert!(
                                        actual.is_finite()
                                            && reference[2] <= actual
                                            && actual <= reference[3],
                                        "{} {intent:?} spots {spots} frame {frame} pixel {p} channel {c}: GPU {actual}, independent {reference:?}",
                                        case.name
                                    );
                                    components += 1;
                                }
                            }
                            presentations += 1;
                        }
                        if limit.is_none() {
                            whole = frames;
                        } else {
                            assert_eq!(frames, whole, "{} bounded", case.name);
                        }
                    }
                }
            }
        }
        // Numeric selections expose the original complemented CMY and actual extras,
        // without the display-only spot inks or any ICC evaluation.
        for channel in 0..6 {
            let request =
                numeric_float().with_max_frame_slots(std::num::NonZeroUsize::new(3).unwrap());
            let request = if channel < 3 {
                request.with_color_channel(channel as u32)
            } else {
                request.with_extra_channel(channel as u32 - 3)
            }
            .unwrap();
            let frames = super::output::frames(&backend, &data, request, None);
            assert_eq!(frames.len(), manifest.frames);
            for (frame, pixels) in frames.iter().enumerate() {
                assert_eq!(pixels.len(), count);
                for (p, &word) in pixels.iter().enumerate() {
                    let expected = u32::from_le_bytes(original[(frame * count + p) * 6 + channel]);
                    if case.mode == 0 || channel >= 3 {
                        assert_eq!(
                            word, expected,
                            "{} frame {frame} pixel {p} channel {channel}",
                            case.name
                        );
                    } else {
                        assert!(
                            (f32::from_bits(word) - f32::from_bits(expected)).abs() <= 2e-5,
                            "{} numeric frame {frame} pixel {p} channel {channel}",
                            case.name
                        );
                    }
                }
            }
        }
    }
    assert_eq!(components, 528768);
    assert_eq!(presentations, 1728);
    eprintln!("CMYK: {presentations} color presentations, {components} independent components");
}
