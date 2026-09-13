use super::output::frames;
use super::{corpus, inventory, numeric, profile};
use jxl_gpu_bitstream::FrameEncoding;
use jxl_gpu_formats::{ColorSpecification, PixelFormat, RgbChannelOrder};
use jxl_gpu_protocol::WhitePointAdaptation;
use jxl_gpu_protocol::icc::{IccLimits, IccProfile, IccRenderingIntent};
use jxl_test_support::{fixtures::modular_ycbcr, fixtures::original_color};
use jxl_wgpu::WgpuBackend;
use jxl_wgpu_decode::{AlphaOutputPolicy, GpuOutputRequest, OrientationPolicy};
use std::num::{NonZeroU64, NonZeroUsize};

mod converted;

fn donor(gray: bool) -> corpus::Case {
    corpus::cases()
        .find(|case| case.gray == gray && !case.xyb)
        .unwrap()
}

fn device_format(donor: corpus::Case) -> PixelFormat {
    let profile = IccProfile::parse(donor.profile().into(), IccLimits::default()).unwrap();
    let specification = ColorSpecification::Icc(profile);
    if donor.gray {
        PixelFormat::gray_f32(true, false, specification)
    } else {
        PixelFormat::rgb_f32(RgbChannelOrder::Rgba, false, specification)
    }
}

fn device_request(format: PixelFormat) -> GpuOutputRequest {
    // Reconstructing YCbCr into its original device space needs no ICC method or adaptation.
    // These choices would fail if the matrix/TRC CMS were accidentally selected.
    GpuOutputRequest::color(format)
        .unwrap()
        .with_alpha_output_policy(AlphaOutputPolicy::Preserve)
        .with_orientation_policy(OrientationPolicy::Keep)
        .with_icc_rendering_intent(IccRenderingIntent::Perceptual)
        .with_white_point_adaptation(WhitePointAdaptation::None)
        // Retain all four sequence frames while reserving one slot for end-of-stream.
        .with_max_frame_slots(NonZeroUsize::new(5).unwrap())
}

fn compare(
    actual: &[u32],
    expected: &[f32],
    channels: usize,
    bound: impl Fn(f32) -> f32,
    alpha_bound: impl Fn(f32) -> f32,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let mut maximum = 0.0_f32;
    for (index, (&word, &expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f32::from_bits(word);
        let error = (actual - expected).abs();
        let tolerance = if index % channels == channels - 1 {
            alpha_bound(expected)
        } else {
            bound(expected)
        };
        assert!(
            actual.is_finite() && expected.is_finite() && error <= tolerance,
            "{context}: {index}: {actual} vs {expected}, error {error}"
        );
        maximum = maximum.max(error);
    }
    eprintln!("{context}: maxAE={maximum}");
}

#[test]
fn modular_icc_ycbcr_reconstructs_sampling_precision_restoration_and_alpha() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let cases: Vec<_> = modular_ycbcr::cases()
        .into_iter()
        .filter(|case| case.global_transforms.is_empty() && !case.has_local_transforms())
        .collect();
    assert_eq!(cases.len(), 102);
    for case in cases {
        let (source, reference) = numeric::sample_fixture("modular_ycbcr", &case.name);
        case.validate(&inventory(&source));
        let donor = donor(case.grayscale);
        let data = profile::replace(&source, &donor.bytes());
        let format = device_format(donor);
        let pixels = (case.size[0] * case.size[1]) as usize;
        // Filtered references use the independently expanded 4:4:4 equivalent streams;
        // the native fast renderer is not a valid reference for original restoration_1/3.
        let channels: &[usize] = if case.grayscale {
            &[0, 3]
        } else {
            &[0, 1, 2, 3]
        };
        let expected: Vec<_> = reference[..pixels * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|pixel| {
                channels
                    .iter()
                    .map(move |&channel| f32::from_bits(pixel[channel]))
            })
            .collect();
        let mut whole = None;
        for limit in [None, NonZeroU64::new(40)] {
            let frames = frames(&backend, &data, device_request(format.clone()), limit);
            assert_eq!(frames.len(), 1, "{}", case.name);
            compare(
                &frames[0],
                &expected,
                if case.grayscale { 2 } else { 4 },
                |_| 2e-6,
                |_| 2e-6,
                &format!("{} ICC {limit:?}", case.name),
            );
            if let Some(whole) = &whole {
                assert_eq!(&frames, whole, "{} fragmented", case.name);
            }
            whole = Some(frames);
        }
    }
}

#[test]
fn both_icc_ycbcr_codecs_compose_original_device_values_and_retain_completed_frames() {
    let backend = pollster::block_on(WgpuBackend::request_default(Default::default())).unwrap();
    let cases: Vec<_> = original_color::cases()
        .into_iter()
        .chain(original_color::analytic_cases())
        .filter(|case| case.mode.ycbcr())
        .collect();
    assert_eq!(cases.len(), 44);
    let donor = donor(false);
    let format = device_format(donor);
    for case in cases {
        let source = case.bytes();
        let source_inventory = inventory(&source);
        case.validate(&source_inventory);
        assert!(source_inventory.frames.iter().all(|frame| frame.do_ycbcr));
        let data = profile::replace(&source, &donor.bytes());
        let expected = case.reference();
        // Preserve the established original_color reconstruction bound for each codec.
        let tolerance = if case.mode.encoding() == FrameEncoding::Modular {
            1e-5
        } else {
            1.0 / 1024.0
        };
        let mut whole = None;
        for limit in [None, NonZeroU64::new(256)] {
            let frames = frames(&backend, &data, device_request(format.clone()), limit);
            let actual: Vec<_> = frames.iter().flatten().copied().collect();
            compare(
                &actual,
                &expected,
                4,
                |value| tolerance * (1.0 + value.abs()),
                |value| 2e-6 * (1.0 + value.abs()),
                &format!("{} ICC {limit:?}", case.name),
            );
            assert_eq!(frames.len(), if case.sequence { 4 } else { 1 });
            if let Some(whole) = &whole {
                assert_eq!(&frames, whole, "{} fragmented", case.name);
            }
            whole = Some(frames);
        }
    }
}
