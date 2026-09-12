use super::*;
use jxl_test_support::fixtures::patch_references::{self as corpus, Pattern};

#[test]
fn subsampled_component_references_match_independent_decoders_with_bounded_progression() {
    check_references(false);
}

#[test]
fn mixed_component_references_match_native_with_bounded_progression() {
    check_references(true);
}

fn check_references(mixed: bool) {
    let backend = backend();
    for case in corpus::cases()
        .into_iter()
        .filter(|case| case.name.starts_with("mixed_") == mixed)
    {
        let name = format!("references/{}", case.name);
        let data = encoded(&name);
        assert_eq!(data, case.encode(), "fixture recipe drift: {name}");
        let info = inventory(&data);
        if mixed {
            let first = info.frames.first().unwrap();
            assert!(
                info.frames
                    .iter()
                    .any(|frame| frame.encoding != first.encoding
                        || frame.do_ycbcr != first.do_ycbcr)
            );
        } else {
            assert!(info.frames.iter().all(|frame| frame.do_ycbcr));
        }
        let expected = reference(&name);
        eprintln!("{name}: {:?} linear reference", case.reference_source());
        if case.pattern == Pattern::AllModes && mixed {
            let control = name.strip_suffix("_patches").unwrap().to_owned() + "_empty";
            assert_ne!(expected, reference(&control), "patches must affect {name}");
        }
        if case.zero_noise
            && let Some(nonzero) = name.strip_suffix("_zero")
        {
            assert_ne!(expected, reference(nonzero), "noise must affect {name}");
        }
        let tolerance = match case.reference_source() {
            corpus::ReferenceSource::NativeLinear | corpus::ReferenceSource::NativeSrgb => {
                1.0 / 1024.0
            }
            corpus::ReferenceSource::JxlOxideLinear => 1e-5,
        };
        features::check_image(&backend, &name, &data, &expected, tolerance);
    }
}

#[test]
fn subsampled_patches_reject_references_saved_after_color_conversion() {
    let backend = backend();
    let original = source("noise/jpeg_420");
    let frame = inventory(&original).frames.remove(0);
    let values = fixtures::values(&frame, 0, 16);
    let data = frame_features::assemble_frames(&[
        frame_features::Frame {
            codestream: &original,
            reference: Some((3, false)),
            patches: None,
            splines: None,
        },
        frame_features::Frame {
            codestream: &original,
            reference: None,
            patches: Some(&values),
            splines: None,
        },
    ]);
    for limit in [None, NonZeroU64::new(256)] {
        let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
        if let Some(limit) = limit {
            engine = engine.with_stream_window_limit(limit);
        }
        let decoder = GpuDecoder::new(engine);
        let mut session = planes::open_fragmented(&decoder, &data, request());
        assert!(matches!(
            session.next_frame(),
            Err(Error::PatchDictionary { code: 13 })
        ));
        drop(session);
        progression::drain(&backend, 0);
        assert_eq!(
            decoder.incremental_input_budget().snapshot().reserved_bytes,
            0
        );
    }
}

#[test]
fn jpeg_patch_destinations_cannot_cross_the_padded_component_extent() {
    let backend = backend();
    // Expected padded extents include raw-factor alignment for equal nonzero selectors.
    for (selectors, width, height) in [
        ("000", 264, 24),
        ("111", 272, 32),
        ("222", 272, 24),
        ("333", 264, 32),
        ("020", 272, 24),
        ("030", 264, 32),
        ("010", 272, 32),
        ("320", 272, 32),
    ] {
        let original = source(&format!("jpeg_sampling/odd_{selectors}"));
        for [x, y] in [[width - 1, 0], [0, height - 1]] {
            let values = vec![1, 3, 0, 0, 1, 1, 0, x, y, 1];
            let data = fixtures::assemble(&original, &values);
            for limit in [None, NonZeroU64::new(256)] {
                let mut engine = WgpuDecodeEngine::new(backend.clone()).unwrap();
                if let Some(limit) = limit {
                    engine = engine.with_stream_window_limit(limit);
                }
                let decoder = GpuDecoder::new(engine);
                let mut session = if limit.is_some() {
                    planes::open_fragmented(&decoder, &data, request())
                } else {
                    decoder.open(&data, request()).unwrap()
                };
                assert!(
                    matches!(
                        session.next_frame(),
                        Err(Error::PatchDictionary { code: 14 })
                    ),
                    "out-of-bounds patch at {selectors} {x},{y} {limit:?}"
                );
                drop(session);
                progression::drain(&backend, 0);
                assert_eq!(
                    decoder.incremental_input_budget().snapshot().reserved_bytes,
                    0
                );
            }
        }
    }
}
