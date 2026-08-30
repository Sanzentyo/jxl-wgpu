use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use jxl_gpu_harness::codec::request_backend;
use jxl_gpu_harness::conformance::{
    AlphaPattern, ConformanceCorpus, ConformanceExpectation, DEFAULT_MAX_ROW_BYTES,
    ExternalFixtureOptions, ExternalFixtureStatus, LazyImage, PixelModel, ResolutionClass,
    SampleDepth, external_fixture, run_stock_gpu_round_trip, write_encoded_output,
};
use jxl_gpu_harness::report::CaseStatus;

fn manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("conformance-corpus.toml")
}

#[test]
fn checked_in_manifest_exposes_required_inventory() {
    let corpus = ConformanceCorpus::load(manifest_path()).unwrap();
    let categories = corpus
        .cases
        .iter()
        .map(|case| case.category)
        .collect::<BTreeSet<_>>();
    for required in [
        ResolutionClass::Tiny,
        ResolutionClass::Odd,
        ResolutionClass::Square,
        ResolutionClass::Portrait,
        ResolutionClass::Landscape,
        ResolutionClass::Panorama,
        ResolutionClass::Tall,
        ResolutionClass::GroupBoundary255,
        ResolutionClass::GroupBoundary256,
        ResolutionClass::GroupBoundary257,
        ResolutionClass::Hd,
        ResolutionClass::Fhd,
        ResolutionClass::Uhd4k,
        ResolutionClass::Uhd8k,
        ResolutionClass::Uhd16k,
    ] {
        assert!(categories.contains(&required), "missing {required:?}");
    }
    let models = corpus
        .cases
        .iter()
        .map(|case| case.source.model)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        models,
        BTreeSet::from([PixelModel::Gray, PixelModel::Rgb, PixelModel::Rgba])
    );
    let depths = corpus
        .cases
        .iter()
        .map(|case| case.source.depth)
        .collect::<BTreeSet<_>>();
    assert!(depths.contains(&SampleDepth::U8));
    assert!(depths.contains(&SampleDepth::U10));
    assert!(depths.contains(&SampleDepth::U12));
    assert!(depths.contains(&SampleDepth::U16));
    let alpha = corpus
        .cases
        .iter()
        .map(|case| case.source.alpha)
        .collect::<BTreeSet<_>>();
    assert!(alpha.contains(&AlphaPattern::Opaque));
    assert!(alpha.contains(&AlphaPattern::Checkerboard));
    assert!(alpha.contains(&AlphaPattern::HorizontalRamp));

    let panorama_16k = corpus
        .cases
        .iter()
        .find(|case| case.name == "extreme-panorama-gray8-16384x1")
        .unwrap();
    assert_eq!(panorama_16k.extent.width, 16_384);
    assert_eq!(panorama_16k.extent.height, 1);
    assert_eq!(
        panorama_16k.expectation,
        ConformanceExpectation::StockGpuRoundTrip
    );

    let uhd_8k = corpus
        .cases
        .iter()
        .find(|case| case.category == ResolutionClass::Uhd8k)
        .unwrap();
    assert_eq!((uhd_8k.extent.width, uhd_8k.extent.height), (7680, 4320));
    let uhd_16k = corpus
        .cases
        .iter()
        .find(|case| case.category == ResolutionClass::Uhd16k)
        .unwrap();
    assert_eq!(
        (uhd_16k.extent.width, uhd_16k.extent.height),
        (15_360, 8640)
    );
    let uhd_16k_image = LazyImage::new(uhd_16k, DEFAULT_MAX_ROW_BYTES).unwrap();
    assert_eq!(uhd_16k_image.layout().row_stride, 15_360);
    assert_eq!(uhd_16k_image.layout().storage_bytes, 132_710_400);

    for case in &corpus.cases {
        let image = LazyImage::new(case, DEFAULT_MAX_ROW_BYTES).unwrap();
        assert!(image.layout().row_stride >= image.layout().active_row_bytes);
        assert_eq!(image.rows().len(), case.extent.height as usize);
    }

    let odd = corpus
        .cases
        .iter()
        .find(|case| case.name == "odd-gray8-17x13")
        .unwrap();
    let hashes = LazyImage::new(odd, DEFAULT_MAX_ROW_BYTES)
        .unwrap()
        .hashes()
        .unwrap();
    assert_eq!(
        hashes.input_hash,
        "3464f0dd45fdfd16b5e1b462dbc9eac659c86f0762846614bc2a8885fab1350b"
    );
    assert_eq!(
        hashes.pixel_hash,
        "581f568b0098a0edb2fc693c5b332c2d0270fe241f48377e96a4d793a6688c66"
    );
}

#[test]
fn stock_profile_cases_round_trip_exactly_on_an_available_gpu() {
    let corpus = ConformanceCorpus::load(manifest_path()).unwrap();
    let Some(backend) = request_backend().unwrap() else {
        eprintln!("skipping GPU conformance because no adapter is available");
        return;
    };
    let stock = corpus
        .cases
        .iter()
        .filter(|case| case.expectation == ConformanceExpectation::StockGpuRoundTrip)
        .collect::<Vec<_>>();
    assert!(stock.len() >= 2);
    for case in stock {
        let result = run_stock_gpu_round_trip(case, Some(&backend), DEFAULT_MAX_ROW_BYTES).unwrap();
        let expected = result.inventory.hashes.pixel_hash;
        let report = result.report;
        assert!(result.encoded.is_some());
        eprintln!(
            "conformance case {}: status={:?}, codec_submissions={}, readback_staging_bytes={}",
            case.name, report.status, report.codec_submissions, report.readback_staging_bytes
        );
        assert_eq!(report.status, CaseStatus::Passed, "{report:#?}");
        assert_eq!(report.output_hash.as_deref(), Some(expected.as_str()));
    }
}

#[test]
fn stock_small_case_container_can_be_saved_without_reencoding() {
    let corpus = ConformanceCorpus::load(manifest_path()).unwrap();
    let Some(backend) = request_backend().unwrap() else {
        eprintln!("skipping encoded save test because no adapter is available");
        return;
    };
    let case = corpus
        .cases
        .iter()
        .find(|case| case.name == "tiny-gray8-2x2")
        .unwrap();
    let result = run_stock_gpu_round_trip(case, Some(&backend), DEFAULT_MAX_ROW_BYTES).unwrap();
    assert_eq!(result.report.status, CaseStatus::Passed);
    let encoded = result
        .encoded
        .expect("a completed GPU encode returns its container");
    assert_eq!(std::sync::Arc::strong_count(&encoded), 1);
    let path = std::env::temp_dir().join(format!(
        "jxl-wgpu-conformance-encoded-save-{}.jxl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    write_encoded_output(&path, encoded.as_ref()).unwrap();
    let saved = std::fs::read(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(saved.as_slice(), encoded.as_ref());
}

#[test]
fn installed_reference_tools_make_an_exact_standard_fixture() {
    let cjxl = PathBuf::from("/opt/homebrew/bin/cjxl");
    let djxl = PathBuf::from("/opt/homebrew/bin/djxl");
    if !cjxl.is_file() || !djxl.is_file() {
        eprintln!("skipping external fixture test because cjxl/djxl are unavailable");
        return;
    }
    let corpus = ConformanceCorpus::load(manifest_path()).unwrap();
    let case = corpus
        .cases
        .iter()
        .find(|case| case.name == "tiny-gray8-2x2")
        .unwrap();
    let output_dir = std::env::temp_dir().join(format!(
        "jxl-wgpu-external-conformance-test-{}",
        std::process::id()
    ));
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }
    let report = external_fixture(
        case,
        DEFAULT_MAX_ROW_BYTES,
        &ExternalFixtureOptions {
            apply: true,
            force: false,
            output_dir: output_dir.clone(),
            cjxl,
            djxl,
        },
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(output_dir);
    assert_eq!(report.status, ExternalFixtureStatus::Passed, "{report:#?}");
    assert_eq!(report.exact, Some(true));
    assert!(report.jxl_bytes.is_some_and(|bytes| bytes > 0));
}
