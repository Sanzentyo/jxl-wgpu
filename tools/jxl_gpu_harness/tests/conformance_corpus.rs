use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use jxl_gpu_harness::codec::request_backend;
use jxl_gpu_harness::conformance::{
    AlphaPattern, ConformanceCorpus, ConformanceExpectation, DEFAULT_MAX_ROW_BYTES,
    ExternalFixtureOptions, ExternalFixtureStatus, LazyImage, PixelModel, ResolutionClass,
    SampleDepth, external_fixture, run_stock_gpu_round_trip,
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
        let expected = LazyImage::new(case, DEFAULT_MAX_ROW_BYTES)
            .unwrap()
            .hashes()
            .unwrap()
            .pixel_hash;
        let report = run_stock_gpu_round_trip(case, Some(&backend), DEFAULT_MAX_ROW_BYTES).unwrap();
        assert_eq!(report.status, CaseStatus::Passed, "{report:#?}");
        assert_eq!(report.output_hash.as_deref(), Some(expected.as_str()));
    }
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
