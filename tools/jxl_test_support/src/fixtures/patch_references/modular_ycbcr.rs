use super::{Case, ColorErrorScale, Family, Noise, Pattern, ReferenceSource};

fn case(name: String, sources: [String; 2], pattern: Pattern) -> Case {
    let mut case = Case::new(
        name,
        sources.map(|source| format!("modular_ycbcr/{source}")),
        Family::ModularYcbcr,
        pattern,
    );
    case.reference_source = ReferenceSource::NativeSrgb;
    case.linear_tolerance = 1e-5;
    case.encoded_tolerance = Some(2e-6);
    case
}

pub(super) fn extend(cases: &mut Vec<Case>) {
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                for zero in [false, true] {
                    let name = format!("modular_ycbcr_{cb}{y}{cr}");
                    let mut case = case(
                        format!("{name}{}", if zero { "_zero" } else { "" }),
                        [
                            format!("sampling_{cr}{cb}{y}"),
                            format!("sampling_{cb}{y}{cr}"),
                        ],
                        Pattern::AllModes,
                    );
                    case.sources.iter_mut().for_each(|source| {
                        source.noise = if zero { Noise::Zero } else { Noise::Inject }
                    });
                    if zero {
                        case.controls.push(name);
                    }
                    cases.push(case);
                }
            }
        }
    }
    for (name, sources) in [
        ("rct", ["sampling_123", "rct_41"]),
        ("squeeze", ["rct_41", "squeeze_sampling_123"]),
        ("palette", ["palette_squeeze", "rct_palette_squeeze"]),
        ("thin", ["squeeze_thin_000_1x19", "rct_empty_41_1x19"]),
        ("passes", ["local_squeeze_sampling_123", "local_passes"]),
        ("local", ["local_rct_41", "local_squeeze_append"]),
        (
            "local_edge",
            ["local_rct_palette_squeeze", "local_squeeze_residual_rct"],
        ),
        ("lf", ["local_lf_rct", "local_lf_palette"]),
    ] {
        extend_family(cases, name, sources, ColorErrorScale::Component);
    }
    for (name, sources) in [
        ("extras", ["feature_local_up1", "feature_global_up1"]),
        ("resampling_2", ["feature_local_up2", "feature_global_up2"]),
        ("resampling_4", ["feature_local_up4", "feature_global_up4"]),
        ("resampling_8", ["feature_local_up8", "feature_global_up8"]),
    ] {
        extend_family(cases, name, sources, ColorErrorScale::PixelRgb);
    }
    for (name, sources) in [
        ("restoration", ["restoration_1", "restoration_3"]),
        (
            "global_restoration",
            ["restored_palette_squeeze", "restoration_2"],
        ),
        (
            "local_restoration",
            [
                "local_restored_palette_squeeze",
                "local_restored_palette_squeeze",
            ],
        ),
    ] {
        for (suffix, pattern) in [
            ("empty", Pattern::Empty),
            ("patches", Pattern::AllModes),
            ("overwrite", Pattern::Overwrite),
        ] {
            for zero in [false, true] {
                let prefix = format!("modular_ycbcr_{name}_{suffix}");
                let noise = if zero { Noise::Zero } else { Noise::Inject };
                let mut case = case(
                    format!("{prefix}{}", if zero { "_zero" } else { "" }),
                    sources.map(str::to_owned),
                    pattern,
                );
                case.sources
                    .iter_mut()
                    .for_each(|source| source.noise = noise);
                case.reference_source = ReferenceSource::ExpandedSrgb {
                    sources: sources.map(|source| super::Source {
                        name: format!("modular_ycbcr/{source}.expanded"),
                        noise,
                    }),
                };
                if zero {
                    case.controls.push(prefix);
                }
                if pattern != Pattern::Empty {
                    case.controls.push(format!(
                        "modular_ycbcr_{name}_empty{}",
                        if zero { "_zero" } else { "" }
                    ));
                }
                cases.push(case);
            }
        }
    }
}

fn extend_family(cases: &mut Vec<Case>, name: &str, sources: [&str; 2], scale: ColorErrorScale) {
    for (suffix, pattern) in [
        ("empty", Pattern::Empty),
        ("patches", Pattern::AllModes),
        ("overwrite", Pattern::Overwrite),
    ] {
        let mut case = case(
            format!("modular_ycbcr_{name}_{suffix}"),
            sources.map(str::to_owned),
            pattern,
        );
        case.color_error_scale = scale;
        if pattern != Pattern::Empty {
            case.controls.push(format!("modular_ycbcr_{name}_empty"));
        }
        cases.push(case);
    }
}
