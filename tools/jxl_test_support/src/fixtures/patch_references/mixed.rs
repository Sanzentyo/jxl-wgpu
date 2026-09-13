use super::{Case, Family, Pattern, ReferenceSource};

pub(super) fn extend(cases: &mut Vec<Case>) {
    for (name, sources, overwrite, reference) in [
        (
            "xyb",
            ["noise/modular_257x17", "noise/vardct_257x17"],
            true,
            ReferenceSource::NativeLinear,
        ),
        (
            "xyb_up2",
            ["noise/modular_up2", "noise/vardct_up2"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "xyb_up4",
            ["noise/modular_up4", "noise/vardct_up4"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "xyb_up8",
            ["noise/modular_up8", "noise/vardct_up8"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "rgb",
            ["noise/modular_rgb_group256", "noise/vardct_rgb_257x17"],
            true,
            ReferenceSource::NativeLinear,
        ),
        (
            "rgb_up2",
            ["noise/modular_rgb_up2", "noise/vardct_rgb_up2"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "rgb_up4",
            ["noise/modular_rgb_up4", "noise/vardct_rgb_up4"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "rgb_up8",
            ["noise/modular_rgb_up8", "noise/vardct_rgb_up8"],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "extras_up2",
            [
                "lf_patch_features/equal_up2_modular.lf1",
                "lf_patch_features/equal_up2_vardct.lf1",
            ],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "extras_up4",
            [
                "lf_patch_features/equal_up4_modular.lf1",
                "lf_patch_features/equal_up4_vardct.lf1",
            ],
            false,
            ReferenceSource::NativeLinear,
        ),
        (
            "extras_up8",
            [
                "lf_patch_features/equal_up8_modular.lf1",
                "lf_patch_features/equal_up8_vardct.lf1",
            ],
            true,
            ReferenceSource::NativeLinear,
        ),
        (
            "lf_up8",
            [
                "lf_patch_features/equal_up8_modular",
                "lf_patch_features/equal_up8_vardct",
            ],
            false,
            ReferenceSource::NativeLinear,
        ),
        // Native CMS loses accuracy above sRGB two; evaluate the specified transfer independently.
        (
            "jpeg_modular",
            ["noise/modular_rgb_group256", "noise/jpeg_420"],
            true,
            ReferenceSource::NativeSrgb,
        ),
        (
            "jpeg_rgb",
            ["noise/vardct_rgb_257x17", "noise/jpeg_422"],
            true,
            ReferenceSource::NativeSrgb,
        ),
    ] {
        for reverse in [false, true] {
            let sources = if reverse {
                [sources[1], sources[0]]
            } else {
                sources
            };
            for (suffix, pattern) in [("empty", Pattern::Empty), ("patches", Pattern::AllModes)] {
                let mut case = Case::new(
                    format!("mixed_{name}_{}_{suffix}", u32::from(reverse)),
                    sources.map(str::to_owned),
                    Family::Mixed,
                    pattern,
                );
                case.reference_source = reference.clone();
                if pattern == Pattern::AllModes {
                    case.controls
                        .push(format!("mixed_{name}_{}_empty", u32::from(reverse)));
                }
                cases.push(case);
            }
            if overwrite {
                let mut case = Case::new(
                    format!("mixed_{name}_{}_overwrite", u32::from(reverse)),
                    sources.map(str::to_owned),
                    Family::Mixed,
                    Pattern::Overwrite,
                );
                case.reference_source = reference.clone();
                cases.push(case);
            }
        }
    }
    for selectors in ["010", "123"] {
        for (name, source, vardct) in [
            ("modular_rgb", "noise/modular_rgb_group256", false),
            ("vardct_rgb", "noise/vardct_rgb_257x17", true),
            ("vardct_ycbcr", "noise/jpeg_420", true),
        ] {
            for reverse in [false, true] {
                let sources = [
                    source.to_owned(),
                    format!("modular_ycbcr/feature_mixed_{selectors}"),
                ];
                let sources = if reverse {
                    [sources[1].clone(), sources[0].clone()]
                } else {
                    sources
                };
                let name = format!(
                    "mixed_modular_ycbcr_{selectors}_{name}_{}",
                    u32::from(reverse)
                );
                for (suffix, pattern) in [
                    ("empty", Pattern::Empty),
                    ("patches", Pattern::AllModes),
                    ("overwrite", Pattern::Overwrite),
                ] {
                    let mut case = Case::new(
                        format!("{name}_{suffix}"),
                        sources.clone(),
                        Family::Mixed,
                        pattern,
                    );
                    case.reference_source = ReferenceSource::NativeSrgb;
                    // VarDCT keeps the established independent-IDCT error bound; lossless
                    // Modular components use the tighter native F32 comparison in both domains.
                    case.linear_tolerance = if vardct { 1.0 / 1024.0 } else { 1e-5 };
                    case.encoded_tolerance = Some(if vardct { 1.0 / 1024.0 } else { 2e-6 });
                    if pattern != Pattern::Empty {
                        case.controls.push(format!("{name}_empty"));
                    }
                    cases.push(case);
                }
            }
        }
    }
}
