use super::{Case, Family, Noise, Pattern, ReferenceSource};

pub(super) fn extend(cases: &mut Vec<Case>) {
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                for zero in [false, true] {
                    let name = format!("jpeg_{cb}{y}{cr}");
                    let mut case = Case::new(
                        format!("{name}{}", if zero { "_zero" } else { "" }),
                        [
                            format!("jpeg_sampling/odd_{cr}{cb}{y}"),
                            format!("jpeg_sampling/odd_{cb}{y}{cr}"),
                        ],
                        Family::Jpeg,
                        Pattern::AllModes,
                    );
                    if zero {
                        case.sources
                            .iter_mut()
                            .for_each(|source| source.noise = Noise::Zero);
                        case.controls.push(name);
                    }
                    cases.push(case);
                }
                let mut case = Case::new(
                    format!("jpeg_{cb}{y}{cr}_padding"),
                    [
                        format!("jpeg_sampling/odd_{cr}{cb}{y}"),
                        format!("jpeg_sampling/odd_{cb}{y}{cr}"),
                    ],
                    Family::Jpeg,
                    Pattern::JpegPadding,
                );
                case.sources
                    .iter_mut()
                    .for_each(|source| source.noise = Noise::Zero);
                cases.push(case);
            }
        }
    }
    for sampling in ["422", "440", "420"] {
        for filter in ["gab", "epf1", "gab_epf2", "gab_epf3"] {
            let mut case = Case::new(
                format!("jpeg_{sampling}_{filter}"),
                [
                    format!("noise/jpeg_{sampling}"),
                    format!("noise/jpeg_{sampling}_{filter}"),
                ],
                Family::Jpeg,
                Pattern::AllModes,
            );
            // Independent Gaborish and jxl-oxide checks establish libjxl's vertical-restoration defect.
            if matches!(sampling, "440" | "420") && matches!(filter, "gab" | "gab_epf3") {
                case.reference_source = ReferenceSource::JxlOxideLinear;
                case.linear_tolerance = 1e-5;
            }
            cases.push(case);
        }
        for (suffix, pattern) in [("empty", Pattern::Empty), ("edge", Pattern::PaddedEdge)] {
            let mut case = Case::new(
                format!("jpeg_{sampling}_{suffix}"),
                std::array::from_fn(|_| format!("noise/jpeg_{sampling}")),
                Family::Jpeg,
                pattern,
            );
            case.sources
                .iter_mut()
                .for_each(|source| source.noise = Noise::Zero);
            cases.push(case);
        }
    }
}
