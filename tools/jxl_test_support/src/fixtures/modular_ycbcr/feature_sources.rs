use super::{Case, SampleBitDepth, Transform, transforms::split};

pub(super) fn extend(cases: &mut Vec<Case>) {
    for factor in [1, 2, 4, 8] {
        for local in [false, true] {
            let location = if local { "local" } else { "global" };
            let mut case = Case::new(format!("feature_{location}_up{factor}"));
            case.size = [256 * factor + 3, 37];
            case.group_size_shift = 0;
            case.bit_depth = SampleBitDepth::Float {
                bits_per_sample: 32,
                exponent_bits_per_sample: 8,
            };
            case.upsampling = factor;
            case.extra_factors = vec![factor; 2];
            case.selectors = if local { [1, 2, 3] } else { [0, 1, 0] };
            let transforms = if local {
                &mut case.pass_transforms
            } else {
                &mut case.global_transforms
            };
            *transforms = vec![Transform::Squeeze(vec![
                split(true, false, 0, 3),
                split(false, false, 0, 3),
            ])];
            cases.push(case);
        }
    }
    for selectors in [[0, 1, 0], [1, 2, 3]] {
        let mut case = Case::new(format!(
            "feature_mixed_{}{}{}",
            selectors[0], selectors[1], selectors[2]
        ));
        case.size = [257, 17];
        case.bit_depth = SampleBitDepth::Integer { bits_per_sample: 8 };
        case.selectors = selectors;
        cases.push(case);
    }
}
