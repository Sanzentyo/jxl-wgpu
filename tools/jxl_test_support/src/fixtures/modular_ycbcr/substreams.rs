use jxl_gpu_bitstream::SampleBitDepth;

use super::transforms::{default_squeeze, palette, rct, split};
use super::{Case, Transform};

fn grouped(name: String) -> Case {
    let mut case = Case::new(name);
    case.size = [259, 37];
    case.group_size_shift = 0;
    case
}

pub(super) fn extend(cases: &mut Vec<Case>) {
    for kind in 0..42 {
        let mut case = grouped(format!("local_rct_{kind}"));
        case.size[1] = 129;
        case.selectors = [0; 3];
        case.pass_transforms = vec![rct(kind, 0)];
        cases.push(case);
    }
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                let mut case = grouped(format!("local_squeeze_sampling_{cb}{y}{cr}"));
                case.selectors = [cb, y, cr];
                case.pass_transforms = vec![default_squeeze()];
                cases.push(case);
            }
        }
    }
    for channel in 0..3 {
        let mut case = grouped(format!("local_palette_{channel}"));
        case.selectors = [1, 2, 3];
        case.pass_transforms = vec![palette(channel, 1)];
        cases.push(case);
    }
    let mut case = grouped("local_palette_rgb".into());
    case.selectors = [0; 3];
    case.pass_transforms = vec![palette(0, 3)];
    cases.push(case.clone());
    case.name = "local_rct_palette_squeeze".into();
    case.size = [257, 129];
    case.pass_transforms = vec![rct(6, 0), palette(0, 3), default_squeeze()];
    cases.push(case);

    let mut residual = grouped("local_squeeze_residual_rct".into());
    residual.size = [257, 129];
    residual.selectors = [0; 3];
    residual.pass_transforms = vec![
        Transform::Squeeze(vec![split(true, false, 0, 3)]),
        rct(41, 3),
    ];
    cases.push(residual);
    for in_place in [false, true] {
        let mut case = grouped(
            if in_place {
                "local_squeeze_in_place"
            } else {
                "local_squeeze_append"
            }
            .into(),
        );
        case.size = [259, 129];
        case.selectors = [1, 2, 3];
        case.pass_transforms = vec![Transform::Squeeze(vec![
            split(true, in_place, 0, 3),
            split(false, in_place, 0, 3),
        ])];
        cases.push(case);
    }
    for shift in 0..=3 {
        for vertical in [false, true] {
            let direction = if vertical { "vertical" } else { "horizontal" };
            let mut case = Case::new(format!("local_squeeze_groups_{}_{direction}", 128 << shift));
            case.group_size_shift = shift;
            case.size[usize::from(vertical)] = (256 << shift) + 1;
            case.pass_transforms = vec![default_squeeze()];
            cases.push(case);
        }
    }
    let mut extras = grouped("local_squeeze_extras".into());
    extras.extra_factors = vec![2, 8];
    extras.pass_transforms = vec![Transform::Squeeze(vec![
        split(true, false, 0, 3),
        split(false, false, 0, 3),
    ])];
    cases.push(extras);

    for (name, transforms) in [
        ("local_lf_squeeze", vec![default_squeeze()]),
        ("local_lf_palette", vec![palette(0, 3), default_squeeze()]),
        ("local_lf_rct", vec![rct(41, 0)]),
    ] {
        let mut case = Case::new(name.into());
        case.size[0] = 2051;
        case.group_size_shift = 0;
        case.selectors = [0; 3];
        case.passes = 2;
        case.global_transforms = vec![Transform::Squeeze(
            (0..3)
                .flat_map(|_| [split(true, true, 0, 3), split(false, true, 0, 3)])
                .collect(),
        )];
        case.lf_transforms = transforms;
        case.pass_transforms = vec![default_squeeze()];
        cases.push(case);
    }
    let mut passes = grouped("local_passes".into());
    passes.passes = 2;
    passes.global_transforms = vec![default_squeeze()];
    passes.pass_transforms = vec![default_squeeze()];
    cases.push(passes);

    let mut restored = grouped("local_restored_palette_squeeze".into());
    restored.selectors = [1, 2, 3];
    restored.gaborish = true;
    restored.epf_iterations = 3;
    restored.pass_transforms = vec![palette(0, 1), default_squeeze()];
    cases.push(restored);
    for bits in [8, 31] {
        let mut case = grouped(format!("local_squeeze_integer_{bits}"));
        case.bit_depth = SampleBitDepth::Integer {
            bits_per_sample: bits,
        };
        case.scalar_inverse_reference = bits == 31;
        case.pass_transforms = vec![default_squeeze()];
        cases.push(case);
    }
    for (bits, exponent) in [(16, 5), (24, 7), (32, 8)] {
        let mut case = grouped(format!("local_squeeze_float_{bits}"));
        case.bit_depth = SampleBitDepth::Float {
            bits_per_sample: bits,
            exponent_bits_per_sample: exponent,
        };
        case.pass_transforms = vec![default_squeeze()];
        cases.push(case);
    }
    for factor in [2, 4, 8] {
        let mut case = grouped(format!("local_resampling_{factor}"));
        case.size[0] = 256 * factor + 3;
        case.bit_depth = SampleBitDepth::Float {
            bits_per_sample: 32,
            exponent_bits_per_sample: 8,
        };
        case.upsampling = factor;
        case.extra_factors = vec![factor, 8];
        case.pass_transforms = vec![Transform::Squeeze(vec![
            split(true, false, 0, 3),
            split(false, false, 0, 3),
        ])];
        cases.push(case);
    }
}
