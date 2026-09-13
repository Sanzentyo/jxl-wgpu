use super::Case;
use jxl_gpu_bitstream::SampleBitDepth;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Squeeze {
    pub horizontal: bool,
    pub in_place: bool,
    pub begin_channel: u32,
    pub channel_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Transform {
    Rct {
        begin_channel: u32,
        rct_type: u32,
    },
    Palette {
        begin_channel: u32,
        channel_count: u32,
    },
    /// An empty parameter list requests the normative default schedule.
    Squeeze(Vec<Squeeze>),
}

pub(super) fn split(
    horizontal: bool,
    in_place: bool,
    begin_channel: u32,
    channel_count: u32,
) -> Squeeze {
    Squeeze {
        horizontal,
        in_place,
        begin_channel,
        channel_count,
    }
}

pub(super) fn rct(rct_type: u32, begin_channel: u32) -> Transform {
    Transform::Rct {
        begin_channel,
        rct_type,
    }
}

pub(super) fn palette(begin_channel: u32, channel_count: u32) -> Transform {
    Transform::Palette {
        begin_channel,
        channel_count,
    }
}

pub(super) fn default_squeeze() -> Transform {
    Transform::Squeeze(Vec::new())
}

pub(super) fn extend(cases: &mut Vec<Case>) {
    for kind in 0..42 {
        let mut case = Case::new(format!("rct_{kind}"));
        case.selectors = [0; 3];
        case.global_transforms = vec![rct(kind, 0)];
        cases.push(case);
    }
    for cb in 0..4 {
        for y in 0..4 {
            for cr in 0..4 {
                let mut case = Case::new(format!("squeeze_sampling_{cb}{y}{cr}"));
                case.selectors = [cb, y, cr];
                case.global_transforms = vec![default_squeeze()];
                cases.push(case);
            }
        }
    }
    for selectors in [[0, 0, 0], [0, 1, 0], [1, 2, 3]] {
        for size in [[1, 1], [1, 19], [37, 1]] {
            let [cb, y, cr] = selectors;
            let [width, height] = size;
            let mut case = Case::new(format!("squeeze_thin_{cb}{y}{cr}_{width}x{height}"));
            case.selectors = selectors;
            case.size = size;
            case.global_transforms = vec![default_squeeze()];
            cases.push(case);
        }
    }
    for shift in 0..=3 {
        for vertical in [false, true] {
            let direction = if vertical { "vertical" } else { "horizontal" };
            let mut case = Case::new(format!("squeeze_groups_{}_{direction}", 128 << shift));
            case.group_size_shift = shift;
            case.size[usize::from(vertical)] = (256 << shift) + 3;
            case.global_transforms = vec![default_squeeze()];
            cases.push(case);
        }
    }
    for in_place in [false, true] {
        let mut case = Case::new(
            if in_place {
                "squeeze_in_place"
            } else {
                "squeeze_append"
            }
            .into(),
        );
        case.selectors = [1, 2, 3];
        case.size = [259, 129];
        case.group_size_shift = 0;
        case.global_transforms = vec![Transform::Squeeze(vec![
            split(true, in_place, 0, 3),
            split(false, in_place, 0, 3),
        ])];
        cases.push(case);
    }
    let mut lf = Case::new("squeeze_lf".into());
    lf.size[0] = 2051;
    lf.group_size_shift = 0;
    lf.passes = 2;
    lf.global_transforms = vec![Transform::Squeeze(
        (0..3)
            .flat_map(|_| [split(true, true, 0, 3), split(false, true, 0, 3)])
            .collect(),
    )];
    cases.push(lf);
    let mut passes = Case::new("squeeze_passes".into());
    passes.size = [259, 37];
    passes.group_size_shift = 0;
    passes.passes = 2;
    passes.global_transforms = vec![default_squeeze()];
    cases.push(passes);
    for bits in [8, 31] {
        let mut case = Case::new(format!("squeeze_integer_{bits}"));
        case.bit_depth = SampleBitDepth::Integer {
            bits_per_sample: bits,
        };
        case.global_transforms = vec![default_squeeze()];
        cases.push(case);
    }
    for (bits, exponent) in [(16, 5), (24, 7), (32, 8)] {
        let mut case = Case::new(format!("squeeze_float_{bits}"));
        case.bit_depth = SampleBitDepth::Float {
            bits_per_sample: bits,
            exponent_bits_per_sample: exponent,
        };
        case.global_transforms = vec![default_squeeze()];
        cases.push(case);
    }
    for channel in 0..3 {
        for grouped in [false, true] {
            let mut case = Case::new(format!(
                "{}_{channel}",
                if grouped { "palette_groups" } else { "palette" }
            ));
            case.selectors = [1, 2, 3];
            if grouped {
                case.size = [259, 37];
                case.group_size_shift = 0;
            }
            case.global_transforms = vec![palette(channel, 1)];
            cases.push(case);
        }
    }
    let mut case = Case::new("palette_rgb".into());
    case.selectors = [0; 3];
    case.global_transforms = vec![palette(0, 3)];
    cases.push(case.clone());
    case.name = "rct_palette_squeeze".into();
    case.global_transforms = vec![rct(6, 0), palette(0, 3), default_squeeze()];
    cases.push(case.clone());
    case.name = "palette_squeeze".into();
    case.selectors = [1, 2, 3];
    case.global_transforms = vec![palette(0, 1), default_squeeze()];
    cases.push(case);
    let mut residual = Case::new("squeeze_residual_rct".into());
    residual.selectors = [0; 3];
    residual.global_transforms = vec![
        Transform::Squeeze(vec![split(true, false, 0, 3)]),
        rct(41, 3),
    ];
    cases.push(residual);
    let mut extras = Case::new("squeeze_extras".into());
    extras.size = [259, 37];
    extras.group_size_shift = 0;
    extras.extra_factors = vec![2, 8];
    extras.global_transforms = vec![Transform::Squeeze(vec![
        split(true, false, 0, 3),
        split(false, false, 0, 3),
        split(true, false, 3, 1),
    ])];
    cases.push(extras);
    let mut restored = Case::new("restored_palette_squeeze".into());
    restored.selectors = [1, 2, 3];
    restored.gaborish = true;
    restored.epf_iterations = 3;
    restored.global_transforms = vec![palette(0, 1), default_squeeze()];
    cases.push(restored);
    for kind in 0..42 {
        for horizontal in [true, false] {
            let size = if horizontal { [1, 19] } else { [37, 1] };
            let mut case = Case::new(format!("rct_empty_{kind}_{}x{}", size[0], size[1]));
            case.selectors = [0; 3];
            case.size = size;
            case.global_transforms = vec![
                Transform::Squeeze(vec![split(horizontal, false, 0, 3)]),
                rct(kind, 3),
            ];
            cases.push(case);
        }
    }
}
