use jxl_gpu_formats::TransferFunction;
use jxl_test_support::oracles::hdr::interval;
type Interval = [f64; 2];

pub(super) fn original_bounds(
    case: &super::corpus::Case,
    original: &[f32],
    linear: Option<&[f32]>,
    pixel: usize,
) -> [Interval; 4] {
    let expected = &original[pixel * 4..][..4];
    if case.xyb && !case.sequence {
        // PQ's derivative grows sharply around black. Apply the existing native
        // linear reconstruction budget before the OETF; an encoded fixed epsilon
        // would conflate IDCT/XYB rounding with the separately tested transfer.
        let linear = &linear.unwrap()[pixel * 4..][..4];
        let rgb = [linear[0], linear[1], linear[2]].map(f64::from);
        let bounds = interval(
            rgb,
            TransferFunction::Linear,
            case.transfer,
            case.space,
            case.space,
            case.nits,
            f64::from(case.tolerance()),
        );
        std::array::from_fn(|c| {
            if c == 3 {
                [f64::from(expected[c]) - 2e-6, f64::from(expected[c]) + 2e-6]
            } else {
                let packing = 5e-5 * (1.0 + f64::from(expected[c]).abs());
                [bounds[c][0] - packing, bounds[c][1] + packing]
            }
        })
    } else {
        std::array::from_fn(|c| {
            let value = f64::from(expected[c]);
            let tolerance = if c == 3 {
                2e-6
            } else {
                f64::from(case.tolerance()) * (1.0 + value.abs())
            };
            [value - tolerance, value + tolerance]
        })
    }
}
