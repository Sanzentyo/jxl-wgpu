use super::{ColorSpace, Mapping, gamut, oracle};

pub(super) fn map(
    light: [f64; 3],
    interval: [[f64; 2]; 3],
    matrix: [[f64; 3]; 3],
    target: ColorSpace,
    tone: Option<Mapping>,
    preference: f64,
) -> ([f64; 3], [[f64; 2]; 3]) {
    let converted = matrix.map(|row| (0..3).map(|c| row[c] * light[c]).sum());
    let bounds = matrix.map(|row| {
        std::array::from_fn(|edge| {
            (0..3)
                .map(|c| row[c] * interval[c][if row[c] >= 0.0 { edge } else { 1 - edge }])
                .sum()
        })
    });
    let luminance = oracle::luminance(target);
    let protected = tone.map_or(0.0, |tone| tone.protected / tone.source[1]);
    let globally_protected =
        tone.is_some_and(|tone| tone.protected >= tone.source[1].min(tone.target[1]));
    let y = gamut::luminance(converted, luminance);
    let ys: [f64; 2] =
        std::array::from_fn(|edge| gamut::luminance(bounds.map(|c| c[edge]), luminance));
    let (mapped, bounds) = if let Some(tone) = tone {
        let mapped = tone.apply(converted, luminance, [1.0; 3]);
        let bounds = tone.interval(bounds, luminance, [1.0; 3]);
        (mapped, expand(bounds, mapped, 8e-5))
    } else {
        (converted, bounds)
    };
    let expected = if globally_protected || (protected > 0.0 && y < protected) {
        mapped
    } else {
        gamut::apply(mapped, luminance, preference)
    };
    let bounds = if globally_protected || (protected > 0.0 && ys[1] < protected) {
        bounds
    } else {
        let gamut_bounds = gamut::interval(bounds, luminance, preference);
        if protected > 0.0 && ys[0] < protected {
            std::array::from_fn(|c| {
                [
                    bounds[c][0].min(gamut_bounds[c][0]),
                    bounds[c][1].max(gamut_bounds[c][1]),
                ]
            })
        } else {
            gamut_bounds
        }
    };
    (expected, expand(bounds, expected, 4e-6))
}

fn expand(bounds: [[f64; 2]; 3], center: [f64; 3], allowance: f64) -> [[f64; 2]; 3] {
    std::array::from_fn(|c| {
        let error = allowance * (1.0 + center[c].abs());
        [bounds[c][0] - error, bounds[c][1] + error]
    })
}
