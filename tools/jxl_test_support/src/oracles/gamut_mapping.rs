//! Independent F64 intersections of a color-to-neutral line with the RGB cube.

pub fn apply(rgb: [f64; 3], weights: [f64; 3], preference: f64) -> [f64; 3] {
    if rgb.iter().all(|v| (0.0..=1.0).contains(v)) {
        return rgb;
    }
    let y = luminance(rgb, weights);
    if y <= 0.0 {
        return [0.0; 3];
    }
    let negative = rgb
        .iter()
        .copied()
        .filter(|c| *c < 0.0)
        .map(|c| -c / (y - c))
        .fold(0.0_f64, f64::max);
    let upper = rgb
        .iter()
        .copied()
        .filter(|c| *c > 1.0 && *c > y)
        .map(|c| (c - 1.0) / (c - y))
        .fold(negative, f64::max);
    let neutral = (preference * negative + (1.0 - preference) * upper).clamp(0.0, 1.0);
    let mixed = rgb.map(|c| (1.0 - neutral) * c + neutral * y);
    let maximum = mixed.iter().copied().fold(1.0_f64, f64::max);
    mixed.map(|c| (c / maximum).clamp(0.0, 1.0))
}

pub fn luminance(rgb: [f64; 3], weights: [f64; 3]) -> f64 {
    rgb.into_iter().zip(weights).map(|(c, w)| c * w).sum()
}

/// Conservative interval arithmetic over the line/cube intersections. Intervals crossing
/// the zero-luminance boundary include black. No tolerances are inferred from GPU output.
pub fn interval(rgb: [[f64; 2]; 3], weights: [f64; 3], preference: f64) -> [[f64; 2]; 3] {
    assert!(weights.iter().all(|v| *v >= 0.0));
    let y: [f64; 2] = std::array::from_fn(|e| luminance(rgb.map(|c| c[e]), weights));
    if y[1] <= 0.0 {
        return [[0.0; 2]; 3];
    }
    if rgb.iter().all(|c| c[0] >= 0.0 && c[1] <= 1.0) {
        return rgb;
    }
    let mut negative = [0.0_f64; 2];
    for c in rgb {
        if c[1] < 0.0 {
            negative[0] = negative[0].max(-c[1] / (y[1] - c[1]));
        }
        if c[0] < 0.0 {
            negative[1] = negative[1].max(-c[0] / (y[0].max(0.0) - c[0]));
        }
    }
    let mut upper = negative;
    if preference < 1.0 {
        let cap = 1.0 / (1.0 - preference);
        for c in rgb {
            if c[0] > y[1].max(1.0) {
                upper[0] = upper[0].max(((c[0] - 1.0) / (c[1] - y[0].max(0.0))).min(cap));
            }
            if c[1] > y[0].max(1.0) {
                let denominator = c[0] - y[1];
                let bound = if denominator <= 0.0 {
                    cap
                } else {
                    ((c[1] - 1.0) / denominator).min(cap)
                };
                upper[1] = upper[1].max(bound);
            }
        }
    }
    let mix: [f64; 2] = std::array::from_fn(|e| {
        (preference * negative[e] + (1.0 - preference) * upper[e]).clamp(0.0, 1.0)
    });
    let mixed = rgb.map(|c| {
        let color = product(c, [1.0 - mix[1], 1.0 - mix[0]]);
        let gray = product([y[0].max(0.0), y[1]], mix);
        [color[0] + gray[0], color[1] + gray[1]]
    });
    let peak: [f64; 2] = std::array::from_fn(|e| mixed.iter().map(|c| c[e]).fold(1.0, f64::max));
    mixed.map(|c| {
        let low = if y[0] <= 0.0 {
            0.0
        } else {
            (c[0] / peak[1]).clamp(0.0, 1.0)
        };
        [low, (c[1] / peak[0]).clamp(0.0, 1.0)]
    })
}

fn product(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let values = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
    [
        values.into_iter().fold(f64::INFINITY, f64::min),
        values.into_iter().fold(f64::NEG_INFINITY, f64::max),
    ]
}

#[cfg(test)]
mod tests;
