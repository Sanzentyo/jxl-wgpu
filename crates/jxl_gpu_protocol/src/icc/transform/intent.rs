use super::{IccMatrixTrc, IccRenderingIntent, IccSignature, IccTransformEndpoint};

// Decimal PCS D50 used by the ICC CMM connection policy. Profile matrices retain their
// original fixed-point encoding; neither their colorants nor their CHAD is rewritten.
const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

pub(super) struct Connection {
    pub(super) scale: [f64; 3],
    pub(super) offset: [f64; 3],
}

impl Connection {
    pub(super) fn new(
        source: &IccTransformEndpoint,
        target: &IccTransformEndpoint,
        intent: IccRenderingIntent,
    ) -> Self {
        if intent == IccRenderingIntent::Absolute {
            let source = media_white(source);
            let target = media_white(target);
            return Self {
                scale: std::array::from_fn(|c| source[c] / target[c]),
                offset: [0.0; 3],
            };
        }
        let compensate = matches!(
            intent,
            IccRenderingIntent::Perceptual | IccRenderingIntent::Saturation
        ) && target
            .profile()
            .is_none_or(|profile| profile.header.version >> 24 == 4);
        if compensate {
            let source = black(source);
            let target = black(target);
            if source != target {
                let scale = std::array::from_fn(|c| (D50[c] - target[c]) / (D50[c] - source[c]));
                return Self {
                    scale,
                    offset: std::array::from_fn(|c| target[c] - scale[c] * source[c]),
                };
            }
        }
        Self {
            scale: [1.0; 3],
            offset: [0.0; 3],
        }
    }

    pub(super) fn is_identity(&self) -> bool {
        self.scale == [1.0; 3] && self.offset == [0.0; 3]
    }
}

fn media_white(endpoint: &IccTransformEndpoint) -> [f64; 3] {
    match endpoint {
        // The virtual endpoint is an already adapted, ideal v4 display in PCS D50.
        IccTransformEndpoint::LinearRgb(_) => D50,
        IccTransformEndpoint::Profile(profile)
            if profile.header.version >> 24 == 2
                && profile.header.class == IccSignature(*b"mntr") =>
        {
            D50
        }
        IccTransformEndpoint::Profile(profile) => {
            profile.media_white.map(|v| f64::from(v) / 65536.0)
        }
    }
}

fn black(endpoint: &IccTransformEndpoint) -> [f64; 3] {
    match endpoint {
        IccTransformEndpoint::LinearRgb(_) => [0.0; 3],
        IccTransformEndpoint::Profile(profile) => profile_black(profile),
    }
}

fn profile_black(profile: &IccMatrixTrc) -> [f64; 3] {
    // At most three metadata endpoints are evaluated. Pixel curves remain GPU work.
    let values: [f64; 3] = std::array::from_fn(|c| {
        profile
            .curves
            .get(c)
            .map_or(0.0, |curve| curve.black_level())
    });
    let xyz = profile
        .matrix
        .map(|row| row.into_iter().zip(values).map(|(a, b)| a * b).sum::<f64>());
    let fy = lab_function(xyz[1]);
    let lightness = 116.0 * fy - 16.0;
    // Matrix-shaper BPC follows the CMM darker-colorant policy, including synthetic
    // negative profiles. Clipping L* must retain a* and b*, not neutralize the black.
    let clipped = if lightness > 95.0 {
        0.0
    } else {
        lightness.clamp(0.0, 50.0)
    };
    if lightness == clipped {
        return xyz;
    }
    let delta = (clipped + 16.0) / 116.0 - fy;
    std::array::from_fn(|c| D50[c] * inverse_lab_function(lab_function(xyz[c] / D50[c]) + delta))
}

fn lab_function(value: f64) -> f64 {
    if value > 216.0 / 24389.0 {
        value.cbrt()
    } else {
        (24389.0 / 27.0 * value + 16.0) / 116.0
    }
}

fn inverse_lab_function(value: f64) -> f64 {
    if value > 6.0 / 29.0 {
        value * value * value
    } else {
        (116.0 * value - 16.0) * 27.0 / 24389.0
    }
}
