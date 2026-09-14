use super::{IccError, IccMatrixTrc, IccRenderingIntent, IccSignature, IccTransformEndpoint};
use crate::icc::{IccAffine, IccBlackPointConnection, IccStage};

// Decimal PCS D50 used by the ICC CMM connection policy. Profile matrices retain their
// original fixed-point encoding; neither their colorants nor their CHAD is rewritten.
const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

pub(super) struct Connection {
    scale: [f64; 3],
    offset: [f64; 3],
    probe: Option<IccBlackPointConnection>,
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
                probe: None,
            };
        }
        let compensate = matches!(
            intent,
            IccRenderingIntent::Perceptual | IccRenderingIntent::Saturation
        ) && target
            .profile()
            .is_none_or(|profile| profile.header.version >> 24 == 4);
        if compensate {
            let target = black(target).expect("automatic compensation targets a v4 endpoint");
            let Some(source_black) = black(source) else {
                let profile = source.profile().expect("selected v2 LUT source");
                return Self {
                    scale: [1.0; 3],
                    offset: [0.0; 3],
                    probe: Some(IccBlackPointConnection {
                        source: profile.program.clone(),
                        input: dark_input(profile.header.device_space)
                            .expect("known darker colorant")
                            .into(),
                        target,
                    }),
                };
            };
            if source_black != target {
                let scale =
                    std::array::from_fn(|c| (D50[c] - target[c]) / (D50[c] - source_black[c]));
                return Self {
                    scale,
                    offset: std::array::from_fn(|c| target[c] - scale[c] * source_black[c]),
                    probe: None,
                };
            }
        }
        Self {
            scale: [1.0; 3],
            offset: [0.0; 3],
            probe: None,
        }
    }

    pub(super) fn is_identity(&self) -> bool {
        self.probe.is_none() && self.scale == [1.0; 3] && self.offset == [0.0; 3]
    }

    pub(super) fn into_stage(self) -> Result<IccStage, IccError> {
        if let Some(probe) = self.probe {
            return Ok(IccStage::BlackPointConnection(probe));
        }
        Ok(IccStage::Matrix(IccAffine::new(
            3,
            (0..9)
                .map(|i| {
                    if i / 3 == i % 3 {
                        self.scale[i / 3]
                    } else {
                        0.0
                    }
                })
                .collect(),
            self.offset.to_vec(),
            false,
        )?))
    }
}

fn media_white(endpoint: &IccTransformEndpoint) -> [f64; 3] {
    match endpoint {
        // The virtual endpoint is an already adapted, ideal v4 display in PCS D50.
        IccTransformEndpoint::Rgb { .. } => D50,
        IccTransformEndpoint::Profile(profile)
            if profile.tag.is_some_and(|tag| tag.0[3] == b'3') =>
        {
            D50
        }
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

fn black(endpoint: &IccTransformEndpoint) -> Option<[f64; 3]> {
    Some(match endpoint {
        IccTransformEndpoint::Rgb { .. } => [0.0; 3],
        IccTransformEndpoint::Profile(profile) => match &profile.matrix_trc {
            Some(matrix) => profile_black(matrix),
            // Selected v4 LUT/MPE perceptual/saturation methods use the PCS reference black.
            // Unused matrix-shaper tags do not change the selected method's meaning.
            None if profile.header.version >> 24 == 4 => [0.00336, 0.0034731, 0.0028646],
            None if dark_input(profile.header.device_space).is_some() => return None,
            // The CMM has no darker-colorant endpoint for uncommon device spaces.
            // An unavailable estimate means zero PCS black, not the v4 reference black.
            None => [0.0; 3],
        },
    })
}

fn dark_input(space: IccSignature) -> Option<&'static [f64]> {
    Some(match &space.0 {
        b"GRAY" => &[0.0],
        b"RGB " => &[0.0; 3],
        b"CMY " => &[1.0; 3],
        b"CMYK" => &[1.0; 4],
        b"Lab " => &[0.0, 128.0 / 255.0, 128.0 / 255.0],
        _ => return None,
    })
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
