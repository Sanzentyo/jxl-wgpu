use super::{IccCurve, IccDirection, IccError, IccProfile, IccRenderingIntent, IccSignature};

/// Exact profile matrix and independent curves selected for one direction. Colorants are
/// stored PCS-relative: applying `chad` to them again would double-adapt the profile.
#[derive(Clone, Debug, PartialEq)]
pub struct IccMatrixTrc {
    curves: Vec<IccCurve>,
    matrix: [[f64; 3]; 3],
    media_white: [i32; 3],
    chromatic_adaptation: Option<[i32; 9]>,
}

impl IccMatrixTrc {
    #[must_use]
    pub fn curves(&self) -> &[IccCurve] {
        &self.curves
    }
    /// RGB-to-PCS rows. A gray profile uses the first column to scale its one channel to D50.
    #[must_use]
    pub const fn matrix(&self) -> &[[f64; 3]; 3] {
        &self.matrix
    }
    #[must_use]
    pub const fn media_white(&self) -> [i32; 3] {
        self.media_white
    }
    #[must_use]
    pub const fn chromatic_adaptation(&self) -> Option<[i32; 9]> {
        self.chromatic_adaptation
    }
}

impl IccProfile {
    /// Selects a matrix/TRC transform without bypassing a higher-priority LUT. Unimplemented
    /// DToB/BToD elements currently return an explicit error instead of silently using a
    /// lower-priority profile method. This conservative policy can expand with MPE support.
    pub fn matrix_trc(
        &self,
        direction: IccDirection,
        intent: IccRenderingIntent,
    ) -> Result<IccMatrixTrc, IccError> {
        let header = self.header();
        if !matches!(&header.class.0, b"scnr" | b"mntr" | b"prtr") {
            return Err(IccError::Unsupported {
                field: "profile class",
                signature: header.class,
            });
        }
        let suffix = b'0' + intent as u8;
        let mpe = match direction {
            IccDirection::DeviceToPcs => IccSignature([b'D', b'2', b'B', suffix]),
            IccDirection::PcsToDevice => IccSignature([b'B', b'2', b'D', suffix]),
        };
        let base = match direction {
            IccDirection::DeviceToPcs => *b"A2B0",
            IccDirection::PcsToDevice => *b"B2A0",
        };
        let mut selected = base;
        selected[3] = b'0'
            + if intent == IccRenderingIntent::Absolute {
                1
            } else {
                intent as u8
            };
        for tag in [mpe, IccSignature(selected), IccSignature(base)] {
            if self.tag(tag).is_some() {
                return Err(IccError::TransformTag { tag });
            }
        }
        if intent != IccRenderingIntent::Relative {
            return Err(IccError::RenderingIntent { intent });
        }
        if header.pcs != IccSignature(*b"XYZ ") {
            return Err(IccError::Unsupported {
                field: "PCS",
                signature: header.pcs,
            });
        }
        let media_white = xyz_tag(self, *b"wtpt")?;
        if media_white.iter().any(|v| *v <= 0) {
            return Err(IccError::Invalid {
                field: "media white point",
                offset: u64::from(
                    self.tag(IccSignature(*b"wtpt"))
                        .expect("read white tag")
                        .offset,
                ),
            });
        }
        let chromatic_adaptation = if let Some(tag) = self.tag(IccSignature(*b"chad")) {
            if tag.kind != IccSignature(*b"sf32") || tag.size != 44 {
                return Err(IccError::TagType {
                    tag: tag.signature,
                    kind: tag.kind,
                });
            }
            let data = self.required(tag.signature)?;
            let mut values = [0; 9];
            for (i, value) in values.iter_mut().enumerate() {
                *value = data.i32(8 + i as u64 * 4)?;
            }
            inverse(std::array::from_fn(|r| {
                std::array::from_fn(|c| f64::from(values[r * 3 + c]) / 65536.0)
            }))?;
            Some(values)
        } else {
            None
        };
        let (matrix, curve_tags) = match &header.device_space.0 {
            b"RGB " if header.class != IccSignature(*b"prtr") => {
                let columns = [
                    xyz_tag(self, *b"rXYZ")?,
                    xyz_tag(self, *b"gXYZ")?,
                    xyz_tag(self, *b"bXYZ")?,
                ];
                let matrix = std::array::from_fn(|r| {
                    std::array::from_fn(|c| f64::from(columns[c][r]) / 65536.0)
                });
                if direction == IccDirection::PcsToDevice {
                    inverse(matrix)?;
                }
                (matrix, vec![*b"rTRC", *b"gTRC", *b"bTRC"])
            }
            b"GRAY" => {
                let white = header.illuminant.map(|v| f64::from(v) / 65536.0);
                (white.map(|v| [v, 0.0, 0.0]), vec![*b"kTRC"])
            }
            _ => {
                return Err(IccError::Unsupported {
                    field: "device space / profile class",
                    signature: header.device_space,
                });
            }
        };
        let curves = curve_tags
            .into_iter()
            .map(|tag| IccCurve::parse(self, IccSignature(tag)))
            .collect::<Result<Vec<_>, _>>()?;
        if direction == IccDirection::PcsToDevice {
            for curve in &curves {
                curve.inverse_direction()?;
            }
        }
        Ok(IccMatrixTrc {
            curves,
            matrix,
            media_white,
            chromatic_adaptation,
        })
    }
}

/// Device-to-device colorimetric program. Input/output channel counts follow their profiles;
/// alpha and extra channels are deliberately outside this pixel color transform.
#[derive(Clone, Debug, PartialEq)]
pub struct IccTransform {
    source: IccMatrixTrc,
    target: IccMatrixTrc,
    matrix: [[f64; 3]; 3],
}

impl IccTransform {
    pub fn new(
        source: &IccProfile,
        target: &IccProfile,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        let source = source.matrix_trc(IccDirection::DeviceToPcs, intent)?;
        let target = target.matrix_trc(IccDirection::PcsToDevice, intent)?;
        let target_inverse = if target.curves.len() == 1 {
            // Gray uses PCS Y; no chromatic or black-point adaptation is implicit here.
            [[0.0, 1.0, 0.0], [0.0; 3], [0.0; 3]]
        } else {
            inverse(target.matrix)?
        };
        let matrix = if source.matrix == target.matrix && source.curves.len() == target.curves.len()
        {
            // Preserve exact cancellation. A tiny residual can select a different end of a
            // sampled plateau even though the mathematical matrix product is identity.
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        } else {
            std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    (0..3)
                        .map(|k| target_inverse[r][k] * source.matrix[k][c])
                        .sum::<f64>()
                })
            })
        };
        if matrix.iter().flatten().any(|v| !v.is_finite()) {
            return Err(IccError::Matrix);
        }
        Ok(Self {
            source,
            target,
            matrix,
        })
    }

    #[must_use]
    pub const fn source(&self) -> &IccMatrixTrc {
        &self.source
    }
    #[must_use]
    pub const fn target(&self) -> &IccMatrixTrc {
        &self.target
    }
    #[must_use]
    pub const fn matrix(&self) -> &[[f64; 3]; 3] {
        &self.matrix
    }
}

fn xyz_tag(profile: &IccProfile, signature: [u8; 4]) -> Result<[i32; 3], IccError> {
    let tag = IccSignature(signature);
    let data = profile.required(tag)?;
    let kind = data.signature(0)?;
    if kind != IccSignature(*b"XYZ ") {
        return Err(IccError::TagType { tag, kind });
    }
    if data.0.len() != 20 {
        return Err(IccError::Invalid {
            field: "XYZ element size",
            offset: 0,
        });
    }
    Ok([data.i32(8)?, data.i32(12)?, data.i32(16)?])
}

fn inverse(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], IccError> {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let adj = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let det = a * adj[0][0] + b * adj[1][0] + c * adj[2][0];
    if !det.is_finite() || det == 0.0 {
        return Err(IccError::Matrix);
    }
    let result = adj.map(|row| row.map(|v| v / det));
    if result.iter().flatten().any(|v| !v.is_finite()) {
        return Err(IccError::Matrix);
    }
    Ok(result)
}
