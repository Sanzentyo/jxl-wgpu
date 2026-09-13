use crate::color::matrix::{IDENTITY, multiply};
use crate::{Chromaticity, ColorMatrix, RgbColorSpace, WhitePointAdaptation};

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

/// A selected ICC profile endpoint or an unbounded linear RGB connection. Linear RGB has
/// three channels and no ICC device-domain clipping or fixed-point profile approximation.
#[derive(Clone, Debug, PartialEq)]
pub enum IccTransformEndpoint {
    Profile(IccMatrixTrc),
    LinearRgb(RgbColorSpace),
}

impl IccTransformEndpoint {
    #[must_use]
    pub fn channels(&self) -> usize {
        match self {
            Self::Profile(profile) => profile.curves.len(),
            Self::LinearRgb(_) => 3,
        }
    }

    /// Selected device curves; a linear RGB connection has no curve descriptors.
    #[must_use]
    pub fn curves(&self) -> &[IccCurve] {
        match self {
            Self::Profile(profile) => &profile.curves,
            Self::LinearRgb(_) => &[],
        }
    }

    #[must_use]
    pub const fn profile(&self) -> Option<&IccMatrixTrc> {
        match self {
            Self::Profile(profile) => Some(profile),
            Self::LinearRgb(_) => None,
        }
    }
}

/// ICC colorimetric program. Device endpoints follow their profile's bounded curve rules;
/// linear RGB endpoints preserve signed values and values above one. Alpha and extra channels
/// are outside this transform. Relative intent connects PCS D50 and RGB whites using Bradford.
#[derive(Clone, Debug, PartialEq)]
pub struct IccTransform {
    source: IccTransformEndpoint,
    target: IccTransformEndpoint,
    matrix: [[f64; 3]; 3],
}

impl IccTransform {
    pub fn new(
        source: &IccProfile,
        target: &IccProfile,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::Profile(source.matrix_trc(IccDirection::DeviceToPcs, intent)?),
            IccTransformEndpoint::Profile(target.matrix_trc(IccDirection::PcsToDevice, intent)?),
        )
    }

    /// Convert a profile's device channels to unbounded linear RGB. Source colorants and
    /// independent curves retain their exact ICC values; no synthetic profile is serialized.
    pub fn to_linear_rgb(
        source: &IccProfile,
        target: RgbColorSpace,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::Profile(source.matrix_trc(IccDirection::DeviceToPcs, intent)?),
            IccTransformEndpoint::LinearRgb(target),
        )
    }

    /// Convert unbounded linear RGB to a profile's device channels. Inputs are not clamped
    /// before the PCS matrix; only the selected ICC inverse curve applies device bounds.
    pub fn from_linear_rgb(
        source: RgbColorSpace,
        target: &IccProfile,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::LinearRgb(source),
            IccTransformEndpoint::Profile(target.matrix_trc(IccDirection::PcsToDevice, intent)?),
        )
    }

    fn connect(
        source: IccTransformEndpoint,
        target: IccTransformEndpoint,
    ) -> Result<Self, IccError> {
        let source_matrix = match &source {
            IccTransformEndpoint::Profile(profile) => profile.matrix,
            IccTransformEndpoint::LinearRgb(space) => *ColorMatrix::rgb_to_xyz(
                *space,
                Chromaticity::ICC_D50,
                WhitePointAdaptation::Bradford,
            )?
            .rows(),
        };
        let target_inverse = match &target {
            IccTransformEndpoint::Profile(profile) if profile.curves.len() == 1 => {
                // Gray uses PCS Y; no implicit black-point adaptation.
                [[0.0, 1.0, 0.0], [0.0; 3], [0.0; 3]]
            }
            IccTransformEndpoint::Profile(profile) => inverse(profile.matrix)?,
            IccTransformEndpoint::LinearRgb(space) => *ColorMatrix::xyz_to_rgb(
                Chromaticity::ICC_D50,
                *space,
                WhitePointAdaptation::Bradford,
            )?
            .rows(),
        };
        let matrix = if matches!((&source, &target),
            (IccTransformEndpoint::Profile(s), IccTransformEndpoint::Profile(t))
                if s.matrix == t.matrix && s.curves.len() == t.curves.len())
        {
            // Exact cancellation also retains the specified endpoint of sampled plateaus.
            IDENTITY
        } else {
            multiply(target_inverse, source_matrix)
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
    pub const fn source(&self) -> &IccTransformEndpoint {
        &self.source
    }
    #[must_use]
    pub const fn target(&self) -> &IccTransformEndpoint {
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

fn inverse(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], IccError> {
    crate::color::matrix::inverse(matrix).map_err(|_| IccError::Matrix)
}
