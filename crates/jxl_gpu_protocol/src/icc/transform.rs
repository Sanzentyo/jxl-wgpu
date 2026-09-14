use crate::{Chromaticity, ColorMatrix, RgbColorSpace, WhitePointAdaptation};

use super::{
    IccAffine, IccCurve, IccDirection, IccError, IccHeader, IccProfile, IccProgram,
    IccRenderingIntent, IccSignature, IccStage,
};

mod intent;

/// Exact profile matrix and independent curves selected for one direction. Colorants are
/// stored PCS-relative: applying `chad` to them again would double-adapt the profile.
#[derive(Clone, Debug, PartialEq)]
pub struct IccMatrixTrc {
    header: IccHeader,
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
    /// Inspect a selected matrix/TRC method. MPE and legacy LUT methods cannot be
    /// represented by this descriptor; use `select` for general profile execution.
    pub fn matrix_trc(
        &self,
        direction: IccDirection,
        intent: IccRenderingIntent,
    ) -> Result<IccMatrixTrc, IccError> {
        let selected = self.select(direction, intent)?;
        selected.matrix_trc.ok_or_else(|| IccError::TransformTag {
            tag: selected.tag.expect("non-matrix method"),
        })
    }

    fn parse_matrix_trc(&self, direction: IccDirection) -> Result<IccMatrixTrc, IccError> {
        let header = self.header();
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
            header: *header,
            curves,
            matrix,
            media_white,
            chromatic_adaptation,
        })
    }
}

/// One selected directional method, expressed in physical PCS XYZ at its connection end.
/// Original profile metadata stays available; unused TRC/LUT methods are never interpreted.
#[derive(Clone, Debug, PartialEq)]
pub struct IccProfileProgram {
    header: IccHeader,
    tag: Option<IccSignature>,
    program: IccProgram,
    matrix_trc: Option<IccMatrixTrc>,
    media_white: [i32; 3],
    device_channels: usize,
}

impl IccProfileProgram {
    #[must_use]
    pub const fn header(&self) -> &IccHeader {
        &self.header
    }
    #[must_use]
    pub const fn tag(&self) -> Option<IccSignature> {
        self.tag
    }
    #[must_use]
    pub const fn program(&self) -> &IccProgram {
        &self.program
    }
    #[must_use]
    pub const fn matrix_trc(&self) -> Option<&IccMatrixTrc> {
        self.matrix_trc.as_ref()
    }
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.device_channels
    }
}

impl IccProfile {
    /// Select the requested DToB/BToD method, then its legacy LUT, then the intent-zero
    /// legacy LUT, then matrix/TRC. Unknown MPE element types discard that method as
    /// required by ICC.1 §10.16.1. Broken supported elements remain an error.
    pub fn select(
        &self,
        direction: IccDirection,
        intent: IccRenderingIntent,
    ) -> Result<IccProfileProgram, IccError> {
        let header = *self.header();
        if !matches!(&header.class.0, b"scnr" | b"mntr" | b"prtr") {
            return Err(IccError::Unsupported {
                field: "profile class",
                signature: header.class,
            });
        }
        if !matches!(&header.pcs.0, b"XYZ " | b"Lab ") {
            return Err(IccError::Unsupported {
                field: "PCS",
                signature: header.pcs,
            });
        }
        let device_channels = device_channels(header.device_space)?;
        let (inputs, outputs, mut mpe, base) = match direction {
            IccDirection::DeviceToPcs => (device_channels, 3, *b"D2B0", *b"A2B0"),
            IccDirection::PcsToDevice => (3, device_channels, *b"B2D0", *b"B2A0"),
        };
        mpe[3] += intent as u8;
        let mpe = IccSignature(mpe);
        if self.tag(mpe).is_some() {
            if header.version < 0x0430_0000 {
                return Err(IccError::Invalid {
                    field: "MPE requires v4.3",
                    offset: 8,
                });
            }
            if let Some(program) = super::mpe::parse(self, mpe, inputs, outputs)? {
                let mut stages = program.stages().to_vec();
                if header.pcs == IccSignature(*b"Lab ") {
                    match direction {
                        IccDirection::DeviceToPcs => stages.push(IccStage::LabToXyz),
                        IccDirection::PcsToDevice => stages.insert(0, IccStage::XyzToLab),
                    }
                }
                let media_white = validated_white(self)?;
                return Ok(IccProfileProgram {
                    header,
                    tag: Some(mpe),
                    program: IccProgram::new(inputs, outputs, stages)?,
                    matrix_trc: None,
                    media_white,
                    device_channels,
                });
            }
        }
        let mut requested = base;
        requested[3] += if intent == IccRenderingIntent::Absolute {
            1
        } else {
            intent as u8
        };
        for tag in [IccSignature(requested), IccSignature(base)] {
            if self.tag(tag).is_some() {
                return Ok(IccProfileProgram {
                    header,
                    tag: Some(tag),
                    program: super::lut::parse(self, tag, direction, inputs, outputs)?,
                    matrix_trc: None,
                    media_white: validated_white(self)?,
                    device_channels,
                });
            }
        }
        let matrix_trc = self.parse_matrix_trc(direction)?;
        let curves = IccStage::Curves {
            curves: matrix_trc.curves.clone().into(),
            inverse: direction == IccDirection::PcsToDevice,
        };
        let stages = match direction {
            IccDirection::DeviceToPcs => vec![
                curves,
                IccStage::Matrix(IccAffine::new(
                    device_channels,
                    matrix_trc
                        .matrix
                        .iter()
                        .flat_map(|row| row[..device_channels].iter().copied())
                        .collect(),
                    vec![0.0; 3],
                    false,
                )?),
            ],
            IccDirection::PcsToDevice => {
                let matrix = if device_channels == 1 {
                    vec![0.0, 1.0, 0.0]
                } else {
                    inverse(matrix_trc.matrix)?.into_iter().flatten().collect()
                };
                vec![
                    IccStage::Matrix(IccAffine::new(
                        3,
                        matrix,
                        vec![0.0; device_channels],
                        false,
                    )?),
                    curves,
                ]
            }
        };
        Ok(IccProfileProgram {
            header,
            tag: None,
            program: IccProgram::new(inputs, outputs, stages)?,
            media_white: matrix_trc.media_white,
            matrix_trc: Some(matrix_trc),
            device_channels,
        })
    }
}

fn validated_white(profile: &IccProfile) -> Result<[i32; 3], IccError> {
    let white = xyz_tag(profile, *b"wtpt")?;
    if white.iter().any(|v| *v <= 0) {
        return Err(IccError::Invalid {
            field: "media white point",
            offset: u64::from(
                profile
                    .tag(IccSignature(*b"wtpt"))
                    .expect("read white tag")
                    .offset,
            ),
        });
    }
    Ok(white)
}

fn device_channels(signature: IccSignature) -> Result<usize, IccError> {
    Ok(match &signature.0 {
        b"GRAY" => 1,
        b"CMYK" => 4,
        b"RGB " | b"XYZ " | b"Lab " | b"Luv " | b"YCbr" | b"Yxy " | b"HSV " | b"HLS " | b"CMY " => {
            3
        }
        [n @ b'2'..=b'9', b'C', b'L', b'R'] => usize::from(n - b'0'),
        [n @ b'A'..=b'F', b'C', b'L', b'R'] => usize::from(n - b'A' + 10),
        _ => {
            return Err(IccError::Unsupported {
                field: "device space",
                signature,
            });
        }
    })
}

/// A selected directional profile or an unbounded linear RGB connection.
#[derive(Clone, Debug, PartialEq)]
pub enum IccTransformEndpoint {
    Profile(Box<IccProfileProgram>),
    LinearRgb(RgbColorSpace),
}

impl IccTransformEndpoint {
    #[must_use]
    pub fn channels(&self) -> usize {
        match self {
            Self::Profile(p) => p.channels(),
            Self::LinearRgb(_) => 3,
        }
    }
    #[must_use]
    pub fn profile(&self) -> Option<&IccProfileProgram> {
        match self {
            Self::Profile(p) => Some(p),
            Self::LinearRgb(_) => None,
        }
    }

    fn stages(&self, direction: IccDirection) -> Result<Vec<IccStage>, IccError> {
        match self {
            Self::Profile(p) => Ok(p.program.stages().to_vec()),
            Self::LinearRgb(space) => {
                let matrix = match direction {
                    IccDirection::DeviceToPcs => ColorMatrix::rgb_to_xyz(
                        *space,
                        Chromaticity::ICC_D50,
                        WhitePointAdaptation::Bradford,
                    )?,
                    IccDirection::PcsToDevice => ColorMatrix::xyz_to_rgb(
                        Chromaticity::ICC_D50,
                        *space,
                        WhitePointAdaptation::Bradford,
                    )?,
                };
                Ok(vec![IccStage::Matrix(IccAffine::new(
                    3,
                    matrix.rows().iter().flatten().copied().collect(),
                    vec![0.0; 3],
                    false,
                )?)])
            }
        }
    }
}

/// Ordered ICC program. Alpha and extra channels are outside this transform. Matrix/TRC
/// profiles use relative PCS, fully adapted absolute white, or v4 black compensation.
/// MPE stages preserve unbounded values except for their explicitly bounded CLUT inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct IccTransform {
    source: IccTransformEndpoint,
    target: IccTransformEndpoint,
    program: IccProgram,
}

impl IccTransform {
    pub fn new(
        source: &IccProfile,
        target: &IccProfile,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::Profile(source.select(IccDirection::DeviceToPcs, intent)?.into()),
            IccTransformEndpoint::Profile(target.select(IccDirection::PcsToDevice, intent)?.into()),
            intent,
        )
    }
    pub fn to_linear_rgb(
        source: &IccProfile,
        target: RgbColorSpace,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::Profile(source.select(IccDirection::DeviceToPcs, intent)?.into()),
            IccTransformEndpoint::LinearRgb(target),
            intent,
        )
    }
    pub fn from_linear_rgb(
        source: RgbColorSpace,
        target: &IccProfile,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        Self::connect(
            IccTransformEndpoint::LinearRgb(source),
            IccTransformEndpoint::Profile(target.select(IccDirection::PcsToDevice, intent)?.into()),
            intent,
        )
    }

    fn connect(
        source: IccTransformEndpoint,
        target: IccTransformEndpoint,
        intent: IccRenderingIntent,
    ) -> Result<Self, IccError> {
        let connection = intent::Connection::new(&source, &target, intent);
        let mut first = source.stages(IccDirection::DeviceToPcs)?;
        let mut last = target.stages(IccDirection::PcsToDevice)?;
        let same_geometry = source
            .profile()
            .and_then(IccProfileProgram::matrix_trc)
            .zip(target.profile().and_then(IccProfileProgram::matrix_trc))
            .is_some_and(|(s, t)| s.matrix == t.matrix && s.curves.len() == t.curves.len());
        let mut stages = Vec::new();
        if connection.is_identity() && same_geometry {
            // Exact cancellation retains the specified endpoint of sampled plateaus.
            first.pop();
            last.remove(0);
            stages.extend(first);
            stages.extend(last);
        } else {
            let connection = connection.into_stage()?;
            for stage in first.into_iter().chain([connection]).chain(last) {
                super::program::append(&mut stages, stage)?;
            }
        }
        let program = IccProgram::new(source.channels(), target.channels(), stages)?;
        Ok(Self {
            source,
            target,
            program,
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
    pub const fn program(&self) -> &IccProgram {
        &self.program
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
