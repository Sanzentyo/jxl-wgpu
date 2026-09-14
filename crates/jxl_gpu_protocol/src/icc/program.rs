//! Selected ICC processing order, independent of GPU storage and dispatch layout.
use std::sync::Arc;

use super::{IccCurve, IccError};

/// A checked sequence of processing elements. No image samples are stored or evaluated here.
#[derive(Clone, Debug, PartialEq)]
pub struct IccProgram {
    input_channels: usize,
    output_channels: usize,
    stages: Arc<[IccStage]>,
}

impl IccProgram {
    pub(super) fn new(
        input_channels: usize,
        output_channels: usize,
        stages: Vec<IccStage>,
    ) -> Result<Self, IccError> {
        channels(input_channels)?;
        channels(output_channels)?;
        let mut current = input_channels;
        for stage in &stages {
            channels(stage.input_channels())?;
            channels(stage.output_channels())?;
            if stage.input_channels() != current {
                return Err(IccError::Invalid {
                    field: "processing element channel continuity",
                    offset: 0,
                });
            }
            current = stage.output_channels();
        }
        if current != output_channels {
            return Err(IccError::Invalid {
                field: "processing program output channels",
                offset: 0,
            });
        }
        Ok(Self {
            input_channels,
            output_channels,
            stages: stages.into(),
        })
    }

    #[must_use]
    pub const fn input_channels(&self) -> usize {
        self.input_channels
    }

    #[must_use]
    pub const fn output_channels(&self) -> usize {
        self.output_channels
    }

    #[must_use]
    pub fn stages(&self) -> &[IccStage] {
        &self.stages
    }

    #[must_use]
    pub fn max_channels(&self) -> usize {
        self.stages
            .iter()
            .map(IccStage::output_channels)
            .fold(self.input_channels, usize::max)
    }
}

/// Clipping is part of an element's semantics, never an implicit connection between elements.
#[derive(Clone, Debug, PartialEq)]
pub enum IccStage {
    Curves {
        curves: Arc<[IccCurve]>,
        inverse: bool,
    },
    Matrix(IccAffine),
    Clut(IccClut),
    SegmentedCurves(Arc<[IccSegmentedCurve]>),
    /// CIE Lab in physical units (L*, a*, b*) to PCS XYZ, without unit-range clipping.
    LabToXyz,
    XyzToLab,
}

impl IccStage {
    #[must_use]
    pub fn input_channels(&self) -> usize {
        match self {
            Self::Curves { curves, .. } => curves.len(),
            Self::Matrix(matrix) => matrix.input_channels,
            Self::Clut(clut) => clut.grid.len(),
            Self::SegmentedCurves(curves) => curves.len(),
            Self::LabToXyz | Self::XyzToLab => 3,
        }
    }

    #[must_use]
    pub fn output_channels(&self) -> usize {
        match self {
            Self::Matrix(matrix) => matrix.offset.len(),
            Self::Clut(clut) => clut.output_channels,
            _ => self.input_channels(),
        }
    }
}

/// Row-major affine coefficients. Host connection matrices retain f64 precision until upload.
#[derive(Clone, Debug, PartialEq)]
pub struct IccAffine {
    pub(super) input_channels: usize,
    pub(super) matrix: Arc<[f64]>,
    pub(super) offset: Arc<[f64]>,
    pub(super) clamp_output: bool,
}

impl IccAffine {
    pub(super) fn new(
        input_channels: usize,
        matrix: Vec<f64>,
        offset: Vec<f64>,
        clamp_output: bool,
    ) -> Result<Self, IccError> {
        channels(input_channels)?;
        channels(offset.len())?;
        if matrix.len() != input_channels * offset.len()
            || matrix.iter().chain(&offset).any(|v| !v.is_finite())
        {
            return Err(IccError::Matrix);
        }
        Ok(Self {
            input_channels,
            matrix: matrix.into(),
            offset: offset.into(),
            clamp_output,
        })
    }

    pub(super) fn is_identity(&self) -> bool {
        !self.clamp_output
            && self.input_channels == self.offset.len()
            && self.offset.iter().all(|v| *v == 0.0)
            && self
                .matrix
                .iter()
                .enumerate()
                .all(|(i, v)| *v == f64::from(i / self.input_channels == i % self.input_channels))
    }

    /// Fuse only unclipped affine stages; no curve or CLUT boundary can be crossed.
    pub(super) fn after(&self, source: &Self) -> Result<Self, IccError> {
        let inputs = source.input_channels;
        let middle = self.input_channels;
        let outputs = self.offset.len();
        let matrix = (0..outputs * inputs)
            .map(|i| {
                let (r, c) = (i / inputs, i % inputs);
                (0..middle)
                    .map(|k| self.matrix[r * middle + k] * source.matrix[k * inputs + c])
                    .sum()
            })
            .collect();
        let offset = (0..outputs)
            .map(|r| {
                self.offset[r]
                    + (0..middle)
                        .map(|k| self.matrix[r * middle + k] * source.offset[k])
                        .sum::<f64>()
            })
            .collect();
        Self::new(inputs, matrix, offset, false)
    }
    #[must_use]
    pub const fn input_channels(&self) -> usize {
        self.input_channels
    }

    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    #[must_use]
    pub fn offset(&self) -> &[f64] {
        &self.offset
    }

    #[must_use]
    pub const fn clamp_output(&self) -> bool {
        self.clamp_output
    }
}

/// Last input dimension varies fastest; every grid point contains all output channels.
#[derive(Clone, Debug, PartialEq)]
pub struct IccClut {
    pub(super) grid: Arc<[u8]>,
    pub(super) output_channels: usize,
    pub(super) values: Arc<[f32]>,
}

impl IccClut {
    #[must_use]
    pub fn grid(&self) -> &[u8] {
        &self.grid
    }

    #[must_use]
    pub const fn output_channels(&self) -> usize {
        self.output_channels
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Float curves preserve first-segment ownership of shared breakpoints (ICC.1, 10.16.2.2).
#[derive(Clone, Debug, PartialEq)]
pub struct IccSegmentedCurve {
    pub(super) segments: Arc<[IccCurveSegment]>,
}

impl IccSegmentedCurve {
    #[must_use]
    pub fn segments(&self) -> &[IccCurveSegment] {
        &self.segments
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IccCurveSegment {
    /// Exclusive lower breakpoint; negative infinity for the first segment.
    pub lower: f32,
    /// Inclusive upper breakpoint; positive infinity for the last segment.
    pub upper: f32,
    pub kind: IccCurveSegmentKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum IccCurveSegmentKind {
    /// (a*x + b)^gamma + c.
    Power { gamma: f32, a: f32, b: f32, c: f32 },
    /// a*log10(b*x^gamma + c) + d.
    Logarithmic {
        gamma: f32,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    },
    /// a*b^(c*x + d) + e.
    Exponential {
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        e: f32,
    },
    /// Includes the implicit initial value derived from the preceding segment at `lower`.
    Samples(Arc<[f32]>),
}

pub(super) fn channels(count: usize) -> Result<(), IccError> {
    if count == 0 {
        return Err(IccError::Invalid {
            field: "zero processing channels",
            offset: 0,
        });
    }
    super::profile::limit("processing channels", count as u64, u64::from(u16::MAX))
}

pub(super) fn append(stages: &mut Vec<IccStage>, stage: IccStage) -> Result<(), IccError> {
    if matches!(&stage, IccStage::Matrix(matrix) if matrix.is_identity()) {
        return Ok(());
    }
    if let (Some(IccStage::Matrix(previous)), IccStage::Matrix(next)) = (stages.last_mut(), &stage)
        && !previous.clamp_output
        && !next.clamp_output
        && previous.offset.len() == next.input_channels
    {
        *previous = next.after(previous)?;
    } else {
        stages.push(stage);
    }
    Ok(())
}
