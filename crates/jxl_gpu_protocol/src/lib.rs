// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![deny(unsafe_code)]

//! Backend-neutral protocol for accelerating JPEG XL rendering.
//!
//! The decoder deliberately exposes no `wgpu` types here. A backend can
//! therefore live in a separate crate, share an application's existing device,
//! and be omitted entirely on builds that do not need GPU support.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

/// Stable identifier for a logical image plane in a [`RenderPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlaneId(pub u32);

/// Stable identifier for an output requested by the decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OutputId(pub u32);

/// Stable identifier for a decoded JPEG XL group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupId(pub u32);

/// Stable identifier for data that becomes available after plan construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u32);

/// A two-dimensional extent in pixels or samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extent2d {
    pub width: u32,
    pub height: u32,
}

impl Extent2d {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn area(self) -> Option<usize> {
        usize::try_from(self.width)
            .ok()?
            .checked_mul(usize::try_from(self.height).ok()?)
    }

    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// A signed rectangle. Origins may be negative while dependency halos are propagated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn expand(self, border: Border2d) -> Self {
        let left = i32::from(border.left);
        let top = i32::from(border.top);
        Self {
            x: self.x.saturating_sub(left),
            y: self.y.saturating_sub(top),
            width: self
                .width
                .saturating_add(u32::from(border.left) + u32::from(border.right)),
            height: self
                .height
                .saturating_add(u32::from(border.top) + u32::from(border.bottom)),
        }
    }

    pub fn clamp_to(self, extent: Extent2d) -> Self {
        let x0 = (self.x.max(0) as u32).min(extent.width);
        let y0 = (self.y.max(0) as u32).min(extent.height);
        let x1 = self
            .x
            .saturating_add_unsigned(self.width)
            .max(0)
            .min(i32::try_from(extent.width).unwrap_or(i32::MAX)) as u32;
        let y1 = self
            .y
            .saturating_add_unsigned(self.height)
            .max(0)
            .min(i32::try_from(extent.height).unwrap_or(i32::MAX)) as u32;
        Self::new(
            x0 as i32,
            y0 as i32,
            x1.saturating_sub(x0),
            y1.saturating_sub(y0),
        )
    }
}

/// Input dependency around an output region.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Border2d {
    pub left: u16,
    pub right: u16,
    pub top: u16,
    pub bottom: u16,
}

impl Border2d {
    pub const ZERO: Self = Self::symmetric(0, 0);

    pub const fn symmetric(x: u16, y: u16) -> Self {
        Self {
            left: x,
            right: x,
            top: y,
            bottom: y,
        }
    }
}

/// Integer output samples produced for one integer input sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale2d {
    pub x: u8,
    pub y: u8,
}

impl Scale2d {
    pub const IDENTITY: Self = Self { x: 1, y: 1 };

    pub const fn new(x: u8, y: u8) -> Self {
        Self { x, y }
    }
}

/// Scalar representation stored in a logical plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SampleType {
    I32,
    F32,
    F16,
    U16,
    U8,
}

impl SampleType {
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::F16 | Self::U16 => 2,
            Self::U8 => 1,
        }
    }
}

/// Intended lifetime and ownership of a logical plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneRole {
    Source,
    Intermediate,
    Parameter,
    Output,
    /// Pre-allocated GPU-resident plane supplied directly by the caller.
    /// Bypasses arena slot allocation and host upload copying.
    ImportedResident,
}

/// Description of a single planar image allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneDesc {
    pub id: PlaneId,
    pub extent: Extent2d,
    /// Row stride in samples. A zero stride asks the backend to choose an aligned stride.
    pub stride: u32,
    pub sample_type: SampleType,
    pub role: PlaneRole,
}

impl PlaneDesc {
    pub fn minimum_len(&self) -> Option<usize> {
        let stride = if self.stride == 0 {
            self.extent.width
        } else {
            self.stride
        };
        usize::try_from(stride)
            .ok()?
            .checked_mul(usize::try_from(self.extent.height).ok()?)
    }
}

/// Coarse operation kinds used for capability negotiation and kernel planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderOpKind {
    Copy,
    ModularToF32,
    ChromaUpsample,
    Gaborish,
    Epf,
    Upsample,
    AdaptiveLf,
    VarDct,
    AddNoise,
    XybToRgb,
    YcbcrToRgb,
    TransferFunction,
    Blend,
    PremultiplyAlpha,
    Convert,
    Extend,
    Save,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaAxis {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpfPass {
    Pass0,
    Pass1,
    Pass2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferFunction {
    Linear,
    Srgb,
    Bt709,
    Pq,
    Hlg,
    Gamma,
}

/// RGB primary chromaticities attached to an output signal.
///
/// The portable backend converts BT.709, BT.2020, and D65 Display-P3 primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RgbPrimaries {
    Bt709,
    Bt2020,
    DisplayP3,
    Undefined,
}

/// Primaries and transfer function of three floating-point RGB channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RgbColorEncoding {
    pub primaries: RgbPrimaries,
    pub transfer: TransferFunction,
}

impl RgbColorEncoding {
    pub const LINEAR_BT709: Self = Self {
        primaries: RgbPrimaries::Bt709,
        transfer: TransferFunction::Linear,
    };
    pub const SRGB_BT709: Self = Self {
        primaries: RgbPrimaries::Bt709,
        transfer: TransferFunction::Srgb,
    };
    pub const BT709: Self = Self {
        primaries: RgbPrimaries::Bt709,
        transfer: TransferFunction::Bt709,
    };
}

/// Color interpretation of a render-plan output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OutputColorEncoding {
    /// The output is not a three-channel RGB color signal.
    NonColor,
    Rgb(RgbColorEncoding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Preserve the base sample.
    Keep,
    /// Replace the base sample with the source sample.
    Replace,
    /// Add the source sample to the base sample.
    Add,
    /// Multiply the base sample by the optionally clamped source sample.
    Multiply,
    /// Composite the source above the base.
    BlendAbove,
    /// Composite the source below the base.
    BlendBelow,
    /// Add the source weighted by its alpha to the base.
    AlphaWeightedAddAbove,
    /// Place the source below the base, weighting the base by its alpha.
    AlphaWeightedAddBelow,
}

/// Interpretation of one scalar [`RenderOp::Blend`] result.
///
/// A scalar contract keeps the portable shader below the minimum WebGPU storage-buffer binding
/// limit. Frontends emit one node per color or extra channel and may fuse or pack those nodes on a
/// backend with stronger binding-array capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendComponent {
    /// A color or non-alpha extra-channel sample.
    ///
    /// With alpha, node inputs are `[base, source, base_alpha, source_alpha]`. Without alpha,
    /// inputs are `[base, source]`; blend modes then use JPEG XL's no-alpha fallbacks.
    Color {
        /// Whether both color inputs use associated (premultiplied) alpha.
        alpha_associated: bool,
    },
    /// The alpha channel itself. Inputs are `[base_alpha, source_alpha]`.
    Alpha,
}

#[derive(Clone, Debug)]
pub struct GaborishParams {
    pub channel: u16,
    pub weight0: f32,
    pub weight1: f32,
    pub weight2: f32,
}

#[derive(Clone, Debug)]
pub struct EpfParams {
    pub pass: EpfPass,
    pub sigma_scale: f32,
    pub border_sad_mul: f32,
    pub channel_scale: [f32; 3],
    /// Late-bound sigma data. The resource contains either a single constant or the plane named
    /// by [`sigma_plane`](Self::sigma_plane).
    pub sigma_resource: Option<ResourceId>,
    /// Optional plane containing one sigma value per 8x8 block.
    pub sigma_plane: Option<PlaneId>,
}

/// Shape of a plane supplied by a late-bound rendering resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourcePlaneDesc {
    pub extent: Extent2d,
    /// Row stride in samples. A zero stride requests a tightly packed plane.
    pub stride: u32,
    pub sample_type: SampleType,
}

/// A late-bound resource requested by a semantic render stage.
///
/// The provider is retained by the CPU pipeline and is only snapshotted at the coordinator
/// boundary, after parallel entropy-decoding workers have completed.
#[derive(Clone)]
pub struct RenderResourceSpec {
    pub provider: Arc<dyn RenderResourceProvider>,
    pub plane: Option<ResourcePlaneDesc>,
}

impl fmt::Debug for RenderResourceSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RenderResourceSpec")
            .field("plane", &self.plane)
            .finish_non_exhaustive()
    }
}

/// Supplies data that is decoded after [`RenderPlan`] construction.
pub trait RenderResourceProvider: Send + Sync {
    /// Takes an owned snapshot for one frame submission. `plane` is present when the resource is
    /// addressable as a parameter plane in the render plan.
    fn snapshot(&self, plane: Option<&PlaneDesc>) -> Result<ResourceData, BackendError>;
}

/// Non-identity 2×, 4×, or 8× upsampling factor for image and channel interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum UpsamplingFactor {
    X2 = 2,
    X4 = 4,
    X8 = 8,
}

impl UpsamplingFactor {
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u8 as u32
    }
}

/// Error returned when an upsampling factor is not 2, 4, or 8.
#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
#[error("unsupported upsampling factor {factor}; expected 2, 4, or 8")]
pub struct UnsupportedUpsamplingFactor {
    pub factor: u32,
}

impl TryFrom<u8> for UpsamplingFactor {
    type Error = UnsupportedUpsamplingFactor;

    fn try_from(factor: u8) -> Result<Self, Self::Error> {
        match factor {
            2 => Ok(Self::X2),
            4 => Ok(Self::X4),
            8 => Ok(Self::X8),
            _ => Err(UnsupportedUpsamplingFactor {
                factor: u32::from(factor),
            }),
        }
    }
}

impl TryFrom<u32> for UpsamplingFactor {
    type Error = UnsupportedUpsamplingFactor;

    fn try_from(factor: u32) -> Result<Self, Self::Error> {
        match factor {
            2 => Ok(Self::X2),
            4 => Ok(Self::X4),
            8 => Ok(Self::X8),
            _ => Err(UnsupportedUpsamplingFactor { factor }),
        }
    }
}

impl From<UpsamplingFactor> for u8 {
    fn from(factor: UpsamplingFactor) -> Self {
        factor as Self
    }
}

impl From<UpsamplingFactor> for u32 {
    fn from(factor: UpsamplingFactor) -> Self {
        factor as u32
    }
}

/// Storage buffer allocation requirement for a retained or shared resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoragePlan {
    pub bytes: u64,
}

impl StoragePlan {
    #[must_use]
    pub const fn new(bytes: u64) -> Self {
        Self { bytes }
    }
}

impl UpsamplingFactor {
    /// Storage buffer plan for the prepared weights of this upsampling factor.
    #[must_use]
    pub const fn weights_storage_plan(self) -> StoragePlan {
        let factor = self.as_u32() as u64;
        let floats = factor * factor * 25;
        StoragePlan::new(floats * std::mem::size_of::<f32>() as u64)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UpsampleParams {
    pub factor: UpsamplingFactor,
    pub weights: ResourceId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveLfParams {
    pub thresholds: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct XybParams {
    pub opsin_bias: [f32; 3],
    pub inverse_opsin_matrix: [[f32; 3]; 3],
    pub intensity_target: f32,
}

impl Default for XybParams {
    fn default() -> Self {
        Self {
            opsin_bias: [-0.003_793_073_4; 3],
            inverse_opsin_matrix: [
                [11.031_567, -9.866_944, -0.164_622_99],
                [-3.254_147_3, 4.418_770_3, -0.164_622_99],
                [-3.658_851_4, 2.712_923, 1.945_928_2],
            ],
            intensity_target: 255.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransferParams {
    pub function: TransferFunction,
    /// Encoding exponent for [`TransferFunction::Gamma`]. JPEG XL stores the inverse display
    /// gamma (for example, `1 / 2.2`), not the display gamma itself.
    pub gamma: f32,
    pub intensity_target: f32,
    pub min_nits: f32,
    /// Linear-light RGB luminance coefficients used by the HLG inverse OOTF. Frontends must
    /// derive these from the encoded primaries; keeping them explicit prevents BT.709 constants
    /// from being applied to BT.2020 or custom primaries.
    pub luminance_rgb: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct BlendParams {
    pub mode: BlendMode,
    pub component: BlendComponent,
    /// Apply the JPEG XL blend-mode clamp at the point required by the specification.
    pub clamp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputLayout {
    Planar,
    Interleaved,
}

/// Mapping from codestream coordinates to display-oriented output coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum OutputOrientation {
    #[default]
    Identity,
    FlipHorizontal,
    Rotate180,
    FlipVertical,
    Transpose,
    Rotate90Cw,
    AntiTranspose,
    Rotate90Ccw,
}

impl OutputOrientation {
    pub const fn is_transposing(self) -> bool {
        matches!(
            self,
            Self::Transpose | Self::Rotate90Cw | Self::AntiTranspose | Self::Rotate90Ccw
        )
    }

    pub const fn map_extent(self, extent: Extent2d) -> Extent2d {
        if self.is_transposing() {
            Extent2d::new(extent.height, extent.width)
        } else {
            extent
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaveParams {
    pub output: OutputId,
    pub sample_type: SampleType,
    pub channels: Vec<PlaneId>,
    pub layout: OutputLayout,
    pub orientation: OutputOrientation,
}

/// Backend-neutral operation with all immutable parameters needed for replay.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum RenderOp {
    Copy,
    ModularToF32 {
        multiplier: f32,
        bias: f32,
    },
    ChromaUpsample {
        axis: ChromaAxis,
    },
    Gaborish(GaborishParams),
    Epf(EpfParams),
    Upsample(UpsampleParams),
    AdaptiveLf(AdaptiveLfParams),
    /// Dequantize and inverse-transform the strategy buckets in [`VarDctPacket`].
    ///
    /// Transform coverage is carried by each packet bucket rather than summarized as a maximum
    /// edge. This prevents capability negotiation from silently treating rectangular and special
    /// JPEG XL transforms as equivalent to a square DCT with the same largest dimension.
    VarDct,
    AddNoise {
        seed0: u32,
        seed1: u32,
    },
    XybToRgb(XybParams),
    YcbcrToRgb,
    TransferFunction(TransferParams),
    Blend(BlendParams),
    PremultiplyAlpha {
        alpha_plane: PlaneId,
    },
    Convert {
        output_type: SampleType,
    },
    /// Extend or crop a frame plane to the full image canvas.
    ///
    /// Inputs are `[frame]` to fill uncovered pixels with zero, or `[frame, reference]` to fill
    /// them from an equal-sized reference canvas. The single output has `image_extent`; `origin`
    /// places the frame's `(0, 0)` sample in canvas coordinates and may be negative.
    Extend {
        image_extent: Extent2d,
        origin: (i32, i32),
    },
    Save(SaveParams),
}

impl RenderOp {
    pub const fn kind(&self) -> RenderOpKind {
        match self {
            Self::Copy => RenderOpKind::Copy,
            Self::ModularToF32 { .. } => RenderOpKind::ModularToF32,
            Self::ChromaUpsample { .. } => RenderOpKind::ChromaUpsample,
            Self::Gaborish(_) => RenderOpKind::Gaborish,
            Self::Epf(_) => RenderOpKind::Epf,
            Self::Upsample(_) => RenderOpKind::Upsample,
            Self::AdaptiveLf(_) => RenderOpKind::AdaptiveLf,
            Self::VarDct => RenderOpKind::VarDct,
            Self::AddNoise { .. } => RenderOpKind::AddNoise,
            Self::XybToRgb(_) => RenderOpKind::XybToRgb,
            Self::YcbcrToRgb => RenderOpKind::YcbcrToRgb,
            Self::TransferFunction(_) => RenderOpKind::TransferFunction,
            Self::Blend(_) => RenderOpKind::Blend,
            Self::PremultiplyAlpha { .. } => RenderOpKind::PremultiplyAlpha,
            Self::Convert { .. } => RenderOpKind::Convert,
            Self::Extend { .. } => RenderOpKind::Extend,
            Self::Save(_) => RenderOpKind::Save,
        }
    }
}

/// A node in topological order.
#[derive(Clone, Debug)]
pub struct RenderNode {
    pub name: Arc<str>,
    pub op: RenderOp,
    pub inputs: Vec<PlaneId>,
    pub outputs: Vec<PlaneId>,
    /// Late-bound parameter resources consumed by this node.
    pub resources: Vec<ResourceId>,
    pub scale: Scale2d,
    pub border: Border2d,
    pub precision: PrecisionContract,
}

/// Accuracy gate for an operation. Backends may be stricter, but never looser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrecisionContract {
    Exact,
    Float {
        absolute: f32,
        relative: f32,
        rmse: f32,
    },
    Perceptual {
        max_lsb: u16,
        min_psnr: f32,
    },
}

impl Default for PrecisionContract {
    fn default() -> Self {
        Self::Float {
            absolute: 2.0e-5,
            relative: 2.0e-5,
            rmse: 2.0e-6,
        }
    }
}

/// An output contract, independent of the concrete API buffer or GPU resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputDesc {
    pub id: OutputId,
    pub extent: Extent2d,
    pub sample_type: SampleType,
    pub channels: u8,
    pub layout: OutputLayout,
    /// Encoding of the values presented to a generic color-output conversion.
    pub color_encoding: OutputColorEncoding,
}

/// A validated, backend-neutral rendering graph.
#[derive(Clone, Debug, Default)]
pub struct RenderPlan {
    pub planes: Vec<PlaneDesc>,
    pub nodes: Vec<RenderNode>,
    pub outputs: Vec<OutputDesc>,
}

impl RenderPlan {
    pub fn validate(&self) -> Result<(), PlanError> {
        let mut planes = BTreeMap::new();
        for plane in &self.planes {
            if plane.extent.is_empty() {
                return Err(PlanError::EmptyPlane(plane.id));
            }
            if plane.stride != 0 && plane.stride < plane.extent.width {
                return Err(PlanError::ShortStride(plane.id));
            }
            if planes.insert(plane.id, plane).is_some() {
                return Err(PlanError::DuplicatePlane(plane.id));
            }
        }

        let mut available: BTreeSet<_> = self
            .planes
            .iter()
            .filter(|plane| {
                matches!(
                    plane.role,
                    PlaneRole::Source | PlaneRole::Parameter | PlaneRole::ImportedResident
                )
            })
            .map(|plane| plane.id)
            .collect();
        let mut written = BTreeSet::new();
        for (node_index, node) in self.nodes.iter().enumerate() {
            if node.scale.x == 0 || node.scale.y == 0 {
                return Err(PlanError::ZeroScale(node_index));
            }
            if let RenderOp::PremultiplyAlpha { alpha_plane } = node.op
                && node
                    .inputs
                    .iter()
                    .filter(|&&input| input == alpha_plane)
                    .count()
                    != 1
            {
                return Err(PlanError::OperationInput {
                    node: node_index,
                    operation: RenderOpKind::PremultiplyAlpha,
                    plane: alpha_plane,
                });
            }
            for input in &node.inputs {
                if !planes.contains_key(input) {
                    return Err(PlanError::UnknownPlane(*input));
                }
                if !available.contains(input) {
                    return Err(PlanError::ReadBeforeWrite {
                        node: node_index,
                        plane: *input,
                    });
                }
            }
            for output in &node.outputs {
                let output_desc = planes.get(output).ok_or(PlanError::UnknownPlane(*output))?;
                if matches!(
                    output_desc.role,
                    PlaneRole::Source | PlaneRole::Parameter | PlaneRole::ImportedResident
                ) {
                    return Err(PlanError::ReadOnlyPlaneWritten {
                        node: node_index,
                        plane: *output,
                    });
                }
                if !written.insert(*output) {
                    return Err(PlanError::MultipleWriters(*output));
                }
                available.insert(*output);
            }
        }

        let output_ids: BTreeSet<_> = self.outputs.iter().map(|output| output.id).collect();
        if output_ids.len() != self.outputs.len() {
            return Err(PlanError::DuplicateOutput);
        }
        for node in &self.nodes {
            if let RenderOp::Save(save) = &node.op
                && !output_ids.contains(&save.output)
            {
                return Err(PlanError::UnknownOutput(save.output));
            }
        }
        Ok(())
    }

    /// First and last node indices that access each plane. Useful for arena aliasing.
    pub fn lifetimes(&self) -> BTreeMap<PlaneId, PlaneLifetime> {
        let mut lifetimes = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for plane in node.inputs.iter().chain(&node.outputs) {
                lifetimes
                    .entry(*plane)
                    .and_modify(|lifetime: &mut PlaneLifetime| lifetime.last = index)
                    .or_insert(PlaneLifetime {
                        first: index,
                        last: index,
                    });
            }
        }
        lifetimes
    }

    /// Propagates a changed source rectangle through scales and dependency halos.
    pub fn propagate_dirty(&self, source: PlaneId, region: Region) -> BTreeMap<PlaneId, Region> {
        let extents: BTreeMap<_, _> = self
            .planes
            .iter()
            .map(|plane| (plane.id, plane.extent))
            .collect();
        let mut dirty = BTreeMap::from([(source, region)]);
        for node in &self.nodes {
            let Some(input_region) = node
                .inputs
                .iter()
                .filter_map(|input| dirty.get(input).copied())
                .reduce(union_region)
            else {
                continue;
            };
            let expanded = input_region.expand(node.border);
            let scaled = Region::new(
                expanded.x.saturating_mul(i32::from(node.scale.x)),
                expanded.y.saturating_mul(i32::from(node.scale.y)),
                expanded.width.saturating_mul(u32::from(node.scale.x)),
                expanded.height.saturating_mul(u32::from(node.scale.y)),
            );
            for output in &node.outputs {
                let value = extents
                    .get(output)
                    .map_or(scaled, |extent| scaled.clamp_to(*extent));
                dirty
                    .entry(*output)
                    .and_modify(|current| *current = union_region(*current, value))
                    .or_insert(value);
            }
        }
        dirty
    }

    /// Appends an upsampling node to the plan with the canonical 2-sample symmetric border
    /// and scale matching the factor.
    pub fn add_upsample_node(
        &mut self,
        name: impl Into<std::sync::Arc<str>>,
        factor: UpsamplingFactor,
        weights: ResourceId,
        input: PlaneId,
        output: PlaneId,
    ) {
        let f = factor.as_u8();
        let node = RenderNode {
            name: name.into(),
            op: RenderOp::Upsample(UpsampleParams { factor, weights }),
            inputs: vec![input],
            outputs: vec![output],
            resources: vec![weights],
            scale: Scale2d::new(f, f),
            border: Border2d::symmetric(2, 2),
            precision: PrecisionContract::default(),
        };
        self.nodes.push(node);
    }
}

fn union_region(a: Region, b: Region) -> Region {
    let x0 = a.x.min(b.x);
    let y0 = a.y.min(b.y);
    let x1 =
        a.x.saturating_add_unsigned(a.width)
            .max(b.x.saturating_add_unsigned(b.width));
    let y1 =
        a.y.saturating_add_unsigned(a.height)
            .max(b.y.saturating_add_unsigned(b.height));
    Region::new(
        x0,
        y0,
        x1.saturating_sub(x0) as u32,
        y1.saturating_sub(y0) as u32,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLifetime {
    pub first: usize,
    pub last: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("plane {0:?} has an empty extent")]
    EmptyPlane(PlaneId),
    #[error("plane {0:?} has a stride shorter than its width")]
    ShortStride(PlaneId),
    #[error("plane {0:?} is declared more than once")]
    DuplicatePlane(PlaneId),
    #[error("render plan refers to unknown plane {0:?}")]
    UnknownPlane(PlaneId),
    #[error("node {node} reads plane {plane:?} before it is produced")]
    ReadBeforeWrite { node: usize, plane: PlaneId },
    #[error("node {node} writes read-only source or parameter plane {plane:?}")]
    ReadOnlyPlaneWritten { node: usize, plane: PlaneId },
    #[error("plane {0:?} has multiple writers")]
    MultipleWriters(PlaneId),
    #[error("node {0} has a zero scale")]
    ZeroScale(usize),
    #[error("node {node} {operation:?} must declare plane {plane:?} exactly once as an input")]
    OperationInput {
        node: usize,
        operation: RenderOpKind,
        plane: PlaneId,
    },
    #[error("render plan declares the same output more than once")]
    DuplicateOutput,
    #[error("save node refers to unknown output {0:?}")]
    UnknownOutput(OutputId),
    #[error(
        "node {node} declares {actual} late-bound resources, but its operation requires {expected}"
    )]
    InvalidOperationResourceCount {
        node: usize,
        expected: usize,
        actual: usize,
    },
    #[error("node {node} requests an incompatible plane shape for resource {resource:?}")]
    IncompatibleResourcePlane { node: usize, resource: ResourceId },
}

/// Precision contract negotiated before creating a frame session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrecisionPolicy {
    MatchDecoder,
    #[default]
    F32Only,
    AllowF16Storage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryMode {
    #[default]
    Auto,
    Resident,
    Streaming,
}

#[derive(Clone, Debug)]
pub struct FrameSessionDesc {
    pub frame_extent: Extent2d,
    pub group_extent: Extent2d,
    pub group_count: u32,
    pub precision: PrecisionPolicy,
    pub memory_mode: MemoryMode,
    pub max_resident_bytes: u64,
    pub max_scratch_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    pub name: String,
    pub supported_ops: BTreeSet<RenderOpKind>,
    pub max_buffer_bytes: u64,
    pub max_workgroup_storage_bytes: u32,
    pub max_invocations_per_workgroup: u32,
    pub supports_timestamps: bool,
    pub supports_f16: bool,
}

impl BackendCapabilities {
    pub fn supports_plan(&self, plan: &RenderPlan) -> bool {
        plan.nodes
            .iter()
            .all(|node| self.supported_ops.contains(&node.op.kind()))
    }
}

#[derive(Clone, Debug)]
pub enum PlaneData {
    I32(Vec<i32>),
    F32(Vec<f32>),
    F16(Vec<u16>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

impl PlaneData {
    pub const fn sample_type(&self) -> SampleType {
        match self {
            Self::I32(_) => SampleType::I32,
            Self::F32(_) => SampleType::F32,
            Self::F16(_) => SampleType::F16,
            Self::U16(_) => SampleType::U16,
            Self::U8(_) => SampleType::U8,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::I32(data) => data.len(),
            Self::F32(data) => data.len(),
            Self::F16(data) | Self::U16(data) => data.len(),
            Self::U8(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
pub struct HostPlane {
    pub id: PlaneId,
    pub extent: Extent2d,
    pub stride: u32,
    pub origin: (i32, i32),
    pub data: PlaneData,
}

impl HostPlane {
    pub fn validate(&self) -> Result<(), BackendError> {
        let stride = if self.stride == 0 {
            self.extent.width
        } else {
            self.stride
        };
        if stride < self.extent.width {
            return Err(BackendError::InvalidPayload(format!(
                "plane {:?} stride {stride} is shorter than width {}",
                self.id, self.extent.width
            )));
        }
        let required = usize::try_from(stride)
            .ok()
            .and_then(|stride| stride.checked_mul(self.extent.height as usize))
            .ok_or_else(|| BackendError::InvalidPayload("plane size overflow".into()))?;
        if self.data.len() < required {
            return Err(BackendError::InvalidPayload(format!(
                "plane {:?} has {} samples, expected at least {required}",
                self.id,
                self.data.len()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct GroupPayload {
    pub group: GroupId,
    /// Monotonic progressive revision for this group.
    pub revision: u32,
    pub complete: bool,
    pub planes: Vec<HostPlane>,
    /// Present when VarDCT processing is transferred before dequantization and IDCT.
    pub vardct: Option<VarDctPacket>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransformKind {
    Dct8,
    Hornuss,
    Dct2x2,
    Dct4x4,
    Dct16x16,
    Dct32x32,
    Dct16x8,
    Dct8x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Dct4x8,
    Dct8x4,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Dct64x64,
    Dct64x32,
    Dct32x64,
    Dct128x128,
    Dct128x64,
    Dct64x128,
    Dct256x256,
    Dct256x128,
    Dct128x256,
}

impl TransformKind {
    /// Every transform strategy defined by the JPEG XL codestream, in codestream order.
    pub const ALL: [Self; 27] = [
        Self::Dct8,
        Self::Hornuss,
        Self::Dct2x2,
        Self::Dct4x4,
        Self::Dct16x16,
        Self::Dct32x32,
        Self::Dct16x8,
        Self::Dct8x16,
        Self::Dct32x8,
        Self::Dct8x32,
        Self::Dct32x16,
        Self::Dct16x32,
        Self::Dct4x8,
        Self::Dct8x4,
        Self::Afv0,
        Self::Afv1,
        Self::Afv2,
        Self::Afv3,
        Self::Dct64x64,
        Self::Dct64x32,
        Self::Dct32x64,
        Self::Dct128x128,
        Self::Dct128x64,
        Self::Dct64x128,
        Self::Dct256x256,
        Self::Dct256x128,
        Self::Dct128x256,
    ];

    /// Spatial output extent `(width, height)` for one transform task.
    ///
    /// JPEG XL names rectangular transforms as rows-by-columns, so `Dct16x8` covers an 8-pixel
    /// wide by 16-pixel high rectangle.
    pub const fn pixel_extent(self) -> Extent2d {
        match self {
            Self::Dct8
            | Self::Hornuss
            | Self::Dct2x2
            | Self::Dct4x4
            | Self::Dct4x8
            | Self::Dct8x4
            | Self::Afv0
            | Self::Afv1
            | Self::Afv2
            | Self::Afv3 => Extent2d::new(8, 8),
            Self::Dct16x16 => Extent2d::new(16, 16),
            Self::Dct32x32 => Extent2d::new(32, 32),
            Self::Dct16x8 => Extent2d::new(8, 16),
            Self::Dct8x16 => Extent2d::new(16, 8),
            Self::Dct32x8 => Extent2d::new(8, 32),
            Self::Dct8x32 => Extent2d::new(32, 8),
            Self::Dct32x16 => Extent2d::new(16, 32),
            Self::Dct16x32 => Extent2d::new(32, 16),
            Self::Dct64x64 => Extent2d::new(64, 64),
            Self::Dct64x32 => Extent2d::new(32, 64),
            Self::Dct32x64 => Extent2d::new(64, 32),
            Self::Dct128x128 => Extent2d::new(128, 128),
            Self::Dct128x64 => Extent2d::new(64, 128),
            Self::Dct64x128 => Extent2d::new(128, 64),
            Self::Dct256x256 => Extent2d::new(256, 256),
            Self::Dct256x128 => Extent2d::new(128, 256),
            Self::Dct128x256 => Extent2d::new(256, 128),
        }
    }

    /// Low-frequency sample rectangle consumed before the inverse transform.
    pub const fn lf_extent(self) -> Extent2d {
        let pixels = self.pixel_extent();
        Extent2d::new(pixels.width / 8, pixels.height / 8)
    }

    /// Whether this strategy uses JPEG XL's special 8x8 inverse rather than a regular 2D DCT.
    pub const fn is_special(self) -> bool {
        matches!(
            self,
            Self::Hornuss
                | Self::Dct2x2
                | Self::Dct4x4
                | Self::Dct4x8
                | Self::Dct8x4
                | Self::Afv0
                | Self::Afv1
                | Self::Afv2
                | Self::Afv3
        )
    }

    /// Whether coefficient coordinates and the canonical dequantization matrix are transposed for
    /// this wire strategy.
    pub const fn needs_transpose(self) -> bool {
        !self.is_special() && self.pixel_extent().height >= self.pixel_extent().width
    }
}

#[derive(Clone, Debug)]
pub enum PackedCoefficients {
    DenseI32(Vec<i32>),
    /// Two signed 16-bit values in each word plus explicit out-of-range replacements.
    PackedI16 {
        words: Vec<u32>,
        overflow: Vec<CoefficientOverflow>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientOverflow {
    pub index: u32,
    pub value: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformTask {
    /// Scalar offset into [`VarDctPacket::coefficients`]. Data is channel-major X/Y/B, followed
    /// by `TransformKind::pixel_extent().area()` coefficients per channel in JPEG XL's linear
    /// transform-buffer order. This is the order consumed by the selected strategy, not a generic
    /// row-major frequency matrix.
    pub coefficient_offset: u32,
    /// Per-channel destination origins. `None` skips a chroma-subsampled channel for this task.
    /// The transform extent itself is unchanged; frontends supply downsampled coordinates.
    pub destinations: [Option<(u32, u32)>; 3],
    pub quant_index: u16,
    pub dequant_matrix_index: u16,
    /// Top-left coordinate of this transform in the global HF coefficient grid.  CfL parameters
    /// are selected per frequency from [`VarDctResource::hf_correlation`] rather than once per
    /// varblock, so this origin is intentionally independent of the spatial destinations.
    pub coefficient_origin: (u32, u32),
    /// First tightly packed X/Y/B LF tuple in [`VarDctResource::lf_coefficients`].
    pub lf_offset: u32,
}

#[derive(Clone, Debug)]
pub struct TransformBucket {
    pub transform: TransformKind,
    pub tasks: Vec<TransformTask>,
}

/// Owned group packet that can outlive entropy-decoder worker callbacks.
#[derive(Clone, Debug)]
pub struct VarDctPacket {
    pub revision: u32,
    pub last_pass: u16,
    pub coefficients: PackedCoefficients,
    pub buckets: Vec<TransformBucket>,
}

/// Per-frequency dequantization multipliers for one transform strategy.
///
/// `scales` contains exactly `transform.pixel_extent().area()` entries in the same JPEG XL
/// transform-buffer order as the task coefficients. Each entry stores the X/Y/B channel
/// multipliers applied after quantization-bias adjustment.
#[derive(Clone, Debug)]
pub struct VarDctDequantMatrix {
    pub transform: TransformKind,
    pub scales: Vec<[f32; 3]>,
}

/// Global 64x64-cell HF chroma-from-luma grid.
///
/// `values` is row-major and contains exactly `extent.width * extent.height` `[Y→X, Y→B]`
/// multipliers.  A transform frequency at global coefficient coordinate `(x, y)` selects cell
/// `(x / 64, y / 64)`.  This per-frequency lookup is required for large transforms that span
/// several correlation cells.
#[derive(Clone, Debug)]
pub struct VarDctCorrelationGrid {
    pub extent: Extent2d,
    pub values: Vec<[f32; 2]>,
}

/// Typed late-bound parameters for GPU VarDCT rendering.
///
/// A [`TransformTask`] selects one `quant_scales` entry with `quant_index` and one
/// `dequant_matrices` entry with `dequant_matrix_index`. Its `coefficient_origin` addresses the
/// global HF correlation grid, while `lf_offset` selects the separately decoded LF rectangle. HF
/// coefficients are bias-adjusted using `quant_biases`, multiplied by both selected dequantization
/// factors, then Y is correlated into X/B through the global 64x64-cell grid before LF
/// reinterpretation and inverse transform. Producers can therefore represent global scale, raw
/// quantization, channel multipliers, quantization matrices, color correlation, and the LF image
/// without an untyped positional `Vec<f32>` contract.
#[derive(Clone, Debug)]
pub struct VarDctResource {
    /// Biases for X/Y/B small coefficients followed by the large-coefficient numerator.
    pub quant_biases: [f32; 4],
    /// Per-quantization-index X/Y/B multipliers.
    pub quant_scales: Vec<[f32; 3]>,
    pub dequant_matrices: Vec<VarDctDequantMatrix>,
    pub hf_correlation: VarDctCorrelationGrid,
    /// Tightly packed LF tuples in X/Y/B order. Regular large transforms reinterpret an `N/8` by
    /// `M/8` rectangle into their lowest frequencies before inverse DCT; special 8x8 transforms
    /// consume one tuple.
    pub lf_coefficients: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub enum ResourceData {
    Plane(HostPlane),
    F32(Vec<f32>),
    I32(Vec<i32>),
    Bytes(Vec<u8>),
    VarDct(VarDctResource),
}

#[derive(Clone, Debug)]
pub struct ResourceUpdate {
    pub id: ResourceId,
    pub revision: u32,
    pub data: ResourceData,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RenderIntent {
    Progressive,
    #[default]
    Final,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmissionToken(pub u64);

#[derive(Clone, Debug)]
pub struct RenderedOutput {
    pub id: OutputId,
    pub extent: Extent2d,
    pub data: PlaneData,
}

#[derive(Clone, Debug, Default)]
pub struct ChangedRegions {
    pub outputs: BTreeMap<OutputId, Vec<Region>>,
}

#[derive(Clone, Debug)]
pub struct RenderedFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<RenderedOutput>,
    pub changed: ChangedRegions,
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("render plan is unsupported: {0}")]
    Unsupported(String),
    #[error("invalid backend payload: {0}")]
    InvalidPayload(String),
    #[error("backend resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("backend device was lost: {0}")]
    DeviceLost(String),
    #[error("backend execution failed: {0}")]
    Execution(String),
}

/// Factory supplied by a render backend crate.
pub trait RenderBackend: Send + Sync + fmt::Debug {
    fn capabilities(&self) -> BackendCapabilities;

    fn create_frame_session(
        &self,
        frame: &FrameSessionDesc,
        plan: Arc<RenderPlan>,
    ) -> Result<Box<dyn FrameSession>, BackendError>;
}

/// Per-frame state. Enqueue never waits for GPU completion; synchronization is explicit in
/// [`submit`](Self::submit) and [`wait`](Self::wait).
pub trait FrameSession: Send {
    /// Updates a late-bound plan resource. Revisions must be monotonically increasing per ID.
    fn update_resource(&mut self, update: ResourceUpdate) -> Result<(), BackendError>;

    fn enqueue(&mut self, payload: GroupPayload) -> Result<(), BackendError>;

    fn submit(&mut self, intent: RenderIntent) -> Result<SubmissionToken, BackendError>;

    fn wait(&mut self, token: SubmissionToken) -> Result<RenderedFrame, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_plan() -> RenderPlan {
        RenderPlan {
            planes: vec![
                PlaneDesc {
                    id: PlaneId(0),
                    extent: Extent2d::new(16, 16),
                    stride: 16,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Source,
                },
                PlaneDesc {
                    id: PlaneId(1),
                    extent: Extent2d::new(32, 32),
                    stride: 32,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Intermediate,
                },
                PlaneDesc {
                    id: PlaneId(2),
                    extent: Extent2d::new(32, 32),
                    stride: 32,
                    sample_type: SampleType::F32,
                    role: PlaneRole::Output,
                },
            ],
            nodes: vec![
                RenderNode {
                    name: "upsample".into(),
                    op: RenderOp::Upsample(UpsampleParams {
                        factor: UpsamplingFactor::X2,
                        weights: ResourceId(0),
                    }),
                    inputs: vec![PlaneId(0)],
                    outputs: vec![PlaneId(1)],
                    resources: vec![ResourceId(0)],
                    scale: Scale2d::new(2, 2),
                    border: Border2d::symmetric(2, 2),
                    precision: PrecisionContract::default(),
                },
                RenderNode {
                    name: "copy".into(),
                    op: RenderOp::Copy,
                    inputs: vec![PlaneId(1)],
                    outputs: vec![PlaneId(2)],
                    resources: Vec::new(),
                    scale: Scale2d::IDENTITY,
                    border: Border2d::default(),
                    precision: PrecisionContract::Exact,
                },
            ],
            outputs: Vec::new(),
        }
    }

    #[test]
    fn validates_topological_plan() {
        assert_eq!(test_plan().validate(), Ok(()));
    }

    #[test]
    fn rejects_read_before_write() {
        let mut plan = test_plan();
        plan.nodes.swap(0, 1);
        assert!(matches!(
            plan.validate(),
            Err(PlanError::ReadBeforeWrite {
                node: 0,
                plane: PlaneId(1)
            })
        ));
    }

    #[test]
    fn premultiply_alpha_dependency_must_be_an_explicit_available_input() {
        let alpha = PlaneId(3);
        let mut missing = test_plan();
        missing.nodes[1].op = RenderOp::PremultiplyAlpha { alpha_plane: alpha };
        assert_eq!(
            missing.validate(),
            Err(PlanError::OperationInput {
                node: 1,
                operation: RenderOpKind::PremultiplyAlpha,
                plane: alpha,
            })
        );

        let mut unknown = test_plan();
        unknown.nodes[1].op = RenderOp::PremultiplyAlpha { alpha_plane: alpha };
        unknown.nodes[1].inputs.push(alpha);
        assert_eq!(unknown.validate(), Err(PlanError::UnknownPlane(alpha)));

        let mut read_before_write = test_plan();
        read_before_write.planes.push(PlaneDesc {
            id: alpha,
            extent: Extent2d::new(32, 32),
            stride: 32,
            sample_type: SampleType::F32,
            role: PlaneRole::Intermediate,
        });
        read_before_write.nodes[1].op = RenderOp::PremultiplyAlpha { alpha_plane: alpha };
        read_before_write.nodes[1].inputs.push(alpha);
        assert_eq!(
            read_before_write.validate(),
            Err(PlanError::ReadBeforeWrite {
                node: 1,
                plane: alpha,
            })
        );
    }

    #[test]
    fn rejects_writes_to_source_and_parameter_planes() {
        for role in [PlaneRole::Source, PlaneRole::Parameter] {
            let mut plan = test_plan();
            plan.planes[1].role = role;
            assert_eq!(
                plan.validate(),
                Err(PlanError::ReadOnlyPlaneWritten {
                    node: 0,
                    plane: PlaneId(1),
                })
            );
        }
    }

    #[test]
    fn computes_plane_lifetimes() {
        let lifetimes = test_plan().lifetimes();
        assert_eq!(lifetimes[&PlaneId(0)], PlaneLifetime { first: 0, last: 0 });
        assert_eq!(lifetimes[&PlaneId(1)], PlaneLifetime { first: 0, last: 1 });
        assert_eq!(lifetimes[&PlaneId(2)], PlaneLifetime { first: 1, last: 1 });
    }

    #[test]
    fn propagates_scale_and_halo() {
        let dirty = test_plan().propagate_dirty(PlaneId(0), Region::new(4, 4, 2, 2));
        assert_eq!(dirty[&PlaneId(1)], Region::new(4, 4, 12, 12));
        assert_eq!(dirty[&PlaneId(2)], Region::new(4, 4, 12, 12));
    }

    #[test]
    fn validates_host_plane_length() {
        let plane = HostPlane {
            id: PlaneId(0),
            extent: Extent2d::new(4, 3),
            stride: 8,
            origin: (0, 0),
            data: PlaneData::F32(vec![0.0; 24]),
        };
        assert!(plane.validate().is_ok());
    }

    #[test]
    fn clamps_fully_outside_region_to_extent_edge() {
        assert_eq!(
            Region::new(100, 200, 5, 7).clamp_to(Extent2d::new(16, 9)),
            Region::new(16, 9, 0, 0)
        );
    }

    #[test]
    fn capabilities_match_supported_operation_kinds() {
        let capabilities = BackendCapabilities {
            name: "operation set".into(),
            supported_ops: BTreeSet::from([RenderOpKind::Upsample, RenderOpKind::Copy]),
            max_buffer_bytes: 0,
            max_workgroup_storage_bytes: 0,
            max_invocations_per_workgroup: 0,
            supports_timestamps: false,
            supports_f16: false,
        };
        assert!(capabilities.supports_plan(&test_plan()));

        let mut unsupported = test_plan();
        unsupported.nodes[0].op = RenderOp::AddNoise { seed0: 1, seed1: 2 };
        assert!(!capabilities.supports_plan(&unsupported));
    }

    #[test]
    fn vardct_strategy_table_has_all_codestream_extents() {
        let expected = [
            (TransformKind::Dct8, (8, 8)),
            (TransformKind::Hornuss, (8, 8)),
            (TransformKind::Dct2x2, (8, 8)),
            (TransformKind::Dct4x4, (8, 8)),
            (TransformKind::Dct16x16, (16, 16)),
            (TransformKind::Dct32x32, (32, 32)),
            (TransformKind::Dct16x8, (8, 16)),
            (TransformKind::Dct8x16, (16, 8)),
            (TransformKind::Dct32x8, (8, 32)),
            (TransformKind::Dct8x32, (32, 8)),
            (TransformKind::Dct32x16, (16, 32)),
            (TransformKind::Dct16x32, (32, 16)),
            (TransformKind::Dct4x8, (8, 8)),
            (TransformKind::Dct8x4, (8, 8)),
            (TransformKind::Afv0, (8, 8)),
            (TransformKind::Afv1, (8, 8)),
            (TransformKind::Afv2, (8, 8)),
            (TransformKind::Afv3, (8, 8)),
            (TransformKind::Dct64x64, (64, 64)),
            (TransformKind::Dct64x32, (32, 64)),
            (TransformKind::Dct32x64, (64, 32)),
            (TransformKind::Dct128x128, (128, 128)),
            (TransformKind::Dct128x64, (64, 128)),
            (TransformKind::Dct64x128, (128, 64)),
            (TransformKind::Dct256x256, (256, 256)),
            (TransformKind::Dct256x128, (128, 256)),
            (TransformKind::Dct128x256, (256, 128)),
        ];

        assert_eq!(TransformKind::ALL.len(), expected.len());
        for (actual, (strategy, (width, height))) in TransformKind::ALL.into_iter().zip(expected) {
            assert_eq!(actual, strategy);
            assert_eq!(actual.pixel_extent(), Extent2d::new(width, height));
            assert_eq!(actual.lf_extent(), Extent2d::new(width / 8, height / 8));
        }
        assert_eq!(
            TransformKind::ALL
                .into_iter()
                .filter(|strategy| strategy.is_special())
                .count(),
            9
        );
    }
}
