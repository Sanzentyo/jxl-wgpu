// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Backend-neutral protocol for accelerating JPEG XL rendering.
//!
//! The decoder deliberately exposes no `wgpu` types here.  An accelerator can
//! therefore live in a separate crate, share an application's existing device,
//! and be omitted entirely on builds that do not need GPU support.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

/// Wire/API version of the backend-neutral render protocol.
pub const PROTOCOL_VERSION: u32 = 1;

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
    CpuFallback,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferFunction {
    Linear,
    Srgb,
    Bt709,
    Pq,
    Hlg,
    Gamma,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Replace,
    Add,
    Blend,
    Multiply,
    MulAdd,
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
    fn snapshot(&self, plane: Option<&PlaneDesc>) -> Result<ResourceData, AcceleratorError>;
}

#[derive(Clone, Debug)]
pub struct UpsampleParams {
    pub factor: u8,
    /// Phase-major 5x5 kernels. Custom transform data is intentionally not compiled into WGSL.
    pub weights: Arc<[f32]>,
}

#[derive(Clone, Debug)]
pub struct XybParams {
    pub opsin_bias: [f32; 3],
    pub inverse_opsin_matrix: [[f32; 3]; 3],
    pub intensity_target: f32,
}

#[derive(Clone, Debug)]
pub struct TransferParams {
    pub function: TransferFunction,
    pub gamma: f32,
    pub intensity_target: f32,
    pub min_nits: f32,
}

#[derive(Clone, Debug)]
pub struct BlendParams {
    pub mode: BlendMode,
    pub alpha_plane: Option<PlaneId>,
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
    /// VarDCT packet stage. `transform` is the maximum square DCT edge handled natively by the
    /// backend; the baseline contract uses `8` and packet fallbacks cover other shapes.
    VarDct {
        transform: u16,
    },
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
    Extend {
        image_extent: Extent2d,
        origin: (i32, i32),
    },
    Save(SaveParams),
    /// An explicit CPU boundary. Backends must never silently approximate it.
    CpuFallback {
        reason: Arc<str>,
    },
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
            Self::VarDct { .. } => RenderOpKind::VarDct,
            Self::AddNoise { .. } => RenderOpKind::AddNoise,
            Self::XybToRgb(_) => RenderOpKind::XybToRgb,
            Self::YcbcrToRgb => RenderOpKind::YcbcrToRgb,
            Self::TransferFunction(_) => RenderOpKind::TransferFunction,
            Self::Blend(_) => RenderOpKind::Blend,
            Self::PremultiplyAlpha { .. } => RenderOpKind::PremultiplyAlpha,
            Self::Convert { .. } => RenderOpKind::Convert,
            Self::Extend { .. } => RenderOpKind::Extend,
            Self::Save(_) => RenderOpKind::Save,
            Self::CpuFallback { .. } => RenderOpKind::CpuFallback,
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
            .filter(|plane| matches!(plane.role, PlaneRole::Source | PlaneRole::Parameter))
            .map(|plane| plane.id)
            .collect();
        let mut written = BTreeSet::new();
        for (node_index, node) in self.nodes.iter().enumerate() {
            if node.scale.x == 0 || node.scale.y == 0 {
                return Err(PlanError::ZeroScale(node_index));
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
                if matches!(output_desc.role, PlaneRole::Source | PlaneRole::Parameter) {
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FallbackGranularity {
    #[default]
    WholeFrame,
    TransformTile,
    CpuSuffix,
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
    pub fallback: FallbackGranularity,
}

#[derive(Clone, Debug)]
pub struct AcceleratorCapabilities {
    pub name: String,
    pub supported_ops: BTreeSet<RenderOpKind>,
    /// Whole-frame pixel count below which automatic decoder integration should keep the
    /// low-memory CPU pipeline. `None` disables automatic integration, while `Some(0)` always
    /// opts into acceleration.
    pub minimum_frame_pixels: Option<u64>,
    pub max_buffer_bytes: u64,
    pub max_workgroup_storage_bytes: u32,
    pub max_invocations_per_workgroup: u32,
    pub supports_timestamps: bool,
    pub supports_f16: bool,
}

impl AcceleratorCapabilities {
    pub fn supports_plan(&self, plan: &RenderPlan) -> bool {
        plan.nodes
            .iter()
            .all(|node| self.supported_ops.contains(&node.op.kind()))
    }

    pub fn prefers_frame_size(&self, size: (usize, usize)) -> bool {
        let Some(minimum_frame_pixels) = self.minimum_frame_pixels else {
            return false;
        };
        let width = u64::try_from(size.0).unwrap_or(u64::MAX);
        let height = u64::try_from(size.1).unwrap_or(u64::MAX);
        width.saturating_mul(height) >= minimum_frame_pixels
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
    pub fn validate(&self) -> Result<(), AcceleratorError> {
        let stride = if self.stride == 0 {
            self.extent.width
        } else {
            self.stride
        };
        if stride < self.extent.width {
            return Err(AcceleratorError::InvalidPayload(format!(
                "plane {:?} stride {stride} is shorter than width {}",
                self.id, self.extent.width
            )));
        }
        let required = usize::try_from(stride)
            .ok()
            .and_then(|stride| stride.checked_mul(self.extent.height as usize))
            .ok_or_else(|| AcceleratorError::InvalidPayload("plane size overflow".into()))?;
        if self.data.len() < required {
            return Err(AcceleratorError::InvalidPayload(format!(
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
    Dct2,
    Dct4,
    Dct4x8,
    Dct8x4,
    Dct8,
    Dct8x16,
    Dct16x8,
    Dct16,
    Dct16x32,
    Dct32x16,
    Dct32,
    Dct32x64,
    Dct64x32,
    Dct64,
    Dct128,
    Dct256,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Hornuss,
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
    /// Scalar offset into [`VarDctPacket::coefficients`]. DCT8 data is channel-major X/Y/B,
    /// followed by 64 coefficients per channel in the exact linear transform-buffer order
    /// consumed by `jxl_transforms::transform::transform_to_pixels`.
    pub coefficient_offset: u32,
    pub coefficient_count: u32,
    /// Absolute destination coordinates in the three output planes.
    pub destination_x: u32,
    pub destination_y: u32,
    pub block_width: u16,
    pub block_height: u16,
    pub quant_index: u16,
    pub dequant_matrix_index: u16,
    pub correlation_index: u16,
    /// Index of the dequantized X/Y/B LF coefficient that replaces coefficient zero before IDCT.
    pub lf_index: u32,
    pub lf_x: u16,
    pub lf_y: u16,
    pub hshift: u8,
    pub vshift: u8,
}

#[derive(Clone, Debug)]
pub struct TransformBucket {
    pub transform: TransformKind,
    pub tasks: Vec<TransformTask>,
}

#[derive(Clone, Debug)]
pub struct CpuRenderedTile {
    pub region: Region,
    pub channels: [Vec<f32>; 3],
}

/// Owned group packet that can outlive entropy-decoder worker callbacks.
#[derive(Clone, Debug)]
pub struct VarDctPacket {
    pub revision: u32,
    pub last_pass: u16,
    pub coefficients: PackedCoefficients,
    pub buckets: Vec<TransformBucket>,
    pub cpu_fallback_tiles: Vec<CpuRenderedTile>,
}

/// Per-frequency dequantization multipliers for one DCT8 matrix.
///
/// `scales` contains exactly 64 entries in the same linear order as the DCT8 coefficients passed
/// to `jxl_transforms::transform::transform_to_pixels`. Each entry stores the X/Y/B channel
/// multipliers applied after quantization-bias adjustment.
#[derive(Clone, Debug)]
pub struct VarDctDct8DequantMatrix {
    pub scales: Vec<[f32; 3]>,
}

/// Typed late-bound parameters for the bounded DCT8 GPU path.
///
/// A [`TransformTask`] selects one `quant_scales` entry with `quant_index`, one
/// `dequant_matrices` entry with `dequant_matrix_index`, and one `correlations` entry with
/// `correlation_index`. Its `lf_index` selects the separately decoded coefficient zero.
/// HF coefficients are bias-adjusted using `quant_biases`, multiplied by both selected
/// dequantization factors, then Y is correlated into X/B using `[y_to_x, y_to_b]` before the LF
/// replacement and inverse transform. Producers can therefore represent global scale, raw
/// quantization, channel multipliers, quantization matrices, color correlation, and the LF image
/// without an untyped positional `Vec<f32>` contract.
#[derive(Clone, Debug)]
pub struct VarDctDct8Resource {
    /// Biases for X/Y/B small coefficients followed by the large-coefficient numerator.
    pub quant_biases: [f32; 4],
    /// Per-quantization-index X/Y/B multipliers.
    pub quant_scales: Vec<[f32; 3]>,
    pub dequant_matrices: Vec<VarDctDct8DequantMatrix>,
    /// Per-correlation-index `[y_to_x, y_to_b]` multipliers.
    pub correlations: Vec<[f32; 2]>,
    /// Per-task dequantized LF coefficients in X/Y/B order. JPEG XL transmits these separately
    /// from the HF coefficient stream; `transform_to_pixels` replaces coefficient zero with this
    /// value immediately before DCT8.
    pub lf_coefficients: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub enum ResourceData {
    Plane(HostPlane),
    F32(Vec<f32>),
    I32(Vec<i32>),
    Bytes(Vec<u8>),
    VarDctDct8(VarDctDct8Resource),
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
pub struct AcceleratedFrame {
    pub token: SubmissionToken,
    pub outputs: Vec<RenderedOutput>,
    pub changed: ChangedRegions,
}

#[derive(Debug, thiserror::Error)]
pub enum AcceleratorError {
    #[error("render plan is unsupported: {0}")]
    Unsupported(String),
    #[error("invalid accelerator payload: {0}")]
    InvalidPayload(String),
    #[error("accelerator resource limit exceeded: {0}")]
    ResourceLimit(String),
    #[error("accelerator device was lost: {0}")]
    DeviceLost(String),
    #[error("accelerator execution failed: {0}")]
    Execution(String),
}

/// Factory supplied by an optional acceleration crate.
pub trait JxlAccelerator: Send + Sync + fmt::Debug {
    fn capabilities(&self) -> AcceleratorCapabilities;

    fn create_frame_session(
        &self,
        frame: &FrameSessionDesc,
        plan: Arc<RenderPlan>,
    ) -> Result<Box<dyn AcceleratedFrameSession>, AcceleratorError>;
}

/// Per-frame state. Enqueue never waits for GPU completion; synchronization is explicit in
/// [`submit`](Self::submit) and [`wait`](Self::wait).
pub trait AcceleratedFrameSession: Send {
    /// Updates a late-bound plan resource. Revisions must be monotonically increasing per ID.
    fn update_resource(&mut self, update: ResourceUpdate) -> Result<(), AcceleratorError>;

    fn enqueue(&mut self, payload: GroupPayload) -> Result<(), AcceleratorError>;

    fn submit(&mut self, intent: RenderIntent) -> Result<SubmissionToken, AcceleratorError>;

    fn wait(&mut self, token: SubmissionToken) -> Result<AcceleratedFrame, AcceleratorError>;
}

/// Neutral names preferred by standalone producers and backends.
///
/// The original names remain available so the source-tree prototype can be adapted without a
/// flag-day rename.
pub use AcceleratedFrameSession as FrameSession;
pub use AcceleratorCapabilities as BackendCapabilities;
pub use AcceleratorError as BackendError;
pub use JxlAccelerator as RenderBackend;

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
                        factor: 2,
                        weights: vec![0.0; 4 * 25].into(),
                    }),
                    inputs: vec![PlaneId(0)],
                    outputs: vec![PlaneId(1)],
                    resources: Vec::new(),
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
    fn accelerator_cost_hint_handles_thresholds_and_overflow() {
        let capabilities = AcceleratorCapabilities {
            name: "cost model".into(),
            supported_ops: BTreeSet::new(),
            minimum_frame_pixels: Some(65_536),
            max_buffer_bytes: 0,
            max_workgroup_storage_bytes: 0,
            max_invocations_per_workgroup: 0,
            supports_timestamps: false,
            supports_f16: false,
        };
        assert!(!capabilities.prefers_frame_size((255, 255)));
        assert!(capabilities.prefers_frame_size((256, 256)));
        assert!(capabilities.prefers_frame_size((usize::MAX, usize::MAX)));

        let disabled = AcceleratorCapabilities {
            minimum_frame_pixels: None,
            ..capabilities
        };
        assert!(!disabled.prefers_frame_size((usize::MAX, usize::MAX)));
    }
}
