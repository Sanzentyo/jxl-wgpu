// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Extra channel metadata and decode execution planning.
//!
//! Tracks image-header extra channel properties (alpha, depth, spot color, etc.),
//! dimensional shifts, bit depths, and provides plan validation for rendering pipelines.

use jxl_gpu_bitstream::{ExtraChannelInventory, ExtraChannelTypeInventory, SampleBitDepth};
use jxl_gpu_protocol::{
    BlendComponent, BlendMode, BlendParams, Border2d, Extent2d, OutputId, PlaneId,
    PrecisionContract, RenderNode, RenderOp, Scale2d, UpsamplingFactor,
};

use crate::{Error, Result};

/// Semantic type and properties of an extra channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExtraChannelType {
    /// Alpha transparency. `associated: true` indicates premultiplied alpha.
    Alpha { associated: bool },
    /// Depth map.
    Depth,
    /// Spot colour with sRGB coordinates and solidity.
    SpotColour {
        red: f32,
        green: f32,
        blue: f32,
        solidity: f32,
    },
    /// Selection mask.
    SelectionMask,
    /// Black (K in CMYK).
    Black,
    /// Color Filter Array data for raw sensor pixels.
    Cfa { channel: u32 },
    /// Thermal image data.
    Thermal,
    /// Non-optional generic extra channel.
    NonOptional,
    /// Optional generic extra channel.
    Optional,
}

impl From<ExtraChannelTypeInventory> for ExtraChannelType {
    fn from(inv: ExtraChannelTypeInventory) -> Self {
        match inv {
            ExtraChannelTypeInventory::Alpha { associated } => Self::Alpha { associated },
            ExtraChannelTypeInventory::Depth => Self::Depth,
            ExtraChannelTypeInventory::SpotColour {
                red,
                green,
                blue,
                solidity,
            } => Self::SpotColour {
                red: red.to_f32(),
                green: green.to_f32(),
                blue: blue.to_f32(),
                solidity: solidity.to_f32(),
            },
            ExtraChannelTypeInventory::SelectionMask => Self::SelectionMask,
            ExtraChannelTypeInventory::Black => Self::Black,
            ExtraChannelTypeInventory::Cfa { channel } => Self::Cfa { channel },
            ExtraChannelTypeInventory::Thermal => Self::Thermal,
            ExtraChannelTypeInventory::NonOptional => Self::NonOptional,
            ExtraChannelTypeInventory::Optional => Self::Optional,
        }
    }
}

/// Metadata description for one extra channel in an image.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraChannelInfo {
    /// 0-based channel index among all extra channels in the image.
    pub index: u32,
    /// Semantic type of this extra channel.
    pub channel_type: ExtraChannelType,
    /// Declared bit depth (integer or float).
    pub bit_depth: SampleBitDepth,
    /// Dimension downsampling shift factor (0 = full resolution, 1 = 1/2 size, etc.).
    pub dimension_shift: u32,
    /// Optional channel name (UTF-8 encoded).
    pub name: String,
}

impl ExtraChannelInfo {
    /// Returns the bits per sample of this extra channel.
    #[must_use]
    pub const fn bits_per_sample(&self) -> u32 {
        match self.bit_depth {
            SampleBitDepth::Integer { bits_per_sample } => bits_per_sample,
            SampleBitDepth::Float {
                bits_per_sample, ..
            } => bits_per_sample,
        }
    }

    /// Returns `true` if this is an alpha channel.
    #[must_use]
    pub const fn is_alpha(&self) -> bool {
        matches!(self.channel_type, ExtraChannelType::Alpha { .. })
    }

    /// Returns `true` if this is an associated (premultiplied) alpha channel.
    #[must_use]
    pub const fn is_premultiplied_alpha(&self) -> bool {
        matches!(
            self.channel_type,
            ExtraChannelType::Alpha { associated: true }
        )
    }

    /// Computes the downsampled extent for this extra channel given the main image extent.
    #[must_use]
    pub fn shifted_extent(&self, image_extent: Extent2d) -> Extent2d {
        Extent2d::new(
            (image_extent.width >> self.dimension_shift).max(1),
            (image_extent.height >> self.dimension_shift).max(1),
        )
    }

    /// Returns the upsampling factor needed to restore this channel to full resolution,
    /// or `None` if it is already full resolution (`dimension_shift == 0`).
    #[must_use]
    pub const fn upsampling_factor(&self) -> Option<UpsamplingFactor> {
        match self.dimension_shift {
            0 => None,
            1 => Some(UpsamplingFactor::X2),
            2 => Some(UpsamplingFactor::X4),
            3 => Some(UpsamplingFactor::X8),
            _ => None,
        }
    }
}

/// Complete plan and inventory of all extra channels declared by an image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtraChannelPlan {
    channels: Vec<ExtraChannelInfo>,
    alpha_channel_index: Option<u32>,
}

impl ExtraChannelPlan {
    /// Constructs an [`ExtraChannelPlan`] from a slice of parsed bitstream inventories.
    pub fn from_inventory(inventory: &[ExtraChannelInventory]) -> Result<Self> {
        let mut channels = Vec::with_capacity(inventory.len());
        let mut alpha_channel_index = None;

        for (index, inv) in inventory.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| {
                Error::UnsupportedOutputFormat("extra channel index overflow".into())
            })?;
            let name = String::from_utf8_lossy(&inv.name_bytes).into_owned();
            let info = ExtraChannelInfo {
                index,
                channel_type: inv.channel_type.into(),
                bit_depth: inv.bit_depth,
                dimension_shift: inv.dimension_shift,
                name,
            };

            if info.is_alpha() && alpha_channel_index.is_none() {
                alpha_channel_index = Some(index);
            }

            channels.push(info);
        }

        let plan = Self {
            channels,
            alpha_channel_index,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Validates internal consistency of the extra channel plan.
    pub fn validate(&self) -> Result<()> {
        for (i, channel) in self.channels.iter().enumerate() {
            if channel.index != i as u32 {
                return Err(Error::UnsupportedOutputFormat(format!(
                    "extra channel index mismatch: declared {}, actual {}",
                    channel.index, i
                )));
            }
            if channel.dimension_shift > 3 {
                return Err(Error::UnsupportedOutputFormat(format!(
                    "extra channel {} dimension shift {} exceeds supported limit (3)",
                    channel.index, channel.dimension_shift
                )));
            }
        }
        Ok(())
    }

    /// Returns all declared extra channels.
    #[must_use]
    pub fn channels(&self) -> &[ExtraChannelInfo] {
        &self.channels
    }

    /// Number of extra channels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Returns `true` if there are no extra channels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    /// Gets metadata for an extra channel by 0-based index.
    #[must_use]
    pub fn channel(&self, index: u32) -> Option<&ExtraChannelInfo> {
        self.channels.get(index as usize)
    }

    /// Gets metadata for the primary alpha channel, if declared.
    #[must_use]
    pub fn alpha_channel(&self) -> Option<&ExtraChannelInfo> {
        self.alpha_channel_index
            .and_then(|index| self.channel(index))
    }

    /// Returns `true` if this image has at least one alpha channel.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.alpha_channel_index.is_some()
    }
}

/// Extra channel descriptor for routing and composition within a render tail.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtraChannelDescriptor {
    pub info: ExtraChannelInfo,
    /// Mode of output routing for this extra channel.
    pub output_mode: ExtraChannelOutputRouting,
}

/// Output routing mode for an extra channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtraChannelOutputRouting {
    /// Save as an independent single-channel output target.
    SeparateOutput(OutputId),
    /// Pack into primary RGBA output (only valid for Alpha channels).
    PackPrimaryAlpha,
    /// Do not save directly (used solely for intermediate blending/composition).
    InternalOnly,
}

/// Specification for blending a source frame plane onto a base plane.
#[derive(Clone, Debug)]
pub struct FrameBlendSpecification {
    pub mode: BlendMode,
    pub clamp: bool,
    pub alpha_associated: bool,
    pub base_plane: PlaneId,
    pub source_plane: PlaneId,
    pub base_alpha_plane: Option<PlaneId>,
    pub source_alpha_plane: Option<PlaneId>,
    pub target_plane: PlaneId,
}

impl FrameBlendSpecification {
    /// Builds a [`RenderNode`] executing this blend specification.
    pub fn build_render_node(&self, name: impl Into<std::sync::Arc<str>>) -> RenderNode {
        let (component, inputs) = if let (Some(base_alpha), Some(source_alpha)) =
            (self.base_alpha_plane, self.source_alpha_plane)
        {
            (
                BlendComponent::Color {
                    alpha_associated: self.alpha_associated,
                },
                vec![
                    self.base_plane,
                    self.source_plane,
                    base_alpha,
                    source_alpha,
                ],
            )
        } else {
            (
                BlendComponent::Color {
                    alpha_associated: false,
                },
                vec![self.base_plane, self.source_plane],
            )
        };

        RenderNode {
            name: name.into(),
            op: RenderOp::Blend(BlendParams {
                mode: self.mode,
                component,
                clamp: self.clamp,
            }),
            inputs,
            outputs: vec![self.target_plane],
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::ZERO,
            precision: PrecisionContract::Float {
                absolute: 1.0e-6,
                relative: 1.0e-6,
                rmse: 1.0e-6,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_channel_plan_parses_and_tracks_alpha() {
        let inv = vec![
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::Alpha { associated: false },
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 8 },
                dimension_shift: 0,
                name_bytes: b"alpha".to_vec(),
            },
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::Depth,
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 16 },
                dimension_shift: 1,
                name_bytes: b"depth".to_vec(),
            },
        ];

        let plan = ExtraChannelPlan::from_inventory(&inv).expect("plan creation");
        assert_eq!(plan.len(), 2);
        assert!(!plan.is_empty());
        assert!(plan.has_alpha());

        let alpha = plan.alpha_channel().expect("alpha channel");
        assert_eq!(alpha.index, 0);
        assert_eq!(alpha.name, "alpha");
        assert_eq!(alpha.bits_per_sample(), 8);
        assert!(!alpha.is_premultiplied_alpha());
        assert_eq!(
            alpha.shifted_extent(Extent2d::new(100, 100)),
            Extent2d::new(100, 100)
        );

        let depth = plan.channel(1).expect("depth channel");
        assert_eq!(depth.index, 1);
        assert_eq!(depth.name, "depth");
        assert_eq!(depth.bits_per_sample(), 16);
        assert_eq!(
            depth.shifted_extent(Extent2d::new(100, 100)),
            Extent2d::new(50, 50)
        );
    }

    #[test]
    fn extra_channel_plan_handles_no_extra_channels() {
        let plan = ExtraChannelPlan::from_inventory(&[]).expect("empty plan");
        assert_eq!(plan.len(), 0);
        assert!(plan.is_empty());
        assert!(!plan.has_alpha());
        assert!(plan.alpha_channel().is_none());
    }

    #[test]
    fn extra_channel_plan_rejects_excessive_dimension_shift() {
        let inv = vec![ExtraChannelInventory {
            channel_type: ExtraChannelTypeInventory::Depth,
            bit_depth: SampleBitDepth::Integer { bits_per_sample: 16 },
            dimension_shift: 4, // > 3
            name_bytes: Vec::new(),
        }];

        let result = ExtraChannelPlan::from_inventory(&inv);
        assert!(result.is_err());
    }

    #[test]
    fn multi_extra_channel_plan_and_routing() {
        use jxl_gpu_bitstream::FiniteF32;

        let inv = vec![
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::Alpha { associated: false },
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 8 },
                dimension_shift: 0,
                name_bytes: b"alpha".to_vec(),
            },
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::Depth,
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 16 },
                dimension_shift: 1,
                name_bytes: b"depth".to_vec(),
            },
            ExtraChannelInventory {
                channel_type: ExtraChannelTypeInventory::SpotColour {
                    red: FiniteF32::from_f32(1.0).unwrap(),
                    green: FiniteF32::from_f32(0.5).unwrap(),
                    blue: FiniteF32::from_f32(0.0).unwrap(),
                    solidity: FiniteF32::from_f32(0.8).unwrap(),
                },
                bit_depth: SampleBitDepth::Integer { bits_per_sample: 8 },
                dimension_shift: 0,
                name_bytes: b"spot".to_vec(),
            },
        ];

        let plan = ExtraChannelPlan::from_inventory(&inv).expect("multi plan creation");
        assert_eq!(plan.len(), 3);
        assert!(plan.has_alpha());

        let descriptors = vec![
            ExtraChannelDescriptor {
                info: plan.channel(0).unwrap().clone(),
                output_mode: ExtraChannelOutputRouting::PackPrimaryAlpha,
            },
            ExtraChannelDescriptor {
                info: plan.channel(1).unwrap().clone(),
                output_mode: ExtraChannelOutputRouting::SeparateOutput(OutputId(1)),
            },
            ExtraChannelDescriptor {
                info: plan.channel(2).unwrap().clone(),
                output_mode: ExtraChannelOutputRouting::SeparateOutput(OutputId(2)),
            },
        ];

        assert_eq!(descriptors.len(), 3);
        assert_eq!(descriptors[0].output_mode, ExtraChannelOutputRouting::PackPrimaryAlpha);
        assert_eq!(descriptors[1].output_mode, ExtraChannelOutputRouting::SeparateOutput(OutputId(1)));
        assert_eq!(descriptors[2].output_mode, ExtraChannelOutputRouting::SeparateOutput(OutputId(2)));
    }

    #[test]
    fn frame_blend_specification_builds_valid_node() {
        let blend_spec = FrameBlendSpecification {
            mode: BlendMode::BlendAbove,
            clamp: true,
            alpha_associated: false,
            base_plane: PlaneId(10),
            source_plane: PlaneId(11),
            base_alpha_plane: Some(PlaneId(12)),
            source_alpha_plane: Some(PlaneId(13)),
            target_plane: PlaneId(14),
        };

        let node = blend_spec.build_render_node("test_blend");
        assert_eq!(&*node.name, "test_blend");
        assert_eq!(node.inputs.len(), 4);
        assert_eq!(node.inputs, vec![PlaneId(10), PlaneId(11), PlaneId(12), PlaneId(13)]);
        assert_eq!(node.outputs, vec![PlaneId(14)]);

        let RenderOp::Blend(params) = &node.op else { panic!("expected blend op") };
        assert_eq!(params.mode, BlendMode::BlendAbove);
        assert!(params.clamp);
        assert_eq!(params.component, BlendComponent::Color { alpha_associated: false });

        // Without alpha
        let blend_no_alpha = FrameBlendSpecification {
            mode: BlendMode::Add,
            clamp: false,
            alpha_associated: false,
            base_plane: PlaneId(20),
            source_plane: PlaneId(21),
            base_alpha_plane: None,
            source_alpha_plane: None,
            target_plane: PlaneId(22),
        };
        let node_no_alpha = blend_no_alpha.build_render_node("add_no_alpha");
        assert_eq!(node_no_alpha.inputs.len(), 2);
        assert_eq!(node_no_alpha.inputs, vec![PlaneId(20), PlaneId(21)]);
    }
}
