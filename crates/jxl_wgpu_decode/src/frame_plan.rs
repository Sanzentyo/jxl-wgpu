//! Frame semantics shared by every GPU coding mode. No pixel or entropy decoding occurs here.

use std::num::NonZeroU32;
use std::ops::Range;

use jxl_gpu_bitstream::{CodestreamInventory, FrameBlendMode, FrameEncoding, FrameType};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};

use crate::{AnimationMetadata, FrameDuration, FrameMetadata, FrameTimebase};

/// The exact producer occupying a reference slot before a physical frame executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameReference {
    pub frame_index: u32,
    pub before_color_transform: bool,
}

/// One physical decode node, including invisible layers and progressive-DC producers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameExecutionNode {
    pub frame_index: u32,
    pub encoding: FrameEncoding,
    pub lf_source_frame: Option<u32>,
    /// Last physical consumer of this LF slot version. Unused LF nodes still execute.
    pub lf_last_use: Option<u32>,
    /// Snapshot before this frame writes a slot. Empty slots represent a zero background.
    /// Patch references discovered later in entropy use this same snapshot.
    pub references: [Option<FrameReference>; 4],
    pub save_reference: Option<u32>,
    /// Whether color or an extra channel requires canvas composition after reconstruction.
    pub needs_composition: bool,
}

/// Physical work coalesced into one presentation. Ranges index `FrameExecutionPlan::nodes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePresentation {
    pub physical_frames: Range<usize>,
    pub metadata: FrameMetadata,
}

/// Backend-neutral frame ordering, reference versions, and exact presentation timing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameExecutionPlan {
    pub metadata: AnimationMetadata,
    pub nodes: Vec<FrameExecutionNode>,
    pub presentations: Vec<FramePresentation>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FramePlanError {
    #[error("frame execution requires at least one physical frame")]
    MissingFrame,
    #[error("invalid frame sequence at physical frame {frame_index}: {reason}")]
    InvalidFrame {
        frame_index: u32,
        reason: &'static str,
    },
    #[error("invalid animation timebase")]
    InvalidTimebase,
    #[error("invalid image orientation {orientation}")]
    InvalidOrientation { orientation: u32 },
    #[error("frame execution requires a selected image reconstruction inventory")]
    ImageNotSelected,
}

impl FrameExecutionPlan {
    /// Plans an image-domain inventory, such as
    /// [`SelectedImageInventory::reconstruction_inventory`](crate::SelectedImageInventory::reconstruction_inventory).
    /// Physical frame IDs may start after zero and are never treated as vector positions.
    pub fn negotiate(inventory: &CodestreamInventory) -> Result<Self, FramePlanError> {
        Self::negotiate_with_orientation(inventory, crate::OrientationPolicy::Apply)
    }

    /// Builds the same physical dependency graph with the requested presentation coordinates.
    pub fn negotiate_with_orientation(
        inventory: &CodestreamInventory,
        orientation_policy: crate::OrientationPolicy,
    ) -> Result<Self, FramePlanError> {
        let image = &inventory.image_header;
        if image.preview_size.is_some() || inventory.frames.iter().any(|frame| frame.is_preview) {
            return Err(FramePlanError::ImageNotSelected);
        }
        if inventory.frames.is_empty() {
            return Err(FramePlanError::MissingFrame);
        }
        let orientation = OutputOrientation::from_exif_value(image.orientation).ok_or(
            FramePlanError::InvalidOrientation {
                orientation: image.orientation,
            },
        )?;
        let extent = orientation_policy
            .resolve(orientation)
            .map_extent(Extent2d::new(image.width, image.height));
        let mut metadata = if let Some(animation) = image.animation {
            let timebase = FrameTimebase {
                ticks_per_second_numerator: NonZeroU32::new(animation.ticks_per_second_numerator)
                    .ok_or(FramePlanError::InvalidTimebase)?,
                ticks_per_second_denominator: NonZeroU32::new(
                    animation.ticks_per_second_denominator,
                )
                .ok_or(FramePlanError::InvalidTimebase)?,
            };
            AnimationMetadata::animation(
                extent,
                timebase,
                animation.num_loops,
                animation.have_timecodes,
                None,
            )
        } else {
            AnimationMetadata::still(extent)
        };
        metadata.extra_channels = image.extra_channels.clone();
        let mut references = [None; 4];
        let mut lf_references = [None; 4];
        let mut nodes: Vec<FrameExecutionNode> = Vec::with_capacity(inventory.frames.len());
        let mut presentations = Vec::new();
        let mut first = 0;
        let mut ticks = 0_u64;
        let first_frame_index = inventory.frames[0].frame_index;
        for (index, frame) in inventory.frames.iter().enumerate() {
            let invalid = |reason| FramePlanError::InvalidFrame {
                frame_index: frame.frame_index,
                reason,
            };
            let normal = matches!(
                frame.frame_type,
                FrameType::Regular | FrameType::SkipProgressive
            );
            if frame
                .frame_index
                .checked_sub(first_frame_index)
                .and_then(|value| usize::try_from(value).ok())
                != Some(index)
            {
                return Err(invalid("noncontiguous physical indices"));
            }
            if frame.is_last != (index + 1 == inventory.frames.len()) || (frame.is_last && !normal)
            {
                return Err(invalid("missing or misplaced final frame"));
            }
            if frame.width == 0 || frame.height == 0 {
                return Err(invalid("empty frame rectangle"));
            }
            if frame.timecode.is_some() != (normal && metadata.has_timecodes.unwrap_or(false))
                || (metadata.timebase.is_none() && frame.duration_ticks != 0)
            {
                return Err(invalid("timing does not match the image header"));
            }
            let is_lf = frame.frame_type == FrameType::LowFrequency;
            if (is_lf && !(1..=4).contains(&frame.lf_level)) || (!is_lf && frame.lf_level != 0) {
                return Err(invalid("invalid LF frame level"));
            }
            if frame.uses_lf_frame() != frame.lf_source_frame.is_some() {
                return Err(invalid("LF dependency does not match the frame flags"));
            }
            if let Some(source) = frame.lf_source_frame {
                if frame.encoding != FrameEncoding::VarDct
                    || lf_references.get(frame.lf_level as usize) != Some(&Some(source))
                {
                    return Err(invalid(
                        "LF dependency is not the current producer at the next level",
                    ));
                }
                let source_position = inventory
                    .frame_position(source)
                    .ok_or_else(|| invalid("LF producer is outside the selected image"))?;
                let producer = &inventory.frames[source_position];
                let (width, height) = frame
                    .color_sample_extent()
                    .ok_or_else(|| invalid("invalid encoded sample extent"))?;
                // LF slots store the reconstructed frame after restoration and frame upsampling.
                let lf_scale = 1_u32 << (3 * producer.lf_level);
                let source_extent = (
                    producer.width.div_ceil(lf_scale),
                    producer.height.div_ceil(lf_scale),
                );
                if source_extent != (width.div_ceil(8), height.div_ceil(8)) {
                    return Err(invalid(
                        "LF producer extent does not match the consumer block extent",
                    ));
                }
                nodes[source_position].lf_last_use = Some(frame.frame_index);
            }
            if is_lf {
                lf_references[frame.lf_level as usize - 1] = Some(frame.frame_index);
            }
            if frame.save_as_reference >= 4
                || frame.color_blend.source >= 4
                || frame
                    .extra_channel_blends
                    .iter()
                    .any(|blend| blend.source >= 4)
                || frame.extra_channel_blends.len() != image.extra_channels.len()
            {
                return Err(invalid("invalid reference or extra-channel topology"));
            }
            let covers_canvas = i64::from(frame.x0) <= 0
                && i64::from(frame.y0) <= 0
                && i64::from(frame.x0) + i64::from(frame.width) >= i64::from(image.width)
                && i64::from(frame.y0) + i64::from(frame.height) >= i64::from(image.height);
            let replaces_all = std::iter::once(&frame.color_blend)
                .chain(&frame.extra_channel_blends)
                .all(|blend| blend.mode == FrameBlendMode::Replace);
            let needs_composition = normal
                && (!covers_canvas
                    || !replaces_all
                    || frame.x0 != 0
                    || frame.y0 != 0
                    || frame.width != image.width
                    || frame.height != image.height);
            let can_reference = !frame.is_last
                && frame.frame_type != FrameType::LowFrequency
                && (frame.duration_ticks == 0 || frame.save_as_reference != 0);
            nodes.push(FrameExecutionNode {
                frame_index: frame.frame_index,
                encoding: frame.encoding,
                lf_source_frame: frame.lf_source_frame,
                lf_last_use: None,
                references,
                save_reference: can_reference.then_some(frame.save_as_reference),
                needs_composition,
            });
            if can_reference {
                references[frame.save_as_reference as usize] = Some(FrameReference {
                    frame_index: frame.frame_index,
                    before_color_transform: frame.save_before_color_transform,
                });
            }
            if normal && (frame.duration_ticks != 0 || frame.is_last) {
                let name = String::from_utf8(frame.name_bytes.clone())
                    .map_err(|_| invalid("frame name is not UTF-8"))?;
                presentations.push(FramePresentation {
                    physical_frames: first..index + 1,
                    metadata: FrameMetadata {
                        index: presentations.len(),
                        duration: FrameDuration {
                            ticks: frame.duration_ticks,
                            timebase: metadata.timebase,
                        },
                        presentation_ticks: ticks,
                        timecode: frame.timecode,
                        is_last: frame.is_last,
                        // Conservative independence: composition and patch references require a
                        // later dependency closure before a seek can use this as a restart point.
                        is_keyframe: nodes[first..].iter().all(|node| {
                            !node.needs_composition
                                && node.lf_source_frame.is_none_or(|source| {
                                    source >= inventory.frames[first].frame_index
                                })
                        }) && inventory.frames[first..=index]
                            .iter()
                            .all(|layer| layer.flags & 2 == 0),
                        name,
                    },
                });
                first = index + 1;
                ticks = ticks
                    .checked_add(u64::from(frame.duration_ticks))
                    .ok_or_else(|| invalid("presentation clock overflow"))?;
            }
        }
        if first != nodes.len() || presentations.is_empty() {
            return Err(FramePlanError::MissingFrame);
        }
        metadata.frame_count_hint = Some(presentations.len());
        Ok(Self {
            metadata,
            nodes,
            presentations,
        })
    }
}
