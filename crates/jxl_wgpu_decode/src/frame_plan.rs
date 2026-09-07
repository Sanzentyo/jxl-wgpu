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
    #[error("preview presentation is not connected to the GPU frame executor")]
    PreviewUnsupported,
    #[error("physical frame {frame_index} requires GPU canvas composition")]
    CompositionRequired { frame_index: u32 },
    #[error("physical frame {frame_index} requires GPU reference retention")]
    ReferenceRequired { frame_index: u32 },
}

impl FrameExecutionPlan {
    pub fn negotiate(inventory: &CodestreamInventory) -> Result<Self, FramePlanError> {
        let image = &inventory.image_header;
        if image.preview_size.is_some() || inventory.frames.iter().any(|frame| frame.is_preview) {
            return Err(FramePlanError::PreviewUnsupported);
        }
        if inventory.frames.is_empty() {
            return Err(FramePlanError::MissingFrame);
        }
        let orientation = OutputOrientation::from_exif_value(image.orientation).ok_or(
            FramePlanError::InvalidOrientation {
                orientation: image.orientation,
            },
        )?;
        let extent = orientation.map_extent(Extent2d::new(image.width, image.height));
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
        let mut references = [None; 4];
        let mut nodes = Vec::with_capacity(inventory.frames.len());
        let mut presentations = Vec::new();
        let mut first = 0;
        let mut ticks = 0_u64;
        for (index, frame) in inventory.frames.iter().enumerate() {
            let invalid = |reason| FramePlanError::InvalidFrame {
                frame_index: frame.frame_index,
                reason,
            };
            let normal = matches!(
                frame.frame_type,
                FrameType::Regular | FrameType::SkipProgressive
            );
            if usize::try_from(frame.frame_index).ok() != Some(index) {
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
            if let Some(source) = frame.lf_source_frame
                && (source >= frame.frame_index
                    || inventory.frames[source as usize].frame_type != FrameType::LowFrequency)
            {
                return Err(invalid("LF dependency is not an earlier LF producer"));
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
                                && node
                                    .lf_source_frame
                                    .is_none_or(|source| source as usize >= first)
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

    pub(crate) fn validate_independent_frames(
        &self,
        inventory: &CodestreamInventory,
    ) -> Result<(), FramePlanError> {
        for (node, frame) in self.nodes.iter().zip(&inventory.frames) {
            if node.needs_composition {
                return Err(FramePlanError::CompositionRequired {
                    frame_index: node.frame_index,
                });
            }
            if frame.frame_type == FrameType::ReferenceOnly {
                return Err(FramePlanError::ReferenceRequired {
                    frame_index: node.frame_index,
                });
            }
        }
        Ok(())
    }
}
