//! Presentation coordinates and output metadata over the shared checked frame sequence.
use crate::{AnimationMetadata, FrameDuration, FrameMetadata, FrameTimebase};
use jxl_gpu_bitstream::{CodestreamInventory, FrameSequenceEnd, FrameSequencePlan};
pub use jxl_gpu_bitstream::{FrameExecutionNode, FramePlanError, FrameReference};
use jxl_gpu_protocol::{Extent2d, OutputOrientation};
use std::num::NonZeroU32;
use std::ops::Range;

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
        Self::negotiate_interval(inventory, orientation_policy, true)
    }

    /// Plans a validated image selection, including a dependency-complete seek interval.
    /// Seek boundaries do not change original frame finality or reference-saving semantics.
    pub fn negotiate_selected(
        selection: &crate::SelectedImageInventory,
        orientation_policy: crate::OrientationPolicy,
    ) -> Result<Self, FramePlanError> {
        Self::negotiate_interval(
            selection.reconstruction_inventory(),
            orientation_policy,
            selection.reconstruction_is_complete(),
        )
    }

    fn negotiate_interval(
        inventory: &CodestreamInventory,
        orientation_policy: crate::OrientationPolicy,
        complete: bool,
    ) -> Result<Self, FramePlanError> {
        let sequence = FrameSequencePlan::negotiate_interval(
            inventory,
            if complete {
                FrameSequenceEnd::Complete
            } else {
                FrameSequenceEnd::Presentation
            },
        )?;
        Self::from_sequence(inventory, orientation_policy, sequence)
    }

    pub(crate) fn from_sequence(
        inventory: &CodestreamInventory,
        orientation_policy: crate::OrientationPolicy,
        sequence: FrameSequencePlan,
    ) -> Result<Self, FramePlanError> {
        let image = &inventory.image_header;
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
        let (nodes, presentations) = sequence.into_parts();
        let presentations: Vec<_> = presentations
            .into_iter()
            .enumerate()
            .map(|(index, frame)| FramePresentation {
                physical_frames: frame.physical_frames,
                metadata: FrameMetadata {
                    index,
                    duration: FrameDuration {
                        ticks: frame.duration_ticks,
                        timebase: metadata.timebase,
                    },
                    presentation_ticks: frame.presentation_ticks,
                    timecode: frame.timecode,
                    is_last: frame.is_last,
                    is_keyframe: frame.is_keyframe,
                    name: frame.name,
                },
            })
            .collect();
        metadata.frame_count_hint = Some(presentations.len());
        Ok(Self {
            metadata,
            nodes,
            presentations,
        })
    }
}
