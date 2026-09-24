//! Bind optional container indexes to real headers and retain the complete restart dependency span.
use std::ops::Range;
use std::sync::Arc;

use jxl_gpu_bitstream::{CodestreamInventory, FrameIndex, FrameIndexLimits, FrameSequencePlan};

use crate::{FrameExecutionPlan, FrameMetadata, ImageSelection, SelectedImageInventory};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSeekLimits {
    /// Complete presentations decoded and discarded before the requested presentation.
    pub max_preroll_presentations: usize,
    /// All physical frames from the restart anchor through the target, including hidden/LF nodes.
    pub max_physical_frames: usize,
}

impl Default for FrameSeekLimits {
    fn default() -> Self {
        Self {
            max_preroll_presentations: 16_383,
            max_physical_frames: 16_384,
        }
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FrameSeekError {
    #[error(transparent)]
    Index(jxl_gpu_bitstream::FrameIndexError),
    #[error(transparent)]
    Image(#[from] crate::ImageSelectionError),
    #[error(transparent)]
    Plan(#[from] crate::FramePlanError),
    #[error("frame seeking selects the main image, not an embedded preview")]
    PreviewSelection,
    #[error("requested presentation {target}, but the stream has {frames}")]
    Target { target: usize, frames: usize },
    #[error("jxli entry {entry} does not identify the expected presentation boundary")]
    Offset { entry: usize },
    #[error("jxli entry {entry} depends on an earlier reference or LF producer")]
    DependentAnchor { entry: usize },
    #[error("jxli displayed-frame intervals do not match the codestream")]
    FrameCount,
    #[error("jxli entry {entry} has a duration inconsistent with the image timebase")]
    Duration { entry: usize },
    #[error("seek preroll requires {required} presentations, limit is {limit}")]
    PrerollLimit { required: usize, limit: usize },
    #[error("seek requires {required} physical frames, limit is {limit}")]
    PhysicalLimit { required: usize, limit: usize },
    #[error("invalid seek inventory: {0}")]
    Inventory(&'static str),
}

/// A header-validated index. Neither binding nor planning validates skipped frame entropy.
#[derive(Clone, Debug)]
pub struct BoundFrameIndex {
    selection: SelectedImageInventory,
    execution: FrameExecutionPlan,
    index: FrameIndex,
    anchors: Vec<usize>,
    earliest: Vec<usize>,
}

impl BoundFrameIndex {
    /// Binds a supplied index, or generates all independently restartable presentation anchors.
    /// Every interval is checked against original logical offsets, frame counts and rational time.
    pub fn new(
        inventory: Arc<CodestreamInventory>,
        index: Option<FrameIndex>,
        limits: FrameIndexLimits,
    ) -> Result<Self, FrameSeekError> {
        let selection = SelectedImageInventory::new(inventory, ImageSelection::Main)?;
        let inventory = selection.reconstruction_inventory();
        let sequence = FrameSequencePlan::negotiate(inventory)?;
        let index = match index {
            Some(index) => index,
            None => FrameIndex::from_sequence(&sequence, limits)?,
        };
        let anchors = index.bind_sequence(&sequence, limits)?;
        let earliest = sequence.earliest_dependencies().to_vec();
        let execution = FrameExecutionPlan::from_sequence(
            inventory,
            crate::OrientationPolicy::Apply,
            sequence,
        )?;
        Ok(Self {
            selection,
            execution,
            index,
            anchors,
            earliest,
        })
    }

    #[must_use]
    pub const fn index(&self) -> &FrameIndex {
        &self.index
    }

    /// Uses the latest indexed anchor whose entire preroll interval has no earlier dependency.
    /// A later independent frame alone cannot discard references used by the requested target.
    pub fn seek(
        &self,
        target: usize,
        limits: FrameSeekLimits,
    ) -> Result<FrameSeekPlan, FrameSeekError> {
        let presentation =
            self.execution
                .presentations
                .get(target)
                .ok_or(FrameSeekError::Target {
                    target,
                    frames: self.execution.presentations.len(),
                })?;
        let end = presentation.physical_frames.end;
        let mut scan = end;
        let mut first_dependency = end;
        let mut chosen = None;
        for &anchor in self
            .anchors
            .iter()
            .rev()
            .filter(|&&anchor| anchor <= target)
        {
            let start = self.execution.presentations[anchor].physical_frames.start;
            first_dependency = first_dependency.min(
                *self.earliest[start..scan]
                    .iter()
                    .min()
                    .ok_or(FrameSeekError::Inventory("empty seek interval"))?,
            );
            scan = start;
            if first_dependency >= start {
                chosen = Some((anchor, start));
                break;
            }
        }
        let (anchor, start) = chosen.ok_or(FrameSeekError::Inventory(
            "no dependency-complete restart anchor",
        ))?;
        let preroll = target - anchor;
        if preroll > limits.max_preroll_presentations {
            return Err(FrameSeekError::PrerollLimit {
                required: preroll,
                limit: limits.max_preroll_presentations,
            });
        }
        if end - start > limits.max_physical_frames {
            return Err(FrameSeekError::PhysicalLimit {
                required: end - start,
                limit: limits.max_physical_frames,
            });
        }
        let frames = &self.selection.reconstruction_inventory().frames;
        let first = frames[start].frame_index;
        let last = frames[end - 1]
            .frame_index
            .checked_add(1)
            .ok_or(FrameSeekError::Inventory("physical frame overflow"))?;
        Ok(FrameSeekPlan {
            selection: self.selection.for_seek(start..end)?,
            target: presentation.metadata.clone(),
            restart_presentation: anchor,
            physical_frames: first..last,
            preroll,
        })
    }
}

/// One target plus the bounded dependency span needed to reconstruct it from empty GPU caches.
#[derive(Clone, Debug)]
pub struct FrameSeekPlan {
    pub(crate) selection: SelectedImageInventory,
    target: FrameMetadata,
    restart_presentation: usize,
    physical_frames: Range<u32>,
    preroll: usize,
}

impl FrameSeekPlan {
    #[must_use]
    pub const fn target(&self) -> &FrameMetadata {
        &self.target
    }
    #[must_use]
    pub const fn restart_presentation(&self) -> usize {
        self.restart_presentation
    }
    #[must_use]
    pub fn physical_frames(&self) -> Range<u32> {
        self.physical_frames.clone()
    }
    #[must_use]
    pub const fn preroll_presentations(&self) -> usize {
        self.preroll
    }
}

// Preserve decoder-facing typed errors while sharing their validation with the writer.
impl From<jxl_gpu_bitstream::FrameIndexError> for FrameSeekError {
    fn from(error: jxl_gpu_bitstream::FrameIndexError) -> Self {
        use jxl_gpu_bitstream::FrameIndexError;
        match error {
            FrameIndexError::Offset { entry } => Self::Offset { entry },
            FrameIndexError::DependentAnchor { entry } => Self::DependentAnchor { entry },
            FrameIndexError::FrameCount => Self::FrameCount,
            FrameIndexError::Duration { entry } => Self::Duration { entry },
            FrameIndexError::UnalignedHeader => Self::Inventory("unaligned frame header"),
            error => Self::Index(error),
        }
    }
}
