//! Bind optional container indexes to real headers and retain the complete restart dependency span.
use std::num::NonZeroU32;
use std::ops::Range;
use std::sync::Arc;

use jxl_gpu_bitstream::{
    CodestreamInventory, FrameBlendMode, FrameIndex, FrameIndexEntry, FrameIndexLimits, FrameType,
};

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
    Index(#[from] jxl_gpu_bitstream::FrameIndexError),
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
        let execution = FrameExecutionPlan::negotiate(inventory)?;
        if execution.presentations.len() as u64 > limits.max_frames {
            return Err(jxl_gpu_bitstream::FrameIndexError::FrameLimit.into());
        }
        let earliest = dependencies(inventory, &execution)?;
        let independent = |presentation: usize| {
            let span = &execution.presentations[presentation].physical_frames;
            earliest[span.clone()]
                .iter()
                .all(|&source| source >= span.start)
        };
        let index = match index {
            Some(index) => {
                // Reapply caller limits, even when the parsed object had looser limits.
                index.encode(limits)?;
                index
            }
            None => {
                let anchors: Vec<_> = (0..execution.presentations.len())
                    .filter(|&i| independent(i))
                    .collect();
                if anchors.first() != Some(&0) {
                    return Err(FrameSeekError::DependentAnchor { entry: 0 });
                }
                let entries = anchors
                    .iter()
                    .enumerate()
                    .map(|(i, &start)| {
                        let end = anchors
                            .get(i + 1)
                            .copied()
                            .unwrap_or(execution.presentations.len());
                        Ok(FrameIndexEntry {
                            codestream_offset: offset(inventory, &execution, start)?,
                            duration_ticks: duration(&execution, start..end),
                            frames: (end - start) as u64,
                        })
                    })
                    .collect::<Result<Vec<_>, FrameSeekError>>()?;
                let (numerator, denominator) = execution.metadata.timebase.map_or(
                    (1, NonZeroU32::new(1).expect("one is nonzero")),
                    |time| {
                        (
                            time.ticks_per_second_denominator.get(),
                            time.ticks_per_second_numerator,
                        )
                    },
                );
                FrameIndex::new(numerator, denominator, entries, limits)?
            }
        };
        let mut anchors = Vec::with_capacity(index.entries().len());
        let mut presentation = 0_usize;
        for (entry, record) in index.entries().iter().enumerate() {
            let next = usize::try_from(record.frames)
                .ok()
                .and_then(|count| presentation.checked_add(count))
                .filter(|&end| end <= execution.presentations.len())
                .ok_or(FrameSeekError::FrameCount)?;
            if offset(inventory, &execution, presentation)? != record.codestream_offset {
                return Err(FrameSeekError::Offset { entry });
            }
            if !independent(presentation) {
                return Err(FrameSeekError::DependentAnchor { entry });
            }
            let ticks = duration(&execution, presentation..next);
            let (tps_num, tps_den) = execution.metadata.timebase.map_or((1, 1), |t| {
                (
                    t.ticks_per_second_numerator.get(),
                    t.ticks_per_second_denominator.get(),
                )
            });
            if u128::from(record.duration_ticks)
                * u128::from(index.tick_numerator())
                * u128::from(tps_num)
                != u128::from(ticks)
                    * u128::from(tps_den)
                    * u128::from(index.tick_denominator().get())
            {
                return Err(FrameSeekError::Duration { entry });
            }
            anchors.push(presentation);
            presentation = next;
        }
        if presentation != execution.presentations.len() {
            return Err(FrameSeekError::FrameCount);
        }
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

fn offset(
    inventory: &CodestreamInventory,
    plan: &FrameExecutionPlan,
    presentation: usize,
) -> Result<u64, FrameSeekError> {
    let span = &plan
        .presentations
        .get(presentation)
        .ok_or(FrameSeekError::FrameCount)?
        .physical_frames;
    let bits = inventory.frames[span.start].header_bits.offset;
    if !bits.is_multiple_of(8) {
        return Err(FrameSeekError::Inventory("unaligned frame header"));
    }
    Ok(bits / 8)
}

fn duration(plan: &FrameExecutionPlan, range: Range<usize>) -> u64 {
    // The full execution plan has already checked the cumulative u64 presentation clock.
    plan.presentations[range]
        .iter()
        .map(|p| u64::from(p.metadata.duration.ticks))
        .sum()
}

fn dependencies(
    inventory: &CodestreamInventory,
    plan: &FrameExecutionPlan,
) -> Result<Vec<usize>, FrameSeekError> {
    let mut earliest = Vec::with_capacity(plan.nodes.len());
    for (position, (frame, node)) in inventory.frames.iter().zip(&plan.nodes).enumerate() {
        let mut first = position;
        let mut use_source = |source: u32| -> Result<(), FrameSeekError> {
            let source = inventory
                .frame_position(source)
                .filter(|&index| index < position)
                .ok_or(FrameSeekError::Inventory(
                    "reference is not an earlier physical frame",
                ))?;
            first = first.min(earliest[source]);
            Ok(())
        };
        if let Some(source) = node.lf_source_frame {
            use_source(source)?;
        }
        if frame.flags & 2 != 0 {
            // Patch selectors are GPU entropy. All live slot versions are conservative dependencies.
            for reference in node.references.iter().flatten() {
                use_source(reference.frame_index)?;
            }
        }
        if matches!(
            frame.frame_type,
            FrameType::Regular | FrameType::SkipProgressive
        ) {
            let covers = i64::from(frame.x0) <= 0
                && i64::from(frame.y0) <= 0
                && i64::from(frame.x0) + i64::from(frame.width)
                    >= i64::from(inventory.image_header.width)
                && i64::from(frame.y0) + i64::from(frame.height)
                    >= i64::from(inventory.image_header.height);
            for blend in std::iter::once(&frame.color_blend).chain(&frame.extra_channel_blends) {
                if (!covers || blend.mode != FrameBlendMode::Replace)
                    && let Some(reference) = node.references[blend.source as usize]
                {
                    use_source(reference.frame_index)?;
                }
                // Unassociated source-over and the selected alpha's output read that alpha's
                // own background slot, even when its declared extra-channel mode is Replace.
                if matches!(
                    blend.mode,
                    FrameBlendMode::Blend | FrameBlendMode::MultiplyAdd
                ) && let Some(alpha) = frame
                    .extra_channel_blends
                    .get(blend.alpha_channel.unwrap_or(0) as usize)
                    && let Some(reference) = node.references[alpha.source as usize]
                {
                    use_source(reference.frame_index)?;
                }
            }
        }
        earliest.push(first);
    }
    Ok(earliest)
}
