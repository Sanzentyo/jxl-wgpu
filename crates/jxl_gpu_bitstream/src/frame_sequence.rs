//! Checked physical-frame semantics shared by container indexing and GPU execution.
//! Planning reads metadata only; it does not validate entropy or reconstruct pixels.

use crate::{AnimationInventory, CodestreamInventory, FrameBlendMode, FrameEncoding, FrameType};
use std::ops::Range;

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

/// Physical frames coalesced into one displayed presentation, in original coordinates/time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSequencePresentation {
    pub physical_frames: Range<usize>,
    pub header_bit_offset: u64,
    pub duration_ticks: u32,
    pub presentation_ticks: u64,
    pub timecode: Option<u32>,
    pub is_last: bool,
    pub is_keyframe: bool,
    pub name: String,
}

/// Whether a selected reconstruction interval must contain the original final frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameSequenceEnd {
    Complete,
    Presentation,
}

/// Immutable header-validated ordering, reference versions and transitive dependencies.
/// Inventories must already select one image domain, with no preview header/frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameSequencePlan {
    complete: bool,
    animation: Option<AnimationInventory>,
    nodes: Vec<FrameExecutionNode>,
    presentations: Vec<FrameSequencePresentation>,
    earliest: Vec<usize>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FramePlanError {
    #[error("frame execution requires at least one physical frame")]
    MissingFrame,
    #[error("invalid header at physical frame {frame_index}: {source}")]
    InvalidHeader {
        frame_index: u32,
        #[source]
        source: crate::InventoryError,
    },
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

impl FrameSequencePlan {
    /// Plans a complete selected image. Physical IDs need not start at zero.
    pub fn negotiate(inventory: &CodestreamInventory) -> Result<Self, FramePlanError> {
        Self::negotiate_interval(inventory, FrameSequenceEnd::Complete)
    }

    /// Plans a selected interval ending at a presentation. References must be contained in
    /// the interval; original finality and reference-saving semantics remain unchanged.
    pub fn negotiate_interval(
        inventory: &CodestreamInventory,
        end: FrameSequenceEnd,
    ) -> Result<Self, FramePlanError> {
        let complete = end == FrameSequenceEnd::Complete;
        let image = &inventory.image_header;
        if image.preview_size.is_some() || inventory.frames.iter().any(|frame| frame.is_preview) {
            return Err(FramePlanError::ImageNotSelected);
        }
        if inventory.frames.is_empty() {
            return Err(FramePlanError::MissingFrame);
        }
        if !(1..=8).contains(&image.orientation) {
            return Err(FramePlanError::InvalidOrientation {
                orientation: image.orientation,
            });
        }
        if image.animation.is_some_and(|a| {
            a.ticks_per_second_numerator == 0 || a.ticks_per_second_denominator == 0
        }) {
            return Err(FramePlanError::InvalidTimebase);
        }
        let mut references = [None; 4];
        let mut lf_references = [None; 4];
        let mut nodes: Vec<FrameExecutionNode> = Vec::with_capacity(inventory.frames.len());
        let mut presentations = Vec::new();
        let mut first = 0;
        let mut ticks = 0_u64;
        let first_frame_index = inventory.frames[0].frame_index;
        for (index, frame) in inventory.frames.iter().enumerate() {
            frame.validate_color_reference(image).map_err(|source| {
                FramePlanError::InvalidHeader {
                    frame_index: frame.frame_index,
                    source,
                }
            })?;
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
            let at_end = index + 1 == inventory.frames.len();
            if (frame.is_last && (!at_end || !normal)) || (complete && at_end && !frame.is_last) {
                return Err(invalid("missing or misplaced final frame"));
            }
            if frame.width == 0 || frame.height == 0 {
                return Err(invalid("empty frame rectangle"));
            }
            if frame.timecode.is_some()
                != (normal && image.animation.is_some_and(|a| a.have_timecodes))
                || (image.animation.is_none() && frame.duration_ticks != 0)
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
                // A cropped consumer reads the top-left block rectangle of the LF image.
                // Its signed canvas origin affects composition, not prediction coordinates.
                if source_extent.0 < width.div_ceil(8) || source_extent.1 < height.div_ceil(8) {
                    return Err(invalid(
                        "LF producer extent is smaller than the consumer block extent",
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
            let covers_canvas = covers_canvas(frame, image);
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
            let can_reference = frame.can_be_referenced();
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
                presentations.push(FrameSequencePresentation {
                    physical_frames: first..index + 1,
                    header_bit_offset: inventory.frames[first].header_bits.offset,
                    duration_ticks: frame.duration_ticks,
                    presentation_ticks: ticks,
                    timecode: frame.timecode,
                    is_last: frame.is_last,
                    // Conservative independence: composition and patch references require a
                    // later dependency closure before a seek can use this as a restart point.
                    is_keyframe: nodes[first..].iter().all(|node| {
                        !node.needs_composition
                            && node
                                .lf_source_frame
                                .is_none_or(|source| source >= inventory.frames[first].frame_index)
                    }) && inventory.frames[first..=index]
                        .iter()
                        .all(|layer| layer.flags & 2 == 0),
                    name,
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
        let earliest = dependencies(inventory, &nodes)?;
        Ok(Self {
            complete,
            animation: image.animation,
            nodes,
            presentations,
            earliest,
        })
    }

    #[must_use]
    pub const fn animation(&self) -> Option<AnimationInventory> {
        self.animation
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete
    }

    #[must_use]
    pub fn nodes(&self) -> &[FrameExecutionNode] {
        &self.nodes
    }

    #[must_use]
    pub fn presentations(&self) -> &[FrameSequencePresentation] {
        &self.presentations
    }

    /// Earliest transitive dependency, as a physical position, for each node.
    #[must_use]
    pub fn earliest_dependencies(&self) -> &[usize] {
        &self.earliest
    }

    /// Transfers checked physical/presentation metadata to a backend-specific execution plan.
    #[must_use]
    pub fn into_parts(self) -> (Vec<FrameExecutionNode>, Vec<FrameSequencePresentation>) {
        (self.nodes, self.presentations)
    }
}

fn dependencies(
    inventory: &CodestreamInventory,
    nodes: &[FrameExecutionNode],
) -> Result<Vec<usize>, FramePlanError> {
    let mut earliest = Vec::with_capacity(nodes.len());
    for (position, (frame, node)) in inventory.frames.iter().zip(nodes).enumerate() {
        let mut first = position;
        let mut use_source = |source: u32| -> Result<(), FramePlanError> {
            let source = inventory
                .frame_position(source)
                .filter(|&index| index < position)
                .ok_or(FramePlanError::InvalidFrame {
                    frame_index: frame.frame_index,
                    reason: "reference is not an earlier physical frame",
                })?;
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
            let covers = covers_canvas(frame, &inventory.image_header);
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

fn covers_canvas(frame: &crate::FrameInventory, image: &crate::ImageHeaderInventory) -> bool {
    i64::from(frame.x0) <= 0
        && i64::from(frame.y0) <= 0
        && i64::from(frame.x0) + i64::from(frame.width) >= i64::from(image.width)
        && i64::from(frame.y0) + i64::from(frame.height) >= i64::from(image.height)
}
