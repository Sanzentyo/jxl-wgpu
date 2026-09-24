//! Index boundaries and durations have one owner for both encoding and decoding.

use std::num::NonZeroU32;
use std::ops::Range;

use crate::FrameSequencePlan;

use super::{FrameIndex, FrameIndexEntry, FrameIndexError, FrameIndexLimits};

impl FrameIndex {
    /// Generates every independently restartable presentation in a checked sequence.
    /// Entries include leading hidden/LF frames and retain the original rational clock.
    /// This validates headers/dependencies only, never skipped frame entropy.
    pub fn from_sequence(
        sequence: &FrameSequencePlan,
        limits: FrameIndexLimits,
    ) -> Result<Self, FrameIndexError> {
        check_frame_limit(sequence, limits)?;
        let count = sequence.presentations().len();
        let mut entries: Vec<FrameIndexEntry> = Vec::new();
        for presentation in 0..count {
            if independent(sequence, presentation) {
                if entries.len() == limits.max_entries {
                    return Err(FrameIndexError::EntryLimit);
                }
                entries
                    .try_reserve(1)
                    .map_err(|_| FrameIndexError::Allocation)?;
                entries.push(FrameIndexEntry {
                    codestream_offset: offset(sequence, presentation)?,
                    duration_ticks: 0,
                    frames: 0,
                });
            }
            let entry = entries
                .last_mut()
                .ok_or(FrameIndexError::DependentAnchor { entry: 0 })?;
            entry.frames += 1;
            // The sequence already checked the cumulative u64 presentation clock.
            entry.duration_ticks +=
                u64::from(sequence.presentations()[presentation].duration_ticks);
        }
        let (numerator, denominator) = sequence.animation().map_or((1, 1), |a| {
            (a.ticks_per_second_denominator, a.ticks_per_second_numerator)
        });
        Self::new(
            numerator,
            NonZeroU32::new(denominator).expect("checked sequence timebase"),
            entries,
            limits,
        )
    }

    /// Checks exact offsets, complete displayed-frame spans, restart dependencies and rational
    /// durations. Returns the presentation position of each entry; limits are reapplied.
    pub fn bind_sequence(
        &self,
        sequence: &FrameSequencePlan,
        limits: FrameIndexLimits,
    ) -> Result<Vec<usize>, FrameIndexError> {
        check_frame_limit(sequence, limits)?;
        super::validate(&self.entries, limits)?;
        let mut anchors = Vec::new();
        anchors
            .try_reserve_exact(self.entries.len())
            .map_err(|_| FrameIndexError::Allocation)?;
        let mut presentation = 0_usize;
        for (entry, record) in self.entries.iter().enumerate() {
            let next = usize::try_from(record.frames)
                .ok()
                .and_then(|count| presentation.checked_add(count))
                .filter(|&end| end <= sequence.presentations().len())
                .ok_or(FrameIndexError::FrameCount)?;
            if offset(sequence, presentation)? != record.codestream_offset {
                return Err(FrameIndexError::Offset { entry });
            }
            if !independent(sequence, presentation) {
                return Err(FrameIndexError::DependentAnchor { entry });
            }
            let ticks = duration(sequence, presentation..next);
            let (tps_num, tps_den) = sequence.animation().map_or((1, 1), |a| {
                (a.ticks_per_second_numerator, a.ticks_per_second_denominator)
            });
            if u128::from(record.duration_ticks)
                * u128::from(self.tick_numerator())
                * u128::from(tps_num)
                != u128::from(ticks)
                    * u128::from(tps_den)
                    * u128::from(self.tick_denominator().get())
            {
                return Err(FrameIndexError::Duration { entry });
            }
            anchors.push(presentation);
            presentation = next;
        }
        if presentation != sequence.presentations().len() {
            return Err(FrameIndexError::FrameCount);
        }
        Ok(anchors)
    }
}

fn check_frame_limit(
    sequence: &FrameSequencePlan,
    limits: FrameIndexLimits,
) -> Result<(), FrameIndexError> {
    if !sequence.is_complete() {
        return Err(FrameIndexError::IncompleteSequence);
    }
    if sequence.presentations().len() as u64 > limits.max_frames {
        return Err(FrameIndexError::FrameLimit);
    }
    Ok(())
}

fn independent(sequence: &FrameSequencePlan, presentation: usize) -> bool {
    let span = &sequence.presentations()[presentation].physical_frames;
    sequence.earliest_dependencies()[span.clone()]
        .iter()
        .all(|&source| source >= span.start)
}

fn offset(sequence: &FrameSequencePlan, presentation: usize) -> Result<u64, FrameIndexError> {
    let bits = sequence
        .presentations()
        .get(presentation)
        .ok_or(FrameIndexError::FrameCount)?
        .header_bit_offset;
    if !bits.is_multiple_of(8) {
        return Err(FrameIndexError::UnalignedHeader);
    }
    Ok(bits / 8)
}

fn duration(sequence: &FrameSequencePlan, range: Range<usize>) -> u64 {
    sequence.presentations()[range]
        .iter()
        .map(|p| u64::from(p.duration_ticks))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnimationInventory, CodestreamInventory, FrameBlendMode, FrameSequenceEnd, FrameType,
    };

    // Header-only graph probe based on a parsed fixture. Offsets/timings below are an explicit
    // independent expectation; this does not claim to manufacture valid frame entropy.
    fn inventory() -> CodestreamInventory {
        let bytes = crate::test_fixtures::fragmented_animation();
        let mut inventory = crate::parse(&bytes, Default::default())
            .unwrap()
            .codestream_inventory(Default::default())
            .unwrap();
        inventory.image_header.animation = Some(AnimationInventory {
            ticks_per_second_numerator: 60_000,
            ticks_per_second_denominator: 1001,
            num_loops: 2,
            have_timecodes: false,
        });
        let prototype = inventory.frames[0].clone();
        inventory.frames = [
            (0, FrameBlendMode::Replace, 0, 0),
            (3, FrameBlendMode::Add, 0, 1),
            (5, FrameBlendMode::Replace, 0, 0),
            (7, FrameBlendMode::Add, 1, 2),
            (0, FrameBlendMode::Replace, 0, 0),
            (11, FrameBlendMode::Add, 0, 3),
            (0, FrameBlendMode::Replace, 0, 0),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (duration, mode, source, save))| {
            let mut frame = prototype.clone();
            frame.frame_index = 23 + i as u32;
            frame.frame_type = FrameType::Regular;
            frame.header_bits.offset = (100 + i as u64 * 24) * 8;
            frame.duration_ticks = duration;
            frame.timecode = None;
            frame.is_last = i == 6;
            frame.flags = 0;
            frame.lf_level = 0;
            frame.lf_source_frame = None;
            frame.x0 = 0;
            frame.y0 = 0;
            frame.width = inventory.image_header.width;
            frame.height = inventory.image_header.height;
            frame.have_crop = false;
            frame.save_as_reference = save;
            frame.save_before_color_transform = false;
            frame.color_blend.mode = mode;
            frame.color_blend.source = source;
            frame
        })
        .collect();
        inventory
    }

    #[test]
    fn indexes_count_presentations_keep_hidden_offsets_and_sum_exact_ticks() {
        let plan = FrameSequencePlan::negotiate(&inventory()).unwrap();
        assert_eq!(plan.earliest_dependencies(), [0, 0, 2, 0, 4, 4, 6]);
        let index = FrameIndex::from_sequence(&plan, Default::default()).unwrap();
        assert_eq!(index.tick_numerator(), 1001);
        assert_eq!(index.tick_denominator().get(), 60_000);
        let expected = [(100, 3, 1), (148, 12, 2), (196, 11, 1), (244, 0, 1)];
        assert_eq!(
            index.entries(),
            expected.map(|(codestream_offset, duration_ticks, frames)| {
                FrameIndexEntry {
                    codestream_offset,
                    duration_ticks,
                    frames,
                }
            })
        );
        assert_eq!(
            index.bind_sequence(&plan, Default::default()).unwrap(),
            [0, 1, 3, 4]
        );
        for (limits, error) in [
            (
                FrameIndexLimits {
                    max_entries: 3,
                    ..Default::default()
                },
                FrameIndexError::EntryLimit,
            ),
            (
                FrameIndexLimits {
                    max_frames: 4,
                    ..Default::default()
                },
                FrameIndexError::FrameLimit,
            ),
            (
                FrameIndexLimits {
                    max_payload_bytes: 8,
                    ..Default::default()
                },
                FrameIndexError::PayloadLimit,
            ),
        ] {
            assert_eq!(FrameIndex::from_sequence(&plan, limits), Err(error.clone()));
            assert_eq!(index.bind_sequence(&plan, limits), Err(error));
        }
    }

    #[test]
    fn unaligned_headers_and_incomplete_intervals_cannot_authorize_indexes() {
        let inventory = inventory();
        let complete = FrameSequencePlan::negotiate(&inventory).unwrap();
        let index = FrameIndex::from_sequence(&complete, Default::default()).unwrap();
        let interval =
            FrameSequencePlan::negotiate_interval(&inventory, FrameSequenceEnd::Presentation)
                .unwrap();
        assert_eq!(
            FrameIndex::from_sequence(&interval, Default::default()),
            Err(FrameIndexError::IncompleteSequence)
        );
        assert_eq!(
            index.bind_sequence(&interval, Default::default()),
            Err(FrameIndexError::IncompleteSequence)
        );
        let mut unaligned = inventory;
        unaligned.frames[0].header_bits.offset += 1;
        let plan = FrameSequencePlan::negotiate(&unaligned).unwrap();
        assert_eq!(
            FrameIndex::from_sequence(&plan, Default::default()),
            Err(FrameIndexError::UnalignedHeader)
        );
        assert_eq!(
            index.bind_sequence(&plan, Default::default()),
            Err(FrameIndexError::UnalignedHeader)
        );
    }
}
