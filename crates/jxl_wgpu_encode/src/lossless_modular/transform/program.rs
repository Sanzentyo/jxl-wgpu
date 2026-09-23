//! Ordered forward Squeeze operations and their disjoint, lifetime-colored GPU arena.
use super::{ChannelRange, PlannedChannel, SampleSource, SqueezeStep};
use crate::{EncodeError, LosslessModularSqueezeStep};

/// Offsets are relative to the program's sample arena, never host byte pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::lossless_modular) struct SqueezeJob {
    source: u32,
    source_offset: u32,
    width: u32,
    height: u32,
    average_offset: u32,
    residual_offset: u32,
    horizontal: u32,
    padding: u32,
}

const _: () = {
    assert!(std::mem::size_of::<SqueezeJob>() == 32);
    assert!(std::mem::align_of::<SqueezeJob>() == 4);
};

#[derive(Clone, Debug)]
pub(in crate::lossless_modular) struct SqueezeProgram {
    pub(in crate::lossless_modular) jobs: Vec<SqueezeJob>,
    pub(in crate::lossless_modular) arena_words: u32,
}

impl SqueezeProgram {
    pub(super) fn new(
        channels: &mut Vec<PlannedChannel>,
        meta: usize,
        steps: &[LosslessModularSqueezeStep],
    ) -> Result<(Self, Vec<SqueezeStep>), EncodeError> {
        let mut arena = Arena::default();
        let mut jobs = Vec::new();
        let mut wire = Vec::with_capacity(steps.len());
        for (step_index, step) in steps.iter().enumerate() {
            let image_channels = channels.len() - meta;
            let begin = step.begin as usize;
            let end = begin + step.count as usize;
            if end > image_channels {
                return Err(EncodeError::InvalidModularSqueezeChannels {
                    begin: step.begin,
                    count: step.count,
                    channels: image_channels as u32,
                });
            }
            let mut residuals = Vec::with_capacity(step.count as usize);
            for (index, channel) in channels[meta + begin..meta + end].iter_mut().enumerate() {
                let [width, height] = channel.extent;
                if width == 0 || height == 0 {
                    return Err(EncodeError::EmptyModularSqueezeChannel {
                        step: step_index as u32,
                        channel: (begin + index) as u32,
                    });
                }
                if channel.shifts.iter().any(|shift| *shift > 30) {
                    return Err(EncodeError::ModularSqueezeShiftLimit {
                        step: step_index as u32,
                        channel: (begin + index) as u32,
                        horizontal: channel.shifts[0],
                        vertical: channel.shifts[1],
                    });
                }
                let pixels = width * height; // source pass-group area bounds every descendant
                let offset = arena.allocate(pixels)?;
                let axis = usize::from(!step.horizontal);
                let mut average = *channel;
                let mut residual = *channel;
                average.extent[axis] = channel.extent[axis].div_ceil(2);
                residual.extent[axis] = channel.extent[axis] / 2;
                average.shifts[axis] += 1;
                residual.shifts[axis] += 1;
                let residual_offset = offset + average.extent[0] * average.extent[1];
                jobs.push(SqueezeJob {
                    source: channel.source.kernel_value(),
                    source_offset: match channel.source {
                        SampleSource::Arena(value) => value,
                        _ => 0,
                    },
                    width,
                    height,
                    average_offset: offset,
                    residual_offset,
                    horizontal: u32::from(step.horizontal),
                    padding: 0,
                });
                // Both new views must be disjoint from the input until this GPU operation ends.
                if let SampleSource::Arena(old) = channel.source {
                    arena.release(old..old + pixels);
                }
                average.source = SampleSource::Arena(offset);
                residual.source = SampleSource::Arena(residual_offset);
                *channel = average;
                residuals.push(residual);
            }
            if step.in_place {
                channels.splice(meta + end..meta + end, residuals);
            } else {
                channels.extend(residuals);
            }
            wire.push(SqueezeStep {
                horizontal: step.horizontal,
                in_place: step.in_place,
                range: ChannelRange {
                    begin: (meta + begin) as u32,
                    count: step.count,
                },
            });
        }
        Ok((
            Self {
                jobs,
                arena_words: arena.high_water,
            },
            wire,
        ))
    }

    pub(in crate::lossless_modular) fn metadata_words(&self) -> u64 {
        1 + self.jobs.len() as u64 * 8
    }
    pub(in crate::lossless_modular) fn scratch_bytes(&self) -> u64 {
        4 * (self.metadata_words() + u64::from(self.arena_words))
    }
    pub(in crate::lossless_modular) fn metadata(&self) -> Vec<u32> {
        let mut words = Vec::with_capacity(self.metadata_words() as usize);
        words.push(self.jobs.len() as u32);
        words.extend_from_slice(bytemuck::cast_slice(&self.jobs));
        words
    }
}

#[derive(Default)]
struct Arena {
    high_water: u32,
    free: Vec<std::ops::Range<u32>>,
}

impl Arena {
    fn allocate(&mut self, words: u32) -> Result<u32, EncodeError> {
        if let Some((index, _)) = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, span)| span.end - span.start >= words)
            .min_by_key(|(_, span)| span.end - span.start)
        {
            let start = self.free[index].start;
            self.free[index].start += words;
            if self.free[index].is_empty() {
                self.free.remove(index);
            }
            return Ok(start);
        }
        let start = self.high_water;
        self.high_water = start.checked_add(words).ok_or(EncodeError::InvalidSource(
            "Squeeze arena exceeds WGSL indexing",
        ))?;
        Ok(start)
    }

    fn release(&mut self, span: std::ops::Range<u32>) {
        self.free.push(span);
        self.free.sort_unstable_by_key(|span| span.start);
        let mut index = 1;
        while index < self.free.len() {
            if self.free[index - 1].end == self.free[index].start {
                self.free[index - 1].end = self.free[index].end;
                self.free.remove(index);
            } else {
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LosslessModularSqueeze;
    use crate::lossless_modular::squeeze::SqueezeAxes;
    use crate::lossless_modular::transform::WorkingComponent;

    #[test]
    fn ordered_jobs_keep_live_views_disjoint_and_reuse_consumed_spans() {
        let mut channels: Vec<_> = (0..4)
            .map(|component| PlannedChannel {
                extent: [513, 257],
                source: SampleSource::Component(WorkingComponent(component)),
                shifts: [0; 2],
                squeeze: SqueezeAxes::None,
                band: 0,
            })
            .collect();
        let steps: Vec<_> = (0..18)
            .map(|step| LosslessModularSqueezeStep::new(step % 2 == 0, 0, 4, false).unwrap())
            .collect();
        let (program, wire) = SqueezeProgram::new(&mut channels, 0, &steps).unwrap();
        assert_eq!(wire.len(), 18);
        assert_eq!(channels.len(), 76);
        let mut live = Vec::<std::ops::Range<u32>>::new();
        let mut total_allocated = 0;
        for job in &program.jobs {
            let pixels = job.width * job.height;
            let output = job.average_offset..job.average_offset + pixels;
            assert!(
                live.iter()
                    .all(|span| span.end <= output.start || output.end <= span.start)
            );
            assert!(output.end <= program.arena_words);
            if job.source == 6 {
                let source = job.source_offset..job.source_offset + pixels;
                let index = live.iter().position(|span| *span == source).unwrap();
                live.remove(index);
            }
            for span in [
                output.start..job.residual_offset,
                job.residual_offset..output.end,
            ] {
                if !span.is_empty() {
                    live.push(span);
                }
            }
            total_allocated += pixels;
        }
        assert!(program.arena_words < total_allocated);
        assert!(program.arena_words <= 2 * 4 * 513 * 257);
        assert_eq!(
            live.iter().map(|span| span.end - span.start).sum::<u32>(),
            4 * 513 * 257
        );
        for channel in channels {
            if channel.extent[0] * channel.extent[1] == 0 {
                continue;
            }
            let SampleSource::Arena(offset) = channel.source else {
                panic!("missing arena view");
            };
            assert!(live.contains(&(offset..offset + channel.extent[0] * channel.extent[1])));
        }
        let job = SqueezeJob {
            source: 1,
            source_offset: 2,
            width: 3,
            height: 4,
            average_offset: 5,
            residual_offset: 6,
            horizontal: 7,
            padding: 8,
        };
        assert_eq!(bytemuck::cast::<_, [u32; 8]>(job), [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn sequence_limits_and_empty_intermediate_inputs_are_typed() {
        let step = LosslessModularSqueezeStep::new(true, 0, 1, false).unwrap();
        for count in [0, 297] {
            assert!(
                matches!(LosslessModularSqueeze::sequence(vec![step; count]), Err(EncodeError::InvalidModularSqueezeStepCount { count: actual }) if actual == count)
            );
        }
        for (begin, count) in [(9288, 1), (0, 0), (0, 20), (u32::MAX, 1)] {
            assert!(matches!(
                LosslessModularSqueezeStep::new(false, begin, count, false),
                Err(EncodeError::InvalidModularSqueezeStep { .. })
            ));
        }
        for count in [1, 16, 17, 72, 73, 296] {
            let policy = LosslessModularSqueeze::sequence(vec![step; count]).unwrap();
            let changed = policy.clone().with_in_place(true);
            assert!(policy.steps().unwrap().iter().all(|step| !step.in_place()));
            assert!(changed.steps().unwrap().iter().all(|step| step.in_place()));
            assert!(policy.with_channels(0, 1).is_err());
        }
        let mut channels = vec![PlannedChannel {
            extent: [1, 3],
            source: SampleSource::Component(WorkingComponent(0)),
            shifts: [0; 2],
            squeeze: SqueezeAxes::None,
            band: 0,
        }];
        assert!(matches!(
            SqueezeProgram::new(
                &mut channels,
                0,
                &[
                    step,
                    LosslessModularSqueezeStep::new(true, 1, 1, true).unwrap()
                ]
            ),
            Err(EncodeError::EmptyModularSqueezeChannel {
                step: 1,
                channel: 1
            })
        ));
    }
}
