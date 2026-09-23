//! Ordered forward transforms and their disjoint, lifetime-colored GPU arena.
use super::{
    ChannelRange, PlannedChannel, PlannedRct, SampleSource, SqueezeStep, TransformOperation,
};
use crate::{
    EncodeError, LosslessModularRctType, LosslessModularSqueezeStep, LosslessModularTransform,
};

/// Offsets are relative to the program's sample arena, never host byte pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(in crate::lossless_modular) struct TransformJob {
    operation: u32,
    mode: u32,
    width: u32,
    height: u32,
    sources: [u32; 3],
    source_offsets: [u32; 3],
    output_offsets: [u32; 3],
    padding: [u32; 3],
}

const _: () = {
    assert!(std::mem::size_of::<TransformJob>() == 64);
    assert!(std::mem::align_of::<TransformJob>() == 4);
};

#[derive(Clone, Debug)]
pub(in crate::lossless_modular) struct TransformProgram {
    pub(in crate::lossless_modular) jobs: Vec<TransformJob>,
    pub(in crate::lossless_modular) arena_words: u32,
}

impl TransformProgram {
    pub(super) fn squeeze(
        channels: &mut Vec<PlannedChannel>,
        meta: usize,
        steps: &[LosslessModularSqueezeStep],
    ) -> Result<(Self, Vec<SqueezeStep>), EncodeError> {
        let mut builder = Builder::default();
        let wire = steps
            .iter()
            .enumerate()
            .map(|(index, step)| builder.squeeze(channels, meta, *step, index))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((builder.finish(), wire))
    }

    pub(super) fn new(
        channels: &mut Vec<PlannedChannel>,
        meta: usize,
        operations: &[LosslessModularTransform],
    ) -> Result<(Self, Vec<TransformOperation>), EncodeError> {
        let mut builder = Builder::default();
        let mut wire = Vec::with_capacity(operations.len());
        for (index, operation) in operations.iter().enumerate() {
            wire.push(match *operation {
                LosslessModularTransform::Squeeze(step) => {
                    TransformOperation::Squeeze(vec![builder.squeeze(channels, meta, step, index)?])
                }
                LosslessModularTransform::Rct {
                    begin_channel,
                    rct_type,
                } => TransformOperation::Rct(builder.rct(
                    channels,
                    meta,
                    begin_channel,
                    rct_type,
                    index,
                )?),
            });
        }
        Ok((builder.finish(), wire))
    }

    pub(in crate::lossless_modular) fn metadata_words(&self) -> u64 {
        1 + self.jobs.len() as u64 * 16
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
struct Builder {
    arena: Arena,
    jobs: Vec<TransformJob>,
}

impl Builder {
    fn finish(self) -> TransformProgram {
        TransformProgram {
            jobs: self.jobs,
            arena_words: self.arena.high_water,
        }
    }

    fn rct(
        &mut self,
        channels: &mut [PlannedChannel],
        meta: usize,
        begin: u32,
        rct_type: LosslessModularRctType,
        index: usize,
    ) -> Result<PlannedRct, EncodeError> {
        let image_channels = channels.len() - meta;
        let range = meta + begin as usize..meta + begin as usize + 3;
        let selected = channels
            .get_mut(range)
            .ok_or(EncodeError::InvalidModularRctChannels {
                operation: index as u32,
                begin,
                channels: image_channels as u32,
            })?;
        let first = selected[0];
        if selected
            .iter()
            .any(|channel| channel.extent != first.extent || channel.shifts != first.shifts)
        {
            return Err(EncodeError::UnequalModularRctChannels {
                operation: index as u32,
                begin,
            });
        }
        let pixels = first.extent[0] * first.extent[1];
        // Empty residual channels still participate in the wire topology, but have no samples.
        if pixels != 0 {
            let offset = self.arena.allocate(3 * pixels)?;
            let mut job = TransformJob {
                operation: 1,
                mode: rct_type.value(),
                width: first.extent[0],
                height: first.extent[1],
                sources: [0; 3],
                source_offsets: [0; 3],
                output_offsets: [offset, offset + pixels, offset + 2 * pixels],
                padding: [0; 3],
            };
            for (component, channel) in selected.iter_mut().enumerate() {
                job.sources[component] = channel.source.kernel_value();
                if let SampleSource::Arena(old) = channel.source {
                    job.source_offsets[component] = old;
                    self.arena.release(old..old + pixels);
                }
                channel.source = SampleSource::Arena(job.output_offsets[component]);
            }
            self.jobs.push(job);
        }
        Ok(PlannedRct {
            begin: meta as u32 + begin,
            rct_type,
        })
    }

    fn squeeze(
        &mut self,
        channels: &mut Vec<PlannedChannel>,
        meta: usize,
        step: LosslessModularSqueezeStep,
        step_index: usize,
    ) -> Result<SqueezeStep, EncodeError> {
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
            let offset = self.arena.allocate(pixels)?;
            let axis = usize::from(!step.horizontal);
            let mut average = *channel;
            let mut residual = *channel;
            average.extent[axis] = channel.extent[axis].div_ceil(2);
            residual.extent[axis] = channel.extent[axis] / 2;
            average.shifts[axis] += 1;
            residual.shifts[axis] += 1;
            let residual_offset = offset + average.extent[0] * average.extent[1];
            self.jobs.push(TransformJob {
                operation: 0,
                mode: u32::from(step.horizontal),
                sources: [channel.source.kernel_value(), 0, 0],
                source_offsets: [
                    match channel.source {
                        SampleSource::Arena(value) => value,
                        _ => 0,
                    },
                    0,
                    0,
                ],
                width,
                height,
                output_offsets: [offset, residual_offset, 0],
                padding: [0; 3],
            });
            // Both new views must be disjoint from the input until this GPU operation ends.
            if let SampleSource::Arena(old) = channel.source {
                self.arena.release(old..old + pixels);
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
        Ok(SqueezeStep {
            horizontal: step.horizontal,
            in_place: step.in_place,
            range: ChannelRange {
                begin: (meta + begin) as u32,
                count: step.count,
            },
        })
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
            "Modular transform arena exceeds WGSL indexing",
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
    fn empty_rct_keeps_the_wire_operation_without_an_arena_job() {
        let mut channels: Vec<_> = (0..3)
            .map(|component| PlannedChannel {
                extent: [1, 1],
                source: SampleSource::Component(WorkingComponent(component)),
                shifts: [0; 2],
                squeeze: SqueezeAxes::None,
                band: 0,
            })
            .collect();
        let operations = [
            LosslessModularTransform::Squeeze(
                LosslessModularSqueezeStep::new(true, 0, 3, false).unwrap(),
            ),
            LosslessModularTransform::Rct {
                begin_channel: 3,
                rct_type: LosslessModularRctType::YCOCG,
            },
        ];
        let (program, wire) = TransformProgram::new(&mut channels, 0, &operations).unwrap();
        assert_eq!(wire.len(), 2);
        assert_eq!(program.jobs.len(), 3);
        assert!(program.jobs.iter().all(|job| job.operation == 0));
        assert_eq!(program.arena_words, 3);
        assert!(
            channels[3..]
                .iter()
                .all(|channel| channel.extent == [0, 1] && channel.shifts == [1, 0])
        );
    }

    #[test]
    fn mixed_jobs_allocate_outputs_before_retiring_any_rct_input() {
        let mut channels: Vec<_> = (0..4)
            .map(|component| PlannedChannel {
                extent: [64, 32],
                source: SampleSource::Component(WorkingComponent(component)),
                shifts: [0; 2],
                squeeze: SqueezeAxes::None,
                band: 0,
            })
            .collect();
        let mut operations = Vec::new();
        for level in 0..5 {
            operations.push(LosslessModularTransform::Rct {
                begin_channel: 1,
                rct_type: LosslessModularRctType::new(level * 7).unwrap(),
            });
            operations.push(LosslessModularTransform::Squeeze(
                LosslessModularSqueezeStep::new(level % 2 == 0, 1, 3, false).unwrap(),
            ));
        }
        let (program, wire) = TransformProgram::new(&mut channels, 0, &operations).unwrap();
        assert_eq!(wire.len(), 10);
        let mut live = Vec::<std::ops::Range<u32>>::new();
        let mut total_allocated = 0;
        for job in &program.jobs {
            let pixels = job.width * job.height;
            let outputs = if job.operation == 1 {
                job.output_offsets
                    .map(|offset| offset..offset + pixels)
                    .to_vec()
            } else {
                vec![
                    job.output_offsets[0]..job.output_offsets[1],
                    job.output_offsets[1]..job.output_offsets[0] + pixels,
                ]
            };
            for (index, output) in outputs.iter().enumerate() {
                assert!(output.end <= program.arena_words);
                assert!(
                    live.iter()
                        .chain(&outputs[..index])
                        .all(|span| span.end <= output.start || output.end <= span.start)
                );
                total_allocated += output.end - output.start;
            }
            for component in 0..if job.operation == 1 { 3 } else { 1 } {
                if job.sources[component] == 6 {
                    let source =
                        job.source_offsets[component]..job.source_offsets[component] + pixels;
                    live.remove(live.iter().position(|span| *span == source).unwrap());
                }
            }
            live.extend(outputs);
        }
        assert!(program.arena_words < total_allocated);
        assert_eq!(
            live.iter().map(|span| span.end - span.start).sum::<u32>(),
            3 * 64 * 32
        );
        for channel in &channels[1..] {
            let SampleSource::Arena(offset) = channel.source else {
                panic!("missing arena view")
            };
            assert!(live.contains(&(offset..offset + channel.extent[0] * channel.extent[1])));
        }
    }

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
        let (program, wire) = TransformProgram::squeeze(&mut channels, 0, &steps).unwrap();
        assert_eq!(wire.len(), 18);
        assert_eq!(channels.len(), 76);
        let mut live = Vec::<std::ops::Range<u32>>::new();
        let mut total_allocated = 0;
        for job in &program.jobs {
            let pixels = job.width * job.height;
            let output = job.output_offsets[0]..job.output_offsets[0] + pixels;
            assert!(
                live.iter()
                    .all(|span| span.end <= output.start || output.end <= span.start)
            );
            assert!(output.end <= program.arena_words);
            if job.sources[0] == 6 {
                let source = job.source_offsets[0]..job.source_offsets[0] + pixels;
                let index = live.iter().position(|span| *span == source).unwrap();
                live.remove(index);
            }
            for span in [
                output.start..job.output_offsets[1],
                job.output_offsets[1]..output.end,
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
        let job = TransformJob {
            operation: 1,
            mode: 2,
            width: 3,
            height: 4,
            sources: [5, 6, 7],
            source_offsets: [8, 9, 10],
            output_offsets: [11, 12, 13],
            padding: [14, 15, 16],
        };
        assert_eq!(
            bytemuck::cast::<_, [u32; 16]>(job),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
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
            TransformProgram::squeeze(
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
