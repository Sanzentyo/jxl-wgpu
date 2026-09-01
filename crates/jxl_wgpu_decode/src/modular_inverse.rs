//! Lifetime planning and command recording for GPU-resident Modular inverse transforms.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    ModularInversePlanError, ModularTransformFeature, Result,
    modular_rct::{ModularRctParams, ModularRctPlane},
    modular_squeeze::{ModularSqueezeDirection, ModularSqueezeParams, ModularSqueezePlane},
    modular_transform::{
        ModularChannelGeometry, ModularChannelTopology, ModularInverseTransform, ModularRct,
        ModularSqueezeParameter, ModularTransformPlan,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WordSpan {
    offset: u32,
    length: u32,
}

impl WordSpan {
    const fn end(self) -> Option<u32> {
        self.offset.checked_add(self.length)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModularArenaPlane {
    pub geometry: ModularChannelGeometry,
    pub offset_words: u32,
}

impl ModularArenaPlane {
    fn span(self) -> Result<WordSpan> {
        let length =
            u32::try_from(u64::from(self.geometry.width) * u64::from(self.geometry.height))
                .map_err(|_| ModularInversePlanError::ArenaAddressSpace)?;
        Ok(WordSpan {
            offset: self.offset_words,
            length,
        })
    }

    const fn squeeze_view(self) -> ModularSqueezePlane {
        ModularSqueezePlane {
            width: self.geometry.width,
            height: self.geometry.height,
            stride: self.geometry.width,
            offset_words: self.offset_words,
        }
    }

    const fn rct_view(self) -> ModularRctPlane {
        ModularRctPlane {
            width: self.geometry.width,
            height: self.geometry.height,
            stride: self.geometry.width,
            offset_words: self.offset_words,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModularInverseJob {
    Squeeze { params: ModularSqueezeParams },
    Rct { params: ModularRctParams },
}

/// Complete reverse schedule over one storage arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModularInversePlan {
    entropy_words: u32,
    arena_words: u32,
    jobs: Vec<ModularInverseJob>,
    final_planes: Vec<ModularArenaPlane>,
}

impl ModularInversePlan {
    pub(crate) const fn entropy_words(&self) -> u32 {
        self.entropy_words
    }

    pub(crate) const fn arena_words(&self) -> u32 {
        self.arena_words
    }

    pub(crate) const fn arena_bytes(&self) -> u64 {
        self.arena_words as u64 * 4
    }

    pub(crate) fn jobs(&self) -> &[ModularInverseJob] {
        &self.jobs
    }

    pub(crate) fn final_planes(&self) -> &[ModularArenaPlane] {
        &self.final_planes
    }
}

/// Lowers RCT/Squeeze operations into a resident lifetime schedule.
pub(crate) fn plan_modular_inverse(
    transform_plan: &ModularTransformPlan,
) -> Result<ModularInversePlan> {
    let initial_layout = transform_plan.topology.gpu_layout()?;
    let mut live = transform_plan
        .topology
        .channels()
        .iter()
        .copied()
        .zip(initial_layout)
        .map(|(geometry, layout)| ModularArenaPlane {
            geometry,
            offset_words: layout.word_offset,
        })
        .collect::<Vec<_>>();
    let entropy_words = topology_words(&transform_plan.topology)?;
    let mut allocator = WordAllocator::new(entropy_words);
    let mut jobs = Vec::new();

    transform_plan.visit_inverse(|operation, source, destination| {
        ensure_live_topology(&live, source)?;
        match operation {
            ModularInverseTransform::Squeeze(parameter) => lower_squeeze(
                parameter,
                source,
                destination,
                &mut live,
                &mut allocator,
                &mut jobs,
            )?,
            ModularInverseTransform::Rct(rct) => lower_rct(rct, &live, &mut jobs)?,
            ModularInverseTransform::Palette(_) => {
                return Err(ModularInversePlanError::UnsupportedTransform {
                    feature: ModularTransformFeature::Palette,
                }
                .into());
            }
        }
        ensure_live_topology(&live, destination)
    })?;

    ensure_live_topology(&live, transform_plan.source_topology())?;
    Ok(ModularInversePlan {
        entropy_words,
        arena_words: allocator.high_water,
        jobs,
        final_planes: live,
    })
}

fn lower_squeeze(
    parameter: ModularSqueezeParameter,
    source: &ModularChannelTopology,
    destination: &ModularChannelTopology,
    live: &mut Vec<ModularArenaPlane>,
    allocator: &mut WordAllocator,
    jobs: &mut Vec<ModularInverseJob>,
) -> Result<()> {
    let average_start = usize::try_from(parameter.begin_channel).map_err(|_| {
        ModularInversePlanError::TopologyState {
            reason: "Squeeze begin channel exceeds host space",
        }
    })?;
    let channel_count = usize::try_from(parameter.channel_count).map_err(|_| {
        ModularInversePlanError::TopologyState {
            reason: "Squeeze channel count exceeds host space",
        }
    })?;
    let average_end =
        average_start
            .checked_add(channel_count)
            .ok_or(ModularInversePlanError::TopologyState {
                reason: "Squeeze average range overflows",
            })?;
    let residual_start = if parameter.in_place {
        average_end
    } else {
        live.len()
            .checked_sub(channel_count)
            .ok_or(ModularInversePlanError::TopologyState {
                reason: "Squeeze residual count exceeds live planes",
            })?
    };
    let residual_end = residual_start.checked_add(channel_count).ok_or(
        ModularInversePlanError::TopologyState {
            reason: "Squeeze residual range overflows",
        },
    )?;
    if average_end > live.len()
        || residual_start < average_end
        || residual_end > live.len()
        || destination.channels().len() + channel_count != source.channels().len()
    {
        return Err(ModularInversePlanError::TopologyState {
            reason: "Squeeze ranges do not match inverse topology",
        }
        .into());
    }

    for index in 0..channel_count {
        let average_index = average_start + index;
        let residual_index = residual_start + index;
        let average = live[average_index];
        let residual = live[residual_index];
        let geometry = destination.channels()[average_index];
        let output_span = allocator.allocate(geometry_words(geometry)?)?;
        let output = ModularArenaPlane {
            geometry,
            offset_words: output_span.offset,
        };
        let direction = if parameter.horizontal {
            ModularSqueezeDirection::Horizontal
        } else {
            ModularSqueezeDirection::Vertical
        };
        jobs.push(ModularInverseJob::Squeeze {
            params: ModularSqueezeParams::new(
                direction,
                average.squeeze_view(),
                residual.squeeze_view(),
                output.squeeze_view(),
            ),
        });
        live[average_index] = output;
        allocator.release(average.span()?)?;
        allocator.release(residual.span()?)?;
    }
    live.drain(residual_start..residual_end);
    Ok(())
}

fn lower_rct(
    rct: ModularRct,
    live: &[ModularArenaPlane],
    jobs: &mut Vec<ModularInverseJob>,
) -> Result<()> {
    let begin =
        usize::try_from(rct.begin_channel).map_err(|_| ModularInversePlanError::TopologyState {
            reason: "RCT begin channel exceeds host space",
        })?;
    let end = begin
        .checked_add(3)
        .ok_or(ModularInversePlanError::TopologyState {
            reason: "RCT channel range overflows",
        })?;
    let planes = live
        .get(begin..end)
        .ok_or(ModularInversePlanError::TopologyState {
            reason: "RCT channel range exceeds live planes",
        })?;
    let [first, second, third] = planes else {
        return Err(ModularInversePlanError::TopologyState {
            reason: "RCT did not resolve exactly three live planes",
        }
        .into());
    };
    jobs.push(ModularInverseJob::Rct {
        params: ModularRctParams::new(
            rct.rct_type,
            first.rct_view(),
            second.rct_view(),
            third.rct_view(),
        ),
    });
    Ok(())
}

fn ensure_live_topology(
    live: &[ModularArenaPlane],
    topology: &ModularChannelTopology,
) -> Result<()> {
    if live.len() != topology.channels().len()
        || live
            .iter()
            .zip(topology.channels())
            .any(|(plane, geometry)| plane.geometry != *geometry)
    {
        return Err(ModularInversePlanError::TopologyState {
            reason: "live plane geometry differs from transform topology",
        }
        .into());
    }
    Ok(())
}

fn geometry_words(geometry: ModularChannelGeometry) -> Result<u32> {
    u32::try_from(u64::from(geometry.width) * u64::from(geometry.height))
        .map_err(|_| ModularInversePlanError::ArenaAddressSpace.into())
}

fn topology_words(topology: &ModularChannelTopology) -> Result<u32> {
    u32::try_from(
        topology
            .sample_count()
            .ok_or(ModularInversePlanError::ArenaAddressSpace)?,
    )
    .map_err(|_| ModularInversePlanError::ArenaAddressSpace.into())
}

#[derive(Debug)]
struct WordAllocator {
    high_water: u32,
    free_by_offset: BTreeMap<u32, u32>,
    free_by_size: BTreeSet<(u32, u32)>,
}

impl WordAllocator {
    fn new(high_water: u32) -> Self {
        Self {
            high_water,
            free_by_offset: BTreeMap::new(),
            free_by_size: BTreeSet::new(),
        }
    }

    fn allocate(&mut self, length: u32) -> Result<WordSpan> {
        if length == 0 {
            return Ok(WordSpan {
                offset: self.high_water,
                length: 0,
            });
        }
        if let Some((available, offset)) = self.free_by_size.range((length, 0)..).next().copied() {
            self.remove_free(offset, available);
            if available > length {
                self.insert_free(WordSpan {
                    offset: offset
                        .checked_add(length)
                        .ok_or(ModularInversePlanError::ArenaAddressSpace)?,
                    length: available - length,
                })?;
            }
            return Ok(WordSpan { offset, length });
        }
        let offset = self.high_water;
        self.high_water = self
            .high_water
            .checked_add(length)
            .ok_or(ModularInversePlanError::ArenaAddressSpace)?;
        Ok(WordSpan { offset, length })
    }

    fn release(&mut self, span: WordSpan) -> Result<()> {
        if span.length == 0 {
            return Ok(());
        }
        self.insert_free(span)
    }

    fn insert_free(&mut self, span: WordSpan) -> Result<()> {
        let mut offset = span.offset;
        let mut end = span
            .end()
            .ok_or(ModularInversePlanError::ArenaAddressSpace)?;
        if let Some((&previous_offset, &previous_length)) =
            self.free_by_offset.range(..=offset).next_back()
        {
            let previous_end = previous_offset
                .checked_add(previous_length)
                .ok_or(ModularInversePlanError::ArenaAddressSpace)?;
            if previous_end > offset {
                return Err(ModularInversePlanError::FreeListOverlap.into());
            }
            if previous_end == offset {
                self.remove_free(previous_offset, previous_length);
                offset = previous_offset;
            }
        }
        if let Some((&next_offset, &next_length)) = self.free_by_offset.range(offset..).next() {
            if next_offset < end {
                return Err(ModularInversePlanError::FreeListOverlap.into());
            }
            if next_offset == end {
                self.remove_free(next_offset, next_length);
                end = next_offset
                    .checked_add(next_length)
                    .ok_or(ModularInversePlanError::ArenaAddressSpace)?;
            }
        }
        let length = end
            .checked_sub(offset)
            .ok_or(ModularInversePlanError::FreeListOverlap)?;
        self.free_by_offset.insert(offset, length);
        self.free_by_size.insert((length, offset));
        Ok(())
    }

    fn remove_free(&mut self, offset: u32, length: u32) {
        self.free_by_offset.remove(&offset);
        self.free_by_size.remove(&(length, offset));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::modular_rct::{ModularRctArena, ModularRctPipeline};
    #[cfg(not(target_arch = "wasm32"))]
    use crate::modular_squeeze::{ModularSqueezeArena, ModularSqueezePipeline};
    use crate::modular_transform::{
        ModularChannelTopology, ModularRct, ModularSqueezeParameter, ModularTransformIr,
        ModularTransformLimits, ModularTransformPlan,
    };

    fn nested_squeeze_plan() -> ModularInversePlan {
        let limits = ModularTransformLimits::default();
        let topology = ModularChannelTopology::full_resolution(9, 5, 8, 1, limits).unwrap();
        let transform = ModularTransformPlan::squeeze_only_for_test(
            topology,
            vec![
                ModularSqueezeParameter {
                    horizontal: true,
                    in_place: true,
                    begin_channel: 0,
                    channel_count: 1,
                },
                ModularSqueezeParameter {
                    horizontal: false,
                    in_place: true,
                    begin_channel: 0,
                    channel_count: 2,
                },
            ],
            limits,
        )
        .unwrap();
        plan_modular_inverse(&transform).unwrap()
    }

    fn one_squeeze_plan(
        width: u32,
        height: u32,
        channel_count: u32,
        parameter: ModularSqueezeParameter,
    ) -> ModularInversePlan {
        let limits = ModularTransformLimits::default();
        let topology =
            ModularChannelTopology::full_resolution(width, height, 8, channel_count, limits)
                .unwrap();
        let transform =
            ModularTransformPlan::squeeze_only_for_test(topology, vec![parameter], limits).unwrap();
        plan_modular_inverse(&transform).unwrap()
    }

    fn mixed_rct_squeeze_plan() -> ModularInversePlan {
        let limits = ModularTransformLimits::default();
        let topology = ModularChannelTopology::full_resolution(9, 5, 8, 3, limits).unwrap();
        let transform = ModularTransformPlan::from_transforms_for_test(
            topology,
            vec![
                ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type: 5,
                }),
                ModularTransformIr::Squeeze {
                    used_default_parameters: false,
                    parameters: vec![ModularSqueezeParameter {
                        horizontal: true,
                        in_place: true,
                        begin_channel: 0,
                        channel_count: 3,
                    }],
                },
                ModularTransformIr::Rct(ModularRct {
                    begin_channel: 0,
                    rct_type: 41,
                }),
            ],
            limits,
        )
        .unwrap();
        plan_modular_inverse(&transform).unwrap()
    }

    #[test]
    fn best_fit_allocator_merges_both_neighbors_and_rejects_overlap() {
        let mut allocator = WordAllocator::new(100);
        allocator
            .release(WordSpan {
                offset: 20,
                length: 10,
            })
            .unwrap();
        allocator
            .release(WordSpan {
                offset: 40,
                length: 10,
            })
            .unwrap();
        allocator
            .release(WordSpan {
                offset: 30,
                length: 10,
            })
            .unwrap();
        assert_eq!(allocator.free_by_offset, BTreeMap::from([(20, 30)]));
        assert_eq!(
            allocator.allocate(24).unwrap(),
            WordSpan {
                offset: 20,
                length: 24
            }
        );
        assert_eq!(allocator.free_by_offset, BTreeMap::from([(44, 6)]));
        assert!(matches!(
            allocator
                .release(WordSpan {
                    offset: 43,
                    length: 2,
                })
                .unwrap_err(),
            crate::Error::ModularInversePlan(ModularInversePlanError::FreeListOverlap)
        ));
    }

    #[test]
    fn nested_squeeze_schedule_reuses_retired_entropy_ranges() {
        let plan = nested_squeeze_plan();
        assert_eq!(plan.entropy_words(), 45);
        assert_eq!(plan.arena_words(), 90);
        assert_eq!(plan.jobs().len(), 3);
        assert_eq!(
            plan.jobs()
                .iter()
                .map(|job| match job {
                    ModularInverseJob::Squeeze { params } => params.direction(),
                    ModularInverseJob::Rct { .. } => panic!("Squeeze-only plan contains RCT"),
                })
                .collect::<Vec<_>>(),
            vec![
                ModularSqueezeDirection::Vertical,
                ModularSqueezeDirection::Vertical,
                ModularSqueezeDirection::Horizontal,
            ]
        );
        assert_eq!(plan.final_planes().len(), 1);
        assert_eq!(plan.final_planes()[0].offset_words, 0);
        assert_eq!(plan.final_planes()[0].geometry.width, 9);
        assert_eq!(plan.final_planes()[0].geometry.height, 5);
    }

    #[test]
    fn out_of_place_and_zero_residual_schedules_preserve_channel_order() {
        let out_of_place = one_squeeze_plan(
            5,
            3,
            2,
            ModularSqueezeParameter {
                horizontal: true,
                in_place: false,
                begin_channel: 0,
                channel_count: 1,
            },
        );
        assert_eq!(out_of_place.entropy_words(), 30);
        assert_eq!(out_of_place.arena_words(), 45);
        assert_eq!(out_of_place.final_planes().len(), 2);
        assert_eq!(out_of_place.final_planes()[0].geometry.width, 5);
        assert_eq!(out_of_place.final_planes()[0].offset_words, 30);
        assert_eq!(out_of_place.final_planes()[1].geometry.width, 5);
        assert_eq!(out_of_place.final_planes()[1].offset_words, 9);

        let single_column = one_squeeze_plan(
            1,
            7,
            1,
            ModularSqueezeParameter {
                horizontal: true,
                in_place: true,
                begin_channel: 0,
                channel_count: 1,
            },
        );
        assert_eq!(single_column.entropy_words(), 7);
        assert_eq!(single_column.arena_words(), 14);
        let ModularInverseJob::Squeeze { params } = single_column.jobs()[0] else {
            panic!("single Squeeze plan contains RCT");
        };
        assert_eq!(params.residual_plane().width, 0);
        assert_eq!(single_column.final_planes()[0].geometry.width, 1);
        assert_eq!(single_column.final_planes()[0].geometry.height, 7);
    }

    #[test]
    fn mixed_schedule_preserves_rct_and_squeeze_inverse_order() {
        let plan = mixed_rct_squeeze_plan();
        assert_eq!(plan.entropy_words(), 135);
        assert_eq!(plan.jobs().len(), 5);
        assert!(matches!(
            plan.jobs(),
            [
                ModularInverseJob::Rct { params: first },
                ModularInverseJob::Squeeze { .. },
                ModularInverseJob::Squeeze { .. },
                ModularInverseJob::Squeeze { .. },
                ModularInverseJob::Rct { params: last },
            ] if first.rct_type() == 41 && last.rct_type() == 5
        ));
        assert_eq!(plan.final_planes().len(), 3);
        assert!(plan.final_planes().iter().all(|plane| {
            plane.geometry.width == 9 && plane.geometry.height == 5 && plane.geometry.bit_depth == 8
        }));
    }

    fn scalar_tendency(previous: i32, average: i32, next: i32) -> i64 {
        let (previous, average, next) = (i64::from(previous), i64::from(average), i64::from(next));
        if previous >= average && average >= next {
            let mut tendency = (4 * previous - 3 * next - average + 6) / 12;
            if tendency - (tendency & 1) > 2 * (previous - average) {
                tendency = 2 * (previous - average) + 1;
            }
            if tendency + (tendency & 1) > 2 * (average - next) {
                tendency = 2 * (average - next);
            }
            tendency
        } else if previous <= average && average <= next {
            let mut tendency = (4 * previous - 3 * next - average - 6) / 12;
            if tendency + (tendency & 1) < 2 * (previous - average) {
                tendency = 2 * (previous - average) - 1;
            }
            if tendency - (tendency & 1) < 2 * (average - next) {
                tendency = 2 * (average - next);
            }
            tendency
        } else {
            0
        }
    }

    fn scalar_pair(average: i32, residual: i32, next: i32, previous: i32) -> (i32, i32) {
        let difference = i64::from(residual) + scalar_tendency(previous, average, next);
        let first = i64::from(average) + difference / 2;
        (first as i32, (first - difference) as i32)
    }

    fn execute_scalar_squeeze(arena: &mut [i32], params: ModularSqueezeParams) {
        let average = params.average_plane();
        let residual = params.residual_plane();
        let output = params.output_plane();
        match params.direction() {
            ModularSqueezeDirection::Horizontal => {
                for y in 0..average.height {
                    let average_base = average.offset_words + y * average.stride;
                    let residual_base = residual.offset_words + y * residual.stride;
                    let output_base = output.offset_words + y * output.stride;
                    let mut previous = arena[average_base as usize];
                    for x in 0..residual.width {
                        let value = arena[(average_base + x) as usize];
                        let next = if x + 1 < average.width {
                            arena[(average_base + x + 1) as usize]
                        } else if output.width % 2 == 1 {
                            arena[(average_base + residual.width) as usize]
                        } else {
                            value
                        };
                        let (first, second) =
                            scalar_pair(value, arena[(residual_base + x) as usize], next, previous);
                        arena[(output_base + 2 * x) as usize] = first;
                        arena[(output_base + 2 * x + 1) as usize] = second;
                        previous = second;
                    }
                    if output.width % 2 == 1 {
                        arena[(output_base + output.width - 1) as usize] =
                            arena[(average_base + average.width - 1) as usize];
                    }
                }
            }
            ModularSqueezeDirection::Vertical => {
                for x in 0..average.width {
                    let mut previous = arena[(average.offset_words + x) as usize];
                    for y in 0..residual.height {
                        let average_index = average.offset_words + y * average.stride + x;
                        let residual_index = residual.offset_words + y * residual.stride + x;
                        let output_index = output.offset_words + 2 * y * output.stride + x;
                        let value = arena[average_index as usize];
                        let next = if y + 1 < average.height {
                            arena[(average.offset_words + (y + 1) * average.stride + x) as usize]
                        } else if output.height % 2 == 1 {
                            arena[(average.offset_words + residual.height * average.stride + x)
                                as usize]
                        } else {
                            value
                        };
                        let (first, second) =
                            scalar_pair(value, arena[residual_index as usize], next, previous);
                        arena[output_index as usize] = first;
                        arena[(output_index + output.stride) as usize] = second;
                        previous = second;
                    }
                    if output.height % 2 == 1 {
                        arena[(output.offset_words + (output.height - 1) * output.stride + x)
                            as usize] = arena[(average.offset_words
                            + (average.height - 1) * average.stride
                            + x) as usize];
                    }
                }
            }
        }
    }

    fn wrapping_rct(first: i32, second: i32, third: i32, rct_type: u32) -> [i32; 3] {
        let values = match rct_type % 7 {
            0 => [first, second, third],
            1 => [first, second, third.wrapping_add(first)],
            2 => [first, second.wrapping_add(first), third],
            3 => [first, second.wrapping_add(first), third.wrapping_add(first)],
            4 => [
                first,
                second.wrapping_add(first.wrapping_add(third) >> 1),
                third,
            ],
            5 => {
                let third = first.wrapping_add(third);
                [
                    first,
                    second.wrapping_add(first.wrapping_add(third) >> 1),
                    third,
                ]
            }
            6 => {
                let y = first.wrapping_sub(third >> 1);
                let green = third.wrapping_add(y);
                let y = y.wrapping_sub(second >> 1);
                [y.wrapping_add(second), green, y]
            }
            _ => unreachable!(),
        };
        match rct_type / 7 {
            0 => values,
            1 => [values[2], values[0], values[1]],
            2 => [values[1], values[2], values[0]],
            3 => [values[0], values[2], values[1]],
            4 => [values[1], values[0], values[2]],
            5 => [values[2], values[1], values[0]],
            _ => unreachable!(),
        }
    }

    fn execute_scalar_rct(arena: &mut [i32], params: ModularRctParams) {
        let [first, second, third] = params.planes();
        for y in 0..first.height {
            for x in 0..first.width {
                let indices = [first, second, third].map(|plane| {
                    usize::try_from(plane.offset_words + y * plane.stride + x).unwrap()
                });
                let values = wrapping_rct(
                    arena[indices[0]],
                    arena[indices[1]],
                    arena[indices[2]],
                    params.rct_type(),
                );
                for (index, value) in indices.into_iter().zip(values) {
                    arena[index] = value;
                }
            }
        }
    }

    fn execute_scalar_job(arena: &mut [i32], job: ModularInverseJob) {
        match job {
            ModularInverseJob::Squeeze { params } => execute_scalar_squeeze(arena, params),
            ModularInverseJob::Rct { params } => execute_scalar_rct(arena, params),
        }
    }

    fn entropy_value(index: usize) -> i32 {
        match index % 9 {
            0 => i32::MIN,
            1 => i32::MAX,
            2 => index as i32 * 17,
            3 => -(index as i32) * 31,
            4 => 0,
            5 => 1,
            6 => -1,
            7 => 0x1234_5678,
            _ => -0x1234_5678,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn actual_adapter_executes_mixed_schedule_without_intermediate_map() {
        use std::sync::mpsc;

        use wgpu::util::DeviceExt;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            }))
        else {
            eprintln!("skipping Modular inverse schedule GPU test: no adapter");
            return;
        };
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("jxl-wgpu Modular inverse schedule test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
        else {
            eprintln!("skipping Modular inverse schedule GPU test: device request failed");
            return;
        };

        let plan = mixed_rct_squeeze_plan();
        let mut initial = (0..plan.entropy_words() as usize)
            .map(entropy_value)
            .collect::<Vec<_>>();
        initial.resize(plan.arena_words() as usize, 0);
        let mut expected = initial.clone();
        for &job in plan.jobs() {
            execute_scalar_job(&mut expected, job);
        }
        let final_spans = plan
            .final_planes()
            .iter()
            .copied()
            .map(ModularArenaPlane::span)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let expected = final_spans
            .iter()
            .flat_map(|span| expected[span.offset as usize..span.end().unwrap() as usize].iter())
            .copied()
            .collect::<Vec<_>>();

        let arena = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Modular inverse scheduled arena"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Modular inverse scheduled readback"),
            size: expected.len() as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let squeeze_pipeline =
            ModularSqueezePipeline::with_variant(&device, jxl_wgpu::KernelVariant::Lanes128)
                .unwrap();
        let rct_pipeline =
            ModularRctPipeline::with_variant(&device, jxl_wgpu::KernelVariant::Lanes128).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Modular inverse scheduled encoder"),
        });
        let arena_binding = ModularSqueezeArena::entire(&arena).unwrap().storage;
        let uniforms = plan
            .jobs()
            .iter()
            .map(|job| match *job {
                ModularInverseJob::Squeeze { params } => squeeze_pipeline
                    .encode(
                        &device,
                        &mut encoder,
                        ModularSqueezeArena::from_storage(arena_binding),
                        params,
                    )
                    .unwrap(),
                ModularInverseJob::Rct { params } => rct_pipeline
                    .encode(
                        &device,
                        &mut encoder,
                        ModularRctArena::from_storage(arena_binding),
                        params,
                    )
                    .unwrap(),
            })
            .collect::<Vec<_>>();
        let mut staging_offset = 0u64;
        for span in final_spans {
            let bytes = u64::from(span.length) * 4;
            encoder.copy_buffer_to_buffer(
                &arena,
                u64::from(span.offset) * 4,
                &staging,
                staging_offset,
                bytes,
            );
            staging_offset += bytes;
        }
        queue.submit(Some(encoder.finish()));
        drop(uniforms);

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv().unwrap().unwrap();
        let mapped = slice
            .get_mapped_range()
            .expect("mapped Modular inverse output");
        let actual = bytemuck::cast_slice::<u8, i32>(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        assert_eq!(actual, expected);
    }
}
