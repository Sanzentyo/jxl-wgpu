//! Physical stream geometry before local transforms; channel order is wire order.
use super::*;
use crate::extra_channel::input::{ScalarRoute, route};
use crate::extra_channel::sampling::ExtraChannelSamplingPlan;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::lossless_modular) struct ExtraRegion {
    pub source: usize,
    pub origin: [u32; 2],
    pub extent: [u32; 2],
    pub shift: u8,
}

#[derive(Clone, Debug)]
pub(in crate::lossless_modular) struct PlannedStream {
    pub route: ScalarRoute,
    pub region: LosslessModularGroup,
    pub extras: Vec<ExtraRegion>,
    pub shape: usize,
}

impl PlannedStream {
    pub(in crate::lossless_modular) fn has_color(&self) -> bool {
        self.route != ScalarRoute::Lf
    }
}

pub(super) fn streams(
    grid: LosslessModularGroupGrid,
    sampling: &ExtraChannelSamplingPlan,
) -> Vec<PlannedStream> {
    let mut global_prefix = grid.groups == 1;
    let routes: Vec<_> = sampling
        .channels
        .iter()
        .filter_map(|sampled| {
            sampled.source.map(|source| {
                let (route, _) = route(
                    &mut global_prefix,
                    sampled.extent,
                    sampled.shift,
                    grid.group_size.dimension(),
                );
                (source, sampled, route)
            })
        })
        .collect();
    let mut streams = Vec::new();
    if routes.iter().any(|(_, _, route)| *route == ScalarRoute::Lf) {
        let dimension = grid.group_size.dimension() * 8;
        for index in 0..grid.lf_groups {
            let column = index % grid.lf_columns;
            let row = index / grid.lf_columns;
            let x = column * dimension;
            let y = row * dimension;
            streams.push(PlannedStream {
                route: ScalarRoute::Lf,
                region: LosslessModularGroup {
                    index,
                    column,
                    row,
                    x,
                    y,
                    width: (grid.width - x).min(dimension),
                    height: (grid.height - y).min(dimension),
                },
                extras: Vec::new(),
                shape: 0,
            });
        }
    }
    for region in grid.ordered_groups() {
        streams.push(PlannedStream {
            route: if grid.groups == 1 {
                ScalarRoute::Global
            } else {
                ScalarRoute::Pass
            },
            region,
            extras: Vec::new(),
            shape: 0,
        });
    }
    for stream in &mut streams {
        for &(source, sampled, route) in &routes {
            if route != stream.route {
                continue;
            }
            let origin = [
                stream.region.x >> sampled.shift,
                stream.region.y >> sampled.shift,
            ];
            let factor = 1u32 << sampled.shift;
            stream.extras.push(ExtraRegion {
                source,
                origin,
                extent: [
                    stream.region.width.div_ceil(factor),
                    stream.region.height.div_ceil(factor),
                ],
                shift: sampled.shift,
            });
        }
    }
    streams
}
