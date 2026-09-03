// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear-scan allocation planning for transient GPU planes.

use std::collections::BTreeMap;

use jxl_gpu_protocol::{PlaneId, PlaneLifetime, PlaneRole, RenderOp, RenderPlan};

use crate::{Error, Result};

const DEFAULT_ALIGNMENT: u64 = 256;

/// Physical storage assigned to one logical plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaAllocation {
    pub plane: PlaneId,
    pub offset: u64,
    /// Number of bytes visible to the logical plane, excluding alignment padding.
    pub size: u64,
    pub first_use: usize,
    pub last_use: usize,
}

/// A deterministic aliasing plan for independent GPU buffer slots.
///
/// `ArenaAllocation::offset` is a stable slot identifier. Allocations with the same offset reuse
/// one physical buffer only when their lifetimes do not overlap. `size_bytes` is the aggregate
/// capacity of all unique slots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaPlan {
    pub size_bytes: u64,
    pub peak_live_bytes: u64,
    pub peak_scratch_bytes: u64,
    pub allocations: Vec<ArenaAllocation>,
}

impl ArenaPlan {
    pub fn allocation(&self, plane: PlaneId) -> Option<&ArenaAllocation> {
        self.allocations
            .binary_search_by_key(&plane, |allocation| allocation.plane)
            .ok()
            .map(|index| &self.allocations[index])
    }

    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }
}

/// Builds lifetime slots using non-overlapping plane lifetimes.
#[derive(Clone, Copy, Debug)]
pub struct ArenaPlanner {
    alignment: u64,
    max_buffer_bytes: u64,
}

impl ArenaPlanner {
    pub const fn new(max_buffer_bytes: u64) -> Self {
        Self {
            alignment: DEFAULT_ALIGNMENT,
            max_buffer_bytes,
        }
    }

    pub fn with_alignment(mut self, alignment: u64) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::InvalidPayload(format!(
                "arena alignment {alignment} is not a non-zero power of two"
            )));
        }
        self.alignment = alignment;
        Ok(self)
    }

    pub fn plan(&self, plan: &RenderPlan) -> Result<ArenaPlan> {
        self.plan_impl(plan, &BTreeMap::new(), None)
    }

    /// Plans an arena against the scheduler's physical dispatch sequence.
    ///
    /// Nodes carrying the same step execute in one compute dispatch. Mapping logical node
    /// lifetimes onto those steps prevents buffers that become simultaneous read/write bindings
    /// after fusion from sharing a physical slot.
    pub(crate) fn plan_with_node_steps(
        &self,
        plan: &RenderPlan,
        node_steps: &[usize],
    ) -> Result<ArenaPlan> {
        self.plan_impl(plan, &BTreeMap::new(), Some(node_steps))
    }

    /// Plans an arena with optional byte-size overrides.
    ///
    /// Streaming execution uses this to replace full-frame plane sizes with tile sizes while
    /// retaining the graph's original lifetimes.
    pub fn plan_with_sizes(
        &self,
        plan: &RenderPlan,
        size_overrides: &BTreeMap<PlaneId, u64>,
    ) -> Result<ArenaPlan> {
        self.plan_impl(plan, size_overrides, None)
    }

    fn plan_impl(
        &self,
        plan: &RenderPlan,
        size_overrides: &BTreeMap<PlaneId, u64>,
        node_steps: Option<&[usize]>,
    ) -> Result<ArenaPlan> {
        plan.validate()?;

        if let Some(node_steps) = node_steps {
            validate_node_steps(plan.nodes.len(), node_steps)?;
        }

        let mut graph_lifetimes = plan.lifetimes();
        // Parameter planes are late-bound resources rather than ordinary node inputs. Account
        // for their real use interval explicitly so the physical arena includes them and never
        // aliases their contents before the last consuming EPF pass.
        for (index, node) in plan.nodes.iter().enumerate() {
            if let RenderOp::Epf(params) = &node.op
                && let Some(plane) = params.sigma_plane
            {
                graph_lifetimes
                    .entry(plane)
                    .and_modify(|lifetime| {
                        lifetime.first = 0;
                        lifetime.last = lifetime.last.max(index);
                    })
                    .or_insert(PlaneLifetime {
                        first: 0,
                        last: index,
                    });
            }
        }
        if let Some(node_steps) = node_steps {
            for lifetime in graph_lifetimes.values_mut() {
                lifetime.first = node_steps[lifetime.first];
                lifetime.last = node_steps[lifetime.last];
            }
        }
        let terminal = node_steps
            .and_then(|steps| steps.last().copied())
            .map_or(0, |last| last + 1);
        let terminal = if node_steps.is_some() {
            terminal
        } else {
            plan.nodes.len()
        };
        let mut requests = Vec::new();
        for plane in &plan.planes {
            if plane.role == PlaneRole::ImportedResident {
                continue;
            }
            let Some(mut lifetime) = graph_lifetimes.get(&plane.id).copied() else {
                continue;
            };
            // Source and parameter uploads happen before any dispatch. Outputs remain live until
            // the submission has been copied or mapped for readback.
            if matches!(plane.role, PlaneRole::Source | PlaneRole::Parameter) {
                lifetime.first = 0;
            }
            if plane.role == PlaneRole::Output {
                lifetime.last = terminal;
            }

            let size = size_overrides
                .get(&plane.id)
                .copied()
                .map(Ok)
                .unwrap_or_else(|| plane_byte_size(plane))?;
            let physical_size = align_up(size, self.alignment)?;
            requests.push(Request {
                plane: plane.id,
                role: plane.role,
                lifetime,
                size,
                physical_size,
            });
        }

        requests.sort_by_key(|request| {
            (
                request.lifetime.first,
                std::cmp::Reverse(request.physical_size),
                request.plane,
            )
        });

        let mut slots: Vec<Slot> = Vec::new();
        let mut allocations = Vec::with_capacity(requests.len());
        let mut arena_size = 0u64;
        for request in &requests {
            let reusable = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| {
                    slot.last_use < request.lifetime.first && slot.capacity >= request.physical_size
                })
                .min_by_key(|(_, slot)| (slot.capacity, slot.offset))
                .map(|(index, _)| index);

            let offset = if let Some(index) = reusable {
                let slot = &mut slots[index];
                slot.last_use = request.lifetime.last;
                slot.offset
            } else {
                let offset = align_up(arena_size, self.alignment)?;
                arena_size = offset
                    .checked_add(request.physical_size)
                    .ok_or(Error::BufferSizeOverflow)?;
                slots.push(Slot {
                    offset,
                    capacity: request.physical_size,
                    last_use: request.lifetime.last,
                });
                offset
            };

            allocations.push(ArenaAllocation {
                plane: request.plane,
                offset,
                size: request.size,
                first_use: request.lifetime.first,
                last_use: request.lifetime.last,
            });
        }

        if self.max_buffer_bytes != 0 && arena_size > self.max_buffer_bytes {
            return Err(Error::ResourceLimit(format!(
                "arena requires {arena_size} bytes, exceeding the device buffer limit of {} bytes",
                self.max_buffer_bytes
            )));
        }

        let (peak_live_bytes, peak_scratch_bytes) = peak_live_bytes(&requests, terminal)?;
        allocations.sort_by_key(|allocation| allocation.plane);
        Ok(ArenaPlan {
            size_bytes: arena_size,
            peak_live_bytes,
            peak_scratch_bytes,
            allocations,
        })
    }
}

fn validate_node_steps(node_count: usize, node_steps: &[usize]) -> Result<()> {
    if node_steps.len() != node_count {
        return Err(Error::InvalidPayload(format!(
            "arena node schedule has {} entries for {node_count} nodes",
            node_steps.len()
        )));
    }
    if let Some(first) = node_steps.first()
        && *first != 0
    {
        return Err(Error::InvalidPayload(
            "arena node schedule must start at step 0".into(),
        ));
    }
    if node_steps
        .windows(2)
        .any(|pair| pair[1] < pair[0] || pair[1] > pair[0] + 1)
    {
        return Err(Error::InvalidPayload(
            "arena node schedule must be non-decreasing without gaps".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Request {
    plane: PlaneId,
    role: PlaneRole,
    lifetime: PlaneLifetime,
    size: u64,
    physical_size: u64,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    offset: u64,
    capacity: u64,
    last_use: usize,
}

fn plane_byte_size(plane: &jxl_gpu_protocol::PlaneDesc) -> Result<u64> {
    let stride = if plane.stride == 0 {
        plane.extent.width
    } else {
        plane.stride
    };
    u64::from(stride)
        .checked_mul(u64::from(plane.extent.height))
        .and_then(|samples| {
            samples.checked_mul(u64::try_from(plane.sample_type.bytes_per_sample()).ok()?)
        })
        .ok_or(Error::BufferSizeOverflow)
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    debug_assert!(alignment.is_power_of_two());
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(Error::BufferSizeOverflow)
}

fn peak_live_bytes(requests: &[Request], terminal: usize) -> Result<(u64, u64)> {
    let mut peak_live = 0u64;
    let mut peak_scratch = 0u64;
    for index in 0..=terminal {
        let mut live = 0u64;
        let mut scratch = 0u64;
        for request in requests
            .iter()
            .filter(|request| request.lifetime.first <= index && index <= request.lifetime.last)
        {
            live = live
                .checked_add(request.physical_size)
                .ok_or(Error::BufferSizeOverflow)?;
            if request.role == PlaneRole::Intermediate {
                scratch = scratch
                    .checked_add(request.physical_size)
                    .ok_or(Error::BufferSizeOverflow)?;
            }
        }
        peak_live = peak_live.max(live);
        peak_scratch = peak_scratch.max(scratch);
    }
    Ok((peak_live, peak_scratch))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use jxl_gpu_protocol::{
        Border2d, EpfParams, EpfPass, Extent2d, PlaneDesc, PrecisionContract, RenderNode, RenderOp,
        ResourceId, SampleType, Scale2d,
    };

    use super::*;

    fn plane(id: u32, role: PlaneRole) -> PlaneDesc {
        PlaneDesc {
            id: PlaneId(id),
            extent: Extent2d::new(8, 8),
            stride: 8,
            sample_type: SampleType::F32,
            role,
        }
    }

    fn node(input: u32, output: u32) -> RenderNode {
        RenderNode {
            name: Arc::from("copy"),
            op: RenderOp::Copy,
            inputs: vec![PlaneId(input)],
            outputs: vec![PlaneId(output)],
            resources: Vec::new(),
            scale: Scale2d::IDENTITY,
            border: Border2d::default(),
            precision: PrecisionContract::Exact,
        }
    }

    #[test]
    fn aliases_planes_with_disjoint_lifetimes() {
        let plan = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source),
                plane(1, PlaneRole::Intermediate),
                plane(2, PlaneRole::Intermediate),
                plane(3, PlaneRole::Output),
            ],
            nodes: vec![node(0, 1), node(1, 2), node(2, 3)],
            outputs: Vec::new(),
        };

        let arena = ArenaPlanner::new(u64::MAX)
            .with_alignment(16)
            .unwrap()
            .plan(&plan)
            .unwrap();
        assert_eq!(arena.allocation(PlaneId(0)).unwrap().offset, 0);
        assert_eq!(arena.allocation(PlaneId(2)).unwrap().offset, 0);
        assert_eq!(arena.size_bytes, 512);
    }

    #[test]
    fn does_not_alias_inputs_and_outputs_of_same_node() {
        let plan = RenderPlan {
            planes: vec![plane(0, PlaneRole::Source), plane(1, PlaneRole::Output)],
            nodes: vec![node(0, 1)],
            outputs: Vec::new(),
        };
        let arena = ArenaPlanner::new(u64::MAX)
            .with_alignment(16)
            .unwrap()
            .plan(&plan)
            .unwrap();
        assert_ne!(
            arena.allocation(PlaneId(0)).unwrap().offset,
            arena.allocation(PlaneId(1)).unwrap().offset
        );
    }

    #[test]
    fn fused_node_steps_do_not_alias_simultaneous_bindings() {
        let plan = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source),
                plane(1, PlaneRole::Intermediate),
                plane(2, PlaneRole::Output),
            ],
            nodes: vec![node(0, 1), node(1, 2)],
            outputs: Vec::new(),
        };

        let unfused = ArenaPlanner::new(u64::MAX)
            .with_alignment(16)
            .unwrap()
            .plan(&plan)
            .unwrap();
        assert_eq!(
            unfused.allocation(PlaneId(0)).unwrap().offset,
            unfused.allocation(PlaneId(2)).unwrap().offset
        );

        let fused = ArenaPlanner::new(u64::MAX)
            .with_alignment(16)
            .unwrap()
            .plan_with_node_steps(&plan, &[0, 0])
            .unwrap();
        let source = fused.allocation(PlaneId(0)).unwrap().offset;
        let intermediate = fused.allocation(PlaneId(1)).unwrap().offset;
        let output = fused.allocation(PlaneId(2)).unwrap().offset;
        assert_ne!(source, intermediate);
        assert_ne!(source, output);
        assert_ne!(intermediate, output);
    }

    #[test]
    fn applies_streaming_size_overrides() {
        let plan = RenderPlan {
            planes: vec![plane(0, PlaneRole::Source), plane(1, PlaneRole::Output)],
            nodes: vec![node(0, 1)],
            outputs: Vec::new(),
        };
        let sizes = BTreeMap::from([(PlaneId(0), 64), (PlaneId(1), 32)]);
        let arena = ArenaPlanner::new(u64::MAX)
            .with_alignment(16)
            .unwrap()
            .plan_with_sizes(&plan, &sizes)
            .unwrap();
        assert_eq!(arena.size_bytes, 96);
        assert_eq!(arena.allocation(PlaneId(1)).unwrap().size, 32);
    }

    #[test]
    fn rejects_arena_larger_than_device_limit() {
        let plan = RenderPlan {
            planes: vec![plane(0, PlaneRole::Source), plane(1, PlaneRole::Output)],
            nodes: vec![node(0, 1)],
            outputs: Vec::new(),
        };
        assert!(matches!(
            ArenaPlanner::new(128).plan(&plan),
            Err(Error::ResourceLimit(_))
        ));
    }

    #[test]
    fn includes_late_bound_epf_parameter_plane_in_arena_lifetime() {
        let sigma = PlaneId(2);
        let plan = RenderPlan {
            planes: vec![
                plane(0, PlaneRole::Source),
                plane(1, PlaneRole::Output),
                plane(2, PlaneRole::Parameter),
            ],
            nodes: vec![RenderNode {
                name: Arc::from("epf"),
                op: RenderOp::Epf(EpfParams {
                    pass: EpfPass::Pass2,
                    sigma_scale: 1.0,
                    border_sad_mul: 1.0,
                    channel_scale: [1.0; 3],
                    sigma_resource: Some(ResourceId(0)),
                    sigma_plane: Some(sigma),
                }),
                inputs: vec![PlaneId(0)],
                outputs: vec![PlaneId(1)],
                resources: vec![ResourceId(0)],
                scale: Scale2d::IDENTITY,
                border: Border2d::symmetric(1, 1),
                precision: PrecisionContract::default(),
            }],
            outputs: Vec::new(),
        };

        let arena = ArenaPlanner::new(u64::MAX).plan(&plan).unwrap();
        let allocation = arena
            .allocation(sigma)
            .expect("late-bound sigma plane must consume physical storage");
        assert_eq!(allocation.first_use, 0);
        assert_eq!(allocation.last_use, 0);
    }
}
