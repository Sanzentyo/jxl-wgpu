//! Metadata-only resolution shared by allocation, kernel lowering and wire assembly.
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
use super::palette::LosslessModularPalette;
use super::predictor::LosslessModularPredictor;
use super::rct::LosslessModularRctType;
use super::squeeze::SqueezeAxes;
use super::types::{LosslessModularConfig, LosslessModularFormat};
use crate::{BackendError, EncodeError};
pub(super) mod program;
use program::SqueezeProgram;

#[cfg(test)]
mod tests;

/// A component of the working image after the resolved RCT, before Palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WorkingComponent(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SampleSource {
    Component(WorkingComponent),
    PaletteIndex,
    PaletteTable,
    Arena(u32),
}

impl SampleSource {
    pub(super) const fn kernel_value(self) -> u32 {
        match self {
            Self::Component(WorkingComponent(value)) => value,
            Self::PaletteIndex => 4,
            Self::PaletteTable => 5,
            Self::Arena(_) => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ChannelRange {
    pub(super) begin: u32,
    pub(super) count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlannedChannel {
    pub(super) extent: [u32; 2],
    pub(super) source: SampleSource,
    pub(super) shifts: [u8; 2],
    pub(super) squeeze: SqueezeAxes,
    /// Bit n selects the residual from Squeeze stage n; zero selects its average.
    pub(super) band: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PaletteCapacity {
    pub(super) colors: u32,
    pub(super) deltas: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PaletteArtifactPlan {
    pub(super) counts_byte_offset: u64,
    pub(super) capacity: PaletteCapacity,
    pub(super) scratch_bytes: u64,
}

impl PaletteCapacity {
    pub(super) const fn entries(self) -> u32 {
        self.colors + self.deltas
    }

    pub(super) fn validate(
        self,
        entries: u32,
        deltas: u32,
    ) -> Result<ValidatedPaletteCounts, EncodeError> {
        if entries == 0
            || deltas > entries
            || deltas > self.deltas
            || entries - deltas > self.colors
        {
            return Err(
                BackendError::InvalidArtifact("palette count exceeds planned capacity").into(),
            );
        }
        Ok(ValidatedPaletteCounts { entries, deltas })
    }
}

/// Counts acquire wire authority only through the matching planned capacity.
#[derive(Clone, Copy, Debug)]
pub(super) struct ValidatedPaletteCounts {
    entries: u32,
    deltas: u32,
}

impl ValidatedPaletteCounts {
    pub(super) const fn entries(self) -> u32 {
        self.entries
    }
    pub(super) const fn deltas(self) -> u32 {
        self.deltas
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PlannedPalette {
    pub(super) range: ChannelRange,
    pub(super) capacity: PaletteCapacity,
    pub(super) delta_predictor: Option<LosslessModularPredictor>,
    pub(super) implicit_depth: u32,
    pub(super) hash_entries: u32,
    pub(super) scratch_words: u64,
}

impl PlannedPalette {
    fn new(
        policy: LosslessModularPalette,
        format: LosslessModularFormat,
        depth: u8,
        extent: [u32; 2],
    ) -> Self {
        let pixels = extent[0] * extent[1]; // checked pass-group edges are at most 1024
        let range = ChannelRange {
            begin: policy.begin(),
            count: policy.components(format),
        };
        let capacity = PaletteCapacity {
            colors: (policy.max_colors() - policy.max_deltas()).min(pixels),
            deltas: policy
                .max_deltas()
                .min(pixels + u32::from(policy.uses_implicit_entries())),
        };
        let hash_entries = (2
            * (capacity.entries()
                + if policy.uses_implicit_entries() {
                    143
                } else {
                    0
                }))
        .next_power_of_two();
        let delta_predictor = policy.delta_predictor();
        let scratch_words = 2
            + u64::from(capacity.entries()) * u64::from(range.count)
            + u64::from(hash_entries)
            + if delta_predictor.is_some() {
                u64::from(pixels) * u64::from(range.count)
            } else {
                0
            }
            + if delta_predictor == Some(LosslessModularPredictor::Weighted) {
                5 * u64::from(extent[0])
            } else {
                0
            };
        Self {
            range,
            capacity,
            delta_predictor,
            implicit_depth: if policy.uses_implicit_entries() {
                u32::from(depth)
            } else {
                0
            },
            hash_entries,
            scratch_words,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SqueezeStep {
    pub(super) horizontal: bool,
    pub(super) in_place: bool,
    pub(super) range: ChannelRange,
}

#[derive(Clone, Debug)]
pub(super) enum TransformOperation {
    Rct(LosslessModularRctType),
    Palette(PlannedPalette),
    Squeeze(Vec<SqueezeStep>),
}

#[derive(Clone, Debug)]
pub(super) struct GroupTransformPlan {
    pub(super) extent: [u32; 2],
    pub(super) channels: Vec<PlannedChannel>,
    pub(super) operations: Vec<TransformOperation>,
    pub(super) palette: Option<PlannedPalette>,
    pub(super) squeeze_program: Option<SqueezeProgram>,
}

impl GroupTransformPlan {
    fn new(
        extent: [u32; 2],
        format: LosslessModularFormat,
        depth: u8,
        config: &LosslessModularConfig,
        local_rct: Option<LosslessModularRctType>,
        squeeze_range: &std::ops::Range<u32>,
    ) -> Result<Self, EncodeError> {
        let mut channels: Vec<_> = (0..format.channel_count())
            .map(|component| PlannedChannel {
                extent,
                source: SampleSource::Component(WorkingComponent(component)),
                shifts: [0; 2],
                squeeze: SqueezeAxes::None,
                band: 0,
            })
            .collect();
        let mut operations = Vec::new();
        if let Some(rct) = local_rct {
            operations.push(TransformOperation::Rct(rct));
        }
        let palette = config
            .palette
            .map(|policy| PlannedPalette::new(policy, format, depth, extent));
        if let Some(palette) = palette {
            let range = palette.range;
            channels.splice(
                range.begin as usize..(range.begin + range.count) as usize,
                [PlannedChannel {
                    extent,
                    source: SampleSource::PaletteIndex,
                    shifts: [0; 2],
                    squeeze: SqueezeAxes::None,
                    band: 0,
                }],
            );
            channels.insert(
                0,
                PlannedChannel {
                    extent: [palette.capacity.entries(), range.count],
                    source: SampleSource::PaletteTable,
                    shifts: [0; 2],
                    squeeze: SqueezeAxes::None,
                    band: 0,
                },
            );
            operations.push(TransformOperation::Palette(palette));
        }
        let squeeze = config.squeeze.for_extent(extent[0], extent[1]);
        let meta = usize::from(palette.is_some());
        for channel in
            &mut channels[meta + squeeze_range.start as usize..meta + squeeze_range.end as usize]
        {
            channel.squeeze = squeeze;
        }
        let mut steps = Vec::new();
        for stage in 0..squeeze.stages() {
            let horizontal = squeeze.first_horizontal() ^ (stage != 0);
            // Snapshot the selected lineages before this axis. Tail residuals may be separated
            // from their averages by unselected channels, requiring distinct wire steps.
            let mut ranges = Vec::new();
            let mut cursor = 0;
            while cursor < channels.len() {
                if channels[cursor].squeeze == SqueezeAxes::None {
                    cursor += 1;
                    continue;
                }
                let begin = cursor;
                while cursor < channels.len() && channels[cursor].squeeze != SqueezeAxes::None {
                    cursor += 1;
                }
                ranges.push(begin..cursor);
            }
            let mut inserted = 0;
            for range in ranges {
                let begin = range.start + inserted;
                let end = range.end + inserted;
                let axis = usize::from(!horizontal);
                let residuals: Vec<_> = channels[begin..end]
                    .iter()
                    .map(|channel| {
                        let mut residual = *channel;
                        residual.extent[axis] /= 2;
                        residual.shifts[axis] += 1;
                        residual.band |= 1 << stage;
                        residual
                    })
                    .collect();
                for channel in &mut channels[begin..end] {
                    channel.extent[axis] = channel.extent[axis].div_ceil(2);
                    channel.shifts[axis] += 1;
                }
                if config.squeeze.in_place() == Some(true) {
                    channels.splice(end..end, residuals);
                    inserted += end - begin;
                } else {
                    channels.extend(residuals);
                }
                steps.push(SqueezeStep {
                    horizontal,
                    in_place: config.squeeze.in_place() == Some(true),
                    range: ChannelRange {
                        begin: begin as u32,
                        count: (end - begin) as u32,
                    },
                });
            }
        }
        let squeeze_program = if let Some(sequence) = config.squeeze.steps() {
            let (program, resolved) = SqueezeProgram::new(&mut channels, meta, sequence)?;
            steps = resolved;
            Some(program)
        } else {
            None
        };
        if !steps.is_empty() {
            operations.push(TransformOperation::Squeeze(steps));
        }
        Ok(Self {
            extent,
            channels,
            operations,
            palette,
            squeeze_program,
        })
    }
}

/// A frame shares at most four concrete edge-group topologies across every consumer.
#[derive(Clone, Debug)]
pub(super) struct ModularTransformPlan {
    pub(super) global_operations: Vec<TransformOperation>,
    pub(super) rct_type: u32,
    pub(super) max_channels: u32,
    pub(super) dispatches: u32,
    pub(super) prefix_channels: usize,
    pub(super) extended_prediction_domain: bool,
    shapes: Vec<GroupTransformPlan>,
}

impl ModularTransformPlan {
    pub(super) fn new(
        grid: LosslessModularGroupGrid,
        format: LosslessModularFormat,
        depth: u8,
        exponent_bits: u8,
        config: LosslessModularConfig,
    ) -> Result<Self, EncodeError> {
        if let Some(palette) = config.palette {
            palette.validate(format)?;
        }
        let image_count = config.palette.map_or(format.channel_count(), |palette| {
            format.channel_count() - palette.components(format) + 1
        });
        let squeeze_range = config.squeeze.resolve_range(image_count)?;
        let rct = config.color_transform.resolve(format, exponent_bits)?;
        let mut global_operations = Vec::new();
        if let Some(rct) = rct.filter(|rct| !rct.local && grid.groups > 1) {
            global_operations.push(TransformOperation::Rct(rct.rct_type));
        }
        let local_rct = rct
            .filter(|rct| rct.local || grid.groups == 1)
            .map(|rct| rct.rct_type);
        let mut shapes: Vec<GroupTransformPlan> = Vec::with_capacity(4);
        let mut dispatches = 0u32;
        for group in grid.ordered_groups() {
            let extent = [group.width, group.height];
            let shape = if let Some(index) = shapes.iter().position(|shape| shape.extent == extent)
            {
                index
            } else {
                shapes.push(GroupTransformPlan::new(
                    extent,
                    format,
                    depth,
                    &config,
                    local_rct,
                    &squeeze_range,
                )?);
                shapes.len() - 1
            };
            dispatches = dispatches
                .checked_add(shapes[shape].channels.len() as u32)
                .ok_or(EncodeError::InvalidSource(
                    "Modular dispatch count overflow",
                ))?;
        }
        let max_channels = shapes
            .iter()
            .map(|shape| shape.channels.len() as u32)
            .max()
            .ok_or(BackendError::Invariant("empty Modular topology"))?;
        // Keep the established prefix-table policy even when a group's one-pixel axes elide
        // Squeeze. This is an entropy upper bound, not another physical channel topology.
        let prefix_channels = if config.squeeze.steps().is_some() {
            max_channels.min(4) as usize
        } else {
            (u32::from(config.palette.is_some())
                + image_count
                + squeeze_range.len() as u32 * ((1 << config.squeeze.stages()) - 1))
                .min(4) as usize
        };
        Ok(Self {
            global_operations,
            rct_type: rct.map_or(42, |rct| rct.rct_type.value()),
            max_channels,
            dispatches,
            prefix_channels,
            extended_prediction_domain: config.palette.is_some() || config.squeeze.enabled(),
            shapes,
        })
    }

    pub(super) fn group(
        &self,
        group: LosslessModularGroup,
    ) -> Result<&GroupTransformPlan, EncodeError> {
        self.shapes
            .iter()
            .find(|shape| shape.extent == [group.width, group.height])
            .ok_or_else(|| BackendError::Invariant("Modular group has no planned topology").into())
    }
}
