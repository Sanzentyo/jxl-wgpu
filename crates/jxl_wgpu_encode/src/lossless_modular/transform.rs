//! Metadata-only resolution shared by allocation, kernel lowering and wire assembly.
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
use super::palette::LosslessModularPalette;
use super::predictor::LosslessModularPredictor;
use super::rct::LosslessModularRctType;
use super::squeeze::LosslessModularSqueeze;
use super::types::{LosslessModularConfig, LosslessModularFormat};
use crate::{BackendError, EncodeError};

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
}

impl SampleSource {
    pub(super) const fn kernel_value(self) -> u32 {
        match self {
            Self::Component(WorkingComponent(value)) => value,
            Self::PaletteIndex => 4,
            Self::PaletteTable => 5,
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
    pub(super) squeeze: LosslessModularSqueeze,
}

impl GroupTransformPlan {
    fn new(
        extent: [u32; 2],
        format: LosslessModularFormat,
        depth: u8,
        config: LosslessModularConfig,
        local_rct: Option<LosslessModularRctType>,
    ) -> Self {
        let mut channels: Vec<_> = (0..format.channel_count())
            .map(|component| PlannedChannel {
                extent,
                source: SampleSource::Component(WorkingComponent(component)),
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
                    band: 0,
                }],
            );
            channels.insert(
                0,
                PlannedChannel {
                    extent: [palette.capacity.entries(), range.count],
                    source: SampleSource::PaletteTable,
                    band: 0,
                },
            );
            operations.push(TransformOperation::Palette(palette));
        }
        let squeeze = config.squeeze.for_extent(extent[0], extent[1]);
        let mut steps = Vec::new();
        for stage in 0..squeeze.stages() {
            let horizontal = squeeze.first_horizontal() ^ (stage != 0);
            let begin = usize::from(palette.is_some());
            let count = channels.len() - begin;
            let axis = usize::from(!horizontal);
            let residuals: Vec<_> = channels[begin..]
                .iter()
                .map(|channel| {
                    let mut residual = *channel;
                    residual.extent[axis] /= 2;
                    residual.band |= 1 << stage;
                    residual
                })
                .collect();
            for channel in &mut channels[begin..] {
                channel.extent[axis] = channel.extent[axis].div_ceil(2);
            }
            channels.extend(residuals);
            steps.push(SqueezeStep {
                horizontal,
                range: ChannelRange {
                    begin: begin as u32,
                    count: count as u32,
                },
            });
        }
        if !steps.is_empty() {
            operations.push(TransformOperation::Squeeze(steps));
        }
        Self {
            extent,
            channels,
            operations,
            palette,
            squeeze,
        }
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
                    extent, format, depth, config, local_rct,
                ));
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
        let image_count = config.palette.map_or(format.channel_count(), |palette| {
            format.channel_count() - palette.components(format) + 1
        });
        let prefix_channels = (u32::from(config.palette.is_some())
            + (image_count << config.squeeze.stages()))
        .min(4) as usize;
        Ok(Self {
            global_operations,
            rct_type: rct.map_or(42, |rct| rct.rct_type.value()),
            max_channels,
            dispatches,
            prefix_channels,
            extended_prediction_domain: config.palette.is_some()
                || config.squeeze != LosslessModularSqueeze::None,
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
