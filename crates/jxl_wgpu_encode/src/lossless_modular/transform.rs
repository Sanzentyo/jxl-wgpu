//! Metadata-only resolution shared by allocation, kernel lowering and wire assembly.
use super::grid::{LosslessModularGroup, LosslessModularGroupGrid};
use super::palette::LosslessModularPalette;
use super::predictor::LosslessModularPredictor;
use super::rct::LosslessModularRctType;
use super::squeeze::SqueezeAxes;
use super::types::{LosslessModularConfig, LosslessModularFormat};
use crate::{BackendError, EncodeError};
pub(super) mod input;
pub(super) mod program;
use program::TransformProgram;

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
    Independent(usize),
}

impl SampleSource {
    pub(super) fn kernel_value(self) -> u32 {
        match self {
            Self::Component(WorkingComponent(value)) => value,
            Self::PaletteIndex => 4,
            Self::PaletteTable => 5,
            Self::Arena(_) => 6,
            Self::Independent(_) => unreachable!("source inputs must be lowered into the arena"),
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
    Rct(PlannedRct),
    Palette(PlannedPalette),
    Squeeze(Vec<SqueezeStep>),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PlannedRct {
    pub(super) begin: u32,
    pub(super) rct_type: LosslessModularRctType,
}

impl PlannedRct {
    pub(super) const fn source(rct_type: LosslessModularRctType) -> Self {
        Self { begin: 0, rct_type }
    }
}

#[derive(Clone, Debug)]
pub(super) struct GroupTransformPlan {
    pub(super) extent: [u32; 2],
    pub(super) channels: Vec<PlannedChannel>,
    pub(super) operations: Vec<TransformOperation>,
    pub(super) palette: Option<PlannedPalette>,
    pub(super) transform_program: Option<TransformProgram>,
    input_shape: Vec<([u32; 2], u8, usize)>,
    has_color: bool,
}

impl GroupTransformPlan {
    fn new(
        extent: [u32; 2],
        format: LosslessModularFormat,
        depth: u8,
        config: &LosslessModularConfig,
        local_rct: Option<LosslessModularRctType>,
        squeeze_range: &std::ops::Range<u32>,
        stream: &input::PlannedStream,
    ) -> Result<Self, EncodeError> {
        let mut channels: Vec<_> = (0..if stream.has_color() {
            format.channel_count()
        } else {
            0
        })
            .map(|component| PlannedChannel {
                extent,
                source: SampleSource::Component(WorkingComponent(component)),
                shifts: [0; 2],
                squeeze: SqueezeAxes::None,
                band: 0,
            })
            .collect();
        channels.extend(stream.extras.iter().map(|extra| PlannedChannel {
            extent: extra.extent,
            source: SampleSource::Independent(extra.source),
            shifts: [extra.shift; 2],
            squeeze: SqueezeAxes::None,
            band: 0,
        }));
        let mut operations = Vec::new();
        if let Some(rct) = local_rct {
            operations.push(TransformOperation::Rct(PlannedRct::source(rct)));
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
        let input_shape = stream
            .extras
            .iter()
            .map(|extra| (extra.extent, extra.shift, extra.source))
            .collect();
        let has_color = stream.has_color();
        let squeeze_policy = config.local_transforms.squeeze_policy();
        let squeeze = squeeze_policy.map_or(SqueezeAxes::None, |policy| {
            policy.for_extent(extent[0], extent[1])
        });
        let meta = usize::from(palette.is_some());
        for channel in
            &mut channels[meta + squeeze_range.start as usize..meta + squeeze_range.end as usize]
        {
            channel.squeeze = squeeze;
        }
        if !stream.extras.is_empty() {
            let (transform_program, steps) = if let Some(sequence) =
                squeeze_policy.and_then(|policy| policy.steps())
            {
                let (program, steps) = TransformProgram::squeeze(&mut channels, meta, sequence)?;
                (program, steps)
            } else if let Some(sequence) = config.local_transforms.operations() {
                let (program, resolved) = TransformProgram::new(&mut channels, meta, sequence)?;
                operations.extend(resolved);
                (program, Vec::new())
            } else {
                TransformProgram::named(
                    &mut channels,
                    meta,
                    squeeze,
                    squeeze_policy.and_then(|policy| policy.in_place()) == Some(true),
                )?
            };
            if !steps.is_empty() {
                operations.push(TransformOperation::Squeeze(steps));
            }
            if operations.len() > 273 {
                return Err(EncodeError::InvalidModularTransformCount {
                    count: operations.len(),
                });
            }
            return Ok(Self {
                extent,
                channels,
                operations,
                palette,
                transform_program: Some(transform_program),
                input_shape,
                has_color,
            });
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
                let in_place = squeeze_policy.and_then(|policy| policy.in_place()) == Some(true);
                if in_place {
                    channels.splice(end..end, residuals);
                    inserted += end - begin;
                } else {
                    channels.extend(residuals);
                }
                steps.push(SqueezeStep {
                    horizontal,
                    in_place,
                    range: ChannelRange {
                        begin: begin as u32,
                        count: (end - begin) as u32,
                    },
                });
            }
        }
        let transform_program =
            if let Some(sequence) = squeeze_policy.and_then(|policy| policy.steps()) {
                let (program, resolved) = TransformProgram::squeeze(&mut channels, meta, sequence)?;
                steps = resolved;
                Some(program)
            } else if let Some(sequence) = config.local_transforms.operations() {
                let (program, resolved) = TransformProgram::new(&mut channels, meta, sequence)?;
                operations.extend(resolved);
                Some(program)
            } else {
                None
            };
        if !steps.is_empty() {
            operations.push(TransformOperation::Squeeze(steps));
        }
        if operations.len() > 273 {
            return Err(EncodeError::InvalidModularTransformCount {
                count: operations.len(),
            });
        }
        Ok(Self {
            extent,
            channels,
            operations,
            palette,
            transform_program,
            input_shape,
            has_color,
        })
    }
}

/// A frame shares checked color/scalar stream topologies across every consumer.
#[derive(Clone, Debug)]
pub(super) struct ModularTransformPlan {
    pub(super) global_operations: Vec<TransformOperation>,
    pub(super) rct_type: u32,
    pub(super) max_channels: u32,
    pub(super) dispatches: u32,
    pub(super) prefix_channels: usize,
    pub(super) extended_prediction_domain: bool,
    shapes: Vec<GroupTransformPlan>,
    pub(super) streams: Vec<input::PlannedStream>,
}

impl ModularTransformPlan {
    #[cfg(test)]
    pub(super) fn new(
        grid: LosslessModularGroupGrid,
        format: LosslessModularFormat,
        depth: u8,
        exponent_bits: u8,
        config: LosslessModularConfig,
    ) -> Result<Self, EncodeError> {
        let samples = config.samples(format, depth, exponent_bits, false)?;
        let sampling = crate::sampling::FrameSamplingPlan::unscaled(
            jxl_gpu_protocol::Extent2d::new(grid.width, grid.height),
            &samples,
        )?;
        Self::with_sampling(grid, format, depth, exponent_bits, config, &sampling.extras)
    }

    pub(super) fn with_sampling(
        grid: LosslessModularGroupGrid,
        format: LosslessModularFormat,
        depth: u8,
        exponent_bits: u8,
        config: LosslessModularConfig,
        sampling: &crate::extra_channel::sampling::ExtraChannelSamplingPlan,
    ) -> Result<Self, EncodeError> {
        if let Some(palette) = config.palette {
            palette.validate(format)?;
        }
        let image_count = config.palette.map_or(format.channel_count(), |palette| {
            format.channel_count() - palette.components(format) + 1
        });
        let rct = config.color_transform.resolve(format, exponent_bits)?;
        let mut global_operations = Vec::new();
        if let Some(rct) = rct.filter(|rct| !rct.local && grid.groups > 1) {
            global_operations.push(TransformOperation::Rct(PlannedRct::source(rct.rct_type)));
        }
        let local_rct = rct
            .filter(|rct| rct.local || grid.groups == 1)
            .map(|rct| rct.rct_type);
        let mut shapes: Vec<GroupTransformPlan> = Vec::with_capacity(4);
        let mut dispatches = 0u32;
        let mut streams = input::streams(grid, sampling);
        for stream in &mut streams {
            let group = stream.region;
            let extent = [group.width, group.height];
            let input_shape: Vec<_> = stream
                .extras
                .iter()
                .map(|extra| (extra.extent, extra.shift, extra.source))
                .collect();
            let shape = if let Some(index) = shapes.iter().position(|shape| {
                shape.extent == extent
                    && shape.input_shape == input_shape
                    && shape.has_color == stream.has_color()
            }) {
                index
            } else {
                // Local color transforms belong to color streams. LF scalar streams retain
                // their intrinsic topology and share only the frame's prediction/entropy policy.
                let scalar_config = LosslessModularConfig::default();
                let (policy, rct, count) = if stream.has_color() {
                    (&config, local_rct, image_count + stream.extras.len() as u32)
                } else {
                    (&scalar_config, None, stream.extras.len() as u32)
                };
                let range = policy
                    .local_transforms
                    .squeeze_policy()
                    .map_or(Ok(0..count), |squeeze| squeeze.resolve_range(count))?;
                shapes.push(GroupTransformPlan::new(
                    extent, format, depth, policy, rct, &range, stream,
                )?);
                shapes.len() - 1
            };
            stream.shape = shape;
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
        let independent_inputs = sampling
            .channels
            .iter()
            .any(|channel| channel.source.is_some());
        let prefix_channels = if independent_inputs || config.local_transforms.uses_program() {
            max_channels.min(4) as usize
        } else {
            let squeeze_range = config
                .local_transforms
                .squeeze_policy()
                .map_or(Ok(0..image_count), |squeeze| {
                    squeeze.resolve_range(image_count)
                })?;
            (u32::from(config.palette.is_some())
                + image_count
                + squeeze_range.len() as u32
                    * ((1
                        << config
                            .local_transforms
                            .squeeze_policy()
                            .map_or(0, |squeeze| squeeze.stages()))
                        - 1))
                .min(4) as usize
        };
        Ok(Self {
            global_operations,
            rct_type: rct.map_or(42, |rct| rct.rct_type.value()),
            max_channels,
            dispatches,
            prefix_channels,
            extended_prediction_domain: independent_inputs
                || config.palette.is_some()
                || config.local_transforms.uses_program()
                || config.local_transforms.uses_squeeze(),
            shapes,
            streams,
        })
    }

    #[cfg(test)]
    pub(super) fn group(
        &self,
        group: LosslessModularGroup,
    ) -> Result<&GroupTransformPlan, EncodeError> {
        self.streams
            .iter()
            .find(|stream| stream.has_color() && stream.region.index == group.index)
            .map(|stream| &self.shapes[stream.shape])
            .ok_or_else(|| BackendError::Invariant("Modular group has no planned topology").into())
    }

    pub(super) fn stream(&self, index: u32) -> Result<&GroupTransformPlan, EncodeError> {
        self.streams
            .get(index as usize)
            .map(|stream| &self.shapes[stream.shape])
            .ok_or_else(|| BackendError::Invariant("Modular stream has no planned topology").into())
    }
}
