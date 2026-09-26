use crate::EncodeError;
use std::sync::Arc;

/// Squeeze within each Modular pass group, after RCT and Palette.
/// The named policies select one axis or both axis orders. By default they transform every
/// image channel and append residuals at the end; Palette's meta channel is always excluded.
/// Selection is in the post-Palette image-channel domain, before any Squeeze. Both axes act
/// on all descendants of the selected channels. Axes of length one are skipped per group.
/// No cross-group filtering or image analysis is performed. `None` preserves unsqueezed bytes.
/// [`Self::sequence`] instead applies ordered steps to the changing channel list, including
/// repeated axes and zero-sized residual slots. Its immutable step storage is shared on clone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessModularSqueeze {
    policy: SqueezePolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SqueezePolicy {
    Separable {
        axes: SqueezeAxes,
        channel_range: Option<(u32, u32)>,
        in_place: bool,
    },
    Sequence(Arc<[LosslessModularSqueezeStep]>),
}

impl Default for LosslessModularSqueeze {
    fn default() -> Self {
        Self::None
    }
}

/// One explicit Squeeze parameter applied to the current image-channel list, excluding metadata.
/// Unlike named separable policies, explicit steps retain zero-sized residual channels on
/// one-pixel axes. Later steps may not target an empty channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessModularSqueezeStep {
    pub(super) horizontal: bool,
    pub(super) begin: u32,
    pub(super) count: u32,
    pub(super) in_place: bool,
}

impl LosslessModularSqueezeStep {
    /// Validates the wire range (begin at most 9,287; count in 1..=19).
    /// Actual topology bounds are checked for every group before GPU admission.
    pub fn new(
        horizontal: bool,
        begin: u32,
        count: u32,
        in_place: bool,
    ) -> Result<Self, EncodeError> {
        if begin > 9287 || !(1..=19).contains(&count) {
            return Err(EncodeError::InvalidModularSqueezeStep { begin, count });
        }
        Ok(Self {
            horizontal,
            begin,
            count,
            in_place,
        })
    }

    #[must_use]
    pub const fn horizontal(self) -> bool {
        self.horizontal
    }
    #[must_use]
    pub const fn in_place(self) -> bool {
        self.in_place
    }
    #[must_use]
    pub fn channel_range(self) -> std::ops::Range<u32> {
        self.begin..self.begin + self.count
    }
}

#[expect(
    non_upper_case_globals,
    reason = "preserve the established named Squeeze policies"
)]
impl LosslessModularSqueeze {
    pub const None: Self = Self::new(SqueezeAxes::None);
    pub const Horizontal: Self = Self::new(SqueezeAxes::Horizontal);
    pub const Vertical: Self = Self::new(SqueezeAxes::Vertical);
    pub const HorizontalThenVertical: Self = Self::new(SqueezeAxes::HorizontalThenVertical);
    pub const VerticalThenHorizontal: Self = Self::new(SqueezeAxes::VerticalThenHorizontal);

    const fn new(axes: SqueezeAxes) -> Self {
        Self {
            policy: SqueezePolicy::Separable {
                axes,
                channel_range: None,
                in_place: false,
            },
        }
    }

    /// Selects 1–296 ordered explicit parameters. Ranges address each step's current image
    /// topology after preceding steps, rather than the original source components.
    /// Each group's range, nonempty inputs and cumulative shifts (at most 30 on both axes
    /// before a step) are validated before GPU admission.
    pub fn sequence(
        steps: impl Into<Arc<[LosslessModularSqueezeStep]>>,
    ) -> Result<Self, EncodeError> {
        let steps = steps.into();
        if !(1..=296).contains(&steps.len()) {
            return Err(EncodeError::InvalidModularSqueezeStepCount { count: steps.len() });
        }
        Ok(Self {
            policy: SqueezePolicy::Sequence(steps),
        })
    }

    #[must_use]
    pub fn steps(&self) -> Option<&[LosslessModularSqueezeStep]> {
        match &self.policy {
            SqueezePolicy::Sequence(steps) => Some(steps),
            SqueezePolicy::Separable { .. } => None,
        }
    }

    /// Selects a nonempty contiguous range of image channels after RCT and optional Palette.
    /// Channel zero is the first image channel, excluding Palette's table. Unselected channels
    /// keep their dimensions and samples. The range must fit the image input limit and each
    /// color stream's actual post-Palette topology, including its independent scalar inputs.
    /// Scalar channels routed to LF streams are outside this color-local selection.
    pub fn with_channels(mut self, begin: u32, count: u32) -> Result<Self, EncodeError> {
        validate_range(
            begin,
            count,
            3 + crate::extra_channel::MAX_EXTRA_CHANNELS as u32,
        )?;
        let SqueezePolicy::Separable { channel_range, .. } = &mut self.policy else {
            return Err(EncodeError::InvalidConfiguration(
                "explicit Squeeze sequences specify a range in each step",
            ));
        };
        *channel_range = Some((begin, count));
        Ok(self)
    }

    /// Places each step's residuals immediately after that step's selected range when true;
    /// otherwise appends them to the end of the channel list. This is JPEG XL's `in_place`
    /// channel ordering, not a promise about GPU buffer aliasing.
    #[must_use]
    pub fn with_in_place(mut self, in_place: bool) -> Self {
        match &mut self.policy {
            SqueezePolicy::Separable {
                in_place: placement,
                ..
            } => *placement = in_place,
            SqueezePolicy::Sequence(steps) => {
                for step in Arc::make_mut(steps) {
                    step.in_place = in_place;
                }
            }
        }
        self
    }

    /// The named policy's explicit post-Palette range, or `None` for its all-channel default.
    /// Returns `None` for sequences; inspect [`Self::steps`] for their per-step ranges.
    #[must_use]
    pub fn channel_range(&self) -> Option<std::ops::Range<u32>> {
        match self.policy {
            SqueezePolicy::Separable { channel_range, .. } => {
                channel_range.map(|(begin, count)| begin..begin + count)
            }
            SqueezePolicy::Sequence(_) => None,
        }
    }

    /// The named policy's residual placement. Sequences return `None` because placement is
    /// specified independently in each step.
    #[must_use]
    pub fn in_place(&self) -> Option<bool> {
        match self.policy {
            SqueezePolicy::Separable { in_place, .. } => Some(in_place),
            SqueezePolicy::Sequence(_) => None,
        }
    }

    pub(super) fn resolve_range(&self, channels: u32) -> Result<std::ops::Range<u32>, EncodeError> {
        let range = self.channel_range().unwrap_or(0..channels);
        let (begin, count) = (range.start, range.end - range.start);
        validate_range(begin, count, channels)?;
        Ok(begin..begin + count)
    }

    pub(super) fn for_extent(&self, width: u32, height: u32) -> SqueezeAxes {
        match self.policy {
            SqueezePolicy::Separable { axes, .. } => axes.for_extent(width, height),
            SqueezePolicy::Sequence(_) => SqueezeAxes::None,
        }
    }

    pub(super) fn stages(&self) -> u32 {
        self.for_extent(2, 2).stages()
    }

    pub(super) fn enabled(&self) -> bool {
        self.steps().is_some() || self.stages() != 0
    }
}

fn validate_range(begin: u32, count: u32, channels: u32) -> Result<(), EncodeError> {
    if count != 0 && begin.checked_add(count).is_some_and(|end| end <= channels) {
        Ok(())
    } else {
        Err(EncodeError::InvalidModularSqueezeChannels {
            begin,
            count,
            channels,
        })
    }
}

/// Resolved axes for a single image-channel lineage; also the kernel's mode encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum SqueezeAxes {
    #[default]
    None = 0,
    Horizontal = 1,
    Vertical = 2,
    HorizontalThenVertical = 3,
    VerticalThenHorizontal = 4,
}

impl SqueezeAxes {
    const fn for_extent(self, width: u32, height: u32) -> Self {
        match self {
            Self::Horizontal if width == 1 => Self::None,
            Self::Vertical if height == 1 => Self::None,
            Self::HorizontalThenVertical | Self::VerticalThenHorizontal => {
                match (width > 1, height > 1) {
                    (true, true) => self,
                    (true, false) => Self::Horizontal,
                    (false, true) => Self::Vertical,
                    (false, false) => Self::None,
                }
            }
            _ => self,
        }
    }

    pub(super) const fn stages(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Horizontal | Self::Vertical => 1,
            Self::HorizontalThenVertical | Self::VerticalThenHorizontal => 2,
        }
    }

    pub(super) const fn first_horizontal(self) -> bool {
        matches!(self, Self::Horizontal | Self::HorizontalThenVertical)
    }
}
