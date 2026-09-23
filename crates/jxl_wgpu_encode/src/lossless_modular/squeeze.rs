use crate::EncodeError;

/// Explicit separable Squeeze within each Modular pass group, after RCT and Palette.
/// The named policies select one axis or both axis orders. By default they transform every
/// image channel and append residuals at the end; Palette's meta channel is always excluded.
/// Selection is in the post-Palette image-channel domain, before any Squeeze. Both axes act
/// on all descendants of the selected channels. Axes of length one are skipped per group.
/// No cross-group filtering or image analysis is performed. `None` preserves unsqueezed bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LosslessModularSqueeze {
    axes: SqueezeAxes,
    channel_range: Option<(u32, u32)>,
    in_place: bool,
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
            axes,
            channel_range: None,
            in_place: false,
        }
    }

    /// Selects a nonempty contiguous range of image channels after RCT and optional Palette.
    /// Channel zero is the first image channel, excluding Palette's table. Unselected channels
    /// keep their dimensions and samples. The range must fit both the four-channel policy
    /// domain and the actual post-Palette image topology of every submitted source.
    pub fn with_channels(mut self, begin: u32, count: u32) -> Result<Self, EncodeError> {
        validate_range(begin, count, 4)?;
        self.channel_range = Some((begin, count));
        Ok(self)
    }

    /// Places each step's residuals immediately after that step's selected range when true;
    /// otherwise appends them to the end of the channel list. This is JPEG XL's `in_place`
    /// channel ordering, not a promise about GPU buffer aliasing.
    #[must_use]
    pub const fn with_in_place(mut self, in_place: bool) -> Self {
        self.in_place = in_place;
        self
    }

    /// Explicit post-Palette image-channel range, or `None` for all image channels.
    #[must_use]
    pub fn channel_range(self) -> Option<std::ops::Range<u32>> {
        self.channel_range
            .map(|(begin, count)| begin..begin + count)
    }

    #[must_use]
    pub const fn in_place(self) -> bool {
        self.in_place
    }

    pub(super) fn resolve_range(self, channels: u32) -> Result<std::ops::Range<u32>, EncodeError> {
        let (begin, count) = self.channel_range.unwrap_or((0, channels));
        validate_range(begin, count, channels)?;
        Ok(begin..begin + count)
    }

    pub(super) const fn for_extent(self, width: u32, height: u32) -> SqueezeAxes {
        self.axes.for_extent(width, height)
    }

    pub(super) const fn stages(self) -> u32 {
        self.axes.stages()
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
