//! Checked image-wide declarations for losslessly encoded scalar planes.

use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;

use crate::{AlphaAssociation, EncodeError, FiniteF16, SamplePrecision, UpsamplingFactor};

pub(crate) mod sampling;

// JPEG XL level 10 limits the wire-representable channel count to 256.
pub(crate) const MAX_EXTRA_CHANNELS: usize = 256;

/// Standard extra-channel semantics. Spot values retain exact binary16 metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExtraChannelKind {
    Alpha(AlphaAssociation),
    Depth,
    SpotColor { rgba: [FiniteF16; 4] },
    SelectionMask,
    Black,
    Cfa { channel: u32 },
    Thermal,
    Optional,
}

/// One scalar source, with independently checked precision and intrinsic sampling.
/// The default source extent is `ceil(frame_extent / 2^dimension_shift)` on each axis.
/// [`crate::FrameOptions::extra_channel_upsampling`] can further reduce it per frame.
/// Its raw samples are compressed losslessly; presentation upsampling belongs to the decoder.
/// Shifts are 0..=3. Names must be UTF-8 and at most 1071 bytes.
///
/// ```
/// use jxl_wgpu_encode::{ExtraChannel, ExtraChannelKind, UpsamplingFactor, SamplePrecision};
/// use jxl_gpu_protocol::Extent2d;
/// let depth = ExtraChannel::new(ExtraChannelKind::Depth,
///     SamplePrecision::integer(13)?, 1, b"depth".to_vec())?;
/// assert_eq!(depth.source_extent(Extent2d::new(17, 13)), Extent2d::new(9, 7));
/// assert_eq!(depth.source_extent_with_upsampling(Extent2d::new(17, 13),
///     UpsamplingFactor::Four), Extent2d::new(3, 2));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExtraChannel {
    kind: ExtraChannelKind,
    precision: SamplePrecision,
    dimension_shift: u8,
    name: Vec<u8>,
}

impl ExtraChannel {
    pub fn new(
        kind: ExtraChannelKind,
        precision: SamplePrecision,
        dimension_shift: u8,
        name: Vec<u8>,
    ) -> Result<Self, EncodeError> {
        if dimension_shift > 3 || name.len() > 1071 || std::str::from_utf8(&name).is_err() {
            return Err(EncodeError::InvalidConfiguration(
                "extra channels require shifts 0..=3 and UTF-8 names of at most 1071 bytes",
            ));
        }
        if matches!(kind, ExtraChannelKind::Cfa { channel } if channel > 274) {
            return Err(EncodeError::InvalidConfiguration(
                "CFA channel exceeds the JPEG XL syntax",
            ));
        }
        Ok(Self {
            kind,
            precision,
            dimension_shift,
            name,
        })
    }

    #[must_use]
    pub const fn kind(&self) -> ExtraChannelKind {
        self.kind
    }

    #[must_use]
    pub const fn precision(&self) -> SamplePrecision {
        self.precision
    }

    #[must_use]
    pub const fn dimension_shift(&self) -> u8 {
        self.dimension_shift
    }

    #[must_use]
    pub fn name(&self) -> &[u8] {
        &self.name
    }

    #[must_use]
    pub fn source_extent(&self, frame: Extent2d) -> Extent2d {
        self.source_extent_with_upsampling(frame, UpsamplingFactor::One)
    }

    /// Required supplied extent after both intrinsic and per-frame sampling factors.
    #[must_use]
    pub fn source_extent_with_upsampling(
        &self,
        frame: Extent2d,
        upsampling: UpsamplingFactor,
    ) -> Extent2d {
        let factor = upsampling.factor() << self.dimension_shift;
        Extent2d::new(frame.width.div_ceil(factor), frame.height.div_ceil(factor))
    }

    pub(crate) fn packed_alpha(precision: SamplePrecision, association: AlphaAssociation) -> Self {
        Self {
            kind: ExtraChannelKind::Alpha(association),
            precision,
            dimension_shift: 0,
            name: Vec::new(),
        }
    }

    pub(crate) fn write(&self, output: &mut BitWriter) -> Result<(), EncodeError> {
        let samples = self.precision.color(crate::ColorChannels::Gray);
        let default = self.kind == ExtraChannelKind::Alpha(AlphaAssociation::Unassociated)
            && samples.bits_per_sample() == 8
            && samples.exponent_bits() == 0
            && self.dimension_shift == 0
            && self.name.is_empty();
        output.write_bits(u64::from(default), 1)?;
        if default {
            return Ok(());
        }
        let kind = match self.kind {
            ExtraChannelKind::Alpha(_) => 0,
            ExtraChannelKind::Depth => 1,
            ExtraChannelKind::SpotColor { .. } => 2,
            ExtraChannelKind::SelectionMask => 3,
            ExtraChannelKind::Black => 4,
            ExtraChannelKind::Cfa { .. } => 5,
            ExtraChannelKind::Thermal => 6,
            ExtraChannelKind::Optional => 16,
        };
        // Enum U32(0, 1, 2 + u4, 18 + u6).
        match kind {
            0..=1 => output.write_bits(kind, 2)?,
            _ => {
                output.write_bits(2, 2)?;
                output.write_bits(kind - 2, 4)?;
            }
        }
        crate::sample_format::write_sample_bit_depth(
            output,
            samples.bits_per_sample(),
            samples.exponent_bits(),
        )?;
        match self.dimension_shift {
            0 => output.write_bits(0, 2)?,
            3 => output.write_bits(1, 2)?,
            shift => {
                output.write_bits(3, 2)?;
                output.write_bits(u64::from(shift - 1), 3)?;
            }
        }
        let length = self.name.len() as u64;
        match length {
            0 => output.write_bits(0, 2)?,
            1..=15 => {
                output.write_bits(1, 2)?;
                output.write_bits(length, 4)?;
            }
            16..=47 => {
                output.write_bits(2, 2)?;
                output.write_bits(length - 16, 5)?;
            }
            _ => {
                output.write_bits(3, 2)?;
                output.write_bits(length - 48, 10)?;
            }
        }
        for &byte in &self.name {
            output.write_bits(u64::from(byte), 8)?;
        }
        match self.kind {
            ExtraChannelKind::Alpha(association) => {
                output.write_bits(u64::from(association == AlphaAssociation::Associated), 1)?
            }
            ExtraChannelKind::SpotColor { rgba } => {
                for component in rgba {
                    output.write_bits(u64::from(component.to_bits()), 16)?;
                }
            }
            ExtraChannelKind::Cfa { channel } => match channel {
                1 => output.write_bits(0, 2)?,
                0..=3 => {
                    output.write_bits(1, 2)?;
                    output.write_bits(u64::from(channel), 2)?;
                }
                4..=18 => {
                    output.write_bits(2, 2)?;
                    output.write_bits(u64::from(channel - 3), 4)?;
                }
                _ => {
                    output.write_bits(3, 2)?;
                    output.write_bits(u64::from(channel - 19), 8)?;
                }
            },
            _ => {}
        }
        Ok(())
    }
}

pub(crate) fn write_count(output: &mut BitWriter, count: usize) -> Result<(), EncodeError> {
    match count {
        0..=1 => output.write_bits(count as u64, 2)?,
        2..=17 => {
            output.write_bits(2, 2)?;
            output.write_bits(count as u64 - 2, 4)?;
        }
        18..=MAX_EXTRA_CHANNELS => {
            output.write_bits(3, 2)?;
            output.write_bits(count as u64 - 1, 12)?;
        }
        _ => {
            return Err(EncodeError::InvalidConfiguration(
                "extra-channel count exceeds the JPEG XL profile limit",
            ));
        }
    }
    Ok(())
}
