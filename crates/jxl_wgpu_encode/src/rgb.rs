//! Checked RGB image contract shared by VarDCT and mixed-codec sequences.

use crate::frame_header::write_animation_header;
use crate::sample_format::write_sample_bit_depth;
use crate::{AnimationHeader, BitFragment, EncodeError, RgbSampleFormat};
use jxl_gpu_bitstream::BitWriter;

/// Stream-wide canvas and optional timebase for an RGB sRGB/D65 layered still or animation.
///
/// Every frame uses the encoder's configured integer RGB precision and sRGB/D65 sources. The encoder binds the image's coding domain:
/// XYB or original RGB for VarDCT sequences, original RGB for mixed-codec sequences.
/// Modular and tiled DCT8 sources may vary in extent; a single VarDCT transform or checked
/// strategy map constrains the source extent of that codec's frames only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbSequenceDescriptor {
    canvas_width: u32,
    canvas_height: u32,
    animation: AnimationHeader,
    header: ImageHeaderPlan,
}

impl RgbSequenceDescriptor {
    /// Checks the canvas/timebase. The encoder binds its color policy at `begin_sequence`.
    pub fn new(
        canvas_width: u32,
        canvas_height: u32,
        animation: AnimationHeader,
    ) -> Result<Self, EncodeError> {
        let header = ImageHeaderPlan::new(canvas_width, canvas_height, animation)?;
        Ok(Self {
            canvas_width,
            canvas_height,
            animation,
            header,
        })
    }

    pub(crate) fn image_header(
        &self,
        xyb_encoded: bool,
        samples: RgbSampleFormat,
    ) -> Result<crate::BitFragment, EncodeError> {
        self.header.encode(xyb_encoded, samples)
    }

    #[must_use]
    pub const fn canvas_width(&self) -> u32 {
        self.canvas_width
    }

    #[must_use]
    pub const fn canvas_height(&self) -> u32 {
        self.canvas_height
    }

    #[must_use]
    pub const fn animation(&self) -> AnimationHeader {
        self.animation
    }
}

fn write_size(output: &mut BitWriter, size: u32, ratio: bool) -> Result<(), EncodeError> {
    if !(1..(1 << 30)).contains(&size) {
        return Err(EncodeError::InvalidConfiguration(
            "RGB image dimensions must be in 1..2^30",
        ));
    }
    let value = size - 1;
    let (selector, bits) = if value < 1 << 9 {
        (0, 9)
    } else if value < 1 << 13 {
        (1, 13)
    } else if value < 1 << 18 {
        (2, 18)
    } else {
        (3, 30)
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value), bits)?;
    if ratio {
        output.write_bits(0, 3)?;
    }
    Ok(())
}

/// Checked geometry/timebase, bound to the backend's color plan when a sequence begins.
/// Geometry and timebase are independent of the selected frame codecs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ImageHeaderPlan {
    prefix: BitFragment,
    animation: Option<BitFragment>,
}

impl ImageHeaderPlan {
    fn new(width: u32, height: u32, animation: AnimationHeader) -> Result<Self, EncodeError> {
        let fragment = |writer: BitWriter| {
            let bits = writer.bit_len();
            BitFragment::new(writer.into_bytes(), bits).map_err(EncodeError::from)
        };
        let mut prefix = BitWriter::new();
        prefix.write_bits(0x0aff, 16)?;
        prefix.write_bits(0, 1)?; // dimensions are not multiples of eight
        write_size(&mut prefix, height, true)?;
        write_size(&mut prefix, width, false)?;
        let animation = if animation.is_animation() {
            let mut writer = BitWriter::new();
            write_animation_header(&mut writer, animation)?;
            Some(fragment(writer)?)
        } else {
            None
        };
        Ok(Self {
            prefix: fragment(prefix)?,
            animation,
        })
    }

    fn encode(
        &self,
        xyb_encoded: bool,
        samples: RgbSampleFormat,
    ) -> Result<BitFragment, EncodeError> {
        let mut output = BitWriter::new();
        crate::packet::append_fragment(&mut output, &self.prefix)?;
        let has_animation = self.animation.is_some();
        output.write_bits(0, 1)?; // explicit image metadata
        output.write_bits(u64::from(has_animation), 1)?;
        if let Some(animation) = &self.animation {
            output.write_bits(0, 3)?; // identity orientation
            output.write_bits(0, 1)?; // no intrinsic size
            output.write_bits(0, 1)?; // no preview
            output.write_bits(1, 1)?; // animation present
            crate::packet::append_fragment(&mut output, animation)?;
        }
        write_sample_bit_depth(&mut output, samples.bits_per_sample(), 0)?;
        // VarDCT LF coefficients are checked i32 values independently of input depth.
        // Mixed sequences must retain that same image-wide working-buffer contract.
        output.write_bits(0, 1)?; // 32-bit Modular buffers
        output.write_bits(0, 2)?; // no extra channels
        output.write_bits(u64::from(xyb_encoded), 1)?;
        output.write_bits(1, 1)?; // default sRGB presentation
        if has_animation {
            output.write_bits(1, 1)?; // default tone mapping (present only with extra fields)
        }
        output.write_bits(0, 2)?; // no image extensions
        output.write_bits(1, 1)?; // default opsin inverse matrix and upsampling weights
        output.align_to_byte()?;
        Ok(BitFragment::byte_aligned(output.into_bytes())?)
    }
}

/// Compatibility name for [`RgbSequenceDescriptor`]; precision is bound by the encoder.
pub type Rgb8SequenceDescriptor = RgbSequenceDescriptor;
