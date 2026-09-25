//! Codec-independent frame control. Caller options are validated and lowered once, before
//! admission; GPU jobs retain this bounded immutable plan instead of reinterpreting options.
use crate::{
    AnimationHeader, BitFragment, BlendMode, EncodeError, FrameBlend, FrameEncodeRequest,
    FrameIndex, FrameKind, ProgressivePlan,
};
use crate::{extra_channel::sampling::ExtraChannelSamplingPlan, sample_format::ImageSamplePlan};
use jxl_gpu_bitstream::BitWriter;
use jxl_gpu_protocol::Extent2d;

/// Checked frame kind, scalar sampling, crop, blending, timing, references and restoration.
/// The suffix occupies at most 256 + 12 bits per extra channel, independent of pixels;
/// each extra's sampling factor adds two bits in the codec-specific prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FrameHeaderPlan {
    frame_index: FrameIndex,
    is_last: bool,
    kind: FrameKind,
    post_color_reference: bool,
    extras: ExtraChannelSamplingPlan,
    suffix: BitFragment,
}

impl FrameHeaderPlan {
    pub(crate) fn new(
        request: &FrameEncodeRequest,
        source_extent: (u32, u32),
        has_alpha: bool,
    ) -> Result<Self, EncodeError> {
        let extras = ExtraChannelSamplingPlan::for_packed_alpha(
            Extent2d::new(source_extent.0, source_extent.1),
            has_alpha,
            &request.options.extra_channel_upsampling,
        )?;
        Self::with_sampling(request, source_extent, extras)
    }

    pub(crate) fn with_extra_channels(
        request: &FrameEncodeRequest,
        source_extent: (u32, u32),
        samples: &ImageSamplePlan,
    ) -> Result<Self, EncodeError> {
        let extras = ExtraChannelSamplingPlan::for_image(
            samples,
            Extent2d::new(source_extent.0, source_extent.1),
            &request.options.extra_channel_upsampling,
        )?;
        Self::with_sampling(request, source_extent, extras)
    }

    fn with_sampling(
        request: &FrameEncodeRequest,
        source_extent: (u32, u32),
        extras: ExtraChannelSamplingPlan,
    ) -> Result<Self, EncodeError> {
        let extra_channels = extras.channels.len();
        validate_frame(request, source_extent, extra_channels)?;
        let has_alpha = extra_channels != 0;
        let regular = request.options.kind == FrameKind::Regular;
        let can_be_referenced = can_be_referenced(request);
        let mut output = BitWriter::new();
        let have_crop = request.options.crop.is_some();
        output.write_bits(u64::from(have_crop), 1)?;
        if let Some(crop) = request.options.crop {
            if regular {
                write_frame_dimension(&mut output, pack_signed(crop.x()))?;
                write_frame_dimension(&mut output, pack_signed(crop.y()))?;
            }
            write_frame_dimension(&mut output, crop.width())?;
            write_frame_dimension(&mut output, crop.height())?;
        }

        let full_frame = frame_covers_canvas(
            request.options.crop,
            request.canvas_width,
            request.canvas_height,
        );
        if regular {
            write_blending_info(
                &mut output,
                request.options.color_blend,
                has_alpha,
                full_frame,
            )?;
            for channel in 0..extra_channels {
                let alpha_blend = request
                    .options
                    .extra_channel_blends
                    .get(channel)
                    .copied()
                    .unwrap_or_default();
                write_blending_info(&mut output, alpha_blend, true, full_frame)?;
            }

            if let AnimationHeader::Animation { have_timecodes, .. } = request.animation {
                write_frame_duration(&mut output, request.options.timing.duration_ticks)?;
                if have_timecodes {
                    output.write_bits(
                        u64::from(request.options.timing.timecode.ok_or(
                            EncodeError::InvalidConfiguration(
                                "animated frame is missing its declared timecode",
                            ),
                        )?),
                        32,
                    )?;
                }
            }
            output.write_bits(u64::from(request.is_last), 1)?;
        }
        if !request.is_last {
            output.write_bits(u64::from(request.options.save_as_reference.get()), 2)?;
            if !regular
                || (request.options.color_blend.mode == BlendMode::Replace
                    && full_frame
                    && can_be_referenced)
            {
                output.write_bits(u64::from(request.options.save_before_color_transform), 1)?;
            }
        }

        output.write_bits(0, 2)?; // empty frame name
        output.write_bits(0, 1)?; // non-default restoration filter
        output.write_bits(0, 1)?; // no Gaborish
        output.write_bits(0, 2)?; // no EPF iterations
        output.write_bits(0, 2)?; // no restoration-filter extensions
        output.write_bits(0, 2)?; // no frame extensions
        let bit_len = output.bit_len();
        if bit_len > 256 + 12 * extra_channels {
            return Err(EncodeError::InvalidConfiguration(
                "frame control exceeds its fixed storage bound",
            ));
        }
        Ok(Self {
            frame_index: request.frame_index,
            is_last: request.is_last,
            kind: request.options.kind,
            post_color_reference: can_be_referenced && !request.options.save_before_color_transform,
            extras,
            suffix: BitFragment::new(output.into_bytes(), bit_len)?,
        })
    }

    pub(crate) fn extra_channels(&self) -> &ExtraChannelSamplingPlan {
        &self.extras
    }

    pub(crate) const fn frame_index(&self) -> FrameIndex {
        self.frame_index
    }

    pub(crate) const fn is_last(&self) -> bool {
        self.is_last
    }

    pub(crate) const fn requires_post_color_reference(&self) -> bool {
        self.post_color_reference
    }

    pub(crate) fn write_kind(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        writer.write_bits(
            match self.kind {
                FrameKind::Regular => 0,
                FrameKind::ReferenceOnly => 2,
            },
            2,
        )?;
        Ok(())
    }

    pub(crate) const fn has_passes(&self) -> bool {
        matches!(self.kind, FrameKind::Regular)
    }

    /// Reference-only syntax omits the pass bundle and implies one complete coefficient pass.
    pub(crate) fn effective_progressive(&self, requested: &ProgressivePlan) -> ProgressivePlan {
        if self.has_passes() {
            requested.clone()
        } else {
            ProgressivePlan::single()
        }
    }

    pub(crate) fn append_to(&self, writer: &mut BitWriter) -> Result<(), EncodeError> {
        crate::packet::append_fragment(writer, &self.suffix).map_err(Into::into)
    }
}

fn validate_frame(
    request: &FrameEncodeRequest,
    source_extent: (u32, u32),
    extra_channels: usize,
) -> Result<(), EncodeError> {
    let has_alpha = extra_channels != 0;
    if request.canvas_width == 0 || request.canvas_height == 0 {
        return Err(EncodeError::InvalidConfiguration(
            "the JPEG XL canvas must be non-empty",
        ));
    }
    if request.animation.is_animation() {
        write_animation_header(&mut BitWriter::new(), request.animation)?;
    }
    crate::session::validate_frame_timing(request.animation, &request.options)?;
    validate_extent(request, source_extent)?;
    if request.options.kind == FrameKind::ReferenceOnly {
        if request.is_last
            || request.options.color_blend != FrameBlend::default()
            || !request.options.extra_channel_blends.is_empty()
        {
            return Err(EncodeError::InvalidConfiguration(
                "reference-only frames cannot be final or carry timing/blending fields",
            ));
        }
        if request
            .options
            .crop
            .is_some_and(|crop| crop.x() != 0 || crop.y() != 0)
        {
            return Err(EncodeError::InvalidConfiguration(
                "reference-only frame origins must be zero",
            ));
        }
        if !request.options.save_before_color_transform
            && !frame_covers_canvas(
                request.options.crop,
                request.canvas_width,
                request.canvas_height,
            )
        {
            return Err(EncodeError::InvalidConfiguration(
                "post-color-transform reference-only frames must cover the canvas",
            ));
        }
        return Ok(());
    }
    for blend in
        std::iter::once(&request.options.color_blend).chain(&request.options.extra_channel_blends)
    {
        let weighted = matches!(blend.mode, BlendMode::Blend | BlendMode::MultiplyAdd);
        if (weighted
            && (blend.alpha_channel > 10 || blend.alpha_channel as usize >= extra_channels))
            || (!weighted && blend.alpha_channel != 0)
        {
            return Err(EncodeError::InvalidConfiguration(
                "blend alpha selector is absent or outside its channel/syntax bounds",
            ));
        }
    }
    if !request.options.extra_channel_blends.is_empty()
        && request.options.extra_channel_blends.len() != extra_channels
    {
        return Err(EncodeError::InvalidConfiguration(
            "extra-channel blend count does not match the source format",
        ));
    }
    if !has_alpha
        && matches!(
            request.options.color_blend.mode,
            crate::BlendMode::Blend | crate::BlendMode::MultiplyAdd
        )
    {
        return Err(EncodeError::InvalidConfiguration(
            "alpha-weighted blending requires an alpha source",
        ));
    }
    let color_uses_clamp = request.options.color_blend.mode == crate::BlendMode::Multiply
        || (has_alpha
            && matches!(
                request.options.color_blend.mode,
                crate::BlendMode::Blend | crate::BlendMode::MultiplyAdd
            ));
    if request.options.color_blend.clamp && !color_uses_clamp {
        return Err(EncodeError::InvalidConfiguration(
            "the selected JPEG XL color blend mode has no clamp field",
        ));
    }
    if request.options.extra_channel_blends.iter().any(|blend| {
        blend.clamp
            && !matches!(
                blend.mode,
                crate::BlendMode::Blend
                    | crate::BlendMode::MultiplyAdd
                    | crate::BlendMode::Multiply
            )
    }) {
        return Err(EncodeError::InvalidConfiguration(
            "the selected JPEG XL extra-channel blend mode has no clamp field",
        ));
    }
    if request.is_last && request.options.save_as_reference != Default::default() {
        return Err(EncodeError::InvalidConfiguration(
            "the final JPEG XL frame cannot be saved as a reference",
        ));
    }
    let full_frame = frame_covers_canvas(
        request.options.crop,
        request.canvas_width,
        request.canvas_height,
    );
    let resets_canvas = request.options.color_blend.mode == crate::BlendMode::Replace && full_frame;
    let can_be_referenced = can_be_referenced(request);
    let writes_save_before = resets_canvas && can_be_referenced;
    if request.options.save_before_color_transform && !writes_save_before {
        return Err(EncodeError::InvalidConfiguration(
            "save-before-color-transform is not present for this frame contract",
        ));
    }
    Ok(())
}

fn can_be_referenced(request: &FrameEncodeRequest) -> bool {
    !request.is_last
        && (request.options.kind == FrameKind::ReferenceOnly
            || request.options.timing.duration_ticks == 0
            || request.options.save_as_reference.get() != 0)
}

fn validate_extent(
    request: &FrameEncodeRequest,
    source_extent: (u32, u32),
) -> Result<(), EncodeError> {
    let (frame_width, frame_height) = request
        .options
        .crop
        .map_or((request.canvas_width, request.canvas_height), |crop| {
            (crop.width(), crop.height())
        });
    if let Some(crop) = request.options.crop {
        for value in [
            pack_signed(crop.x()),
            pack_signed(crop.y()),
            crop.width(),
            crop.height(),
        ] {
            if value >= 18_688 + (1 << 30) {
                return Err(EncodeError::InvalidConfiguration(
                    "frame crop coordinate exceeds the JPEG XL limit",
                ));
            }
        }
    }
    if frame_width != source_extent.0 || frame_height != source_extent.1 {
        return Err(EncodeError::InvalidConfiguration(
            "the GPU source extent must match the frame crop",
        ));
    }
    Ok(())
}

fn frame_covers_canvas(
    crop: Option<crate::FrameCrop>,
    canvas_width: u32,
    canvas_height: u32,
) -> bool {
    let Some(crop) = crop else {
        return true;
    };
    i64::from(crop.x()) <= 0
        && i64::from(crop.y()) <= 0
        && i64::from(crop.x()) + i64::from(crop.width()) >= i64::from(canvas_width)
        && i64::from(crop.y()) + i64::from(crop.height()) >= i64::from(canvas_height)
}

pub(crate) fn write_animation_header(
    output: &mut BitWriter,
    animation: AnimationHeader,
) -> Result<(), EncodeError> {
    let AnimationHeader::Animation {
        ticks_per_second_numerator,
        ticks_per_second_denominator,
        num_loops,
        have_timecodes,
    } = animation
    else {
        return Err(EncodeError::InvalidConfiguration(
            "animation metadata requires an animation header",
        ));
    };
    let numerator = ticks_per_second_numerator.get();
    match numerator {
        100 => output.write_bits(0, 2)?,
        1000 => output.write_bits(1, 2)?,
        1..=1024 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(numerator - 1), 10)?;
        }
        1025..=1_073_741_824 => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(numerator - 1), 30)?;
        }
        _ => {
            return Err(EncodeError::InvalidConfiguration(
                "animation ticks-per-second numerator exceeds the JPEG XL limit",
            ));
        }
    }
    let denominator = ticks_per_second_denominator.get();
    match denominator {
        1 => output.write_bits(0, 2)?,
        1001 => output.write_bits(1, 2)?,
        2..=256 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(denominator - 1), 8)?;
        }
        257..=1024 => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(denominator - 1), 10)?;
        }
        _ => {
            return Err(EncodeError::InvalidConfiguration(
                "animation ticks-per-second denominator exceeds the JPEG XL limit",
            ));
        }
    }
    match num_loops {
        0 => output.write_bits(0, 2)?,
        1..=7 => {
            output.write_bits(1, 2)?;
            output.write_bits(u64::from(num_loops), 3)?;
        }
        8..=65_535 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(num_loops), 16)?;
        }
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(num_loops), 32)?;
        }
    }
    output.write_bits(u64::from(have_timecodes), 1)?;
    Ok(())
}

fn write_blending_info(
    output: &mut BitWriter,
    blend: FrameBlend,
    has_alpha: bool,
    full_frame: bool,
) -> Result<(), EncodeError> {
    write_blend_mode(output, blend.mode)?;
    let uses_alpha = matches!(blend.mode, BlendMode::Blend | BlendMode::MultiplyAdd);
    if has_alpha && uses_alpha {
        match blend.alpha_channel {
            0..=2 => output.write_bits(u64::from(blend.alpha_channel), 2)?,
            index => {
                output.write_bits(3, 2)?;
                output.write_bits(u64::from(index - 3), 3)?;
            }
        }
    }
    if (has_alpha && uses_alpha) || blend.mode == BlendMode::Multiply {
        output.write_bits(u64::from(blend.clamp), 1)?;
    } else if blend.clamp {
        return Err(EncodeError::InvalidConfiguration(
            "the selected JPEG XL blend mode has no clamp field",
        ));
    }
    // Each channel's own mode determines whether its reference source is present.
    if blend.mode != BlendMode::Replace || !full_frame {
        output.write_bits(u64::from(blend.source_reference.get()), 2)?;
    }
    Ok(())
}

fn write_blend_mode(output: &mut BitWriter, mode: BlendMode) -> Result<(), EncodeError> {
    match mode {
        BlendMode::Replace => output.write_bits(0, 2)?,
        BlendMode::Add => output.write_bits(1, 2)?,
        BlendMode::Blend => output.write_bits(2, 2)?,
        BlendMode::MultiplyAdd => {
            output.write_bits(3, 2)?;
            output.write_bits(0, 2)?;
        }
        BlendMode::Multiply => {
            output.write_bits(3, 2)?;
            output.write_bits(1, 2)?;
        }
    }
    Ok(())
}

pub(crate) fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        (u32::try_from(-i64::from(value)).expect("an i32 magnitude fits u32") << 1).wrapping_sub(1)
    }
}

fn write_frame_dimension(output: &mut BitWriter, value: u32) -> Result<(), EncodeError> {
    let (selector, offset, bits) = if value < 256 {
        (0, 0, 8)
    } else if value < 2_304 {
        (1, 256, 11)
    } else if value < 18_688 {
        (2, 2_304, 14)
    } else if value < 18_688 + (1 << 30) {
        (3, 18_688, 30)
    } else {
        return Err(EncodeError::InvalidConfiguration(
            "frame crop coordinate exceeds the JPEG XL limit",
        ));
    };
    output.write_bits(selector, 2)?;
    output.write_bits(u64::from(value - offset), bits)?;
    Ok(())
}

fn write_frame_duration(output: &mut BitWriter, duration: u32) -> Result<(), EncodeError> {
    match duration {
        0 => output.write_bits(0, 2)?,
        1 => output.write_bits(1, 2)?,
        2..=255 => {
            output.write_bits(2, 2)?;
            output.write_bits(u64::from(duration), 8)?;
        }
        _ => {
            output.write_bits(3, 2)?;
            output.write_bits(u64::from(duration), 32)?;
        }
    }
    Ok(())
}
