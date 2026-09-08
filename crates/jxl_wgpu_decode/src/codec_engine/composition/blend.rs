//! Lower independent color/extra blend selectors to one bounded per-channel GPU operation.

use crate::{Error, Result};
use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{ExtraChannelInventory, ExtraChannelTypeInventory, FrameInventory};
use jxl_gpu_protocol::Extent2d;

const _: () = assert!(std::mem::size_of::<BlendParams>() == 128);
const _: () = assert!(std::mem::align_of::<BlendParams>() == 16);
const _: () = assert!(std::mem::offset_of!(BlendParams, references) == 64);
const _: () = assert!(std::mem::size_of::<BlendChannel>() == 32);
const _: () = assert!(std::mem::align_of::<BlendChannel>() == 16);
const _: () = assert!(std::mem::offset_of!(BlendChannel, alpha) == 16);

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct BlendParams {
    pub canvas: [u32; 4],
    pub intersection: [u32; 4],
    pub source: [u32; 4],
    pub dispatch: [u32; 4],
    pub references: [[u32; 4]; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct BlendChannel {
    operation: [u32; 4], // mode, background slot, absolute alpha plane, clamp/association flags
    alpha: [u32; 4],     // alpha background slot, reserved
}

pub(super) fn blend_channels(
    frame: &FrameInventory,
    extras: &[ExtraChannelInventory],
) -> Result<Vec<BlendChannel>> {
    if frame.extra_channel_blends.len() != extras.len() {
        return Err(Error::EngineContract(
            "frame blend declarations disagree with extra channels",
        ));
    }
    for blend in std::iter::once(&frame.color_blend).chain(&frame.extra_channel_blends) {
        if blend.source >= 4
            || blend
                .alpha_channel
                .is_some_and(|index| index as usize >= extras.len())
        {
            return Err(Error::EngineContract(
                "frame blend selector exceeds its reference or channel range",
            ));
        }
    }
    let has_alpha = extras
        .iter()
        .any(|extra| matches!(extra.channel_type, ExtraChannelTypeInventory::Alpha { .. }));
    let color_alpha = frame.color_blend.alpha_channel.unwrap_or(0);
    (0..3 + extras.len())
        .map(|channel| {
            let blend = if channel < 3 {
                &frame.color_blend
            } else {
                &frame.extra_channel_blends[channel - 3]
            };
            let selector = blend.alpha_channel.unwrap_or(0) as usize;
            let mut mode = blend.mode as u32;
            let mut clamp = blend.clamp;
            let mut alpha_channel = 3 + selector as u32;
            if channel < 3 && !has_alpha {
                mode = match mode {
                    2 => 0,
                    3 => 1,
                    mode => mode,
                };
            } else if channel >= 3 && channel == 3 + selector {
                mode = match mode {
                    2 => 5,
                    3 => 6,
                    mode => mode,
                };
            }
            if has_alpha && frame.color_blend.mode as u32 == 2 && channel as u32 == 3 + color_alpha
            {
                mode = 5;
                alpha_channel = 3 + color_alpha;
                clamp = frame.color_blend.clamp;
            }
            let selected = alpha_channel.saturating_sub(3) as usize;
            if matches!(mode, 2 | 3 | 5) && selected >= extras.len() {
                return Err(Error::EngineContract(
                    "blend alpha channel is outside the frame surface",
                ));
            }
            let associated = extras.get(selected).is_some_and(|extra| {
                matches!(
                    extra.channel_type,
                    ExtraChannelTypeInventory::Alpha { associated: true }
                )
            });
            Ok(BlendChannel {
                operation: [
                    mode,
                    blend.source,
                    alpha_channel,
                    u32::from(clamp) | (u32::from(associated) << 1),
                ],
                alpha: [
                    frame
                        .extra_channel_blends
                        .get(selected)
                        .map_or(0, |blend| blend.source),
                    0,
                    0,
                    0,
                ],
            })
        })
        .collect()
}

pub(super) fn intersection(canvas: Extent2d, frame: &FrameInventory) -> ([u32; 4], [u32; 2]) {
    fn axis(limit: u32, origin: i32, size: u32) -> (u32, u32, u32) {
        let start = i64::from(origin).clamp(0, i64::from(limit));
        let end = (i64::from(origin) + i64::from(size)).clamp(start, i64::from(limit));
        let width = (end - start) as u32;
        let source = if width == 0 {
            0
        } else {
            (start - i64::from(origin)) as u32
        };
        (start as u32, width, source)
    }
    let (x, width, sx) = axis(canvas.width, frame.x0, frame.width);
    let (y, height, sy) = axis(canvas.height, frame.y0, frame.height);
    ([x, y, width, height], [sx, sy])
}
