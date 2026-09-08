//! Declaration-order spot inks, applied only to the presentation after reference storage.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{ExtraChannelInventory, ExtraChannelTypeInventory};

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct SpotColor {
    source: [u32; 4],
    rgba: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SpotColor>() == 32);
const _: () = assert!(std::mem::align_of::<SpotColor>() == 16);
const _: () = assert!(std::mem::offset_of!(SpotColor, rgba) == 16);

pub(super) fn spot_colors(extras: &[ExtraChannelInventory], plane_words: u32) -> Vec<SpotColor> {
    extras
        .iter()
        .enumerate()
        .filter_map(|(index, extra)| {
            let ExtraChannelTypeInventory::SpotColour {
                red,
                green,
                blue,
                solidity,
            } = extra.channel_type
            else {
                return None;
            };
            Some(SpotColor {
                source: [(3 + index as u32) * plane_words, 0, 0, 0],
                rgba: [
                    red.to_f32(),
                    green.to_f32(),
                    blue.to_f32(),
                    solidity.to_f32(),
                ],
            })
        })
        .collect()
}

pub(super) fn shader(enabled: bool) -> &'static str {
    if enabled {
        include_str!("spot.wgsl")
    } else {
        "fn present_rgb(rgb: vec3<f32>, position: u32) -> vec3<f32> { return rgb; }"
    }
}
