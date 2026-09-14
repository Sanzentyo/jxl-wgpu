//! Declaration-order spot inks, applied only to the presentation after reference storage.

use bytemuck::{Pod, Zeroable};
use jxl_gpu_bitstream::{ExtraChannelInventory, ExtraChannelTypeInventory};
use jxl_gpu_formats::ImageLayout;

pub(super) mod render;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct SpotColor {
    source: [u32; 4],
    rgba: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SpotColor>() == 32);
const _: () = assert!(std::mem::align_of::<SpotColor>() == 16);
const _: () = assert!(std::mem::offset_of!(SpotColor, rgba) == 16);

pub(super) fn spot_colors(
    extras: &[ExtraChannelInventory],
    planes: &[ImageLayout],
) -> Vec<SpotColor> {
    assert_eq!(extras.len(), planes.len());
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
                source: [(planes[index].planes[0].offset / 4) as u32, 0, 0, 0],
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
        "fn present_rgb_words(rgb: vec3<u32>, position: u32) -> vec3<u32> { return rgb; }"
    }
}
