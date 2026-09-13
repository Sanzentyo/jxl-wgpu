use std::sync::Arc;

use jxl_gpu_protocol::{Extent2d, OutputOrientation, RenderIntent, RenderOp};
use jxl_wgpu::{
    ColorSpace, ImageLayout, ImageOutputRequest, PixelFormat, RgbChannelOrder, RgbColorEncoding,
    TransferFunction,
};

use super::{backend, enqueue, frame_desc, plan, rgb_color};

#[test]
fn identity_f32_packing_preserves_ieee_words_in_every_orientation_and_channel_order() {
    let Some(backend) = backend() else {
        return;
    };
    // Integer representations include both signs of zero, subnormals, infinities,
    // and NaNs with distinct signs and signaling/payload bits.
    let words = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x8000_0001,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3e80_0000,
        0xc000_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc4_5678,
        0x7f80_0042,
        0xff80_0246,
    ];
    let extent = Extent2d::new(3, 2);
    let channels = std::array::from_fn(|channel| {
        words[channel * 6..(channel + 1) * 6]
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>()
    });
    for (exif, pixels) in [
        (1, [0, 1, 2, 3, 4, 5]),
        (2, [2, 1, 0, 5, 4, 3]),
        (3, [5, 4, 3, 2, 1, 0]),
        (4, [3, 4, 5, 0, 1, 2]),
        (5, [0, 3, 1, 4, 2, 5]),
        (6, [3, 0, 4, 1, 5, 2]),
        (7, [5, 2, 4, 1, 3, 0]),
        (8, [2, 5, 1, 4, 0, 3]),
    ] {
        let orientation = OutputOrientation::from_exif_value(exif).unwrap();
        let output_extent = if exif < 5 {
            extent
        } else {
            Extent2d::new(2, 3)
        };
        for (order, stored_channels) in [
            (RgbChannelOrder::Rgb, &[0, 1, 2][..]),
            (RgbChannelOrder::Bgr, &[2, 1, 0][..]),
            (RgbChannelOrder::Rgba, &[0, 1, 2, 3][..]),
            (RgbChannelOrder::Bgra, &[2, 1, 0, 3][..]),
        ] {
            for planar in [false, true] {
                let format = PixelFormat::rgb_f32(
                    order,
                    planar,
                    rgb_color(ColorSpace::Bt709, TransferFunction::Linear),
                );
                let mut render = plan(extent, RgbColorEncoding::LINEAR_BT709);
                let render_mut = Arc::get_mut(&mut render).unwrap();
                let RenderOp::Save(save) = &mut render_mut.nodes[0].op else {
                    unreachable!()
                };
                save.orientation = orientation;
                render_mut.outputs[0].extent = output_extent;
                let mut session = backend.create_session(&frame_desc(extent), render).unwrap();
                enqueue(&mut session, extent, &channels);
                let token = session
                    .submit_image(
                        RenderIntent::Final,
                        ImageOutputRequest::new(RgbColorEncoding::LINEAR_BT709, format.clone()),
                    )
                    .unwrap();
                let actual = session.wait_image(token).unwrap().outputs.remove(0);
                let layout = ImageLayout::packed(output_extent, format).unwrap();
                assert_eq!(actual.layout, layout);
                let mut expected = vec![0; layout.logical_size as usize];
                for (pixel, source) in pixels.into_iter().enumerate() {
                    for (stored, &channel) in stored_channels.iter().enumerate() {
                        let word = if channel == 3 {
                            0x3f80_0000_u32
                        } else {
                            words[channel * 6 + source]
                        };
                        let plane = &layout.planes[if planar { stored } else { 0 }];
                        let stride = if planar { 4 } else { stored_channels.len() * 4 };
                        let offset = plane.offset as usize
                            + pixel / output_extent.width as usize * plane.row_stride as usize
                            + pixel % output_extent.width as usize * stride
                            + if planar { 0 } else { stored * 4 };
                        expected[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
                    }
                }
                assert_eq!(actual.bytes, expected, "{orientation:?} {order:?} {planar}");
            }
        }
    }
}
