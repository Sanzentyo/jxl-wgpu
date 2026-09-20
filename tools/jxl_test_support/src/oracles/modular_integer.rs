//! Exact original integer planes from jxl-oxide, before any floating-point conversion.

/// Decode one original-color Modular presentation with its declared color encoding.
///
/// This oracle intentionally requires retained integer planes. A filtered or composed F32
/// result cannot prove exact high-depth source words and is rejected instead of rounded back.
pub fn original_planes(data: &[u8], frame_index: usize) -> Vec<Vec<i32>> {
    let mut image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    assert!(!image.image_header().metadata.xyb_encoded);
    let jxl_image::color::ColourEncoding::Enum(encoding) =
        &image.image_header().metadata.colour_encoding
    else {
        panic!("integer oracle requires an enumerated original color encoding")
    };
    image.request_color_encoding(encoding.clone());
    image.set_render_spot_color(false);
    let render = image.render_frame(frame_index).unwrap();
    render
        .color_channels()
        .iter()
        .chain(render.extra_channels().1)
        .map(|channel| match channel {
            jxl_render::ImageBuffer::I32(grid) => grid.buf().to_vec(),
            jxl_render::ImageBuffer::I16(grid) => {
                grid.buf().iter().map(|&v| i32::from(v)).collect()
            }
            jxl_render::ImageBuffer::F32(_) => {
                panic!("integer oracle must retain original sample words")
            }
        })
        .collect()
}
