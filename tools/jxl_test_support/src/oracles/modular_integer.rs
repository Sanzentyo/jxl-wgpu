//! Exact original working words from jxl-oxide, before any floating-point conversion.

mod vardct_extras;
pub use vardct_extras::{ExtraWords, modular_channel_words, vardct_extra_words};

/// Exact retained extra-channel words, including Modular planes accompanying VarDCT color.
/// Presentation arithmetic must not have converted these to F32; that is a separate oracle.
pub fn extra_planes(data: &[u8], frame_index: usize) -> Vec<Vec<i32>> {
    let image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    let render = image.render_frame(frame_index).unwrap();
    render
        .extra_channels()
        .1
        .iter()
        .map(|channel| match channel {
            jxl_render::ImageBuffer::I32(grid) => grid.buf().to_vec(),
            jxl_render::ImageBuffer::I16(grid) => {
                grid.buf().iter().map(|&v| i32::from(v)).collect()
            }
            jxl_render::ImageBuffer::F32(_) => {
                panic!("extra-channel oracle must retain original sample words")
            }
        })
        .collect()
}

/// Independently inspect local headers of a one-pass Modular encoder frame.
/// Returns `(use_global_tree, rct_type)` for each separate pass group. Fused single-group
/// frames have no local header. This helper accepts zero or one RCT, and intentionally
/// rejects broader transform stacks.
pub fn local_rct_headers(data: &[u8], frame_index: usize) -> Vec<(bool, Option<u32>)> {
    local_modular_headers(data, frame_index)
        .into_iter()
        .map(|header| (header.global_tree, header.rct))
        .collect()
}

/// Independently parsed declarations in a separate Modular pass group.
#[derive(Debug, PartialEq, Eq)]
pub struct LocalModularHeader {
    pub global_tree: bool,
    pub coefficients: [u8; 7],
    pub max_weights: [u8; 4],
    pub rct: Option<u32>,
}

/// Reads the WP parameters even when the selected tree does not use Weighted prediction.
pub fn local_modular_headers(data: &[u8], frame_index: usize) -> Vec<LocalModularHeader> {
    use jxl_bitstream::U;
    let image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    let frame = image.frame(frame_index).unwrap();
    if frame.toc().is_single_entry() {
        return Vec::new();
    }
    (0..frame.header().num_groups())
        .map(|group| {
            let mut stream = frame.pass_group_bitstream(0, group).unwrap().unwrap();
            assert!(!stream.partial);
            let bits = &mut stream.bitstream;
            let global_tree = bits.read_bool().unwrap();
            let default_wp = bits.read_bool().unwrap();
            let coefficients = if default_wp {
                [16, 10, 7, 7, 7, 0, 0]
            } else {
                std::array::from_fn(|_| bits.read_bits(5).unwrap() as u8)
            };
            let max_weights = if default_wp {
                [13, 12, 12, 12]
            } else {
                std::array::from_fn(|_| bits.read_bits(4).unwrap() as u8)
            };
            let count = bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap();
            assert!(count <= 1);
            let rct = (count == 1).then(|| {
                assert_eq!(bits.read_bits(2).unwrap(), 0);
                assert_eq!(
                    bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                        .unwrap(),
                    0
                );
                bits.read_u32(6, U(2), 2 + U(4), 10 + U(6)).unwrap()
            });
            LocalModularHeader {
                global_tree,
                coefficients,
                max_weights,
                rct,
            }
        })
        .collect()
}

/// Decode one original-color Modular presentation with its declared color encoding.
///
/// Integer samples and raw floating-point representations both use integer Modular storage.
/// This oracle intentionally requires those retained planes. A filtered or composed F32
/// result cannot prove exact high-depth source words and is rejected instead of rounded back.
pub fn original_planes(data: &[u8], frame_index: usize) -> Vec<Vec<i32>> {
    let mut image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    assert!(!image.image_header().metadata.xyb_encoded);
    match &image.image_header().metadata.colour_encoding {
        jxl_image::color::ColourEncoding::Enum(encoding) => {
            image.request_color_encoding(encoding.clone());
        }
        jxl_image::color::ColourEncoding::IccProfile(_) => {
            let profile = image
                .original_icc()
                .expect("embedded original ICC")
                .to_vec();
            image.request_icc(&profile).unwrap();
        }
    }
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
