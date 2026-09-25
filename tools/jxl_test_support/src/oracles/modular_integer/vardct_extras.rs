//! Decode physical VarDCT extra planes before presentation resampling or blending.
use jxl_frame::data::{PassGroupParams, PassGroupParamsVardct, decode_pass_group};
use jxl_grid::AlignedGrid;

/// Original scalar dimensions and exact integer/float representation words.
#[derive(Debug, PartialEq, Eq)]
pub struct ExtraWords {
    pub width: u32,
    pub height: u32,
    pub words: Vec<u32>,
}

/// Uses jxl-oxide's header, LF/HF, entropy and Modular decoders without rendering.
/// Color coefficients are decoded only to locate each stream's Modular suffix.
/// This intentionally requires a complete VarDCT frame and never rounds a rendered F32 plane.
pub fn vardct_extra_words(data: &[u8], frame_index: usize) -> Vec<ExtraWords> {
    let image = jxl_oxide::JxlImage::read_with_defaults(data).unwrap();
    let frame = image.frame(frame_index).unwrap();
    let header = frame.header();
    assert_eq!(header.encoding, jxl_frame::header::Encoding::VarDct);
    assert_eq!(header.jpeg_upsampling, [0; 3]);
    let lf = frame.try_parse_lf_global::<i32>().unwrap().unwrap();
    assert!(!lf.gmodular.is_partial());
    let mut modular = lf.gmodular.try_clone().unwrap();
    let pool = jxl_threadpool::JxlThreadPool::none();
    let groups = modular
        .modular
        .image_mut()
        .unwrap()
        .prepare_groups(frame.pass_shifts())
        .unwrap();
    let mut lf_images = groups.lf_groups.into_iter();
    let lf_groups: Vec<_> = (0..header.num_lf_groups())
        .map(|group| {
            frame
                .try_parse_lf_group(
                    lf.vardct.as_ref(),
                    lf.gmodular.ma_config(),
                    lf_images.next(),
                    group,
                )
                .unwrap()
                .unwrap()
        })
        .collect();
    assert!(lf_images.next().is_none());
    let hf = frame.try_parse_hf_global(Some(&lf)).unwrap().unwrap();
    let mut coefficients: Vec<[AlignedGrid<i32>; 3]> = (0..header.num_groups())
        .map(|group| {
            let (width, height) = header.group_size_for(group);
            std::array::from_fn(|_| {
                AlignedGrid::with_alloc_tracker(
                    width.div_ceil(8) as usize * 8,
                    height.div_ceil(8) as usize * 8,
                    None,
                )
                .unwrap()
            })
        })
        .collect();
    for (pass, images) in groups.pass_groups.into_iter().enumerate() {
        let mut images = images.into_iter();
        for group in 0..header.num_groups() {
            let mut input = frame
                .pass_group_bitstream(pass as u32, group)
                .unwrap()
                .unwrap();
            assert!(!input.partial);
            let [x, y, b] = &mut coefficients[group as usize];
            let mut output = [x.as_subgrid_mut(), y.as_subgrid_mut(), b.as_subgrid_mut()];
            decode_pass_group(
                &mut input.bitstream,
                PassGroupParams {
                    frame_header: header,
                    lf_group: &lf_groups[header.lf_group_idx_from_group_idx(group) as usize],
                    pass_idx: pass as u32,
                    group_idx: group,
                    global_ma_config: lf.gmodular.ma_config(),
                    modular: images.next(),
                    vardct: Some(PassGroupParamsVardct {
                        lf_vardct: lf.vardct.as_ref().unwrap(),
                        hf_global: &hf,
                        hf_coeff_output: &mut output,
                    }),
                    allow_partial: false,
                    tracker: None,
                    pool: &pool,
                },
            )
            .unwrap();
        }
        assert!(images.next().is_none());
    }
    let image = modular.modular.image_mut().unwrap();
    image.prepare_subimage().unwrap().finish(&pool);
    image
        .image_channels()
        .iter()
        .map(|grid| ExtraWords {
            width: grid.width() as u32,
            height: grid.height() as u32,
            words: grid.buf().iter().map(|&word| word as u32).collect(),
        })
        .collect()
}
