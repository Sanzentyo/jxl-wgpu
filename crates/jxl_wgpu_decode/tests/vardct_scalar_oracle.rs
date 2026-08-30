//! Dev-only scalar oracle for the standard VarDCT GPU frontend fixtures.

use std::collections::BTreeMap;
use std::sync::Arc;

use jxl_frame::{Frame, FrameContext, data::PassGroupParams};
use jxl_gpu_bitstream::FrameSectionKind;
use jxl_grid::AlignedGrid;
use jxl_modular::Sample;
use jxl_oxide_common::Bundle;
use jxl_threadpool::JxlThreadPool;

fn decode_hex(source: &str) -> Vec<u8> {
    let digits: Vec<_> = source
        .bytes()
        .filter(|byte| byte.is_ascii_hexdigit())
        .collect();
    digits
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap();
            let low = (pair[1] as char).to_digit(16).unwrap();
            ((high << 4) | low) as u8
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct ScalarOracle {
    dimensions: (u32, u32),
    groups: u32,
    lf_cursor_after_coefficients: Option<usize>,
    strategies: BTreeMap<(String, i32), usize>,
    lf_extrema: [(i32, i32); 3],
    cfl_nonzero: [usize; 2],
    pass_nonzero: Vec<[usize; 3]>,
    pass_extrema: Vec<[(i32, i32); 3]>,
}

fn scalar_oracle<S: Sample + std::fmt::Debug>(bytes: &[u8]) -> ScalarOracle {
    let pool = JxlThreadPool::none();
    let mut bitstream = jxl_bitstream::Bitstream::new(bytes);
    let image_header = Arc::new(jxl_image::ImageHeader::parse(&mut bitstream, ()).unwrap());
    let mut frame = Frame::parse(
        &mut bitstream,
        FrameContext {
            image_header,
            tracker: None,
            pool: pool.clone(),
        },
    )
    .unwrap();
    let frame_data_offset = bitstream.num_read_bits() / 8;
    frame.feed_bytes(&bytes[frame_data_offset..]).unwrap();
    assert!(frame.is_loading_done());

    let lf_global = frame.try_parse_lf_global::<S>().unwrap().unwrap();
    let lf_vardct = lf_global.vardct.as_ref().unwrap();
    let mut gmodular = lf_global.gmodular.try_clone().unwrap();
    let prepared = gmodular
        .modular
        .image_mut()
        .map(|image| image.prepare_groups(frame.pass_shifts()).unwrap());
    let (mut mlf_groups, pass_groups) = prepared
        .map(|prepared| (prepared.lf_groups, prepared.pass_groups))
        .unwrap_or_default();

    let inventory = jxl_gpu_bitstream::parse(bytes, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    let lf_cursor_after_coefficients = inventory.frames[0]
        .sections
        .iter()
        .find(|section| section.kind == FrameSectionKind::LowFrequencyGroup { group_index: 0 })
        .map(|section| {
            let mut cursor = jxl_bitstream::Bitstream::new(bytes);
            cursor.skip_bits(section.bits.offset as usize).unwrap();
            let (lf_width, lf_height) = frame.header().lf_group_size_for(0);
            let _ = jxl_vardct::LfCoeff::<S>::parse(
                &mut cursor,
                jxl_vardct::LfCoeffParams {
                    lf_group_idx: 0,
                    lf_width,
                    lf_height,
                    jpeg_upsampling: frame.header().jpeg_upsampling,
                    bits_per_sample: frame.header().bit_depth.bits_per_sample(),
                    global_ma_config: lf_global.gmodular.ma_config(),
                    allow_partial: false,
                    tracker: None,
                    pool: &pool,
                },
            )
            .unwrap();
            cursor.num_read_bits()
        });

    let mut lf_groups = Vec::new();
    let mut strategies = BTreeMap::new();
    let mut lf_extrema = [(0, 0); 3];
    let mut cfl_nonzero = [0; 2];
    for lf_group_idx in 0..frame.header().num_lf_groups() {
        let modular = if mlf_groups.is_empty() {
            None
        } else {
            Some(mlf_groups.remove(0))
        };
        let lf_group = frame
            .try_parse_lf_group(
                Some(lf_vardct),
                lf_global.gmodular.ma_config(),
                modular,
                lf_group_idx,
            )
            .unwrap()
            .unwrap();
        let hf_meta = lf_group.hf_meta.as_ref().unwrap();
        for value in hf_meta.block_info.buf() {
            if let jxl_vardct::BlockInfo::Data { dct_select, hf_mul } = value {
                *strategies
                    .entry((format!("{dct_select:?}"), *hf_mul))
                    .or_insert(0) += 1;
            }
        }
        let lf_coeff = lf_group.lf_coeff.as_ref().unwrap();
        for (channel, extrema) in lf_coeff
            .lf_quant
            .image()
            .unwrap()
            .image_channels()
            .iter()
            .zip(&mut lf_extrema)
        {
            *extrema = channel
                .buf()
                .iter()
                .map(|sample| sample.to_i32())
                .fold((0, 0), |(low, high), value| {
                    (low.min(value), high.max(value))
                });
        }
        cfl_nonzero[0] += hf_meta
            .x_from_y
            .buf()
            .iter()
            .filter(|&&value| value != 0)
            .count();
        cfl_nonzero[1] += hf_meta
            .b_from_y
            .buf()
            .iter()
            .filter(|&&value| value != 0)
            .count();
        lf_groups.push(lf_group);
    }

    let hf_global = frame
        .try_parse_hf_global(Some(&lf_global))
        .unwrap()
        .unwrap();
    let mut pass_nonzero = Vec::new();
    let mut pass_extrema = Vec::new();
    for group_idx in 0..frame.header().num_groups() {
        let (width, height) = frame.header().group_size_for(group_idx);
        let width = width.div_ceil(8) as usize * 8;
        let height = height.div_ceil(8) as usize * 8;
        let mut grids = [
            AlignedGrid::<i32>::with_alloc_tracker(width, height, None).unwrap(),
            AlignedGrid::<i32>::with_alloc_tracker(width, height, None).unwrap(),
            AlignedGrid::<i32>::with_alloc_tracker(width, height, None).unwrap(),
        ];
        let mut subgrids = grids.each_mut().map(AlignedGrid::as_subgrid_mut);
        let pass_group = frame.pass_group_bitstream(0, group_idx).unwrap().unwrap();
        let mut pass_bitstream = pass_group.bitstream;
        let lf_group_idx = frame.header().lf_group_idx_from_group_idx(group_idx);
        assert!(pass_groups.iter().all(Vec::is_empty));
        jxl_frame::data::decode_pass_group(
            &mut pass_bitstream,
            PassGroupParams {
                frame_header: frame.header(),
                lf_group: &lf_groups[lf_group_idx as usize],
                pass_idx: 0,
                group_idx,
                global_ma_config: lf_global.gmodular.ma_config(),
                modular: None,
                vardct: Some(jxl_frame::data::PassGroupParamsVardct {
                    lf_vardct,
                    hf_global: &hf_global,
                    hf_coeff_output: &mut subgrids,
                }),
                allow_partial: pass_group.partial,
                tracker: None,
                pool: &pool,
            },
        )
        .unwrap();
        pass_nonzero.push(
            grids
                .each_ref()
                .map(|grid| grid.buf().iter().filter(|&&value| value != 0).count()),
        );
        pass_extrema.push(grids.each_ref().map(|grid| {
            grid.buf()
                .iter()
                .copied()
                .fold((0, 0), |(low, high), value| {
                    (low.min(value), high.max(value))
                })
        }));
    }

    ScalarOracle {
        dimensions: (frame.header().width, frame.header().height),
        groups: frame.header().num_groups(),
        lf_cursor_after_coefficients,
        strategies,
        lf_extrema,
        cfl_nonzero,
        pass_nonzero,
        pass_extrema,
    }
}

#[test]
fn basic_scalar_oracle_pins_special_transform_packet() {
    let bytes = decode_hex(include_str!("../test-data/basic.jxl.hex"));
    let mut strategies = BTreeMap::new();
    strategies.insert(("Dct4x8".into(), 10), 1);
    assert_eq!(
        scalar_oracle::<i32>(&bytes),
        ScalarOracle {
            dimensions: (1, 1),
            groups: 1,
            lf_cursor_after_coefficients: None,
            strategies,
            lf_extrema: [(0, 288), (0, 121), (-5, 0)],
            cfl_nonzero: [0, 0],
            pass_nonzero: vec![[0, 0, 0]],
            pass_extrema: vec![[(0, 0), (0, 0), (0, 0)]],
        }
    );
}

#[test]
fn green_queen_scalar_oracle_pins_grouped_dct8_packets() {
    let bytes = decode_hex(include_str!(
        "../../jxl_gpu_bitstream/test-data/green_queen_vardct_e3.jxl.hex"
    ));
    let mut strategies = BTreeMap::new();
    strategies.insert(("Dct8".into(), 6), 4_070);
    assert_eq!(
        scalar_oracle::<i32>(&bytes),
        ScalarOracle {
            dimensions: (438, 589),
            groups: 6,
            lf_cursor_after_coefficients: Some(67_171),
            strategies,
            lf_extrema: [(0, 560), (-46, 53), (-32, 27)],
            cfl_nonzero: [0, 0],
            pass_nonzero: vec![
                [2_257, 26_028, 1_942],
                [1_259, 23_509, 1_606],
                [1_922, 25_654, 1_438],
                [1_695, 27_221, 1_616],
                [545, 6_944, 413],
                [519, 6_905, 426],
            ],
            pass_extrema: vec![
                [(-6, 8), (-83, 98), (-8, 10)],
                [(-5, 5), (-74, 88), (-7, 8)],
                [(-7, 9), (-100, 106), (-8, 8)],
                [(-10, 6), (-123, 108), (-8, 8)],
                [(-5, 9), (-82, 87), (-7, 7)],
                [(-8, 6), (-77, 87), (-6, 7)],
            ],
        }
    );
}
