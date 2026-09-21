use super::*;
use jxl_wgpu_encode::{LosslessModularConfig, LosslessModularGroupSize};

pub(super) mod animation;
pub(super) mod lifetime;

fn encoder(
    rig: &Rig,
    group_size: LosslessModularGroupSize,
    tree_mode: LosslessModularTreeMode,
) -> LosslessModularEncoder {
    let config = LosslessModularConfig {
        group_size,
        tree_mode,
        ..Default::default()
    };
    let encoder = LosslessModularEncoder::with_config(rig.context.clone(), config);
    assert_eq!(encoder.config(), config);
    encoder
}

fn check_header(encoded: &[u8], size: LosslessModularGroupSize, extents: &[Extent2d]) {
    let inventory = jxl_gpu_bitstream::parse(encoded, Default::default())
        .unwrap()
        .codestream_inventory(Default::default())
        .unwrap();
    assert_eq!(inventory.frames.len(), extents.len());
    for (frame, extent) in inventory.frames.iter().zip(extents) {
        assert_eq!(frame.group_size_shift, u32::from(size.size_shift()));
        assert_eq!((frame.width, frame.height), (extent.width, extent.height));
        let groups =
            extent.width.div_ceil(size.dimension()) * extent.height.div_ceil(size.dimension());
        let lf_groups = extent.width.div_ceil(size.dimension() * 8)
            * extent.height.div_ceil(size.dimension() * 8);
        assert_eq!(
            frame.sections.len(),
            if groups == 1 {
                1
            } else {
                (groups + lf_groups + 2) as usize
            }
        );
    }
}

#[test]
fn all_group_sizes_preserve_integer_ieee_layouts_and_lf_boundaries() {
    let rig = Rig::new();
    for size in LosslessModularGroupSize::ALL {
        let edge = size.dimension();
        for tree in TREES {
            let encoder = encoder(&rig, size, tree);
            for (format_index, format) in [
                LosslessModularFormat::Gray,
                LosslessModularFormat::GrayAlpha,
                LosslessModularFormat::Rgb,
                LosslessModularFormat::Rgba,
            ]
            .into_iter()
            .enumerate()
            {
                for (index, (kind, bits)) in [
                    (SampleKind::Unsigned, 8),
                    (SampleKind::Unsigned, 31),
                    (SampleKind::Float, 16),
                    (SampleKind::Float, 32),
                ]
                .into_iter()
                .enumerate()
                {
                    let extent = [
                        Extent2d::new(edge - 1, 3),
                        Extent2d::new(edge, 2),
                        Extent2d::new(edge + 1, 3),
                        Extent2d::new(1, 8 * edge + 1),
                    ][(index + format_index) % 4];
                    let case = Case {
                        format,
                        bits,
                        kind,
                        storage: Storage::Planar,
                        reversed: true,
                        byte_order: ByteOrder::Big,
                        shifted: true,
                    };
                    let samples = case.samples(extent);
                    let input = upload(&rig.context, &case, extent, &samples, 4099);
                    let plan = encoder.memory_plan(&input).unwrap();
                    assert_eq!(plan.group_grid.group_size, size);
                    let mut job = encoder.submit_container(input).unwrap();
                    assert_eq!(job.group_grid(), plan.group_grid);
                    let groups: Vec<_> = job.ordered_groups().collect();
                    assert_eq!(
                        groups
                            .iter()
                            .map(|group| u64::from(group.width) * u64::from(group.height))
                            .sum::<u64>(),
                        u64::from(extent.width) * u64::from(extent.height)
                    );
                    let encoded = pollster::block_on(&mut job).unwrap();
                    assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
                    assert_eq!(
                        encoded,
                        encoder
                            .encode_container(upload(
                                &rig.context,
                                &case.canonical(),
                                extent,
                                &samples,
                                0
                            ))
                            .unwrap()
                    );
                    check_header(&encoded, size, &[extent]);
                    check_oracles(&encoded, &samples, &case);
                    color::check_numeric(&rig, &encoded, &[samples], &case);
                }
            }
        }
    }
}

#[test]
fn full_tiles_and_two_dimensional_edges_keep_long_runs_and_dense_words() {
    let mut rig = Rig::new();
    // Large dense groups still use fragmented, bounded input; 64 KiB windows avoid issuing
    // thousands of tiny submissions for a megasample case. The thin matrix uses 256 bytes.
    rig.decoders[1] = GpuDecoder::new(
        WgpuDecodeEngine::new(rig.backend.clone())
            .unwrap()
            .with_stream_window_limit(NonZeroU64::new(65536).unwrap()),
    );
    for size in LosslessModularGroupSize::ALL {
        let encoder = encoder(&rig, size, TREES[0]);
        let edge = size.dimension();
        for zero in [true, false] {
            let case = Case {
                format: if zero {
                    LosslessModularFormat::Gray
                } else {
                    LosslessModularFormat::Rgba
                },
                bits: 8,
                kind: SampleKind::Unsigned,
                storage: Storage::Packed,
                reversed: true,
                byte_order: ByteOrder::Big,
                shifted: true,
            };
            let extent = Extent2d::new(edge + u32::from(!zero), edge + u32::from(!zero));
            let samples = if zero {
                vec![0; (edge * edge) as usize]
            } else {
                case.samples(extent)
            };
            let source = upload(&rig.context, &case, extent, &samples, 0);
            let encoded = encoder.encode(source).unwrap();
            check_header(&encoded, size, &[extent]);
            let native = check_oracles(&encoded, &samples, &case);
            rig.check_gpu(&encoded, &samples, &case, &native);
            assert_eq!(encoder.in_flight_memory_stats().reserved_bytes, 0);
        }
    }
}
