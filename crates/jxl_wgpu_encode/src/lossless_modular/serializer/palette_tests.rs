use super::*;
use crate::lossless_modular::transform::{PaletteArtifactPlan, PaletteCapacity};
use crate::{
    LosslessModularColorTransform, LosslessModularConfig, LosslessModularGroupSize,
    LosslessModularPalette, LosslessModularSqueeze,
};

fn write_planned_transforms(
    output: &mut BitWriter,
    channels: u32,
    rct: Option<LosslessModularRctType>,
    squeeze: LosslessModularSqueeze,
    palette: LosslessModularPalette,
    counts: (u32, u32),
) {
    let format = [
        LosslessModularFormat::Gray,
        LosslessModularFormat::GrayAlpha,
        LosslessModularFormat::Rgb,
        LosslessModularFormat::Rgba,
    ][channels as usize - 1];
    let grid =
        LosslessModularGroupGrid::for_extent(1024, 1024, LosslessModularGroupSize::Pixels1024)
            .unwrap();
    let plan = ModularTransformPlan::new(
        grid,
        format,
        31,
        0,
        LosslessModularConfig {
            palette: Some(palette),
            squeeze,
            color_transform: rct.map_or(
                LosslessModularColorTransform::None,
                LosslessModularColorTransform::GlobalRct,
            ),
            ..Default::default()
        },
    )
    .unwrap();
    let group = plan.group(grid.group(0).unwrap()).unwrap();
    write_transforms(
        output,
        TransformHeader {
            operations: &group.operations,
            palette_counts: Some(
                group
                    .palette
                    .unwrap()
                    .capacity
                    .validate(counts.0, counts.1)
                    .unwrap(),
            ),
        },
    )
    .unwrap();
}

use jxl_bitstream::{Bitstream, U};

#[test]
fn independent_wire_reader_checks_delta_counts_and_every_predictor() {
    for predictor in LosslessModularPredictor::ALL {
        for entries in [1, 256, 257, 1280, 1281, 66816] {
            let mut output = BitWriter::new();
            write_planned_transforms(
                &mut output,
                4,
                None,
                LosslessModularSqueeze::None,
                LosslessModularPalette::deltas(entries, predictor).unwrap(),
                (entries, entries),
            );
            let bit_len = output.bit_len();
            let bytes = output.into_bytes();
            let mut bits = Bitstream::new(&bytes);
            assert_eq!(bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap(), 1);
            assert_eq!(bits.read_bits(2).unwrap(), 1);
            assert_eq!(
                bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                    .unwrap(),
                0
            );
            assert_eq!(bits.read_u32(1, 3, 4, 1 + U(13)).unwrap(), 4);
            assert_eq!(
                bits.read_u32(U(8), 256 + U(10), 1280 + U(12), 5376 + U(16))
                    .unwrap(),
                0
            );
            assert_eq!(
                bits.read_u32(0, 1 + U(8), 257 + U(10), 1281 + U(16))
                    .unwrap(),
                entries
            );
            assert_eq!(bits.read_bits(4).unwrap(), predictor.value());
            assert_eq!(bits.num_read_bits(), bit_len);
        }
    }
}

#[test]
fn wire_headers_require_counts_validated_against_their_own_plan() {
    let grid = LosslessModularGroupGrid::for_extent(5, 3, Default::default()).unwrap();
    let plan = ModularTransformPlan::new(
        grid,
        LosslessModularFormat::Gray,
        8,
        0,
        LosslessModularConfig {
            palette: Some(LosslessModularPalette::new(1).unwrap()),
            ..Default::default()
        },
    )
    .unwrap();
    let group = plan.group(grid.group(0).unwrap()).unwrap();
    let other_counts = PaletteCapacity {
        colors: 2,
        deltas: 0,
    }
    .validate(2, 0)
    .unwrap();
    for header in [
        TransformHeader {
            operations: &group.operations,
            palette_counts: None,
        },
        TransformHeader {
            operations: &group.operations,
            palette_counts: Some(other_counts),
        },
        TransformHeader {
            operations: &[],
            palette_counts: Some(other_counts),
        },
    ] {
        assert!(matches!(
            write_transforms(&mut BitWriter::new(), header),
            Err(EncodeError::Backend(BackendError::InvalidArtifact(_)))
        ));
    }
}

#[test]
fn independent_wire_reader_keeps_palette_geometry_and_skips_its_meta_channel() {
    for channels in 1..=4 {
        for (begin, selected) in
            (0..channels).flat_map(|begin| (1..=channels - begin).map(move |count| (begin, count)))
        {
            for colors in [1, 255, 256, 1279, 1280, 5375, 5376, 70911] {
                for (squeeze, axes) in [
                    (LosslessModularSqueeze::None, &[][..]),
                    (LosslessModularSqueeze::Horizontal, &[true][..]),
                    (LosslessModularSqueeze::Vertical, &[false][..]),
                    (
                        LosslessModularSqueeze::HorizontalThenVertical,
                        &[true, false][..],
                    ),
                    (
                        LosslessModularSqueeze::VerticalThenHorizontal,
                        &[false, true][..],
                    ),
                ] {
                    let rct = (channels >= 3).then(|| LosslessModularRctType::new(41).unwrap());
                    let mut output = BitWriter::new();
                    write_planned_transforms(
                        &mut output,
                        channels,
                        rct,
                        squeeze,
                        LosslessModularPalette::new(colors)
                            .unwrap()
                            .with_components(begin, selected)
                            .unwrap(),
                        (colors, 0),
                    );
                    let bit_len = output.bit_len();
                    let bytes = output.into_bytes();
                    let mut bits = Bitstream::new(&bytes);
                    assert_eq!(
                        bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap(),
                        1 + u32::from(rct.is_some()) + u32::from(!axes.is_empty())
                    );
                    if rct.is_some() {
                        assert_eq!(bits.read_bits(2).unwrap(), 0);
                        assert_eq!(
                            bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                                .unwrap(),
                            0
                        );
                        assert_eq!(bits.read_u32(6, U(2), 2 + U(4), 10 + U(6)).unwrap(), 41);
                    }
                    assert_eq!(bits.read_bits(2).unwrap(), 1);
                    assert_eq!(
                        bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                            .unwrap(),
                        begin
                    );
                    assert_eq!(bits.read_u32(1, 3, 4, 1 + U(13)).unwrap(), selected);
                    assert_eq!(
                        bits.read_u32(U(8), 256 + U(10), 1280 + U(12), 5376 + U(16))
                            .unwrap(),
                        colors
                    );
                    assert_eq!(
                        bits.read_u32(0, 1 + U(8), 257 + U(10), 1281 + U(16))
                            .unwrap(),
                        0
                    );
                    assert_eq!(bits.read_bits(4).unwrap(), 0);
                    if !axes.is_empty() {
                        assert_eq!(bits.read_bits(2).unwrap(), 2);
                        assert_eq!(
                            bits.read_u32(0, 1 + U(4), 9 + U(6), 41 + U(8)).unwrap(),
                            axes.len() as u32
                        );
                        for (stage, &horizontal) in axes.iter().enumerate() {
                            assert_eq!(bits.read_bool().unwrap(), horizontal);
                            assert!(!bits.read_bool().unwrap());
                            assert_eq!(
                                bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                                    .unwrap(),
                                1
                            );
                            assert_eq!(
                                bits.read_u32(1, 2, 3, 4 + U(4)).unwrap(),
                                (channels - selected + 1) << stage
                            );
                        }
                    }
                    assert_eq!(bits.num_read_bits(), bit_len);
                }
            }
        }
    }
}

#[test]
fn palette_artifacts_validate_status_count_bounds_and_exact_dynamic_coverage() {
    let mut header = ModularArtifactHeader {
        event_count: 2,
        raw_counts: [0; RAW_SYMBOLS],
        lz77_counts: [0; LZ77_SYMBOLS],
        distance_counts: [0; RAW_SYMBOLS],
    };
    header.raw_counts[0] = 2;
    let events = [ModularEvent {
        kind: 0,
        token: 0,
        extra_bit_count: 0,
        extra_bits: 0,
    }; 2];
    let mut bytes = bytemuck::bytes_of(&header).to_vec();
    bytes.extend_from_slice(bytemuck::cast_slice(&events));
    let count_offset = bytes.len();
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let mut plan = ModularGroupPlan {
        group_index: 0,
        width: 4,
        height: 2,
        channel: 0,
        artifact_byte_offset: 0,
        output_size: bytes.len() as u64,
        max_events: 8,
        palette: Some(PaletteArtifactPlan {
            counts_byte_offset: count_offset as u64,
            capacity: PaletteCapacity {
                colors: 4,
                deltas: 0,
            },
            scratch_bytes: 8,
        }),
    };
    assert_eq!(
        parse_planned_artifact(&plan, &bytes)
            .unwrap()
            .palette_counts
            .map(|counts| (counts.entries(), counts.deltas())),
        Some((1, 0))
    );
    // All counts inside the capacity must still cover their actual table, not its allocation.
    for count in [0u32, 2, 4, 5, u32::MAX] {
        bytes[count_offset..count_offset + 4].copy_from_slice(&count.to_le_bytes());
        assert!(parse_planned_artifact(&plan, &bytes).is_err(), "{count}");
    }
    bytes[count_offset..count_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    assert!(parse_planned_artifact(&plan, &bytes[..count_offset + 3]).is_err());
    plan.palette.as_mut().unwrap().counts_byte_offset = u64::MAX;
    assert!(parse_planned_artifact(&plan, &bytes).is_err());
    bytes[..4].copy_from_slice(&(u32::MAX - 2).to_le_bytes());
    assert!(matches!(
        parse_planned_artifact(&plan, &bytes),
        Err(EncodeError::Backend(BackendError::ModularPaletteOverflow))
    ));
    bytes[..4].copy_from_slice(&(u32::MAX - 3).to_le_bytes());
    assert!(matches!(
        parse_planned_artifact(&plan, &bytes),
        Err(EncodeError::Backend(BackendError::InvalidArtifact(_)))
    ));
}

#[test]
fn mixed_palette_metadata_rejects_partition_overflow_and_truncation_before_tokens() {
    let mut header = ModularArtifactHeader {
        event_count: 4,
        raw_counts: [0; RAW_SYMBOLS],
        lz77_counts: [0; LZ77_SYMBOLS],
        distance_counts: [0; RAW_SYMBOLS],
    };
    header.raw_counts[0] = 4;
    let events = [ModularEvent {
        kind: 0,
        token: 0,
        extra_bit_count: 0,
        extra_bits: 0,
    }; 4];
    let mut bytes = bytemuck::bytes_of(&header).to_vec();
    bytes.extend_from_slice(bytemuck::cast_slice(&events));
    let offset = bytes.len();
    bytes.resize(offset + 8, 0);
    let mut plan = ModularGroupPlan {
        group_index: 0,
        width: 3,
        height: 2,
        channel: 0,
        artifact_byte_offset: 0,
        output_size: bytes.len() as u64,
        max_events: 8,
        palette: Some(PaletteArtifactPlan {
            counts_byte_offset: offset as u64,
            capacity: PaletteCapacity {
                colors: 1,
                deltas: 2,
            },
            scratch_bytes: 8,
        }),
    };
    for (entries, deltas) in [(2u32, 1u32), (2, 2)] {
        bytes[offset..offset + 4].copy_from_slice(&entries.to_le_bytes());
        bytes[offset + 4..].copy_from_slice(&deltas.to_le_bytes());
        assert_eq!(
            parse_planned_artifact(&plan, &bytes)
                .unwrap()
                .palette_counts
                .map(|counts| (counts.entries(), counts.deltas())),
            Some((entries, deltas))
        );
    }
    for (entries, deltas) in [
        (0u32, 0u32),
        (2, 0),
        (2, 3),
        (3, 3),
        (4, 2),
        (u32::MAX, 1),
        (2, u32::MAX),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&entries.to_le_bytes());
        bytes[offset + 4..].copy_from_slice(&deltas.to_le_bytes());
        assert!(
            parse_planned_artifact(&plan, &bytes).is_err(),
            "{entries}, {deltas}"
        );
    }
    bytes[offset..offset + 4].copy_from_slice(&2u32.to_le_bytes());
    bytes[offset + 4..].copy_from_slice(&1u32.to_le_bytes());
    for end in offset..offset + 8 {
        assert!(parse_planned_artifact(&plan, &bytes[..end]).is_err());
    }
    plan.palette.as_mut().unwrap().capacity.deltas = 4;
    assert!(parse_planned_artifact(&plan, &bytes).is_err());
}

#[test]
fn independent_wire_reader_checks_both_mixed_count_buckets_and_zero_used_deltas() {
    for predictor in LosslessModularPredictor::ALL {
        for colors in [1, 255, 256, 1279, 1280, 5375, 5376, 70911] {
            for deltas in [0, 1, 256, 257, 1280, 1281, 66816] {
                let mut output = BitWriter::new();
                write_planned_transforms(
                    &mut output,
                    3,
                    None,
                    LosslessModularSqueeze::None,
                    LosslessModularPalette::mixed(colors, deltas.max(1), predictor).unwrap(),
                    (colors + deltas, deltas),
                );
                let bit_len = output.bit_len();
                let bytes = output.into_bytes();
                let mut bits = Bitstream::new(&bytes);
                assert_eq!(bits.read_u32(0, 1, 2 + U(4), 18 + U(8)).unwrap(), 1);
                assert_eq!(bits.read_bits(2).unwrap(), 1);
                assert_eq!(
                    bits.read_u32(U(3), 8 + U(6), 72 + U(10), 1096 + U(13))
                        .unwrap(),
                    0
                );
                assert_eq!(bits.read_u32(1, 3, 4, 1 + U(13)).unwrap(), 3);
                assert_eq!(
                    bits.read_u32(U(8), 256 + U(10), 1280 + U(12), 5376 + U(16))
                        .unwrap(),
                    colors
                );
                assert_eq!(
                    bits.read_u32(0, 1 + U(8), 257 + U(10), 1281 + U(16))
                        .unwrap(),
                    deltas
                );
                assert_eq!(bits.read_bits(4).unwrap(), predictor.value());
                assert_eq!(bits.num_read_bits(), bit_len);
            }
        }
    }
}
