use super::*;
use jxl_bitstream::{Bitstream, U};

#[test]
fn independent_wire_reader_keeps_palette_geometry_and_skips_its_meta_channel() {
    for channels in 1..=4 {
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
                write_transforms(
                    &mut output,
                    TransformHeader {
                        rct,
                        squeeze,
                        palette_colors: Some(colors),
                        channels,
                    },
                )
                .unwrap();
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
                    0
                );
                assert_eq!(bits.read_u32(1, 3, 4, 1 + U(13)).unwrap(), channels);
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
                        assert_eq!(bits.read_u32(1, 2, 3, 4 + U(4)).unwrap(), 1 << stage);
                    }
                }
                assert_eq!(bits.num_read_bits(), bit_len);
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
    let mut plan = ModularGroupPlan {
        group_index: 0,
        width: 4,
        height: 2,
        channel: 0,
        artifact_byte_offset: 0,
        output_size: bytes.len() as u64,
        max_events: 8,
        palette_colors_byte_offset: Some(count_offset as u64),
    };
    assert_eq!(
        parse_planned_artifact(&plan, &bytes)
            .unwrap()
            .palette_colors,
        Some(1)
    );
    // All counts inside the capacity must still cover their actual table, not its allocation.
    for count in [0u32, 2, 4, 5, u32::MAX] {
        bytes[count_offset..].copy_from_slice(&count.to_le_bytes());
        assert!(parse_planned_artifact(&plan, &bytes).is_err(), "{count}");
    }
    bytes[count_offset..].copy_from_slice(&1u32.to_le_bytes());
    assert!(parse_planned_artifact(&plan, &bytes[..count_offset + 3]).is_err());
    plan.palette_colors_byte_offset = Some(u64::MAX);
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
