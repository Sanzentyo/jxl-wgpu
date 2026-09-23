use super::*;

fn raw(value: u32, kind: u32) -> ModularEvent {
    let token = 32 - value.leading_zeros();
    let extra_bit_count = token.saturating_sub(1);
    ModularEvent {
        kind,
        token,
        extra_bit_count,
        extra_bits: if value == 0 {
            0
        } else {
            value - (1 << extra_bit_count)
        },
    }
}

fn read_entropy(bits: &mut jxl_bitstream::Bitstream<'_>) -> jxl_coding::Decoder {
    let mut tree = jxl_coding::Decoder::parse(bits, 6).unwrap();
    tree.begin(bits).unwrap();
    let mut remaining = 1;
    while remaining != 0 {
        remaining -= 1;
        if tree.read_varint(bits, 1).unwrap() == 0 {
            for context in 2..6 {
                tree.read_varint(bits, context).unwrap();
            }
        } else {
            tree.read_varint(bits, 0).unwrap();
            remaining += 2;
        }
    }
    tree.finalize().unwrap();
    jxl_coding::Decoder::parse(bits, 4).unwrap()
}

#[test]
fn independent_decoder_reads_general_lz77_distances_and_overlaps() {
    for distance in [
        1u32,
        2,
        7,
        16,
        119,
        120,
        121,
        255,
        256,
        257,
        65_535,
        65_536,
        1 << 20,
    ] {
        let seeds: Vec<_> = (0..distance)
            .map(|index| index.wrapping_mul(0x9e37_79b9).rotate_left(13))
            .collect();
        let mut events: Vec<_> = seeds.iter().map(|&value| raw(value, 0)).collect();
        events.push(ModularEvent {
            kind: 2,
            token: 16,
            extra_bit_count: 4,
            extra_bits: 8,
        }); // 24 + 7 = 31 copied samples
        let distance_event = raw(distance + 119, 3);
        events.push(distance_event);
        let mut raw_counts = [[0; RAW_SYMBOLS]; 4];
        for event in &events[..seeds.len()] {
            raw_counts[0][event.token as usize] += 1;
        }
        let mut lz77_counts = [[0; LZ77_SYMBOLS]; 4];
        lz77_counts[0][16] = 1;
        let mut distance_counts = [0; RAW_SYMBOLS];
        distance_counts[distance_event.token as usize] = 1;
        let codes = build_prefix_codes(
            LosslessModularFormat::Gray,
            31,
            LosslessModularPredictor::Zero,
            LosslessModularSqueeze::None,
            None,
            &raw_counts,
            &lz77_counts,
        )
        .unwrap();
        let distance_code =
            build_distance_code(LosslessModularLz77::Greedy, &distance_counts).unwrap();
        let mut output = BitWriter::new();
        write_ma_config(
            &mut output,
            &codes,
            LosslessModularPredictor::Zero,
            distance_code.as_ref(),
        )
        .unwrap();
        write_events(&mut output, &codes[0], distance_code.as_ref(), &events).unwrap();
        let bit_len = output.bit_len();
        output.align_to_byte().unwrap();
        let bytes = output.into_bytes();
        for multiplier in [1, 3, 129, 1024] {
            let mut bits = jxl_bitstream::Bitstream::new(&bytes);
            let mut entropy = read_entropy(&mut bits);
            entropy.begin(&mut bits).unwrap();
            // The channel-index tree's last leaf is channel zero.
            for &value in &seeds {
                assert_eq!(
                    entropy
                        .read_varint_with_multiplier(&mut bits, 3, multiplier)
                        .unwrap(),
                    value
                );
            }
            for index in 0..31 {
                assert_eq!(
                    entropy
                        .read_varint_with_multiplier(&mut bits, 3, multiplier)
                        .unwrap(),
                    seeds[index % seeds.len()]
                );
            }
            entropy.finalize().unwrap();
            assert_eq!(bits.num_read_bits(), bit_len);
        }
    }
}

#[test]
fn lz77_artifacts_reject_invalid_distances_order_histograms_and_coverage() {
    let literal = raw(2, 0);
    let matched = ModularEvent {
        kind: 2,
        token: 0,
        extra_bit_count: 0,
        extra_bits: 0,
    };
    let distance = raw(120, 3);
    let mut header = ModularArtifactHeader {
        event_count: 3,
        raw_counts: [0; RAW_SYMBOLS],
        lz77_counts: [0; LZ77_SYMBOLS],
        distance_counts: [0; RAW_SYMBOLS],
    };
    header.raw_counts[2] = 1;
    header.lz77_counts[0] = 1;
    header.distance_counts[7] = 1;
    let events = [literal, matched, distance];
    validate_gpu_artifacts(8, 1, &header, &events).unwrap();
    for invalid in [
        raw(0, 3),
        raw(119, 3),
        raw(121, 3),
        raw((1 << 20) + 120, 3),
        ModularEvent {
            token: 33,
            ..distance
        },
        ModularEvent {
            extra_bit_count: 5,
            ..distance
        },
        ModularEvent {
            extra_bits: 64,
            ..distance
        },
        ModularEvent {
            kind: 0,
            ..distance
        },
    ] {
        assert!(validate_gpu_artifacts(8, 1, &header, &[literal, matched, invalid]).is_err());
    }
    for invalid in [
        vec![matched, distance],
        vec![literal, matched],
        vec![literal, distance, matched],
        vec![literal, matched, distance, distance],
    ] {
        assert!(validate_gpu_artifacts(8, 1, &header, &invalid).is_err());
    }
    assert!(validate_gpu_artifacts(7, 1, &header, &events).is_err());
    header.distance_counts[7] += 1;
    assert!(validate_gpu_artifacts(8, 1, &header, &events).is_err());
    assert!(
        build_distance_code(
            LosslessModularLz77::ZeroRuns,
            &header.distance_counts.map(u64::from)
        )
        .is_err()
    );
    let code = PrefixCode::fixed_unused_channel();
    assert!(write_events(&mut BitWriter::new(), &code, None, &events).is_err());
    let distances = build_distance_code(LosslessModularLz77::Greedy, &[0; RAW_SYMBOLS]).unwrap();
    assert!(
        write_events(
            &mut BitWriter::new(),
            &code,
            distances.as_ref(),
            &[ModularEvent { kind: 1, ..matched }]
        )
        .is_err()
    );
}
